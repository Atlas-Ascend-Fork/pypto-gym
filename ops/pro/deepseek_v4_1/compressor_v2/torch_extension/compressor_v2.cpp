/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gym_aclnn.h"
#include "compressor_v2_plan.h"
#include <acl/acl_rt.h>
#include <array>
#include <torch/library.h>
#include <type_traits>

namespace {
namespace host = compressor_v2_host;
using GetWorkspace = aclnnStatus (*)(const aclTensor *, const aclTensor *,
                                     const aclTensor *, const aclTensor *,
                                     const aclTensor *, const aclTensor *,
                                     const aclTensor *, const aclTensor *,
                                     const aclTensor *, const aclTensor *,
                                     const aclTensor *, const aclTensor *,
                                     const aclTensor *, const aclTensor *,
                                     int64_t, int64_t, bool, bool, uint64_t *,
                                     aclOpExecutor **);

static host::TensorSpec MakeSpec(const at::Tensor &tensor) {
  host::DType dtype = host::DType::Other;
  switch (tensor.scalar_type()) {
  case at::kHalf:
    dtype = host::DType::Float16;
    break;
  case at::kBFloat16:
    dtype = host::DType::BFloat16;
    break;
  case at::kFloat:
    dtype = host::DType::Float32;
    break;
  case at::kInt:
    dtype = host::DType::Int32;
    break;
  default:
    break;
  }
  return {dtype, tensor.sizes().vec(), tensor.strides().vec()};
}

static void RunStage(int64_t stage, int64_t ratio, bool has_used,
                     bool has_start,
                     const std::array<gym_torch::TensorHandle, 14> &tensors,
                     c10_npu::NPUStream stream) {
  static_assert(std::is_same_v<GetWorkspace,
                               decltype(&aclnnPyptoCompressorV2StageGetWorkspaceSize)>);
  static const std::array<GetWorkspace, 2> queries = {
      gym_torch::Symbol<GetWorkspace>(
          "aclnnPyptoCompressorV2StageGetWorkspaceSize"),
      gym_torch::Symbol<GetWorkspace>(
          "aclnnPyptoCompressorV2VectorGetWorkspaceSize")};
  static const std::array<gym_torch::Execute, 2> executes = {
      gym_torch::Symbol<gym_torch::Execute>("aclnnPyptoCompressorV2Stage"),
      gym_torch::Symbol<gym_torch::Execute>("aclnnPyptoCompressorV2Vector")};
  const size_t entry = stage == 1 ? 1 : 0;
  uint64_t workspace_size = 0;
  aclOpExecutor *executor = nullptr;
  const auto status = queries[entry](
      tensors[0].get(), tensors[1].get(), tensors[2].get(), tensors[3].get(),
      tensors[4].get(), tensors[5].get(), tensors[6].get(), tensors[7].get(),
      tensors[8].get(), tensors[9].get(), tensors[10].get(), tensors[11].get(),
      tensors[12].get(), tensors[13].get(), stage, ratio, has_used, has_start,
      &workspace_size, &executor);
  TORCH_CHECK(status == 0, "CompressorV2 GetWorkspaceSize failed at stage ",
              stage, ", status=", status, ": ", aclGetRecentErrMsg());
  gym_torch::ExecuteOp(executes[entry], workspace_size, executor, stream);
}

static at::Tensor Run(const at::Tensor &x, const at::Tensor &wkv,
                      const at::Tensor &wgate, const at::Tensor &cache,
                      const at::Tensor &table,
                      const c10::optional<at::Tensor> &cu_seqlens,
                      const c10::optional<at::Tensor> &seqused,
                      const c10::optional<at::Tensor> &start_pos,
                      int64_t cmp_ratio) {
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
              "compressor_v2: x must be on NPU");
  const c10::OptionalDeviceGuard guard(x.device());
  for (const auto &tensor : {x, wkv, wgate, cache, table})
    TORCH_CHECK(tensor.device() == x.device(),
                "compressor_v2: all tensors must live on the same NPU device");
  for (const auto &tensor : {cu_seqlens, seqused, start_pos})
    TORCH_CHECK(!tensor.has_value() || tensor->device() == x.device(),
                "compressor_v2: metadata must live on the same NPU device");
  const auto x_spec = MakeSpec(x), kv_spec = MakeSpec(wkv),
             gate_spec = MakeSpec(wgate), cache_spec = MakeSpec(cache),
             table_spec = MakeSpec(table);
  const auto cu_spec =
      cu_seqlens.has_value() ? MakeSpec(*cu_seqlens) : host::TensorSpec{};
  const auto used_spec =
      seqused.has_value() ? MakeSpec(*seqused) : host::TensorSpec{};
  const auto start_spec =
      start_pos.has_value() ? MakeSpec(*start_pos) : host::TensorSpec{};
  const host::Inputs inputs{
      x_spec, kv_spec, gate_spec, cache_spec, cmp_ratio, &table_spec,
      cu_seqlens.has_value() ? &cu_spec : nullptr,
      seqused.has_value() ? &used_spec : nullptr,
      start_pos.has_value() ? &start_spec : nullptr};
  int64_t cube_cores = 0, vector_cores = 0;
  TORCH_CHECK(aclrtGetDeviceInfo(x.device().index(), ACL_DEV_ATTR_CUBE_CORE_NUM,
                                 &cube_cores) == ACL_SUCCESS,
              "compressor_v2: aclrtGetDeviceInfo failed for cube core count");
  TORCH_CHECK(aclrtGetDeviceInfo(
                  x.device().index(), ACL_DEV_ATTR_VECTOR_CORE_NUM,
                  &vector_cores) == ACL_SUCCESS,
              "compressor_v2: aclrtGetDeviceInfo failed for vector cores");
  host::Plan plan;
  const auto status = host::BuildPlan(inputs, cube_cores, vector_cores, plan);
  TORCH_CHECK(static_cast<bool>(status),
              "compressor_v2: invalid inputs: ", status.error);
  auto out = at::empty(std::vector<int64_t>(plan.output_shape.begin(),
                                            plan.output_shape.begin() +
                                                plan.output_rank),
                       x.options());
  if (plan.route_count == 0)
    return out;
  const int64_t projected_size =
      plan.route_count == 2 ? static_cast<int64_t>(plan.projected_elements) : 1;
  const int64_t history_size =
      plan.route_count == 2 ? static_cast<int64_t>(plan.history_elements) : 1;
  auto projected = at::empty({projected_size}, x.options().dtype(at::kFloat));
  auto history = at::empty({history_size}, x.options().dtype(at::kFloat));
  const auto &cu_tensor = cu_seqlens.has_value() ? *cu_seqlens : table;
  const auto &used_tensor = seqused.has_value() ? *seqused : table;
  const auto &start_tensor = start_pos.has_value() ? *start_pos : table;
  const std::vector<at::Tensor> tensors = {
      x,    wkv,   wgate,     cache,      table,    cu_tensor, used_tensor,
      start_tensor, projected, history, out,        cache,     projected,
      history};
  std::vector<gym_torch::TensorInfo> info;
  info.reserve(tensors.size());
  for (const auto &value : tensors)
    info.push_back(gym_torch::TensorInfo(value));
  std::array<gym_torch::TensorHandle, 14> handles;
  for (size_t i = 0; i < handles.size(); ++i)
    handles[i] = gym_torch::Tensor(info[i]);
  const auto stream = c10_npu::getCurrentNPUStream(x.device().index());
  for (int64_t route = 0; route < plan.route_count; ++route)
    RunStage(plan.keys[route].Stage, cmp_ratio, seqused.has_value(),
             start_pos.has_value(), handles, stream);
  return out;
}

static at::Tensor Meta(const at::Tensor &x, const at::Tensor &wkv,
                       const at::Tensor &wgate, const at::Tensor &cache,
                       const at::Tensor &table,
                       const c10::optional<at::Tensor> &cu_seqlens,
                       const c10::optional<at::Tensor> &seqused,
                       const c10::optional<at::Tensor> &start_pos,
                       int64_t cmp_ratio) {
  const auto tokens = x.sym_numel() / x.sym_size(-1);
  if (x.dim() == 3) {
    const auto sequence = x.sym_size(1);
    const auto dim = x.sym_size(2);
    return at::empty_symint(
        {x.sym_size(0), (sequence + cmp_ratio - 1) / cmp_ratio, dim},
        x.options());
  }
  const auto dim = x.sym_size(-1);
  const auto batches = cache.sym_size(0);
  return at::empty_symint({tokens.min(tokens / cmp_ratio + batches), dim},
                          x.options());
}
} // namespace

TORCH_LIBRARY_FRAGMENT(pypto_gym, m) {
  m.def("compressor_v2(Tensor x, Tensor wkv, Tensor wgate, Tensor cache, "
        "Tensor table, Tensor? cu_seqlens=None, Tensor? seqused=None, "
        "Tensor? start_pos=None, int cmp_ratio=4) -> Tensor");
}
TORCH_LIBRARY_IMPL(pypto_gym, PrivateUse1, m) {
  m.impl("compressor_v2", TORCH_FN(Run));
}
TORCH_LIBRARY_IMPL(pypto_gym, Meta, m) {
  m.impl("compressor_v2", TORCH_FN(Meta));
}
