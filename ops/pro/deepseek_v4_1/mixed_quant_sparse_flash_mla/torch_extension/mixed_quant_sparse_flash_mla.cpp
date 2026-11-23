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
#include <string>
#include <torch/library.h>
#include <tuple>

namespace {
using GetWorkspace = aclnnStatus (*)(const aclTensor *, const aclTensor *,
                                     const aclTensor *, const aclTensor *,
                                     const aclIntArray *, const aclTensor *,
                                     const aclTensor *, const aclTensor *,
                                     const aclTensor *, const aclIntArray *,
                                     const aclIntArray *, const aclTensor *,
                                     char *, char *, double, const aclTensor *,
                                     const aclTensor *, uint64_t *,
                                     aclOpExecutor **);

static std::vector<int64_t> HostInt64Values(const at::Tensor &tensor) {
  auto cpu = tensor.to(at::kCPU, at::kLong).contiguous();
  return std::vector<int64_t>(cpu.data_ptr<int64_t>(),
                              cpu.data_ptr<int64_t>() + cpu.numel());
}

static std::tuple<at::Tensor, at::Tensor> Run(
    const at::Tensor &q, const at::Tensor &ori_kv,
    const at::Tensor &ori_sparse_indices, const at::Tensor &ori_block_table,
    const at::Tensor &cu_seqlens_q, const at::Tensor &sinks,
    const at::Tensor &metadata, double softmax_scale,
    const std::string &layout_q, const std::string &layout_kv,
    const c10::optional<at::Tensor> &cmp_kv,
    const c10::optional<at::Tensor> &cmp_sparse_indices,
    const c10::optional<at::Tensor> &cmp_block_table,
    const c10::optional<at::Tensor> &ori_topk_length,
    const c10::optional<at::Tensor> &cmp_topk_length,
    const c10::optional<at::Tensor> &out, const c10::optional<at::Tensor> &lse) {
  TORCH_CHECK(layout_q == "TND" && layout_kv == "PA_BBND",
              "mixed_quant_sparse_flash_mla: only layout_q='TND' and "
              "layout_kv='PA_BBND' are supported");
  for (const auto &value : {q, ori_kv, ori_sparse_indices, ori_block_table,
                            cu_seqlens_q, sinks, metadata})
    TORCH_CHECK(value.device() == q.device(),
                "mixed_quant_sparse_flash_mla: all tensors must live on the "
                "same NPU device");
  const c10::OptionalDeviceGuard guard(q.device());
  static const auto get_workspace = gym_torch::Symbol<GetWorkspace>(
      "aclnnMixedQuantSparseFlashMlaGetWorkspaceSize");
  static const auto execute = gym_torch::Symbol<gym_torch::Execute>(
      "aclnnMixedQuantSparseFlashMla");
  auto out_tensor = out.has_value() && out->defined()
                        ? *out
                        : at::empty(q.sizes(), q.options().dtype(at::kBFloat16));
  auto lse_tensor =
      lse.has_value() && lse->defined()
          ? *lse
          : at::empty({1, q.size(0), q.size(1)},
                      q.options().dtype(at::kFloat));
  // ValueDepend int32 inputs ride as aclIntArray int64 host values.
  auto ori_length_values = ori_topk_length.has_value()
                               ? HostInt64Values(*ori_topk_length)
                               : std::vector<int64_t>{};
  auto cmp_length_values = cmp_topk_length.has_value()
                                ? HostInt64Values(*cmp_topk_length)
                                : std::vector<int64_t>{};
  auto cu_values = HostInt64Values(cu_seqlens_q);
  auto ori_length = ori_length_values.empty()
                        ? gym_torch::IntArrayHandle{}
                        : gym_torch::IntArray(ori_length_values);
  auto cmp_length = cmp_length_values.empty()
                        ? gym_torch::IntArrayHandle{}
                        : gym_torch::IntArray(cmp_length_values);
  auto cu_array = gym_torch::IntArray(cu_values);
  const bool has_cmp =
      cmp_kv.has_value() && cmp_kv->defined() && cmp_sparse_indices.has_value() &&
      cmp_sparse_indices->defined() && cmp_block_table.has_value() &&
      cmp_block_table->defined();
  TORCH_CHECK(!has_cmp || (cmp_kv->device() == q.device() &&
                           cmp_sparse_indices->device() == q.device() &&
                           cmp_block_table->device() == q.device()),
              "mixed_quant_sparse_flash_mla: compression tensors must live on "
              "the same NPU device");
  const std::vector<at::Tensor> tensors = {
      q,            ori_kv,           ori_sparse_indices, ori_block_table,
      sinks,        metadata,         out_tensor,         lse_tensor,
      has_cmp ? *cmp_kv : q, // placeholder slot; consumed via handles below
      has_cmp ? *cmp_sparse_indices : q,
      has_cmp ? *cmp_block_table : q};
  std::vector<gym_torch::TensorInfo> info;
  info.reserve(tensors.size());
  for (const auto &value : tensors)
    info.push_back(gym_torch::TensorInfo(value));
  std::vector<gym_torch::TensorHandle> handles;
  handles.reserve(info.size());
  for (const auto &value : info)
    handles.push_back(gym_torch::Tensor(value));
  uint64_t bytes = 0;
  aclOpExecutor *executor = nullptr;
  const auto status = get_workspace(
      handles[0].get(), handles[1].get(), handles[2].get(), handles[3].get(),
      ori_length.get(), handles[4].get(),
      has_cmp ? handles[8].get() : nullptr,
      has_cmp ? handles[9].get() : nullptr,
      has_cmp ? handles[10].get() : nullptr, cmp_length.get(),
      cu_array.get(), handles[5].get(), const_cast<char *>(layout_q.c_str()),
      const_cast<char *>(layout_kv.c_str()), softmax_scale, handles[6].get(),
      handles[7].get(), &bytes, &executor);
  TORCH_CHECK(status == 0,
              "aclnnMixedQuantSparseFlashMlaGetWorkspaceSize failed, "
              "status=",
              status, ": ", aclGetRecentErrMsg());
  const auto stream = c10_npu::getCurrentNPUStream(q.device().index());
  gym_torch::ExecuteOp(execute, bytes, executor, stream);
  return {out_tensor, lse_tensor};
}

static std::tuple<at::Tensor, at::Tensor> Meta(
    const at::Tensor &q, const at::Tensor &ori_kv,
    const at::Tensor &ori_sparse_indices, const at::Tensor &ori_block_table,
    const at::Tensor &cu_seqlens_q, const at::Tensor &sinks,
    const at::Tensor &metadata, double softmax_scale,
    const std::string &layout_q, const std::string &layout_kv,
    const c10::optional<at::Tensor> &cmp_kv,
    const c10::optional<at::Tensor> &cmp_sparse_indices,
    const c10::optional<at::Tensor> &cmp_block_table,
    const c10::optional<at::Tensor> &ori_topk_length,
    const c10::optional<at::Tensor> &cmp_topk_length,
    const c10::optional<at::Tensor> &out, const c10::optional<at::Tensor> &lse) {
  auto out_tensor = at::empty_symint(q.sym_sizes(), q.options().dtype(at::kBFloat16));
  auto lse_tensor = at::empty_symint({1, q.sym_size(0), q.sym_size(1)},
                                      q.options().dtype(at::kFloat));
  return {out_tensor, lse_tensor};
}
} // namespace

TORCH_LIBRARY_FRAGMENT(pypto_gym, m) {
  m.def("mixed_quant_sparse_flash_mla(Tensor q, Tensor ori_kv, Tensor "
        "ori_sparse_indices, Tensor ori_block_table, Tensor cu_seqlens_q, "
        "Tensor sinks, Tensor metadata, float softmax_scale, str layout_q, "
        "str layout_kv, Tensor? cmp_kv=None, Tensor? cmp_sparse_indices=None, "
        "Tensor? cmp_block_table=None, Tensor? ori_topk_length=None, "
        "Tensor? cmp_topk_length=None, Tensor? out=None, Tensor? lse=None) "
        "-> (Tensor out, Tensor lse)");
}
TORCH_LIBRARY_IMPL(pypto_gym, PrivateUse1, m) {
  m.impl("mixed_quant_sparse_flash_mla", TORCH_FN(Run));
}
TORCH_LIBRARY_IMPL(pypto_gym, Meta, m) {
  m.impl("mixed_quant_sparse_flash_mla", TORCH_FN(Meta));
}
