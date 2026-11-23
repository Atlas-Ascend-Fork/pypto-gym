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
#include <torch/library.h>
#include <tuple>

namespace {
using GetWorkspace = aclnnStatus (*)(const aclTensor *, const aclTensor *,
                                     const aclTensor *, const aclTensor *,
                                     const aclTensor *, const aclTensor *,
                                     const aclTensor *, const aclTensor *,
                                     const aclTensor *, const aclTensor *,
                                     const aclTensor *, uint64_t *,
                                     aclOpExecutor **);

static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
Run(const at::Tensor &grad_out, const at::Tensor &x, const at::Tensor &key,
    const at::Tensor &value, const at::Tensor &weight,
    const at::Tensor &rstd_x, const at::Tensor &rstd_k,
    const at::Tensor &raw_dot, const at::Tensor &gate,
    const at::Tensor &image_mask) {
  const int64_t t = x.size(0);
  const int64_t hc = x.size(1);
  const int64_t dim = x.size(2);
  for (const auto &value :
       {grad_out, x, key, weight, rstd_x, rstd_k, raw_dot, gate, image_mask})
    TORCH_CHECK(value.device() == x.device(),
                "engram_gate_bwd: all tensors must live on the same NPU device");
  const c10::OptionalDeviceGuard guard(x.device());
  // [raw_dot | gate | rstd_x | rstd_k], the layout the kernel expects.
  auto saved = at::cat({raw_dot.to(at::kFloat), gate.to(at::kFloat),
                        rstd_x.to(at::kFloat), rstd_k.to(at::kFloat)}, 1)
                   .contiguous();
  auto grad_x = at::empty({t, hc, dim}, x.options());
  auto grad_key = at::empty({t, hc, dim}, key.options());
  auto grad_value = at::empty({t, dim}, value.options());
  // Zeroed: the kernel writes only (AIV / hc) rows; the rest must contribute
  // nothing to the sum(0) reduction.
  auto grad_w_partial = at::zeros(
      {16, hc, dim},
      at::TensorOptions().dtype(at::kFloat).device(x.device()));
  static const auto get_workspace =
      gym_torch::Symbol<GetWorkspace>("aclnnEngramGateBwdGetWorkspaceSize");
  static const auto execute =
      gym_torch::Symbol<gym_torch::Execute>("aclnnEngramGateBwd");
  const std::vector<at::Tensor> tensors = {grad_out, x,    key,    value,
                                           weight,    saved, image_mask,
                                           grad_x,    grad_key, grad_value,
                                           grad_w_partial};
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
      handles[4].get(), handles[5].get(), handles[6].get(), handles[7].get(),
      handles[8].get(), handles[9].get(), handles[10].get(), &bytes,
      &executor);
  TORCH_CHECK(status == 0,
              "aclnnEngramGateBwdGetWorkspaceSize failed, status=", status,
              ": ", aclGetRecentErrMsg());
  const auto stream = c10_npu::getCurrentNPUStream(x.device().index());
  gym_torch::ExecuteOp(execute, bytes, executor, stream);
  // Same-stream reduction; ordered after the accumulator kernels.
  auto grad_weight = grad_w_partial.sum(0);
  return {grad_x, grad_key, grad_value, grad_weight};
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
Meta(const at::Tensor &grad_out, const at::Tensor &x, const at::Tensor &key,
     const at::Tensor &value, const at::Tensor &weight,
     const at::Tensor &rstd_x, const at::Tensor &rstd_k,
     const at::Tensor &raw_dot, const at::Tensor &gate,
     const at::Tensor &image_mask) {
  auto grad_x = at::empty_symint(x.sym_sizes(), x.options());
  auto grad_key = at::empty_symint(key.sym_sizes(), key.options());
  auto grad_value = at::empty_symint(value.sym_sizes(), value.options());
  auto grad_weight =
      at::empty_symint(weight.sym_sizes(), weight.options().dtype(at::kFloat));
  return {grad_x, grad_key, grad_value, grad_weight};
}
} // namespace

TORCH_LIBRARY_FRAGMENT(pypto_gym, m) {
  m.def("engram_gate_bwd(Tensor grad_out, Tensor x, Tensor key, Tensor value, "
        "Tensor weight, Tensor rstd_x, Tensor rstd_k, Tensor raw_dot, "
        "Tensor gate, Tensor image_mask) -> (Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(pypto_gym, PrivateUse1, m) {
  m.impl("engram_gate_bwd", TORCH_FN(Run));
}
TORCH_LIBRARY_IMPL(pypto_gym, Meta, m) {
  m.impl("engram_gate_bwd", TORCH_FN(Meta));
}
