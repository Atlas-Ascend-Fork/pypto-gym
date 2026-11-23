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

namespace {
using GetWorkspace = aclnnStatus (*)(const aclTensor *, const aclTensor *,
                                     const aclTensor *, const aclTensor *,
                                     const aclTensor *, uint64_t *,
                                     aclOpExecutor **);

static at::Tensor Run(const at::Tensor &x, const at::Tensor &residual,
                      const at::Tensor &post_layer_mix,
                      const at::Tensor &comb_res_mix, const at::Tensor &out) {
  TORCH_CHECK(out.scalar_type() == at::kBFloat16,
              "mhc_post_fwd: out must be bfloat16");
  TORCH_CHECK(out.sizes() == residual.sizes(),
              "mhc_post_fwd: out shape must equal residual shape");
  TORCH_CHECK(out.is_contiguous(), "mhc_post_fwd: out must be contiguous");
  for (const auto &value : {x, residual, post_layer_mix, comb_res_mix, out})
    TORCH_CHECK(value.device() == residual.device(),
                "mhc_post_fwd: all tensors must live on the same NPU device");
  const c10::OptionalDeviceGuard guard(residual.device());
  static const auto get_workspace =
      gym_torch::Symbol<GetWorkspace>("aclnnMhcPostFwdGetWorkspaceSize");
  static const auto execute =
      gym_torch::Symbol<gym_torch::Execute>("aclnnMhcPostFwd");
  const std::vector<at::Tensor> tensors = {x, residual, post_layer_mix,
                                           comb_res_mix, out};
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
      handles[4].get(), &bytes, &executor);
  TORCH_CHECK(status == 0, "aclnnMhcPostFwdGetWorkspaceSize failed, status=",
              status, ": ", aclGetRecentErrMsg());
  const auto stream = c10_npu::getCurrentNPUStream(residual.device().index());
  gym_torch::ExecuteOp(execute, bytes, executor, stream);
  return out;
}

static at::Tensor Meta(const at::Tensor &x, const at::Tensor &residual,
                       const at::Tensor &post_layer_mix,
                       const at::Tensor &comb_res_mix, const at::Tensor &out) {
  return at::empty_symint(residual.sym_sizes(),
                          residual.options().dtype(at::kBFloat16));
}
} // namespace

TORCH_LIBRARY_FRAGMENT(pypto_gym, m) {
  m.def("mhc_post_fwd(Tensor x, Tensor residual, Tensor post_layer_mix, "
        "Tensor comb_res_mix, Tensor out) -> Tensor");
}
TORCH_LIBRARY_IMPL(pypto_gym, PrivateUse1, m) {
  m.impl("mhc_post_fwd", TORCH_FN(Run));
}
TORCH_LIBRARY_IMPL(pypto_gym, Meta, m) {
  m.impl("mhc_post_fwd", TORCH_FN(Meta));
}
