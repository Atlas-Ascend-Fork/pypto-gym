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
#include <cmath>
#include <torch/library.h>

namespace {
using GetWorkspace = aclnnStatus (*)(const aclTensor *, const aclTensor *,
                                     const aclTensor *, const aclTensor *,
                                     const aclTensor *, const aclTensor *,
                                     uint64_t *, aclOpExecutor **);

static at::Tensor Run(const at::Tensor &x, const at::Tensor &key,
                      const at::Tensor &value, const at::Tensor &weight,
                      const at::Tensor &image_mask) {
  const int64_t t = x.size(0);
  const int64_t hc = x.size(1);
  const int64_t dim = x.size(2);
  TORCH_CHECK(x.dim() == 3 && key.dim() == 3 && value.dim() == 2 &&
                  weight.dim() == 2 && image_mask.dim() == 1,
              "engram_gate: unexpected input ranks");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 &&
                  key.scalar_type() == at::kBFloat16 &&
                  value.scalar_type() == at::kBFloat16,
              "engram_gate: x/key/value must be bfloat16");
  TORCH_CHECK(weight.scalar_type() == at::kFloat,
              "engram_gate: weight must be float32");
  TORCH_CHECK(image_mask.scalar_type() == at::kBool,
              "engram_gate: image_mask must be bool");
  for (const auto &value : {x, key, weight, image_mask})
    TORCH_CHECK(value.device() == x.device(),
                "engram_gate: all tensors must live on the same NPU device");
  const c10::OptionalDeviceGuard guard(x.device());
  static const auto get_workspace =
      gym_torch::Symbol<GetWorkspace>("aclnnEngramGateGetWorkspaceSize");
  static const auto execute =
      gym_torch::Symbol<gym_torch::Execute>("aclnnEngramGate");
  auto out = at::empty({t, hc, dim}, x.options());
  const std::vector<at::Tensor> tensors = {x, key, value, weight, image_mask,
                                           out};
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
      handles[4].get(), handles[5].get(), &bytes, &executor);
  TORCH_CHECK(status == 0, "aclnnEngramGateGetWorkspaceSize failed, status=",
              status, ": ", aclGetRecentErrMsg());
  const auto stream = c10_npu::getCurrentNPUStream(x.device().index());
  gym_torch::ExecuteOp(execute, bytes, executor, stream);
  return out;
}

static at::Tensor Meta(const at::Tensor &x, const at::Tensor &key,
                       const at::Tensor &value, const at::Tensor &weight,
                       const at::Tensor &image_mask) {
  return at::empty_symint(x.sym_sizes(), x.options());
}
} // namespace

TORCH_LIBRARY_FRAGMENT(pypto_gym, m) {
  m.def("engram_gate(Tensor x, Tensor key, Tensor value, Tensor weight, "
        "Tensor image_mask) -> Tensor");
}
TORCH_LIBRARY_IMPL(pypto_gym, PrivateUse1, m) {
  m.impl("engram_gate", TORCH_FN(Run));
}
TORCH_LIBRARY_IMPL(pypto_gym, Meta, m) {
  m.impl("engram_gate", TORCH_FN(Meta));
}
