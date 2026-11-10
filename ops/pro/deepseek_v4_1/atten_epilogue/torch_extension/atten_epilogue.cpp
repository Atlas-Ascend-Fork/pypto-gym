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
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <vector>

namespace {
// Fixed operator geometry (op_host registration; see the operator README).
constexpr int64_t kHeads = 64;
constexpr int64_t kHeadDim = 512;
constexpr int64_t kGroups = 8;
constexpr int64_t kGroupN = 1024;
constexpr int64_t kGemm1K = 4096;
constexpr int64_t kGemm2K = 8192;
constexpr int64_t kOutFeatures = 5120;

struct Spec {
  const char *name;
  at::ScalarType dtype;
  std::vector<int64_t> shape;
};

static void ValidateInputs(const at::Tensor &o,
                           const std::vector<at::Tensor> &inputs) {
  const int64_t t = o.size(0);
  const Spec specs[] = {
      {"o", at::kBFloat16, {t, kHeads, kHeadDim}},
      {"woa", at::kFloat8_e4m3fn, {kGroups, kGroupN, kGemm1K}},
      {"wob", at::kFloat8_e4m3fn, {kOutFeatures, kGemm2K}},
      {"descale_woa", at::kFloat8_e8m0fnu, {kGroups, kGroupN, kGemm1K / 64, 2}},
      {"descale_wob", at::kFloat8_e8m0fnu, {kOutFeatures, kGemm2K / 64, 2}},
      {"rope_sin", at::kFloat, {t, kHeads}},
      {"rope_cos", at::kFloat, {t, kHeads}},
  };
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &spec = specs[i];
    TORCH_CHECK(inputs[i].device() == o.device() && inputs[i].is_contiguous() &&
                    inputs[i].scalar_type() == spec.dtype &&
                    inputs[i].sizes().vec() == spec.shape,
                "invalid ", spec.name,
                ": expected the registered shape/dtype, contiguous, on the "
                "same NPU device as o");
    gym_torch::CheckPlainStorage(inputs[i], spec.name);
  }
}

static void CheckNoGrad(const std::vector<at::Tensor> &inputs) {
  for (const auto &value : inputs)
    TORCH_CHECK(!at::GradMode::is_enabled() || !value.requires_grad(),
                "AttenEpilogue provides forward computation only; use "
                "torch.no_grad() for inference");
}

static void EnqueueAttenEpilogue(const std::vector<at::Tensor> &tensors) {
  using GetWorkspace = aclnnStatus (*)(
      const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
      const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
      uint64_t *, aclOpExecutor **);
  static const auto get_workspace =
      gym_torch::Symbol<GetWorkspace>("aclnnAttenEpilogueGetWorkspaceSize");
  static const auto execute =
      gym_torch::Symbol<gym_torch::Execute>("aclnnAttenEpilogue");
  gym_torch::Enqueue(
      "aclnnAttenEpilogue", tensors,
      [](const std::vector<gym_torch::TensorInfo> &values, auto stream) {
        std::vector<gym_torch::TensorHandle> handles;
        handles.reserve(values.size());
        for (const auto &value : values)
          handles.push_back(gym_torch::Tensor(value));
        uint64_t bytes = 0;
        aclOpExecutor *executor = nullptr;
        const auto status = get_workspace(
            handles[0].get(), handles[1].get(), handles[2].get(),
            handles[3].get(), handles[4].get(), handles[5].get(),
            handles[6].get(), handles[7].get(), &bytes, &executor);
        TORCH_CHECK(status == 0,
                    "aclnnAttenEpilogueGetWorkspaceSize failed, status=", status,
                    ": ", aclGetRecentErrMsg());
        gym_torch::ExecuteOp(execute, bytes, executor, stream);
        return 0;
      });
}

static at::Tensor Run(const at::Tensor &o, const at::Tensor &woa,
                      const at::Tensor &wob, const at::Tensor &descale_woa,
                      const at::Tensor &descale_wob, const at::Tensor &rope_sin,
                      const at::Tensor &rope_cos) {
  TORCH_CHECK(o.dim() == 3 && o.scalar_type() == at::kBFloat16 &&
                  o.device().is_privateuseone(),
              "o must be a rank-3 BF16 NPU tensor [T,64,512]");
  TORCH_CHECK(o.size(0) >= 1, "T must be positive");
  TORCH_CHECK(o.size(1) == kHeads && o.size(2) == kHeadDim,
              "o must have shape [T,", kHeads, ",", kHeadDim, "]");
  const c10::OptionalDeviceGuard guard(o.device());
  std::vector<at::Tensor> inputs = {o,          woa,         wob, descale_woa,
                                    descale_wob, rope_sin,    rope_cos};
  ValidateInputs(o, inputs);
  CheckNoGrad(inputs);
  auto out = at::empty({o.size(0), kOutFeatures},
                       o.options().dtype(at::kBFloat16));
  gym_torch::CheckPlainStorage(out, "out");
  inputs.push_back(out);
  EnqueueAttenEpilogue(inputs);
  return out;
}

static at::Tensor Meta(const at::Tensor &o, const at::Tensor &woa,
                       const at::Tensor &wob, const at::Tensor &descale_woa,
                       const at::Tensor &descale_wob, const at::Tensor &rope_sin,
                       const at::Tensor &rope_cos) {
  TORCH_CHECK(o.dim() == 3 && o.scalar_type() == at::kBFloat16,
              "o must be a rank-3 BF16 tensor [T,64,512]");
  TORCH_CHECK(o.size(1) == kHeads && o.size(2) == kHeadDim,
              "o must have shape [T,", kHeads, ",", kHeadDim, "]");
  return at::empty_symint({o.sym_size(0), kOutFeatures},
                          o.options().dtype(at::kBFloat16));
}
} // namespace

TORCH_LIBRARY_FRAGMENT(pypto_gym, m) {
  m.def("atten_epilogue(Tensor o, Tensor woa, Tensor wob, Tensor descale_woa, "
        "Tensor descale_wob, Tensor rope_sin, Tensor rope_cos) -> Tensor");
}
TORCH_LIBRARY_IMPL(pypto_gym, PrivateUse1, m) {
  m.impl("atten_epilogue", TORCH_FN(Run));
}
TORCH_LIBRARY_IMPL(pypto_gym, Meta, m) {
  m.impl("atten_epilogue", TORCH_FN(Meta));
}
