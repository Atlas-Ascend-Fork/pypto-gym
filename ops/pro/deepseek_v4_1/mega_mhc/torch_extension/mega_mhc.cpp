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
#include <cmath>
#include <limits>
#include <torch/library.h>
#include <tuple>

namespace {
using Result =
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
static void ValidateInputs(const at::Tensor &x,
                           const std::vector<at::Tensor> &inputs) {
  const int64_t t = x.size(0), h = x.size(1);
  const char *const names[] = {"x",          "residual",     "previous_mix",
                               "post_mix",   "comb_res_mix", "fn",
                               "mix_scales", "mix_bases",    "rmsnorm_weight"};
  const std::vector<std::vector<int64_t>> shapes = {
      {t, h},      {t, 4, h}, {t, 4}, {t, 4, 1}, {t, 4, 4},
      {24, 4 * h}, {3},       {24},   {h}};
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto dtype =
        (i == 0 || i == 1 || i == 8) ? at::kBFloat16 : at::kFloat;
    TORCH_CHECK(inputs[i].device() == x.device() && inputs[i].is_contiguous() &&
                    inputs[i].scalar_type() == dtype &&
                    inputs[i].sizes().vec() == shapes[i],
                "invalid input ", i);
    gym_torch::CheckNDStorage(inputs[i], names[i]);
  }
}

static std::vector<at::Tensor> AllocateOutputs(
    const at::Tensor &x, const std::vector<std::vector<int64_t>> &outShapes,
    const char *const outputNames[]) {
  std::vector<at::Tensor> outputs;
  outputs.reserve(5);
  for (size_t i = 0; i < 5; ++i)
    outputs.push_back(at::empty(
        outShapes[i],
        x.options().dtype((i == 0 || i == 3) ? at::kBFloat16 : at::kFloat)));
  for (size_t i = 0; i < 5; ++i) {
    TORCH_CHECK(outputs[i].device() == x.device() &&
                    outputs[i].is_contiguous() &&
                    outputs[i].sizes().vec() == outShapes[i],
                "invalid output ", i);
    gym_torch::CheckNDStorage(outputs[i], outputNames[i]);
  }
  return outputs;
}

static void CheckNoGrad(const std::vector<at::Tensor> &inputs) {
  for (const auto &value : inputs)
    TORCH_CHECK(!at::GradMode::is_enabled() || !value.requires_grad(),
                "MegaMhc provides forward computation only; use "
                "torch.no_grad() for inference");
}

static void EnqueueMegaMhc(const std::vector<at::Tensor> &inputs,
                           const std::vector<at::Tensor> &outputs, double hc,
                           double pre, double post, double sink,
                           int64_t repeat, double norm, int64_t batch_attr) {
  using GetWorkspace = aclnnStatus (*)(
      const aclTensor *, const aclTensor *, const aclTensor *,
      const aclTensor *, const aclTensor *, const aclTensor *,
      const aclTensor *, const aclTensor *, const aclTensor *, double, double,
      double, double, int64_t, double, int64_t, const aclTensor *,
      const aclTensor *, const aclTensor *, const aclTensor *,
      const aclTensor *, uint64_t *, aclOpExecutor **);
  static const auto get_workspace =
      gym_torch::Symbol<GetWorkspace>("aclnnMegaMhcGetWorkspaceSize");
  static const auto execute =
      gym_torch::Symbol<gym_torch::Execute>("aclnnMegaMhc");
  auto tensors = inputs;
  tensors.insert(tensors.end(), outputs.begin(), outputs.end());
  gym_torch::Enqueue(
      "aclnnMegaMhc", tensors,
      [hc, pre, post, sink, repeat, norm, batch_attr](const auto &values,
                                                      auto stream) {
        std::vector<gym_torch::TensorHandle> handles;
        for (const auto &value : values)
          handles.push_back(gym_torch::Tensor(value));
        uint64_t bytes = 0;
        aclOpExecutor *executor = nullptr;
        const auto status = get_workspace(
            handles[0].get(), handles[1].get(), handles[2].get(),
            handles[3].get(), handles[4].get(), handles[5].get(),
            handles[6].get(), handles[7].get(), handles[8].get(), hc, pre, post,
            sink, repeat, norm, batch_attr, handles[9].get(), handles[10].get(),
            handles[11].get(), handles[12].get(), handles[13].get(), &bytes,
            &executor);
        TORCH_CHECK(status == 0,
                    "aclnnMegaMhcGetWorkspaceSize failed, status=", status,
                    ": ", aclGetRecentErrMsg());
        gym_torch::ExecuteOp(execute, bytes, executor, stream);
        return 0;
      });
}

static std::vector<at::Tensor>
Run(at::Tensor x, at::Tensor residual, at::Tensor previous_mix,
    at::Tensor post_mix, at::Tensor comb_res_mix, at::Tensor fn,
    at::Tensor mix_scales, at::Tensor mix_bases, at::Tensor rmsnorm_weight,
    double hc, double pre, double post, double sink, int64_t repeat,
    double norm, c10::optional<int64_t> batch) {
  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kBFloat16 &&
                  x.device().is_privateuseone(),
              "x must be a rank-2 BF16 NPU tensor");
  const c10::OptionalDeviceGuard guard(x.device());
  const int64_t t = x.size(0), h = x.size(1);
  TORCH_CHECK(h >= 1024 && h <= 7168 && h % 1024 == 0, "unsupported H");
  TORCH_CHECK(!batch.has_value() || *batch == 0 || *batch == 16,
              "batch must be 0 (automatic) or 16");
  for (double v : {hc, pre, post, sink, norm})
    TORCH_CHECK(std::isfinite(v) && v >= 0 &&
                    v <= std::numeric_limits<float>::max(),
                "invalid scalar");
  TORCH_CHECK(hc > 0 && norm > 0 && repeat >= 1 && repeat <= 65535,
              "invalid RMS epsilon/iterations");
  std::vector<at::Tensor> inputs = {x,          residual,     previous_mix,
                                    post_mix,   comb_res_mix, fn,
                                    mix_scales, mix_bases,    rmsnorm_weight};
  ValidateInputs(x, inputs);
  const std::vector<std::vector<int64_t>> outShapes = {
      {t, 4, h}, {t, 4, 1}, {t, 4, 4}, {t, h}, {t, 4}};
  const char *const outputNames[] = {
      "new_residual", "new_post_mix", "new_comb_res_mix", "y", "new_previous_mix"};
  auto outputs = AllocateOutputs(x, outShapes, outputNames);
  CheckNoGrad(inputs);
  if (t == 0) return outputs;
  EnqueueMegaMhc(inputs, outputs, hc, pre, post, sink, repeat, norm,
                 batch.value_or(0));
  return outputs;
}

static Result AsTuple(const std::vector<at::Tensor> &outputs) {
  return {outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]};
}
static Result MegaMhc(const at::Tensor &x, const at::Tensor &residual,
                      const at::Tensor &previous_mix,
                      const at::Tensor &post_mix,
                      const at::Tensor &comb_res_mix, const at::Tensor &fn,
                      const at::Tensor &mix_scales, const at::Tensor &mix_bases,
                      const at::Tensor &rmsnorm_weight, double hc, double pre,
                      double post, double sink, int64_t repeat, double norm,
                      int64_t batch) {
  return AsTuple(Run(x, residual, previous_mix, post_mix, comb_res_mix, fn,
                     mix_scales, mix_bases, rmsnorm_weight, hc, pre, post, sink,
                     repeat, norm, batch));
}

static Result Meta(const at::Tensor &x, const at::Tensor &residual,
                   const at::Tensor &previous_mix, const at::Tensor &post_mix,
                   const at::Tensor &comb_res_mix, const at::Tensor &fn,
                   const at::Tensor &mix_scales, const at::Tensor &mix_bases,
                   const at::Tensor &rmsnorm_weight, double hc, double pre,
                   double post, double sink, int64_t repeat, double norm,
                   int64_t batch) {
  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kBFloat16,
              "x must be a rank-2 BF16 tensor");
  const auto t = x.sym_size(0), h = x.sym_size(1);
  auto bf16 = x.options().dtype(at::kBFloat16);
  auto fp32 = x.options().dtype(at::kFloat);
  return {at::empty_symint({t, 4, h}, bf16), at::empty_symint({t, 4, 1}, fp32),
          at::empty_symint({t, 4, 4}, fp32), at::empty_symint({t, h}, bf16),
          at::empty_symint({t, 4}, fp32)};
}

} // namespace

TORCH_LIBRARY_FRAGMENT(pypto_gym, m) {
  m.def("mega_mhc(Tensor x, Tensor residual, Tensor previous_mix, Tensor "
        "post_mix, Tensor comb_res_mix, Tensor fn, Tensor mix_scales, Tensor "
        "mix_bases, Tensor rmsnorm_weight, *, float hc_norm_eps=2e-5, float "
        "hc_pre_eps=3e-4, float hc_post_scale=1.25, float sinkhorn_eps=2e-6, "
        "int num_sinkhorn_iters=10, float rmsnorm_eps=7e-6, int batch=0) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(pypto_gym, PrivateUse1, m) {
  m.impl("mega_mhc", TORCH_FN(MegaMhc));
}
TORCH_LIBRARY_IMPL(pypto_gym, Meta, m) {
  m.impl("mega_mhc", TORCH_FN(Meta));
}
