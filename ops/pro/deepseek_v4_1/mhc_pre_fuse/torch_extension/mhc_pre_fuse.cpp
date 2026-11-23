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
                                     const aclTensor *, double, double, double,
                                     double, int64_t, const aclTensor *,
                                     const aclTensor *, const aclTensor *,
                                     uint64_t *, aclOpExecutor **);

static std::tuple<at::Tensor, at::Tensor, at::Tensor>
Run(const at::Tensor &gemm_out_mul, const at::Tensor &gemm_out_sqrsum,
    const at::Tensor &mhc_scale,
    const at::Tensor &mhc_base,
    const at::Tensor &residual,
    double rms_eps, double mhc_pre_eps,
    double mhc_sinkhorn_eps,
    double mhc_post_mult_value,
    int64_t sinkhorn_repeat) {
  const int64_t t = residual.size(0);
  const int64_t h = residual.size(2);
  for (const auto &value :
       {gemm_out_mul, gemm_out_sqrsum, mhc_scale, mhc_base, residual})
    TORCH_CHECK(value.device() == residual.device(),
                "mhc_pre_fuse: all tensors must live on the same NPU device");
  TORCH_CHECK(sinkhorn_repeat >= 1,
              "mhc_pre_fuse: sinkhorn_repeat must be >= 1, got ",
              sinkhorn_repeat);
  const c10::OptionalDeviceGuard guard(residual.device());
  static const auto get_workspace =
      gym_torch::Symbol<GetWorkspace>("aclnnMhcPreFuseGetWorkspaceSize");
  static const auto execute =
      gym_torch::Symbol<gym_torch::Execute>("aclnnMhcPreFuse");
  auto post_mix = at::empty({t, 4}, residual.options().dtype(at::kFloat));
  auto comb_mix = at::empty({t, 16}, residual.options().dtype(at::kFloat));
  auto layer_input = at::empty({t, h}, residual.options().dtype(at::kBFloat16));
  const std::vector<at::Tensor> tensors = {gemm_out_mul, gemm_out_sqrsum,
                                           mhc_scale,    mhc_base,
                                           residual,     post_mix,
                                           comb_mix,     layer_input};
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
      handles[4].get(), rms_eps, mhc_pre_eps, mhc_sinkhorn_eps,
      mhc_post_mult_value, sinkhorn_repeat, handles[5].get(), handles[6].get(),
      handles[7].get(), &bytes, &executor);
  TORCH_CHECK(status == 0,
              "aclnnMhcPreFuseGetWorkspaceSize failed, status=", status, ": ",
              aclGetRecentErrMsg());
  const auto stream = c10_npu::getCurrentNPUStream(residual.device().index());
  gym_torch::ExecuteOp(execute, bytes, executor, stream);
  return {post_mix, comb_mix, layer_input};
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor>
Meta(const at::Tensor &gemm_out_mul, const at::Tensor &gemm_out_sqrsum,
     const at::Tensor &mhc_scale,
     const at::Tensor &mhc_base,
     const at::Tensor &residual,
     double rms_eps, double mhc_pre_eps,
     double mhc_sinkhorn_eps,
     double mhc_post_mult_value,
     int64_t sinkhorn_repeat) {
  const auto t = residual.sym_size(0);
  const auto h = residual.sym_size(2);
  auto post_mix = at::empty_symint({t, 4}, residual.options().dtype(at::kFloat));
  auto comb_mix = at::empty_symint({t, 16}, residual.options().dtype(at::kFloat));
  auto layer_input =
      at::empty_symint({t, h}, residual.options().dtype(at::kBFloat16));
  return {post_mix, comb_mix, layer_input};
}
} // namespace

TORCH_LIBRARY_FRAGMENT(pypto_gym, m) {
  m.def("mhc_pre_fuse(Tensor gemm_out_mul, Tensor gemm_out_sqrsum, "
        "Tensor mhc_scale, Tensor mhc_base, Tensor residual, "
        "float rms_eps, float mhc_pre_eps, float mhc_sinkhorn_eps, "
        "float mhc_post_mult_value, int sinkhorn_repeat) "
        "-> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(pypto_gym, PrivateUse1, m) {
  m.impl("mhc_pre_fuse", TORCH_FN(Run));
}
TORCH_LIBRARY_IMPL(pypto_gym, Meta, m) {
  m.impl("mhc_pre_fuse", TORCH_FN(Meta));
}
