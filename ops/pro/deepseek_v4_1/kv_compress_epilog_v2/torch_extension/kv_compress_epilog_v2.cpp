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
using GetWorkspace = aclnnStatus (*)(aclTensor *, const aclTensor *,
                                     const aclTensor *, int64_t, int64_t, bool,
                                     double, int64_t, int64_t, uint64_t *,
                                     aclOpExecutor **);

static at::Tensor Run(const at::Tensor &cache, const at::Tensor &x,
                      const at::Tensor &slot_mapping, int64_t quant_group_size,
                      int64_t quant_mode, bool round_scale, double x_scale,
                      int64_t block_stride) {
  TORCH_CHECK(cache.device() == x.device() && slot_mapping.device() == x.device(),
              "kv_compress_epilog_v2: all tensors must live on the same NPU device");
  const c10::OptionalDeviceGuard guard(x.device());
  auto slot64 = slot_mapping.scalar_type() == at::kLong ? slot_mapping
                                                        : slot_mapping.to(at::kLong);
  const int64_t storage_bytes =
      static_cast<int64_t>(cache.storage().nbytes()) -
      cache.storage_offset() * cache.itemsize();
  static const auto get_workspace = gym_torch::Symbol<GetWorkspace>(
      "aclnnKvCompressEpilogV2GetWorkspaceSize");
  static const auto execute =
      gym_torch::Symbol<gym_torch::Execute>("aclnnKvCompressEpilogV2");
  const std::vector<at::Tensor> tensors = {cache, x, slot64};
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
      handles[0].get(), handles[1].get(), handles[2].get(), quant_group_size,
      quant_mode, round_scale, x_scale, block_stride, storage_bytes, &bytes,
      &executor);
  TORCH_CHECK(status == 0,
              "aclnnKvCompressEpilogV2GetWorkspaceSize failed, status=",
              status, ": ", aclGetRecentErrMsg());
  const auto stream = c10_npu::getCurrentNPUStream(x.device().index());
  gym_torch::ExecuteOp(execute, bytes, executor, stream);
  return cache;
}

static at::Tensor Meta(const at::Tensor &cache, const at::Tensor &x,
                       const at::Tensor &slot_mapping, int64_t quant_group_size,
                       int64_t quant_mode, bool round_scale, double x_scale,
                       int64_t block_stride) {
  return at::empty_symint(cache.sym_sizes(), cache.options());
}
} // namespace

TORCH_LIBRARY_FRAGMENT(pypto_gym, m) {
  m.def("kv_compress_epilog_v2(Tensor cache, Tensor x, Tensor slot_mapping, "
        "int quant_group_size, int quant_mode, bool round_scale, "
        "float x_scale, int block_stride) -> Tensor");
}
TORCH_LIBRARY_IMPL(pypto_gym, PrivateUse1, m) {
  m.impl("kv_compress_epilog_v2", TORCH_FN(Run));
}
TORCH_LIBRARY_IMPL(pypto_gym, Meta, m) {
  m.impl("kv_compress_epilog_v2", TORCH_FN(Meta));
}
