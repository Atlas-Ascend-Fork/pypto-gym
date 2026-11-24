/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file mixed_quant_sparse_flash_mla_metadata.cpp
 * \brief Precompiled Torch binding for aclnnMixedQuantSparseFlashMlaMetadata.
 */

#include "gym_aclnn.h"

#include <array>
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <vector>

namespace {

constexpr int64_t kMetadataElements = 1024;

static bool HasValue(const c10::optional<at::Tensor> &value)
{
  return value.has_value() && value->defined();
}

static void CheckInt32(const at::Tensor &tensor, const char *name)
{
  TORCH_CHECK(tensor.device().is_privateuseone(), name, " must be an NPU tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  gym_torch::CheckPlainStorage(tensor, name);
}

static at::Tensor Run(const at::Tensor &cu_seqlens_q,
                      const at::Tensor &ori_topk_length,
                      const c10::optional<at::Tensor> &cmp_topk_length,
                      int64_t blocks, int64_t has_cmp, int64_t k_ori,
                      int64_t k_cmp)
{
  TORCH_CHECK(blocks >= 1 && blocks <= 36, "blocks must be in [1,36]");
  TORCH_CHECK(has_cmp == 0 || has_cmp == 1, "has_cmp must be 0 or 1");
  TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_q.numel() >= 2,
              "cu_seqlens_q must be rank-1 with at least 2 entries");
  CheckInt32(cu_seqlens_q, "cu_seqlens_q");
  CheckInt32(ori_topk_length, "ori_topk_length");
  if (HasValue(cmp_topk_length))
    CheckInt32(*cmp_topk_length, "cmp_topk_length");
  const c10::OptionalDeviceGuard guard(cu_seqlens_q.device());

  auto metadata = at::empty({kMetadataElements}, cu_seqlens_q.options().dtype(at::kInt));
  gym_torch::CheckPlainStorage(metadata, "metadata");

  // Slot order = ACLNN signature: 3 inputs, then the metadata output.
  // Enqueue() only accepts defined tensors, so track which slots are present
  // and scatter the handles back with nullptr holes.
  std::array<bool, 4> present{};
  std::vector<at::Tensor> defined;
  size_t slot = 0;
  const auto add_required = [&](const at::Tensor &value) {
    present[slot++] = true;
    defined.push_back(value);
  };
  const auto add_optional = [&](const c10::optional<at::Tensor> &value) {
    const bool has = HasValue(value);
    present[slot++] = has;
    if (has)
      defined.push_back(*value);
  };
  add_required(cu_seqlens_q);
  add_required(ori_topk_length);
  add_optional(cmp_topk_length);
  add_required(metadata);

  using GetWorkspace = aclnnStatus (*)(
      const aclTensor *, const aclTensor *, const aclTensor *, int64_t, int64_t,
      int64_t, int64_t, const aclTensor *, uint64_t *, aclOpExecutor **);
  static const auto get_workspace = gym_torch::Symbol<GetWorkspace>(
      "aclnnMixedQuantSparseFlashMlaMetadataGetWorkspaceSize");
  static const auto execute = gym_torch::Symbol<gym_torch::Execute>(
      "aclnnMixedQuantSparseFlashMlaMetadata");
  gym_torch::Enqueue(
      "aclnnMixedQuantSparseFlashMlaMetadata", defined,
      [present, blocks, has_cmp, k_ori,
       k_cmp](const std::vector<gym_torch::TensorInfo> &infos, auto stream) {
        std::array<gym_torch::TensorHandle, 4> handles;
        size_t index = 0;
        for (size_t i = 0; i < handles.size(); ++i) {
          if (present[i])
            handles[i] = gym_torch::Tensor(infos[index++]);
        }
        uint64_t bytes = 0;
        aclOpExecutor *executor = nullptr;
        const auto status = get_workspace(
            handles[0].get(), handles[1].get(), handles[2].get(), blocks, has_cmp,
            k_ori, k_cmp, handles[3].get(), &bytes, &executor);
        TORCH_CHECK(status == 0,
                    "aclnnMixedQuantSparseFlashMlaMetadataGetWorkspaceSize "
                    "failed, status=", status, ": ", aclGetRecentErrMsg());
        gym_torch::ExecuteOp(execute, bytes, executor, stream);
        return 0;
      });
  return metadata;
}

} // namespace

TORCH_LIBRARY_FRAGMENT(pypto_gym, m) {
  m.def("mixed_quant_sparse_flash_mla_metadata(Tensor cu_seqlens_q, "
        "Tensor ori_topk_length, Tensor? cmp_topk_length=None, int blocks=28, "
        "int has_cmp=1, int k_ori=0, int k_cmp=0) -> Tensor");
}

TORCH_LIBRARY_IMPL(pypto_gym, PrivateUse1, m) {
  m.impl("mixed_quant_sparse_flash_mla_metadata", TORCH_FN(Run));
}
