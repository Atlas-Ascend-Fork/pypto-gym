/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file mixed_quant_sparse_flash_mla_metadata_pto_bnn.cpp
 * \brief Precompiled Torch binding for aclnnMixedQuantSparseFlashMlaMetadataPtoBnn.
 */

#include "gym_aclnn.h"

#include <array>
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <tuple>
#include <vector>

namespace {

constexpr int64_t kMetadataElements = 1024;
constexpr int64_t kTilingDeviceElements = 16;

using Result = std::tuple<at::Tensor, at::Tensor>;

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

// Run/Meta 共用的入参标量与形状校验(Run 追加 NPU 放置/存储检查, Meta 只查值)。
static void CheckCommonArgs(const at::Tensor &cu_seqlens_q,
                            const at::Tensor &ori_topk_length,
                            const c10::optional<at::Tensor> &cmp_topk_length,
                            int64_t blocks, int64_t has_cmp, bool npu_checks)
{
  TORCH_CHECK(blocks >= 1 && blocks <= 36, "blocks must be in [1,36]");
  TORCH_CHECK(has_cmp == 0 || has_cmp == 1, "has_cmp must be 0 or 1");
  TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_q.numel() >= 2,
              "cu_seqlens_q must be rank-1 with at least 2 entries");
  if (npu_checks) {
    CheckInt32(cu_seqlens_q, "cu_seqlens_q");
    CheckInt32(ori_topk_length, "ori_topk_length");
    if (HasValue(cmp_topk_length))
      CheckInt32(*cmp_topk_length, "cmp_topk_length");
  } else {
    TORCH_CHECK(cu_seqlens_q.scalar_type() == at::kInt &&
                    ori_topk_length.scalar_type() == at::kInt,
                "cu_seqlens_q/ori_topk_length must be int32");
    if (HasValue(cmp_topk_length))
      TORCH_CHECK(cmp_topk_length->scalar_type() == at::kInt,
                  "cmp_topk_length must be int32");
  }
}

// Run 的 aclnn 下发段: 12 个标量捕获进任务闭包, 张量按槽位 scatter。
static void EnqueuePlan(const std::array<bool, 5> &present,
                        const std::vector<at::Tensor> &defined,
                        int64_t blocks, int64_t has_cmp, int64_t k_ori,
                        int64_t k_cmp, int64_t bs_ori, int64_t bs_cmp,
                        int64_t bt_ori, int64_t bt_cmp, int64_t ori_pages,
                        int64_t cmp_pages, int64_t ori_kv_stride0,
                        int64_t cmp_kv_stride0)
{
  using GetWorkspace = aclnnStatus (*)(
      const aclTensor *, const aclTensor *, const aclTensor *, int64_t, int64_t,
      int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
      int64_t, int64_t, const aclTensor *, const aclTensor *, uint64_t *,
      aclOpExecutor **);
  static const auto get_workspace = gym_torch::Symbol<GetWorkspace>(
      "aclnnMixedQuantSparseFlashMlaMetadataPtoBnnGetWorkspaceSize");
  static const auto execute = gym_torch::Symbol<gym_torch::Execute>(
      "aclnnMixedQuantSparseFlashMlaMetadataPtoBnn");
  gym_torch::Enqueue(
      "aclnnMixedQuantSparseFlashMlaMetadataPtoBnn", defined,
      [present, blocks, has_cmp, k_ori, k_cmp, bs_ori, bs_cmp, bt_ori, bt_cmp,
       ori_pages, cmp_pages, ori_kv_stride0,
       cmp_kv_stride0](const std::vector<gym_torch::TensorInfo> &infos, auto stream) {
        std::array<gym_torch::TensorHandle, 5> handles;
        size_t index = 0;
        for (size_t i = 0; i < handles.size(); ++i) {
          if (present[i])
            handles[i] = gym_torch::Tensor(infos[index++]);
        }
        uint64_t bytes = 0;
        aclOpExecutor *executor = nullptr;
        const auto status = get_workspace(
            handles[0].get(), handles[1].get(), handles[2].get(), blocks, has_cmp,
            k_ori, k_cmp, bs_ori, bs_cmp, bt_ori, bt_cmp, ori_pages, cmp_pages,
            ori_kv_stride0, cmp_kv_stride0, handles[3].get(), handles[4].get(),
            &bytes, &executor);
        TORCH_CHECK(status == 0,
                    "aclnnMixedQuantSparseFlashMlaMetadataPtoBnnGetWorkspaceSize "
                    "failed, status=", status, ": ", aclGetRecentErrMsg());
        gym_torch::ExecuteOp(execute, bytes, executor, stream);
        return 0;
      });
}

static Result Run(const at::Tensor &cu_seqlens_q,
                  const at::Tensor &ori_topk_length,
                  const c10::optional<at::Tensor> &cmp_topk_length,
                  int64_t blocks, int64_t has_cmp, int64_t k_ori, int64_t k_cmp,
                  int64_t bs_ori, int64_t bs_cmp, int64_t bt_ori, int64_t bt_cmp,
                  int64_t ori_pages, int64_t cmp_pages, int64_t ori_kv_stride0,
                  int64_t cmp_kv_stride0)
{
  CheckCommonArgs(cu_seqlens_q, ori_topk_length, cmp_topk_length, blocks,
                  has_cmp, true);
  const c10::OptionalDeviceGuard guard(cu_seqlens_q.device());

  auto metadata = at::empty({kMetadataElements}, cu_seqlens_q.options().dtype(at::kInt));
  gym_torch::CheckPlainStorage(metadata, "metadata");
  auto tiling_device = at::empty({kTilingDeviceElements}, cu_seqlens_q.options().dtype(at::kInt));
  gym_torch::CheckPlainStorage(tiling_device, "tiling_device");

  // Slot order = ACLNN signature: 3 inputs, then the metadata/tiling_device outputs.
  std::array<bool, 5> present{};
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
  add_required(tiling_device);

  EnqueuePlan(present, defined, blocks, has_cmp, k_ori, k_cmp, bs_ori, bs_cmp,
              bt_ori, bt_cmp, ori_pages, cmp_pages, ori_kv_stride0,
              cmp_kv_stride0);
  return {metadata, tiling_device};
}

// Meta: 输出为与输入无关的常量 int32 占位(metadata[1024]/tiling_device[16]), 供 meta 调用与 PT2 追踪走通(内容依赖设备数据, 追踪阶段无语义)。
static Result Meta(const at::Tensor &cu_seqlens_q,
                   const at::Tensor &ori_topk_length,
                   const c10::optional<at::Tensor> &cmp_topk_length,
                   int64_t blocks, int64_t has_cmp, int64_t k_ori, int64_t k_cmp,
                   int64_t bs_ori, int64_t bs_cmp, int64_t bt_ori, int64_t bt_cmp,
                   int64_t ori_pages, int64_t cmp_pages, int64_t ori_kv_stride0,
                   int64_t cmp_kv_stride0)
{
  CheckCommonArgs(cu_seqlens_q, ori_topk_length, cmp_topk_length, blocks,
                  has_cmp, false);
  (void)k_ori;
  (void)k_cmp;
  (void)bs_ori;
  (void)bs_cmp;
  (void)bt_ori;
  (void)bt_cmp;
  (void)ori_pages;
  (void)cmp_pages;
  (void)ori_kv_stride0;
  (void)cmp_kv_stride0;
  const auto options = cu_seqlens_q.options().dtype(at::kInt);
  return {at::empty({kMetadataElements}, options),
          at::empty({kTilingDeviceElements}, options)};
}

} // namespace

TORCH_LIBRARY_FRAGMENT(pypto_gym, m) {
  m.def("mixed_quant_sparse_flash_mla_metadata_pto_bnn(Tensor cu_seqlens_q, "
        "Tensor ori_topk_length, Tensor? cmp_topk_length=None, int blocks=28, "
        "int has_cmp=1, int k_ori=0, int k_cmp=0, int bs_ori=0, int bs_cmp=0, "
        "int bt_ori=0, int bt_cmp=0, int ori_pages=0, int cmp_pages=0, "
        "int ori_kv_stride0=0, int cmp_kv_stride0=0) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(pypto_gym, PrivateUse1, m) {
  m.impl("mixed_quant_sparse_flash_mla_metadata_pto_bnn", TORCH_FN(Run));
}

TORCH_LIBRARY_IMPL(pypto_gym, Meta, m) {
  m.impl("mixed_quant_sparse_flash_mla_metadata_pto_bnn", TORCH_FN(Meta));
}
