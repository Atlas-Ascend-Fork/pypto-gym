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
#include <array>
#include <string>
#include <tuple>
#include <vector>

namespace {
using Result = std::tuple<at::Tensor, at::Tensor>;

// Fixed operator geometry (op_host registration; see the operator README).
constexpr int64_t kHeadsQ = 32;  // N1
constexpr int64_t kPackedD = 64; // 128 logical FP4 dims, two per byte
constexpr int64_t kCandidateCapacity = 2048;
constexpr int64_t kMaxTopk = 8192;
constexpr int64_t kMaxCmpRatio = 128;
constexpr int64_t kMetadataElements = 1024;

// E8M0 scales are registered as raw UINT8 exponent bytes; callers may hold
// float8_e8m0fnu views, so normalize them exactly like the aclnn test bridge.
static at::Tensor AsScaleBytes(const at::Tensor &tensor) {
  if (tensor.defined() && tensor.scalar_type() == at::kFloat8_e8m0fnu)
    return tensor.view(at::kByte);
  return tensor;
}

static bool IsScaleType(const at::Tensor &tensor) {
  return tensor.scalar_type() == at::kByte ||
         tensor.scalar_type() == at::kFloat8_e8m0fnu;
}

static bool HasValue(const c10::optional<at::Tensor> &value) {
  return value.has_value() && value->defined();
}

static void CheckOptionalInt32(const c10::optional<at::Tensor> &value,
                               const char *name, const at::Tensor &q) {
  if (!HasValue(value))
    return;
  TORCH_CHECK(value->scalar_type() == at::kInt &&
                  value->device() == q.device() && value->is_contiguous(),
              name, " must be a contiguous int32 tensor on the q device");
  gym_torch::CheckPlainStorage(*value, name);
}

static void Validate(const at::Tensor &q, const at::Tensor &k,
                     const at::Tensor &w, const at::Tensor &descale_q,
                     const c10::optional<at::Tensor> &descale_k,
                     const at::Tensor &candidate_block_indices,
                     const at::Tensor &candidate_block_length,
                     const c10::optional<at::Tensor> &cu_seqlens_q,
                     const c10::optional<at::Tensor> &cu_seqlens_k,
                     const c10::optional<at::Tensor> &seqused_q,
                     const c10::optional<at::Tensor> &seqused_k,
                     const c10::optional<at::Tensor> &cmp_residual_k,
                     const c10::optional<at::Tensor> &block_table,
                     const c10::optional<at::Tensor> &output_idx_offset,
                     const c10::optional<at::Tensor> &metadata) {
  const int64_t t = q.size(0);
  TORCH_CHECK(q.device().is_privateuseone() && k.device() == q.device(),
              "q/k must live on one NPU device");
  TORCH_CHECK(q.scalar_type() == at::kByte && k.scalar_type() == at::kByte,
              "q/k storage must be uint8 packed MXFP4");
  TORCH_CHECK(q.dim() == 3 && q.size(1) == kHeadsQ && q.size(2) == kPackedD &&
                  q.is_contiguous(),
              "q must be contiguous with shape (T1,32,64)");
  gym_torch::CheckPlainStorage(q, "q");
  // k and descale_k deliberately skip the contiguity check: the TND path
  // supports axis-0 padding and host tiling reads the strides.
  gym_torch::CheckPlainStorage(k, "k");
  TORCH_CHECK(w.scalar_type() == at::kFloat && w.dim() == 2 && w.size(0) == t &&
                  w.size(1) == kHeadsQ && w.device() == q.device() &&
                  w.is_contiguous(),
              "w must be contiguous FP32 with shape (T1,32)");
  gym_torch::CheckPlainStorage(w, "w");
  TORCH_CHECK(IsScaleType(descale_q) && descale_q.dim() == 4 &&
                  descale_q.size(0) == t && descale_q.size(1) == kHeadsQ &&
                  descale_q.size(2) == 2 && descale_q.size(3) == 2 &&
                  descale_q.device() == q.device() && descale_q.is_contiguous(),
              "descale_q must be contiguous E8M0/uint8 with shape (T1,32,2,2)");
  gym_torch::CheckPlainStorage(descale_q, "descale_q");
  if (HasValue(descale_k)) {
    TORCH_CHECK(IsScaleType(*descale_k) && descale_k->device() == q.device(),
                "descale_k must be an E8M0/uint8 tensor on the q device");
    gym_torch::CheckPlainStorage(*descale_k, "descale_k");
  }
  TORCH_CHECK(candidate_block_indices.scalar_type() == at::kInt &&
                  candidate_block_indices.dim() == 3 &&
                  candidate_block_indices.size(0) == t &&
                  candidate_block_indices.size(1) == 1 &&
                  candidate_block_indices.size(2) == kCandidateCapacity &&
                  candidate_block_indices.device() == q.device() &&
                  candidate_block_indices.is_contiguous(),
              "candidate_block_indices must be contiguous int32 (T1,1,2048)");
  gym_torch::CheckPlainStorage(candidate_block_indices, "candidate_block_indices");
  TORCH_CHECK(candidate_block_length.scalar_type() == at::kInt &&
                  candidate_block_length.device() == q.device() &&
                  candidate_block_length.is_contiguous() &&
                  candidate_block_length.numel() == t,
              "candidate_block_length must be contiguous int32 with T1 entries");
  gym_torch::CheckPlainStorage(candidate_block_length, "candidate_block_length");
  CheckOptionalInt32(cu_seqlens_q, "cu_seqlens_q", q);
  CheckOptionalInt32(cu_seqlens_k, "cu_seqlens_k", q);
  CheckOptionalInt32(seqused_q, "seqused_q", q);
  CheckOptionalInt32(seqused_k, "seqused_k", q);
  CheckOptionalInt32(cmp_residual_k, "cmp_residual_k", q);
  CheckOptionalInt32(output_idx_offset, "output_idx_offset", q);
  if (HasValue(block_table)) {
    TORCH_CHECK(block_table->scalar_type() == at::kInt &&
                    block_table->dim() == 2 &&
                    block_table->device() == q.device() &&
                    block_table->is_contiguous(),
                "block_table must be a contiguous int32 (B,max_blocks) tensor");
    gym_torch::CheckPlainStorage(*block_table, "block_table");
  }
  TORCH_CHECK(HasValue(metadata),
              "metadata must contain the current LI/LD schedule (int32[1024])");
  TORCH_CHECK(metadata->scalar_type() == at::kInt &&
                  metadata->device() == q.device() &&
                  metadata->is_contiguous() &&
                  metadata->numel() == kMetadataElements,
              "metadata must be a contiguous int32[1024] tensor");
  gym_torch::CheckPlainStorage(*metadata, "metadata");
}

static void CheckNoGrad(const std::vector<at::Tensor> &inputs) {
  for (const auto &value : inputs)
    TORCH_CHECK(!at::GradMode::is_enabled() || !value.requires_grad(),
                "QuantSparseLightningIndexer provides forward computation "
                "only; use torch.no_grad() for inference");
}

static Result FinishOutputs(const at::Tensor &indices, const at::Tensor &values,
                            bool return_value) {
  // return_value=False still allocates the full score buffer (the
  // WriteValues=0 specialization simply does not store into it); the caller
  // sees a zero-element view so the public contract stays shape-stable.
  if (return_value)
    return {indices, values};
  return {indices, values.reshape({-1}).slice(0, 0, 0)};
}

static Result
Run(const at::Tensor &q, const at::Tensor &k, const at::Tensor &w,
    const at::Tensor &descale_q, const c10::optional<at::Tensor> &descale_k,
    const at::Tensor &candidate_block_indices,
    const at::Tensor &candidate_block_length,
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_k,
    const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_k,
    const c10::optional<at::Tensor> &cmp_residual_k,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &output_idx_offset,
    const c10::optional<at::Tensor> &metadata, int64_t topk,
    int64_t candidate_block_size, int64_t quant_mode, int64_t max_seqlen_q,
    int64_t mask_mode, int64_t cmp_ratio, std::string layout_q,
    std::string layout_k, bool return_value) {
  TORCH_CHECK(quant_mode == 1,
              "Quant_sparse_lightning_indexer supports quant_mode=1 only");
  TORCH_CHECK(layout_q == "TND" && (layout_k == "PA_BBND" || layout_k == "TND"),
              "Current QSLI layouts must be TND Q with PA_BBND or TND K");
  TORCH_CHECK(topk >= 1 && topk <= kMaxTopk, "Topk must be in [1,", kMaxTopk,
                                               "]");
  TORCH_CHECK(candidate_block_size == 8, "Candidate_block_size must be 8");
  TORCH_CHECK(mask_mode == 0 || mask_mode == 3, "Mask_mode must be 0 or 3");
  TORCH_CHECK(cmp_ratio >= 1 && cmp_ratio <= kMaxCmpRatio,
              "Cmp_ratio must be in [1,", kMaxCmpRatio, "]");
  TORCH_CHECK(max_seqlen_q >= -1, "Max_seqlen_q must be -1 or nonnegative");
  const bool is_pa = layout_k == "PA_BBND";
  TORCH_CHECK(!is_pa || !HasValue(descale_k),
              "Descale_k is only supported with TND K; PA_BBND stores scales "
              "in packed k");
  TORCH_CHECK(is_pa || !HasValue(block_table),
              "Block_table must be None for TND K");
  TORCH_CHECK(q.dim() == 3 && q.device().is_privateuseone(),
              "q must be a rank-3 NPU tensor (T1,32,64)");
  TORCH_CHECK(q.size(0) >= 1, "T1 must be positive");
  const c10::OptionalDeviceGuard guard(q.device());
  const auto scale_q = AsScaleBytes(descale_q);
  const auto scale_k = HasValue(descale_k)
                           ? c10::optional<at::Tensor>(AsScaleBytes(*descale_k))
                           : c10::optional<at::Tensor>(c10::nullopt);
  Validate(q, k, w, scale_q, scale_k, candidate_block_indices,
           candidate_block_length, cu_seqlens_q, cu_seqlens_k, seqused_q,
           seqused_k, cmp_residual_k, block_table, output_idx_offset, metadata);

  const int64_t t = q.size(0);
  auto indices = at::empty({t, 1, topk}, q.options().dtype(at::kInt));
  auto values = at::empty({t, 1, topk}, q.options().dtype(at::kBFloat16));
  gym_torch::CheckPlainStorage(indices, "indices");
  gym_torch::CheckPlainStorage(values, "values");

  // Slot order = ACLNN signature: 15 inputs, then indices, then values.
  // Enqueue() only accepts defined tensors, so track which slots are present
  // and let the callback scatter the handles back with nullptr holes.
  std::array<bool, 17> present{};
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
  add_required(q);
  add_required(k);
  add_required(w);
  add_required(scale_q);
  add_optional(scale_k);
  add_required(candidate_block_indices);
  add_required(candidate_block_length);
  add_optional(cu_seqlens_q);
  add_optional(cu_seqlens_k);
  add_optional(seqused_q);
  add_optional(seqused_k);
  add_optional(cmp_residual_k);
  add_optional(block_table);
  add_optional(output_idx_offset);
  add_optional(metadata);
  add_required(indices);
  add_required(values);
  CheckNoGrad(defined);

  using GetWorkspace = aclnnStatus (*)(
      const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
      const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
      const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
      const aclTensor *, const aclTensor *, const aclTensor *, int64_t, int64_t,
      int64_t, int64_t, int64_t, int64_t, char *, char *, bool,
      const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
  static const auto get_workspace = gym_torch::Symbol<GetWorkspace>(
      "aclnnQuantSparseLightningIndexerGetWorkspaceSize");
  static const auto execute = gym_torch::Symbol<gym_torch::Execute>(
      "aclnnQuantSparseLightningIndexer");
  gym_torch::Enqueue(
      "aclnnQuantSparseLightningIndexer", defined,
      [present, topk, candidate_block_size, quant_mode, max_seqlen_q, mask_mode,
       cmp_ratio, layout_q, layout_k,
       return_value](const std::vector<gym_torch::TensorInfo> &infos,
                     auto stream) {
        std::vector<gym_torch::TensorHandle> handles;
        handles.reserve(infos.size());
        for (const auto &info : infos)
          handles.push_back(gym_torch::Tensor(info));
        std::array<const aclTensor *, 17> slots{};
        size_t cursor = 0;
        for (size_t i = 0; i < slots.size(); ++i)
          if (present[i])
            slots[i] = handles[cursor++].get();
        uint64_t bytes = 0;
        aclOpExecutor *executor = nullptr;
        const auto status = get_workspace(
            slots[0], slots[1], slots[2], slots[3], slots[4], slots[5], slots[6],
            slots[7], slots[8], slots[9], slots[10], slots[11], slots[12],
            slots[13], slots[14], topk, candidate_block_size, quant_mode,
            max_seqlen_q, mask_mode, cmp_ratio, const_cast<char *>(layout_q.c_str()),
            const_cast<char *>(layout_k.c_str()), return_value, slots[15],
            slots[16], &bytes, &executor);
        TORCH_CHECK(status == 0,
                    "aclnnQuantSparseLightningIndexerGetWorkspaceSize failed, "
                    "status=",
                    status, ": ", aclGetRecentErrMsg());
        gym_torch::ExecuteOp(execute, bytes, executor, stream);
        return 0;
      });
  return FinishOutputs(indices, values, return_value);
}

static Result
Meta(const at::Tensor &q, const at::Tensor &k, const at::Tensor &w,
     const at::Tensor &descale_q, const c10::optional<at::Tensor> &descale_k,
     const at::Tensor &candidate_block_indices,
     const at::Tensor &candidate_block_length,
     const c10::optional<at::Tensor> &cu_seqlens_q,
     const c10::optional<at::Tensor> &cu_seqlens_k,
     const c10::optional<at::Tensor> &seqused_q,
     const c10::optional<at::Tensor> &seqused_k,
     const c10::optional<at::Tensor> &cmp_residual_k,
     const c10::optional<at::Tensor> &block_table,
     const c10::optional<at::Tensor> &output_idx_offset,
     const c10::optional<at::Tensor> &metadata, int64_t topk,
     int64_t candidate_block_size, int64_t quant_mode, int64_t max_seqlen_q,
     int64_t mask_mode, int64_t cmp_ratio, std::string layout_q,
     std::string layout_k, bool return_value) {
  TORCH_CHECK(q.dim() == 3 && q.scalar_type() == at::kByte,
              "q must be a rank-3 uint8 tensor (T1,32,64)");
  TORCH_CHECK(topk >= 1, "Topk must be positive");
  const auto t = q.sym_size(0);
  auto indices = at::empty_symint({t, 1, topk}, q.options().dtype(at::kInt));
  auto values = at::empty_symint({t, 1, topk}, q.options().dtype(at::kBFloat16));
  return FinishOutputs(indices, values, return_value);
}
} // namespace

TORCH_LIBRARY_FRAGMENT(pypto_gym, m) {
  m.def("quant_sparse_lightning_indexer(Tensor q, Tensor k, Tensor w, Tensor "
        "descale_q, Tensor? descale_k, Tensor candidate_block_indices, "
        "Tensor candidate_block_length, Tensor? cu_seqlens_q=None, Tensor? "
        "cu_seqlens_k=None, Tensor? seqused_q=None, Tensor? seqused_k=None, "
        "Tensor? cmp_residual_k=None, Tensor? block_table=None, Tensor? "
        "output_idx_offset=None, Tensor? metadata=None, *, int topk, int "
        "candidate_block_size=8, int quant_mode=1, int max_seqlen_q=-1, int "
        "mask_mode=0, int cmp_ratio=1, str layout_q=\"TND\", str "
        "layout_k=\"TND\", bool return_value=False) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(pypto_gym, PrivateUse1, m) {
  m.impl("quant_sparse_lightning_indexer", TORCH_FN(Run));
}
TORCH_LIBRARY_IMPL(pypto_gym, Meta, m) {
  m.impl("quant_sparse_lightning_indexer", TORCH_FN(Meta));
}
