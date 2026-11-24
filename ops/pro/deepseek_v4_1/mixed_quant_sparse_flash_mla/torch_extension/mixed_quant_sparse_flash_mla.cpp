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
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gym_aclnn.h"
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
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
constexpr int64_t kHeads = 64;
constexpr int64_t kHeadDim = 512;
constexpr int64_t kOriKvBytes = 544;
constexpr int64_t kCmpKvBytes = 320;
constexpr int64_t kMetadataElements = 1024;

using CreateIntArray = aclIntArray *(*)(const int64_t *, uint64_t);
using DestroyIntArray = aclnnStatus (*)(const aclIntArray *);

static CreateIntArray CreateIntArrayFn() {
  static const auto function = [] {
    void *library = dlopen("libnnopbase.so", RTLD_LAZY | RTLD_GLOBAL);
    TORCH_CHECK(library != nullptr, "dlopen libnnopbase.so failed: ", dlerror());
    void *symbol = dlsym(library, "aclCreateIntArray");
    TORCH_CHECK(symbol != nullptr, "dlsym aclCreateIntArray failed");
    return reinterpret_cast<CreateIntArray>(symbol);
  }();
  return function;
}

static DestroyIntArray DestroyIntArrayFn() {
  static const auto function = [] {
    void *library = dlopen("libnnopbase.so", RTLD_LAZY | RTLD_GLOBAL);
    TORCH_CHECK(library != nullptr, "dlopen libnnopbase.so failed: ", dlerror());
    void *symbol = dlsym(library, "aclDestroyIntArray");
    TORCH_CHECK(symbol != nullptr, "dlsym aclDestroyIntArray failed");
    return reinterpret_cast<DestroyIntArray>(symbol);
  }();
  return function;
}

static std::vector<int64_t> HostValues(const c10::optional<at::Tensor> &value) {
  if (!value.has_value() || !value->defined())
    return {};
  const auto host = value->to(at::kCPU, at::kLong).contiguous();
  const auto *data = host.data_ptr<int64_t>();
  return std::vector<int64_t>(data, data + host.numel());
}

static bool HasValue(const c10::optional<at::Tensor> &value) {
  return value.has_value() && value->defined();
}

static bool IsLengthDtype(at::ScalarType type) {
  return type == at::kInt || type == at::kLong;
}

static void CheckOptionalLength(const c10::optional<at::Tensor> &value,
                                const at::Tensor &q, const char *name,
                                int64_t rows) {
  if (!HasValue(value))
    return;
  TORCH_CHECK(IsLengthDtype(value->scalar_type()) &&
                  value->device() == q.device() && value->is_contiguous() &&
                  value->dim() == 2 && value->size(0) == rows &&
                  value->size(1) == 1,
              name, " must be a contiguous int32/int64 [T,1] tensor on q.device");
  gym_torch::CheckPlainStorage(*value, name);
}

static void CheckNoGrad(const std::vector<at::Tensor> &inputs) {
  for (const auto &value : inputs)
    TORCH_CHECK(!at::GradMode::is_enabled() || !value.requires_grad(),
                "MixedQuantSparseFlashMla provides forward computation "
                "only; use torch.no_grad() for inference");
}

static void ValidateInputs(
    const at::Tensor &q, const at::Tensor &ori_kv,
    const at::Tensor &ori_sparse_indices, const at::Tensor &ori_block_table,
    const at::Tensor &cu_seqlens_q, const at::Tensor &sinks,
    const at::Tensor &metadata,
    const c10::optional<at::Tensor> &ori_topk_length,
    const c10::optional<at::Tensor> &cmp_kv,
    const c10::optional<at::Tensor> &cmp_sparse_indices,
    const c10::optional<at::Tensor> &cmp_block_table,
    const c10::optional<at::Tensor> &cmp_topk_length) {
  const int64_t t = q.size(0);
  TORCH_CHECK(q.dim() == 3 && q.size(1) == kHeads && q.size(2) == kHeadDim &&
                  q.scalar_type() == at::kBFloat16 &&
                  q.device().is_privateuseone() && q.is_contiguous(),
              "q must be a contiguous BF16 [T,64,512] NPU tensor");
  gym_torch::CheckPlainStorage(q, "q");

  TORCH_CHECK(ori_kv.dim() == 4 && ori_kv.size(2) == 1 &&
                  ori_kv.size(3) == kOriKvBytes &&
                  ori_kv.scalar_type() == at::kByte &&
                  ori_kv.device() == q.device(),
              "ori_kv must be uint8 [pages,bs,1,544] on q.device "
              "(axis-0 padding allowed)");
  gym_torch::CheckPlainStorage(ori_kv, "ori_kv");

  TORCH_CHECK(ori_sparse_indices.dim() == 3 &&
                  ori_sparse_indices.size(0) == t &&
                  ori_sparse_indices.size(1) == 1 &&
                  ori_sparse_indices.scalar_type() == at::kInt &&
                  ori_sparse_indices.device() == q.device() &&
                  ori_sparse_indices.is_contiguous(),
              "ori_sparse_indices must be contiguous int32 [T,1,K] on q.device");
  gym_torch::CheckPlainStorage(ori_sparse_indices, "ori_sparse_indices");

  TORCH_CHECK(ori_block_table.dim() == 2 &&
                  ori_block_table.scalar_type() == at::kInt &&
                  ori_block_table.device() == q.device() &&
                  ori_block_table.is_contiguous(),
              "ori_block_table must be a contiguous int32 [B,bt] tensor");
  gym_torch::CheckPlainStorage(ori_block_table, "ori_block_table");

  TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_q.numel() >= 2 &&
                  IsLengthDtype(cu_seqlens_q.scalar_type()) &&
                  cu_seqlens_q.device() == q.device() &&
                  cu_seqlens_q.is_contiguous(),
              "cu_seqlens_q must be a contiguous int32/int64 [B+1] tensor");
  gym_torch::CheckPlainStorage(cu_seqlens_q, "cu_seqlens_q");

  TORCH_CHECK(sinks.dim() == 1 && sinks.numel() == q.size(1) &&
                  sinks.scalar_type() == at::kFloat &&
                  sinks.device() == q.device() && sinks.is_contiguous(),
              "sinks must be a contiguous FP32 [heads] tensor");
  gym_torch::CheckPlainStorage(sinks, "sinks");

  TORCH_CHECK(metadata.scalar_type() == at::kInt &&
                  metadata.numel() == kMetadataElements &&
                  metadata.device() == q.device() && metadata.is_contiguous(),
              "metadata must be a contiguous int32[1024] tensor");
  gym_torch::CheckPlainStorage(metadata, "metadata");

  CheckOptionalLength(ori_topk_length, q, "ori_topk_length", t);
  CheckOptionalLength(cmp_topk_length, q, "cmp_topk_length", t);

  if (HasValue(cmp_kv)) {
    TORCH_CHECK(HasValue(cmp_sparse_indices) && HasValue(cmp_block_table),
                "cmp_kv is given but cmp_sparse_indices/cmp_block_table is missing");
    TORCH_CHECK(cmp_kv->dim() == 4 && cmp_kv->size(2) == 1 &&
                    cmp_kv->size(3) == kCmpKvBytes &&
                    cmp_kv->scalar_type() == at::kByte &&
                    cmp_kv->device() == q.device(),
                "cmp_kv must be uint8 [pages,bs,1,320] on q.device "
                "(axis-0 padding allowed)");
    gym_torch::CheckPlainStorage(*cmp_kv, "cmp_kv");
  }
  if (HasValue(cmp_sparse_indices)) {
    TORCH_CHECK(cmp_sparse_indices->dim() == 3 &&
                    cmp_sparse_indices->size(0) == t &&
                    cmp_sparse_indices->size(1) == 1 &&
                    cmp_sparse_indices->scalar_type() == at::kInt &&
                    cmp_sparse_indices->device() == q.device() &&
                    cmp_sparse_indices->is_contiguous(),
                "cmp_sparse_indices must be contiguous int32 [T,1,K] on q.device");
    gym_torch::CheckPlainStorage(*cmp_sparse_indices, "cmp_sparse_indices");
  }
  if (HasValue(cmp_block_table)) {
    TORCH_CHECK(cmp_block_table->dim() == 2 &&
                    cmp_block_table->scalar_type() == at::kInt &&
                    cmp_block_table->device() == q.device() &&
                    cmp_block_table->is_contiguous(),
                "cmp_block_table must be a contiguous int32 [B,bt] tensor");
    gym_torch::CheckPlainStorage(*cmp_block_table, "cmp_block_table");
  }
}

static void ValidateOutput(const at::Tensor &q, const at::Tensor &out,
                           const at::Tensor &lse) {
  TORCH_CHECK(out.defined() && out.dim() == 3 && out.sizes() == q.sizes() &&
                  out.scalar_type() == at::kBFloat16 &&
                  out.device() == q.device() && out.is_contiguous(),
              "out must be a contiguous BF16 [T,64,512] tensor on q.device");
  TORCH_CHECK(lse.defined() && lse.dim() == 3 && lse.size(0) == 1 &&
                  lse.size(1) == q.size(0) && lse.size(2) == q.size(1) &&
                  lse.scalar_type() == at::kFloat &&
                  lse.device() == q.device() && lse.is_contiguous(),
              "lse must be a contiguous FP32 [1,T,heads] tensor on q.device");
}

static Result Run(
    const at::Tensor &q, const at::Tensor &ori_kv,
    const at::Tensor &ori_sparse_indices, const at::Tensor &ori_block_table,
    const at::Tensor &cu_seqlens_q, const at::Tensor &sinks,
    const at::Tensor &metadata,
    const c10::optional<at::Tensor> &ori_topk_length,
    const c10::optional<at::Tensor> &cmp_kv,
    const c10::optional<at::Tensor> &cmp_sparse_indices,
    const c10::optional<at::Tensor> &cmp_block_table,
    const c10::optional<at::Tensor> &cmp_topk_length,
    const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_ori_kv,
    const c10::optional<at::Tensor> &seqused_cmp_kv, int64_t quant_mode,
    double softmax_scale, std::string layout_q, std::string layout_kv,
    bool return_softmax_lse, const c10::optional<at::Tensor> &provided_out,
    const c10::optional<at::Tensor> &provided_lse,
    const c10::optional<int64_t> &block_dim,
    const c10::optional<int64_t> &fd_enable) {
  (void)seqused_q;
  (void)seqused_ori_kv;
  (void)seqused_cmp_kv;
  (void)return_softmax_lse;
  (void)block_dim;
  (void)fd_enable;
  TORCH_CHECK(quant_mode == 1,
              "MixedQuantSparseFlashMla supports quant_mode=1 only");
  TORCH_CHECK(layout_q == "TND" && layout_kv == "PA_BBND",
              "only layout_q='TND', layout_kv='PA_BBND' are supported");
  const c10::OptionalDeviceGuard guard(q.device());
  ValidateInputs(q, ori_kv, ori_sparse_indices, ori_block_table, cu_seqlens_q,
                 sinks, metadata, ori_topk_length, cmp_kv, cmp_sparse_indices,
                 cmp_block_table, cmp_topk_length);

  at::Tensor out;
  at::Tensor lse;
  if (provided_out.has_value() && provided_out->defined()) {
    out = *provided_out;
  } else {
    out = at::empty(q.sizes(), q.options().dtype(at::kBFloat16));
  }
  if (provided_lse.has_value() && provided_lse->defined()) {
    lse = *provided_lse;
  } else {
    lse = at::empty({1, q.size(0), q.size(1)}, q.options().dtype(at::kFloat));
  }
  ValidateOutput(q, out, lse);
  gym_torch::CheckPlainStorage(out, "out");
  gym_torch::CheckPlainStorage(lse, "lse");
  if (provided_out.has_value() && provided_out->defined())
    at::assert_no_overlap(out, q);
  if (provided_lse.has_value() && provided_lse->defined())
    at::assert_no_overlap(lse, q);

  std::vector<at::Tensor> inputs = {
      q, ori_kv, ori_sparse_indices, ori_block_table, cu_seqlens_q, sinks,
      metadata};
  if (HasValue(ori_topk_length))
    inputs.push_back(*ori_topk_length);
  if (HasValue(cmp_kv))
    inputs.push_back(*cmp_kv);
  if (HasValue(cmp_sparse_indices))
    inputs.push_back(*cmp_sparse_indices);
  if (HasValue(cmp_block_table))
    inputs.push_back(*cmp_block_table);
  if (HasValue(cmp_topk_length))
    inputs.push_back(*cmp_topk_length);
  CheckNoGrad(inputs);

  // The default ACLNN signature carries ValueDepend inputs as host aclIntArray
  // values; the Tensor variant does not feed nnopbase's host placement and the
  // tiling rejects it.
  const auto ori_len_host = HostValues(ori_topk_length);
  const auto cmp_len_host = HostValues(cmp_topk_length);
  const auto cu_seqlens_host =
      HostValues(c10::optional<at::Tensor>(cu_seqlens_q));
  const bool has_ori_len = !ori_len_host.empty();
  const bool has_cmp_len = !cmp_len_host.empty();
  TORCH_CHECK(!cu_seqlens_host.empty(),
              "cu_seqlens_q must contain host values for tiling");

  // Tensor slots in ACLNN order: q, ori_kv, ori_sparse_indices,
  // ori_block_table, sinks, cmp_kv, cmp_sparse_indices, cmp_block_table,
  // metadata, out, lse. The three int-array slots are filled separately.
  static constexpr std::array<size_t, 11> kTensorSlots{0, 1, 2, 3, 5, 6,
                                                        7, 8, 11, 12, 13};
  std::array<bool, 11> present{};
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
  add_required(ori_kv);
  add_required(ori_sparse_indices);
  add_required(ori_block_table);
  add_required(sinks);
  add_optional(cmp_kv);
  add_optional(cmp_sparse_indices);
  add_optional(cmp_block_table);
  add_required(metadata);
  add_required(out);
  add_required(lse);

  using GetWorkspace = aclnnStatus (*)(
      const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
      const aclIntArray *, const aclTensor *, const aclTensor *,
      const aclTensor *, const aclTensor *, const aclIntArray *,
      const aclIntArray *, const aclTensor *, char *, char *, double,
      const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
  static const auto get_workspace = gym_torch::Symbol<GetWorkspace>(
      "aclnnMixedQuantSparseFlashMlaGetWorkspaceSize");
  static const auto execute = gym_torch::Symbol<gym_torch::Execute>(
      "aclnnMixedQuantSparseFlashMla");
  gym_torch::Enqueue(
      "aclnnMixedQuantSparseFlashMla", defined,
      [present, softmax_scale, layout_q, layout_kv, ori_len_host,
       cmp_len_host, cu_seqlens_host, has_ori_len,
       has_cmp_len](const std::vector<gym_torch::TensorInfo> &infos, auto stream) {
        std::vector<gym_torch::TensorHandle> handles;
        handles.reserve(infos.size());
        for (const auto &info : infos)
          handles.push_back(gym_torch::Tensor(info));
        std::array<const aclTensor *, 14> slots{};
        size_t cursor = 0;
        for (size_t i = 0; i < kTensorSlots.size(); ++i)
          if (present[i])
            slots[kTensorSlots[i]] = handles[cursor++].get();
        TORCH_CHECK(cursor == handles.size(),
                    "MixedQuantSparseFlashMla argument mapping mismatch");
        aclIntArray *ori_len =
            has_ori_len ? CreateIntArrayFn()(ori_len_host.data(),
                                             ori_len_host.size())
                        : nullptr;
        aclIntArray *cmp_len =
            has_cmp_len ? CreateIntArrayFn()(cmp_len_host.data(),
                                             cmp_len_host.size())
                        : nullptr;
        aclIntArray *cu_len = CreateIntArrayFn()(cu_seqlens_host.data(),
                                                 cu_seqlens_host.size());
        uint64_t bytes = 0;
        aclOpExecutor *executor = nullptr;
        const auto status = get_workspace(
            slots[0], slots[1], slots[2], slots[3], ori_len, slots[5],
            slots[6], slots[7], slots[8], cmp_len, cu_len, slots[11],
            const_cast<char *>(layout_q.c_str()),
            const_cast<char *>(layout_kv.c_str()), softmax_scale, slots[12],
            slots[13], &bytes, &executor);
        if (status == 0)
          gym_torch::ExecuteOp(execute, bytes, executor, stream);
        if (ori_len != nullptr)
          DestroyIntArrayFn()(ori_len);
        if (cmp_len != nullptr)
          DestroyIntArrayFn()(cmp_len);
        DestroyIntArrayFn()(cu_len);
        TORCH_CHECK(status == 0,
                    "aclnnMixedQuantSparseFlashMlaGetWorkspaceSize "
                    "failed, status=",
                    status, ": ", aclGetRecentErrMsg());
        return 0;
      });
  return {out, lse};
}

static Result Meta(
    const at::Tensor &q, const at::Tensor &ori_kv,
    const at::Tensor &ori_sparse_indices, const at::Tensor &ori_block_table,
    const at::Tensor &cu_seqlens_q, const at::Tensor &sinks,
    const at::Tensor &metadata,
    const c10::optional<at::Tensor> &ori_topk_length,
    const c10::optional<at::Tensor> &cmp_kv,
    const c10::optional<at::Tensor> &cmp_sparse_indices,
    const c10::optional<at::Tensor> &cmp_block_table,
    const c10::optional<at::Tensor> &cmp_topk_length,
    const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_ori_kv,
    const c10::optional<at::Tensor> &seqused_cmp_kv, int64_t quant_mode,
    double softmax_scale, std::string layout_q, std::string layout_kv,
    bool return_softmax_lse, const c10::optional<at::Tensor> &provided_out,
    const c10::optional<at::Tensor> &provided_lse,
    const c10::optional<int64_t> &block_dim,
    const c10::optional<int64_t> &fd_enable) {
  (void)ori_kv;
  (void)ori_sparse_indices;
  (void)ori_block_table;
  (void)cu_seqlens_q;
  (void)sinks;
  (void)metadata;
  (void)ori_topk_length;
  (void)cmp_kv;
  (void)cmp_sparse_indices;
  (void)cmp_block_table;
  (void)cmp_topk_length;
  (void)seqused_q;
  (void)seqused_ori_kv;
  (void)seqused_cmp_kv;
  (void)quant_mode;
  (void)softmax_scale;
  (void)layout_q;
  (void)layout_kv;
  (void)return_softmax_lse;
  (void)block_dim;
  (void)fd_enable;
  TORCH_CHECK(q.dim() == 3 && q.size(1) == kHeads && q.size(2) == kHeadDim &&
                  q.scalar_type() == at::kBFloat16,
              "q must be a BF16 [T,64,512] tensor");
  at::Tensor out = provided_out.has_value() && provided_out->defined()
                       ? *provided_out
                       : at::empty_symint(q.sym_sizes(),
                                          q.options().dtype(at::kBFloat16));
  at::Tensor lse = provided_lse.has_value() && provided_lse->defined()
                       ? *provided_lse
                       : at::empty_symint({1, q.sym_size(0), q.sym_size(1)},
                                          q.options().dtype(at::kFloat));
  TORCH_CHECK(out.sizes() == q.sizes() &&
                  out.scalar_type() == at::kBFloat16,
              "out must be BF16 with the same shape as q");
  TORCH_CHECK(lse.dim() == 3 && lse.size(0) == 1 &&
                  lse.size(1) == q.size(0) && lse.size(2) == q.size(1) &&
                  lse.scalar_type() == at::kFloat,
              "lse must be FP32 [1,T,heads]");
  return {out, lse};
}
} // namespace

TORCH_LIBRARY_FRAGMENT(pypto_gym, m) {
  m.def("mixed_quant_sparse_flash_mla(Tensor q, Tensor ori_kv, "
        "Tensor ori_sparse_indices, Tensor ori_block_table, "
        "Tensor cu_seqlens_q, Tensor sinks, Tensor metadata, "
        "Tensor? ori_topk_length=None, Tensor? cmp_kv=None, "
        "Tensor? cmp_sparse_indices=None, Tensor? cmp_block_table=None, "
        "Tensor? cmp_topk_length=None, Tensor? seqused_q=None, "
        "Tensor? seqused_ori_kv=None, Tensor? seqused_cmp_kv=None, *, "
        "int quant_mode=1, float softmax_scale=0.0, str layout_q=\"TND\", "
        "str layout_kv=\"PA_BBND\", bool return_softmax_lse=False, "
        "Tensor(a!)? out=None, Tensor(b!)? lse=None, int? block_dim=None, "
        "int? fd_enable=None) -> (Tensor(a!), Tensor(b!))");
}
TORCH_LIBRARY_IMPL(pypto_gym, PrivateUse1, m) {
  m.impl("mixed_quant_sparse_flash_mla", TORCH_FN(Run));
}
TORCH_LIBRARY_IMPL(pypto_gym, Meta, m) {
  m.impl("mixed_quant_sparse_flash_mla", TORCH_FN(Meta));
}
