/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file attn_prologue.cpp
 * \brief Precompiled Torch binding for aclnnAttnPrologue.
 *
 * The three projection weights use FRACTAL_NZ storage; gym_aclnn.h now carries
 * the tensor's real NPU format and physical storage shape, while all other
 * operands stay ND.
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
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

constexpr int64_t H = 5120;
constexpr int64_t R = 1280;
constexpr int64_t N = 64;
constexpr int64_t D = 512;
constexpr int64_t DR = 64;
constexpr int64_t KV_RECORD_BYTES = 544;
constexpr int64_t FRACTAL_NZ = 29;

void CheckND(const at::Tensor &tensor, at::ScalarType dtype,
             const std::vector<int64_t> &shape, const char *name) {
  TORCH_CHECK(tensor.device().is_privateuseone(), name, " must be an NPU tensor");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.sizes().vec() == shape, name, " has an invalid shape");
  gym_torch::CheckNDStorage(tensor, name);
}

void CheckNZ(const at::Tensor &tensor, const std::vector<int64_t> &shape,
             const char *name) {
  TORCH_CHECK(tensor.device().is_privateuseone(), name, " must be an NPU tensor");
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float8_e4m3fn,
              name, " must be E4M3");
  TORCH_CHECK(tensor.sizes().vec() == shape, name, " has an invalid shape");
  const auto format = at_npu::native::get_npu_format(tensor);
  TORCH_CHECK(format == FRACTAL_NZ, name,
              " must use FRACTAL_NZ storage; got NPU format ", format);
}

void CheckNoGrad(const std::vector<at::Tensor> &inputs) {
  for (const auto &value : inputs)
    TORCH_CHECK(!at::GradMode::is_enabled() || !value.requires_grad(),
                "AttnPrologue provides forward computation only; use "
                "torch.no_grad() for inference");
}

void EnqueueAttnPrologue(const std::vector<at::Tensor> &inputs,
                         const std::vector<at::Tensor> &outputs,
                         double norm_eps, int64_t block_stride,
                         int64_t row_stride, int64_t storage_bytes,
                         int64_t emit_qr_norm) {
  using GetWorkspace = aclnnStatus (*)(
      const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
      const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
      const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
      const aclTensor *, const aclTensor *, const aclTensor *, double,
      int64_t, int64_t, int64_t, int64_t,
      const aclTensor *, const aclTensor *, const aclTensor *,
      uint64_t *, aclOpExecutor **);
  static const auto get_workspace =
      gym_torch::Symbol<GetWorkspace>("aclnnAttnPrologueGetWorkspaceSize");
  static const auto execute =
      gym_torch::Symbol<gym_torch::Execute>("aclnnAttnPrologue");

  auto tensors = inputs;
  tensors.insert(tensors.end(), outputs.begin(), outputs.end());
  gym_torch::Enqueue(
      "aclnnAttnPrologue", tensors,
      [norm_eps, block_stride, row_stride,
       storage_bytes, emit_qr_norm](const auto &values, auto stream) {
        std::vector<gym_torch::TensorHandle> handles;
        handles.reserve(values.size());
        for (const auto &value : values)
          handles.push_back(gym_torch::Tensor(value));
        uint64_t bytes = 0;
        aclOpExecutor *executor = nullptr;
        const auto status = get_workspace(
            handles[0].get(), handles[1].get(), handles[2].get(),
            handles[3].get(), handles[4].get(), handles[5].get(),
            handles[6].get(), handles[7].get(), handles[8].get(),
            handles[9].get(), handles[10].get(), handles[11].get(),
            handles[12].get(), handles[13].get(), handles[14].get(), norm_eps,
            block_stride, row_stride, storage_bytes, emit_qr_norm,
            handles[15].get(), handles[16].get(), handles[17].get(), &bytes,
            &executor);
        TORCH_CHECK(status == 0,
                    "aclnnAttnPrologueGetWorkspaceSize failed, status=", status,
                    ": ", aclGetRecentErrMsg());
        gym_torch::ExecuteOp(execute, bytes, executor, stream);
        return 0;
      });
}

struct CacheLayout {
  int64_t pageStride;
  int64_t rowStride;
  int64_t bytes;
};

static CacheLayout CheckCache(const at::Tensor &kv_cache) {
  TORCH_CHECK(kv_cache.device().is_privateuseone() &&
                  kv_cache.scalar_type() == at::ScalarType::Byte &&
                  kv_cache.dim() == 4 && kv_cache.size(2) == 1 &&
                  kv_cache.size(3) >= KV_RECORD_BYTES,
              "kv_cache must be UINT8 [P,BS,1,C>=544]");
  // torch_npu reports a plain 4-D uint8 buffer as NCHW (0) rather than ND (2).
  // The ACLNN kernel treats kv_cache as a flat record buffer, so both are valid.
  const auto kv_format = at_npu::native::get_npu_format(kv_cache);
  TORCH_CHECK(kv_format == ACL_FORMAT_ND || kv_format == ACL_FORMAT_NCHW,
              "kv_cache must use ND/NCHW storage; got NPU format ", kv_format);
  const int64_t pages = kv_cache.size(0);
  const int64_t block_size = kv_cache.size(1);
  TORCH_CHECK(pages > 0 && block_size > 0,
              "kv_cache P/BS must be positive");
  TORCH_CHECK(kv_cache.stride(3) == 1,
              "kv_cache records must be contiguous on the last dim");
  const int64_t row_stride_resolved =
      kv_cache.stride(1) * kv_cache.element_size();
  const int64_t page_stride_resolved =
      kv_cache.stride(0) * kv_cache.element_size();
  TORCH_CHECK(row_stride_resolved >= KV_RECORD_BYTES &&
                  row_stride_resolved % 32 == 0,
              "row_stride must be >= 544 and 32B aligned");
  TORCH_CHECK(page_stride_resolved >= block_size * row_stride_resolved &&
                  page_stride_resolved % 32 == 0,
              "page stride must be >= BS*row_stride and 32B aligned");
  const int64_t needed = (pages - 1) * page_stride_resolved +
                         (block_size - 1) * row_stride_resolved +
                         KV_RECORD_BYTES;
  const int64_t storage_bytes = static_cast<int64_t>(
      kv_cache.storage().nbytes() -
      kv_cache.storage_offset() * kv_cache.element_size());
  TORCH_CHECK(storage_bytes >= needed,
              "kv_cache backing storage ", storage_bytes,
              " < physical write range ", needed);
  return {page_stride_resolved, row_stride_resolved, storage_bytes};
}

static void CheckScales(int64_t t, const at::Tensor &descale_x, const at::Tensor &descale_wqa,
                        const at::Tensor &descale_wqb, const at::Tensor &descale_wkv) {
  CheckND(descale_x, at::ScalarType::Float8_e8m0fnu, {t, H / 64, 2},
          "descale_x");
  CheckND(descale_wqa, at::ScalarType::Float8_e8m0fnu, {R, H / 64, 2},
          "descale_wqa");
  CheckND(descale_wqb, at::ScalarType::Float8_e8m0fnu, {N * D, R / 64, 2},
          "descale_wqb");
  CheckND(descale_wkv, at::ScalarType::Float8_e8m0fnu, {D, H / 64, 2},
          "descale_wkv");
}

static at::Tensor QrNormBuffer(const c10::optional<at::Tensor> &qr_norm, const at::Tensor &x, int64_t t) {
  if (qr_norm.has_value()) {
    TORCH_CHECK(qr_norm->device() == x.device() &&
                    qr_norm->scalar_type() == at::ScalarType::BFloat16 &&
                    qr_norm->dim() == 2 && qr_norm->size(0) == t &&
                    qr_norm->size(1) == R && qr_norm->is_contiguous(),
                "qr_norm must be contiguous BF16 [T,1280] on the same NPU");
    gym_torch::CheckNDStorage(*qr_norm, "qr_norm");
    return *qr_norm;
  }
  return at::empty({0, R}, x.options().dtype(at::ScalarType::BFloat16));
}

static Result Run(const at::Tensor &x, const at::Tensor &wqa,
                  const at::Tensor &wqb, const at::Tensor &wkv,
                  const at::Tensor &descale_x, const at::Tensor &descale_wqa,
                  const at::Tensor &descale_wqb, const at::Tensor &descale_wkv,
                  const at::Tensor &norm_weight_qr,
                  const at::Tensor &norm_weight_kv,
                  const at::Tensor &rope_sin, const at::Tensor &rope_cos,
                  const at::Tensor &cache_index, const at::Tensor &kv_cache,
                  const c10::optional<at::Tensor> &qr_norm, double norm_eps) {
  TORCH_CHECK(x.dim() == 2 && x.size(1) == H &&
                  x.scalar_type() == at::ScalarType::Float8_e4m3fn &&
                  x.device().is_privateuseone(),
              "x must be a [T,5120] E4M3 NPU tensor");
  const c10::OptionalDeviceGuard guard(x.device());
  const int64_t t = x.size(0);
  TORCH_CHECK(t >= 1 && t <= 262144, "T outside [1,262144]");
  TORCH_CHECK(std::isfinite(norm_eps) && norm_eps > 0 &&
                  norm_eps <= std::numeric_limits<float>::max(),
              "norm_eps must be finite and positive");

  CheckND(x, at::ScalarType::Float8_e4m3fn, {t, H}, "x");
  CheckNZ(wqa, {R, H}, "wqa");
  CheckNZ(wqb, {N * D, R}, "wqb");
  CheckNZ(wkv, {D, H}, "wkv");
  CheckScales(t, descale_x, descale_wqa, descale_wqb, descale_wkv);
  CheckND(norm_weight_qr, at::ScalarType::Float, {R}, "norm_weight_qr");
  CheckND(norm_weight_kv, at::ScalarType::Float, {D}, "norm_weight_kv");
  CheckND(rope_sin, at::ScalarType::Float, {t, DR}, "rope_sin");
  CheckND(rope_cos, at::ScalarType::Float, {t, DR}, "rope_cos");
  CheckND(cache_index, at::ScalarType::Long, {t}, "cache_index");
  const auto cache = CheckCache(kv_cache);
  const auto qr_norm_buf = QrNormBuffer(qr_norm, x, t);
  const int64_t emit_qr_norm = qr_norm_buf.numel() != 0;
  std::vector<at::Tensor> inputs = {
      x, wqa, wqb, wkv, descale_x, descale_wqa, descale_wqb, descale_wkv,
      norm_weight_qr, norm_weight_kv, rope_sin, rope_cos, cache_index,
      kv_cache, qr_norm_buf};
  CheckNoGrad(inputs);

  auto q = at::empty({t, N, D}, x.options().dtype(at::ScalarType::BFloat16));
  auto qr = at::empty({t, R}, x.options().dtype(at::ScalarType::Float8_e4m3fn));
  auto descale_qr =
      at::empty({t, R / 64, 2}, x.options().dtype(at::ScalarType::Float8_e8m0fnu));
  gym_torch::CheckNDStorage(q, "q");
  gym_torch::CheckNDStorage(qr, "qr");
  gym_torch::CheckNDStorage(descale_qr, "descale_qr");
  std::vector<at::Tensor> outputs = {q, qr, descale_qr};

  EnqueueAttnPrologue(inputs, outputs, norm_eps, cache.pageStride, cache.rowStride,
                      cache.bytes, emit_qr_norm);
  return {q, qr, descale_qr, kv_cache};
}

static Result Meta(const at::Tensor &x, const at::Tensor &wqa,
                   const at::Tensor &wqb, const at::Tensor &wkv,
                   const at::Tensor &descale_x, const at::Tensor &descale_wqa,
                   const at::Tensor &descale_wqb, const at::Tensor &descale_wkv,
                   const at::Tensor &norm_weight_qr,
                   const at::Tensor &norm_weight_kv,
                   const at::Tensor &rope_sin, const at::Tensor &rope_cos,
                   const at::Tensor &cache_index, const at::Tensor &kv_cache,
                   const c10::optional<at::Tensor> &qr_norm, double norm_eps) {
  TORCH_CHECK(x.dim() == 2, "x must be rank-2");
  (void)wqa; (void)wqb; (void)wkv; (void)descale_x; (void)descale_wqa;
  (void)descale_wqb; (void)descale_wkv; (void)norm_weight_qr;
  (void)norm_weight_kv; (void)rope_sin; (void)rope_cos; (void)cache_index;
  (void)norm_eps; (void)qr_norm;
  const auto t = x.sym_size(0);
  return {at::empty_symint({t, N, D}, x.options().dtype(at::ScalarType::BFloat16)),
          at::empty_symint({t, R}, x.options().dtype(at::ScalarType::Float8_e4m3fn)),
          at::empty_symint({t, R / 64, 2},
                           x.options().dtype(at::ScalarType::Float8_e8m0fnu)),
          kv_cache};
}

} // namespace

TORCH_LIBRARY_FRAGMENT(pypto_gym, m) {
  m.def("attn_prologue(Tensor x, Tensor wqa, Tensor wqb, Tensor wkv, "
        "Tensor descale_x, Tensor descale_wqa, Tensor descale_wqb, "
        "Tensor descale_wkv, Tensor norm_weight_qr, Tensor norm_weight_kv, "
        "Tensor rope_sin, Tensor rope_cos, Tensor cache_index, "
        "Tensor(a!) kv_cache, Tensor(b!)? qr_norm=None, *, float norm_eps=1e-6) -> "
        "(Tensor, Tensor, Tensor, Tensor(a!))");
}
TORCH_LIBRARY_IMPL(pypto_gym, PrivateUse1, m) {
  m.impl("attn_prologue", TORCH_FN(Run));
}
TORCH_LIBRARY_IMPL(pypto_gym, Meta, m) {
  m.impl("attn_prologue", TORCH_FN(Meta));
}
