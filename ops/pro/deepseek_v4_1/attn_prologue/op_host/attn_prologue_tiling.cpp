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
 * \file attn_prologue_tiling.cpp
 * \brief AttnPrologue host tiling.
 *
 * C++ port of the Python host wrapper
 * (``op_kernel/attn_prologue.py::attn_prologue``). The two must stay
 * equivalent:
 *   - T is x.dim(0); n_slots is kv_cache.dim(0) * dim(1);
 *   - grid = max(1, min(cube_num, vector_num / 2));
 *   - decode split-K workspace bands use the same gate as the kernel:
 *       n_tiles * NA1 * 2 <= grid  -> 2 bands, else 1 band;
 *   - workspace holds ws_qa [ws_rows, R] followed by ws_kv [ws_rows, D] fp32;
 *   - tiling key selects the weight L2 policy: 0 (T <= BASE_M, NotAllocKeep)
 *     or 1 (T > BASE_M, default) — matches AttnPrologueTilingKey.Profile.
 */

#include "attn_prologue_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>

#ifndef OP_LOGE
#include <cstdio>
#define OP_LOGE(nodeName, fmt, ...) \
  do { \
    (void)(nodeName); \
    std::printf(fmt "\n", ##__VA_ARGS__); \
  } while (0)
#endif

namespace optiling {

constexpr int64_t H = 5120;
constexpr int64_t R = 1280;
constexpr int64_t N_HEADS = 64;
constexpr int64_t D = 512;
constexpr int64_t DR = 64;
constexpr int64_t BASE_M = 128;
constexpr int64_t NA1 = 14;
constexpr int64_t KV_RECORD_BYTES = 544;
constexpr int64_t T_MAX = 262144;

enum InputIndex {
    X = 0,
    WQA,
    WQB,
    WKV,
    DESCALE_X,
    DESCALE_WQA,
    DESCALE_WQB,
    DESCALE_WKV,
    NORM_WEIGHT_QR,
    NORM_WEIGHT_KV,
    ROPE_SIN,
    ROPE_COS,
    CACHE_INDEX,
    KV_CACHE,
};

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return b > 0 ? (a + b - 1) / b : 0;
}

class TilingAttnPrologueFunc {
public:
    explicit TilingAttnPrologueFunc(gert::TilingContext *context) : context_(context) {}
    ge::graphStatus RunTiling();

private:
    ge::graphStatus ParseShapesAndAttrs();
    ge::graphStatus ReadTokens();
    ge::graphStatus CheckWeights();
    ge::graphStatus CheckScales();
    ge::graphStatus CheckRopeAndNorm();
    ge::graphStatus ReadCache();
    ge::graphStatus ReadModeAttrs();
    ge::graphStatus ReadStrides();
    static bool CheckShape(const gert::StorageShape *shape,
                           std::initializer_list<int64_t> dims);
    static ge::DataType GetInputDtype(gert::TilingContext *context, size_t index);

    gert::TilingContext *context_ = nullptr;
    AttnPrologueTilingData tiling_data_;
    int64_t t_ = 0;
    int64_t pages_ = 0;
    int64_t cache_rows_ = 0;
    int64_t block_size_ = 0;
    int64_t block_stride_ = 0;
    int64_t row_stride_ = 0;
    int64_t storage_bytes_ = 0;
    int64_t grid_ = 1;
    float norm_eps_ = 1e-6f;
    int64_t emit_qr_norm_ = 0;
};

ge::DataType TilingAttnPrologueFunc::GetInputDtype(
    gert::TilingContext *context, size_t index)
{
    const auto *desc = context->GetInputDesc(index);
    return desc == nullptr ? ge::DT_UNDEFINED : desc->GetDataType();
}

bool TilingAttnPrologueFunc::CheckShape(
    const gert::StorageShape *shape, std::initializer_list<int64_t> dims)
{
    if (shape == nullptr) {
        return false;
    }
    const auto &storage = shape->GetStorageShape();
    if (dims.size() != storage.GetDimNum()) {
        return false;
    }
    int64_t i = 0;
    for (int64_t dim : dims) {
        if (storage.GetDim(static_cast<size_t>(i)) != dim) {
            return false;
        }
        ++i;
    }
    return true;
}

ge::graphStatus TilingAttnPrologueFunc::ReadTokens()
{
    const auto *xShape = context_->GetInputShape(X);
    if (xShape == nullptr) {
        OP_LOGE(context_->GetNodeName(),
                "attn_prologue: x shape is null.");
        return ge::GRAPH_FAILED;
    }
    const auto &xs = xShape->GetStorageShape();
    if (xs.GetDimNum() != 2 || xs.GetDim(1) != H) {
        OP_LOGE(context_->GetNodeName(),
                "attn_prologue: x must be [T,%ld], got rank %zu.", H,
                xs.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    t_ = xs.GetDim(0);
    if (t_ < 1 || t_ > T_MAX) {
        OP_LOGE(context_->GetNodeName(),
                "attn_prologue: T=%ld outside [1,%ld].", t_, T_MAX);
        return ge::GRAPH_FAILED;
    }
    if (GetInputDtype(context_, X) != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE(context_->GetNodeName(),
                "attn_prologue: x must be FLOAT8_E4M3FN.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingAttnPrologueFunc::CheckWeights()
{
    if (!CheckShape(context_->GetInputShape(WQA), {R, H}) ||
        !CheckShape(context_->GetInputShape(WQB), {N_HEADS * D, R}) ||
        !CheckShape(context_->GetInputShape(WKV), {D, H})) {
        OP_LOGE(context_->GetNodeName(),
                "attn_prologue: wqa/wqb/wkv shape mismatch.");
        return ge::GRAPH_FAILED;
    }
    for (uint32_t index : {static_cast<uint32_t>(WQA), static_cast<uint32_t>(WQB),
                           static_cast<uint32_t>(WKV)}) {
        if (GetInputDtype(context_, index) != ge::DT_FLOAT8_E4M3FN) {
            OP_LOGE(context_->GetNodeName(),
                    "attn_prologue: weights must be FLOAT8_E4M3FN.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingAttnPrologueFunc::CheckScales()
{
    if (!CheckShape(context_->GetInputShape(DESCALE_X), {t_, H / 64, 2}) ||
        !CheckShape(context_->GetInputShape(DESCALE_WQA), {R, H / 64, 2}) ||
        !CheckShape(context_->GetInputShape(DESCALE_WQB), {N_HEADS * D, R / 64, 2}) ||
        !CheckShape(context_->GetInputShape(DESCALE_WKV), {D, H / 64, 2})) {
        OP_LOGE(context_->GetNodeName(),
                "attn_prologue: descale shape mismatch.");
        return ge::GRAPH_FAILED;
    }
    for (uint32_t index : {static_cast<uint32_t>(DESCALE_X),
                           static_cast<uint32_t>(DESCALE_WQA),
                           static_cast<uint32_t>(DESCALE_WQB),
                           static_cast<uint32_t>(DESCALE_WKV)}) {
        if (GetInputDtype(context_, index) != ge::DT_FLOAT8_E8M0) {
            OP_LOGE(context_->GetNodeName(),
                    "attn_prologue: descale tensors must be FLOAT8_E8M0.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingAttnPrologueFunc::CheckRopeAndNorm()
{
    if (!CheckShape(context_->GetInputShape(NORM_WEIGHT_QR), {R}) ||
        !CheckShape(context_->GetInputShape(NORM_WEIGHT_KV), {D}) ||
        GetInputDtype(context_, NORM_WEIGHT_QR) != ge::DT_FLOAT ||
        GetInputDtype(context_, NORM_WEIGHT_KV) != ge::DT_FLOAT) {
        OP_LOGE(context_->GetNodeName(),
                "attn_prologue: norm weight shape/dtype mismatch.");
        return ge::GRAPH_FAILED;
    }
    if (!CheckShape(context_->GetInputShape(ROPE_SIN), {t_, DR}) ||
        !CheckShape(context_->GetInputShape(ROPE_COS), {t_, DR}) ||
        GetInputDtype(context_, ROPE_SIN) != ge::DT_FLOAT ||
        GetInputDtype(context_, ROPE_COS) != ge::DT_FLOAT) {
        OP_LOGE(context_->GetNodeName(),
                "attn_prologue: rope table shape/dtype mismatch.");
        return ge::GRAPH_FAILED;
    }
    if (!CheckShape(context_->GetInputShape(CACHE_INDEX), {t_}) ||
        GetInputDtype(context_, CACHE_INDEX) != ge::DT_INT64) {
        OP_LOGE(context_->GetNodeName(),
                "attn_prologue: cache_index must be [T] INT64.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingAttnPrologueFunc::ReadCache()
{
    const auto *kvShape = context_->GetInputShape(KV_CACHE);
    if (kvShape == nullptr) return ge::GRAPH_FAILED;
    const auto &ks = kvShape->GetStorageShape();
    if (ks.GetDimNum() != 4 || ks.GetDim(2) != 1 ||
        ks.GetDim(3) < KV_RECORD_BYTES || ks.GetDim(0) <= 0 || ks.GetDim(1) <= 0) {
        OP_LOGE(context_->GetNodeName(),
                "attn_prologue: kv_cache must be [P,BS,1,C>=%ld] with P,BS>0.",
                KV_RECORD_BYTES);
        return ge::GRAPH_FAILED;
    }
    if (GetInputDtype(context_, KV_CACHE) != ge::DT_UINT8) {
        OP_LOGE(context_->GetNodeName(),
                "attn_prologue: kv_cache must be UINT8.");
        return ge::GRAPH_FAILED;
    }
    pages_ = ks.GetDim(0);
    block_size_ = ks.GetDim(1);
    cache_rows_ = pages_ * block_size_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingAttnPrologueFunc::ReadModeAttrs()
{
    const auto *attrs = context_->GetAttrs();
    const float *eps = (attrs == nullptr) ? nullptr : attrs->GetFloat(0);
    const int64_t *emit_attr = (attrs == nullptr) ? nullptr : attrs->GetInt(4);
    norm_eps_ = (eps == nullptr) ? 1e-6f : *eps;
    emit_qr_norm_ = (emit_attr == nullptr || *emit_attr == 0) ? 0 : 1;
    if (!std::isfinite(norm_eps_) || norm_eps_ <= 0.0f) {
        OP_LOGE(context_->GetNodeName(),
                "attn_prologue: norm_eps must be finite and positive.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingAttnPrologueFunc::ReadStrides()
{
    const auto *attrs = context_->GetAttrs();
    const int64_t *page_attr = (attrs == nullptr) ? nullptr : attrs->GetInt(1);
    const int64_t *row_attr = (attrs == nullptr) ? nullptr : attrs->GetInt(2);
    const int64_t *storage_attr = (attrs == nullptr) ? nullptr : attrs->GetInt(3);
    row_stride_ = (row_attr == nullptr || *row_attr == 0)
        ? KV_RECORD_BYTES : *row_attr;
    if (row_stride_ < KV_RECORD_BYTES || row_stride_ % 32 != 0) {
        OP_LOGE(context_->GetNodeName(),
                "attn_prologue: invalid row_stride=%ld (>= %ld and 32B aligned).",
                row_stride_, KV_RECORD_BYTES);
        return ge::GRAPH_FAILED;
    }
    block_stride_ = (page_attr == nullptr || *page_attr == 0)
        ? block_size_ * row_stride_ : *page_attr;
    if (block_stride_ < block_size_ * row_stride_ || block_stride_ % 32 != 0) {
        OP_LOGE(context_->GetNodeName(),
                "attn_prologue: invalid block_stride=%ld (>= BS*row_stride=%ld and 32B aligned).",
                block_stride_, block_size_ * row_stride_);
        return ge::GRAPH_FAILED;
    }
    const int64_t needed = (pages_ - 1) * block_stride_
        + (block_size_ - 1) * row_stride_ + KV_RECORD_BYTES;
    storage_bytes_ = (storage_attr == nullptr || *storage_attr <= 0)
        ? needed : *storage_attr;
    if (storage_bytes_ < needed) {
        OP_LOGE(context_->GetNodeName(),
                "attn_prologue: cache storage %ld < physical write range %ld.",
                storage_bytes_, needed);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingAttnPrologueFunc::ParseShapesAndAttrs()
{
    if (ReadTokens() != ge::GRAPH_SUCCESS || CheckWeights() != ge::GRAPH_SUCCESS ||
        CheckScales() != ge::GRAPH_SUCCESS || CheckRopeAndNorm() != ge::GRAPH_SUCCESS ||
        ReadCache() != ge::GRAPH_SUCCESS || ReadModeAttrs() != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    return ReadStrides();
}

ge::graphStatus TilingAttnPrologueFunc::RunTiling()
{
    auto ret = ParseShapesAndAttrs();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    auto platform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    int64_t cube = static_cast<int64_t>(platform.GetCoreNumAic());
    int64_t vector = static_cast<int64_t>(platform.GetCoreNumAiv());
    if (cube <= 0 || vector < cube) {
        cube = 32;
        vector = 64;
    }
    grid_ = std::max<int64_t>(1, std::min<int64_t>(cube, vector / 2));

    const int64_t n_tiles = CeilDiv(t_, BASE_M);
    const bool split_k = (n_tiles * NA1 * 2 <= grid_);
    const int64_t ws_rows = split_k ? (2 * n_tiles * BASE_M) : t_;

    tiling_data_.set_t(t_);
    tiling_data_.set_cache_rows(cache_rows_);
    tiling_data_.set_block_size(block_size_);
    tiling_data_.set_block_stride(block_stride_);
    tiling_data_.set_row_stride(row_stride_);
    tiling_data_.set_ws_rows(ws_rows);
    tiling_data_.set_norm_eps(norm_eps_);
    tiling_data_.set_emit_qr_norm(emit_qr_norm_);

    size_t *workspace = context_->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        OP_LOGE(context_->GetNodeName(),
                "attn_prologue: workspace size pointer is null.");
        return ge::GRAPH_FAILED;
    }
    const size_t ws_bytes =
        static_cast<size_t>(ws_rows) * static_cast<size_t>(R + D) * sizeof(float);
    workspace[0] = (ws_bytes + 511U) & ~static_cast<size_t>(511U);

    context_->SetTilingKey(((t_ <= BASE_M) ? 0U : 1U)
                           | (emit_qr_norm_ ? 2U : 0U));
    tiling_data_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                              context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling_data_.GetDataSize());
    context_->SetBlockDim(static_cast<uint32_t>(grid_));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingAttnPrologueFunc tiling_object(context);
    auto ret = tiling_object.RunTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "attn_prologue tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AttnPrologue).Tiling(TilingFunc);

} // namespace optiling
