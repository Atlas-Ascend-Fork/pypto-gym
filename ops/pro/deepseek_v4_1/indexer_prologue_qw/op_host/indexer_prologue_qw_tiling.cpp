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
 * \file indexer_prologue_qw_tiling.cpp
 * \brief IndexerPrologueQw host tiling.
 *
 * Only ``T`` is dynamic.  ``x`` may carry more rows than ``qr``; the extra rows
 * are ignored.  The flat workspace is consumed only by the split-K W reduction
 * when ``ceil(T/128) <= 12``.
 */

#include "indexer_prologue_qw_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

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

constexpr int64_t X_INDEX = 0;
constexpr int64_t QR_INDEX = 1;
constexpr int64_t WQB_INDEX = 2;
constexpr int64_t WW_INDEX = 3;
constexpr int64_t DESCALE_QR_INDEX = 4;
constexpr int64_t DESCALE_WQB_INDEX = 5;
constexpr int64_t ROPE_SIN_INDEX = 6;
constexpr int64_t ROPE_COS_INDEX = 7;

constexpr int64_t DIM = 5120;
constexpr int64_t QLORA = 1280;
constexpr int64_t N_HEADS = 32;
constexpr int64_t DR = 64;
constexpr int64_t Q_N = 4096;
constexpr int64_t BASE_M = 128;
constexpr int64_t SPLIT_W_TILES = 12;
constexpr int64_t HB_K = 16;
constexpr int64_t T_MAX = 262144;

static bool ShapeIs(const gert::Shape &shape, std::initializer_list<int64_t> dims)
{
    if (shape.GetDimNum() != dims.size()) {
        return false;
    }
    size_t i = 0;
    for (int64_t dim : dims) {
        if (shape.GetDim(i) != dim) {
            return false;
        }
        ++i;
    }
    return true;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const char *node = context->GetNodeName();

    const auto *xShape = context->GetInputShape(X_INDEX);
    const auto *qrShape = context->GetInputShape(QR_INDEX);
    const auto *wqbShape = context->GetInputShape(WQB_INDEX);
    const auto *wwShape = context->GetInputShape(WW_INDEX);
    const auto *dqrShape = context->GetInputShape(DESCALE_QR_INDEX);
    const auto *dwqbShape = context->GetInputShape(DESCALE_WQB_INDEX);
    const auto *sinShape = context->GetInputShape(ROPE_SIN_INDEX);
    const auto *cosShape = context->GetInputShape(ROPE_COS_INDEX);
    if (xShape == nullptr || qrShape == nullptr || wqbShape == nullptr || wwShape == nullptr ||
        dqrShape == nullptr || dwqbShape == nullptr || sinShape == nullptr || cosShape == nullptr) {
        OP_LOGE(node, "indexer_prologue_qw: missing input shape.");
        return ge::GRAPH_FAILED;
    }

    const auto &xs = xShape->GetStorageShape();
    const auto &qrs = qrShape->GetStorageShape();
    const auto &wqbs = wqbShape->GetStorageShape();
    const auto &wws = wwShape->GetStorageShape();
    const auto &dqrs = dqrShape->GetStorageShape();
    const auto &dwqbs = dwqbShape->GetStorageShape();
    const auto &sins = sinShape->GetStorageShape();
    const auto &coss = cosShape->GetStorageShape();

    if (qrs.GetDimNum() != 2 || qrs.GetDim(1) != QLORA) {
        OP_LOGE(node, "indexer_prologue_qw: qr must be [T,%ld].", QLORA);
        return ge::GRAPH_FAILED;
    }
    const int64_t t = qrs.GetDim(0);
    if (t < 1 || t > T_MAX) {
        OP_LOGE(node, "indexer_prologue_qw: T=%ld outside [1,%ld].", t, T_MAX);
        return ge::GRAPH_FAILED;
    }
    if (xs.GetDimNum() != 2 || xs.GetDim(1) != DIM || xs.GetDim(0) < t) {
        OP_LOGE(node, "indexer_prologue_qw: x must be [>=T,%ld].", DIM);
        return ge::GRAPH_FAILED;
    }
    if (!ShapeIs(wqbs, {Q_N, QLORA}) || wws.GetDimNum() != 2 ||
        wws.GetDim(0) != N_HEADS || wws.GetDim(1) != DIM) {
        OP_LOGE(node, "indexer_prologue_qw: wqb must be [4096,%ld] and ww [%ld,%ld].",
                QLORA, N_HEADS, DIM);
        return ge::GRAPH_FAILED;
    }
    if (!ShapeIs(dqrs, {t, 20, 2}) || !ShapeIs(dwqbs, {Q_N, 20, 2})) {
        OP_LOGE(node, "indexer_prologue_qw: descale shapes mismatch.");
        return ge::GRAPH_FAILED;
    }
    if (!ShapeIs(sins, {t, DR}) || !ShapeIs(coss, {t, DR})) {
        OP_LOGE(node, "indexer_prologue_qw: rope tables must be [T,%ld].", DR);
        return ge::GRAPH_FAILED;
    }

    float softmaxScale = 0.08838834764831845F;
    const auto *attrs = context->GetAttrs();
    if (attrs != nullptr) {
        if (const float *value = attrs->GetFloat(0)) {
            softmaxScale = *value;
        }
    }

    IndexerPrologueQwTilingData tilingData;
    tilingData.set_t(t);
    tilingData.set_softmax_scale(softmaxScale);

    const int64_t nTiles = (t + BASE_M - 1) / BASE_M;
    const int64_t wsRows = (nTiles <= SPLIT_W_TILES) ? HB_K * t : 1;
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = static_cast<size_t>(wsRows) * static_cast<size_t>(N_HEADS) * sizeof(float);

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    int64_t blockDim = static_cast<int64_t>(platform.GetCoreNumAic());
    if (blockDim <= 0) {
        blockDim = static_cast<int64_t>(platform.GetCoreNumAiv()) / 2;
    }
    if (blockDim <= 0) {
        blockDim = 1;
    }

    context->SetTilingKey(0);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(IndexerPrologueQw).Tiling(TilingFunc);

} // namespace optiling
