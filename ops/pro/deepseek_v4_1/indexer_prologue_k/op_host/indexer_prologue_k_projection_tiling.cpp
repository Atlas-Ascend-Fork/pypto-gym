/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "indexer_prologue_k_projection_tiling.h"
#include "indexer_prologue_k_plan.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

#ifndef OP_LOGE
#include <cstdio>
#define OP_LOGE(nodeName, fmt, ...) \
  do { \
    (void)(nodeName); \
    std::printf(fmt "\n", ##__VA_ARGS__); \
  } while (0)
#endif

namespace optiling {

using namespace indexer_prologue_k_plan;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const char *node = context->GetNodeName();
    const auto *latentShape = context->GetInputShape(0);
    const auto *wkShape = context->GetInputShape(1);
    if (latentShape == nullptr || wkShape == nullptr) {
        OP_LOGE(node, "indexer_prologue_k_projection: missing input shape.");
        return ge::GRAPH_FAILED;
    }
    const auto &ls = latentShape->GetStorageShape();
    const auto &ws = wkShape->GetStorageShape();
    if (ls.GetDimNum() != 2 || ws.GetDimNum() != 2 || ls.GetDim(1) != ws.GetDim(1)) {
        OP_LOGE(node, "indexer_prologue_k_projection: latent/wk shapes mismatch.");
        return ge::GRAPH_FAILED;
    }
    const int64_t t = ls.GetDim(0);
    const int64_t headDim = ls.GetDim(1);
    const int64_t dim = ws.GetDim(0);
    if (t < 1 || t > T_MAX) {
        OP_LOGE(node, "indexer_prologue_k_projection: T=%ld outside [1,%ld].", t, T_MAX);
        return ge::GRAPH_FAILED;
    }
    if (headDim <= 0 || headDim % 16 != 0) {
        OP_LOGE(node, "indexer_prologue_k_projection: H must be a positive multiple of 16.");
        return ge::GRAPH_FAILED;
    }
    if (dim < 32 || dim % 32 != 0 || dim > MAX_DIM) {
        OP_LOGE(node, "indexer_prologue_k_projection: D must be a multiple of 32 in [32,2048].");
        return ge::GRAPH_FAILED;
    }

    IndexerPrologueKProjectionTilingData tilingData;
    tilingData.set_t(t);
    tilingData.set_head_dim(headDim);
    tilingData.set_dim(dim);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    const auto axes = GetProjectionAxes(dim, headDim);
    const int64_t nmode = NModeOf(axes.base_n);
    const int64_t kmode = KModeOf(axes.base_k);
    const int64_t mTiles = CeilDiv(t, BASE_M);
    const int64_t totalTiles = mTiles * axes.n_tiles;
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    int64_t blockDim = std::min<int64_t>(static_cast<int64_t>(platform.GetCoreNumAic()), totalTiles);
    if (blockDim <= 0) {
        blockDim = 1;
    }
    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        OP_LOGE(node, "indexer_prologue_k_projection: workspace descriptor is null.");
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 32;
    context->SetTilingKey(nmode | (kmode << 2));
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(IndexerPrologueKProjection).Tiling(TilingFunc);

} // namespace optiling
