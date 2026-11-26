/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <limits>
#include "engram_hash_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
struct CompileInfo {};
static bool ValidDimensions(int64_t t, int64_t g, int64_t l, int64_t n) {
    return t >= 0 && g >= 2 && l > 0 && n > 0 && n <= std::numeric_limits<int64_t>::max() / (g - 1);
}

static ge::graphStatus ReadDimensions(gert::TilingContext *ctx, int64_t &t, int64_t &g, int64_t &l, int64_t &n) {
    const auto *ids = ctx->GetRequiredInputShape(0);
    const auto *mult = ctx->GetRequiredInputShape(1);
    const auto *vocab = ctx->GetRequiredInputShape(2);
    const auto *offs = ctx->GetRequiredInputShape(3);
    if (!ids || !mult || !vocab || !offs) return ge::GRAPH_FAILED;
    const auto &x = ids->GetStorageShape();
    const auto &m = mult->GetStorageShape();
    const auto &v = vocab->GetStorageShape();
    const auto &o = offs->GetStorageShape();
    if (x.GetDimNum()!=2 || m.GetDimNum()!=2 || v.GetDimNum()!=3 || o.GetDimNum()!=2)
        return ge::GRAPH_FAILED;
    t=x.GetDim(0); g=x.GetDim(1); l=m.GetDim(0); n=v.GetDim(2);
    if (!ValidDimensions(t, g, l, n)
        || m.GetDim(1)!=g || v.GetDim(0)!=l || v.GetDim(1)!=g-1
        || o.GetDim(0)!=l || o.GetDim(1)!=(g-1)*n) return ge::GRAPH_FAILED;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling(gert::TilingContext *ctx)
{
    if (!ctx || !ctx->GetPlatformInfo()) return ge::GRAPH_FAILED;
    int64_t t, g, l, n;
    if (ReadDimensions(ctx, t, g, l, n) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    const auto *mask=ctx->GetOptionalInputShape(4);
    if (mask && (mask->GetStorageShape().GetDimNum()!=1 || mask->GetStorageShape().GetDim(0)!=t))
        return ge::GRAPH_FAILED;
    auto *td=ctx->GetTilingData<EngramHashTiling>();
    auto *ws=ctx->GetWorkspaceSizes(1);
    if (!td || !ws) return ge::GRAPH_FAILED;
    td->tokens=t; td->layers=l; td->ngram=g; td->tables=n;
    ctx->GetRawTilingData()->SetDataSize(sizeof(*td));
    const auto cores=platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo()).GetCoreNumAiv();
    if (!cores) return ge::GRAPH_FAILED;
    constexpr int64_t lanes = 128;
    const int64_t rowTiles = t / lanes + (t % lanes != 0);
    const bool standardTables = n == 8 && (g == 3 || g == 4);
    constexpr int64_t singleTokenKeyOffset = 2;
    int64_t geometry = 0;
    int64_t jobs = t;
    if (standardTables && mask != nullptr && t == 1 && l == 2) {
        geometry = g + singleTokenKeyOffset;
        jobs = 1;
    } else if (standardTables && rowTiles * 4 >= cores) {
        geometry = g;
        jobs = rowTiles;
    } else if (g <= 16 && n <= 32) {
        // Match the SIMT dispatch in engram_hash.py. Clamp before multiplying
        // by the layer count; only the physical launch width is needed here.
        const int64_t tiles = (t / lanes) * n + ((t % lanes) * n + lanes - 1) / lanes;
        jobs = std::min<int64_t>(tiles, cores) * std::min<int64_t>(l, cores);
    }
    ctx->SetBlockDim(std::clamp<int64_t>(jobs, 1, cores));
    ctx->SetTilingKey(GET_TPL_TILING_KEY(static_cast<uint64_t>(mask != nullptr),
                                       static_cast<uint64_t>(geometry)));
    ws[0]=0;
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus Parse(gert::TilingParseContext *) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(EngramHash).Tiling(Tiling).TilingParse<CompileInfo>(Parse);
}
