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
 * \file indexer_prologue_k_fused_tiling.cpp
 * \brief IndexerPrologueK fused host tiling.
 */

#include "indexer_prologue_k_fused_tiling.h"
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

constexpr int64_t LATENT = 0;
constexpr int64_t WK = 1;
constexpr int64_t NORM_WEIGHT = 2;
constexpr int64_t ROPE_SIN = 3;
constexpr int64_t ROPE_COS = 4;
constexpr int64_t CACHE_INDEX = 5;
constexpr int64_t K_CACHE = 6;
constexpr int64_t K_SCALE_CACHE = 7;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const char *node = context->GetNodeName();

    const auto *latentShape = context->GetInputShape(LATENT);
    const auto *wkShape = context->GetInputShape(WK);
    const auto *normShape = context->GetInputShape(NORM_WEIGHT);
    const auto *sinShape = context->GetInputShape(ROPE_SIN);
    const auto *cosShape = context->GetInputShape(ROPE_COS);
    const auto *ciShape = context->GetInputShape(CACHE_INDEX);
    const auto *cacheShape = context->GetInputShape(K_CACHE);
    const auto *scaleShape = context->GetInputShape(K_SCALE_CACHE);
    if (latentShape == nullptr || wkShape == nullptr || normShape == nullptr ||
        sinShape == nullptr || cosShape == nullptr || ciShape == nullptr ||
        cacheShape == nullptr) {
        OP_LOGE(node, "indexer_prologue_k: missing input shape.");
        return ge::GRAPH_FAILED;
    }

    const auto &ls = latentShape->GetStorageShape();
    const auto &ws = wkShape->GetStorageShape();
    const auto &ns = normShape->GetStorageShape();
    const auto &ss = sinShape->GetStorageShape();
    const auto &cs = cosShape->GetStorageShape();
    const auto &cis = ciShape->GetStorageShape();
    const auto &caches = cacheShape->GetStorageShape();

    if (ls.GetDimNum() != 2 || ws.GetDimNum() != 2 || ls.GetDim(1) != ws.GetDim(1)) {
        OP_LOGE(node, "indexer_prologue_k: latent/wk shapes mismatch.");
        return ge::GRAPH_FAILED;
    }
    const int64_t t = ls.GetDim(0);
    const int64_t headDim = ls.GetDim(1);
    const int64_t dim = ws.GetDim(0);
    if (t < 1 || t > T_MAX) {
        OP_LOGE(node, "indexer_prologue_k: T=%ld outside [1,%ld].", t, T_MAX);
        return ge::GRAPH_FAILED;
    }
    if (headDim <= 0 || headDim % 16 != 0) {
        OP_LOGE(node, "indexer_prologue_k: H must be a positive multiple of 16.");
        return ge::GRAPH_FAILED;
    }
    if (dim < 32 || dim % 32 != 0 || dim > MAX_DIM) {
        OP_LOGE(node, "indexer_prologue_k: D must be a multiple of 32 in [32,2048].");
        return ge::GRAPH_FAILED;
    }
    if (ns.GetDimNum() != 1 || ns.GetDim(0) != dim || ss.GetDimNum() != 2 ||
        cs.GetDimNum() != 2 || ss.GetDim(0) != t || cs.GetDim(0) != t ||
        ss.GetDim(1) != cs.GetDim(1)) {
        OP_LOGE(node, "indexer_prologue_k: norm_weight/rope shapes mismatch.");
        return ge::GRAPH_FAILED;
    }
    const int64_t dr = ss.GetDim(1);
    if (dr <= 0 || dr > dim || dr % 2 != 0) {
        OP_LOGE(node, "indexer_prologue_k: Dr must be positive even and <= D.");
        return ge::GRAPH_FAILED;
    }
    if (cis.GetDimNum() != 1 || cis.GetDim(0) != t) {
        OP_LOGE(node, "indexer_prologue_k: cache_index must be [T].");
        return ge::GRAPH_FAILED;
    }

    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(node, "indexer_prologue_k: missing attributes.");
        return ge::GRAPH_FAILED;
    }
    const int64_t *storageModePtr = attrs->GetInt(0);
    const float *normEpsPtr = attrs->GetFloat(1);
    const int64_t *gPtr = attrs->GetInt(2);
    const bool *writeScalePtr = attrs->GetBool(3);
    if (storageModePtr == nullptr || normEpsPtr == nullptr) {
        OP_LOGE(node, "indexer_prologue_k: storage_mode/norm_eps are required.");
        return ge::GRAPH_FAILED;
    }
    const int64_t storageMode = *storageModePtr;
    const int64_t groupSize = (gPtr == nullptr) ? -1 : *gPtr;
    if (storageMode != 0 && storageMode != 1) {
        OP_LOGE(node, "indexer_prologue_k: storage_mode must be 0 or 1.");
        return ge::GRAPH_FAILED;
    }

    const int64_t valueBytes = dim / 2;
    const int64_t scaleBytes = dim / FP4_GROUP_SIZE;
    const int64_t writeScale = (storageMode == 0 && writeScalePtr != nullptr && *writeScalePtr) ? 1 : 0;
    const int64_t scaleBoxes = writeScale != 0 ? scaleBytes : 1;
    const int64_t cacheWidth = (storageMode == 0) ? valueBytes
        : groupSize * (valueBytes + scaleBytes);
    if (storageMode == 1 && groupSize <= 0) {
        OP_LOGE(node, "indexer_prologue_k: mode 1 requires combined_block_size > 0.");
        return ge::GRAPH_FAILED;
    }
    if (caches.GetDimNum() != 4 || caches.GetDim(2) != 1 || caches.GetDim(3) != cacheWidth) {
        OP_LOGE(node, "indexer_prologue_k: k_cache shape mismatch.");
        return ge::GRAPH_FAILED;
    }
    const int64_t cacheBlocks = caches.GetDim(0);
    const int64_t cacheSlots = caches.GetDim(1);
    if (writeScale != 0) {
        const auto &scs = scaleShape->GetStorageShape();
        if (scs.GetDimNum() != 4 || scs.GetDim(0) != cacheBlocks || scs.GetDim(1) != cacheSlots ||
            scs.GetDim(2) != 1 || scs.GetDim(3) != scaleBytes) {
            OP_LOGE(node, "indexer_prologue_k: k_scale_cache shape mismatch.");
            return ge::GRAPH_FAILED;
        }
    }

    const int64_t baseN = (dim % BASE_N_WIDE == 0) ? BASE_N_WIDE
        : ((dim % BASE_N == 0) ? BASE_N : NZ_ALIGN);
    const int64_t baseK = (headDim % BASE_K == 0) ? BASE_K : NZ_ALIGN;
    const int64_t nmode = (baseN == BASE_N_WIDE) ? 0 : ((baseN == BASE_N) ? 1 : 2);
    const int64_t kmode = (baseK == BASE_K) ? 0 : 1;
    const int64_t dimCap = std::max<int64_t>(512, NextPow2(dim));

    IndexerPrologueKFusedTilingData tilingData;
    tilingData.set_t(t);
    tilingData.set_head_dim(headDim);
    tilingData.set_dim(dim);
    tilingData.set_dr(dr);
    tilingData.set_storage_mode(storageMode);
    tilingData.set_combined_block_size(groupSize);
    tilingData.set_write_scale_cache(writeScale);
    tilingData.set_cache_blocks(cacheBlocks);
    tilingData.set_cache_slots(cacheSlots);
    tilingData.set_cache_width(cacheWidth);
    tilingData.set_scale_boxes(scaleBoxes);
    tilingData.set_epsilon(*normEpsPtr);

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = static_cast<size_t>(t) * static_cast<size_t>(dim) * sizeof(uint16_t);

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    int64_t blockDim = std::min<int64_t>(static_cast<int64_t>(platform.GetCoreNumAic()),
                                         static_cast<int64_t>(platform.GetCoreNumAiv()) / 2);
    if (blockDim <= 0) {
        blockDim = 1;
    }

    context->SetTilingKey(nmode | (kmode << 2) | (DimCapIndex(dimCap) << 3));
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetDynUBufSize(DynUBufSize(dimCap));
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(IndexerPrologueK).Tiling(TilingFunc);

} // namespace optiling
