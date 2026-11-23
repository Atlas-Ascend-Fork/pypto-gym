/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "indexer_prologue_k_postprocess_tiling.h"
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
    const auto *projectedShape = context->GetInputShape(0);
    const auto *normShape = context->GetInputShape(1);
    const auto *sinShape = context->GetInputShape(2);
    const auto *cosShape = context->GetInputShape(3);
    const auto *ciShape = context->GetInputShape(4);
    const auto *cacheShape = context->GetInputShape(5);
    const auto *scaleShape = context->GetInputShape(6);
    if (projectedShape == nullptr || normShape == nullptr || sinShape == nullptr ||
        cosShape == nullptr || ciShape == nullptr || cacheShape == nullptr) {
        OP_LOGE(node, "indexer_prologue_k_postprocess: missing input shape.");
        return ge::GRAPH_FAILED;
    }
    const auto &ps = projectedShape->GetStorageShape();
    const auto &ns = normShape->GetStorageShape();
    const auto &ss = sinShape->GetStorageShape();
    const auto &cs = cosShape->GetStorageShape();
    const auto &cis = ciShape->GetStorageShape();
    const auto &caches = cacheShape->GetStorageShape();
    if (ps.GetDimNum() != 2) {
        OP_LOGE(node, "indexer_prologue_k_postprocess: projected must be [T,D].");
        return ge::GRAPH_FAILED;
    }
    const int64_t t = ps.GetDim(0);
    const int64_t dim = ps.GetDim(1);
    if (t < 1 || t > T_MAX) {
        OP_LOGE(node, "indexer_prologue_k_postprocess: T=%ld outside [1,%ld].", t, T_MAX);
        return ge::GRAPH_FAILED;
    }
    if (dim < 32 || dim % 32 != 0 || dim > MAX_DIM) {
        OP_LOGE(node, "indexer_prologue_k_postprocess: D must be a multiple of 32 in [32,2048].");
        return ge::GRAPH_FAILED;
    }
    if (ns.GetDimNum() != 1 || ns.GetDim(0) != dim || ss.GetDimNum() != 2 ||
        cs.GetDimNum() != 2 || ss.GetDim(0) != t || cs.GetDim(0) != t ||
        ss.GetDim(1) != cs.GetDim(1)) {
        OP_LOGE(node, "indexer_prologue_k_postprocess: norm_weight/rope shapes mismatch.");
        return ge::GRAPH_FAILED;
    }
    const int64_t dr = ss.GetDim(1);
    if (dr <= 0 || dr > dim || dr % 2 != 0) {
        OP_LOGE(node, "indexer_prologue_k_postprocess: Dr must be positive even and <= D.");
        return ge::GRAPH_FAILED;
    }
    if (cis.GetDimNum() != 1 || cis.GetDim(0) != t) {
        OP_LOGE(node, "indexer_prologue_k_postprocess: cache_index must be [T].");
        return ge::GRAPH_FAILED;
    }

    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(node, "indexer_prologue_k_postprocess: missing attributes.");
        return ge::GRAPH_FAILED;
    }
    const int64_t *storageModePtr = attrs->GetInt(0);
    const float *normEpsPtr = attrs->GetFloat(1);
    const int64_t *gPtr = attrs->GetInt(2);
    const bool *writeScalePtr = attrs->GetBool(3);
    if (storageModePtr == nullptr || normEpsPtr == nullptr) {
        OP_LOGE(node, "indexer_prologue_k_postprocess: storage_mode/norm_eps are required.");
        return ge::GRAPH_FAILED;
    }
    const int64_t storageMode = *storageModePtr;
    const int64_t groupSize = (gPtr == nullptr) ? -1 : *gPtr;
    if (storageMode != 0 && storageMode != 1) {
        OP_LOGE(node, "indexer_prologue_k_postprocess: storage_mode must be 0 or 1.");
        return ge::GRAPH_FAILED;
    }
    const int64_t valueBytes = dim / E2M1_VALUES_PER_BYTE;
    const int64_t scaleBytes = dim / FP4_GROUP_SIZE;
    const int64_t writeScale = (storageMode == 0 && writeScalePtr != nullptr && *writeScalePtr) ? 1 : 0;
    const int64_t scaleBoxes = writeScale != 0 ? scaleBytes : 1;
    const int64_t cacheWidth = (storageMode == 0) ? valueBytes
        : groupSize * (valueBytes + scaleBytes);
    if (storageMode == 1 && groupSize <= 0) {
        OP_LOGE(node, "indexer_prologue_k_postprocess: mode 1 requires combined_block_size > 0.");
        return ge::GRAPH_FAILED;
    }
    if (caches.GetDimNum() != 4 || caches.GetDim(2) != 1 || caches.GetDim(3) != cacheWidth) {
        OP_LOGE(node, "indexer_prologue_k_postprocess: k_cache shape mismatch.");
        return ge::GRAPH_FAILED;
    }
    const int64_t cacheBlocks = caches.GetDim(0);
    const int64_t cacheSlots = caches.GetDim(1);
    if (writeScale != 0) {
        const auto &scs = scaleShape->GetStorageShape();
        if (scs.GetDimNum() != 4 || scs.GetDim(0) != cacheBlocks || scs.GetDim(1) != cacheSlots ||
            scs.GetDim(2) != 1 || scs.GetDim(3) != scaleBytes) {
            OP_LOGE(node, "indexer_prologue_k_postprocess: k_scale_cache shape mismatch.");
            return ge::GRAPH_FAILED;
        }
    }

    const int64_t dimCap = std::max<int64_t>(512, NextPow2(dim));
    IndexerPrologueKPostprocessTilingData tilingData;
    tilingData.set_t(t);
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
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const int64_t workItems = (storageMode == 0) ? t : CeilDiv(t, groupSize);
    int64_t blockDim = std::min<int64_t>(static_cast<int64_t>(platform.GetCoreNumAiv()), workItems);
    if (blockDim <= 0) {
        blockDim = 1;
    }
    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        OP_LOGE(node, "indexer_prologue_k_postprocess: workspace descriptor is null.");
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 32;
    context->SetTilingKey(DimCapIndex(dimCap));
    context->SetDynUBufSize(DynUBufSize(dimCap));
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(IndexerPrologueKPostprocess).Tiling(TilingFunc);

} // namespace optiling
