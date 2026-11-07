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
 * \file kv_compress_epilog_v2_tiling.cpp
 * \brief KvCompressEpilogV2 host tiling.
 *
 * C++ port of the Python host validation/planning in
 * op_kernel/kv_compress_epilog_v2.py (_validate / _cache_geometry /
 * _validate_storage):
 *   - the cache geometry, quant format/group, scale rounding and 2D/4D layout
 *     are resolved into the tiling struct and the packed tiling key;
 *   - storage_bytes is the caller-declared backing storage (attr) when
 *     positive, otherwise the geometry lower bound.
 *
 * The delivery-form tiling key keeps the compile-time specializations that
 * change generated code: Kind / Group / RoundScale / Slot64 / Width /
 * LargeScale / Layout4.  Width selects the on-chip tile capacity and must match
 * the kernel's per-dim tile width; fixing it to 8192 measurably slows small-D
 * rows.  The aclnn interface normalizes slot_mapping to INT64, so Slot64 is
 * always 1 here; the jit path also selects Slot64=0 for INT32 slots.
 */

#include "kv_compress_epilog_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cmath>
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

constexpr int64_t MAX_D = 8192;
constexpr int64_t MAX_WORKERS = 40;
constexpr int64_t MIN_WORKSPACE = 1024;

enum InputIndex {
    CACHE = 0,
    X = 1,
    SLOT_MAPPING = 2,
};

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    const int64_t divisor = (b > 0) ? b : 1;
    return (a + divisor - 1) / divisor;
}

inline int64_t Ceil32(int64_t value)
{
    return CeilDiv(value, 32) * 32;
}

class TilingKvCompressEpilogV2Func {
public:
    explicit TilingKvCompressEpilogV2Func(gert::TilingContext *context) : tilingContext_(context) {}

    ge::graphStatus RunTiling();

private:
    ge::DataType GetInputDtype(size_t index) const
    {
        const auto *desc = tilingContext_->GetInputDesc(index);
        return desc == nullptr ? ge::DT_UNDEFINED : desc->GetDataType();
    }

    const gert::StorageShape *GetInputShape(size_t index) const
    {
        return tilingContext_->GetInputShape(index);
    }

    ge::graphStatus ParseShapesAndAttrs();
    ge::graphStatus ValidateCache();

    gert::TilingContext *tilingContext_ = nullptr;
    KvCompressEpilogV2TilingData tilingData_;

    int64_t tokens_ = 0;
    int64_t dim_ = 0;
    int64_t cacheRows_ = 0;
    int64_t cacheCols_ = 0;
    int64_t blockSize_ = 1;
    int64_t blockStride_ = 0;
    int64_t storageBytes_ = 0;
    int64_t kind_ = 0;
    int64_t group_ = 32;
    int64_t mode_ = 2;
    int64_t largeScale_ = 0;
    int64_t layout4_ = 0;
    bool roundScale_ = true;
};

ge::graphStatus TilingKvCompressEpilogV2Func::ParseShapesAndAttrs()
{
    const auto *attrs = tilingContext_->GetAttrs();
    if (attrs != nullptr) {
        if (const int64_t *value = attrs->GetInt(0)) {
            group_ = *value;
        }
        if (const int64_t *value = attrs->GetInt(1)) {
            mode_ = *value;
        }
        if (const bool *value = attrs->GetBool(2)) {
            roundScale_ = *value;
        }
        if (const float *value = attrs->GetFloat(3)) {
            if (std::fabs(*value - 1.0F) > 1.0e-6F) {
                OP_LOGE(tilingContext_->GetNodeName(),
                        "kv_compress_epilog_v2: x_scale must be 1.0, got %f.", static_cast<double>(*value));
                return ge::GRAPH_FAILED;
            }
        }
        if (const int64_t *value = attrs->GetInt(4)) {
            blockStride_ = *value;
        }
        if (const int64_t *value = attrs->GetInt(5)) {
            storageBytes_ = *value;
        }
    }

    if (mode_ != 2 && mode_ != 4) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "kv_compress_epilog_v2: quant_mode must be 2 or 4, got %ld.", mode_);
        return ge::GRAPH_FAILED;
    }
    if (mode_ == 2 && group_ != 32) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "kv_compress_epilog_v2: FP8 quantization requires quant_group_size=32, got %ld.", group_);
        return ge::GRAPH_FAILED;
    }

    const auto *xShape = GetInputShape(X);
    if (xShape == nullptr) {
        OP_LOGE(tilingContext_->GetNodeName(), "kv_compress_epilog_v2: x shape is null.");
        return ge::GRAPH_FAILED;
    }
    const auto &xs = xShape->GetStorageShape();
    if (xs.GetDimNum() != 2) {
        OP_LOGE(tilingContext_->GetNodeName(), "kv_compress_epilog_v2: x must be rank 2 [T,D].");
        return ge::GRAPH_FAILED;
    }
    tokens_ = xs.GetDim(0);
    dim_ = xs.GetDim(1);
    if (GetInputDtype(X) != ge::DT_BF16) {
        OP_LOGE(tilingContext_->GetNodeName(), "kv_compress_epilog_v2: x must be BF16.");
        return ge::GRAPH_FAILED;
    }
    if (tokens_ <= 0 || dim_ <= 0 || dim_ > MAX_D) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "kv_compress_epilog_v2: invalid T=%ld or D=%ld.", tokens_, dim_);
        return ge::GRAPH_FAILED;
    }
    if (group_ != 16 && group_ != 32) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "kv_compress_epilog_v2: quant_group_size must be 16 or 32, got %ld.", group_);
        return ge::GRAPH_FAILED;
    }
    if (dim_ % group_ != 0) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "kv_compress_epilog_v2: D=%ld must be divisible by quant_group_size=%ld.", dim_, group_);
        return ge::GRAPH_FAILED;
    }

    const auto *slotShape = GetInputShape(SLOT_MAPPING);
    if (slotShape == nullptr) {
        OP_LOGE(tilingContext_->GetNodeName(), "kv_compress_epilog_v2: slot_mapping shape is null.");
        return ge::GRAPH_FAILED;
    }
    const auto &ss = slotShape->GetStorageShape();
    if (ss.GetDimNum() != 1 || ss.GetDim(0) != tokens_) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "kv_compress_epilog_v2: slot_mapping must be [T]=[%ld].", tokens_);
        return ge::GRAPH_FAILED;
    }
    if (GetInputDtype(SLOT_MAPPING) != ge::DT_INT64) {
        OP_LOGE(tilingContext_->GetNodeName(), "kv_compress_epilog_v2: slot_mapping must be INT64.");
        return ge::GRAPH_FAILED;
    }

    const ge::DataType cacheDtype = GetInputDtype(CACHE);
    if (cacheDtype == ge::DT_FLOAT8_E4M3FN) {
        kind_ = 0;
    } else if (cacheDtype == ge::DT_FLOAT8_E5M2) {
        kind_ = 1;
    } else if (cacheDtype == ge::DT_UINT8) {
        kind_ = 2;
    } else {
        OP_LOGE(tilingContext_->GetNodeName(),
                "kv_compress_epilog_v2: cache dtype got %d, expected E4M3FN/E5M2/UINT8.",
                static_cast<int>(cacheDtype));
        return ge::GRAPH_FAILED;
    }
    if ((mode_ == 2) != (kind_ < 2)) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "kv_compress_epilog_v2: quant_mode=%ld is incompatible with cache dtype.", mode_);
        return ge::GRAPH_FAILED;
    }

    return ValidateCache();
}

ge::graphStatus TilingKvCompressEpilogV2Func::ValidateCache()
{
    const auto *cacheShape = GetInputShape(CACHE);
    if (cacheShape == nullptr) {
        OP_LOGE(tilingContext_->GetNodeName(), "kv_compress_epilog_v2: cache shape is null.");
        return ge::GRAPH_FAILED;
    }
    const auto &cs = cacheShape->GetStorageShape();

    const int64_t data = (kind_ < 2) ? dim_ : dim_ / 2;
    const int64_t bundle = data + 2 * (dim_ / group_);
    const int64_t aligned = Ceil32(bundle);

    if (cs.GetDimNum() == 2) {
        const int64_t rows = cs.GetDim(0);
        const int64_t cols = cs.GetDim(1);
        if (rows <= 0 || cols < aligned) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "kv_compress_epilog_v2: invalid 2D cache [%ld,%ld], C must be >= %ld.",
                    rows, cols, aligned);
            return ge::GRAPH_FAILED;
        }
        cacheRows_ = rows;
        cacheCols_ = cols;
        blockSize_ = 1;
        blockStride_ = cols;
        layout4_ = 0;
        const int64_t needed = rows * cols;
        if (storageBytes_ > 0 && storageBytes_ < needed) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "kv_compress_epilog_v2: cache storage %ld < physical write range %ld.",
                    storageBytes_, needed);
            return ge::GRAPH_FAILED;
        }
        if (storageBytes_ <= 0) {
            storageBytes_ = needed;
        }
    } else if (cs.GetDimNum() == 4) {
        const int64_t batch = cs.GetDim(0);
        const int64_t blockSize = cs.GetDim(1);
        const int64_t one = cs.GetDim(2);
        const int64_t cols = cs.GetDim(3);
        if (one != 1 || batch <= 0 || blockSize <= 0) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "kv_compress_epilog_v2: invalid 4D cache [%ld,%ld,%ld,%ld].",
                    batch, blockSize, one, cols);
            return ge::GRAPH_FAILED;
        }
        if (cols < bundle) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "kv_compress_epilog_v2: 4D cache C=%ld must be >= bundle=%ld.", cols, bundle);
            return ge::GRAPH_FAILED;
        }
        const int64_t stride = (blockStride_ != 0) ? blockStride_ : blockSize * cols;
        if (stride < blockSize * cols || stride % 32 != 0) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "kv_compress_epilog_v2: invalid block_stride=%ld (>= %ld and 32B aligned).",
                    stride, blockSize * cols);
            return ge::GRAPH_FAILED;
        }
        cacheRows_ = batch * blockSize;
        cacheCols_ = cols;
        blockSize_ = blockSize;
        blockStride_ = stride;
        layout4_ = 1;
        const int64_t needed = (batch - 1) * stride + blockSize * bundle;
        if (storageBytes_ > 0 && storageBytes_ < needed) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "kv_compress_epilog_v2: cache storage %ld < physical write range %ld.",
                    storageBytes_, needed);
            return ge::GRAPH_FAILED;
        }
        if (storageBytes_ <= 0) {
            storageBytes_ = needed;
        }
    } else {
        OP_LOGE(tilingContext_->GetNodeName(),
                "kv_compress_epilog_v2: cache must be rank 2 or 4, got %zu.", cs.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    largeScale_ = (kind_ == 2 && group_ == 16 && dim_ / group_ >= 16) ? 1 : 0;
    if (kind_ < 2 && group_ != 32) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "kv_compress_epilog_v2: FP8 only supports quant_group_size=32.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingKvCompressEpilogV2Func::RunTiling()
{
    auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext_->GetPlatformInfo());
    auto ret = ParseShapesAndAttrs();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    tilingData_.set_tokens(tokens_);
    tilingData_.set_dim(dim_);
    tilingData_.set_cache_rows(cacheRows_);
    tilingData_.set_cache_cols(cacheCols_);
    tilingData_.set_block_size(blockSize_);
    tilingData_.set_block_stride(blockStride_);
    tilingData_.set_storage_bytes(storageBytes_);

    size_t *workspace = tilingContext_->GetWorkspaceSizes(1);
    workspace[0] = MIN_WORKSPACE;

    // Same rule as the jit host: next power of two, floored at 32 (FP8) or
    // 128 (FP4).  Width is a key dimension because it sizes the physical tiles.
    int64_t width = 1;
    while (width < dim_) {
        width <<= 1;
    }
    if (width < ((kind_ < 2) ? 32 : 128)) {
        width = (kind_ < 2) ? 32 : 128;
    }
    int64_t widthIdx = 0;
    for (int64_t w = 32; w < width; w <<= 1) {
        ++widthIdx;
    }

    const int64_t kindIdx = kind_;
    const int64_t groupIdx = (group_ == 16) ? 0 : 1;
    const int64_t roundIdx = (kind_ == 2) ? 1 : (roundScale_ ? 1 : 0);
    const int64_t slot64Idx = 1;  // op def registers INT64 only
    const int64_t largeIdx = largeScale_;
    const int64_t layoutIdx = layout4_;
    // Field bit offsets: Kind 2 | Group 6 | RoundScale 1 | Slot64 1 | Width 14 |
    // LargeScale 1 | Layout4 1.
    const int64_t tilingKey = kindIdx | (groupIdx << 2) | (roundIdx << 8) |
                              (slot64Idx << 9) | (widthIdx << 10) |
                              (largeIdx << 24) | (layoutIdx << 25);
    tilingContext_->SetTilingKey(tilingKey);

    tilingData_.SaveToBuffer(tilingContext_->GetRawTilingData()->GetData(),
                             tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    int64_t blockDim = std::min<int64_t>(tokens_, MAX_WORKERS);
    const uint32_t aivCoreNum = platformInfo.GetCoreNumAiv();
    if (aivCoreNum > 0) {
        blockDim = std::min<int64_t>(blockDim, static_cast<int64_t>(aivCoreNum));
    }
    if (blockDim <= 0) {
        blockDim = 1;
    }
    tilingContext_->SetBlockDim(static_cast<uint32_t>(blockDim));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    TilingKvCompressEpilogV2Func tilingObject(context);
    auto ret = tilingObject.RunTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "kv_compress_epilog_v2 tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KvCompressEpilogV2).Tiling(TilingFunc);

} // namespace optiling
