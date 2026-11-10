/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mega_mhc_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>

namespace optiling {
static bool ShapeIs(const gert::StorageShape *shape, std::initializer_list<int64_t> dims)
{
    if (shape == nullptr || shape->GetStorageShape().GetDimNum() != dims.size()) return false;
    size_t i = 0;
    for (auto dim : dims) if (shape->GetStorageShape().GetDim(i++) != dim) return false;
    return true;
}
static size_t Align(size_t bytes) { return (bytes + 511) / 512 * 512; }
struct MegaMhcAttrs {
    const float *hc = nullptr;
    const float *pre = nullptr;
    const float *post = nullptr;
    const float *sink = nullptr;
    const float *norm = nullptr;
    const int64_t *repeat = nullptr;
    const int64_t *batch = nullptr;
};

static bool ReadAttrs(gert::TilingContext *ctx, MegaMhcAttrs &attr)
{
    const auto *attrs = ctx->GetAttrs();
    if (attrs == nullptr) return false;
    attr.hc = attrs->GetFloat(0);
    attr.pre = attrs->GetFloat(1);
    attr.post = attrs->GetFloat(2);
    attr.sink = attrs->GetFloat(3);
    attr.norm = attrs->GetFloat(5);
    attr.repeat = attrs->GetInt(4);
    attr.batch = attrs->GetInt(6);
    if (!attr.hc || !attr.pre || !attr.post || !attr.sink || !attr.norm ||
        !attr.repeat || !attr.batch) return false;
    for (float value : {*attr.hc, *attr.pre, *attr.post, *attr.sink, *attr.norm})
        if (!std::isfinite(value) || value < 0) return false;
    if (*attr.hc <= 0 || *attr.norm <= 0 || *attr.repeat < 1 || *attr.repeat > 65535 ||
        (*attr.batch != 0 && *attr.batch != 16)) return false;
    return true;
}

static ge::graphStatus CheckInputs(gert::TilingContext *ctx, int64_t t, int64_t h)
{
    const std::initializer_list<int64_t> shapes[] = {
        {t,h}, {t,4,h}, {t,4}, {t,4,1}, {t,4,4}, {24,4*h}, {3}, {24}, {h}};
    const ge::DataType types[] = {ge::DT_BF16,ge::DT_BF16,ge::DT_FLOAT,ge::DT_FLOAT,
        ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_BF16};
    for (size_t i = 0; i < 9; ++i) {
        const auto *desc = ctx->GetInputDesc(i);
        if (!ShapeIs(ctx->GetInputShape(i), shapes[i]) || desc == nullptr ||
            desc->GetDataType() != types[i] || desc->GetStorageFormat() != ge::FORMAT_ND)
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutputs(gert::TilingContext *ctx, int64_t t, int64_t h)
{
    const std::initializer_list<int64_t> shapes[] = {{t,4,h},{t,4,1},{t,4,4},{t,h},{t,4}};
    const ge::DataType types[] = {ge::DT_BF16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_BF16,ge::DT_FLOAT};
    for (size_t i = 0; i < 5; ++i) {
        const auto *desc = ctx->GetOutputDesc(i);
        if (!ShapeIs(ctx->GetOutputShape(i), shapes[i]) || desc == nullptr ||
            desc->GetDataType() != types[i]) return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BuildTiling(gert::TilingContext *ctx, int64_t t, int64_t h,
                                   const MegaMhcAttrs &attr)
{
    const bool b8 = *attr.batch == 0 && h == 7168 && t >= 4096;
    const int64_t jobs = b8 ? 27 : (h == 7168 ? 7 : 8);
    const int64_t rows = std::min(t, 2 * (b8 ? 8 : 16) * jobs);
    const int64_t shards = h / 256;
    // Cross-core barriers require exactly one resident AIC group per width shard.
    auto platform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    if (platform.GetCoreNumAic() < shards) return ge::GRAPH_FAILED;
    const size_t invOffset = Align(rows * shards * 32 * sizeof(float));
    const size_t flagsOffset = invOffset + Align(rows * 16 * sizeof(float));
    size_t *workspace = ctx->GetWorkspaceSizes(1);
    workspace[0] = std::max<size_t>(1024, flagsOffset + Align((shards + 1) * 16 * sizeof(int32_t)));
    MegaMhcTilingData data;
    data.set_tokens(t); data.set_rows(rows);
    data.set_inv_offset(invOffset); data.set_flags_offset(flagsOffset);
    data.set_repeat(*attr.repeat); data.set_hc_eps(*attr.hc); data.set_pre_eps(*attr.pre);
    data.set_post_scale(*attr.post); data.set_sink_eps(*attr.sink); data.set_norm_eps(*attr.norm);
    // Key packs candidate INDEX, not literal value: Width[0:2], BlockJobs[3:4].
    ctx->SetTilingKey((h / 1024 - 1) | ((b8 ? 2 : (h == 7168 ? 0 : 1)) << 3));
    ctx->SetBlockDim(static_cast<uint32_t>(shards));
    data.SaveToBuffer(ctx->GetRawTilingData()->GetData(), ctx->GetRawTilingData()->GetCapacity());
    ctx->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling(gert::TilingContext *ctx)
{
    const auto *x = ctx->GetInputShape(0);
    if (x == nullptr || x->GetStorageShape().GetDimNum() != 2) return ge::GRAPH_FAILED;
    const int64_t t = x->GetStorageShape().GetDim(0), h = x->GetStorageShape().GetDim(1);
    if (t < 0 || h < 1024 || h > 7168 || h % 1024 != 0 ||
        t > std::numeric_limits<int64_t>::max() / (4 * h)) return ge::GRAPH_FAILED;
    if (CheckInputs(ctx, t, h) != ge::GRAPH_SUCCESS ||
        CheckOutputs(ctx, t, h) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    MegaMhcAttrs attr;
    if (!ReadAttrs(ctx, attr)) return ge::GRAPH_FAILED;
    return BuildTiling(ctx, t, h, attr);
}
IMPL_OP_OPTILING(MegaMhc).Tiling(Tiling);
} // namespace optiling
