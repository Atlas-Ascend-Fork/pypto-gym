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
 * \file mhc_pre_fuse_tiling.cpp
 * \brief MhcPreFuse host tiling.
 *
 * Mirrors the Python host selection:
 *   tb    = 8 / 4 / 2 / 1 (largest divisor of T)
 *   h_blk = widest of 512/256/192/128/64 dividing H with H/h_blk >= 5 and
 *           fitting the 5-deep residual stream in 253952 B of UB
 *   r_rem = (H/h_blk - 5) % 5
 *
 * The structural (tb, h_blk, r_rem) tuple is shipped as the tiling key; T, H
 * and the five knobs are runtime tiling fields.
 */

#include "mhc_pre_fuse_tiling.h"
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

constexpr int64_t MHC = 4;
constexpr int64_t MHC2 = 16;
constexpr int64_t MHC3 = 24;
constexpr int64_t VL = 64;
constexpr int64_t PIPE_DEPTH = 5;
constexpr int64_t UB_BYTES = 253952;
constexpr int64_t STREAM_BASE = 0xC000;
constexpr int64_t MIN_WORKSPACE = 1024;

enum InputIndex {
    GEMM_OUT_MUL = 0,
    GEMM_OUT_SQRSUM = 1,
    MHC_SCALE = 2,
    MHC_BASE = 3,
    RESIDUAL = 4,
};

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    const int64_t divisor = (b > 0) ? b : 1;
    return (a + divisor - 1) / divisor;
}

inline int64_t PickTb(int64_t t)
{
    for (int64_t c : {8, 4, 2}) {
        if (t % c == 0) {
            return c;
        }
    }
    return 1;
}

inline int64_t PickHBlk(int64_t hidden, int64_t tb)
{
    for (int64_t c : {512, 256, 192, 128, 64}) {
        if (hidden % c != 0) {
            continue;
        }
        if (hidden / c < PIPE_DEPTH) {
            continue;
        }
        if (STREAM_BASE + PIPE_DEPTH * (tb * MHC * c * 2) +
                PIPE_DEPTH * (tb * c * 2) > UB_BYTES) {
            continue;
        }
        return c;
    }
    return 0;
}

inline int64_t TbIndex(int64_t tb)
{
    switch (tb) {
        case 8: return 0;
        case 4: return 1;
        case 2: return 2;
        default: return 3;
    }
}

inline int64_t HBlkIndex(int64_t hblk)
{
    switch (hblk) {
        case 512: return 0;
        case 256: return 1;
        case 192: return 2;
        case 128: return 3;
        default: return 4;
    }
}

class TilingMhcPreFuseFunc {
public:
    explicit TilingMhcPreFuseFunc(gert::TilingContext *context) : tilingContext_(context) {}

    ge::graphStatus RunTiling();

private:
    ge::DataType GetInputDtype(size_t index) const
    {
        const auto *desc = tilingContext_->GetInputDesc(index);
        return desc == nullptr ? ge::DT_UNDEFINED : desc->GetDataType();
    }

    gert::TilingContext *tilingContext_ = nullptr;
    MhcPreFuseTilingData tilingData_;
};

ge::graphStatus TilingMhcPreFuseFunc::RunTiling()
{
    const auto *resShape = tilingContext_->GetInputShape(RESIDUAL);
    const auto *mulShape = tilingContext_->GetInputShape(GEMM_OUT_MUL);
    const auto *sqShape = tilingContext_->GetInputShape(GEMM_OUT_SQRSUM);
    const auto *scaleShape = tilingContext_->GetInputShape(MHC_SCALE);
    const auto *baseShape = tilingContext_->GetInputShape(MHC_BASE);
    if (resShape == nullptr || mulShape == nullptr || sqShape == nullptr ||
        scaleShape == nullptr || baseShape == nullptr) {
        OP_LOGE(tilingContext_->GetNodeName(), "mhc_pre_fuse: input shape is null.");
        return ge::GRAPH_FAILED;
    }
    const auto &rs = resShape->GetStorageShape();
    const auto &ms = mulShape->GetStorageShape();
    const auto &ss = sqShape->GetStorageShape();
    const auto &scs = scaleShape->GetStorageShape();
    const auto &bs = baseShape->GetStorageShape();
    if (rs.GetDimNum() != 3 || ms.GetDimNum() != 2 || ss.GetDimNum() != 1 ||
        scs.GetDimNum() != 1 || bs.GetDimNum() != 1) {
        OP_LOGE(tilingContext_->GetNodeName(), "mhc_pre_fuse: unexpected input ranks.");
        return ge::GRAPH_FAILED;
    }

    const int64_t t = rs.GetDim(0);
    const int64_t mhc = rs.GetDim(1);
    const int64_t hidden = rs.GetDim(2);
    if (mhc != MHC) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mhc_pre_fuse: residual mix axis must be %ld, got %ld.", MHC, mhc);
        return ge::GRAPH_FAILED;
    }
    if (t < 0 || hidden <= 0) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mhc_pre_fuse: invalid T=%ld or H=%ld.", t, hidden);
        return ge::GRAPH_FAILED;
    }
    if (hidden % VL != 0 || hidden / VL < PIPE_DEPTH) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mhc_pre_fuse: H must be a multiple of %ld with H/%ld >= %ld, got %ld.",
                VL, VL, PIPE_DEPTH, hidden);
        return ge::GRAPH_FAILED;
    }
    if (ms.GetDim(0) != t || ms.GetDim(1) != MHC3) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mhc_pre_fuse: gemm_out_mul must be [T,24] = [%ld,24].", t);
        return ge::GRAPH_FAILED;
    }
    if (ss.GetDim(0) != t || scs.GetDim(0) != 3 || bs.GetDim(0) != MHC3) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mhc_pre_fuse: gemm_out_sqrsum/mhc_scale/mhc_base shapes mismatch.");
        return ge::GRAPH_FAILED;
    }
    if (GetInputDtype(GEMM_OUT_MUL) != ge::DT_FLOAT ||
        GetInputDtype(GEMM_OUT_SQRSUM) != ge::DT_FLOAT ||
        GetInputDtype(MHC_SCALE) != ge::DT_FLOAT ||
        GetInputDtype(MHC_BASE) != ge::DT_FLOAT ||
        GetInputDtype(RESIDUAL) != ge::DT_BF16) {
        OP_LOGE(tilingContext_->GetNodeName(), "mhc_pre_fuse: input dtype mismatch.");
        return ge::GRAPH_FAILED;
    }

    const int64_t tb = PickTb(t);
    if (tb == 0) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mhc_pre_fuse: T=%ld produced a zero token block.", t);
        return ge::GRAPH_FAILED;
    }
    const int64_t hblk = PickHBlk(hidden, tb);
    if (hblk == 0) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mhc_pre_fuse: hidden=%ld is not supported by the residual stream.", hidden);
        return ge::GRAPH_FAILED;
    }
    const int64_t nCh = hidden / hblk;
    const int64_t rRem = (nCh - PIPE_DEPTH) % PIPE_DEPTH;
    const int64_t nT = t / tb;

    float rmsEps = 1.0e-6F;
    float preEps = 1.0e-6F;
    float sinkhornEps = 1.0e-6F;
    float postMult = 1.0F;
    int64_t sinkhornRepeat = 10;
    const auto *attrs = tilingContext_->GetAttrs();
    if (attrs != nullptr) {
        if (const float *p = attrs->GetFloat(0)) {
            rmsEps = *p;
        }
        if (const float *p = attrs->GetFloat(1)) {
            preEps = *p;
        }
        if (const float *p = attrs->GetFloat(2)) {
            sinkhornEps = *p;
        }
        if (const float *p = attrs->GetFloat(3)) {
            postMult = *p;
        }
        if (const int64_t *p = attrs->GetInt(4)) {
            sinkhornRepeat = *p;
        }
    }
    if (sinkhornRepeat < 1) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mhc_pre_fuse: sinkhorn_repeat must be >= 1, got %ld.", sinkhornRepeat);
        return ge::GRAPH_FAILED;
    }

    tilingData_.set_t(t);
    tilingData_.set_h(hidden);
    tilingData_.set_inv(1.0F / static_cast<float>(MHC * hidden));
    tilingData_.set_rms_eps(rmsEps);
    tilingData_.set_pre_eps(preEps);
    tilingData_.set_sinkhorn_eps(sinkhornEps);
    tilingData_.set_post_mult(postMult);
    tilingData_.set_sinkhorn_repeat(sinkhornRepeat);

    auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext_->GetPlatformInfo());
    int64_t coreNum = static_cast<int64_t>(platformInfo.GetCoreNumAiv());
    if (coreNum <= 0) {
        coreNum = std::max<int64_t>(1, static_cast<int64_t>(platformInfo.GetCoreNumAic()) * 2);
    }
    int64_t blockDim = std::min<int64_t>(std::max<int64_t>(nT, 1), coreNum);
    if (blockDim <= 0) {
        blockDim = 1;
    }

    size_t *workspace = tilingContext_->GetWorkspaceSizes(1);
    workspace[0] = MIN_WORKSPACE;

    const int64_t tilingKey = TbIndex(tb) | (HBlkIndex(hblk) << 2) | (rRem << 5);
    tilingContext_->SetTilingKey(tilingKey);
    tilingData_.SaveToBuffer(tilingContext_->GetRawTilingData()->GetData(),
                             tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    tilingContext_->SetBlockDim(static_cast<uint32_t>(blockDim));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    TilingMhcPreFuseFunc tilingObject(context);
    auto ret = tilingObject.RunTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "mhc_pre_fuse tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MhcPreFuse).Tiling(TilingFunc);

} // namespace optiling
