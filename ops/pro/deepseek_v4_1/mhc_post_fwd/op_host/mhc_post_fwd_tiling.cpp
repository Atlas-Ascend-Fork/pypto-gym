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
 * \file mhc_post_fwd_tiling.cpp
 * \brief MhcPostFwd host tiling.
 *
 * Mirrors the Python wrapper validation:
 *   x                BF16 [n0, n1, h]
 *   residual         BF16 [n0, n1, MHC, h]
 *   post_layer_mix   FP32 [n0, n1, MHC, 1]
 *   comb_res_mix     FP32 [n0, n1, MHC, MHC]
 *   out              BF16 [n0, n1, MHC, h]
 *
 * h_blk = align(h if h <= 4096 else min(ceil(h/2), 4096), 64) is the compile-time
 * hidden tile width; it is shipped as the tiling key (index = h_blk/64 - 1).
 */

#include "mhc_post_fwd_tiling.h"
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
constexpr int64_t VL = 64;
constexpr int64_t MAX_H_BLK = 4096;
constexpr int64_t MIN_WORKSPACE = 1024;

enum InputIndex {
    X = 0,
    RESIDUAL = 1,
    POST_LAYER_MIX = 2,
    COMB_RES_MIX = 3,
};

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    const int64_t divisor = (b > 0) ? b : 1;
    return a / divisor + ((a % divisor > 0) ? 1 : 0);
}

inline int64_t HBlkOf(int64_t hidden)
{
    const int64_t blk = (hidden <= MAX_H_BLK) ? hidden : std::min(CeilDiv(hidden, 2), MAX_H_BLK);
    return CeilDiv(blk, VL) * VL;
}

class TilingMhcPostFwdFunc {
public:
    explicit TilingMhcPostFwdFunc(gert::TilingContext *context) : tilingContext_(context) {}

    ge::graphStatus RunTiling();

private:
    ge::DataType GetInputDtype(size_t index) const
    {
        const auto *desc = tilingContext_->GetInputDesc(index);
        return desc == nullptr ? ge::DT_UNDEFINED : desc->GetDataType();
    }

    gert::TilingContext *tilingContext_ = nullptr;
    MhcPostFwdTilingData tilingData_;
    int64_t nTok_ = 0;
    int64_t hidden_ = 0;
    int64_t hBlk_ = 0;
};

ge::graphStatus TilingMhcPostFwdFunc::RunTiling()
{
    const auto *xShape = tilingContext_->GetInputShape(X);
    const auto *resShape = tilingContext_->GetInputShape(RESIDUAL);
    const auto *postShape = tilingContext_->GetInputShape(POST_LAYER_MIX);
    const auto *combShape = tilingContext_->GetInputShape(COMB_RES_MIX);
    if (xShape == nullptr || resShape == nullptr || postShape == nullptr || combShape == nullptr) {
        OP_LOGE(tilingContext_->GetNodeName(), "mhc_post_fwd: input shape is null.");
        return ge::GRAPH_FAILED;
    }
    const auto &xs = xShape->GetStorageShape();
    const auto &rs = resShape->GetStorageShape();
    const auto &ps = postShape->GetStorageShape();
    const auto &cs = combShape->GetStorageShape();
    if (xs.GetDimNum() != 3 || rs.GetDimNum() != 4 || ps.GetDimNum() != 4 || cs.GetDimNum() != 4) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mhc_post_fwd: unexpected ranks x=%zu residual=%zu post=%zu comb=%zu.",
                xs.GetDimNum(), rs.GetDimNum(), ps.GetDimNum(), cs.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    const int64_t n0 = rs.GetDim(0);
    const int64_t n1 = rs.GetDim(1);
    const int64_t mhc = rs.GetDim(2);
    hidden_ = rs.GetDim(3);
    if (mhc != MHC) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mhc_post_fwd: residual dim2 (mhc) must be %ld, got %ld.", MHC, mhc);
        return ge::GRAPH_FAILED;
    }
    if (xs.GetDim(0) != n0 || xs.GetDim(1) != n1 || xs.GetDim(2) != hidden_) {
        OP_LOGE(tilingContext_->GetNodeName(), "mhc_post_fwd: x shape must be [n0,n1,h].");
        return ge::GRAPH_FAILED;
    }
    if (ps.GetDim(0) != n0 || ps.GetDim(1) != n1 || ps.GetDim(2) != MHC || ps.GetDim(3) != 1) {
        OP_LOGE(tilingContext_->GetNodeName(), "mhc_post_fwd: post_layer_mix must be [n0,n1,4,1].");
        return ge::GRAPH_FAILED;
    }
    if (cs.GetDim(0) != n0 || cs.GetDim(1) != n1 || cs.GetDim(2) != MHC || cs.GetDim(3) != MHC) {
        OP_LOGE(tilingContext_->GetNodeName(), "mhc_post_fwd: comb_res_mix must be [n0,n1,4,4].");
        return ge::GRAPH_FAILED;
    }
    if (GetInputDtype(X) != ge::DT_BF16 || GetInputDtype(RESIDUAL) != ge::DT_BF16) {
        OP_LOGE(tilingContext_->GetNodeName(), "mhc_post_fwd: x/residual must be BF16.");
        return ge::GRAPH_FAILED;
    }
    if (GetInputDtype(POST_LAYER_MIX) != ge::DT_FLOAT ||
        GetInputDtype(COMB_RES_MIX) != ge::DT_FLOAT) {
        OP_LOGE(tilingContext_->GetNodeName(), "mhc_post_fwd: post/comb must be FP32.");
        return ge::GRAPH_FAILED;
    }
    if (hidden_ <= 0 || hidden_ % VL != 0) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mhc_post_fwd: hidden must be a positive multiple of %ld, got %ld.", VL, hidden_);
        return ge::GRAPH_FAILED;
    }

    hBlk_ = HBlkOf(hidden_);
    const int64_t keyIndex = hBlk_ / VL - 1;
    if (hBlk_ < VL || hBlk_ > MAX_H_BLK || keyIndex < 0 || keyIndex >= 64) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mhc_post_fwd: unsupported hidden tile width h_blk=%ld.", hBlk_);
        return ge::GRAPH_FAILED;
    }

    nTok_ = n0 * n1;
    tilingData_.set_n_tok(nTok_);
    tilingData_.set_h(hidden_);

    auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext_->GetPlatformInfo());
    const int64_t total = nTok_ * CeilDiv(hidden_, hBlk_);
    int64_t coreNum = static_cast<int64_t>(platformInfo.GetCoreNumAiv());
    if (coreNum <= 0) {
        coreNum = std::max<int64_t>(1, static_cast<int64_t>(platformInfo.GetCoreNumAic()) * 2);
    }
    int64_t blockDim = std::min<int64_t>(std::max<int64_t>(total, 1), coreNum);
    if (blockDim <= 0) {
        blockDim = 1;
    }

    size_t *workspace = tilingContext_->GetWorkspaceSizes(1);
    workspace[0] = MIN_WORKSPACE;

    tilingContext_->SetTilingKey(static_cast<int64_t>(keyIndex));
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
    TilingMhcPostFwdFunc tilingObject(context);
    auto ret = tilingObject.RunTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "mhc_post_fwd tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MhcPostFwd).Tiling(TilingFunc);

} // namespace optiling
