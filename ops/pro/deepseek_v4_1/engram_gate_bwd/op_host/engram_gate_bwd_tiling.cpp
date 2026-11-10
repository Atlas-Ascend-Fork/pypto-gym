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
 * \file engram_gate_bwd_tiling.cpp
 * \brief EngramGateBwd host tiling.
 *
 * t/dim are runtime axes; hc is baked to 4 by the kernel.  The delivered binary
 * is AIV-only (new PyPTO codegen), so SetBlockDim is the AIV count.  The kernel
 * derives its worker count as block_num * subblock_num (subblock_num == 1 for
 * AIV-only), which must equal the tiling-side AIV count and stay a multiple
 * of hc.
 */

#include "engram_gate_bwd_tiling.h"
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

constexpr int64_t HC = 4;
constexpr int64_t STEP = 128;
constexpr int64_t MAXC = 7168;
constexpr int64_t NPART = 16;
constexpr int64_t MIN_WORKSPACE = 1024;

enum InputIndex {
    GRAD_OUT = 0,
    X = 1,
    KEY = 2,
    VALUE = 3,
    WEIGHT = 4,
    SAVED = 5,
    IMAGE_MASK = 6,
};

class TilingEngramGateBwdFunc {
public:
    explicit TilingEngramGateBwdFunc(gert::TilingContext *context) : tilingContext_(context) {}

    ge::graphStatus RunTiling();

private:
    ge::DataType GetInputDtype(size_t index) const
    {
        const auto *desc = tilingContext_->GetInputDesc(index);
        return desc == nullptr ? ge::DT_UNDEFINED : desc->GetDataType();
    }

    gert::TilingContext *tilingContext_ = nullptr;
    EngramGateBwdTilingData tilingData_;
};

ge::graphStatus TilingEngramGateBwdFunc::RunTiling()
{
    const auto *gradOutShape = tilingContext_->GetInputShape(GRAD_OUT);
    const auto *xShape = tilingContext_->GetInputShape(X);
    const auto *keyShape = tilingContext_->GetInputShape(KEY);
    const auto *valueShape = tilingContext_->GetInputShape(VALUE);
    const auto *weightShape = tilingContext_->GetInputShape(WEIGHT);
    const auto *savedShape = tilingContext_->GetInputShape(SAVED);
    const auto *maskShape = tilingContext_->GetInputShape(IMAGE_MASK);
    if (gradOutShape == nullptr || xShape == nullptr || keyShape == nullptr ||
        valueShape == nullptr || weightShape == nullptr || savedShape == nullptr ||
        maskShape == nullptr) {
        OP_LOGE(tilingContext_->GetNodeName(), "engram_gate_bwd: input shape is null.");
        return ge::GRAPH_FAILED;
    }
    const auto &gos = gradOutShape->GetStorageShape();
    const auto &xs = xShape->GetStorageShape();
    const auto &ks = keyShape->GetStorageShape();
    const auto &vs = valueShape->GetStorageShape();
    const auto &ws = weightShape->GetStorageShape();
    const auto &ss = savedShape->GetStorageShape();
    const auto &ms = maskShape->GetStorageShape();
    if (gos.GetDimNum() != 3 || xs.GetDimNum() != 3 || ks.GetDimNum() != 3 ||
        vs.GetDimNum() != 2 || ws.GetDimNum() != 2 || ss.GetDimNum() != 2 ||
        ms.GetDimNum() != 1) {
        OP_LOGE(tilingContext_->GetNodeName(), "engram_gate_bwd: unexpected input ranks.");
        return ge::GRAPH_FAILED;
    }

    const int64_t t = xs.GetDim(0);
    const int64_t hc = xs.GetDim(1);
    const int64_t dim = xs.GetDim(2);
    if (t < 1 || hc != HC || dim < 1 || dim > MAXC) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "engram_gate_bwd: invalid t=%ld hc=%ld dim=%ld (hc=%ld, dim <= %ld).",
                t, hc, dim, HC, MAXC);
        return ge::GRAPH_FAILED;
    }
    if (dim % (HC * STEP) != 0) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "engram_gate_bwd: dim=%ld must be a multiple of hc * STEP = %ld.",
                dim, HC * STEP);
        return ge::GRAPH_FAILED;
    }
    if (gos.GetDim(0) != t || gos.GetDim(1) != hc || gos.GetDim(2) != dim ||
        ks.GetDim(0) != t || ks.GetDim(1) != hc || ks.GetDim(2) != dim) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "engram_gate_bwd: grad_out/key shape must equal x shape.");
        return ge::GRAPH_FAILED;
    }
    if (vs.GetDim(0) != t || vs.GetDim(1) != dim) {
        OP_LOGE(tilingContext_->GetNodeName(), "engram_gate_bwd: value must be [t,dim].");
        return ge::GRAPH_FAILED;
    }
    if (ws.GetDim(0) != hc || ws.GetDim(1) != dim) {
        OP_LOGE(tilingContext_->GetNodeName(), "engram_gate_bwd: weight must be [hc,dim].");
        return ge::GRAPH_FAILED;
    }
    if (ss.GetDim(0) != t || ss.GetDim(1) != 4 * hc || ms.GetDim(0) != t) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "engram_gate_bwd: saved must be [t,4*hc] and image_mask [t].");
        return ge::GRAPH_FAILED;
    }
    if (GetInputDtype(GRAD_OUT) != ge::DT_BF16 || GetInputDtype(X) != ge::DT_BF16 ||
        GetInputDtype(KEY) != ge::DT_BF16 || GetInputDtype(VALUE) != ge::DT_BF16 ||
        GetInputDtype(WEIGHT) != ge::DT_FLOAT || GetInputDtype(SAVED) != ge::DT_FLOAT ||
        GetInputDtype(IMAGE_MASK) != ge::DT_BOOL) {
        OP_LOGE(tilingContext_->GetNodeName(), "engram_gate_bwd: input dtype mismatch.");
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext_->GetPlatformInfo());
    int64_t aiv = static_cast<int64_t>(platformInfo.GetCoreNumAiv());
    if (aiv <= 0) {
        aiv = std::max<int64_t>(1, static_cast<int64_t>(platformInfo.GetCoreNumAic()) * 2);
    }
    int64_t nc = std::min<int64_t>(aiv, NPART * HC);
    nc = (nc / HC) * HC;
    if (nc <= 0) {
        OP_LOGE(tilingContext_->GetNodeName(), "engram_gate_bwd: launch width rounded to zero.");
        return ge::GRAPH_FAILED;
    }
    const int64_t blockDim = std::max<int64_t>(1, nc);

    tilingData_.set_t(t);
    tilingData_.set_dim(dim);

    size_t *workspace = tilingContext_->GetWorkspaceSizes(1);
    workspace[0] = MIN_WORKSPACE;

    tilingContext_->SetTilingKey(0);
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
    TilingEngramGateBwdFunc tilingObject(context);
    auto ret = tilingObject.RunTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "engram_gate_bwd tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EngramGateBwd).Tiling(TilingFunc);

} // namespace optiling
