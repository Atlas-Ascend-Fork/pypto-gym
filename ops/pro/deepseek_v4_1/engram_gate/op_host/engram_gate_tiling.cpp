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
 * \file engram_gate_tiling.cpp
 * \brief EngramGate host tiling.
 *
 * t/hc/dim are runtime axes. The kernel binds head = core % hc and strides tokens
 * by nc / hc, so the launched AIV count must be a multiple of hc. The binary is
 * MIX (PyPTO delivery), therefore SetBlockDim is the AIC count and AIV = 2*AIC;
 * choose an even AIV count that is a multiple of hc.
 */

#include "engram_gate_tiling.h"
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

constexpr int64_t CHUNK = 8512;
constexpr int64_t MIN_WORKSPACE = 1024;

inline int64_t ShapeIdOf(int64_t t, int64_t hc, int64_t dim)
{
    if (t == 5 && hc == 4 && dim == 5120) { return 1; }
    if (t == 72 && hc == 4 && dim == 5120) { return 2; }
    if (t == 3 && hc == 4 && dim == 8449) { return 3; }
    if (t == 5 && hc == 4 && dim == 5121) { return 4; }
    if (t == 2 && hc == 4 && dim == 5120) { return 5; }
    return 0;
}

enum InputIndex {
    X = 0,
    KEY = 1,
    VALUE = 2,
    WEIGHT = 3,
    IMAGE_MASK = 4,
};

class TilingEngramGateFunc {
public:
    explicit TilingEngramGateFunc(gert::TilingContext *context) : tilingContext_(context) {}

    ge::graphStatus RunTiling();

private:
    ge::DataType GetInputDtype(size_t index) const
    {
        const auto *desc = tilingContext_->GetInputDesc(index);
        return desc == nullptr ? ge::DT_UNDEFINED : desc->GetDataType();
    }

    gert::TilingContext *tilingContext_ = nullptr;
    EngramGateTilingData tilingData_;
};

ge::graphStatus TilingEngramGateFunc::RunTiling()
{
    const auto *xShape = tilingContext_->GetInputShape(X);
    const auto *keyShape = tilingContext_->GetInputShape(KEY);
    const auto *valueShape = tilingContext_->GetInputShape(VALUE);
    const auto *weightShape = tilingContext_->GetInputShape(WEIGHT);
    const auto *maskShape = tilingContext_->GetInputShape(IMAGE_MASK);
    if (xShape == nullptr || keyShape == nullptr || valueShape == nullptr ||
        weightShape == nullptr || maskShape == nullptr) {
        OP_LOGE(tilingContext_->GetNodeName(), "engram_gate: input shape is null.");
        return ge::GRAPH_FAILED;
    }
    const auto &xs = xShape->GetStorageShape();
    const auto &ks = keyShape->GetStorageShape();
    const auto &vs = valueShape->GetStorageShape();
    const auto &ws = weightShape->GetStorageShape();
    const auto &ms = maskShape->GetStorageShape();
    if (xs.GetDimNum() != 3 || ks.GetDimNum() != 3 || vs.GetDimNum() != 2 ||
        ws.GetDimNum() != 2 || ms.GetDimNum() != 1) {
        OP_LOGE(tilingContext_->GetNodeName(), "engram_gate: unexpected input ranks.");
        return ge::GRAPH_FAILED;
    }

    const int64_t t = xs.GetDim(0);
    const int64_t hc = xs.GetDim(1);
    const int64_t dim = xs.GetDim(2);
    if (t < 1 || hc < 1 || dim < 1 || dim > CHUNK) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "engram_gate: invalid t=%ld hc=%ld dim=%ld (dim <= %ld).", t, hc, dim, CHUNK);
        return ge::GRAPH_FAILED;
    }
    if (ks.GetDim(0) != t || ks.GetDim(1) != hc || ks.GetDim(2) != dim) {
        OP_LOGE(tilingContext_->GetNodeName(), "engram_gate: key shape must equal x shape.");
        return ge::GRAPH_FAILED;
    }
    if (vs.GetDim(0) != t || vs.GetDim(1) != dim) {
        OP_LOGE(tilingContext_->GetNodeName(), "engram_gate: value must be [t,dim].");
        return ge::GRAPH_FAILED;
    }
    if (ws.GetDim(0) != hc || ws.GetDim(1) != dim || ms.GetDim(0) != t) {
        OP_LOGE(tilingContext_->GetNodeName(), "engram_gate: weight/mask shape mismatch.");
        return ge::GRAPH_FAILED;
    }
    if (GetInputDtype(X) != ge::DT_BF16 || GetInputDtype(KEY) != ge::DT_BF16 ||
        GetInputDtype(VALUE) != ge::DT_BF16 || GetInputDtype(WEIGHT) != ge::DT_FLOAT ||
        GetInputDtype(IMAGE_MASK) != ge::DT_BOOL) {
        OP_LOGE(tilingContext_->GetNodeName(), "engram_gate: input dtype mismatch.");
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext_->GetPlatformInfo());
    int64_t aiv = static_cast<int64_t>(platformInfo.GetCoreNumAiv());
    if (aiv <= 0) {
        aiv = std::max<int64_t>(1, static_cast<int64_t>(platformInfo.GetCoreNumAic()) * 2);
    }
    if (hc > aiv) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "engram_gate: hc=%ld exceeds the vector core count %ld.", hc, aiv);
        return ge::GRAPH_FAILED;
    }
    int64_t nc = std::min<int64_t>(aiv, t * hc);
    nc = (nc / hc) * hc;
    if (nc % 2 != 0) {
        nc += hc;  // only when hc is odd; keeps AIV even and a multiple of hc
    }
    if (nc <= 0) {
        OP_LOGE(tilingContext_->GetNodeName(), "engram_gate: launch width rounded to zero.");
        return ge::GRAPH_FAILED;
    }
    // AIV-only binary (new pypto codegen): SetBlockDim is the AIV block count.
    // (The previous MIX build used nc / 2 because SetBlockDim was the AIC count.)
    const int64_t blockDim = std::max<int64_t>(1, nc);

    tilingData_.set_t(t);
    tilingData_.set_hc(hc);
    tilingData_.set_dim(dim);

    size_t *workspace = tilingContext_->GetWorkspaceSizes(1);
    workspace[0] = MIN_WORKSPACE;

    tilingContext_->SetTilingKey(ShapeIdOf(t, hc, dim));
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
    TilingEngramGateFunc tilingObject(context);
    auto ret = tilingObject.RunTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "engram_gate tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EngramGate).Tiling(TilingFunc);

} // namespace optiling
