/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file atten_epilogue_tiling.cpp
 * \brief AttenEpilogue host tiling.
 *
 * C++ port of the Python host wrapper
 * (`op_kernel/atten_epilogue.py::atten_epilogue`).  The two must stay
 * equivalent:
 *   - full input shape/dtype validation against the fixed fused contract;
 *   - padded = ceil(T / 16) * 16 and the flat GM workspace it implies;
 *   - tiling key = the EpilogueKey capacity profile chosen by (T, cube cores);
 *   - blockDim = cube core num (MIX_AIC_1_2 -> 1 AIC : 2 AIV per block).
 *
 * Workspace: the kernel no longer receives four caller-allocated scratch tensors.
 * It carves aq/asc/yq/ys out of the single framework workspace with pl.addptr,
 * so tiling reports exactly WORKSPACE_BYTES_PER_ROW * padded bytes and the kernel
 * uses the workspace pointer as the aq base (same convention the PyPTO delivery
 * form compiles with: -DENABLE_CV_COMM_VIA_SSBUF=true makes the user workspace
 * start at the raw workspace argument, with no system reserve in front of it).
 *
 * No tiling data dependency: T is a static input-shape dimension, so nothing has
 * to be read back from the device.
 */

#include "atten_epilogue_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <cstdint>
#include <initializer_list>
#include <limits>

#ifndef OP_LOGE
#include <cstdio>
#define OP_LOGE(nodeName, fmt, ...) \
  do { \
    (void)(nodeName); \
    std::printf(fmt "\n", ##__VA_ARGS__); \
  } while (0)
#endif

namespace optiling {

// ---- fixed fused shape contract (mirrors the op_kernel constants) ----
constexpr int64_t HEADS = 64;
constexpr int64_t HEAD_DIM = 512;
constexpr int64_t GROUPS = 8;
constexpr int64_t GROUP_N = 1024;
constexpr int64_t GEMM1_K = 4096;
constexpr int64_t OUT_FEATURES = 5120;
constexpr int64_t GEMM2_K = 8192;
constexpr int64_t PAD_MULTIPLE = 16;

// ---- flat GM workspace layout, bytes per padded row (must match op_kernel) ----
constexpr int64_t AQ_BYTES_PER_ROW = GROUPS * GEMM1_K;             // 32768
constexpr int64_t ASC_BYTES_PER_ROW = GROUPS * GEMM1_K / 32;       // 1024
constexpr int64_t YQ_BYTES_PER_ROW = GEMM2_K;                      // 8192
constexpr int64_t YS_BYTES_PER_ROW = GEMM2_K / 32;                 // 256
constexpr int64_t WORKSPACE_BYTES_PER_ROW =
    AQ_BYTES_PER_ROW + ASC_BYTES_PER_ROW + YQ_BYTES_PER_ROW + YS_BYTES_PER_ROW;  // 42240

// ---- EpilogueKey bit layout (offsets assigned by TilingKeySchema in field order) ----
constexpr uint32_t WIDE_CUBE_CORES = 32;   // 32-AIC layout selector of tiling_key_for

enum InputIndex {
    O = 0,
    WOA,
    WOB,
    DESCALE_WOA,
    DESCALE_WOB,
    ROPE_SIN,
    ROPE_COS,
};

enum OutputIndex {
    OUT = 0,
};

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    // G.EXP.22-CPP: guard the divisor. Every call site passes the compile-time
    // constant PAD_MULTIPLE, so this never triggers; the guard keeps the helper
    // safe by construction (same shape as flex_attention_tiling.cpp).
    if (b <= 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

class TilingAttenEpilogueFunc {
public:
    explicit TilingAttenEpilogueFunc(gert::TilingContext *tiling_context)
        : tiling_context_(tiling_context) {}
    ge::graphStatus RunTiling();

private:
    ge::graphStatus ValidateInputs();
    ge::graphStatus ValidateOutput();
    uint64_t ComputeTilingKey() const;

    static bool CheckShape(const gert::StorageShape *shape, std::initializer_list<int64_t> dims);
    static ge::DataType GetInputDtype(gert::TilingContext *context, size_t index);

    gert::TilingContext *tiling_context_ = nullptr;
    int64_t t_ = 0;
    int64_t padded_ = 0;
    uint32_t aicCoreNum_ = 0;
};

bool TilingAttenEpilogueFunc::CheckShape(const gert::StorageShape *shape,
                                            std::initializer_list<int64_t> dims)
{
    if (shape == nullptr) {
        return false;
    }
    const auto &storage = shape->GetStorageShape();
    if (dims.size() != storage.GetDimNum()) {
        return false;
    }
    int64_t i = 0;
    for (int64_t dim : dims) {
        if (storage.GetDim(static_cast<size_t>(i)) != dim) {
            return false;
        }
        ++i;
    }
    return true;
}

ge::DataType TilingAttenEpilogueFunc::GetInputDtype(gert::TilingContext *context, size_t index)
{
    const auto *desc = context->GetInputDesc(index);
    return desc == nullptr ? ge::DT_UNDEFINED : desc->GetDataType();
}

ge::graphStatus TilingAttenEpilogueFunc::ValidateInputs()
{
    const auto *oShape = tiling_context_->GetInputShape(O);
    if (oShape == nullptr) {
        OP_LOGE(tiling_context_->GetNodeName(), "atten_epilogue: o shape is null.");
        return ge::GRAPH_FAILED;
    }
    const auto &os = oShape->GetStorageShape();
    if (os.GetDimNum() != 3) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "atten_epilogue: o must be rank-3 [T, %ld, %ld], got rank %zu.",
                HEADS, HEAD_DIM, os.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    t_ = os.GetDim(0);
    if (os.GetDim(1) != HEADS || os.GetDim(2) != HEAD_DIM) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "atten_epilogue: o trailing dims must be (%ld, %ld), got (%ld, %ld).",
                HEADS, HEAD_DIM, os.GetDim(1), os.GetDim(2));
        return ge::GRAPH_FAILED;
    }
    if (t_ < 1) {
        OP_LOGE(tiling_context_->GetNodeName(), "atten_epilogue: T must be positive, got %ld.", t_);
        return ge::GRAPH_FAILED;
    }
    if (GetInputDtype(tiling_context_, O) != ge::DT_BF16) {
        OP_LOGE(tiling_context_->GetNodeName(), "atten_epilogue: o must be BF16.");
        return ge::GRAPH_FAILED;
    }

    // weights + block scales (all T-independent)
    if (!CheckShape(tiling_context_->GetInputShape(WOA), {GROUPS, GROUP_N, GEMM1_K}) ||
        GetInputDtype(tiling_context_, WOA) != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "atten_epilogue: woa must be FLOAT8_E4M3FN [%ld, %ld, %ld].",
                GROUPS, GROUP_N, GEMM1_K);
        return ge::GRAPH_FAILED;
    }
    if (!CheckShape(tiling_context_->GetInputShape(WOB), {OUT_FEATURES, GEMM2_K}) ||
        GetInputDtype(tiling_context_, WOB) != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "atten_epilogue: wob must be FLOAT8_E4M3FN [%ld, %ld].", OUT_FEATURES, GEMM2_K);
        return ge::GRAPH_FAILED;
    }
    if (!CheckShape(tiling_context_->GetInputShape(DESCALE_WOA), {GROUPS, GROUP_N, GEMM1_K / 64, 2}) ||
        GetInputDtype(tiling_context_, DESCALE_WOA) != ge::DT_FLOAT8_E8M0) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "atten_epilogue: descale_woa must be FLOAT8_E8M0 [%ld, %ld, %ld, 2].",
                GROUPS, GROUP_N, GEMM1_K / 64);
        return ge::GRAPH_FAILED;
    }
    if (!CheckShape(tiling_context_->GetInputShape(DESCALE_WOB), {OUT_FEATURES, GEMM2_K / 64, 2}) ||
        GetInputDtype(tiling_context_, DESCALE_WOB) != ge::DT_FLOAT8_E8M0) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "atten_epilogue: descale_wob must be FLOAT8_E8M0 [%ld, %ld, 2].",
                OUT_FEATURES, GEMM2_K / 64);
        return ge::GRAPH_FAILED;
    }

    // RoPE tables follow T
    if (!CheckShape(tiling_context_->GetInputShape(ROPE_SIN), {t_, HEADS}) ||
        !CheckShape(tiling_context_->GetInputShape(ROPE_COS), {t_, HEADS})) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "atten_epilogue: rope_sin/rope_cos must be [%ld, %ld].", t_, HEADS);
        return ge::GRAPH_FAILED;
    }
    if (GetInputDtype(tiling_context_, ROPE_SIN) != ge::DT_FLOAT ||
        GetInputDtype(tiling_context_, ROPE_COS) != ge::DT_FLOAT) {
        OP_LOGE(tiling_context_->GetNodeName(), "atten_epilogue: rope_sin/rope_cos must be FLOAT.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingAttenEpilogueFunc::ValidateOutput()
{
    if (!CheckShape(tiling_context_->GetOutputShape(OUT), {t_, OUT_FEATURES})) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "atten_epilogue: out shape must be [T=%ld, %ld].", t_, OUT_FEATURES);
        return ge::GRAPH_FAILED;
    }
    const auto *outDesc = tiling_context_->GetOutputDesc(OUT);
    if (outDesc == nullptr || outDesc->GetDataType() != ge::DT_BF16) {
        OP_LOGE(tiling_context_->GetNodeName(), "atten_epilogue: out must be BF16.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t TilingAttenEpilogueFunc::ComputeTilingKey() const
{
    // Keep this selector equivalent to tiling_key_for(T, actual Cube core count).
    // Pass actual candidate values in the generated schema's declaration order.
    const bool wide = aicCoreNum_ == WIDE_CUBE_CORES;
    uint32_t bm = t_ <= 64 ? 64 : (t_ <= 128 ? 128 : 256);
    uint32_t bk = t_ <= 64 ? 512 : 256;
    uint32_t slots = (wide || t_ > 128) ? 1 : 2;
    uint32_t k0 = wide && bm >= 128 ? 128 : 256;
    uint32_t bn1 = wide ? 256 : 160;
    uint32_t bn = wide ? 160 : 192;
    uint32_t algo = 0;
    uint32_t mcap = 16;
    if (wide) {
        if (t_ <= 64) {
            algo = 1;
            mcap = static_cast<uint32_t>(CeilDiv(t_, PAD_MULTIPLE) * PAD_MULTIPLE);
        } else if (t_ <= 256) {
            algo = 3;
        } else if (t_ <= 384) {
            algo = 2;
        } else if (t_ <= 1024) {
            algo = 4;
        }
    } else if (aicCoreNum_ == 28 && t_ <= 64) {
        bn1 = 128;
        algo = 1;
        mcap = static_cast<uint32_t>(CeilDiv(t_, PAD_MULTIPLE) * PAD_MULTIPLE);
    } else if (aicCoreNum_ == 28 && t_ <= 1024) {
        k0 = 128;
        bn1 = t_ <= 256 ? 160 : 256;
        algo = 4;
    }
    return GET_TPL_TILING_KEY(bm, bk, slots, k0, bn1, bn, algo, mcap);
}

ge::graphStatus TilingAttenEpilogueFunc::RunTiling()
{
    if (tiling_context_->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platform_info = platform_ascendc::PlatformAscendC(tiling_context_->GetPlatformInfo());
    aicCoreNum_ = platform_info.GetCoreNumAic();
    if (aicCoreNum_ == 0) {
        OP_LOGE(tiling_context_->GetNodeName(), "atten_epilogue: platform reports 0 AIC cores.");
        return ge::GRAPH_FAILED;
    }

    auto ret = ValidateInputs();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = ValidateOutput();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // Guard the rounding and workspace products before any signed arithmetic.
    const int64_t maxRows = std::numeric_limits<int64_t>::max() / WORKSPACE_BYTES_PER_ROW;
    const int64_t maxPaddedRows = (maxRows / PAD_MULTIPLE) * PAD_MULTIPLE;
    if (t_ > maxPaddedRows) {
        OP_LOGE(tiling_context_->GetNodeName(), "atten_epilogue: workspace size overflows.");
        return ge::GRAPH_FAILED;
    }
    padded_ = CeilDiv(t_, PAD_MULTIPLE) * PAD_MULTIPLE;
    auto *data = tiling_context_->GetTilingData<EpilogueTiling>();
    size_t *workspace = tiling_context_->GetWorkspaceSizes(1);
    if (data == nullptr || workspace == nullptr) {
        OP_LOGE(tiling_context_->GetNodeName(), "atten_epilogue: tiling/workspace buffer is null or too small.");
        return ge::GRAPH_FAILED;
    }
    data->t = t_;
    data->padded = padded_;
    workspace[0] = static_cast<size_t>(WORKSPACE_BYTES_PER_ROW * padded_);
    tiling_context_->SetTilingKey(ComputeTilingKey());

    tiling_context_->SetBlockDim(aicCoreNum_);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    TilingAttenEpilogueFunc tilingObject(context);
    auto ret = tilingObject.RunTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "atten_epilogue tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParse(gert::TilingParseContext *context)
{
    return context == nullptr ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AttenEpilogue)
    .Tiling(TilingFunc)
    .TilingParse<AttenEpilogueCompileInfo>(TilingParse);

} // namespace optiling
