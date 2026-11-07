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
 * \file mixed_quant_sparse_flash_mla_tiling.cpp
 * \brief MixedQuantSparseFlashMla host tiling.
 *
 * C++ port of the Python host planner (`op_kernel/mixed_quant_sparse_flash_mla.py`
 * plan_metadata_uniform / plan_metadata_fd / plan_metadata / build_metadata,
 * cross-checked against the independent re-implementation in
 * tests/assets/mixed_quant_sparse_flash_mla_golden_cpu.py). The two must stay
 * equivalent:
 *   - the int32[1024] metadata schedule (uniform whole-row or flat FD plan)
 *     is generated here for FD/blockDim planning; the kernel reads the
 *     caller-provided metadata input tensor (large arrays stay out of tiling);
 *   - FD_ENABLE tiling-key bit and blockDim follow the plan;
 *   - uniform blocks = min(device cube cores, rows), clamped to AIC_CORE_MAX_NUM.
 *
 * Value dependency:
 *   cu_seqlens_q / ori_topk_length / cmp_topk_length are registered
 *   TilingInputsDataDependency (ValueDepend(OPTIONAL)). When their Host data is
 *   available the plan is exact; without per-row lengths the FD plan is never
 *   chosen (uniform only), and without cu_seqlens_q host data the tiling
 *   synthesizes the single-batch default ([0, T1]) for b_num == 2 and fails
 *   otherwise (multi-batch plans cannot be derived from shapes alone).
 */

#include "mixed_quant_sparse_flash_mla_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <vector>

#ifndef OP_LOGE
#include <cstdio>
#define OP_LOGE(nodeName, fmt, ...) \
  do { \
    (void)(nodeName); \
    std::printf(fmt "\n", ##__VA_ARGS__); \
  } while (0)
#endif

namespace optiling {

constexpr int64_t AIC_CORE_MAX_NUM = 36;
constexpr int64_t MQSMLA_METADATA_TOTAL_SIZE = 1024;
constexpr int64_t FA_METADATA_SIZE = 9;
constexpr int64_t FA_CORE_ENABLE_INDEX = 0;
constexpr int64_t FA_BN2_START_INDEX = 1;
constexpr int64_t FA_M_START_INDEX = 2;
constexpr int64_t FA_S2_START_INDEX = 3;
constexpr int64_t FA_BN2_END_INDEX = 4;
constexpr int64_t FA_M_END_INDEX = 5;
constexpr int64_t FA_S2_END_INDEX = 6;
constexpr int64_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7;
constexpr int64_t FD_METADATA_BASE = AIC_CORE_MAX_NUM * FA_METADATA_SIZE;  // 324
constexpr int64_t FD_CORE_ENABLE_INDEX = 0;
constexpr int64_t FD_BN2_IDX_INDEX = 1;
constexpr int64_t FD_M_IDX_INDEX = 2;
constexpr int64_t FD_WORKSPACE_IDX_INDEX = 3;
constexpr int64_t FD_WORKSPACE_NUM_INDEX = 4;
constexpr int64_t FD_M_START_INDEX = 5;
constexpr int64_t FD_M_NUM_INDEX = 6;
constexpr int64_t FD_METADATA_SIZE = 8;
constexpr int64_t FD_HEADS_PER_AIV = 32;
constexpr int64_t FD_USED_VEC_NUM_WORD = 900;
constexpr int64_t TILE_N = 128;
constexpr double FD_MIN_GAIN = 1.25;
constexpr int64_t FD_MIN_TOTAL_TILES = 64;
constexpr int64_t FD_MIN_ROW_TILES = 4;
constexpr double FD_SPLIT_COST_TILES = 0.5;
// FD split partials workspace (72 splits x 64 heads x 512 fp32 + m/l planes).
constexpr int64_t WORKSPACE_BYTES_FD = (72 * 64 * 512 + 2 * 72 * 64) * 4;
// A non-zero workspace is required for PyPTO's _gen_infer_cpp to correctly
// route SetSysWorkspaceForce + tiling data.
constexpr int64_t MIN_WORKSPACE = 1024;

constexpr int64_t D = 512;
constexpr int64_t N1 = 64;
constexpr int64_t ROW_BYTES_ORI = 544;
constexpr int64_t ROW_BYTES_CMP = 320;

enum InputIndex {
    QUERY = 0,
    ORI_KV,
    ORI_SPARSE_INDICES,
    ORI_BLOCK_TABLE,
    ORI_TOP_K_LEN,
    SINKS,
    CMP_KV,
    CMP_SPARSE_INDICES,
    CMP_BLOCK_TABLE,
    CMP_TOP_K_LEN,
    CU_SEQLENS_Q,
    METADATA,
};

enum OutputIndex {
    OUT = 0,
    LSE,
};

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    const int64_t divisor = (b > 0) ? b : 1;
    return (a + divisor - 1) / divisor;
}

// Port of the Python `_row_tiles` (kernel planner).
inline int64_t RowTiles(int64_t wOri, int64_t wCmp, bool hasCmp)
{
    int64_t n = wOri > 0 ? CeilDiv(wOri, TILE_N) : 0;
    if (hasCmp && wCmp > 0) {
        n += CeilDiv(wCmp, TILE_N);
    }
    return n;
}

// Port of the Python `_batch_of` (kernel planner).
inline int64_t BatchOf(const std::vector<int64_t> &cuQ, int64_t row)
{
    int64_t b = 0;
    for (size_t i = 0; i < cuQ.size(); ++i) {
        if (cuQ[i] <= row) {
            b = static_cast<int64_t>(i);
        }
    }
    return b;
}

// Port of the Python `_uniform_critical`.
static int64_t UniformCritical(int64_t rows, const std::vector<int64_t> &rowTiles, int64_t blocks)
{
    int64_t ublocks = std::min(blocks, rows);
    if (ublocks <= 0) {
        return 0;
    }
    int64_t worst = 0;
    int64_t start = 0;
    const int64_t divisor = (ublocks > 0) ? ublocks : 1;
    for (int64_t c = 0; c < ublocks; ++c) {
        int64_t cnt = rows / divisor + (c < rows % divisor ? 1 : 0);
        int64_t sum = 0;
        for (int64_t i = start; i < start + cnt && i < static_cast<int64_t>(rowTiles.size()); ++i) {
            sum += rowTiles[static_cast<size_t>(i)];
        }
        worst = std::max(worst, sum);
        start += cnt;
    }
    return worst;
}

// Port of the Python `plan_metadata_uniform`.
static void PlanUniform(const std::vector<int64_t> &cuQ, int64_t blocks, std::vector<int64_t> &md)
{
    int64_t rows = cuQ.back();
    md.assign(MQSMLA_METADATA_TOTAL_SIZE, 0);
    const int64_t divisor = (blocks > 0) ? blocks : 1;
    int64_t rowsPerCore = rows / divisor;
    int64_t extra = rows % divisor;
    int64_t start = 0;
    for (int64_t core = 0; core < blocks; ++core) {
        int64_t count = rowsPerCore + (core < extra ? 1 : 0);
        int64_t end = start + count;
        if (count > 0) {
            int64_t bn2S = BatchOf(cuQ, start);
            int64_t bn2E = BatchOf(cuQ, end);
            int64_t mS = start - cuQ[static_cast<size_t>(bn2S)];
            int64_t mE = end - cuQ[static_cast<size_t>(bn2E)];
            int64_t fa = FA_METADATA_SIZE * core;
            md[static_cast<size_t>(fa + FA_CORE_ENABLE_INDEX)] = 1;
            md[static_cast<size_t>(fa + FA_BN2_START_INDEX)] = bn2S;
            md[static_cast<size_t>(fa + FA_M_START_INDEX)] = mS;
            md[static_cast<size_t>(fa + FA_S2_START_INDEX)] = 0;
            md[static_cast<size_t>(fa + FA_BN2_END_INDEX)] = bn2E;
            md[static_cast<size_t>(fa + FA_M_END_INDEX)] = mE;
            md[static_cast<size_t>(fa + FA_S2_END_INDEX)] = 0;
            md[static_cast<size_t>(fa + FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX)] = 0;
        }
        start = end;
    }
    md[FD_USED_VEC_NUM_WORD] = 0;
}

// Port of the Python `plan_metadata_fd`. Returns the blockDim when the FD plan
// is adopted, 0 otherwise.
static int64_t PlanFd(const std::vector<int64_t> &cuQ, const std::vector<int64_t> &lenOri,
                      const std::vector<int64_t> &lenCmp, bool hasCmp, int64_t blocks,
                      std::vector<int64_t> &md)
{
    int64_t rows = cuQ.back();
    if (rows <= 0 || blocks <= 0) {
        return 0;
    }
    std::vector<int64_t> rowTiles(static_cast<size_t>(rows), 0);
    int64_t total = 0;
    int64_t maxTiles = 0;
    for (int64_t r = 0; r < rows; ++r) {
        int64_t wCmp = (hasCmp && !lenCmp.empty()) ? lenCmp[static_cast<size_t>(r)] : 0;
        int64_t tiles = RowTiles(lenOri[static_cast<size_t>(r)], wCmp, hasCmp);
        rowTiles[static_cast<size_t>(r)] = tiles;
        total += tiles;
        maxTiles = std::max(maxTiles, tiles);
    }
    if (total < FD_MIN_TOTAL_TILES || maxTiles < FD_MIN_ROW_TILES) {
        return 0;
    }

    int64_t ubMax = UniformCritical(rows, rowTiles, blocks);
    int64_t nBlocks = std::min(blocks, total);
    const int64_t blockDivisor = (nBlocks > 0) ? nBlocks : 1;
    std::vector<int64_t> flat(static_cast<size_t>(nBlocks) + 1, 0);
    for (int64_t c = 0; c <= nBlocks; ++c) {
        flat[static_cast<size_t>(c)] = total * c / blockDivisor;
    }
    int64_t flMax = 0;
    for (int64_t c = 0; c < nBlocks; ++c) {
        flMax = std::max(flMax, flat[static_cast<size_t>(c) + 1] - flat[static_cast<size_t>(c)]);
    }
    if (flMax <= 0 || static_cast<double>(ubMax) < static_cast<double>(flMax) * FD_MIN_GAIN) {
        return 0;
    }

    std::vector<int64_t> rstart(static_cast<size_t>(rows), 0);
    int64_t acc = 0;
    for (int64_t r = 0; r < rows; ++r) {
        rstart[static_cast<size_t>(r)] = acc;
        acc += rowTiles[static_cast<size_t>(r)];
    }
    auto rowAt = [&](int64_t flatPos) -> int64_t {
        int64_t r = 0;
        for (int64_t i = 0; i < rows; ++i) {
            if (rstart[static_cast<size_t>(i)] <= flatPos) {
                r = i;
            }
        }
        return r;
    };

    std::vector<std::pair<int64_t, int64_t>> splits;
    for (int64_t c = 1; c < nBlocks; ++c) {
        int64_t f = flat[static_cast<size_t>(c)];
        int64_t r = rowAt(f);
        if (rstart[static_cast<size_t>(r)] < f && f < rstart[static_cast<size_t>(r)] + rowTiles[static_cast<size_t>(r)]) {
            splits.emplace_back(c, r);
        }
    }
    if (static_cast<double>(ubMax - flMax) < FD_SPLIT_COST_TILES * static_cast<double>(splits.size())) {
        return 0;
    }
    if (static_cast<int64_t>(splits.size()) > nBlocks ||
        static_cast<int64_t>(splits.size()) * 2 > 2 * AIC_CORE_MAX_NUM) {
        return 0;
    }
    for (size_t i = 1; i < splits.size(); ++i) {
        if (splits[i].second == splits[i - 1].second) {
            return 0;
        }
    }

    md.assign(MQSMLA_METADATA_TOTAL_SIZE, 0);
    for (int64_t c = 0; c < nBlocks; ++c) {
        int64_t f0 = flat[static_cast<size_t>(c)];
        int64_t f1 = flat[static_cast<size_t>(c) + 1];
        if (f1 == f0) {
            continue;
        }
        int64_t r0 = rowAt(f0);
        int64_t r1 = rowAt(f1 - 1);
        int64_t e = f1 - rstart[static_cast<size_t>(r1)];
        int64_t r1e = e < rowTiles[static_cast<size_t>(r1)] ? r1 : r1 + 1;
        int64_t b0 = BatchOf(cuQ, r0);
        int64_t b1 = BatchOf(cuQ, r1e);
        int64_t fa = FA_METADATA_SIZE * c;
        md[static_cast<size_t>(fa + FA_CORE_ENABLE_INDEX)] = 1;
        md[static_cast<size_t>(fa + FA_BN2_START_INDEX)] = b0;
        md[static_cast<size_t>(fa + FA_M_START_INDEX)] = r0 - cuQ[static_cast<size_t>(b0)];
        md[static_cast<size_t>(fa + FA_S2_START_INDEX)] = f0 - rstart[static_cast<size_t>(r0)];
        md[static_cast<size_t>(fa + FA_BN2_END_INDEX)] = b1;
        md[static_cast<size_t>(fa + FA_M_END_INDEX)] = r1e - cuQ[static_cast<size_t>(b1)];
        md[static_cast<size_t>(fa + FA_S2_END_INDEX)] = e >= rowTiles[static_cast<size_t>(r1)] ? 0 : e;
    }

    for (const auto &split : splits) {
        int64_t c = split.first;
        int64_t r = split.second;
        int64_t headWs = 2 * (c - 1);
        int64_t tailWs = 2 * (c - 1) + 1;
        md[static_cast<size_t>(FA_METADATA_SIZE * c + FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX)] = headWs + 1;
        md[static_cast<size_t>(FA_METADATA_SIZE * (c - 1) + 8)] = tailWs + 1;
        int64_t b = BatchOf(cuQ, r);
        for (int64_t half = 0; half < 2; ++half) {
            int64_t base = FD_METADATA_BASE + (2 * (c - 1) + half) * FD_METADATA_SIZE;
            md[static_cast<size_t>(base + FD_CORE_ENABLE_INDEX)] = 1;
            md[static_cast<size_t>(base + FD_BN2_IDX_INDEX)] = b;
            md[static_cast<size_t>(base + FD_M_IDX_INDEX)] = r - cuQ[static_cast<size_t>(b)];
            md[static_cast<size_t>(base + FD_WORKSPACE_IDX_INDEX)] = headWs;
            md[static_cast<size_t>(base + FD_WORKSPACE_NUM_INDEX)] = 2;
            md[static_cast<size_t>(base + FD_M_START_INDEX)] = FD_HEADS_PER_AIV * half;
            md[static_cast<size_t>(base + FD_M_NUM_INDEX)] = FD_HEADS_PER_AIV;
        }
    }
    md[FD_USED_VEC_NUM_WORD] = static_cast<int64_t>(splits.size());
    return nBlocks;
}

// Port of the Python `plan_metadata` dispatch. Returns the blockDim.
static int64_t PlanMetadata(const std::vector<int64_t> &cuQ, const std::vector<int64_t> &lenOri,
                            const std::vector<int64_t> &lenCmp, bool hasCmp, int64_t blocks,
                            std::vector<int64_t> &md)
{
    if (!lenOri.empty()) {
        int64_t fdBlocks = PlanFd(cuQ, lenOri, lenCmp, hasCmp, blocks, md);
        if (fdBlocks > 0) {
            return fdBlocks;
        }
    }
    int64_t ublocks = std::min(blocks, std::max(cuQ.back(), static_cast<int64_t>(1)));
    PlanUniform(cuQ, ublocks, md);
    return ublocks;
}

class TilingMixedQuantSparseFlashMlaFunc {
public:
    explicit TilingMixedQuantSparseFlashMlaFunc(gert::TilingContext *tilingContext)
        : tilingContext_(tilingContext) {}
    ge::graphStatus RunTiling();

private:
    ge::graphStatus ParseShapesAndAttrs();
    ge::graphStatus ValidateTensors();
    ge::graphStatus ComputeTiling();

    static bool IsHostPlacement(const gert::Tensor *tensor)
    {
        return tensor != nullptr && gert::TensorPlacementUtils::IsOnHost(tensor->GetPlacement());
    }
    static bool CheckShape(const gert::StorageShape *shape, std::initializer_list<int64_t> dims);
    static ge::DataType GetInputDtype(gert::TilingContext *context, size_t index);
    static ge::DataType GetOutputDtype(gert::TilingContext *context, size_t index);

    gert::TilingContext *tilingContext_ = nullptr;
    MixedQuantSparseFlashMlaTilingData tilingData_;
    int64_t t1_ = 0;
    int64_t bNum_ = 0;
    int64_t kOri_ = 0;
    int64_t kCmp_ = 0;
    int64_t bsOri_ = 0;
    int64_t bsCmp_ = 0;
    int64_t btOri_ = 0;
    int64_t btCmp_ = 0;
    int64_t oriPages_ = 0;
    int64_t cmpPages_ = 0;
    int64_t blockDim_ = 1;
    int64_t fdEnable_ = 0;
    int64_t vdep64_ = 0;
    float scale_ = 0.0F;
    bool hasCmp_ = false;
    bool hasOriLen_ = false;
    bool hasCmpLen_ = false;
    uint32_t cubeCoreNum_ = 0;
    std::vector<int64_t> cuQ_;
    std::vector<int64_t> lenOri_;
    std::vector<int64_t> lenCmp_;
};

ge::DataType TilingMixedQuantSparseFlashMlaFunc::GetInputDtype(gert::TilingContext *context, size_t index)
{
    const auto *desc = context->GetInputDesc(index);
    return desc == nullptr ? ge::DT_UNDEFINED : desc->GetDataType();
}

ge::DataType TilingMixedQuantSparseFlashMlaFunc::GetOutputDtype(gert::TilingContext *context, size_t index)
{
    const auto *desc = context->GetOutputDesc(index);
    return desc == nullptr ? ge::DT_UNDEFINED : desc->GetDataType();
}

bool TilingMixedQuantSparseFlashMlaFunc::CheckShape(const gert::StorageShape *shape,
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

ge::graphStatus TilingMixedQuantSparseFlashMlaFunc::ParseShapesAndAttrs()
{
    // ---- attributes (mirrors the Python wrapper) ----
    const char *layoutQ = tilingContext_->GetAttrs() == nullptr ? nullptr : tilingContext_->GetAttrs()->GetStr(0);
    const char *layoutKv = tilingContext_->GetAttrs() == nullptr ? nullptr : tilingContext_->GetAttrs()->GetStr(1);
    if (layoutQ == nullptr || std::strcmp(layoutQ, "TND") != 0) {
        OP_LOGE(tilingContext_->GetNodeName(), "mixed_quant_sparse_flash_mla: layout_q must be 'TND'.");
        return ge::GRAPH_FAILED;
    }
    if (layoutKv == nullptr || std::strcmp(layoutKv, "PA_BBND") != 0) {
        OP_LOGE(tilingContext_->GetNodeName(), "mixed_quant_sparse_flash_mla: layout_kv must be 'PA_BBND'.");
        return ge::GRAPH_FAILED;
    }
    const float *scaleAttr = nullptr;
    if (tilingContext_->GetAttrs() != nullptr) {
        scaleAttr = tilingContext_->GetAttrs()->GetFloat(2);
    }
    scale_ = (scaleAttr != nullptr && *scaleAttr > 0.0F)
        ? *scaleAttr
        : static_cast<float>(1.0 / std::sqrt(static_cast<double>(D)));

    const auto *qShape = tilingContext_->GetInputShape(QUERY);
    const auto *oriKvShape = tilingContext_->GetInputShape(ORI_KV);
    const auto *idxShape = tilingContext_->GetInputShape(ORI_SPARSE_INDICES);
    const auto *btShape = tilingContext_->GetInputShape(ORI_BLOCK_TABLE);
    const auto *sinksShape = tilingContext_->GetInputShape(SINKS);
    const auto *cuShape = tilingContext_->GetInputShape(CU_SEQLENS_Q);
    if (qShape == nullptr || oriKvShape == nullptr || idxShape == nullptr ||
        btShape == nullptr || sinksShape == nullptr || cuShape == nullptr) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla: a required input shape is null.");
        return ge::GRAPH_FAILED;
    }
    if (GetInputDtype(tilingContext_, QUERY) != ge::DT_BF16) {
        OP_LOGE(tilingContext_->GetNodeName(), "mixed_quant_sparse_flash_mla: q must be BF16.");
        return ge::GRAPH_FAILED;
    }
    if (GetInputDtype(tilingContext_, ORI_KV) != ge::DT_UINT8 ||
        GetInputDtype(tilingContext_, ORI_SPARSE_INDICES) != ge::DT_INT32 ||
        GetInputDtype(tilingContext_, ORI_BLOCK_TABLE) != ge::DT_INT32 ||
        GetInputDtype(tilingContext_, SINKS) != ge::DT_FLOAT) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla: input dtype contract violated.");
        return ge::GRAPH_FAILED;
    }
    // ValueDepend planning inputs accept INT32 (Tensor variant) or INT64
    // (aclIntArray variant of the generated aclnn interface).
    for (size_t idx : {static_cast<size_t>(CU_SEQLENS_Q)}) {
        auto dtype = GetInputDtype(tilingContext_, idx);
        if (dtype != ge::DT_INT32 && dtype != ge::DT_INT64) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "mixed_quant_sparse_flash_mla: cu_seqlens_q must be int32/int64.");
            return ge::GRAPH_FAILED;
        }
    }

    // q [T1, 64, 512]
    const auto &qs = qShape->GetStorageShape();
    if (qs.GetDimNum() != 3 || qs.GetDim(1) != N1 || qs.GetDim(2) != D) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla: q must be [T1, 64, 512].");
        return ge::GRAPH_FAILED;
    }
    t1_ = qs.GetDim(0);

    // ori_kv [pages, bs, 1, 544]
    const auto &oks = oriKvShape->GetStorageShape();
    if (oks.GetDimNum() != 4 || oks.GetDim(2) != 1 || oks.GetDim(3) != ROW_BYTES_ORI) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla: ori_kv must be [pages, bs, 1, 544].");
        return ge::GRAPH_FAILED;
    }
    oriPages_ = oks.GetDim(0);
    bsOri_ = oks.GetDim(1);

    // ori_sparse_indices [T1, 1, k_ori]
    const auto &is = idxShape->GetStorageShape();
    if (is.GetDimNum() != 3 || is.GetDim(0) != t1_ || is.GetDim(1) != 1) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla: ori_sparse_indices must be [T1, 1, k_ori].");
        return ge::GRAPH_FAILED;
    }
    kOri_ = is.GetDim(2);

    // cu_seqlens_q [B + 1]
    const auto &cus = cuShape->GetStorageShape();
    if (cus.GetDimNum() != 1) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla: cu_seqlens_q must be rank 1.");
        return ge::GRAPH_FAILED;
    }
    bNum_ = cus.GetDim(0);

    // ori_block_table [B, bt_ori] with B = b_num - 1
    const auto &bts = btShape->GetStorageShape();
    if (bts.GetDimNum() != 2 || bts.GetDim(0) != bNum_ - 1) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla: ori_block_table must be [B, bt_ori].");
        return ge::GRAPH_FAILED;
    }
    btOri_ = bts.GetDim(1);

    // sinks [64]
    if (!CheckShape(sinksShape, {N1})) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla: sinks must be [64].");
        return ge::GRAPH_FAILED;
    }

    // ---- optional cmp group: all-or-none ----
    hasCmp_ = tilingContext_->GetOptionalInputShape(CMP_KV) != nullptr;
    if (hasCmp_) {
        const auto *cmpKvShape = tilingContext_->GetOptionalInputShape(CMP_KV);
        const auto *cmpIdxShape = tilingContext_->GetOptionalInputShape(CMP_SPARSE_INDICES);
        const auto *cmpBtShape = tilingContext_->GetOptionalInputShape(CMP_BLOCK_TABLE);
        if (cmpIdxShape == nullptr || cmpBtShape == nullptr) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "mixed_quant_sparse_flash_mla: cmp_kv given but cmp_sparse_indices/cmp_block_table missing.");
            return ge::GRAPH_FAILED;
        }
        if (GetInputDtype(tilingContext_, CMP_KV) != ge::DT_UINT8 ||
            GetInputDtype(tilingContext_, CMP_SPARSE_INDICES) != ge::DT_INT32 ||
            GetInputDtype(tilingContext_, CMP_BLOCK_TABLE) != ge::DT_INT32) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "mixed_quant_sparse_flash_mla: cmp group dtype contract violated.");
            return ge::GRAPH_FAILED;
        }
        const auto &cks = cmpKvShape->GetStorageShape();
        if (cks.GetDimNum() != 4 || cks.GetDim(2) != 1 || cks.GetDim(3) != ROW_BYTES_CMP) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "mixed_quant_sparse_flash_mla: cmp_kv must be [pages, bs, 1, 320].");
            return ge::GRAPH_FAILED;
        }
        cmpPages_ = cks.GetDim(0);
        bsCmp_ = cks.GetDim(1);
        const auto &cis = cmpIdxShape->GetStorageShape();
        if (cis.GetDimNum() != 3 || cis.GetDim(0) != t1_ || cis.GetDim(1) != 1) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "mixed_quant_sparse_flash_mla: cmp_sparse_indices must be [T1, 1, k_cmp].");
            return ge::GRAPH_FAILED;
        }
        kCmp_ = cis.GetDim(2);
        const auto &cbts = cmpBtShape->GetStorageShape();
        if (cbts.GetDimNum() != 2 || cbts.GetDim(0) != bNum_ - 1) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "mixed_quant_sparse_flash_mla: cmp_block_table must be [B, bt_cmp].");
            return ge::GRAPH_FAILED;
        }
        btCmp_ = cbts.GetDim(1);
        hasCmpLen_ = tilingContext_->GetOptionalInputShape(CMP_TOP_K_LEN) != nullptr;
    } else {
        cmpPages_ = 0;
        bsCmp_ = 0;
        kCmp_ = 0;
        btCmp_ = 0;
    }
    hasOriLen_ = tilingContext_->GetOptionalInputShape(ORI_TOP_K_LEN) != nullptr;
    if (hasOriLen_) {
        auto dtype = GetInputDtype(tilingContext_, ORI_TOP_K_LEN);
        if (dtype != ge::DT_INT32 && dtype != ge::DT_INT64) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "mixed_quant_sparse_flash_mla: ori_topk_length must be int32/int64.");
            return ge::GRAPH_FAILED;
        }
    }
    if (hasCmpLen_) {
        auto dtype = GetInputDtype(tilingContext_, CMP_TOP_K_LEN);
        if (dtype != ge::DT_INT32 && dtype != ge::DT_INT64) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "mixed_quant_sparse_flash_mla: cmp_topk_length must be int32/int64.");
            return ge::GRAPH_FAILED;
        }
    }
    vdep64_ = (GetInputDtype(tilingContext_, CU_SEQLENS_Q) == ge::DT_INT64) ? 1 : 0;
    if (hasOriLen_ && GetInputDtype(tilingContext_, ORI_TOP_K_LEN) == ge::DT_INT64) {
        vdep64_ = 1;
    }
    if (hasCmpLen_ && GetInputDtype(tilingContext_, CMP_TOP_K_LEN) == ge::DT_INT64) {
        vdep64_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingMixedQuantSparseFlashMlaFunc::ValidateTensors()
{
    // ---- metadata plan (input tensor, not a tiling field) ----
    const auto *mdShape = tilingContext_->GetInputShape(METADATA);
    if (!CheckShape(mdShape, {MQSMLA_METADATA_TOTAL_SIZE})) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla: metadata must be int32[%ld].",
                MQSMLA_METADATA_TOTAL_SIZE);
        return ge::GRAPH_FAILED;
    }
    if (GetInputDtype(tilingContext_, METADATA) != ge::DT_INT32) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla: metadata must be int32.");
        return ge::GRAPH_FAILED;
    }

    // ---- outputs ----
    if (!CheckShape(tilingContext_->GetOutputShape(OUT), {t1_, N1, D})) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla: out must be [T1=%ld, 64, 512].", t1_);
        return ge::GRAPH_FAILED;
    }
    if (GetOutputDtype(tilingContext_, OUT) != ge::DT_BF16) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla: out must be BF16.");
        return ge::GRAPH_FAILED;
    }
    if (!CheckShape(tilingContext_->GetOutputShape(LSE), {1, t1_, N1})) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla: lse must be [1, T1=%ld, 64].", t1_);
        return ge::GRAPH_FAILED;
    }
    if (GetOutputDtype(tilingContext_, LSE) != ge::DT_FLOAT) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla: lse must be float32.");
        return ge::GRAPH_FAILED;
    }

    // ---- host data for the planner ----
    // cu_seqlens_q: prefix sum [B + 1] (int32 Tensor variant or int64
    // aclIntArray variant). ValueDepend host data is preferred; a single-batch
    // graph (b_num == 2) can be synthesized from shapes.
    const auto *cuTensor = tilingContext_->GetInputTensor(CU_SEQLENS_Q);
    if (IsHostPlacement(cuTensor)) {
        if (GetInputDtype(tilingContext_, CU_SEQLENS_Q) == ge::DT_INT64) {
            const int64_t *cu = cuTensor->GetData<int64_t>();
            if (cu == nullptr) {
                OP_LOGE(tilingContext_->GetNodeName(),
                        "mixed_quant_sparse_flash_mla: cu_seqlens_q data is null.");
                return ge::GRAPH_FAILED;
            }
            cuQ_.assign(cu, cu + bNum_);
        } else {
            const int32_t *cu = cuTensor->GetData<int32_t>();
            if (cu == nullptr) {
                OP_LOGE(tilingContext_->GetNodeName(),
                        "mixed_quant_sparse_flash_mla: cu_seqlens_q data is null.");
                return ge::GRAPH_FAILED;
            }
            cuQ_.assign(cu, cu + bNum_);
        }
        if (cuQ_.empty() || cuQ_[0] != 0) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "mixed_quant_sparse_flash_mla: cu_seqlens_q must start with 0.");
            return ge::GRAPH_FAILED;
        }
        for (int64_t i = 1; i < bNum_; ++i) {
            if (cuQ_[static_cast<size_t>(i)] < cuQ_[static_cast<size_t>(i) - 1]) {
                OP_LOGE(tilingContext_->GetNodeName(),
                        "mixed_quant_sparse_flash_mla: cu_seqlens_q must be non-decreasing.");
                return ge::GRAPH_FAILED;
            }
        }
        if (cuQ_.back() != t1_) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "mixed_quant_sparse_flash_mla: cu_seqlens_q[-1](%ld) != T1(%ld).",
                    cuQ_.back(), t1_);
            return ge::GRAPH_FAILED;
        }
    } else if (bNum_ == 2) {
        cuQ_ = {0, t1_};
    } else {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla: cu_seqlens_q host data unavailable for "
                "multi-batch graph (B=%ld); pass it as a ValueDepend host array.",
                bNum_ - 1);
        return ge::GRAPH_FAILED;
    }

    // per-row TopK lengths (clamped to K like the Python build_metadata).
    // The aclIntArray variant arrives as a flat 1D buffer (numel only);
    // the Tensor variant keeps the declared [T1, 1] rank. Accept both.
    auto readLengths = [&](size_t index, int64_t kLimit, std::vector<int64_t> &out) -> bool {
        const auto *tensor = tilingContext_->GetOptionalInputTensor(index);
        if (!IsHostPlacement(tensor)) {
            return false;
        }
        const auto *shape = tilingContext_->GetOptionalInputShape(index);
        if (shape == nullptr) {
            return false;
        }
        const auto &storage = shape->GetStorageShape();
        int64_t numel = 1;
        for (size_t i = 0; i < storage.GetDimNum(); ++i) {
            numel *= storage.GetDim(i);
        }
        if (numel != t1_) {
            return false;
        }
        out.resize(static_cast<size_t>(t1_));
        if (GetInputDtype(tilingContext_, index) == ge::DT_INT64) {
            const int64_t *data = tensor->GetData<int64_t>();
            if (data == nullptr) {
                return false;
            }
            for (int64_t i = 0; i < t1_; ++i) {
                out[static_cast<size_t>(i)] = std::min(kLimit, std::max(data[i], static_cast<int64_t>(0)));
            }
        } else {
            const int32_t *data = tensor->GetData<int32_t>();
            if (data == nullptr) {
                return false;
            }
            for (int64_t i = 0; i < t1_; ++i) {
                out[static_cast<size_t>(i)] = std::min(kLimit, std::max(static_cast<int64_t>(data[i]), static_cast<int64_t>(0)));
            }
        }
        return true;
    };
    if (hasOriLen_ && !readLengths(ORI_TOP_K_LEN, kOri_, lenOri_)) {
        // Host data unavailable: the FD cost model is skipped (uniform plan),
        // which stays correct -- the kernel still reads the device tensor.
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla: ori_topk_length host data unavailable, "
                "falling back to the uniform schedule.");
        lenOri_.clear();
    }
    if (hasCmp_ && hasCmpLen_ && !readLengths(CMP_TOP_K_LEN, kCmp_, lenCmp_)) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla: cmp_topk_length host data unavailable, "
                "falling back to the uniform schedule.");
        lenCmp_.clear();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingMixedQuantSparseFlashMlaFunc::ComputeTiling()
{
    auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext_->GetPlatformInfo());
    cubeCoreNum_ = platformInfo.GetCoreNumAic();
    if (cubeCoreNum_ <= 0) {
        cubeCoreNum_ = 1;
    }
    if (cubeCoreNum_ > AIC_CORE_MAX_NUM) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla: device cube cores(%u) exceed AIC_CORE_MAX_NUM(%ld).",
                cubeCoreNum_, AIC_CORE_MAX_NUM);
        return ge::GRAPH_FAILED;
    }
    int64_t blocks = static_cast<int64_t>(cubeCoreNum_);

    std::vector<int64_t> md;
    blockDim_ = PlanMetadata(cuQ_, lenOri_, lenCmp_, hasCmp_, blocks, md);
    fdEnable_ = md[FD_USED_VEC_NUM_WORD] > 0 ? 1 : 0;
    if (blockDim_ <= 0) {
        blockDim_ = 1;
    }
    if (blockDim_ > blocks) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla: planned blockDim(%ld) exceeds device cores(%ld).",
                blockDim_, blocks);
        return ge::GRAPH_FAILED;
    }

    tilingData_.set_scale(scale_);
    tilingData_.set_t1(t1_);
    tilingData_.set_ori_pages(oriPages_);
    tilingData_.set_cmp_pages(cmpPages_);
    tilingData_.set_b_num(bNum_);
    tilingData_.set_k_ori(kOri_);
    tilingData_.set_k_cmp(hasCmp_ ? kCmp_ : 129);
    tilingData_.set_bs_ori(bsOri_);
    tilingData_.set_bs_cmp(hasCmp_ ? bsCmp_ : 128);
    tilingData_.set_bt_ori(btOri_);
    tilingData_.set_bt_cmp(hasCmp_ ? btCmp_ : 0);
    // ND + AutoContiguous: page stride equals one page row (no axis-0 padding).
    tilingData_.set_ori_kv_stride0(bsOri_ * ROW_BYTES_ORI);
    tilingData_.set_cmp_kv_stride0(hasCmp_ ? bsCmp_ * ROW_BYTES_CMP : 0);

    // Tiling key encodes (HAS_ORI_LEN, HAS_CMP_LEN, HAS_CMP, FD_ENABLE, VDEP64)
    // at bits 0..4 in the MqsmlaTilingKey declaration order.
    const int64_t tilingKey = (hasOriLen_ ? 1 : 0) | (hasCmpLen_ ? 1 : 0) << 1 |
                              (hasCmp_ ? 1 : 0) << 2 | (fdEnable_ << 3) |
                              (vdep64_ << 4);
    tilingContext_->SetTilingKey(tilingKey);

    // FD partials workspace; keep a non-zero minimum for the non-FD variants
    // (SetSysWorkspaceForce routing requirement).
    size_t *workspace = tilingContext_->GetWorkspaceSizes(1);
    workspace[0] = fdEnable_ ? static_cast<size_t>(WORKSPACE_BYTES_FD) : static_cast<size_t>(MIN_WORKSPACE);

    tilingContext_->SetBlockDim(static_cast<uint32_t>(blockDim_));

    tilingData_.SaveToBuffer(tilingContext_->GetRawTilingData()->GetData(),
                             tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingMixedQuantSparseFlashMlaFunc::RunTiling()
{
    auto ret = ParseShapesAndAttrs();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = ValidateTensors();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = ComputeTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingMixedQuantSparseFlashMlaFunc tilingObject(context);
    auto ret = tilingObject.RunTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "mixed_quant_sparse_flash_mla tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MixedQuantSparseFlashMla)
    .Tiling(TilingFunc)
    .TilingInputsDataDependency({ORI_TOP_K_LEN, CMP_TOP_K_LEN, CU_SEQLENS_Q});

} // namespace optiling