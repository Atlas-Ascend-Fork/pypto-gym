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
 * \file quant_sparse_lightning_indexer_tiling.cpp
 * \brief QuantSparseLightningIndexer host tiling.
 *
 * C++ port of the Python host wrapper
 * (`op_kernel/quant_sparse_lightning_indexer.py::quant_sparse_lightning_indexer`).
 * The two must stay equivalent:
 *   - the full input/attribute contract (layouts, dtypes, ranks, page geometry);
 *   - every rt_* runtime scalar the kernel reads (batch, strides, presence flags,
 *     page reciprocal parameter, capacity, topk);
 *   - the QsliKey selection (WriteValues, IsPA, TopKCapacity, TndStrideMode, PairMin);
 *   - the flat workspace size the kernel carves with pl.addptr;
 *   - blockDim = num_workers = min(32, AIC core num).
 *
 * Strides: callers may pass k / descale_k with axis-0 padding, and def.cpp
 * deliberately omits AutoContiguous() on those two inputs -- but nnopbase still
 * makes every non-contiguous input contiguous before launch (aclnnContiguous,
 * whose buffer is appended to the workspace size reported here; measured
 * 5,767,168 B requested vs 22,545,408 B returned by
 * aclnnQuantSparseLightningIndexerGetWorkspaceSize). The kernel therefore only
 * ever sees COMPACT strides, so this tiling reports the compact values and
 * TndStrideMode is always 3 on the built-in path. GetInputStride /
 * GetOptionalInputStride are still read, but only to reject overlapping axis-0
 * rows -- feeding the caller's padded stride to the kernel makes it walk the
 * contiguous copy with the wrong pitch (indices stay right, every score wrong).
 * The strided specializations (TndStrideMode 0/1/2) remain reachable only through
 * the jit host entry, which hands the caller's tensors to the kernel uncopied.
 *
 * PairMin reproduces the jit host's _batch_consistency_enabled(): determinism
 * level 3 forces the single-record DMA path (PairMin=0), otherwise the paired
 * 544-byte record path is used.
 *
 * No tiling data dependency: everything derives from input shapes/strides and
 * attributes. The LI/LD schedule table (`metadata`) is caller-generated content
 * that the op never reads back to host, exactly as in the jit form.
 */

#include "quant_sparse_lightning_indexer_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>

#ifndef OP_LOGE
#include <cstdio>
#define OP_LOGE(nodeName, fmt, ...) \
  do { \
    (void)(nodeName); \
    std::printf(fmt "\n", ##__VA_ARGS__); \
  } while (0)
#endif

namespace optiling {

// ---- tile geometry (mirrors the op_kernel constants) ----
constexpr int64_t TILE_N = 512;
constexpr int64_t LOGICAL_D = 128;
constexpr int64_t PACKED_D = LOGICAL_D / 2;             // 64
constexpr int64_t N1 = 32;
constexpr int64_t N2 = 1;
constexpr int64_t SCALE_K = LOGICAL_D / 32;             // 4
constexpr int64_t CANDIDATE_BLOCK_SIZE = 8;
constexpr int64_t CANDIDATE_CAPACITY = 2048;
constexpr int64_t KEY_BLOCK_BYTES = CANDIDATE_BLOCK_SIZE * PACKED_D;      // 512
constexpr int64_t SCALE_BLOCK_BYTES = CANDIDATE_BLOCK_SIZE * SCALE_K;     // 32
constexpr int64_t PACKED_BLOCK_BYTES = KEY_BLOCK_BYTES + SCALE_BLOCK_BYTES;  // 544
constexpr int64_t STAGE_DEPTH = 4;
constexpr int64_t MAX_TOKENS = CANDIDATE_CAPACITY * CANDIDATE_BLOCK_SIZE;  // 16384
constexpr int64_t HALF_TOKENS = TILE_N / 2;                               // 256
constexpr int64_t MAX_TILES = MAX_TOKENS / TILE_N;                        // 32
constexpr int64_t MAX_CUBE_WORKERS = 32;
constexpr int64_t WORKSPACE_SLOTS = 4 * MAX_CUBE_WORKERS;                 // 128
constexpr int64_t METADATA_ELEMENTS = 1024;
constexpr int64_t PA_TABLE_CACHE_ENTRIES = 2048;
constexpr int64_t MAX_ADDRESS32_BYTES = static_cast<int64_t>(1) << 31;
constexpr int64_t PA_BLOCK_ALIGNMENT = 16;
constexpr int64_t MAX_PA_BLOCK_SIZE = 1024;
constexpr int64_t MAX_TOPK = 8192;
constexpr int64_t MAX_CMP_RATIO = 128;
constexpr int64_t DETERMINISTIC_LEVEL_BATCH_CONSISTENT = 3;

// TopKCapacity candidate list, in QsliKey declaration order (index == packed value).
constexpr int64_t TOPK_CAPACITIES[] = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
constexpr size_t TOPK_CAPACITY_COUNT = sizeof(TOPK_CAPACITIES) / sizeof(TOPK_CAPACITIES[0]);

// QsliKey bit layout (TilingKeySchema assigns offsets in declaration order):
//   WriteValues bits=1 @0 [0,1] | IsPA bits=1 @1 [0,1] | TopKCapacity bits=4 @2
//   TndStrideMode bits=2 @6 [0,1,2,3] | PairMin bits=1 @8 [544,0]
constexpr uint32_t KEY_SHIFT_IS_PA = 1;
constexpr uint32_t KEY_SHIFT_TOPK = 2;
constexpr uint32_t KEY_SHIFT_TND_STRIDE = 6;
constexpr uint32_t KEY_SHIFT_PAIR_MIN = 8;

enum InputIndex {
    Q = 0,
    K,
    W,
    DESCALE_Q,
    DESCALE_K,
    CANDIDATE_BLOCK_INDICES,
    CANDIDATE_BLOCK_LENGTH,
    CU_SEQLENS_Q,
    CU_SEQLENS_K,
    SEQUSED_Q,
    SEQUSED_K,
    CMP_RESIDUAL_K,
    BLOCK_TABLE,
    OUTPUT_IDX_OFFSET,
    METADATA,
};

enum OutputIndex {
    INDICES = 0,
    VALUES,
};

enum AttrIndex {
    ATTR_TOPK = 0,
    ATTR_CANDIDATE_BLOCK_SIZE,
    ATTR_QUANT_MODE,
    ATTR_MAX_SEQLEN_Q,
    ATTR_MASK_MODE,
    ATTR_CMP_RATIO,
    ATTR_LAYOUT_Q,
    ATTR_LAYOUT_K,
    ATTR_RETURN_VALUE,
};

// GetInt/GetBool hand back pointers into the attribute blob; a missing or
// ill-typed attribute yields nullptr, which must be rejected, not dereferenced.
static bool ReadIntAttr(const gert::RuntimeAttrs *attrs, size_t index, int64_t &value)
{
    const int64_t *ptr = attrs->GetInt(index);
    if (ptr == nullptr) {
        return false;
    }
    value = *ptr;
    return true;
}

static bool ReadBoolAttr(const gert::RuntimeAttrs *attrs, size_t index, bool &value)
{
    const bool *ptr = attrs->GetBool(index);
    if (ptr == nullptr) {
        return false;
    }
    value = *ptr;
    return true;
}

// PAGE_PARAMS[n] from the kernel module: 0 for n==0, log2(n) for a power of two,
// otherwise the 32-bit reciprocal used by the vectorised page-number computation.
static int64_t PageParam(int64_t n)
{
    if (n <= 0) {
        return 0;
    }
    if ((n & (n - 1)) == 0) {
        int64_t log2n = 0;
        while ((static_cast<int64_t>(1) << log2n) < n) {
            ++log2n;
        }
        return log2n;
    }
    // Guarded at the division site as well as by the early return above, so the
    // non-negativity of the divisor is locally visible (G.EXP.22-CPP).
    return (n > 0) ? ((static_cast<int64_t>(1) << 32) / n + 1) : 0;
}

class TilingQsliFunc {
public:
    explicit TilingQsliFunc(gert::TilingContext *tiling_context)
        : tiling_context_(tiling_context) {}
    ge::graphStatus RunTiling();

private:
    ge::graphStatus ParseAttrs();
    ge::graphStatus ValidateCommonInputs();
    ge::graphStatus DerivePa();
    ge::graphStatus DeriveTnd();
    ge::graphStatus ValidateSequences();
    ge::graphStatus ValidateOutputs();
    uint32_t ComputeTilingKey() const;
    int64_t WorkspaceBytes() const;

    static bool CheckShape(const gert::StorageShape *shape, std::initializer_list<int64_t> dims);
    static int64_t Numel(const gert::StorageShape *shape);
    static ge::DataType InputDtype(gert::TilingContext *context, size_t index);
    static int64_t Axis0Stride(gert::TilingContext *context, size_t index, bool optional,
                               int64_t fallback);
    static bool Present(gert::TilingContext *context, size_t index);

    gert::TilingContext *tiling_context_ = nullptr;
    QsliTilingData tiling_data_;

    // attributes
    int64_t topk_ = 0;
    int64_t maskMode_ = 0;
    int64_t cmpRatio_ = 1;
    bool returnValue_ = false;
    bool isPa_ = false;

    // derived geometry
    int64_t totalQ_ = 0;
    int64_t batch_ = 0;
    int64_t logicalKCapacity_ = 0;
    int64_t keyStride_ = 0;
    int64_t tableWidth_ = 0;
    bool tableCached_ = true;
    bool hasTable_ = false;
    bool addr32_ = false;
    int64_t tndKStride_ = PACKED_D;
    int64_t tndSStride_ = SCALE_K;
    int64_t tndRows_ = 1;
    int64_t blocksPerPage_ = 1;
    bool hasCuQ_ = false;
    bool hasSeqQ_ = false;
    bool hasSeqK_ = false;
    bool hasResidual_ = false;
    bool hasIdxOff_ = false;
    bool resEqUsedQ_ = true;
    int64_t numWorkers_ = MAX_CUBE_WORKERS;
    int64_t topkCapacity_ = 16;
    int64_t pairMin_ = PACKED_BLOCK_BYTES;
};

bool TilingQsliFunc::CheckShape(const gert::StorageShape *shape,
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

int64_t TilingQsliFunc::Numel(const gert::StorageShape *shape)
{
    if (shape == nullptr) {
        return 0;
    }
    const auto &storage = shape->GetStorageShape();
    int64_t numel = 1;
    for (size_t i = 0; i < storage.GetDimNum(); ++i) {
        numel *= storage.GetDim(i);
    }
    return numel;
}

ge::DataType TilingQsliFunc::InputDtype(gert::TilingContext *context, size_t index)
{
    const auto *desc = context->GetInputDesc(index);
    return desc == nullptr ? ge::DT_UNDEFINED : desc->GetDataType();
}

bool TilingQsliFunc::Present(gert::TilingContext *context, size_t index)
{
    return context->GetOptionalInputShape(index) != nullptr;
}

int64_t TilingQsliFunc::Axis0Stride(gert::TilingContext *context, size_t index, bool optional,
                                    int64_t fallback)
{
    const gert::Stride *stride = optional ? context->GetOptionalInputStride(index)
                                          : context->GetInputStride(index);
    if (stride == nullptr) {
        return fallback;
    }
    return stride->GetStride(0);
}

ge::graphStatus TilingQsliFunc::ParseAttrs()
{
    const auto *attrs = tiling_context_->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: attrs is null.");
        return ge::GRAPH_FAILED;
    }
    int64_t blockSize = 0;
    int64_t quantMode = 0;
    int64_t maxSeqlenQ = 0;
    if (!ReadIntAttr(attrs, ATTR_TOPK, topk_) ||
        !ReadIntAttr(attrs, ATTR_CANDIDATE_BLOCK_SIZE, blockSize) ||
        !ReadIntAttr(attrs, ATTR_QUANT_MODE, quantMode) ||
        !ReadIntAttr(attrs, ATTR_MAX_SEQLEN_Q, maxSeqlenQ) ||
        !ReadIntAttr(attrs, ATTR_MASK_MODE, maskMode_) ||
        !ReadIntAttr(attrs, ATTR_CMP_RATIO, cmpRatio_) ||
        !ReadBoolAttr(attrs, ATTR_RETURN_VALUE, returnValue_)) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: failed to read the scalar attributes.");
        return ge::GRAPH_FAILED;
    }
    const char *layoutQ = attrs->GetStr(ATTR_LAYOUT_Q);
    const char *layoutK = attrs->GetStr(ATTR_LAYOUT_K);

    if (blockSize != CANDIDATE_BLOCK_SIZE) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: candidate_block_size must be 8, got %ld.",
                blockSize);
        return ge::GRAPH_FAILED;
    }
    if (quantMode != 1) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: quant_mode must be 1, got %ld.", quantMode);
        return ge::GRAPH_FAILED;
    }
    if (topk_ < 1 || topk_ > MAX_TOPK) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: topk must be in [1,%ld], got %ld.",
                MAX_TOPK, topk_);
        return ge::GRAPH_FAILED;
    }
    if (maskMode_ != 0 && maskMode_ != 3) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: mask_mode must be 0 or 3, got %ld.",
                maskMode_);
        return ge::GRAPH_FAILED;
    }
    if (cmpRatio_ < 1 || cmpRatio_ > MAX_CMP_RATIO) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: cmp_ratio must be in [1,%ld], got %ld.",
                MAX_CMP_RATIO, cmpRatio_);
        return ge::GRAPH_FAILED;
    }
    if (maxSeqlenQ < -1) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: max_seqlen_q must be >= -1, got %ld.",
                maxSeqlenQ);
        return ge::GRAPH_FAILED;
    }
    if (layoutQ == nullptr || std::strcmp(layoutQ, "TND") != 0) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: layout_q must be 'TND'.");
        return ge::GRAPH_FAILED;
    }
    if (layoutK != nullptr && std::strcmp(layoutK, "PA_BBND") == 0) {
        isPa_ = true;
    } else if (layoutK == nullptr || std::strcmp(layoutK, "TND") == 0) {
        isPa_ = false;
    } else {
        OP_LOGE(tiling_context_->GetNodeName(),
                "qsli: layout_k must be 'TND' or 'PA_BBND', got '%s'.", layoutK);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingQsliFunc::ValidateCommonInputs()
{
    if (!CheckShape(tiling_context_->GetInputShape(Q), {totalQ_, N1, PACKED_D})) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "qsli: q shape must be [T=%ld, %ld, %ld].", totalQ_, N1, PACKED_D);
        return ge::GRAPH_FAILED;
    }
    if (InputDtype(tiling_context_, Q) != ge::DT_UINT8) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: q must be UINT8 packed MXFP4.");
        return ge::GRAPH_FAILED;
    }
    if (!CheckShape(tiling_context_->GetInputShape(W), {totalQ_, N1}) ||
        InputDtype(tiling_context_, W) != ge::DT_FLOAT) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: w must be FLOAT [T=%ld, %ld].",
                totalQ_, N1);
        return ge::GRAPH_FAILED;
    }
    if (!CheckShape(tiling_context_->GetInputShape(DESCALE_Q), {totalQ_, N1, 2, 2}) ||
        InputDtype(tiling_context_, DESCALE_Q) != ge::DT_UINT8) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "qsli: descale_q must be UINT8 [T=%ld, %ld, 2, 2].", totalQ_, N1);
        return ge::GRAPH_FAILED;
    }
    if (!CheckShape(tiling_context_->GetInputShape(CANDIDATE_BLOCK_INDICES),
                    {totalQ_, N2, CANDIDATE_CAPACITY}) ||
        InputDtype(tiling_context_, CANDIDATE_BLOCK_INDICES) != ge::DT_INT32) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "qsli: candidate_block_indices must be INT32 [T=%ld, 1, %ld].",
                totalQ_, CANDIDATE_CAPACITY);
        return ge::GRAPH_FAILED;
    }
    const auto *candLenShape = tiling_context_->GetInputShape(CANDIDATE_BLOCK_LENGTH);
    if (candLenShape == nullptr || Numel(candLenShape) != totalQ_ * N2 ||
        InputDtype(tiling_context_, CANDIDATE_BLOCK_LENGTH) != ge::DT_INT32) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "qsli: candidate_block_length must be INT32 with T*1=%ld entries.", totalQ_);
        return ge::GRAPH_FAILED;
    }
    if (!CheckShape(tiling_context_->GetInputShape(METADATA), {METADATA_ELEMENTS}) ||
        InputDtype(tiling_context_, METADATA) != ge::DT_INT32) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "qsli: metadata must be contiguous INT32 [%ld].", METADATA_ELEMENTS);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingQsliFunc::DerivePa()
{
    if (Present(tiling_context_, DESCALE_K)) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "qsli: descale_k must be absent for PA_BBND (scales live in the 544B record).");
        return ge::GRAPH_FAILED;
    }
    const auto *kShape = tiling_context_->GetInputShape(K);
    if (kShape == nullptr || kShape->GetStorageShape().GetDimNum() != 3) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: PA k must be rank 3.");
        return ge::GRAPH_FAILED;
    }
    const auto &ks = kShape->GetStorageShape();
    if (ks.GetDim(2) != PACKED_BLOCK_BYTES) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "qsli: PA k dim2 must be %ld (K[512]+scale[32]), got %ld.",
                PACKED_BLOCK_BYTES, ks.GetDim(2));
        return ge::GRAPH_FAILED;
    }
    blocksPerPage_ = ks.GetDim(1);
    const int64_t pageSize = blocksPerPage_ * CANDIDATE_BLOCK_SIZE;
    if (pageSize <= 0 || pageSize > MAX_PA_BLOCK_SIZE || pageSize % PA_BLOCK_ALIGNMENT != 0) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "qsli: PA block_size must be a multiple of %ld in (0,%ld], got %ld.",
                PA_BLOCK_ALIGNMENT, MAX_PA_BLOCK_SIZE, pageSize);
        return ge::GRAPH_FAILED;
    }
    const int64_t physicalBlocks = ks.GetDim(0);
    if (totalQ_ <= 0 || physicalBlocks <= 0) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "qsli: T(%ld) and the physical K block count(%ld) must be positive.",
                totalQ_, physicalBlocks);
        return ge::GRAPH_FAILED;
    }
    if (InputDtype(tiling_context_, K) != ge::DT_UINT8) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: k must be UINT8 packed MXFP4.");
        return ge::GRAPH_FAILED;
    }
    // axis-0 padding is allowed; inner strides must stay ND-contiguous and rows
    // must not overlap (the jit host enforces the same via _pa_axis0_stride).
    // Same reasoning as the TND path: aclnn hands the kernel a contiguous copy, so
    // the record pitch is the compact one, not the caller's padded axis-0 stride.
    const int64_t compactKeyStride = blocksPerPage_ * PACKED_BLOCK_BYTES;
    const int64_t declaredKeyStride =
        Axis0Stride(tiling_context_, K, false, compactKeyStride);
    if (declaredKeyStride < compactKeyStride) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "qsli: k axis-0 rows must not overlap (stride %ld < %ld).",
                declaredKeyStride, compactKeyStride);
        return ge::GRAPH_FAILED;
    }
    keyStride_ = compactKeyStride;
    const int64_t span = (physicalBlocks - 1) * keyStride_ + blocksPerPage_ * PACKED_BLOCK_BYTES;
    addr32_ = span < MAX_ADDRESS32_BYTES;
    tndKStride_ = PACKED_D;
    tndSStride_ = SCALE_K;
    tndRows_ = 1;

    hasTable_ = Present(tiling_context_, BLOCK_TABLE);
    if (hasTable_) {
        const auto *tableShape = tiling_context_->GetOptionalInputShape(BLOCK_TABLE);
        if (tableShape == nullptr || tableShape->GetStorageShape().GetDimNum() != 2 ||
            InputDtype(tiling_context_, BLOCK_TABLE) != ge::DT_INT32) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "qsli: block_table must be INT32 [B, max_pages].");
            return ge::GRAPH_FAILED;
        }
        tableWidth_ = tableShape->GetStorageShape().GetDim(1);
        logicalKCapacity_ = tableWidth_ * pageSize;
    } else {
        tableWidth_ = physicalBlocks;
        logicalKCapacity_ = physicalBlocks * pageSize;
    }
    tableCached_ = tableWidth_ <= PA_TABLE_CACHE_ENTRIES;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingQsliFunc::DeriveTnd()
{
    if (Present(tiling_context_, BLOCK_TABLE)) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: block_table must be absent for TND k.");
        return ge::GRAPH_FAILED;
    }
    const auto *kShape = tiling_context_->GetInputShape(K);
    if (kShape == nullptr || kShape->GetStorageShape().GetDimNum() != 3 ||
        kShape->GetStorageShape().GetDim(1) != N2 ||
        kShape->GetStorageShape().GetDim(2) != PACKED_D) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: TND k must have shape [T2, 1, %ld].",
                PACKED_D);
        return ge::GRAPH_FAILED;
    }
    tndRows_ = kShape->GetStorageShape().GetDim(0);
    if (tndRows_ <= 0 || totalQ_ <= 0) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: T1(%ld) and T2(%ld) must be positive.",
                totalQ_, tndRows_);
        return ge::GRAPH_FAILED;
    }
    if (InputDtype(tiling_context_, K) != ge::DT_UINT8) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: k must be UINT8 packed MXFP4.");
        return ge::GRAPH_FAILED;
    }
    if (!Present(tiling_context_, DESCALE_K)) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "qsli: descale_k is required for TND k, shape [T2, 1, 2, 2].");
        return ge::GRAPH_FAILED;
    }
    const auto *dkShape = tiling_context_->GetOptionalInputShape(DESCALE_K);
    if (dkShape == nullptr || dkShape->GetStorageShape().GetDimNum() != 4 ||
        dkShape->GetStorageShape().GetDim(0) != tndRows_ ||
        dkShape->GetStorageShape().GetDim(1) != N2 ||
        dkShape->GetStorageShape().GetDim(2) != 2 ||
        dkShape->GetStorageShape().GetDim(3) != 2) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "qsli: descale_k must have shape [T2=%ld, 1, 2, 2].", tndRows_);
        return ge::GRAPH_FAILED;
    }
    if (InputDtype(tiling_context_, DESCALE_K) != ge::DT_UINT8) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: descale_k must carry E8M0/uint8 bytes.");
        return ge::GRAPH_FAILED;
    }
    blocksPerPage_ = 1;
    keyStride_ = 0;
    // The aclnn/nnopbase layer copies every non-contiguous input into a contiguous
    // buffer before launch (NnopbaseGetUnContiguousTensors + aclnnContiguous, whose
    // workspace is appended to the size this tiling reports -- measured: tiling asks
    // for 5,767,168 B and aclnnXxxGetWorkspaceSize returns 22,545,408 B). The kernel
    // therefore always sees compact axis-0 strides, even when the caller passed an
    // axis-0-padded view, so the strides reported here must be the COMPACT ones.
    // Reporting the caller's padded stride (what GetInputStride returns, i.e. the
    // pre-copy descriptor) makes the kernel walk the contiguous copy with the wrong
    // pitch: indices stay right but every score is wrong.
    // Consequence: TndStrideMode is always 3 on the built-in path; the strided
    // specializations (modes 0/1/2) remain reachable only through the jit host,
    // which passes the caller's tensors straight to the kernel without a copy.
    const int64_t declaredKStride = Axis0Stride(tiling_context_, K, false, PACKED_D);
    const int64_t declaredSStride = Axis0Stride(tiling_context_, DESCALE_K, true, SCALE_K);
    if (declaredKStride < PACKED_D) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "qsli: k axis-0 rows must not overlap (stride %ld < %ld).",
                declaredKStride, PACKED_D);
        return ge::GRAPH_FAILED;
    }
    if (declaredSStride < SCALE_K) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "qsli: descale_k axis-0 rows must not overlap (stride %ld < %ld).",
                declaredSStride, SCALE_K);
        return ge::GRAPH_FAILED;
    }
    tndKStride_ = PACKED_D;
    tndSStride_ = SCALE_K;
    hasTable_ = false;
    tableWidth_ = 0;
    tableCached_ = true;
    addr32_ = false;
    logicalKCapacity_ = tndRows_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingQsliFunc::ValidateSequences()
{
    hasCuQ_ = Present(tiling_context_, CU_SEQLENS_Q);
    hasSeqQ_ = Present(tiling_context_, SEQUSED_Q);
    hasSeqK_ = Present(tiling_context_, SEQUSED_K);
    hasIdxOff_ = Present(tiling_context_, OUTPUT_IDX_OFFSET);
    resEqUsedQ_ = (maskMode_ == 0);
    if (resEqUsedQ_) {
        cmpRatio_ = 1;
    }
    const bool wantResidual = !resEqUsedQ_ && cmpRatio_ != 1;
    if (wantResidual != Present(tiling_context_, CMP_RESIDUAL_K)) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "qsli: cmp_residual_k is required iff mask_mode=3 and cmp_ratio != 1 "
                "(mask_mode=%ld cmp_ratio=%ld).", maskMode_, cmpRatio_);
        return ge::GRAPH_FAILED;
    }
    hasResidual_ = wantResidual;

    if (hasCuQ_) {
        const auto *cuShape = tiling_context_->GetOptionalInputShape(CU_SEQLENS_Q);
        if (cuShape == nullptr || Numel(cuShape) < 2 ||
            InputDtype(tiling_context_, CU_SEQLENS_Q) != ge::DT_INT32) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "qsli: cu_seqlens_q must be INT32 [B+1] with at least two boundaries.");
            return ge::GRAPH_FAILED;
        }
        batch_ = Numel(cuShape) - 1;
    } else {
        if (isPa_) {
            batch_ = 1;   // PA without cu_seqlens_q is a single batch
        } else {
            OP_LOGE(tiling_context_->GetNodeName(), "qsli: cu_seqlens_q is required for TND k.");
            return ge::GRAPH_FAILED;
        }
    }
    if (!isPa_) {
        if (!Present(tiling_context_, CU_SEQLENS_K) ||
            Numel(tiling_context_->GetOptionalInputShape(CU_SEQLENS_K)) != batch_ + 1 ||
            InputDtype(tiling_context_, CU_SEQLENS_K) != ge::DT_INT32) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "qsli: cu_seqlens_k must be INT32 [B+1=%ld] for TND k.", batch_ + 1);
            return ge::GRAPH_FAILED;
        }
    }
    if (batch_ <= 0) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: batch must be positive, got %ld.", batch_);
        return ge::GRAPH_FAILED;
    }
    if (hasSeqQ_ && (Numel(tiling_context_->GetOptionalInputShape(SEQUSED_Q)) != batch_ ||
                     InputDtype(tiling_context_, SEQUSED_Q) != ge::DT_INT32)) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: seqused_q must be INT32 [B=%ld].", batch_);
        return ge::GRAPH_FAILED;
    }
    if (hasSeqK_ && (Numel(tiling_context_->GetOptionalInputShape(SEQUSED_K)) != batch_ ||
                     InputDtype(tiling_context_, SEQUSED_K) != ge::DT_INT32)) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: seqused_k must be INT32 [B=%ld].", batch_);
        return ge::GRAPH_FAILED;
    }
    if (hasResidual_ &&
        (Numel(tiling_context_->GetOptionalInputShape(CMP_RESIDUAL_K)) != batch_ ||
         InputDtype(tiling_context_, CMP_RESIDUAL_K) != ge::DT_INT32)) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "qsli: cmp_residual_k must be INT32 [B=%ld].", batch_);
        return ge::GRAPH_FAILED;
    }
    if (hasIdxOff_ &&
        (Numel(tiling_context_->GetOptionalInputShape(OUTPUT_IDX_OFFSET)) != totalQ_ * N2 ||
         InputDtype(tiling_context_, OUTPUT_IDX_OFFSET) != ge::DT_INT32)) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "qsli: output_idx_offset must be INT32 with T*1=%ld entries.", totalQ_);
        return ge::GRAPH_FAILED;
    }
    if (hasTable_ &&
        tiling_context_->GetOptionalInputShape(BLOCK_TABLE)->GetStorageShape().GetDim(0) != batch_) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "qsli: block_table dim0 must equal batch(%ld).", batch_);
        return ge::GRAPH_FAILED;
    }
    if (isPa_ && !hasTable_ && batch_ != 1) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "qsli: block_table is required when batch(%ld) > 1 for PA_BBND.", batch_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingQsliFunc::ValidateOutputs()
{
    if (!CheckShape(tiling_context_->GetOutputShape(INDICES), {totalQ_, N2, topk_})) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "qsli: indices shape must be [T=%ld, 1, topk=%ld].", totalQ_, topk_);
        return ge::GRAPH_FAILED;
    }
    if (!CheckShape(tiling_context_->GetOutputShape(VALUES), {totalQ_, N2, topk_})) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "qsli: values shape must be [T=%ld, 1, topk=%ld].", totalQ_, topk_);
        return ge::GRAPH_FAILED;
    }
    const auto *idxDesc = tiling_context_->GetOutputDesc(INDICES);
    const auto *valDesc = tiling_context_->GetOutputDesc(VALUES);
    if (idxDesc == nullptr || idxDesc->GetDataType() != ge::DT_INT32) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: indices must be INT32.");
        return ge::GRAPH_FAILED;
    }
    if (valDesc == nullptr || valDesc->GetDataType() != ge::DT_BF16) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: values must be BF16.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

uint32_t TilingQsliFunc::ComputeTilingKey() const
{
    // Byte-for-byte port of the jit host's `key` dict + TilingKeySchema.pack.
    uint32_t capacityIndex = 0;
    for (size_t i = 0; i < TOPK_CAPACITY_COUNT; ++i) {
        if (TOPK_CAPACITIES[i] == topkCapacity_) {
            capacityIndex = static_cast<uint32_t>(i);
            break;
        }
    }
    const uint32_t writeValues = returnValue_ ? 1U : 0U;
    const uint32_t isPa = isPa_ ? 1U : 0U;
    // TndStrideMode bit0: K stride is the compact 64 bytes; bit1: scale stride is
    // the compact 4 bytes. Forced to 0 on the PA path (mirrors the jit host).
    uint32_t tndStrideMode = 0;
    if (!isPa_) {
        tndStrideMode = static_cast<uint32_t>((tndKStride_ == PACKED_D ? 1 : 0) |
                                              ((tndSStride_ == SCALE_K ? 1 : 0) << 1));
    }
    // PairMin is 544 (paired record DMA) or 0 (single record); the jit host forces
    // 0 on the TND path, where the field is not read at all.
    const uint32_t pairMinIndex = (!isPa_ || pairMin_ == 0) ? 1U : 0U;
    return writeValues | (isPa << KEY_SHIFT_IS_PA) | (capacityIndex << KEY_SHIFT_TOPK) |
           (tndStrideMode << KEY_SHIFT_TND_STRIDE) | (pairMinIndex << KEY_SHIFT_PAIR_MIN);
}

int64_t TilingQsliFunc::WorkspaceBytes() const
{
    // Mirrors ws_layout() in the kernel module: one flat buffer the kernel slices
    // with pl.addptr into stg_k / stg_s / key_ws / part_key / part_pos.
    const int64_t stageRows = numWorkers_ * STAGE_DEPTH * TILE_N;
    const int64_t stgK = stageRows * PACKED_D;
    const int64_t stgS = stageRows * SCALE_K;
    const int64_t keyWs = numWorkers_ * MAX_TILES * 2 * HALF_TOKENS * 2;
    const int64_t part = WORKSPACE_SLOTS * topkCapacity_ * 2;
    return stgK + stgS + keyWs + 2 * part;
}

ge::graphStatus TilingQsliFunc::RunTiling()
{
    auto platform_info = platform_ascendc::PlatformAscendC(tiling_context_->GetPlatformInfo());
    const uint32_t aicCores = platform_info.GetCoreNumAic();
    if (aicCores == 0) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: platform reports 0 AIC cores.");
        return ge::GRAPH_FAILED;
    }
    numWorkers_ = std::min(static_cast<int64_t>(aicCores), MAX_CUBE_WORKERS);

    auto ret = ParseAttrs();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    const auto *qShape = tiling_context_->GetInputShape(Q);
    if (qShape == nullptr || qShape->GetStorageShape().GetDimNum() != 3) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: q must be rank 3 [T, 32, 64].");
        return ge::GRAPH_FAILED;
    }
    totalQ_ = qShape->GetStorageShape().GetDim(0);
    if (totalQ_ <= 0) {
        OP_LOGE(tiling_context_->GetNodeName(), "qsli: T must be positive, got %ld.", totalQ_);
        return ge::GRAPH_FAILED;
    }

    ret = ValidateCommonInputs();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = isPa_ ? DerivePa() : DeriveTnd();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = ValidateSequences();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = ValidateOutputs();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // next power of two >= topk, floored at 16 (jit: max(16, 1 << (topk-1).bit_length()))
    topkCapacity_ = 16;
    while (topkCapacity_ < topk_) {
        topkCapacity_ <<= 1;
    }
    // determinism level 3 disables the paired-record DMA path (jit:
    // _batch_consistency_enabled()).
    pairMin_ = (tiling_context_->GetDeterministicLevel() == DETERMINISTIC_LEVEL_BATCH_CONSISTENT)
                   ? 0
                   : PACKED_BLOCK_BYTES;

    tiling_data_.set_rt_batch(batch_);
    tiling_data_.set_rt_key_stride(keyStride_);
    tiling_data_.set_rt_table_width(isPa_ ? tableWidth_ : 0);
    tiling_data_.set_rt_total_q(totalQ_);
    tiling_data_.set_rt_table_cached(isPa_ ? static_cast<int64_t>(tableCached_) : 1);
    tiling_data_.set_rt_tnd_k_stride(tndKStride_);
    tiling_data_.set_rt_tnd_s_stride(tndSStride_);
    tiling_data_.set_rt_tnd_rows(tndRows_);
    tiling_data_.set_rt_addr32(isPa_ ? static_cast<int64_t>(addr32_) : 0);
    tiling_data_.set_rt_has_cu_q(static_cast<int64_t>(hasCuQ_));
    tiling_data_.set_rt_has_seq_q(static_cast<int64_t>(hasSeqQ_));
    tiling_data_.set_rt_has_seq_k(static_cast<int64_t>(hasSeqK_));
    tiling_data_.set_rt_has_residual(static_cast<int64_t>(hasResidual_));
    tiling_data_.set_rt_res_eq_used_q(static_cast<int64_t>(resEqUsedQ_));
    tiling_data_.set_rt_has_idx_off(static_cast<int64_t>(hasIdxOff_));
    tiling_data_.set_rt_has_table(isPa_ ? static_cast<int64_t>(hasTable_) : 0);
    tiling_data_.set_rt_k_capacity(logicalKCapacity_);
    tiling_data_.set_rt_cmp_ratio(cmpRatio_);
    tiling_data_.set_rt_blocks_per_page(blocksPerPage_);
    tiling_data_.set_rt_page_param(PageParam(blocksPerPage_));
    tiling_data_.set_rt_topk(topk_);
    tiling_data_.set_num_workers(numWorkers_);

    size_t *workspace = tiling_context_->GetWorkspaceSizes(1);
    workspace[0] = static_cast<size_t>(WorkspaceBytes());

    tiling_context_->SetTilingKey(ComputeTilingKey());
    tiling_data_.SaveToBuffer(tiling_context_->GetRawTilingData()->GetData(),
                              tiling_context_->GetRawTilingData()->GetCapacity());
    tiling_context_->GetRawTilingData()->SetDataSize(tiling_data_.GetDataSize());
    tiling_context_->SetBlockDim(static_cast<uint32_t>(numWorkers_));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingQsliFunc tilingObject(context);
    auto ret = tilingObject.RunTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "quant_sparse_lightning_indexer tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantSparseLightningIndexer)
    .Tiling(TilingFunc);

} // namespace optiling
