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
 * \file flex_attention_tiling.cpp
 * \brief FlexAttention host tiling.
 *
 * C++ port of the Python host wrapper (`op_kernel/flex_attention.py::flex_attention`).
 * The two implementations must stay equivalent:
 *   - varlen/bshd layout contract and full input validation;
 *   - exact per-batch seq lists from actual_seq_q / actual_seq_kv (Host ValueDepend);
 *   - descriptor shapes derived from seq lists (r_f/p_f/w_f) and cross-checked;
 *   - per-work_id cost-balanced core split via binary-search (build_work_ranges);
 *   - tiling key = (Schedule, Layout, IsDN, HoleChunk);
 *   - blockDim = min(platform core num, total_work).
 *
 * Value dependency:
 *   actual_seq_q/actual_seq_kv and the schedule cost tensor (block_compute) are
 *   registered via TilingInputsDataDependency. The auto-generated aclnn interface
 *   exposes them as aclIntArray (host values), so the framework hands tiling a
 *   Host placement tensor. If a caller uses the Tensor variant instead, the data
 *   may still be device-resident: we then fall back to a *safe upper bound*
 *   (never under-estimating total_work) plus a uniform split, and let the kernel
 *   self-clamp phantom work ids.
 */

#include "flex_attention_tiling.h"
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

constexpr int64_t TS = 128;      // kernel Q-tile row cap
constexpr int64_t TKV = 128;     // kernel KV-tile width
constexpr int64_t MAX_TILING_CORES = 63;  // work_starts capacity (63 cores + sentinel)
constexpr int64_t MASK_NUM_LIMIT = 256;   // host sanity limit (kernel has no upper bound)
constexpr int64_t HOLE_CHUNK = 16;        // kernel CHUNK

enum InputIndex {
    QUERY = 0,
    KEY,
    VALUE,
    MASKR,
    MASK_START,
    MASK_LEN,
    BLOCK_RANGE,
    BLOCK_COMPUTE,
    BLOCK_MASK,
    ACTUAL_SEQ_Q,
    ACTUAL_SEQ_KV,
};

enum OutputIndex {
    ATTN_OUT = 0,
    SOFTMAX_MAX_OUT,
    SOFTMAX_SUM_OUT,
};

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b <= 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

// Build contiguous per-core ranges minimizing the maximum cost (exact min-max
// via binary search) -- byte-for-byte port of the Python `build_work_ranges`.
static std::vector<int64_t> BuildWorkRanges(const std::vector<int64_t> &costs, int64_t numCores)
{
    int64_t totalWork = static_cast<int64_t>(costs.size());
    if (numCores <= 0) {
        numCores = 1;
    }
    std::vector<int64_t> prefix(static_cast<size_t>(totalWork) + 1, 0);
    for (int64_t i = 0; i < totalWork; ++i) {
        prefix[static_cast<size_t>(i) + 1] = prefix[static_cast<size_t>(i)] + costs[static_cast<size_t>(i)];
    }

    auto numParts = [&](int64_t cap) -> int64_t {
        int64_t parts = 1;
        int64_t cur = 0;
        for (int64_t i = 0; i < totalWork; ++i) {
            int64_t c = costs[static_cast<size_t>(i)];
            if (c > cap) {
                return totalWork + 1;
            }
            if (cur + c > cap) {
                ++parts;
                cur = 0;
            }
            cur += c;
        }
        return parts;
    };

    int64_t maxCost = 0;
    for (auto c : costs) {
        maxCost = std::max(maxCost, c);
    }
    int64_t lo = maxCost;
    int64_t hi = (prefix[static_cast<size_t>(totalWork)] > 0) ? prefix[static_cast<size_t>(totalWork)] : 1;
    while (lo < hi) {
        int64_t mid = (lo + hi) / 2;
        if (numParts(mid) <= numCores) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    int64_t cap = lo;

    std::vector<int64_t> starts;
    int64_t core = 0;
    int64_t start = 0;
    int64_t cur = 0;
    int64_t i = 0;
    while (i < totalWork && core < numCores) {
        int64_t c = costs[static_cast<size_t>(i)];
        if (cur + c > cap && cur > 0) {
            starts.push_back(start);
            ++core;
            start = i;
            cur = 0;
            continue;
        }
        cur += c;
        ++i;
    }
    starts.push_back(start);
    for (int64_t k = core + 1; k < numCores; ++k) {
        starts.push_back(totalWork);
    }
    starts.push_back(totalWork);
    return starts;
}

static std::vector<int64_t> BuildUniformRanges(int64_t totalWork, int64_t numCores)
{
    std::vector<int64_t> starts;
    if (numCores <= 0) {
        starts.push_back(totalWork);
        return starts;
    }
    int64_t per = totalWork / numCores;
    int64_t rem = totalWork % numCores;
    int64_t acc = 0;
    for (int64_t i = 0; i < numCores; ++i) {
        starts.push_back(acc);
        acc += per + (static_cast<int64_t>(i) < rem ? 1 : 0);
    }
    starts.push_back(totalWork);
    return starts;
}

class TilingFlexAttentionFunc {
public:
    explicit TilingFlexAttentionFunc(gert::TilingContext *tiling_context)
        : tiling_context_(tiling_context) {}
    ge::graphStatus RunTiling();

private:
    ge::graphStatus ParseShapesAndAttrs();
    ge::graphStatus ValidateDescriptors();
    ge::graphStatus ValidateOutputs();
    std::vector<int64_t> ComputeWorkRanges();

    static bool IsHostPlacement(const gert::Tensor *tensor)
    {
        return tensor != nullptr && gert::TensorPlacementUtils::IsOnHost(tensor->GetPlacement());
    }
    static bool CheckShape(const gert::StorageShape *shape, std::initializer_list<int64_t> dims);
    static ge::DataType GetInputDtype(gert::TilingContext *context, size_t index);
    static ge::DataType GetOutputDtype(gert::TilingContext *context, size_t index);

    gert::TilingContext *tiling_context_ = nullptr;
    FlexAttentionTilingData tiling_data_;
    int64_t tq_ = 0;
    int64_t tkv_ = 0;
    int64_t nDim_ = 0;
    int64_t kvHeads_ = 0;
    int64_t headDim_ = 0;
    int64_t batch_ = 0;
    int64_t rF_ = 0;
    int64_t wF_ = 0;
    int64_t maskNum_ = 0;
    int64_t s1_ = 0;
    int64_t s2_ = 0;
    int64_t layout_ = 0;  // 0 = varlen (TND), 1 = bshd
    uint32_t cubeCoreNum_ = 0;
    bool seqQExact_ = false;      // per-batch q lengths from actual_seq_q Host data
    bool seqKVExact_ = false;     // per-batch kv lengths from actual_seq_kv Host data
    bool haveCostData_ = false;   // block_compute Host data available
    int64_t totalWork_ = 0;
    int64_t actualCores_ = 1;
    std::vector<int64_t> seqQ_;
    std::vector<int64_t> seqKV_;
    std::vector<int64_t> costs_;  // per work_id (bi outer, then n, then q-tile)
};

ge::DataType TilingFlexAttentionFunc::GetInputDtype(gert::TilingContext *context, size_t index)
{
    const auto *desc = context->GetInputDesc(index);
    return desc == nullptr ? ge::DT_UNDEFINED : desc->GetDataType();
}

ge::DataType TilingFlexAttentionFunc::GetOutputDtype(gert::TilingContext *context, size_t index)
{
    const auto *desc = context->GetOutputDesc(index);
    return desc == nullptr ? ge::DT_UNDEFINED : desc->GetDataType();
}

bool TilingFlexAttentionFunc::CheckShape(const gert::StorageShape *shape, std::initializer_list<int64_t> dims)
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

ge::graphStatus TilingFlexAttentionFunc::ParseShapesAndAttrs()
{
    // ---- layout attribute (mirrors the Python wrapper's `layout` argument) ----
    const char *layout = tiling_context_->GetAttrs() == nullptr ? nullptr : tiling_context_->GetAttrs()->GetStr(0);
    if (layout == nullptr || std::strcmp(layout, "varlen") == 0) {
        layout_ = 0;
    } else if (std::strcmp(layout, "bshd") == 0) {
        layout_ = 1;
    } else {
        OP_LOGE(tiling_context_->GetNodeName(),
                "flex_attention: layout must be 'varlen' or 'bshd', got '%s'.", layout);
        return ge::GRAPH_FAILED;
    }

    const auto *qShape = tiling_context_->GetInputShape(QUERY);
    const auto *kShape = tiling_context_->GetInputShape(KEY);
    const auto *vShape = tiling_context_->GetInputShape(VALUE);
    if (qShape == nullptr || kShape == nullptr || vShape == nullptr) {
        OP_LOGE(tiling_context_->GetNodeName(), "flex_attention: query/key/value shape is null.");
        return ge::GRAPH_FAILED;
    }
    const auto &qs = qShape->GetStorageShape();
    const auto &ks = kShape->GetStorageShape();
    const auto &vs = vShape->GetStorageShape();

    // ---- dtype contract (mirrors the Python asserts) ----
    const ge::DataType qdtype = GetInputDtype(tiling_context_, QUERY);
    const ge::DataType kdtype = GetInputDtype(tiling_context_, KEY);
    const ge::DataType vdtype = GetInputDtype(tiling_context_, VALUE);
    if (qdtype != ge::DT_FLOAT16 && qdtype != ge::DT_BF16) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "flex_attention: query dtype must be FLOAT16/BFLOAT16, got %d.", static_cast<int>(qdtype));
        return ge::GRAPH_FAILED;
    }
    if (kdtype != qdtype || vdtype != qdtype) {
        OP_LOGE(tiling_context_->GetNodeName(), "flex_attention: query/key/value must share the same dtype.");
        return ge::GRAPH_FAILED;
    }

    if (layout_ == 0) {
        // varlen: q/k/v [L, H, D]
        if (qs.GetDimNum() != 3 || ks.GetDimNum() != 3 || vs.GetDimNum() != 3) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "flex_attention: varlen layout requires q/k/v of rank 3.");
            return ge::GRAPH_FAILED;
        }
        tq_ = qs.GetDim(0);
        nDim_ = qs.GetDim(1);
        headDim_ = qs.GetDim(2);
        tkv_ = ks.GetDim(0);
        kvHeads_ = ks.GetDim(1);
        if (ks.GetDim(2) != headDim_) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "flex_attention: k head_dim(%ld) != q head_dim(%ld).", ks.GetDim(2), headDim_);
            return ge::GRAPH_FAILED;
        }
        if (!CheckShape(vShape, {tkv_, kvHeads_, headDim_})) {
            OP_LOGE(tiling_context_->GetNodeName(), "flex_attention: value shape must be [tkv, kv_heads, d].");
            return ge::GRAPH_FAILED;
        }
        s1_ = 0;
        s2_ = 0;

        const auto *aqShape = tiling_context_->GetInputShape(ACTUAL_SEQ_Q);
        const auto *akvShape = tiling_context_->GetInputShape(ACTUAL_SEQ_KV);
        if (aqShape == nullptr || akvShape == nullptr) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "flex_attention: varlen layout requires actual_seq_q/actual_seq_kv.");
            return ge::GRAPH_FAILED;
        }
        const int64_t b = aqShape->GetStorageShape().GetDim(0);
        if (b <= 0 || akvShape->GetStorageShape().GetDim(0) != b) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "flex_attention: actual_seq_q/actual_seq_kv batch mismatch or empty.");
            return ge::GRAPH_FAILED;
        }
        if (GetInputDtype(tiling_context_, ACTUAL_SEQ_Q) != ge::DT_INT64 ||
            GetInputDtype(tiling_context_, ACTUAL_SEQ_KV) != ge::DT_INT32) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "flex_attention: actual_seq_q must be int64 and actual_seq_kv int32.");
            return ge::GRAPH_FAILED;
        }

        const auto *aqTensor = tiling_context_->GetInputTensor(ACTUAL_SEQ_Q);
        const auto *akvTensor = tiling_context_->GetInputTensor(ACTUAL_SEQ_KV);
        if (IsHostPlacement(aqTensor)) {
            const int64_t *aq = aqTensor->GetData<int64_t>();
            if (aq == nullptr) {
                OP_LOGE(tiling_context_->GetNodeName(), "flex_attention: actual_seq_q data is null.");
                return ge::GRAPH_FAILED;
            }
            seqQ_.resize(static_cast<size_t>(b));
            int64_t prevQ = 0;
            int64_t sumQ = 0;
            for (int64_t i = 0; i < b; ++i) {
                const int64_t curQ = aq[i];
                seqQ_[static_cast<size_t>(i)] = curQ - prevQ;
                prevQ = curQ;
                sumQ += seqQ_[static_cast<size_t>(i)];
            }
            if (sumQ != tq_) {
                OP_LOGE(tiling_context_->GetNodeName(),
                        "flex_attention: sum(actual_seq_q)=%ld != tq %ld.", sumQ, tq_);
                return ge::GRAPH_FAILED;
            }
            seqQExact_ = true;
        }
        if (IsHostPlacement(akvTensor)) {
            const int32_t *akv = akvTensor->GetData<int32_t>();
            if (akv == nullptr) {
                OP_LOGE(tiling_context_->GetNodeName(), "flex_attention: actual_seq_kv data is null.");
                return ge::GRAPH_FAILED;
            }
            seqKV_.resize(static_cast<size_t>(b));
            int64_t prevKV = 0;
            int64_t sumKV = 0;
            for (int64_t i = 0; i < b; ++i) {
                const int64_t curKV = akv[i];
                seqKV_[static_cast<size_t>(i)] = curKV - prevKV;
                prevKV = curKV;
                sumKV += seqKV_[static_cast<size_t>(i)];
            }
            if (sumKV != tkv_) {
                OP_LOGE(tiling_context_->GetNodeName(),
                        "flex_attention: sum(actual_seq_kv)=%ld != tkv %ld.", sumKV, tkv_);
                return ge::GRAPH_FAILED;
            }
            seqKVExact_ = true;
        }
        // actual_seq_q is an int64 ValueDepend input, so the auto-generated aclnn
        // interface passes it as aclIntArray and the framework provides Host data.
        // actual_seq_kv is int32 and stays a Tensor in that interface: when its
        // Host copy is unavailable, r_f/w_f are taken from the descriptors and
        // validated against them (the kernel still reads the device tensor).
        batch_ = b;
    } else {
        // bshd: q [B, S, H, D], k/v [B, S_kv, H_kv, D]
        if (qs.GetDimNum() != 4 || ks.GetDimNum() != 4 || vs.GetDimNum() != 4) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "flex_attention: bshd layout requires q/k/v of rank 4.");
            return ge::GRAPH_FAILED;
        }
        batch_ = qs.GetDim(0);
        s1_ = qs.GetDim(1);
        nDim_ = qs.GetDim(2);
        headDim_ = qs.GetDim(3);
        if (ks.GetDim(0) != batch_ || ks.GetDim(3) != headDim_) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "flex_attention: k shape must be [B, S_kv, H_kv, D] matching q.");
            return ge::GRAPH_FAILED;
        }
        s2_ = ks.GetDim(1);
        kvHeads_ = ks.GetDim(2);
        if (!CheckShape(vShape, {batch_, s2_, kvHeads_, headDim_})) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "flex_attention: value shape must be [B, S_kv, H_kv, D].");
            return ge::GRAPH_FAILED;
        }
        tq_ = batch_ * s1_;
        tkv_ = batch_ * s2_;
        seqQ_.assign(static_cast<size_t>(batch_), s1_);
        seqKV_.assign(static_cast<size_t>(batch_), s2_);
        seqQExact_ = true;
        seqKVExact_ = true;
    }

    if (headDim_ < 8 || headDim_ > TS) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "flex_attention: head_dim must be in [8, %ld], got %ld.", TS, headDim_);
        return ge::GRAPH_FAILED;
    }
    if (nDim_ <= 0 || batch_ <= 0 || kvHeads_ <= 0) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "flex_attention: invalid n_dim(%ld)/batch(%ld)/kv_heads(%ld).", nDim_, batch_, kvHeads_);
        return ge::GRAPH_FAILED;
    }
    if (nDim_ % kvHeads_ != 0) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "flex_attention: n_dim(%ld) not divisible by kv_heads(%ld).", nDim_, kvHeads_);
        return ge::GRAPH_FAILED;
    }

    // ---- derive descriptor sizes exactly like the Python wrapper ----
    if (seqQExact_) {
        int64_t maxSq = 0;
        for (int64_t sq : seqQ_) {
            maxSq = std::max(maxSq, sq);
        }
        rF_ = CeilDiv(maxSq, TS);
    } else {
        // No Host seq: p_f must come from the maskr descriptor.
        const auto *maskrShape = tiling_context_->GetInputShape(MASKR);
        if (maskrShape == nullptr || maskrShape->GetStorageShape().GetDimNum() != 2) {
            OP_LOGE(tiling_context_->GetNodeName(), "flex_attention: maskr must be rank-2 [B, Pf].");
            return ge::GRAPH_FAILED;
        }
        const int64_t pF = maskrShape->GetStorageShape().GetDim(1);
        if (pF <= 0 || pF % TS != 0) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "flex_attention: maskr dim1(%ld) must be a positive multiple of %ld.", pF, TS);
            return ge::GRAPH_FAILED;
        }
        rF_ = pF / TS;
    }

    if (seqKVExact_) {
        int64_t maxSkv = 0;
        for (int64_t skv : seqKV_) {
            maxSkv = std::max(maxSkv, skv);
        }
        wF_ = (CeilDiv(maxSkv, TKV) + 31) / 32;
    } else {
        // w_f from the block_compute descriptor: [B, r_f, w_f] (Tensor variant)
        // or a flat ValueDepend buffer (array variant) -> numel / (B * r_f).
        const auto *scShape = tiling_context_->GetOptionalInputShape(BLOCK_COMPUTE);
        if (scShape == nullptr) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "flex_attention: block_compute [B, r_f, w_f] is required.");
            return ge::GRAPH_FAILED;
        }
        const auto &sc = scShape->GetStorageShape();
        if (sc.GetDimNum() == 3) {
            wF_ = sc.GetDim(2);
        } else if (sc.GetDimNum() == 1 && batch_ > 0 && rF_ > 0) {
            wF_ = sc.GetDim(0) / (batch_ * rF_);
        } else {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "flex_attention: block_compute descriptor has unsupported rank %zu.", sc.GetDimNum());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingFlexAttentionFunc::ValidateDescriptors()
{
    // maskr [B, Pf] int32
    if (!CheckShape(tiling_context_->GetInputShape(MASKR), {batch_, rF_ * TS})) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "flex_attention: maskr shape must be [B=%ld, Pf=%ld].", batch_, rF_ * TS);
        return ge::GRAPH_FAILED;
    }
    if (GetInputDtype(tiling_context_, MASKR) != ge::DT_INT32) {
        OP_LOGE(tiling_context_->GetNodeName(), "flex_attention: maskr must be int32.");
        return ge::GRAPH_FAILED;
    }

    // mask_start/mask_len [B, mask_num, Pf] int32 (mask_num derived from dim1)
    const auto *maskStartShape = tiling_context_->GetOptionalInputShape(MASK_START);
    if (maskStartShape != nullptr) {
        const auto &ms = maskStartShape->GetStorageShape();
        if (ms.GetDimNum() != 3) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "flex_attention: mask_start must be rank-3 [B, mask_num, Pf], got rank %zu.",
                    static_cast<size_t>(ms.GetDimNum()));
            return ge::GRAPH_FAILED;
        }
        maskNum_ = ms.GetDim(1);
        if (maskNum_ < 0 || maskNum_ > MASK_NUM_LIMIT) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "flex_attention: mask_num %ld exceeds sanity limit %ld.", maskNum_, MASK_NUM_LIMIT);
            return ge::GRAPH_FAILED;
        }
        if (!CheckShape(maskStartShape, {batch_, maskNum_, rF_ * TS})) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "flex_attention: mask_start shape must be [B=%ld, mask_num=%ld, Pf=%ld].",
                    batch_, maskNum_, rF_ * TS);
            return ge::GRAPH_FAILED;
        }
        const auto *maskLenShape = tiling_context_->GetOptionalInputShape(MASK_LEN);
        if (maskLenShape == nullptr) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "flex_attention: mask_len is required when mask_start is present.");
            return ge::GRAPH_FAILED;
        }
        if (!CheckShape(maskLenShape, {batch_, maskNum_, rF_ * TS})) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "flex_attention: mask_len shape must match mask_start.");
            return ge::GRAPH_FAILED;
        }
        if (GetInputDtype(tiling_context_, MASK_START) != ge::DT_INT32 ||
            GetInputDtype(tiling_context_, MASK_LEN) != ge::DT_INT32) {
            OP_LOGE(tiling_context_->GetNodeName(), "flex_attention: mask_start/mask_len must be int32.");
            return ge::GRAPH_FAILED;
        }
    } else {
        maskNum_ = 0;
    }

    // block_range [B, r_f] int32; block_compute/block_mask [B, r_f, w_f]
    if (!CheckShape(tiling_context_->GetOptionalInputShape(BLOCK_RANGE), {batch_, rF_})) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "flex_attention: block_range shape must be [B=%ld, r_f=%ld].", batch_, rF_);
        return ge::GRAPH_FAILED;
    }
    // aclIntArray ValueDepend inputs arrive as a flat 1D buffer (numel only);
    // the Tensor variant keeps the declared [B, r_f, w_f] rank. Accept both.
    const auto *computeShape = tiling_context_->GetOptionalInputShape(BLOCK_COMPUTE);
    if (computeShape == nullptr) {
        OP_LOGE(tiling_context_->GetNodeName(), "flex_attention: block_compute is required.");
        return ge::GRAPH_FAILED;
    }
    const auto &computeStorage = computeShape->GetStorageShape();
    if (computeStorage.GetDimNum() == 3) {
        if (!CheckShape(computeShape, {batch_, rF_, wF_})) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "flex_attention: block_compute shape must be [B=%ld, r_f=%ld, w_f=%ld].", batch_, rF_, wF_);
            return ge::GRAPH_FAILED;
        }
    } else {
        int64_t numel = 1;
        for (size_t i = 0; i < computeStorage.GetDimNum(); ++i) {
            numel *= computeStorage.GetDim(i);
        }
        if (numel != batch_ * rF_ * wF_) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "flex_attention: block_compute numel(%ld) != B*r_f*w_f (%ld*%ld*%ld).",
                    numel, batch_, rF_, wF_);
            return ge::GRAPH_FAILED;
        }
    }
    if (!CheckShape(tiling_context_->GetOptionalInputShape(BLOCK_MASK), {batch_, rF_, wF_})) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "flex_attention: block_mask shape must be [B=%ld, r_f=%ld, w_f=%ld].", batch_, rF_, wF_);
        return ge::GRAPH_FAILED;
    }
    if (GetInputDtype(tiling_context_, BLOCK_RANGE) != ge::DT_INT32) {
        OP_LOGE(tiling_context_->GetNodeName(), "flex_attention: block_range must be int32.");
        return ge::GRAPH_FAILED;
    }
    if (GetInputDtype(tiling_context_, BLOCK_COMPUTE) != ge::DT_INT64) {
        OP_LOGE(tiling_context_->GetNodeName(), "flex_attention: block_compute must be int64.");
        return ge::GRAPH_FAILED;
    }
    if (GetInputDtype(tiling_context_, BLOCK_MASK) != ge::DT_UINT32) {
        OP_LOGE(tiling_context_->GetNodeName(), "flex_attention: block_mask must be uint32.");
        return ge::GRAPH_FAILED;
    }
    // actual_seq_q is also a flat ValueDepend buffer; actual_seq_kv stays [B].
    if (layout_ == 0) {
        const auto *aqShape = tiling_context_->GetInputShape(ACTUAL_SEQ_Q);
        const auto *akvShape = tiling_context_->GetInputShape(ACTUAL_SEQ_KV);
        if (aqShape == nullptr || akvShape == nullptr ||
            !CheckShape(akvShape, {batch_})) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "flex_attention: actual_seq_q/actual_seq_kv must be [B=%ld].", batch_);
            return ge::GRAPH_FAILED;
        }
        int64_t aqNumel = 1;
        const auto &aqStorage = aqShape->GetStorageShape();
        for (size_t i = 0; i < aqStorage.GetDimNum(); ++i) {
            aqNumel *= aqStorage.GetDim(i);
        }
        if (aqNumel != batch_) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "flex_attention: actual_seq_q numel(%ld) != B(%ld).", aqNumel, batch_);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingFlexAttentionFunc::ValidateOutputs()
{
    if (!CheckShape(tiling_context_->GetOutputShape(ATTN_OUT), {tq_, nDim_, headDim_})) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "flex_attention: attn_out shape must be [tq=%ld, n=%ld, d=%ld].", tq_, nDim_, headDim_);
        return ge::GRAPH_FAILED;
    }
    if (GetOutputDtype(tiling_context_, ATTN_OUT) != GetInputDtype(tiling_context_, QUERY)) {
        OP_LOGE(tiling_context_->GetNodeName(), "flex_attention: attn_out dtype must match query.");
        return ge::GRAPH_FAILED;
    }
    if (!CheckShape(tiling_context_->GetOutputShape(SOFTMAX_MAX_OUT), {nDim_, tq_}) ||
        !CheckShape(tiling_context_->GetOutputShape(SOFTMAX_SUM_OUT), {nDim_, tq_})) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "flex_attention: softmax_max/sum shape must be [n=%ld, tq=%ld].", nDim_, tq_);
        return ge::GRAPH_FAILED;
    }
    if (GetOutputDtype(tiling_context_, SOFTMAX_MAX_OUT) != ge::DT_FLOAT ||
        GetOutputDtype(tiling_context_, SOFTMAX_SUM_OUT) != ge::DT_FLOAT) {
        OP_LOGE(tiling_context_->GetNodeName(), "flex_attention: softmax_max/sum must be float32.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

std::vector<int64_t> TilingFlexAttentionFunc::ComputeWorkRanges()
{
    // ---- per-work_id cost from block_compute popcount (Python tile_costs) ----
    const auto *scTensor = tiling_context_->GetOptionalInputTensor(BLOCK_COMPUTE);
    if (IsHostPlacement(scTensor)) {
        const int64_t *sc = scTensor->GetData<int64_t>();
        if (sc != nullptr) {
            haveCostData_ = true;
            costs_.clear();
            costs_.reserve(static_cast<size_t>(totalWork_));
            for (int64_t bi = 0; bi < batch_; ++bi) {
                const int64_t blkM = CeilDiv(seqQ_[static_cast<size_t>(bi)], TS);
                for (int64_t n = 0; n < nDim_; ++n) {
                    for (int64_t m = 0; m < blkM; ++m) {
                        int64_t cnt = 0;
                        for (int64_t w = 0; w < wF_; ++w) {
                            const int64_t word = sc[(bi * rF_ + m) * wF_ + w];
                            cnt += __builtin_popcountll(static_cast<uint64_t>(word));
                        }
                        costs_.push_back(cnt);
                    }
                }
            }
        }
    }

    if (seqQExact_) {
        totalWork_ = 0;
        for (int64_t sq : seqQ_) {
            totalWork_ += CeilDiv(sq, TS);
        }
        totalWork_ *= nDim_;
        if (static_cast<int64_t>(costs_.size()) != totalWork_) {
            haveCostData_ = false;
            costs_.clear();
        }
    } else {
        // Safe upper bound: sum_b ceil(sq_b/TS) <= ceil(sum(sq_b)/TS) + B - 1.
        // Over-estimation is harmless (kernel clamps phantom work ids); under-
        // estimation would silently drop work, so never use an even split here.
        totalWork_ = nDim_ * (CeilDiv(tq_, TS) + batch_ - 1);
    }

    const int64_t numCores = static_cast<int64_t>(cubeCoreNum_);
    actualCores_ = std::min(numCores, totalWork_);
    if (actualCores_ <= 0) {
        actualCores_ = 1;
    }

    std::vector<int64_t> starts;
    if (haveCostData_ && !costs_.empty()) {
        starts = BuildWorkRanges(costs_, actualCores_);
    } else {
        starts = BuildUniformRanges(totalWork_, actualCores_);
    }
    return starts;
}

ge::graphStatus TilingFlexAttentionFunc::RunTiling()
{
    auto platform_info = platform_ascendc::PlatformAscendC(tiling_context_->GetPlatformInfo());
    cubeCoreNum_ = platform_info.GetCoreNumAic();

    auto ret = ParseShapesAndAttrs();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = ValidateDescriptors();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = ValidateOutputs();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    std::vector<int64_t> workStarts = ComputeWorkRanges();
    if (static_cast<int64_t>(workStarts.size()) > MAX_TILING_CORES + 1) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "flex_attention: work_starts size %zu > capacity %ld.",
                workStarts.size(), MAX_TILING_CORES + 1);
        return ge::GRAPH_FAILED;
    }

    tiling_data_.set_mask_num(maskNum_);
    tiling_data_.set_n_dim(nDim_);
    tiling_data_.set_kv_heads(kvHeads_);
    tiling_data_.set_batch(batch_);
    tiling_data_.set_r_f(rF_);
    tiling_data_.set_w_f(wF_);
    tiling_data_.set_s1(s1_);
    tiling_data_.set_s2(s2_);
    tiling_data_.set_d(headDim_);
    // Match the Python host: math.sqrt(d) in double, then round to the float field.
    tiling_data_.set_scale(static_cast<float>(1.0 / std::sqrt(static_cast<double>(headDim_))));
    // Python pads work_starts to 64 with zeros; only [0, blockDim] is consumed.
    std::vector<int64_t> ws(64, 0);
    for (size_t i = 0; i < workStarts.size() && i < 64; ++i) {
        ws[i] = workStarts[i];
    }
    tiling_data_.set_work_starts(ws.data());

    // A non-zero workspace is required for PyPTO's _gen_infer_cpp to
    // correctly route SetSysWorkspaceForce + tiling data.
    size_t *workspace = tiling_context_->GetWorkspaceSizes(1);
    workspace[0] = 1024;

    // Tiling key encodes (Schedule, Layout, IsDN, HoleChunk) at bits 0..3.
    // Schedule=0 and IsDN=0 are fixed by this delivery-form kernel; Layout
    // follows the host contract and HoleChunk selects the chunked decoder.
    const int64_t tilingKey = (layout_ << 1) | ((maskNum_ > HOLE_CHUNK) ? 8 : 0);
    tiling_context_->SetTilingKey(tilingKey);

    tiling_data_.SaveToBuffer(tiling_context_->GetRawTilingData()->GetData(),
                              tiling_context_->GetRawTilingData()->GetCapacity());
    tiling_context_->GetRawTilingData()->SetDataSize(tiling_data_.GetDataSize());

    tiling_context_->SetBlockDim(static_cast<uint32_t>(actualCores_));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingFlexAttentionFunc tilingObject(context);
    auto ret = tilingObject.RunTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "flex_attention tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FlexAttention)
    .Tiling(TilingFunc)
    .TilingInputsDataDependency({ACTUAL_SEQ_Q, BLOCK_COMPUTE});

} // namespace optiling
