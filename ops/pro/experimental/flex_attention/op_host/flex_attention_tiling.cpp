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
 * C++ port of the (verified) Python host tiling:
 *   - per-work_id cost-balanced core split via binary-search (build_work_ranges)
 *   - cost from popcount of block_compute
 *   - seq_q_list derived from actual_seq_q prefix sums
 *
 * Value dependency:
 *   actual_seq_q is needed for total_work + per-batch block counts; the
 *   schedule cost tensor (block_compute) is needed for balancing. These are
 *   marked ValueDepend(OPTIONAL) in def.cpp and registered via
 *   TilingInputsDataDependency so the framework may populate their Host
 *   addresses. When the data IS available the tiling emits cost-balanced
 *   work_starts; when it is nullptr it falls back to a uniform contiguous
 *   split -- the kernel self-clamps work_end to the real total_work, so the
 *   output stays correct in both paths.
 */

#include "flex_attention_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
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

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b <= 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

static std::vector<int64_t> BuildWorkRanges(const std::vector<int64_t> &costs, int64_t num_cores)
{
    int64_t totalWork = static_cast<int64_t>(costs.size());
    if (num_cores <= 0) {
        num_cores = 1;
    }
    std::vector<int64_t> prefix(totalWork + 1, 0);
    for (int64_t i = 0; i < totalWork; ++i) {
        prefix[i + 1] = prefix[i] + costs[i];
    }

    auto numParts = [&](int64_t cap) -> int64_t {
        int64_t parts = 1, cur = 0;
        for (int64_t i = 0; i < totalWork; ++i) {
            int64_t c = costs[i];
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
    for (auto c : costs) { maxCost = std::max(maxCost, c); }
    int64_t lo = maxCost;
    int64_t hi = (prefix[totalWork] > 0) ? prefix[totalWork] : 1;
    while (lo < hi) {
        int64_t mid = (lo + hi) / 2;
        if (numParts(mid) <= num_cores) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    int64_t cap = lo;

    std::vector<int64_t> starts;
    int64_t core = 0, start = 0, cur = 0;
    for (int64_t i = 0; i < totalWork && core < num_cores; ++i) {
        int64_t c = costs[i];
        if (cur + c > cap && cur > 0) {
            starts.push_back(start);
            ++core;
            start = i;
            cur = 0;
            continue;
        }
        cur += c;
    }
    starts.push_back(start);
    for (int64_t i = core + 1; i < num_cores; ++i) {
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
    std::vector<int64_t> ComputeWorkRanges();

    gert::TilingContext *tiling_context_ = nullptr;
    FlexAttentionTilingData tiling_data_;
    int64_t tq_ = 0;
    int64_t nDim_ = 0;
    int64_t kvHeads_ = 0;
    int64_t headDim_ = 0;
    int64_t batch_ = 0;
    int64_t rF_ = 0;
    int64_t wF_ = 0;
    int64_t maskNum_ = 0;
    uint32_t cubeCoreNum_ = 0;
};

ge::graphStatus TilingFlexAttentionFunc::ParseShapesAndAttrs()
{
    auto queryShape = tiling_context_->GetInputShape(QUERY)->GetStorageShape();
    auto keyShape = tiling_context_->GetInputShape(KEY)->GetStorageShape();
    auto maskrShape = tiling_context_->GetInputShape(MASKR)->GetStorageShape();
    tq_ = queryShape.GetDim(0);
    nDim_ = queryShape.GetDim(1);
    headDim_ = queryShape.GetDim(2);
    kvHeads_ = keyShape.GetDim(1);
    batch_ = maskrShape.GetDim(0);
    int64_t pF = maskrShape.GetDim(1);
    rF_ = pF / TS;

    // block_compute [B, r_f, w_f] is OPTIONAL. It is a ValueDepend (aclIntArray*)
    // input: the framework exposes a FLAT 1D host buffer, so recover wF_ from
    // numel = B * r_f * w_f  =>  wF_ = numel / (batch * rF_).
    auto scShapePtr = tiling_context_->GetOptionalInputShape(BLOCK_COMPUTE);
    if (scShapePtr != nullptr) {
        const auto &scShape = scShapePtr->GetStorageShape();
        const size_t dimNum = scShape.GetDimNum();
        if (dimNum >= 3) {
            wF_ = scShape.GetDim(2);
        } else if (dimNum == 1) {
            int64_t numel = scShape.GetDim(0);
            int64_t denom = batch_ * rF_;
            wF_ = (denom > 0) ? (numel / denom) : 0;
        }
    }

    // mask_num from mask_start shape (descriptor condition: mask_start [B, mask_num, Pf]).
    // mask_num is not an operator attribute: it is derived here and carried to
    // the kernel via the tiling data. mask_len must be present with the same
    // shape whenever mask_start is present.
    auto mask_startShapePtr = tiling_context_->GetOptionalInputShape(MASK_START);
    if (mask_startShapePtr != nullptr) {
        const auto &msShape = mask_startShapePtr->GetStorageShape();
        if (msShape.GetDimNum() < 3) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "flex_attention: mask_start must be rank-3 [B, mask_num, Pf], got rank %zu.",
                    static_cast<size_t>(msShape.GetDimNum()));
            return ge::GRAPH_FAILED;
        }
        maskNum_ = msShape.GetDim(1);
        auto mask_lenShapePtr = tiling_context_->GetOptionalInputShape(MASK_LEN);
        if (mask_lenShapePtr == nullptr) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "flex_attention: mask_len is required when mask_start is present.");
            return ge::GRAPH_FAILED;
        }
        const auto &mlShape = mask_lenShapePtr->GetStorageShape();
        if (mlShape.GetDimNum() != msShape.GetDimNum()) {
            OP_LOGE(tiling_context_->GetNodeName(),
                    "flex_attention: mask_len rank %zu != mask_start rank %zu.",
                    static_cast<size_t>(mlShape.GetDimNum()),
                    static_cast<size_t>(msShape.GetDimNum()));
            return ge::GRAPH_FAILED;
        }
        for (size_t axis = 0; axis < msShape.GetDimNum(); ++axis) {
            if (mlShape.GetDim(axis) != msShape.GetDim(axis)) {
                OP_LOGE(tiling_context_->GetNodeName(),
                        "flex_attention: mask_len dim %zu (%ld) != mask_start dim %zu (%ld).",
                        axis, mlShape.GetDim(axis), axis, msShape.GetDim(axis));
                return ge::GRAPH_FAILED;
            }
        }
    }

    if (headDim_ != TS) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "flex_attention: head_dim must be %ld, got %ld.", TS, headDim_);
        return ge::GRAPH_FAILED;
    }
    if (nDim_ <= 0 || batch_ <= 0) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "flex_attention: invalid n_dim(%ld) or batch(%ld).", nDim_, batch_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

std::vector<int64_t> TilingFlexAttentionFunc::ComputeWorkRanges()
{
    const auto *seqQTensor = tiling_context_->GetInputTensor(ACTUAL_SEQ_Q);
    const int64_t *seqQData = nullptr;
    if (seqQTensor != nullptr) {
        seqQData = seqQTensor->GetData<int64_t>();
    }

    std::vector<int64_t> seqQList(batch_);
    if (seqQData != nullptr) {
        int64_t prev = 0;
        for (int64_t i = 0; i < batch_; ++i) {
            seqQList[i] = seqQData[i] - prev;
            prev = seqQData[i];
        }
    } else {
        int64_t per = tq_ / batch_;
        int64_t rem = tq_ % batch_;
        for (int64_t i = 0; i < batch_; ++i) {
            seqQList[i] = per + (i < rem ? 1 : 0);
        }
    }

    int64_t totalWork = 0;
    for (int64_t sq : seqQList) {
        totalWork += CeilDiv(sq, TS);
    }
    totalWork *= nDim_;

    int64_t numCores = static_cast<int64_t>(cubeCoreNum_);
    int64_t actualCores = std::min(numCores, totalWork);
    if (actualCores <= 0) {
        actualCores = 1;
    }

    // ---- per-work_id cost from block_compute popcount ----
    std::vector<int64_t> costs;
    costs.reserve(static_cast<size_t>(totalWork));
    bool haveCostData = false;

    const auto *scTensor = tiling_context_->GetOptionalInputTensor(BLOCK_COMPUTE);
    const int64_t *sc = (scTensor != nullptr) ? scTensor->GetData<int64_t>() : nullptr;
    if (sc != nullptr) {
        haveCostData = true;
        for (int64_t bi = 0; bi < batch_; ++bi) {
            int64_t blkM = CeilDiv(seqQList[bi], TS);
            for (int64_t n = 0; n < nDim_; ++n) {
                for (int64_t m = 0; m < blkM; ++m) {
                    int64_t cnt = 0;
                    for (int64_t w = 0; w < wF_; ++w) {
                        int64_t word = sc[(bi * rF_ + m) * wF_ + w];
                        cnt += __builtin_popcountll(static_cast<uint64_t>(word));
                    }
                    costs.push_back(cnt);
                }
            }
        }
    }

    std::vector<int64_t> starts;
    if (haveCostData && !costs.empty()) {
        starts = BuildWorkRanges(costs, actualCores);
    } else {
        starts = BuildUniformRanges(totalWork, actualCores);
    }

    while (static_cast<int64_t>(starts.size()) < numCores + 1) {
        starts.push_back(totalWork);
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
    // varlen-only entry (mirrors the JIT host entry: s1 = s2 = 0, layout_id = 0;
    // tiling.s1/s2 are consumed only by the bshd Layout=1 path).
    tiling_data_.set_s1(0);
    tiling_data_.set_s2(0);
    tiling_data_.set_d(headDim_);
    tiling_data_.set_scale(1.0f / std::sqrt(static_cast<float>(headDim_)));
    std::vector<int64_t> ws(64, 0);
    for (size_t i = 0; i < workStarts.size() && i < 64; ++i) {
        ws[i] = workStarts[i];
    }
    tiling_data_.set_work_starts(ws.data());

    // A non-zero workspace is required for PyPTO's _gen_infer_cpp to
    // correctly route SetSysWorkspaceForce + tiling data.
    size_t *workspace = tiling_context_->GetWorkspaceSizes(1);
    workspace[0] = 1024;

    tiling_context_->SetTilingKey(0);

    tiling_data_.SaveToBuffer(tiling_context_->GetRawTilingData()->GetData(),
                              tiling_context_->GetRawTilingData()->GetCapacity());
    tiling_context_->GetRawTilingData()->SetDataSize(tiling_data_.GetDataSize());

    int64_t totalWork = (workStarts.size() >= 2) ? workStarts.back() : 0;
    int64_t blockDim = std::min(static_cast<int64_t>(cubeCoreNum_), std::max<int64_t>(1, totalWork));
    tiling_context_->SetBlockDim(static_cast<uint32_t>(blockDim));
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