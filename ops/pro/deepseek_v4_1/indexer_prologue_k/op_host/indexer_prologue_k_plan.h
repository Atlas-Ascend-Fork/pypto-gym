/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDEXER_PROLOGUE_K_PLAN_H
#define INDEXER_PROLOGUE_K_PLAN_H

#include <algorithm>
#include <cstdint>

namespace indexer_prologue_k_plan {

constexpr int64_t BASE_M = 16;
constexpr int64_t BASE_N_WIDE = 128;
constexpr int64_t BASE_N = 64;
constexpr int64_t BASE_K = 128;
constexpr int64_t NZ_ALIGN = 16;
constexpr int64_t FP4_GROUP_SIZE = 32;
constexpr int64_t E2M1_VALUES_PER_BYTE = 2;
constexpr int64_t CACHE_ABI_DIM2 = 1;
constexpr int64_t MAX_DIM = 2048;
constexpr int64_t T_MAX = 262144;
constexpr int64_t FUSED_T_MAX = 2048;

inline int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return value / divisor + (value % divisor != 0);
}

inline int64_t NextPow2(int64_t value)
{
    int64_t cap = 32;
    while (cap < value) {
        cap <<= 1;
    }
    return std::min<int64_t>(cap, MAX_DIM);
}

inline int64_t DimCapIndex(int64_t cap)
{
    if (cap <= 512) {
        return 0;
    }
    if (cap <= 1024) {
        return 1;
    }
    return 2;
}

inline uint32_t DynUBufSize(int64_t dimCap)
{
    if (dimCap == 1024) {
        return 12832;
    }
    if (dimCap == 2048) {
        return 29888;
    }
    return 20768;
}

struct ProjectionAxes {
    int64_t base_n = BASE_N_WIDE;
    int64_t base_k = BASE_K;
    int64_t n_tiles = 0;
    int64_t k_tiles = 0;
};

inline ProjectionAxes GetProjectionAxes(int64_t dim, int64_t headDim)
{
    ProjectionAxes axes;
    axes.base_n = (dim % BASE_N_WIDE == 0) ? BASE_N_WIDE
        : ((dim % BASE_N == 0) ? BASE_N : NZ_ALIGN);
    axes.base_k = (headDim % BASE_K == 0) ? BASE_K : NZ_ALIGN;
    axes.n_tiles = CeilDiv(dim, axes.base_n);
    axes.k_tiles = CeilDiv(headDim, axes.base_k);
    return axes;
}

inline int64_t NModeOf(int64_t baseN)
{
    return (baseN == BASE_N_WIDE) ? 0 : ((baseN == BASE_N) ? 1 : 2);
}

inline int64_t KModeOf(int64_t baseK)
{
    return (baseK == BASE_K) ? 0 : 1;
}

} // namespace indexer_prologue_k_plan

#endif
