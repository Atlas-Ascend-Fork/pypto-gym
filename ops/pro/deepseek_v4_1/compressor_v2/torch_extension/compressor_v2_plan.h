/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Host shape and launch selection shared by ACLNN API and host tiling.
#ifndef COMPRESSOR_V2_PLAN_H
#define COMPRESSOR_V2_PLAN_H

#include "exe_graph/runtime/tiling_context.h"
#include "register/tilingdata_base.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

namespace optiling {

// Shared registration metadata; both entries use the generated 96-byte CompressorV2Tiling layout.
BEGIN_TILING_DATA_DEF(PyptoCompressorV2TilingData)
TILING_DATA_FIELD_DEF(int64_t, tokens)
TILING_DATA_FIELD_DEF(int64_t, hidden)
TILING_DATA_FIELD_DEF(int64_t, dim)
TILING_DATA_FIELD_DEF(int64_t, batches)
TILING_DATA_FIELD_DEF(int64_t, sequence)
TILING_DATA_FIELD_DEF(int64_t, block_size)
TILING_DATA_FIELD_DEF(int64_t, cache_stride)
TILING_DATA_FIELD_DEF(int64_t, block_count)
TILING_DATA_FIELD_DEF(int64_t, out_rows)
TILING_DATA_FIELD_DEF(int64_t, packed_layout)
TILING_DATA_FIELD_DEF(int64_t, has_used)
TILING_DATA_FIELD_DEF(int64_t, has_start)
END_TILING_DATA_DEF

} // namespace optiling

namespace compressor_v2_host {

// GCC/Clang host extension: selector cost arithmetic must retain Python integer semantics.
using Wide = unsigned __int128;
constexpr uint64_t kMaxElements = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());

struct Status {
    const char *error = nullptr;
    explicit operator bool() const { return error == nullptr; }
};

enum class DType { Float16, BFloat16, Float32, Int32, Other };

struct TensorSpec {
    DType dtype = DType::Other;
    // Strides use elements; storage offset is not included.
    std::vector<int64_t> shape, strides;
};

struct Inputs {
    const TensorSpec &x;
    const TensorSpec &wkv;
    const TensorSpec &wgate;
    const TensorSpec &cache;
    int64_t ratio = 4;
    const TensorSpec *table = nullptr, *cu = nullptr;
    const TensorSpec *used = nullptr, *starts = nullptr;
};

struct KernelShape {
    int64_t tokens = 0, hidden = 0, dim = 0;
    int64_t batches = 0, sequence = 0;
    int64_t block_size = 0, cache_stride = 0, block_count = 0;
    int64_t out_rows = 0, packed_layout = 0, has_used = 0, has_start = 0;
};

// Keep this declaration order. Feed the fields into the actual generated template-key macro.
struct Key {
    int64_t Stage = 0, Half = 0;
    int64_t Rows = 0, Ratio = 0, Fuse = 0, Cols = 0;
    int64_t Direct = 0, Parts = 1, Dim = 0, Mode = 0;
};

struct Plan {
    KernelShape shape;
    std::array<int64_t, 3> output_shape{};
    std::size_t output_rank = 0;
    uint64_t output_elements = 0;
    std::array<Key, 2> keys{};
    std::array<int64_t, 2> workers{};
    std::size_t route_count = 0;
    uint64_t projected_elements = 0, history_elements = 0;
    uint64_t projected_bytes = 0, history_bytes = 0;
    uint64_t raw_temporary_bytes = 0;
};

inline Wide CeilDiv(Wide numerator, Wide denominator)
{
    if (denominator == 0) {
        throw std::invalid_argument("CeilDiv requires a positive denominator");
    }
    return numerator / denominator + (numerator % denominator != 0);
}

inline bool CheckedProduct(std::initializer_list<uint64_t> factors, uint64_t limit, uint64_t &result)
{
    for (uint64_t factor : factors) {
        if (factor == 0) {
            result = 0;
            return true;
        }
    }
    result = 1;
    for (uint64_t factor : factors) {
        if (result > limit / factor) {
            return false;
        }
        result *= factor;
    }
    return true;
}

inline bool ValidDimensions(const TensorSpec &tensor)
{
    if (tensor.shape.size() != tensor.strides.size()) {
        return false;
    }
    for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
        if (tensor.shape[i] < 0 || tensor.strides[i] < 0) {
            return false;
        }
    }
    return true;
}

inline bool IsContiguous(const TensorSpec &tensor)
{
    if (!ValidDimensions(tensor)) {
        return false;
    }
    if (std::find(tensor.shape.begin(), tensor.shape.end(), 0) != tensor.shape.end()) {
        return true;
    }
    Wide expected = 1;
    for (std::size_t i = tensor.shape.size(); i-- > 0;) {
        if (tensor.shape[i] != 1 && static_cast<Wide>(tensor.strides[i]) != expected) {
            return false;
        }
        expected *= static_cast<Wide>(tensor.shape[i]);
    }
    return true;
}

inline bool ValidRatio(int64_t ratio)
{
    constexpr std::array<int64_t, 7> ratios{2, 4, 8, 16, 32, 64, 128};
    return std::find(ratios.begin(), ratios.end(), ratio) != ratios.end();
}

inline Status ValidateActivation(const TensorSpec &x)
{
    if (!ValidDimensions(x) || (x.shape.size() != 2 && x.shape.size() != 3) ||
        (x.dtype != DType::Float16 && x.dtype != DType::BFloat16)) {
        return {"x must be a rank-2/rank-3 FP16 or BF16 tensor"};
    }
    int64_t hidden = x.shape.back();
    if (hidden < 1024 || hidden > 10240 || hidden % 512 != 0) {
        return {"H must be in [1024,10240] and divisible by 512"};
    }
    return {};
}

inline Status ValidateWeights(const Inputs &inputs)
{
    const auto &shape = inputs.wkv.shape;
    if (!ValidDimensions(inputs.wkv) || !ValidDimensions(inputs.wgate) || shape.size() != 2 ||
        (shape[0] != 128 && shape[0] != 512) || shape[1] != inputs.x.shape.back() ||
        inputs.wgate.shape != shape) {
        return {"weights must have shape [D,H], D=128 or 512"};
    }
    if (inputs.wkv.dtype != inputs.x.dtype || inputs.wgate.dtype != inputs.x.dtype) {
        return {"activation and weight dtypes must match"};
    }
    return ValidRatio(inputs.ratio) ? Status{} : Status{"cmp_ratio must be one of 2,4,8,16,32,64,128"};
}

inline Status ValidateCache(const TensorSpec &cache, int64_t dim)
{
    if (!ValidDimensions(cache) || cache.shape.size() != 3 || cache.dtype != DType::Float32 ||
        cache.shape[2] != 2 * dim) {
        return {"state_cache must be FP32 [block_count,block_size,2*D]"};
    }
    if (cache.shape[0] < 1 || cache.shape[1] < 1 || cache.shape[1] > 1024) {
        return {"state_cache needs positive block_count and block_size in [1,1024]"};
    }
    if (cache.strides[2] != 1 || cache.strides[1] != 2 * dim ||
        cache.strides[0] < cache.shape[1] * 2 * dim) {
        return {"only the cache block stride may be noncontiguous"};
    }
    Wide span = static_cast<Wide>(cache.shape[0] - 1) * cache.strides[0] + cache.shape[1] * 2 * dim;
    return span <= kMaxElements ? Status{} : Status{"cache element address span exceeds int64 range"};
}

inline Status BatchCount(const Inputs &inputs, int64_t &batches)
{
    if (inputs.x.shape.size() == 2) {
        if (inputs.cu == nullptr || !ValidDimensions(*inputs.cu) || inputs.cu->shape.size() != 1 ||
            inputs.cu->shape[0] < 1) {
            return {"TH layout requires cu_seqlens[B+1]"};
        }
        batches = inputs.cu->shape[0] - 1;
        return {};
    }
    if (inputs.cu != nullptr) {
        return {"BSH layout requires cu_seqlens=None"};
    }
    batches = inputs.x.shape[0];
    return {};
}

inline bool ValidMetadata(const TensorSpec *tensor, int64_t size)
{
    return tensor == nullptr || (ValidDimensions(*tensor) && tensor->dtype == DType::Int32 &&
        tensor->shape.size() == 1 && tensor->shape[0] == size);
}

inline Status ValidateMetadata(const Inputs &inputs, int64_t batches)
{
    if (inputs.table == nullptr || !ValidMetadata(inputs.table, batches)) {
        return {"state_block_table is required and must be int32 [B]"};
    }
    // cu is present only for TH, where batches was computed as cu.shape[0]-1.
    if (inputs.cu != nullptr && !ValidMetadata(inputs.cu, batches + 1)) {
        return {"cu_seqlens must be int32 [B+1]"};
    }
    if (!ValidMetadata(inputs.used, batches) || !ValidMetadata(inputs.starts, batches)) {
        return {"seqused and start_pos, when present, must be int32 [B]"};
    }
    return {};
}

inline Status ValidateContiguity(const Inputs &inputs)
{
    const std::array<const TensorSpec *, 7> tensors{
        &inputs.x, &inputs.wkv, &inputs.wgate, inputs.table, inputs.cu, inputs.used, inputs.starts};
    for (const TensorSpec *tensor : tensors) {
        if (tensor != nullptr && !IsContiguous(*tensor)) {
            return {"inputs must be contiguous except the cache block stride"};
        }
    }
    return {};
}

inline Status ValidateInputs(const Inputs &inputs, int64_t &batches)
{
    Status status = ValidateActivation(inputs.x);
    if (status) {
        status = ValidateWeights(inputs);
    }
    if (status) {
        status = ValidateCache(inputs.cache, inputs.wkv.shape[0]);
    }
    if (status) {
        status = BatchCount(inputs, batches);
    }
    if (status) {
        status = ValidateMetadata(inputs, batches);
    }
    return status ? ValidateContiguity(inputs) : status;
}

inline int64_t RowCapacity(int64_t tokens, int64_t dim, int64_t cols, int64_t cores)
{
    if (cols == 0) {
        throw std::invalid_argument("RowCapacity requires a nonzero column capacity");
    }
    if (cols < 0 || dim < cols || cores <= 0) {
        throw std::invalid_argument("RowCapacity requires positive column tiles and core count");
    }
    int64_t parallel_rows = std::max<int64_t>(1, cores / (2 * (dim / cols)));
    Wide ideal = CeilDiv(tokens, parallel_rows);
    for (int64_t rows : {16, 32, 64, 96, 128}) {
        if (static_cast<Wide>(rows) >= ideal) {
            return rows;
        }
    }
    return 128;
}

inline int64_t ColumnCapacity(int64_t tokens)
{
    return tokens <= 128 ? 64 : 128;
}

inline bool SingleFusedPanel(const Key &route)
{
    return route.Parts == 1 && route.Rows == 128 && route.Fuse == 1 && route.Cols == 128;
}

inline void RefinePanelKey(const KernelShape &shape, int64_t cores, Key &route)
{
    if (SingleFusedPanel(route) && shape.hidden % 512 == 0) {
        Wide grid = CeilDiv(shape.tokens, 128) * (shape.dim / 128);
        Wide current_panel_work = CeilDiv(grid, cores) * (shape.hidden / 256);
        Wide split_panel_work = CeilDiv(grid * 2, cores) * (shape.hidden / 512);
        if (split_panel_work * 8 <= current_panel_work * 7) {
            route.Parts = 2;
        }
    }
    if (SingleFusedPanel(route) && shape.tokens % 256 == 0) {
        Wide narrow_grid = static_cast<Wide>(shape.tokens / 128) * (shape.dim / 128);
        Wide wide_grid = static_cast<Wide>(shape.tokens / 256) * (shape.dim / 128);
        if (2 * CeilDiv(wide_grid, cores) <= CeilDiv(narrow_grid, cores)) {
            route.Rows = 256;
        }
    }
}

inline Key CubeKey(const KernelShape &shape, int64_t half, int64_t ratio, int64_t cores)
{
    int64_t cols = ColumnCapacity(shape.tokens);
    if (cols == 0) {
        throw std::invalid_argument("CubeKey requires a positive column capacity");
    }
    int64_t rows = RowCapacity(shape.tokens, shape.dim, cols, cores);
    int64_t fuse = CeilDiv(shape.tokens, rows) * (shape.dim / cols) >= static_cast<Wide>(cores);
    Key route{0, half, rows, ratio, fuse, cols, 0, 1, shape.dim};
    Wide split_grid = std::max<Wide>(1, CeilDiv(shape.tokens, 256) * (shape.dim / 64));
    int64_t parts = static_cast<int64_t>(std::min<Wide>(4, static_cast<Wide>(cores) / split_grid));
    if (parts >= 2 && shape.hidden >= 1024 * parts) {
        Wide current_grid = (2 - fuse) * CeilDiv(shape.tokens, rows) * (shape.dim / cols);
        Wide current_cost = CeilDiv(current_grid, cores) *
            (std::min(shape.tokens, rows) + (1 + fuse) * cols) * shape.hidden;
        Wide split_cost = CeilDiv(split_grid * parts, cores) * (std::min<int64_t>(shape.tokens, 256) + 128);
        split_cost *= CeilDiv(shape.hidden, parts);
        if (split_cost * 5 < current_cost * 4) {
            route.Rows = 256;
            route.Cols = 64;
            route.Fuse = 1;
            route.Parts = parts;
        }
    }
    if (shape.tokens >= 1 && shape.tokens <= 128 && route.Parts > 1) {
        for (int64_t selected_rows : {16, 32, 64, 96, 128}) {
            if (selected_rows >= shape.tokens) {
                route.Rows = selected_rows;
                break;
            }
        }
    }
    if (shape.tokens > 128 && shape.tokens <= 256 && route.Parts > 1) {
        Wide grid = CeilDiv(shape.tokens, 128) * (shape.dim / route.Cols) * route.Parts;
        if (grid <= static_cast<Wide>(cores)) {
            route.Rows = 128;
        }
    }
    RefinePanelKey(shape, cores, route);
    return route;
}

struct Geometry {
    int64_t rows = 0, cols = 0;
};

inline Wide OnchipCost(Geometry geometry, const KernelShape &shape, int64_t cores)
{
    Wide grid = static_cast<Wide>(shape.tokens / geometry.rows) * (shape.dim / geometry.cols);
    return CeilDiv(grid, cores) *
        (geometry.rows + 2 * geometry.cols + geometry.rows * geometry.cols / 64);
}

inline Geometry OnchipGeometry(const KernelShape &shape, int64_t ratio, int64_t cores)
{
    if (shape.tokens < 32) {
        return {};
    }
    const std::array<Geometry, 4> choices{{{256, 128}, {128, 128}, {64, 64}, {32, 64}}};
    Geometry selected;
    Wide best_cost = 0;
    for (Geometry choice : choices) {
        if (choice.rows < 2 * ratio || shape.tokens % choice.rows != 0 ||
            (choice.rows == 256 && shape.block_size % (choice.rows / 2) != 0)) {
            continue;
        }
        Wide cost = OnchipCost(choice, shape, cores);
        if (selected.rows == 0 || cost < best_cost) { // Strict comparison preserves Python min's first tie.
            selected = choice;
            best_cost = cost;
        }
    }
    return selected;
}

inline int64_t VectorMode(const KernelShape &shape, int64_t ratio)
{
    int64_t mode = shape.packed_layout | (shape.has_used << 1) | (shape.has_start << 2);
    if (shape.packed_layout == 0 && shape.has_used == 0 && shape.sequence == ratio) {
        mode |= 8;
    }
    if ((mode == 4 || mode == 5) && shape.tokens <= (ratio == 64 ? 64 : 128)) {
        mode |= 16;
    }
    if (shape.batches == 1 && (mode == 5 || mode == 12 || mode == 20 || mode == 21)) {
        mode |= 32;
    }
    return mode;
}

inline void LaunchKeys(Plan &plan, int64_t half, int64_t ratio, int64_t cube_cores)
{
    const auto &shape = plan.shape;
    if (shape.batches == 1 && shape.has_used == 0 && shape.has_start == 0) {
        Geometry geometry = OnchipGeometry(shape, ratio, cube_cores);
        if (geometry.rows != 0) {
            plan.keys[0] = {2, half, geometry.rows, ratio, 1, geometry.cols,
                shape.block_size % (geometry.rows / 2) == 0, 1, shape.dim};
            plan.route_count = 1;
            return;
        }
    }
    plan.keys[0] = CubeKey(shape, half, ratio, cube_cores);
    plan.keys[1] = plan.keys[0];
    plan.keys[1].Stage = 1;
    plan.keys[1].Rows = 16;
    plan.keys[1].Fuse = 0;
    plan.keys[1].Cols = 64;
    plan.keys[1].Mode = VectorMode(shape, ratio);
    if (shape.batches == 1 && shape.tokens == 1 &&
        (plan.keys[1].Mode == 52 || plan.keys[1].Mode == 53)) {
        plan.keys[1].Rows = 1;
    }
    plan.route_count = 2;
}

inline Status SetOutputShape(int64_t ratio, Plan &plan)
{
    if (ratio == 0 || !ValidRatio(ratio)) {
        return {"cmp_ratio must be one of 2,4,8,16,32,64,128"};
    }
    auto &shape = plan.shape;
    Wide out_rows = shape.packed_layout != 0 ? std::min<Wide>(shape.tokens,
        shape.tokens / ratio + static_cast<Wide>(shape.batches)) :
        static_cast<Wide>(shape.batches) * CeilDiv(shape.sequence, ratio);
    if (out_rows > kMaxElements || !CheckedProduct({static_cast<uint64_t>(out_rows),
        static_cast<uint64_t>(shape.dim)}, kMaxElements, plan.output_elements)) {
        return {"output element count exceeds int64 range"};
    }
    shape.out_rows = static_cast<int64_t>(out_rows);
    plan.output_rank = shape.packed_layout != 0 ? 2 : 3;
    plan.output_shape = shape.packed_layout != 0 ?
        std::array<int64_t, 3>{shape.out_rows, shape.dim, 0} :
        std::array<int64_t, 3>{shape.batches, static_cast<int64_t>(CeilDiv(shape.sequence, ratio)), shape.dim};
    return {};
}

inline Status MakeShape(const Inputs &inputs, int64_t batches, Plan &plan)
{
    int64_t hidden = inputs.x.shape.back();
    uint64_t tokens = 0;
    uint64_t x_elements = 0;
    bool packed = inputs.x.shape.size() == 2;
    uint64_t sequence = packed ? 0 : static_cast<uint64_t>(inputs.x.shape[1]);
    if (!CheckedProduct({static_cast<uint64_t>(inputs.x.shape[0]), packed ? 1 : sequence},
        kMaxElements, tokens) || !CheckedProduct({tokens, static_cast<uint64_t>(hidden)},
        kMaxElements, x_elements)) {
        return {"x element count exceeds int64 range"};
    }
    int64_t dim = inputs.wkv.shape[0];
    plan.shape = {static_cast<int64_t>(tokens), hidden, dim, batches, static_cast<int64_t>(sequence),
        inputs.cache.shape[1], inputs.cache.strides[0], inputs.cache.shape[0], 0,
        packed, inputs.used != nullptr, inputs.starts != nullptr};
    return SetOutputShape(inputs.ratio, plan);
}

inline Status MakeTemporaries(int64_t ratio, Plan &plan)
{
    if (plan.route_count != 2) {
        return {}; // Stage 2 uses cache for ignored projected/history slots, just as the wrapper does.
    }
    const auto &shape = plan.shape;
    if (!CheckedProduct({static_cast<uint64_t>(plan.keys[0].Parts), 2,
        static_cast<uint64_t>(shape.tokens), static_cast<uint64_t>(shape.dim)},
        kMaxElements, plan.projected_elements) || !CheckedProduct({static_cast<uint64_t>(shape.batches),
        static_cast<uint64_t>(ratio), 2, static_cast<uint64_t>(shape.dim)},
        kMaxElements, plan.history_elements)) {
        return {"temporary tensor element count exceeds int64 range"};
    }
    constexpr uint64_t limit = std::numeric_limits<uint64_t>::max();
    if (!CheckedProduct({plan.projected_elements, 4}, limit, plan.projected_bytes) ||
        !CheckedProduct({plan.history_elements, 4}, limit, plan.history_bytes) ||
        plan.projected_bytes > limit - plan.history_bytes) {
        return {"raw temporary byte count exceeds uint64 range"};
    }
    plan.raw_temporary_bytes = plan.projected_bytes + plan.history_bytes;
    return {};
}

inline Status MakeRoutes(int64_t half, int64_t ratio, int64_t cube_cores, int64_t vector_cores, Plan &plan)
{
    if (plan.shape.tokens == 0) {
        return {}; // No route, platform requirement, or temporary allocation on the empty-input path.
    }
    if (cube_cores <= 0 || vector_cores < 0) {
        return {"invalid platform core counts"};
    }
    LaunchKeys(plan, half, ratio, cube_cores);
    const auto &shape = plan.shape;
    const Wide ncols = static_cast<Wide>(shape.dim / 64);
    Wide vector_workers = vector_cores;
    if ((shape.packed_layout == 0 && shape.sequence <= 128) ||
        (plan.keys[plan.route_count - 1].Stage == 1 && plan.keys[plan.route_count - 1].Mode == 21)) {
        vector_workers = std::min<Wide>((vector_cores / ncols) * ncols,
            static_cast<Wide>(shape.batches) * ncols);
    }
    for (std::size_t i = 0; i < plan.route_count; ++i) {
        Wide workers = cube_cores;
        const auto &key = plan.keys[i];
        if (key.Stage == 0) {
            Wide projection = CeilDiv(shape.tokens, key.Rows) * (shape.dim / key.Cols);
            projection *= (2 - key.Fuse) * key.Parts;
            Wide snapshot = static_cast<Wide>(shape.batches) * (2 * shape.dim / 64);
            workers = std::min(workers, std::max(projection, CeilDiv(snapshot, 2)));
        } else if (key.Stage == 1) {
            workers = vector_workers;
        }
        plan.workers[i] = static_cast<uint32_t>(workers);
    }
    return MakeTemporaries(ratio, plan);
}

inline Status BuildPlan(const Inputs &inputs, int64_t cube_cores, int64_t vector_cores, Plan &plan)
{
    plan = {};
    int64_t batches = 0;
    Status status = ValidateInputs(inputs, batches);
    if (status) {
        status = MakeShape(inputs, batches, plan);
    }
    return status ? MakeRoutes(inputs.x.dtype == DType::Float16, inputs.ratio, cube_cores, vector_cores, plan) : status;
}

// Shared host callbacks; the generated tiling representation is private to the
// implementation and is not exposed to the ACLNN planner.
struct StageSelection {
    Plan plan;
    Key key;
    uint32_t workers = 0;
};

Status BuildContextSelection(gert::TilingContext *context, bool vector_entry, StageSelection &selection);
ge::graphStatus ReportTilingFailure(gert::TilingContext *context, const char *message);
ge::graphStatus CompleteTiling(gert::TilingContext *context, const StageSelection &selection,
    uint64_t key, std::size_t system_workspace);

} // namespace compressor_v2_host
#endif // COMPRESSOR_V2_PLAN_H
