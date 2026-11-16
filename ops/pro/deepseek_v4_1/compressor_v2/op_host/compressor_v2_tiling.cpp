/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "compressor_v2_tiling.h"
#include "compressor_v2_plan.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"

#include <cstdio>

namespace compressor_v2_host {

constexpr std::size_t kInputCount = 10;
constexpr std::size_t kOutputCount = 4;

struct StageAttributes {
    int64_t stage = 0;
    int64_t ratio = 4;
    bool has_used = false;
    bool has_start = false;
};

struct ContextSpecs {
    std::array<TensorSpec, kInputCount> inputs;
    std::array<TensorSpec, kOutputCount> outputs;
};


inline DType ConvertDType(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16: return DType::Float16;
        case ge::DT_BF16: return DType::BFloat16;
        case ge::DT_FLOAT: return DType::Float32;
        case ge::DT_INT32: return DType::Int32;
        default: return DType::Other;
    }
}

inline Status ReadAttributes(gert::TilingContext *context, StageAttributes &attributes)
{
    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return {"missing stage attributes"};
    }
    const int64_t *stage = attrs->GetInt(0);
    const int64_t *ratio = attrs->GetInt(1);
    const bool *has_used = attrs->GetBool(2);
    const bool *has_start = attrs->GetBool(3);
    if (stage == nullptr || ratio == nullptr || has_used == nullptr || has_start == nullptr) {
        return {"stage, cmp_ratio, has_used and has_start are required"};
    }
    attributes = {*stage, *ratio, *has_used, *has_start};
    return {};
}

inline Status MakeContiguousStrides(TensorSpec &spec)
{
    spec.strides.assign(spec.shape.size(), 1);
    if (std::find(spec.shape.begin(), spec.shape.end(), 0) != spec.shape.end()) {
        return {}; // No element is addressable; IsContiguous also treats empty tensors as contiguous.
    }
    Wide stride = 1;
    for (std::size_t i = spec.shape.size(); i-- > 0;) {
        if (stride > kMaxElements || spec.shape[i] < 0) {
            return {"invalid contiguous tensor dimensions"};
        }
        spec.strides[i] = static_cast<int64_t>(stride);
        stride *= static_cast<Wide>(spec.shape[i]);
    }
    return {};
}

inline Status ReadContextTensor(gert::TilingContext *context, std::size_t index,
    bool output, TensorSpec &spec)
{
    const auto *shape = output ? context->GetOutputShape(index) : context->GetInputShape(index);
    const auto *desc = output ? context->GetOutputDesc(index) : context->GetInputDesc(index);
    if (shape == nullptr || desc == nullptr) {
        return {"all ten input and four output descriptors are required"};
    }
    const auto &origin = shape->GetOriginShape();
    spec.dtype = ConvertDType(desc->GetDataType());
    spec.shape.clear();
    for (std::size_t i = 0; i < origin.GetDimNum(); ++i) {
        if (origin.GetDim(i) < 0) {
            return {"runtime tensor dimensions must be nonnegative"};
        }
        spec.shape.push_back(origin.GetDim(i));
    }
    bool view = output ? context->OutputIsView(index) : context->InputIsView(index);
    if (!view) {
        return MakeContiguousStrides(spec);
    }
    const auto *strides = output ? context->GetOutputStride(index) : context->GetInputStride(index);
    int64_t offset = output ? context->GetOutputOffset(index) : context->GetInputOffset(index);
    if (strides == nullptr || strides->GetDimNum() != spec.shape.size() || offset < 0) {
        std::fprintf(stderr, "compressor_v2: %s tensor %zu has shape rank %zu, stride rank %zu, offset %lld\n",
            output ? "output" : "input", index, spec.shape.size(),
            strides == nullptr ? 0 : strides->GetDimNum(), static_cast<long long>(offset));
        return {"view tensor requires matching element strides and a nonnegative offset"};
    }
    spec.strides.clear();
    for (std::size_t i = 0; i < strides->GetDimNum(); ++i) {
        spec.strides.push_back(strides->GetStride(i));
    }
    return ValidDimensions(spec) ? Status{} : Status{"invalid tensor strides"};
}

inline Status ReadContextSpecs(gert::TilingContext *context, ContextSpecs &specs)
{
    for (std::size_t i = 0; i < kInputCount; ++i) {
        Status status = ReadContextTensor(context, i, false, specs.inputs[i]);
        if (!status) {
            return status;
        }
    }
    for (std::size_t i = 0; i < kOutputCount; ++i) {
        Status status = ReadContextTensor(context, i, true, specs.outputs[i]);
        if (!status) {
            return status;
        }
    }
    return {};
}

inline Inputs RestorePublicInputs(const ContextSpecs &specs, const StageAttributes &attributes)
{
    const auto &in = specs.inputs;
    return {in[0], in[1], in[2], in[3], attributes.ratio, &in[4],
        in[0].shape.size() == 2 ? &in[5] : nullptr,
        attributes.has_used ? &in[6] : nullptr, attributes.has_start ? &in[7] : nullptr};
}

inline Status ValidatePlaceholderInputs(const ContextSpecs &specs)
{
    for (std::size_t i = 4; i < kInputCount; ++i) {
        DType expected = i < 8 ? DType::Int32 : DType::Float32;
        if (specs.inputs[i].dtype != expected || !IsContiguous(specs.inputs[i])) {
            return {"metadata must be contiguous INT32 and intermediate inputs contiguous FP32"};
        }
    }
    return {};
}

inline bool SameTensorView(const TensorSpec &left, const TensorSpec &right)
{
    return left.dtype == right.dtype && left.shape == right.shape && left.strides == right.strides;
}

inline bool SameOffset(gert::TilingContext *context, std::size_t input, std::size_t output)
{
    int64_t in_offset = context->InputIsView(input) ? context->GetInputOffset(input) : 0;
    int64_t out_offset = context->OutputIsView(output) ? context->GetOutputOffset(output) : 0;
    return in_offset == out_offset;
}

inline Status ElementCount(const TensorSpec &spec, uint64_t &count)
{
    if (std::find(spec.shape.begin(), spec.shape.end(), 0) != spec.shape.end()) {
        count = 0;
        return {};
    }
    count = 1;
    for (int64_t dimension : spec.shape) {
        if (dimension < 0 || count > kMaxElements / static_cast<uint64_t>(dimension)) {
            return {"tensor element count exceeds int64 range"};
        }
        count *= static_cast<uint64_t>(dimension);
    }
    return {};
}

inline Status ValidateIntermediate(gert::TilingContext *context, const ContextSpecs &specs,
    std::size_t input, std::size_t output, uint64_t needed)
{
    const auto &in = specs.inputs[input];
    const auto &out = specs.outputs[output];
    if (!SameTensorView(in, out) || !SameOffset(context, input, output) ||
        in.dtype != DType::Float32 || !IsContiguous(in)) {
        return {"intermediate input/output views must match and be contiguous FP32"};
    }
    uint64_t capacity = 0;
    Status status = ElementCount(in, capacity);
    if (!status) {
        return status;
    }
    return capacity >= needed ? Status{} : Status{"intermediate tensor capacity is too small"};
}

inline Status ValidateContextOutputs(gert::TilingContext *context, const ContextSpecs &specs, const Plan &plan)
{
    const auto &out = specs.outputs[0];
    std::vector<int64_t> expected(plan.output_shape.begin(), plan.output_shape.begin() + plan.output_rank);
    if (out.dtype != specs.inputs[0].dtype || out.shape != expected || !IsContiguous(out)) {
        return {"out must have the planned shape, activation dtype and contiguous strides"};
    }
    if (!SameTensorView(specs.inputs[3], specs.outputs[1]) || !SameOffset(context, 3, 1)) {
        return {"cache input/output shapes, dtypes, strides and offsets must match"};
    }
    uint64_t projected = plan.route_count == 2 ? plan.projected_elements : 1;
    uint64_t history = plan.route_count == 2 ? plan.history_elements : 1;
    Status status = ValidateIntermediate(context, specs, 8, 2, projected);
    return status ? ValidateIntermediate(context, specs, 9, 3, history) : status;
}

inline Status SelectStage(const Plan &plan, int64_t stage, bool vector_entry, StageSelection &selection)
{
    if ((vector_entry && stage != 1) || (!vector_entry && stage != 0 && stage != 2)) {
        return {"stage does not belong to this entry's core type"};
    }
    for (std::size_t i = 0; i < plan.route_count; ++i) {
        if (plan.keys[i].Stage != stage) {
            continue;
        }
        if (plan.workers[i] <= 0 || static_cast<uint64_t>(plan.workers[i]) > UINT32_MAX) {
            return {"selected blockDim is not a positive uint32 value"};
        }
        selection.key = plan.keys[i];
        selection.workers = static_cast<uint32_t>(plan.workers[i]);
        return {};
    }
    return {"stage is not selected by the original wrapper for these inputs"};
}

Status BuildContextSelection(gert::TilingContext *context, bool vector_entry, StageSelection &selection)
{
    if (context == nullptr) {
        return {"tiling context is null"};
    }
    selection = {};
    StageAttributes attributes;
    ContextSpecs specs;
    Status status = ReadAttributes(context, attributes);
    if (status) {
        status = ReadContextSpecs(context, specs);
    }
    if (status) {
        status = ValidatePlaceholderInputs(specs);
    }
    if (!status) {
        return status;
    }
    auto *platform_info = context->GetPlatformInfo();
    if (platform_info == nullptr) {
        return {"platform information is required"};
    }
    platform_ascendc::PlatformAscendC platform(platform_info);
    status = BuildPlan(RestorePublicInputs(specs, attributes), platform.GetCoreNumAic(),
        platform.GetCoreNumAiv(), selection.plan);
    if (status) {
        status = ValidateContextOutputs(context, specs, selection.plan);
    }
    return status ? SelectStage(selection.plan, attributes.stage, vector_entry, selection) : status;
}

inline Status WriteGeneratedTiling(gert::TilingContext *context, const KernelShape &shape)
{
    static_assert(sizeof(::CompressorV2Tiling) == 12 * sizeof(int64_t), "unexpected generated tiling size");
    auto *tiling = context->GetTilingData<::CompressorV2Tiling>();
    if (tiling == nullptr) {
        return {"generated tiling buffer is null or too small"};
    }
    tiling->tokens = shape.tokens;
    tiling->hidden = shape.hidden;
    tiling->dim = shape.dim;
    tiling->batches = shape.batches;
    tiling->sequence = shape.sequence;
    tiling->block_size = shape.block_size;
    tiling->cache_stride = shape.cache_stride;
    tiling->block_count = shape.block_count;
    tiling->out_rows = shape.out_rows;
    tiling->packed_layout = shape.packed_layout;
    tiling->has_used = shape.has_used;
    tiling->has_start = shape.has_start;
    return {};
}

ge::graphStatus ReportTilingFailure(gert::TilingContext *context, const char *message)
{
    const char *name = context == nullptr ? "null" : context->GetNodeName();
    std::fprintf(stderr, "compressor_v2 [%s]: %s\n", name == nullptr ? "unnamed" : name, message);
    return ge::GRAPH_FAILED;
}

ge::graphStatus CompleteTiling(gert::TilingContext *context, const StageSelection &selection,
    uint64_t key, std::size_t system_workspace)
{
    Status status = WriteGeneratedTiling(context, selection.plan.shape);
    if (!status) {
        return ReportTilingFailure(context, status.error);
    }
    std::size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ReportTilingFailure(context, "workspace descriptor is null");
    }
    workspace[0] = system_workspace;
    if (context->SetBlockDim(selection.workers) != ge::GRAPH_SUCCESS ||
        context->SetTilingKey(key) != ge::GRAPH_SUCCESS) {
        return ReportTilingFailure(context, "could not set blockDim or tiling key");
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace compressor_v2_host

namespace optiling {

static ge::graphStatus TilingPyptoCompressorV2Stage(gert::TilingContext *context)
{
    compressor_v2_host::StageSelection selection;
    compressor_v2_host::Status status = compressor_v2_host::BuildContextSelection(context, false, selection);
    if (!status) {
        return compressor_v2_host::ReportTilingFailure(context, status.error);
    }
    const auto &key = selection.key;
    uint64_t encoded_key = GET_TPL_TILING_KEY(
        static_cast<uint64_t>(key.Stage), static_cast<uint64_t>(key.Rows),
        static_cast<uint64_t>(key.Ratio), static_cast<uint64_t>(key.Fuse),
        static_cast<uint64_t>(key.Cols), static_cast<uint64_t>(key.Direct),
        static_cast<uint64_t>(key.Parts), static_cast<uint64_t>(key.Dim));
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    return compressor_v2_host::CompleteTiling(context, selection, encoded_key, platform.GetLibApiWorkSpaceSize());
}

static ge::graphStatus TilingParsePyptoCompressorV2Stage([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PyptoCompressorV2Stage)
    .Tiling(TilingPyptoCompressorV2Stage)
    .TilingParse<PyptoCompressorV2StageCompileInfo>(TilingParsePyptoCompressorV2Stage);

} // namespace optiling
