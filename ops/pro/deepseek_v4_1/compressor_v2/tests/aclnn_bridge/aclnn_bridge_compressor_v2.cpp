/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/** Test caller of the two generated ACLNN interfaces; no custom op_api library. */
#include "aclnn_bridge_common.h"
#include "aclnn_pypto_compressor_v2_stage.h"
#include "aclnn_pypto_compressor_v2_vector.h"
#include "../../op_host/compressor_v2_plan.h"

#include <acl/acl_rt.h>
#include <c10/core/DeviceGuard.h>
#include <limits>
#include <memory>
#include <type_traits>

namespace {
namespace host = compressor_v2_host;
using TensorHandle = std::unique_ptr<aclTensor, decltype(&gym_test_bridge::destroy_acl_tensor)>;

host::TensorSpec MakeSpec(const torch::Tensor &tensor)
{
    host::DType dtype = host::DType::Other;
    switch (tensor.scalar_type()) {
        case at::kHalf: dtype = host::DType::Float16; break;
        case at::kBFloat16: dtype = host::DType::BFloat16; break;
        case at::kFloat: dtype = host::DType::Float32; break;
        case at::kInt: dtype = host::DType::Int32; break;
        default: break;
    }
    return {dtype, tensor.sizes().vec(), tensor.strides().vec()};
}

TensorHandle MakeTensor(const torch::Tensor &tensor)
{
    auto *descriptor = gym_test_bridge::make_acl_tensor(tensor);
    TORCH_CHECK(descriptor != nullptr, "aclCreateTensor failed");
    return TensorHandle(descriptor, gym_test_bridge::destroy_acl_tensor);
}

void RunStage(int64_t stage, int64_t ratio, bool hasUsed, bool hasStart,
              const std::array<const aclTensor *, 14> &tensors,
              const torch::TensorOptions &options, aclrtStream stream)
{
    using Query = decltype(&aclnnPyptoCompressorV2StageGetWorkspaceSize);
    using Execute = decltype(&aclnnPyptoCompressorV2Stage);
    static_assert(std::is_same_v<Query, decltype(&aclnnPyptoCompressorV2VectorGetWorkspaceSize)>);
    static_assert(std::is_same_v<Execute, decltype(&aclnnPyptoCompressorV2Vector)>);
    static const std::array<Query, 2> queries = {
        reinterpret_cast<Query>(gym_test_bridge::get_opapi_func("aclnnPyptoCompressorV2StageGetWorkspaceSize")),
        reinterpret_cast<Query>(gym_test_bridge::get_opapi_func("aclnnPyptoCompressorV2VectorGetWorkspaceSize"))};
    static const std::array<Execute, 2> executes = {
        reinterpret_cast<Execute>(gym_test_bridge::get_opapi_func("aclnnPyptoCompressorV2Stage")),
        reinterpret_cast<Execute>(gym_test_bridge::get_opapi_func("aclnnPyptoCompressorV2Vector"))};
    const size_t entry = stage == 1 ? 1 : 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = queries[entry](
        tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], tensors[6], tensors[7],
        tensors[8], tensors[9], stage, ratio, hasUsed, hasStart,
        tensors[10], tensors[11], tensors[12], tensors[13], &workspaceSize, &executor);
    TORCH_CHECK(status == 0, "generated ACLNN GetWorkspaceSize failed at stage ", stage, ", status=", status);
    TORCH_CHECK(executor != nullptr, "generated ACLNN returned a null executor");
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "workspace size exceeds the torch allocation limit");
    auto workspace = torch::empty({static_cast<int64_t>(workspaceSize)}, options.dtype(at::kByte));
    status = executes[entry](workspaceSize == 0 ? nullptr : workspace.data_ptr(), workspaceSize, executor, stream);
    TORCH_CHECK(status == 0, "generated ACLNN execution failed at stage ", stage, ", status=", status);
}

torch::Tensor Compressor(torch::Tensor x, torch::Tensor wkv, torch::Tensor wgate,
                         torch::Tensor cache, int64_t ratio,
                         c10::optional<torch::Tensor> table, c10::optional<torch::Tensor> cu,
                         c10::optional<torch::Tensor> used, c10::optional<torch::Tensor> starts)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    const c10::DeviceGuard guard(x.device());
    TORCH_CHECK(table.has_value() && table->defined(), "state_block_table is required");
    for (const auto &tensor : {x, wkv, wgate, cache, *table}) {
        TORCH_CHECK(tensor.device() == x.device(), "all tensors must be on the same NPU");
    }
    for (const auto &tensor : {cu, used, starts}) {
        TORCH_CHECK(!tensor.has_value() || tensor->device() == x.device(), "metadata must be on the same NPU");
    }
    const auto xSpec = MakeSpec(x), kvSpec = MakeSpec(wkv), gateSpec = MakeSpec(wgate), cacheSpec = MakeSpec(cache);
    const auto tableSpec = MakeSpec(*table);
    const auto cuSpec = cu.has_value() ? MakeSpec(*cu) : host::TensorSpec{};
    const auto usedSpec = used.has_value() ? MakeSpec(*used) : host::TensorSpec{};
    const auto startSpec = starts.has_value() ? MakeSpec(*starts) : host::TensorSpec{};
    const host::Inputs inputs{xSpec, kvSpec, gateSpec, cacheSpec, ratio, &tableSpec,
        cu.has_value() ? &cuSpec : nullptr, used.has_value() ? &usedSpec : nullptr,
        starts.has_value() ? &startSpec : nullptr};
    int64_t cubeCores = 0, vectorCores = 0;
    TORCH_CHECK(aclrtGetDeviceInfo(x.device().index(), ACL_DEV_ATTR_CUBE_CORE_NUM, &cubeCores) == ACL_SUCCESS,
                "aclrtGetDeviceInfo failed for Cube core count");
    TORCH_CHECK(aclrtGetDeviceInfo(x.device().index(), ACL_DEV_ATTR_VECTOR_CORE_NUM, &vectorCores) == ACL_SUCCESS,
                "aclrtGetDeviceInfo failed for Vector core count");
    host::Plan plan;
    const auto status = host::BuildPlan(inputs, cubeCores, vectorCores, plan);
    TORCH_CHECK(static_cast<bool>(status), "invalid CompressorV2 inputs: ", status.error);
    auto out = torch::empty(std::vector<int64_t>(plan.output_shape.begin(),
        plan.output_shape.begin() + plan.output_rank), x.options());
    if (plan.route_count == 0) {
        return out;
    }
    const int64_t projectedSize = plan.route_count == 2 ? static_cast<int64_t>(plan.projected_elements) : 1;
    const int64_t historySize = plan.route_count == 2 ? static_cast<int64_t>(plan.history_elements) : 1;
    auto projected = torch::empty({projectedSize}, x.options().dtype(at::kFloat));
    auto history = torch::empty({historySize}, x.options().dtype(at::kFloat));
    auto aclX = MakeTensor(x), aclKv = MakeTensor(wkv), aclGate = MakeTensor(wgate), aclCache = MakeTensor(cache);
    auto aclTable = MakeTensor(*table), aclOut = MakeTensor(out);
    auto aclCu = MakeTensor(cu.value_or(*table)), aclUsed = MakeTensor(used.value_or(*table));
    auto aclStarts = MakeTensor(starts.value_or(*table));
    auto aclProjected = MakeTensor(projected), aclHistory = MakeTensor(history);
    const std::array<const aclTensor *, 14> tensors = {aclX.get(), aclKv.get(), aclGate.get(), aclCache.get(),
        aclTable.get(), aclCu.get(), aclUsed.get(), aclStarts.get(), aclProjected.get(), aclHistory.get(),
        aclOut.get(), aclCache.get(), aclProjected.get(), aclHistory.get()};
    const auto stream = gym_test_bridge::get_current_npu_stream();
    for (size_t i = 0; i < plan.route_count; ++i) {
        RunStage(plan.keys[i].Stage, ratio, used.has_value(), starts.has_value(), tensors, x.options(), stream);
    }
    return out;
}
} // namespace

PYBIND11_MODULE(aclnn_bridge_compressor_v2, module)
{
    module.doc() = "CompressorV2 test caller using generated ACLNN interfaces";
    module.def("compressor", &Compressor,
        pybind11::arg("x"), pybind11::arg("wkv"), pybind11::arg("wgate"), pybind11::arg("state_cache"),
        pybind11::arg("cmp_ratio") = 4, pybind11::kw_only(), pybind11::arg("state_block_table") = pybind11::none(),
        pybind11::arg("cu_seqlens") = pybind11::none(), pybind11::arg("seqused") = pybind11::none(),
        pybind11::arg("start_pos") = pybind11::none());
}
