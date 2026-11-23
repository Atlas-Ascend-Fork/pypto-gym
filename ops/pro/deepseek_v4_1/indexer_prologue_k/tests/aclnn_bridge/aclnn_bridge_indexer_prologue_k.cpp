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

/** Test caller of the generated ACLNN interface for IndexerPrologueK. */
#include "aclnn_bridge_common.h"
#include "../../op_host/indexer_prologue_k_plan.h"

#include <acl/acl_base_rt.h>
#include <acl/acl_rt.h>
#include <c10/core/DeviceGuard.h>
#include <vector>

namespace {

using gym_test_bridge::destroy_acl_tensor;
using gym_test_bridge::get_acl_meta_func;
using gym_test_bridge::get_current_npu_stream;
using gym_test_bridge::get_opapi_func;

constexpr int kAclFloat = 0;
constexpr int kAclUint8 = 4;
constexpr int kAclInt64 = 9;
constexpr int kAclBf16 = 27;

aclDataType KAclDtype(at::ScalarType dtype)
{
    switch (dtype) {
        case at::ScalarType::Float: return static_cast<aclDataType>(kAclFloat);
        case at::ScalarType::Byte: return static_cast<aclDataType>(kAclUint8);
        case at::ScalarType::Long: return static_cast<aclDataType>(kAclInt64);
        case at::ScalarType::BFloat16: return static_cast<aclDataType>(kAclBf16);
        default:
            TORCH_CHECK(false, "unsupported tensor dtype for aclTensor: ", c10::toString(dtype));
    }
    return ACL_DT_UNDEFINED;
}

aclTensor *MakeAclTensor(const torch::Tensor &tensor)
{
    TORCH_CHECK(tensor.defined(), "undefined tensor passed to the aclnn bridge");
    TORCH_CHECK(tensor.itemsize() > 0, "tensor itemsize must be non-zero");
    auto create_tensor = reinterpret_cast<aclTensor *(*)(const int64_t *, uint64_t, aclDataType,
                                                         const int64_t *, int64_t, aclFormat,
                                                         const int64_t *, uint64_t, void *)>(
        get_acl_meta_func("aclCreateTensor"));
    const auto sizes = tensor.sizes();
    const auto strides = tensor.strides();
    std::vector<int64_t> storage_dims = {
        static_cast<int64_t>(tensor.storage().nbytes() / tensor.itemsize())};
    return create_tensor(sizes.data(), static_cast<uint64_t>(sizes.size()),
                         KAclDtype(tensor.scalar_type()), strides.data(),
                         tensor.storage_offset(), ACL_FORMAT_ND, storage_dims.data(),
                         static_cast<uint64_t>(storage_dims.size()),
                         const_cast<void *>(tensor.storage().data()));
}

void RunProjection(const torch::Tensor &latent, const torch::Tensor &wk, torch::Tensor &projected,
                   aclrtStream stream)
{
    using GetWs = aclnnStatus (*)(const aclTensor *, const aclTensor *, const aclTensor *,
                                  uint64_t *, aclOpExecutor **);
    using Execute = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    static auto get_ws = reinterpret_cast<GetWs>(
        get_opapi_func("aclnnIndexerPrologueKProjectionGetWorkspaceSize"));
    static auto execute = reinterpret_cast<Execute>(
        get_opapi_func("aclnnIndexerPrologueKProjection"));
    std::vector<aclTensor *> tensors = {
        MakeAclTensor(latent), MakeAclTensor(wk), MakeAclTensor(projected)};
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus status = get_ws(tensors[0], tensors[1], tensors[2], &workspace_size, &executor);
    if (status == 0) {
        torch::Tensor workspace;
        void *workspace_addr = nullptr;
        if (workspace_size != 0) {
            workspace = torch::empty({static_cast<int64_t>(workspace_size)},
                                     projected.options().dtype(torch::kUInt8));
            workspace_addr = workspace.data_ptr();
        }
        status = execute(workspace_addr, workspace_size, executor, stream);
    }
    for (auto *tensor : tensors) {
        destroy_acl_tensor(tensor);
    }
    TORCH_CHECK(status == 0, "aclnnIndexerPrologueKProjection failed, status=", status,
                ": ", aclGetRecentErrMsg());
}

void RunPostprocess(const torch::Tensor &projected, const torch::Tensor &norm_weight,
                    const torch::Tensor &rope_sin, const torch::Tensor &rope_cos,
                    const torch::Tensor &cache_index, torch::Tensor &k_cache,
                    const torch::Tensor &k_scale_cache, int64_t storage_mode, double norm_eps,
                    int64_t combined_block_size, bool write_scale_cache, aclrtStream stream)
{
    using GetWs = aclnnStatus (*)(
        const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
        const aclTensor *, aclTensor *, aclTensor *,
        int64_t, double, int64_t, bool, uint64_t *, aclOpExecutor **);
    using Execute = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    static auto get_ws = reinterpret_cast<GetWs>(
        get_opapi_func("aclnnIndexerPrologueKPostprocessGetWorkspaceSize"));
    static auto execute = reinterpret_cast<Execute>(
        get_opapi_func("aclnnIndexerPrologueKPostprocess"));
    std::vector<aclTensor *> tensors = {
        MakeAclTensor(projected), MakeAclTensor(norm_weight), MakeAclTensor(rope_sin),
        MakeAclTensor(rope_cos), MakeAclTensor(cache_index), MakeAclTensor(k_cache),
        MakeAclTensor(k_scale_cache)};
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus status = get_ws(
        tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], tensors[6],
        storage_mode, norm_eps, combined_block_size, write_scale_cache,
        &workspace_size, &executor);
    if (status == 0) {
        torch::Tensor workspace;
        void *workspace_addr = nullptr;
        if (workspace_size != 0) {
            workspace = torch::empty({static_cast<int64_t>(workspace_size)},
                                     projected.options().dtype(torch::kUInt8));
            workspace_addr = workspace.data_ptr();
        }
        status = execute(workspace_addr, workspace_size, executor, stream);
    }
    for (auto *tensor : tensors) {
        destroy_acl_tensor(tensor);
    }
    TORCH_CHECK(status == 0, "aclnnIndexerPrologueKPostprocess failed, status=", status,
                ": ", aclGetRecentErrMsg());
}

torch::Tensor IndexerPrologueK(
    const torch::Tensor &latent, const torch::Tensor &wk, const torch::Tensor &norm_weight,
    const torch::Tensor &rope_sin, const torch::Tensor &rope_cos, torch::Tensor k_cache,
    const c10::optional<torch::Tensor> &k_scale_cache, const torch::Tensor &cache_index,
    int64_t storage_mode, double norm_eps, int64_t combined_block_size)
{
    TORCH_CHECK(latent.device().type() == c10::DeviceType::PrivateUse1, "latent must be on NPU");
    const c10::DeviceGuard guard(latent.device());
    auto dummy_scale = torch::zeros({1}, k_cache.options().dtype(torch::kUInt8));
    const torch::Tensor &scale_cache = k_scale_cache.has_value() ? *k_scale_cache : dummy_scale;
    const bool write_scale_cache = storage_mode == 0 && k_scale_cache.has_value();

    if (latent.size(0) > indexer_prologue_k_plan::FUSED_T_MAX) {
        auto stream = get_current_npu_stream();
        auto projected = torch::empty({latent.size(0), wk.size(0)},
                                      latent.options().dtype(torch::kBFloat16));
        RunProjection(latent, wk, projected, stream);
        RunPostprocess(projected, norm_weight, rope_sin, rope_cos, cache_index, k_cache,
                       scale_cache, storage_mode, norm_eps, combined_block_size,
                       write_scale_cache, stream);
        return k_cache;
    }

    std::vector<aclTensor *> tensors;
    tensors.push_back(MakeAclTensor(latent));
    tensors.push_back(MakeAclTensor(wk));
    tensors.push_back(MakeAclTensor(norm_weight));
    tensors.push_back(MakeAclTensor(rope_sin));
    tensors.push_back(MakeAclTensor(rope_cos));
    tensors.push_back(MakeAclTensor(cache_index));
    tensors.push_back(MakeAclTensor(k_cache));
    tensors.push_back(MakeAclTensor(scale_cache));

    using GetWs = aclnnStatus (*)(
        const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
        const aclTensor *, const aclTensor *, aclTensor *, aclTensor *,
        int64_t, double, int64_t, bool, uint64_t *, aclOpExecutor **);
    using Execute = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    static auto get_ws = reinterpret_cast<GetWs>(
        get_opapi_func("aclnnIndexerPrologueKGetWorkspaceSize"));
    static auto execute = reinterpret_cast<Execute>(
        get_opapi_func("aclnnIndexerPrologueK"));

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus status = get_ws(
        tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5],
        tensors[6], tensors[7], storage_mode, norm_eps, combined_block_size,
        write_scale_cache, &workspace_size, &executor);
    if (status == 0) {
        torch::Tensor workspace;
        void *workspace_addr = nullptr;
        if (workspace_size != 0) {
            workspace = torch::empty({static_cast<int64_t>(workspace_size)},
                                     k_cache.options().dtype(torch::kUInt8));
            workspace_addr = workspace.data_ptr();
        }
        status = execute(workspace_addr, workspace_size, executor, get_current_npu_stream());
    }

    for (auto *tensor : tensors) {
        destroy_acl_tensor(tensor);
    }
    TORCH_CHECK(status == 0, "aclnnIndexerPrologueK failed, status=", status, ": ", aclGetRecentErrMsg());
    return k_cache;
}

} // namespace

PYBIND11_MODULE(aclnn_bridge_indexer_prologue_k, m)
{
    m.doc() = "pypto-gym pybind11 test bridge: aclnn C++ -> Python";
    m.def("indexer_prologue_k_wrapper", &IndexerPrologueK,
          pybind11::arg("latent"), pybind11::arg("wk"), pybind11::arg("norm_weight"),
          pybind11::arg("rope_sin"), pybind11::arg("rope_cos"), pybind11::arg("k_cache"),
          pybind11::arg("k_scale_cache") = pybind11::none(),
          pybind11::arg("cache_index"), pybind11::arg("storage_mode"),
          pybind11::arg("norm_eps"), pybind11::arg("combined_block_size") = -1);
}
