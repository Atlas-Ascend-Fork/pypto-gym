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

/** Test caller of the generated ACLNN interface for IndexerPrologueQw. */
#include "aclnn_bridge_common.h"

#include <acl/acl_base_rt.h>
#include <c10/core/DeviceGuard.h>
#include <vector>

namespace {

using gym_test_bridge::destroy_acl_tensor;
using gym_test_bridge::get_acl_meta_func;
using gym_test_bridge::get_current_npu_stream;
using gym_test_bridge::get_opapi_func;

constexpr int kAclFloat = 0;
constexpr int kAclUint8 = 4;
constexpr int kAclBf16 = 27;
constexpr int kAclFloat8E4m3fn = 36;
constexpr int kAclFloat8E8m0 = 37;

aclDataType QwAclDtype(at::ScalarType dtype)
{
    switch (dtype) {
        case at::ScalarType::Float: return static_cast<aclDataType>(kAclFloat);
        case at::ScalarType::Byte: return static_cast<aclDataType>(kAclUint8);
        case at::ScalarType::BFloat16: return static_cast<aclDataType>(kAclBf16);
        case at::ScalarType::Float8_e4m3fn: return static_cast<aclDataType>(kAclFloat8E4m3fn);
        case at::ScalarType::Float8_e8m0fnu: return static_cast<aclDataType>(kAclFloat8E8m0);
        default:
            TORCH_CHECK(false, "unsupported tensor dtype for aclTensor: ", c10::toString(dtype));
    }
    return ACL_DT_UNDEFINED;
}

aclTensor *MakeAclTensor(const torch::Tensor &tensor, aclFormat format)
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
                         QwAclDtype(tensor.scalar_type()), strides.data(),
                         tensor.storage_offset(), format, storage_dims.data(),
                         static_cast<uint64_t>(storage_dims.size()),
                         const_cast<void *>(tensor.storage().data()));
}

void CheckNz(const torch::Tensor &tensor, const char *name)
{
    auto format = pybind11::module_::import("torch_npu")
                      .attr("get_npu_format")(tensor)
                      .cast<int64_t>();
    TORCH_CHECK(format == 29, name, " must be FRACTAL_NZ; got NPU format ", format);
}

std::vector<torch::Tensor> IndexerPrologueQw(
    const torch::Tensor &x, const torch::Tensor &qr, const torch::Tensor &wqb,
    const torch::Tensor &ww, const torch::Tensor &descale_qr, const torch::Tensor &descale_wqb,
    const torch::Tensor &rope_sin, const torch::Tensor &rope_cos, double softmax_scale)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    CheckNz(wqb, "wqb");
    CheckNz(ww, "ww");
    const c10::DeviceGuard guard(x.device());
    const int64_t t = qr.size(0);
    auto q = torch::empty({t, 32, 64}, x.options().dtype(torch::kUInt8));
    auto descale_q = torch::empty({t, 32, 2, 2}, x.options().dtype(torch::kUInt8));
    auto w = torch::empty({t, 32}, x.options().dtype(torch::kFloat32));

    std::vector<aclTensor *> tensors;
    tensors.push_back(MakeAclTensor(x, ACL_FORMAT_ND));
    tensors.push_back(MakeAclTensor(qr, ACL_FORMAT_ND));
    tensors.push_back(MakeAclTensor(wqb, ACL_FORMAT_FRACTAL_NZ));
    tensors.push_back(MakeAclTensor(ww, ACL_FORMAT_FRACTAL_NZ));
    tensors.push_back(MakeAclTensor(descale_qr, ACL_FORMAT_ND));
    tensors.push_back(MakeAclTensor(descale_wqb, ACL_FORMAT_ND));
    tensors.push_back(MakeAclTensor(rope_sin, ACL_FORMAT_ND));
    tensors.push_back(MakeAclTensor(rope_cos, ACL_FORMAT_ND));
    tensors.push_back(MakeAclTensor(q, ACL_FORMAT_ND));
    tensors.push_back(MakeAclTensor(descale_q, ACL_FORMAT_ND));
    tensors.push_back(MakeAclTensor(w, ACL_FORMAT_ND));

    using GetWs = aclnnStatus (*)(
        const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
        const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
        double, const aclTensor *, const aclTensor *, const aclTensor *,
        uint64_t *, aclOpExecutor **);
    using Execute = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    static auto get_ws = reinterpret_cast<GetWs>(
        get_opapi_func("aclnnIndexerPrologueQwGetWorkspaceSize"));
    static auto execute = reinterpret_cast<Execute>(
        get_opapi_func("aclnnIndexerPrologueQw"));

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus status = get_ws(
        tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5],
        tensors[6], tensors[7], softmax_scale, tensors[8], tensors[9], tensors[10],
        &workspace_size, &executor);
    if (status == 0) {
        torch::Tensor workspace;
        void *workspace_addr = nullptr;
        if (workspace_size != 0) {
            workspace = torch::empty({static_cast<int64_t>(workspace_size)},
                                     x.options().dtype(torch::kUInt8));
            workspace_addr = workspace.data_ptr();
        }
        status = execute(workspace_addr, workspace_size, executor, get_current_npu_stream());
    }

    for (auto *tensor : tensors) {
        destroy_acl_tensor(tensor);
    }
    TORCH_CHECK(status == 0, "aclnnIndexerPrologueQw failed, status=", status);
    return {q, descale_q, w};
}

} // namespace

PYBIND11_MODULE(aclnn_bridge_indexer_prologue_qw, m)
{
    m.doc() = "pypto-gym pybind11 test bridge: aclnn C++ -> Python";
    m.def("indexer_prologue_qw", &IndexerPrologueQw,
          pybind11::arg("x"), pybind11::arg("qr"), pybind11::arg("wqb"), pybind11::arg("ww"),
          pybind11::arg("descale_qr"), pybind11::arg("descale_wqb"),
          pybind11::arg("rope_sin"), pybind11::arg("rope_cos"),
          pybind11::arg("softmax_scale") = 0.08838834764831845);
}
