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
 * \file aclnn_bridge_attn_prologue.cpp
 * \brief pybind11 bridge: torch.Tensor -> aclnnAttnPrologue.
 */

#include "aclnn_bridge_common.h"

#include <op_plugin/utils/op_api_common_base.h>

namespace {

using gym_test_bridge::destroy_acl_tensor;
using gym_test_bridge::get_current_npu_stream;
using gym_test_bridge::get_opapi_func;

typedef aclnnStatus (*AttnPrologueGetWorkspaceSizeFunc)(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, double,
    int64_t, int64_t, int64_t, int64_t,
    const aclTensor *, const aclTensor *, const aclTensor *,
    uint64_t *, aclOpExecutor **);
typedef aclnnStatus (*AttnPrologueFunc)(void *, uint64_t, aclOpExecutor *, aclrtStream);

aclnnStatus ExecuteAttn(std::vector<aclTensor *> &acl_inputs, std::vector<aclTensor *> &acl_outputs,
                        const c10::Device &device, double norm_eps, int64_t block_stride,
                        int64_t row_stride, int64_t storage_bytes, int64_t emit_qr_norm) {
    static auto get_ws_size = reinterpret_cast<AttnPrologueGetWorkspaceSizeFunc>(
        get_opapi_func("aclnnAttnPrologueGetWorkspaceSize"));
    static auto aclnn_op = reinterpret_cast<AttnPrologueFunc>(
        get_opapi_func("aclnnAttnPrologue"));

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus status = get_ws_size(
        acl_inputs[0], acl_inputs[1], acl_inputs[2], acl_inputs[3],
        acl_inputs[4], acl_inputs[5], acl_inputs[6], acl_inputs[7],
        acl_inputs[8], acl_inputs[9], acl_inputs[10], acl_inputs[11],
        acl_inputs[12], acl_inputs[13], acl_inputs[14],
        norm_eps, block_stride, row_stride, storage_bytes,
        emit_qr_norm,
        acl_outputs[0], acl_outputs[1], acl_outputs[2],
        &workspace_size, &executor);

    if (status == 0) {
        at::Tensor workspace;
        void *workspace_addr = nullptr;
        if (workspace_size != 0) {
            workspace = at::empty(
                {static_cast<int64_t>(workspace_size)},
                at::TensorOptions().dtype(at::kByte).device(device));
            workspace_addr = workspace.data_ptr();
        }
        aclrtStream stream = get_current_npu_stream();
        status = aclnn_op(workspace_addr, workspace_size, executor, stream);
    }

    for (auto *tensor : acl_inputs) {
        destroy_acl_tensor(tensor);
    }
    for (auto *tensor : acl_outputs) {
        destroy_acl_tensor(tensor);
    }
    return status;
}

} // namespace

PYBIND11_MODULE(aclnn_bridge_attn_prologue, m) {
    m.doc() = "pypto-gym pybind11 test bridge: aclnn C++ -> Python";

    m.def("attn_prologue",
          [](torch::Tensor x, torch::Tensor wqa, torch::Tensor wqb, torch::Tensor wkv,
             torch::Tensor descale_x, torch::Tensor descale_wqa, torch::Tensor descale_wqb,
             torch::Tensor descale_wkv, torch::Tensor norm_weight_qr,
             torch::Tensor norm_weight_kv, torch::Tensor rope_sin, torch::Tensor rope_cos,
             torch::Tensor cache_index, torch::Tensor kv_cache,
             torch::Tensor qr_norm,
             torch::Tensor q, torch::Tensor qr, torch::Tensor descale_qr, double norm_eps,
             int64_t block_stride, int64_t row_stride, int64_t storage_bytes) {
              if (block_stride == 0) {
                  block_stride = kv_cache.stride(0) * kv_cache.element_size();
              }
              if (row_stride == 0) {
                  row_stride = kv_cache.stride(1) * kv_cache.element_size();
              }
              std::vector<aclTensor *> acl_inputs = {
                  ConvertType(x), ConvertType(wqa), ConvertType(wqb), ConvertType(wkv),
                  ConvertType(descale_x), ConvertType(descale_wqa), ConvertType(descale_wqb),
                  ConvertType(descale_wkv), ConvertType(norm_weight_qr),
                  ConvertType(norm_weight_kv), ConvertType(rope_sin), ConvertType(rope_cos),
                  ConvertType(cache_index), ConvertType(kv_cache),
                  ConvertType(qr_norm)};
              std::vector<aclTensor *> acl_outputs = {
                  ConvertType(q), ConvertType(qr), ConvertType(descale_qr)};

              const auto status = ExecuteAttn(acl_inputs, acl_outputs, x.device(), norm_eps,
                                              block_stride, row_stride, storage_bytes, qr_norm.numel() != 0);
              TORCH_CHECK(status == 0, "aclnnAttnPrologue call failed, status=", status);
              return std::make_tuple(q, qr, descale_qr);
          },
          pybind11::arg("x"), pybind11::arg("wqa"), pybind11::arg("wqb"), pybind11::arg("wkv"),
          pybind11::arg("descale_x"), pybind11::arg("descale_wqa"),
          pybind11::arg("descale_wqb"), pybind11::arg("descale_wkv"),
          pybind11::arg("norm_weight_qr"), pybind11::arg("norm_weight_kv"),
          pybind11::arg("rope_sin"), pybind11::arg("rope_cos"),
          pybind11::arg("cache_index"), pybind11::arg("kv_cache"),
          pybind11::arg("qr_norm"),
          pybind11::arg("q"), pybind11::arg("qr"), pybind11::arg("descale_qr"),
          pybind11::arg("norm_eps") = 1e-6,
          pybind11::arg("block_stride") = 0,
          pybind11::arg("row_stride") = 0,
          pybind11::arg("storage_bytes") = 0,
          "aclnnAttnPrologue wrapper (torch.Tensor interface)");
}
