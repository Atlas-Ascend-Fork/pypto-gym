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

/*!
 * \file aclnn_bridge_mixed_quant_sparse_flash_mla_metadata.cpp
 * \brief pybind11 bridge for the AICPU metadata planner op (aclnn C API -> Python).
 */

#include "aclnn_bridge_common.h"

namespace {

using gym_test_bridge::destroy_acl_tensor;
using gym_test_bridge::get_current_npu_stream;
using gym_test_bridge::get_opapi_func;
using gym_test_bridge::make_acl_tensor;

typedef aclnnStatus (*GetWorkspaceSizeFunc)(
    const aclTensor *, const aclTensor *, const aclTensor *,
    int64_t, int64_t, int64_t, int64_t,
    const aclTensor *, uint64_t *, aclOpExecutor **);
typedef aclnnStatus (*RunFunc)(void *, uint64_t, aclOpExecutor *, aclrtStream);

} // namespace

PYBIND11_MODULE(aclnn_bridge_mixed_quant_sparse_flash_mla_metadata, m) {
    m.doc() = "pypto-gym pybind11 test bridge: AICPU metadata planner (aclnn)";

    m.def("mixed_quant_sparse_flash_mla_metadata",
          [](torch::Tensor cu_seqlens_q,
             torch::Tensor ori_topk_length,
             c10::optional<torch::Tensor> cmp_topk_length,
             int64_t blocks,
             int64_t has_cmp,
             int64_t k_ori,
             int64_t k_cmp) {
              static auto get_ws_size = reinterpret_cast<GetWorkspaceSizeFunc>(
                  get_opapi_func("aclnnMixedQuantSparseFlashMlaMetadataGetWorkspaceSize"));
              static auto aclnn_op = reinterpret_cast<RunFunc>(
                  get_opapi_func("aclnnMixedQuantSparseFlashMlaMetadata"));

              torch::Tensor md = torch::empty(
                  {1024}, cu_seqlens_q.options().dtype(at::kInt));
              aclTensor *cu_t = make_acl_tensor(cu_seqlens_q);
              aclTensor *ori_t = make_acl_tensor(ori_topk_length);
              // A null input tensor aborts the AICPU task; an absent cmp side passes a
              // one-element placeholder and gates on has_cmp instead.
              torch::Tensor cmp_holder;
              if (!(cmp_topk_length.has_value() && cmp_topk_length->defined())) {
                  cmp_holder = torch::zeros({1}, cu_seqlens_q.options().dtype(at::kInt));
              }
              aclTensor *cmp_t = make_acl_tensor(
                  cmp_topk_length.has_value() && cmp_topk_length->defined() ? *cmp_topk_length : cmp_holder);
              aclTensor *md_t = make_acl_tensor(md);

              uint64_t workspace_size = 0;
              aclOpExecutor *executor = nullptr;
              aclnnStatus status = get_ws_size(
                  cu_t, ori_t, cmp_t, blocks, has_cmp, k_ori, k_cmp, md_t,
                  &workspace_size, &executor);
              if (status == 0) {
                  at::Tensor workspace;
                  void *workspace_addr = nullptr;
                  if (workspace_size != 0) {
                      workspace = at::empty(
                          {static_cast<int64_t>(workspace_size)},
                          at::TensorOptions().dtype(at::kByte).device(cu_seqlens_q.device()));
                      workspace_addr = workspace.data_ptr();
                  }
                  status = aclnn_op(workspace_addr, workspace_size, executor,
                                    get_current_npu_stream());
              }

              destroy_acl_tensor(cu_t);
              destroy_acl_tensor(ori_t);
              destroy_acl_tensor(cmp_t);
              destroy_acl_tensor(md_t);
              TORCH_CHECK(status == 0, "aclnn metadata planner failed, status=", status);
              return md;
          },
          pybind11::arg("cu_seqlens_q"),
          pybind11::arg("ori_topk_length"),
          pybind11::arg("cmp_topk_length") = pybind11::none(),
          pybind11::arg("blocks") = 28,
          pybind11::arg("has_cmp") = 1,
          pybind11::arg("k_ori") = 0,
          pybind11::arg("k_cmp") = 0);
}
