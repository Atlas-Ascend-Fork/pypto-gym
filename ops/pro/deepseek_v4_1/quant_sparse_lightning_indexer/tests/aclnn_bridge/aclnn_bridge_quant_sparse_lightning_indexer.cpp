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
 * \file aclnn_bridge_quant_sparse_lightning_indexer.cpp
 * \brief pybind11 桥: 将 aclnn C++ 接口转为 Python 可调用的 torch.Tensor 接口.
 *
 * 运行时通过 dlopen/dlsym 从 libcust_opapi.so(ASCEND_CUSTOM_OPP_PATH / LD_LIBRARY_PATH)查找 aclnn 符号.
 * 公共层(符号加载/tensor 构造/销毁/stream 获取)见 tests/ops/pro/aclnn_bridge_common/aclnn_bridge_common.h.
 *
 * 签名与 op_kernel 的宿主入口逐参数一致(15 输入 + 2 输出 + 9 属性), 因此同一组用例、
 * 同一组 golden 可在 --mode jit 与 --mode aclnn 下复用(见 tests/ 的 _load_operators 适配层).
 *
 * E8M0 scale 在算子里注册为 UINT8(原始指数字节), 而调用方可能持有 float8_e8m0fnu
 * 视图, 故这里对 descale_q/descale_k 做一次 view 归一.
 */

#include "aclnn_bridge_common.h"

namespace {

using gym_test_bridge::destroy_acl_tensor;
using gym_test_bridge::get_current_npu_stream;
using gym_test_bridge::get_opapi_func;
using gym_test_bridge::make_acl_tensor;

// E8M0 scale 张量按原始字节传给算子(UINT8 注册); float8_e8m0fnu 视图等价.
torch::Tensor as_scale_bytes(const torch::Tensor &tensor)
{
    if (tensor.scalar_type() == at::ScalarType::Float8_e8m0fnu) {
        return tensor.view(at::ScalarType::Byte);
    }
    return tensor;
}

aclTensor *make_scale_tensor(const c10::optional<torch::Tensor> &opt)
{
    if (!opt.has_value() || !opt->defined()) {
        return nullptr;
    }
    return make_acl_tensor(as_scale_bytes(*opt));
}

typedef aclnnStatus (*QsliGetWorkspaceSizeFunc)(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    char *, char *, bool,
    const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
typedef aclnnStatus (*QsliRunFunc)(void *, uint64_t, aclOpExecutor *, aclrtStream);

} // namespace

PYBIND11_MODULE(aclnn_bridge_quant_sparse_lightning_indexer, m) {
    m.doc() = "pypto-gym pybind11 test bridge: aclnn C++ -> Python";

    m.def("quant_sparse_lightning_indexer",
          [](torch::Tensor q, torch::Tensor k, torch::Tensor w, torch::Tensor descale_q,
             c10::optional<torch::Tensor> descale_k,
             torch::Tensor candidate_block_indices, torch::Tensor candidate_block_length,
             c10::optional<torch::Tensor> cu_seqlens_q, c10::optional<torch::Tensor> cu_seqlens_k,
             c10::optional<torch::Tensor> seqused_q, c10::optional<torch::Tensor> seqused_k,
             c10::optional<torch::Tensor> cmp_residual_k, c10::optional<torch::Tensor> block_table,
             c10::optional<torch::Tensor> output_idx_offset, torch::Tensor metadata,
             torch::Tensor indices, torch::Tensor values,
             int64_t topk, int64_t candidate_block_size, int64_t quant_mode,
             int64_t max_seqlen_q, int64_t mask_mode, int64_t cmp_ratio,
             const std::string &layout_q, const std::string &layout_k, bool return_value) {
              static auto get_ws_size = reinterpret_cast<QsliGetWorkspaceSizeFunc>(
                  get_opapi_func("aclnnQuantSparseLightningIndexerGetWorkspaceSize"));
              static auto aclnn_op = reinterpret_cast<QsliRunFunc>(
                  get_opapi_func("aclnnQuantSparseLightningIndexer"));

              std::vector<aclTensor *> acl_tensors = {
                  make_acl_tensor(q), make_acl_tensor(k), make_acl_tensor(w),
                  make_scale_tensor(descale_q), make_scale_tensor(descale_k),
                  make_acl_tensor(candidate_block_indices),
                  make_acl_tensor(candidate_block_length),
                  make_acl_tensor(cu_seqlens_q), make_acl_tensor(cu_seqlens_k),
                  make_acl_tensor(seqused_q), make_acl_tensor(seqused_k),
                  make_acl_tensor(cmp_residual_k), make_acl_tensor(block_table),
                  make_acl_tensor(output_idx_offset), make_acl_tensor(metadata),
                  make_acl_tensor(indices), make_acl_tensor(values)};

              std::string layout_q_value = layout_q;
              std::string layout_k_value = layout_k;
              uint64_t workspace_size = 0;
              aclOpExecutor *executor = nullptr;
              aclnnStatus status = get_ws_size(
                  acl_tensors[0], acl_tensors[1], acl_tensors[2], acl_tensors[3],
                  acl_tensors[4], acl_tensors[5], acl_tensors[6], acl_tensors[7],
                  acl_tensors[8], acl_tensors[9], acl_tensors[10], acl_tensors[11],
                  acl_tensors[12], acl_tensors[13], acl_tensors[14],
                  topk, candidate_block_size, quant_mode, max_seqlen_q, mask_mode, cmp_ratio,
                  layout_q_value.data(), layout_k_value.data(), return_value,
                  acl_tensors[15], acl_tensors[16], &workspace_size, &executor);

              if (status == 0) {
                  at::Tensor workspace;
                  void *workspace_addr = nullptr;
                  if (workspace_size != 0) {
                      workspace = at::empty(
                          {static_cast<int64_t>(workspace_size)},
                          at::TensorOptions().dtype(at::kByte).device(q.device()));
                      workspace_addr = workspace.data_ptr();
                  }
                  aclrtStream stream = get_current_npu_stream();
                  status = aclnn_op(workspace_addr, workspace_size, executor, stream);
              }

              for (auto *tensor : acl_tensors) {
                  destroy_acl_tensor(tensor);
              }
              TORCH_CHECK(status == 0,
                          "aclnnQuantSparseLightningIndexer call failed, status=", status);
              // return_value=False still requires a full-size score buffer (the
              // WriteValues=0 specialization simply does not store into it); the caller
              // sees a zero-element view so the public contract stays shape-stable.
              torch::Tensor values_out =
                  return_value ? values : values.reshape({-1}).slice(0, 0, 0);
              return pybind11::make_tuple(indices, values_out);
          },
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("w"),
          pybind11::arg("descale_q"), pybind11::arg("descale_k"),
          pybind11::arg("candidate_block_indices"), pybind11::arg("candidate_block_length"),
          pybind11::arg("cu_seqlens_q"), pybind11::arg("cu_seqlens_k"),
          pybind11::arg("seqused_q"), pybind11::arg("seqused_k"),
          pybind11::arg("cmp_residual_k"), pybind11::arg("block_table"),
          pybind11::arg("output_idx_offset"), pybind11::arg("metadata"),
          pybind11::arg("indices"), pybind11::arg("values"),
          pybind11::arg("topk"), pybind11::arg("candidate_block_size") = 8,
          pybind11::arg("quant_mode") = 1, pybind11::arg("max_seqlen_q") = -1,
          pybind11::arg("mask_mode") = 0, pybind11::arg("cmp_ratio") = 1,
          pybind11::arg("layout_q") = "TND", pybind11::arg("layout_k") = "TND",
          pybind11::arg("return_value") = false,
          "aclnnQuantSparseLightningIndexer wrapper (torch.Tensor interface)");
}
