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
 * \file aclnn_bridge_mixed_quant_sparse_flash_mla.cpp
 * \brief
 *
 * pybind11 桥: 将 aclnn C++ 接口转为 Python 可调用的 torch.Tensor 接口.
 *
 * 运行时通过 dlopen/dlsym 从 libcust_opapi.so(ASCEND_CUSTOM_OPP_PATH / LD_LIBRARY_PATH)查找 aclnn 符号,
 * 运行时按动态库符号查找链加载 aclnn 入口(dlopen/dlsym).
 *
 * 公共层(torch<->aclnn 转换/符号加载)见 tests/ops/pro/aclnn_bridge_common/aclnn_bridge_common.h.
 *
 * 编译: ENABLE_TESTS=ON 时由 CMake 链编译为 aclnn_bridge_mixed_quant_sparse_flash_mla.so
 *       → build/<soc>/tests/
 * 消费: pytest --mode aclnn → 根 conftest.py 注入 sys.path →
 *       op_caller.get_op_caller("mixed_quant_sparse_flash_mla")
 *
 * jit/aclnn 入口签名一致约定: `metadata` 作为普通输入张量传给 aclnn 接口;
 * `block_dim` / `fd_enable` 由框架 host tiling 决定, 参数被接受但忽略,
 * 其余参数语义与 jit 入口一一对应.
 *
 * 注意: ValueDepend(int32) 输入(cu_seqlens_q / *_topk_length)在自动生成的
 * aclnn 接口中的形态以生成头 aclnn_mixed_quant_sparse_flash_mla.h 为准;
 * 若生成器为它们产出了 aclIntArray 形参, 需按头文件调整本桥的转换.
 */

#include "aclnn_bridge_common.h"

namespace {

using gym_test_bridge::destroy_acl_int_array;
using gym_test_bridge::destroy_acl_tensor;
using gym_test_bridge::get_current_npu_stream;
using gym_test_bridge::get_opapi_func;
using gym_test_bridge::make_acl_int_array;
using gym_test_bridge::make_acl_tensor;

// 形参顺序 = 自动生成头 aclnn_mixed_quant_sparse_flash_mla.h
// (ValueDepend int32 输入经 aclIntArray int64 host 值承载).
typedef aclnnStatus (*aclnnMixedQuantSparseFlashMlaGetWorkspaceSizeFunc)(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclIntArray *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *,
    const aclIntArray *, const aclIntArray *,
    const aclTensor *,
    char *, char *, double,
    const aclTensor *, const aclTensor *,
    uint64_t *, aclOpExecutor **);
typedef aclnnStatus (*aclnnMixedQuantSparseFlashMlaFunc)(
    void *, uint64_t, aclOpExecutor *, aclrtStream);

} // namespace

PYBIND11_MODULE(aclnn_bridge_mixed_quant_sparse_flash_mla, m) {
    m.doc() = "pypto-gym pybind11 test bridge: aclnn C++ -> Python";

    m.def("mixed_quant_sparse_flash_mla",
          [](torch::Tensor q,
             torch::Tensor ori_kv,
             torch::Tensor ori_sparse_indices,
             torch::Tensor ori_block_table,
             torch::Tensor cu_seqlens_q,
             torch::Tensor sinks,
             c10::optional<torch::Tensor> metadata,
             int64_t quant_mode,
             c10::optional<double> softmax_scale,
             c10::optional<torch::Tensor> cmp_kv,
             c10::optional<torch::Tensor> cmp_sparse_indices,
             c10::optional<torch::Tensor> cmp_block_table,
             c10::optional<torch::Tensor> ori_topk_length,
             c10::optional<torch::Tensor> cmp_topk_length,
             c10::optional<torch::Tensor> seqused_q,
             c10::optional<torch::Tensor> seqused_ori_kv,
             c10::optional<torch::Tensor> seqused_cmp_kv,
             const std::string &layout_q,
             const std::string &layout_kv,
             bool return_softmax_lse,
             c10::optional<torch::Tensor> out,
             c10::optional<torch::Tensor> lse,
             c10::optional<int64_t> block_dim,
             c10::optional<int64_t> fd_enable) {
              (void)seqused_q;
              (void)seqused_ori_kv;
              (void)seqused_cmp_kv;
              (void)block_dim;
              (void)fd_enable;
              TORCH_CHECK(quant_mode == 1, "aclnn: only quant_mode=1 is supported");
              TORCH_CHECK(metadata.has_value() && metadata->defined(),
                          "aclnn: metadata is required");
              TORCH_CHECK(layout_q == "TND" && layout_kv == "PA_BBND",
                          "aclnn: only layout_q='TND', layout_kv='PA_BBND' are supported");

              static auto get_ws_size = reinterpret_cast<aclnnMixedQuantSparseFlashMlaGetWorkspaceSizeFunc>(
                  get_opapi_func("aclnnMixedQuantSparseFlashMlaGetWorkspaceSize"));
              static auto aclnn_op = reinterpret_cast<aclnnMixedQuantSparseFlashMlaFunc>(
                  get_opapi_func("aclnnMixedQuantSparseFlashMla"));

              torch::Tensor out_t = out.has_value() && out->defined()
                  ? *out
                  : torch::empty(q.sizes(), q.options().dtype(at::kBFloat16));
              torch::Tensor lse_t = lse.has_value() && lse->defined()
                  ? *lse
                  : torch::empty({1, q.size(0), q.size(1)}, q.options().dtype(at::kFloat));

              double scale_v = softmax_scale.has_value() ? *softmax_scale : 0.0;
              std::string layout_q_value = layout_q;
              std::string layout_kv_value = layout_kv;

              // ValueDepend inputs become aclIntArray (int64 host values); the
              // kernel reads them as int64 device tensors after framework sync.
              std::vector<torch::Tensor> host_arrays;
              auto to_int_array = [&host_arrays](const c10::optional<torch::Tensor> &tensor) -> aclIntArray * {
                  if (!tensor.has_value() || !tensor->defined()) {
                      return nullptr;
                  }
                  host_arrays.push_back(tensor->to(at::kCPU, at::kLong).contiguous());
                  return make_acl_int_array(host_arrays.back(), host_arrays);
              };
              aclIntArray *ori_len_arr = to_int_array(ori_topk_length);
              aclIntArray *cmp_len_arr = to_int_array(cmp_topk_length);
              aclIntArray *cu_arr = to_int_array(cu_seqlens_q);

              std::vector<aclTensor *> acl_inputs = {
                  make_acl_tensor(q),
                  make_acl_tensor(ori_kv),
                  make_acl_tensor(ori_sparse_indices),
                  make_acl_tensor(ori_block_table),
                  make_acl_tensor(sinks),
                  make_acl_tensor(cmp_kv),
                  make_acl_tensor(cmp_sparse_indices),
                  make_acl_tensor(cmp_block_table)};
              aclTensor *metadata_t = make_acl_tensor(*metadata);
              std::vector<aclTensor *> acl_outputs = {
                  make_acl_tensor(out_t), make_acl_tensor(lse_t)};

              uint64_t workspace_size = 0;
              aclOpExecutor *executor = nullptr;
              aclnnStatus status = get_ws_size(
                  acl_inputs[0], acl_inputs[1], acl_inputs[2], acl_inputs[3],
                  ori_len_arr, acl_inputs[4],
                  acl_inputs[5], acl_inputs[6], acl_inputs[7],
                  cmp_len_arr, cu_arr,
                  metadata_t,
                  layout_q_value.data(), layout_kv_value.data(), scale_v,
                  acl_outputs[0], acl_outputs[1],
                  &workspace_size, &executor);

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

              for (auto *tensor : acl_inputs) {
                  destroy_acl_tensor(tensor);
              }
              for (auto *tensor : acl_outputs) {
                  destroy_acl_tensor(tensor);
              }
              destroy_acl_tensor(metadata_t);
              destroy_acl_int_array(ori_len_arr);
              destroy_acl_int_array(cmp_len_arr);
              destroy_acl_int_array(cu_arr);
              TORCH_CHECK(status == 0, "aclnnMixedQuantSparseFlashMla call failed, status=", status);

              pybind11::dict result;
              result["out"] = out_t;
              result["lse"] = lse_t;
              return result;
          },
          pybind11::arg("q"),
          pybind11::arg("ori_kv"),
          pybind11::arg("ori_sparse_indices"),
          pybind11::arg("ori_block_table"),
          pybind11::arg("cu_seqlens_q"),
          pybind11::arg("sinks"),
          pybind11::arg("metadata") = pybind11::none(),
          pybind11::arg("quant_mode") = 1,
          pybind11::arg("softmax_scale") = pybind11::none(),
          pybind11::arg("cmp_kv") = pybind11::none(),
          pybind11::arg("cmp_sparse_indices") = pybind11::none(),
          pybind11::arg("cmp_block_table") = pybind11::none(),
          pybind11::arg("ori_topk_length") = pybind11::none(),
          pybind11::arg("cmp_topk_length") = pybind11::none(),
          pybind11::arg("seqused_q") = pybind11::none(),
          pybind11::arg("seqused_ori_kv") = pybind11::none(),
          pybind11::arg("seqused_cmp_kv") = pybind11::none(),
          pybind11::arg("layout_q") = "TND",
          pybind11::arg("layout_kv") = "PA_BBND",
          pybind11::arg("return_softmax_lse") = false,
          pybind11::arg("out") = pybind11::none(),
          pybind11::arg("lse") = pybind11::none(),
          pybind11::arg("block_dim") = pybind11::none(),
          pybind11::arg("fd_enable") = pybind11::none(),
          "aclnnMixedQuantSparseFlashMla wrapper (torch.Tensor interface)");
}