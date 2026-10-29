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
 * \file aclnn_bridge_flex_attention.h
 * \brief
 *
 * pybind11 桥: 将 aclnn C++ 接口转为 Python 可调用的 torch.Tensor 接口.
 *
 * 运行时通过 dlopen/dlsym 从 libcust_opapi.so(ASCEND_CUSTOM_OPP_PATH / LD_LIBRARY_PATH)查找 aclnn 符号,
 * 运行时按动态库符号查找链加载 aclnn 入口(dlopen/dlsym).
 *
 * 公共层(torch<->aclnn 转换/符号加载)见 tests/ops/pro/aclnn_bridge_common/aclnn_bridge_common.h.
 *
 * 编译: ENABLE_TESTS=ON 时由 CMake 链编译为 aclnn_bridge_flex_attention.so → build/<soc>/tests/
 * 消费: pytest --mode aclnn → 根 conftest.py 注入 sys.path → op_caller.get_op_caller("flex_attention")
 */

#include "aclnn_bridge_common.h"

namespace {

using gym_test_bridge::destroy_acl_int_array;
using gym_test_bridge::destroy_acl_tensor;
using gym_test_bridge::get_current_npu_stream;
using gym_test_bridge::get_opapi_func;
using gym_test_bridge::make_acl_int_array;
using gym_test_bridge::make_acl_tensor;

typedef aclnnStatus (*aclnnFlexAttentionGetWorkspaceSizeFunc)(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclIntArray *,
    const aclTensor *, const aclIntArray *, const aclTensor *, char *,
    const aclTensor *, const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
typedef aclnnStatus (*aclnnFlexAttentionFunc)(
    void *, uint64_t, aclOpExecutor *, aclrtStream);

} // namespace

PYBIND11_MODULE(aclnn_bridge_flex_attention, m) {
    m.doc() = "pypto-gym pybind11 test bridge: aclnn C++ -> Python";

    m.def("flex_attention",
          [](torch::Tensor query, torch::Tensor key, torch::Tensor value,
             c10::optional<torch::Tensor> actual_seq_q,
             c10::optional<torch::Tensor> actual_seq_kv,
             c10::optional<torch::Tensor> maskr,
             c10::optional<torch::Tensor> mask_start,
             c10::optional<torch::Tensor> mask_len,
             c10::optional<torch::Tensor> block_range,
             c10::optional<torch::Tensor> block_compute,
             c10::optional<torch::Tensor> block_mask,
             torch::Tensor attn_out,
             c10::optional<torch::Tensor> softmax_max,
             c10::optional<torch::Tensor> softmax_sum,
             const std::string &layout) {
              static auto get_ws_size = reinterpret_cast<aclnnFlexAttentionGetWorkspaceSizeFunc>(
                  get_opapi_func("aclnnFlexAttentionGetWorkspaceSize"));
              static auto aclnn_op = reinterpret_cast<aclnnFlexAttentionFunc>(
                  get_opapi_func("aclnnFlexAttention"));

              std::vector<torch::Tensor> host_arrays;
              std::vector<aclTensor *> acl_inputs = {
                  make_acl_tensor(query), make_acl_tensor(key), make_acl_tensor(value),
                  make_acl_tensor(maskr), make_acl_tensor(mask_start), make_acl_tensor(mask_len),
                  make_acl_tensor(block_range), nullptr, make_acl_tensor(block_mask), nullptr,
                  make_acl_tensor(actual_seq_kv)};
              aclIntArray *block_compute_array = make_acl_int_array(block_compute, host_arrays);
              aclIntArray *actual_seq_q_array = make_acl_int_array(actual_seq_q, host_arrays);
              std::string layout_value = layout;
              std::vector<aclTensor *> acl_outputs = {
                  make_acl_tensor(attn_out), make_acl_tensor(softmax_max),
                  make_acl_tensor(softmax_sum)};

              uint64_t workspace_size = 0;
              aclOpExecutor *executor = nullptr;
              aclnnStatus status = get_ws_size(
                  acl_inputs[0], acl_inputs[1], acl_inputs[2], acl_inputs[3],
                  acl_inputs[4], acl_inputs[5], acl_inputs[6], block_compute_array,
                  acl_inputs[8], actual_seq_q_array, acl_inputs[10],
                  layout_value.data(),
                  acl_outputs[0], acl_outputs[1], acl_outputs[2],
                  &workspace_size, &executor);

              if (status == 0) {
                  at::Tensor workspace;
                  void *workspace_addr = nullptr;
                  if (workspace_size != 0) {
                      workspace = at::empty(
                          {static_cast<int64_t>(workspace_size)},
                          at::TensorOptions().dtype(at::kByte).device(query.device()));
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
              destroy_acl_int_array(block_compute_array);
              destroy_acl_int_array(actual_seq_q_array);
              TORCH_CHECK(status == 0, "aclnnFlexAttention call failed, status=", status);
              return attn_out;
          },
          pybind11::arg("query"), pybind11::arg("key"), pybind11::arg("value"),
          pybind11::arg("actual_seq_q"), pybind11::arg("actual_seq_kv"),
          pybind11::arg("maskr"),
          pybind11::arg("mask_start") = pybind11::none(),
          pybind11::arg("mask_len") = pybind11::none(),
          pybind11::arg("block_range") = pybind11::none(),
          pybind11::arg("block_compute") = pybind11::none(),
          pybind11::arg("block_mask") = pybind11::none(),
          pybind11::arg("attn_out"),
          pybind11::arg("softmax_max") = pybind11::none(),
          pybind11::arg("softmax_sum") = pybind11::none(),
          pybind11::arg("layout") = "varlen",
          "aclnnFlexAttention wrapper (torch.Tensor interface)");
}
