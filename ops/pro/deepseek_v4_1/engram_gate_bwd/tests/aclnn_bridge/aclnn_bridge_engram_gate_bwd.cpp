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
 * \file aclnn_bridge_engram_gate_bwd.cpp
 * \brief
 *
 * pybind11 桥: 将 aclnn C++ 接口转为 Python 可调用的 torch.Tensor 接口.
 *
 * 运行时通过 dlopen/dlsym 从 libcust_opapi.so(ASCEND_CUSTOM_OPP_PATH / LD_LIBRARY_PATH)查找 aclnn 符号.
 *
 * 编译: ENABLE_TESTS=ON 时由 CMake 链编译为 aclnn_bridge_engram_gate_bwd.so → build/<soc>/tests/
 * 消费: pytest --mode aclnn → 根 conftest.py 注入 sys.path →
 *       op_caller.get_op_caller("engram_gate_bwd",
 *           jit_attr="engram_gate_bwd_wrapper", aclnn_attr="engram_gate_bwd_wrapper")
 *
 * jit/aclnn 入口签名一致: clamp_value 在 kernel 内是编译期立即数，桥接受参数但只允许默认值.
 * saved 由四份 forward 统计量在桥上打包; grad_w_partial 是 kernel 的 16 行归约缓冲,
 * 桥在同一个 stream 上求和成公开的 grad_weight.
 */

#include "aclnn_bridge_common.h"

#include <cmath>
#include <stdexcept>

namespace {

using gym_test_bridge::destroy_acl_tensor;
using gym_test_bridge::get_current_npu_stream;
using gym_test_bridge::get_opapi_func;
using gym_test_bridge::make_acl_tensor;

// 形参顺序 = 自动生成头 aclnn_engram_gate_bwd.h
typedef aclnnStatus (*aclnnEngramGateBwdGetWorkspaceSizeFunc)(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    uint64_t *, aclOpExecutor **);
typedef aclnnStatus (*aclnnEngramGateBwdFunc)(
    void *, uint64_t, aclOpExecutor *, aclrtStream);

} // namespace

PYBIND11_MODULE(aclnn_bridge_engram_gate_bwd, m) {
    m.doc() = "pypto-gym pybind11 test bridge: aclnn C++ -> Python";

    m.def("engram_gate_bwd_wrapper",
          [](torch::Tensor grad_out,
             torch::Tensor x,
             torch::Tensor key,
             torch::Tensor value,
             torch::Tensor weight,
             torch::Tensor rstd_x,
             torch::Tensor rstd_k,
             torch::Tensor raw_dot,
             torch::Tensor gate,
             torch::Tensor image_mask,
             double clamp_value) {
              static auto get_ws_size = reinterpret_cast<aclnnEngramGateBwdGetWorkspaceSizeFunc>(
                  get_opapi_func("aclnnEngramGateBwdGetWorkspaceSize"));
              static auto aclnn_op = reinterpret_cast<aclnnEngramGateBwdFunc>(
                  get_opapi_func("aclnnEngramGateBwd"));

              if (std::abs(clamp_value - 1.0e-6) > 0.0) {
                  throw std::invalid_argument("this kernel bakes clamp_value=1e-06");
              }
              const int64_t t = x.size(0);
              const int64_t hc = x.size(1);
              const int64_t dim = x.size(2);

              // [raw_dot | gate | rstd_x | rstd_k], the layout the kernel expects.
              at::Tensor saved = at::cat(
                  {raw_dot.to(at::kFloat), gate.to(at::kFloat),
                   rstd_x.to(at::kFloat), rstd_k.to(at::kFloat)}, 1).contiguous();

              at::Tensor grad_x = at::empty({t, hc, dim}, x.options());
              at::Tensor grad_key = at::empty({t, hc, dim}, key.options());
              at::Tensor grad_value = at::empty({t, dim}, value.options());
              // Zeroed: the kernel writes only (AIV / hc) rows; the rest must
              // contribute nothing to the sum(0) reduction.
              at::Tensor grad_w_partial = at::zeros(
                  {16, hc, dim}, at::TensorOptions().dtype(at::kFloat).device(x.device()));

              std::vector<aclTensor *> acl_inputs = {
                  make_acl_tensor(grad_out), make_acl_tensor(x), make_acl_tensor(key),
                  make_acl_tensor(value), make_acl_tensor(weight),
                  make_acl_tensor(saved), make_acl_tensor(image_mask)};
              std::vector<aclTensor *> acl_outputs = {
                  make_acl_tensor(grad_x), make_acl_tensor(grad_key),
                  make_acl_tensor(grad_value), make_acl_tensor(grad_w_partial)};

              uint64_t workspace_size = 0;
              aclOpExecutor *executor = nullptr;
              aclnnStatus status = get_ws_size(
                  acl_inputs[0], acl_inputs[1], acl_inputs[2], acl_inputs[3],
                  acl_inputs[4], acl_inputs[5], acl_inputs[6], acl_outputs[0],
                  acl_outputs[1], acl_outputs[2], acl_outputs[3],
                  &workspace_size, &executor);

              if (status == 0) {
                  at::Tensor workspace;
                  void *workspace_addr = nullptr;
                  if (workspace_size != 0) {
                      workspace = at::empty(
                          {static_cast<int64_t>(workspace_size)},
                          at::TensorOptions().dtype(at::kByte).device(x.device()));
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
              TORCH_CHECK(status == 0, "aclnnEngramGateBwd call failed, status=", status);

              // Same-stream reduction; ordered after the acc kernels.
              at::Tensor grad_weight = grad_w_partial.sum(0);
              return pybind11::make_tuple(grad_x, grad_key, grad_value, grad_weight);
          },
          pybind11::arg("grad_out"), pybind11::arg("x"), pybind11::arg("key"),
          pybind11::arg("value"), pybind11::arg("weight"), pybind11::arg("rstd_x"),
          pybind11::arg("rstd_k"), pybind11::arg("raw_dot"), pybind11::arg("gate"),
          pybind11::arg("image_mask"), pybind11::arg("clamp_value") = 1.0e-6,
          "aclnnEngramGateBwd wrapper (torch.Tensor interface)");
}
