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
 * \file aclnn_bridge_engram_gate.cpp
 * \brief
 *
 * pybind11 桥: 将 aclnn C++ 接口转为 Python 可调用的 torch.Tensor 接口.
 *
 * 运行时通过 dlopen/dlsym 从 libcust_opapi.so(ASCEND_CUSTOM_OPP_PATH / LD_LIBRARY_PATH)查找 aclnn 符号.
 *
 * 编译: ENABLE_TESTS=ON 时由 CMake 链编译为 aclnn_bridge_engram_gate.so → build/<soc>/tests/
 * 消费: pytest --mode aclnn → 根 conftest.py 注入 sys.path →
 *       op_caller.get_op_caller("engram_gate",
 *           jit_attr="engram_gate_wrapper", aclnn_attr="engram_gate_wrapper")
 *
 * jit/aclnn 入口签名一致: eps/clamp_value 在 kernel 内是编译期立即数，桥接受参数但只允许默认值.
 */

#include "aclnn_bridge_common.h"

#include <cmath>
#include <stdexcept>

namespace {

using gym_test_bridge::destroy_acl_tensor;
using gym_test_bridge::get_current_npu_stream;
using gym_test_bridge::get_opapi_func;
using gym_test_bridge::make_acl_tensor;

// 形参顺序 = 自动生成头 aclnn_engram_gate.h
typedef aclnnStatus (*aclnnEngramGateGetWorkspaceSizeFunc)(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
typedef aclnnStatus (*aclnnEngramGateFunc)(
    void *, uint64_t, aclOpExecutor *, aclrtStream);

} // namespace

PYBIND11_MODULE(aclnn_bridge_engram_gate, m) {
    m.doc() = "pypto-gym pybind11 test bridge: aclnn C++ -> Python";

    m.def("engram_gate_wrapper",
          [](torch::Tensor x,
             torch::Tensor key,
             torch::Tensor value,
             torch::Tensor weight,
             torch::Tensor image_mask,
             double eps,
             double clamp_value) {
              static auto get_ws_size = reinterpret_cast<aclnnEngramGateGetWorkspaceSizeFunc>(
                  get_opapi_func("aclnnEngramGateGetWorkspaceSize"));
              static auto aclnn_op = reinterpret_cast<aclnnEngramGateFunc>(
                  get_opapi_func("aclnnEngramGate"));

              if (std::abs(eps - 1.0e-6) > 0.0 || std::abs(clamp_value - 1.0e-6) > 0.0) {
                  throw std::invalid_argument(
                      "this kernel bakes eps=1e-06 and clamp_value=1e-06");
              }
              const int64_t t = x.size(0);
              const int64_t hc = x.size(1);
              const int64_t dim = x.size(2);
              at::Tensor y = at::empty({t, hc, dim}, x.options());

              std::vector<aclTensor *> acl_inputs = {
                  make_acl_tensor(x), make_acl_tensor(key), make_acl_tensor(value),
                  make_acl_tensor(weight), make_acl_tensor(image_mask)};
              std::vector<aclTensor *> acl_outputs = {make_acl_tensor(y)};

              uint64_t workspace_size = 0;
              aclOpExecutor *executor = nullptr;
              aclnnStatus status = get_ws_size(
                  acl_inputs[0], acl_inputs[1], acl_inputs[2], acl_inputs[3],
                  acl_inputs[4], acl_outputs[0], &workspace_size, &executor);

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
              TORCH_CHECK(status == 0, "aclnnEngramGate call failed, status=", status);
              return y;
          },
          pybind11::arg("x"), pybind11::arg("key"), pybind11::arg("value"),
          pybind11::arg("weight"), pybind11::arg("image_mask"),
          pybind11::arg("eps") = 1.0e-6, pybind11::arg("clamp_value") = 1.0e-6,
          "aclnnEngramGate wrapper (torch.Tensor interface)");
}
