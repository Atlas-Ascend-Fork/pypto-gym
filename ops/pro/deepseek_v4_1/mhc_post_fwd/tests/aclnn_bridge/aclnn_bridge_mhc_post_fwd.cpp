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
 * \file aclnn_bridge_mhc_post_fwd.cpp
 * \brief
 *
 * pybind11 桥: 将 aclnn C++ 接口转为 Python 可调用的 torch.Tensor 接口.
 *
 * 运行时通过 dlopen/dlsym 从 libcust_opapi.so(ASCEND_CUSTOM_OPP_PATH / LD_LIBRARY_PATH)查找 aclnn 符号.
 *
 * 编译: ENABLE_TESTS=ON 时由 CMake 链编译为 aclnn_bridge_mhc_post_fwd.so → build/<soc>/tests/
 * 消费: pytest --mode aclnn → 根 conftest.py 注入 sys.path →
 *       op_caller.get_op_caller("mhc_post_fwd")
 *
 * jit/aclnn 入口签名一致约定:
 *   - out 是普通输出(kernel 只写不读); 未传入时由桥按 residual 形状分配;
 *   - 其余参数与 jit 宿主入口一一对应.
 */

#include "aclnn_bridge_common.h"

namespace {

using gym_test_bridge::destroy_acl_tensor;
using gym_test_bridge::get_current_npu_stream;
using gym_test_bridge::get_opapi_func;
using gym_test_bridge::make_acl_tensor;

// 形参顺序 = 自动生成头 aclnn_mhc_post_fwd.h
typedef aclnnStatus (*aclnnMhcPostFwdGetWorkspaceSizeFunc)(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, uint64_t *, aclOpExecutor **);
typedef aclnnStatus (*aclnnMhcPostFwdFunc)(
    void *, uint64_t, aclOpExecutor *, aclrtStream);

} // namespace

PYBIND11_MODULE(aclnn_bridge_mhc_post_fwd, m) {
    m.doc() = "pypto-gym pybind11 test bridge: aclnn C++ -> Python";

    m.def("mhc_post_fwd",
          [](torch::Tensor x,
             torch::Tensor residual,
             torch::Tensor post_layer_mix,
             torch::Tensor comb_res_mix,
             c10::optional<torch::Tensor> out) {
              static auto get_ws_size = reinterpret_cast<aclnnMhcPostFwdGetWorkspaceSizeFunc>(
                  get_opapi_func("aclnnMhcPostFwdGetWorkspaceSize"));
              static auto aclnn_op = reinterpret_cast<aclnnMhcPostFwdFunc>(
                  get_opapi_func("aclnnMhcPostFwd"));

              at::Tensor out_tensor = out.has_value() && out->defined()
                  ? *out
                  : at::empty(residual.sizes(), residual.options());
              if (out.has_value() && out->defined()) {
                  TORCH_CHECK(out_tensor.scalar_type() == at::kBFloat16,
                              "mhc_post_fwd: out must be bfloat16");
                  TORCH_CHECK(out_tensor.sizes() == residual.sizes(),
                              "mhc_post_fwd: out shape must equal residual shape");
                  TORCH_CHECK(out_tensor.is_contiguous(),
                              "mhc_post_fwd: out must be contiguous");
                  TORCH_CHECK(out_tensor.device() == residual.device(),
                              "mhc_post_fwd: out must live on the residual device");
              }

              std::vector<aclTensor *> acl_inputs = {
                  make_acl_tensor(x), make_acl_tensor(residual),
                  make_acl_tensor(post_layer_mix), make_acl_tensor(comb_res_mix)};
              std::vector<aclTensor *> acl_outputs = {make_acl_tensor(out_tensor)};

              uint64_t workspace_size = 0;
              aclOpExecutor *executor = nullptr;
              aclnnStatus status = get_ws_size(
                  acl_inputs[0], acl_inputs[1], acl_inputs[2], acl_inputs[3],
                  acl_outputs[0], &workspace_size, &executor);

              if (status == 0) {
                  at::Tensor workspace;
                  void *workspace_addr = nullptr;
                  if (workspace_size != 0) {
                      workspace = at::empty(
                          {static_cast<int64_t>(workspace_size)},
                          at::TensorOptions().dtype(at::kByte).device(residual.device()));
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
              TORCH_CHECK(status == 0, "aclnnMhcPostFwd call failed, status=", status);
              return out_tensor;
          },
          pybind11::arg("x"), pybind11::arg("residual"),
          pybind11::arg("post_layer_mix"), pybind11::arg("comb_res_mix"),
          pybind11::arg("out") = pybind11::none(),
          "aclnnMhcPostFwd wrapper (torch.Tensor interface)");
}
