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
 * \file aclnn_bridge_mhc_pre_fuse.cpp
 * \brief
 *
 * pybind11 桥: 将 aclnn C++ 接口转为 Python 可调用的 torch.Tensor 接口.
 *
 * 运行时通过 dlopen/dlsym 从 libcust_opapi.so(ASCEND_CUSTOM_OPP_PATH / LD_LIBRARY_PATH)查找 aclnn 符号.
 *
 * 编译: ENABLE_TESTS=ON 时由 CMake 链编译为 aclnn_bridge_mhc_pre_fuse.so → build/<soc>/tests/
 * 消费: pytest --mode aclnn → 根 conftest.py 注入 sys.path →
 *       op_caller.get_op_caller("mhc_pre_fuse",
 *           jit_attr="mhc_pre_fuse_wrapper", aclnn_attr="mhc_pre_fuse_wrapper")
 *
 * jit/aclnn 入口签名一致约定:
 *   - gemm_out_mul 带前导 split 轴(3D) / gemm_out_sqrsum(2D) 时，桥只取第 0 份，
 *     与 jit wrapper 的 ``[0]`` 语义一致;
 *   - knobs 是 Python dataclass，桥按属性名读取五个字段；缺省用契约默认值;
 *   - 三个输出由桥分配并返回.
 */

#include "aclnn_bridge_common.h"

namespace {

using gym_test_bridge::destroy_acl_tensor;
using gym_test_bridge::get_current_npu_stream;
using gym_test_bridge::get_opapi_func;
using gym_test_bridge::make_acl_tensor;

// 形参顺序 = 自动生成头 aclnn_mhc_pre_fuse.h
typedef aclnnStatus (*aclnnMhcPreFuseGetWorkspaceSizeFunc)(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *,
    double, double, double, double, int64_t,
    const aclTensor *, const aclTensor *, const aclTensor *,
    uint64_t *, aclOpExecutor **);
typedef aclnnStatus (*aclnnMhcPreFuseFunc)(
    void *, uint64_t, aclOpExecutor *, aclrtStream);

} // namespace

PYBIND11_MODULE(aclnn_bridge_mhc_pre_fuse, m) {
    m.doc() = "pypto-gym pybind11 test bridge: aclnn C++ -> Python";

    m.def("mhc_pre_fuse_wrapper",
          [](torch::Tensor gemm_out_mul,
             torch::Tensor gemm_out_sqrsum,
             torch::Tensor mhc_scale,
             torch::Tensor mhc_base,
             torch::Tensor residual,
             pybind11::object knobs) {
              static auto get_ws_size = reinterpret_cast<aclnnMhcPreFuseGetWorkspaceSizeFunc>(
                  get_opapi_func("aclnnMhcPreFuseGetWorkspaceSize"));
              static auto aclnn_op = reinterpret_cast<aclnnMhcPreFuseFunc>(
                  get_opapi_func("aclnnMhcPreFuse"));

              if (gemm_out_mul.dim() == 3) {
                  gemm_out_mul = gemm_out_mul[0];
              }
              if (gemm_out_sqrsum.dim() == 2) {
                  gemm_out_sqrsum = gemm_out_sqrsum[0];
              }

              double rms_eps = 1.0e-6;
              double pre_eps = 1.0e-6;
              double sinkhorn_eps = 1.0e-6;
              double post_mult = 1.0;
              int64_t sinkhorn_repeat = 10;
              if (!knobs.is_none()) {
                  rms_eps = knobs.attr("rms_eps").cast<double>();
                  pre_eps = knobs.attr("mhc_pre_eps").cast<double>();
                  sinkhorn_eps = knobs.attr("mhc_sinkhorn_eps").cast<double>();
                  post_mult = knobs.attr("mhc_post_mult_value").cast<double>();
                  sinkhorn_repeat = knobs.attr("sinkhorn_repeat").cast<int64_t>();
              }

              const int64_t t = residual.size(0);
              const int64_t h = residual.size(2);
              at::Tensor post_mix = at::empty(
                  {t, 4}, residual.options().dtype(at::kFloat));
              at::Tensor comb_mix = at::empty(
                  {t, 16}, residual.options().dtype(at::kFloat));
              at::Tensor layer_input = at::empty(
                  {t, h}, residual.options().dtype(at::kBFloat16));

              std::vector<aclTensor *> acl_inputs = {
                  make_acl_tensor(gemm_out_mul), make_acl_tensor(gemm_out_sqrsum),
                  make_acl_tensor(mhc_scale), make_acl_tensor(mhc_base),
                  make_acl_tensor(residual)};
              std::vector<aclTensor *> acl_outputs = {
                  make_acl_tensor(post_mix), make_acl_tensor(comb_mix),
                  make_acl_tensor(layer_input)};

              uint64_t workspace_size = 0;
              aclOpExecutor *executor = nullptr;
              aclnnStatus status = get_ws_size(
                  acl_inputs[0], acl_inputs[1], acl_inputs[2], acl_inputs[3], acl_inputs[4],
                  rms_eps, pre_eps, sinkhorn_eps, post_mult, sinkhorn_repeat,
                  acl_outputs[0], acl_outputs[1], acl_outputs[2],
                  &workspace_size, &executor);

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
              TORCH_CHECK(status == 0, "aclnnMhcPreFuse call failed, status=", status);
              return std::make_tuple(post_mix, comb_mix, layer_input);
          },
          pybind11::arg("gemm_out_mul"), pybind11::arg("gemm_out_sqrsum"),
          pybind11::arg("mhc_scale"), pybind11::arg("mhc_base"),
          pybind11::arg("residual"), pybind11::arg("knobs") = pybind11::none(),
          "aclnnMhcPreFuse wrapper (torch.Tensor interface)");
}
