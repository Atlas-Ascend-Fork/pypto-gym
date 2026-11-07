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
 * \file aclnn_bridge_kv_compress_epilog_v2.cpp
 * \brief
 *
 * pybind11 桥: 将 aclnn C++ 接口转为 Python 可调用的 torch.Tensor 接口.
 *
 * 运行时通过 dlopen/dlsym 从 libcust_opapi.so(ASCEND_CUSTOM_OPP_PATH / LD_LIBRARY_PATH)查找 aclnn 符号.
 *
 * 编译: ENABLE_TESTS=ON 时由 CMake 链编译为 aclnn_bridge_kv_compress_epilog_v2.so → build/<soc>/tests/
 * 消费: pytest --mode aclnn → 根 conftest.py 注入 sys.path →
 *       op_caller.get_op_caller("kv_compress_epilog_v2")
 *
 * jit/aclnn 入口签名一致约定:
 *   - cache 是原地更新对象, 桥把它同时作为输入和输出传给 aclnn;
 *   - slot_mapping 在 aclnn 侧统一为 INT64, INT32 调用方在桥内转换;
 *   - storage_bytes 由桥从 torch storage 计算, 供 host tiling 校验物理写入范围;
 *   - 其余属性与 jit 宿主入口一一对应.
 *
 * 注意: 形参顺序/类型以自动生成的 autogen/aclnn_kv_compress_epilog_v2.h 为准.
 */

#include "aclnn_bridge_common.h"

namespace {

using gym_test_bridge::destroy_acl_tensor;
using gym_test_bridge::get_current_npu_stream;
using gym_test_bridge::get_opapi_func;
using gym_test_bridge::make_acl_tensor;

// 形参顺序 = 自动生成头 aclnn_kv_compress_epilog_v2.h
// cache 是 ref 输出(输入/输出同名): 生成接口只有一个 aclTensor *cacheRef,
// 不再有独立的输出形参; 框架通过 NnopbaseSetRef(0, 0) 建立原地别名.
typedef aclnnStatus (*aclnnKvCompressEpilogV2GetWorkspaceSizeFunc)(
    aclTensor *, const aclTensor *, const aclTensor *,
    int64_t, int64_t, bool, double, int64_t, int64_t,
    uint64_t *, aclOpExecutor **);
typedef aclnnStatus (*aclnnKvCompressEpilogV2Func)(
    void *, uint64_t, aclOpExecutor *, aclrtStream);

} // namespace

PYBIND11_MODULE(aclnn_bridge_kv_compress_epilog_v2, m) {
    m.doc() = "pypto-gym pybind11 test bridge: aclnn C++ -> Python";

    m.def("kv_compress_epilog_v2",
          [](torch::Tensor cache,
             torch::Tensor x,
             torch::Tensor slot_mapping,
             int64_t quant_group_size,
             int64_t quant_mode,
             bool round_scale,
             double x_scale,
             int64_t block_stride) {
              static auto get_ws_size = reinterpret_cast<aclnnKvCompressEpilogV2GetWorkspaceSizeFunc>(
                  get_opapi_func("aclnnKvCompressEpilogV2GetWorkspaceSize"));
              static auto aclnn_op = reinterpret_cast<aclnnKvCompressEpilogV2Func>(
                  get_opapi_func("aclnnKvCompressEpilogV2"));

              at::Tensor slot64 = slot_mapping.scalar_type() == at::kLong
                  ? slot_mapping
                  : slot_mapping.to(at::kLong);
              const int64_t storage_bytes =
                  static_cast<int64_t>(cache.storage().nbytes()) -
                  cache.storage_offset() * cache.itemsize();

              std::vector<aclTensor *> acl_inputs = {
                  make_acl_tensor(cache), make_acl_tensor(x), make_acl_tensor(slot64)};

              uint64_t workspace_size = 0;
              aclOpExecutor *executor = nullptr;
              aclnnStatus status = get_ws_size(
                  acl_inputs[0], acl_inputs[1], acl_inputs[2],
                  quant_group_size, quant_mode, round_scale, x_scale, block_stride, storage_bytes,
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
              TORCH_CHECK(status == 0, "aclnnKvCompressEpilogV2 call failed, status=", status);
              return cache;
          },
          pybind11::arg("cache"), pybind11::arg("x"), pybind11::arg("slot_mapping"),
          pybind11::arg("quant_group_size") = 32,
          pybind11::arg("quant_mode") = 2,
          pybind11::arg("round_scale") = true,
          pybind11::arg("x_scale") = 1.0,
          pybind11::arg("block_stride") = 0,
          "aclnnKvCompressEpilogV2 wrapper (torch.Tensor interface)");
}
