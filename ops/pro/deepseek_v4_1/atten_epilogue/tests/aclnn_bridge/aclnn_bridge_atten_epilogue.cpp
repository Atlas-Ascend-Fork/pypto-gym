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
 * \file aclnn_bridge_atten_epilogue.cpp
 * \brief pybind11 桥: 将 aclnn C++ 接口转为 Python 可调用的 torch.Tensor 接口.
 *
 * 运行时通过 dlopen/dlsym 从 libcust_opapi.so(ASCEND_CUSTOM_OPP_PATH / LD_LIBRARY_PATH)查找 aclnn 符号.
 * 公共层(符号加载/tensor 销毁/stream 获取)见 tests/ops/pro/aclnn_bridge_common/aclnn_bridge_common.h.
 *
 * 本算子的输入含 FP8E4M3FN / FP8E8M0, 公共层的 to_acl_dtype 不覆盖这两种类型,
 * 因此这里自带一份 dtype 映射与 aclTensor 构造(make_fp8_acl_tensor), 其余复用公共层.
 *
 * 编译: ENABLE_TESTS=ON 时由 CMake 链编译为 aclnn_bridge_atten_epilogue.so → build/<soc>/tests/
 * 消费: pytest --mode aclnn → 根 conftest.py 注入 sys.path → op_caller.get_op_caller("atten_epilogue")
 */

#include "aclnn_bridge_common.h"

namespace {

using gym_test_bridge::destroy_acl_tensor;
using gym_test_bridge::get_acl_meta_func;
using gym_test_bridge::get_current_npu_stream;
using gym_test_bridge::get_opapi_func;

// aclDataType (include/acl/acl_base_rt.h); the common header only maps the non-FP8 subset.
constexpr int kAclFloat = 0;
constexpr int kAclBf16 = 27;
constexpr int kAclFloat8E4m3fn = 36;
constexpr int kAclFloat8E8m0 = 37;

aclDataType epilogue_acl_dtype(at::ScalarType dtype)
{
    switch (dtype) {
        case at::ScalarType::BFloat16: return static_cast<aclDataType>(kAclBf16);
        case at::ScalarType::Float: return static_cast<aclDataType>(kAclFloat);
        case at::ScalarType::Float8_e4m3fn: return static_cast<aclDataType>(kAclFloat8E4m3fn);
        case at::ScalarType::Float8_e8m0fnu: return static_cast<aclDataType>(kAclFloat8E8m0);
        default:
            TORCH_CHECK(false, "unsupported tensor dtype for aclTensor: ", c10::toString(dtype));
    }
    return ACL_DT_UNDEFINED;
}

// 与公共层 make_acl_tensor 同构, 仅 dtype 映射不同(FP8 感知).
aclTensor *make_epilogue_acl_tensor(const torch::Tensor &tensor)
{
    TORCH_CHECK(tensor.defined(), "undefined tensor passed to the aclnn bridge");
    TORCH_CHECK(tensor.itemsize() > 0, "tensor itemsize must be non-zero");
    auto create_tensor = reinterpret_cast<aclTensor *(*)(const int64_t *, uint64_t, aclDataType,
                                                         const int64_t *, int64_t, aclFormat,
                                                         const int64_t *, uint64_t, void *)>(
        get_acl_meta_func("aclCreateTensor"));
    const auto sizes = tensor.sizes();
    const auto strides = tensor.strides();
    std::vector<int64_t> storage_dims = {
        static_cast<int64_t>(tensor.storage().nbytes() / tensor.itemsize())};
    // 算子注册的所有输入/输出均为 FORMAT_ND, 不能按 rank 推断 NCL/NCHW.
    return create_tensor(sizes.data(), static_cast<uint64_t>(sizes.size()),
                         epilogue_acl_dtype(tensor.scalar_type()), strides.data(),
                         tensor.storage_offset(), ACL_FORMAT_ND, storage_dims.data(),
                         static_cast<uint64_t>(storage_dims.size()),
                         const_cast<void *>(tensor.storage().data()));
}

typedef aclnnStatus (*aclnnAttenEpilogueGetWorkspaceSizeFunc)(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    uint64_t *, aclOpExecutor **);
typedef aclnnStatus (*aclnnAttenEpilogueFunc)(void *, uint64_t, aclOpExecutor *, aclrtStream);

} // namespace

PYBIND11_MODULE(aclnn_bridge_atten_epilogue, m) {
    m.doc() = "pypto-gym pybind11 test bridge: aclnn C++ -> Python";

    m.def("atten_epilogue",
          [](torch::Tensor o, torch::Tensor woa, torch::Tensor wob,
             torch::Tensor descale_woa, torch::Tensor descale_wob,
             torch::Tensor rope_sin, torch::Tensor rope_cos, torch::Tensor out) {
              static auto get_ws_size = reinterpret_cast<aclnnAttenEpilogueGetWorkspaceSizeFunc>(
                  get_opapi_func("aclnnAttenEpilogueGetWorkspaceSize"));
              static auto aclnn_op = reinterpret_cast<aclnnAttenEpilogueFunc>(
                  get_opapi_func("aclnnAttenEpilogue"));

              std::vector<aclTensor *> acl_tensors = {
                  make_epilogue_acl_tensor(o), make_epilogue_acl_tensor(woa),
                  make_epilogue_acl_tensor(wob), make_epilogue_acl_tensor(descale_woa),
                  make_epilogue_acl_tensor(descale_wob), make_epilogue_acl_tensor(rope_sin),
                  make_epilogue_acl_tensor(rope_cos), make_epilogue_acl_tensor(out)};

              uint64_t workspace_size = 0;
              aclOpExecutor *executor = nullptr;
              aclnnStatus status = get_ws_size(
                  acl_tensors[0], acl_tensors[1], acl_tensors[2], acl_tensors[3],
                  acl_tensors[4], acl_tensors[5], acl_tensors[6], acl_tensors[7],
                  &workspace_size, &executor);

              if (status == 0) {
                  at::Tensor workspace;
                  void *workspace_addr = nullptr;
                  if (workspace_size != 0) {
                      workspace = at::empty(
                          {static_cast<int64_t>(workspace_size)},
                          at::TensorOptions().dtype(at::kByte).device(o.device()));
                      workspace_addr = workspace.data_ptr();
                  }
                  aclrtStream stream = get_current_npu_stream();
                  status = aclnn_op(workspace_addr, workspace_size, executor, stream);
              }

              for (auto *tensor : acl_tensors) {
                  destroy_acl_tensor(tensor);
              }
              TORCH_CHECK(status == 0, "aclnnAttenEpilogue call failed, status=", status);
              return out;
          },
          pybind11::arg("o"), pybind11::arg("woa"), pybind11::arg("wob"),
          pybind11::arg("descale_woa"), pybind11::arg("descale_wob"),
          pybind11::arg("rope_sin"), pybind11::arg("rope_cos"), pybind11::arg("out"),
          "aclnnAttenEpilogue wrapper (torch.Tensor interface)");
}
