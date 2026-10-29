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
 *//**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_bridge_flex_common.h
 * \brief
 *
 * aclnn 测试桥公共层: torch <-> aclnn 的类型转换与运行库符号加载.
 *
 * 仅测试使用(ENABLE_TESTS=ON), 不随 whl/run 交付.
 * 运行时按动态库符号查找链加载 aclnn 入口(dlopen/dlsym).
 *
 * 公共层(torch<->aclnn 转换/符号加载)见 tests/ops/pro/aclnn_bridge_common/aclnn_bridge_common.h.
 *
 * 消费: 算子 tests/ 下的 pybind11 桥 include 本头, 并 link CMake 目标 aclnn_bridge_common.
 * 实现为 header-only(inline + 函数内 static 缓存), 各算子模块独立编译、无公共库链接依赖.
 */

#ifndef PYPTO_GYM_ACLNN_BRIDGE_COMMON_H
#define PYPTO_GYM_ACLNN_BRIDGE_COMMON_H

#include <torch/extension.h>
#include <aclnn/acl_meta.h>
#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace gym_test_bridge {

// 查找链: ① ASCEND_CUSTOM_OPP_PATH → ② LD_LIBRARY_PATH
inline void *load_opapi_lib() {
    static void *handle = []() -> void * {
        const char *custom_opp = std::getenv("ASCEND_CUSTOM_OPP_PATH");
        if (custom_opp != nullptr) {
            const std::string lib_path =
                std::string(custom_opp) + "/op_api/lib/libcust_opapi.so";
            void *lib = dlopen(lib_path.c_str(), RTLD_LAZY);
            if (lib != nullptr) {
                return lib;
            }
        }
        return dlopen("libcust_opapi.so", RTLD_LAZY);
    }();
    return handle;
}

inline void *get_opapi_func(const char *name) {
    void *lib = load_opapi_lib();
    TORCH_CHECK(lib != nullptr, "libcust_opapi.so not found; "
                "install .run first and source <vendor>/bin/set_env.bash "
                "or export ASCEND_CUSTOM_OPP_PATH=<vendor_path>");
    void *func = dlsym(lib, name);
    TORCH_CHECK(func != nullptr, "dlsym ", name, " from libcust_opapi.so failed");
    return func;
}

// aclCreateTensor/aclDestroyTensor 等由 libnnopbase.so 提供(libcust_opapi.so 的运行时依赖).
inline void *get_acl_meta_func(const char *name) {
    struct Cache {
        void *create_tensor = nullptr;
        void *destroy_tensor = nullptr;
        void *create_int_array = nullptr;
        void *destroy_int_array = nullptr;
    };
    static Cache cache;
    const std::string func_name(name);
    void *&entry = (func_name == "aclCreateTensor") ? cache.create_tensor
                 : (func_name == "aclDestroyTensor") ? cache.destroy_tensor
                 : (func_name == "aclCreateIntArray") ? cache.create_int_array
                 : cache.destroy_int_array;
    if (entry != nullptr) {
        return entry;
    }
    void *lib = dlopen("libnnopbase.so", RTLD_LAZY | RTLD_GLOBAL);
    if (lib == nullptr) {
        void *cust = load_opapi_lib();
        TORCH_CHECK(cust != nullptr, "libcust_opapi.so not found");
        lib = dlopen("libnnopbase.so", RTLD_LAZY | RTLD_GLOBAL);
    }
    TORCH_CHECK(lib != nullptr, "dlopen libnnopbase.so failed: ", dlerror());
    entry = dlsym(lib, name);
    TORCH_CHECK(entry != nullptr, "dlsym ", name, " from libnnopbase.so failed");
    return entry;
}

inline aclDataType to_acl_dtype(at::ScalarType dtype) {
    switch (dtype) {
        case at::ScalarType::Half: return ACL_FLOAT16;
        case at::ScalarType::BFloat16: return ACL_BF16;
        case at::ScalarType::Float: return ACL_FLOAT;
        case at::ScalarType::Int: return ACL_INT32;
        case at::ScalarType::Long: return ACL_INT64;
        case at::ScalarType::UInt32: return ACL_UINT32;
        case at::ScalarType::Byte: return ACL_UINT8;
        case at::ScalarType::Char: return ACL_INT8;
        case at::ScalarType::Short: return ACL_INT16;
        case at::ScalarType::Bool: return ACL_BOOL;
        default:
            TORCH_CHECK(false, "unsupported tensor dtype for aclTensor: ", c10::toString(dtype));
    }
    return ACL_DT_UNDEFINED;
}

inline aclTensor *make_acl_tensor(const c10::optional<torch::Tensor> &opt_tensor) {
    if (!opt_tensor.has_value() || !opt_tensor->defined()) {
        return nullptr;
    }
    const torch::Tensor &tensor = *opt_tensor;
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
                         to_acl_dtype(tensor.scalar_type()), strides.data(),
                         tensor.storage_offset(), ACL_FORMAT_ND, storage_dims.data(),
                         static_cast<uint64_t>(storage_dims.size()),
                         const_cast<void *>(tensor.storage().data()));
}

// ValueDepend 输入由 host tiling 消费: op_build 默认接口把它生成为 aclIntArray
// (host 值), nnopbase 只对数组输入提供 Host placement, 因此这里物化 host 拷贝.
inline aclIntArray *make_acl_int_array(const c10::optional<torch::Tensor> &opt_tensor,
                                       std::vector<torch::Tensor> &keepalive) {
    if (!opt_tensor.has_value() || !opt_tensor->defined()) {
        return nullptr;
    }
    TORCH_CHECK(opt_tensor->scalar_type() == at::kLong, "ValueDepend array input must be int64");
    keepalive.push_back(opt_tensor->to(at::kCPU).contiguous());
    const torch::Tensor &host = keepalive.back();
    auto create_int_array = reinterpret_cast<aclIntArray *(*)(const int64_t *, uint64_t)>(
        get_acl_meta_func("aclCreateIntArray"));
    return create_int_array(host.data_ptr<int64_t>(), static_cast<uint64_t>(host.numel()));
}

inline void destroy_acl_tensor(aclTensor *tensor) {
    if (tensor == nullptr) {
        return;
    }
    auto destroy_tensor = reinterpret_cast<aclnnStatus (*)(const aclTensor *)>(
        get_acl_meta_func("aclDestroyTensor"));
    destroy_tensor(tensor);
}

inline void destroy_acl_int_array(aclIntArray *array) {
    if (array == nullptr) {
        return;
    }
    auto destroy_array = reinterpret_cast<aclnnStatus (*)(const aclIntArray *)>(
        get_acl_meta_func("aclDestroyIntArray"));
    destroy_array(array);
}

inline aclrtStream get_current_npu_stream() {
    // 从 Python 侧取 torch_npu 当前流, 避免桥直接依赖 torch_npu C++ 头/库.
    pybind11::object stream_obj =
        pybind11::module_::import("torch_npu").attr("npu").attr("current_stream")();
    int64_t raw_stream = stream_obj.attr("npu_stream").cast<int64_t>();
    return reinterpret_cast<aclrtStream>(raw_stream);
}

} // namespace gym_test_bridge

#endif // PYPTO_GYM_ACLNN_BRIDGE_COMMON_H
