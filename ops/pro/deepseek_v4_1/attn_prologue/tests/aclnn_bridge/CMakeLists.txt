# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# attn_prologue 的 aclnn 模式 pybind11 桥. 权重是 FRACTAL_NZ, 因此走 torch_npu 的
# ConvertType 而不是公共层的 ACL_FORMAT_ND 手工构造.
gym_add_aclnn_bridge(attn_prologue)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch_npu,os;print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT TORCH_NPU_ROOT)
    message(FATAL_ERROR "torch_npu not importable; required for the attn_prologue NZ bridge")
endif()

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig;print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

target_include_directories(aclnn_bridge_attn_prologue PRIVATE
    ${TORCH_NPU_ROOT}/include
    ${TORCH_NPU_ROOT}/include/third_party/op-plugin
    ${PYTHON_INCLUDE_DIR})
target_link_libraries(aclnn_bridge_attn_prologue PRIVATE ${TORCH_NPU_ROOT}/lib/libtorch_npu.so)
