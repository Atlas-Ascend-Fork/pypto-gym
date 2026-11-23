# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.18.4)
add_op_to_compiled_list()

# 纯 AICPU 算子: 无 def/proto/tiling, aclnn 入口手写在 op_api/ 下直接进 gym_opapi_manual;
# AICPU kernel 由 op_kernel_aicpu/CMakeLists.txt 的 enable_pypto_aicpu_kernel 接入.
file(GLOB _opapi_srcs ${CMAKE_CURRENT_SOURCE_DIR}/op_api/*.cpp)
set_property(GLOBAL APPEND PROPERTY GYM_OPAPI_SRCS ${_opapi_srcs})
