# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# pro 范式 aclnn_bridge_common 公共层: header-only INTERFACE 目标 + 算子桥构建辅助函数.
# 由根 CMakeLists 在 ENABLE_TESTS 时逐级 add_subdirectory 引入;
# torch 依赖由 cmake/config.cmake 在根作用域统一查找, 此处直接复用其变量与 imported target.

add_library(aclnn_bridge_common INTERFACE)
target_include_directories(aclnn_bridge_common
        INTERFACE
                ${CMAKE_CURRENT_SOURCE_DIR}
                ${TORCH_INCLUDE_DIRS}
                ${ASCEND_CANN_PACKAGE_PATH}/include
)
target_link_libraries(aclnn_bridge_common
        INTERFACE
                ${TORCH_LIBRARIES}
                ${TORCH_PYTHON_LIB}
                dl
)

# 按命名约定创建算子 aclnn 桥: 源文件/目标/产物/模块名统一为 aclnn_bridge_<op>.
# 在算子 tests/aclnn_bridge/CMakeLists.txt 内调用(CMAKE_CURRENT_SOURCE_DIR 即该目录).
function(gym_add_aclnn_bridge op_name)
    set(_bridge_target "aclnn_bridge_${op_name}")
    set(_bridge_source "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_bridge_${op_name}.cpp")
    if (NOT EXISTS "${_bridge_source}")
        message(FATAL_ERROR "aclnn bridge source not found: ${_bridge_source}")
    endif()
    add_library(${_bridge_target} SHARED EXCLUDE_FROM_ALL "${_bridge_source}")
    set_target_properties(${_bridge_target} PROPERTIES
            PREFIX ""
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    target_link_libraries(${_bridge_target} PRIVATE aclnn_bridge_common)
    add_dependencies(gym_test_bridges ${_bridge_target})
    message(STATUS "aclnn bridge: ${_bridge_source} -> ${CMAKE_BINARY_DIR}/tests/${_bridge_target}.so")
endfunction()
