# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# 算子级 dispatcher(对齐 ops-transformer): 派发算子子目录(op_host 必选, tests 等按需).
# 算子接入契约: gym_op_discovery 锚定本文件, 缺失即 FATAL.

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(NOT ENABLE_TESTS_EXECUTE)
    list(REMOVE_ITEM CURRENT_DIRS tests)
endif()
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
