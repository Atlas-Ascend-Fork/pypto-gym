# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# 算子 tests 构建入口(由算子级 dispatcher 在 ENABLE_TESTS 时派发进入).
# 目录约定:
#   aclnn_bridge/  aclnn 模式的 pybind11 桥(C++ 目标, 按需构建)
#   assets/        golden/参考实现与共享常量(Python, 不参与 CMake)
#   test_*.py      用例, 位于本目录(pytest 直接收集, 不参与 CMake)
# 后续如需新增测试侧 C++ 目标, 在此 add_subdirectory.

add_subdirectory(aclnn_bridge)
