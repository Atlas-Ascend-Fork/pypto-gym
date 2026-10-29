# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# 测试侧 CMake 入口: 仅承载测试用 C++ 目标(pytest 用例不参与构建).
# 由根 CMakeLists 在 ENABLE_TESTS 时引入; 逐级 add_subdirectory 向下扩展.
# 聚合目标: 默认构建时编译所有测试侧桥(ENABLE_TESTS 已开启, 仅测试构建使用).
add_custom_target(gym_test_bridges ALL)
add_subdirectory(ops)
