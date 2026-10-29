#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Kernel 常量(测试共享): 从 op_kernel/flex_attention.py 提取, 两种模式(jit/aclnn)均可 import."""

QK_PRELOAD = 3
FIFO_SIZE = QK_PRELOAD + 1
TS = 128
TKV = 128
TD = 128
NEG_INF = -1e9
NEG_INF_TAIL = -1e10
CHUNK = 16
MASK_NUM_LIMIT = 256
