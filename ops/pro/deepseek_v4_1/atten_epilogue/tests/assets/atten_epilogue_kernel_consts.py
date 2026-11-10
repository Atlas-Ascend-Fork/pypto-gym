#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Kernel 常量(测试共享): 从 op_kernel/atten_epilogue.py 提取, 两种模式(jit/aclnn)均可 import.

用例只依赖本模块, 不 import kernel 模块本身——aclnn 模式下 kernel 源码不参与运行,
常量单独落地才能保证同一组用例在两种模式下的期望值一致.
"""

HEADS = 64
HEAD_DIM = 512
GROUPS = 8
GROUP_N = 1024
GEMM1_K = 4096
OUT_FEATURES = 5120
GEMM2_K = 8192
PAD_MULTIPLE = 16
MX_BLOCK = 32

# 扁平 GM workspace 布局(每 padded 行的字节数), 与 op_host tiling / kernel 的 pl.addptr 切分一致.
AQ_BYTES_PER_ROW = GROUPS * GEMM1_K              # 32768
ASC_BYTES_PER_ROW = GROUPS * GEMM1_K // 32       # 1024
YQ_BYTES_PER_ROW = GEMM2_K                       # 8192
YS_BYTES_PER_ROW = GEMM2_K // 32                 # 256
ASC_OFFSET_PER_ROW = AQ_BYTES_PER_ROW
YQ_OFFSET_PER_ROW = ASC_OFFSET_PER_ROW + ASC_BYTES_PER_ROW
YS_OFFSET_PER_ROW = YQ_OFFSET_PER_ROW + YQ_BYTES_PER_ROW
WORKSPACE_BYTES_PER_ROW = YS_OFFSET_PER_ROW + YS_BYTES_PER_ROW   # 42240

# 32 核、28 核和通用回退共用有限的19个合法 profile，T 本身仍为运行时数据。
WIDE_CUBE_CORES = 32
PROFILE_COUNT = 19


def padded_tokens(t):
    """T rounded up to the kernel's row granularity."""
    return (t + PAD_MULTIPLE - 1) // PAD_MULTIPLE * PAD_MULTIPLE


def workspace_bytes(t):
    """Flat GM workspace size in bytes the host tiling must report for `t` tokens."""
    return WORKSPACE_BYTES_PER_ROW * padded_tokens(t)
