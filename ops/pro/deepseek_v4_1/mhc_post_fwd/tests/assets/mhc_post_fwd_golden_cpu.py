#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""mhc_post_fwd 的独立 CPU golden 与输入构造器。

数学参考（与 SPEC 的 ``formula`` 逐字对应）：

.. code-block:: text

    out[n, mo, h] = post_layer_mix[n, mo] * x[n, h]
                    + sum_{mi=0}^{3} comb_res_mix[n, mi, mo] * residual[n, mi, h]

BF16 输入先提升到 FP32，乘加与 4 项求和全程 FP32，最后一次性舍入到 BF16。
``comb_res_mix`` 的两个下标是 ``(mi, mo)``，``mi`` 被求和、``mo`` 进入输出——写成 einsum 时
必然是 ``'nji,njh->nih'``（``j`` 求和）。轴序写反会得到一个"看起来合理但全错"的参考，
早期排查中正是这里耗掉了最多时间。
"""

import logging
import math

import torch

MHC = 4

logger = logging.getLogger(__name__)

# P0 = 本算子的代表测试矩阵（覆盖 n0 ∈ {1,2} 与四个常用 hidden）。
P0_CASES = [(n0, 4096, h) for n0 in (1, 2) for h in (1280, 2560, 5120, 7168)]


def mhc_post_fwd_golden_cpu(x, residual, post_layer_mix, comb_res_mix):
    """CPU 数学参考；返回 BF16 结果（CPU 上计算，与设备无关）。

    x               [n0,n1,h]        BF16
    residual        [n0,n1,mhc,h]    BF16
    post_layer_mix  [n0,n1,mhc,1]    FP32
    comb_res_mix    [n0,n1,mhc,mhc]  FP32
    """
    x = x.cpu()
    residual = residual.cpu()
    post_layer_mix = post_layer_mix.cpu()
    comb_res_mix = comb_res_mix.cpu()
    h = x.shape[-1]
    n = x.numel() // h
    term2 = torch.einsum('nji,njh->nih',
                         comb_res_mix.reshape(n, MHC, MHC),
                         residual.float().reshape(n, MHC, h))
    out = (x.float().reshape(n, h).unsqueeze(1)
           * post_layer_mix.reshape(n, MHC, 1) + term2).bfloat16()
    return out.reshape(*x.shape[:-1], MHC, h)


def make_inputs(n0, n1, h, seed=0, device="cpu"):
    """确定性输入构造器；``generator`` 固定，CPU 造数再搬到设备（设备上不造随机数）。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((n0, n1, h), dtype=torch.bfloat16, generator=g)
    residual = torch.randn((n0, n1, MHC, h), dtype=torch.bfloat16, generator=g)
    post_layer_mix = torch.randn((n0, n1, MHC, 1), dtype=torch.float32, generator=g)
    comb_res_mix = torch.randn((n0, n1, MHC, MHC), dtype=torch.float32, generator=g)
    if device is not None and str(device) != "cpu":
        x = x.to(device)
        residual = residual.to(device)
        post_layer_mix = post_layer_mix.to(device)
        comb_res_mix = comb_res_mix.to(device)
    return x, residual, post_layer_mix, comb_res_mix


def h_blk_reference(h: int) -> int:
    """hidden 分块：``h ≤ 4096`` 取整段，否则取一半并封顶 4096，再 64 对齐。"""
    blk = h if h <= 4096 else min(math.ceil(h / 2), 4096)
    return (blk + 63) // 64 * 64


def h_tiles_reference(h: int) -> int:
    blk = h_blk_reference(h)
    return (h + blk - 1) // blk


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    g = torch.Generator().manual_seed(0)
    args = torch.randn((1, 8, 128), dtype=torch.bfloat16, generator=g)
    res = torch.randn((1, 8, MHC, 128), dtype=torch.bfloat16, generator=g)
    post = torch.randn((1, 8, MHC, 1), dtype=torch.float32, generator=g)
    comb = torch.randn((1, 8, MHC, MHC), dtype=torch.float32, generator=g)
    out = mhc_post_fwd_golden_cpu(args, res, post, comb)
    logger.info("golden out: %s %s", tuple(out.shape), out.dtype)
    for h in (1280, 2560, 5120, 7168):
        logger.info("  h=%d: h_blk=%d -> %d tile(s)", h, h_blk_reference(h), h_tiles_reference(h))
