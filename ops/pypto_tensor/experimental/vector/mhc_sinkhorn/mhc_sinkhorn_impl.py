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

"""PyPTO mhc_sinkhorn kernel implementation.

mhc_sinkhorn 实现 Sinkhorn-Knopp 双随机矩阵迭代归一化算法。
通过交替行列归一化迭代，将矩阵转换为双随机矩阵。
"""

import pypto
import torch


@pypto.frontend.jit(
    runtime_options={"device_sched_mode": 1},
    pass_options={"vec_nbuffer_setting": {-2: 1, -1: 2}}
)
def mhc_sinkhorn_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, 4, 4], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, 4, 4], pypto.DT_FP32),
    eps: float,
    num_iters: int,
):
    """Sinkhorn-Knopp 双随机矩阵迭代归一化 kernel。

    Args:
        x: 输入 tensor, shape [B*S, 8, 8], dtype float32。
           B*S 为动态轴。
        out: 输出 tensor, shape 与 x 相同。
        eps: 数值稳定性参数，float 标量。
        num_iters: 迭代次数，固定值 20。
    """
    pypto.experimental.set_operation_options(combine_axis=True)
    t = x.shape[0]
    hc = x.shape[1]
    unroll_list = [1, 8]

    x_flat = pypto.reshape(x, [t, hc * hc], inplace=True)
    s = 32

    for s_idx, unrollLength in pypto.loop_unroll(
    0, (t + s - 1) // s, 1, name="tLoop", idx_name="tIdx", unroll_list=unroll_list):
        tile_t = unrollLength * s
        t_idx = s_idx * s
        t_valid = (t - t_idx).min(tile_t)

        pypto.set_vec_tile_shapes(256, 16)
        comb_flag = pypto.view(x_flat, [tile_t, hc * hc], [t_idx, 0], valid_shape=[t_valid, hc * hc])
        comb_flag = pypto.transpose(comb_flag, 1, 0)
        comb_flag = pypto.reshape(comb_flag, [hc, hc, tile_t], inplace=True)

        pypto.set_vec_tile_shapes(4, 4, 256)
        row_max = pypto.amax(comb_flag, 1, True)
        comb_flag = pypto.exp(comb_flag - row_max)

        row_sum = pypto.sum(comb_flag, 1, True)
        comb_flag = comb_flag / (row_sum + eps)
        comb_flag = comb_flag + eps

        col_sum = pypto.sum(comb_flag, 0, True)
        comb_flag = comb_flag / (col_sum + eps)

        for _ in range(num_iters - 1):
            row_sum = comb_flag.sum(1, keepdim=True)
            comb_flag = comb_flag / (row_sum + eps)
            col_sum = comb_flag.sum(0, keepdim=True)
            comb_flag = comb_flag / (col_sum + eps)

        comb_flag = pypto.reshape(comb_flag, [hc * hc, tile_t])
        pypto.set_vec_tile_shapes(16, 256)
        comb_flag = pypto.transpose(comb_flag, 1, 0)
        pypto.set_vec_tile_shapes(256, 16)

        comb_flag = pypto.reshape(comb_flag, [tile_t, hc, hc])
        out[t_idx:, :, :] = comb_flag


def mhc_sinkhorn_wrapper(
    x: torch.Tensor,
    eps: float = 1e-6,
    num_iters: int = 20
) -> torch.Tensor:
    """算子 wrapper，供 test_mhc_sinkhorn.py 调用。

    Args:
        x: 输入 torch.Tensor, shape [B*S, 8, 8], dtype float32。
        eps: 数值稳定性参数，默认 1e-6。
        num_iters: 迭代次数，默认 20。

    Returns:
        输出 torch.Tensor, shape 与输入相同。
    """
    output = torch.empty_like(x)
    mhc_sinkhorn_kernel(x, output, eps, num_iters)
    return output