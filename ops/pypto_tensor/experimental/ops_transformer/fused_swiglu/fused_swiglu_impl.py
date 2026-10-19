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
"""
Fused SwiGLU Operator Implementation

This module implements the forward kernels for the Fused SwiGLU operator.
Features:
- Dynamic shape support for batch dimension
- Bias fusion via addition
"""
import os
import pypto


@pypto.frontend.jit(
    pass_options={
        "vec_nbuffer_setting": {-1: 2, 0: 4},
        "cube_l1_reuse_setting": {-1: 2}
    },
    runtime_options={
        "stitch_function_max_num": 128,
        "device_sched_mode": 3
    }
)
def fused_swiglu_fwd_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    w_g: pypto.Tensor([], pypto.DT_BF16),
    w_fc: pypto.Tensor([], pypto.DT_BF16),
    b_g: pypto.Tensor([], pypto.DT_BF16),
    b_fc: pypto.Tensor([], pypto.DT_BF16),
    y: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16)
):
    """
    Fused SwiGLU forward kernel.
    Computes: y = SiLU(x @ w_g + b_g) * (x @ w_fc + b_fc)
    """
    pypto.experimental.set_operation_options(combine_axis=True)
    m = x.shape[0]
    k = x.shape[1]
    tile_m = 512
    loop_count = (m + tile_m - 1) // tile_m
    pypto.set_vec_tile_shapes(128, 128)
    b_g_fp32 = pypto.cast(b_g, pypto.DT_FP32)
    b_fc_fp32 = pypto.cast(b_fc, pypto.DT_FP32)

    for idx in pypto.loop(loop_count, name="LOOP_BWD_DW", idx_name="idx"):
        tile_offset = idx * tile_m
        valid_m = (m - tile_offset).min(tile_m)
        x_tile = pypto.view(x, [tile_m, k], [tile_offset, 0], valid_shape=[valid_m, k])
        pypto.set_cube_tile_shapes([128, 128], [128, 256], [128, 128])
        g_tile = pypto.matmul(x_tile, w_g, pypto.DT_FP32, extend_params={'bias_tensor': b_g_fp32})
        fc_tile = pypto.matmul(x_tile, w_fc, pypto.DT_FP32, extend_params={'bias_tensor': b_fc_fp32})
        pypto.set_vec_tile_shapes(128, 128)
        sigmoid_g = pypto.sigmoid(g_tile)
        silu_g = g_tile * sigmoid_g
        y_result = silu_g * fc_tile
        y_bf16 = pypto.cast(y_result, pypto.DT_BF16)
        y[tile_offset:, 0:] = y_bf16