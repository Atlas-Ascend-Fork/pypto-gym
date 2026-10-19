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
Fused SwiGLU Grad Operator Implementation

This module implements the backward kernels for the Fused SwiGLU operator.
Features:
- Dynamic shape support for batch dimension
- Bias fusion via addition
"""
import os
import pypto


@pypto.frontend.jit(
    pass_options={
        "vec_nbuffer_setting": {-1: 2, 0: 4}
    },
    runtime_options={
        "stitch_function_max_num": 128,
        "device_sched_mode": 3
    }
)
def fused_swiglu_bwd_b_kernel(
    dy: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    g: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    fc: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    dg: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    dfc: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    db_g: pypto.Tensor([], pypto.DT_BF16),
    db_fc: pypto.Tensor([], pypto.DT_BF16)
):
    """
    Fused SwiGLU backward kernel - Compute gradients for dg and dfc,
    and accumulate db_g and db_fc.
    """
    pypto.experimental.set_operation_options(combine_axis=True)
    m = dy.shape[0]
    n = dy.shape[1]
    tile_m = 1024
    loop_count = (m + tile_m - 1) // tile_m

    for idx in pypto.loop(loop_count, name="LOOP_BWD_DG", idx_name="idx"):
        tile_offset = idx * tile_m
        valid_m = (m - tile_offset).min(tile_m)
        dy_tile = pypto.view(dy, [tile_m, n], [tile_offset, 0], valid_shape=[valid_m, n])
        g_tile = pypto.view(g, [tile_m, n], [tile_offset, 0], valid_shape=[valid_m, n])
        fc_tile = pypto.view(fc, [tile_m, n], [tile_offset, 0], valid_shape=[valid_m, n])

        pypto.set_vec_tile_shapes(128, 128)
        dy_fp32 = pypto.cast(dy_tile, pypto.DT_FP32)
        g_fp32 = pypto.cast(g_tile, pypto.DT_FP32)
        fc_fp32 = pypto.cast(fc_tile, pypto.DT_FP32)
        exp_g = pypto.exp(g_fp32)
        sigmoid_g = pypto.div(exp_g, (1.0 + exp_g), precision_type=pypto.PrecisionType.INTRINSIC)
        silu_g = g_fp32 * sigmoid_g
        dy_mul_fc = dy_fp32 * fc_fp32
        silu_bwd = sigmoid_g * (1.0 + g_fp32 * (1.0 - sigmoid_g))
        dg_fp32 = dy_mul_fc * silu_bwd
        dfc_fp32 = dy_fp32 * silu_g
        dg_tile = pypto.cast(dg_fp32, pypto.DT_BF16)
        dfc_tile = pypto.cast(dfc_fp32, pypto.DT_BF16)
        dg[tile_offset:, 0:] = dg_tile
        dfc[tile_offset:, 0:] = dfc_tile

        dg_sum = pypto.sum(dg_fp32, dim=0, keepdim=True)
        dfc_sum = pypto.sum(dfc_fp32, dim=0, keepdim=True)
        db_g[:] = db_g + pypto.cast(dg_sum, pypto.DT_BF16)
        db_fc[:] = db_fc + pypto.cast(dfc_sum, pypto.DT_BF16)


@pypto.frontend.jit(
    pass_options={
        "vec_nbuffer_setting": {-1: 2, 0: 8},
        "cube_l1_reuse_setting": {-1: 2}
    },
    runtime_options={
        "stitch_function_max_num": 128,
        "device_sched_mode": 3
    }
)
def fused_swiglu_bwd_w_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    dg: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    dfc: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    dw_g: pypto.Tensor([], pypto.DT_BF16),
    dw_fc: pypto.Tensor([], pypto.DT_BF16)
):
    """
    Fused SwiGLU backward kernel - Compute weight gradients dw_g and dw_fc.
    """
    pypto.experimental.set_operation_options(combine_axis=True)
    m = x.shape[0]
    k = x.shape[1]
    n = dg.shape[1]
    tile_m = 2048
    loop_count = (m + tile_m - 1) // tile_m

    for idx in pypto.loop(loop_count, name="LOOP_BWD_DW", idx_name="idx"):
        tile_offset = idx * tile_m
        valid_m = (m - tile_offset).min(tile_m)
        x_tile = pypto.view(x, [tile_m, k], [tile_offset, 0], valid_shape=[valid_m, k])
        dg_tile = pypto.view(dg, [tile_m, n], [tile_offset, 0], valid_shape=[valid_m, n])
        dfc_tile = pypto.view(dfc, [tile_m, n], [tile_offset, 0], valid_shape=[valid_m, n])
        pypto.set_vec_tile_shapes(128, 128)
        x_fp32 = pypto.cast(x_tile, pypto.DT_FP32)
        dg_fp32 = pypto.cast(dg_tile, pypto.DT_FP32)
        dfc_fp32 = pypto.cast(dfc_tile, pypto.DT_FP32)
        pypto.set_cube_tile_shapes([128, 128], [128, 256], [128, 256])
        dw_g_fp32 = pypto.matmul(x_fp32, dg_fp32, pypto.DT_FP32, a_trans=True, b_trans=False)
        dw_fc_fp32 = pypto.matmul(x_fp32, dfc_fp32, pypto.DT_FP32, a_trans=True, b_trans=False)
        pypto.set_vec_tile_shapes(128, 128)
        dw_g[:] = dw_g + pypto.cast(dw_g_fp32, pypto.DT_BF16)
        dw_fc[:] = dw_fc + pypto.cast(dw_fc_fp32, pypto.DT_BF16)


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
def fused_swiglu_bwd_x_kernel(
    dg: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    dfc: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    w_g: pypto.Tensor([], pypto.DT_BF16),
    w_fc: pypto.Tensor([], pypto.DT_BF16),
    dx: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16)
):
    """
    Fused SwiGLU backward kernel - Compute input gradient dx.
    """
    pypto.experimental.set_operation_options(combine_axis=True)
    m = dg.shape[0]
    n = dg.shape[1]
    tile_m = 1024
    loop_count = (m + tile_m - 1) // tile_m

    for idx in pypto.loop(loop_count, name="LOOP_BWD_DX", idx_name="idx"):
        tile_offset = idx * tile_m
        valid_m = (m - tile_offset).min(tile_m)
        dg_tile = pypto.view(dg, [tile_m, n], [tile_offset, 0], valid_shape=[valid_m, n])
        dfc_tile = pypto.view(dfc, [tile_m, n], [tile_offset, 0], valid_shape=[valid_m, n])
        pypto.set_vec_tile_shapes(128, 128)
        dg_fp32 = pypto.cast(dg_tile, pypto.DT_FP32)
        dfc_fp32 = pypto.cast(dfc_tile, pypto.DT_FP32)
        w_g_fp32 = pypto.cast(w_g, pypto.DT_FP32)
        w_fc_fp32 = pypto.cast(w_fc, pypto.DT_FP32)
        pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256])
        dx_g_tile = pypto.matmul(dg_fp32, w_g_fp32, pypto.DT_FP32, a_trans=False, b_trans=True)
        dx_fc_tile = pypto.matmul(dfc_fp32, w_fc_fp32, pypto.DT_FP32, a_trans=False, b_trans=True)
        pypto.set_vec_tile_shapes(128, 128)
        dx_result = pypto.cast(dx_g_tile + dx_fc_tile, pypto.DT_BF16)
        dx[tile_offset:, 0:] = dx_result