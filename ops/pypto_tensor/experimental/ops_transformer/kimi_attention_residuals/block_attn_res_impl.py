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
Block Attention Residuals Implementation Module

This module implements the core computation functions for Block Attention Residuals.
It provides both forward and backward passes, with optional RMSNorm support and cache outputs.

Main Functions:
    - ai_infra_block_attn_res: Forward pass for block attention residuals
    - ai_infra_block_attn_res_backward: Backward pass for block attention residuals

Kernels:
    - ai_infra_block_attn_res_kernel: 前向 kernel
    - ai_infra_block_attn_res_backward_kernel: 反向 kernel
"""

from typing import List, Optional

import torch

import pypto


def ai_infra_block_attn_res_forward_kernel(
    v_flat, proj_weight, gamma_fp32, h_out, rms_cache, alpha_cache_3d, scale, rms_norm_eps, enable_rmsnorm, l_max
):
    """PyPTO function ai_infra_block_attn_res_forward_kernel

    v_flat:         [B*T, L, D]    bf16/fp16
    proj_weight:    [1, 1, D]      fp32
    gamma_fp32:     [1, 1, D]      fp32
    h_out:          [B*T, 1, D]    bf16/fp16
    rms_cache:      [B*T, L, 1]    fp32
    alpha_cache_3d: [B*T, L]       fp32
    scale:          softmax 缩放因子
    rms_norm_eps:   RMSNorm epsilon
    enable_rmsnorm: 是否启用 RMSNorm
    l_max:          L分档大小
    """
    bt, l, d = v_flat.shape  # noqa: E741
    dtype = v_flat.dtype
    unroll_list = [1, 64, 128]
    norm_m_tile = 64 * 1024 // (4 * d)
    norm_m_tile = max(norm_m_tile, 1)

    if d == 512:
        pypto.set_pass_options(
            vec_nbuffer_setting={"DEFAULT": 8},
            cube_nbuffer_setting={"DEFAULT": 8, "func8_0": 16},
            cube_l1_reuse_setting={"DEFAULT": 2},
        )
    elif d == 1536:
        pypto.set_pass_options(
            vec_nbuffer_setting={"DEFAULT": 8, "func8_2": 48},
            cube_nbuffer_setting={"DEFAULT": 16},
            cube_l1_reuse_setting={"DEFAULT": 2},
        )

    pypto.experimental.set_operation_options(combine_axis=True)

    for bt_idx, unroll_length in pypto.loop_unroll(
        0,
        bt,
        1,
        name="Block_Attn_Res_T_Loop",
        idx_name="bt_idx",
        unroll_list=unroll_list,
    ):
        tile_bt = unroll_length

        v = pypto.view(v_flat, [tile_bt, l_max, d], [bt_idx, 0, 0], valid_shape=[tile_bt, l, d])

        if enable_rmsnorm:
            pypto.set_semantic_label("RmsNorm")
            pypto.set_vec_tile_shapes(1, norm_m_tile, d)
            v_fp32 = pypto.cast(v, pypto.DT_FP32)
            v_sq = pypto.mul(v_fp32, v_fp32)
            v_sq_sum = pypto.sum(v_sq, dim=-1, keepdim=True)
            mean_val = pypto.div(v_sq_sum, d, pypto.PrecisionType.INTRINSIC)
            rms_val = pypto.sqrt(pypto.add(mean_val, rms_norm_eps))
            k_fp32 = pypto.div(v_fp32, rms_val, pypto.PrecisionType.INTRINSIC)
            k_fp32 = pypto.mul(k_fp32, gamma_fp32)

            pypto.assemble(rms_val, [bt_idx, 0, 0], rms_cache)

        pypto.set_semantic_label("Attention Scores")
        if enable_rmsnorm:
            matmul_lhs = k_fp32
        else:
            pypto.set_vec_tile_shapes(1, norm_m_tile, d)
            matmul_lhs = pypto.cast(v, pypto.DT_FP32)
        pypto.set_vec_tile_shapes(1, 1, d)

        if l_max == 32:
            pypto.set_cube_tile_shapes([l_max, l_max], [512, 512], [16, 16])
        else:
            pypto.set_cube_tile_shapes([l_max, l_max], [128, 512], [16, 16])
        logits_3d = pypto.matmul(proj_weight, matmul_lhs, pypto.DT_FP32, a_trans=False, b_trans=True)
        logits = pypto.reshape(logits_3d, [tile_bt, l_max], valid_shape=[tile_bt, l])

        pypto.set_semantic_label("Attention Softmax")
        pypto.set_vec_tile_shapes(128, l_max)
        if scale != 1.0:
            logits = pypto.mul(logits, scale)
        alpha = pypto.softmax(logits, dim=-1)

        pypto.assemble(alpha, [bt_idx, 0], alpha_cache_3d)

        alpha_16 = pypto.cast(alpha, dtype)
        alpha_3d = pypto.reshape(alpha_16, [tile_bt, 1, l_max], valid_shape=[tile_bt, 1, l])

        pypto.set_semantic_label("Weighted Summation")
        if l_max == 32:
            pypto.set_cube_tile_shapes([1, 1], [l_max, l_max], [256, 256])
        else:
            pypto.set_cube_tile_shapes([1, 1], [128, 128], [128, 128])
        h = pypto.matmul(alpha_3d, v, dtype)

        pypto.assemble(h, [bt_idx, 0, 0], h_out)


@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 64, "device_sched_mode": 0, "max_workspace_kb": 300000},
)
def ai_infra_block_attn_res_forward_kernel_l_max_32(
    v_flat: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    proj_weight: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    gamma_fp32: pypto.Tensor([pypto.STATIC, pypto.STATIC]),
    h_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    rms_cache: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    alpha_cache_3d: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    scale: float,
    rms_norm_eps: float,
    enable_rmsnorm: bool,
):
    ai_infra_block_attn_res_forward_kernel(
        v_flat, proj_weight, gamma_fp32, h_out, rms_cache, alpha_cache_3d, scale, rms_norm_eps, enable_rmsnorm, 32
    )


@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 64, "device_sched_mode": 0},
)
def ai_infra_block_attn_res_forward_kernel_l_max_64(
    v_flat: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    proj_weight: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    gamma_fp32: pypto.Tensor([pypto.STATIC, pypto.STATIC]),
    h_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    rms_cache: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    alpha_cache_3d: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    scale: float,
    rms_norm_eps: float,
    enable_rmsnorm: bool,
):
    ai_infra_block_attn_res_forward_kernel(
        v_flat, proj_weight, gamma_fp32, h_out, rms_cache, alpha_cache_3d, scale, rms_norm_eps, enable_rmsnorm, 64
    )


@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 64, "device_sched_mode": 0},
)
def ai_infra_block_attn_res_forward_kernel_l_max_96(
    v_flat: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    proj_weight: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    gamma_fp32: pypto.Tensor([pypto.STATIC, pypto.STATIC]),
    h_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    rms_cache: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    alpha_cache_3d: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    scale: float,
    rms_norm_eps: float,
    enable_rmsnorm: bool,
):
    ai_infra_block_attn_res_forward_kernel(
        v_flat, proj_weight, gamma_fp32, h_out, rms_cache, alpha_cache_3d, scale, rms_norm_eps, enable_rmsnorm, 96
    )


@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 64, "device_sched_mode": 0},
)
def ai_infra_block_attn_res_forward_kernel_l_max_128(
    v_flat: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    proj_weight: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    gamma_fp32: pypto.Tensor([pypto.STATIC, pypto.STATIC]),
    h_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    rms_cache: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    alpha_cache_3d: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    scale: float,
    rms_norm_eps: float,
    enable_rmsnorm: bool,
):
    ai_infra_block_attn_res_forward_kernel(
        v_flat, proj_weight, gamma_fp32, h_out, rms_cache, alpha_cache_3d, scale, rms_norm_eps, enable_rmsnorm, 128
    )


def _validate_forward_inputs(
    blocks: List[torch.Tensor],
    proj_weight: torch.Tensor,
    partial_block: Optional[torch.Tensor],
    rmsnorm_gamma: Optional[torch.Tensor],
    enable_rmsnorm: bool,
    rms_out_flag: bool,
):
    if not blocks:
        raise ValueError("blocks must not be empty")
    n = len(blocks)
    if n < 1 or n > 127:
        raise ValueError(f"N (number of blocks) must be in [1, 127], got {n}")

    b, t, d = blocks[0].shape
    dtype = blocks[0].dtype

    for i, block in enumerate(blocks):
        if block.shape != (b, t, d):
            raise ValueError(f"blocks[{i}] shape {block.shape} != expected {(b, t, d)}")
        if block.dtype != dtype:
            raise ValueError(f"blocks[{i}] dtype {block.dtype} != expected {dtype}")

    if partial_block is not None:
        if partial_block.shape != (b, t, d):
            raise ValueError(f"partial_block shape {partial_block.shape} != expected {(b, t, d)}")
        if partial_block.dtype != dtype:
            raise ValueError(f"partial_block dtype {partial_block.dtype} != expected {dtype}")

    if proj_weight.shape != (d,):
        raise ValueError(f"proj_weight shape must be [D], got {proj_weight.shape}")
    if proj_weight.dtype != dtype:
        raise ValueError(f"proj_weight dtype {proj_weight.dtype} != expected {dtype}")

    if enable_rmsnorm and rmsnorm_gamma is None:
        raise ValueError("rmsnorm_gamma is required when enable_rmsnorm=True")

    if rmsnorm_gamma is not None:
        if rmsnorm_gamma.shape != (d,):
            raise ValueError(f"rmsnorm_gamma shape must be [D], got {rmsnorm_gamma.shape}")
        if rmsnorm_gamma.dtype != dtype:
            raise ValueError(f"rmsnorm_gamma dtype {rmsnorm_gamma.dtype} != expected {dtype}")

    if rms_out_flag and not enable_rmsnorm:
        raise ValueError("rms_out_flag=True is invalid when enable_rmsnorm=False (no RMSNorm output to cache)")


def ai_infra_block_attn_res(
    blocks: List[torch.Tensor],
    proj_weight: torch.Tensor,
    partial_block: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    rmsnorm_eps: float = 1e-6,
    rmsnorm_gamma: Optional[torch.Tensor] = None,
    enable_rmsnorm: bool = True,
    rms_out_flag: bool = False,
    alpha_out_flag: bool = False,
):
    """Block Attention Residuals 前向算子调用入口。

    Args:
        blocks: N 个已完成 block 表示, 每个 [B, T, D]
        proj_weight: 伪查询投影权重 [D]
        partial_block: 当前 block 部分和 [B, T, D]
        scale: softmax 缩放因子
        rmsnorm_eps: RMSNorm epsilon
        rmsnorm_gamma: RMSNorm 缩放参数 [D]
        enable_rmsnorm: 是否启用 RMSNorm
        rms_out_flag: 是否输出 rms_cache 供反向计算复用, 默认 False
        alpha_out_flag: 是否输出 alpha_cache 供反向计算复用, 默认 False

    Returns:
        - 仅 output [B,T,D] (两个 flag 均为 False)
        - (output, rms_cache [B,T,L,1]) (仅 rms_out_flag=True)
        - (output, alpha_cache [B,T,L]) (仅 alpha_out_flag=True)
        - (output, rms_cache, alpha_cache) (两个 flag 均为 True)
    """
    _validate_forward_inputs(blocks, proj_weight, partial_block, rmsnorm_gamma, enable_rmsnorm, rms_out_flag)
    n = len(blocks)
    l = n + 1 if partial_block is not None else n  # noqa: E741
    b, t, d = blocks[0].shape
    dtype = blocks[0].dtype
    device = blocks[0].device

    need_cache = rms_out_flag or alpha_out_flag

    if partial_block is not None:
        v = torch.stack(blocks + [partial_block], dim=2)
    else:
        v = torch.stack(blocks, dim=2)

    v = v.reshape(b * t, l, d)
    q = proj_weight.reshape(1, 1, d).to(torch.float32)
    gamma_fp32 = (
        rmsnorm_gamma.reshape(1, 1, d) if rmsnorm_gamma is not None else torch.ones(1, 1, d, dtype=dtype, device=device)
    )
    gamma_fp32 = gamma_fp32.to(torch.float32)

    h_out = torch.zeros(b * t, 1, d, dtype=dtype, device=device)

    rms_cache_flat = torch.zeros(b * t, l, 1, dtype=torch.float32, device=device)
    alpha_cache_flat = torch.zeros(b * t, l, dtype=torch.float32, device=device)
    if l <= 32:
        ai_infra_block_attn_res_forward_kernel_l_max_32(
            v, q, gamma_fp32, h_out, rms_cache_flat, alpha_cache_flat, scale, rmsnorm_eps, enable_rmsnorm
        )
    elif l <= 64:
        ai_infra_block_attn_res_forward_kernel_l_max_64(
            v, q, gamma_fp32, h_out, rms_cache_flat, alpha_cache_flat, scale, rmsnorm_eps, enable_rmsnorm
        )
    elif l <= 96:
        ai_infra_block_attn_res_forward_kernel_l_max_96(
            v, q, gamma_fp32, h_out, rms_cache_flat, alpha_cache_flat, scale, rmsnorm_eps, enable_rmsnorm
        )
    elif l <= 128:
        ai_infra_block_attn_res_forward_kernel_l_max_128(
            v, q, gamma_fp32, h_out, rms_cache_flat, alpha_cache_flat, scale, rmsnorm_eps, enable_rmsnorm
        )
    else:
        raise ValueError(f"Unsupported l={l}, expected l <= 128")

    output = h_out.reshape(b, t, d)

    if not need_cache:
        return output

    results = [output]
    if rms_out_flag:
        results.append(rms_cache_flat.reshape(b, t, l, 1))
    if alpha_out_flag:
        results.append(alpha_cache_flat.reshape(b, t, l))
    return tuple(results)


@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 32, "device_sched_mode": 1})
def ai_infra_block_attn_res_backward_kernel_l_max_32(
    v_flat: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    grad_h_3d: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    rms_cache: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    alpha_cache_3d: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.DYNAMIC]),
    proj_weight: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    gamma_fp32: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    grad_v_flat: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    grad_proj_weight: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    grad_gamma: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    scale: float,
    enable_rmsnorm: bool,
):
    l_max = 32
    bt, l, d = v_flat.shape  # noqa: E741
    if d == 512:
        configs = {
            "vec_nbuffer_setting": {"DEFAULT": 16, "func7_7": 2, "func7_1": 8, "func7_0": 32, "func7_8": 32},
            "cube_nbuffer_setting": {"DEFAULT": 16},
        }
    elif d == 1536:
        configs = {
            "vec_nbuffer_setting": {"DEFAULT": 16},
            "cube_nbuffer_setting": {"DEFAULT": 16},
        }
    ai_infra_block_attn_res_backward_kernel(
        v_flat,
        grad_h_3d,
        rms_cache,
        alpha_cache_3d,
        proj_weight,
        gamma_fp32,
        grad_v_flat,
        grad_proj_weight,
        grad_gamma,
        scale,
        enable_rmsnorm,
        l_max,
        configs,
    )


@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 32, "device_sched_mode": 1})
def ai_infra_block_attn_res_backward_kernel_l_max_64(
    v_flat: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    grad_h_3d: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    rms_cache: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    alpha_cache_3d: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.DYNAMIC]),
    proj_weight: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    gamma_fp32: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    grad_v_flat: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    grad_proj_weight: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    grad_gamma: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    scale: float,
    enable_rmsnorm: bool,
):
    l_max = 64
    bt, l, d = v_flat.shape  # noqa: E741
    if d == 512:
        configs = {"vec_nbuffer_setting": {"DEFAULT": 16}, "cube_nbuffer_setting": {"DEFAULT": 16}}
    elif d == 1536:
        configs = {
            "vec_nbuffer_setting": {"DEFAULT": 16},
            "cube_nbuffer_setting": {"DEFAULT": 16},
        }
    ai_infra_block_attn_res_backward_kernel(
        v_flat,
        grad_h_3d,
        rms_cache,
        alpha_cache_3d,
        proj_weight,
        gamma_fp32,
        grad_v_flat,
        grad_proj_weight,
        grad_gamma,
        scale,
        enable_rmsnorm,
        l_max,
        configs,
    )


@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 32, "device_sched_mode": 1})
def ai_infra_block_attn_res_backward_kernel_l_max_96(
    v_flat: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    grad_h_3d: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    rms_cache: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    alpha_cache_3d: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.DYNAMIC]),
    proj_weight: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    gamma_fp32: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    grad_v_flat: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    grad_proj_weight: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    grad_gamma: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    scale: float,
    enable_rmsnorm: bool,
):
    l_max = 96
    bt, l, d = v_flat.shape  # noqa: E741
    if d == 512:
        configs = {"vec_nbuffer_setting": {"DEFAULT": 16}, "cube_nbuffer_setting": {"DEFAULT": 16}}
    elif d == 1536:
        configs = {
            "vec_nbuffer_setting": {"DEFAULT": 16},
            "cube_nbuffer_setting": {"DEFAULT": 16},
        }
    ai_infra_block_attn_res_backward_kernel(
        v_flat,
        grad_h_3d,
        rms_cache,
        alpha_cache_3d,
        proj_weight,
        gamma_fp32,
        grad_v_flat,
        grad_proj_weight,
        grad_gamma,
        scale,
        enable_rmsnorm,
        l_max,
        configs,
    )


@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 32, "device_sched_mode": 1})
def ai_infra_block_attn_res_backward_kernel_l_max_128(
    v_flat: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    grad_h_3d: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    rms_cache: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    alpha_cache_3d: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.DYNAMIC]),
    proj_weight: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    gamma_fp32: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    grad_v_flat: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    grad_proj_weight: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    grad_gamma: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    scale: float,
    enable_rmsnorm: bool,
):
    l_max = 128
    bt, l, d = v_flat.shape  # noqa: E741
    if d == 512:
        configs = {"vec_nbuffer_setting": {"DEFAULT": 16}, "cube_nbuffer_setting": {"DEFAULT": 16}}
    elif d == 1536:
        configs = {
            "vec_nbuffer_setting": {"DEFAULT": 16},
            "cube_nbuffer_setting": {"DEFAULT": 16},
        }
    ai_infra_block_attn_res_backward_kernel(
        v_flat,
        grad_h_3d,
        rms_cache,
        alpha_cache_3d,
        proj_weight,
        gamma_fp32,
        grad_v_flat,
        grad_proj_weight,
        grad_gamma,
        scale,
        enable_rmsnorm,
        l_max,
        configs,
    )


def ai_infra_block_attn_res_backward_kernel(
    v_flat,
    grad_h_3d,
    rms_cache,
    alpha_cache_3d,
    proj_weight,
    gamma_fp32,
    grad_v_flat,
    grad_proj_weight,
    grad_gamma,
    scale,
    enable_rmsnorm,
    l_max,
    configs,
):
    """PyPTO jit ai_infra_block_attn_res_backward_kernel

    v_flat:            [B*T, L, D]    bf16/fp16
    grad_h_3d:         [B*T, 1, D]    bf16/fp16
    rms_cache:         [B*T, L, 1]    fp32 (enable_rmsnorm=False 时为空)
    alpha_cache_3d:    [B*T, 1, L]    fp32
    proj_weight:       [1, 1, D]      bf16/fp16
    gamma_fp32:        [1, 1, D]      fp32
    grad_v_flat:       [B*T, L, D]    bf16/fp16
    grad_proj_weight:  [1, 1, D]      bf16/fp16
    grad_gamma:        [1, 1, D]      bf16/fp16
    scale:          softmax 缩放因子
    enable_rmsnorm: 是否启用 RMSNorm
    l_max:          L分档大小
    """
    bt, l, d = v_flat.shape  # noqa: E741
    dtype = v_flat.dtype
    unroll_list = [1, 64]
    l_tile = max(16384 // d, 1)
    l_tile = max((l_tile // 4) * 4, 4)
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_pass_options(
        vec_nbuffer_setting=configs["vec_nbuffer_setting"],
        cube_nbuffer_setting=configs["cube_nbuffer_setting"],
    )

    pypto.set_vec_tile_shapes(1, 1, d)
    grad_proj_weight_acc = pypto.zeros([1, 1, d], dtype=pypto.DT_FP32)
    if enable_rmsnorm:
        grad_gamma_acc = pypto.zeros([1, 1, d], dtype=pypto.DT_FP32)

    for bt_idx, tile_bt in pypto.loop_unroll(0, bt, 1, name="BT_LOOP", idx="bt_idx", unroll_list=unroll_list):
        pypto.set_semantic_label("Weighted Summation Backward")
        pypto.set_vec_tile_shapes(1, l_tile, d)
        grad_h_tile_3d = pypto.view(grad_h_3d, [tile_bt, 1, d], [bt_idx, 0, 0])

        v_tile = pypto.view(v_flat, [tile_bt, l_max, d], [bt_idx, 0, 0], valid_shape=[tile_bt, l, d])
        pypto.set_cube_tile_shapes([1, 1], [32768 // l_max, 32768 // l_max], [l_max, l_max])
        grad_alpha_fp32 = pypto.matmul(grad_h_tile_3d, v_tile, pypto.DT_FP32, b_trans=True)

        alpha_tile_3d = pypto.view(alpha_cache_3d, [tile_bt, 1, l_max], [bt_idx, 0, 0], valid_shape=[tile_bt, 1, l])
        pypto.set_cube_tile_shapes([l_max, l_max], [16, 16], [16384 // l_max, 16384 // l_max])
        pypto.set_vec_tile_shapes(l_tile, 1, d)
        grad_h_tile_3d_fp32 = pypto.cast(grad_h_tile_3d, pypto.DT_FP32)
        grad_v_agg_fp32 = pypto.matmul(alpha_tile_3d, grad_h_tile_3d_fp32, pypto.DT_FP32, a_trans=True)

        pypto.set_semantic_label("Attention Softmax Backward")
        pypto.set_vec_tile_shapes(128, 1, 128)
        grad_alpha_2d_fp32 = pypto.reshape(grad_alpha_fp32, [tile_bt, l_max], valid_shape=[tile_bt, l])
        alpha_2d = pypto.reshape(alpha_tile_3d, [tile_bt, l_max], valid_shape=[tile_bt, l])
        pypto.set_vec_tile_shapes(128, 128)
        alpha_fp32_2d = pypto.cast(alpha_2d, pypto.DT_FP32)

        dot_fp32 = pypto.sum(grad_alpha_2d_fp32 * alpha_fp32_2d, dim=-1, keepdim=True)

        grad_logits_fp32 = alpha_fp32_2d * (grad_alpha_2d_fp32 - dot_fp32)
        grad_logits_reshaped = pypto.reshape(grad_logits_fp32, [tile_bt, 1, l_max], valid_shape=[tile_bt, 1, l])

        pypto.set_semantic_label("Attention Scores Backward")
        pypto.set_vec_tile_shapes(1, l_tile, d)
        v_tile_fp32 = pypto.cast(v_tile, pypto.DT_FP32)
        proj_weight_fp32 = pypto.cast(proj_weight, pypto.DT_FP32)
        proj_weight_scale = scale * proj_weight_fp32
        proj_weight_scale_fp32 = proj_weight_scale
        if enable_rmsnorm:
            pypto.set_vec_tile_shapes(1, l_tile, d)
            rms_tile = pypto.view(rms_cache, [tile_bt, l_max, 1], [bt_idx, 0, 0], valid_shape=[tile_bt, l, 1])
            rms_tile_fp32 = pypto.cast(rms_tile, pypto.DT_FP32)
            k_tile_fp32 = v_tile_fp32 / rms_tile_fp32 * gamma_fp32
        else:
            k_tile_fp32 = v_tile_fp32

        pypto.set_cube_tile_shapes([l_max, l_max], [16, 16], [16384 // l_max, 16384 // l_max])
        grad_k_fp32 = pypto.matmul(grad_logits_reshaped, proj_weight_scale_fp32, pypto.DT_FP32, a_trans=True)

        pypto.set_vec_tile_shapes(1, l_tile, d)
        grad_logits_reshaped = pypto.reshape(grad_logits_fp32 + 0.0, [tile_bt, l_max, 1], valid_shape=[tile_bt, l, 1])
        grad_proj_weight_tile_fp32 = pypto.mul(grad_logits_reshaped, k_tile_fp32)
        pypto.set_vec_tile_shapes(tile_bt, 1, 16384 // tile_bt)
        grad_proj_weight_tile_fp32 = pypto.sum(grad_proj_weight_tile_fp32, dim=0, keepdim=True)
        pypto.set_pass_options(sg_set_scope=1)
        pypto.set_vec_tile_shapes(1, l_max, 16384 // l_max)
        grad_proj_weight_tile_fp32 = pypto.sum(grad_proj_weight_tile_fp32, dim=1, keepdim=True)
        pypto.set_pass_options(sg_set_scope=-1)

        pypto.set_vec_tile_shapes(1, 1, d)
        grad_proj_weight_scaled_fp32 = grad_proj_weight_tile_fp32 * scale
        grad_proj_weight_acc[:] = grad_proj_weight_acc + grad_proj_weight_scaled_fp32

        if enable_rmsnorm:
            pypto.set_semantic_label("RmsNorm Backward")
            pypto.set_vec_tile_shapes(1, l_tile, d)
            c_fp32 = pypto.sum(gamma_fp32 * grad_k_fp32 * v_tile_fp32, dim=-1, keepdim=True)

            rms_sq_fp32 = rms_tile_fp32 * rms_tile_fp32
            grad_v_rms_fp32 = (gamma_fp32 * grad_k_fp32 - v_tile_fp32 * c_fp32 / (d * rms_sq_fp32)) / rms_tile_fp32

            grad_gamma_tile_fp32 = grad_k_fp32 * (v_tile_fp32 / rms_tile_fp32)
            pypto.set_vec_tile_shapes(tile_bt, 1, 16384 // tile_bt)
            grad_gamma_tile_fp32 = pypto.sum(grad_gamma_tile_fp32, dim=0, keepdim=True)
            pypto.set_pass_options(sg_set_scope=1)
            pypto.set_vec_tile_shapes(1, l_max, 16384 // l_max)
            grad_gamma_tile_fp32 = pypto.sum(grad_gamma_tile_fp32, dim=1, keepdim=True)
            pypto.set_pass_options(sg_set_scope=-1)

            pypto.set_vec_tile_shapes(1, 1, d)
            grad_gamma_acc[:] = grad_gamma_acc + grad_gamma_tile_fp32
        else:
            grad_v_rms_fp32 = grad_k_fp32

        pypto.set_vec_tile_shapes(1, l_tile, d)
        grad_v_fp32 = grad_v_agg_fp32 + grad_v_rms_fp32
        del grad_v_agg_fp32, grad_alpha_fp32, grad_k_fp32
        grad_v_tile = pypto.cast(grad_v_fp32, dtype)
        pypto.assemble(grad_v_tile, [bt_idx, 0, 0], grad_v_flat)

    grad_proj_weight[:] = pypto.cast(grad_proj_weight_acc, dtype)
    if enable_rmsnorm:
        grad_gamma[:] = pypto.cast(grad_gamma_acc, dtype)


def _validate_backward_inputs(
    grad_h: torch.Tensor,
    blocks: List[torch.Tensor],
    proj_weight: torch.Tensor,
    alpha_cache: torch.Tensor,
    partial_block: Optional[torch.Tensor],
    rmsnorm_gamma: Optional[torch.Tensor],
    rms_cache: Optional[torch.Tensor],
    enable_rmsnorm: bool,
):
    n = len(blocks)
    if n == 0:
        raise ValueError("blocks must not be empty")

    has_partial = partial_block is not None
    b, t, d = blocks[0].shape
    l = n + 1 if has_partial else n  # noqa: E741
    dtype = blocks[0].dtype
    device = blocks[0].device

    if grad_h.shape != (b, t, d):
        raise ValueError(f"grad_h shape must be {(b, t, d)}, but got {tuple(grad_h.shape)}")
    if grad_h.dtype != dtype or grad_h.device != device:
        raise ValueError(f"grad_h dtype/device must be {dtype}/{device}, but got {grad_h.dtype}/{grad_h.device}")

    for i, block in enumerate(blocks):
        if block.shape != (b, t, d):
            raise ValueError(f"blocks[{i}] shape must be {(b, t, d)}, but got {tuple(block.shape)}")
        if block.dtype != dtype or block.device != device:
            raise ValueError(f"blocks[{i}] dtype/device must be {dtype}/{device}, but got {block.dtype}/{block.device}")

    if has_partial:
        if partial_block.shape != (b, t, d):
            raise ValueError(f"partial_block shape must be {(b, t, d)}, but got {tuple(partial_block.shape)}")
        if partial_block.dtype != dtype or partial_block.device != device:
            raise ValueError(
                f"partial_block dtype/device must be {dtype}/{device}, "
                f"but got {partial_block.dtype}/{partial_block.device}"
            )

    if proj_weight.shape != (d,):
        raise ValueError(f"proj_weight shape must be {(d,)}, but got {tuple(proj_weight.shape)}")
    if proj_weight.dtype != dtype or proj_weight.device != device:
        raise ValueError(
            f"proj_weight dtype/device must be {dtype}/{device}, but got {proj_weight.dtype}/{proj_weight.device}"
        )

    if alpha_cache is None:
        raise ValueError("alpha_cache must be provided and cannot be None")
    if alpha_cache.shape != (b, t, l):
        raise ValueError(f"alpha_cache shape must be {(b, t, l)}, but got {tuple(alpha_cache.shape)}")

    if enable_rmsnorm:
        if rmsnorm_gamma is None:
            raise ValueError("when enable_rmsnorm is True, rmsnorm_gamma must be provided")
        if rmsnorm_gamma.shape != (d,):
            raise ValueError(f"rmsnorm_gamma shape must be {(d,)}, but got {tuple(rmsnorm_gamma.shape)}")
        if rmsnorm_gamma.dtype != dtype or rmsnorm_gamma.device != device:
            raise ValueError(
                f"rmsnorm_gamma dtype/device must be {dtype}/{device}, "
                f"but got {rmsnorm_gamma.dtype}/{rmsnorm_gamma.device}"
            )

        if rms_cache is None:
            raise ValueError("when enable_rmsnorm is True, rms_cache must be provided")
        if rms_cache.shape != (b, t, l, 1):
            raise ValueError(f"rms_cache shape must be {(b, t, l, 1)}, but got {tuple(rms_cache.shape)}")


def ai_infra_block_attn_res_backward(
    grad_h: torch.Tensor,
    blocks: List[torch.Tensor],
    proj_weight: torch.Tensor,
    alpha_cache: torch.Tensor,
    partial_block: Optional[torch.Tensor] = None,
    rmsnorm_gamma: Optional[torch.Tensor] = None,
    rms_cache: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    enable_rmsnorm: bool = True,
):
    """Block Attention Residuals 反向算子调用入口。

    Args:
        grad_h: 上游传递的输出梯度 [B, T, D]
        blocks: 正向使用的已完成的 block 表示列表, 每个 tensor shape 均为 [B, T, D]
        proj_weight: 伪查询投影权重 [D]
        alpha_cache: 正向缓存的 softmax 权重 [B, T, L]
        partial_block: 正向使用的当前 block 部分和 [B, T, D]
        rmsnorm_gamma: RMSNorm 缩放参数 [D]
        rms_cache: 正向缓存的 RMSNorm 分母 [B, T, L, 1]
        scale: softmax 缩放因子, 默认值为 1.0
        enable_rmsnorm: 是否启用 RMSNorm, 默认为 True

    Returns:
        - (grad_blocks, grad_partial_block, grad_proj_weight, grad_rmsnorm_gamma)
        - (grad_blocks, grad_partial_block, grad_proj_weight, grad_rmsnorm_gamma,
                grad_rmsnorm_gamma) (enable_rmsnorm为True时)
    """
    _validate_backward_inputs(
        grad_h, blocks, proj_weight, alpha_cache, partial_block, rmsnorm_gamma, rms_cache, enable_rmsnorm
    )

    n = len(blocks)
    has_partial = partial_block is not None
    b, t, d = blocks[0].shape
    l = n + 1 if has_partial else n  # noqa: E741
    dtype = blocks[0].dtype
    device = blocks[0].device

    tensors = blocks + ([partial_block] if has_partial else [])
    v = torch.stack(tensors, dim=2)
    v_flat = v.reshape(b * t, l, d)

    if rmsnorm_gamma is not None:
        gamma_reshaped = rmsnorm_gamma.reshape(1, 1, d).to(torch.float32)
    else:
        gamma_reshaped = torch.ones(1, 1, d, dtype=dtype, device=device).to(torch.float32)

    proj_weight_reshaped = proj_weight.reshape(1, 1, d)

    grad_h_3d = grad_h.reshape(b * t, 1, d)

    alpha_cache_3d = alpha_cache.reshape(b * t, 1, l)

    if rms_cache is not None:
        rms_cache_flat = rms_cache.reshape(b * t, l, 1)
    else:
        rms_cache_flat = torch.empty(b * t, l, 1, dtype=dtype, device=device)

    grad_v_flat = torch.zeros(b * t, l, d, dtype=dtype, device=device)
    grad_proj_weight_out = torch.zeros(1, 1, d, dtype=dtype, device=device)
    grad_gamma_out = torch.zeros(1, 1, d, dtype=dtype, device=device)

    kernel_args = (
        v_flat,
        grad_h_3d,
        rms_cache_flat,
        alpha_cache_3d,
        proj_weight_reshaped,
        gamma_reshaped,
        grad_v_flat,
        grad_proj_weight_out,
        grad_gamma_out,
        scale,
        enable_rmsnorm,
    )

    if l <= 32:
        ai_infra_block_attn_res_backward_kernel_l_max_32(*kernel_args)
    elif l <= 64:
        ai_infra_block_attn_res_backward_kernel_l_max_64(*kernel_args)
    elif l <= 96:
        ai_infra_block_attn_res_backward_kernel_l_max_96(*kernel_args)
    elif l <= 128:
        ai_infra_block_attn_res_backward_kernel_l_max_128(*kernel_args)
    else:
        raise ValueError(f"Unsupported l={l}, expected l <= 128")

    grad_v = grad_v_flat.reshape(b, t, l, d)
    grad_blocks = [grad_v[:, :, i, :].contiguous() for i in range(n)]

    if has_partial:
        grad_partial_block = grad_v[:, :, n, :].contiguous()
    else:
        grad_partial_block = None

    grad_proj_weight_final = grad_proj_weight_out.reshape(d)

    if enable_rmsnorm:
        grad_rmsnorm_gamma_final = grad_gamma_out.reshape(d)
        return grad_blocks, grad_partial_block, grad_proj_weight_final, grad_rmsnorm_gamma_final

    return grad_blocks, grad_partial_block, grad_proj_weight_final
