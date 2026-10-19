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
QAT Implementation Module

This module implements the core computation functions for Quantization-Aware Training (QAT).
It provides both symmetric and asymmetric quantization operations with forward and backward passes.

Main Functions:
    - ai_infra_qat_asymmetric_per_group: Asymmetric per-group quantization
    - ai_infra_qat_asymmetric_per_group_backward: Backward pass for asymmetric per-group quantization
    - ai_infra_qat_symmetric_per_channel: Symmetric per-channel quantization
    - ai_infra_qat_symmetric_per_channel_backward: Backward pass for symmetric per-channel quantization
    - ai_infra_qat_symmetric_per_tensor: Symmetric per-tensor quantization
    - ai_infra_qat_symmetric_per_tensor_backward: Backward pass for symmetric per-tensor quantization

Example:
    See ai_infra_pypto_qat.py for usage examples.
"""

import pypto
import torch


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 64,
    },
    pass_options={"vec_nbuffer_setting": {-1: 2, -2: 1}},
)
def ai_infra_qat_asymmetric_per_group_kernel(
    weight: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    scale: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_BF16),
    offset: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_BF16),
    output_bf16: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    eps,
    n_levels,
    neg_clip_val,
    clip_val,
    shift
):
    pypto.experimental.set_operation_options(combine_axis=True)
    unroll_list = [512, 256]
    num_groups = scale.shape[0]
    group_size = weight.shape[1]
    pypto.set_vec_tile_shapes(128, 128)

    for g_offset, unroll_length in pypto.loop_unroll(
        0, num_groups, 1,
        name="LOOP_GROUPS",
        idx_name="g_offset",
        unroll_list=unroll_list
    ):
        tile_groups = unroll_length

        weight_tile = pypto.view(weight, [tile_groups, group_size], [g_offset, 0])
        weight_fp32 = pypto.cast(weight_tile, pypto.DT_FP32)

        scale_tile = pypto.view(scale, [tile_groups, 1], [g_offset, 0])
        offset_tile = pypto.view(offset, [tile_groups, 1], [g_offset, 0])

        scale_fp32 = pypto.cast(scale_tile, pypto.DT_FP32)
        offset_fp32 = pypto.cast(offset_tile, pypto.DT_FP32)

        protected_scale = pypto.maximum(scale_fp32, eps)
        alpha = pypto.mul(protected_scale, n_levels)

        weight_shifted = pypto.sub(weight_fp32, offset_fp32)
        weight_norm = pypto.div(weight_shifted, alpha, precision_type=pypto.PrecisionType.INTRINSIC)
        weight_clipped = pypto.clip(weight_norm, neg_clip_val, clip_val)

        weight_scaled = pypto.mul(weight_clipped, n_levels)
        weight_shifted2 = pypto.sub(weight_scaled, shift)
        weight_rounded = pypto.round(weight_shifted2, decimals=0)

        weight_unshifted = pypto.add(weight_rounded, shift)
        weight_denorm = pypto.div(weight_unshifted, n_levels, precision_type=pypto.PrecisionType.INTRINSIC)

        weight_rescaled = pypto.mul(weight_denorm, alpha)
        output = pypto.add(weight_rescaled, offset_fp32)

        output_tile = pypto.cast(output, pypto.DT_BF16)

        pypto.assemble(output_tile, [g_offset, 0], output_bf16)


def ai_infra_qat_asymmetric_per_group(weight, scale, offset, group_size=128, bit=4,
                                  eps=1e-4, clip_val=0.99):

    n_levels = 2 ** (bit - 1)
    shift = 0.5
    neg_clip_val = -clip_val

    output_bf16 = torch.empty(weight.shape, dtype=weight.dtype, device=weight.device)
    weight_grouped = weight.view(-1, group_size)
    output_bf16_grouped = output_bf16.view(-1, group_size)

    inputs = [
        weight_grouped,
        scale,
        offset,
        output_bf16_grouped,
        eps,
        n_levels,
        neg_clip_val,
        clip_val,
        shift
    ]
    ai_infra_qat_asymmetric_per_group_kernel(*inputs)
    return output_bf16_grouped.view(weight.shape)


@pypto.frontend.jit(
    pass_options={
        "vec_nbuffer_setting": {-1: 2, -2: 1}
        },
    runtime_options={
        "stitch_function_max_num": 64
        },
)
def ai_infra_qat_asymmetric_per_group_backward_kernel(
    grad_output: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    weight: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    scale: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_BF16),
    offset: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_BF16),
    grad_weight_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    grad_scale_out: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_BF16),
    grad_offset_out: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_BF16),
    eps,
    n_levels,
    neg_clip_val,
    clip_val,
    shift
):
    group_size = grad_output.shape[1]
    num_groups = scale.shape[0]
    unroll_list = [512, 256]

    pypto.experimental.set_operation_options(combine_axis=True)
    for g_offset, unroll_length in pypto.loop_unroll(
        0, num_groups, 1,
        name="LOOP_GROUPS",
        idx_name="g_offset",
        unroll_list=unroll_list
    ):
        tile_groups = unroll_length
        pypto.set_vec_tile_shapes(64, 128)
        # --- 1. 数据加载与类型转换 (Load & Cast) ---
        grad_out_tile = pypto.view(grad_output, [tile_groups, group_size], [g_offset, 0])
        weight_tile = pypto.view(weight, [tile_groups, group_size], [g_offset, 0])
        scale_tile = pypto.view(scale, [tile_groups, 1], [g_offset, 0])
        offset_tile = pypto.view(offset, [tile_groups, 1], [g_offset, 0])

        grad_out_fp32 = pypto.cast(grad_out_tile, pypto.DT_FP32)
        weight_fp32 = pypto.cast(weight_tile, pypto.DT_FP32)
        scale_fp32 = pypto.cast(scale_tile, pypto.DT_FP32)
        offset_fp32 = pypto.cast(offset_tile, pypto.DT_FP32)

        # --- 2. 前向状态重计算 (Recompute Intermediates) ---
        protected_scale = pypto.maximum(scale_fp32, eps)
        alpha = pypto.mul(protected_scale, n_levels)

        weight_shifted = pypto.sub(weight_fp32, offset_fp32)
        # 对应前向的 weight_scaled，未截断
        weight_norm = pypto.div(weight_shifted, alpha, precision_type=pypto.PrecisionType.INTRINSIC)

        # 重新计算 weight_denorm 用于求解 scale 梯度
        weight_clipped = pypto.clip(weight_norm, neg_clip_val, clip_val)
        weight_scaled = pypto.mul(weight_clipped, n_levels)
        weight_shifted2 = pypto.sub(weight_scaled, shift)
        weight_rounded = pypto.round(weight_shifted2, decimals=0)
        weight_unshifted = pypto.add(weight_rounded, shift)
        weight_denorm = pypto.div(weight_unshifted, n_levels, precision_type=pypto.PrecisionType.INTRINSIC)

        # --- 3. 掩码生成 (Mask Generation) ---
        # 判断元素是否在 clip 范围内 (-clip_val <= w <= clip_val)
        diff = pypto.sub(weight_norm, weight_clipped)
        abs_diff = pypto.abs(diff)
        big_number = 1e15
        sign = pypto.mul(abs_diff, big_number)
        is_out = pypto.clip(sign, 0.0, 1.0)
        one = pypto.full(is_out.shape, 1.0, is_out.dtype)
        mask_f32 = pypto.sub(one, is_out)

        one_tile_group_gs = pypto.full([tile_groups, group_size], 1.0, pypto.DT_FP32)
        inv_mask_f32 = pypto.sub(one_tile_group_gs, mask_f32)

        # 判断 scale 是否合法（> eps），用于过滤 scale 梯度
        scale_diff = pypto.sub(scale_fp32, eps)
        diff_pos = pypto.maximum(scale_diff, 0.0)
        amplified_diff = pypto.mul(diff_pos, 1e15)
        scale_mask_f32 = pypto.clip(amplified_diff, 0.0, 1.0)

        # --- 4. 梯度计算 (Gradient Computation) ---

        # Grad Weight: 只有在 clip 范围内的元素有梯度
        grad_weight_fp32 = pypto.mul(grad_out_fp32, mask_f32)

        # Grad Offset: 截断元素的梯度累加
        grad_offset_pre = pypto.mul(grad_out_fp32, inv_mask_f32)
        grad_offset_fp32 = pypto.sum(grad_offset_pre, dim=1, keepdim=True)

        # Grad Scale: grad_y * (weight_denorm - weight_norm * mask) * n_levels
        term_w_norm = pypto.mul(weight_norm, mask_f32)
        term_diff = pypto.sub(weight_denorm, term_w_norm)
        grad_alpha_pre = pypto.mul(grad_out_fp32, term_diff)
        grad_alpha_fp32 = pypto.sum(grad_alpha_pre, dim=1, keepdim=True)

        grad_scale_pre = pypto.mul(grad_alpha_fp32, n_levels)
        grad_scale_fp32 = pypto.mul(grad_scale_pre, scale_mask_f32)

        # --- 5. 数据流出 (Cast & Assemble) ---
        grad_w_bf16 = pypto.cast(grad_weight_fp32, pypto.DT_BF16)
        grad_s_bf16 = pypto.cast(grad_scale_fp32, pypto.DT_BF16)
        grad_o_bf16 = pypto.cast(grad_offset_fp32, pypto.DT_BF16)

        pypto.assemble(grad_w_bf16, [g_offset, 0], grad_weight_out)
        pypto.assemble(grad_s_bf16, [g_offset, 0], grad_scale_out)
        pypto.assemble(grad_o_bf16, [g_offset, 0], grad_offset_out)


def ai_infra_qat_asymmetric_per_group_backward(grad_output, weight_pto, scale_pto, offset_pto, 
                                            group_size=128, bit=4, eps=1e-4, clip_val=0.99):
    n_levels = 2 ** (bit - 1)
    shift = 0.5
    neg_clip_val = -clip_val

    grad_weight_out = torch.empty(weight_pto.shape, dtype=weight_pto.dtype, device=weight_pto.device)
    grad_scale_out = torch.empty(scale_pto.shape, dtype=scale_pto.dtype, device=scale_pto.device)
    grad_offset_out = torch.empty(offset_pto.shape, dtype=offset_pto.dtype, device=offset_pto.device)

    grad_output_grouped = grad_output.view(-1, group_size)
    weight_pto_grouped = weight_pto.view(-1, group_size)
    grad_weight_out_grouped = grad_weight_out.view(-1, group_size)

    inputs = [
        grad_output_grouped,
        weight_pto_grouped,
        scale_pto,
        offset_pto,
        grad_weight_out_grouped,
        grad_scale_out,
        grad_offset_out,
        eps,
        n_levels,
        neg_clip_val,
        clip_val,
        shift,
    ]
    ai_infra_qat_asymmetric_per_group_backward_kernel(*inputs)

    return grad_weight_out_grouped.view(weight_pto.shape), grad_scale_out, grad_offset_out


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 64,
    },
)
def ai_infra_qat_symmetric_per_channel_kernel(
    weight: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    scale: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_BF16),
    output_bf16: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    eps,
    min_v,
    max_v
):
    pypto.experimental.set_operation_options(combine_axis=True)
    n, m = weight.shape
    unroll_list = [512, 32, 8]
    pypto.set_vec_tile_shapes(32, 512)

    for n_offset, unroll_length in pypto.loop_unroll(
        0, n, 1,
        name="LOOP_N_UNROLL",
        idx_name="n_offset",
        unroll_list=unroll_list
    ):
        tile_n = unroll_length

        weight_tile = pypto.view(weight, [tile_n, m], [n_offset, 0])
        weight_fp32 = pypto.cast(weight_tile, pypto.DT_FP32)
        scale_tile = pypto.view(scale, [tile_n, 1], [n_offset, 0])
        scale_fp32 = pypto.cast(scale_tile, pypto.DT_FP32)

        protected_scale = pypto.maximum(scale_fp32, eps)
        normalized = pypto.div(weight_fp32, protected_scale, precision_type=pypto.PrecisionType.INTRINSIC)
        rounded = pypto.round(normalized, decimals=0)
        clamped = pypto.clip(rounded, min_v, max_v)
        output = pypto.mul(clamped, protected_scale)

        output_tile = pypto.cast(output, pypto.DT_BF16)

        pypto.assemble(output_tile, [n_offset, 0], output_bf16)


def ai_infra_qat_symmetric_per_channel(weight, scale, eps, min_v, max_v):
    output_pto = torch.empty(weight.shape, dtype=torch.bfloat16, device=weight.device)
    inputs = [
        weight,
        scale,
        output_pto,
        eps,
        min_v,
        max_v
    ]
    ai_infra_qat_symmetric_per_channel_kernel(*inputs)
    return output_pto


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 64,
    },
    pass_options={"vec_nbuffer_setting": {-1: 2, -2: 1}},
)
def ai_infra_qat_symmetric_per_channel_backward_kernel(
    grad_output: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    weight: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    scale: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_BF16),
    grad_weight_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    grad_scale_out: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_BF16),
    eps,
    min_v,
    max_v,
):

    n, m = weight.shape
    unroll_list = [512, 32, 8]

    tile_shapes_m = min(m, 4096)
    pypto.set_vec_tile_shapes(4, tile_shapes_m)

    for n_offset, unroll_length in pypto.loop_unroll(
        0, n, 1,
        name="BACKWARD_LOOP_N_UNROLL",
        idx_name="n_offset",
        unroll_list=unroll_list
    ):
        tile_n = unroll_length

        grad_out_tile = pypto.view(grad_output, [tile_n, m], [n_offset, 0])
        weight_tile = pypto.view(weight, [tile_n, m], [n_offset, 0])
        scale_tile = pypto.view(scale, [tile_n, 1], [n_offset, 0])

        grad_out_fp32 = pypto.cast(grad_out_tile, pypto.DT_FP32)
        weight_fp32 = pypto.cast(weight_tile, pypto.DT_FP32)
        scale_fp32_tile = pypto.cast(scale_tile, pypto.DT_FP32)

        protected_scale_tile = pypto.maximum(scale_fp32_tile, eps)
        scale_mask_tile = pypto.ge(scale_fp32_tile, eps)
        scale_mask_fp32_tile = pypto.where(scale_mask_tile, 1.0, 0.0)

        # 重算前向计算
        normalized = pypto.div(weight_fp32, protected_scale_tile, precision_type=pypto.PrecisionType.INTRINSIC)
        rounded = pypto.round(normalized, decimals=0)
        clamped = pypto.clip(rounded, min_v, max_v)

        # 计算mask: (rounded >= min_v) & (rounded <= max_v)
        # 等价与计算 equal(rounded, clamped) + where(相同, 1.0, 0.0)
        # 规避where, 求两者差，其都是整数型浮点数(xxx.0)
        diff = pypto.sub(rounded, clamped)
        abs_diff = pypto.abs(diff)
        out_of_bounds = pypto.clip(abs_diff, 0.0, 1.0)
        neg_out_of_bounds = pypto.mul(out_of_bounds, -1.0)
        mask_float = pypto.add(neg_out_of_bounds, 1.0)

        # ================= 计算 grad_weight =================
        grad_weight_fp32 = pypto.mul(grad_out_fp32, mask_float)
        grad_weight_tile = pypto.cast(grad_weight_fp32, pypto.DT_BF16)
        pypto.assemble(grad_weight_tile, [n_offset, 0], grad_weight_out)

        # ================= 计算 grad_scale =================
        # 乘法路径: grad_output * clamped
        grad_scale_mul_tile = pypto.mul(grad_out_fp32, clamped) # [tile_n, m]
        grad_scale_mul_tile_sum = pypto.sum(grad_scale_mul_tile, dim=1, keepdim=True)

        # 除法路径: grad_output * mask * (-weight / protected_scale)
        neg_weight_fp32 = pypto.mul(weight_fp32, -1.0)
        weight_div_scale_tile = pypto.div(
            neg_weight_fp32, protected_scale_tile, precision_type=pypto.PrecisionType.INTRINSIC)
        grad_scale_div_step1 = pypto.mul(grad_out_fp32, mask_float)
        grad_scale_div_tile = pypto.mul(grad_scale_div_step1, weight_div_scale_tile)
        grad_scale_div_tile_sum = pypto.sum(grad_scale_div_tile, dim=1, keepdim=True)

        # 合并两条路径然后mask
        grad_scale_sum_m = pypto.add(grad_scale_mul_tile_sum, grad_scale_div_tile_sum)
        grad_scale_fp32_masked = pypto.mul(grad_scale_sum_m, scale_mask_fp32_tile)
        grad_scale_bf16_tile = pypto.cast(grad_scale_fp32_masked, pypto.DT_BF16)

        pypto.assemble(grad_scale_bf16_tile, [n_offset, 0], grad_scale_out)


def ai_infra_qat_symmetric_per_channel_backward(grad_output, weight, scale, eps, min_v, max_v):
    grad_weight_out = torch.empty(weight.shape, dtype=weight.dtype, device=weight.device)
    grad_scale_out = torch.empty(scale.shape, dtype=scale.dtype, device=scale.device)
    inputs = [
        grad_output,
        weight,
        scale,
        grad_weight_out,
        grad_scale_out,
        eps,
        min_v,
        max_v
    ]
    ai_infra_qat_symmetric_per_channel_backward_kernel(*inputs)
    return grad_weight_out, grad_scale_out


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 64
    },
)
def ai_infra_qat_symmetric_per_tensor_kernel(
    weight: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    scale: pypto.Tensor([1, 1], pypto.DT_BF16),
    output_bf16: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    eps,
    min_v,
    max_v
):

    unroll_list = [512, 32, 8]
    pypto.experimental.set_operation_options(combine_axis=True)
    n, m = weight.shape
    pypto.set_vec_tile_shapes(32, 512)
    scale_fp32 = pypto.cast(scale, pypto.DT_FP32)
    protected_scale = pypto.maximum(scale_fp32, eps)

    for n_offset, unroll_length in pypto.loop_unroll(
        0, n, 1,
        name="LOOP_N_UNROLL",
        idx_name="n_offset",
        unroll_list=unroll_list
    ):
        tile_n = unroll_length

        weight_tile = pypto.view(weight, [tile_n, m], [n_offset, 0])
        weight_fp32 = pypto.cast(weight_tile, pypto.DT_FP32)

        scale_n = pypto.expand_clone(protected_scale, [tile_n, 1])
        normalized = pypto.div(weight_fp32, scale_n, precision_type=pypto.PrecisionType.INTRINSIC)
        rounded = pypto.round(normalized, decimals=0)
        clamped = pypto.clip(rounded, min_v, max_v)
        output = pypto.mul(clamped, scale_n)

        output_tile = pypto.cast(output, pypto.DT_BF16)

        pypto.assemble(output_tile, [n_offset, 0], output_bf16)


def ai_infra_qat_symmetric_per_tensor(weight, scale, eps, min_v, max_v):
    output_pto = torch.empty(weight.shape, dtype=torch.bfloat16, device=weight.device)
    inputs = [
        weight,
        scale,
        output_pto,
        eps,
        min_v,
        max_v
    ]
    ai_infra_qat_symmetric_per_tensor_kernel(*inputs)
    return output_pto


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 64,
    },
    pass_options={"vec_nbuffer_setting": {-1: 4}},
)
def ai_infra_qat_symmetric_per_tensor_backward_kernel(
    grad_output: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    weight: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    scale: pypto.Tensor([1, 1], pypto.DT_BF16),
    grad_weight_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    grad_scale_out: pypto.Tensor([1, 1], pypto.DT_BF16),
    eps,
    min_v,
    max_v,
):
    pypto.experimental.set_operation_options(combine_axis=False)
    n, m = weight.shape
    unroll_list = [512, 32, 8]
    tile_shapes_m = min(m, 4096)
    pypto.set_vec_tile_shapes(4, tile_shapes_m)

    scale_fp32 = pypto.cast(scale, pypto.DT_FP32)
    protected_scale = pypto.maximum(scale_fp32, eps)

    # 计算全局的 scale_mask
    scale_mask = pypto.ge(scale_fp32, eps)
    scale_mask_fp32 = pypto.where(scale_mask, 1.0, 0.0)

    # 在循环外初始化 FP32 的局部累加器
    grad_scale_acc = pypto.full([1, 1], 0.0, pypto.DT_FP32)

    for n_offset, unroll_length in pypto.loop_unroll(
        0, n, 1,
        name="BACKWARD_LOOP_N_UNROLL",
        idx_name="n_offset",
        unroll_list=unroll_list
    ):
        tile_n = unroll_length
        grad_out_tile = pypto.view(grad_output, [tile_n, m], [n_offset, 0])
        grad_out_fp32 = pypto.cast(grad_out_tile, pypto.DT_FP32)

        weight_tile = pypto.view(weight, [tile_n, m], [n_offset, 0])
        weight_fp32 = pypto.cast(weight_tile, pypto.DT_FP32)

        # 重算前向计算
        scale_n = pypto.expand_clone(protected_scale, [tile_n, 1])
        normalized = pypto.div(weight_fp32, scale_n, precision_type=pypto.PrecisionType.INTRINSIC)
        rounded = pypto.round(normalized, decimals=0)
        clamped = pypto.clip(rounded, min_v, max_v)

        # 计算mask: (rounded >= min_v) & (rounded <= max_v)
        # 等价与计算 equal(rounded, clamped) + where(相同, 1.0, 0.0)
        # 规避where, 求两者差，其都是整数型浮点数(xxx.0)
        diff = pypto.sub(rounded, clamped)
        abs_diff = pypto.abs(diff)
        out_of_bounds = pypto.clip(abs_diff, 0.0, 1.0)
        neg_out_of_bounds = pypto.mul(out_of_bounds, -1.0)
        mask_float = pypto.add(neg_out_of_bounds, 1.0)

        # ================= 计算 grad_weight =================
        grad_weight_fp32 = pypto.mul(grad_out_fp32, mask_float)
        grad_weight_tile = pypto.cast(grad_weight_fp32, pypto.DT_BF16)
        pypto.assemble(grad_weight_tile, [n_offset, 0], grad_weight_out)

        # ================= 计算 grad_scale =================
        # 乘法路径: grad_output * clamped
        grad_scale_mul_tile = pypto.mul(grad_out_fp32, clamped) # [tile_n, m]
        grad_scale_mul_tile_m = pypto.sum(grad_scale_mul_tile, dim=1, keepdim=True)

        # 除法路径: grad_output * mask * (-weight / protected_scale)
        neg_weight_fp32 = pypto.mul(weight_fp32, -1.0)
        weight_div_scale_tile = pypto.div(neg_weight_fp32, scale_n, precision_type=pypto.PrecisionType.INTRINSIC)
        grad_scale_div_step1 = pypto.mul(grad_out_fp32, mask_float)
        grad_scale_div_tile = pypto.mul(grad_scale_div_step1, weight_div_scale_tile)
        grad_scale_div_tile_m = pypto.sum(grad_scale_div_tile, dim=1, keepdim=True)

        pypto.set_vec_tile_shapes(512, 1)
        grad_scale_mul_tile_n = pypto.sum(grad_scale_mul_tile_m, dim=0, keepdim=True)
        grad_scale_div_tile_n = pypto.sum(grad_scale_div_tile_m, dim=0, keepdim=True)

        # 合并两条路径
        grad_scale_tile = pypto.add(grad_scale_mul_tile_n, grad_scale_div_tile_n)
        # 将当前 Tile 的梯度累加到全局 FP32 寄存器中
        grad_scale_acc[:] = pypto.add(grad_scale_acc, grad_scale_tile)

    # 应用 scale_mask 掩码，并进行最终的类型转换和写回
    final_grad_scale_fp32 = pypto.mul(grad_scale_acc, scale_mask_fp32)
    final_grad_scale_bf16 = pypto.cast(final_grad_scale_fp32, pypto.DT_BF16)
    pypto.assemble(final_grad_scale_bf16, [0, 0], grad_scale_out)


def ai_infra_qat_symmetric_per_tensor_backward(grad_output, weight, scale, eps, min_v, max_v):
    grad_weight_out = torch.empty(weight.shape, dtype=weight.dtype, device=weight.device)
    grad_scale_out = torch.empty(scale.shape, dtype=scale.dtype, device=scale.device)
    inputs = [
        grad_output,
        weight,
        scale,
        grad_weight_out,
        grad_scale_out,
        eps,
        min_v,
        max_v
    ]
    ai_infra_qat_symmetric_per_tensor_backward_kernel(*inputs)
    return grad_weight_out, grad_scale_out