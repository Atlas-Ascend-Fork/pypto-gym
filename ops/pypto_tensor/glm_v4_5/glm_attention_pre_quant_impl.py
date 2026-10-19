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
GLM-4.5 Attention Pre Quant Module

This module implements the fused attention_pre_quant operation for GLM-4.5,
which combines multiple operations:
- Input LayerNorm with residual connection
- Input quantization
- Quantized QKV matrix multiplication
- Q/K LayerNorm
- Rotary Position Embedding (RoPE)

This fused operation significantly improves execution efficiency and memory bandwidth
utilization on NPU by reducing kernel launch overhead.

Main Functions:
    - attention_pre_quant: Main function for attention_pre_quant
    - quant_attention_pre_kernel: JIT compiled kernel implementation
    - rms_norm_bias: RMS normalization with bias
    - rope_data: Rotary position embedding computation
"""
import os
from typing import Optional
import torch
import pypto
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
from .utils.get_format import get_format


def check_args(
    hidden_states,
    residual,
    input_layernorm_weight,
    input_layernorm_bias,
    atten_qkv_input_scale_reciprocal,
    atten_qkv_input_offset,
    atten_qkv_weight,
    atten_qkv_quant_bias,
    atten_qkv_deq_scale,
    atten_q_norm_weight,
    atten_q_norm_bias,
    atten_k_norm_weight,
    atten_k_norm_bias,
    cos,
    sin,
    query,
    key,
    value,
    residual_res,
):
    assert hidden_states.dim() == 2
    assert get_format(hidden_states) == 'ND'
    assert hidden_states.dtype == torch.bfloat16
    assert residual.dim() == 2
    assert get_format(residual) == 'ND'
    assert residual.dtype == torch.bfloat16
    assert input_layernorm_weight.dim() == 1
    assert get_format(input_layernorm_weight) == 'ND'
    assert input_layernorm_weight.dtype == torch.bfloat16
    assert input_layernorm_bias.dim() == 1
    assert get_format(input_layernorm_bias) == 'ND'
    assert input_layernorm_bias.dtype == torch.bfloat16
    assert atten_qkv_input_scale_reciprocal.dim() == 1
    assert get_format(atten_qkv_input_scale_reciprocal) == 'ND'
    assert atten_qkv_input_scale_reciprocal.dtype == torch.bfloat16
    assert atten_qkv_input_offset.dim() == 1
    assert get_format(atten_qkv_input_offset) == 'ND'
    assert atten_qkv_input_offset.dtype == torch.bfloat16
    assert atten_qkv_weight.dim() == 2
    assert get_format(atten_qkv_weight) == 'NZ'
    assert atten_qkv_weight.dtype == torch.int8
    assert atten_qkv_quant_bias.dim() == 1
    assert get_format(atten_qkv_quant_bias) == 'ND'
    assert atten_qkv_quant_bias.dtype == torch.int32
    assert atten_qkv_deq_scale.dim() == 1
    assert get_format(atten_qkv_deq_scale) == 'ND'
    assert atten_qkv_deq_scale.dtype == torch.float32
    assert atten_q_norm_weight.dim() == 1
    assert get_format(atten_q_norm_weight) == 'ND'
    assert atten_q_norm_weight.dtype == torch.bfloat16
    assert atten_q_norm_bias.dim() == 1
    assert get_format(atten_q_norm_bias) == 'ND'
    assert atten_q_norm_bias.dtype == torch.bfloat16
    assert atten_k_norm_weight.dim() == 1
    assert get_format(atten_k_norm_weight) == 'ND'
    assert atten_k_norm_weight.dtype == torch.bfloat16
    assert atten_k_norm_bias.dim() == 1
    assert get_format(atten_k_norm_bias) == 'ND'
    assert atten_k_norm_bias.dtype == torch.bfloat16
    assert cos.dim() == 3
    assert cos.shape[1] == 1
    assert get_format(cos) == 'ND'
    assert cos.dtype == torch.bfloat16
    assert sin.dim() == 3
    assert sin.shape[1] == 1
    assert get_format(sin) == 'ND'
    assert sin.dtype == torch.bfloat16
    assert query.dim() == 2
    assert get_format(query) == 'ND'
    assert query.dtype == torch.bfloat16
    assert key.dim() == 2
    assert get_format(key) == 'ND'
    assert key.dtype == torch.bfloat16
    assert value.dim() == 2
    assert get_format(value) == 'ND'
    assert value.dtype == torch.bfloat16
    assert residual_res.dim() == 2
    assert get_format(residual_res) == 'ND'
    assert residual_res.dtype == torch.bfloat16


def rms_norm_bias(tensor_value, gamma, bias, mean_coff, eps, tile_shape):
    input_dtype = tensor_value.dtype
    pypto.set_vec_tile_shapes(*tile_shape)
    tensor_value_fp32 = pypto.cast(tensor_value, pypto.DT_FP32)

    square = pypto.mul(tensor_value_fp32, tensor_value_fp32)

    mean_res = pypto.mul(square, mean_coff)

    reduce_asum = pypto.sum(mean_res, -1, keepdim=True)
    reduce_sum = pypto.add(reduce_asum, eps)

    reduce_sqrt = pypto.sqrt(reduce_sum)

    res_div = pypto.div(tensor_value_fp32, reduce_sqrt)

    res = pypto.mul(res_div, gamma)

    res_add = pypto.add(res, bias)

    y_bf16 = pypto.cast(res_add, input_dtype)

    return y_bf16


def rope_data(x1, x2, cos, sin, tile_shape):
    pypto.set_vec_tile_shapes(*tile_shape)
    o1 = pypto.sub(pypto.mul(x1, cos), pypto.mul(x2, sin))
    o2 = pypto.add(pypto.mul(x2, cos), pypto.mul(x1, sin))
    res = pypto.concat([o1, o2], 2)

    y_bf16 = pypto.cast(res, pypto.DT_BF16)
    return y_bf16


@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128}
)
def quant_attention_pre_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    residual_input: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    x_gamma: pypto.Tensor([], pypto.DT_BF16),
    x_bias: pypto.Tensor([], pypto.DT_BF16),
    x_scale: pypto.Tensor([], pypto.DT_BF16),
    x_offset: pypto.Tensor([], pypto.DT_BF16),
    weight: pypto.Tensor([], pypto.DT_INT8),
    quant_bias: pypto.Tensor([], pypto.DT_INT32),
    deq_scale: pypto.Tensor([], pypto.DT_FP32),
    q_gamma: pypto.Tensor([], pypto.DT_BF16),
    q_bias: pypto.Tensor([], pypto.DT_BF16),
    k_gamma: pypto.Tensor([], pypto.DT_BF16),
    k_bias: pypto.Tensor([], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    residual: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
):
    """
    JIT compiled kernel for fused attention_pre_quant operation.

    This kernel performs the following operations in sequence:
    1. Add residual connection: x = residual + hidden_states
    2. RMS normalization: x_norm = RMSNorm(x)
    3. Input quantization: x_int8 = Quantize(x_norm)
    4. Quantized QKV projection: qkv = Dequantize(MatMul(x_int8, weight))
    5. Split QKV: q, k, v = Split(qkv)
    6. Q/K normalization: q_norm = RMSNorm(q), k_norm = RMSNorm(k)
    7. Apply RoPE: q_rope = RoPE(q_norm), k_rope = RoPE(k_norm)

    Args:
        x: Input hidden states [num_tokens, hidden_size]
        residual_input: Residual tensor [num_tokens, hidden_size]
        x_gamma: Input LayerNorm weight [hidden_size]
        x_bias: Input LayerNorm bias [hidden_size]
        x_scale: Input quantization scale [hidden_size]
        x_offset: Input quantization offset [hidden_size]
        weight: QKV weight matrix (int8) [hidden_size, total_head_size]
        quant_bias: QKV quantization bias [total_head_size]
        deq_scale: QKV dequantization scale [total_head_size]
        q_gamma: Query LayerNorm weight [head_size]
        q_bias: Query LayerNorm bias [head_size]
        k_gamma: Key LayerNorm weight [head_size]
        k_bias: Key LayerNorm bias [head_size]
        cos: Cosine values for RoPE [num_tokens, 1, half_rotary_dim]
        sin: Sine values for RoPE [num_tokens, 1, half_rotary_dim]
        q: Output query tensor [num_tokens, q_size]
        k: Output key tensor [num_tokens, kv_size]
        v: Output value tensor [num_tokens, kv_size]
        residual: Output residual tensor [num_tokens, hidden_size]

    Note:
        This function processes inputs in tiles of size 8 to support dynamic batch sizes.
        The computation uses FP32 for intermediate calculations to maintain numerical precision.
    """
    hidden_size = x.shape[1]
    total_head_size = weight.shape[1]
    head_size = q_gamma.shape[0]
    bs = x.shape[0]
    half_rotary_dim = cos.shape[-1]
    q_size = q.shape[-1]
    kv_size = k.shape[-1]
    bs_tile = 8

    x_mean_coff = 1.0 / x.shape[-1]
    qk_mean_coff = 1.0 / head_size
    eps = 1e-05
    rotary_dim = half_rotary_dim * 2
    stay_dim = head_size - rotary_dim

    q_num_head = q_size // head_size
    kv_num_head = kv_size // head_size
    kv_index = q_num_head + kv_num_head

    bs_loop = (bs + bs_tile - 1) // bs_tile
    calc_dtype = pypto.DT_FP32
    input_dtype = x.dtype
    tiling_value = 128
    vec_tile_value = 5120
    q_batch_tile = 4
    pypto.set_vec_tile_shapes(vec_tile_value)
    x_gamma_2d = pypto.reshape(x_gamma, [1, hidden_size], inplace=True)
    x_bias_2d = pypto.reshape(x_bias, [1, hidden_size], inplace=True)
    x_scale_2d = pypto.reshape(x_scale, [1, hidden_size], inplace=True)
    x_offset_2d = pypto.reshape(x_offset, [1, hidden_size], inplace=True)
    quant_bias_2d = pypto.reshape(quant_bias, [1, total_head_size], inplace=True)
    deq_scale_2d = pypto.reshape(deq_scale, [1, total_head_size], inplace=True)
    q_gamma_2d = pypto.reshape(q_gamma, [1, 1, head_size], inplace=True)
    q_bias_2d = pypto.reshape(q_bias, [1, 1, head_size], inplace=True)
    k_gamma_2d = pypto.reshape(k_gamma, [1, 1, head_size], inplace=True)
    k_bias_2d = pypto.reshape(k_bias, [1, 1, head_size], inplace=True)

    pypto.set_vec_tile_shapes(1, 1, head_size)
    q_gamma_2d_fp32 = pypto.cast(q_gamma_2d, calc_dtype)
    q_bias_2d_fp32 = pypto.cast(q_bias_2d, calc_dtype)
    k_gamma_2d_fp32 = pypto.cast(k_gamma_2d, calc_dtype)
    k_bias_2d_fp32 = pypto.cast(k_bias_2d, calc_dtype)
    q_gamma_expand = pypto.expand_clone(q_gamma_2d_fp32, [1, q_num_head, head_size])
    q_bias_expand = pypto.expand_clone(q_bias_2d_fp32, [1, q_num_head, head_size])
    k_gamma_expand = pypto.expand_clone(k_gamma_2d_fp32, [1, kv_num_head, head_size])
    k_bias_expand = pypto.expand_clone(k_bias_2d_fp32, [1, kv_num_head, head_size])

    for bs_idx in pypto.loop(bs_loop, name="LOOP_ATT_PRE_L0", idx_name="bs_idx"):
        act_bs_tile = (bs - bs_idx * bs_tile).min(bs_tile)

        x_tile = pypto.view(x, [bs_tile, hidden_size], [bs_idx * bs_tile, 0],
                                valid_shape=[act_bs_tile, hidden_size])
        pypto.set_vec_tile_shapes(1, vec_tile_value)
        x_tile_fp32 = pypto.cast(x_tile, calc_dtype)
        residual_input_tile = pypto.view(residual_input, [bs_tile, hidden_size], [bs_idx * bs_tile, 0],
            valid_shape=[act_bs_tile, hidden_size])
        residual_input_tile_fp32 = pypto.cast(residual_input_tile, calc_dtype)
        x_f32 = pypto.add(residual_input_tile_fp32, x_tile_fp32)

        square = pypto.mul(x_f32, x_f32)
        mean_res = pypto.mul(square, x_mean_coff)
        reduce_asum = pypto.sum(mean_res, -1, keepdim=True)
        reduce_sum = pypto.add(reduce_asum, eps)
        reduce_sqrt = pypto.sqrt(reduce_sum)
        res_div = pypto.div(x_f32, reduce_sqrt)
        residual_bf16 = pypto.cast(x_f32, input_dtype)
        x_int8 = pypto.tensor([bs_tile, hidden_size], pypto.DT_INT8, "x_int8")

        for tmp_idx in range(bs_tile):
            pypto.set_vec_tile_shapes(1, vec_tile_value)
            x_gamma_2d_fp32 = pypto.cast(x_gamma_2d, calc_dtype)
            x_bias_2d_fp32 = pypto.cast(x_bias_2d, calc_dtype)
            x_scale_2d_fp32 = pypto.cast(x_scale_2d, calc_dtype)
            x_offset_2d_fp32 = pypto.cast(x_offset_2d, calc_dtype)

            res_div_single = pypto.view(res_div, [1, hidden_size], [tmp_idx, 0])

            res = pypto.mul(res_div_single, x_gamma_2d_fp32)
            res_add = pypto.add(res, x_bias_2d_fp32)
            x_norm = pypto.cast(res_add, input_dtype)

            pypto.set_vec_tile_shapes(1, vec_tile_value)
            x_norm_fp32 = pypto.cast(x_norm, calc_dtype)
            x_mul = pypto.mul(x_norm_fp32, x_scale_2d_fp32)
            x_add = pypto.add(x_mul, x_offset_2d_fp32)
            x_int32 = pypto.cast(x_add, pypto.DT_INT32, pypto.CastMode.CAST_RINT)
            x_fp16 = pypto.cast(x_int32, pypto.DT_FP16)
            x_int8[tmp_idx:tmp_idx + 1, 0:] = pypto.cast(x_fp16, pypto.DT_INT8, satmode=pypto.SaturationMode.ON)

        pypto.set_cube_tile_shapes([32, 32], [256, 512], [256, 256])
        tmp_c = pypto.matmul(x_int8, weight, pypto.DT_INT32)
        pypto.set_vec_tile_shapes(bs_tile, total_head_size)
        mm_add = pypto.add(tmp_c, quant_bias_2d)
        mm_fp32 = pypto.cast(mm_add, calc_dtype)
        mm_deq_scale = pypto.mul(mm_fp32, deq_scale_2d)
        mm_bf16 = pypto.cast(mm_deq_scale, input_dtype)

        pypto.set_vec_tile_shapes(bs_tile, head_size)
        mm_3d = pypto.reshape(mm_bf16, [bs_tile, total_head_size // head_size, head_size], inplace=True)
        pypto.set_vec_tile_shapes(bs_tile, tiling_value, head_size)

        q_tile = pypto.view(mm_3d, [bs_tile, q_num_head, head_size], [0, 0, 0],
            valid_shape=[act_bs_tile, q_num_head, head_size])
        k_tile = pypto.view(mm_3d, [bs_tile, kv_num_head, head_size], [0, q_num_head, 0],
            valid_shape=[act_bs_tile, kv_num_head, head_size])
        v_tile = pypto.view(mm_3d, [bs_tile, kv_num_head, head_size], [0, kv_index, 0],
            valid_shape=[act_bs_tile, kv_num_head, head_size])

        q_norm = rms_norm_bias(q_tile, q_gamma_expand, q_bias_expand, qk_mean_coff, eps,
            [q_batch_tile, q_num_head, head_size])
        k_norm = rms_norm_bias(k_tile, k_gamma_expand, k_bias_expand, qk_mean_coff, eps,
            [q_batch_tile, kv_num_head, head_size])

        q_rot = pypto.view(q_norm, [bs_tile, q_num_head, rotary_dim], [0, 0, 0],
            valid_shape=[act_bs_tile, q_num_head, rotary_dim])
        q_pass = pypto.view(q_norm, [bs_tile, q_num_head, stay_dim], [0, 0, rotary_dim],
            valid_shape=[act_bs_tile, q_num_head, stay_dim])

        k_rot = pypto.view(k_norm, [bs_tile, kv_num_head, rotary_dim], [0, 0, 0],
            valid_shape=[act_bs_tile, kv_num_head, rotary_dim])
        k_pass = pypto.view(k_norm, [bs_tile, kv_num_head, stay_dim], [0, 0, rotary_dim],
            valid_shape=[act_bs_tile, kv_num_head, stay_dim])

        pypto.set_vec_tile_shapes(q_batch_tile, q_num_head, head_size)
        cos_tile = pypto.view(cos, [bs_tile, 1, half_rotary_dim], [bs_idx * bs_tile, 0, 0],
            valid_shape=[act_bs_tile, 1, half_rotary_dim])
        sin_tile = pypto.view(sin, [bs_tile, 1, half_rotary_dim], [bs_idx * bs_tile, 0, 0],
            valid_shape=[act_bs_tile, 1, half_rotary_dim])
        q_fp32 = pypto.cast(q_rot, calc_dtype)
        k_fp32 = pypto.cast(k_rot, calc_dtype)
        cos_fp32 = pypto.cast(cos_tile, calc_dtype)
        sin_fp32 = pypto.cast(sin_tile, calc_dtype)

        q1 = pypto.view(q_fp32, [bs_tile, q_num_head, half_rotary_dim], [0, 0, 0],
            valid_shape=[act_bs_tile, q_num_head, half_rotary_dim])
        q2 = pypto.view(q_fp32, [bs_tile, q_num_head, half_rotary_dim], [0, 0, half_rotary_dim],
            valid_shape=[act_bs_tile, q_num_head, half_rotary_dim])

        q_rope = rope_data(q1, q2, cos_fp32, sin_fp32, [q_batch_tile, q_num_head, half_rotary_dim])
        q_cat = pypto.concat([q_rope, q_pass], 2)

        k1 = pypto.view(k_fp32, [bs_tile, kv_num_head, half_rotary_dim], [0, 0, 0],
            valid_shape=[act_bs_tile, kv_num_head, half_rotary_dim])
        k2 = pypto.view(k_fp32, [bs_tile, kv_num_head, half_rotary_dim], [0, 0, half_rotary_dim],
            valid_shape=[act_bs_tile, kv_num_head, half_rotary_dim])

        k_rope = rope_data(k1, k2, cos_fp32, sin_fp32, [q_batch_tile, q_num_head, half_rotary_dim])
        k_cat = pypto.concat([k_rope, k_pass], 2)

        q_res = pypto.reshape(q_cat, [bs_tile, q_size], valid_shape=[act_bs_tile, q_size])
        k_res = pypto.reshape(k_cat, [bs_tile, kv_size], valid_shape=[act_bs_tile, kv_size])
        v_res = pypto.reshape(v_tile, [bs_tile, kv_size], valid_shape=[act_bs_tile, kv_size])

        q[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = q_res
        k[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = k_res
        v[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = v_res
        residual[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = residual_bf16


@allow_in_graph
def attention_pre_quant(
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    input_layernorm_weight: torch.Tensor,
    input_layernorm_bias: torch.Tensor,
    atten_qkv_input_scale_reciprocal: torch.Tensor,
    atten_qkv_input_offset: torch.Tensor,
    atten_qkv_weight: torch.Tensor,
    atten_qkv_quant_bias: torch.Tensor,
    atten_qkv_deq_scale: torch.Tensor,
    atten_q_norm_weight: torch.Tensor,
    atten_q_norm_bias: torch.Tensor,
    atten_k_norm_weight: torch.Tensor,
    atten_k_norm_bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    residual_res: torch.Tensor
):
    """
    Main function for attention_pre_quant operation.

    This function fuses multiple operations to compute Q, K, V tensors for attention:
    - Input LayerNorm with residual connection
    - Input quantization
    - Quantized QKV matrix multiplication
    - Q/K LayerNorm
    - Rotary Position Embedding (RoPE)

    Args:
        hidden_states: Input hidden states [num_tokens, hidden_size]
        residual: Optional residual tensor [num_tokens, hidden_size]
        input_layernorm_weight: Input LayerNorm weight [hidden_size]
        input_layernorm_bias: Input LayerNorm bias [hidden_size]
        atten_qkv_input_scale_reciprocal: QKV input quantization scale reciprocal [hidden_size]
        atten_qkv_input_offset: QKV input quantization offset [hidden_size]
        atten_qkv_weight: QKV weight matrix (int8) [hidden_size, total_head_size]
        atten_qkv_quant_bias: QKV quantization bias [total_head_size]
        atten_qkv_deq_scale: QKV dequantization scale [total_head_size]
        atten_q_norm_weight: Query LayerNorm weight [head_size]
        atten_q_norm_bias: Query LayerNorm bias [head_size]
        atten_k_norm_weight: Key LayerNorm weight [head_size]
        atten_k_norm_bias: Key LayerNorm bias [head_size]
        cos: Cosine values for RoPE [num_tokens, 1, half_rotary_dim]
        sin: Sine values for RoPE [num_tokens, 1, half_rotary_dim]
        query: Output query tensor [num_tokens, q_size]
        key: Output key tensor [num_tokens, kv_size]
        value: Output value tensor [num_tokens, kv_size]
        residual_res: Output residual tensor [num_tokens, hidden_size]

    Note:
        This function is decorated with @allow_in_graph to enable integration
        with PyTorch's compilation graph.
    """
    if isinstance(hidden_states, FakeTensor):
        return

    check_args(
        hidden_states,
        residual,
        input_layernorm_weight,
        input_layernorm_bias,
        atten_qkv_input_scale_reciprocal,
        atten_qkv_input_offset,
        atten_qkv_weight,
        atten_qkv_quant_bias,
        atten_qkv_deq_scale,
        atten_q_norm_weight,
        atten_q_norm_bias,
        atten_k_norm_weight,
        atten_k_norm_bias,
        cos,
        sin,
        query,
        key,
        value,
        residual_res,
    )

    _bs = hidden_states.shape[0]
    _hidden_size = hidden_states.shape[1]
    _total_head_size = atten_qkv_weight.shape[1]
    _head_size = atten_q_norm_weight.shape[0]
    _q_size = query.shape[1]
    _kv_size = key.shape[1]
    _half_rotary_dim = cos.shape[2]
    inputs = [hidden_states, residual, input_layernorm_weight, input_layernorm_bias, atten_qkv_input_scale_reciprocal,
         atten_qkv_input_offset, atten_qkv_weight, atten_qkv_quant_bias, atten_qkv_deq_scale, atten_q_norm_weight,
         atten_q_norm_bias, atten_k_norm_weight, atten_k_norm_bias, cos, sin, query, key, value, residual_res]
    quant_attention_pre_kernel(*inputs)
