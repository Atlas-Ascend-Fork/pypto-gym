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
import sys
import logging
from typing import Optional

_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

import torch
import torch_npu
import pytest
import numpy as np
import pypto
from numpy.testing import assert_allclose
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
from common_utils import get_format


logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)


def _check_input_shape_dtype(t, dim, fmt, dtype, name):
    assert t.dim() == dim, f"invalid {name} dim"
    assert get_format(t) == fmt, f"invalid {name} format"
    assert t.dtype == dtype, f"invalid {name} dtype"


def check_args(
    hidden_states, residual, input_layernorm_weight, input_layernorm_bias,
    atten_qkv_input_scale_reciprocal, atten_qkv_input_offset,
    atten_qkv_weight, atten_qkv_quant_bias, atten_qkv_deq_scale,
    atten_q_norm_weight, atten_q_norm_bias, atten_k_norm_weight, atten_k_norm_bias,
    cos, sin, query, key, value, residual_res
):
    bf16 = torch.bfloat16
    _check_input_shape_dtype(hidden_states, 2, 'ND', bf16, "hidden_states")
    _check_input_shape_dtype(residual, 2, 'ND', bf16, "residual")
    _check_input_shape_dtype(input_layernorm_weight, 1, 'ND', bf16, "input_layernorm_weight")
    _check_input_shape_dtype(input_layernorm_bias, 1, 'ND', bf16, "input_layernorm_bias")
    _check_input_shape_dtype(atten_qkv_input_scale_reciprocal, 1, 'ND', bf16, "atten_qkv_input_scale_reciprocal")
    _check_input_shape_dtype(atten_qkv_input_offset, 1, 'ND', bf16, "atten_qkv_input_offset")
    assert atten_qkv_weight.dim() == 2
    assert get_format(atten_qkv_weight) == 'NZ'
    assert atten_qkv_weight.dtype == torch.int8
    _check_input_shape_dtype(atten_qkv_quant_bias, 1, 'ND', torch.int32, "atten_qkv_quant_bias")
    _check_input_shape_dtype(atten_qkv_deq_scale, 1, 'ND', torch.float32, "atten_qkv_deq_scale")
    _check_input_shape_dtype(atten_q_norm_weight, 1, 'ND', bf16, "atten_q_norm_weight")
    _check_input_shape_dtype(atten_q_norm_bias, 1, 'ND', bf16, "atten_q_norm_bias")
    _check_input_shape_dtype(atten_k_norm_weight, 1, 'ND', bf16, "atten_k_norm_weight")
    _check_input_shape_dtype(atten_k_norm_bias, 1, 'ND', bf16, "atten_k_norm_bias")
    assert cos.dim() == 3
    assert cos.shape[1] == 1
    assert get_format(cos) == 'ND'
    assert cos.dtype == bf16
    assert sin.dim() == 3
    assert sin.shape[1] == 1
    assert get_format(sin) == 'ND'
    assert sin.dtype == bf16
    _check_input_shape_dtype(query, 2, 'ND', bf16, "query")
    _check_input_shape_dtype(key, 2, 'ND', bf16, "key")
    _check_input_shape_dtype(value, 2, 'ND', bf16, "value")
    _check_input_shape_dtype(residual_res, 2, 'ND', bf16, "residual_res")


# golden
def add_rms_norm_npu_golden(residual_input, x, x_gamma, x_bias, eps):
    x_bias_fp32 = x_bias.to(torch.float32)
    x_fp32 = x.to(torch.float32)
    residual_input_fp32 = residual_input.to(torch.float32)
    x_fp32 = x_fp32 + residual_input_fp32
    x_mean_coff = 1.0 / x.shape[-1]
    x_square = x_fp32 * x_fp32
    x_mean = x_square * x_mean_coff
    x_reduce_sum = torch.sum(x_mean, dim=-1, keepdim=True) + eps
    x_reduce_sqrt = torch.sqrt(x_reduce_sum)
    x_res_div = x_fp32 / x_reduce_sqrt
    x_mul_res = x_res_div * x_gamma.to(torch.float32)
    x_add_bias = x_mul_res + x_bias_fp32

    return x_add_bias.to(torch.bfloat16), x_fp32.to(torch.bfloat16)


def rms_norm_npu_golden(x, x_gamma, x_bias, eps):
    x_bias_fp32 = x_bias.to(torch.float32)
    x_fp32 = x.to(torch.float32)
    x_mean_coff = 1.0 / x.shape[-1]
    x_square = x_fp32 * x_fp32
    x_mean = x_square * x_mean_coff
    x_reduce_sum = torch.sum(x_mean, dim=-1, keepdim=True) + eps
    x_reduce_sqrt = torch.sqrt(x_reduce_sum)
    x_res_div = x_fp32 / x_reduce_sqrt
    x_mul_res = x_res_div * x_gamma.to(torch.float32)
    x_add_bias = x_mul_res + x_bias_fp32

    return x_add_bias.to(torch.bfloat16)


def _apply_rotary_emb_neuron(x, cos, sin):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    return torch.cat((o1, o2), dim=-1)


def apply_rotary_pos_emb_v2(q, k, cos, sin):
    x_dtype = q.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)

    q_embed = _apply_rotary_emb_neuron(q, cos, sin)
    k_embed = _apply_rotary_emb_neuron(k, cos, sin)

    if x_dtype != torch.float32:
        q_embed = q_embed.to(x_dtype)
        k_embed = k_embed.to(x_dtype)
    return q_embed, k_embed


# pypto
def rms_norm_bias(tensor_value, gamma, bias, mean_coff, eps, tile_shape):
    input_dtype = tensor_value.dtype
    # cast
    pypto.set_vec_tile_shapes(*tile_shape)
    tensor_value_fp32 = pypto.cast(tensor_value, pypto.DT_FP32)

    # square
    square = pypto.mul(tensor_value_fp32, tensor_value_fp32)

    # mean_res
    mean_res = pypto.mul(square, mean_coff)

    # reduce sum
    reduce_asum = pypto.sum(mean_res, -1, keepdim=True)
    reduce_sum = pypto.add(reduce_asum, eps)

    # sqrt
    reduce_sqrt = pypto.sqrt(reduce_sum)

    # div
    res_div = pypto.div(tensor_value_fp32, reduce_sqrt)

    # gamma mul
    res = pypto.mul(res_div, gamma)

    res_add = pypto.add(res, bias)

    # cast
    y_bf16 = pypto.cast(res_add, input_dtype)

    return y_bf16


def rope_data(x1, x2, cos, sin, tile_shape):
    pypto.set_vec_tile_shapes(*tile_shape)
    o1 = pypto.sub(pypto.mul(x1, cos), pypto.mul(x2, sin))
    o2 = pypto.add(pypto.mul(x2, cos), pypto.mul(x1, sin))
    # concat
    res = pypto.concat([o1, o2], 2)

    # cast
    y_bf16 = pypto.cast(res, pypto.DT_BF16)
    return y_bf16


def _init_norm_weights(q_gamma, q_bias, k_gamma, k_bias, q_num_head, kv_num_head,
                       head_size, hidden_size, calc_dtype):
    """Initialize and expand norm weight tensors."""
    vec_tile_value = 5120
    pypto.set_vec_tile_shapes(vec_tile_value)
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
    return q_gamma_expand, q_bias_expand, k_gamma_expand, k_bias_expand


def _process_bs_tile(
    x, residual_input, x_gamma_2d, x_bias_2d, x_scale_2d, x_offset_2d,
    weight, quant_bias_2d, deq_scale_2d,
    q_gamma_expand, q_bias_expand, k_gamma_expand, k_bias_expand,
    cos, sin, q, k, v, residual,
    bs_idx, bs_tile, bs, hidden_size, total_head_size, head_size,
    calc_dtype, input_dtype, vec_tile_value, tiling_value, q_batch_tile,
    x_mean_coff, eps, qk_mean_coff,
    q_num_head, kv_num_head, kv_index,
    rotary_dim, stay_dim, half_rotary_dim, q_size, kv_size):
    """Process a single batch-size tile in the attention pre quant kernel."""
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

    x_int8 = _quantize_x_tile(res_div, x_gamma_2d, x_bias_2d, x_scale_2d, x_offset_2d,
                               bs_tile, hidden_size, calc_dtype, input_dtype, vec_tile_value)
    mm_bf16 = _quantized_matmul(x_int8, weight, quant_bias_2d, deq_scale_2d,
                                 bs_tile, total_head_size, head_size, calc_dtype, input_dtype)
    pypto.set_vec_tile_shapes(bs_tile, tiling_value, head_size)

    q_tile, k_tile, v_tile = _split_qkv(mm_bf16, bs_tile, total_head_size, head_size,
                                         q_num_head, kv_num_head, kv_index, act_bs_tile)

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

    q_cat, k_cat = _apply_rope(q_rot, q_pass, k_rot, k_pass, cos, sin,
                                bs_idx, bs_tile, act_bs_tile,
                                q_num_head, kv_num_head, half_rotary_dim,
                                calc_dtype, q_batch_tile, head_size)

    q_res = pypto.reshape(q_cat, [bs_tile, q_size], valid_shape=[act_bs_tile, q_size])
    k_res = pypto.reshape(k_cat, [bs_tile, kv_size], valid_shape=[act_bs_tile, kv_size])
    v_res = pypto.reshape(v_tile, [bs_tile, kv_size], valid_shape=[act_bs_tile, kv_size])

    q[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = q_res
    k[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = k_res
    v[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = v_res
    residual[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = residual_bf16


def _quantize_x_tile(res_div, x_gamma_2d, x_bias_2d, x_scale_2d, x_offset_2d,
                      bs_tile, hidden_size, calc_dtype, input_dtype, vec_tile_value):
    """Quantize x tile per-row and pack into int8 tensor."""
    x_int8 = pypto.tensor([bs_tile, hidden_size], pypto.DT_INT8, "x_int8")
    for tmp_idx in range(bs_tile):
        pypto.set_vec_tile_shapes(1, vec_tile_value)
        xg = pypto.cast(x_gamma_2d, calc_dtype)
        xb = pypto.cast(x_bias_2d, calc_dtype)
        xs = pypto.cast(x_scale_2d, calc_dtype)
        xo = pypto.cast(x_offset_2d, calc_dtype)
        res_div_single = pypto.view(res_div, [1, hidden_size], [tmp_idx, 0])
        res = pypto.mul(res_div_single, xg)
        res_add = pypto.add(res, xb)
        x_norm = pypto.cast(res_add, input_dtype)
        pypto.set_vec_tile_shapes(1, vec_tile_value)
        x_norm_fp32 = pypto.cast(x_norm, calc_dtype)
        x_mul = pypto.mul(x_norm_fp32, xs)
        x_add = pypto.add(x_mul, xo)
        x_int32 = pypto.cast(x_add, pypto.DT_INT32, pypto.CastMode.CAST_RINT)
        x_fp16 = pypto.cast(x_int32, pypto.DT_FP16)
        x_int8[tmp_idx:tmp_idx + 1, 0:] = pypto.cast(x_fp16, pypto.DT_INT8, satmode=pypto.SaturationMode.ON)
    return x_int8


def _quantized_matmul(x_int8, weight, quant_bias_2d, deq_scale_2d,
                       bs_tile, total_head_size, head_size, calc_dtype, input_dtype):
    """Perform quantized matmul with dequant."""
    pypto.set_cube_tile_shapes([32, 32], [256, 512], [256, 256])
    tmp_c = pypto.matmul(x_int8, weight, pypto.DT_INT32)
    pypto.set_vec_tile_shapes(bs_tile, total_head_size)
    mm_add = pypto.add(tmp_c, quant_bias_2d)
    mm_fp32 = pypto.cast(mm_add, calc_dtype)
    mm_deq_scale = pypto.mul(mm_fp32, deq_scale_2d)
    mm_bf16 = pypto.cast(mm_deq_scale, input_dtype)
    pypto.set_vec_tile_shapes(bs_tile, head_size)
    return mm_bf16


def _split_qkv(mm_bf16, bs_tile, total_head_size, head_size,
                q_num_head, kv_num_head, kv_index, act_bs_tile):
    """Split matmul output into q, k, v tiles."""
    mm_3d = pypto.reshape(mm_bf16, [bs_tile, total_head_size // head_size, head_size], inplace=True)
    q_tile = pypto.view(mm_3d, [bs_tile, q_num_head, head_size], [0, 0, 0],
        valid_shape=[act_bs_tile, q_num_head, head_size])
    k_tile = pypto.view(mm_3d, [bs_tile, kv_num_head, head_size], [0, q_num_head, 0],
        valid_shape=[act_bs_tile, kv_num_head, head_size])
    v_tile = pypto.view(mm_3d, [bs_tile, kv_num_head, head_size], [0, kv_index, 0],
        valid_shape=[act_bs_tile, kv_num_head, head_size])
    return q_tile, k_tile, v_tile


def _apply_rope(q_rot, q_pass, k_rot, k_pass, cos, sin,
                bs_idx, bs_tile, act_bs_tile,
                q_num_head, kv_num_head, half_rotary_dim,
                calc_dtype, q_batch_tile, head_size):
    """Apply RoPE to q and k tiles."""
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
    return q_cat, k_cat


def _init_kernel_cfg(x, weight, q_gamma, cos, q, k, head_size):
    """Initialize kernel configuration parameters from tensor shapes."""
    bs_tile = 8
    bs = x.shape[0]
    q_size = q.shape[-1]
    kv_size = k.shape[-1]
    half_rotary_dim = cos.shape[-1]
    x_mean_coff = 1.0 / x.shape[-1]
    qk_mean_coff = 1.0 / head_size
    eps = 1e-05
    rotary_dim = half_rotary_dim * 2
    stay_dim = head_size - rotary_dim
    q_num_head = q_size // head_size
    kv_num_head = kv_size // head_size
    kv_index = q_num_head + kv_num_head
    bs_loop = (bs + bs_tile - 1) // bs_tile
    return {
        "bs_tile": bs_tile, "bs_loop": bs_loop,
        "half_rotary_dim": half_rotary_dim,
        "q_size": q_size, "kv_size": kv_size,
        "x_mean_coff": x_mean_coff, "qk_mean_coff": qk_mean_coff,
        "eps": eps, "rotary_dim": rotary_dim, "stay_dim": stay_dim,
        "q_num_head": q_num_head, "kv_num_head": kv_num_head, "kv_index": kv_index,
    }


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
    """JIT fused kernel: RMSNorm(x+residual) → quantize → QKV matmul → split → QK norm → RoPE."""
    hidden_size = x.shape[1]
    total_head_size = weight.shape[1]
    head_size = q_gamma.shape[0]
    bs = x.shape[0]
    half_rotary_dim = cos.shape[-1]
    q_size = q.shape[-1]
    kv_size = k.shape[-1]

    cfg = _init_kernel_cfg(x, weight, q_gamma, cos, q, k, head_size)
    bs_tile = cfg["bs_tile"]
    bs_loop = cfg["bs_loop"]
    calc_dtype = pypto.DT_FP32
    input_dtype = x.dtype
    tiling_value = 128
    vec_tile_value = 5120
    q_batch_tile = 4

    q_gamma_expand, q_bias_expand, k_gamma_expand, k_bias_expand = \
        _init_norm_weights(q_gamma, q_bias, k_gamma, k_bias, cfg["q_num_head"], cfg["kv_num_head"],
                           head_size, hidden_size, calc_dtype)

    # 4. 定义动态函数
    pypto.set_vec_tile_shapes(vec_tile_value)
    x_gamma_2d = pypto.reshape(x_gamma, [1, hidden_size], inplace=True)
    x_bias_2d = pypto.reshape(x_bias, [1, hidden_size], inplace=True)
    x_scale_2d = pypto.reshape(x_scale, [1, hidden_size], inplace=True)
    x_offset_2d = pypto.reshape(x_offset, [1, hidden_size], inplace=True)
    quant_bias_2d = pypto.reshape(quant_bias, [1, total_head_size], inplace=True)
    deq_scale_2d = pypto.reshape(deq_scale, [1, total_head_size], inplace=True)

    # 5. 实现kernel逻辑，循环展开BS动态轴
    for bs_idx in pypto.loop(cfg["bs_loop"], name="LOOP_ATT_PRE_L0", idx_name="bs_idx"):
        _process_bs_tile(
            x, residual_input, x_gamma_2d, x_bias_2d, x_scale_2d, x_offset_2d,
            weight, quant_bias_2d, deq_scale_2d,
            q_gamma_expand, q_bias_expand, k_gamma_expand, k_bias_expand,
            cos, sin, q, k, v, residual,
            bs_idx, cfg["bs_tile"], bs, hidden_size, total_head_size, head_size,
            calc_dtype, input_dtype, vec_tile_value, tiling_value, q_batch_tile,
            cfg["x_mean_coff"], cfg["eps"], cfg["qk_mean_coff"],
            cfg["q_num_head"], cfg["kv_num_head"], cfg["kv_index"],
            cfg["rotary_dim"], cfg["stay_dim"], cfg["half_rotary_dim"], cfg["q_size"], cfg["kv_size"])


def _generate_test_data(device_id, bs, hidden_size, total_head_size, head_size,
                         q_size, kv_size, half_rotary_dim):
    """Generate test input tensors on NPU."""
    np.random.seed(0)
    x = torch.rand(bs, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    residual_input = torch.rand(bs, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    x_gamma = torch.rand(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    x_bias = torch.rand(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    x_scale = torch.rand(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    x_offset = torch.rand(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    weight = torch.randint(-128, 128, size=(hidden_size, total_head_size), dtype=torch.int8,
        device=f'npu:{device_id}')
    quant_bias = torch.randint(-128, 128, size=(total_head_size,), dtype=torch.int32, device=f'npu:{device_id}')
    deq_scale = torch.rand(total_head_size, dtype=torch.float32, device=f'npu:{device_id}')
    q_gamma = torch.rand(head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    q_bias = torch.rand(head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    k_gamma = torch.rand(head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    k_bias = torch.rand(head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    cos = torch.rand(bs, 1, half_rotary_dim, dtype=torch.bfloat16, device=f'npu:{device_id}')
    sin = torch.rand(bs, 1, half_rotary_dim, dtype=torch.bfloat16, device=f'npu:{device_id}')
    query = torch.rand(bs, q_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    key = torch.rand(bs, kv_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    value = torch.rand(bs, kv_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    residual_res = torch.rand(bs, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    return (x, residual_input, x_gamma, x_bias, x_scale, x_offset, weight,
            quant_bias, deq_scale, q_gamma, q_bias, k_gamma, k_bias,
            cos, sin, query, key, value, residual_res)


def _compute_golden(x, residual_input, x_gamma, x_bias, x_scale, x_offset,
                    weight, quant_bias, deq_scale, q_gamma, q_bias, k_gamma, k_bias,
                    cos, sin, eps, q_size, kv_size, head_size, bs, rotary_dim):
    """Compute golden reference for attention pre quant."""
    x_g, residual_g = add_rms_norm_npu_golden(x, residual_input, x_gamma, x_bias, eps)
    x_quant = torch_npu.npu_quantize(x_g, x_scale, x_offset, torch.qint8, -1, False)
    mm_golden = torch_npu.npu_quant_matmul(x_quant, weight, deq_scale,
                                           bias=quant_bias, output_dtype=torch.bfloat16)
    q_g, k_g, v_g = mm_golden.split([q_size, kv_size, kv_size], dim=-1)
    q_by_head = q_g.view(*q_g.shape[:-1], q_g.shape[-1] // head_size, head_size)
    q_by_head = rms_norm_npu_golden(q_by_head, q_gamma, q_bias, eps)
    k_by_head = k_g.view(*k_g.shape[:-1], k_g.shape[-1] // head_size, head_size)
    k_by_head = rms_norm_npu_golden(k_by_head, k_gamma, k_bias, eps)
    q_rot = q_by_head[..., :rotary_dim]
    q_pass = q_by_head[..., rotary_dim:]
    k_rot = k_by_head[..., :rotary_dim]
    k_pass = k_by_head[..., rotary_dim:]
    q_r, k_r = apply_rotary_pos_emb_v2(q_rot, k_rot, cos, sin)
    q_cat = torch.cat((q_r, q_pass), dim=-1)
    k_cat = torch.cat((k_r, k_pass), dim=-1)
    q_r = q_cat.view(bs, q_size)
    k_r = k_cat.view(bs, kv_size)
    return residual_g, q_r, k_r, v_g


@pytest.mark.soc("950", "910")
def test_quant_attention_pre():
    bs = 8
    hidden_size = 5120
    total_head_size = 1792
    head_size = 128
    q_size = 1536
    kv_size = 128
    rotary_dim = 64
    half_rotary_dim = rotary_dim // 2
    eps = 1e-05

    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    for i in range(0, 1):
        if i == 1:
            bs = 5
        elif i == 2:
            bs = 11
        elif i == 3:
            bs = 2

        (x, residual_input, x_gamma, x_bias, x_scale, x_offset, weight,
         quant_bias, deq_scale, q_gamma, q_bias, k_gamma, k_bias,
         cos, sin, query, key, value, residual_res) = \
            _generate_test_data(device_id, bs, hidden_size, total_head_size, head_size,
                                q_size, kv_size, half_rotary_dim)

        inputs = [x, residual_input, x_gamma, x_bias, x_scale, x_offset,
                  torch_npu.npu_format_cast(weight, 29),
                  quant_bias, deq_scale, q_gamma, q_bias, k_gamma, k_bias,
                  cos, sin, query, key, value, residual_res]
        attention_pre_quant(*inputs)

        residual_g, q_r, k_r, v_g = _compute_golden(
            x, residual_input, x_gamma, x_bias, x_scale, x_offset,
            weight, quant_bias, deq_scale, q_gamma, q_bias, k_gamma, k_bias,
            cos, sin, eps, q_size, kv_size, head_size, bs, rotary_dim)

        assert_allclose(np.array(residual_g.cpu().flatten().tolist()),
                        np.array(residual_res.cpu().flatten().tolist()),
                        rtol=0.0078125, atol=0.0001)
        assert_allclose(np.array(q_r.cpu().flatten().tolist()),
                        np.array(query.cpu().flatten().tolist()),
                        rtol=0.0078125, atol=0.0001)
        assert_allclose(np.array(k_r.cpu().flatten().tolist()),
                        np.array(key.cpu().flatten().tolist()),
                        rtol=0.0078125, atol=0.0001)
        assert_allclose(np.array(v_g.cpu().flatten().tolist()),
                        np.array(value.cpu().flatten().tolist()),
                        rtol=0.0078125, atol=0.0001)
        logging.info("PASS")


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
        residual_res
    )

    bs = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    total_head_size = atten_qkv_weight.shape[1]
    head_size = atten_q_norm_weight.shape[0]
    q_size = query.shape[1]
    kv_size = key.shape[1]
    half_rotary_dim = cos.shape[2]
    inputs = [hidden_states, residual, input_layernorm_weight, input_layernorm_bias, atten_qkv_input_scale_reciprocal,
         atten_qkv_input_offset, atten_qkv_weight, atten_qkv_quant_bias, atten_qkv_deq_scale, atten_q_norm_weight,
         atten_q_norm_bias, atten_k_norm_weight, atten_k_norm_bias, cos, sin, query, key, value, residual_res]
    quant_attention_pre_kernel(*inputs)


def main():
    test_quant_attention_pre()


if __name__ == "__main__":
    main()
