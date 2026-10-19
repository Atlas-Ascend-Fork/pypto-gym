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
Lightning Indexer Prolog Quantization Module

This module implements the Lightning Indexer Prolog quantization computation
for DeepSeek V32 model. It handles:
- Query computation with dynamic quantization
- Key computation with LayerNorm and RoPE
- Weight computation for indexer attention

Main Functions:
    - lightning_indexer_prolog_quant_compute: Main computation function
    - quant_layer_norm: Quantized LayerNorm implementation
    - prolog_quant: Per-token quantization function
    - quant_rope_2d: 2D RoPE (Rotary Position Embedding) computation
    - rope_3d: 3D RoPE computation

Example:
    See tests/ops/deepseek_v32_exp/test_lightning_indexer_prolog_quant.py for usage examples.
"""
import math
from typing import List
from dataclasses import dataclass
import torch
import pypto

SHAPE_DIM_2 = 2
SHAPE_DIM_3 = 3

NUM_0 = 0
NUM_1 = 1
NUM_2 = 2
NUM_3 = 3
NUM_7168 = 7168

TILE_CUBE_DIM = 6
Q_PARAM_DIM = 2
NZ_DIM = 4
COS_SIN_DIM = 2
L0M_INDEX = 0
L1M_INDEX = 1
L0K_INDEX = 2
L1K_INDEX = 3
L0N_INDEX = 4
L1N_INDEX = 5
SCATTER_DIM = -2
NZ_FIRST_DIM = 16
NZ_B8_C0 = 32
NZ_B16_C0 = 16

VEC_TILE_256 = 256
VEC_TILE_128 = 128
VEC_TILE_64 = 64
VEC_TILE_8 = 8
VEC_TILE_4 = 4
VEC_TILE_32 = 32


@dataclass
class IndexerPrologQuantInput:
    x: torch.tensor  # BF16, (t, h)
    q_norm: torch.tensor  # INT8, (t, qLoraRank)
    q_norm_scale: torch.tensor  # FP32, (t, 1)
    w_qb: torch.tensor  # INT8, (headNum * headDim // NZ_B8_C0, qLoraRank // NZ_FIRST_DIM, NZ_FIRST_DIM, NZ_B8_C0), NZ
    w_qb_scale: torch.tensor  # FP32, (headNum * headDim, 1)
    wk: torch.tensor  # BF16, (headDim // NZ_B16_C0, h // NZ_FIRST_DIM, NZ_FIRST_DIM, NZ_B16_C0), NZ
    w_proj: torch.tensor  # BF16, (headNum // NZ_B16_C0, h // NZ_FIRST_DIM, NZ_FIRST_DIM, NZ_B16_C0), NZ
    ln_gamma_k: torch.tensor  # BF16, (headDim,)
    ln_beta_k: torch.tensor  # BF16, (headDim,)
    cos_idx_rope: torch.tensor  # BF16, (t, ropeHeadDim)
    sin_idx_rope: torch.tensor  # BF16, (t, ropeHeadDim)
    hadamard_q: torch.tensor  # BF16, (headDim, headDim)
    hadamard_k: torch.tensor  # BF16, (headDim, headDim)
    k_cache: torch.tensor  # INT8, (blockNum, blockSize, nKv, headDim)
    k_cache_scale: torch.tensor  # FP16, (blockNum, blockSize, nKv, 1)
    k_cache_index: torch.tensor  # INT64, (t,)


@dataclass
class IndexerPrologQuantOutput:
    q_int8: torch.tensor
    q_scale: torch.tensor
    k_int8: torch.tensor
    k_scale: torch.tensor
    weights: torch.tensor


@dataclass
class IndexerPrologQuantAttr:
    eps: float
    layerout_query: str
    layerout_key: str


@dataclass
class IndexerPrologQuantConfigs:
    q_linear: List[int]
    q_hd: List[int]
    k_linear: List[int]
    w_linear: List[int]
    unroll_list: List[int]

    cube_l1_reuse_setting: dict[int, int]
    block_size: int
    t_sub_tile: int
    chunk_size: int
    vec_nbuffer_setting: dict[int, int]


def quant_layer_norm(x: pypto.Tensor, gamma: pypto.Tensor, beta: pypto.Tensor, dim: int, epsilon: float):
    """Compute quantized LayerNorm operation.

    Applies Layer Normalization with quantization support. The function normalizes
    the input tensor along the specified dimension using mean and variance,
    then applies learnable scale (gamma) and shift (beta) parameters.

    Args:
        x: Input tensor to normalize, shape depends on input
        gamma: Scale parameter tensor, shape should match the normalization dimension
        beta: Shift parameter tensor, shape should match the normalization dimension
        dim: Dimension along which to normalize. Can be -1 (last dimension) or
             len(x.shape) - 1 (last dimension explicitly)
        epsilon: Small constant added to variance to avoid division by zero

    Returns:
        Normalized tensor with the same shape as input x, with scale and shift applied

    Note:
        The function performs normalization in FP32 precision to maintain numerical
        stability, then casts back to the original dtype.
    """
    pypto.set_semantic_label("Key-LayerNorm")
    actual_dim = dim + len(x.shape) if dim < 0 else dim
    x_dtype = x.dtype

    x_fp32 = pypto.cast(x, pypto.DT_FP32)
    # do division first to avoid overflow
    x_scaled = x_fp32 * (1.0 / x.shape[actual_dim])
    mean = pypto.sum(x_scaled, actual_dim, keepdim=True)

    diff = x_fp32 - mean
    squared_diff = diff * diff
    squared_diff_scaled = squared_diff * (1.0 / x.shape[actual_dim])
    var = pypto.sum(squared_diff_scaled, actual_dim, keepdim=True)
    # add epsilon to avoid division by zero
    var_eps = var + epsilon
    std_var = pypto.sqrt(var_eps)
    res32 = diff / std_var

    gamma32 = pypto.cast(gamma, pypto.DT_FP32)
    beta32 = pypto.cast(beta, pypto.DT_FP32)
    return pypto.cast((res32 * gamma32) + beta32, x_dtype)


def quant_rope_2d(x: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor):
    """Apply 2D Rotary Position Embedding (RoPE) to input tensor.

    Implements RoPE transformation for 2D tensors. RoPE encodes positional
    information by rotating the input tensor using cosine and sine values.

    Args:
        x: Input tensor of shape (t_tile, rope_dim), where t_tile is the
           sequence length and rope_dim is the RoPE dimension
        cos: Cosine values for RoPE, shape (t_tile, rope_dim)
        sin: Sine values for RoPE, shape (t_tile, rope_dim)

    Returns:
        Tensor with RoPE applied, same shape as input x

    Note:
        The function performs rotation in FP32 precision for numerical stability,
        then casts back to the original dtype.
    """
    pypto.set_semantic_label("Key-Rope2D")
    _key_rope_dim = 2
    x_dtype = x.dtype
    t_tile = x.shape[0]
    rope_dim = x.shape[1]

    pypto.set_vec_tile_shapes(t_tile, rope_dim)
    cast_cos = pypto.cast(cos, pypto.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DT_FP32)
    x_view = pypto.cast(x, pypto.DT_FP32)

    pypto.set_vec_tile_shapes(t_tile, rope_dim)
    x_embed = (x_view * cast_cos) + ((rotate_half(x_view)) * cast_sin)
    res = pypto.cast(x_embed, x_dtype)
    return res


def prolog_quant(x: pypto.Tensor):
    """Perform per-token quantization to INT8.

    Quantizes the input tensor to INT8 format using dynamic quantization.
    The quantization scale is computed per-token based on the maximum absolute
    value, ensuring the full INT8 range [-127, 127] is utilized.

    Args:
        input: Input tensor to quantize, can be any shape. Quantization is
               performed along the last dimension per token.

    Returns:
        Tuple of (quantized_tensor, dequant_scale):
            - quantized_tensor: INT8 quantized tensor, same shape as input
            - dequant_scale: FP32 scale factor for dequantization, shape matches
                            input with last dimension reduced to 1

    Note:
        The quantization process:
        1. Find per-token maximum absolute value
        2. Compute scale = 127.0 / max_value
        3. Quantize: int8 = round(input * scale)
        4. Return dequantization scale = 1.0 / scale
    """
    pypto.set_semantic_label("Prolog-Quant")
    s8_max_value = 127.0
    s8_one_value = 1.0
    input_fp32 = pypto.cast(x, pypto.DT_FP32, pypto.CastMode.CAST_NONE)

    abs_res = pypto.abs(input_fp32)
    max_value = pypto.amax(abs_res, dim=-1, keepdim=True)
    temp127 = pypto.full(max_value.shape, s8_max_value, pypto.DT_FP32)

    scale_quant = temp127 / max_value
    out_fp32 = input_fp32 * scale_quant
    out_int32 = pypto.cast(out_fp32, pypto.DT_INT32, pypto.CastMode.CAST_RINT)
    out_half = pypto.cast(out_int32, pypto.DT_FP16, pypto.CastMode.CAST_ROUND)
    out_int8 = pypto.cast(out_half, pypto.DT_INT8, pypto.CastMode.CAST_TRUNC, satmode=pypto.SaturationMode.ON)
    temp1 = pypto.full(scale_quant.shape, s8_one_value, pypto.DT_FP32)
    scale_dequant = temp1 / scale_quant
    return (out_int8, scale_dequant)


def rotate_half(input_tensor: pypto.Tensor) -> pypto.Tensor:
    """Rotate half of the tensor dimensions for RoPE computation.

    Splits the last dimension in half and applies rotation transformation:
    [-x2, x1] where x1 is the first half and x2 is the second half.
    This is a key component of RoPE (Rotary Position Embedding).

    Args:
        input_tensor: Input tensor with last dimension divisible by 2

    Returns:
        Rotated tensor with same shape as input, where the first half of
        the last dimension is negated and swapped with the second half

    Raises:
        AssertionError: If the last dimension is not divisible by 2

    Example:
        If input is [a, b, c, d] along last dim, output is [-c, -d, a, b]
    """
    chunk_size = 2
    shape = input_tensor.shape
    shape_size = len(shape)
    shape[shape_size - 1] //= chunk_size
    offset1 = [0] * shape_size
    offset2 = [0] * shape_size
    offset2[shape_size - 1] = shape[shape_size - 1]
    x1 = pypto.view(input_tensor, shape, offset1)
    x2 = pypto.view(input_tensor, shape, offset2)
    return pypto.concat([x2 * (-1.0), x1 + 0.0], -1)


def rope_3d(x: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor, configs: IndexerPrologQuantConfigs) -> pypto.Tensor:
    """Apply 3D Rotary Position Embedding (RoPE) to input tensor.

    Implements RoPE transformation for 3D tensors with shape (t_tile, head_num, rope_dim).
    The RoPE is applied independently to each head using the provided cosine and sine values.

    Args:
        x: Input tensor of shape (t_tile, head_num, rope_dim)
        cos: Cosine values for RoPE, shape (t_tile, rope_dim)
        sin: Sine values for RoPE, shape (t_tile, rope_dim)
        configs: Configuration object containing tiling parameters:
            - t_sub_tile: Sub-tile size for t dimension
            - chunk_size: Chunk size for head dimension processing

    Returns:
        Tensor with RoPE applied, same shape as input x

    Note:
        The function broadcasts cos and sin to match the head dimension,
        then applies rotation: x_rotated = x * cos + rotate_half(x) * sin
    """
    head_num_axis = 1
    head_dim_axis = 2

    x_dtype = x.dtype
    t_tile = x.shape[0]
    head_num = x.shape[head_num_axis]
    rope_dim = x.shape[head_dim_axis]

    pypto.set_vec_tile_shapes(1, rope_dim)
    cast_cos = pypto.cast(cos, pypto.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DT_FP32)

    pypto.set_vec_tile_shapes(configs.t_sub_tile, head_num // configs.chunk_size, rope_dim)
    x_view = pypto.cast(x, pypto.DT_FP32)
    cast_cos = pypto.reshape(cast_cos, [t_tile, 1, rope_dim])
    cast_sin = pypto.reshape(cast_sin, [t_tile, 1, rope_dim])

    x_embed = (x_view * cast_cos) + ((rotate_half(x_view)) * cast_sin)
    res = pypto.cast(x_embed, x_dtype)
    return res


@pypto.frontend.jit(
    pass_options={"cube_l1_reuse_setting": {"Query-Linear": 4}},
    runtime_options={"stitch_function_max_num": 128,
                    "device_sched_mode": 1}
)
def lightning_indexer_prolog_quant(
    x_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    q_norm_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT8),
    q_norm_scale_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    w_qb_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT8, format=pypto.TileOpFormat.TILEOP_NZ),
    w_qb_scale_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    wk_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_NZ),
    w_proj_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_NZ),
    ln_gamma_k_in: pypto.Tensor([pypto.STATIC], pypto.DT_BF16),
    ln_beta_k_in: pypto.Tensor([pypto.STATIC], pypto.DT_BF16),
    cos_idx_rope_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    sin_idx_rope_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    hadamard_q_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    hadamard_k_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    k_int8_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    k_scale_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    k_cache_index_in: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT64),
    q_int8_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    q_scale_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    k_int8_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    k_scale_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    weights_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP16),

    configs,
    attrs
):
    """Compute Lightning Indexer Prolog with quantization.
        Main computation function for Lightning Indexer Prolog quantization.
        This function processes input tokens to generate quantized query, key, and weights
        for the indexer attention mechanism. The computation includes:

        1. Query Path:
        - Dequantize q_norm (INT8) to FP32
        - Apply linear transformation with w_qb
        - Apply RoPE (Rotary Position Embedding)
        - Apply Hadamard transformation
        - Quantize to INT8 with per-token-head scale

        2. Key Path:
        - Linear transformation with wk
        - LayerNorm normalization
        - Apply RoPE
        - Apply Hadamard transformation
        - Quantize to INT8 with per-token-head scale
        - Update key cache using scatter_update

        3. Weights Path:
        - Linear transformation with w_proj
        - Normalize by sqrt(head_num * head_dim)
        - Convert to FP16

        Args:
            x_in: Input hidden states tensor, shape (t, h), dtype BF16
            q_norm_in: Quantized query norm tensor, shape (t, q_lora_rank), dtype INT8
            q_norm_scale_in: Query norm dequantization scale, shape (t, 1), dtype FP32
            w_qb_in: Query projection weight matrix, INT8 format with NZ layout
            w_qb_scale_in: Query weight dequantization scale, shape (head_num * head_dim, 1), dtype FP32
            wk_in: Key projection weight matrix, BF16 format with NZ layout
            w_proj_in: Weight projection matrix, BF16 format with NZ layout
            ln_gamma_k_in: LayerNorm scale parameter for key, shape (head_dim,), dtype BF16
            ln_beta_k_in: LayerNorm shift parameter for key, shape (head_dim,), dtype BF16
            cos_idx_rope_in: Cosine values for RoPE, shape (t, rope_head_dim), dtype BF16
            sin_idx_rope_in: Sine values for RoPE, shape (t, rope_head_dim), dtype BF16
            hadamard_q_in: Hadamard transformation matrix for query, shape (head_dim, head_dim), dtype BF16
            hadamard_k_in: Hadamard transformation matrix for key, shape (head_dim, head_dim), dtype BF16
            k_int8_in: Input key cache, shape (block_num, block_size, n_kv, head_dim), dtype INT8
            k_scale_in: Key cache scale, shape (block_num, block_size, n_kv, 1), dtype FP16
            k_cache_index_in: Cache index for scatter update, shape (t,), dtype INT64
            q_int8_out: Output quantized query tensor, shape (t, head_num, head_dim), dtype INT8
            q_scale_out: Output query quantization scale, shape (t, head_num, 1), dtype FP16
            k_int8_out: Output key cache (updated in-place), shape (block_num, block_size, n_kv, head_dim), dtype INT8
            k_scale_out: Output key cache scale (updated in-place), shape (block_num, block_size, n_kv, 1), dtype FP16
            weights_out: Output weights tensor, shape (t, head_num), dtype FP16
            attrs: IndexerPrologQuantAttr object containing:
                - eps: LayerNorm epsilon value
                - layerout_query: Query layout format (e.g., "TND")
                - layerout_key: Key layout format (e.g., "PA_BSND")
            configs: IndexerPrologQuantConfigs object containing tiling and optimization parameters

        Note:
            - The function processes tokens in tiles using loop_unroll for optimization
            - All outputs are written in-place using pypto.assemble or scatter_update
            - The computation uses dynamic tiling based on configs.unroll_list
    """
    x_dtype = x_in.dtype
    # 动态轴
    t = x_in.shape[0]
    h = x_in.shape[1]
    q_lora_rank = q_norm_in.shape[1]
    head_num = w_proj_in.shape[1]
    head_dim = hadamard_q_in.shape[0]
    rope_head_dim = cos_idx_rope_in.shape[1]

    k_cache_index = pypto.reshape(k_cache_index_in, [t, 1], inplace=True)
    w_qb_scale = pypto.reshape(w_qb_scale_in, [1, head_num * head_dim], inplace=True)
    gamma_2d = pypto.reshape(ln_gamma_k_in, [1, ln_gamma_k_in.shape[0]], inplace=True)
    beta_2d = pypto.reshape(ln_beta_k_in, [1, ln_beta_k_in.shape[0]], inplace=True)

    unroll_list = configs.unroll_list
    for t_idx, unroll_length in pypto.loop_unroll(0, t, 1, name="IndexerPrologQuantQuantLoop", idx_name="tIdx",
                                                unroll_list=unroll_list, ):
        t_tile = unroll_length
        # 获取query计算的各阶段Tile参数
        q_linear = configs.q_linear
        q_hd = configs.q_hd
        # 多分档内会将t_tile作为档位，offset无需乘t_tile
        q_norm = pypto.view(q_norm_in, [t_tile, q_lora_rank], [t_idx, 0], valid_shape=[t_tile, q_lora_rank])
        q_norm_scale = pypto.view(q_norm_scale_in, [t_tile, 1], [t_idx, 0], valid_shape=[t_tile, 1])
        pypto.set_semantic_label("Query-Linear")
        pypto.set_cube_tile_shapes([q_linear[L0M_INDEX], q_linear[L1M_INDEX]],
                                [q_linear[L0K_INDEX], q_linear[L1K_INDEX]],
                                [q_linear[L0N_INDEX], q_linear[L1N_INDEX]])
        q_s32 = pypto.matmul(q_norm, w_qb_in, pypto.DT_INT32)  # (t_tile, head_num * head_dim)

        pypto.set_semantic_label("Query-Dequant")

        pypto.set_vec_tile_shapes(configs.t_sub_tile, head_num * head_dim // configs.chunk_size)
        q_f32 = pypto.cast(q_s32, pypto.DT_FP32)  # (t_tile, head_num * head_dim), fp32
        q_f32 = q_f32 * q_norm_scale  # (t_tile, head_num * head_dim), fp32
        q_f32 = q_f32 * w_qb_scale  # (t_tile, head_num * head_dim), fp32
        q_cast = pypto.cast(q_f32, x_dtype)

        q_bf16 = pypto.reshape(q_cast, [t_tile, head_num, head_dim], valid_shape=[t_tile, head_num, head_dim])
        # UB view
        q_rope = pypto.view(q_bf16, [t_tile, head_num, rope_head_dim], [0, 0, 0],
                            valid_shape=[t_tile, head_num, rope_head_dim])
        q_nope = pypto.view(q_bf16, [t_tile, head_num, head_dim - rope_head_dim], [0, 0, rope_head_dim],
                            valid_shape=[t_tile, head_num, head_dim - rope_head_dim])
        rope_cos = pypto.view(cos_idx_rope_in, [t_tile, rope_head_dim], [t_idx, 0],
                            valid_shape=[t_tile, rope_head_dim])
        rope_sin = pypto.view(sin_idx_rope_in, [t_tile, rope_head_dim], [t_idx, 0],
                            valid_shape=[t_tile, rope_head_dim])

        q_roped = rope_3d(q_rope, rope_cos, rope_sin, configs)  # [t_tile, head_num, rope_head_dim]
        pypto.set_vec_tile_shapes(configs.t_sub_tile, head_num // configs.chunk_size, head_dim)
        q_nope = pypto.cast(pypto.cast(q_nope, pypto.DT_FP32), q_bf16.dtype)
        q_cat = pypto.concat([q_roped, q_nope], -1)  # [t_tile, head_num, head_dim]
        hadamard_q = pypto.reshape(hadamard_q_in, [1, head_dim, head_dim], valid_shape=[1, head_dim, head_dim])

        pypto.set_semantic_label("Query-Hadamard")
        cur_max_unroll = 32
        q_hd_m_tile = cur_max_unroll if t_tile < cur_max_unroll else q_hd[L0M_INDEX]
        pypto.set_cube_tile_shapes([q_hd_m_tile, q_hd_m_tile], [q_hd[L0K_INDEX], q_hd[L1K_INDEX]],
                                [q_hd[L0N_INDEX], q_hd[L1N_INDEX]])
        q_hadamard = pypto.matmul(q_cat, hadamard_q, x_dtype)  # (t_tile, head_num, head_dim)

        pypto.set_semantic_label("Query-Quant")
        pypto.set_vec_tile_shapes(configs.t_sub_tile, head_num // configs.chunk_size, head_dim)
        q_res = prolog_quant(q_hadamard)
        q_scale = pypto.cast(q_res[1], pypto.DT_FP16)

        pypto.assemble(q_res[0], [t_idx, 0, 0], q_int8_out)
        pypto.assemble(q_scale, [t_idx, 0, 0], q_scale_out)

        # 获取key计算的各阶段Tile参数
        k_linear = configs.k_linear
        pypto.set_semantic_label("Key-Linear")
        pypto.set_cube_tile_shapes([k_linear[L0M_INDEX], k_linear[L1M_INDEX]],
                                [k_linear[L0K_INDEX], k_linear[L1K_INDEX]],
                                [k_linear[L0N_INDEX], k_linear[L1N_INDEX]])
        x = pypto.view(x_in, [t_tile, h], [t_idx, 0], valid_shape=[t_tile, h])  # 这里将t_tile分档，offset不需要乘t_tile
        k = pypto.matmul(x, wk_in, pypto.DT_FP32)  # (t_tile, head_dim)

        if t_tile <= 32:
            pypto.set_vec_tile_shapes(min(t_tile, VEC_TILE_4), head_dim)
        else:
            pypto.set_vec_tile_shapes(min(t_tile, VEC_TILE_32), head_dim)
        k_bf16 = pypto.cast(quant_layer_norm(k, gamma_2d, beta_2d, -1, attrs.eps), x_dtype)

        k_rope = pypto.view(k_bf16, [t_tile, rope_head_dim], [0, 0], valid_shape=[t_tile, rope_head_dim])
        k_nope = pypto.view(k_bf16, [t_tile, head_dim - rope_head_dim], [0, rope_head_dim],
                            valid_shape=[t_tile, head_dim - rope_head_dim])
        k_roped = quant_rope_2d(k_rope, rope_cos, rope_sin)  # (t_tile, rope_head_dim)
        pypto.set_vec_tile_shapes(t_tile, head_dim)
        k_nope = pypto.cast(pypto.cast(k_nope, pypto.DT_FP32), k_bf16.dtype)
        k_concat = pypto.concat([k_roped, k_nope], -1)
        pypto.set_semantic_label("Key-Hadamard")
        hadamard_k = pypto.matmul(k_concat, hadamard_k_in, x_dtype)  # (t_tile, head_dim), bf16
        pypto.set_semantic_label("Key-Quant")
        k_res = prolog_quant(hadamard_k)
        k_cache_4d = pypto.reshape(k_res[0], [t_tile, 1, 1, head_dim], valid_shape=[t_tile, 1, 1, head_dim])
        k_scale_4d = pypto.reshape(pypto.cast(k_res[1], pypto.DT_FP16), [t_tile, 1, 1, 1],
                                valid_shape=[t_tile, 1, 1, 1])

        index = pypto.view(k_cache_index, [t_tile, 1], [t_idx, 0], valid_shape=[t_tile, 1])
        pypto.set_vec_tile_shapes(t_tile, 1, 1, head_dim)
        k_int8_out.move(pypto.scatter_update(k_int8_in, SCATTER_DIM, index, k_cache_4d))
        k_scale_out.move(pypto.scatter_update(k_scale_in, SCATTER_DIM, index, k_scale_4d))

        pypto.set_semantic_label("Weight-Linear")
        w_linear = configs.w_linear
        pypto.set_cube_tile_shapes([w_linear[L0M_INDEX], w_linear[L1M_INDEX]],
                                [w_linear[L0K_INDEX], w_linear[L1K_INDEX]],
                                [w_linear[L0N_INDEX], w_linear[L1N_INDEX]])
        pypto.set_vec_tile_shapes(t_tile, head_num)
        weights = pypto.cast(pypto.matmul(x, w_proj_in, x_dtype), pypto.DT_FP32)
        weights = pypto.mul(weights, 1.0 / (math.sqrt(head_num) * math.sqrt(head_dim)))
        weights_f16 = pypto.cast(weights, pypto.DT_FP16)
        pypto.assemble(weights_f16, [t_idx, 0], weights_out)
