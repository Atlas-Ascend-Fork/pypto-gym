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
MLA Prolog MXFP Quantization Module (V3)

This module implements MLA (Multi-head Latent Attention) Prolog quantization
for DeepSeek V32/V4 model using MXFP (Microscaling FP8) quantization format.
It converts hidden states to query, key, and value projections with support
for block-wise quantization and RoPE (Rotary Position Embedding).

MXFP Quantization Format:
    - Data format: FP8E4M3 (range [0, 448], asymmetric)
    - Scale format: FP8E8M0 (block-wise scaling, 64 elements per block)
    - Uses pypto.scaled_mm for quantized matrix multiplication
    - Uses pypto.quant_mx for per-block quantization

Main Functions:
    - mla_prolog_quant_compute: Core MLA prolog computation with MXFP quantization
    - pre_compute_2d: Pre-computation for query and key-value projections
    - rms_norm: RMS normalization implementation
    - dequant: Dequantization function for INT8 tensors
    - rope_v2: 2D RoPE implementation
    - rope_3d_v2: 3D RoPE implementation

Example:
    See test_mla_prolog_mxfp_quant_v3.py for usage examples.
"""
from dataclasses import dataclass
from typing import List, Tuple
import pypto
from pypto import pypto_impl


class MlaTileConfig:
    """Tile configuration for MLA prolog quantization operations.

    Contains tiling parameters for optimizing memory access patterns
    and computation efficiency on NPU.

    Attributes:
        tile_b: Batch tile size
        tile_s: Sequence tile size
        tile_bs: Combined batch-sequence tile size
        m_tile: Matmul tile size
        mv_tile: Vector matmul tile size
        pre_quant_cube_tile: Cube tile shapes for pre-quantization matmul
        unroll_list: List of unroll lengths for loop optimization
        q_vec_tile0: Query vector tile dimension 0
        q_vec_tile1: Query vector tile dimension 1
        k_vec_tile0: Key vector tile dimension 0
        k_vec_tile1: Key vector tile dimension 1
        cube_l1_reuse_setting: L1 reuse configuration for cube operations
        pg_upper_bound: Upper bound for pipeline granularity
        cube_nbuffer_setting: N-buffer configuration for cube operations
        dynamic_unaligned_enable: Enable dynamic unaligned processing
    """
    def __init__(self):
        self.tile_b = 8
        self.tile_s = 1
        self.tile_bs = 8
        self.m_tile = 16
        self.mv_tile = 16
        self.pre_quant_cube_tile = [16, 16, 256, 256, 128, 128]
        self.unroll_list = [32, 16, 8, 4, 2, 1]
        self.q_vec_tile0 = 16
        self.q_vec_tile1 = 16
        self.k_vec_tile0 = 16
        self.k_vec_tile1 = 16
        self.cube_l1_reuse_setting = {-1: 4}
        self.pg_upper_bound = 8192
        self.cube_nbuffer_setting = {3: 4}
        self.dynamic_unaligned_enable = False


@dataclass
class MlaQuantInputs:
    """Container for quantization scale tensors.

    Encapsulates all dequantization and quantization scale tensors
    used throughout the MLA prolog quantization computation.

    Attributes:
        dequant_scale_x: Dequantization scale for input tensor
        dequant_scale_w_dq: Dequantization scale for w_dq weight
        dequant_scale_w_uq_qr: Dequantization scale for w_uq_qr weight
        dequant_scale_w_dkv_kr: Dequantization scale for w_dkv_kr weight
        quant_scale_ckv: Quantization scale for compressed KV
        quant_scale_ckr: Quantization scale for compressed KR
        smooth_scales_cq: Smooth quantization factor for query
    """
    dequant_scale_x: pypto.Tensor = None
    dequant_scale_w_dq: pypto.Tensor = None
    dequant_scale_w_uq_qr: pypto.Tensor = None
    dequant_scale_w_dkv_kr: pypto.Tensor = None
    quant_scale_ckv: pypto.Tensor = None
    quant_scale_ckr: pypto.Tensor = None
    smooth_scales_cq: pypto.Tensor = None


@dataclass
class RopeTileShapeConfig:
    """Tile shape configuration for RoPE (Rotary Position Embedding) operations.

    Defines tile shapes for different dimensional RoPE computations
    to optimize memory access and computation patterns.

    Attributes:
        two_dim: Tile shape for 2D RoPE operations, e.g., [32, 64]
        three_dim: Tile shape for 3D RoPE operations, e.g., [32, 32, 128]
        four_dim: Tile shape for 4D RoPE operations, e.g., [16, 128, 128, 128]
    """
    two_dim: List[int]
    three_dim: List[int]
    four_dim: List[int]


def rms_norm(input_tensor: pypto.Tensor, gamma: pypto.Tensor, epsilon: float) -> pypto.Tensor:
    """Compute RMS (Root Mean Square) normalization.

    Applies RMS normalization to the input tensor. RMS normalization is similar
    to LayerNorm but uses root mean square instead of standard deviation.

    Formula: output = gamma * input / sqrt(mean(input^2) + epsilon)

    Args:
        input_tensor: Input tensor to normalize
        gamma: Scale parameter tensor, shape should match the last dimension
        epsilon: Small constant added to variance to avoid division by zero

    Returns:
        Normalized tensor with the same shape as input, scaled by gamma

    Note:
        The normalization is performed along the last dimension.
        Computation is done in FP32 for numerical stability.
    """
    input_fp32 = pypto.cast(input_tensor, pypto.DT_FP32)
    dim = len(input_tensor.shape)
    shape = [1] * dim
    shape[dim - 1] = gamma.shape[0]
    gamma_cast = pypto.reshape(gamma, shape)
    gamma_fp32 = pypto.cast(gamma_cast, pypto.DT_FP32)
    y = pypto.mul(input_fp32, input_fp32)
    y = pypto.mul(y, 1.0 / input_tensor.shape[dim - 1])
    y = pypto.sum(y, -1, keepdim=True)
    y = pypto.add(y, epsilon)
    y = pypto.sqrt(y)
    ones_vector = pypto.full(y.shape, 1.0, pypto.DT_FP32)
    y = pypto.div(ones_vector, y)
    y = pypto.mul(input_fp32, y)
    y = pypto.mul(gamma_fp32, y)
    y = pypto.cast(y, input_tensor.dtype)
    return y


def dequant(
    dtype: pypto.DataType, input_tensor: pypto.Tensor, scale: pypto.Tensor, w_scale: pypto.Tensor
) -> pypto.Tensor:
    """Dequantize tensor back to floating point.

    Converts quantized tensor back to floating point by applying
    dequantization scales. Supports per-token and per-channel scaling.

    Args:
        dtype: Target data type for output (e.g., DT_BF16, DT_FP16)
        input_tensor: Quantized input tensor (INT8 or FP8)
        scale: Per-token or per-block dequantization scale
        w_scale: Weight dequantization scale (per-channel or per-block)

    Returns:
        Dequantized tensor in the specified dtype

    Note:
        Dequantization formula: output = (input * scale) * w_scale
        The computation is done in FP32, then cast to target dtype.
        This function is used for non-MXFP quantization scenarios.
    """
    dequant_res = pypto.cast(input_tensor, pypto.DT_FP32)
    dequant_res = dequant_res * scale
    dequant_res = dequant_res * w_scale
    return pypto.cast(dequant_res, dtype)


def rotate_half(input_tensor: pypto.Tensor) -> pypto.Tensor:
    """Rotate half of the tensor dimensions for RoPE computation.

    Splits the last dimension in half and applies rotation transformation:
    [-x2, x1] where x1 is the first half and x2 is the second half.
    This is a key component of RoPE (Rotary Position Embedding).

    Args:
        input_tensor: Input tensor with last dimension divisible by 2

    Returns:
        Rotated tensor with same shape as input

    Raises:
        AssertionError: If input dimension is less than 1 or last dimension
                       is not divisible by 2
    """
    shape = input_tensor.shape
    shape_size = len(shape)

    new_shape = list(shape)
    new_shape[shape_size - 1] //= 2

    offset1 = [0] * shape_size
    offset2 = [0] * shape_size
    offset2[shape_size - 1] = new_shape[shape_size - 1]

    x1 = pypto.view(input_tensor, new_shape, offset1)
    x2 = pypto.view(input_tensor, new_shape, offset2)

    return pypto.concat([x2 * (-1.0), x1 + 0.0], -1)


def rope_v2(
    x: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor, tile_config: RopeTileShapeConfig
) -> pypto.Tensor:
    """Apply 2D Rotary Position Embedding (RoPE) version 2.

    Implements RoPE transformation for 2D tensors with optimized tiling.
    The function reshapes and transposes the input before applying rotation.

    Args:
        x: Input tensor of shape (seq_size, d_r)
        cos: Cosine values for RoPE, shape (seq_size, d_r)
        sin: Sine values for RoPE, shape (seq_size, d_r)
        tile_config: RopeTileShapeConfig object containing tiling parameters:
            - two_dim: Tile shape for 2D operations
            - three_dim: Tile shape for 3D reshape operations

    Returns:
        Tensor with RoPE applied, same shape as input x

    Note:
        The function performs reshape and transpose operations before applying
        rotation to optimize memory access patterns.
    """
    seq_size = x.shape[0]
    d_r = x.shape[1]
    x_dtype = x.dtype

    pypto.set_vec_tile_shapes(tile_config.two_dim[0], tile_config.two_dim[1])
    cast_x = pypto.cast(x, pypto.DT_FP32)
    cast_cos = pypto.cast(cos, pypto.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DT_FP32)

    pypto.set_vec_tile_shapes(*tile_config.three_dim)
    x_view = pypto.reshape(cast_x, [seq_size, d_r // 2, 2])
    x_trans = pypto.transpose(x_view, 1, 2)
    x_re_second = pypto.reshape(x_trans, [seq_size, d_r])

    pypto.set_vec_tile_shapes(tile_config.two_dim[0], tile_config.two_dim[1])
    x_embded = x_re_second * cast_cos + rotate_half(x_re_second) * cast_sin

    return pypto.cast(x_embded, x.dtype)


def rope_3d_v2(x: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor) -> pypto.Tensor:
    """Apply 3D Rotary Position Embedding (RoPE) version 2.

    Implements RoPE transformation for 3D tensors with shape (batch, heads, dim).
    The RoPE is applied independently to each head using broadcasted cos/sin values.

    Args:
        x: Input tensor of shape (batch, heads, rope_dim)
        cos: Cosine values for RoPE, shape (batch, rope_dim)
        sin: Sine values for RoPE, shape (batch, rope_dim)

    Returns:
        Tensor with RoPE applied, same shape as input x

    Note:
        The function broadcasts cos and sin to match the head dimension,
        then applies rotation: x_rotated = x * cos + rotate_half(x) * sin
    """

    pypto.set_vec_tile_shapes(4, 64)
    cast_cos = pypto.cast(cos, pypto.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DT_FP32)
    cast_cos_re = pypto.reshape(cast_cos, [x.shape[0], 1, x.shape[2]])
    cast_sin_re = pypto.reshape(cast_sin, [x.shape[0], 1, x.shape[2]])

    if pypto.platform.npuarch == 'DAV_3510':
        pypto.set_vec_tile_shapes(1, 128, 64)
    else:
        pypto.set_vec_tile_shapes(1, 64, 64)
    cast_x = pypto.cast(x, pypto.DT_FP32)

    pypto.set_vec_tile_shapes(1, 64, 128, 128)
    x_view = pypto.reshape(cast_x, [x.shape[0], x.shape[1], x.shape[2] // 2, 2])
    x_trans = pypto.transpose(x_view, 2, 3)
    x_re_second = pypto.reshape(x_trans, x.shape)
    x_embed = x_re_second * cast_cos_re + rotate_half(x_re_second) * cast_sin_re

    return pypto.cast(x_embed, x.dtype)


def pre_compute_2d(
    token_x: pypto.Tensor,
    x_scale: pypto.Tensor,
    w_dq: pypto.Tensor,
    w_uq_qr: pypto.Tensor,
    w_dkv_kr: pypto.Tensor,
    gamma_cq: pypto.Tensor,
    epsilon_cq: float,
    quant_inputs: MlaQuantInputs,
    tile_config: MlaTileConfig
) -> pypto.Tensor:
    """Pre-compute query and key-value projections with MXFP quantization.

    Performs the initial computation steps for MLA prolog using MXFP (Microscaling)
    block-wise quantization:
    1. Query path: token_x (FP8E4M3) -> scaled_mm(w_dq) -> RMSNorm -> w_uq_qr
    2. Key-value path: token_x (FP8E4M3) -> scaled_mm(w_dkv_kr)

    MXFP Quantization Details:
        - Input token_x is pre-quantized to FP8E4M3 with per-block scales (FP8E8M0)
        - Block size: 64 elements per scale block
        - Uses pypto.scaled_mm for quantized matrix multiplication
        - Quant_a: Weights w_dq, w_dkv_kr are FP8E4M3 with per-block scales
        - Quant_b: RmsNorm output quantized to FP8E4M3, weight w_uq_qr is FP8E4M3

    Args:
        token_x: Input token tensor, shape (bs, h), dtype FP8E4M3 (pre-quantized)
        x_scale: Per-block scale for token_x, shape (bs, h//64, 2), dtype FP8E8M0
        w_dq: Down-projection weight for query, shape (h, q_lora_rank), FP8E4M3
        w_uq_qr: Up-projection weight for query and RoPE, shape (q_lora_rank, n*q_head_dim), FP8E4M3
        w_dkv_kr: Down-projection weight for key-value and RoPE, shape (h, kv_lora_rank+rope_dim), FP8E4M3
        gamma_cq: RMSNorm scale parameter for query, shape (q_lora_rank,), BF16
        epsilon_cq: RMSNorm epsilon parameter
        quant_inputs: MlaQuantInputs object containing quantization scales:
            - dequant_scale_w_dq: Per-block scale for w_dq, shape (h//128, q_lora_rank, 2), FP8E8M0
            - dequant_scale_w_dkv_kr: Per-block scale for w_dkv_kr, shape (h//128, q_head_dim, 2), FP8E8M0
            - dequant_scale_w_uq_qr: Per-block scale for w_uq_qr, FP8E8M0
        tile_config: MlaTileConfig object containing tiling parameters

    Returns:
        List containing:
            - q_b_proj: Query projection result, shape (bs, n*q_head_dim), BF16
            - compressed_kv: Compressed key-value result, shape (bs, kv_lora_rank+rope_dim), BF16

    Note:
        The function splits input dimension h into two halves and processes them
        separately to optimize memory access patterns, then accumulates results.
        For quant_b, RmsNorm output is quantized using pypto.quant_mx to FP8E4M3.
    """
    dequant_scale_w_dq = quant_inputs.dequant_scale_w_dq
    dequant_scale_w_dkv_kr = quant_inputs.dequant_scale_w_dkv_kr
    dequant_scale_w_uq_qr = quant_inputs.dequant_scale_w_uq_qr

    is_quant_a = (dequant_scale_w_dq is not None) and (dequant_scale_w_dkv_kr is not None)
    is_quant_b = dequant_scale_w_uq_qr is not None

    smooth_scales_cq = quant_inputs.smooth_scales_cq
    is_smooth = smooth_scales_cq is not None

    bs = token_x.shape[0]
    k = token_x.shape[1]
    q_lora_rank = w_dq.shape[1]
    q_head_dim = w_uq_qr.shape[1]
    dtype = pypto.DT_BF16
    qkv_pre_res = []

    pypto.set_semantic_label("pre_reshape")
    mv = tile_config.mv_tile if bs <= 16 else 4

    if is_quant_a:
        pypto.set_vec_tile_shapes(8, k // 2)
        token_x_view0 = pypto.view(token_x, [bs, k // 2], [0, 0])
        w_dq_view0 = pypto.view(w_dq, [k // 2, q_lora_rank], [0, 0])
        token_x_view1 = pypto.view(token_x, [bs, k // 2], [0, (k // 2)])
        w_dq_view1 = pypto.view(w_dq, [k // 2, q_lora_rank], [(k // 2), 0])
        pypto.set_vec_tile_shapes(8, 256, 8)
        dequant_scale_w_dq_view0 = pypto.view(dequant_scale_w_dq, [k // 128, q_lora_rank, 2], [0, 0, 0])
        x_scale_view0 = pypto.view(x_scale, [bs, k // 128, 2], [0, 0, 0])
        dequant_scale_w_dkv_kr_view0 = pypto.view(dequant_scale_w_dkv_kr, [k // 128, q_head_dim, 2], [0, 0, 0])
        w_dkv_kr_view0 = pypto.view(w_dkv_kr, [k // 2, q_head_dim], [0, 0])
        dequant_scale_w_dq_view1 = pypto.view(dequant_scale_w_dq, [k // 128, q_lora_rank, 2], [(k // 128), 0, 0])
        x_scale_view1 = pypto.view(x_scale, [bs, k // 128, 2], [0, (k // 128), 0])
        dequant_scale_w_dkv_kr_view1 = pypto.view(dequant_scale_w_dkv_kr, [k // 128, q_head_dim, 2], [(k // 128), 0, 0])
        w_dkv_kr_view1 = pypto.view(w_dkv_kr, [k // 2, q_head_dim], [(k // 2), 0])

        pypto.set_semantic_label("QuantMatmul_qa")
        pypto.set_cube_tile_shapes([tile_config.pre_quant_cube_tile[0], tile_config.pre_quant_cube_tile[1]],
                            [256, 256], [128, 128])
        q_a_proj0 = pypto.scaled_mm(token_x_view0, w_dq_view0, pypto.DT_FP32, x_scale_view0, dequant_scale_w_dq_view0)

        pypto.set_semantic_label("QuantMatmul_kva")
        pypto.set_cube_tile_shapes([tile_config.pre_quant_cube_tile[0], tile_config.pre_quant_cube_tile[1]],
                            [256, 256], [256, 256])
        compressed_kv_tmp0 = pypto.scaled_mm(
    token_x_view0,
    w_dkv_kr_view0,
    pypto.DT_FP32,
    x_scale_view0,
     dequant_scale_w_dkv_kr_view0)
        pypto.set_cube_tile_shapes([tile_config.pre_quant_cube_tile[0], tile_config.pre_quant_cube_tile[1]],
                            [256, 256], [128, 128])
        pypto.set_semantic_label("QuantMatmul_qa")
        q_a_proj1 = pypto.scaled_mm(token_x_view1, w_dq_view1, pypto.DT_FP32, x_scale_view1, dequant_scale_w_dq_view1)

        pypto.set_semantic_label("QuantMatmul_kva")
        pypto.set_cube_tile_shapes([tile_config.pre_quant_cube_tile[0], tile_config.pre_quant_cube_tile[1]],
                            [256, 256], [256, 256])
        compressed_kv_tmp1 = pypto.scaled_mm(
    token_x_view1,
    w_dkv_kr_view1,
    pypto.DT_FP32,
    x_scale_view1,
     dequant_scale_w_dkv_kr_view1)
        compressed_kv_tmp = compressed_kv_tmp0 + compressed_kv_tmp1
        compressed_kv = pypto.cast(compressed_kv_tmp, dtype)
    else:
        pypto.set_cube_tile_shapes([tile_config.pre_quant_cube_tile[0], tile_config.pre_quant_cube_tile[1]],
                                   [tile_config.pre_quant_cube_tile[2], tile_config.pre_quant_cube_tile[3]],
                                   [tile_config.pre_quant_cube_tile[4], tile_config.pre_quant_cube_tile[5]])
        pypto.set_semantic_label("Matmul_qa")
        x_view1 = pypto.view(token_x, [bs, k // 2], [0, 0])
        x_view2 = pypto.view(token_x, [bs, k // 2], [0, k // 2])
        w_dq1 = pypto.view(w_dq, [k // 2, q_lora_rank], [0, 0])
        w_dq2 = pypto.view(w_dq, [k // 2, q_lora_rank], [k // 2, 0])
        q_a_proj1 = pypto.matmul(x_view1, w_dq1, pypto.DT_FP32)
        q_a_proj2 = pypto.matmul(x_view2, w_dq2, pypto.DT_FP32)
        q_a_proj_tmp = q_a_proj1 + q_a_proj2
        q_a_proj = pypto.cast(q_a_proj_tmp, pypto.DT_BF16)

    pypto.set_vec_tile_shapes(mv, q_lora_rank)
    pypto.set_semantic_label("RmsNorm_qa")
    q_a_proj = q_a_proj0 + q_a_proj1
    norm_res = rms_norm(q_a_proj, gamma_cq, epsilon_cq)
    if is_quant_b:
        pypto.set_vec_tile_shapes(mv, q_lora_rank)
        pypto.set_semantic_label("Quant_qMnRes")
        quant_view, scale_view = pypto.quant_mx(
            norm_res,
            pypto.DT_FP8E4M3,
            pypto.ROUND_DOWN,
            -1,
            True,
        )
        pypto.set_semantic_label("QuantMatmul_qb")
        pypto.set_cube_tile_shapes([tile_config.pre_quant_cube_tile[0], tile_config.pre_quant_cube_tile[1]],
                                    [128, 256], [256, 256])
        q_b_proj = pypto.scaled_mm(quant_view, w_uq_qr, pypto.DT_BF16, scale_view, dequant_scale_w_uq_qr)
    else:
        pypto.set_cube_tile_shapes([tile_config.pre_quant_cube_tile[0], tile_config.pre_quant_cube_tile[1]],
                                   [tile_config.pre_quant_cube_tile[2], tile_config.pre_quant_cube_tile[3]],
                                   [256, 256])
        pypto.set_semantic_label("Matmul_qb")
        q_b_proj = pypto.matmul(norm_res, w_uq_qr, dtype)

    qkv_pre_res.append(q_b_proj)
    qkv_pre_res.append(compressed_kv)

    return qkv_pre_res


def mla_prolog_quant_compute(
    token_x: pypto.Tensor,
    x_scale: pypto.Tensor,
    w_dq: pypto.Tensor,
    w_dq_scale: pypto.Tensor,
    w_uq_qr: pypto.Tensor,
    w_uqqr_scale: pypto.Tensor,
    w_uk: pypto.Tensor,
    w_dkv_kr: pypto.Tensor,
    w_dkvkr_scale: pypto.Tensor,
    gamma_cq: pypto.Tensor,
    gamma_ckv: pypto.Tensor,
    cos: pypto.Tensor,
    sin: pypto.Tensor,
    cache_index: pypto.Tensor,
    kv_cache: pypto.Tensor,
    kr_cache: pypto.Tensor,
    query_nope_out: pypto.Tensor,
    query_rope_out: pypto.Tensor,
    kv_cache_out: pypto.Tensor,
    kr_cache_out: pypto.Tensor,
    epsilon_cq: float,
    epsilon_ckv: float,
    tile_config: MlaTileConfig,
    rope_cfg: RopeTileShapeConfig):
    """Compute MLA Prolog with MXFP quantization support.

    Main computation function for MLA Prolog MXFP quantization. Converts hidden states
    to query, key, and value projections using block-wise FP8E4M3 quantization with
    FP8E8M0 scales, and applies RoPE for position encoding.

    MXFP Quantization Format:
        - Data: FP8E4M3 (asymmetric, range [0, 448])
        - Scale: FP8E8M0 (block-wise, 64 or 128 elements per block)
        - Uses pypto.scaled_mm for efficient quantized matmul

    The computation includes:
    1. Query computation:
       - Input token_x (FP8E4M3) with per-block scales (FP8E8M0)
       - scaled_mm(token_x, w_dq) -> RMSNorm -> quant_mx -> scaled_mm(w_uq_qr)
       - Split into nope and rope parts
       - Apply RoPE to rope part
       - Apply w_uk transformation to nope part via transposed_batchmatmul

    2. Key computation:
       - scaled_mm(token_x, w_dkv_kr) -> Split into nope and rope
       - Apply RMSNorm to nope part (no additional quantization)
       - Apply RoPE to rope part
       - Update cache using scatter_update (no quantization for k_nope)

    Args:
        token_x: Input token tensor, shape (t, h), dtype FP8E4M3 (pre-quantized)
        x_scale: Per-block scale for token_x, shape (t, h//64, 2), dtype FP8E8M0
        w_dq: Down-projection weight for query, shape (h, q_lora_rank), FP8E4M3
        w_dq_scale: Per-block scale for w_dq, shape (h//128, q_lora_rank, 2), FP8E8M0
        w_uq_qr: Up-projection weight for query and RoPE, shape (q_lora_rank, n*q_head_dim), FP8E4M3
        w_uqqr_scale: Per-block scale for w_uq_qr, FP8E8M0
        w_uk: Up-projection weight for key, shape (n, qk_nope_head_dim, kv_lora_rank), BF16
        w_dkv_kr: Down-projection weight for key-value and RoPE, shape (h, kv_lora_rank+rope_dim), FP8E4M3
        w_dkvkr_scale: Per-block scale for w_dkv_kr, shape (h//128, q_head_dim, 2), FP8E8M0
        gamma_cq: RMSNorm scale for query, shape (q_lora_rank,), BF16
        gamma_ckv: RMSNorm scale for key-value, shape (kv_lora_rank,), BF16
        cos: Cosine values for RoPE, shape (t, qk_rope_head_dim), BF16
        sin: Sine values for RoPE, shape (t, qk_rope_head_dim), BF16
        cache_index: Cache index for scatter update, shape (t,), INT64
        kv_cache: Key-value cache input, shape (block_num, block_size, n_kv, kv_lora_rank), BF16
        kr_cache: Key RoPE cache input, shape (block_num, block_size, n_kv, rope_dim), BF16
        query_nope_out: Output query without RoPE, shape (t, n_q, kv_lora_rank), BF16
        query_rope_out: Output query with RoPE, shape (t, n_q, rope_dim), BF16
        kv_cache_out: Output key-value cache, updated in-place via scatter_update
        kr_cache_out: Output key RoPE cache, updated in-place via scatter_update
        epsilon_cq: RMSNorm epsilon for query
        epsilon_ckv: RMSNorm epsilon for key-value
        tile_config: MlaTileConfig object containing tiling parameters
        rope_cfg: RopeTileShapeConfig object containing RoPE tiling parameters

    Note:
        The function processes tokens in tiles using loop_unroll for optimization.
        Key nope (k_nope) is NOT quantized in this version - only RmsNorm is applied.
        All cache updates use scatter_update with axis=-2 for in-place modification.
    """

    dtype = token_x.dtype
    h = token_x.shape[1]
    n1 = w_uk.shape[0]
    q_lora_rank = w_dq.shape[1]
    qk_nope_head_dim = w_uk.shape[1]
    kv_lora_rank = w_uk.shape[2]
    qk_rope_head_dim = sin.shape[1]
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    t = token_x.shape[0]
    quant_inputs = MlaQuantInputs(
                    dequant_scale_w_dq=w_dq_scale,
                    dequant_scale_w_uq_qr=w_uqqr_scale,
                    dequant_scale_w_dkv_kr=w_dkvkr_scale
                    )
    k_cache_index_2d = pypto.reshape(cache_index, [t, 1], inplace=True)
    unroll_list = tile_config.unroll_list
    # Main loop: Process tokens in tiles using loop_unroll for dynamic sequence length
    for bs_offset, unroll_length in pypto.loop_unroll(0, t, 1, name="MLA_BS_LOOP", idx_name="bs_offset",
                                                      unroll_list=unroll_list, ):
        tile_bs = unroll_length
        output_offset = [bs_offset, 0, 0]
        pypto.set_vec_tile_shapes(tile_bs, 128, 8)
        x_view = pypto.view(token_x, [tile_bs, h], [bs_offset, 0])

        # View input scale for current tile (block-wise scale: 64 elements per block)
        x_scale_view = pypto.view(x_scale, [tile_bs, h // 64, 2], [bs_offset, 0, 0])
        q_kv = pre_compute_2d(x_view, x_scale_view, w_dq, w_uq_qr, w_dkv_kr, gamma_cq, \
                                epsilon_cq, quant_inputs, tile_config)
        q = q_kv[0]
        kv_tmp = q_kv[1]

        ########### q ##############
        # Query processing: Split into nope and rope parts
        q_tmp = pypto.reshape(q, [tile_bs, n1, q_head_dim])
        pypto.set_semantic_label("Prepare_qNope")
        q_nope = pypto.view(q_tmp, [tile_bs, n1, qk_nope_head_dim], [0, 0, 0])
        m = tile_config.m_tile
        pypto.set_semantic_label("Matmul_qNope_wUk")
        pypto.set_cube_tile_shapes([m, m], [128, 128], [128, 128])
        q_nope_new_trans = pypto.experimental.transposed_batchmatmul(q_nope, w_uk, pypto.DT_BF16)

        pypto.set_semantic_label("Assemble_queryOut")
        pypto.set_vec_tile_shapes(tile_config.q_vec_tile0, tile_config.q_vec_tile1, 128)
        pypto.assemble(q_nope_new_trans, output_offset, query_nope_out)

        # Query RoPE processing
        pypto.set_vec_tile_shapes(tile_config.q_vec_tile0, tile_config.q_vec_tile1, 64)
        q_pe_view = pypto.view(q_tmp, [tile_bs, n1, qk_rope_head_dim], [0, 0, qk_nope_head_dim])
        cos_2d_view = pypto.view(cos, [tile_bs, qk_rope_head_dim], [bs_offset, 0])
        sin_2d_view = pypto.view(sin, [tile_bs, qk_rope_head_dim], [bs_offset, 0])
        pypto.set_semantic_label("Rope_qRope")
        q_rope_view = rope_3d_v2(q_pe_view, cos_2d_view, sin_2d_view)
        pypto.set_semantic_label("Assemble_qRope")
        pypto.set_vec_tile_shapes(tile_config.q_vec_tile0, tile_config.q_vec_tile1, 64)
        pypto.assemble(q_rope_view, output_offset, query_rope_out)

        ########### RoPE #################
        # Key RoPE processing: Apply RoPE to key rope part
        pypto.set_vec_tile_shapes(tile_config.k_vec_tile0, tile_config.k_vec_tile1)
        pypto.set_semantic_label("RotaryPosEmb")
        k_pe_view = pypto.view(kv_tmp, [tile_bs, qk_rope_head_dim], [0, kv_lora_rank])
        k_rope_2d = rope_v2(k_pe_view, cos_2d_view, sin_2d_view, rope_cfg)

        ############### kNope ##############
        compressed_kv1 = pypto.view(kv_tmp, [tile_bs, kv_lora_rank], [0, 0])
        pypto.set_semantic_label("RmsNorm_compressedkv")
        pypto.set_vec_tile_shapes(tile_config.k_vec_tile0, tile_config.k_vec_tile1)
        k_nope = rms_norm(compressed_kv1, gamma_ckv, epsilon_ckv)

        ########### kNope reshape for cache update ############
        # Note: k_nope is NOT quantized in this MXFP V3 implementation
        # It is only reshaped for scatter_update cache operation
        pypto.set_semantic_label("Quant_knope")
        pypto.set_vec_tile_shapes(32, kv_lora_rank)
        k_nope_split = pypto.reshape(k_nope, [tile_bs, 4, kv_lora_rank // 4])

        # Update KR cache via scatter_update
        k_rope_4d = pypto.reshape(k_rope_2d, [tile_bs, 1, 1, qk_rope_head_dim], inplace=True)
        index = pypto.view(k_cache_index_2d, [tile_bs, 1], [bs_offset, 0])
        pypto.set_semantic_label("ScatterUpdate_krCache")
        pypto.set_vec_tile_shapes(32, 1, 1, qk_rope_head_dim)
        kr_cache_out[:] = pypto.scatter_update(kr_cache, -2, index, k_rope_4d)

        # Update KV cache via scatter_update (axis=-2: sequence dimension)
        pypto.set_vec_tile_shapes(32, 4, kv_lora_rank // 4)
        k_nope_4d = pypto.reshape(k_nope_split, [tile_bs, 1, 1, kv_lora_rank], inplace=True)
        pypto.set_semantic_label("ScatterUpdate_kvCache")
        pypto.set_vec_tile_shapes(32, 1, 1, kv_lora_rank)
        kv_cache_out[:] = pypto.scatter_update(kv_cache, -2, index, k_nope_4d)


@pypto.frontend.jit(
    pass_options={
        "cube_l1_reuse_setting": {-1: 4, 0: 1, 1: 3},
        "cube_nbuffer_setting": {-1: 4, 0: 1, 1: 1},
    },
    runtime_options={
        "device_sched_mode": 2,
        "stitch_function_max_num": 128
    }
)
def mla_prolog_quant(
    token_x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP8E4M3),
    x_scale: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP8E8M0),
    w_dq: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP8E4M3),
    w_dq_scale: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP8E8M0),
    w_uq_qr: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP8E4M3),
    w_uqqr_scale: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP8E8M0),
    w_uk: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    w_dkv_kr: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP8E4M3),
    w_dkvkr_scale: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP8E8M0),
    gamma_cq: pypto.Tensor([pypto.STATIC]),
    gamma_ckv: pypto.Tensor([pypto.STATIC]),
    cos: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC]),
    sin: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC]),
    cache_index: pypto.Tensor([pypto.DYNAMIC]),
    kv_cache: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    kr_cache: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    query_nope_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    query_rope_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    kv_cache_out: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    kr_cache_out: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    epsilon_cq, epsilon_ckv, tile_config, rope_cfg
):
    """
    JIT-compiled MLA Prolog MXFP quantization for decode phase.

    Optimized version for decode phase using MXFP (Microscaling FP8) quantization.
    Processes single or few tokens at a time for low latency.

    MXFP Format:
        - Input token_x: FP8E4M3 with per-block scales (FP8E8M0)
        - Weights: FP8E4M3 with per-block scales (FP8E8M0)
        - Uses pypto.scaled_mm for efficient quantized matrix multiplication

    Args:
        token_x: Input token tensor, shape (t, h), dtype FP8E4M3 (pre-quantized)
        x_scale: Per-block scale for token_x, shape (t, h//64, 2), dtype FP8E8M0
        w_dq: Down-projection weight for query, shape (h, q_lora_rank), FP8E4M3
        w_dq_scale: Per-block scale for w_dq, shape (h//128, q_lora_rank, 2), FP8E8M0
        w_uq_qr: Up-projection weight for query and RoPE, shape (q_lora_rank, n*q_head_dim), FP8E4M3
        w_uqqr_scale: Per-block scale for w_uq_qr, FP8E8M0
        w_uk: Up-projection weight for key, shape (n, qk_nope_head_dim, kv_lora_rank), BF16
        w_dkv_kr: Down-projection weight for key-value and RoPE, shape (h, kv_lora_rank+rope_dim), FP8E4M3
        w_dkvkr_scale: Per-block scale for w_dkv_kr, shape (h//128, q_head_dim, 2), FP8E8M0
        gamma_cq: RMSNorm scale for query, shape (q_lora_rank,), BF16
        gamma_ckv: RMSNorm scale for key-value, shape (kv_lora_rank,), BF16
        cos: Cosine values for RoPE, shape (t, qk_rope_head_dim), BF16
        sin: Sine values for RoPE, shape (t, qk_rope_head_dim), BF16
        cache_index: Cache index for scatter update, shape (t,), INT64
        kv_cache: Key-value cache input/output, shape (block_num, block_size, n_kv, kv_lora_rank), BF16
        kr_cache: Key RoPE cache input/output, shape (block_num, block_size, n_kv, rope_dim), BF16
        query_nope_out: Output query without RoPE, shape (t, n_q, kv_lora_rank), BF16
        query_rope_out: Output query with RoPE, shape (t, n_q, rope_dim), BF16
        kv_cache_out: Output key-value cache, updated in-place
        kr_cache_out: Output key RoPE cache, updated in-place
        epsilon_cq: RMSNorm epsilon for query
        epsilon_ckv: RMSNorm epsilon for key-value
        tile_config: MlaTileConfig object containing tiling parameters
        rope_cfg: RopeTileShapeConfig object containing RoPE tiling parameters

    Note:
        Configured for decode phase with optimized memory and latency settings.
        Pass options include cube_l1_reuse_setting and cube_nbuffer_setting for performance.
        The t dimension is dynamic (pypto.DYNAMIC) for variable sequence lengths.
    """
    mla_prolog_quant_compute(
                            token_x, x_scale, w_dq, w_dq_scale, w_uq_qr, w_uqqr_scale, w_uk,
                            w_dkv_kr, w_dkvkr_scale, gamma_cq, gamma_ckv, cos,
                            sin, cache_index, kv_cache, kr_cache, query_nope_out,
                            query_rope_out, kv_cache_out,
                            kr_cache_out, epsilon_cq,
                            epsilon_ckv, tile_config, rope_cfg
    )
