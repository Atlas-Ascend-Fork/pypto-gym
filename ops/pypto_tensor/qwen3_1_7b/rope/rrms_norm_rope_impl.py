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
Qwen3-1.7B RoPE PyPTO Kernel - 部分融合算子

功能: Q/K per-head RMSNorm + RoPE
输入: [seq_len, num_heads, D] - q_proj/k_proj 输出（已完成projection）
输出: [seq_len, num_heads, D] - 经过 RMSNorm + RoPE 的 Q/K

融合范围:
- Q/K RMSNorm (per-head normalization)
- RoPE (Rotary Position Embedding)

未融合部分:
- q_proj/k_proj/v_proj (在 PyTorch 中完成)
- pre-attention RMSNorm (在 PyTorch 中完成)

性能优化:
- 支持 3D tensor 输入，避免 reshape
- Tile-based 处理，符合 UB 限制
- BF16 计算精度
"""

__all__ = [
    'qwen3_qk_rope_q',
    'qwen3_qk_rope_k',
    '_make_qk_rope_kernel',
]

import pypto

D = 128
HALF_D = D // 2
EPS = 1e-6
BS_TILE = 8


def _rms_norm_per_d(x_3d_fp32, w_fp32_n, mean_coff, eps):
    """
    RMSNorm along last dimension

    Args:
        x_3d_fp32: [BS_TILE, num_heads, D] FP32
        w_fp32_n: [1, num_heads, D] FP32 (broadcast)
        mean_coff: 1.0 / D
        eps: 1e-6

    Returns:
        normed: [BS_TILE, num_heads, D] FP32
    """
    sq = pypto.mul(x_3d_fp32, x_3d_fp32)
    mean = pypto.sum(sq, -1, keepdim=True)
    mean = pypto.mul(mean, mean_coff)
    rsqrt = pypto.rsqrt(pypto.add(mean, eps))
    out = pypto.mul(x_3d_fp32, rsqrt)
    out = pypto.mul(out, w_fp32_n)
    return out


def _make_qk_rope_kernel(num_heads: int):
    """
    Factory: 创建针对特定 head 数量的 JIT kernel

    Args:
        num_heads: head 数量 (N_q=16 或 N_kv=8)
    Returns:
        JIT kernel function
    """

    @pypto.frontend.jit(
        runtime_options={"stitch_function_max_num": 128, "device_sched_mode": 1},
    )
    def kernel(
        x: pypto.Tensor([pypto.DYNAMIC, num_heads, D], pypto.DT_BF16),
        cos: pypto.Tensor([pypto.DYNAMIC, D], pypto.DT_BF16),
        sin: pypto.Tensor([pypto.DYNAMIC, D], pypto.DT_BF16),
        w_norm: pypto.Tensor([D], pypto.DT_BF16),
        out: pypto.Tensor([pypto.DYNAMIC, num_heads, D], pypto.DT_BF16),
    ):
        """
        Q/K RoPE kernel

        输入:
            x: [seq_len, num_heads, D] - q_proj/k_proj 输出
            cos: [seq_len, D] - cos 值（由 position embeddings 生成）
            sin: [seq_len, D] - sin 值
            w_norm: [D] - q_norm/k_norm 权重

        输出:
            out: [seq_len, num_heads, D] - 经过 RMSNorm + RoPE 的结果
        """
        seq_len = x.shape[0]
        bs_loop = (seq_len + BS_TILE - 1) // BS_TILE
        d_mean_coff = 1.0 / D

        # Pre-process: 广播 normalization 权重
        pypto.set_vec_tile_shapes(1, 1, D)
        w_3d = pypto.reshape(w_norm, [1, 1, D], inplace=False)
        w_fp32_1 = pypto.cast(w_3d, pypto.DT_FP32)
        w_fp32_n = pypto.expand_clone(w_fp32_1, [1, num_heads, D])

        # Loop: 遍历 sequence 维度
        for bs_idx in pypto.loop(bs_loop, name="LOOP_BS_QKROPE", idx_name="bs_idx", unroll_list=[8, 4, 2, 1]):
            cur_bs = (seq_len - bs_idx * BS_TILE).min(BS_TILE)

            # Step 1: View input tile
            x_tile = pypto.view(x, [BS_TILE, num_heads, D], [bs_idx * BS_TILE, 0, 0],
                                valid_shape=[cur_bs, num_heads, D])

            # Step 2: RMSNorm
            pypto.set_vec_tile_shapes(BS_TILE, num_heads, D)
            x_fp32 = pypto.cast(x_tile, pypto.DT_FP32)
            normed_fp32 = _rms_norm_per_d(x_fp32, w_fp32_n, d_mean_coff, EPS)

            # Step 3: Prepare cos/sin
            cos_tile = pypto.view(cos, [BS_TILE, D], [bs_idx * BS_TILE, 0],
                                  valid_shape=[cur_bs, D])
            sin_tile = pypto.view(sin, [BS_TILE, D], [bs_idx * BS_TILE, 0],
                                  valid_shape=[cur_bs, D])

            # Take first half for RoPE
            pypto.set_vec_tile_shapes(BS_TILE, HALF_D)
            cos_half = pypto.view(cos_tile, [BS_TILE, HALF_D], [0, 0],
                                  valid_shape=[cur_bs, HALF_D])
            sin_half = pypto.view(sin_tile, [BS_TILE, HALF_D], [0, 0],
                                  valid_shape=[cur_bs, HALF_D])

            cos_half_fp32 = pypto.cast(cos_half, pypto.DT_FP32)
            sin_half_fp32 = pypto.cast(sin_half, pypto.DT_FP32)

            # Reshape for broadcasting: [BS_TILE, HALF_D] -> [BS_TILE, 1, HALF_D]
            cos_b = pypto.reshape(cos_half_fp32, [BS_TILE, 1, HALF_D], inplace=False)
            sin_b = pypto.reshape(sin_half_fp32, [BS_TILE, 1, HALF_D], inplace=False)

            # Step 4: RoPE computation
            pypto.set_vec_tile_shapes(BS_TILE, num_heads, HALF_D)

            # Split into left/right halves
            x_left = pypto.view(normed_fp32, [BS_TILE, num_heads, HALF_D], [0, 0, 0],
                                 valid_shape=[cur_bs, num_heads, HALF_D])
            x_right = pypto.view(normed_fp32, [BS_TILE, num_heads, HALF_D], [0, 0, HALF_D],
                                 valid_shape=[cur_bs, num_heads, HALF_D])


            o1 = pypto.sub(pypto.mul(x_left, cos_b), pypto.mul(x_right, sin_b))
            o2 = pypto.add(pypto.mul(x_right, cos_b), pypto.mul(x_left, sin_b))

            # Concat: [BS_TILE, num_heads, D] FP32
            roped_fp32 = pypto.concat([o1, o2], 2)

            # Cast to BF16
            roped_bf = pypto.cast(roped_fp32, pypto.DT_BF16)

            # Step 5: Write back
            pypto.assemble(roped_bf, [bs_idx * BS_TILE, 0, 0], out)

    return kernel


# Create specialized kernels for Q and K
qwen3_qk_rope_q = _make_qk_rope_kernel(16)   # Q: N_q=16 heads
qwen3_qk_rope_k = _make_qk_rope_kernel(8)    # K: N_kv=8 heads

