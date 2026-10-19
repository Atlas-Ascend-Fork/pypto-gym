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
FlashAttentionScoreGrad PyPTO Kernel 实现 (性能优化版)

优化项:
  1. S_TILE 64 → 128，减少循环迭代
  2. cube_tile_shapes 增大至 [128,128],[64,256],[128,128]
  3. pass_options: cube_l1_reuse_setting Q 常驻 L1
  4. runtime_options: stitch_function_max_num=128, device_sched_mode=1
  5. 内层 unroll_list=[8, 4, 2, 1]
"""

import pypto
import torch

# ============================================================
# 模块级常量
# ============================================================
S_TILE = 128  # 优化: 64 → 128


def compute_tile(q_i, k_j, v_j, dy_i, smax_i, ssum_i, d_i,
                 actual_s1, actual_s2, scale_value, c_tile, v_tile_s, v_tile_d, s_tile_size):
    """计算一个 (s1_tile, s2_tile) 块的 P_ij 和 dS_ij。"""
    # 计算公式：S_ij = Q_i @ K_j^T * scale
    pypto.set_vec_tile_shapes(v_tile_s[0], v_tile_s[1])
    pypto.set_cube_tile_shapes(c_tile[0], c_tile[1], c_tile[2])
    s_ij = pypto.matmul(q_i, k_j, pypto.DT_FP32, b_trans=True)
    s_ij = pypto.view(s_ij, [s_tile_size, s_tile_size], [0, 0],
                      valid_shape=[actual_s1, actual_s2])

    pypto.set_vec_tile_shapes(v_tile_s[0], v_tile_s[1])
    s_ij = pypto.mul(s_ij, scale_value)
    p_ij = pypto.exp(pypto.sub(s_ij, smax_i),
                     precision_type=pypto.PrecisionType.HIGH_PRECISION)
    p_ij = pypto.div(p_ij, ssum_i)

    # 计算公式：dP_ij = dY_i @ V_j^T
    pypto.set_vec_tile_shapes(v_tile_s[0], v_tile_s[1])
    pypto.set_cube_tile_shapes(c_tile[0], c_tile[1], c_tile[2])
    dp_ij = pypto.matmul(dy_i, v_j, pypto.DT_FP32, b_trans=True)
    dp_ij = pypto.view(dp_ij, [s_tile_size, s_tile_size], [0, 0],
                       valid_shape=[actual_s1, actual_s2])

    # 计算公式：dS_ij = P_ij * (dP_ij - D_i)
    pypto.set_vec_tile_shapes(v_tile_s[0], v_tile_s[1])
    ds_ij = pypto.mul(p_ij, pypto.sub(dp_ij, d_i))

    return p_ij, ds_ij


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 128,
        "device_sched_mode": 1,
    },
    pass_options={
        "cube_l1_reuse_setting": {0: 8},
        "cube_nbuffer_setting": {0: 4},
    }
)
def flash_attention_score_grad_kernel(
    q: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    k: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    dy: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    softmax_max: pypto.Tensor([pypto.DYN, ...], pypto.DT_FP32),
    softmax_sum: pypto.Tensor([pypto.DYN, ...], pypto.DT_FP32),
    attention_out: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    dq: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    dk: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    dv: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    batch_size: pypto.Tensor([pypto.DYN], pypto.DT_INT32),
    scale_value: float,
    num_heads: int,
):
    b = batch_size.shape[0]
    total = q.shape[0]
    head_dim = q.shape[1]
    s = total // b // num_heads

    q_2d = q
    k_2d = k
    v_2d = v
    dy_2d = dy
    ao_2d = attention_out
    sm_2d = softmax_max
    ss_2d = softmax_sum
    dq_2d = dq
    dk_2d = dk
    dv_2d = dv

    s_loop = (s + S_TILE - 1) // S_TILE

    c_tile = [[S_TILE, S_TILE], [head_dim, 256], [S_TILE, S_TILE]]
    v_tile_s = [S_TILE, S_TILE]
    v_tile_d = [S_TILE, head_dim]

    for b_idx in pypto.loop(b, name="LOOP_b", idx_name="b_idx"):
        for n_idx in pypto.loop(num_heads, name="LOOP_n", idx_name="n_idx"):
            bn_base = (b_idx * num_heads + n_idx) * s

            # ===== 趟1: 计算 dQ =====
            for s1_idx in pypto.loop(s_loop, name="LOOP_s1_dq", idx_name="s1_idx"):
                s1_off = bn_base + s1_idx * S_TILE
                actual_s1 = (s - s1_idx * S_TILE).min(S_TILE)

                pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                q_i = pypto.view(q_2d, [S_TILE, head_dim], [s1_off, 0], valid_shape=[actual_s1, head_dim])
                dy_i = pypto.view(dy_2d, [S_TILE, head_dim], [s1_off, 0], valid_shape=[actual_s1, head_dim])
                ao_i = pypto.view(ao_2d, [S_TILE, head_dim], [s1_off, 0], valid_shape=[actual_s1, head_dim])

                sm_i_8 = pypto.view(sm_2d, [S_TILE, 8], [s1_off, 0], valid_shape=[actual_s1, 8])
                ss_i_8 = pypto.view(ss_2d, [S_TILE, 8], [s1_off, 0], valid_shape=[actual_s1, 8])
                pypto.set_vec_tile_shapes(S_TILE, 8)
                smax_i = pypto.view(sm_i_8, [S_TILE, 1], [0, 0], valid_shape=[actual_s1, 1])
                ssum_i = pypto.view(ss_i_8, [S_TILE, 1], [0, 0], valid_shape=[actual_s1, 1])

                pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                dy_ao_fp32 = pypto.cast(pypto.mul(dy_i, ao_i), pypto.DT_FP32)
                d_i = pypto.sum(dy_ao_fp32, -1, keepdim=True)

                dq_acc = pypto.tensor([S_TILE, head_dim], pypto.DT_FP32, "dq_acc")

                for s2_idx in pypto.loop(s_loop, name="LOOP_s2_dq", idx_name="s2_idx",
                                         unroll_list=[8, 4, 2, 1]):
                    s2_off = bn_base + s2_idx * S_TILE
                    actual_s2 = (s - s2_idx * S_TILE).min(S_TILE)

                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    k_j = pypto.view(k_2d, [S_TILE, head_dim], [s2_off, 0], valid_shape=[actual_s2, head_dim])
                    v_j = pypto.view(v_2d, [S_TILE, head_dim], [s2_off, 0], valid_shape=[actual_s2, head_dim])

                    _, ds_ij = compute_tile(q_i, k_j, v_j, dy_i, smax_i, ssum_i, d_i,
                                            actual_s1, actual_s2, scale_value,
                                            c_tile, v_tile_s, v_tile_d, S_TILE)

                    ds_bf16 = pypto.cast(ds_ij, pypto.DT_BF16)
                    pypto.set_cube_tile_shapes(c_tile[0], c_tile[1], c_tile[2])
                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    dq_tile = pypto.matmul(ds_bf16, k_j, pypto.DT_FP32)

                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    if pypto.is_loop_begin(s2_idx):
                        dq_acc[:] = dq_tile
                    else:
                        dq_acc[:] = dq_acc + dq_tile

                    if pypto.is_loop_end(s2_idx):
                        pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                        dq_final = pypto.cast(pypto.mul(dq_acc, scale_value), pypto.DT_BF16)
                        dq_final_v = pypto.view(dq_final, [S_TILE, head_dim], [0, 0],
                                                valid_shape=[actual_s1, head_dim])
                        pypto.assemble(dq_final_v, [s1_off, 0], dq_2d)

            # ===== 趟2: 计算 dK, dV =====
            for s2_idx in pypto.loop(s_loop, name="LOOP_s2_dkv", idx_name="s2_idx"):
                s2_off = bn_base + s2_idx * S_TILE
                actual_s2 = (s - s2_idx * S_TILE).min(S_TILE)

                pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                k_j = pypto.view(k_2d, [S_TILE, head_dim], [s2_off, 0], valid_shape=[actual_s2, head_dim])
                v_j = pypto.view(v_2d, [S_TILE, head_dim], [s2_off, 0], valid_shape=[actual_s2, head_dim])

                dk_acc = pypto.tensor([S_TILE, head_dim], pypto.DT_FP32, "dk_acc")
                dv_acc = pypto.tensor([S_TILE, head_dim], pypto.DT_FP32, "dv_acc")

                for s1_idx in pypto.loop(s_loop, name="LOOP_s1_dkv", idx_name="s1_idx",
                                         unroll_list=[8, 4, 2, 1]):
                    s1_off = bn_base + s1_idx * S_TILE
                    actual_s1 = (s - s1_idx * S_TILE).min(S_TILE)

                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    q_i = pypto.view(q_2d, [S_TILE, head_dim], [s1_off, 0], valid_shape=[actual_s1, head_dim])
                    dy_i = pypto.view(dy_2d, [S_TILE, head_dim], [s1_off, 0], valid_shape=[actual_s1, head_dim])
                    ao_i = pypto.view(ao_2d, [S_TILE, head_dim], [s1_off, 0], valid_shape=[actual_s1, head_dim])

                    sm_i_8 = pypto.view(sm_2d, [S_TILE, 8], [s1_off, 0], valid_shape=[actual_s1, 8])
                    ss_i_8 = pypto.view(ss_2d, [S_TILE, 8], [s1_off, 0], valid_shape=[actual_s1, 8])
                    pypto.set_vec_tile_shapes(S_TILE, 8)
                    smax_i = pypto.view(sm_i_8, [S_TILE, 1], [0, 0], valid_shape=[actual_s1, 1])
                    ssum_i = pypto.view(ss_i_8, [S_TILE, 1], [0, 0], valid_shape=[actual_s1, 1])

                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    dy_ao_fp32 = pypto.cast(pypto.mul(dy_i, ao_i), pypto.DT_FP32)
                    d_i = pypto.sum(dy_ao_fp32, -1, keepdim=True)

                    p_ij, ds_ij = compute_tile(q_i, k_j, v_j, dy_i, smax_i, ssum_i, d_i,
                                               actual_s1, actual_s2, scale_value,
                                               c_tile, v_tile_s, v_tile_d, S_TILE)

                    ds_bf16 = pypto.cast(ds_ij, pypto.DT_BF16)
                    p_bf16 = pypto.cast(p_ij, pypto.DT_BF16)
                    pypto.set_cube_tile_shapes(c_tile[0], c_tile[1], c_tile[2])
                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    dk_tile = pypto.matmul(ds_bf16, q_i, pypto.DT_FP32, a_trans=True)
                    dv_tile = pypto.matmul(p_bf16, dy_i, pypto.DT_FP32, a_trans=True)

                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    if pypto.is_loop_begin(s1_idx):
                        dk_acc[:] = dk_tile
                        dv_acc[:] = dv_tile
                    else:
                        dk_acc[:] = dk_acc + dk_tile
                        dv_acc[:] = dv_acc + dv_tile

                    if pypto.is_loop_end(s1_idx):
                        pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                        dk_final = pypto.cast(pypto.mul(dk_acc, scale_value), pypto.DT_BF16)
                        dv_final = pypto.cast(dv_acc, pypto.DT_BF16)
                        dk_final_v = pypto.view(dk_final, [S_TILE, head_dim], [0, 0],
                                                valid_shape=[actual_s2, head_dim])
                        dv_final_v = pypto.view(dv_final, [S_TILE, head_dim], [0, 0],
                                                valid_shape=[actual_s2, head_dim])
                        pypto.assemble(dk_final_v, [s2_off, 0], dk_2d)
                        pypto.assemble(dv_final_v, [s2_off, 0], dv_2d)


def flash_attention_score_grad_wrapper(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dy: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    attention_out: torch.Tensor,
    scale_value: float,
    num_heads: int,
    head_dim: int,
):
    """算子 wrapper，供测试调用。"""
    batch_size, num_heads_out, seq_len, head_dim_out = query.shape
    if num_heads_out != num_heads or head_dim_out != head_dim:
        raise ValueError(
            f"Shape mismatch: expected num_heads={num_heads}, head_dim={head_dim}, "
            f"got num_heads={num_heads_out}, head_dim={head_dim_out}"
        )

    q_flat = query.reshape(-1, head_dim_out).contiguous()
    k_flat = key.reshape(-1, head_dim_out).contiguous()
    v_flat = value.reshape(-1, head_dim_out).contiguous()
    dy_flat = dy.reshape(-1, head_dim_out).contiguous()
    ao_flat = attention_out.reshape(-1, head_dim_out).contiguous()
    sm_flat = softmax_max.reshape(-1, 8).contiguous()
    ss_flat = softmax_sum.reshape(-1, 8).contiguous()

    dq_flat = torch.empty_like(q_flat)
    dk_flat = torch.empty_like(k_flat)
    dv_flat = torch.empty_like(v_flat)

    batch_tensor = torch.zeros(batch_size, dtype=torch.int32, device=query.device)

    flash_attention_score_grad_kernel(
        q_flat, k_flat, v_flat, dy_flat,
        sm_flat, ss_flat, ao_flat,
        dq_flat, dk_flat, dv_flat,
        batch_tensor,
        scale_value,
        num_heads,
    )

    return (dq_flat.reshape(batch_size, num_heads, seq_len, head_dim),
            dk_flat.reshape(batch_size, num_heads, seq_len, head_dim),
            dv_flat.reshape(batch_size, num_heads, seq_len, head_dim))
