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
Sparse Flash Attention Forward - TND Format v2

Input Tensors:
    - q_nope:              [T1, N1, D]             BF16
    - compressed_kv_norm:  [T2, N2, D]             BF16  (T2 静态)
    - topk_indices:        [T1, N2, sparse_size]   INT32  (indices into T2)
    - q_pe:                [T1, N1, D_ROPE]        BF16
    - k_pe:                [T2, N2, D_ROPE]        BF16   (T2 静态, 与compressed_kv_norm同维)
    - npu_actual_q_len:    [B]                     INT32
    - npu_actual_kv_len:   [B]                     INT32

Output Tensors:
    - core_attn_out:       [T1, N1, D]             BF16
    - softmax_max:         [N2, T1, N1/N2]         FP32
    - softmax_sum:         [N2, T1, N1/N2]         FP32

Loop structure:
    - outer loop over B (batch, dynamic from npu_actual_q_len.shape[0])
      - inner loop over s (per-batch query tokens, dynamic from prefix-sum)
        - static for range(N2) kv heads
          - process group=N1/N2 query heads via matmul

Key design:
    - npu_actual_q_len is prefix-sum, s = q_len[i] - q_len[i-1] * (i > 0)
    - npu_actual_kv_len is NOT prefix-sum (per-batch KV length)
    - compressed_kv_norm 保持 3D，不合轴
    - topk_indices 直接对 compressed_kv_norm[:, kv_head, :] 取值
    - index_select 前对 kv_gather_idx 取 valid_shape
    - 输出用 assemble 写到对应 (t_idx, kv_head_idx) 位置
"""
import os
import math
from dataclasses import dataclass
import numpy as np
import pypto


@dataclass
class SaTileShapeConfig:
    s_kv_tile: int
    c1_tile_shape: list
    v1_tile_shape: list
    c2_tile_shape: list
    v2_tile_shape: list


def sfa_forward_tnd_compute(q_nope, compressed_kv_norm, topk_indices,
                            q_pe, k_pe,
                            npu_actual_q_len, npu_actual_kv_len,
                            core_attn_out, softmax_max_out, softmax_sum_out,
                            nq, n_kv, scale, sparse_size, max_total_kv, tile_config):
    """Compute sparse flash attention in TND format v2.

    Nested loop: outer loop over B (batch), inner loop over s (per-batch query tokens).
    B is derived from npu_actual_q_len.shape[0].
    npu_actual_q_len and npu_actual_kv_len are both prefix-sums.

    compressed_kv_norm and k_pe have dynamic T2. After reshape inplace to
    [T2*N2, D], they are viewed with fixed shape [max_total_kv, D] and
    valid_shape [T2*N2, D] so that index_select works on a static-shape tensor.

    Args:
        q_nope:              [T1, N1, D], BF16
        compressed_kv_norm:  [T2, N2, D], BF16 (T2 dynamic)
        topk_indices:        [T1, N2, sparse_size], INT32
        q_pe:                [T1, N1, D_ROPE], BF16
        k_pe:                [T2, N2, D_ROPE], BF16 (T2 dynamic)
        npu_actual_q_len:    [B], INT32 (prefix-sum of query lengths)
        npu_actual_kv_len:   [B], INT32 (prefix-sum of KV lengths)
        core_attn_out:       [T1, N1, D], BF16
        softmax_max_out:     [N2, T1, group], FP32
        softmax_sum_out:     [N2, T1, group], FP32
        max_total_kv:        int, static upper bound for T2*N2 (default 128K)
    """
    dtype = q_nope.dtype
    dn = q_nope.shape[-1]        # D
    dr = q_pe.shape[-1]          # D_ROPE
    group = nq // n_kv
    s2_tile = tile_config.s_kv_tile  # = sparse_size
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape

    t1_sym = q_nope.shape[0]
    nq1 = q_nope.shape[1]
    t2_sym = compressed_kv_norm.shape[0]  # dynamic
    b_sym = npu_actual_q_len.shape[0]

    # Reshape inputs to 2D (inplace) for per-head view access
    for _ in pypto.loop(0, 1, 1, name="loop_reshape"):
        q_nope_2d = pypto.reshape(q_nope, [t1_sym * nq, dn], inplace=True)
        q_pe_2d = pypto.reshape(q_pe, [t1_sym * nq, dr], inplace=True)
        topk_1d = pypto.reshape(topk_indices, [t1_sym * n_kv * sparse_size], inplace=True)
        # Reshape KV to 2D: [T2*N2, D] / [T2*N2, D_ROPE] (dynamic shape)
        kv_2d_dyn = pypto.reshape(compressed_kv_norm, [t2_sym * n_kv, dn], inplace=True)
        k_pe_2d_dyn = pypto.reshape(k_pe, [t2_sym * n_kv, dr], inplace=True)

    out_2d = pypto.tensor([t1_sym * nq1, dn], dtype=pypto.DT_BF16)
    # Outer loop over B (batch dimension)
    for batch_idx in pypto.loop(0, b_sym, 1, name="LOOP_B", idx_name="bIdx"):
        cur_q_prefix = npu_actual_q_len[batch_idx]
        prev_q_prefix = npu_actual_q_len[(batch_idx - 1).max(0)] * (batch_idx > 0)
        s_per_batch = cur_q_prefix - prev_q_prefix

        cur_kv_prefix = npu_actual_kv_len[batch_idx]
        prev_kv_prefix = npu_actual_kv_len[(batch_idx - 1).max(0)] * (batch_idx > 0)
        cur_kv_len = cur_kv_prefix - prev_kv_prefix

        # Inner loop over s (per-batch query tokens)
        for s_idx in pypto.loop(0, s_per_batch, 1, name="LOOP_S", idx_name="sIdx"):
            eff_topk = (cur_kv_len - s_per_batch + 1 + s_idx).max(0).min(sparse_size)
            eff_topk_cond = (cur_kv_len - s_per_batch + 1 + s_idx).max(0)
            t_idx = prev_q_prefix + s_idx
            # View to fixed shape [max_total_kv, D] with valid_shape [T2*N2, D]
            kv_2d = pypto.view(kv_2d_dyn, [max_total_kv, dn], [0, 0],
                        valid_shape=[t2_sym * n_kv, dn])
            k_pe_2d = pypto.view(k_pe_2d_dyn, [max_total_kv, dr], [0, 0],
                            valid_shape=[t2_sym * n_kv, dr])
            for kv_head_idx in range(n_kv):                  
                if pypto.cond(eff_topk_cond <= sparse_size):
                    kv_sel = pypto.view(kv_2d_dyn, [sparse_size, dn], [prev_kv_prefix, 0],
                                valid_shape=[eff_topk, dn])
                    k_pe_sel = pypto.view(k_pe_2d_dyn, [sparse_size, dr], [prev_kv_prefix, 0],
                                    valid_shape=[eff_topk, dr])
                    # ==== Step 1: Get topk indices for (t_idx, kv_head_idx) ====
                else:
                    topk_row = t_idx * n_kv + kv_head_idx
                    cur_topk_1d = pypto.view(topk_1d, [s2_tile], [topk_row * sparse_size],
                                            valid_shape=[eff_topk])

                    # ==== Step 2: Transform indices & gather KV nope + K_pe ====
                    if n_kv > 1:
                        kv_gather_idx = pypto.add(pypto.mul(cur_topk_1d, n_kv), kv_head_idx)
                    else:
                        kv_gather_idx = cur_topk_1d

                    pypto.set_vec_tile_shapes(128, dn)
                    kv_sel = pypto.index_select(kv_2d, 0, kv_gather_idx)  # (s2_tile, D)

                    pypto.set_vec_tile_shapes(128, dr)
                    k_pe_sel = pypto.index_select(k_pe_2d, 0, kv_gather_idx)  # (s2_tile, D_ROPE)

                # ==== Step 3: Get Q slices for group query heads ====
                q_head_offset = t_idx * nq + kv_head_idx * group
                qi_nope = pypto.view(q_nope_2d, [group, dn], [q_head_offset, 0])
                qi_pe = pypto.view(q_pe_2d, [group, dr], [q_head_offset, 0])
                # ==== Step 4: S = Q_nope @ KV^T + Q_pe @ K_pe^T ====
                pypto.set_cube_tile_shapes([c1_tile[0], c1_tile[1]],
                    [c1_tile[2], c1_tile[3]], [c1_tile[4], c1_tile[5]])
                s_nope = pypto.matmul(qi_nope, kv_sel, pypto.DT_FP32,
                                    a_trans=False, b_trans=True)
                s_rope = pypto.matmul(qi_pe, k_pe_sel, pypto.DT_FP32,
                                    a_trans=False, b_trans=True)

                pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                sij = pypto.add(s_nope, s_rope)

                # ==== Step 6: Softmax ====
                pypto.set_semantic_label("Sa_V1")
                pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                sij_scale = pypto.mul(sij, scale)
                tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                t_sub = pypto.sub(sij_scale, tilda_mij)
                tilda_pij = pypto.exp(t_sub)
                tilda_lij = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                t_softmax = pypto.div(tilda_pij, tilda_lij)
                tilda_pij_bf16 = pypto.cast(t_softmax, dtype)

                # ==== Step 7: C2: Out = Softmax @ V ====
                pypto.set_semantic_label("Sa_C2")
                pypto.set_cube_tile_shapes([c2_tile[0], c2_tile[1]],
                    [c2_tile[2], c2_tile[3]], [c2_tile[4], c2_tile[5]])
                pypto.set_matrix_size([tilda_pij_bf16.shape[0],
                    tilda_pij_bf16.shape[1], dn])
                out = pypto.matmul(tilda_pij_bf16, kv_sel, dtype)  # (group, D)

                # ==== Step 8: Assemble directly to output tensors ====
                pypto.set_vec_tile_shapes(group, dn)
                out_3d = pypto.reshape(out, [1, group, dn])
                pypto.assemble(out_3d, [t_idx, kv_head_idx * group, 0], core_attn_out)

                pypto.set_vec_tile_shapes(1, 1, 128)
                sm_max_row = pypto.reshape(tilda_mij, [1, 1, group])
                sm_sum_row = pypto.reshape(tilda_lij, [1, 1, group])
                pypto.assemble(sm_max_row, [kv_head_idx, t_idx, 0], softmax_max_out)
                pypto.assemble(sm_sum_row, [kv_head_idx, t_idx, 0], softmax_sum_out)


@pypto.frontend.jit(
    pass_options={
        "cube_l1_reuse_setting": {-1: 16},
        "cube_nbuffer_setting": {2: 4},
        "vec_nbuffer_setting": {-2: 1, -1: 8},
    },
    runtime_options={
        "stitch_function_max_num": 128,
        "device_sched_mode": 1,
    },
)
def sfa_forward_tnd(
    q_nope: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    compressed_kv_norm: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    topk_indices: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    q_pe: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    k_pe: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    npu_actual_q_len: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    npu_actual_kv_len: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    core_attn_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    softmax_max_out: pypto.Tensor([pypto.STATIC, pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    softmax_sum_out: pypto.Tensor([pypto.STATIC, pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    nq, n_kv,
    scale, sparse_size,
    tile_config,
    max_total_kv=1024 * 1024
):
    """Factory function to create the SFA forward TND v2 kernel.

    Args:
        max_total_kv: Static upper bound for T2*N2 (default 128K).
                      compressed_kv_norm and k_pe are reshaped to [T2*N2, D]
                      then viewed to [max_total_kv, D] with valid_shape=[T2*N2, D].
    """
    
    pypto.experimental.set_operation_options(combine_axis=True)

    sfa_forward_tnd_compute(
        q_nope, compressed_kv_norm, topk_indices,
        q_pe, k_pe,
        npu_actual_q_len, npu_actual_kv_len,
        core_attn_out, softmax_max_out, softmax_sum_out,
        nq, n_kv, scale, sparse_size, max_total_kv, tile_config)

