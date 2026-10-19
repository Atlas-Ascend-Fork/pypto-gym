#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Flash Attention HiFP8 Forward with Dynamic Variable Length Sequences

HiFP8 (HF8) input tensors are dequantized to FP32 (per-token multiplicative scales)
before flash attention computation. P (attention probabilities) are quantized to HF8
using p_scale before the P@V matmul; p_scale cancels in the final output O = O / L.

4 loops: batch + head + q_tile + kv_tile.
"""

import pypto


Q_TILE = 256
K_TILE = 256


@pypto.frontend.jit(
    runtime_options={
        "device_sched_mode": 1,
        "max_workspace_kb": 4194304,
        "ready_on_host_tensors": ["cu_seqlens_q", "cu_seqlens_k"]
    },
    pass_options={
        "cube_l1_reuse_setting": {-1: 8},
        "vec_nbuffer_setting": {-1: 8},
        "cube_nbuffer_setting": {-1: 1},
        "ooo_sched_mode": "HLF",
    },
    host_options={"compile_monitor_enable": 0},
    codegen_options={
        "vf_options": "-mllvm -cce-vf-enable-vloopv2-recognizer=true -mllvm -enable-pto-colop-fusion=true"
    }
)
def flash_attention_fp8_varlen_forward_kernel(
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_HF8, format=pypto.TileOpFormat.TILEOP_ND),
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_HF8, format=pypto.TileOpFormat.TILEOP_ND),
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_HF8, format=pypto.TileOpFormat.TILEOP_ND),
    d_scale_q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_ND),
    d_scale_k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_ND),
    d_scale_v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_ND),
    p_scale: pypto.Tensor([pypto.DYNAMIC], pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_ND),
    output: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    l_output: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    m_output: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    cu_seqlens_q: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    cu_seqlens_k: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
):
    """
    Flash Attention HiFP8 Forward - 4 loops (batch + head + q_tile + kv_tile).

    Input tensors:
      q/k/v: 3D [total_seq, N, D], HF8 (HiFP8)
      d_scale_q/k/v: 3D [total_seq, N, 1], FP32 per-token dequant scales (multiplicative)
      p_scale: 1D [1], FP32 scalar for P quantization to HF8

    Output tensors:
      output: 2D [total_q, hidden_dim], BF16
      l_output/m_output: 2D [total_q, N], FP32

    Computation flow (per batch, per head, per q_tile, per kv_tile):
      1. Q_fp32 = cast(Q_hf8, FP32) * d_scale_q, K_fp32 = cast(K_hf8, FP32) * d_scale_k
      2. S_tile = Q_fp32 @ K_fp32^T * scale             [sq, sk]  FP32
      3. M_tile = max(S_tile, dim=-1)                    [sq, 1]   FP32
      4. P_tile = exp(S_tile - M_tile)                   [sq, sk]  FP32
      5. P_hf8 = cast(cast(P_tile * p_scale, HF8), FP32) [sq, sk]  FP32 (quantized)
      6. L_tile = sum(P_hf8, dim=-1)                     [sq, 1]   FP32 (includes p_scale)
      7. V_fp32 = cast(V_hf8, FP32) * d_scale_v
      8. O_tile = P_hf8 @ V_fp32                         [sq, D]   FP32
      O/L/M accumulated across kv tiles (online softmax). p_scale cancels in final O = O / L.
    """
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    hidden_dim = num_heads * head_dim
    total_q = q.shape[0]
    total_kv = k.shape[0]
    scale = 1.0 / (head_dim ** 0.5)

    q_2d = pypto.reshape(q, [total_q, hidden_dim], inplace=True)
    k_2d = pypto.reshape(k, [total_kv, hidden_dim], inplace=True)
    v_2d = pypto.reshape(v, [total_kv, hidden_dim], inplace=True)

    d_scale_q_2d = pypto.reshape(d_scale_q, [total_q, num_heads], inplace=True)
    d_scale_k_2d = pypto.reshape(d_scale_k, [total_kv, num_heads], inplace=True)
    d_scale_v_2d = pypto.reshape(d_scale_v, [total_kv, num_heads], inplace=True)
    p_scale_2d = pypto.reshape(p_scale, [1, 1], inplace=True)

    v1_tile = [64, 128]
    v2_tile = [64, 128]

    q_tile = Q_TILE
    k_tile = K_TILE

    pypto.experimental.set_operation_options(combine_axis=True)

    batch_size = cu_seqlens_q.shape[0] - 1
    for b_idx in pypto.loop(batch_size, name="batch_loop"):
        q_start = cu_seqlens_q[b_idx]
        q_end = cu_seqlens_q[b_idx + 1]
        seq_len_q = q_end - q_start

        k_start = cu_seqlens_k[b_idx]
        k_end = cu_seqlens_k[b_idx + 1]
        seq_len_k = k_end - k_start

        q_tile_count = (seq_len_q + q_tile - 1) // q_tile
        k_tile_count = (seq_len_k + k_tile - 1) // k_tile

        for h_idx in pypto.loop(num_heads, name="head_loop"):

            for q_tile_idx in pypto.loop(q_tile_count, name="q_tile_loop"):
                oi_update = pypto.tensor([q_tile, head_dim], pypto.DT_FP32, "oi_update")
                li_update = pypto.tensor([q_tile, 1], pypto.DT_FP32, "li_update")
                mi_update = pypto.tensor([q_tile, 1], pypto.DT_FP32, "mi_update")
                q_tile_start = q_tile_idx * q_tile
                q_tile_end = pypto.min(q_tile_start + q_tile, seq_len_q)
                q_tile_len = q_tile_end - q_tile_start

                for k_tile_idx in pypto.loop(k_tile_count, name="k_tile_loop", unroll_list=[16]):
                    k_tile_start = k_tile_idx * k_tile
                    k_tile_end = pypto.min(k_tile_start + k_tile, seq_len_k)
                    k_tile_len = k_tile_end - k_tile_start

                    h_offset = h_idx * head_dim
                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                    pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                    if pypto.platform.npuarch == 'DAV_3510':
                        pypto.set_pass_options(sg_set_scope=5001)

                    q_tile_view = pypto.view(q_2d, [q_tile, head_dim],
                                                [q_start + q_tile_start, h_offset],
                                                valid_shape=[q_tile_len, head_dim])
                    k_tile_view = pypto.view(k_2d, [k_tile, head_dim],
                                                [k_start + k_tile_start, h_offset],
                                                valid_shape=[k_tile_len, head_dim])
                    v_tile_view = pypto.view(v_2d, [k_tile, head_dim],
                                                [k_start + k_tile_start, h_offset],
                                                valid_shape=[k_tile_len, head_dim])
                    dscale_q_view = pypto.view(d_scale_q_2d, [q_tile, 1],
                                                [q_start + q_tile_start, h_idx],
                                                valid_shape=[q_tile_len, 1])
                    dscale_k_view = pypto.view(d_scale_k_2d, [k_tile, 1],
                                                [k_start + k_tile_start, h_idx],
                                                valid_shape=[k_tile_len, 1])
                    dscale_v_view = pypto.view(d_scale_v_2d, [k_tile, 1],
                                                [k_start + k_tile_start, h_idx],
                                                valid_shape=[k_tile_len, 1])

                    q_fp32 = pypto.cast(q_tile_view, pypto.DT_FP32) * dscale_q_view
                    k_fp32 = pypto.cast(k_tile_view, pypto.DT_FP32) * dscale_k_view
                    v_fp32 = pypto.cast(v_tile_view, pypto.DT_FP32) * dscale_v_view

                    scores = pypto.matmul(q_fp32, k_fp32, out_dtype=pypto.DT_FP32, b_trans=True)

                    scores_scaled = pypto.mul(scores, scale)
                    mij = pypto.amax(scores_scaled, dim=-1, keepdim=True)
                    s_shifted = pypto.sub(scores_scaled, mij)
                    pij = pypto.exp(s_shifted)
                    pij_scaled = pypto.mul(pij, p_scale_2d)
                    pij_hf8 = pypto.cast(pij_scaled, pypto.DT_HF8)
                    pij = pypto.cast(pij_hf8, pypto.DT_FP32)
                    lij = pypto.sum(pij, dim=-1, keepdim=True)

                    pij_bf16 = pypto.cast(pij, pypto.DT_BF16)
                    v_bf16 = pypto.cast(v_fp32, pypto.DT_BF16)
                    oij = pypto.matmul(pij_bf16, v_bf16, out_dtype=pypto.DT_FP32)
                    if pypto.platform.npuarch == 'DAV_3510':
                        pypto.set_pass_options(sg_set_scope=-1)

                        pypto.set_pass_options(sg_set_scope=1)
                    pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                    if pypto.is_loop_begin(k_tile_idx):
                        if pypto.is_loop_end(k_tile_idx):
                            oij_tmp = pypto.div(oij, lij, precision_type=pypto.PrecisionType.INTRINSIC)
                            oij_final = pypto.cast(oij_tmp, pypto.DT_BF16)
                            pypto.assemble(lij, [q_start + q_tile_start, h_idx], l_output)
                            pypto.assemble(mij, [q_start + q_tile_start, h_idx], m_output)
                            pypto.assemble(oij_final, [q_start + q_tile_start, h_offset], output)

                        else:
                            oi_update[:] = oij
                            li_update[:] = lij
                            mi_update[:] = mij
                    else:
                        li = pypto.view(li_update, [q_tile, 1], [0, 0], valid_shape=[q_tile_len, 1])
                        mi = pypto.view(mi_update, [q_tile, 1], [0, 0], valid_shape=[q_tile_len, 1])
                        oi = pypto.view(oi_update, [q_tile, head_dim], [0, 0], valid_shape=[q_tile_len, head_dim])

                        mi_new = pypto.maximum(mi, mij)
                        t1 = pypto.sub(mi, mi_new)
                        t2 = pypto.exp(t1)
                        t3 = pypto.sub(mij, mi_new)
                        t4 = pypto.exp(t3)

                        t2_li = pypto.mul(t2, li)
                        t4_lij = pypto.mul(t4, lij)
                        li_new = pypto.add(t2_li, t4_lij)

                        oi_t2 = pypto.mul(oi, t2)
                        oij_t4 = pypto.mul(oij, t4)
                        oi_tmp = pypto.add(oi_t2, oij_t4)

                        if pypto.is_loop_end(k_tile_idx):
                            out_fp32 = pypto.div(oi_tmp, li_new, precision_type=pypto.PrecisionType.INTRINSIC)
                            out_bf16 = pypto.cast(out_fp32, pypto.DT_BF16)

                            pypto.assemble(li_new, [q_start + q_tile_start, h_idx], l_output)
                            pypto.assemble(mi_new, [q_start + q_tile_start, h_idx], m_output)
                            pypto.assemble(out_bf16, [q_start + q_tile_start, h_offset], output)
                        else:
                            oi_update[:] = oi_tmp
                            li_update[:] = li_new
                            mi_update[:] = mi_new
                    if pypto.platform.npuarch == 'DAV_3510':
                        pypto.set_pass_options(sg_set_scope=-1)