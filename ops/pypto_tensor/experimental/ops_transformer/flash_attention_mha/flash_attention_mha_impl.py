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
Flash Attention Forward with Dynamic Variable Length Sequences

语义约定:
  - Q 侧: s1_size (Q seqlen), 张量包括 Q/O/L/M
  - KV 侧: s2_size (KV seqlen), 张量包括 K/V
  - Q_TILE/K_TILE: 序列维度的分块大小 (将 s1/s2 切分为多个 tile 迭代)

4 loops: batch + head + q_tile + kv_tile.
Tiles Q and KV sequence dimensions by Q_TILE/K_TILE to reduce intermediate
attention matrix from [s1_size, s2_size] to [Q_TILE, K_TILE] per iteration.
O, L, M are accumulated across kv tiles (online softmax algorithm).
"""

from dataclasses import dataclass
import pypto


@dataclass
class FlashAttentionTileShapeConfig:
    q_tile: int
    k_tile: int
    c1_cube_tile: list
    v1_tile: list
    c2_cube_tile: list
    v2_tile: list


def _validate_tile_config(tile_config):
    """
    校验 FlashAttentionTileShapeConfig 的 tile_config 是否符合预期。
    预期：
      - q_tile == k_tile == 128
      - c1_cube_tile == c2_cube_tile == [[128, 128], [128, 128], [128, 128]]
      - v1_tile == v2_tile == [128, 64]
    """
    # 校验 q_tile 和 k_tile
    if tile_config.q_tile != 128 or tile_config.k_tile != 128:
        return False

    # 校验 c1_cube_tile 和 c2_cube_tile
    expected_cube_tile = [[128, 128], [128, 128], [128, 128]]
    if tile_config.c1_cube_tile != expected_cube_tile or tile_config.c2_cube_tile != expected_cube_tile:
        return False

    # 校验 v1_tile 和 v2_tile
    expected_v_tile = [128, 64]
    if tile_config.v1_tile != expected_v_tile or tile_config.v2_tile != expected_v_tile:
        return False

    return True


def flash_attention_varlen_forward_kernel(
    # Q侧输入: shape=[total_q, N, D], total_q=DYNAMIC, N=num_heads, D=head_dim
    q: pypto.Tensor,
    # KV侧输入: shape=[total_kv, N, D]
    k: pypto.Tensor,
    v: pypto.Tensor,
    # Q侧输出: shape=[total_q, hidden_dim] (二维)
    output: pypto.Tensor,
    # Q侧: softmax中间量L, shape=[total_q, n] (二维)
    l_output: pypto.Tensor,
    # Q侧: softmax中间量M, shape=[total_q, n] (二维)
    m_output: pypto.Tensor,
    # 累积序列长度: shape=[batch_size + 1]
    cu_seqlens_q: pypto.Tensor,
    cu_seqlens_k: pypto.Tensor,
    # TileShape配置
    tile_config: FlashAttentionTileShapeConfig,
):
    """
    Flash Attention Forward - 4 loops (batch + head + q_tile + kv_tile).

    输入张量为三维 [total_seq, N, D] (total_seq=DYNAMIC, N=num_heads, D=head_dim),
    输出张量为二维 [total_seq, hidden_dim] (hidden_dim = N*D)。

    Kernel入口处从输入tensor shape 获取 num_heads 和 head_dim:
      num_heads = q.shape[1]
      head_dim = q.shape[2]
      hidden_dim = num_heads * head_dim

    然后 reshape inplace 输入为二维 [total_seq, hidden_dim] 以便按 head 做 view 切片，
    输出保持二维布局。

    张量布局 (Q: s1_size, KV: s2_size):
      Q 侧: Q/O/L/M — 每个 batch 占用 s1 行 (通过 cu_seqlens_q 定位)
      KV 侧: K/V    — 每个 batch 占用 s2_total 行 (通过 cu_seqlens_k 定位)

    cu_seqlens_q.shape[0] = batch_size + 1
    cu_seqlens_q[i]       = 前 i 个 batch 的累积 Q seqlen
    cu_seqlens_k[i]       = 前 i 个 batch 的累积 KV seqlen

    计算流程 (per batch, per head, per q_tile, per kv_tile):
      对 Q seqlen 按 q_tile 分块, 对 KV seqlen 按 k_tile 分块:
        S_tile = Q_tile @ K_tile^T * scale         [sq, sk]       BF16→FP32
        M_tile = max(S_tile, dim=-1)               [sq, 1]        FP32
        P_tile = exp(S_tile - M_tile)              [sq, sk]       FP32
        L_tile = sum(P_tile, dim=-1)               [sq, 1]        FP32
        P_norm = P_tile / L_tile                   [sq, sk]       FP32→BF16
        O_tile = P_norm @ V_tile                   [sq, D]        BF16
      O/L/M 在 kv_tile 循环中累加 (online softmax), 最终写回。

    Kernel dtype 转换流程:
      1. Q/K/V: BF16 输入
      2. scores = Q(BF16) @ K^T(BF16) → FP32      (matmul out_dtype=FP32)
      3. scores_scaled = scores * scale → FP32    (mul)
      4. M = max(scores_scaled) → FP32            (amax)
      5. P = exp(scores_scaled - M) → FP32        (exp after sub)
      6. L = sum(P) → FP32                        (sum)
      7. P_norm = P / L → FP32                    (div)
      8. P_bf16 = cast(P_norm, BF16)              (pypto.cast → DT_BF16)
      9. O = P_bf16 @ V(BF16) → BF16              (matmul out_dtype=BF16)
     10. L, M 保持 FP32 写回
    """
    if pypto.platform.npuarch == 'DAV_3510' and _validate_tile_config(tile_config):
        flash_attention_varlen_forward_950(q, k, v, output, l_output, m_output, cu_seqlens_q, cu_seqlens_k, tile_config)
    else:
        flash_attention_varlen_forward(q, k, v, output, l_output, m_output, cu_seqlens_q, cu_seqlens_k, tile_config)


@pypto.frontend.jit(
    runtime_options={
        "device_sched_mode": 1,
        "stitch_function_max_num": 1024,
        "ready_on_host_tensors": ["cu_seqlens_q", "cu_seqlens_k"]
    },
    pass_options={
        "cube_l1_reuse_setting": {-1: 1},
        "vec_nbuffer_setting": {-1: 1},
        "ooo_sched_mode": "HLF",
    },
)
def flash_attention_varlen_forward_950(
    # Q侧输入: shape=[total_q, N, D], total_q=DYNAMIC, N=num_heads, D=head_dim
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    # KV侧输入: shape=[total_kv, N, D]
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    # Q侧输出: shape=[total_q, hidden_dim] (二维)
    output: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    # Q侧: softmax中间量L, shape=[total_q, n] (二维)
    l_output: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    # Q侧: softmax中间量M, shape=[total_q, n] (二维)
    m_output: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    # 累积序列长度: shape=[batch_size + 1]
    cu_seqlens_q: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    cu_seqlens_k: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    # TileShape配置
    tile_config: FlashAttentionTileShapeConfig,
):
    # ---- 从三维输入获取 N(num_heads) 和 D(head_dim), 然后 reshape 为二维 ----
    total_kv = k.shape[0]
    total_q = q.shape[0]
    head_dim = q.shape[2]
    num_heads = q.shape[1]
    hidden_dim = num_heads * head_dim
    scale = 1.0 / (head_dim ** 0.5)

    # reshape inplace: q/k/v [total_seq, N, D] → [total_seq, N*D]
    v_2d = pypto.reshape(v, [total_kv, hidden_dim], inplace=True)
    k_2d = pypto.reshape(k, [total_kv, hidden_dim], inplace=True)
    q_2d = pypto.reshape(q, [total_q, hidden_dim], inplace=True)
    # output/l/m 保持二维，无需 reshape

    q_tile = tile_config.q_tile
    k_tile = tile_config.k_tile
    v1_tile = tile_config.v1_tile
    v2_tile = tile_config.v2_tile

    pypto.experimental.set_operation_options(combine_axis=True)

    # 累计Q序列长度 batch_size + 1
    batch_size = cu_seqlens_q.shape[0] - 1
    for b_idx in pypto.loop(batch_size, name="batch_loop"):
        q_end = cu_seqlens_q[b_idx + 1]
        q_start = cu_seqlens_q[b_idx]
        seq_len_q = q_end - q_start
        seq_len_q = pypto.experimental.assume_divisible(seq_len_q, q_tile)

        k_start = cu_seqlens_k[b_idx]
        k_end = cu_seqlens_k[b_idx + 1]
        seq_len_k = k_end - k_start
        seq_len_k = pypto.experimental.assume_divisible(seq_len_k, k_tile)

        k_tile_count = (seq_len_k + k_tile - 1) // k_tile
        q_tile_count = (seq_len_q + q_tile - 1) // q_tile

        for h_idx in pypto.loop(num_heads, name="head_loop"):

            for q_tile_idx in pypto.loop(q_tile_count, name="q_tile_loop"):
                mi_update = pypto.tensor([1, q_tile], pypto.DT_FP32, "mi_update")
                li_update = pypto.tensor([1, q_tile], pypto.DT_FP32, "li_update")
                oi_update = pypto.tensor([head_dim, q_tile], pypto.DT_FP32, "oi_update")

                q_tile_start = q_tile_idx * q_tile
                q_tile_end = pypto.min(q_tile_start + q_tile, seq_len_q)
                q_tile_len = q_tile_end - q_tile_start

                for k_tile_idx in pypto.loop(k_tile_count, name="k_tile_loop", unroll_list=[32]):
                    k_tile_start = k_tile_idx * k_tile
                    k_tile_end = pypto.min(k_tile_start + k_tile, seq_len_k)
                    k_tile_len = k_tile_end - k_tile_start

                    h_offset = h_idx * head_dim
                    v_tile_view = pypto.view(v_2d, [k_tile, head_dim],
                                        [k_start + k_tile_start, h_offset],
                                        valid_shape=[k_tile_len, head_dim])
                    k_tile_view = pypto.view(k_2d, [k_tile, head_dim],
                                        [k_start + k_tile_start, h_offset],
                                        valid_shape=[k_tile_len, head_dim])
                    q_tile_view = pypto.view(q_2d, [q_tile, head_dim],
                                        [q_start + q_tile_start, h_offset],
                                        valid_shape=[q_tile_len, head_dim])

                    pypto.set_pass_options(sg_set_scope=5001)

                    pypto.set_cube_tile_shapes(
                        tile_config.c1_cube_tile[0],
                        tile_config.c1_cube_tile[1],
                        tile_config.c1_cube_tile[2])
                    scores = pypto.matmul(k_tile_view, q_tile_view, out_dtype=pypto.DT_FP32, b_trans=True)

                    pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])


                    if pypto.is_loop_begin(k_tile_idx):

                        if pypto.is_loop_end(k_tile_idx):
                            mij_a = pypto.amax(scores, dim=0, keepdim=True)
                            mij = pypto.mul(mij_a, scale)
                            scores_scaled = pypto.mul(scores, scale)

                            s_shifted = pypto.sub(scores_scaled, mij)
                            pij = pypto.exp(s_shifted)
                            pij_bf16 = pypto.cast(pij, pypto.DT_BF16)
                            lij = pypto.sum(pij, dim=0, keepdim=True)

                            pij_div = pypto.div(pij, lij, precision_type=pypto.PrecisionType.INTRINSIC)
                            pij_bf16 = pypto.cast(pij_div, pypto.DT_BF16)

                            oij = pypto.matmul(v_tile_view, pij_bf16, out_dtype=pypto.DT_BF16, a_trans=True)

                            mij_r = pypto.transpose(mij, 0, 1)
                            lij_r = pypto.transpose(lij, 0, 1)
                            oij_t = pypto.transpose(oij, 0, 1)

                            pypto.set_vec_tile_shapes(v2_tile[1], v2_tile[0])

                            pypto.assemble(mij_r, [q_start + q_tile_start, h_idx], m_output)
                            pypto.assemble(lij_r, [q_start + q_tile_start, h_idx], l_output)
                            pypto.assemble(oij_t, [q_start + q_tile_start, h_offset], output)

                        else:
                            pypto.set_pass_options(sg_set_ooo_scope=1)
                            pij_bf16, mij, lij = pypto.experimental.online_softmax(scores, scale)
                            pypto.set_pass_options(sg_set_ooo_scope=-1)

                            pypto.set_cube_tile_shapes(
                                tile_config.c2_cube_tile[0], tile_config.c2_cube_tile[1], tile_config.c2_cube_tile[2])
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])

                            oij = pypto.matmul(v_tile_view, pij_bf16, out_dtype=pypto.DT_FP32, a_trans=True)

                            mi_update[:] = mij
                            li_update[:] = lij
                            oi_update[:] = oij

                    else:
                        pypto.set_pass_options(sg_set_ooo_scope=1)
                        pij_bf16, mij, lij = pypto.experimental.online_softmax(scores, scale)
                        pypto.set_pass_options(sg_set_ooo_scope=-1)

                        pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                        pypto.set_cube_tile_shapes(
                            tile_config.c2_cube_tile[0], tile_config.c2_cube_tile[1], tile_config.c2_cube_tile[2])
                        oij = pypto.matmul(v_tile_view, pij_bf16, out_dtype=pypto.DT_FP32, a_trans=True)

                        mi = pypto.view(mi_update, [1, q_tile], [0, 0], valid_shape=[1, q_tile_len])
                        li = pypto.view(li_update, [1, q_tile], [0, 0], valid_shape=[1, q_tile_len])
                        oi = pypto.view(oi_update, [head_dim, q_tile], [0, 0], valid_shape=[head_dim, q_tile_len])

                        pypto.set_pass_options(sg_set_ooo_scope=2)
                        mi_new, li_new, oi_tmp = pypto.experimental.online_softmax_update(mi, li, oi, mij, lij, oij)
                        if pypto.is_loop_end(k_tile_idx):
                            out_fp32 = pypto.div(oi_tmp, li_new, precision_type=pypto.PrecisionType.INTRINSIC)
                            out_bf16 = pypto.cast(out_fp32, pypto.DT_BF16)
                            pypto.set_pass_options(sg_set_ooo_scope=-1)

                            mi_new_r = pypto.transpose(mi_new, 0, 1)
                            li_new_r = pypto.transpose(li_new, 0, 1)
                            out_bf16_t = pypto.transpose(out_bf16, 0, 1)

                            pypto.set_vec_tile_shapes(v2_tile[1], v2_tile[0])

                            pypto.assemble(mi_new_r, [q_start + q_tile_start, h_idx], m_output)
                            pypto.assemble(li_new_r, [q_start + q_tile_start, h_idx], l_output)
                            pypto.assemble(out_bf16_t, [q_start + q_tile_start, h_offset], output)

                        else:
                            pypto.set_pass_options(sg_set_ooo_scope=-1)

                            oi_update[:] = oi_tmp
                            li_update[:] = li_new
                            mi_update[:] = mi_new

                    pypto.set_pass_options(sg_set_scope=-1)


@pypto.frontend.jit(
    runtime_options={
        "device_sched_mode": 0,
        "stitch_function_max_num": 1024,
        "ready_on_host_tensors": ["cu_seqlens_q", "cu_seqlens_k"],
    },
    pass_options={
        "cube_l1_reuse_setting": {-1: 8},
        "vec_nbuffer_setting": {-1: 8},
        "cube_nbuffer_setting": {-1: 8},
    }
)
def flash_attention_varlen_forward(
    # Q侧输入: shape=[total_q, N, D], total_q=DYNAMIC, N=num_heads, D=head_dim
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    # KV侧输入: shape=[total_kv, N, D]
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    # Q侧输出: shape=[total_q, hidden_dim] (二维)
    output: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    # Q侧: softmax中间量L, shape=[total_q, n] (二维)
    l_output: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    # Q侧: softmax中间量M, shape=[total_q, n] (二维)
    m_output: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    # 累积序列长度: shape=[batch_size + 1]
    cu_seqlens_q: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    cu_seqlens_k: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    # TileShape配置
    tile_config: FlashAttentionTileShapeConfig,
):
    # ---- 从三维输入获取 N(num_heads) 和 D(head_dim), 然后 reshape 为二维 ----
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    hidden_dim = num_heads * head_dim
    total_q = q.shape[0]
    total_kv = k.shape[0]
    scale = 1.0 / (head_dim ** 0.5)

    # reshape inplace: q/k/v [total_seq, N, D] → [total_seq, N*D]
    q_2d = pypto.reshape(q, [total_q, hidden_dim], inplace=True)
    k_2d = pypto.reshape(k, [total_kv, hidden_dim], inplace=True)
    v_2d = pypto.reshape(v, [total_kv, hidden_dim], inplace=True)
    # output/l/m 保持二维，无需 reshape

    q_tile = tile_config.q_tile
    k_tile = tile_config.k_tile
    v1_tile = tile_config.v1_tile
    v2_tile = tile_config.v2_tile

    pypto.experimental.set_operation_options(combine_axis=True)

    # 累计Q序列长度 batch_size + 1
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

        h_num = num_heads // 2
        for h_idx in pypto.loop(h_num, name="head_loop"):

            for q_tile_idx in pypto.loop(q_tile_count, name="q_tile_loop"):
                # 创建2个head独立的累加器
                oi_update_0 = pypto.tensor([q_tile, head_dim], pypto.DT_FP32, "oi_update")
                li_update_0 = pypto.tensor([q_tile, 1], pypto.DT_FP32, "li_update")
                mi_update_0 = pypto.tensor([q_tile, 1], pypto.DT_FP32, "mi_update")
                oi_update_1 = pypto.tensor([q_tile, head_dim], pypto.DT_FP32, "oi_update")
                li_update_1 = pypto.tensor([q_tile, 1], pypto.DT_FP32, "li_update")
                mi_update_1 = pypto.tensor([q_tile, 1], pypto.DT_FP32, "mi_update")

                q_tile_start = q_tile_idx * q_tile
                q_tile_end = pypto.min(q_tile_start + q_tile, seq_len_q)
                q_tile_len = q_tile_end - q_tile_start

                for k_tile_idx in pypto.loop(k_tile_count, name="k_tile_loop"):
                    k_tile_start = k_tile_idx * k_tile
                    k_tile_end = pypto.min(k_tile_start + k_tile, seq_len_k)
                    k_tile_len = k_tile_end - k_tile_start

                    for h_s_idx in range(2):
                        h_act_idx = h_idx * 2 + h_s_idx
                        h_offset = h_act_idx * head_dim

                        if h_s_idx == 0:
                            li_update = li_update_0
                            mi_update = mi_update_0
                            oi_update = oi_update_0
                        else:
                            li_update = li_update_1
                            mi_update = mi_update_1
                            oi_update = oi_update_1

                        q_tile_view = pypto.view(q_2d, [q_tile, head_dim],
                                            [q_start + q_tile_start, h_offset],
                                            valid_shape=[q_tile_len, head_dim])

                        k_tile_view = pypto.view(k_2d, [k_tile, head_dim],
                                            [k_start + k_tile_start, h_offset],
                                            valid_shape=[k_tile_len, head_dim])
                        v_tile_view = pypto.view(v_2d, [k_tile, head_dim],
                                            [k_start + k_tile_start, h_offset],
                                            valid_shape=[k_tile_len, head_dim])

                        pypto.set_cube_tile_shapes(
                            tile_config.c1_cube_tile[0],
                            tile_config.c1_cube_tile[1],
                            tile_config.c1_cube_tile[2])

                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])

                        if pypto.platform.npuarch == 'DAV_3510':
                            pypto.set_pass_options(sg_set_scope=5001)

                        scores = pypto.matmul(q_tile_view, k_tile_view, out_dtype=pypto.DT_FP32, b_trans=True)

                        scores_scaled = pypto.mul(scores, scale)
                        mij = pypto.amax(scores_scaled, dim=-1, keepdim=True)
                        s_shifted = pypto.sub(scores_scaled, mij)
                        pij = pypto.exp(s_shifted)
                        lij = pypto.sum(pij, dim=-1, keepdim=True)

                        pypto.set_cube_tile_shapes(
                            tile_config.c2_cube_tile[0],
                            tile_config.c2_cube_tile[1],
                            tile_config.c2_cube_tile[2])

                        if pypto.is_loop_begin(k_tile_idx):
                            if pypto.is_loop_end(k_tile_idx):
                                pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                                pij_div = pypto.div(pij, lij, precision_type=pypto.PrecisionType.INTRINSIC)
                                pij_bf16 = pypto.cast(pij_div, pypto.DT_BF16)

                                oij = pypto.matmul(pij_bf16, v_tile_view, out_dtype=pypto.DT_BF16)

                                if pypto.platform.npuarch == 'DAV_3510':
                                    pypto.set_pass_options(sg_set_scope=-1)

                                pypto.assemble(lij, [q_start + q_tile_start, h_act_idx], l_output)
                                pypto.assemble(mij, [q_start + q_tile_start, h_act_idx], m_output)
                                pypto.assemble(oij, [q_start + q_tile_start, h_offset], output)

                            else:
                                pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                                pij_bf16 = pypto.cast(pij, pypto.DT_BF16)

                                oij = pypto.matmul(pij_bf16, v_tile_view, out_dtype=pypto.DT_FP32)

                                if pypto.platform.npuarch == 'DAV_3510':
                                    pypto.set_pass_options(sg_set_scope=-1)

                                oi_update[:] = oij
                                li_update[:] = lij
                                mi_update[:] = mij
                        else:
                            pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                            pij_bf16 = pypto.cast(pij, pypto.DT_BF16)

                            oij = pypto.matmul(pij_bf16, v_tile_view, out_dtype=pypto.DT_FP32)

                            if pypto.platform.npuarch == 'DAV_3510':
                                pypto.set_pass_options(sg_set_scope=-1)

                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])

                            li = pypto.view(li_update, [q_tile, 1], [0, 0], valid_shape=[q_tile_len, 1])
                            mi = pypto.view(mi_update, [q_tile, 1], [0, 0], valid_shape=[q_tile_len, 1])
                            oi = pypto.view(oi_update, [q_tile, head_dim], [0, 0], valid_shape=[q_tile_len, head_dim])

                            mi_new = pypto.maximum(mi, mij)
                            t1 = pypto.sub(mi, mi_new)
                            t2 = pypto.exp(t1)
                            t3 = pypto.sub(mij, mi_new)
                            t4 = pypto.exp(t3)

                            li_new = pypto.add(pypto.mul(t2, li), pypto.mul(t4, lij))
                            oi_tmp = pypto.add(pypto.mul(oi, t2), pypto.mul(oij, t4))

                            if pypto.is_loop_end(k_tile_idx):
                                out_fp32 = pypto.div(oi_tmp, li_new, precision_type=pypto.PrecisionType.INTRINSIC)
                                out_bf16 = pypto.cast(out_fp32, pypto.DT_BF16)

                                pypto.assemble(li_new, [q_start + q_tile_start, h_act_idx], l_output)
                                pypto.assemble(mi_new, [q_start + q_tile_start, h_act_idx], m_output)
                                pypto.assemble(out_bf16, [q_start + q_tile_start, h_offset], output)
                            else:
                                oi_update[:] = oi_tmp
                                li_update[:] = li_new
                                mi_update[:] = mi_new


@pypto.frontend.jit(
    runtime_options={
        "device_sched_mode": 3,
        "stitch_function_max_num": 256,
        "ready_on_host_tensors": ["cu_seqlens_q", "cu_seqlens_k"],
    },
    pass_options={
        "cube_l1_reuse_setting": {0: 8, 1: 1},
        "cube_nbuffer_setting": {0: 1, 1: 1},
        "vec_nbuffer_setting": {-2: 1, 0: 8, 1: 2},
    },
    host_options={
        "compile_monitor_enable": 1,
    },
)
def flash_attention_varlen_forward_kernel_910(
    # Q侧输入: shape=[total_q, N, D], total_q=DYNAMIC, N=num_heads, D=head_dim
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    # KV侧输入: shape=[total_kv, N, D]
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    # Q侧输出: shape=[total_q, hidden_dim] (二维)
    output: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    # Q侧: softmax中间量L, shape=[total_q, n] (二维)
    l_output: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    # Q侧: softmax中间量M, shape=[total_q, n] (二维)
    m_output: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    # 累积序列长度: shape=[batch_size + 1]
    cu_seqlens_q: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    cu_seqlens_k: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    # TileShape配置
    tile_config: FlashAttentionTileShapeConfig,
):
    """
    Flash Attention Forward - 4 loops (batch + head + q_tile + kv_tile).

    输入张量为三维 [total_seq, N, D] (total_seq=DYNAMIC, N=num_heads, D=head_dim),
    输出张量为二维 [total_seq, hidden_dim] (hidden_dim = N*D)。

    Kernel入口处从输入tensor shape 获取 num_heads 和 head_dim:
      num_heads = q.shape[1]
      head_dim = q.shape[2]
      hidden_dim = num_heads * head_dim

    然后 reshape inplace 输入为二维 [total_seq, hidden_dim] 以便按 head 做 view 切片，
    输出保持二维布局。

    张量布局 (Q: s1_size, KV: s2_size):
      Q 侧: Q/O/L/M — 每个 batch 占用 s1 行 (通过 cu_seqlens_q 定位)
      KV 侧: K/V    — 每个 batch 占用 s2_total 行 (通过 cu_seqlens_k 定位)

    cu_seqlens_q.shape[0] = batch_size + 1
    cu_seqlens_q[i]       = 前 i 个 batch 的累积 Q seqlen
    cu_seqlens_k[i]       = 前 i 个 batch 的累积 KV seqlen

    计算流程 (per batch, per head, per q_tile, per kv_tile):
      对 Q seqlen 按 q_tile 分块, 对 KV seqlen 按 k_tile 分块:
        S_tile = Q_tile @ K_tile^T * scale         [sq, sk]       BF16→FP32
        M_tile = max(S_tile, dim=-1)               [sq, 1]        FP32
        P_tile = exp(S_tile - M_tile)              [sq, sk]       FP32
        L_tile = sum(P_tile, dim=-1)               [sq, 1]        FP32
        P_norm = P_tile / L_tile                   [sq, sk]       FP32→BF16
        O_tile = P_norm @ V_tile                   [sq, D]        BF16
      O/L/M 在 kv_tile 循环中累加 (online softmax), 最终写回。

    Kernel dtype 转换流程:
      1. Q/K/V: BF16 输入
      2. scores = Q(BF16) @ K^T(BF16) → FP32      (matmul out_dtype=FP32)
      3. scores_scaled = scores * scale → FP32    (mul)
      4. M = max(scores_scaled) → FP32            (amax)
      5. P = exp(scores_scaled - M) → FP32        (exp after sub)
      6. L = sum(P) → FP32                        (sum)
      7. P_norm = P / L → FP32                    (div)
      8. P_bf16 = cast(P_norm, BF16)              (pypto.cast → DT_BF16)
      9. O = P_bf16 @ V(BF16) → BF16              (matmul out_dtype=BF16)
     10. L, M 保持 FP32 写回
    """

    # ---- 从三维输入获取 N(num_heads) 和 D(head_dim), 然后 reshape 为二维 ----
    num_heads = q.shape[1] # 8
    head_dim = q.shape[2] # 128
    hidden_dim = num_heads * head_dim # 8*128 = 1024
    total_q = q.shape[0] # 4096
    total_kv = k.shape[0] # 4096
    scale = 1.0 / (head_dim ** 0.5) # 1 / (128 ** 0.5)

    # reshape inplace: q/k/v [total_seq, N, D] → [total_seq, N*D]
    q_2d = pypto.reshape(q, [total_q, hidden_dim], inplace=True)
    k_2d = pypto.reshape(k, [total_kv, hidden_dim], inplace=True)
    v_2d = pypto.reshape(v, [total_kv, hidden_dim], inplace=True)
    # output/l/m 保持二维，无需 reshape

    v1_tile = tile_config.v1_tile
    v2_tile = tile_config.v2_tile

    q_tile = tile_config.q_tile
    k_tile = tile_config.k_tile

    pypto.experimental.set_operation_options(combine_axis=True)

    # 累计Q序列长度 batch_size + 1
    batch_size = cu_seqlens_q.shape[0] - 1
    for b_idx in pypto.loop(batch_size, name="batch_loop"):
        q_start = cu_seqlens_q[b_idx]
        q_end = cu_seqlens_q[b_idx + 1]
        seq_len_q = q_end - q_start

        k_start = cu_seqlens_k[b_idx]
        k_end = cu_seqlens_k[b_idx + 1]
        seq_len_k = 4096

        q_tile_count = (seq_len_q + q_tile - 1) // q_tile
        k_tile_count = (seq_len_k + k_tile - 1) // k_tile

        sta_num = 1
        h_num = num_heads // sta_num
        for h_idx in pypto.loop(h_num, name="head_loop"):
            for q_tile_idx in pypto.loop(q_tile_count, name="q_tile_loop"):
                oi_update = pypto.tensor([q_tile, head_dim], pypto.DT_FP32, "oi_update")
                li_update = pypto.tensor([q_tile, 1], pypto.DT_FP32, "li_update")
                mi_update = pypto.tensor([q_tile, 1], pypto.DT_FP32, "mi_update")

                q_tile_start = q_tile_idx * q_tile
                q_tile_end = pypto.min(q_tile_start + q_tile, seq_len_q)
                q_tile_len = q_tile_end - q_tile_start

                for k_tile_idx in range(k_tile_count):
                    k_tile_start = k_tile_idx * k_tile
                    k_tile_end = pypto.min(k_tile_start + k_tile, seq_len_k)
                    k_tile_len = k_tile_end - k_tile_start

                    for h_s_idx in range(sta_num):
                        h_act_idx = h_idx * sta_num + h_s_idx
                        h_offset = h_act_idx * head_dim

                        q_tile_view = pypto.view(q_2d, [q_tile, head_dim],
                                            [q_start + q_tile_start, h_offset],
                                            valid_shape=[q_tile_len, head_dim])

                        k_tile_view = pypto.view(k_2d, [k_tile, head_dim],
                                            [k_start + k_tile_start, h_offset],
                                            valid_shape=[k_tile_len, head_dim])
                        v_tile_view = pypto.view(v_2d, [k_tile, head_dim],
                                            [k_start + k_tile_start, h_offset],
                                            valid_shape=[k_tile_len, head_dim])

                        # C1
                        pypto.set_cube_tile_shapes(
                            tile_config.c1_cube_tile[0],
                            tile_config.c1_cube_tile[1],
                            tile_config.c1_cube_tile[2])
                        scores = pypto.matmul(q_tile_view, k_tile_view, out_dtype=pypto.DT_FP32, b_trans=True)

                        # V1
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        pypto.set_pass_options(sg_set_scope=6)
                        scores_scaled = pypto.mul(scores, scale)
                        mij = pypto.amax(scores_scaled, dim=-1, keepdim=True)
                        s_shifted = pypto.sub(scores_scaled, mij)
                        pij = pypto.exp(s_shifted)
                        lij = pypto.sum(pij, dim=-1, keepdim=True)
                        pij_bf16 = pypto.cast(pij, pypto.DT_BF16)
                        pypto.set_pass_options(sg_set_scope=-1)

                        # C2
                        pypto.set_cube_tile_shapes(
                            tile_config.c2_cube_tile[0],
                            tile_config.c2_cube_tile[1],
                            tile_config.c2_cube_tile[2])
                        oij = pypto.matmul(pij_bf16, v_tile_view, out_dtype=pypto.DT_FP32)

                        # V2
                        if k_tile_idx == 0:
                            oi_update = oij
                            li_update = lij
                            mi_update = mij
                        else:
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            mi_new = pypto.maximum(mi_update, mij)
                            t1 = pypto.sub(mi_update, mi_new)
                            t2 = pypto.exp(t1)
                            t3 = pypto.sub(mij, mi_new)
                            t4 = pypto.exp(t3)

                            li_new = pypto.add(pypto.mul(t2, li_update), pypto.mul(t4, lij))
                            oi_tmp = pypto.add(pypto.mul(oi_update, t2), pypto.mul(oij, t4))

                            out_fp32 = pypto.div(oi_tmp, li_new, precision_type=pypto.PrecisionType.INTRINSIC)
                            out_bf16 = pypto.cast(out_fp32, pypto.DT_BF16)
                            pypto.assemble(li_new, [q_start + q_tile_start, h_act_idx], l_output)
                            pypto.assemble(mi_new, [q_start + q_tile_start, h_act_idx], m_output)
                            pypto.assemble(out_bf16, [q_start + q_tile_start, h_offset], output)
