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
Flash Attention MHA Backward

布局 (varlen, 通过 actual_q / actual_kv cumsum 描述每 batch 的 s1 / s2):
  - q/k/v/o/do: [total, num_heads, head_dim] BF16
                Q 侧 total = actual_q[-1], KV 侧 total = actual_kv[-1]
  - l/m:        [total_q, num_heads, 1] FP32
  - dq/dk/dv:   [total, hidden_dim] BF16
  - actual_q/actual_kv: [batch + 1] INT32, 前缀累加
                第 i 个 batch: offset = actual_*[i], seq = actual_*[i+1] - actual_*[i]

单趟计算:
  dQ/dK/dV 在同一双层循环内完成，dQ/dK/dV 用 atomic_add 累加输出。
  输出 tensor 由 host 端 torch.zeros 预初始化，kernel 不再做 assemble 清零。

关键: s1_loop / s2_loop 基于 per-batch 的 s1 / s2 动态计算;
is_loop_begin/end 用 Python if 即可。
"""

from dataclasses import dataclass
import pypto


@dataclass
class FlashAttentionGradTileShapeConfig:
    s1_tile: int
    s2_tile: int
    c_tile: list
    v_tile_s: list
    v_tile_d: list



@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 512,
        "device_sched_mode": 3,
        "ready_on_host_tensors": ["actual_q", "actual_kv"],
        "max_workspace_kb": 25170624,
    },
    pass_options={
        "cube_l1_reuse_setting": {-1: 1, 0: 4},
    }
)
def flash_attention_mha_grad_kernel_impl(
    q: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    k: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    o: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    do: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    l_input: pypto.Tensor([pypto.DYN, ...], pypto.DT_FP32),
    m_input: pypto.Tensor([pypto.DYN, ...], pypto.DT_FP32),
    dq: pypto.Tensor([pypto.DYN, ...], pypto.DT_FP32),
    dk: pypto.Tensor([pypto.DYN, ...], pypto.DT_FP32),
    dv: pypto.Tensor([pypto.DYN, ...], pypto.DT_FP32),
    actual_q: pypto.Tensor([pypto.DYN], pypto.DT_INT32),
    actual_kv: pypto.Tensor([pypto.DYN], pypto.DT_INT32),
    tile_config: FlashAttentionGradTileShapeConfig,
):
    """合一 kernel: 两趟 (dQ, dK/dV) 共享外层 batch+head 循环。

    actual_q / actual_kv 为前缀累加, shape=[batch_size + 1], 故 batch_size = shape[0] - 1
    第 i 个 batch: q_start=actual_q[i], s1=actual_q[i+1]-actual_q[i]
                   kv_start=actual_kv[i], s2=actual_kv[i+1]-actual_kv[i]
    s1 / s2 tile 循环次数基于 per-batch 的 s1 / s2 动态计算。
    """
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    hidden_dim = num_heads * head_dim
    total = q.shape[0]
    b = actual_q.shape[0] - 1
    scale = 1.0 / (head_dim ** 0.5)
    pypto.experimental.set_operation_options(combine_axis=True)

    # reshape 三维 → 二维, 便于按 [seq, hidden_dim] 切片
    q_2d = pypto.reshape(q, [total, hidden_dim], inplace=True)
    k_2d = pypto.reshape(k, [total, hidden_dim], inplace=True)
    v_2d = pypto.reshape(v, [total, hidden_dim], inplace=True)
    o_2d = pypto.reshape(o, [total, hidden_dim], inplace=True)
    do_2d = pypto.reshape(do, [total, hidden_dim], inplace=True)
    l_2d = pypto.reshape(l_input, [total, num_heads], inplace=True)
    m_2d = pypto.reshape(m_input, [total, num_heads], inplace=True)

    s1_tile = tile_config.s1_tile
    s2_tile = tile_config.s2_tile
    c_tile = tile_config.c_tile
    v_tile_s = tile_config.v_tile_s
    v_tile_d = tile_config.v_tile_d

    for b_idx in pypto.loop(b, name="LOOP_b_calc", idx_name="b_idx"):
        # per-batch 派生 Q/KV 偏移与 seqlen, 循环次数基于 per-batch 动态值
        q_start = actual_q[b_idx]
        s1 = actual_q[b_idx + 1] - q_start
        kv_start = actual_kv[b_idx]
        s2 = actual_kv[b_idx + 1] - kv_start

        s1_loop = (s1 + s1_tile - 1) // s1_tile
        s2_loop = (s2 + s2_tile - 1) // s2_tile

        for n_idx in pypto.loop(num_heads, name="LOOP_n_calc", idx_name="n_idx"):
            h_ofs = n_idx * head_dim
            for s1_idx in pypto.loop(s1_loop, name="LOOP_s1_dq", idx_name="s1_idx"):
                for s2_idx in pypto.loop(s2_loop, name="LOOP_s2_dq", idx_name="s2_idx"):
                    s1_off = q_start + s1_idx * s1_tile
                    actual_s1 = (s1 - s1_idx * s1_tile).min(s1_tile)
                    s2_off = kv_start + s2_idx * s2_tile
                    actual_s2 = (s2 - s2_idx * s2_tile).min(s2_tile)

                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    q_i = pypto.view(q_2d, [s1_tile, head_dim], [s1_off, h_ofs], valid_shape=[actual_s1, head_dim])
                    k_j = pypto.view(k_2d, [s2_tile, head_dim], [s2_off, h_ofs], valid_shape=[actual_s2, head_dim])
                    v_j = pypto.view(v_2d, [s2_tile, head_dim], [s2_off, h_ofs], valid_shape=[actual_s2, head_dim])
                    do_i = pypto.view(do_2d, [s1_tile, head_dim], [s1_off, h_ofs], valid_shape=[actual_s1, head_dim])
                    o_i = pypto.view(o_2d, [s1_tile, head_dim], [s1_off, h_ofs], valid_shape=[actual_s1, head_dim])
                    m_i = pypto.view(m_2d, [s1_tile, 1], [s1_off, n_idx], valid_shape=[actual_s1, 1])
                    l_i = pypto.view(l_2d, [s1_tile, 1], [s1_off, n_idx], valid_shape=[actual_s1, 1])

                    pypto.set_cube_tile_shapes(c_tile[0], c_tile[1], c_tile[2])
                    s_ij = pypto.matmul(q_i, k_j, pypto.DT_FP32, b_trans=True)
                    dp_ij = pypto.matmul(do_i, v_j, pypto.DT_FP32, b_trans=True)

                    pypto.set_pass_options(sg_set_scope=1)
                    pypto.set_vec_tile_shapes(v_tile_s[0], v_tile_s[1])
                    do_i_fp32 = pypto.cast(do_i, pypto.DT_FP32)
                    o_i_fp32 = pypto.cast(o_i, pypto.DT_FP32)
                    do_mul_oi = pypto.mul(o_i_fp32, do_i_fp32)
                    d_i = pypto.sum(do_mul_oi, -1, keepdim=True)

                    s_ij = pypto.mul(s_ij, scale)
                    p_ij = pypto.exp(pypto.sub(s_ij, m_i))
                    p_ij = pypto.div(p_ij, l_i, precision_type=pypto.PrecisionType.INTRINSIC)
                    ds_ij = pypto.mul(p_ij, pypto.sub(dp_ij, d_i))

                    ds_bf16 = pypto.cast(ds_ij, pypto.DT_BF16)
                    p_bf16 = pypto.cast(p_ij, pypto.DT_BF16)
                    pypto.set_pass_options(sg_set_scope=-1)

                    pypto.set_cube_tile_shapes(c_tile[0], c_tile[1], c_tile[2])
                    dv_tile = pypto.matmul(p_bf16, do_i, pypto.DT_FP32, a_trans=True)
                    dq_tile = pypto.matmul(ds_bf16, k_j, pypto.DT_FP32)
                    dk_tile = pypto.matmul(ds_bf16, q_i, pypto.DT_FP32, a_trans=True)

                    pypto.set_pass_options(sg_set_scope=2)
                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    pypto.atomic_add(dv_tile, [s2_off, h_ofs], dv)
                    dq_final = pypto.mul(dq_tile, scale)
                    pypto.atomic_add(dq_final, [s1_off, h_ofs], dq)
                    dk_final = pypto.mul(dk_tile, scale)
                    pypto.atomic_add(dk_final, [s2_off, h_ofs], dk)
                    pypto.set_pass_options(sg_set_scope=-1)