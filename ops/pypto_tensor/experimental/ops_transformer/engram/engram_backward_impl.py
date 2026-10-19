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
Engram Backward PyPTO Kernel Implementation (Multi-Head, MHC)

多头 (m heads) Engram 反向算子: bs 合轴 + 逐头 RMSNorm 融合反传 + matmul BF16 分派。
BF16 输入/输出, FP32 中间累加 (atomic_add 进 FP32 acc, 末尾降 BF16)。

反向两条主路径:
  1. Value 路径: grad_value_bf16 = Σ_m(grad_out·gate) → value_linear_backward → grad_embeddings_out
  2. Key/Query 路径(per head m):
     d_gate → d_score (sigmoid' + sign-sqrt' 链式) → grad_query_normed/grad_key_normed (点积反传)
     → rms_norm_backward → grad_key → key_linear_backward → grad_embeddings_out (累加)
  gate_cache_in/score_cache_in 输入为 FP32; key_cache_in/value_cache_in 输入 BF16, 内部 cast FP32 计算。

输入: grad_out_in, hidden_states_in, embeddings_in, weight_key_in, weight_value_in, gamma_key_in, gamma_query_in,
      key_cache_in(BF16), value_cache_in(BF16), gate_cache_in(FP32), score_cache_in(FP32)
输出: grad_hidden_out, grad_embeddings_out, grad_weight_key_out,
      grad_weight_value_out, grad_gamma_key_out, grad_gamma_query_out (均 BF16)
"""

import math

import torch
import pypto


@pypto.frontend.jit(
    pass_options={
        "cube_l1_reuse_setting": {"DEFAULT": 32},
        "vec_nbuffer_setting": {"DEFAULT": 32, "func9_1": 2, "func9_2": 8, "func13_2": 4, "func23_0": 8},
    },
    runtime_options={
        "stitch_function_max_num": 32,
    },
)
def engram_backward_kernel(
    # inputs
    grad_out_in: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),  # [bs, m, h]
    hidden_states_in: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),  # [bs, m, h]
    embeddings_in: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),  # [bs, de]
    weight_key_in: pypto.Tensor([], pypto.DT_BF16),  # [m, de, h]
    weight_value_in: pypto.Tensor([], pypto.DT_BF16),  # [de, h]
    gamma_key_in: pypto.Tensor([], pypto.DT_BF16),  # [m, h]
    gamma_query_in: pypto.Tensor([], pypto.DT_BF16),  # [m, h]
    key_cache_in: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),  # [bs, m, h]
    value_cache_in: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),  # [bs, h]
    gate_cache_in: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),  # [bs, m]
    score_cache_in: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),  # [bs, m]
    # workspace
    grad_embeddings_acc: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),  # [bs, de]
    grad_value_acc: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),  # [bs, h]
    # outputs
    grad_hidden_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),  # [bs, m, h]
    grad_embeddings_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),  # [bs, de]
    grad_weight_key_out: pypto.Tensor([], pypto.DT_BF16),  # [m, de, h]
    grad_weight_value_out: pypto.Tensor([], pypto.DT_BF16),  # [de, h]
    grad_gamma_key_out: pypto.Tensor([], pypto.DT_BF16),  # [m, h]
    grad_gamma_query_out: pypto.Tensor([], pypto.DT_BF16),  # [m, h]
    # attrs
    clamp_value,
    eps,
):
    pypto.experimental.set_operation_options(combine_axis=True)
    bs, m, h = hidden_states_in.shape
    de = embeddings_in.shape[-1]
    sqrt_h = math.sqrt(h)
    inv_sqrt_h = 1.0 / sqrt_h
    unroll_list = [128]
    first_axis = 8
    if h > 1536:
        first_axis = 4
        pypto.set_pass_options(vec_nbuffer_setting={
            "DEFAULT": 32, "func9_1": 4, "func9_2": 8, "func13_2": 4, "func23_0": 8})

    # ── Value weight accumulators (FP32: 跨 head/tile 累加保精度) ──
    pypto.set_vec_tile_shapes(128, 128)  # vec tile (de, h)
    grad_weight_v_acc = pypto.full([de, h], 0.0, pypto.DT_FP32)  # [de, h] FP32

    for m_idx in pypto.loop(m, name="m_loop"):
        # ── Per-head accumulators (FP32) ──
        pypto.set_vec_tile_shapes(128, 128)
        grad_weight_k_acc = pypto.full([de, h], 0.0, pypto.DT_FP32)
        pypto.set_vec_tile_shapes(1, h)
        grad_gamma_k_acc = pypto.full([1, h], 0.0, pypto.DT_FP32)
        grad_gamma_q_acc = pypto.full([1, h], 0.0, pypto.DT_FP32)
        # gamma FP32
        gamma_key = pypto.cast(gamma_key_in[m_idx:m_idx + 1], pypto.DT_FP32)
        gamma_query = pypto.cast(gamma_query_in[m_idx:m_idx + 1], pypto.DT_FP32)

        for bs_idx, tile in pypto.loop_unroll(0, bs, 1, name="bl_loop", unroll_list=unroll_list):
            pypto.set_vec_tile_shapes(first_axis, h)
            grad_out = grad_out_in[bs_idx:bs_idx + tile, m_idx]
            key = key_cache_in[bs_idx:bs_idx + tile, m_idx]
            query = hidden_states_in[bs_idx:bs_idx + tile, m_idx]

            pypto.set_semantic_label("grad_value")
            pypto.set_pass_options(sg_set_scope=1)
            pypto.set_vec_tile_shapes(first_axis, h)
            gate = gate_cache_in[bs_idx:bs_idx + tile, m_idx]
            pypto.set_vec_tile_shapes(first_axis)
            gate_2d = gate.unsqueeze(-1)
            pypto.set_vec_tile_shapes(first_axis, h)
            grad_value = pypto.cast(grad_out, pypto.DT_FP32) * gate_2d
            pypto.atomic_add(grad_value, [bs_idx, 0], grad_value_acc)

            pypto.set_semantic_label("grad_score")
            pypto.set_vec_tile_shapes(first_axis, h)
            score = score_cache_in[bs_idx:bs_idx + tile, m_idx]
            value = value_cache_in[bs_idx:bs_idx + tile]
            grad_gate = (pypto.cast(grad_out, pypto.DT_FP32) * pypto.cast(value, pypto.DT_FP32)).sum(-1)
            pypto.set_vec_tile_shapes(first_axis)
            sigmoid_grad = gate * (1.0 - gate)
            abs_score = pypto.abs(score)
            sqrt_abs = pypto.sqrt(pypto.maximum(abs_score, clamp_value))
            mask = pypto.where(pypto.gt(abs_score, clamp_value),
                               grad_gate * 0.0 + 1.0, grad_gate * 0.0)
            logits_grad = pypto.div(mask, 2.0 * sqrt_abs + 1e-12, pypto.PrecisionType.INTRINSIC)
            grad_score = grad_gate * sigmoid_grad * logits_grad
            grad_score_scaled = grad_score * inv_sqrt_h

            pypto.set_semantic_label("key_rmsnorm")
            pypto.set_vec_tile_shapes(first_axis, h)  # reduce -1轴不切
            key_fp32 = pypto.cast(key, pypto.DT_FP32)
            sum_sq_k = pypto.sum(pypto.div((key_fp32 * key_fp32), h, pypto.PrecisionType.INTRINSIC), -1, keepdim=True)
            rms_k = pypto.sqrt(sum_sq_k + eps)
            key_hat = pypto.div(key_fp32, rms_k, pypto.PrecisionType.INTRINSIC)
            key_normed = key_hat * gamma_key

            pypto.set_semantic_label("query_rmsnorm")
            query_fp32 = pypto.cast(query, pypto.DT_FP32)
            sum_sq_q = pypto.sum(pypto.div((query_fp32 * query_fp32),
                                        h, pypto.PrecisionType.INTRINSIC), -1, keepdim=True)
            rms_q = pypto.sqrt(sum_sq_q + eps)
            query_hat = pypto.div(query_fp32, rms_q, pypto.PrecisionType.INTRINSIC)
            query_normed = query_hat * gamma_query

            pypto.set_semantic_label("grad_key/query_normed")
            pypto.set_vec_tile_shapes(first_axis)
            grad_score_scaled_2d = grad_score_scaled.unsqueeze(-1)
            pypto.set_vec_tile_shapes(first_axis, h)
            grad_query_normed = grad_score_scaled_2d * key_normed
            grad_key_normed = grad_score_scaled_2d * query_normed

            pypto.set_semantic_label("key_rmsnorm_bwd")
            grad_n_k = grad_key_normed * gamma_key
            gn_xh_k = pypto.sum(pypto.div((grad_n_k * key_hat), h, pypto.PrecisionType.INTRINSIC), -1, keepdim=True)
            grad_key = pypto.div((grad_n_k - key_hat * gn_xh_k), rms_k, pypto.PrecisionType.INTRINSIC)
            grad_key_bf16 = pypto.cast(grad_key, pypto.DT_BF16)
            grad_gamma_key = grad_key_normed * key_hat

            pypto.set_semantic_label("query_rmsnorm_bwd")
            grad_n_q = grad_query_normed * gamma_query
            gn_xh_q = pypto.sum(pypto.div((grad_n_q * query_hat), h, pypto.PrecisionType.INTRINSIC), -1, keepdim=True)
            grad_query = pypto.div((grad_n_q - query_hat * gn_xh_q), rms_q, pypto.PrecisionType.INTRINSIC)
            grad_query_bf16 = pypto.cast(grad_query, pypto.DT_BF16)
            grad_gamma_query = grad_query_normed * query_hat
            pypto.set_pass_options(sg_set_scope=-1)

            grad_hidden_out[bs_idx:bs_idx + tile, m_idx] = grad_query_bf16
            pypto.set_vec_tile_shapes(128, 128)
            grad_gamma_key_sum = pypto.sum(grad_gamma_key, 0, keepdim=True)
            pypto.atomic_add(grad_gamma_key_sum, [0, 0], grad_gamma_k_acc)
            grad_gamma_query_sum = pypto.sum(grad_gamma_query, 0, keepdim=True)
            pypto.atomic_add(grad_gamma_query_sum, [0, 0], grad_gamma_q_acc)

            pypto.set_semantic_label("key_linear_bwd")
            pypto.set_cube_tile_shapes([128, 128], [128, 256], [128, 128])  
            embeddings = embeddings_in[bs_idx:bs_idx + tile]
            grad_embeddings_key = pypto.matmul(grad_key_bf16, weight_key_in[m_idx], pypto.DT_FP32, b_trans=True)
            grad_weight_key = pypto.matmul(embeddings, grad_key_bf16, pypto.DT_FP32, a_trans=True)
            pypto.atomic_add(grad_weight_key, [0, 0], grad_weight_k_acc)
            pypto.atomic_add(grad_embeddings_key, [bs_idx, 0], grad_embeddings_acc)

        # ── Write per-head outputs ──
        pypto.set_vec_tile_shapes(1, 128, 128)
        grad_weight_key_out[m_idx:m_idx + 1] = pypto.cast(pypto.reshape(grad_weight_k_acc, [1, de, h]), pypto.DT_BF16)
        pypto.set_vec_tile_shapes(1, h)
        grad_gamma_key_out[m_idx:m_idx + 1] = pypto.cast(grad_gamma_k_acc, pypto.DT_BF16)
        grad_gamma_query_out[m_idx:m_idx + 1] = pypto.cast(grad_gamma_q_acc, pypto.DT_BF16)

    for bs_idx, tile in pypto.loop_unroll(0, bs, 1, name="value_matmul_loop", unroll_list=unroll_list):
        pypto.set_semantic_label("value_linear_bwd")
        pypto.set_vec_tile_shapes(first_axis, h)
        embeddings = embeddings_in[bs_idx:bs_idx + tile]
        grad_value_bf16 = pypto.cast(grad_value_acc[bs_idx:bs_idx + tile], pypto.DT_BF16)
        pypto.set_cube_tile_shapes([128, 128], [128, 256], [128, 128])
        grad_embeddings_value = pypto.matmul(grad_value_bf16, weight_value_in, pypto.DT_FP32, b_trans=True)
        grad_weight_vlaue = pypto.matmul(embeddings, grad_value_bf16, pypto.DT_FP32, a_trans=True)
        pypto.set_semantic_label("value_store")
        pypto.atomic_add(grad_embeddings_value, [bs_idx, 0], grad_embeddings_acc)  # value 分支对应 emb
        pypto.atomic_add(grad_weight_vlaue, [0, 0], grad_weight_v_acc)

    for bs_idx, tile in pypto.loop_unroll(0, bs, 1, name="value_matmul_loop", unroll_list=unroll_list):
        grad_embeddings_out[bs_idx:bs_idx + tile] = pypto.cast(grad_embeddings_acc[bs_idx:bs_idx + tile], pypto.DT_BF16)

    # ── Write value weight outputs ──
    pypto.set_vec_tile_shapes(128, 128)
    grad_weight_value_out[:, :] = pypto.cast(grad_weight_v_acc, pypto.DT_BF16)


def engram_backward_wrapper(
    grad_out_in,        # [b, s, m, h] BF16
    hidden_states_in,   # [b, s, m, h] BF16
    embeddings_in,      # [b, s, de]   BF16
    weight_key_in,      # [m, de, h]   BF16
    weight_value_in,    # [de, h]      BF16
    gamma_key_in,       # [m, h]       BF16
    gamma_query_in,     # [m, h]       BF16
    score_cache_in,     # [b, s, m]    FP32
    gate_cache_in,      # [b, s, m]    FP32
    key_cache_in,       # [b, s, m, h] BF16
    value_cache_in,     # [b, s, h]    BF16
    clamp_value=1e-6,
    eps=1e-6,
):
    """
    Inputs:
      grad_out_in,        [b, s, m, h]  BF16
      hidden_states_in,   [b, s, m, h]  BF16
      embeddings_in,      [b, s, de]    BF16
      weight_key_in,      [m, de, h]    BF16
      weight_value_in,    [de, h]       BF16
      gamma_key_in,       [m, h]        BF16
      gamma_query_in,     [m, h]        BF16
      score_cache_in,     [b, s, m, 1]  FP32
      gate_cache_in,      [b, s, m, 1]  FP32
      key_cache_in,       [b, s, m, h]  BF16
      value_cache_in,     [b, s, h]     BF16

    Returns:
      grad_hidden_out:         [b, s, m, h]  BF16
      grad_embeddings_out:     [b, s, de]    BF16
      grad_weight_key_out:     [m, de, h]    BF16
      grad_weight_value_out:   [de, h]       BF16
      grad_gamma_key_out:      [m, h]        BF16
      grad_gamma_query_out:    [m, h]        BF16
    """
    # ── 输入校验: 拦截空 tensor 和非连续 tensor ──
    for name, t in (("grad_out_in", grad_out_in),
                    ("hidden_states_in", hidden_states_in),
                    ("embeddings_in", embeddings_in),
                    ("weight_key_in", weight_key_in),
                    ("weight_value_in", weight_value_in),
                    ("gamma_key_in", gamma_key_in),
                    ("gamma_query_in", gamma_query_in),
                    ("score_cache_in", score_cache_in),
                    ("gate_cache_in", gate_cache_in),
                    ("key_cache_in", key_cache_in),
                    ("value_cache_in", value_cache_in)):
        if t.numel() == 0:
            raise ValueError(f"engram_backward: input '{name}' must not be empty, got shape {tuple(t.shape)}")
        if not t.is_contiguous():
            raise ValueError(f"engram_backward: input '{name}' must be contiguous, got shape {tuple(t.shape)}")

    b, s, m, h = hidden_states_in.shape
    de = embeddings_in.shape[-1]
    device = grad_out_in.device
    grad_hidden_out = torch.zeros([b * s, m, h], dtype=torch.bfloat16, device=device)
    grad_embeddings_out = torch.zeros([b * s, de], dtype=torch.bfloat16, device=device)
    grad_embeddings_acc = torch.zeros([b * s, de], dtype=torch.float32, device=device)
    grad_value_acc = torch.zeros([b * s, h], dtype=torch.float32, device=device)
    grad_weight_key_out = torch.zeros_like(weight_key_in)
    grad_weight_value_out = torch.zeros_like(weight_value_in)
    grad_gamma_key_out = torch.zeros_like(gamma_key_in)
    grad_gamma_query_out = torch.zeros_like(gamma_query_in)

    engram_backward_kernel(
        grad_out_in.reshape(b * s, m, h),
        hidden_states_in.reshape(b * s, m, h),
        embeddings_in.reshape(b * s, de),
        weight_key_in, weight_value_in, gamma_key_in, gamma_query_in,
        key_cache_in.reshape(b * s, m, h),
        value_cache_in.reshape(b * s, h),
        gate_cache_in.reshape(b * s, m),
        score_cache_in.reshape(b * s, m),
        grad_embeddings_acc,
        grad_value_acc,
        grad_hidden_out.reshape(b * s, m, h),
        grad_embeddings_out.reshape(b * s, de),
        grad_weight_key_out,
        grad_weight_value_out,
        grad_gamma_key_out,
        grad_gamma_query_out,
        clamp_value,
        eps,
    )

    return (grad_hidden_out.reshape(b, s, m, h),
            grad_embeddings_out.reshape(b, s, de),
            grad_weight_key_out,
            grad_weight_value_out,
            grad_gamma_key_out,
            grad_gamma_query_out)
