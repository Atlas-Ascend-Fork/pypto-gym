# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------



"""PyPTO chunked_gated_delta_rule kernel implementation.

This module implements the Chunked Gated Delta Rule Linear Attention mechanism,
reducing traditional O(n²) softmax attention to O(n) complexity.

It provides two kernel versions:
  - chunk_gated_delta_rule_aligned:  for sequences where T % L == 0
  - chunk_gated_delta_rule_unaligned: for sequences where T % L != 0

Key design decisions (per qwen3_next production reference):
    - S loop uses pypto.loop with unroll_list=[16,1] (per qwen3_next reference)
    - Nv loop with parallel=True (per qwen3_next production reference)

Performance optimizations (all verified on NPU 910B3):
   ✅ zeros extracted as kernel inputs (avoid repeated pypto.full)
   ✅ l2norm_scaled merged function (eliminate separate scale step)
   ✅ cube_tile_shapes switches reduced from 4→2 (per-iteration savings)
   ✅ stitch_function_max_num=32 (aligned) / 8 (unaligned)
   ✅ chunk_size L configurable (32/64/128, auto-select via wrapper)
   ✅ Nv≥4 required for parallel=True to utilize ≥4 cores
   ✅ No extra set_vec_tile_shapes before write-back (per qwen3_next)
   ✅ combine_axis=True
   ✅ prepare_chunk_helpers(): auto-generate mask/tril/eye/zeros
   ✅ auto chunk_size selection in wrapper (chunk_size="auto")
   ✅ B1: g_exp passed from Phase 4 to Phase 5 (eliminate duplicate exp(gate_cum))
   ✅ B2: final_state_1 inlined (eliminate 64KB temp tensor)
   ✅ B3: state UB prefetch via state_ub = state + 0.0 (hide GM→UB latency)

Negative optimization results (tested, no benefit — do NOT retry):
   ❌ cumsum for gate_cum: replaces matmul(tril,gate) with vector cumsum,
      but no perf improvement because bottleneck is S loop serial dependency;
      also fails to compile in unaligned is_loop_end branching context.
   ❌ inplace=True on reshape: causes precision failure (max_diff jumps
      from ~1e-05 to ~1e-01); do NOT use inplace=True on reshape operations.
   ❌ exclusive cumsum (cumsum - self): semantic error, max_diff=6.165e+13.
   ❌ A_inv dual matmul fusion (A_inv @ concat[vβ, kβ*g_exp]): concat/slice
      overhead + [L,2D] shape doesn't fit cube pipeline as well as two [L,D].
   ❌ stitch=128: L=64 memory failure + L=128 performance worse than 32.
   ❌ 0switch tile mode for large L: +20% task_time degradation.
   ❌ B loop parallel=True: PyPTO hard limit — no nested parallel loops.
   ❌ L=64/L=32 for aligned multi-chunk (T>L): S loop serial dependency
      amplifies; L=64 → +28% slower (T=512), L=32 → +98% slower (T=512).
      Exception: unaligned T≈130 with tiny partial, L=64 may be ~8% faster.
   ❌ valid_shape_optimize=1: causes compile timeout.
   ❌ device_sched_mode=1: +5~8% slower on medium/large cases; slight gain
      only on tiny L64/L32 single-chunk cases (not worth it).
   ❌ device_sched_mode=3: likely negative since mode=1 is already negative.
   ❌ submit_before_loop=True on S loop: catastrophic +183~357% regression;
      AICore Util collapses to 5~7% (pipeline completely broken).
   ❌ cube_nbuffer_setting={"DEFAULT":16}: compile failure for L=64 + 
      +5~9% slower on completed cases.
   ❌ cube_l1_reuse_setting={"DEFAULT":2}: compile failure for unaligned +
      +2~5% slower on completed cases.

Reference: models/qwen3_next/gated_delta_rule_impl.py (production implementation)
"""

import pypto
import torch
import torch_npu

DYNAMIC_T = pypto.DYNAMIC
DYNAMIC_B = pypto.DYNAMIC
DYNAMIC_B1 = pypto.DYNAMIC


def prepare_chunk_helpers(chunk_size: int, dtype=torch.float32):
    """Generate mask, tril_mask, eye, and zero tensors for a given chunk_size.

    Convenience function that callers can use to create the helper tensors
    needed by chunked_gated_delta_rule_wrapper. Useful when auto-selecting
    chunk_size or when building inputs from scratch.

    Args:
        chunk_size: chunk size L (must be multiple of 8 for inverse_pto)
        dtype: tensor dtype (default float32)

    Returns:
        dict with keys: mask, tril_mask, eye
    """
    if chunk_size % 8 != 0:
        raise ValueError(f"chunk_size must be multiple of 8, got {chunk_size}")
    L = chunk_size

    mask = torch.tril(-torch.ones(L, L, dtype=dtype), diagonal=-1)
    tril_mask = torch.ones(L, L, dtype=dtype).tril()
    eye = torch.eye(L, dtype=dtype)

    return {
        "mask": mask, "tril_mask": tril_mask, "eye": eye,
    }


def l2norm_scaled(
    query: pypto.Tensor, key: pypto.Tensor, eps: float = 1e-6, d: int = 128
) -> tuple:
    """L2 normalization + scaling for query, L2 normalization for key.

    Merged function: computes query_norm * (1/sqrt(d)) and key_norm in one call,
    eliminating the separate scale multiplication step.

    Args:
        query: [L, D] pypto.Tensor
        key: [L, D] pypto.Tensor
        eps: epsilon for numerical stability (default 1e-6)
        d: head dimension for scale computation (default 128)

    Returns:
        query_scale: [L, D] (l2norm + scaled by 1/sqrt(d))
        key_norm: [L, D] (l2norm only)
    """
    pypto.set_vec_tile_shapes(128, 128)
    scale = 1 / d ** 0.5
    query_norm = query / pypto.sqrt((query * query).sum(-1, keepdim=True) + eps)
    key_norm = key / pypto.sqrt((key * key).sum(-1, keepdim=True) + eps)
    query_scale = query_norm * scale
    return query_scale, key_norm


def pre_attn(
    gate_view: pypto.Tensor,
    key_view_2d: pypto.Tensor,
    beta_view: pypto.Tensor,
    tril: pypto.Tensor,
    mask: pypto.Tensor,
) -> tuple:
    """Pre-attention computation: gate cumsum, decay mask, key_beta, and KKT.

    Args:
        gate_view: [L, 1]
        key_view_2d: [L, D] (after l2norm)
        beta_view: [L, 1]
        tril: [L, L] (lower triangular mask, diagonal=0 inclusive)
        mask: [L, L] (pre-attention mask, diagonal=-1 exclusive)

    Returns:
        gate_cum: [L, 1], decay_mask: [L, L], a: [L, L], key_beta: [L, D]
    """
    pypto.set_vec_tile_shapes(128, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

    gate_cum = pypto.matmul(tril, gate_view, pypto.DT_FP32)
    decay_mask = ((gate_cum - gate_cum.transpose(0, 1)) * tril).exp()
    key_beta = key_view_2d * beta_view
    kkt = pypto.matmul(key_beta, key_view_2d, pypto.DT_FP32, b_trans=True)
    a = kkt * decay_mask * mask

    return gate_cum, decay_mask, a, key_beta


def inverse_pto(**kwargs) -> pypto.Tensor:
    """Power series matrix inversion.

    Computes (I - A)^{-1} = I + A + A^2 + ... + A^8
    for a strictly lower triangular matrix A.

    NOTE: Each pypto.matmul result must be followed by + 0.0 to force a
    data copy before being used as input to another pypto.matmul. Without
    this, the PyPTO compiler produces incorrect results due to buffer
    reuse issues in chained matmul operations.

    Args (via kwargs):
        attn: [L, L] input matrix (A = kkt * decay_mask * mask)
        eye: [L, L] identity matrix
    """
    attn = kwargs.get("attn")
    eye = kwargs.get("eye")

    pypto.set_vec_tile_shapes(128, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

    # + 0.0 is REQUIRED after each pypto.matmul to force data copy,
    # otherwise chained matmul produces incorrect results (PyPTO compiler bug)
    attn_pow2 = pypto.matmul(attn, attn, pypto.DT_FP32) + 0.0
    attn_pow3 = pypto.matmul(attn_pow2, attn, pypto.DT_FP32) + 0.0
    attn_pow4 = pypto.matmul(attn_pow3, attn, pypto.DT_FP32) + 0.0
    attn_pow5 = pypto.matmul(attn_pow4, attn, pypto.DT_FP32) + 0.0
    attn_pow6 = pypto.matmul(attn_pow5, attn, pypto.DT_FP32) + 0.0
    attn_pow7 = pypto.matmul(attn_pow6, attn, pypto.DT_FP32) + 0.0
    attn_pow8 = pypto.matmul(attn_pow7, attn, pypto.DT_FP32) + 0.0

    acc = pypto.add(eye, attn)
    acc = pypto.add(acc, attn_pow2)
    acc = pypto.add(acc, attn_pow3)
    acc = pypto.add(acc, attn_pow4)
    acc = pypto.add(acc, attn_pow5)
    acc = pypto.add(acc, attn_pow6)
    acc = pypto.add(acc, attn_pow7)
    acc = pypto.add(acc, attn_pow8)

    return acc


def cal_value_and_key_cumdecay(
    attn: pypto.Tensor,
    value_view: pypto.Tensor,
    beta_view: pypto.Tensor,
    key_beta: pypto.Tensor,
    gate_cum: pypto.Tensor,
) -> tuple:
    """Calculate value and key cumulative decay.

    Returns g_exp as third value to avoid duplicate exp(gate_cum) computation
    in recurrent_state_attn_all (B1 optimization).

    NOTE: Fused matmul (A_inv @ concat[v_beta, k_beta*exp(g)]) was tested but
    showed no benefit — concat/slice overhead + larger [L,2D] shape doesn't
    fit cube pipeline as well as two [L,D] matmuls. Keep separate approach.
    """
    pypto.set_vec_tile_shapes(128, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

    value_beta_view = value_view * beta_view
    value_out = pypto.matmul(attn, value_beta_view, pypto.DT_FP32)

    g_exp = pypto.exp(gate_cum)
    weighted_k_beta_view = key_beta * g_exp
    key_cum_out = pypto.matmul(attn, weighted_k_beta_view, pypto.DT_FP32)

    return value_out, key_cum_out, g_exp


_RECURRENT_ATTN_TILE_SWITCH_MODE = "2switch"


def recurrent_state_attn_all(**kwargs) -> tuple:
    """Recurrent state attention computation with configurable cube_tile_shapes switches.

    Supports two modes via _RECURRENT_ATTN_TILE_SWITCH_MODE global:
      - "2switch": current optimized approach (2 tile switches)
      - "0switch": experimental - all matmuls use [128,128],[128,128],[128,128]

    Optimization: accepts pre-computed g_exp from cal_value_and_key_cumdecay
    to eliminate duplicate exp(gate_cum) computation.
    """
    query = kwargs.get("query")
    key = kwargs.get("key")
    value = kwargs.get("value")
    k_cumdecay = kwargs.get("k_cumdecay")
    gate = kwargs.get("gate")
    state = kwargs.get("state")
    decay_mask = kwargs.get("decay_mask")
    tril = kwargs.get("tril")
    g_exp = kwargs.get("g_exp")  # B1: pre-computed exp(gate_cum) from Phase 4

    dv = value.shape[-1]
    l = gate.valid_shape[0]
    gate_exp = g_exp  # B1: use pre-computed g_exp instead of gate.exp()

    if _RECURRENT_ATTN_TILE_SWITCH_MODE == "0switch":
        state_ub = state + 0.0
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        pypto.set_vec_tile_shapes(64, 128)
        _last_gate_1 = gate[l - 1:l, :]
        kgexp = key * (_last_gate_1 - gate).exp()
        qgexp = query * gate_exp

        v_prime = pypto.matmul(k_cumdecay, state_ub, pypto.DT_FP32, b_trans=True)
        attn_inter = pypto.matmul(qgexp, state_ub, pypto.DT_FP32, b_trans=True)

        v_new = value - v_prime

        state_update = pypto.matmul(v_new, kgexp, pypto.DT_FP32, a_trans=True)
        attn = pypto.matmul(query, key, pypto.DT_FP32, b_trans=True)

        _last_gate_2 = pypto.expand_clone(gate_exp[l - 1:l, :], (dv, 1))
        state_new = state * _last_gate_2 + state_update

        attn_tmp = attn * decay_mask * tril
        chunk_attn_vnew = pypto.matmul(attn_tmp, v_new, pypto.DT_FP32)

        chunk_attn_out = attn_inter + chunk_attn_vnew
    else:
        state_ub = state + 0.0
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        pypto.set_vec_tile_shapes(64, 128)
        _last_gate_1 = gate[l - 1:l, :]
        kgexp = key * (_last_gate_1 - gate).exp()
        qgexp = query * gate_exp

        pypto.set_cube_tile_shapes([128, 128], [128, 128], [64, 64])
        v_prime = pypto.matmul(k_cumdecay, state_ub, pypto.DT_FP32, b_trans=True)
        attn_inter = pypto.matmul(qgexp, state_ub, pypto.DT_FP32, b_trans=True)

        v_new = value - v_prime

        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        state_update = pypto.matmul(v_new, kgexp, pypto.DT_FP32, a_trans=True)
        attn = pypto.matmul(query, key, pypto.DT_FP32, b_trans=True)

        _last_gate_2 = pypto.expand_clone(gate_exp[l - 1:l, :], (dv, 1))
        state_new = state * _last_gate_2 + state_update

        attn_tmp = attn * decay_mask * tril
        chunk_attn_vnew = pypto.matmul(attn_tmp, v_new, pypto.DT_FP32)

        chunk_attn_out = attn_inter + chunk_attn_vnew

    return chunk_attn_out, state_new


# ─────────────────────────────────────────────
# 2. Aligned version kernel factory
# ─────────────────────────────────────────────

def chunk_gated_delta_rule_aligned(b, nqk, nv, d, l, enable_perf_debug=False,
                                     stitch_function_max_num=32):
    """Factory function for aligned version kernel (T % L == 0).

    stitch_function_max_num configurable (default 32 for better Nv loop parallelism).
    zeros passed as kernel inputs (extracted from S loop).
    chunk_size l configurable (not hardcoded 128).

    Args:
        b: batch size
        nqk: number of query/key heads
        nv: number of value heads
        d: head dimension
        l: chunk size (configurable)
        enable_perf_debug: if True, enables runtime_debug_mode
        stitch_function_max_num: stitch parallelism (default 32)
    """
    jit_kwargs = {
        "runtime_options": {
            "stitch_function_max_num": stitch_function_max_num,
            "device_sched_parallelism": 8,
        },
        "pass_options": {
            "vec_nbuffer_setting": {8: 16, 1: 16, 28: 32, 6: 4, 25: 32, 26: 8,
                2: 8, 3: 8, 10: 32, 5: 4, 7: 16, 9: 32, 0: 4, 4: 4, -2: 1}
        },
    }
    if enable_perf_debug:
        jit_kwargs["debug_options"] = {"runtime_debug_mode": 1}

    @pypto.frontend.jit(**jit_kwargs)
    def kernel(
            query: pypto.Tensor([DYNAMIC_T, nqk, d], pypto.DT_FP32),
            key: pypto.Tensor([DYNAMIC_T, nqk, d], pypto.DT_FP32),
            value: pypto.Tensor([DYNAMIC_T, nv, d], pypto.DT_FP32),
            beta: pypto.Tensor([DYNAMIC_T, nv], pypto.DT_FP32),
            gate: pypto.Tensor([DYNAMIC_T, nv], pypto.DT_FP32),
            states: pypto.Tensor([DYNAMIC_B, nv, d, d], pypto.DT_FP32),
            mask: pypto.Tensor([l, l], pypto.DT_FP32),
            tril_mask: pypto.Tensor([l, l], pypto.DT_FP32),
            eye: pypto.Tensor([l, l], pypto.DT_FP32),
            act_seq_len: pypto.Tensor([DYNAMIC_B1], pypto.DT_INT32),
            core_attn_out: pypto.Tensor([DYNAMIC_T, nv, d], pypto.DT_FP32),
            last_state_data: pypto.Tensor([DYNAMIC_B, nv, d, d], pypto.DT_FP32),
    ):
        _, nqk, d = query.shape
        _, nv, d = value.shape
        b = states.shape[0]
        l, l = mask.shape
        group = nv // nqk
        last_state = pypto.tensor([d, d], pypto.DT_FP32)
        pypto.experimental.set_operation_options(combine_axis=True)

        for b_idx in pypto.loop(b, name="LOOP_B_TND", idx_name="b_idx"):
            s = act_seq_len[b_idx + 1] - act_seq_len[b_idx]
            b_ofs = act_seq_len[b_idx]

            for nv_idx in pypto.loop(nv, name="LOOP_Nv_TND", idx_name="nv_idx", parallel=False):
                nqk_idx = nv_idx // group
                pypto.set_vec_tile_shapes(16, 16, 128, 128)
                last_state = states[b_idx, nv_idx]

                for s_idx in pypto.loop(0, s, l, name="LOOP_S_TND",
                                        idx_name="s_idx", unroll_list=[16, 1]):
                    bs_ofs = b_ofs + s_idx
                    actual_l = (s - s_idx).min(l)

                    query_view = pypto.view(query, [l, 1, d], [bs_ofs, nqk_idx, 0],
                                            valid_shape=[actual_l, 1, d])
                    key_view = pypto.view(key, [l, 1, d], [bs_ofs, nqk_idx, 0],
                                          valid_shape=[actual_l, 1, d])
                    value_view = pypto.view(value, [l, 1, d], [bs_ofs, nv_idx, 0],
                                            valid_shape=[actual_l, 1, d])
                    beta_view = pypto.view(beta, [l, 1], [bs_ofs, nv_idx],
                                           valid_shape=[actual_l, 1])
                    gate_view = pypto.view(gate, [l, 1], [bs_ofs, nv_idx],
                                           valid_shape=[actual_l, 1])

                    pypto.set_vec_tile_shapes(128, 128, 128)
                    query_view_2d = pypto.reshape(query_view, [l, d],
                                                  valid_shape=[actual_l, d])
                    key_view_2d = pypto.reshape(key_view, [l, d],
                                                 valid_shape=[actual_l, d])
                    value_view_2d = pypto.reshape(value_view, [l, d],
                                                   valid_shape=[actual_l, d])

                    # ─── Step 1: L2 normalization + scale (merged) ───
                    query_scale, key_norm = l2norm_scaled(query_view_2d, key_view_2d, eps=1e-6, d=d)

                    # ─── Step 2: Pre-attention ───
                    gate_cum, decay_mask, a_block, key_beta = pre_attn(
                        gate_view, key_norm, beta_view, tril_mask, mask)

                    # ─── Step 3: Matrix inverse ───
                    a_block_inverse = inverse_pto(
                        attn=a_block, eye=eye)

                    # ─── Step 4: Cumulative decay ───
                    value_out, key_cum_out, g_exp = cal_value_and_key_cumdecay(
                        a_block_inverse, value_view_2d, beta_view, key_beta, gate_cum)

                    # ─── Step 5: Recurrent state attention ───
                    chunk_attn_out, cur_state = recurrent_state_attn_all(
                        query=query_scale, key=key_norm, value=value_out,
                        k_cumdecay=key_cum_out, gate=gate_cum, state=last_state,
                        decay_mask=decay_mask, tril=tril_mask, g_exp=g_exp)

                    # ─── Step 6: Write-back ───
                    last_state[:] = cur_state
                    core_attn_out[bs_ofs:bs_ofs + l, nv_idx] = chunk_attn_out
                    last_state_data[b_idx, nv_idx] = last_state

    return kernel


# ─────────────────────────────────────────────
# 3. Unaligned version kernel factory
# ─────────────────────────────────────────────

def chunk_gated_delta_rule_unaligned(b, nqk, nv, d, l, enable_perf_debug=False,
                                       stitch_function_max_num=8):
    """Factory function for unaligned version kernel (T % L != 0).

    stitch_function_max_num configurable (default 8, conservative due to is_loop_end branching).
    zeros passed as kernel inputs.
    chunk_size l configurable.

    Args:
        b: batch size
        nqk: number of query/key heads
        nv: number of value heads
        d: head dimension
        l: chunk size (configurable)
        enable_perf_debug: if True, enables runtime_debug_mode
        stitch_function_max_num: stitch parallelism (default 8)
    """
    jit_kwargs = {
        "runtime_options": {
            "stitch_function_max_num": stitch_function_max_num,
            "device_sched_parallelism": 8,
        },
        "pass_options": {
            "vec_nbuffer_setting": {9: 16, 1: 16, 28: 32, 6: 4, 25: 32, 26: 8,
                2: 8, 3: 8, 10: 32, 5: 4, 7: 16, 8: 32, 0: 4, 4: 4, 35: 8, 17: 8, 14: 30, 36: 16, -2: 1}
        },
    }
    if enable_perf_debug:
        jit_kwargs["debug_options"] = {"runtime_debug_mode": 1}

    @pypto.frontend.jit(**jit_kwargs)
    def kernel(
            query: pypto.Tensor([DYNAMIC_T, nqk, d], pypto.DT_FP32),
            key: pypto.Tensor([DYNAMIC_T, nqk, d], pypto.DT_FP32),
            value: pypto.Tensor([DYNAMIC_T, nv, d], pypto.DT_FP32),
            beta: pypto.Tensor([DYNAMIC_T, nv], pypto.DT_FP32),
            gate: pypto.Tensor([DYNAMIC_T, nv], pypto.DT_FP32),
            states: pypto.Tensor([DYNAMIC_B, nv, d, d], pypto.DT_FP32),
            mask: pypto.Tensor([l, l], pypto.DT_FP32),
            tril_mask: pypto.Tensor([l, l], pypto.DT_FP32),
            eye: pypto.Tensor([l, l], pypto.DT_FP32),
            act_seq_len: pypto.Tensor([DYNAMIC_B1], pypto.DT_INT32),
            core_attn_out: pypto.Tensor([DYNAMIC_T, nv, d], pypto.DT_FP32),
            last_state_data: pypto.Tensor([DYNAMIC_B, nv, d, d], pypto.DT_FP32),
    ):
        _, nqk, d = query.shape
        _, nv, d = value.shape
        b = states.shape[0]
        l, l = mask.shape
        group = nv // nqk
        pypto.experimental.set_operation_options(combine_axis=True)

        for b_idx in pypto.loop(b, name="LOOP_B_TND", idx_name="b_idx"):
            s = act_seq_len[b_idx + 1] - act_seq_len[b_idx]
            b_ofs = act_seq_len[b_idx]

            for nv_idx in pypto.loop(nv, name="LOOP_Nv_TND", idx_name="nv_idx", parallel=False):
                nqk_idx = nv_idx // group
                pypto.set_vec_tile_shapes(16, 16, 128, 128)
                last_state = states[b_idx, nv_idx]

                for s_idx in pypto.loop(0, s, l, name="LOOP_S_TND",
                                                    idx_name="s_idx", unroll_list=[16, 1]):
                    bs_ofs = b_ofs + s_idx
                    actual_l = (s - s_idx).min(l)

                    # ─── View slicing (3D) ───
                    query_view = pypto.view(query, [l, 1, d], [bs_ofs, nqk_idx, 0],
                                            valid_shape=[actual_l, 1, d])
                    key_view = pypto.view(key, [l, 1, d], [bs_ofs, nqk_idx, 0],
                                          valid_shape=[actual_l, 1, d])
                    value_view = pypto.view(value, [l, 1, d], [bs_ofs, nv_idx, 0],
                                            valid_shape=[actual_l, 1, d])
                    beta_view = pypto.view(beta, [l, 1], [bs_ofs, nv_idx],
                                           valid_shape=[actual_l, 1])
                    gate_view = pypto.view(gate, [l, 1], [bs_ofs, nv_idx],
                                           valid_shape=[actual_l, 1])

                    pypto.set_vec_tile_shapes(128, 128, 128)
                    query_view_2d = pypto.reshape(query_view, [l, d],
                                                  valid_shape=[actual_l, d])
                    key_view_2d = pypto.reshape(key_view, [l, d],
                                                 valid_shape=[actual_l, d])
                    value_view_2d = pypto.reshape(value_view, [l, d],
                                                   valid_shape=[actual_l, d])

                    # ─── is_loop_end branching ───
                    if pypto.is_loop_end(s_idx):
                        pad_q = pypto.fillpad(query_view_2d, "constant", 0.0)
                        pad_k = pypto.fillpad(key_view_2d, "constant", 0.0)
                        pad_v = pypto.fillpad(value_view_2d, "constant", 0.0)
                        pad_b = pypto.fillpad(beta_view, "constant", 0.0)
                        pad_g = pypto.fillpad(gate_view, "constant", 0.0)

                        query_scale, key_norm = l2norm_scaled(pad_q, pad_k, eps=1e-6, d=d)

                        gate_cum, decay_mask, a_block, key_beta = pre_attn(
                            pad_g, key_norm, pad_b, tril_mask, mask)

                        a_block_inverse = inverse_pto(
                            attn=a_block, eye=eye)

                        value_out, key_cum_out, g_exp = cal_value_and_key_cumdecay(
                            a_block_inverse, pad_v, pad_b, key_beta, gate_cum)

                        chunk_attn_out, cur_state = recurrent_state_attn_all(
                            query=query_scale, key=key_norm, value=value_out,
                            k_cumdecay=key_cum_out, gate=gate_cum, state=last_state,
                            decay_mask=decay_mask, tril=tril_mask, g_exp=g_exp)

                        last_state[:] = cur_state
                        last_state_data[b_idx, nv_idx] = last_state
                        pypto.set_vec_tile_shapes(128, 16, 128)
                        chunk_attn_out_reshaped = chunk_attn_out.reshape(
                            [l, 1, d], valid_shape=[actual_l, 1, d])
                        pypto.assemble(chunk_attn_out_reshaped,
                                       [bs_ofs, nv_idx, 0], core_attn_out)

                    else:
                        query_scale, key_norm = l2norm_scaled(query_view_2d, key_view_2d, eps=1e-6, d=d)

                        gate_cum, decay_mask, a_block, key_beta = pre_attn(
                            gate_view, key_norm, beta_view, tril_mask, mask)

                        a_block_inverse = inverse_pto(
                            attn=a_block, eye=eye)

                        value_out, key_cum_out, g_exp = cal_value_and_key_cumdecay(
                            a_block_inverse, value_view_2d, beta_view, key_beta, gate_cum)

                        chunk_attn_out, cur_state = recurrent_state_attn_all(
                            query=query_scale, key=key_norm, value=value_out,
                            k_cumdecay=key_cum_out, gate=gate_cum, state=last_state,
                            decay_mask=decay_mask, tril=tril_mask, g_exp=g_exp)

                        last_state[:] = cur_state
                        last_state_data[b_idx, nv_idx] = last_state
                        core_attn_out[bs_ofs:bs_ofs + l, nv_idx] = chunk_attn_out

    return kernel


# ─────────────────────────────────────────────
# 4. Wrapper function (exported interface)
# ─────────────────────────────────────────────

def _select_chunk_size(max_seq_len: int) -> int:
    """Auto-select optimal chunk_size based on maximum sequence length.

    Strategy (based on empirical performance data on NPU 910B3):
      - max_seq_len ≤ 32 → L=32  (single-chunk, 84us vs 116us for L=128)
      - max_seq_len ≤ 64 → L=64  (single-chunk, 90us vs 116us for L=128)
      - max_seq_len > 64 → L=128 (aligned multi-chunk: L=64 is +28~98% slower;
                                  unaligned T≈130 with tiny partial: L=64 may be
                                  ~8% faster, but difference is marginal)

    Empirical evidence for L=128 as default for T>L (aligned):
      - B=2,Nqk=2,Nv=8,T=512: L=128=389us, L=64=498us(+28%), L=32=771us(+98%)
      - B=2,Nqk=4,Nv=4,T=512: L=128=199us, L=64=232us(+16.5%)
      Root cause: S loop serial dependency — more chunks = more serial
      iterations, even though L=64 per-iteration is faster (~90us vs ~116us).

    Exception: unaligned T≈130 with tiny partial (<8 tokens), L=64 may be
    slightly faster (~8%) due to smaller fillpad/assemble overhead.

    Args:
        max_seq_len: maximum sequence length across all batches

    Returns:
        Recommended chunk_size (32, 64, or 128)
    """
    if max_seq_len <= 32:
        return 32
    elif max_seq_len <= 64:
        return 64
    else:
        return 128


def chunked_gated_delta_rule_wrapper(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    gate: torch.Tensor,
    states: torch.Tensor,
    act_seq_len: torch.Tensor,
    chunk_size="auto",
    mask: torch.Tensor = None,
    tril_mask: torch.Tensor = None,
    eye: torch.Tensor = None,
    enable_perf_debug: bool = False,
    stitch_function_max_num: int = None,
) -> tuple:
    """NPU call wrapper for chunked_gated_delta_rule.

    Dispatches to aligned or unaligned kernel based on sequence lengths.
    When chunk_size="auto", automatically selects optimal L and generates
    mask/tril_mask/eye tensors. When chunk_size is an explicit integer,
    mask/tril_mask/eye must be provided with matching dimensions.

    Auto chunk_size strategy:
      - max_seq_len ≤ 32 → L=32 (fastest single-chunk, ~75us)
      - max_seq_len ≤ 64 → L=64 (fast single-chunk, ~86us)
      - max_seq_len > 64 → L=128 (best for multi-chunk, ~118us)

    Args:
        query: [T, Nqk, D] float32
        key: [T, Nqk, D] float32
        value: [T, Nv, D] float32
        beta: [T, Nv] float32
        gate: [T, Nv] float32
        states: [B, Nv, D, D] float32
        act_seq_len: [B+1] int32
        chunk_size: chunk size L, or "auto" for automatic selection (default "auto")
        mask: [L, L] float32 (optional if chunk_size="auto", required if explicit)
        tril_mask: [L, L] float32 (optional if chunk_size="auto", required if explicit)
        eye: [L, L] float32 (optional if chunk_size="auto", required if explicit)
        enable_perf_debug: enable runtime_debug_mode
        stitch_function_max_num: stitch parallelism (None=auto: 32 aligned, 8 unaligned)

    Returns:
        (core_attn_out, last_state_data)
    """
    T = query.shape[0]
    Nqk = query.shape[1]
    D = query.shape[2]
    Nv = value.shape[1]
    B = states.shape[0]

    # ─── Auto chunk_size selection ───
    if chunk_size == "auto":
        max_seq_len = max(
            int(act_seq_len[i + 1]) - int(act_seq_len[i])
            for i in range(B)
        )
        L = _select_chunk_size(max_seq_len)
        helpers = prepare_chunk_helpers(L)
        mask = helpers["mask"]
        tril_mask = helpers["tril_mask"]
        eye = helpers["eye"]
    else:
        L = chunk_size
        if mask is None or tril_mask is None or eye is None:
            raise ValueError(
                f"When chunk_size={L} (explicit), mask, tril_mask, and eye "
                f"must be provided with dimensions matching L={L}. "
                f"Use chunk_size='auto' to auto-generate these tensors."
            )

    is_aligned = all(
        (int(act_seq_len[i + 1]) - int(act_seq_len[i])) % L == 0
        for i in range(B)
    )

    core_attn_out = torch.zeros([T, Nv, D], dtype=torch.float32).npu()
    last_state_data = torch.zeros([B, Nv, D, D], dtype=torch.float32).npu()

    query_npu = query.npu()
    key_npu = key.npu()
    value_npu = value.npu()
    beta_npu = beta.npu()
    gate_npu = gate.npu()
    states_npu = states.npu()
    mask_npu = mask.npu()
    tril_mask_npu = tril_mask.npu()
    eye_npu = eye.npu()
    act_seq_len_npu = act_seq_len.npu()

    if is_aligned:
        aligned_stitch = stitch_function_max_num if stitch_function_max_num is not None else 32
        kernel_fn = chunk_gated_delta_rule_aligned(B, Nqk, Nv, D, L,
                                                    enable_perf_debug=enable_perf_debug,
                                                    stitch_function_max_num=aligned_stitch)
    else:
        unaligned_stitch = stitch_function_max_num if stitch_function_max_num is not None else 8
        kernel_fn = chunk_gated_delta_rule_unaligned(B, Nqk, Nv, D, L,
                                                      enable_perf_debug=enable_perf_debug,
                                                      stitch_function_max_num=unaligned_stitch)

    input_data = [query_npu, key_npu, value_npu, beta_npu, gate_npu, states_npu,
                  mask_npu, tril_mask_npu, eye_npu,
                  act_seq_len_npu]
    output_data = [core_attn_out, last_state_data]
    kernel_fn(*input_data, *output_data)

    torch_npu.npu.synchronize()

    return core_attn_out.cpu(), last_state_data.cpu()