# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F821
"""PyPTO-Pro implementation of block_attn_res_update_rms_norm (Ascend 950PR / dav3510).

Native single-`@pl.jit` vector kernel. Numerics match the upstream reference contract:

    p      = partial_block + fp32(delta)                                   (FP32, in-place)
    score  = sum(p*q) / sqrt(mean(p^2) + score_eps)                        (precise div)
    m      = max(logit_max, score); a = exp(lmax-m); b = exp(score-m)
    h      = bf16_rn(numerator*alpha + p*beta), alpha/beta = a/den, b/den  (precise div)
    rstd   = 1 / sqrt(mean(fp32(h)^2) + norm_eps)                          (precise div)
    y      = bf16_rn(fp32(h) * rstd * fp32(gamma))

Layout / execution shape
    D = 7168 fixed (112 x 64 FP32 lanes), T >= 0 dynamic. One row per worker
    (`pl.get_block_idx()` stride), launched as `kernel[None, min(cores, T), key]`.
    Row metadata (logit_max / exp_sum) is read with `pl.getval` in the outer SIMD
    kernel and passed to the VF as scalars: no metadata tile, load or valid-shape
    bookkeeping per row.

Tiling key (internal specialization, no public-interface change)
    Path = 0  monolithic row: one VF for phases 1..3. Fewer VF boundaries, best
              when no concurrency can hide a boundary (measured T <= 2).
    Path = 1  staged row: phase1 VF -> p write-back -> phase2/3 VF, so the p DMA
              overlaps phases 2/3. Better once per-core rows overlap DMA (T >= 3).

UB plan (bytes, 32B aligned, peak 229632 <= 253952):
    partial 2x28672 | delta 2x14336 | numerator 2x28672 | h(f32) 28672 | y 14336
    | query 28672 | gamma 14336 | score 256

Ownership: each worker owns whole rows (no cross-core communication, no scratch
tensor). It writes y[row, :] (BF16) and p into partial_block[row, :] (FP32), so
launching min(vec_cores, T) workers covers every row exactly once. Transfers are
full-width (7168 elements); there is no padding, tail or valid-shape bookkeeping.

Numerical range (every division; score runs on the scalar core, the rest in VF bodies):
    den = exp_sum*a + b, a = exp(lmax - m) in (0, 1], b = exp(score - m) in (0, 1]
    with m = max(lmax, score), so den >= b > 0 and alpha = a/den, beta = b/den stay
    in [0, 1]. b underflows to +0 only when score - m < -103, where alpha -> 1 and
    beta -> 0 still hold. score_eps, norm_eps > 0 keep both sqrt arguments >= eps,
    so score = dot/sqrt(mean(p^2)+eps) and rstd = 1/sqrt(mean(h^2)+eps) are finite
    even for all-zero (or FTZ-flushed) p and h. The only cancellation-sensitive
    term is dot = sum(p*q): symmetric p makes it ~0 while p != 0, so both divisions
    use the precise (non-reciprocal) form and the cancellation regression in the
    test covers that input class.
"""

import torch

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import get_platform_info
from pypto_pro.runtime.tilingkey import TilingKeyField

HIDDEN_SIZE = 7168
N_CHUNKS = 56  # 7168 / 128 elements per even/odd register pair
INV_D = 1.0 / HIDDEN_SIZE

ROW_F32_BYTES = HIDDEN_SIZE * 4
ROW_B16_BYTES = HIDDEN_SIZE * 2

VA_PARTIAL = [0, ROW_F32_BYTES]
VA_DELTA = [2 * ROW_F32_BYTES, 2 * ROW_F32_BYTES + ROW_B16_BYTES]
VA_NUMER = [2 * ROW_F32_BYTES + 2 * ROW_B16_BYTES, 3 * ROW_F32_BYTES + 2 * ROW_B16_BYTES]
VA_H = 4 * ROW_F32_BYTES + 2 * ROW_B16_BYTES
VA_Y = VA_H + ROW_F32_BYTES
VA_QUERY = VA_Y + ROW_B16_BYTES
VA_GAMMA = VA_QUERY + ROW_F32_BYTES
VA_SCORE = VA_GAMMA + ROW_B16_BYTES

SMALL_T_THRESHOLD = 3  # below this, the monolithic row wins (measured crossover)


class RowPath:
    """Internal tiling key: algorithm class of the per-row schedule."""

    Path = TilingKeyField(bits=1, values=[0, 1])


@pl.vector_function
def phase1_vf(partial_t, delta_t, query_t, score_t, score_eps: pl.DT_FP32):
    """Update p in place, reduce the two row sums and store the broadcast score."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_u16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)

    dot0 = vf.full(0.0, preg, dtype=pl.DT_FP32)
    dot1 = vf.full(0.0, preg, dtype=pl.DT_FP32)
    ssq0 = vf.full(0.0, preg, dtype=pl.DT_FP32)
    ssq1 = vf.full(0.0, preg, dtype=pl.DT_FP32)
    for c in pl.range(0, N_CHUNKS // 2):
        base = c * 256
        d_lo_u = vf.load_align(delta_t, base, dist=pl.LoadDist.UNPK_B16)
        d_hi_u = vf.load_align(delta_t, base + 64, dist=pl.LoadDist.UNPK_B16)
        d_lo = vf.astype(d_lo_u, preg_u16, dtype=pl.DT_FP32)
        d_hi = vf.astype(d_hi_u, preg_u16, dtype=pl.DT_FP32)
        p_lo = vf.load_align(partial_t, base)
        p_hi = vf.load_align(partial_t, base + 64)
        p_lo = vf.add(p_lo, d_lo, preg)
        p_hi = vf.add(p_hi, d_hi, preg)
        vf.store_align(partial_t + base, p_lo, preg)
        vf.store_align(partial_t + base + 64, p_hi, preg)
        q_lo = vf.load_align(query_t, base)
        q_hi = vf.load_align(query_t, base + 64)
        dot0 = vf.mul_add_dst(p_lo, q_lo, preg)
        dot1 = vf.mul_add_dst(p_hi, q_hi, preg)
        ssq0 = vf.mul_add_dst(p_lo, p_lo, preg)
        ssq1 = vf.mul_add_dst(p_hi, p_hi, preg)
        d_lo_u = vf.load_align(delta_t, base + 128, dist=pl.LoadDist.UNPK_B16)
        d_hi_u = vf.load_align(delta_t, base + 128 + 64, dist=pl.LoadDist.UNPK_B16)
        d_lo = vf.astype(d_lo_u, preg_u16, dtype=pl.DT_FP32)
        d_hi = vf.astype(d_hi_u, preg_u16, dtype=pl.DT_FP32)
        p_lo = vf.load_align(partial_t, base + 128)
        p_hi = vf.load_align(partial_t, base + 128 + 64)
        p_lo = vf.add(p_lo, d_lo, preg)
        p_hi = vf.add(p_hi, d_hi, preg)
        vf.store_align(partial_t + base + 128, p_lo, preg)
        vf.store_align(partial_t + base + 128 + 64, p_hi, preg)
        q_lo = vf.load_align(query_t, base + 128)
        q_hi = vf.load_align(query_t, base + 128 + 64)
        dot0 = vf.mul_add_dst(p_lo, q_lo, preg)
        dot1 = vf.mul_add_dst(p_hi, q_hi, preg)
        ssq0 = vf.mul_add_dst(p_lo, p_lo, preg)
        ssq1 = vf.mul_add_dst(p_hi, p_hi, preg)
    dot_sum = vf.add(dot0, dot1, preg)
    ssq_sum = vf.add(ssq0, ssq1, preg)
    dot_red = vf.reduce_sum(dot_sum, preg)
    ssq_red = vf.reduce_sum(ssq_sum, preg)
    dot_b = vf.full(dot_red, preg)
    ssq_b = vf.full(ssq_red, preg)
    rms = vf.muls(ssq_b, INV_D, preg)
    rms = vf.adds(rms, score_eps, preg)
    rms = vf.sqrt(rms, preg)
    vf.store_align(score_t, dot_b, preg)
    vf.store_align(score_t + 32, rms, preg)


@pl.vector_function
def phase23_vf(
    partial_t, numer_t, h_t, y_t, gamma_t,
    score_val: pl.DT_FP32, lmax_val: pl.DT_FP32, esum_val: pl.DT_FP32,
    norm_eps: pl.DT_FP32,
):
    """Merge online-softmax state, emit h (bf16) and y (bf16); rstd from rounded h."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_u16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)

    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)  # p written in phase 1, read here
    score_b = vf.full(score_val, preg)
    lmax_b = vf.full(lmax_val, preg)
    esum_b = vf.full(esum_val, preg)
    m_b = vf.max(lmax_b, score_b, preg)
    a_b = vf.exp_sub(lmax_b, m_b, preg)
    b_b = vf.exp_sub(score_b, m_b, preg)
    denom = vf.mul(esum_b, a_b, preg)
    denom = vf.add(denom, b_b, preg)
    inv_den = vf.div(one, denom, preg, precision=True)
    alpha = vf.mul(a_b, inv_den, preg)
    beta = vf.mul(b_b, inv_den, preg)

    ssq_h0 = vf.full(0.0, preg, dtype=pl.DT_FP32)
    ssq_h1 = vf.full(0.0, preg, dtype=pl.DT_FP32)
    for c in pl.range(0, N_CHUNKS // 2):
        base = c * 256
        p_lo = vf.load_align(partial_t, base)
        p_hi = vf.load_align(partial_t, base + 64)
        n_lo = vf.load_align(numer_t, base)
        n_hi = vf.load_align(numer_t, base + 64)
        out_lo = vf.mul(n_lo, alpha, preg)
        out_hi = vf.mul(n_hi, alpha, preg)
        out_lo = vf.mul_add_dst(p_lo, beta, preg)
        out_hi = vf.mul_add_dst(p_hi, beta, preg)
        bf_even = vf.astype(out_lo, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                            round_mode=pl.VFRoundMode.CAST_RINT)
        bf_odd = vf.astype(out_hi, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                           round_mode=pl.VFRoundMode.CAST_RINT)
        h_lo = vf.astype(bf_even, preg_u16, dtype=pl.DT_FP32)
        h_hi = vf.astype(bf_odd, preg_u16, dtype=pl.DT_FP32)
        ssq_h0 = vf.mul_add_dst(h_lo, h_lo, preg)
        ssq_h1 = vf.mul_add_dst(h_hi, h_hi, preg)
        vf.store_align(h_t + base, h_lo, preg)
        vf.store_align(h_t + base + 64, h_hi, preg)

        p_lo = vf.load_align(partial_t, base + 128)
        p_hi = vf.load_align(partial_t, base + 128 + 64)
        n_lo = vf.load_align(numer_t, base + 128)
        n_hi = vf.load_align(numer_t, base + 128 + 64)
        out_lo = vf.mul(n_lo, alpha, preg)
        out_hi = vf.mul(n_hi, alpha, preg)
        out_lo = vf.mul_add_dst(p_lo, beta, preg)
        out_hi = vf.mul_add_dst(p_hi, beta, preg)
        bf_even = vf.astype(out_lo, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                            round_mode=pl.VFRoundMode.CAST_RINT)
        bf_odd = vf.astype(out_hi, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                           round_mode=pl.VFRoundMode.CAST_RINT)
        h_lo = vf.astype(bf_even, preg_u16, dtype=pl.DT_FP32)
        h_hi = vf.astype(bf_odd, preg_u16, dtype=pl.DT_FP32)
        ssq_h0 = vf.mul_add_dst(h_lo, h_lo, preg)
        ssq_h1 = vf.mul_add_dst(h_hi, h_hi, preg)
        vf.store_align(h_t + base + 128, h_lo, preg)
        vf.store_align(h_t + base + 128 + 64, h_hi, preg)

    ssq_h_sum = vf.add(ssq_h0, ssq_h1, preg)
    ssq_h_red = vf.reduce_sum(ssq_h_sum, preg)
    ssq_h_b = vf.full(ssq_h_red, preg)
    rms_h = vf.muls(ssq_h_b, INV_D, preg)
    rms_h = vf.adds(rms_h, norm_eps, preg)
    rms_h = vf.sqrt(rms_h, preg)
    rstd_b = vf.div(one, rms_h, preg, precision=True)

    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)  # h written above, re-read below
    for c in pl.range(0, N_CHUNKS // 2):
        base = c * 256
        h_e = vf.load_align(h_t, base)
        h_o = vf.load_align(h_t, base + 64)
        g_lo_u = vf.load_align(gamma_t, base, dist=pl.LoadDist.UNPK_B16)
        g_e = vf.astype(g_lo_u, preg_u16, dtype=pl.DT_FP32)
        g_hi_u = vf.load_align(gamma_t, base + 64, dist=pl.LoadDist.UNPK_B16)
        g_o = vf.astype(g_hi_u, preg_u16, dtype=pl.DT_FP32)
        y_e = vf.mul(h_e, rstd_b, preg)
        y_o = vf.mul(h_o, rstd_b, preg)
        y_e = vf.mul(y_e, g_e, preg)
        y_o = vf.mul(y_o, g_o, preg)
        yb_even = vf.astype(y_e, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                            round_mode=pl.VFRoundMode.CAST_RINT)
        yb_odd = vf.astype(y_o, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                           round_mode=pl.VFRoundMode.CAST_RINT)
        vf.store_align(y_t + base, yb_even, preg_u16, dist=pl.StoreDist.PACK_B32)
        vf.store_align(y_t + base + 64, yb_odd, preg_u16, dist=pl.StoreDist.PACK_B32)

        h_e = vf.load_align(h_t, base + 128)
        h_o = vf.load_align(h_t, base + 128 + 64)
        g_lo_u = vf.load_align(gamma_t, base + 128, dist=pl.LoadDist.UNPK_B16)
        g_e = vf.astype(g_lo_u, preg_u16, dtype=pl.DT_FP32)
        g_hi_u = vf.load_align(gamma_t, base + 128 + 64, dist=pl.LoadDist.UNPK_B16)
        g_o = vf.astype(g_hi_u, preg_u16, dtype=pl.DT_FP32)
        y_e = vf.mul(h_e, rstd_b, preg)
        y_o = vf.mul(h_o, rstd_b, preg)
        y_e = vf.mul(y_e, g_e, preg)
        y_o = vf.mul(y_o, g_o, preg)
        yb_even = vf.astype(y_e, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                            round_mode=pl.VFRoundMode.CAST_RINT)
        yb_odd = vf.astype(y_o, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                           round_mode=pl.VFRoundMode.CAST_RINT)
        vf.store_align(y_t + base + 128, yb_even, preg_u16, dist=pl.StoreDist.PACK_B32)
        vf.store_align(y_t + base + 128 + 64, yb_odd, preg_u16, dist=pl.StoreDist.PACK_B32)


@pl.vector_function
def process_row_vf(
    partial_t, delta_t, numer_t, h_t, y_t, query_t, gamma_t,
    lmax_val: pl.DT_FP32, esum_val: pl.DT_FP32,
    score_eps: pl.DT_FP32, norm_eps: pl.DT_FP32,
):
    """Monolithic row (Path=0): phases 1..3 in one VF, no intermediate VF boundary."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_u16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)

    dot0 = vf.full(0.0, preg, dtype=pl.DT_FP32)
    dot1 = vf.full(0.0, preg, dtype=pl.DT_FP32)
    ssq0 = vf.full(0.0, preg, dtype=pl.DT_FP32)
    ssq1 = vf.full(0.0, preg, dtype=pl.DT_FP32)
    for c in pl.range(0, N_CHUNKS // 2):
        base = c * 256
        d_lo_u = vf.load_align(delta_t, base, dist=pl.LoadDist.UNPK_B16)
        d_hi_u = vf.load_align(delta_t, base + 64, dist=pl.LoadDist.UNPK_B16)
        d_lo = vf.astype(d_lo_u, preg_u16, dtype=pl.DT_FP32)
        d_hi = vf.astype(d_hi_u, preg_u16, dtype=pl.DT_FP32)
        p_lo = vf.load_align(partial_t, base)
        p_hi = vf.load_align(partial_t, base + 64)
        p_lo = vf.add(p_lo, d_lo, preg)
        p_hi = vf.add(p_hi, d_hi, preg)
        vf.store_align(partial_t + base, p_lo, preg)
        vf.store_align(partial_t + base + 64, p_hi, preg)
        q_lo = vf.load_align(query_t, base)
        q_hi = vf.load_align(query_t, base + 64)
        dot0 = vf.mul_add_dst(p_lo, q_lo, preg)
        dot1 = vf.mul_add_dst(p_hi, q_hi, preg)
        ssq0 = vf.mul_add_dst(p_lo, p_lo, preg)
        ssq1 = vf.mul_add_dst(p_hi, p_hi, preg)
        d_lo_u = vf.load_align(delta_t, base + 128, dist=pl.LoadDist.UNPK_B16)
        d_hi_u = vf.load_align(delta_t, base + 128 + 64, dist=pl.LoadDist.UNPK_B16)
        d_lo = vf.astype(d_lo_u, preg_u16, dtype=pl.DT_FP32)
        d_hi = vf.astype(d_hi_u, preg_u16, dtype=pl.DT_FP32)
        p_lo = vf.load_align(partial_t, base + 128)
        p_hi = vf.load_align(partial_t, base + 128 + 64)
        p_lo = vf.add(p_lo, d_lo, preg)
        p_hi = vf.add(p_hi, d_hi, preg)
        vf.store_align(partial_t + base + 128, p_lo, preg)
        vf.store_align(partial_t + base + 128 + 64, p_hi, preg)
        q_lo = vf.load_align(query_t, base + 128)
        q_hi = vf.load_align(query_t, base + 128 + 64)
        dot0 = vf.mul_add_dst(p_lo, q_lo, preg)
        dot1 = vf.mul_add_dst(p_hi, q_hi, preg)
        ssq0 = vf.mul_add_dst(p_lo, p_lo, preg)
        ssq1 = vf.mul_add_dst(p_hi, p_hi, preg)
    dot_sum = vf.add(dot0, dot1, preg)
    ssq_sum = vf.add(ssq0, ssq1, preg)
    dot_red = vf.reduce_sum(dot_sum, preg)
    ssq_red = vf.reduce_sum(ssq_sum, preg)
    dot_b = vf.full(dot_red, preg)
    ssq_b = vf.full(ssq_red, preg)
    rms = vf.muls(ssq_b, INV_D, preg)
    rms = vf.adds(rms, score_eps, preg)
    rms = vf.sqrt(rms, preg)
    score_b = vf.div(dot_b, rms, preg, precision=True)

    lmax_b = vf.full(lmax_val, preg)
    esum_b = vf.full(esum_val, preg)
    m_b = vf.max(lmax_b, score_b, preg)
    a_b = vf.exp_sub(lmax_b, m_b, preg)
    b_b = vf.exp_sub(score_b, m_b, preg)
    denom = vf.mul(esum_b, a_b, preg)
    denom = vf.add(denom, b_b, preg)
    inv_den = vf.div(one, denom, preg, precision=True)
    alpha = vf.mul(a_b, inv_den, preg)
    beta = vf.mul(b_b, inv_den, preg)

    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)  # p written above, re-read below
    ssq_h0 = vf.full(0.0, preg, dtype=pl.DT_FP32)
    ssq_h1 = vf.full(0.0, preg, dtype=pl.DT_FP32)
    for c in pl.range(0, N_CHUNKS // 2):
        base = c * 256
        p_lo = vf.load_align(partial_t, base)
        p_hi = vf.load_align(partial_t, base + 64)
        n_lo = vf.load_align(numer_t, base)
        n_hi = vf.load_align(numer_t, base + 64)
        out_lo = vf.mul(n_lo, alpha, preg)
        out_hi = vf.mul(n_hi, alpha, preg)
        out_lo = vf.mul_add_dst(p_lo, beta, preg)
        out_hi = vf.mul_add_dst(p_hi, beta, preg)
        bf_even = vf.astype(out_lo, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                            round_mode=pl.VFRoundMode.CAST_RINT)
        bf_odd = vf.astype(out_hi, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                           round_mode=pl.VFRoundMode.CAST_RINT)
        h_lo = vf.astype(bf_even, preg_u16, dtype=pl.DT_FP32)
        h_hi = vf.astype(bf_odd, preg_u16, dtype=pl.DT_FP32)
        ssq_h0 = vf.mul_add_dst(h_lo, h_lo, preg)
        ssq_h1 = vf.mul_add_dst(h_hi, h_hi, preg)
        vf.store_align(h_t + base, h_lo, preg)
        vf.store_align(h_t + base + 64, h_hi, preg)

        p_lo = vf.load_align(partial_t, base + 128)
        p_hi = vf.load_align(partial_t, base + 128 + 64)
        n_lo = vf.load_align(numer_t, base + 128)
        n_hi = vf.load_align(numer_t, base + 128 + 64)
        out_lo = vf.mul(n_lo, alpha, preg)
        out_hi = vf.mul(n_hi, alpha, preg)
        out_lo = vf.mul_add_dst(p_lo, beta, preg)
        out_hi = vf.mul_add_dst(p_hi, beta, preg)
        bf_even = vf.astype(out_lo, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                            round_mode=pl.VFRoundMode.CAST_RINT)
        bf_odd = vf.astype(out_hi, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                           round_mode=pl.VFRoundMode.CAST_RINT)
        h_lo = vf.astype(bf_even, preg_u16, dtype=pl.DT_FP32)
        h_hi = vf.astype(bf_odd, preg_u16, dtype=pl.DT_FP32)
        ssq_h0 = vf.mul_add_dst(h_lo, h_lo, preg)
        ssq_h1 = vf.mul_add_dst(h_hi, h_hi, preg)
        vf.store_align(h_t + base + 128, h_lo, preg)
        vf.store_align(h_t + base + 128 + 64, h_hi, preg)

    ssq_h_sum = vf.add(ssq_h0, ssq_h1, preg)
    ssq_h_red = vf.reduce_sum(ssq_h_sum, preg)
    ssq_h_b = vf.full(ssq_h_red, preg)
    rms_h = vf.muls(ssq_h_b, INV_D, preg)
    rms_h = vf.adds(rms_h, norm_eps, preg)
    rms_h = vf.sqrt(rms_h, preg)
    rstd_b = vf.div(one, rms_h, preg, precision=True)

    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)  # h written above, re-read below
    for c in pl.range(0, N_CHUNKS // 2):
        base = c * 256
        h_e = vf.load_align(h_t, base)
        h_o = vf.load_align(h_t, base + 64)
        g_lo_u = vf.load_align(gamma_t, base, dist=pl.LoadDist.UNPK_B16)
        g_e = vf.astype(g_lo_u, preg_u16, dtype=pl.DT_FP32)
        g_hi_u = vf.load_align(gamma_t, base + 64, dist=pl.LoadDist.UNPK_B16)
        g_o = vf.astype(g_hi_u, preg_u16, dtype=pl.DT_FP32)
        y_e = vf.mul(h_e, rstd_b, preg)
        y_o = vf.mul(h_o, rstd_b, preg)
        y_e = vf.mul(y_e, g_e, preg)
        y_o = vf.mul(y_o, g_o, preg)
        yb_even = vf.astype(y_e, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                            round_mode=pl.VFRoundMode.CAST_RINT)
        yb_odd = vf.astype(y_o, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                           round_mode=pl.VFRoundMode.CAST_RINT)
        vf.store_align(y_t + base, yb_even, preg_u16, dist=pl.StoreDist.PACK_B32)
        vf.store_align(y_t + base + 64, yb_odd, preg_u16, dist=pl.StoreDist.PACK_B32)

        h_e = vf.load_align(h_t, base + 128)
        h_o = vf.load_align(h_t, base + 128 + 64)
        g_lo_u = vf.load_align(gamma_t, base + 128, dist=pl.LoadDist.UNPK_B16)
        g_e = vf.astype(g_lo_u, preg_u16, dtype=pl.DT_FP32)
        g_hi_u = vf.load_align(gamma_t, base + 128 + 64, dist=pl.LoadDist.UNPK_B16)
        g_o = vf.astype(g_hi_u, preg_u16, dtype=pl.DT_FP32)
        y_e = vf.mul(h_e, rstd_b, preg)
        y_o = vf.mul(h_o, rstd_b, preg)
        y_e = vf.mul(y_e, g_e, preg)
        y_o = vf.mul(y_o, g_o, preg)
        yb_even = vf.astype(y_e, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                            round_mode=pl.VFRoundMode.CAST_RINT)
        yb_odd = vf.astype(y_o, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                           round_mode=pl.VFRoundMode.CAST_RINT)
        vf.store_align(y_t + base + 128, yb_even, preg_u16, dist=pl.StoreDist.PACK_B32)
        vf.store_align(y_t + base + 128 + 64, yb_odd, preg_u16, dist=pl.StoreDist.PACK_B32)


@pl.jit(auto_mutex=True, tiling_key=RowPath)
def block_attn_res_update_rms_norm_kernel(
    partial_block: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    delta: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    pseudo_query: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32],
    numerator: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    logit_max: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32],
    exp_sum: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32],
    gamma: pl.Tensor[[pl.DYNAMIC], pl.DT_BF16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    score_eps: pl.DT_FP32,
    norm_eps: pl.DT_FP32,
):
    row_tt_f32 = pl.TileType(shape=[1, HIDDEN_SIZE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    row_tt_b16 = pl.TileType(shape=[1, HIDDEN_SIZE], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
    score_tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)

    partial_grp = pl.make_tile_group(type=row_tt_f32, addrs=VA_PARTIAL, mutex_ids=[0, 1])
    delta_grp = pl.make_tile_group(type=row_tt_b16, addrs=VA_DELTA, mutex_ids=[2, 3])
    numer_grp = pl.make_tile_group(type=row_tt_f32, addrs=VA_NUMER, mutex_ids=[4, 5])
    h_grp = pl.make_tile_group(type=row_tt_f32, addrs=VA_H, mutex_ids=[6])
    y_grp = pl.make_tile_group(type=row_tt_b16, addrs=VA_Y, mutex_ids=[7])
    query_grp = pl.make_tile_group(type=row_tt_f32, addrs=VA_QUERY, mutex_ids=[8])
    gamma_grp = pl.make_tile_group(type=row_tt_b16, addrs=VA_GAMMA, mutex_ids=[9])

    with pl.section_vector():
        tokens = partial_block.shape[0]
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()

        query_slot = query_grp.next()
        pl.load(query_slot, pseudo_query, [0])
        gamma_slot = gamma_grp.next()

        if Path == 0:
            # Monolithic row: one VF for phases 1..3 (best when no concurrency exists).
            for row in pl.range(core_id, tokens, num_cores):
                p_slot = partial_grp.next()
                d_slot = delta_grp.next()
                n_slot = numer_grp.next()
                h_slot = h_grp.next()

                pl.load(p_slot, partial_block, [row, 0])
                pl.load(d_slot, delta, [row, 0])
                pl.load(n_slot, numerator, [row, 0])
                if row == core_id:
                    # Keep gamma (14KB) off the phase-1 MTE2 prefix.
                    pl.load(gamma_slot, gamma, [0])
                lmax_val = pl.getval(logit_max, row)
                esum_val = pl.getval(exp_sum, row)

                y_slot = y_grp.next()

                process_row_vf(p_slot, d_slot, n_slot, h_slot, y_slot,
                               query_slot, gamma_slot, lmax_val, esum_val,
                               score_eps, norm_eps)

                pl.store(partial_block, p_slot, [row, 0])
                pl.store(y, y_slot, [row, 0])
        else:
            # Staged row: release p's DMA right after phase 1 so it overlaps phases 2/3.
            score_grp = pl.make_tile_group(type=score_tt, addrs=VA_SCORE, mutex_ids=[10])
            score_slot = score_grp.next()
            for row in pl.range(core_id, tokens, num_cores):
                p_slot = partial_grp.next()
                d_slot = delta_grp.next()

                pl.load(p_slot, partial_block, [row, 0])
                pl.load(d_slot, delta, [row, 0])
                phase1_vf(p_slot, d_slot, query_slot, score_slot, score_eps)

                pl.store(partial_block, p_slot, [row, 0])

                n_slot = numer_grp.next()
                pl.load(n_slot, numerator, [row, 0])
                if row == core_id:
                    pl.load(gamma_slot, gamma, [0])
                lmax_val = pl.getval(logit_max, row)
                esum_val = pl.getval(exp_sum, row)
                # Precise score division runs on the scalar core while the
                # vector pipe drains phase 1 (dot/rms are uniform in-lane).
                dot_val = pl.getval(score_slot, 0)
                rms_val = pl.getval(score_slot, 32)
                score_val = dot_val / rms_val

                h_slot = h_grp.next()
                y_slot = y_grp.next()
                phase23_vf(p_slot, n_slot, h_slot, y_slot, gamma_slot,
                           score_val, lmax_val, esum_val, norm_eps)

                pl.store(y, y_slot, [row, 0])
    return


_CORE_NUM = None


def _vector_core_num() -> int:
    global _CORE_NUM
    if _CORE_NUM is None:
        _CORE_NUM = int(get_platform_info().vector_core_num)
    return _CORE_NUM


def select_path(tokens: int) -> int:
    """Return the tiling-key route for the given row count."""
    return 0 if tokens < SMALL_T_THRESHOLD else 1


def _validate_tensor_meta(inputs, ref):
    """Every input must be a contiguous torch.Tensor on the reference device."""
    for t in inputs:
        if not isinstance(t, torch.Tensor):
            raise ValueError("all inputs must be torch.Tensor")
        if not t.is_contiguous():
            raise ValueError("all inputs must be contiguous")
        if t.device != ref.device:
            raise ValueError("all inputs must be on the same device")


def _validate_row_meta(partial_block):
    """partial_block must be [T, D] with T >= 0 and the fixed D; return (T, D)."""
    if partial_block.dim() != 2:
        raise ValueError("partial_block must be [T, D]")
    tokens, d = int(partial_block.shape[0]), int(partial_block.shape[1])
    if tokens < 0:
        raise ValueError("T must be >= 0")
    if d != HIDDEN_SIZE:
        raise ValueError(f"D must be {HIDDEN_SIZE}")
    return tokens, d


def _validate_operand_shapes(delta, pseudo_query, numerator, logit_max, exp_sum, gamma, tokens, d):
    """Every operand must match its declared dtype and [T, D]-derived shape."""
    if delta.dtype != torch.bfloat16 or tuple(delta.shape) != (tokens, d):
        raise ValueError(f"delta must be [{tokens}, {d}] bfloat16")
    if pseudo_query.dtype != torch.float32 or tuple(pseudo_query.shape) != (d,):
        raise ValueError(f"pseudo_query must be [{d},] float32")
    if numerator.dtype != torch.float32 or tuple(numerator.shape) != (tokens, d):
        raise ValueError(f"numerator must be [{tokens}, {d}] float32")
    if logit_max.dtype != torch.float32 or tuple(logit_max.shape) != (tokens,):
        raise ValueError(f"logit_max must be [{tokens},] float32")
    if exp_sum.dtype != torch.float32 or tuple(exp_sum.shape) != (tokens,):
        raise ValueError(f"exp_sum must be [{tokens},] float32")
    if gamma.dtype != torch.bfloat16 or tuple(gamma.shape) != (d,):
        raise ValueError(f"gamma must be [{d},] bfloat16")


def _validate_eps(score_eps, norm_eps):
    """Both epsilons must be finite positive floats (ints accepted)."""
    for name, v in (("score_eps", score_eps), ("norm_eps", norm_eps)):
        if not (isinstance(v, (int, float)) and v > 0.0 and v == v and v != float("inf")):
            raise ValueError(f"{name} must be a finite positive float")


def _validate_inputs(partial_block, delta, pseudo_query, numerator, logit_max, exp_sum, gamma,
                     score_eps, norm_eps):
    """Validate every public input; return (T, D)."""
    inputs = (partial_block, delta, pseudo_query, numerator, logit_max, exp_sum, gamma)
    _validate_tensor_meta(inputs, partial_block)
    tokens, d = _validate_row_meta(partial_block)
    _validate_operand_shapes(delta, pseudo_query, numerator, logit_max, exp_sum, gamma, tokens, d)
    _validate_eps(score_eps, norm_eps)
    if partial_block.device.type != "npu":
        raise ValueError("block_attn_res_update_rms_norm requires NPU tensors")
    return tokens, d


def block_attn_res_update_rms_norm(
    partial_block: torch.Tensor,
    delta: torch.Tensor,
    pseudo_query: torch.Tensor,
    numerator: torch.Tensor,
    logit_max: torch.Tensor,
    exp_sum: torch.Tensor,
    gamma: torch.Tensor,
    score_eps: float = 1e-6,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    """Public callable. partial_block is updated in place (p); returns y [T, D] BF16."""
    tokens, d = _validate_inputs(partial_block, delta, pseudo_query, numerator,
                                 logit_max, exp_sum, gamma, score_eps, norm_eps)
    # An int is accepted above; normalize once here because the kernel parameter is fp32.
    score_eps = float(score_eps)
    norm_eps = float(norm_eps)

    y = torch.empty((tokens, d), dtype=torch.bfloat16, device=partial_block.device)
    if tokens == 0:
        return y
    workers = min(_vector_core_num(), tokens)
    path = select_path(tokens)
    block_attn_res_update_rms_norm_kernel[None, workers, {"Path": path}](
        partial_block, delta, pseudo_query, numerator, logit_max, exp_sum, gamma, y,
        score_eps, norm_eps,
    )
    return y
