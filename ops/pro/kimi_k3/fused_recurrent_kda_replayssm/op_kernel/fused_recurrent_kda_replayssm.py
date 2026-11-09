#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO-Pro fused recurrent KDA ReplaySSM verify kernel (Ascend 950).

One packed ``mixedqkv`` in ``[Q(P) | K(P) | V(P)]`` order (``P = heads * 128``);
per (batch, head) the operator normalizes Q/K rows, activates gate/beta, runs
the KDA token recurrence on a read-only FP32 ``[128, 128]`` checkpoint, writes
the BF16 output and the FP16 U/K + FP32 decay replay records.

Math per token (S is the pre-token state)::

    d     = exp(lower * sigmoid(exp(a_log) * (g + dt_bias)))
    beta  = sigmoid(beta_raw)
    S_bar = S * diag(d)
    u     = beta * (V - S_bar @ k)
    S_new = S_bar + outer(u, k)
    o     = scale * (S_new @ q)

Design notes
------------
* Pure Vector kernel.  One work item covers ``rows`` state rows of one
  (batch, head); rows are independent in the recurrence, so a head splits.
  Items are assigned block-cyclically and ``rows`` is picked from the measured
  work-size thresholds so every vector worker is filled.
* Folding removes one per-row reduction from the update pass::

      kq[r] = dot(S_row, d * q_hat * scale)     (pass 1, with sk)
      sk[r] = dot(S_row, beta * d * k_hat)      (pass 1)
      u[r]  = beta*V[r] - sk[r]                 (flat over rows)
      o[r]  = kq[r] + u[r] * dot(k_hat, q_hat*scale)
      S_new = d * S_row + u[r] * k_hat          (pass 2)

  The pass-1 weights are built in registers and consumed by the same VF call
  (the state dot is fused into prep), so they never round-trip through UB.
* Cross-token prefetch: the five per-token input tiles use two explicit parity
  groups.  Token ``t+1`` is loaded while token ``t`` computes, with
  ``set_flag``/``wait_flag`` pairs ordering (a) the V reads of the parity being
  overwritten and (b) the MTE2 load against the next V read.
* The working state lives in a 2-slot UB tile ring (VF-only after the initial
  MTE2 load, so the vector pipe orders the swap).
* BF16 inputs widen with ``LoadDist.UNPK_B16`` + ``astype(FP32, ZERO)``;
  BF16 output / FP16 records narrow with ``PACK_B32`` dense stores.
"""

from __future__ import annotations

import torch

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf
from pypto_pro.runtime.tilingkey import TilingKeyField

FP32 = pl.DT_FP32
BF16 = pl.DT_BF16
FP16 = pl.DT_FP16
I32 = pl.DT_INT32

HEAD_DIM = 128
MAX_HEADS = 96
VL = 64
SUPPORTED_ROWS = (16, 32, 64)
DEFAULT_ROWS = 64
STATE_SLOTS = 3
_EPS = 1e-6


@pl.vector_function
def _prep_dot(state, q_bf, k_bf, g_bf, v_bf, beta_bf, dtb,
              kn, dec, kqt, vf32, rk_t, rd_t, skf, sqf,
              scale: pl.DT_FP32, lower: pl.DT_FP32, alog: pl.DT_FP32,
              rows: pl.DT_INT64):
    """Prep and both state dots in one call: dk/dq never leave registers."""
    full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    mask_r = vf.update_mask(rows, dtype=FP32)
    one = vf.full(1.0, full, dtype=FP32)

    bp = vf.load_align(beta_bf, [0, 0], dist=pl.LoadDist.UNPK_B16)
    bv = vf.astype(bp, full, dtype=FP32, layout=pl.CastLayout.ZERO)
    bn = vf.neg(bv, full)
    be = vf.exp(bn, full)
    bd = vf.adds(be, 1.0, full)
    beta_s = vf.div(one, bd, full)
    beta_b = vf.full(beta_s, full)

    q_lo = vf.load_align(q_bf, [0, 0], dist=pl.LoadDist.UNPK_B16)
    q_hi = vf.load_align(q_bf, [0, VL], dist=pl.LoadDist.UNPK_B16)
    q_pre = vf.astype(q_lo, full, dtype=FP32, layout=pl.CastLayout.ZERO)
    q_post = vf.astype(q_hi, full, dtype=FP32, layout=pl.CastLayout.ZERO)
    sqs = vf.mul(q_pre, q_pre, full)
    sqs = vf.mul_add_dst(q_post, q_post, full)
    q_sum = vf.reduce_sum(sqs, full)
    q_eps = vf.adds(q_sum, _EPS, full)
    q_root = vf.sqrt(q_eps, full)
    q_inv = vf.div(one, q_root, full)
    q_inv_b = vf.full(q_inv, full)
    qn_pre = vf.mul(q_pre, q_inv_b, full)
    qn_pre = vf.muls(qn_pre, scale, full)
    qn_post = vf.mul(q_post, q_inv_b, full)
    qn_post = vf.muls(qn_post, scale, full)

    k_lo = vf.load_align(k_bf, [0, 0], dist=pl.LoadDist.UNPK_B16)
    k_hi = vf.load_align(k_bf, [0, VL], dist=pl.LoadDist.UNPK_B16)
    k_pre = vf.astype(k_lo, full, dtype=FP32, layout=pl.CastLayout.ZERO)
    k_post = vf.astype(k_hi, full, dtype=FP32, layout=pl.CastLayout.ZERO)
    ks = vf.mul(k_pre, k_pre, full)
    ks = vf.mul_add_dst(k_post, k_post, full)
    k_sum = vf.reduce_sum(ks, full)
    k_eps = vf.adds(k_sum, _EPS, full)
    k_root = vf.sqrt(k_eps, full)
    k_inv = vf.div(one, k_root, full)
    k_inv_b = vf.full(k_inv, full)
    kn_pre = vf.mul(k_pre, k_inv_b, full)
    kn_post = vf.mul(k_post, k_inv_b, full)
    vf.store_align(kn, kn_pre, full, [0, 0])
    vf.store_align(kn, kn_post, full, [0, VL])

    kq = vf.mul(kn_pre, qn_pre, full)
    kq = vf.mul_add_dst(kn_post, qn_post, full)
    kq = vf.reduce_sum(kq, full)
    vf.store_align(kqt, kq, full, [0, 0], dist=pl.StoreDist.FIRST_ELEMENT_B32)

    g_lo = vf.load_align(g_bf, [0, 0], dist=pl.LoadDist.UNPK_B16)
    g_hi = vf.load_align(g_bf, [0, VL], dist=pl.LoadDist.UNPK_B16)
    g_pre = vf.astype(g_lo, full, dtype=FP32, layout=pl.CastLayout.ZERO)
    g_post = vf.astype(g_hi, full, dtype=FP32, layout=pl.CastLayout.ZERO)
    b_pre = vf.load_align(dtb, [0, 0])
    b_post = vf.load_align(dtb, [0, VL])
    t_pre = vf.add(g_pre, b_pre, full)
    t_post = vf.add(g_post, b_post, full)
    alpha_v = vf.full(alog, full, dtype=FP32)
    alpha_e = vf.exp(alpha_v, full)
    neg_alpha = vf.neg(alpha_e, full)
    t_pre = vf.mul(t_pre, neg_alpha, full)
    t_post = vf.mul(t_post, neg_alpha, full)
    e_pre = vf.exp(t_pre, full)
    e_post = vf.exp(t_post, full)
    d_pre = vf.adds(e_pre, 1.0, full)
    d_post = vf.adds(e_post, 1.0, full)
    s_pre = vf.div(one, d_pre, full)
    s_post = vf.div(one, d_post, full)
    a_pre = vf.muls(s_pre, lower, full)
    a_post = vf.muls(s_post, lower, full)
    dec_pre = vf.exp(a_pre, full)
    dec_post = vf.exp(a_post, full)
    vf.store_align(dec, dec_pre, full, [0, 0])
    vf.store_align(dec, dec_post, full, [0, VL])
    vf.store_align(rd_t, dec_pre, full, [0, 0])
    vf.store_align(rd_t, dec_post, full, [0, VL])

    dk_pre = vf.mul(dec_pre, kn_pre, full)
    dk_pre = vf.mul(dk_pre, beta_b, full)
    dk_post = vf.mul(dec_post, kn_post, full)
    dk_post = vf.mul(dk_post, beta_b, full)
    dq_pre = vf.mul(dec_pre, qn_pre, full)
    dq_post = vf.mul(dec_post, qn_post, full)

    kn_pre_h = vf.astype(kn_pre, full, dtype=FP16, layout=pl.CastLayout.ZERO)
    kn_post_h = vf.astype(kn_post, full, dtype=FP16, layout=pl.CastLayout.ZERO)
    vf.store_align(rk_t, kn_pre_h, full, [0, 0], dist=pl.StoreDist.PACK_B32)
    vf.store_align(rk_t, kn_post_h, full, [0, VL], dist=pl.StoreDist.PACK_B32)

    v_lo = vf.load_align(v_bf, [0, 0], dist=pl.LoadDist.UNPK_B16)
    v_w = vf.astype(v_lo, mask_r, dtype=FP32, layout=pl.CastLayout.ZERO)
    v_b = vf.mul(v_w, beta_b, mask_r)
    vf.store_align(vf32, v_b, mask_r, [0, 0])

    for r in pl.range(0, rows, 4):
        s0 = vf.load_align(state, [r, 0])
        s0h = vf.load_align(state, [r, VL])
        s1 = vf.load_align(state, [r + 1, 0])
        s1h = vf.load_align(state, [r + 1, VL])
        s2 = vf.load_align(state, [r + 2, 0])
        s2h = vf.load_align(state, [r + 2, VL])
        s3 = vf.load_align(state, [r + 3, 0])
        s3h = vf.load_align(state, [r + 3, VL])
        p0 = vf.mul(s0, dk_pre, full)
        p0 = vf.mul_add_dst(s0h, dk_post, full)
        p1 = vf.mul(s1, dk_pre, full)
        p1 = vf.mul_add_dst(s1h, dk_post, full)
        p2 = vf.mul(s2, dk_pre, full)
        p2 = vf.mul_add_dst(s2h, dk_post, full)
        p3 = vf.mul(s3, dk_pre, full)
        p3 = vf.mul_add_dst(s3h, dk_post, full)
        k0 = vf.reduce_sum(p0, full)
        k1 = vf.reduce_sum(p1, full)
        k2 = vf.reduce_sum(p2, full)
        k3 = vf.reduce_sum(p3, full)
        vf.store_align(skf, k0, full, [0, r], dist=pl.StoreDist.FIRST_ELEMENT_B32)
        vf.store_align(skf, k1, full, [0, r + 1], dist=pl.StoreDist.FIRST_ELEMENT_B32)
        vf.store_align(skf, k2, full, [0, r + 2], dist=pl.StoreDist.FIRST_ELEMENT_B32)
        vf.store_align(skf, k3, full, [0, r + 3], dist=pl.StoreDist.FIRST_ELEMENT_B32)
        q0 = vf.mul(s0, dq_pre, full)
        q0 = vf.mul_add_dst(s0h, dq_post, full)
        q1 = vf.mul(s1, dq_pre, full)
        q1 = vf.mul_add_dst(s1h, dq_post, full)
        q2 = vf.mul(s2, dq_pre, full)
        q2 = vf.mul_add_dst(s2h, dq_post, full)
        q3 = vf.mul(s3, dq_pre, full)
        q3 = vf.mul_add_dst(s3h, dq_post, full)
        o0 = vf.reduce_sum(q0, full)
        o1 = vf.reduce_sum(q1, full)
        o2 = vf.reduce_sum(q2, full)
        o3 = vf.reduce_sum(q3, full)
        vf.store_align(sqf, o0, full, [0, r], dist=pl.StoreDist.FIRST_ELEMENT_B32)
        vf.store_align(sqf, o1, full, [0, r + 1], dist=pl.StoreDist.FIRST_ELEMENT_B32)
        vf.store_align(sqf, o2, full, [0, r + 2], dist=pl.StoreDist.FIRST_ELEMENT_B32)
        vf.store_align(sqf, o3, full, [0, r + 3], dist=pl.StoreDist.FIRST_ELEMENT_B32)


@pl.vector_function
def _delta_out(vf32, skf, sqf, kqt, ug, outf, ru_t, outb, rows: pl.DT_INT64):
    """u = beta*V - sk (flat), o = sq + u*kq (flat); stage U/out records."""
    mask_r = vf.update_mask(rows, dtype=FP32)
    value_v = vf.load_align(vf32, [0, 0])
    sk_v = vf.load_align(skf, [0, 0])
    sq_v = vf.load_align(sqf, [0, 0])
    u = vf.sub(value_v, sk_v, mask_r)
    vf.store_align(ug, u, mask_r, [0, 0])
    kq_b = vf.load_align(kqt, [0, 0], dist=pl.LoadDist.BRC)
    o = vf.mul(u, kq_b, mask_r)
    o = vf.add(o, sq_v, mask_r)
    o_bf = vf.astype(o, mask_r, dtype=BF16, layout=pl.CastLayout.ZERO)
    vf.store_align(outb, o_bf, mask_r, [0, 0], dist=pl.StoreDist.PACK_B32)
    u_h = vf.astype(u, mask_r, dtype=FP16, layout=pl.CastLayout.ZERO)
    vf.store_align(ru_t, u_h, mask_r, [0, 0], dist=pl.StoreDist.PACK_B32)


@pl.vector_function
def _step_upd_1(state_a, state_b, kn, dec, ug, rows: pl.DT_INT64):
    """Pass 2: S_new = d * S_row + u * k_hat (one row per iteration)."""
    full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    kn_pre = vf.load_align(kn, [0, 0])
    kn_post = vf.load_align(kn, [0, VL])
    dec_pre = vf.load_align(dec, [0, 0])
    dec_post = vf.load_align(dec, [0, VL])
    for r in pl.range(rows):
        s_pre = vf.load_align(state_a, [r, 0])
        s_post = vf.load_align(state_a, [r, VL])
        u = vf.load_align(ug, [0, r], dist=pl.LoadDist.BRC)
        n_pre = vf.mul(u, kn_pre, full)
        n_pre = vf.mul_add_dst(s_pre, dec_pre, full)
        n_post = vf.mul(u, kn_post, full)
        n_post = vf.mul_add_dst(s_post, dec_post, full)
        vf.store_align(state_b, n_pre, full, [r, 0])
        vf.store_align(state_b, n_post, full, [r, VL])


@pl.vector_function
def _step_upd_2(state_a, state_b, kn, dec, ug, rows: pl.DT_INT64):
    """Pass 2 unrolled two rows per iteration."""
    full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    kn_pre = vf.load_align(kn, [0, 0])
    kn_post = vf.load_align(kn, [0, VL])
    dec_pre = vf.load_align(dec, [0, 0])
    dec_post = vf.load_align(dec, [0, VL])
    for r in pl.range(0, rows, 2):
        s0 = vf.load_align(state_a, [r, 0])
        s0h = vf.load_align(state_a, [r, VL])
        s1 = vf.load_align(state_a, [r + 1, 0])
        s1h = vf.load_align(state_a, [r + 1, VL])
        u0 = vf.load_align(ug, [0, r], dist=pl.LoadDist.BRC)
        u1 = vf.load_align(ug, [0, r + 1], dist=pl.LoadDist.BRC)
        n0 = vf.mul(u0, kn_pre, full)
        n0 = vf.mul_add_dst(s0, dec_pre, full)
        n0h = vf.mul(u0, kn_post, full)
        n0h = vf.mul_add_dst(s0h, dec_post, full)
        n1 = vf.mul(u1, kn_pre, full)
        n1 = vf.mul_add_dst(s1, dec_pre, full)
        n1h = vf.mul(u1, kn_post, full)
        n1h = vf.mul_add_dst(s1h, dec_post, full)
        vf.store_align(state_b, n0, full, [r, 0])
        vf.store_align(state_b, n0h, full, [r, VL])
        vf.store_align(state_b, n1, full, [r + 1, 0])
        vf.store_align(state_b, n1h, full, [r + 1, VL])


@pl.vector_function
def _step_upd_4(state_a, state_b, kn, dec, ug, rows: pl.DT_INT64):
    """Pass 2 unrolled four rows per iteration."""
    full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    kn_pre = vf.load_align(kn, [0, 0])
    kn_post = vf.load_align(kn, [0, VL])
    dec_pre = vf.load_align(dec, [0, 0])
    dec_post = vf.load_align(dec, [0, VL])
    for r in pl.range(0, rows, 4):
        s0 = vf.load_align(state_a, [r, 0])
        s0h = vf.load_align(state_a, [r, VL])
        s1 = vf.load_align(state_a, [r + 1, 0])
        s1h = vf.load_align(state_a, [r + 1, VL])
        s2 = vf.load_align(state_a, [r + 2, 0])
        s2h = vf.load_align(state_a, [r + 2, VL])
        s3 = vf.load_align(state_a, [r + 3, 0])
        s3h = vf.load_align(state_a, [r + 3, VL])
        u0 = vf.load_align(ug, [0, r], dist=pl.LoadDist.BRC)
        u1 = vf.load_align(ug, [0, r + 1], dist=pl.LoadDist.BRC)
        u2 = vf.load_align(ug, [0, r + 2], dist=pl.LoadDist.BRC)
        u3 = vf.load_align(ug, [0, r + 3], dist=pl.LoadDist.BRC)
        n0 = vf.mul(u0, kn_pre, full)
        n0 = vf.mul_add_dst(s0, dec_pre, full)
        n0h = vf.mul(u0, kn_post, full)
        n0h = vf.mul_add_dst(s0h, dec_post, full)
        n1 = vf.mul(u1, kn_pre, full)
        n1 = vf.mul_add_dst(s1, dec_pre, full)
        n1h = vf.mul(u1, kn_post, full)
        n1h = vf.mul_add_dst(s1h, dec_post, full)
        n2 = vf.mul(u2, kn_pre, full)
        n2 = vf.mul_add_dst(s2, dec_pre, full)
        n2h = vf.mul(u2, kn_post, full)
        n2h = vf.mul_add_dst(s2h, dec_post, full)
        n3 = vf.mul(u3, kn_pre, full)
        n3 = vf.mul_add_dst(s3, dec_pre, full)
        n3h = vf.mul(u3, kn_post, full)
        n3h = vf.mul_add_dst(s3h, dec_post, full)
        vf.store_align(state_b, n0, full, [r, 0])
        vf.store_align(state_b, n0h, full, [r, VL])
        vf.store_align(state_b, n1, full, [r + 1, 0])
        vf.store_align(state_b, n1h, full, [r + 1, VL])
        vf.store_align(state_b, n2, full, [r + 2, 0])
        vf.store_align(state_b, n2h, full, [r + 2, VL])
        vf.store_align(state_b, n3, full, [r + 3, 0])
        vf.store_align(state_b, n3h, full, [r + 3, VL])


def _prefetch_next(addr, mrow, grow, mix_t, mix_ko, mix_vo, g_t, r0,
                   mix, g, beta, q0, k0, v0, gg0, b0, q1, k1, v1, gg1, b1):
    """Load the next token's Q/K/V/G/beta into the alternate double buffer."""
    nmrow = mrow + mix_t
    ngrow = grow + g_t
    if addr % 2 == 0:
        pl.load(q1.current(), mix, [nmrow, 0])
        pl.load(k1.current(), mix, [nmrow + mix_ko, 0])
        pl.load(v1.current(), mix, [nmrow + mix_vo, r0])
        pl.load(gg1.current(), g, [ngrow, 0])
        pl.load(b1.current(), beta, [ngrow, 0])
    else:
        pl.load(q0.current(), mix, [nmrow, 0])
        pl.load(k0.current(), mix, [nmrow + mix_ko, 0])
        pl.load(v0.current(), mix, [nmrow + mix_vo, r0])
        pl.load(gg0.current(), g, [ngrow, 0])
        pl.load(b0.current(), beta, [ngrow, 0])


class ReplayKey:
    """Internal tiling keys; values are the compiled specializations."""

    Rows = TilingKeyField(bits=2, values=[16, 32, 64])
    Slots = TilingKeyField(bits=2, values=[2, 3, 4])
    Unroll = TilingKeyField(bits=2, values=[1, 2, 4])


@pl.jit(auto_mutex=True, tiling_key=ReplayKey, name="fused_recurrent_kda_replayssm")
def fused_recurrent_kda_replayssm_kernel(mix: pl.Tensor[[pl.DYNAMIC, HEAD_DIM], BF16],
           g: pl.Tensor[[pl.DYNAMIC, HEAD_DIM], BF16],
           beta: pl.Tensor[[pl.DYNAMIC, 1], BF16],
           out: pl.Tensor[[pl.DYNAMIC, HEAD_DIM], BF16],
           a_log: pl.Tensor[[pl.DYNAMIC], FP32],
           dt_bias: pl.Tensor[[pl.DYNAMIC, HEAD_DIM], FP32],
           sinit: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, HEAD_DIM, HEAD_DIM], FP32],
           ru: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, HEAD_DIM], FP16],
           rk: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, HEAD_DIM], FP16],
           rd: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, HEAD_DIM], FP32],
           cu: pl.Tensor[[pl.DYNAMIC], I32],
           scale: pl.DT_FP32, lower: pl.DT_FP32,
           heads: pl.DT_INT64, seq: pl.DT_INT64, batch: pl.DT_INT64,
           mix_b: pl.DT_INT64, mix_n: pl.DT_INT64, mix_t: pl.DT_INT64,
           mix_ko: pl.DT_INT64, mix_vo: pl.DT_INT64,
           g_b: pl.DT_INT64, g_n: pl.DT_INT64, g_t: pl.DT_INT64,
           packed: pl.DT_INT64):
    row_blocks = HEAD_DIM // Rows
    state_bytes = Rows * HEAD_DIM * 4
    va_state = 0
    va_misc = va_state + Slots * state_bytes
    va_dtb = va_misc + 12 * 512
    va_bf = va_dtb + 512
    va_out = va_bf + 3072

    with pl.section_vector():
        state_g = pl.make_tile_group(
            type=pl.TileType(shape=[Rows, HEAD_DIM], dtype=FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_state + i * state_bytes for i in range(Slots)],
            mutex_ids=[0, 1, 2, 3][:Slots])
        dtb_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, HEAD_DIM], dtype=FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_dtb], mutex_ids=[30])
        q0 = pl.make_tile_group(
            type=pl.TileType(shape=[1, HEAD_DIM], dtype=BF16,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_bf], mutex_ids=None, depth=1)
        k0 = pl.make_tile_group(
            type=pl.TileType(shape=[1, HEAD_DIM], dtype=BF16,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_bf + 512], mutex_ids=None, depth=1)
        gg0 = pl.make_tile_group(
            type=pl.TileType(shape=[1, HEAD_DIM], dtype=BF16,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_bf + 1024], mutex_ids=None, depth=1)
        v0 = pl.make_tile_group(
            type=pl.TileType(shape=[1, Rows], dtype=BF16, valid_shape=[1, Rows],
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_bf + 1536], mutex_ids=None, depth=1)
        b0 = pl.make_tile_group(
            type=pl.TileType(shape=[1, VL], dtype=BF16, valid_shape=[1, 1],
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_bf + 2048], mutex_ids=None, depth=1)
        q1 = pl.make_tile_group(
            type=pl.TileType(shape=[1, HEAD_DIM], dtype=BF16,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_bf + 256], mutex_ids=None, depth=1)
        k1 = pl.make_tile_group(
            type=pl.TileType(shape=[1, HEAD_DIM], dtype=BF16,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_bf + 768], mutex_ids=None, depth=1)
        gg1 = pl.make_tile_group(
            type=pl.TileType(shape=[1, HEAD_DIM], dtype=BF16,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_bf + 1280], mutex_ids=None, depth=1)
        v1 = pl.make_tile_group(
            type=pl.TileType(shape=[1, Rows], dtype=BF16, valid_shape=[1, Rows],
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_bf + 1536 + Rows * 2], mutex_ids=None, depth=1)
        b1 = pl.make_tile_group(
            type=pl.TileType(shape=[1, VL], dtype=BF16, valid_shape=[1, 1],
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_bf + 2176], mutex_ids=None, depth=1)
        qn_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, HEAD_DIM], dtype=FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_misc], mutex_ids=[14])
        kn_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, HEAD_DIM], dtype=FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_misc + 512], mutex_ids=[15])
        dec_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, HEAD_DIM], dtype=FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_misc + 1024], mutex_ids=[16])
        dk_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, HEAD_DIM], dtype=FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_misc + 1536], mutex_ids=[17])
        dq_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, HEAD_DIM], dtype=FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_misc + 2048], mutex_ids=[18])
        vf_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, HEAD_DIM], dtype=FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_misc + 2560], mutex_ids=[19])
        kqt_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, VL], dtype=FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_misc + 3072], mutex_ids=[20])
        outf_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, HEAD_DIM], dtype=FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_misc + 3328], mutex_ids=[21])
        sk_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, HEAD_DIM], dtype=FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_misc + 3840], mutex_ids=[22])
        sq_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, HEAD_DIM], dtype=FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_misc + 4352], mutex_ids=[23])
        ug_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, HEAD_DIM], dtype=FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_misc + 4864], mutex_ids=[24])
        outb_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, Rows], dtype=BF16, valid_shape=[-1, -1],
                             target_memory=pl.MemorySpace.Vec),
            addrs=[va_out + i * Rows * 2 for i in range(3)],
            mutex_ids=[25, 26, 4][:3])
        ru_base = va_out + 6 * Rows
        ru_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, Rows], dtype=FP16, valid_shape=[-1, -1],
                             target_memory=pl.MemorySpace.Vec),
            addrs=[ru_base + i * Rows * 2 for i in range(3)],
            mutex_ids=[27, 28, 5][:3])
        rk_base = ru_base + 6 * Rows
        rk_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, HEAD_DIM], dtype=FP16,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[rk_base, rk_base + 256], mutex_ids=[29, 6])
        rd_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, HEAD_DIM], dtype=FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[rk_base + 512, rk_base + 1024], mutex_ids=[31, 7])

        items_per_batch = heads * row_blocks
        total_items = batch * items_per_batch
        for item in pl.range(pl.get_block_idx(), total_items, pl.get_block_num()):
            bb = item // items_per_batch
            rem = item % items_per_batch
            hh = rem // row_blocks
            rblk = rem % row_blocks
            r0 = rblk * Rows
            tok_lo = bb * seq
            tok_hi = tok_lo + seq
            if packed == 1:
                tok_lo = pl.getval(cu, bb)
                tok_hi = pl.getval(cu, bb + 1)
            state = state_g.next()
            pl.load(state, sinit, [bb, hh, r0, 0])
            dtb_t = dtb_g.next()
            pl.load(dtb_t, dt_bias, [hh, 0])
            alog = pl.getval(a_log, hh)
            mix_base = bb * mix_b + hh * mix_n + tok_lo * mix_t * packed
            g_base = bb * g_b + hh * g_n + tok_lo * g_t * packed
            pl.set_validshape(outb_g.next(), [1, Rows])
            pl.set_validshape(ru_g.next(), [1, Rows])
            pl.load(q0.current(), mix, [mix_base, 0])
            pl.load(k0.current(), mix, [mix_base + mix_ko, 0])
            pl.load(v0.current(), mix, [mix_base + mix_vo, r0])
            pl.load(gg0.current(), g, [g_base, 0])
            pl.load(b0.current(), beta, [g_base, 0])
            pl.system.bar_mte2()
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
            for t in pl.range(tok_lo, tok_hi):
                addr = t - tok_lo
                mrow = mix_base + addr * mix_t
                grow = g_base + addr * g_t
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
                if t + 1 < tok_hi:
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    _prefetch_next(addr, mrow, grow, mix_t, mix_ko, mix_vo,
                                   g_t, r0, mix, g, beta, q0, k0, v0, gg0,
                                   b0, q1, k1, v1, gg1, b1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
                if addr % 2 == 0:
                    _prep_dot(state, q0.current(), k0.current(), gg0.current(),
                              v0.current(), b0.current(), dtb_t,
                              kn_g.next(), dec_g.next(), kqt_g.next(),
                              vf_g.next(), rk_g.next(), rd_g.next(),
                              sk_g.next(), sq_g.next(),
                              scale, lower, alog, Rows)
                else:
                    _prep_dot(state, q1.current(), k1.current(), gg1.current(),
                              v1.current(), b1.current(), dtb_t,
                              kn_g.next(), dec_g.next(), kqt_g.next(),
                              vf_g.next(), rk_g.next(), rd_g.next(),
                              sk_g.next(), sq_g.next(),
                              scale, lower, alog, Rows)
                _delta_out(vf_g.current(), sk_g.current(), sq_g.current(),
                           kqt_g.current(), ug_g.next(), outf_g.next(),
                           ru_g.next(), outb_g.next(), Rows)
                nxt = state_g.next()
                if Unroll == 1:
                    _step_upd_1(state, nxt, kn_g.current(), dec_g.current(),
                                ug_g.current(), Rows)
                if Unroll == 2:
                    _step_upd_2(state, nxt, kn_g.current(), dec_g.current(),
                                ug_g.current(), Rows)
                if Unroll == 4:
                    _step_upd_4(state, nxt, kn_g.current(), dec_g.current(),
                                ug_g.current(), Rows)
                pl.store(out, outb_g.current(), [grow, r0])
                if rblk == 0:
                    pl.store(rk, rk_g.current(), [bb, addr, hh, 0])
                    pl.store(rd, rd_g.current(), [bb, addr, hh, 0])
                pl.store(ru, ru_g.current(), [bb, addr, hh, r0])
                state = nxt



_DEFAULTS: dict[tuple, torch.Tensor] = {}


def _default_tensor(key, factory):
    tensor = _DEFAULTS.get(key)
    if tensor is None:
        tensor = factory()
        _DEFAULTS[key] = tensor
    return tensor


def _route(rows, slots, unroll):
    for name, value, values in (("rows", rows, ReplayKey.Rows.values),
                                ("slots", slots, ReplayKey.Slots.values),
                                ("unroll", unroll, ReplayKey.Unroll.values)):
        _require(value in values, f"{name}={value} must be one of {values}")
    return {"Rows": rows, "Slots": slots, "Unroll": unroll}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


LAYOUTS = ("BSND", "BNSD", "TND")


def _check_state(state):
    _require(state.dim() == 4, "state must be rank-4")
    _require(state.dtype == torch.float32, "state must be fp32")
    _require(state.is_contiguous(), "state must be contiguous")
    batch, heads, dv, dk = map(int, state.shape)
    _require(dv == dk == HEAD_DIM, "state trailing shape must be 128x128")
    _require(1 <= heads <= MAX_HEADS, f"heads must be in [1, {MAX_HEADS}]")
    return batch, heads


def _check_inputs(mixedqkv, g, beta, a_log, dt_bias, heads, lower):
    for tensor, want, name in ((mixedqkv, torch.bfloat16, "mixedqkv"),
                               (g, torch.bfloat16, "g"),
                               (beta, torch.bfloat16, "beta"),
                               (a_log, torch.float32, "a_log"),
                               (dt_bias, torch.float32, "dt_bias")):
        _require(tensor.dtype == want, f"{name} must be {want}, got {tensor.dtype}")
    _require(tuple(a_log.shape) == (heads,), "a_log must be [heads]")
    _require(tuple(dt_bias.shape) == (heads, HEAD_DIM), "dt_bias must be [heads, 128]")
    _require(-5.0 <= float(lower) <= 0.0, "lower_bound must be in [-5, 0]")


def _bsnd_views(mixedqkv, g, beta, heads, state):
    b, seq, width = map(int, mixedqkv.shape)
    _require(width == 3 * heads * HEAD_DIM, "BSND mixedqkv last dim must be 3*heads*128")
    _require(tuple(g.shape) == (b, seq, heads, HEAD_DIM), "g shape")
    _require(tuple(beta.shape) == (b, seq, heads, 1), "beta shape")
    out = torch.empty(b, seq, heads, HEAD_DIM, dtype=torch.bfloat16,
                      device=state.device)
    return dict(mix=mixedqkv.view(b * seq * 3 * heads, HEAD_DIM),
                gate=g.view(b * seq * heads, HEAD_DIM),
                beta=beta.view(b * seq * heads, 1),
                out=out, out_v=out.view(b * seq * heads, HEAD_DIM),
                batch=b, seq=seq, token_count=b * seq,
                strides=(seq * 3 * heads, 1, 3 * heads, heads, 2 * heads,
                         seq * heads, 1, heads, 0))


def _bnsd_views(mixedqkv, g, beta, heads, state):
    b, packed_heads, seq, head_dim = map(int, mixedqkv.shape)
    _require(head_dim == HEAD_DIM, "BNSD last dim must be 128")
    _require(packed_heads == 3 * heads, "BNSD packed heads must be 3*heads")
    _require(tuple(g.shape) == (b, heads, seq, HEAD_DIM), "g shape")
    _require(tuple(beta.shape) == (b, heads, seq, 1), "beta shape")
    out = torch.empty(b, heads, seq, HEAD_DIM, dtype=torch.bfloat16,
                      device=state.device)
    return dict(mix=mixedqkv.reshape(b * packed_heads * seq, HEAD_DIM),
                gate=g.reshape(b * heads * seq, HEAD_DIM),
                beta=beta.reshape(b * heads * seq, 1),
                out=out, out_v=out.view(b * heads * seq, HEAD_DIM),
                batch=b, seq=seq, token_count=b * seq,
                strides=(3 * heads * seq, seq, 1, heads * seq, 2 * heads * seq,
                         heads * seq, seq, 1, 0))


def _tnd_views(mixedqkv, g, beta, heads, state, cu_seqlens):
    token_count, width = map(int, mixedqkv.shape)
    _require(width == 3 * heads * HEAD_DIM, "TND mixedqkv last dim must be 3*heads*128")
    _require(tuple(g.shape) == (token_count, heads, HEAD_DIM), "g shape")
    _require(tuple(beta.shape) == (token_count, heads, 1), "beta shape")
    _require(cu_seqlens is not None, "cu_seqlens is required for TND")
    b = int(cu_seqlens.numel()) - 1
    out = torch.empty(token_count, heads, HEAD_DIM, dtype=torch.bfloat16,
                      device=state.device)
    return dict(mix=mixedqkv.view(token_count * 3 * heads, HEAD_DIM),
                gate=g.view(token_count * heads, HEAD_DIM),
                beta=beta.view(token_count * heads, 1),
                out=out, out_v=out.view(token_count * heads, HEAD_DIM),
                batch=b, seq=0, token_count=token_count,
                strides=(0, 1, 3 * heads, heads, 2 * heads, 0, 1, heads, 1))


def _prepare(mixedqkv, state, beta, g, a_log, dt_bias, lower, layout, cu_seqlens):
    _batch, heads = _check_state(state)
    _check_inputs(mixedqkv, g, beta, a_log, dt_bias, heads, lower)
    if layout == "BSND":
        prep = _bsnd_views(mixedqkv, g, beta, heads, state)
    elif layout == "BNSD":
        prep = _bnsd_views(mixedqkv, g, beta, heads, state)
    elif layout == "TND":
        prep = _tnd_views(mixedqkv, g, beta, heads, state, cu_seqlens)
    else:
        raise ValueError(f"layout_qkv must be one of {LAYOUTS}, got {layout}")
    for tensor, name in ((mixedqkv, "mixedqkv"), (prep["gate"], "g"),
                         (prep["beta"], "beta")):
        _require(tensor.is_contiguous(), f"{name} view must be contiguous")
    if cu_seqlens is not None:
        _require(cu_seqlens.dtype == torch.int32 and cu_seqlens.is_contiguous(),
                 "cu_seqlens must be contiguous int32")
    prep.update(heads=heads, a_log=a_log, dt_bias=dt_bias)
    return prep


def _choose_rows(batch, heads, workers):
    """Pick a row block from the measured work-size thresholds.

    Large batch*head counts keep 64-row units (balanced across the 64 vector
    workers); small ones halve the unit so enough units exist to fill cores.
    """
    total_heads = max(int(batch) * int(heads), 1)
    if total_heads >= 24:
        return 64
    if total_heads >= 6:
        return 32
    return 16


def _workers_for():
    from pypto_pro.runtime.platform import get_platform_info
    try:
        vector_cores = int(get_platform_info().vector_core_num)
    except (AttributeError, RuntimeError, TypeError, ValueError):  # platform query fallback
        vector_cores = 64
    if vector_cores <= 0:
        vector_cores = 64
    return min(vector_cores, 64)


def fused_recurrent_kda_functional(mixedqkv, state, beta, g, scale, a_log, dt_bias,
                                   lower_bound=-5.0, layout_qkv="BSND", *, cu_seqlens=None,
                                   replay_u, replay_k, replay_decay,
                                   rows=None, workers=None,
                                   slots=STATE_SLOTS, unroll=2):
    _require(layout_qkv in LAYOUTS, "layout_qkv must be BSND, BNSD or TND")
    _require(tuple(replay_u.shape) == (int(state.shape[0]), int(replay_u.shape[1]),
                                       int(state.shape[1]), HEAD_DIM), "replay_u shape")
    prep = _prepare(mixedqkv, state, beta, g, a_log, dt_bias, lower_bound,
                    layout_qkv, cu_seqlens)
    batch = prep["batch"]
    heads = prep["heads"]
    capacity = int(replay_u.shape[1])
    for name, rec, dt in (("replay_u", replay_u, torch.float16),
                          ("replay_k", replay_k, torch.float16),
                          ("replay_decay", replay_decay, torch.float32)):
        _require(rec.dtype == dt, f"{name} must be {dt}")
        _require(rec.is_contiguous(), f"{name} must be contiguous")
        _require(tuple(rec.shape) == (batch, capacity, heads, HEAD_DIM),
                 f"{name} shape must match state/records")
    cu = cu_seqlens
    if cu is None:
        cu = _default_tensor(
            ("cu", str(state.device)),
            lambda: torch.zeros(batch + 1, dtype=torch.int32, device=state.device))
    ca, cb, ct, cko, cvo, cgb, cgn, cgt, packed = prep["strides"]
    if workers is None:
        workers = _workers_for()
    if rows is None:
        rows = _choose_rows(batch, heads, workers)
    total_items = batch * heads * (HEAD_DIM // rows)
    _require(total_items > 0, "no work items")
    workers = min(int(workers), total_items)
    scale_value = float(scale) if scale is not None else HEAD_DIM ** -0.5
    route = _route(rows, slots, unroll)
    fused_recurrent_kda_replayssm_kernel[None, workers, route](
        prep["mix"], prep["gate"], prep["beta"], prep["out_v"],
        a_log, dt_bias, state, replay_u, replay_k, replay_decay, cu,
        scale_value, float(lower_bound),
        heads, prep["seq"], batch,
        ca, cb, ct, cko, cvo, cgb, cgn, cgt, packed,
    )
    return prep["out"]


def fused_recurrent_kda(mixedqkv, state, beta, g, scale, a_log, dt_bias,
                        lower_bound=-5.0, layout_qkv="BSND", *, replay_u, replay_k,
                        replay_decay, cu_seqlens=None, rows=None,
                        workers=None, slots=STATE_SLOTS, unroll=2):
    """Run the fused recurrent KDA ReplaySSM verify operator."""
    return fused_recurrent_kda_functional(
        mixedqkv, state, beta, g, scale, a_log, dt_bias, lower_bound, layout_qkv,
        cu_seqlens=cu_seqlens, replay_u=replay_u, replay_k=replay_k,
        replay_decay=replay_decay, rows=rows, workers=workers, slots=slots,
        unroll=unroll)
