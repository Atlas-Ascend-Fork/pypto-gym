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

"""PyPTO-Pro implementation of `fused_recurrent_kda_snapshot` (Ascend 950PR).

Contract: one packed `mixedqkv` in `[Q(P) | K(P) | V(P)]` order, `P = heads * 128`; per
(batch, head) the operator normalizes Q/K rows, activates gate/beta, runs the
KDA token recurrence on a `[128, 128]` state and writes every token's snapshot
back into the state pool.  BSND / BNSD / TND storage layouts, optional
`ssm_state_indices` and `num_accepted_tokens` selection are supported.

Design notes
------------
* Pure Vector kernel.  One work item covers `rows` (32/64) state rows of one
  (batch, head); rows are independent in the recurrence.  Items are assigned
  block-cyclically; the default worker count is the device vector-core count
  capped at 64 (the stream budget), so the 192 units of the benchmark shape
  spread evenly 3 per worker.
* The working state lives in a resident 3-slot UB tile group.  Token `t` reads
  slot `t % 3` and writes slot `(t+1) % 3`.  The live slots are touched only by
  the VF recurrence — the snapshot DMA no longer reads them (see below).
* Per token the recurrence is split into separate VF passes instead of one fused
  per-row loop, so a row's `reduce_sum` no longer feeds its own rank-1 update in
  the same iteration (that serial chain was the main vector-pipe stall):
  - `_prep` — normalized q/k, gate decay, folded weights `beta*decay*key` and
    `beta*V`;
  - `_step_dot` — per row `sk[r] = dot(S_row_r, beta*decay*key)`, unrolled 4 rows
    at a time to hide the reduce latency, results written to UB;
  - `_fold_u` — one flat op `u = beta*V - sk` over the owned rows;
  - `_step_upd_snap` — per row the rank-1 update
    `S_new = S*decay + u (x) key`, the output dot `o = dot(S_new, q)` and the
    decoupled snapshot store;
  - `_pack_out` — FP32 -> BF16 output with a PACK_B32 dense store.
* Decoupled snapshot: each token's `S_new` is written to a dedicated
  double-buffered snapshot tile (FP32 pool: directly inside `_step_upd_snap`;
  BF16 pool: narrowed by `_narrow_state`) and DMA'd immediately to
  `state[ssm_state_indices[t]]`.  The MTE3 snapshot therefore never orders
  against the VF recurrence through the live-slot mutex.
* UB byte plan (rows=64, slots=3): state 3*32 KiB = 96 KiB, snapshot/staging
  64 KiB, misc < 16 KiB — inside the 256 KiB Vec capacity, no Cube resources.
* Measured with the torch_npu profiler on Ascend950PR, B=16,N=6,S=8,D=128,
  BSND, FP32 state, accepted=4, 64 AIV: median 17.27-17.46 us
  (min 16.68-16.76 us).  The three levers were: workers 56 -> 64 (load
  balance), the fused per-row body -> split passes with a 4-row dot unroll
  (vector efficiency), and the delayed-store snapshot -> a decoupled snapshot
  tile (VF/MTE3 decouple).

CodeCheck exceptions
--------------------
* G.FNM.03: PyPTO kernel/host-wrapper tensor parameters must follow the positional
  ABI; PyPTO JIT cannot carry torch.Tensor fields in a dataclass, so the signature
  stays positional.
* Large function / depth: the section, token-loop and row-loop structure mirrors
  the hardware pipeline; splitting it further would need a re-validation on
  Ascend 950.
"""

from __future__ import annotations

import torch

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

FP32 = pl.DT_FP32
BF16 = pl.DT_BF16
I32 = pl.DT_INT32

HEAD_DIM = 128
MAX_HEADS = 96
SUPPORTED_ROW_ITEMS = (32, 64)
DEFAULT_ROWS = 64
STATE_SLOTS = 3
_EPS = 1e-6


@pl.vector_function
def _prep(q_bf, k_bf, g_bf, v_bf, beta_bf, dtb, qn, kn, dec, kd, vf32,
          scale: pl.DT_FP32, lower: pl.DT_FP32, alog: pl.DT_FP32, rows: pl.DT_INT64):
    """Build one token's per-head constants.

    Folded form: kd = beta * decay * key and vf32 = beta * V, so the recurrence
    pass computes u = (beta*V) - rowdot(S, beta*decay*key) with one fewer
    per-row instruction; beta multiplies exact operands instead of the residual.
    """
    full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    mask_r = vf.update_mask(rows, dtype=FP32)
    lane = vf.create_mask(pattern=pl.MaskPattern.VL1, dtype=FP32)
    one = vf.full(1.0, full, dtype=FP32)

    # --- beta activation (single scalar) ---
    bp = vf.load_align(beta_bf, [0, 0], dist=pl.LoadDist.UNPK_B16)
    bv = vf.astype(bp, full, dtype=FP32, layout=pl.CastLayout.ZERO)
    bn = vf.neg(bv, full)
    be = vf.exp(bn, full)
    bd = vf.adds(be, 1.0, full)
    beta_s = vf.div(one, bd, full)
    beta_b = vf.full(beta_s, full)

    # --- normalized, scaled query ---
    q_lo = vf.load_align(q_bf, [0, 0], dist=pl.LoadDist.UNPK_B16)
    q_hi = vf.load_align(q_bf, [0, 64], dist=pl.LoadDist.UNPK_B16)
    q_pre = vf.astype(q_lo, full, dtype=FP32, layout=pl.CastLayout.ZERO)
    q_post = vf.astype(q_hi, full, dtype=FP32, layout=pl.CastLayout.ZERO)
    sq = vf.mul(q_pre, q_pre, full)
    sq = vf.mul_add_dst(q_post, q_post, full)
    q_sum = vf.reduce_sum(sq, full)
    q_eps = vf.adds(q_sum, _EPS, full)
    q_root = vf.sqrt(q_eps, full)
    q_inv = vf.div(one, q_root, full)
    q_inv_b = vf.full(q_inv, full)
    qn_pre = vf.mul(q_pre, q_inv_b, full)
    qn_pre = vf.muls(qn_pre, scale, full)
    qn_post = vf.mul(q_post, q_inv_b, full)
    qn_post = vf.muls(qn_post, scale, full)
    vf.store_align(qn, qn_pre, full, [0, 0])
    vf.store_align(qn, qn_post, full, [0, 64])

    # --- normalized key ---
    k_lo = vf.load_align(k_bf, [0, 0], dist=pl.LoadDist.UNPK_B16)
    k_hi = vf.load_align(k_bf, [0, 64], dist=pl.LoadDist.UNPK_B16)
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
    vf.store_align(kn, kn_post, full, [0, 64])

    # --- gate activation and decay ---
    g_lo = vf.load_align(g_bf, [0, 0], dist=pl.LoadDist.UNPK_B16)
    g_hi = vf.load_align(g_bf, [0, 64], dist=pl.LoadDist.UNPK_B16)
    g_pre = vf.astype(g_lo, full, dtype=FP32, layout=pl.CastLayout.ZERO)
    g_post = vf.astype(g_hi, full, dtype=FP32, layout=pl.CastLayout.ZERO)
    b_pre = vf.load_align(dtb, [0, 0])
    b_post = vf.load_align(dtb, [0, 64])
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
    vf.store_align(dec, dec_post, full, [0, 64])

    # --- beta * decay * key (folded weight of the key dot) ---
    kd_pre = vf.mul(dec_pre, kn_pre, full)
    kd_pre = vf.mul(kd_pre, beta_b, full)
    kd_post = vf.mul(dec_post, kn_post, full)
    kd_post = vf.mul(kd_post, beta_b, full)
    vf.store_align(kd, kd_pre, full, [0, 0])
    vf.store_align(kd, kd_post, full, [0, 64])

    # --- beta * V for the owned rows ---
    v_lo = vf.load_align(v_bf, [0, 0], dist=pl.LoadDist.UNPK_B16)
    v_w = vf.astype(v_lo, mask_r, dtype=FP32, layout=pl.CastLayout.ZERO)
    v_b = vf.mul(v_w, beta_b, mask_r)
    vf.store_align(vf32, v_b, mask_r, [0, 0])


@pl.vector_function
def _step_dot(state_a, kd, skf, rows: pl.DT_INT64):
    """Pass 1 — key dot: sk[r] = dot(S_row_r, beta*decay*key), 4 rows in flight.

    Split out of the old fused ``_step`` so the per-row reduce no longer feeds the
    rank-1 update in the same iteration; the 4-row unroll hides the reduce latency
    (a 4-row unroll hides the reduce latency).
    """
    full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    kd_pre = vf.load_align(kd, [0, 0])
    kd_post = vf.load_align(kd, [0, 64])
    for r in pl.range(0, rows, 4):
        s0 = vf.load_align(state_a, [r, 0])
        s0h = vf.load_align(state_a, [r, 64])
        s1 = vf.load_align(state_a, [r + 1, 0])
        s1h = vf.load_align(state_a, [r + 1, 64])
        s2 = vf.load_align(state_a, [r + 2, 0])
        s2h = vf.load_align(state_a, [r + 2, 64])
        s3 = vf.load_align(state_a, [r + 3, 0])
        s3h = vf.load_align(state_a, [r + 3, 64])
        p0 = vf.mul(s0, kd_pre, full)
        p0 = vf.mul_add_dst(s0h, kd_post, full)
        p1 = vf.mul(s1, kd_pre, full)
        p1 = vf.mul_add_dst(s1h, kd_post, full)
        p2 = vf.mul(s2, kd_pre, full)
        p2 = vf.mul_add_dst(s2h, kd_post, full)
        p3 = vf.mul(s3, kd_pre, full)
        p3 = vf.mul_add_dst(s3h, kd_post, full)
        k0 = vf.reduce_sum(p0, full)
        k1 = vf.reduce_sum(p1, full)
        k2 = vf.reduce_sum(p2, full)
        k3 = vf.reduce_sum(p3, full)
        vf.store_align(skf, k0, full, [0, r], dist=pl.StoreDist.FIRST_ELEMENT_B32)
        vf.store_align(skf, k1, full, [0, r + 1], dist=pl.StoreDist.FIRST_ELEMENT_B32)
        vf.store_align(skf, k2, full, [0, r + 2], dist=pl.StoreDist.FIRST_ELEMENT_B32)
        vf.store_align(skf, k3, full, [0, r + 3], dist=pl.StoreDist.FIRST_ELEMENT_B32)


@pl.vector_function
def _fold_u(vf32, skf, ug, rows: pl.DT_INT64):
    """Pass 2 — flat delta: u = beta*V - sk over the owned rows (one vector op)."""
    mask_r = vf.update_mask(rows, dtype=FP32)
    value_v = vf.load_align(vf32, [0, 0])
    sk_v = vf.load_align(skf, [0, 0])
    u = vf.sub(value_v, sk_v, mask_r)
    vf.store_align(ug, u, mask_r, [0, 0])


@pl.vector_function
def _step_upd_snap(state_a, state_b, kn, dec, qn, ug, outf, snap, write_snap,
                   rows: pl.DT_INT64):
    """Pass 3 — rank-1 update, output dot and (optional) snapshot store.

    `write_snap` is a compile-time flag: the FP32 pool passes True and `snap` is
    the dedicated decoupled snapshot tile; the BF16 pool passes False and the
    real snapshot is narrowed from the live slot afterwards.  The FP32 snapshot
    DMA reads its own tile instead of the rotating live state slot, so the VF
    recurrence never orders against MTE3 through the slot mutex.
    """
    full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    lane = vf.create_mask(pattern=pl.MaskPattern.VL1, dtype=FP32)
    kn_pre = vf.load_align(kn, [0, 0])
    kn_post = vf.load_align(kn, [0, 64])
    dec_pre = vf.load_align(dec, [0, 0])
    dec_post = vf.load_align(dec, [0, 64])
    qn_pre = vf.load_align(qn, [0, 0])
    qn_post = vf.load_align(qn, [0, 64])
    for r in pl.range(rows):
        s_pre = vf.load_align(state_a, [r, 0])
        s_post = vf.load_align(state_a, [r, 64])
        u = vf.load_align(ug, [0, r], dist=pl.LoadDist.BRC)
        n_pre = vf.mul(u, kn_pre, full)
        n_pre = vf.mul_add_dst(s_pre, dec_pre, full)
        n_post = vf.mul(u, kn_post, full)
        n_post = vf.mul_add_dst(s_post, dec_post, full)
        vf.store_align(state_b, n_pre, full, [r, 0])
        vf.store_align(state_b, n_post, full, [r, 64])
        if write_snap:
            vf.store_align(snap, n_pre, full, [r, 0])
            vf.store_align(snap, n_post, full, [r, 64])
        o_part = vf.mul(n_pre, qn_pre, full)
        o_part = vf.mul_add_dst(n_post, qn_post, full)
        o = vf.reduce_sum(o_part, full)
        vf.store_align(outf, o, lane, [0, r], dist=pl.StoreDist.FIRST_ELEMENT_B32)


@pl.vector_function
def _pack_out(outf, outb, rows: pl.DT_INT64):
    mask_r = vf.update_mask(rows, dtype=FP32)
    acc = vf.load_align(outf, [0, 0])
    narrow = vf.astype(acc, mask_r, dtype=BF16, layout=pl.CastLayout.ZERO)
    vf.store_align(outb, narrow, mask_r, [0, 0], dist=pl.StoreDist.PACK_B32)


@pl.vector_function
def _widen_state(src_bf, dst_f32, rows: pl.DT_INT64):
    """BF16 pool -> FP32 working state (order preserving, once per item)."""
    full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for r in pl.range(rows):
        lo = vf.load_align(src_bf, [r, 0], dist=pl.LoadDist.UNPK_B16)
        hi = vf.load_align(src_bf, [r, 64], dist=pl.LoadDist.UNPK_B16)
        wlo = vf.astype(lo, full, dtype=FP32, layout=pl.CastLayout.ZERO)
        whi = vf.astype(hi, full, dtype=FP32, layout=pl.CastLayout.ZERO)
        vf.store_align(dst_f32, wlo, full, [r, 0])
        vf.store_align(dst_f32, whi, full, [r, 64])


@pl.vector_function
def _narrow_state(src_f32, dst_bf, rows: pl.DT_INT64):
    """FP32 working state -> BF16 snapshot staging (PACK_B32 dense store)."""
    full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for r in pl.range(rows):
        lo = vf.load_align(src_f32, [r, 0])
        hi = vf.load_align(src_f32, [r, 64])
        nlo = vf.astype(lo, full, dtype=BF16, layout=pl.CastLayout.ZERO)
        nhi = vf.astype(hi, full, dtype=BF16, layout=pl.CastLayout.ZERO)
        vf.store_align(dst_bf, nlo, full, [r, 0], dist=pl.StoreDist.PACK_B32)
        vf.store_align(dst_bf, nhi, full, [r, 64], dist=pl.StoreDist.PACK_B32)


def _make_kernel(rows: int, slots: int = STATE_SLOTS, state_bf16: bool = False):
    if rows not in SUPPORTED_ROW_ITEMS:
        raise ValueError(f"rows must be one of {SUPPORTED_ROW_ITEMS}")
    if not 2 <= slots <= 5:
        raise ValueError("slots must be in [2, 5]")
    row_blocks = HEAD_DIM // rows
    state_bytes = rows * HEAD_DIM * 4  # working state is always FP32 in UB
    va_state = 0
    va_stg = va_state + slots * rows * HEAD_DIM * 4
    # Staging region holds either the BF16 load+snapshot tiles (3 x rows*256) or the
    # FP32 decoupled-snapshot tile (depth 2, 2 x rows*512); mutually exclusive per
    # compiled kernel, so reserve the larger of the two.
    stg_bytes = max(rows * HEAD_DIM * 2 * 3, rows * HEAD_DIM * 4 * 2)
    va_dtb = va_stg + stg_bytes
    va_misc = va_dtb + 1024
    va_bf = va_misc + 6 * 1024
    va_out = va_bf + 2048

    suffix = "_bf" if state_bf16 else ""
    state_dt = BF16 if state_bf16 else FP32

    @pl.jit(auto_mutex=True, name=f"kda_snapshot_r{rows}{suffix}")
    def kernel(mix: pl.Tensor[[pl.DYNAMIC, HEAD_DIM], BF16],
               g: pl.Tensor[[pl.DYNAMIC, HEAD_DIM], BF16],
               beta: pl.Tensor[[pl.DYNAMIC, 1], BF16],
               out: pl.Tensor[[pl.DYNAMIC, HEAD_DIM], BF16],
               a_log: pl.Tensor[[pl.DYNAMIC], FP32],
               dt_bias: pl.Tensor[[pl.DYNAMIC, HEAD_DIM], FP32],
               sinit: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, HEAD_DIM, HEAD_DIM], state_dt],
               sout: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, HEAD_DIM, HEAD_DIM], state_dt],
               state_idx: pl.Tensor[[pl.DYNAMIC], I32],
               accepted: pl.Tensor[[pl.DYNAMIC], I32],
               cu: pl.Tensor[[pl.DYNAMIC], I32],
               scale: pl.DT_FP32, lower: pl.DT_FP32,
               heads: pl.DT_INT64, seq: pl.DT_INT64, batch: pl.DT_INT64,
               mix_b: pl.DT_INT64, mix_n: pl.DT_INT64, mix_t: pl.DT_INT64,
               mix_ko: pl.DT_INT64, mix_vo: pl.DT_INT64,
               g_b: pl.DT_INT64, g_n: pl.DT_INT64, g_t: pl.DT_INT64,
               packed: pl.DT_INT64):
        with pl.section_vector():
            state_g = pl.make_tile_group(
                type=pl.TileType(shape=[rows, HEAD_DIM], dtype=FP32,
                                 target_memory=pl.MemorySpace.Vec),
                addrs=[va_state + i * state_bytes for i in range(slots)],
                mutex_ids=[0, 1, 2, 3, 4][:slots])
            if state_bf16:
                ld_g = pl.make_tile_group(
                    type=pl.TileType(shape=[rows, HEAD_DIM], dtype=BF16,
                                     target_memory=pl.MemorySpace.Vec),
                    addrs=[va_stg], mutex_ids=[20])
                snap_g = pl.make_tile_group(
                    type=pl.TileType(shape=[rows, HEAD_DIM], dtype=BF16,
                                     target_memory=pl.MemorySpace.Vec),
                    addrs=[va_stg + rows * HEAD_DIM * 2,
                           va_stg + 2 * rows * HEAD_DIM * 2], mutex_ids=[21, 22])
                snf_g = None
            else:
                ld_g = None
                snap_g = None
                # FP32 decoupled-snapshot tile (depth 2): _step_upd_snap writes the new
                # state here and the DMA reads it, so the live state slots stay VF-only.
                snf_g = pl.make_tile_group(
                    type=pl.TileType(shape=[rows, HEAD_DIM], dtype=FP32,
                                     target_memory=pl.MemorySpace.Vec),
                    addrs=[va_stg, va_stg + rows * HEAD_DIM * 4], mutex_ids=[21, 22])
            dtb_g = pl.make_tile_group(
                type=pl.TileType(shape=[1, HEAD_DIM], dtype=FP32,
                                 target_memory=pl.MemorySpace.Vec),
                addrs=[va_dtb], mutex_ids=[30])
            q_g = pl.make_tile_group(
                type=pl.TileType(shape=[1, HEAD_DIM], dtype=BF16,
                                 target_memory=pl.MemorySpace.Vec),
                addrs=[va_bf], mutex_ids=[4])
            k_g = pl.make_tile_group(
                type=pl.TileType(shape=[1, HEAD_DIM], dtype=BF16,
                                 target_memory=pl.MemorySpace.Vec),
                addrs=[va_bf + 256], mutex_ids=[5])
            g_g = pl.make_tile_group(
                type=pl.TileType(shape=[1, HEAD_DIM], dtype=BF16,
                                 target_memory=pl.MemorySpace.Vec),
                addrs=[va_bf + 512], mutex_ids=[6])
            v_g = pl.make_tile_group(
                type=pl.TileType(shape=[1, 64], dtype=BF16, valid_shape=[-1, -1],
                                 target_memory=pl.MemorySpace.Vec),
                addrs=[va_bf + 768], mutex_ids=[7])
            beta_g = pl.make_tile_group(
                type=pl.TileType(shape=[1, 64], dtype=BF16, valid_shape=[-1, -1],
                                 target_memory=pl.MemorySpace.Vec),
                addrs=[va_bf + 1024], mutex_ids=[8])
            qn_g = pl.make_tile_group(
                type=pl.TileType(shape=[1, HEAD_DIM], dtype=FP32,
                                 target_memory=pl.MemorySpace.Vec),
                addrs=[va_misc], mutex_ids=[9])
            kn_g = pl.make_tile_group(
                type=pl.TileType(shape=[1, HEAD_DIM], dtype=FP32,
                                 target_memory=pl.MemorySpace.Vec),
                addrs=[va_misc + 512], mutex_ids=[10])
            dec_g = pl.make_tile_group(
                type=pl.TileType(shape=[1, HEAD_DIM], dtype=FP32,
                                 target_memory=pl.MemorySpace.Vec),
                addrs=[va_misc + 1024], mutex_ids=[11])
            kd_g = pl.make_tile_group(
                type=pl.TileType(shape=[1, HEAD_DIM], dtype=FP32,
                                 target_memory=pl.MemorySpace.Vec),
                addrs=[va_misc + 1536], mutex_ids=[12])
            vf_g = pl.make_tile_group(
                type=pl.TileType(shape=[1, HEAD_DIM], dtype=FP32,
                                 target_memory=pl.MemorySpace.Vec),
                addrs=[va_misc + 2048], mutex_ids=[13])
            betaf_g = pl.make_tile_group(
                type=pl.TileType(shape=[1, 64], dtype=FP32,
                                 target_memory=pl.MemorySpace.Vec),
                addrs=[va_misc + 2560], mutex_ids=[14])
            outf_g = pl.make_tile_group(
                type=pl.TileType(shape=[1, HEAD_DIM], dtype=FP32,
                                 target_memory=pl.MemorySpace.Vec),
                addrs=[va_misc + 2816], mutex_ids=[15])
            outb_g = pl.make_tile_group(
                type=pl.TileType(shape=[1, 64], dtype=BF16, valid_shape=[-1, -1],
                                 target_memory=pl.MemorySpace.Vec),
                addrs=[va_out], mutex_ids=[16])
            sk_g = pl.make_tile_group(
                type=pl.TileType(shape=[1, HEAD_DIM], dtype=FP32,
                                 target_memory=pl.MemorySpace.Vec),
                addrs=[va_misc + 3328], mutex_ids=[17])
            ug_g = pl.make_tile_group(
                type=pl.TileType(shape=[1, HEAD_DIM], dtype=FP32,
                                 target_memory=pl.MemorySpace.Vec),
                addrs=[va_misc + 3840], mutex_ids=[18])

            items_per_batch = heads * row_blocks
            total_items = batch * items_per_batch
            for item in pl.range(pl.get_block_idx(), total_items, pl.get_block_num()):
                bb = item // items_per_batch
                rem = item % items_per_batch
                hh = rem // row_blocks
                rblk = rem % row_blocks
                r0 = rblk * rows
                tok_lo = bb * seq
                tok_hi = tok_lo + seq
                if packed == 1:
                    tok_lo = pl.getval(cu, bb)
                    tok_hi = pl.getval(cu, bb + 1)
                acc = pl.getval(accepted, bb)
                acc_prev = pl.max(acc - 1, 0)
                init_tok = tok_lo + acc_prev
                slot0 = pl.getval(state_idx, init_tok)
                state = state_g.next()
                if state_bf16:
                    ld_bf = ld_g.next()
                    pl.load(ld_bf, sinit, [slot0, hh, r0, 0])
                    _widen_state(ld_bf, state, rows)
                else:
                    pl.load(state, sinit, [slot0, hh, r0, 0])
                dtb_t = dtb_g.next()
                pl.load(dtb_t, dt_bias, [hh, 0])
                alog = pl.getval(a_log, hh)
                mix_base = bb * mix_b + hh * mix_n + tok_lo * mix_t * packed
                g_base = bb * g_b + hh * g_n + tok_lo * g_t * packed
                pl.set_validshape(v_g.next(), [1, rows])
                pl.set_validshape(beta_g.next(), [1, 1])
                pl.set_validshape(outb_g.next(), [1, rows])
                # Decoupled snapshot: each token's new state goes to a dedicated
                # snapshot tile (FP32 directly, BF16 via narrowing) and is DMA'd
                # immediately, so the rotating live state slots are touched only by
                # the VF recurrence and MTE3 never orders against them.  Token t's
                # snapshot S_{t+1} is written to slot state_idx[t] (same mapping as
                # the contract; the initial slot is always overwritten by
                # token init_tok's snapshot).
                for t in pl.range(tok_lo, tok_hi):
                    addr = t - tok_lo
                    mrow = mix_base + addr * mix_t
                    grow = g_base + addr * g_t
                    pl.load(q_g.next(), mix, [mrow, 0])
                    pl.load(k_g.next(), mix, [mrow + mix_ko, 0])
                    pl.load(v_g.next(), mix, [mrow + mix_vo, r0])
                    pl.load(g_g.next(), g, [grow, 0])
                    pl.load(beta_g.next(), beta, [grow, 0])
                    _prep(q_g.current(), k_g.current(), g_g.current(),
                          v_g.current(), beta_g.current(), dtb_t,
                          qn_g.next(), kn_g.next(), dec_g.next(), kd_g.next(),
                          vf_g.next(), scale, lower, alog, rows)
                    _step_dot(state, kd_g.current(), sk_g.next(), rows)
                    _fold_u(vf_g.current(), sk_g.current(), ug_g.next(), rows)
                    nxt = state_g.next()
                    if state_bf16:
                        # BF16 pool: snapshot is narrowed from nxt afterwards; the
                        # live slot doubles as the (unused) snapshot target here.
                        snap = nxt
                    else:
                        snap = snf_g.next()
                    _step_upd_snap(state, nxt, kn_g.current(), dec_g.current(),
                                   qn_g.current(), ug_g.current(), outf_g.next(),
                                   snap, not state_bf16, rows)
                    _pack_out(outf_g.current(), outb_g.next(), rows)
                    pl.store(out, outb_g.current(), [grow, r0])
                    slot = pl.getval(state_idx, t)
                    if state_bf16:
                        snap_bf = snap_g.next()
                        _narrow_state(nxt, snap_bf, rows)
                        pl.store(sout, snap_bf, [slot, hh, r0, 0])
                    else:
                        pl.store(sout, snap, [slot, hh, r0, 0])
                    state = nxt

    return kernel


_KERNELS: dict[int, object] = {}
_DEFAULTS: dict[tuple, torch.Tensor] = {}


def _default_tensor(key, factory):
    tensor = _DEFAULTS.get(key)
    if tensor is None:
        tensor = factory()
        _DEFAULTS[key] = tensor
    return tensor


def _get_kernel(rows: int, slots: int = STATE_SLOTS, state_bf16: bool = False):
    key = (rows, slots, state_bf16)
    kernel = _KERNELS.get(key)
    if kernel is None:
        kernel = _make_kernel(rows, slots, state_bf16)
        _KERNELS[key] = kernel
    return kernel


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


LAYOUTS = ("BSND", "BNSD", "TND")


def _check_state(state):
    _require(state.dim() == 4, "state must be rank-4")
    _require(state.dtype in (torch.bfloat16, torch.float32),
             f"state must be bf16 or fp32, got {state.dtype}")
    _require(state.is_contiguous(), "state must be contiguous")
    pool, heads, dv, dk = map(int, state.shape)
    _require(dv == dk == HEAD_DIM, "state trailing shape must be 128x128")
    _require(1 <= heads <= MAX_HEADS, f"heads must be in [1, {MAX_HEADS}]")
    return pool, heads


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
    batch, seq, width = map(int, mixedqkv.shape)
    _require(width == 3 * heads * HEAD_DIM, "BSND mixedqkv last dim must be 3*heads*128")
    _require(tuple(g.shape) == (batch, seq, heads, HEAD_DIM), "g shape")
    _require(tuple(beta.shape) == (batch, seq, heads, 1), "beta shape")
    out = torch.empty(batch, seq, heads, HEAD_DIM, dtype=torch.bfloat16,
                      device=state.device)
    return dict(mix=mixedqkv.view(batch * seq * 3 * heads, HEAD_DIM),
                gate=g.view(batch * seq * heads, HEAD_DIM),
                beta=beta.view(batch * seq * heads, 1),
                out=out, out_v=out.view(batch * seq * heads, HEAD_DIM),
                batch=batch, seq=seq, token_count=batch * seq,
                strides=(seq * 3 * heads, 1, 3 * heads, heads, 2 * heads,
                         seq * heads, 1, heads, 0))


def _bnsd_views(mixedqkv, g, beta, heads, state):
    batch, packed_heads, seq, head_dim = map(int, mixedqkv.shape)
    _require(head_dim == HEAD_DIM, "BNSD last dim must be 128")
    _require(packed_heads == 3 * heads, "BNSD packed heads must be 3*heads")
    _require(tuple(g.shape) == (batch, heads, seq, HEAD_DIM), "g shape")
    _require(tuple(beta.shape) == (batch, heads, seq, 1), "beta shape")
    out = torch.empty(batch, heads, seq, HEAD_DIM, dtype=torch.bfloat16,
                      device=state.device)
    return dict(mix=mixedqkv.reshape(batch * packed_heads * seq, HEAD_DIM),
                gate=g.reshape(batch * heads * seq, HEAD_DIM),
                beta=beta.reshape(batch * heads * seq, 1),
                out=out, out_v=out.view(batch * heads * seq, HEAD_DIM),
                batch=batch, seq=seq, token_count=batch * seq,
                strides=(3 * heads * seq, seq, 1, heads * seq, 2 * heads * seq,
                         heads * seq, seq, 1, 0))


def _tnd_views(mixedqkv, g, beta, heads, state, cu_seqlens):
    token_count, width = map(int, mixedqkv.shape)
    _require(width == 3 * heads * HEAD_DIM, "TND mixedqkv last dim must be 3*heads*128")
    _require(tuple(g.shape) == (token_count, heads, HEAD_DIM), "g shape")
    _require(tuple(beta.shape) == (token_count, heads, 1), "beta shape")
    _require(cu_seqlens is not None, "cu_seqlens is required for TND")
    batch = int(cu_seqlens.numel()) - 1
    out = torch.empty(token_count, heads, HEAD_DIM, dtype=torch.bfloat16,
                      device=state.device)
    return dict(mix=mixedqkv.view(token_count * 3 * heads, HEAD_DIM),
                gate=g.view(token_count * heads, HEAD_DIM),
                beta=beta.view(token_count * heads, 1),
                out=out, out_v=out.view(token_count * heads, HEAD_DIM),
                batch=batch, seq=0, token_count=token_count,
                strides=(0, 1, 3 * heads, heads, 2 * heads, 0, 1, heads, 1))


def _prepare(mixedqkv, state, beta, g, a_log, dt_bias, lower, layout, cu_seqlens):
    pool, heads = _check_state(state)
    _check_inputs(mixedqkv, g, beta, a_log, dt_bias, heads, lower)
    if layout == "BSND":
        prep = _bsnd_views(mixedqkv, g, beta, heads, state)
    elif layout == "BNSD":
        prep = _bnsd_views(mixedqkv, g, beta, heads, state)
    elif layout == "TND":
        prep = _tnd_views(mixedqkv, g, beta, heads, state, cu_seqlens)
    else:
        raise ValueError(f"layout_qkv must be one of {LAYOUTS}, got {layout}")
    _require(pool >= prep["token_count"], "state pool must provide token_count slots")
    for tensor, name in ((mixedqkv, "mixedqkv"), (prep["gate"], "g"),
                         (prep["beta"], "beta")):
        _require(tensor.is_contiguous(), f"{name} view must be contiguous")
    if cu_seqlens is not None:
        _require(cu_seqlens.dtype == torch.int32 and cu_seqlens.is_contiguous(),
                 "cu_seqlens must be contiguous int32")
    prep.update(heads=heads, a_log=a_log, dt_bias=dt_bias)
    return prep


def _workers_for():
    """Vector core count of the current device, capped by the stream budget."""
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
                                   ssm_state_indices=None, num_accepted_tokens=None,
                                   rows=DEFAULT_ROWS, workers=None, slots=STATE_SLOTS):
    _require(layout_qkv in LAYOUTS, "layout_qkv must be BSND, BNSD or TND")
    prep = _prepare(mixedqkv, state, beta, g, a_log, dt_bias, lower_bound,
                    layout_qkv, cu_seqlens)
    token_count = prep["token_count"]
    batch = prep["batch"]
    heads = prep["heads"]
    if ssm_state_indices is None:
        state_idx = _default_tensor(
            ("idx", str(state.device), token_count),
            lambda: torch.arange(token_count, dtype=torch.int32, device=state.device))
    else:
        _require(ssm_state_indices.dtype == torch.int32, "state indices must be int32")
        _require(tuple(ssm_state_indices.shape) == (token_count,),
                 "state indices must have token_count entries")
        state_idx = ssm_state_indices
    if num_accepted_tokens is None:
        accepted = _default_tensor(
            ("acc", str(state.device), batch),
            lambda: torch.ones(batch, dtype=torch.int32, device=state.device))
    else:
        accepted = num_accepted_tokens
    cu = cu_seqlens
    if cu is None:
        cu = state_idx[: batch + 1]
    ca, cb, ct, cko, cvo, cgb, cgn, cgt, packed = prep["strides"]
    if workers is None:
        workers = _workers_for()
    total_items = batch * heads * (HEAD_DIM // rows)
    _require(total_items > 0, "no work items")
    workers = min(int(workers), total_items)
    scale_value = float(scale) if scale is not None else HEAD_DIM ** -0.5
    state_bf16 = state.dtype == torch.bfloat16
    kernel = _get_kernel(rows, slots, state_bf16)
    kernel[None, workers](
        prep["mix"], prep["gate"], prep["beta"], prep["out_v"],
        a_log, dt_bias, state, state, state_idx, accepted, cu,
        scale_value, float(lower_bound),
        heads, prep["seq"], batch,
        ca, cb, ct, cko, cvo, cgb, cgn, cgt, packed,
    )
    return state, prep["out"]


def fused_recurrent_kda(mixedqkv, state, beta, g, scale, a_log, dt_bias,
                        lower_bound=-5.0, layout_qkv="BSND", *, cu_seqlens=None,
                        ssm_state_indices=None, num_accepted_tokens=None,
                        rows=DEFAULT_ROWS, workers=None, slots=STATE_SLOTS):
    """Run the fused recurrent KDA snapshot operator; returns (state, out)."""
    return fused_recurrent_kda_functional(
        mixedqkv, state, beta, g, scale, a_log, dt_bias, lower_bound, layout_qkv,
        cu_seqlens=cu_seqlens, ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens, rows=rows, workers=workers, slots=slots)
