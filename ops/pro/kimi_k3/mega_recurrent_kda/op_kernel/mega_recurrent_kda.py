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
"""Fused ``mega_recurrent_kda`` PyPTO-Pro operator.

Host wrapper plus a single mixed AIC/AIV ``@pl.jit`` kernel implementing the
full SPEC math (``SPEC_ANALYSIS.md`` §8): QKV / decay / beta / output-gate
projections, causal Conv1D+SiLU, Q/K RMS normalisation, KDA delta-rule
recurrence, RMSNorm + output gate and the output projection.

    with section_cube():                     # AIC
        qkv      = hidden @ Wqkv^T                (fp32)
        proj2    = hidden @ [Wa;Wbeta;Wgate]^T       (bf16)
        sync_all(AIC_ONLY)                    # raw_decay consumes proj2
        raw_decay= proj2[:, :128] @ Wb^T          (bf16)
    sync_all(MIX)
    with section_vector():                   # AIV (total = batch*num_heads)
        activation, Conv1D+SiLU + Q/K RMS norm, delta-rule recurrence,
        RMSNorm + output gate                -> gated (bf16), recurrent_state
    sync_all(MIX)
    with section_cube():                     # AIC
        out      = gated @ Wo^T                   (bf16)

The three cube phases and the vector phase share one launch, mirroring the
reference's single mixed AIC/AIV kernel: the vector work is grid-strided over
``total = batch*num_heads`` work items by ``get_block_num()*get_subblock_num()``
and ``gexp`` / ``beta_act`` / ``gate_act`` / ``qhat`` / ``khat`` / ``vconv`` stay
in UB instead of round-tripping through GM.

Caller weights are ND bf16 and are pre-packed to FRACTAL_NZ internally.
``rows = B*S`` is padded to a multiple of 128
(Mpad) so every GEMM tile is full; padded rows are sliced off on return.
"""

import os
from typing import NamedTuple

import torch
import torch_npu

import pypto
import pypto_pro.language as pl
from pypto_pro.runtime.platform import get_platform_info
from pypto_pro.runtime.tilingkey import TilingKeyField
from pypto_pro.language import Vf as vf  # noqa: N813

# torch_npu.Format.FRACTAL_NZ; the projection weights arrive in NZ so the Cube
# GM->L1 load is a pure NZ->NZ copy instead of a per-load ND->NZ reformat.
_FRACTAL_NZ = int(torch_npu.Format.FRACTAL_NZ)

_S = 8
_D = 128
_CONV_K = 4
_QK_EPS = 1e-6
_TILE = 128
_VL = 64          # SIMD vector length (fp32 lanes)
_INV_D = 1.0 / _D  # host-side reciprocal, the VF parser forbids scalar division
_BETA_OFF = 128   # offset of the (padded) beta block inside proj2
_GATE_OFF = 256   # offset of the gate block inside proj2
_BETA_W = 128     # padded beta width


# Compile-time M tile. ``B*S`` is padded to a multiple of it, so the small-batch
# shapes (rows 32 / 64) stop paying 4x / 2x redundant rows in the projection.


class _MegaTilingKey:
    MTile = TilingKeyField(bits=2, values=[32, 64, 128])


# Supported AIC quotas for this operator.  The kernel is accepted/measured for
# these five counts; smaller or intermediate values are not part of the spec.
# The default is the largest supported quota the device can host (9589 -> 32,
# 28-core SKUs -> 28); MEGA_KDA_CUBE_CORES must name one of these.
_SUPPORTED_CUBE_CORES = (8, 16, 24, 28, 32)
# B-adaptive front-end tiling: when the padded M grid has at least this many
# 128-row tiles there is enough Cube work to hide the longer N384 panel chain,
# so the qkv projection reuses each hidden panel for three outputs (A reuse
# R=3, ~1/3 fewer A reads) instead of two (N256, R=2).  Below the threshold the
# extra chain length is not amortised and N256 wins.
_N384_MIN_MTILES = 2
# Below this AIC quota the Cube is MTE2-byte bound enough that the wider N384
# panel (A reuse R=3) pays off even when the M grid is a single 128-row tile.
_N384_LOWCORE_CUT = 16
# At/above this AIC quota with at least two M tiles, the widest qkv projection
# (nt_q >= 72, i.e. N=24) prefers the N256 chain, while a single M tile prefers
# N384 (see the branch comment in the kernel).
_N384_WIDE_CUT_CORES = 28
_N384_WIDE_CUT_NT = 72


# ---------------------------------------------------------------------------
# Conv1D + SiLU + Q/K RMS norm
# ---------------------------------------------------------------------------

@pl.vector_function
def _conv_silu_vf(win, wt, out):
    """Causal Conv1D(K=4)+SiLU over the 8 tokens of one channel slice.

    ``win`` is [16,128] with rows [0,1,2] = 3 history rows and [3..10] = the 8
    raw tokens; ``wt`` is the BF16 [16,128] panel whose rows [0..3] are the 4
    filter taps (loaded with ``UNPK_B16``, so no FP32 copy is staged in UB).
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for t in pl.range(0, _S, 1):
        a0 = vf.full(0.0, preg, dtype=pl.DT_FP32)
        a1 = vf.full(0.0, preg, dtype=pl.DT_FP32)
        for k in pl.range(0, _CONV_K, 1):
            wb0 = vf.load_align(wt, k * _D, dist=pl.LoadDist.UNPK_B16)
            wb1 = vf.load_align(wt, k * _D + _VL, dist=pl.LoadDist.UNPK_B16)
            w0 = vf.astype(wb0, preg, dtype=pl.DT_FP32)
            w1 = vf.astype(wb1, preg, dtype=pl.DT_FP32)
            b0 = vf.load_align(win, (t + k) * _D, dist=pl.LoadDist.UNPK_B16)
            b1 = vf.load_align(win, (t + k) * _D + _VL, dist=pl.LoadDist.UNPK_B16)
            s0 = vf.astype(b0, preg, dtype=pl.DT_FP32)
            s1 = vf.astype(b1, preg, dtype=pl.DT_FP32)
            m0 = vf.mul(w0, s0, preg)
            a0 = vf.add(a0, m0, preg)
            m1 = vf.mul(w1, s1, preg)
            a1 = vf.add(a1, m1, preg)
        one = vf.full(1.0, preg, dtype=pl.DT_FP32)
        n0 = vf.neg(a0, preg)
        e0 = vf.exp(n0, preg)
        d0 = vf.add(one, e0, preg)
        r0 = vf.div(a0, d0, preg)
        n1 = vf.neg(a1, preg)
        e1 = vf.exp(n1, preg)
        d1 = vf.add(one, e1, preg)
        r1 = vf.div(a1, d1, preg)
        vf.store_align(out, r0, preg, t * _D)
        vf.store_align(out, r1, preg, t * _D + _VL)


# ---------------------------------------------------------------------------
# Per-(batch, head) activation stage: decay, beta and output gate.
# The three chains share one register pass instead of a UB->UB tile-op chain
# which wrote an intermediate tile back to UB at every step.
# ---------------------------------------------------------------------------

@pl.vector_function
def _act_vf(rdb, beta_src, gate_src, dtb, gexp, beta, gate,
            alog: pl.DT_FP32, lower_bound: pl.DT_FP32):
    """The whole per-(batch, head) activation stage in one register pass.

    ``decay = exp(lower_bound * sigmoid((raw_decay + dt_bias) * exp(a_log)))``,
    ``beta = sigmoid(beta_in)`` and ``gate = sigmoid(gate_in)`` are three
    independent chains over the same ``[8, 128]`` rows, so they share one
    loop: each panel is loaded once (BF16 via ``UNPK_B16``), the whole chain
    stays in registers and only the three FP32 results are stored.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    ainv = vf.full(alog, preg, dtype=pl.DT_FP32)
    ascale = vf.exp(ainv, preg)
    for r in pl.range(0, _S, 1):
        for hf in pl.range(0, 2, 1):
            off = r * _D + hf * _VL
            b = vf.load_align(rdb, off, dist=pl.LoadDist.UNPK_B16)
            s = vf.astype(b, preg, dtype=pl.DT_FP32)
            d = vf.load_align(dtb, hf * _VL)
            ge = vf.add(s, d, preg)
            ge = vf.mul(ge, ascale, preg)
            ge = vf.neg(ge, preg)
            ge = vf.exp(ge, preg)
            ge = vf.add(ge, one, preg)
            ge = vf.div(one, ge, preg)
            ge = vf.muls(ge, lower_bound, preg)
            ge = vf.exp(ge, preg)
            vf.store_align(gexp, ge, preg, off)

            bb = vf.load_align(beta_src, off, dist=pl.LoadDist.UNPK_B16)
            bs = vf.astype(bb, preg, dtype=pl.DT_FP32)
            bn = vf.neg(bs, preg)
            be = vf.exp(bn, preg)
            be = vf.add(be, one, preg)
            be = vf.div(one, be, preg)
            vf.store_align(beta, be, preg, off)

            gb = vf.load_align(gate_src, off, dist=pl.LoadDist.UNPK_B16)
            gs = vf.astype(gb, preg, dtype=pl.DT_FP32)
            gn = vf.neg(gs, preg)
            ga = vf.exp(gn, preg)
            ga = vf.add(ga, one, preg)
            ga = vf.div(one, ga, preg)
            vf.store_align(gate, ga, preg, off)


@pl.vector_function
def _rmsgate_vf(o, gam, gate_t, rrm, eps: pl.DT_FP32):
    """RMSNorm over the 128 channels of each token, then gamma and output gate.

    ``o`` is the per-token output tile produced by the recurrence; ``rrm`` is a
    ``[1, 8]`` FP32 scratch row that carries the per-token sum of squares from
    the reduce pass to the scale pass (no register-level broadcast in the DSL,
    so it goes through UB once per token instead of ``o`` going through global
    memory).  The final FP32->BF16 pack stays outside the VF: the narrowing
    conversion needs an undeclared ``CastLayout`` on this SDK.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pref = vf.create_mask(pattern=pl.MaskPattern.VL1, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    g0 = vf.load_align(gam, 0)
    g1 = vf.load_align(gam, _VL)
    for t in pl.range(0, _S, 1):
        s0 = vf.load_align(o, t * _D)
        s1 = vf.load_align(o, t * _D + _VL)
        q0 = vf.mul(s0, s0, preg)
        q1 = vf.mul(s1, s1, preg)
        r0 = vf.reduce_sum(q0, preg)
        r1 = vf.reduce_sum(q1, preg)
        ss = vf.add(r0, r1, preg)
        vf.store_align(rrm, ss, pref, t, dist=pl.StoreDist.FIRST_ELEMENT_B32)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    for t in pl.range(0, _S, 1):
        dd = vf.load_align(rrm, t, dist=pl.LoadDist.BRC)
        dd = vf.muls(dd, _INV_D, preg)
        dd = vf.adds(dd, eps, preg)
        rt = vf.sqrt(dd, preg)
        inv = vf.div(one, rt, preg)
        s0 = vf.load_align(o, t * _D)
        s1 = vf.load_align(o, t * _D + _VL)
        s0 = vf.mul(s0, inv, preg)
        s1 = vf.mul(s1, inv, preg)
        s0 = vf.mul(s0, g0, preg)
        s1 = vf.mul(s1, g1, preg)
        f0 = vf.load_align(gate_t, t * _D)
        f1 = vf.load_align(gate_t, t * _D + _VL)
        s0 = vf.mul(s0, f0, preg)
        s1 = vf.mul(s1, f1, preg)
        vf.store_align(o, s0, preg, t * _D)
        vf.store_align(o, s1, preg, t * _D + _VL)


@pl.vector_function
def _qknorm_vf(out, rrm, eps: pl.DT_FP32, scale: pl.DT_FP32):
    """In-place Q/K RMS norm over the 128 channels of each token row.

    ``out`` is one FP32 Conv1D+SiLU panel; ``rrm`` is a ``[1, 8]`` FP32 scratch
    row that carries the per-token sum of squares from the reduce pass to the
    scale pass.  ``scale`` folds the output scale into the same pass (pass a
    literal ``1.0`` for K).
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pref = vf.create_mask(pattern=pl.MaskPattern.VL1, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    for t in pl.range(0, _S, 1):
        s0 = vf.load_align(out, t * _D)
        s1 = vf.load_align(out, t * _D + _VL)
        q0 = vf.mul(s0, s0, preg)
        q1 = vf.mul(s1, s1, preg)
        r0 = vf.reduce_sum(q0, preg)
        r1 = vf.reduce_sum(q1, preg)
        ss = vf.add(r0, r1, preg)
        vf.store_align(rrm, ss, pref, t, dist=pl.StoreDist.FIRST_ELEMENT_B32)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    for t in pl.range(0, _S, 1):
        dd = vf.load_align(rrm, t, dist=pl.LoadDist.BRC)
        dd = vf.adds(dd, eps, preg)
        rt = vf.sqrt(dd, preg)
        inv = vf.div(one, rt, preg)
        s0 = vf.load_align(out, t * _D)
        s1 = vf.load_align(out, t * _D + _VL)
        s0 = vf.mul(s0, inv, preg)
        s1 = vf.mul(s1, inv, preg)
        s0 = vf.muls(s0, scale, preg)
        s1 = vf.muls(s1, scale, preg)
        vf.store_align(out, s0, preg, t * _D)
        vf.store_align(out, s1, preg, t * _D + _VL)


# ---------------------------------------------------------------------------
# Single mixed AIC/AIV kernel: the four projections, activation, Conv1D+SiLU,
# Q/K RMS norm, KDA delta-rule recurrence and RMSNorm + output gate.
# Cube runs qkv/proj2/raw_decay, then the vector section works on the
# grid-strided (batch, head) items while gexp/beta/gate/qhat/khat/vconv stay in
# UB, and finally Cube projects the gated activation to the output.
# The SIMD VFs (``_conv_silu_vf`` / ``_kda_{ksum,delta,update}_vf``) are unchanged.
# ---------------------------------------------------------------------------
# The proj2 weight is three separate public tensors -- Wa, Wbeta (padded to
# 128 columns), Wgate -- selected per 128-wide N tile instead of being concatenated
# on the host.  Tiles 0 and 1 are Wa and the padded beta block, tiles >= 2 are
# the gate block at local column ``(tile - 2) * 128``.
def _load_proj2_panel(dst, wa_t, wbeta_t, wgate_t, k_row, tile):
    if tile == 0:
        pl.load(dst, wa_t, [k_row, 0])
    else:
        if tile == 1:
            pl.load(dst, wbeta_t, [k_row, 0])
        else:
            pl.load(dst, wgate_t, [k_row, (tile - 2) * _TILE])


def _matmul_2tile(ac0, ac1, al, r0, r1, ca, cb0, cb1, k):
    pl.move(al, ca)
    pl.move(r0, cb0)
    pl.move(r1, cb1)
    if k == 0:
        pl.matmul(ac0, al, r0)
        pl.matmul(ac1, al, r1)
    else:
        pl.matmul_acc(ac0, ac0, al, r0)
        pl.matmul_acc(ac1, ac1, al, r1)


def _cube_qkv_fused4p(
    c_a_l0, c_a_l1, c_acc0, c_b0_l0, c_b0_l1, cid_c,
    hidden_pad, k_hidden, nc_c, nt_all, nt_q, proj2,
    qkv, wa_t, wbeta_t, wgate_t, wqkv_t):
    """Cube qkv: single-M 4P fused front end (qkv + proj2 share one panel)."""
    for t in pl.range(cid_c, nt_all, nc_c):
        ac0 = c_acc0.current()
        for k in pl.range(0, k_hidden, 1):
            ca = c_a_l1.next()
            cb0 = c_b0_l1.next()
            pl.load(ca, hidden_pad, [0, k * _TILE])
            if t < nt_q:
                pl.load(cb0, wqkv_t, [k * _TILE, t * _TILE])
            else:
                _load_proj2_panel(cb0, wa_t, wbeta_t, wgate_t,
                                  k * _TILE, t - nt_q)
            al = c_a_l0.next()
            r0 = c_b0_l0.next()
            pl.move(al, ca)
            pl.move(r0, cb0)
            if k == 0:
                pl.matmul(ac0, al, r0)
            else:
                pl.matmul_acc(ac0, ac0, al, r0)
        if t < nt_q:
            pl.store(qkv, ac0, [0, t * _TILE])
        else:
            pl.store(proj2, ac0, [0, (t - nt_q) * _TILE])


def _cube_qkv_n512(
    c_acc0, c_acc1, c_acc2, c_acc3, cid_c, hidden_pad,
    k5_a_l0, k5_a_l1, k5_b0_l0, k5_b0_l1, k5_b1_l0, k5_b1_l1,
    k5_b2_l0, k5_b2_l1, k5_b3_l0, k5_b3_l1, k_hidden, m_tiles,
    n4_q, nc_c, qkv, wqkv_t):
    """Cube qkv: N512 (R=4) panels for a single M tile."""
    for t in pl.range(cid_c, m_tiles * n4_q, nc_c):
        i = t % m_tiles
        j0 = (t // m_tiles) * 4
        ac0 = c_acc0.current()
        ac1 = c_acc1.current()
        ac2 = c_acc2.current()
        ac3 = c_acc3.current()
        for k in pl.range(0, k_hidden, 1):
            for s in pl.range(0, 2, 1):
                a64 = k5_a_l1.next()
                b0 = k5_b0_l1.next()
                b1 = k5_b1_l1.next()
                b2 = k5_b2_l1.next()
                b3 = k5_b3_l1.next()
                pl.load(a64, hidden_pad, [i * MTile, k * _TILE + s * 64])
                pl.load(b0, wqkv_t, [k * _TILE + s * 64, j0 * _TILE])
                pl.load(b1, wqkv_t, [k * _TILE + s * 64, (j0 + 1) * _TILE])
                pl.load(b2, wqkv_t, [k * _TILE + s * 64, (j0 + 2) * _TILE])
                pl.load(b3, wqkv_t, [k * _TILE + s * 64, (j0 + 3) * _TILE])
                al = k5_a_l0.next()
                r0 = k5_b0_l0.next()
                r1 = k5_b1_l0.next()
                r2 = k5_b2_l0.next()
                r3 = k5_b3_l0.next()
                pl.move(al, a64)
                pl.move(r0, b0)
                pl.move(r1, b1)
                pl.move(r2, b2)
                pl.move(r3, b3)
                if k == 0 and s == 0:
                    pl.matmul(ac0, al, r0)
                    pl.matmul(ac1, al, r1)
                    pl.matmul(ac2, al, r2)
                    pl.matmul(ac3, al, r3)
                else:
                    pl.matmul_acc(ac0, ac0, al, r0)
                    pl.matmul_acc(ac1, ac1, al, r1)
                    pl.matmul_acc(ac2, ac2, al, r2)
                    pl.matmul_acc(ac3, ac3, al, r3)
        pl.store(qkv, ac0, [i * MTile, j0 * _TILE])
        pl.store(qkv, ac1, [i * MTile, (j0 + 1) * _TILE])
        pl.store(qkv, ac2, [i * MTile, (j0 + 2) * _TILE])
        pl.store(qkv, ac3, [i * MTile, (j0 + 3) * _TILE])


def _cube_qkv_n128(
    c_a_l0, c_a_l1, c_acc0, c_b0_l0, c_b0_l1, cid_c,
    hidden_pad, k_hidden, m_tiles, nc_c, nt_q, qkv,
    wqkv_t):
    """Cube qkv: N128 (R=1) panels when N256 leaves cores idle."""
    n1_q = nt_q
    for t in pl.range(cid_c, m_tiles * n1_q, nc_c):
        i = t % m_tiles
        j0 = t // m_tiles
        ac0 = c_acc0.current()
        for k in pl.range(0, k_hidden, 1):
            ca = c_a_l1.next()
            cb0 = c_b0_l1.next()
            pl.load(ca, hidden_pad, [i * MTile, k * _TILE])
            pl.load(cb0, wqkv_t, [k * _TILE, j0 * _TILE])
            al = c_a_l0.next()
            r0 = c_b0_l0.next()
            pl.move(al, ca)
            pl.move(r0, cb0)
            if k == 0:
                pl.matmul(ac0, al, r0)
            else:
                pl.matmul_acc(ac0, ac0, al, r0)
        pl.store(qkv, ac0, [i * MTile, j0 * _TILE])


def _cube_qkv_n384(
    c_acc0, c_acc1, c_acc2, cid_c, hidden_pad, k_a64_l0,
    k_a64_l1, k_b0_l0, k_b0_l1, k_b1_l0, k_b1_l1, k_b2_l0,
    k_b2_l1, k_hidden, m_tiles, n3_q, nc_c, qkv,
    wqkv_t):
    for t in pl.range(cid_c, m_tiles * n3_q, nc_c):
        i = t % m_tiles
        j0 = (t // m_tiles) * 3
        ac0 = c_acc0.current()
        ac1 = c_acc1.current()
        ac2 = c_acc2.current()
        for k in pl.range(0, k_hidden, 1):
            for s in pl.range(0, 2, 1):
                a64 = k_a64_l1.next()
                b0 = k_b0_l1.next()
                b1 = k_b1_l1.next()
                b2 = k_b2_l1.next()
                pl.load(a64, hidden_pad, [i * MTile, k * _TILE + s * 64])
                pl.load(b0, wqkv_t, [k * _TILE + s * 64, j0 * _TILE])
                pl.load(b1, wqkv_t, [k * _TILE + s * 64, (j0 + 1) * _TILE])
                pl.load(b2, wqkv_t, [k * _TILE + s * 64, (j0 + 2) * _TILE])
                al = k_a64_l0.next()
                r0 = k_b0_l0.next()
                r1 = k_b1_l0.next()
                r2 = k_b2_l0.next()
                pl.move(al, a64)
                pl.move(r0, b0)
                pl.move(r1, b1)
                pl.move(r2, b2)
                if k == 0 and s == 0:
                    pl.matmul(ac0, al, r0)
                    pl.matmul(ac1, al, r1)
                    pl.matmul(ac2, al, r2)
                else:
                    pl.matmul_acc(ac0, ac0, al, r0)
                    pl.matmul_acc(ac1, ac1, al, r1)
                    pl.matmul_acc(ac2, ac2, al, r2)
        pl.store(qkv, ac0, [i * MTile, j0 * _TILE])
        pl.store(qkv, ac1, [i * MTile, (j0 + 1) * _TILE])
        pl.store(qkv, ac2, [i * MTile, (j0 + 2) * _TILE])


def _cube_qkv_n256(
    c_a_l0, c_a_l1, c_acc0, c_acc1, c_b0_l0, c_b0_l1,
    c_b1_l0, c_b1_l1, cid_c, hidden_pad, k_hidden, m_tiles,
    nc_c, nt_q, qkv, wqkv_t):
    n2_q = (nt_q + 1) // 2
    for t in pl.range(cid_c, m_tiles * n2_q, nc_c):
        i = t % m_tiles
        j0 = (t // m_tiles) * 2
        ac0 = c_acc0.current()
        ac1 = c_acc1.current()
        for k in pl.range(0, k_hidden, 1):
            ca = c_a_l1.next()
            cb0 = c_b0_l1.next()
            cb1 = c_b1_l1.next()
            pl.load(ca, hidden_pad, [i * MTile, k * _TILE])
            pl.load(cb0, wqkv_t, [k * _TILE, j0 * _TILE])
            pl.load(cb1, wqkv_t, [k * _TILE, (j0 + 1) * _TILE])
            al = c_a_l0.next()
            r0 = c_b0_l0.next()
            r1 = c_b1_l0.next()
            _matmul_2tile(ac0, ac1, al, r0, r1, ca, cb0, cb1, k)
        pl.store(qkv, ac0, [i * MTile, j0 * _TILE])
        pl.store(qkv, ac1, [i * MTile, (j0 + 1) * _TILE])


def _cube_proj2_1tile(
    c_a_l0, c_a_l1, c_acc0, c_b0_l0, c_b0_l1, cid_c,
    hidden_pad, k_hidden, nc_c, nt_p, proj2, wa_t,
    wbeta_t, wgate_t):
    """Cube proj2: one N-tile per job."""
    for t in pl.range(cid_c, nt_p, nc_c):
        ac0 = c_acc0.current()
        for k in pl.range(0, k_hidden, 1):
            ca = c_a_l1.next()
            cb0 = c_b0_l1.next()
            pl.load(ca, hidden_pad, [0, k * _TILE])
            _load_proj2_panel(cb0, wa_t, wbeta_t, wgate_t, k * _TILE, t)
            al = c_a_l0.next()
            r0 = c_b0_l0.next()
            pl.move(al, ca)
            pl.move(r0, cb0)
            if k == 0:
                pl.matmul(ac0, al, r0)
            else:
                pl.matmul_acc(ac0, ac0, al, r0)
        pl.store(proj2, ac0, [0, t * _TILE])


def _cube_proj2_2tile(
    c_a_l0, c_a_l1, c_acc0, c_acc1, c_b0_l0, c_b0_l1,
    c_b1_l0, c_b1_l1, cid_c, hidden_pad, k_hidden, m_tiles,
    n2_p, nc_c, proj2, wa_t, wbeta_t, wgate_t):
    """Cube proj2: two N-tiles per job."""
    for t in pl.range(cid_c, m_tiles * n2_p, nc_c):
        i = t % m_tiles
        j0 = (t // m_tiles) * 2
        ac0 = c_acc0.current()
        ac1 = c_acc1.current()
        for k in pl.range(0, k_hidden, 1):
            ca = c_a_l1.next()
            cb0 = c_b0_l1.next()
            cb1 = c_b1_l1.next()
            pl.load(ca, hidden_pad, [i * MTile, k * _TILE])
            _load_proj2_panel(cb0, wa_t, wbeta_t, wgate_t, k * _TILE, j0)
            _load_proj2_panel(cb1, wa_t, wbeta_t, wgate_t, k * _TILE, j0 + 1)
            al = c_a_l0.next()
            r0 = c_b0_l0.next()
            r1 = c_b1_l0.next()
            _matmul_2tile(ac0, ac1, al, r0, r1, ca, cb0, cb1, k)
        pl.store(proj2, ac0, [i * MTile, j0 * _TILE])
        pl.store(proj2, ac1, [i * MTile, (j0 + 1) * _TILE])


def _cube_front_proj(
    c_a_l0, c_a_l1, c_acc0, c_acc1, c_acc2, c_acc3,
    c_b0_l0, c_b0_l1, c_b1_l0, c_b1_l1, hidden_pad, k5_a_l0,
    k5_a_l1, k5_b0_l0, k5_b0_l1, k5_b1_l0, k5_b1_l1, k5_b2_l0,
    k5_b2_l1, k5_b3_l0, k5_b3_l1, k_a64_l0, k_a64_l1, k_b0_l0,
    k_b0_l1, k_b1_l0, k_b1_l1, k_b2_l0, k_b2_l1, k_hidden,
    m_tiles, nt_p, nt_q, proj2, qkv, wa_t,
    wbeta_t, wgate_t, wqkv_t):
    """Cube front projections: qkv and proj2 (Wa/Wbeta/Wgate).

    The branch tree below is driven by five runtime values only: ``m_tiles``,
    ``nc_c``, ``nt_q = 3N``, ``nt_p = 2+N`` and ``nt_all = nt_q + nt_p``.
    Priority: 4P fusion -> (qkv) N512 / N128 / N384 / N256 -> (proj2) 1 or 2
    N-tiles per job.  The conditions, capacity limits (R <= 4, KS = 64) and the
    measured exceptions are documented in ``DESIGN.md`` §2.1.
    """
    nc_c = pl.get_block_num()
    cid_c = pl.get_block_idx()
    n3_q = nt_q // 3
    # See DESIGN.md §2.1 for the decision tree and its measurements.
    n384 = ((m_tiles >= _N384_MIN_MTILES and m_tiles * n3_q >= nc_c)
            or (nc_c <= _N384_LOWCORE_CUT and n3_q * 2 >= nc_c))
    if nc_c >= _N384_WIDE_CUT_CORES:
        if nt_q >= _N384_WIDE_CUT_NT and m_tiles >= _N384_MIN_MTILES:
            n384 = False
    n4_q = nt_q // 4
    nt_all = nt_q + nt_p

    if (m_tiles == 1 and nc_c >= _N384_WIDE_CUT_CORES
            and nt_all <= nc_c):
        # 4P fusion: qkv and proj2 share one A panel and one task pool.
        _cube_qkv_fused4p(
            c_a_l0, c_a_l1, c_acc0, c_b0_l0, c_b0_l1, cid_c,
            hidden_pad, k_hidden, nc_c, nt_all, nt_q, proj2,
            qkv, wa_t, wbeta_t, wgate_t, wqkv_t
        )
        pl.system.bar_fix()
        pl.system.sync_all(core_type=pl.SyncCoreType.AIC_ONLY)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
    else:
        if (nc_c >= _N384_WIDE_CUT_CORES and m_tiles == 1
                and nt_q % 4 == 0 and n4_q * 2 >= nc_c):
            # N512 (A reuse R=4): 4 B panels fill L0B, 4 accumulators fill L0C.
            _cube_qkv_n512(
                c_acc0, c_acc1, c_acc2, c_acc3, cid_c, hidden_pad,
                k5_a_l0, k5_a_l1, k5_b0_l0, k5_b0_l1, k5_b1_l0, k5_b1_l1,
                k5_b2_l0, k5_b2_l1, k5_b3_l0, k5_b3_l1, k_hidden, m_tiles,
                n4_q, nc_c, qkv, wqkv_t
            )
        elif (nc_c >= _N384_WIDE_CUT_CORES and m_tiles == 1
                and ((nt_q + 1) // 2) * 2 < nc_c and nt_q * 2 >= nc_c):
            # N128 (R=1): N256 leaves cores idle, so one tile per job.
            _cube_qkv_n128(
                c_a_l0, c_a_l1, c_acc0, c_b0_l0, c_b0_l1, cid_c,
                hidden_pad, k_hidden, m_tiles, nc_c, nt_q, qkv,
                wqkv_t
            )
        elif n384:
            # N384 (A reuse R=3), KS=64 keeps 3 B panels resident in L0B.
            _cube_qkv_n384(
                c_acc0, c_acc1, c_acc2, cid_c, hidden_pad, k_a64_l0,
                k_a64_l1, k_b0_l0, k_b0_l1, k_b1_l0, k_b1_l1, k_b2_l0,
                k_b2_l1, k_hidden, m_tiles, n3_q, nc_c, qkv,
                wqkv_t
            )
        else:
            _cube_qkv_n256(
                c_a_l0, c_a_l1, c_acc0, c_acc1, c_b0_l0, c_b0_l1,
                c_b1_l0, c_b1_l1, cid_c, hidden_pad, k_hidden, m_tiles,
                nc_c, nt_q, qkv, wqkv_t
            )
        pl.system.bar_fix()
        pl.system.sync_all(core_type=pl.SyncCoreType.AIC_ONLY)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
        n2_p = (nt_p + 1) // 2
        if (m_tiles == 1 and nc_c >= _N384_WIDE_CUT_CORES
                and n2_p * 4 <= nc_c):
            # Very few 2-tile jobs: one N-tile per job doubles the task count.
            _cube_proj2_1tile(
                c_a_l0, c_a_l1, c_acc0, c_b0_l0, c_b0_l1, cid_c,
                hidden_pad, k_hidden, nc_c, nt_p, proj2, wa_t,
                wbeta_t, wgate_t
            )
        else:
            # Two N-tiles per job, aligned with the qkv N256 path.
            _cube_proj2_2tile(
                c_a_l0, c_a_l1, c_acc0, c_acc1, c_b0_l0, c_b0_l1,
                c_b1_l0, c_b1_l1, cid_c, hidden_pad, k_hidden, m_tiles,
                n2_p, nc_c, proj2, wa_t, wbeta_t, wgate_t
            )
        # Final barrier for path B: proj2 must be visible to raw_decay on all AIC cores.
        pl.system.bar_fix()
        pl.system.sync_all(core_type=pl.SyncCoreType.AIC_ONLY)


def _cube_decay_proj(
    c_a_l0, c_a_l1, c_acc0, c_acc1, c_b0_l0, c_b0_l1,
    c_b1_l0, c_b1_l1, m_tiles, nt_d, proj2, raw_decay,
    wb_t):
    nc_c = pl.get_block_num()
    cid_c = pl.get_block_idx()
    n2_d = (nt_d + 1) // 2
    if (m_tiles == 1 and nc_c >= _N384_WIDE_CUT_CORES
            and n2_d * 4 <= nc_c):
        for t in pl.range(cid_c, nt_d, nc_c):
            ac0 = c_acc0.current()
            for k in pl.range(0, 1, 1):
                ca = c_a_l1.next()
                cb0 = c_b0_l1.next()
                pl.load(ca, proj2, [0, 0])
                pl.load(cb0, wb_t, [0, t * _TILE])
                al = c_a_l0.next()
                r0 = c_b0_l0.next()
                pl.move(al, ca)
                pl.move(r0, cb0)
                if k == 0:
                    pl.matmul(ac0, al, r0)
                else:
                    pl.matmul_acc(ac0, ac0, al, r0)
            pl.store(raw_decay, ac0, [0, t * _TILE])
    else:
        for t in pl.range(cid_c, m_tiles * n2_d, nc_c):
            i = t % m_tiles
            j0 = (t // m_tiles) * 2
            ac0 = c_acc0.current()
            ac1 = c_acc1.current()
            for k in pl.range(0, 1, 1):
                ca = c_a_l1.next()
                cb0 = c_b0_l1.next()
                cb1 = c_b1_l1.next()
                pl.load(ca, proj2, [i * MTile, 0])
                pl.load(cb0, wb_t, [0, j0 * _TILE])
                pl.load(cb1, wb_t, [0, (j0 + 1) * _TILE])
                al = c_a_l0.next()
                r0 = c_b0_l0.next()
                r1 = c_b1_l0.next()
                _matmul_2tile(ac0, ac1, al, r0, r1, ca, cb0, cb1, k)
            pl.store(raw_decay, ac0, [i * MTile, j0 * _TILE])
            pl.store(raw_decay, ac1, [i * MTile, (j0 + 1) * _TILE])


def _vector_conv_qknorm(
    P, b_hb1, b_hb2, b_oq, b_ok, b_ov, b_raw8b, b_red,
    b_wbf, b_win, conv_num_accepted, conv_state, conv_state_indices,
    conv_w, kg, n_heads, qg, qkv, scale, total, vg):
    """Vector: causal Conv1D + SiLU and Q/K RMS norm (one (batch, head) item).

    The three conv panels are kept in UB and normalised in place, so there is no
    ``convolved`` global-memory staging buffer and no per-item MTE3 drain.  The
    final stores read register-produced tiles, which the framework's auto_mutex
    does not order against MTE3, hence the explicit ``bar_all()`` before them.
    """
    nc = pl.get_block_num() * pl.get_subblock_num()
    cid = pl.get_block_idx()
    for work in pl.range(cid, total, nc):
        b = work // n_heads
        h = work % n_heads
        # ================= B: Conv1D+SiLU + Q/K RMS norm =================
        oq = b_oq.current()
        ok = b_ok.current()
        ov = b_ov.current()
        cache_index = pl.getval(conv_state_indices, b)
        if cache_index != 0:
            accepted = pl.getval(conv_num_accepted, b)
            off = accepted - 1
            if off < 0:
                off = 0
            for si in pl.range(0, 3, 1):
                base = si * P + h * _D
                w = b_win.current()
                pl.load(w[0:3, :], conv_state, [cache_index, off, base])
                pl.load(w[3:11, :], qkv, [b * _S, base])
                pl.load(b_hb1.current(), conv_state, [cache_index, off + 1, base])
                pl.store(conv_state, b_hb1.current(), [cache_index, 0, base])
                pl.load(b_hb2.current(), conv_state, [cache_index, off + 2, base])
                pl.store(conv_state, b_hb2.current(), [cache_index, 1, base])
                pl.load(b_raw8b.current(), qkv, [b * _S, base])
                pl.store(conv_state, b_raw8b.current(), [cache_index, 2, base])
                wb = b_wbf.current()
                # single [4,128] tap load instead of four [1,128] copies
                pl.load(wb[0:4, :], conv_w, [0, base])
                if si == 0:
                    _conv_silu_vf(w, wb, oq)
                elif si == 1:
                    _conv_silu_vf(w, wb, ok)
                else:
                    _conv_silu_vf(w, wb, ov)
            _qknorm_vf(oq, b_red.current(), _QK_EPS, scale)
            _qknorm_vf(ok, b_red.current(), _QK_EPS, 1.0)
        else:
            # A zero conv panel normalises to zero.
            pl.expands(oq, 0.0)
            pl.expands(ok, 0.0)
            pl.expands(ov, 0.0)
        # The panels are register-produced; order the V stores before MTE3.
        pl.system.bar_all()
        pl.store(qg, oq, [b * _S, h * _D])
        pl.store(kg, ok, [b * _S, h * _D])
        pl.store(vg, ov, [b * _S, h * _D])


def _vector_kda_recur(
    a_log, a_bg, a_gb, a_dtb, a_rdb, batch, beta, c_delta, c_gam, c_ksum,
    c_o8, c_ob, c_rr, c_snap,
    c_st, dt_bias, gate, gated,
    gexp, h_offset, kg, khat, lower_bound, n_heads,
    num_accepted, proj2, qg, qhat, raw_decay, recurrent_state,
    rms_eps, ssm_state_indices, st_stride, vconv, vg):
    """Vector: KDA recurrence + RMSNorm/gate for one head group (h starts at h_offset)."""
    nc = pl.get_block_num() * pl.get_subblock_num()
    cid = pl.get_block_idx()
    c_gam_t = c_gam.current()
    hg2 = n_heads // 2
    for work in pl.range(cid, hg2 * batch, nc):
        b = work % batch
        h = h_offset + work // batch
        # ================= A: decay / beta / gate activation =================
        dt = a_dtb.current()
        pl.load(dt, dt_bias, [h, 0])
        rdb = a_rdb.current()
        pl.load(rdb, raw_decay, [b * _S, h * _D])
        bb = a_bg.current()
        pl.load(bb, proj2, [b * _S, _BETA_OFF])
        gb = a_gb.current()
        pl.load(gb, proj2, [b * _S, _GATE_OFF + h * _D])
        _act_vf(rdb, bb, gb, dt, gexp.current(), beta.current(), gate.current(),
                pl.getval(a_log, h), lower_bound)

        pl.load(khat.current(), kg, [b * _S, h * _D])
        pl.load(qhat.current(), qg, [b * _S, h * _D])
        pl.load(vconv.current(), vg, [b * _S, h * _D])
        # ================= C: KDA recurrence + RMSNorm/gate =================
        accepted = pl.getval(num_accepted, b)
        it = accepted - 1
        if it < 0:
            it = 0
        idx0 = pl.getval(ssm_state_indices, b * _S + it)
        s = c_st.current()
        sn = c_snap.next()
        pl.load(s, recurrent_state, [idx0 * st_stride + h * _D, 0])
        oc = c_o8.current()
        idx_t = pl.getval(ssm_state_indices, b * _S)
        prev_idx = idx_t
        prev_buf = 0
        for t in pl.range(0, _S, 1):
            row = b * _S + t
            idx_nxt = idx_t
            if t + 1 < _S:
                idx_nxt = pl.getval(ssm_state_indices, row + 1)
            kr = khat.current()[t:t + 1, :]
            qr = qhat.current()[t:t + 1, :]
            gr = gexp.current()[t:t + 1, :]
            vr = vconv.current()[t:t + 1, :]
            betav = pl.getval(beta.current(), t * _BETA_W + h)
            _kda_ksum_vf(s, kr, gr, c_ksum.current())
            _kda_delta_vf(vr, c_ksum.current(), c_delta.current(), betav)
            sn = c_snap[t % 2]
            _kda_update_vf(s, sn, kr, qr, gr, c_delta.current(), oc, t * _D)
            # Deferred one token so the state DMA overlaps the recurrence.
            # (l2_cache_mode): add the L2 store hint here once pl.store exposes it.
            if t > 0:
                pl.store(recurrent_state, c_snap[prev_buf], [prev_idx * st_stride + h * _D, 0])
            prev_buf = t % 2
            prev_idx = idx_t
            idx_t = idx_nxt
        pl.store(recurrent_state, c_snap[prev_buf], [prev_idx * st_stride + h * _D, 0])
        # ---- RMSNorm + output gate (o stays in UB, no GM round-trip) ----
        _rmsgate_vf(oc, c_gam_t, gate.current(), c_rr.current(), rms_eps)
        obb = c_ob.current()
        pl.cast(obb, oc)
        pl.store(gated, obb, [b * _S, h * _D])


def _cube_out_proj(c_acc0, c_acc1, c_a_l1, c_b0_l1, c_b1_l1,
                   c_a_l0, c_b0_l0, c_b1_l0, gated, wo_t, out_pad,
                   nt_o, k_gated, m_tiles, accum):
    """Cube output projection: one K segment of ``out = gated @ Wo``.

    Parameters are the L1/L0/Acc tile groups, then the GM tensors, then the
    runtime tile counts; the DSL inlines this helper into the kernel.
    ``accum`` selects the second K half, which atomically adds to ``out_pad``.
    """
    nc_c = pl.get_block_num()
    cid_c = pl.get_block_idx()
    n2_o = (nt_o + 1) // 2
    k_half = k_gated // 2
    for t in pl.range(cid_c, m_tiles * n2_o, nc_c):
        i = t % m_tiles
        j0 = (t // m_tiles) * 2
        ac0 = c_acc0.current()
        ac1 = c_acc1.current()
        for k in pl.range(0, k_half, 1):
            kk = k_half + k if accum else k
            ca = c_a_l1.next()
            cb0 = c_b0_l1.next()
            cb1 = c_b1_l1.next()
            pl.load(ca, gated, [i * MTile, kk * _TILE])
            pl.load(cb0, wo_t, [kk * _TILE, j0 * _TILE])
            pl.load(cb1, wo_t, [kk * _TILE, (j0 + 1) * _TILE])
            al = c_a_l0.next()
            r0 = c_b0_l0.next()
            r1 = c_b1_l0.next()
            _matmul_2tile(ac0, ac1, al, r0, r1, ca, cb0, cb1,
                          0 if accum else k)
        if accum:
            pl.store(out_pad, ac0, [i * MTile, j0 * _TILE],
                     atomic=pl.AtomicType.AtomicAdd)
            pl.store(out_pad, ac1, [i * MTile, (j0 + 1) * _TILE],
                     atomic=pl.AtomicType.AtomicAdd)
        else:
            pl.store(out_pad, ac0, [i * MTile, j0 * _TILE])
            pl.store(out_pad, ac1, [i * MTile, (j0 + 1) * _TILE])


@pl.jit(auto_mutex=True, tiling_key=_MegaTilingKey)
def mega_recurrent_kda_kernel(
    hidden_pad: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    wqkv_t: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16, pl.NZ],
    wa_t: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16, pl.NZ],
    wbeta_t: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16, pl.NZ],
    wgate_t: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16, pl.NZ],
    wb_t: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16, pl.NZ],
    wo_t: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16, pl.NZ],
    out_pad: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    qkv: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    proj2: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    raw_decay: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    qg: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    kg: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    vg: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    gated: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    dt_bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    a_log: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32],
    lower_bound: pl.DT_FP32,
    conv_w: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    conv_state: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    conv_state_indices: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    conv_num_accepted: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    gamma: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32],
    ssm_state_indices: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    num_accepted: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    recurrent_state: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    rms_eps: pl.DT_FP32,
    scale: pl.DT_FP32,
):
    n_heads = dt_bias.shape[0]
    P = raw_decay.shape[1]  # noqa: N806, G.NAM.01
    batch = conv_state_indices.shape[0]
    total = batch * n_heads
    st_stride = n_heads * _D

    vec_mem = pl.MemorySpace.Vec
    t1 = pl.TileType(shape=[1, _D], dtype=pl.DT_FP32, target_memory=vec_mem)
    tb1 = pl.TileType(shape=[1, _D], dtype=pl.DT_BF16, target_memory=vec_mem)
    t8f = pl.TileType(shape=[_S, _D], dtype=pl.DT_FP32, target_memory=vec_mem)
    t8b = pl.TileType(shape=[_S, _D], dtype=pl.DT_BF16, target_memory=vec_mem)
    t8v = pl.TileType(shape=[_S, _D], dtype=pl.DT_BF16, target_memory=vec_mem,
                      valid_shape=[-1, -1])
    t16b = pl.TileType(shape=[16, _D], dtype=pl.DT_BF16, target_memory=vec_mem)
    tstate = pl.TileType(shape=[_D, _D], dtype=pl.DT_FP32, target_memory=vec_mem)
    rmt = pl.TileType(shape=[1, _S], dtype=pl.DT_FP32, target_memory=vec_mem)

    # ---- UB intermediates that live across the three fused stages ----
    gexp = pl.make_tile_group(type=t8f, addrs=0x00000, mutex_ids=[0])
    beta = pl.make_tile_group(type=t8f, addrs=0x01000, mutex_ids=[1])
    gate = pl.make_tile_group(type=t8f, addrs=0x02000, mutex_ids=[2])
    qhat = pl.make_tile_group(type=t8f, addrs=0x03000, mutex_ids=[3])
    khat = pl.make_tile_group(type=t8f, addrs=0x04000, mutex_ids=[4])
    vconv = pl.make_tile_group(type=t8f, addrs=0x05000, mutex_ids=[5])
    # ---- stage A scratch: activation (the chains themselves run in registers) ----
    a_dtb = pl.make_tile_group(type=t1, addrs=0x06000, mutex_ids=[7])
    a_rdb = pl.make_tile_group(type=t8b, addrs=0x06200, mutex_ids=[8])
    a_bg = pl.make_tile_group(type=t8v, addrs=0x06A00, mutex_ids=[11])
    a_gb = pl.make_tile_group(type=t8v, addrs=0x07200, mutex_ids=[29])
    # ---- stage C small scratch (compacted low, before the big state tiles) ----
    c_o8 = pl.make_tile_group(type=t8f, addrs=0x07A00, mutex_ids=[7])
    c_ksum = pl.make_tile_group(type=t1, addrs=0x08A00, mutex_ids=[8])
    c_delta = pl.make_tile_group(type=t1, addrs=0x08C00, mutex_ids=[9])
    c_gam = pl.make_tile_group(type=t1, addrs=0x08E00, mutex_ids=[12])
    c_ob = pl.make_tile_group(type=t8b, addrs=0x09000, mutex_ids=[16])
    c_rr = pl.make_tile_group(type=rmt, addrs=0x09800, mutex_ids=[17])
    # ---- big state + depth-2 snapshot: 3 x 64KB contiguous ----
    c_st = pl.make_tile_group(type=tstate, addrs=0x0A000, mutex_ids=[28])
    c_snap = pl.make_tile_group(type=tstate, addrs=[0x1A000, 0x2A000], mutex_ids=[18, 19])
    # ---- stage B scratch: Conv1D+SiLU + Q/K RMS norm (compacted above the state) ----
    b_wbf = pl.make_tile_group(type=t16b, addrs=0x3A000, mutex_ids=[12])
    b_hb1 = pl.make_tile_group(type=tb1, addrs=0x3B000, mutex_ids=[15])
    b_hb2 = pl.make_tile_group(type=tb1, addrs=0x3B100, mutex_ids=[16])
    b_raw8b = pl.make_tile_group(type=t8b, addrs=0x3B200, mutex_ids=[19])
    b_win = pl.make_tile_group(type=t16b, addrs=0x3BA00, mutex_ids=[20])
    b_oq = pl.make_tile_group(type=t8f, addrs=0x3CA00, mutex_ids=[21])
    b_ok = pl.make_tile_group(type=t8f, addrs=0x3DA00, mutex_ids=[13])
    b_ov = pl.make_tile_group(type=t8f, addrs=0x3EA00, mutex_ids=[14])
    b_red = pl.make_tile_group(type=rmt, addrs=0x3FA00, mutex_ids=[27])

    # ---- Cube tiles (shared by the four projection phases) ----
    # L1 Mat tiles are NZ: the activation/`gated` operands are ND GM (ND->NZ
    # reformat on load), the four projection weights are pre-packed NZ GM, so
    # their load is a pure NZ->NZ copy.
    ct_mat = pl.TileType(shape=[MTile, _TILE], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat,
                         layout=pl.NZ)
    ct_w = pl.TileType(shape=[_TILE, _TILE], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat,
                       layout=pl.NZ)
    ct_left = pl.TileType(shape=[MTile, _TILE], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left)
    ct_right = pl.TileType(shape=[_TILE, _TILE], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right)
    ct_acc = pl.TileType(shape=[MTile, _TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         fractal=1024)
    c_a_l1 = pl.make_tile_group(type=ct_mat, addrs=0x00000, mutex_ids=[0, 1])
    c_b0_l1 = pl.make_tile_group(type=ct_w, addrs=0x10000, mutex_ids=[2, 3])
    c_b1_l1 = pl.make_tile_group(type=ct_w, addrs=0x20000, mutex_ids=[10, 11])
    c_a_l0 = pl.make_tile_group(type=ct_left, addrs=0x0000, mutex_ids=[4, 5])
    c_b0_l0 = pl.make_tile_group(type=ct_right, addrs=0x0000, mutex_ids=[6])
    c_b1_l0 = pl.make_tile_group(type=ct_right, addrs=0x8000, mutex_ids=[7])
    c_acc0 = pl.make_tile_group(type=ct_acc, addrs=0x0000, mutex_ids=[8])
    c_acc1 = pl.make_tile_group(type=ct_acc, addrs=0x10000, mutex_ids=[9])
    c_acc2 = pl.make_tile_group(type=ct_acc, addrs=0x20000, mutex_ids=[14])
    # KS=64 N384 resident panels: a [128,64] A panel and three [64,128] B panels
    # share the L1/L0 space of the N256 path (the two branches are runtime
    # exclusive); three 16 KB L0B panels fit in the 64 KB L0B so the third
    # matmul no longer waits on a reused slot.
    ct_a64 = pl.TileType(shape=[MTile, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat,
                         layout=pl.NZ)
    ct_b64 = pl.TileType(shape=[64, _TILE], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat,
                         layout=pl.NZ)
    ct_l64 = pl.TileType(shape=[MTile, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left)
    ct_r64 = pl.TileType(shape=[64, _TILE], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right)
    k_a64_l1 = pl.make_tile_group(type=ct_a64, addrs=0x40000, mutex_ids=[15, 16])
    k_b0_l1 = pl.make_tile_group(type=ct_b64, addrs=0x48000, mutex_ids=[17, 18])
    k_b1_l1 = pl.make_tile_group(type=ct_b64, addrs=0x50000, mutex_ids=[19, 20])
    k_b2_l1 = pl.make_tile_group(type=ct_b64, addrs=0x58000, mutex_ids=[21, 22])
    k_a64_l0 = pl.make_tile_group(type=ct_l64, addrs=0x0000, mutex_ids=[23, 24])
    k_b0_l0 = pl.make_tile_group(type=ct_r64, addrs=0x0000, mutex_ids=[25])
    k_b1_l0 = pl.make_tile_group(type=ct_r64, addrs=0x4000, mutex_ids=[26])
    k_b2_l0 = pl.make_tile_group(type=ct_r64, addrs=0x8000, mutex_ids=[27])
    # N512 (R=4) small-M path: exclusive with the N256/N384 branches, so it
    # reuses their mutex ids (the DSL allocates buffer ids by lifetime).
    k5_a_l1 = pl.make_tile_group(type=ct_a64, addrs=0x40000, mutex_ids=[15, 16])
    k5_b0_l1 = pl.make_tile_group(type=ct_b64, addrs=0x48000, mutex_ids=[17, 18])
    k5_b1_l1 = pl.make_tile_group(type=ct_b64, addrs=0x50000, mutex_ids=[19, 20])
    k5_b2_l1 = pl.make_tile_group(type=ct_b64, addrs=0x58000, mutex_ids=[21, 22])
    k5_b3_l1 = pl.make_tile_group(type=ct_b64, addrs=0x60000, mutex_ids=[23, 24])
    k5_a_l0 = pl.make_tile_group(type=ct_l64, addrs=0x0000, mutex_ids=[25, 26])
    k5_b0_l0 = pl.make_tile_group(type=ct_r64, addrs=0x0000, mutex_ids=[27])
    k5_b1_l0 = pl.make_tile_group(type=ct_r64, addrs=0x4000, mutex_ids=[0])
    k5_b2_l0 = pl.make_tile_group(type=ct_r64, addrs=0x8000, mutex_ids=[1])
    k5_b3_l0 = pl.make_tile_group(type=ct_r64, addrs=0xC000, mutex_ids=[2])
    c_acc3 = pl.make_tile_group(type=ct_acc, addrs=0x30000, mutex_ids=[3])

    m_tiles = hidden_pad.shape[0] // MTile
    k_hidden = hidden_pad.shape[1] // _TILE
    nt_q = wqkv_t.shape[1] // _TILE
    nt_p = proj2.shape[1] // _TILE
    nt_d = raw_decay.shape[1] // _TILE
    nt_o = out_pad.shape[1] // _TILE
    k_gated = gated.shape[1] // _TILE

    hg2 = n_heads // 2

    with pl.section_cube():
        _cube_front_proj(
            c_a_l0, c_a_l1, c_acc0, c_acc1, c_acc2, c_acc3,
            c_b0_l0, c_b0_l1, c_b1_l0, c_b1_l1, hidden_pad, k5_a_l0,
            k5_a_l1, k5_b0_l0, k5_b0_l1, k5_b1_l0, k5_b1_l1, k5_b2_l0,
            k5_b2_l1, k5_b3_l0, k5_b3_l1, k_a64_l0, k_a64_l1, k_b0_l0,
            k_b0_l1, k_b1_l0, k_b1_l1, k_b2_l0, k_b2_l1, k_hidden,
            m_tiles, nt_p, nt_q, proj2, qkv, wa_t,
            wbeta_t, wgate_t, wqkv_t
        )
        _cube_decay_proj(
            c_a_l0, c_a_l1, c_acc0, c_acc1, c_b0_l0, c_b0_l1,
            c_b1_l0, c_b1_l1, m_tiles, nt_d, proj2, raw_decay,
            wb_t
        )
        pl.system.bar_fix()
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=1)

    with pl.section_vector():
        pl.load(c_gam.current(), gamma, [0])
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)
        _vector_conv_qknorm(
            P, b_hb1, b_hb2, b_oq, b_ok, b_ov, b_raw8b, b_red,
            b_wbf, b_win, conv_num_accepted, conv_state, conv_state_indices,
            conv_w, kg, n_heads, qg, qkv, scale, total, vg
        )
        # qg/kg/vg and gated are MTE3 stores; drain only that pipe.
        pl.system.bar_mte3()
        pl.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=1)
        _vector_kda_recur(
            a_log, a_bg, a_gb, a_dtb, a_rdb, batch, beta, c_delta, c_gam, c_ksum,
            c_o8, c_ob, c_rr, c_snap,
            c_st, dt_bias, gate, gated,
            gexp, 0, kg, khat, lower_bound, n_heads,
            num_accepted, proj2, qg, qhat, raw_decay, recurrent_state,
            rms_eps, ssm_state_indices, st_stride, vconv, vg
        )
        # qg/kg/vg and gated are MTE3 stores; drain only that pipe.
        pl.system.bar_mte3()
        pl.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)
        _vector_kda_recur(
            a_log, a_bg, a_gb, a_dtb, a_rdb, batch, beta, c_delta, c_gam, c_ksum,
            c_o8, c_ob, c_rr, c_snap,
            c_st, dt_bias, gate, gated,
            gexp, hg2, kg, khat, lower_bound, n_heads,
            num_accepted, proj2, qg, qhat, raw_decay, recurrent_state,
            rms_eps, ssm_state_indices, st_stride, vconv, vg
        )
        # qg/kg/vg and gated are MTE3 stores; drain only that pipe.
        pl.system.bar_mte3()
        pl.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=3)

    with pl.section_cube():
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=2)
        _cube_out_proj(
            c_acc0, c_acc1, c_a_l1, c_b0_l1, c_b1_l1, c_a_l0, c_b0_l0,
            c_b1_l0, gated, wo_t, out_pad, nt_o, k_gated, m_tiles, False)
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=3)
        _cube_out_proj(
            c_acc0, c_acc1, c_a_l1, c_b0_l1, c_b1_l1, c_a_l0, c_b0_l0,
            c_b1_l0, gated, wo_t, out_pad, nt_o, k_gated, m_tiles, True)


# ---------------------------------------------------------------------------
# KDA delta-rule recurrence + RMSNorm + output gate
# ---------------------------------------------------------------------------

@pl.vector_function
def _kda_ksum_vf(state, key, decay, ksum):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pref = vf.create_mask(pattern=pl.MaskPattern.VL1, dtype=pl.DT_FP32)
    dec0 = vf.load_align(decay, 0)
    dec1 = vf.load_align(decay, _VL)
    kk0 = vf.load_align(key, 0)
    kk1 = vf.load_align(key, _VL)
    dk0 = vf.mul(dec0, kk0, preg)
    dk1 = vf.mul(dec1, kk1, preg)
    for dv in pl.range(0, _D, 1):
        s0 = vf.load_align(state, dv * _D)
        s1 = vf.load_align(state, dv * _D + _VL)
        t = vf.mul(s0, dk0, preg)
        t2 = vf.mul(s1, dk1, preg)
        t = vf.add(t, t2, preg)
        rr = vf.reduce_sum(t, preg)
        vf.store_align(ksum, rr, pref, dv, dist=pl.StoreDist.FIRST_ELEMENT_B32)


@pl.vector_function
def _kda_delta_vf(value, ksum, delta, beta: pl.DT_FP32):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for h in pl.range(0, _D, _VL):
        bb = vf.full(beta, preg, dtype=pl.DT_FP32)
        va = vf.load_align(value, h)
        ks = vf.load_align(ksum, h)
        dd = vf.sub(va, ks, preg)
        dd = vf.mul(dd, bb, preg)
        vf.store_align(delta, dd, preg, h)


@pl.vector_function
def _kda_update_vf(state, snap, key, query, decay, delta, out_o, out_off: pl.DT_INT32):
    """Step 3: state = state*decay + delta[dv]*key, mirrored into ``snap``.

    ``snap`` is the async-store source: the GM write of token t reads ``snap``
    while token t+1 updates ``state``, so the write no longer serialises with
    the next recurrence step.  ``out_o[out_off+dv] = sum_k state*query``.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pref = vf.create_mask(pattern=pl.MaskPattern.VL1, dtype=pl.DT_FP32)
    dec0 = vf.load_align(decay, 0)
    dec1 = vf.load_align(decay, _VL)
    kk0 = vf.load_align(key, 0)
    kk1 = vf.load_align(key, _VL)
    qq0 = vf.load_align(query, 0)
    qq1 = vf.load_align(query, _VL)
    for dv in pl.range(0, _D, 1):
        db = vf.load_align(delta, dv, dist=pl.LoadDist.BRC)
        sp = vf.load_align(state, dv * _D)
        sp = vf.mul(sp, dec0, preg)
        up = vf.mul(db, kk0, preg)
        sp = vf.add(sp, up, preg)
        vf.store_align(state, sp, preg, dv * _D)
        vf.store_align(snap, sp, preg, dv * _D)
        sq = vf.load_align(state, dv * _D + _VL)
        sq = vf.mul(sq, dec1, preg)
        uq = vf.mul(db, kk1, preg)
        sq = vf.add(sq, uq, preg)
        vf.store_align(state, sq, preg, dv * _D + _VL)
        vf.store_align(snap, sq, preg, dv * _D + _VL)
        o0 = vf.mul(sp, qq0, preg)
        o1 = vf.mul(sq, qq1, preg)
        oo = vf.add(o0, o1, preg)
        ro = vf.reduce_sum(oo, preg)
        vf.store_align(out_o, ro, pref, out_off + dv, dist=pl.StoreDist.FIRST_ELEMENT_B32)


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------
def _m_tile_for(rows):
    """Smallest compile-time M tile that keeps a single M tile for small batches."""
    if rows <= 32:
        return 32
    if rows <= 64:
        return 64
    return _TILE


def _ceil_m(n, m):
    return (n + m - 1) // m * m


_WORKSPACE_CACHE = {}


def _require_nz(tensor, name, shape):
    """The six projection weights must be BF16 FRACTAL_NZ ``[K, N]`` tensors.

    ``K`` is the GEMM reduction dim and ``N`` the output channels, i.e. the
    orientation the kernel consumes directly.  A ``[N, K]`` weight is rejected
    rather than silently transposed.
    """
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be BF16")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"{name} must be logical [K, N] = {tuple(shape)}, got {tuple(tensor.shape)}; "
            "projection weights are reduction-dim-first, not [N, K]")
    if torch_npu.get_npu_format(tensor) != _FRACTAL_NZ:
        raise ValueError(
            f"{name} must be a FRACTAL_NZ allocation-root tensor "
            f"(got {torch_npu.get_npu_format(tensor)}); pass "
            "torch_npu.npu_format_cast(weight, torch_npu.Format.FRACTAL_NZ)")


def _pack_conv_weight(conv1d_weight, num_heads):
    """Normalise the Conv1D taps to the kernel's ``[4, 3*N*128]`` layout.

    Accepts the same three shapes as the reference contract: ``[3,N,4,128]``
    (Q/K/V, head, tap, channel), head-shared ``[3,4,128]`` and the flattened
    ``[4,3*N*128]``.  Column order is always ``[Q(P) | K(P) | V(P)]``.
    """
    dim = _D
    taps = _CONV_K
    width = 3 * num_heads * dim
    shape = tuple(conv1d_weight.shape)
    if shape == (taps, width):
        return conv1d_weight.contiguous()
    if shape == (3, num_heads, taps, dim):
        return conv1d_weight.permute(2, 0, 1, 3).reshape(taps, width).contiguous()
    if shape == (3, taps, dim):
        return (conv1d_weight[:, None, :, :].expand(3, num_heads, taps, dim)
                .permute(2, 0, 1, 3).reshape(taps, width).contiguous())
    raise ValueError("conv1d_weight must be [3,N,4,128], [3,4,128] or [4,3*N*128], "
                     f"got {shape}")


class _ShapeInfo(NamedTuple):
    hidden_size: int
    batch: int
    rows: int
    m_tile: int
    mpad: int
    num_heads: int
    P: int  # noqa: N815, G.NAM.01


def _validate_arguments(hidden_states, qkv_projection_weight, decay_projection_a_weight,
                        decay_projection_b_weight, beta_projection_weight,
                        output_gate_projection_weight, output_projection_weight,
                        conv_state_indices, a_log):
    """Validate the public arguments and return the derived shape values.

    The checks run before any workspace allocation or kernel launch so a
    malformed call fails on the host with a named argument.
    """
    hidden_size = 7168
    batch = int(conv_state_indices.shape[0])
    rows = batch * _S
    m_tile = _m_tile_for(rows)
    mpad = _ceil_m(rows, m_tile)
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError(f"hidden_states must be BF16, got {hidden_states.dtype}")
    if tuple(hidden_states.shape) != (mpad, hidden_size):
        raise ValueError(
            f"hidden_states must be the padded A operand [{mpad}, {hidden_size}] "
            f"(B*S = {rows} rounded up to a multiple of MTile={m_tile}); "
            "call mega_recurrent_kda_pad_input() on the [B, S, H] activation")
    gate_cols = int(output_gate_projection_weight.shape[1])
    num_heads = gate_cols // _D
    if num_heads * _D != gate_cols:
        raise ValueError("output_gate_projection_weight columns must be a multiple of 128")
    if num_heads > _BETA_W:
        raise ValueError(
            f"head count {num_heads} exceeds the padded beta width {_BETA_W}")
    if num_heads % 2 != 0:
        raise ValueError("mega_recurrent_kda requires an even head count")
    P = num_heads * _D  # noqa: N806, G.NAM.01

    _require_nz(qkv_projection_weight, "qkv_projection_weight",
                (hidden_size, 3 * P))
    _require_nz(decay_projection_a_weight, "decay_projection_a_weight",
                (hidden_size, _D))
    _require_nz(decay_projection_b_weight, "decay_projection_b_weight",
                (_D, P))
    if tuple(beta_projection_weight.shape) != (hidden_size, _BETA_W):
        raise ValueError(
            f"beta_projection_weight must be zero-padded to [H, {_BETA_W}] "
            f"(only the first {num_heads} columns are read), got "
            f"{tuple(beta_projection_weight.shape)}")
    _require_nz(beta_projection_weight, "beta_projection_weight", (hidden_size, _BETA_W))
    _require_nz(output_gate_projection_weight, "output_gate_projection_weight",
                (hidden_size, P))
    _require_nz(output_projection_weight, "output_projection_weight",
                (P, hidden_size))

    if a_log.dtype != torch.float32:
        raise ValueError(f"a_log must be FP32, got {a_log.dtype}")
    if not a_log.is_contiguous():
        raise ValueError("a_log must be contiguous (the kernel reads it by linear offset)")
    return _ShapeInfo(hidden_size, batch, rows, m_tile, mpad, num_heads, P)


def mega_recurrent_kda_pad_input(hidden_states):
    """Pad ``[B, S, H]`` to the ``[Mpad, H]`` A operand the wrapper expects.

    ``Mpad`` is ``B*S`` rounded up to a multiple of the compile-time ``MTile``
    (32 / 64 / 128).  The pad rows are never read by the operator; when ``B*S``
    already fills whole tiles this returns a zero-copy view.
    """
    batch, seq, hidden_size = hidden_states.shape
    rows = batch * seq
    m_tile = _m_tile_for(rows)
    mpad = _ceil_m(rows, m_tile)
    flat = hidden_states.reshape(rows, hidden_size)
    if mpad == rows:
        return flat
    out = torch.zeros((mpad, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
    out[:rows].copy_(flat)
    return out


def mega_recurrent_kda_wrapper(
    hidden_states, qkv_projection_weight, decay_projection_a_weight, decay_projection_b_weight,
    beta_projection_weight, output_gate_projection_weight, output_norm_weight, output_projection_weight,
    conv1d_weight, conv_state, recurrent_state, conv_state_indices, ssm_state_indices,
    conv_num_accepted_tokens, num_accepted_tokens, a_log, dt_bias,
    scale, lower_bound, rms_norm_eps=1e-5,
):
    shape = _validate_arguments(
        hidden_states, qkv_projection_weight, decay_projection_a_weight,
        decay_projection_b_weight, beta_projection_weight,
        output_gate_projection_weight, output_projection_weight,
        conv_state_indices, a_log)
    hidden_size = shape.hidden_size
    batch = shape.batch
    rows = shape.rows
    m_tile = shape.m_tile
    mpad = shape.mpad
    num_heads = shape.num_heads
    P = shape.P  # noqa: N806, G.NAM.01
    dev = hidden_states.device
    info = get_platform_info()
    max_cube = int(info.core_num)
    supported = tuple(c for c in _SUPPORTED_CUBE_CORES if c <= max_cube)
    if not supported:
        raise ValueError(
            f"{info.soc_version} exposes {max_cube} AIC; the operator supports "
            f"{_SUPPORTED_CUBE_CORES}")
    default_cube = supported[-1]
    cube = int(os.environ.get("MEGA_KDA_CUBE_CORES", default_cube))
    if cube not in supported:
        raise ValueError(
            f"MEGA_KDA_CUBE_CORES={cube} is not supported on {info.soc_version}; "
            f"choose one of {supported} (core_num={max_cube})")
    conv_packed = _pack_conv_weight(conv1d_weight, num_heads)

    # All six GEMM operands arrive in the kernel's [K, N] NZ layout, so they are
    # handed over as-is: every Cube GM->L1 load is a pure NZ->NZ copy and the
    # host does no format conversion or caching on the weight path.
    w = {
        "wqkv_t": qkv_projection_weight,
        "wa_t": decay_projection_a_weight,
        "wbeta_t": beta_projection_weight,
        "wgate_t": output_gate_projection_weight,
        "wb_t": decay_projection_b_weight,
        "wo_t": output_projection_weight,
    }

    # ---- reusable scratch buffers (depend on batch and head count) ----
    def _build_workspaces():
        ws = {
            "qkv": torch.empty((mpad, 3 * P), dtype=torch.bfloat16,
                               device=dev),
            "qg": torch.empty((mpad, P), dtype=torch.float32, device=dev),
            "kg": torch.empty((mpad, P), dtype=torch.float32, device=dev),
            "vg": torch.empty((mpad, P), dtype=torch.float32, device=dev),
            "proj2": torch.empty((mpad, _GATE_OFF + P),
                                 dtype=torch.bfloat16, device=dev),
            "raw_decay": torch.empty((mpad, P), dtype=torch.bfloat16,
                                     device=dev),
            "gated": torch.zeros((mpad, P), dtype=torch.bfloat16,
                                 device=dev),
        }
        return ws

    ws_key = (dev.index, batch, num_heads)
    c = _WORKSPACE_CACHE.get(ws_key)
    if c is None:
        c = _build_workspaces()
        _WORKSPACE_CACHE[ws_key] = c
    # The caller hands over the padded A operand; the wrapper neither pads nor
    # copies it.
    hidden_pad = hidden_states

    qkv = c["qkv"]
    raw_decay = c["raw_decay"]
    gated = c["gated"]
    # The kernel writes this call's output directly: no cached buffer plus a
    # second D2D copy.  Pad rows are sliced off below.
    out_pad = torch.empty((mpad, hidden_size), dtype=torch.bfloat16, device=dev)

    with pypto.options(pass_options={"enable_slice": False}):
        mega_recurrent_kda_kernel[None, cube, {"MTile": m_tile}](
            hidden_pad, w["wqkv_t"], w["wa_t"], w["wbeta_t"], w["wgate_t"],
            w["wb_t"], w["wo_t"], out_pad,
            qkv, c["proj2"], raw_decay, c["qg"], c["kg"], c["vg"], gated,
            dt_bias, a_log, float(lower_bound),
            conv_packed, conv_state, conv_state_indices, conv_num_accepted_tokens,
            output_norm_weight, ssm_state_indices, num_accepted_tokens,
            recurrent_state.reshape(-1, _D), float(rms_norm_eps), float(scale))
    # ``out_pad`` is freshly allocated per call, so the real rows can be
    # returned as a view: no extra allocation and no D2D copy.
    return out_pad[:rows].view(batch, _S, hidden_size)
