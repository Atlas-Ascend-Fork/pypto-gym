# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Two-stage CP KDA forward candidate.

Public API: flash_kda_stage1_wrapper and flash_kda_stage2_wrapper only.
Each wrapper dispatches one JIT specialization for dense, dense-tail or TND
inputs; the seven internal JIT definitions do not imply seven launches.

Stage1 preprocesses the local chunks and produces a tail E/T summary in the
same kernel. For chunk-aligned dense CP=1 training at a sequence start,
Stage1 also saves its native E-chain h/v_new caches. Stage2 consumes those
caches to compute output chunks in parallel. Other paths retain the exclusive
prefix scan and real local recurrence. The caller gathers summaries in CP-rank
order. Stage2 returns (output, final_state, train_cache_or_None). There is no
host-side padding, head-packing computation, or external cache reconstruction.

The CP contract fixes D=128, chunk=64, V-first FP32 state, bounded sigmoid gate
with lower_bound in [-5,0], and beta=sigmoid(raw_beta). allow_neg_eigval=True
is rejected. MM16 summary/prefix products use BF16 operands and FP32
accumulation; finite-precision CP results need not be bit equal to an unsplit
sequential recurrence. The real recurrence retains FP32 state and Upre, with
BF16 state snapshots and v_new. q_c/k_c use Hk; other head caches use Hv.

Raw input/initial-state head slices can retain row-major storage strides.
Stage2 optional out/final_state_out views receive stores directly. Prepared
tensors, caches and gathered packets remain contiguous, bucket-local tensors.
Only complete key/value-head groups may form a bucket. Packed sequence
metadata is supplied by the caller; the wrapper does not transfer it to CPU.

Training H cache is BF16 V-first; rolling and external states stay FP32.
Inference returns cache=None. Training returns native BF16 h/v_new from
the selected state chain and the Stage1 caches. The initial-state reference in
prepared is authoritative and
must not be modified between stages. Production source is not changed by this
candidate; full integration, multi-card and performance acceptance are separate.
"""

__all__ = ["flash_kda_stage1_wrapper", "flash_kda_stage2_wrapper"]

import os

import torch
import torch_npu  # noqa: F401  (NPU device init)

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.tilingkey import TilingKeyField

class _CPStage1Key:
    LocalStateRows = TilingKeyField(bits=8, values=[32, 64, 128])


CHUNK_C = 64
HEAD_D = 128
L2_EPS = 1e-6
LANES = 64

DV_SPLIT = 8
DV = HEAD_D // DV_SPLIT


ROW_BLOCK = 16
NV_MAX = 96
NK_MAX = 64
B_MAX = 64
BB = 32
HS = CHUNK_C // 2
BLOCK_STRIDE_ND = (CHUNK_C >> 1) | 1
REPEAT_STRIDE_ND = 1

_GATE_FWD_IDS = (0, 1)
_GATE_BWD_IDS = (2, 3)

_M6_UPRE_EVENT = 4
_M7_SNAP_EVENT = 12
_M7_UD_EVENTS = (13, 16)
_M3_GK_DONE = 5
_M5_CS_DONE = 14
_M7_U_EVENT = 14
_M7_D_EVENTS = (15, 17)
_P4A_SNAP_BANK_EVENT = 8

VA_X_B = 0x00000
VA_TMP_F = 0x0C000
VA_QN_B = 0x14000
VA_KN_B = 0x18000
VA_GATE_F = 0x1C000
VA_GATE_NZ = 0x24000
VA_A_LOG = 0x2C000
VA_LB = 0x2C100
VA_DT_B = 0x2C200
VA_SC = 0x2C400
VA_CS0 = 0x2C600
VA_CS1 = 0x34600
VA_M4_SCR_B16 = 0x26000

VA_M5_KKT = 0x8000
VA_M5_MQK = 0x10000
VA_M5_BETA_F = VA_QN_B
VA_M5_VB = VA_KN_B
VA_M5_VF = VA_GATE_NZ
VA_M5_KNF = 0x4000

VA_M7_S = 0x00000
VA_M7_DEL = 0x10000
VA_M7_UD = 0x20000
VA_M7_UB = 0x30000
VA_M7_UTNZ = 0x32000
VA_M7_SNAP = 0x34000
P4A_UB_UPRE0 = 0x28000
P4A_UB_UPRE1 = 0x2C000
P4A_UB_GAM0 = 0x3C000
P4A_UB_GAM1 = 0x3C200

L1_L = 0x00000
L1_G0 = 0x04000
L1_G1 = 0x0C000
L6_OP = 0x66000

L6_OP2 = 0x6E000
L6_LA_VBT = 0x5000
L7_SNAP = 0x38000
L7_QD = 0x2C000
L7_U = 0x30000
P4A_L1_W0 = 0x20000
P4A_L1_W1 = 0x24000
P4A_L1_KR0 = 0x28000
P4A_L1_KR1 = 0x2C000
P4A_L1_QD0 = 0x44000
P4A_L1_QD1 = 0x48000
P4A_L1_MQK0 = 0x4C000
P4A_L1_MQK1 = 0x4E000
P4A_L1_SNAPIN0 = 0x50000
P4A_L1_SNAPIN1 = 0x52000
P4A_L1_UNZ0 = 0x54000
P4A_L1_UNZ1 = 0x55000
L9_MQK = 0x40000

LA0 = 0x0
RA0 = 0x0
CA0 = 0x0
L6_LB_OP = 0x0
L6_LB_OP2 = 0xC000

L6_LC_W = 0x10000

L6_LC_U = 0x18000

P3_UB_X = 0x0A000
P3_UB_XY = 0x0C000

P3_L1_Y0 = 0x24000
P3_L1_Y1 = 0x28000
P3_L1_X = 0x2C000
L7_LC_UD = 0x0
L7_LC_DEL = 0x10000


@pl.vector_function
def m1_gate_vf(x_b, dtb, a_row, out_f, lb_row):
    """S1: gate = lb * sigmoid(exp(a_log[hv]) * (g + dt_bias[hv]))."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    a_b = vf.load_align(a_row, 0, dist=pl.LoadDist.BRC_B32)
    a_h = vf.exp(a_b, preg)
    lb_b = vf.load_align(lb_row, 0, dist=pl.LoadDist.BRC_B32)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    dr0 = vf.load_align(dtb, 0)
    dr1 = vf.load_align(dtb, LANES)
    for m in pl.range(0, HS):
        base = m * HEAD_D
        bg = vf.load_align(x_b, base, dtype=pl.DT_BF16)
        e = vf.astype(bg, pregbf, dtype=pl.DT_FP32)
        o_ = vf.astype(bg, pregbf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ONE)
        gr0, gr1 = vf.interleave(e, o_)
        sm = vf.add(gr0, dr0, preg)
        tm = vf.mul(sm, a_h, preg)
        em = vf.exp_sub(zero, tm, preg)
        dn = vf.adds(em, 1.0, preg)
        sg = vf.div(one, dn, preg)
        gl = vf.mul(sg, lb_b, preg)
        vf.store_align(out_f + base, gl, preg)
        sm = vf.add(gr1, dr1, preg)
        tm = vf.mul(sm, a_h, preg)
        em = vf.exp_sub(zero, tm, preg)
        dn = vf.adds(em, 1.0, preg)
        sg = vf.div(one, dn, preg)
        gl = vf.mul(sg, lb_b, preg)
        vf.store_align(out_f + base + LANES, gl, preg)


@pl.vector_function
def m1_l2norm_vf(x_b, out_b, eps):
    """Row L2 norm: x / sqrt(sum(x*x) + eps), BF16 in and out."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    for m in pl.range(0, HS):
        base = m * HEAD_D
        bx = vf.load_align(x_b, base, dtype=pl.DT_BF16)
        e = vf.astype(bx, pregbf, dtype=pl.DT_FP32)
        o_ = vf.astype(bx, pregbf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ONE)
        x0, x1 = vf.interleave(e, o_)
        s0 = vf.mul(x0, x0, preg)
        p0 = vf.reduce_sum(s0, preg)
        s1 = vf.mul(x1, x1, preg)
        p1 = vf.reduce_sum(s1, preg)
        rsum = vf.add(p0, p1, preg)
        denom = vf.full(rsum, preg)
        denom = vf.adds(denom, eps, preg)
        denom = vf.sqrt(denom, preg)
        y0 = vf.div(x0, denom, preg)
        y1 = vf.div(x1, denom, preg)
        b0 = vf.astype(y0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(y1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        d0, d1 = vf.de_interleave(b0, b1)
        vf.store_align(out_b + base, d0, pregbf)


@pl.vector_function
def m1_l2norm_cache_vf(x_b, out_b, rs, eps):
    """Row L2 norm: x / sqrt(sum(x*x) + eps), BF16 in and out."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    for m in pl.range(0, HS):
        base = m * HEAD_D
        bx = vf.load_align(x_b, base, dtype=pl.DT_BF16)
        e = vf.astype(bx, pregbf, dtype=pl.DT_FP32)
        o_ = vf.astype(bx, pregbf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ONE)
        x0, x1 = vf.interleave(e, o_)
        s0 = vf.mul(x0, x0, preg)
        p0 = vf.reduce_sum(s0, preg)
        s1 = vf.mul(x1, x1, preg)
        p1 = vf.reduce_sum(s1, preg)
        rsum = vf.add(p0, p1, preg)
        denom = vf.full(rsum, preg)
        denom = vf.adds(denom, eps, preg)
        denom = vf.sqrt(denom, preg)
        y0 = vf.div(x0, denom, preg)
        y1 = vf.div(x1, denom, preg)
        one = vf.full(1.0, preg, dtype=pl.DT_FP32)
        inverse = vf.div(one, denom, preg)
        vf.store_align(rs + base, inverse, preg)
        b0 = vf.astype(y0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(y1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        d0, d1 = vf.de_interleave(b0, b1)
        vf.store_align(out_b + base, d0, pregbf)


@pl.vector_function
def m3_qd_vf(qn_b, cs_f, sc_row, out_b, r0):
    """S4: qd = qn * exp(cs) * scale — bf16 in -> bf16 out (v6: both pl.casts"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    sc_b = vf.load_align(sc_row, 0, dist=pl.LoadDist.BRC_B32)
    for m in pl.range(0, HS):
        base = m * HEAD_D
        cbase = (r0 + m) * HEAD_D
        bq = vf.load_align(qn_b, base, dtype=pl.DT_BF16)
        e = vf.astype(bq, pregbf, dtype=pl.DT_FP32)
        o_ = vf.astype(bq, pregbf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ONE)
        q0, q1 = vf.interleave(e, o_)
        c0 = vf.load_align(cs_f, cbase)
        c1 = vf.load_align(cs_f, cbase + LANES)
        e0 = vf.exp(c0, preg)
        e1 = vf.exp(c1, preg)
        t0 = vf.mul(q0, e0, preg)
        t1 = vf.mul(q1, e1, preg)
        d0v = vf.mul(t0, sc_b, preg)
        d1v = vf.mul(t1, sc_b, preg)
        b0 = vf.astype(d0v, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(d1v, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        drow, d1_ = vf.de_interleave(b0, b1)
        vf.store_align(out_b + base, drow, pregbf)


@pl.vector_function
def m3_kr_vf(kn_b, cs_f, csl_f, out_b, r0):
    """S5: kr = kn * exp(cs_last - cs) — bf16 in -> bf16 out (v6: both"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    l0 = vf.load_align(csl_f, 0)
    l1 = vf.load_align(csl_f, LANES)
    for m in pl.range(0, HS):
        base = m * HEAD_D
        cbase = (r0 + m) * HEAD_D
        bk = vf.load_align(kn_b, base, dtype=pl.DT_BF16)
        e = vf.astype(bk, pregbf, dtype=pl.DT_FP32)
        o_ = vf.astype(bk, pregbf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ONE)
        k0, k1 = vf.interleave(e, o_)
        c0 = vf.load_align(cs_f, cbase)
        c1 = vf.load_align(cs_f, cbase + LANES)
        t0_e = vf.exp_sub(l0, c0, preg)
        t0 = vf.mul(k0, t0_e, preg)
        t1_e = vf.exp_sub(l1, c1, preg)
        t1 = vf.mul(k1, t1_e, preg)
        b0 = vf.astype(t0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(t1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        drow, d1_ = vf.de_interleave(b0, b1)
        vf.store_align(out_b + base, drow, pregbf)


@pl.vector_function
def m3_gam_vf(csl_f, gam_f):
    """S6: gam = exp(cs_last).  May legitimately underflow to 0 (no clamp)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 2):
        off = r * LANES
        l_r = vf.load_align(csl_f, off)
        e_r = vf.exp(l_r, preg)
        vf.store_align(gam_f + off, e_r, preg)


@pl.vector_function
def m4_ref_row_vf(cs_f, ref_f, row):
    """Extract cs row ``row`` (16*rb row-block base) into a [1,128] ref tile."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 2):
        off = row * HEAD_D + r * LANES
        v = vf.load_align(cs_f, off)
        vf.store_align(ref_f + r * LANES, v, preg)


@pl.vector_function
def m4_lop_vf(src_f, cs_f, ref_f, out_b, r0):
    """l_op = src[r0:r0+16] * exp(cs[r0:r0+16] - ref) -> out rows 0..15 (compacted)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    rf0 = vf.load_align(ref_f, 0)
    rf1 = vf.load_align(ref_f, LANES)
    for m in pl.range(0, ROW_BLOCK):
        sb = (r0 + m) * HEAD_D
        ob = m * HEAD_D
        sb16 = vf.load_align(src_f, sb, dtype=pl.DT_BF16)
        se0 = vf.astype(sb16, pregbf, dtype=pl.DT_FP32)
        se1 = vf.astype(sb16, pregbf, dtype=pl.DT_FP32,
                        layout=pl.CastLayout.ONE)
        s0, s1 = vf.interleave(se0, se1)
        c0 = vf.load_align(cs_f, sb)
        c1 = vf.load_align(cs_f, sb + LANES)
        t0_e = vf.exp_sub(c0, rf0, preg)
        t0 = vf.mul(s0, t0_e, preg)
        t1_e = vf.exp_sub(c1, rf1, preg)
        t1 = vf.mul(s1, t1_e, preg)
        b0 = vf.astype(t0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(t1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        drow, d1_ = vf.de_interleave(b0, b1)
        vf.store_align(out_b + ob, drow, pregbf)


@pl.vector_function
def m4_rop_chunk_vf(src_f, cs_f, ref_f, out_b, r0c):
    """r_op 16-row chunk: out rows 0..15 = src[r0c..r0c+16) * exp(ref - cs[same])."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    rf0 = vf.load_align(ref_f, 0)
    rf1 = vf.load_align(ref_f, LANES)
    for m in pl.range(0, ROW_BLOCK):
        sb = (r0c + m) * HEAD_D
        ob = m * HEAD_D
        sb16 = vf.load_align(src_f, sb, dtype=pl.DT_BF16)
        se0 = vf.astype(sb16, pregbf, dtype=pl.DT_FP32)
        se1 = vf.astype(sb16, pregbf, dtype=pl.DT_FP32,
                        layout=pl.CastLayout.ONE)
        s0, s1 = vf.interleave(se0, se1)
        c0 = vf.load_align(cs_f, sb)
        c1 = vf.load_align(cs_f, sb + LANES)
        t0_e = vf.exp_sub(rf0, c0, preg)
        t0 = vf.mul(s0, t0_e, preg)
        t1_e = vf.exp_sub(rf1, c1, preg)
        t1 = vf.mul(s1, t1_e, preg)
        b0 = vf.astype(t0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(t1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        drow, d1_ = vf.de_interleave(b0, b1)
        vf.store_align(out_b + ob, drow, pregbf)


@pl.vector_function
def m4_lqop_vf(src_f, cs_f, ref_f, out_b, r0, sc_f):
    """lq_op = src[r0:r0+16] * exp(cs[block] - ref) * scale -> out rows 0..15."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    sc_b = vf.load_align(sc_f, 0, dist=pl.LoadDist.BRC_B32)
    rf0 = vf.load_align(ref_f, 0)
    rf1 = vf.load_align(ref_f, LANES)
    for m in pl.range(0, ROW_BLOCK):
        sb = (r0 + m) * HEAD_D
        ob = m * HEAD_D
        sb16 = vf.load_align(src_f, sb, dtype=pl.DT_BF16)
        se0 = vf.astype(sb16, pregbf, dtype=pl.DT_FP32)
        se1 = vf.astype(sb16, pregbf, dtype=pl.DT_FP32,
                        layout=pl.CastLayout.ONE)
        s0, s1 = vf.interleave(se0, se1)
        c0 = vf.load_align(cs_f, sb)
        c1 = vf.load_align(cs_f, sb + LANES)
        t0_e = vf.exp_sub(c0, rf0, preg)
        t0 = vf.mul(s0, t0_e, preg)
        t1_e = vf.exp_sub(c1, rf1, preg)
        t1 = vf.mul(s1, t1_e, preg)
        t0 = vf.mul(t0, sc_b, preg)
        t1 = vf.mul(t1, sc_b, preg)
        b0 = vf.astype(t0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(t1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        drow, d1_ = vf.de_interleave(b0, b1)
        vf.store_align(out_b + ob, drow, pregbf)


@pl.vector_function
def m5_tmqk_vf(kkt_f, mask_b, mqk_f, beta_blk_f, tb_out_f, mqk_out_b,
               off_base, step, r0):
    """v6 FUSED pair-VF: t_beta (S8 finish) + mqk (S12 finish) in ONE pass."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    neg1 = vf.full(-1.0, preg, dtype=pl.DT_FP32)
    for p in pl.range(0, HS // 2):
        gr0 = r0 + 2 * p
        gr1 = gr0 + 1
        base0 = 2 * p * CHUNK_C
        base1 = (2 * p + 1) * CHUNK_C
        brow = vf.load_align(beta_blk_f, off_base + gr0 * step,
                             dist=pl.LoadDist.BRC_B32)
        e = vf.exp_sub(zero, brow, preg)
        dn = vf.adds(e, 1.0, preg)
        sg0 = vf.div(one, dn, preg)
        mb = vf.load_align(mask_b, gr0 * HEAD_D, dtype=pl.DT_BF16)
        me0 = vf.astype(mb, pregbf, dtype=pl.DT_FP32)
        me1 = vf.astype(mb, pregbf, dtype=pl.DT_FP32,
                        layout=pl.CastLayout.ONE)
        slrow, carow = vf.interleave(me0, me1)

        mq0 = vf.load_align(mqk_f, base0)
        gt = vf.gt(carow, 0.0, preg)
        mq0 = vf.select(mq0, zero, gt)
        krow = vf.load_align(kkt_f, base0)
        t = vf.mul(krow, sg0, preg)
        t = vf.mul(t, neg1, preg)
        gt = vf.gt(slrow, 0.0, preg)
        t = vf.select(t, zero, gt)
        vf.store_align(tb_out_f + base0, t, preg)
        brow = vf.load_align(beta_blk_f, off_base + gr1 * step,
                             dist=pl.LoadDist.BRC_B32)
        e = vf.exp_sub(zero, brow, preg)
        dn = vf.adds(e, 1.0, preg)
        sg1 = vf.div(one, dn, preg)
        mb = vf.load_align(mask_b, gr1 * HEAD_D, dtype=pl.DT_BF16)
        me0 = vf.astype(mb, pregbf, dtype=pl.DT_FP32)
        me1 = vf.astype(mb, pregbf, dtype=pl.DT_FP32,
                        layout=pl.CastLayout.ONE)
        slrow, carow = vf.interleave(me0, me1)
        mq1 = vf.load_align(mqk_f, base1)
        gt = vf.gt(carow, 0.0, preg)
        mq1 = vf.select(mq1, zero, gt)
        krow = vf.load_align(kkt_f, base1)
        t = vf.mul(krow, sg1, preg)
        t = vf.mul(t, neg1, preg)
        gt = vf.gt(slrow, 0.0, preg)
        t = vf.select(t, zero, gt)
        vf.store_align(tb_out_f + base1, t, preg)
        b0 = vf.astype(mq0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(mq1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        drow, d1_ = vf.de_interleave(b0, b1)
        vf.store_align(mqk_out_b + p * 2 * CHUNK_C, drow, pregbf)


@pl.vector_function
def m5_vkn_vf(v_b, kn_b, cs_f, beta_blk_f, v_out_b, kd_out_b,
              beta_cache_b, off_base, step, r0, save_cache):
    """Split beta*V into BF16 high/low parts; also produce BF16 -beta*K*exp(cs).

    Reuse each consumed row of v_b for the low part. The caller stores both
    parts before the M6 handoff; no extra UB tile or mutex is needed.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    neg1 = vf.full(-1.0, preg, dtype=pl.DT_FP32)
    scalar = vf.update_mask(1, dtype=pl.DT_FP32)
    index_mask = vf.update_mask(1, dtype=pl.DT_UINT32)
    for m in pl.range(0, HS):
        base = m * HEAD_D
        brow = vf.load_align(beta_blk_f, off_base + (r0 + m) * step,
                             dist=pl.LoadDist.BRC_B32)
        e = vf.exp_sub(zero, brow, preg)
        dn = vf.adds(e, 1.0, preg)
        sg = vf.div(one, dn, preg)
        if save_cache:
            idx = vf.full(m, index_mask, dtype=pl.DT_UINT32)
            vf.scatter(beta_cache_b, sg, idx, scalar)
        bv = vf.load_align(v_b, base, dtype=pl.DT_BF16)
        e0 = vf.astype(bv, pregbf, dtype=pl.DT_FP32)
        e1 = vf.astype(bv, pregbf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ONE)
        vr0, vr1 = vf.interleave(e0, e1)
        t0 = vf.mul(vr0, sg, preg)
        t1 = vf.mul(vr1, sg, preg)
        b0 = vf.astype(t0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(t1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        packed, unused = vf.de_interleave(b0, b1)
        vf.store_align(v_out_b + base, packed, pregbf)
        # Keep the rounding residual while t0/t1 still hold beta*V in FP32.
        hi0 = vf.astype(b0, pregbf, dtype=pl.DT_FP32)
        hi1 = vf.astype(b1, pregbf, dtype=pl.DT_FP32)
        lo0 = vf.sub(t0, hi0, preg)
        lo1 = vf.sub(t1, hi1, preg)
        lb0 = vf.astype(lo0, pregbf, dtype=pl.DT_BF16,
                        round_mode=pl.VFRoundMode.CAST_ROUND)
        lb1 = vf.astype(lo1, pregbf, dtype=pl.DT_BF16,
                        round_mode=pl.VFRoundMode.CAST_ROUND)
        packed_lo, unused_lo = vf.de_interleave(lb0, lb1)
        vf.store_align(v_b + base, packed_lo, pregbf)
        bk = vf.load_align(kn_b, base, dtype=pl.DT_BF16)
        e0 = vf.astype(bk, pregbf, dtype=pl.DT_FP32)
        e1 = vf.astype(bk, pregbf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ONE)
        k0, k1 = vf.interleave(e0, e1)
        c0 = vf.load_align(cs_f, (r0 + m) * HEAD_D)
        c1 = vf.load_align(cs_f, (r0 + m) * HEAD_D + LANES)
        e0 = vf.exp(c0, preg)
        e1 = vf.exp(c1, preg)
        t0 = vf.mul(k0, e0, preg)
        t1 = vf.mul(k1, e1, preg)
        t0 = vf.mul(t0, sg, preg)
        t1 = vf.mul(t1, sg, preg)
        t0 = vf.mul(t0, neg1, preg)
        t1 = vf.mul(t1, neg1, preg)
        b0 = vf.astype(t0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(t1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        packed, unused = vf.de_interleave(b0, b1)
        vf.store_align(kd_out_b + base, packed, pregbf)


@pl.vector_function
def m7_ut_add_upre_bf16_vf(ud_t, up_f, out_b, row0, rows):
    """Add FP32 ud/upre, then round u once to BF16 for the next matmul."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    for m2 in pl.range(0, (rows // 2) // 2):
        off = m2 * (2 * CHUNK_C)
        r0 = vf.load_align(up_f, off)
        r1 = vf.load_align(up_f, off + CHUNK_C)
        a0 = vf.load_align(ud_t, (row0 + 2 * m2) * CHUNK_C)
        a1 = vf.load_align(ud_t, (row0 + 2 * m2 + 1) * CHUNK_C)
        s0 = vf.add(a0, r0, preg)
        s1 = vf.add(a1, r1, preg)
        b0 = vf.astype(s0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(s1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        drow, d1_ = vf.de_interleave(b0, b1)
        vf.store_align(out_b + off, drow, pregbf)


@pl.vector_function
def nd2nz_sg(tile, out, m):
    """ND→NZ row re-assembly for the P3 X mirror (probe L60-67 verbatim)."""
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for i in pl.range(m):
        inreg = vf.load_align(tile, i * CHUNK_C)
        vf.store_align(out, inreg, preg_all,
                       block_stride=BLOCK_STRIDE_ND, repeat_stride=REPEAT_STRIDE_ND,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)


def m1_prefetch_v(q_v, k_v, g_v, b, hk, hv, s0, x_b_g, xk_g, xg_g):
    """v13 manual preload: issue the NEXT work's q/k/g GM loads so they fly"""
    sub = pl.get_subblock_idx()
    r0 = sub * HS
    xq = x_b_g.next()
    xk = xk_g.next()
    xg = xg_g.next()
    pl.load(xq, q_v, [b, hk, s0 + r0, 0], order=[2, 3])
    pl.load(xk, k_v, [b, hk, s0 + r0, 0], order=[2, 3])
    pl.load(xg, g_v, [b, hv, s0 + r0, 0], order=[2, 3])


def m1_gate_norm_v(qn_o, kn_o, a_view, lb_view, dt_bias,
                   gate_l1, gate_slot_idx, b, hv, hk, s0, have_p2,
                   x_b_g, xk_g, xg_g, qn_h_g, kn_h_g, gate_f_g, gate_nz_g,
                   a_g, d_g, lb_g, q_c_o, k_c_o, c, save_cache, rstd_o):
    """Module-1 vector stage (delay 0): gate/qn/kn for one chunk of one (b,hv)."""
    sub = pl.get_subblock_idx()
    r0 = sub * HS
    if have_p2:
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=3)
    xq = x_b_g.current()
    xk = xk_g.current()
    xg = xg_g.current()
    a_s = a_g.next()
    pl.set_validshape(a_s, [1, 1])
    pl.load(a_s, a_view, [hv, 0])
    lb_s = lb_g.next()
    pl.set_validshape(lb_s, [1, 1])
    pl.load(lb_s, lb_view, [0, 0])
    d_s = d_g.next()
    pl.load(d_s, dt_bias, [hv, 0])
    gatel = gate_f_g.next()
    m1_gate_vf(xg, d_s, a_s, gatel, lb_s)
    gate_nz = gate_nz_g.next()
    pl.set_validshape(gate_nz, [HS, HEAD_D])
    pl.set_validshape(gatel, [HS, HEAD_D])
    pl.move(gate_nz, gatel)
    gate_slot = gate_l1[gate_slot_idx]
    pl.insert(gate_slot, gate_nz, [r0, 0])
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
    qnb = qn_h_g[gate_slot_idx]
    if save_cache:
        m1_l2norm_cache_vf(xq, qnb, gatel, L2_EPS)
        rs_view = pl.make_tensor(pl.addptr(pl.make_ptr(rstd_o), ((
            b * rstd_o.shape[1] + c) * rstd_o.shape[2] + hk) * 128 + 0 * 64), [64, 1], [1, 1])
        pl.set_validshape(gatel, [HS, 1])
        pl.store(rs_view, gatel, [r0, 0])
        pl.set_validshape(gatel, [HS, HEAD_D])
    else:
        m1_l2norm_vf(xq, qnb, L2_EPS)
    pl.store(qn_o, qnb, [b, hk, s0 + r0, 0], order=[2, 3])
    if save_cache:
        pl.store(q_c_o, qnb, [b, c, hk, r0, 0], order=[3, 4])
    knb = kn_h_g[gate_slot_idx]
    if save_cache:
        m1_l2norm_cache_vf(xk, knb, gatel, L2_EPS)
        rs_view = pl.make_tensor(pl.addptr(pl.make_ptr(rstd_o), ((
            b * rstd_o.shape[1] + c) * rstd_o.shape[2] + hk) * 128 + 1 * 64), [64, 1], [1, 1])
        pl.set_validshape(gatel, [HS, 1])
        pl.store(rs_view, gatel, [r0, 0])
        pl.set_validshape(gatel, [HS, HEAD_D])
    else:
        m1_l2norm_vf(xk, knb, L2_EPS)
    pl.store(kn_o, knb, [b, hk, s0 + r0, 0], order=[2, 3])
    if save_cache:
        pl.store(k_c_o, knb, [b, c, hk, r0, 0], order=[3, 4])


def m1_gate_norm_v_cached(qn_o, kn_o, a_view, lb_view, dt_bias,
                   gate_l1, gate_slot_idx, b, hv, hk, s0, have_p2,
                   x_b_g, xk_g, xg_g, qn_h_g, kn_h_g, gate_f_g, gate_nz_g,
                   a_g, d_g, lb_g, q_c_o, k_c_o, c, save_cache, rstd_o):
    """Module-1 vector stage (delay 0): gate/qn/kn for one chunk of one (b,hv)."""
    sub = pl.get_subblock_idx()
    r0 = sub * HS
    if have_p2:
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=3)
    xq = x_b_g.current()
    xk = xk_g.current()
    xg = xg_g.current()
    a_s = a_g.next()
    pl.set_validshape(a_s, [1, 1])
    pl.load(a_s, a_view, [hv, 0])
    lb_s = lb_g.next()
    pl.set_validshape(lb_s, [1, 1])
    pl.load(lb_s, lb_view, [0, 0])
    d_s = d_g.next()
    pl.load(d_s, dt_bias, [hv, 0])
    gatel = gate_f_g.next()
    m1_gate_vf(xg, d_s, a_s, gatel, lb_s)
    gate_nz = gate_nz_g.next()
    pl.set_validshape(gate_nz, [HS, HEAD_D])
    pl.set_validshape(gatel, [HS, HEAD_D])
    pl.move(gate_nz, gatel)
    gate_slot = gate_l1[gate_slot_idx]
    pl.insert(gate_slot, gate_nz, [r0, 0])
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
    qnb = qn_h_g[gate_slot_idx]
    if save_cache:
        m1_l2norm_cache_vf(xq, qnb, gatel, L2_EPS)
        rs_view = pl.make_tensor(pl.addptr(pl.make_ptr(rstd_o), ((
            b * rstd_o.shape[1] + c) * rstd_o.shape[2] + hk) * 128 + 0 * 64), [64, 1], [1, 1])
        pl.set_validshape(gatel, [HS, 1])
        pl.store(rs_view, gatel, [r0, 0])
        pl.set_validshape(gatel, [HS, HEAD_D])
    else:
        m1_l2norm_vf(xq, qnb, L2_EPS)
    if save_cache:
        pl.store(q_c_o, qnb, [b, c, hk, r0, 0], order=[3, 4])
    knb = kn_h_g[gate_slot_idx]
    if save_cache:
        m1_l2norm_cache_vf(xk, knb, gatel, L2_EPS)
        rs_view = pl.make_tensor(pl.addptr(pl.make_ptr(rstd_o), ((
            b * rstd_o.shape[1] + c) * rstd_o.shape[2] + hk) * 128 + 1 * 64), [64, 1], [1, 1])
        pl.set_validshape(gatel, [HS, 1])
        pl.store(rs_view, gatel, [r0, 0])
        pl.set_validshape(gatel, [HS, HEAD_D])
    else:
        m1_l2norm_vf(xk, knb, L2_EPS)
    if save_cache:
        pl.store(k_c_o, knb, [b, c, hk, r0, 0], order=[3, 4])
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=6)

def m2_cumsum_c_cached(gate_l1, cs_g, gate_slot_idx,
                have_prev, lower_l1_g, lower_l0a_g, g_l0b_g, acc_g, save_cache):
    """Module-2 cube body (v13 manual pipeline): cs = L_tril @ gate."""
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=0)
    if have_prev:
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=7)
    gate_slot = gate_l1[gate_slot_idx]
    l_l1 = lower_l1_g.next()
    l_a = lower_l0a_g.next()
    pl.move(l_a, l_l1)
    g_r = g_l0b_g.next()
    pl.move(g_r, gate_slot)
    acc = acc_g.next()
    pl.matmul(acc, l_a, g_r,
              phase=pl.AccPhase.Final)
    pl.set_validshape(acc, [CHUNK_C, HEAD_D])
    cs_slot = cs_g[0]
    pl.set_validshape(cs_slot, [CHUNK_C, HEAD_D])
    if have_prev:
        if save_cache:
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE3, event_id=_M3_GK_DONE,
                sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=_M5_CS_DONE)
    pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=6)
    pl.move(cs_slot, acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0,
            phase=pl.STPhase.Partial)
    pl.move(cs_slot, acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec1,
            phase=pl.STPhase.Final)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=2)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=3)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=11)


def m3_qd_kr_gam_v_cached(qn_o, kn_o, qd_o, kr_o, gam_o, cs_g, scale_view,
                   c, b, hv, hk, s0, gate_slot_idx, num_value_heads, nc,
                   qn_h_g, kn_h_g, m3_out_b_g, m3_csl_g, m3_sc_g,
                   m4_kn_b_g, m4_qn_b_g,
                   m4_opscr_b16_g, m4_ref_g,
                   lk0_g, lk1_g, lk2_g, lk3_g,
                   lq0_g, lq1_g, lq2_g, lq3_g,
                   rw0_g, rw1_g, rw2_g, rw3_g,
                   opnz_g, q_c_o, k_c_o, gk_c_o, save_cache):
    """Module-3 vector stage (delay 2): qd/kr/gam + S8/S12 operand pre-pass."""
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=2)
    cs_slot = cs_g[0]
    pl.set_validshape(cs_slot, [CHUNK_C, HEAD_D])
    sub_m3 = pl.get_subblock_idx()
    r0 = sub_m3 * HS
    csl = m3_csl_g.next()
    pl.set_validshape(csl, [1, HEAD_D])
    pl.move(csl, cs_slot, offset=[CHUNK_C - 1, 0])
    sc_s = m3_sc_g.next()
    pl.set_validshape(sc_s, [1, 1])
    pl.load(sc_s, scale_view, [0, 0])

    qnb = qn_h_g[gate_slot_idx]
    qd_b = m3_out_b_g.next()
    m3_qd_vf(qnb, cs_slot, sc_s, qd_b, r0)
    pl.store(qd_o, qd_b, [b, hv, s0 + r0, 0], order=[2, 3])

    knb = kn_h_g[gate_slot_idx]
    kr_b = m3_out_b_g.next()
    m3_kr_vf(knb, cs_slot, csl, kr_b, r0)
    pl.store(kr_o, kr_b, [b, hv, s0 + r0, 0], order=[2, 3])

    m3_gam_vf(csl, csl)
    pl.store(gam_o, csl, [b, hv, c, 0], order=[2, 3])

    knb4 = m4_kn_b_g.next()
    pl.load(knb4, k_c_o, [b, c, hk, 0, 0], order=[3, 4])
    kn4 = knb4
    qnb4 = m4_qn_b_g.next()
    pl.load(qnb4, q_c_o, [b, c, hk, 0, 0], order=[3, 4])
    qn4 = qnb4
    sub_m3 = pl.get_subblock_idx()
    ref0 = m4_ref_g.next()
    ref1 = m4_ref_g.next()
    ref2 = m4_ref_g.next()
    ref3 = m4_ref_g.next()
    scrb0 = m4_opscr_b16_g[0]
    scrb1 = m4_opscr_b16_g[1]
    scrb2 = m4_opscr_b16_g[0]
    # Balance the triangular right-operand work: (1+4) and (2+3) row blocks.
    # Both subblocks already hold full cs, qn and kn; destinations are unchanged.
    if sub_m3 == 0:
        r0v = 0 * ROW_BLOCK
        pl.set_validshape(ref0, [1, HEAD_D])
        m4_ref_row_vf(cs_slot, ref0, r0v)
        m4_lop_vf(kn4, cs_slot, ref0, scrb0, r0v)

        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb0, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb0)
        dst_s = lk0_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref0, scrb1, 0 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb1, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb1)
        dst_s = rw0_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0 * ROW_BLOCK, 0])
        m4_lqop_vf(qn4, cs_slot, ref0, scrb2, r0v, sc_s)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb2, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb2)
        dst_s = lq0_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
        r3v = 3 * ROW_BLOCK
        pl.set_validshape(ref3, [1, HEAD_D])
        m4_ref_row_vf(cs_slot, ref3, r3v)
        m4_lop_vf(kn4, cs_slot, ref3, scrb0, r3v)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb0, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb0)
        dst_s = lk3_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref3, scrb1, 0 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb1, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb1)
        dst_s = rw3_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0 * ROW_BLOCK, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref3, scrb2, 1 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb2, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb2)
        dst_s = rw3_g[gate_slot_idx]
        pl.insert(dst_s, snz, [1 * ROW_BLOCK, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref3, scrb0, 2 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb0, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb0)
        dst_s = rw3_g[gate_slot_idx]
        pl.insert(dst_s, snz, [2 * ROW_BLOCK, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref3, scrb1, 3 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb1, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb1)
        dst_s = rw3_g[gate_slot_idx]
        pl.insert(dst_s, snz, [3 * ROW_BLOCK, 0])
        m4_lqop_vf(qn4, cs_slot, ref3, scrb2, r3v, sc_s)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb2, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb2)
        dst_s = lq3_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
    else:
        r1v = 1 * ROW_BLOCK
        pl.set_validshape(ref1, [1, HEAD_D])
        m4_ref_row_vf(cs_slot, ref1, r1v)
        m4_lop_vf(kn4, cs_slot, ref1, scrb0, r1v)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb0, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb0)
        dst_s = lk1_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref1, scrb1, 0 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb1, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb1)
        dst_s = rw1_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0 * ROW_BLOCK, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref1, scrb2, 1 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb2, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb2)
        dst_s = rw1_g[gate_slot_idx]
        pl.insert(dst_s, snz, [1 * ROW_BLOCK, 0])
        m4_lqop_vf(qn4, cs_slot, ref1, scrb0, r1v, sc_s)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb0, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb0)
        dst_s = lq1_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
        r2v = 2 * ROW_BLOCK
        pl.set_validshape(ref2, [1, HEAD_D])
        m4_ref_row_vf(cs_slot, ref2, r2v)
        m4_lop_vf(kn4, cs_slot, ref2, scrb1, r2v)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb1, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb1)
        dst_s = lk2_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref2, scrb2, 0 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb2, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb2)
        dst_s = rw2_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0 * ROW_BLOCK, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref2, scrb0, 1 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb0, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb0)
        dst_s = rw2_g[gate_slot_idx]
        pl.insert(dst_s, snz, [1 * ROW_BLOCK, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref2, scrb1, 2 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb1, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb1)
        dst_s = rw2_g[gate_slot_idx]
        pl.insert(dst_s, snz, [2 * ROW_BLOCK, 0])
        m4_lqop_vf(qn4, cs_slot, ref2, scrb2, r2v, sc_s)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb2, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb2)
        dst_s = lq2_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
    if save_cache and sub_m3 == 0:
        pl.store(gk_c_o, cs_slot, [b, c, hv, 0, 0], order=[3, 4])
        pl.system.set_cross_core(
            pipe=pl.PipeType.MTE3, event_id=_M3_GK_DONE,
            sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=7)

    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=10)


def m2_cumsum_c(gate_l1, cs_g, gate_slot_idx,
                have_prev, lower_l1_g, lower_l0a_g, g_l0b_g, acc_g, save_cache):
    """Module-2 cube body (v13 manual pipeline): cs = L_tril @ gate."""
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=0)
    if have_prev:
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=7)
    gate_slot = gate_l1[gate_slot_idx]
    l_l1 = lower_l1_g.next()
    l_a = lower_l0a_g.next()
    pl.move(l_a, l_l1)
    g_r = g_l0b_g.next()
    pl.move(g_r, gate_slot)
    acc = acc_g.next()
    pl.matmul(acc, l_a, g_r,
              phase=pl.AccPhase.Final)
    pl.set_validshape(acc, [CHUNK_C, HEAD_D])
    cs_slot = cs_g[0]
    pl.set_validshape(cs_slot, [CHUNK_C, HEAD_D])
    if have_prev:
        if save_cache:
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE3, event_id=_M3_GK_DONE,
                sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=_M5_CS_DONE)
    pl.move(cs_slot, acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0,
            phase=pl.STPhase.Partial)
    pl.move(cs_slot, acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec1,
            phase=pl.STPhase.Final)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=2)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=3)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=11)


def m3_qd_kr_gam_v(qn_o, kn_o, qd_o, kr_o, gam_o, cs_g, scale_view,
                   c, b, hv, hk, s0, gate_slot_idx, num_value_heads, nc,
                   qn_h_g, kn_h_g, m3_out_b_g, m3_csl_g, m3_sc_g,
                   m4_kn_b_g, m4_qn_b_g,
                   m4_opscr_b16_g, m4_ref_g,
                   lk0_g, lk1_g, lk2_g, lk3_g,
                   lq0_g, lq1_g, lq2_g, lq3_g,
                   rw0_g, rw1_g, rw2_g, rw3_g,
                   opnz_g, gk_c_o, save_cache):
    """Module-3 vector stage (delay 2): qd/kr/gam + S8/S12 operand pre-pass."""
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=2)
    cs_slot = cs_g[0]
    pl.set_validshape(cs_slot, [CHUNK_C, HEAD_D])
    sub_m3 = pl.get_subblock_idx()
    r0 = sub_m3 * HS
    csl = m3_csl_g.next()
    pl.set_validshape(csl, [1, HEAD_D])
    pl.move(csl, cs_slot, offset=[CHUNK_C - 1, 0])
    sc_s = m3_sc_g.next()
    pl.set_validshape(sc_s, [1, 1])
    pl.load(sc_s, scale_view, [0, 0])

    qnb = qn_h_g[gate_slot_idx]
    qd_b = m3_out_b_g.next()
    m3_qd_vf(qnb, cs_slot, sc_s, qd_b, r0)
    pl.store(qd_o, qd_b, [b, hv, s0 + r0, 0], order=[2, 3])

    knb = kn_h_g[gate_slot_idx]
    kr_b = m3_out_b_g.next()
    m3_kr_vf(knb, cs_slot, csl, kr_b, r0)
    pl.store(kr_o, kr_b, [b, hv, s0 + r0, 0], order=[2, 3])

    m3_gam_vf(csl, csl)
    pl.store(gam_o, csl, [b, hv, c, 0], order=[2, 3])

    knb4 = m4_kn_b_g.next()
    pl.load(knb4, kn_o, [b, hk, s0, 0], order=[2, 3])
    kn4 = knb4
    qnb4 = m4_qn_b_g.next()
    pl.load(qnb4, qn_o, [b, hk, s0, 0], order=[2, 3])
    qn4 = qnb4
    sub_m3 = pl.get_subblock_idx()
    ref0 = m4_ref_g.next()
    ref1 = m4_ref_g.next()
    ref2 = m4_ref_g.next()
    ref3 = m4_ref_g.next()
    scrb0 = m4_opscr_b16_g[0]
    scrb1 = m4_opscr_b16_g[1]
    scrb2 = m4_opscr_b16_g[0]
    if sub_m3 == 0:
        r0v = 0 * ROW_BLOCK
        pl.set_validshape(ref0, [1, HEAD_D])
        m4_ref_row_vf(cs_slot, ref0, r0v)
        m4_lop_vf(kn4, cs_slot, ref0, scrb0, r0v)

        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb0, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb0)
        dst_s = lk0_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref0, scrb1, 0 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb1, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb1)
        dst_s = rw0_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0 * ROW_BLOCK, 0])
        m4_lqop_vf(qn4, cs_slot, ref0, scrb2, r0v, sc_s)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb2, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb2)
        dst_s = lq0_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
        r1v = 1 * ROW_BLOCK
        pl.set_validshape(ref1, [1, HEAD_D])
        m4_ref_row_vf(cs_slot, ref1, r1v)
        m4_lop_vf(kn4, cs_slot, ref1, scrb0, r1v)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb0, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb0)
        dst_s = lk1_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref1, scrb1, 0 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb1, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb1)
        dst_s = rw1_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0 * ROW_BLOCK, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref1, scrb2, 1 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb2, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb2)
        dst_s = rw1_g[gate_slot_idx]
        pl.insert(dst_s, snz, [1 * ROW_BLOCK, 0])
        m4_lqop_vf(qn4, cs_slot, ref1, scrb0, r1v, sc_s)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb0, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb0)
        dst_s = lq1_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
    else:
        r2v = 2 * ROW_BLOCK
        pl.set_validshape(ref2, [1, HEAD_D])
        m4_ref_row_vf(cs_slot, ref2, r2v)
        m4_lop_vf(kn4, cs_slot, ref2, scrb1, r2v)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb1, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb1)
        dst_s = lk2_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref2, scrb2, 0 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb2, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb2)
        dst_s = rw2_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0 * ROW_BLOCK, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref2, scrb0, 1 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb0, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb0)
        dst_s = rw2_g[gate_slot_idx]
        pl.insert(dst_s, snz, [1 * ROW_BLOCK, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref2, scrb1, 2 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb1, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb1)
        dst_s = rw2_g[gate_slot_idx]
        pl.insert(dst_s, snz, [2 * ROW_BLOCK, 0])
        m4_lqop_vf(qn4, cs_slot, ref2, scrb2, r2v, sc_s)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb2, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb2)
        dst_s = lq2_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
        r3v = 3 * ROW_BLOCK
        pl.set_validshape(ref3, [1, HEAD_D])
        m4_ref_row_vf(cs_slot, ref3, r3v)
        m4_lop_vf(kn4, cs_slot, ref3, scrb0, r3v)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb0, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb0)
        dst_s = lk3_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref3, scrb1, 0 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb1, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb1)
        dst_s = rw3_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0 * ROW_BLOCK, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref3, scrb2, 1 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb2, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb2)
        dst_s = rw3_g[gate_slot_idx]
        pl.insert(dst_s, snz, [1 * ROW_BLOCK, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref3, scrb0, 2 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb0, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb0)
        dst_s = rw3_g[gate_slot_idx]
        pl.insert(dst_s, snz, [2 * ROW_BLOCK, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref3, scrb1, 3 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb1, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb1)
        dst_s = rw3_g[gate_slot_idx]
        pl.insert(dst_s, snz, [3 * ROW_BLOCK, 0])
        m4_lqop_vf(qn4, cs_slot, ref3, scrb2, r3v, sc_s)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb2, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb2)
        dst_s = lq3_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
    if save_cache and sub_m3 == 0:
        pl.store(gk_c_o, cs_slot, [b, c, hv, 0, 0], order=[3, 4])
        pl.system.set_cross_core(
            pipe=pl.PipeType.MTE3, event_id=_M3_GK_DONE,
            sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=7)

    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=10)


def m3_qd_kr_gam_v_local(qn_o, kn_o, qd_o, kr_o, gam_o, cs_g, scale_view,
                   c, b, hv, hk, s0, gate_slot_idx, num_value_heads, nc,
                   qn_h_g, kn_h_g, m3_out_b_g, m3_csl_g, m3_sc_g,
                   m4_kn_b_g, m4_qn_b_g,
                   m4_opscr_b16_g, m4_ref_g,
                   lk0_g, lk1_g, lk2_g, lk3_g,
                   lq0_g, lq1_g, lq2_g, lq3_g,
                   rw0_g, rw1_g, rw2_g, rw3_g,
                   opnz_g, gk_c_o, save_cache):
    """Module-3 vector stage (delay 2): qd/kr/gam + S8/S12 operand pre-pass."""
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=2)
    cs_slot = cs_g[0]
    pl.set_validshape(cs_slot, [CHUNK_C, HEAD_D])
    sub_m3 = pl.get_subblock_idx()
    r0 = sub_m3 * HS
    csl = m3_csl_g.next()
    pl.set_validshape(csl, [1, HEAD_D])
    pl.move(csl, cs_slot, offset=[CHUNK_C - 1, 0])
    sc_s = m3_sc_g.next()
    pl.set_validshape(sc_s, [1, 1])
    pl.load(sc_s, scale_view, [0, 0])

    qnb = qn_h_g[gate_slot_idx]
    qd_b = m3_out_b_g.next()
    m3_qd_vf(qnb, cs_slot, sc_s, qd_b, r0)
    pl.store(qd_o, qd_b, [b, hv, s0 + r0, 0], order=[2, 3])

    knb = kn_h_g[gate_slot_idx]
    kr_b = m3_out_b_g.next()
    m3_kr_vf(knb, cs_slot, csl, kr_b, r0)
    pl.store(kr_o, kr_b, [b, hv, s0 + r0, 0], order=[2, 3])

    m3_gam_vf(csl, csl)
    pl.store(gam_o, csl, [b, hv, c, 0], order=[2, 3])

    knb4 = m4_kn_b_g.next()
    pl.load(knb4, kn_o, [b, hk, s0, 0], order=[2, 3])
    kn4 = knb4
    qnb4 = m4_qn_b_g.next()
    pl.load(qnb4, qn_o, [b, hk, s0, 0], order=[2, 3])
    qn4 = qnb4
    sub_m3 = pl.get_subblock_idx()
    ref0 = m4_ref_g.next()
    ref1 = m4_ref_g.next()
    ref2 = m4_ref_g.next()
    ref3 = m4_ref_g.next()
    scrb0 = m4_opscr_b16_g[0]
    scrb1 = m4_opscr_b16_g[1]
    scrb2 = m4_opscr_b16_g[0]
    # Balance the triangular right-operand work: (1+4) and (2+3) row blocks.
    # Both subblocks already hold full cs, qn and kn; destinations are unchanged.
    if sub_m3 == 0:
        r0v = 0 * ROW_BLOCK
        pl.set_validshape(ref0, [1, HEAD_D])
        m4_ref_row_vf(cs_slot, ref0, r0v)
        m4_lop_vf(kn4, cs_slot, ref0, scrb0, r0v)

        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb0, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb0)
        dst_s = lk0_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref0, scrb1, 0 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb1, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb1)
        dst_s = rw0_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0 * ROW_BLOCK, 0])
        m4_lqop_vf(qn4, cs_slot, ref0, scrb2, r0v, sc_s)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb2, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb2)
        dst_s = lq0_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
        r3v = 3 * ROW_BLOCK
        pl.set_validshape(ref3, [1, HEAD_D])
        m4_ref_row_vf(cs_slot, ref3, r3v)
        m4_lop_vf(kn4, cs_slot, ref3, scrb0, r3v)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb0, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb0)
        dst_s = lk3_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref3, scrb1, 0 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb1, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb1)
        dst_s = rw3_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0 * ROW_BLOCK, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref3, scrb2, 1 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb2, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb2)
        dst_s = rw3_g[gate_slot_idx]
        pl.insert(dst_s, snz, [1 * ROW_BLOCK, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref3, scrb0, 2 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb0, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb0)
        dst_s = rw3_g[gate_slot_idx]
        pl.insert(dst_s, snz, [2 * ROW_BLOCK, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref3, scrb1, 3 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb1, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb1)
        dst_s = rw3_g[gate_slot_idx]
        pl.insert(dst_s, snz, [3 * ROW_BLOCK, 0])
        m4_lqop_vf(qn4, cs_slot, ref3, scrb2, r3v, sc_s)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb2, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb2)
        dst_s = lq3_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
    else:
        r1v = 1 * ROW_BLOCK
        pl.set_validshape(ref1, [1, HEAD_D])
        m4_ref_row_vf(cs_slot, ref1, r1v)
        m4_lop_vf(kn4, cs_slot, ref1, scrb0, r1v)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb0, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb0)
        dst_s = lk1_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref1, scrb1, 0 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb1, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb1)
        dst_s = rw1_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0 * ROW_BLOCK, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref1, scrb2, 1 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb2, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb2)
        dst_s = rw1_g[gate_slot_idx]
        pl.insert(dst_s, snz, [1 * ROW_BLOCK, 0])
        m4_lqop_vf(qn4, cs_slot, ref1, scrb0, r1v, sc_s)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb0, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb0)
        dst_s = lq1_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
        r2v = 2 * ROW_BLOCK
        pl.set_validshape(ref2, [1, HEAD_D])
        m4_ref_row_vf(cs_slot, ref2, r2v)
        m4_lop_vf(kn4, cs_slot, ref2, scrb1, r2v)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb1, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb1)
        dst_s = lk2_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref2, scrb2, 0 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb2, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb2)
        dst_s = rw2_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0 * ROW_BLOCK, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref2, scrb0, 1 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb0, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb0)
        dst_s = rw2_g[gate_slot_idx]
        pl.insert(dst_s, snz, [1 * ROW_BLOCK, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref2, scrb1, 2 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb1, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb1)
        dst_s = rw2_g[gate_slot_idx]
        pl.insert(dst_s, snz, [2 * ROW_BLOCK, 0])
        m4_lqop_vf(qn4, cs_slot, ref2, scrb2, r2v, sc_s)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb2, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb2)
        dst_s = lq2_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
    if save_cache and sub_m3 == 0:
        pl.store(gk_c_o, cs_slot, [b, c, hv, 0, 0], order=[3, 4])
        pl.system.set_cross_core(
            pipe=pl.PipeType.MTE3, event_id=_M3_GK_DONE,
            sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=7)

    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=10)


def m4_kkt_mqk_blocks(kt_g, mq_g, pslot,
                      lk0_g, lk1_g, lk2_g, lk3_g,
                      lq0_g, lq1_g, lq2_g, lq3_g,
                      rz0_g, rz1_g, rz2_g, rz3_g,
                      m4_la_g, m4_b_g, m4_c_g):
    """Module-4 cube body (T70F = T70E blocks + Acc->Vec delivery): kkt/mqk[rb]"""
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=10)
    pl.system.set_mm_layout_transform(enabled=True)
    kt_t = kt_g[0]
    mq_t = mq_g[0]
    pl.set_validshape(kt_t, [HS, CHUNK_C])
    pl.set_validshape(mq_t, [HS, CHUNK_C])
    for rb in pl.range(0, 4):
        a_t = m4_la_g.next()
        lk_s = lk0_g[pslot] if rb == 0 else (lk1_g[pslot] if rb == 1 else (lk2_g[pslot] if rb == 2 else lk3_g[pslot]))
        lq_s = lq0_g[pslot] if rb == 0 else (lq1_g[pslot] if rb == 1 else (lq2_g[pslot] if rb == 2 else lq3_g[pslot]))
        b_full = rz0_g[pslot] if rb == 0 else (rz1_g[pslot] if rb == 1 else (rz2_g[pslot] if rb == 2 else rz3_g[pslot]))
        rs = (rb % 2) * ROW_BLOCK
        pl.move(a_t, lk_s)
        for j in pl.range(0, rb + 1):
            b_t = m4_b_g.next()
            c_t = m4_c_g.next()
            pl.move(b_t, b_full, [0, j * ROW_BLOCK])
            pl.matmul(c_t, a_t, b_t, phase=pl.AccPhase.Final)
            if rb < 2:
                pl.move(kt_t[rs:rs + ROW_BLOCK, j * ROW_BLOCK:(j + 1) * ROW_BLOCK],
                        c_t, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0,
                        phase=pl.STPhase.Final)
            else:
                pl.move(kt_t[rs:rs + ROW_BLOCK, j * ROW_BLOCK:(j + 1) * ROW_BLOCK],
                        c_t, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec1,
                        phase=pl.STPhase.Final)

        a_t = m4_la_g.next()
        pl.move(a_t, lq_s)
        for j in pl.range(0, rb + 1):
            b_t = m4_b_g.next()
            c_t = m4_c_g.next()
            pl.move(b_t, b_full, [0, j * ROW_BLOCK])
            pl.matmul(c_t, a_t, b_t, phase=pl.AccPhase.Final)
            if rb < 2:
                pl.move(mq_t[rs:rs + ROW_BLOCK, j * ROW_BLOCK:(j + 1) * ROW_BLOCK],
                        c_t, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0,
                        phase=pl.STPhase.Final)
            else:
                pl.move(mq_t[rs:rs + ROW_BLOCK, j * ROW_BLOCK:(j + 1) * ROW_BLOCK],
                        c_t, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec1,
                        phase=pl.STPhase.Final)
    pl.system.set_mm_layout_transform(enabled=False)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=9)


def t71_prepare_neumann_v(m5_mqk_g, t71_n, t71_eye, t71_zero,
                          t71_eye_bf16, t71_nd_h, t71_nz_h,
                          t71_nd_b, t71_nz_b, x_ub_g, eye64, reload_mask):
    """Prepare the Neumann seed tiles and publish event 13 early."""
    t71_sub = pl.get_subblock_idx()
    t71_off = t71_sub * HS
    pl.set_validshape(m5_mqk_g[0], [HS, CHUNK_C])
    pl.set_validshape(t71_nd_h, [HS, CHUNK_C])
    pl.set_validshape(t71_nz_h, [HS, CHUNK_C])
    pl.set_validshape(t71_nd_b, [HS, CHUNK_C])
    pl.set_validshape(t71_nz_b, [HS, CHUNK_C])
    pl.cast(t71_nd_h, m5_mqk_g[0], mode=pl.RoundMode.CAST_RINT)
    pl.move(t71_nz_h, t71_nd_h)
    pl.insert(t71_n, t71_nz_h, [t71_off, 0])
    if reload_mask:
        pl.set_validshape(x_ub_g.current(), [HS, CHUNK_C])
        pl.load(x_ub_g.current(), eye64, [t71_off, 0])
        pl.cast(t71_nd_h, x_ub_g.current(), mode=pl.RoundMode.CAST_RINT)
        pl.move(t71_nz_h, t71_nd_h)
        pl.insert(t71_eye, t71_nz_h, [t71_off, 0])
        pl.cast(t71_nd_b, x_ub_g.current(), mode=pl.RoundMode.CAST_RINT)
        pl.move(t71_nz_b, t71_nd_b)
        pl.insert(t71_eye_bf16, t71_nz_b, [t71_off, 0])
        pl.expands(t71_nd_h, 0.0)
        pl.move(t71_nz_h, t71_nd_h)
        pl.insert(t71_zero, t71_nz_h, [t71_off, 0])
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=13)


@pl.vector_function
def m5_beta_sigmoid_vf_local(beta_blk_f, beta_cache_b, off_base, step, r0, valid):
    """Pack this AIV's 32 beta values into lanes; preserve the scalar formula."""
    preg = vf.update_mask(HS, dtype=pl.DT_FP32)
    ipreg = vf.update_mask(HS, dtype=pl.DT_UINT32)
    rows = vf.arange(0, dtype=pl.DT_UINT32)
    global_rows = vf.adds(rows, r0, ipreg)
    indices = vf.muls(global_rows, step, ipreg)
    indices = vf.adds(indices, off_base, ipreg)
    values = vf.gather(beta_blk_f, indices, preg)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    exponent = vf.exp_sub(zero, values, preg)
    denominator = vf.adds(exponent, 1.0, preg)
    sigmoid = vf.div(one, denominator, preg)
    valid_rows = vf.lt(global_rows, valid, ipreg)
    sigmoid = vf.select(sigmoid, zero, valid_rows)
    vf.store_align(beta_cache_b, sigmoid, preg)


@pl.vector_function
def m5_tmqk_vf_local(kkt_f, mask_b, mqk_f, beta_cache_b, tb_out_f, mqk_out_b,
               r0):
    """v6 FUSED pair-VF: t_beta (S8 finish) + mqk (S12 finish) in ONE pass."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    neg1 = vf.full(-1.0, preg, dtype=pl.DT_FP32)
    for p in pl.range(0, HS // 2):
        gr0 = r0 + 2 * p
        gr1 = gr0 + 1
        base0 = 2 * p * CHUNK_C
        base1 = (2 * p + 1) * CHUNK_C
        sg0 = vf.load_align(beta_cache_b, 2 * p + 0, dist=pl.LoadDist.BRC_B32)
        mb = vf.load_align(mask_b, gr0 * HEAD_D, dtype=pl.DT_BF16)
        me0 = vf.astype(mb, pregbf, dtype=pl.DT_FP32)
        me1 = vf.astype(mb, pregbf, dtype=pl.DT_FP32,
                        layout=pl.CastLayout.ONE)
        slrow, carow = vf.interleave(me0, me1)

        mq0 = vf.load_align(mqk_f, base0)
        gt = vf.gt(carow, 0.0, preg)
        mq0 = vf.select(mq0, zero, gt)
        krow = vf.load_align(kkt_f, base0)
        t = vf.mul(krow, sg0, preg)
        t = vf.mul(t, neg1, preg)
        gt = vf.gt(slrow, 0.0, preg)
        t = vf.select(t, zero, gt)
        vf.store_align(tb_out_f + base0, t, preg)
        sg1 = vf.load_align(beta_cache_b, 2 * p + 1, dist=pl.LoadDist.BRC_B32)
        mb = vf.load_align(mask_b, gr1 * HEAD_D, dtype=pl.DT_BF16)
        me0 = vf.astype(mb, pregbf, dtype=pl.DT_FP32)
        me1 = vf.astype(mb, pregbf, dtype=pl.DT_FP32,
                        layout=pl.CastLayout.ONE)
        slrow, carow = vf.interleave(me0, me1)
        mq1 = vf.load_align(mqk_f, base1)
        gt = vf.gt(carow, 0.0, preg)
        mq1 = vf.select(mq1, zero, gt)
        krow = vf.load_align(kkt_f, base1)
        t = vf.mul(krow, sg1, preg)
        t = vf.mul(t, neg1, preg)
        gt = vf.gt(slrow, 0.0, preg)
        t = vf.select(t, zero, gt)
        vf.store_align(tb_out_f + base1, t, preg)
        b0 = vf.astype(mq0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(mq1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        drow, d1_ = vf.de_interleave(b0, b1)
        vf.store_align(mqk_out_b + p * 2 * CHUNK_C, drow, pregbf)


@pl.vector_function
def m5_vkn_vf_local(v_b, kn_b, cs_f, v_out_b, kd_out_b,
              beta_cache_b, r0):
    """Split beta*V into BF16 high/low parts; also produce BF16 -beta*K*exp(cs).

    Reuse each consumed row of v_b for the low part. The caller stores both
    parts before the M6 handoff; no extra UB tile or mutex is needed.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    neg1 = vf.full(-1.0, preg, dtype=pl.DT_FP32)
    for m in pl.range(0, HS):
        base = m * HEAD_D
        sg = vf.load_align(beta_cache_b, m, dist=pl.LoadDist.BRC_B32)
        bv = vf.load_align(v_b, base, dtype=pl.DT_BF16)
        e0 = vf.astype(bv, pregbf, dtype=pl.DT_FP32)
        e1 = vf.astype(bv, pregbf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ONE)
        vr0, vr1 = vf.interleave(e0, e1)
        t0 = vf.mul(vr0, sg, preg)
        t1 = vf.mul(vr1, sg, preg)
        b0 = vf.astype(t0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(t1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        packed, unused = vf.de_interleave(b0, b1)
        vf.store_align(v_out_b + base, packed, pregbf)
        # Keep the rounding residual while t0/t1 still hold beta*V in FP32.
        hi0 = vf.astype(b0, pregbf, dtype=pl.DT_FP32)
        hi1 = vf.astype(b1, pregbf, dtype=pl.DT_FP32)
        lo0 = vf.sub(t0, hi0, preg)
        lo1 = vf.sub(t1, hi1, preg)
        lb0 = vf.astype(lo0, pregbf, dtype=pl.DT_BF16,
                        round_mode=pl.VFRoundMode.CAST_ROUND)
        lb1 = vf.astype(lo1, pregbf, dtype=pl.DT_BF16,
                        round_mode=pl.VFRoundMode.CAST_ROUND)
        packed_lo, unused_lo = vf.de_interleave(lb0, lb1)
        vf.store_align(v_b + base, packed_lo, pregbf)
        bk = vf.load_align(kn_b, base, dtype=pl.DT_BF16)
        e0 = vf.astype(bk, pregbf, dtype=pl.DT_FP32)
        e1 = vf.astype(bk, pregbf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ONE)
        k0, k1 = vf.interleave(e0, e1)
        c0 = vf.load_align(cs_f, (r0 + m) * HEAD_D)
        c1 = vf.load_align(cs_f, (r0 + m) * HEAD_D + LANES)
        e0 = vf.exp(c0, preg)
        e1 = vf.exp(c1, preg)
        t0 = vf.mul(k0, e0, preg)
        t1 = vf.mul(k1, e1, preg)
        t0 = vf.mul(t0, sg, preg)
        t1 = vf.mul(t1, sg, preg)
        t0 = vf.mul(t0, neg1, preg)
        t1 = vf.mul(t1, neg1, preg)
        b0 = vf.astype(t0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(t1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        packed, unused = vf.de_interleave(b0, b1)
        vf.store_align(kd_out_b + base, packed, pregbf)


def m5_finish_v_local(beta, v_v, kn_slot,
                mqk_o, v_beta_bf16_o, v_beta_lo_o, v_c_o, beta_c_o,
                layout_code, m5_mask_g, cs_g, sl_mask, reload_mask,
                c, b, hv, s0, num_value_heads, save_cache,
                y0_l1, snz_stg_g, snz_nz_g, kd_nz_g,
                kd_l1,
                m5_beta_raw_b_g, m5_beta_blk_f_g, m5_v_b_g, beta_cache_g,
                m5_kkt_g, m5_mqk_g, m5_mqk_b_g, m5_v_operand_g, m5_kd_operand_g,
                t71_n, t71_eye, t71_zero, t71_eye_bf16,
                t71_nd_h, t71_nz_h, t71_nd_b, t71_nz_b, x_ub_g, eye64):
    """v12.11: M5 as a plain function called from the merged loop (lagged)."""
    h0 = (hv // 64) * 64
    hv_local = hv - h0
    off_base = hv_local * (1 - layout_code) + hv_local * 64 * layout_code
    step = 64 * (1 - layout_code) + 1 * layout_code
    brow_b = s0 * (1 - layout_code) + h0 * layout_code
    bcol_b = h0 * (1 - layout_code) + s0 * layout_code
    n_rows = 64 * (1 - layout_code) + pl.min(64, num_value_heads - h0) * layout_code
    n_cols = pl.min(64, num_value_heads - h0) * (1 - layout_code) + 64 * layout_code
    sub_m5 = pl.get_subblock_idx()
    r0 = sub_m5 * HS
    if reload_mask:
        pl.set_validshape(m5_mask_g.current(), [CHUNK_C, HEAD_D])
        pl.load(m5_mask_g.current(), sl_mask, [0, 0])
    mask_b = m5_mask_g.current()
    bb_raw = m5_beta_raw_b_g.next()
    pl.set_validshape(bb_raw, [n_rows, n_cols])
    pl.load(bb_raw, beta, [b, brow_b, bcol_b], order=[1, 2])
    blk_f = m5_beta_blk_f_g.next()
    pl.set_validshape(blk_f, [n_rows, n_cols])
    pl.cast(blk_f, bb_raw)
    beta_cache_b = beta_cache_g.next()
    m5_beta_sigmoid_vf_local(blk_f, beta_cache_b, off_base, step, r0, CHUNK_C)
    vb = m5_v_b_g.next()
    pl.load(vb, v_v, [b, hv, s0 + r0, 0], order=[2, 3])
    if save_cache:
        pl.store(v_c_o, vb, [b, c, hv, r0, 0], order=[3, 4])

    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=11)
    cs_slot = cs_g[0]
    kb = kn_slot
    vf_t = m5_v_operand_g.next()
    kf = m5_kd_operand_g.next()
    m5_vkn_vf_local(vb, kb, cs_slot, vf_t, kf, beta_cache_b, r0)
    if save_cache:
        pl.store(beta_c_o, beta_cache_b, [b, c, hv, r0], order=[2, 3])
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=_M5_CS_DONE)
    pl.store(v_beta_bf16_o, vf_t, [b, hv, s0 + r0, 0], order=[2, 3])
    pl.store(v_beta_lo_o, vb, [b, hv, s0 + r0, 0], order=[2, 3])

    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=9)
    kt = m5_kkt_g[0]
    mq = m5_mqk_g[0]
    mqb = m5_mqk_b_g.next()
    m5_tmqk_vf_local(kt, mask_b, mq, beta_cache_b, mq, mqb, r0)
    pl.set_validshape(mq, [HS, CHUNK_C])
    nd2nz_sg(mq, snz_stg_g.current(), HS)
    pl.set_validshape(snz_nz_g.current(), [HS, CHUNK_C])
    pl.insert(y0_l1, snz_nz_g.current(), [r0, 0])
    pl.store(mqk_o, mqb, [b, hv, c, r0, 0])

    t71_prepare_neumann_v(m5_mqk_g, t71_n, t71_eye, t71_zero,
                          t71_eye_bf16, t71_nd_h, t71_nz_h,
                          t71_nd_b, t71_nz_b, x_ub_g, eye64, reload_mask)
    kd_nz = kd_nz_g.next()
    pl.set_validshape(kd_nz, [HS, HEAD_D])
    pl.move(kd_nz, kf)
    pl.insert(kd_l1, kd_nz, [r0, 0])

    # _M5M6_EVENT had no consumer in the current M6 chain; do not emit a
    # per-work cross-core token that can accumulate and exhaust the event queue.
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=_M6_UPRE_EVENT)


def m5_finish_v(beta, v_v, kn_slot,
                mqk_o, v_beta_bf16_o, v_beta_lo_o, v_c_o, beta_c_o,
                layout_code, m5_mask_g, cs_g, sl_mask, reload_mask,
                c, b, hv, s0, num_value_heads, save_cache,
                y0_l1, snz_stg_g, snz_nz_g, kd_nz_g,
                kd_l1,
                m5_beta_raw_b_g, m5_beta_blk_f_g, m5_v_b_g, beta_cache_g,
                m5_kkt_g, m5_mqk_g, m5_mqk_b_g, m5_v_operand_g, m5_kd_operand_g,
                t71_n, t71_eye, t71_zero, t71_eye_bf16,
                t71_nd_h, t71_nz_h, t71_nd_b, t71_nz_b, x_ub_g, eye64):
    """v12.11: M5 as a plain function called from the merged loop (lagged)."""
    h0 = (hv // 64) * 64
    hv_local = hv - h0
    off_base = hv_local * (1 - layout_code) + hv_local * 64 * layout_code
    step = 64 * (1 - layout_code) + 1 * layout_code
    brow_b = s0 * (1 - layout_code) + h0 * layout_code
    bcol_b = h0 * (1 - layout_code) + s0 * layout_code
    n_rows = 64 * (1 - layout_code) + pl.min(64, num_value_heads - h0) * layout_code
    n_cols = pl.min(64, num_value_heads - h0) * (1 - layout_code) + 64 * layout_code
    sub_m5 = pl.get_subblock_idx()
    r0 = sub_m5 * HS
    if reload_mask:
        pl.set_validshape(m5_mask_g.current(), [CHUNK_C, HEAD_D])
        pl.load(m5_mask_g.current(), sl_mask, [0, 0])
    mask_b = m5_mask_g.current()
    bb_raw = m5_beta_raw_b_g.next()
    pl.set_validshape(bb_raw, [n_rows, n_cols])
    pl.load(bb_raw, beta, [b, brow_b, bcol_b], order=[1, 2])
    blk_f = m5_beta_blk_f_g.next()
    pl.set_validshape(blk_f, [n_rows, n_cols])
    pl.cast(blk_f, bb_raw)
    vb = m5_v_b_g.next()
    pl.load(vb, v_v, [b, hv, s0 + r0, 0], order=[2, 3])
    if save_cache:
        pl.store(v_c_o, vb, [b, c, hv, r0, 0], order=[3, 4])

    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=11)
    cs_slot = cs_g[0]
    kb = kn_slot
    vf_t = m5_v_operand_g.next()
    kf = m5_kd_operand_g.next()
    beta_cache_b = beta_cache_g.next()
    m5_vkn_vf(vb, kb, cs_slot, blk_f, vf_t, kf,
              beta_cache_b, off_base, step, r0, save_cache)
    if save_cache:
        pl.store(beta_c_o, beta_cache_b, [b, c, hv, r0], order=[2, 3])
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=_M5_CS_DONE)
    pl.store(v_beta_bf16_o, vf_t, [b, hv, s0 + r0, 0], order=[2, 3])
    pl.store(v_beta_lo_o, vb, [b, hv, s0 + r0, 0], order=[2, 3])

    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=9)
    kt = m5_kkt_g[0]
    mq = m5_mqk_g[0]
    mqb = m5_mqk_b_g.next()
    m5_tmqk_vf(kt, mask_b, mq, blk_f, mq, mqb, off_base, step, r0)
    pl.set_validshape(mq, [HS, CHUNK_C])
    nd2nz_sg(mq, snz_stg_g.current(), HS)
    pl.set_validshape(snz_nz_g.current(), [HS, CHUNK_C])
    pl.insert(y0_l1, snz_nz_g.current(), [r0, 0])
    pl.store(mqk_o, mqb, [b, hv, c, r0, 0])

    t71_prepare_neumann_v(m5_mqk_g, t71_n, t71_eye, t71_zero,
                          t71_eye_bf16, t71_nd_h, t71_nz_h,
                          t71_nd_b, t71_nz_b, x_ub_g, eye64, reload_mask)
    kd_nz = kd_nz_g.next()
    pl.set_validshape(kd_nz, [HS, HEAD_D])
    pl.move(kd_nz, kf)
    pl.insert(kd_l1, kd_nz, [r0, 0])

    # _M5M6_EVENT had no consumer in the current M6 chain; do not emit a
    # per-work cross-core token that can accumulate and exhaust the event queue.
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=_M6_UPRE_EVENT)


def t71_next_l0_fp16(a_g, b_g, ad0_g, ad1_g, ad2_g, ad3_g, a10_g, a32_g,
                     bd0_g, bd1_g, bd2_g, bd3_g, b10_g, b32_g,
                     bt0_g, bt1_g, bl0_g, bl1_g):
    a_g.next()
    b_g.next()
    ad0_g.next()
    ad1_g.next()
    ad2_g.next()
    ad3_g.next()
    a10_g.next()
    a32_g.next()
    bd0_g.next()
    bd1_g.next()
    bd2_g.next()
    bd3_g.next()
    b10_g.next()
    b32_g.next()
    bt0_g.next()
    bt1_g.next()
    bl0_g.next()
    bl1_g.next()


def t71_next_l0_bf16(a_g, b_g):
    a_g.next()
    b_g.next()


def t71_next_l0c16(c_g, d0_g, d1_g, d2_g, d3_g):
    c_g.next()
    d0_g.next()
    d1_g.next()
    d2_g.next()
    d3_g.next()


def m6_neumann_cube(
        t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g,
        t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g,
        t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g,
        t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g,
        t71_bl0_g, t71_bl1_g, t71_zero, t71_eye,
        t71_aux_g, t71_auxd0_g, t71_auxd1_g, t71_auxd2_g,
        t71_auxd3_g, t71_a_bf16_g, t71_b_bf16_g, t71_eye_bf16,
        v_beta_bf16_o, v_beta_lo_o, w_o, upre_o, t71_c_g,
        t71_n_db_g, p_gsi, pb, phv,
        ps0, t71_scratch_g, t71_tmpd0_g, t71_tmpd1_g,
        t71_tmpd2_g, t71_tmpd3_g, t71_p_g, t71_tmp_g,
        t71_inv_bf16_g, x_ub_g, m6_inv_l1_g, m6_akk_ub_g, m6_lb128_g,
        m6_op_l1_g, m6_la_g, m6_w_lc_g, m6_vbt_l1_g,
        m6_la_vbt_g, m6_lb_invt_g, m6_invt_zn_g, m6_u_lc_g,
        akk_o, save_cache):
    """m6 Neumann cube chain for one work: seed, 6 double-doubling rounds, fused W/Upre tail."""
    with pl.section_cube():
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=13)
        t71_next_l0_fp16(
            t71_a_g,
            t71_b_g,
            t71_ad0_g,
            t71_ad1_g,
            t71_ad2_g,
            t71_ad3_g,
            t71_a10_g,
            t71_a32_g,
            t71_bd0_g,
            t71_bd1_g,
            t71_bd2_g,
            t71_bd3_g,
            t71_b10_g,
            t71_b32_g,
            t71_bt0_g,
            t71_bt1_g,
            t71_bl0_g,
            t71_bl1_g,
        )
        t71_c_g.next()
        pl.move(t71_a_g.current(), t71_zero)
        pl.move(t71_b_g.current(), t71_zero)
        pl.move(t71_a_g.current(), t71_n_db_g[p_gsi])
        pl.move(t71_b_g.current(), t71_eye)
        pl.matmul(t71_c_g.current(), t71_a_g.current(), t71_b_g.current())
        t71_next_l0_fp16(
            t71_a_g,
            t71_b_g,
            t71_ad0_g,
            t71_ad1_g,
            t71_ad2_g,
            t71_ad3_g,
            t71_a10_g,
            t71_a32_g,
            t71_bd0_g,
            t71_bd1_g,
            t71_bd2_g,
            t71_bd3_g,
            t71_b10_g,
            t71_b32_g,
            t71_bt0_g,
            t71_bt1_g,
            t71_bl0_g,
            t71_bl1_g,
        )
        pl.move(t71_a_g.current(), t71_zero)
        pl.move(t71_b_g.current(), t71_zero)
        pl.move(t71_a_g.current(), t71_eye)
        pl.move(t71_b_g.current(), t71_eye)
        pl.matmul_acc(t71_c_g.current(), t71_c_g.current(), t71_a_g.current(), t71_b_g.current())
        pl.move(t71_scratch_g.current(), t71_c_g.current())
        t71_next_l0_fp16(
            t71_a_g,
            t71_b_g,
            t71_ad0_g,
            t71_ad1_g,
            t71_ad2_g,
            t71_ad3_g,
            t71_a10_g,
            t71_a32_g,
            t71_bd0_g,
            t71_bd1_g,
            t71_bd2_g,
            t71_bd3_g,
            t71_b10_g,
            t71_b32_g,
            t71_bt0_g,
            t71_bt1_g,
            t71_bl0_g,
            t71_bl1_g,
        )
        t71_c_g.next()
        t71_next_l0c16(t71_aux_g, t71_auxd0_g, t71_auxd1_g, t71_auxd2_g, t71_auxd3_g)
        pl.move(t71_a_g.current(), t71_zero)
        pl.move(t71_b_g.current(), t71_zero)
        pl.move(t71_ad0_g.current(), t71_n_db_g[p_gsi], [0, 0])
        pl.move(t71_bd0_g.current(), t71_n_db_g[p_gsi], [0, 0])
        pl.move(t71_ad1_g.current(), t71_n_db_g[p_gsi], [16, 16])
        pl.move(t71_bd1_g.current(), t71_n_db_g[p_gsi], [16, 16])
        pl.move(t71_ad2_g.current(), t71_n_db_g[p_gsi], [32, 32])
        pl.move(t71_bd2_g.current(), t71_n_db_g[p_gsi], [32, 32])
        pl.move(t71_ad3_g.current(), t71_n_db_g[p_gsi], [48, 48])
        pl.move(t71_bd3_g.current(), t71_n_db_g[p_gsi], [48, 48])
        pl.matmul(t71_auxd0_g.current(), t71_ad0_g.current(), t71_bd0_g.current())
        pl.matmul(t71_auxd1_g.current(), t71_ad1_g.current(), t71_bd1_g.current())
        pl.matmul(t71_auxd2_g.current(), t71_ad2_g.current(), t71_bd2_g.current())
        pl.matmul(t71_auxd3_g.current(), t71_ad3_g.current(), t71_bd3_g.current())
        pl.move(t71_tmpd0_g.current(), t71_auxd0_g.current())
        pl.move(t71_tmpd1_g.current(), t71_auxd1_g.current())
        pl.move(t71_tmpd2_g.current(), t71_auxd2_g.current())
        pl.move(t71_tmpd3_g.current(), t71_auxd3_g.current())
        t71_next_l0_fp16(
            t71_a_g,
            t71_b_g,
            t71_ad0_g,
            t71_ad1_g,
            t71_ad2_g,
            t71_ad3_g,
            t71_a10_g,
            t71_a32_g,
            t71_bd0_g,
            t71_bd1_g,
            t71_bd2_g,
            t71_bd3_g,
            t71_b10_g,
            t71_b32_g,
            t71_bt0_g,
            t71_bt1_g,
            t71_bl0_g,
            t71_bl1_g,
        )
        pl.move(t71_a_g.current(), t71_zero)
        pl.move(t71_b_g.current(), t71_zero)
        pl.move(t71_ad0_g.current(), t71_scratch_g.current(), [0, 0])
        pl.move(t71_bd0_g.current(), t71_eye, [0, 0])
        pl.move(t71_ad1_g.current(), t71_scratch_g.current(), [16, 16])
        pl.move(t71_bd1_g.current(), t71_eye, [16, 16])
        pl.move(t71_ad2_g.current(), t71_scratch_g.current(), [32, 32])
        pl.move(t71_bd2_g.current(), t71_eye, [32, 32])
        pl.move(t71_ad3_g.current(), t71_scratch_g.current(), [48, 48])
        pl.move(t71_bd3_g.current(), t71_eye, [48, 48])
        pl.matmul(t71_c_g.current(), t71_a_g.current(), t71_b_g.current())
        t71_next_l0_fp16(
            t71_a_g,
            t71_b_g,
            t71_ad0_g,
            t71_ad1_g,
            t71_ad2_g,
            t71_ad3_g,
            t71_a10_g,
            t71_a32_g,
            t71_bd0_g,
            t71_bd1_g,
            t71_bd2_g,
            t71_bd3_g,
            t71_b10_g,
            t71_b32_g,
            t71_bt0_g,
            t71_bt1_g,
            t71_bl0_g,
            t71_bl1_g,
        )
        pl.move(t71_ad0_g.current(), t71_scratch_g.current(), [0, 0])
        pl.move(t71_bd0_g.current(), t71_tmpd0_g.current())
        pl.move(t71_ad1_g.current(), t71_scratch_g.current(), [16, 16])
        pl.move(t71_bd1_g.current(), t71_tmpd1_g.current())
        pl.move(t71_ad2_g.current(), t71_scratch_g.current(), [32, 32])
        pl.move(t71_bd2_g.current(), t71_tmpd2_g.current())
        pl.move(t71_ad3_g.current(), t71_scratch_g.current(), [48, 48])
        pl.move(t71_bd3_g.current(), t71_tmpd3_g.current())
        pl.matmul_acc(t71_c_g.current(), t71_c_g.current(), t71_a_g.current(), t71_b_g.current())
        pl.move(t71_p_g.current(), t71_c_g.current())
        t71_next_l0_fp16(
            t71_a_g,
            t71_b_g,
            t71_ad0_g,
            t71_ad1_g,
            t71_ad2_g,
            t71_ad3_g,
            t71_a10_g,
            t71_a32_g,
            t71_bd0_g,
            t71_bd1_g,
            t71_bd2_g,
            t71_bd3_g,
            t71_b10_g,
            t71_b32_g,
            t71_bt0_g,
            t71_bt1_g,
            t71_bl0_g,
            t71_bl1_g,
        )
        t71_next_l0c16(t71_aux_g, t71_auxd0_g, t71_auxd1_g, t71_auxd2_g, t71_auxd3_g)
        pl.move(t71_ad0_g.current(), t71_tmp_g.current(), [0, 0])
        pl.move(t71_bd0_g.current(), t71_tmpd0_g.current())
        pl.move(t71_ad1_g.current(), t71_tmp_g.current(), [16, 16])
        pl.move(t71_bd1_g.current(), t71_tmpd1_g.current())
        pl.move(t71_ad2_g.current(), t71_tmp_g.current(), [32, 32])
        pl.move(t71_bd2_g.current(), t71_tmpd2_g.current())
        pl.move(t71_ad3_g.current(), t71_tmp_g.current(), [48, 48])
        pl.move(t71_bd3_g.current(), t71_tmpd3_g.current())
        pl.matmul(t71_auxd0_g.current(), t71_ad0_g.current(), t71_bd0_g.current())
        pl.matmul(t71_auxd1_g.current(), t71_ad1_g.current(), t71_bd1_g.current())
        pl.matmul(t71_auxd2_g.current(), t71_ad2_g.current(), t71_bd2_g.current())
        pl.matmul(t71_auxd3_g.current(), t71_ad3_g.current(), t71_bd3_g.current())
        pl.move(t71_tmpd0_g.current(), t71_auxd0_g.current())
        pl.move(t71_tmpd1_g.current(), t71_auxd1_g.current())
        pl.move(t71_tmpd2_g.current(), t71_auxd2_g.current())
        pl.move(t71_tmpd3_g.current(), t71_auxd3_g.current())
        t71_next_l0_fp16(
            t71_a_g,
            t71_b_g,
            t71_ad0_g,
            t71_ad1_g,
            t71_ad2_g,
            t71_ad3_g,
            t71_a10_g,
            t71_a32_g,
            t71_bd0_g,
            t71_bd1_g,
            t71_bd2_g,
            t71_bd3_g,
            t71_b10_g,
            t71_b32_g,
            t71_bt0_g,
            t71_bt1_g,
            t71_bl0_g,
            t71_bl1_g,
        )
        pl.move(t71_a_g.current(), t71_zero)
        pl.move(t71_b_g.current(), t71_zero)
        pl.move(t71_ad0_g.current(), t71_p_g.current(), [0, 0])
        pl.move(t71_bd0_g.current(), t71_tmpd0_g.current())
        pl.move(t71_ad1_g.current(), t71_p_g.current(), [16, 16])
        pl.move(t71_bd1_g.current(), t71_tmpd1_g.current())
        pl.move(t71_ad2_g.current(), t71_p_g.current(), [32, 32])
        pl.move(t71_bd2_g.current(), t71_tmpd2_g.current())
        pl.move(t71_ad3_g.current(), t71_p_g.current(), [48, 48])
        pl.move(t71_bd3_g.current(), t71_tmpd3_g.current())
        pl.matmul_acc(t71_c_g.current(), t71_c_g.current(), t71_a_g.current(), t71_b_g.current())
        pl.move(t71_p_g.current(), t71_c_g.current())
        t71_next_l0_fp16(
            t71_a_g,
            t71_b_g,
            t71_ad0_g,
            t71_ad1_g,
            t71_ad2_g,
            t71_ad3_g,
            t71_a10_g,
            t71_a32_g,
            t71_bd0_g,
            t71_bd1_g,
            t71_bd2_g,
            t71_bd3_g,
            t71_b10_g,
            t71_b32_g,
            t71_bt0_g,
            t71_bt1_g,
            t71_bl0_g,
            t71_bl1_g,
        )
        t71_next_l0c16(t71_aux_g, t71_auxd0_g, t71_auxd1_g, t71_auxd2_g, t71_auxd3_g)
        pl.move(t71_ad0_g.current(), t71_tmp_g.current(), [0, 0])
        pl.move(t71_bd0_g.current(), t71_tmpd0_g.current())
        pl.move(t71_ad1_g.current(), t71_tmp_g.current(), [16, 16])
        pl.move(t71_bd1_g.current(), t71_tmpd1_g.current())
        pl.move(t71_ad2_g.current(), t71_tmp_g.current(), [32, 32])
        pl.move(t71_bd2_g.current(), t71_tmpd2_g.current())
        pl.move(t71_ad3_g.current(), t71_tmp_g.current(), [48, 48])
        pl.move(t71_bd3_g.current(), t71_tmpd3_g.current())
        pl.matmul(t71_auxd0_g.current(), t71_ad0_g.current(), t71_bd0_g.current())
        pl.matmul(t71_auxd1_g.current(), t71_ad1_g.current(), t71_bd1_g.current())
        pl.matmul(t71_auxd2_g.current(), t71_ad2_g.current(), t71_bd2_g.current())
        pl.matmul(t71_auxd3_g.current(), t71_ad3_g.current(), t71_bd3_g.current())
        pl.move(t71_tmpd0_g.current(), t71_auxd0_g.current())
        pl.move(t71_tmpd1_g.current(), t71_auxd1_g.current())
        pl.move(t71_tmpd2_g.current(), t71_auxd2_g.current())
        pl.move(t71_tmpd3_g.current(), t71_auxd3_g.current())
        t71_next_l0_fp16(
            t71_a_g,
            t71_b_g,
            t71_ad0_g,
            t71_ad1_g,
            t71_ad2_g,
            t71_ad3_g,
            t71_a10_g,
            t71_a32_g,
            t71_bd0_g,
            t71_bd1_g,
            t71_bd2_g,
            t71_bd3_g,
            t71_b10_g,
            t71_b32_g,
            t71_bt0_g,
            t71_bt1_g,
            t71_bl0_g,
            t71_bl1_g,
        )
        pl.move(t71_a_g.current(), t71_zero)
        pl.move(t71_b_g.current(), t71_zero)
        pl.move(t71_ad0_g.current(), t71_p_g.current(), [0, 0])
        pl.move(t71_bd0_g.current(), t71_tmpd0_g.current())
        pl.move(t71_ad1_g.current(), t71_p_g.current(), [16, 16])
        pl.move(t71_bd1_g.current(), t71_tmpd1_g.current())
        pl.move(t71_ad2_g.current(), t71_p_g.current(), [32, 32])
        pl.move(t71_bd2_g.current(), t71_tmpd2_g.current())
        pl.move(t71_ad3_g.current(), t71_p_g.current(), [48, 48])
        pl.move(t71_bd3_g.current(), t71_tmpd3_g.current())
        pl.matmul_acc(t71_c_g.current(), t71_c_g.current(), t71_a_g.current(), t71_b_g.current())
        pl.move(t71_p_g.current(), t71_c_g.current())
        t71_next_l0_fp16(
            t71_a_g,
            t71_b_g,
            t71_ad0_g,
            t71_ad1_g,
            t71_ad2_g,
            t71_ad3_g,
            t71_a10_g,
            t71_a32_g,
            t71_bd0_g,
            t71_bd1_g,
            t71_bd2_g,
            t71_bd3_g,
            t71_b10_g,
            t71_b32_g,
            t71_bt0_g,
            t71_bt1_g,
            t71_bl0_g,
            t71_bl1_g,
        )
        t71_next_l0c16(t71_aux_g, t71_auxd0_g, t71_auxd1_g, t71_auxd2_g, t71_auxd3_g)
        pl.move(t71_a_g.current(), t71_zero)
        pl.move(t71_b_g.current(), t71_zero)
        pl.move(t71_a10_g.current(), t71_scratch_g.current(), [16, 0])
        pl.move(t71_a32_g.current(), t71_scratch_g.current(), [48, 32])
        pl.move(t71_bd0_g.current(), t71_p_g.current(), [0, 0])
        pl.move(t71_bd2_g.current(), t71_p_g.current(), [32, 32])
        pl.matmul(t71_aux_g.current(), t71_a_g.current(), t71_b_g.current())
        pl.move(t71_tmp_g.current(), t71_aux_g.current())
        t71_next_l0_fp16(
            t71_a_g,
            t71_b_g,
            t71_ad0_g,
            t71_ad1_g,
            t71_ad2_g,
            t71_ad3_g,
            t71_a10_g,
            t71_a32_g,
            t71_bd0_g,
            t71_bd1_g,
            t71_bd2_g,
            t71_bd3_g,
            t71_b10_g,
            t71_b32_g,
            t71_bt0_g,
            t71_bt1_g,
            t71_bl0_g,
            t71_bl1_g,
        )
        pl.move(t71_a_g.current(), t71_zero)
        pl.move(t71_b_g.current(), t71_zero)
        pl.move(t71_ad1_g.current(), t71_p_g.current(), [16, 16])
        pl.move(t71_ad3_g.current(), t71_p_g.current(), [48, 48])
        pl.move(t71_b10_g.current(), t71_tmp_g.current(), [16, 0])
        pl.move(t71_b32_g.current(), t71_tmp_g.current(), [48, 32])
        pl.matmul_acc(t71_c_g.current(), t71_c_g.current(), t71_a_g.current(), t71_b_g.current())
        pl.move(t71_p_g.current(), t71_c_g.current())
        t71_next_l0_fp16(
            t71_a_g,
            t71_b_g,
            t71_ad0_g,
            t71_ad1_g,
            t71_ad2_g,
            t71_ad3_g,
            t71_a10_g,
            t71_a32_g,
            t71_bd0_g,
            t71_bd1_g,
            t71_bd2_g,
            t71_bd3_g,
            t71_b10_g,
            t71_b32_g,
            t71_bt0_g,
            t71_bt1_g,
            t71_bl0_g,
            t71_bl1_g,
        )
        t71_next_l0c16(t71_aux_g, t71_auxd0_g, t71_auxd1_g, t71_auxd2_g, t71_auxd3_g)
        pl.move(t71_a_g.current(), t71_scratch_g.current())
        pl.move(t71_b_g.current(), t71_zero)
        pl.move(t71_bt0_g.current(), t71_p_g.current(), [0, 0])
        pl.move(t71_bt1_g.current(), t71_p_g.current(), [16, 0])
        pl.matmul(t71_aux_g.current(), t71_a_g.current(), t71_b_g.current())
        pl.move(t71_tmp_g.current(), t71_aux_g.current())
        t71_next_l0_fp16(
            t71_a_g,
            t71_b_g,
            t71_ad0_g,
            t71_ad1_g,
            t71_ad2_g,
            t71_ad3_g,
            t71_a10_g,
            t71_a32_g,
            t71_bd0_g,
            t71_bd1_g,
            t71_bd2_g,
            t71_bd3_g,
            t71_b10_g,
            t71_b32_g,
            t71_bt0_g,
            t71_bt1_g,
            t71_bl0_g,
            t71_bl1_g,
        )
        pl.move(t71_a_g.current(), t71_p_g.current())
        pl.move(t71_b_g.current(), t71_zero)
        pl.move(t71_bl0_g.current(), t71_tmp_g.current(), [32, 0])
        pl.move(t71_bl1_g.current(), t71_tmp_g.current(), [48, 0])
        pl.matmul_acc(t71_c_g.current(), t71_c_g.current(), t71_a_g.current(), t71_b_g.current())
        pl.move(t71_inv_bf16_g.current(), t71_c_g.current())
        t71_next_l0_bf16(t71_a_bf16_g, t71_b_bf16_g)
        t71_next_l0c16(t71_aux_g, t71_auxd0_g, t71_auxd1_g, t71_auxd2_g, t71_auxd3_g)
        pl.move(t71_a_bf16_g.current(), t71_inv_bf16_g.current())
        pl.move(t71_b_bf16_g.current(), t71_eye_bf16)
        pl.matmul(t71_aux_g.current(), t71_a_bf16_g.current(), t71_b_bf16_g.current(), phase=pl.AccPhase.Final)
        pl.set_validshape(x_ub_g.current(), [HS, CHUNK_C])
        pl.move(x_ub_g.current(), t71_aux_g.current(),
                acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM, phase=pl.STPhase.Final)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=12)
    with pl.section_vector():
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=12)
        pl.set_validshape(x_ub_g.current(), [HS, CHUNK_C])
        if save_cache:
            # DualModeSplitM presents one 32x64 row half to each Vector
            # subblock; write those halves directly to the BF16 cache.
            sub = pl.get_subblock_idx()
            akk_b = m6_akk_ub_g.next()
            pl.set_validshape(akk_b, [HS, CHUNK_C])
            pl.cast(akk_b, x_ub_g.current(), mode=pl.RoundMode.CAST_RINT)
            pl.store(akk_o, akk_b,
                     [pb, ps0 // CHUNK_C, phv, sub * HS, 0], order=[3, 4])
    with pl.section_cube():
        pl.move(m6_inv_l1_g.next(), t71_c_g.current())
        pl.move(m6_lb128_g.next(), m6_op_l1_g.current())
        pl.move(m6_la_g.next(), m6_inv_l1_g.current())
        pl.matmul(m6_w_lc_g.next(), m6_la_g.current(), m6_lb128_g.current(),
                  phase=pl.AccPhase.Final)
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=_M6_UPRE_EVENT)
        pl.load(m6_vbt_l1_g.next(), v_beta_bf16_o, [pb, phv, ps0, 0], order=[3, 2])
        pl.move(m6_la_vbt_g.next(), m6_vbt_l1_g.current())
        pl.move(m6_lb_invt_g.next(), m6_invt_zn_g.current())
        pl.matmul(m6_u_lc_g.next(), m6_la_vbt_g.current(), m6_lb_invt_g.current(),
                  phase=pl.AccPhase.Partial)
        # Accumulate the low product before the FP32 Upre store.
        pl.load(m6_vbt_l1_g.next(), v_beta_lo_o, [pb, phv, ps0, 0], order=[3, 2])
        pl.move(m6_la_vbt_g.next(), m6_vbt_l1_g.current())
        pl.matmul_acc(m6_u_lc_g.current(), m6_u_lc_g.current(),
                      m6_la_vbt_g.current(), m6_lb_invt_g.current(),
                      phase=pl.AccPhase.Final)
        pl.store(w_o, m6_w_lc_g.current(), [pb, phv, ps0, 0], order=[2, 3],
                 phase=pl.STPhase.Final)
        pl.store(upre_o, m6_u_lc_g.current(), [pb, phv, 0, ps0], order=[2, 3],
                 phase=pl.STPhase.Final)


def _get_platform_config():
    """Read core counts from the installed platform .ini (A5 SKU), falling back"""
    cube = 28
    vec = 56
    ini_hit = False
    try:
        site_packages = os.path.dirname(os.path.dirname(pl.__file__))
        config_dir = os.path.join(site_packages, "pypto", "lib", "configs")
        prefix = ""
        try:
            dev_name = torch.npu.get_device_name(0)
            if "950DT" in dev_name:
                prefix = "950DT"
            elif "950PR" in dev_name:
                prefix = "950PR"
        except Exception:  # noqa: BLE001
            prefix = ""
        if prefix and os.path.isdir(config_dir):
            for fname in sorted(os.listdir(config_dir)):
                if fname.lower().endswith(".ini") and fname.startswith(prefix):
                    path = os.path.join(config_dir, fname)
                    with open(path, "r", encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if line.startswith("cube_core_cnt"):
                                cube = int(line.split("=")[1].strip())
                            elif line.startswith("vector_core_cnt"):
                                vec = int(line.split("=")[1].strip())
                    ini_hit = True
    except Exception:  # noqa: BLE001 — fall through to device probe / planning values
        pass
    if not ini_hit:
        try:
            props = torch.npu.get_device_properties(0)
            cube = int(getattr(props, "cube_core_num", cube))
            vec = int(getattr(props, "vector_core_num", vec))
        except Exception:  # noqa: BLE001 — keep previous fallback values
            pass
    return cube, vec


# TND uses packed external token addresses and padded internal chunk addresses.
# Keep its scheduling separate so the dense kernels retain their tested pipeline.
_TND_OUT_EVENT = 6


@pl.vector_function
def tnd_zero_bf16_vf(tile, elements):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    zero = vf.full(0.0, mask, dtype=pl.DT_BF16)
    for off in pl.range(0, elements, 128):
        vf.store_align(tile + off, zero, mask)


@pl.vector_function
def tnd_zero_gate_tail_vf(tile, valid_rows):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, mask, dtype=pl.DT_FP32)
    for row in pl.range(valid_rows, HS):
        vf.store_align(tile + row * HEAD_D, zero, mask)
        vf.store_align(tile + row * HEAD_D + LANES, zero, mask)


def tnd_load_half(tile, tensor, head, src, valid):
    sub = pl.get_subblock_idx()
    rows = pl.max(0, pl.min(HS, valid - sub * HS))
    # LOAD leaves the remaining physical rows untouched; initialise them first.
    pl.set_validshape(tile, [HS, HEAD_D])
    if rows < HS:
        tnd_zero_bf16_vf(tile, HS * HEAD_D)
    if rows > 0:
        pl.set_validshape(tile, [rows, HEAD_D])
        pl.load(tile, tensor, [0, head, src + sub * HS, 0], order=[2, 3])
    pl.set_validshape(tile, [HS, HEAD_D])


def tnd_m1_prefetch_v(q_v, k_v, g_v, b, hk, hv, src, valid, x_b_g, xk_g, xg_g):
    xq = x_b_g.next()
    xk = xk_g.next()
    xg = xg_g.next()
    tnd_load_half(xq, q_v, hk, src, valid)
    tnd_load_half(xk, k_v, hk, src, valid)
    tnd_load_half(xg, g_v, hv, src, valid)


def tnd_m1_gate_norm_v(qn_o, kn_o, a_view, lb_view, dt_bias,
                   gate_l1, gate_slot_idx, b, hv, hk, s0, have_p2,
                   x_b_g, xk_g, xg_g, qn_h_g, kn_h_g, gate_f_g, gate_nz_g,
                   a_g, d_g, lb_g, q_c_o, k_c_o, c, save_cache, valid):
    """Module-1 vector stage (delay 0): gate/qn/kn for one chunk of one (b,hv)."""
    sub = pl.get_subblock_idx()
    r0 = sub * HS
    if have_p2:
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=3)
    xq = x_b_g.current()
    xk = xk_g.current()
    xg = xg_g.current()
    a_s = a_g.next()
    pl.set_validshape(a_s, [1, 1])
    pl.load(a_s, a_view, [hv, 0])
    lb_s = lb_g.next()
    pl.set_validshape(lb_s, [1, 1])
    pl.load(lb_s, lb_view, [0, 0])
    d_s = d_g.next()
    pl.load(d_s, dt_bias, [hv, 0])
    gatel = gate_f_g.next()
    m1_gate_vf(xg, d_s, a_s, gatel, lb_s)
    tnd_zero_gate_tail_vf(gatel, pl.max(0, pl.min(HS, valid - r0)))
    gate_nz = gate_nz_g.next()
    pl.set_validshape(gate_nz, [HS, HEAD_D])
    pl.set_validshape(gatel, [HS, HEAD_D])
    pl.move(gate_nz, gatel)
    gate_slot = gate_l1[gate_slot_idx]
    pl.insert(gate_slot, gate_nz, [r0, 0])
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
    qnb = qn_h_g[gate_slot_idx]
    m1_l2norm_vf(xq, qnb, L2_EPS)
    pl.store(qn_o, qnb, [b, hk, s0 + r0, 0], order=[2, 3])
    if save_cache:
        pl.store(q_c_o, qnb, [b, c, hk, r0, 0], order=[3, 4])
    knb = kn_h_g[gate_slot_idx]
    m1_l2norm_vf(xk, knb, L2_EPS)
    pl.store(kn_o, knb, [b, hk, s0 + r0, 0], order=[2, 3])
    if save_cache:
        pl.store(k_c_o, knb, [b, c, hk, r0, 0], order=[3, 4])


@pl.vector_function
def tnd_m5_tmqk_vf(kkt_f, mask_b, mqk_f, beta_blk_f, tb_out_f, mqk_out_b,
               off_base, step, r0, valid):
    """v6 FUSED pair-VF: t_beta (S8 finish) + mqk (S12 finish) in ONE pass."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    neg1 = vf.full(-1.0, preg, dtype=pl.DT_FP32)
    for p in pl.range(0, HS // 2):
        gr0 = r0 + 2 * p
        gr1 = gr0 + 1
        base0 = 2 * p * CHUNK_C
        base1 = (2 * p + 1) * CHUNK_C
        brow = vf.load_align(beta_blk_f, off_base + gr0 * step,
                             dist=pl.LoadDist.BRC_B32)
        e = vf.exp_sub(zero, brow, preg)
        dn = vf.adds(e, 1.0, preg)
        sg0 = vf.div(one, dn, preg)
        if gr0 >= valid:
            sg0 = vf.full(0.0, preg, dtype=pl.DT_FP32)
        mb = vf.load_align(mask_b, gr0 * HEAD_D, dtype=pl.DT_BF16)
        me0 = vf.astype(mb, pregbf, dtype=pl.DT_FP32)
        me1 = vf.astype(mb, pregbf, dtype=pl.DT_FP32,
                        layout=pl.CastLayout.ONE)
        slrow, carow = vf.interleave(me0, me1)

        mq0 = vf.load_align(mqk_f, base0)
        gt = vf.gt(carow, 0.0, preg)
        mq0 = vf.select(mq0, zero, gt)
        krow = vf.load_align(kkt_f, base0)
        t = vf.mul(krow, sg0, preg)
        t = vf.mul(t, neg1, preg)
        gt = vf.gt(slrow, 0.0, preg)
        t = vf.select(t, zero, gt)
        vf.store_align(tb_out_f + base0, t, preg)
        brow = vf.load_align(beta_blk_f, off_base + gr1 * step,
                             dist=pl.LoadDist.BRC_B32)
        e = vf.exp_sub(zero, brow, preg)
        dn = vf.adds(e, 1.0, preg)
        sg1 = vf.div(one, dn, preg)
        if gr1 >= valid:
            sg1 = vf.full(0.0, preg, dtype=pl.DT_FP32)
        mb = vf.load_align(mask_b, gr1 * HEAD_D, dtype=pl.DT_BF16)
        me0 = vf.astype(mb, pregbf, dtype=pl.DT_FP32)
        me1 = vf.astype(mb, pregbf, dtype=pl.DT_FP32,
                        layout=pl.CastLayout.ONE)
        slrow, carow = vf.interleave(me0, me1)
        mq1 = vf.load_align(mqk_f, base1)
        gt = vf.gt(carow, 0.0, preg)
        mq1 = vf.select(mq1, zero, gt)
        krow = vf.load_align(kkt_f, base1)
        t = vf.mul(krow, sg1, preg)
        t = vf.mul(t, neg1, preg)
        gt = vf.gt(slrow, 0.0, preg)
        t = vf.select(t, zero, gt)
        vf.store_align(tb_out_f + base1, t, preg)
        b0 = vf.astype(mq0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(mq1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        drow, d1_ = vf.de_interleave(b0, b1)
        vf.store_align(mqk_out_b + p * 2 * CHUNK_C, drow, pregbf)


@pl.vector_function
def tnd_m5_vkn_vf(v_b, kn_b, cs_f, beta_blk_f, v_out_b, kd_out_b,
              beta_cache_b, off_base, step, r0, save_cache, valid):
    """Split beta*V into BF16 high/low parts; also produce BF16 -beta*K*exp(cs).

    Reuse each consumed row of v_b for the low part. The caller stores both
    parts before the M6 handoff; no extra UB tile or mutex is needed.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    neg1 = vf.full(-1.0, preg, dtype=pl.DT_FP32)
    scalar = vf.update_mask(1, dtype=pl.DT_FP32)
    index_mask = vf.update_mask(1, dtype=pl.DT_UINT32)
    for m in pl.range(0, HS):
        base = m * HEAD_D
        brow = vf.load_align(beta_blk_f, off_base + (r0 + m) * step,
                             dist=pl.LoadDist.BRC_B32)
        e = vf.exp_sub(zero, brow, preg)
        dn = vf.adds(e, 1.0, preg)
        sg = vf.div(one, dn, preg)
        if r0 + m >= valid:
            sg = vf.full(0.0, preg, dtype=pl.DT_FP32)
        if save_cache:
            idx = vf.full(m, index_mask, dtype=pl.DT_UINT32)
            vf.scatter(beta_cache_b, sg, idx, scalar)
        bv = vf.load_align(v_b, base, dtype=pl.DT_BF16)
        e0 = vf.astype(bv, pregbf, dtype=pl.DT_FP32)
        e1 = vf.astype(bv, pregbf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ONE)
        vr0, vr1 = vf.interleave(e0, e1)
        t0 = vf.mul(vr0, sg, preg)
        t1 = vf.mul(vr1, sg, preg)
        b0 = vf.astype(t0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(t1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        packed, unused = vf.de_interleave(b0, b1)
        vf.store_align(v_out_b + base, packed, pregbf)
        # Keep the rounding residual while t0/t1 still hold beta*V in FP32.
        hi0 = vf.astype(b0, pregbf, dtype=pl.DT_FP32)
        hi1 = vf.astype(b1, pregbf, dtype=pl.DT_FP32)
        lo0 = vf.sub(t0, hi0, preg)
        lo1 = vf.sub(t1, hi1, preg)
        lb0 = vf.astype(lo0, pregbf, dtype=pl.DT_BF16,
                        round_mode=pl.VFRoundMode.CAST_ROUND)
        lb1 = vf.astype(lo1, pregbf, dtype=pl.DT_BF16,
                        round_mode=pl.VFRoundMode.CAST_ROUND)
        packed_lo, unused_lo = vf.de_interleave(lb0, lb1)
        vf.store_align(v_b + base, packed_lo, pregbf)
        bk = vf.load_align(kn_b, base, dtype=pl.DT_BF16)
        e0 = vf.astype(bk, pregbf, dtype=pl.DT_FP32)
        e1 = vf.astype(bk, pregbf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ONE)
        k0, k1 = vf.interleave(e0, e1)
        c0 = vf.load_align(cs_f, (r0 + m) * HEAD_D)
        c1 = vf.load_align(cs_f, (r0 + m) * HEAD_D + LANES)
        e0 = vf.exp(c0, preg)
        e1 = vf.exp(c1, preg)
        t0 = vf.mul(k0, e0, preg)
        t1 = vf.mul(k1, e1, preg)
        t0 = vf.mul(t0, sg, preg)
        t1 = vf.mul(t1, sg, preg)
        t0 = vf.mul(t0, neg1, preg)
        t1 = vf.mul(t1, neg1, preg)
        b0 = vf.astype(t0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(t1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        packed, unused = vf.de_interleave(b0, b1)
        vf.store_align(kd_out_b + base, packed, pregbf)


def tnd_m5_finish_v(beta, v_v, kn_slot,
                mqk_o, v_beta_bf16_o, v_beta_lo_o, v_c_o, beta_c_o,
                layout_code, m5_mask_g, cs_g, sl_mask, reload_mask,
                c, b, hv, s0, num_value_heads, save_cache, src, valid,
                y0_l1, snz_stg_g, snz_nz_g, kd_nz_g,
                kd_l1,
                m5_beta_raw_b_g, m5_beta_blk_f_g, m5_v_b_g, beta_cache_g,
                m5_kkt_g, m5_mqk_g, m5_mqk_b_g, m5_v_operand_g, m5_kd_operand_g,
                t71_n, t71_eye, t71_zero, t71_eye_bf16,
                t71_nd_h, t71_nz_h, t71_nd_b, t71_nz_b, x_ub_g, eye64):
    """v12.11: M5 as a plain function called from the merged loop (lagged)."""
    h0 = (hv // 64) * 64
    hv_local = hv - h0
    off_base = hv_local * (1 - layout_code) + hv_local * 64 * layout_code
    step = 64 * (1 - layout_code) + 1 * layout_code
    brow_b = src * (1 - layout_code) + h0 * layout_code
    bcol_b = h0 * (1 - layout_code) + src * layout_code
    n_rows = valid * (1 - layout_code) + pl.min(64, num_value_heads - h0) * layout_code
    n_cols = pl.min(64, num_value_heads - h0) * (1 - layout_code) + 64 * layout_code
    sub_m5 = pl.get_subblock_idx()
    r0 = sub_m5 * HS
    if reload_mask:
        pl.set_validshape(m5_mask_g.current(), [CHUNK_C, HEAD_D])
        pl.load(m5_mask_g.current(), sl_mask, [0, 0])
    mask_b = m5_mask_g.current()
    bb_raw = m5_beta_raw_b_g.next()
    pl.set_validshape(bb_raw, [64, 64])
    if valid < CHUNK_C:
        tnd_zero_bf16_vf(bb_raw, 64 * 64)
    pl.set_validshape(bb_raw, [n_rows, n_cols])
    pl.load(bb_raw, beta, [b, brow_b, bcol_b], order=[1, 2])
    blk_f = m5_beta_blk_f_g.next()
    pl.set_validshape(bb_raw, [64, n_cols])
    pl.set_validshape(blk_f, [64, n_cols])
    pl.cast(blk_f, bb_raw)
    vb = m5_v_b_g.next()
    tnd_load_half(vb, v_v, hv, src, valid)
    if save_cache:
        pl.store(v_c_o, vb, [b, c, hv, r0, 0], order=[3, 4])

    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=11)
    cs_slot = cs_g[0]
    kb = kn_slot
    vf_t = m5_v_operand_g.next()
    kf = m5_kd_operand_g.next()
    beta_cache_b = beta_cache_g.next()
    tnd_m5_vkn_vf(vb, kb, cs_slot, blk_f, vf_t, kf,
              beta_cache_b, off_base, step, r0, save_cache, valid)
    if save_cache:
        pl.store(beta_c_o, beta_cache_b, [b, c, hv, r0], order=[2, 3])
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=_M5_CS_DONE)
    pl.store(v_beta_bf16_o, vf_t, [b, hv, s0 + r0, 0], order=[2, 3])
    pl.store(v_beta_lo_o, vb, [b, hv, s0 + r0, 0], order=[2, 3])

    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=9)
    kt = m5_kkt_g[0]
    mq = m5_mqk_g[0]
    mqb = m5_mqk_b_g.next()
    tnd_m5_tmqk_vf(kt, mask_b, mq, blk_f, mq, mqb, off_base, step, r0, valid)
    pl.set_validshape(mq, [HS, CHUNK_C])
    nd2nz_sg(mq, snz_stg_g.current(), HS)
    pl.set_validshape(snz_nz_g.current(), [HS, CHUNK_C])
    pl.insert(y0_l1, snz_nz_g.current(), [r0, 0])
    pl.store(mqk_o, mqb, [b, hv, c, r0, 0])

    t71_prepare_neumann_v(m5_mqk_g, t71_n, t71_eye, t71_zero,
                          t71_eye_bf16, t71_nd_h, t71_nz_h,
                          t71_nd_b, t71_nz_b, x_ub_g, eye64, reload_mask)
    kd_nz = kd_nz_g.next()
    pl.set_validshape(kd_nz, [HS, HEAD_D])
    pl.move(kd_nz, kf)
    pl.insert(kd_l1, kd_nz, [r0, 0])

    # _M5M6_EVENT had no consumer in the current M6 chain; do not emit a
    # per-work cross-core token that can accumulate and exhaust the event queue.
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=_M6_UPRE_EVENT)


# General activation path: direct causal decay and FP32 triangular solve.


# beta in (0,2): solve only the diagonal 16x16 blocks on Vector.


@pl.jit(auto_mutex=True, tiling_key=_CPStage1Key)
def flash_kda_cp_stage1_kernel(
    # 输入
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    g: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    beta: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    a_log: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32],
    dt_bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    lower_bound: pl.DT_FP32,
    scale: pl.DT_FP32,
    layout_code: pl.DT_INT64,
    # 输出
    qd_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    kr_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    gam_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    mqk_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    w_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    upre_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    q_c_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    k_c_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    v_c_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    gk_c_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    beta_c_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    akk_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    save_cache: pl.DT_INT64,
    # workspace
    workspace: pl.Ptr[pl.DT_UINT8],
    const_workspace: pl.Ptr[pl.DT_UINT8],
    initial_state: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    summary: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    first_continues: pl.DT_INT64,
    has_initial: pl.DT_INT64,
    q_bs: pl.DT_INT64,
    q_ns: pl.DT_INT64,
    q_ss: pl.DT_INT64,
    k_bs: pl.DT_INT64,
    k_ns: pl.DT_INT64,
    k_ss: pl.DT_INT64,
    g_bs: pl.DT_INT64,
    g_ns: pl.DT_INT64,
    g_ss: pl.DT_INT64,
    v_bs: pl.DT_INT64,
    v_ns: pl.DT_INT64,
    v_ss: pl.DT_INT64,
    beta_bs: pl.DT_INT64,
    beta_s1: pl.DT_INT64,
    beta_s2: pl.DT_INT64,
    initial_bs: pl.DT_INT64,
    initial_ns: pl.DT_INT64,
    rstd_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    local_h: pl.Tensor[[pl.DYNAMIC] * 5, pl.DT_BF16],
    local_v_new: pl.Tensor[[pl.DYNAMIC] * 5, pl.DT_BF16],
    reuse_local: pl.DT_INT64,
):
    cc_cid = pl.get_block_idx() // pl.get_subblock_num()
    cc_base = const_workspace + cc_cid * 65536
    lower_tril = pl.make_tensor(cc_base, [64, 64], [64, 1], dtype=pl.DT_FP32)
    sl_mask = pl.make_tensor(cc_base + 16384, [64, 128], [128, 1], dtype=pl.DT_BF16)
    eye64 = pl.make_tensor(cc_base + 32768, [64, 64], [64, 1], dtype=pl.DT_FP32)
    lb_t = pl.make_tensor(cc_base + 49152, [1, 64], [64, 1], dtype=pl.DT_FP32)
    scale_t = pl.make_tensor(cc_base + 49408, [1, 64], [64, 1], dtype=pl.DT_FP32)
    cc_a = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec), addrs=[0], mutex_ids=[0])
    cc_b = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec), addrs=[0x4000], mutex_ids=[1])
    cc_bb = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16,
        target_memory=pl.MemorySpace.Vec), addrs=[0xC000], mutex_ids=[2])
    cc_e = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec), addrs=[0x10000], mutex_ids=[3])
    cc_l = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec), addrs=[0x14000], mutex_ids=[4])
    cc_s = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec), addrs=[0x14100], mutex_ids=[5])
    with pl.section_vector():
        if pl.get_subblock_idx() == 0:
            cp_constants_vf(cc_a.current(), cc_b.current(), cc_e.current(), cc_l.current(),
                            cc_s.current(), lower_bound, scale)
            pl.cast(cc_bb.current(), cc_b.current(), mode=pl.RoundMode.CAST_ROUND)
            pl.store(lower_tril, cc_a.current(), [0, 0])
            pl.store(sl_mask, cc_bb.current(), [0, 0])
            pl.store(eye64, cc_e.current(), [0, 0])
            pl.store(lb_t, cc_l.current(), [0, 0])
            pl.store(scale_t, cc_s.current(), [0, 0])
        pl.system.bar_all()
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=13)
    with pl.section_cube():
        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=13)
        pl.system.bar_all()
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=12)
    with pl.section_vector():
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=12)
    beta_view = pl.make_tensor(beta, [beta.shape[0], beta.shape[1], beta.shape[2]], [beta_bs, beta_s1, beta_s2])
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    num_cores = pl.get_block_num()

    batch_size = g.shape[0]
    seq_len = g.shape[1] * (1 - layout_code) + g.shape[2] * layout_code
    num_key_heads = q.shape[2] * (1 - layout_code) + q.shape[1] * layout_code
    num_value_heads = g.shape[2] * (1 - layout_code) + g.shape[1] * layout_code
    nq = num_value_heads // num_key_heads
    nc = seq_len // CHUNK_C
    total = batch_size * num_value_heads
    tt_b = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                       target_memory=pl.MemorySpace.Vec)
    tt_f32 = pl.TileType(shape=[HS, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec)
    tt_b32 = pl.TileType(shape=[HS, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec)
    tt_c64h = pl.TileType(shape=[HS, CHUNK_C], dtype=pl.DT_FP32,
                          target_memory=pl.MemorySpace.Vec)
    tt_c64bh = pl.TileType(shape=[HS, CHUNK_C], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Vec)
    tt_a = pl.TileType(shape=[CHUNK_C, 1], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Vec,
                       valid_shape=[-1, -1])
    tt_d = pl.TileType(shape=[1, HEAD_D], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Vec)
    tt_c64 = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec)
    tt_c64b = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                          target_memory=pl.MemorySpace.Vec)

    gate_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[L1_G0, L1_G1], mutex_ids=[11, 12],
        fwd_ids=_GATE_FWD_IDS, bwd_ids=_GATE_BWD_IDS,
    )

    cs_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[VA_CS0], mutex_ids=[16],
        fwd_ids=(4,), bwd_ids=(6,),
    )

    x_b_g = pl.make_tile_group(type=tt_b32, addrs=[VA_X_B], mutex_ids=[0])
    qn_h_g = pl.make_tile_group(
        type=tt_b32, addrs=[0x34800, 0x36800], mutex_ids=[10, 13])
    kn_h_g = pl.make_tile_group(
        type=tt_b32, addrs=[0x38800, 0x3A800], mutex_ids=[14, 15])
    xk_g = pl.make_tile_group(type=tt_b32, addrs=[0x21000], mutex_ids=[17])

    xg_g = pl.make_tile_group(type=tt_b32, addrs=[0x2000], mutex_ids=[18])
    gate_f_g = pl.make_tile_group(type=tt_f32, addrs=[VA_GATE_F], mutex_ids=[5])
    gate_nz_g = pl.make_tile_group(
        type=pl.TileType(shape=[HS, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                         valid_shape=[-1, -1], compact=1),
        addrs=[VA_GATE_NZ], mutex_ids=[6])
    a_g = pl.make_tile_group(type=tt_a, addrs=[VA_A_LOG], mutex_ids=[8])
    d_g = pl.make_tile_group(type=tt_d, addrs=[VA_DT_B], mutex_ids=[9])
    lb_g = pl.make_tile_group(type=tt_a, addrs=[VA_LB], mutex_ids=[7])

    m3_out_b_g = pl.make_tile_group(type=tt_b32, addrs=[VA_QN_B], mutex_ids=[3])
    m3_csl_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[0x20800], mutex_ids=[21])
    m3_sc_g = pl.make_tile_group(type=tt_a, addrs=[VA_SC], mutex_ids=[22])

    m4_kn_b_g = pl.make_tile_group(type=tt_b, addrs=[VA_KN_B], mutex_ids=[4])
    m4_qn_b_g = pl.make_tile_group(type=tt_b, addrs=[VA_QN_B], mutex_ids=[3])
    m4_opscr_b16_g = pl.make_tile_group(
        type=pl.TileType(shape=[ROW_BLOCK, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[VA_M4_SCR_B16, VA_M4_SCR_B16 + 0x1000], mutex_ids=[6, 19])
    m4_ref_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[0x20A00], mutex_ids=[30])

    _tt_op16 = pl.TileType(shape=[ROW_BLOCK, HEAD_D], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                           valid_shape=[-1, -1], compact=1)

    lk0_g = pl.make_tile_group(type=_tt_op16, addrs=[0x14000, 0x15000], mutex_ids=[31, 27])
    lk1_g = pl.make_tile_group(type=_tt_op16, addrs=[0x16000, 0x17000], mutex_ids=[28, 29])
    lk2_g = pl.make_tile_group(type=_tt_op16, addrs=[0x18000, 0x19000], mutex_ids=[30, 31])
    lk3_g = pl.make_tile_group(type=_tt_op16, addrs=[0x1A000, 0x1B000], mutex_ids=[27, 28])
    lq0_g = pl.make_tile_group(type=_tt_op16, addrs=[0x1C000, 0x1D000], mutex_ids=[29, 30])
    lq1_g = pl.make_tile_group(type=_tt_op16, addrs=[0x1E000, 0x1F000], mutex_ids=[31, 27])
    lq2_g = pl.make_tile_group(type=_tt_op16, addrs=[0x20000, 0x21000], mutex_ids=[28, 29])
    lq3_g = pl.make_tile_group(type=_tt_op16, addrs=[0x22000, 0x23000], mutex_ids=[30, 31])
    _tt_rw = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                         valid_shape=[-1, -1], compact=1)
    _tt_rz64 = pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Mat, layout=pl.ZN,
                           valid_shape=[-1, -1], compact=1)
    rw0_g = pl.make_tile_group(type=_tt_rw, addrs=[0x34000, 0x56000], mutex_ids=[27, 28])
    rw1_g = pl.make_tile_group(type=_tt_rw, addrs=[0x38000, 0x5A000], mutex_ids=[29, 30])
    rw2_g = pl.make_tile_group(type=_tt_rw, addrs=[0x3C000, 0x5E000], mutex_ids=[31, 27])
    rw3_g = pl.make_tile_group(type=_tt_rw, addrs=[0x40000, 0x62000], mutex_ids=[28, 29])
    rz0_g = pl.make_tile_group(type=_tt_rz64, addrs=[0x34000, 0x56000], mutex_ids=[27, 28])
    rz1_g = pl.make_tile_group(type=_tt_rz64, addrs=[0x38000, 0x5A000], mutex_ids=[29, 30])
    rz2_g = pl.make_tile_group(type=_tt_rz64, addrs=[0x3C000, 0x5E000], mutex_ids=[31, 27])
    rz3_g = pl.make_tile_group(type=_tt_rz64, addrs=[0x40000, 0x62000], mutex_ids=[28, 29])

    # kn_h_g[1] ends at 0x3C800; the lagged M5 still needs its last rows.
    opnz_g = pl.make_tile_group(
        type=pl.TileType(shape=[ROW_BLOCK, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                         valid_shape=[-1, -1], compact=1),
        addrs=[0x3C800],
        mutex_ids=[2])
    m4_la_g = pl.make_tile_group(
        type=pl.TileType(shape=[ROW_BLOCK, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=[0x4000, 0xD000], mutex_ids=[13, 23])
    m4_b_g = pl.make_tile_group(
        type=pl.TileType(shape=[HEAD_D, ROW_BLOCK], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=[0x8000, 0x9000, 0xA000, 0xB000],
        mutex_ids=[14, 16, 17, 18])
    m4_c_g = pl.make_tile_group(
        type=pl.TileType(shape=[ROW_BLOCK, ROW_BLOCK], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         fractal=1024),
        addrs=[0x8000, 0x8400, 0x8800, 0x8C00],
        mutex_ids=[15, 19, 20, 21])

    lower_l1_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[L1_L], mutex_ids=[10])
    lower_l0a_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=[LA0], mutex_ids=[13])
    g_l0b_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=[RA0], mutex_ids=[14])
    acc_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         fractal=1024),
        addrs=[CA0], mutex_ids=[15])

    m5_mask_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[0x28000], mutex_ids=[26])
    m5_kkt_g = pl.make_tile_group(
        type=tt_c64h, addrs=[VA_M5_KKT], mutex_ids=[1])
    m5_mqk_g = pl.make_tile_group(
        type=tt_c64h, addrs=[VA_M5_MQK], mutex_ids=[2])
    m5_mqk_b_g = pl.make_tile_group(
        type=tt_c64bh, addrs=[0xE100], mutex_ids=[2])

    m5_beta_raw_b_g = pl.make_tile_group(
        type=tt_c64b, addrs=[0x12000], mutex_ids=[2])
    m5_beta_blk_f_g = pl.make_tile_group(
        type=tt_c64, addrs=[VA_M5_BETA_F], mutex_ids=[3])
    m5_v_b_g = pl.make_tile_group(
        type=tt_b32, addrs=[VA_M5_VB], mutex_ids=[4])
    m5_v_operand_g = pl.make_tile_group(
        type=tt_b32, addrs=[VA_M5_VF], mutex_ids=[6])
    m5_kd_operand_g = pl.make_tile_group(
        type=tt_b32, addrs=[VA_M5_KNF], mutex_ids=[5])

    beta_cache_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, HS], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[0x3D800, 0x3DC00], mutex_ids=[23, 25])

    m5_kd_nz_g = pl.make_tile_group(
        type=pl.TileType(shape=[HS, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                         valid_shape=[-1, -1], compact=1),
        addrs=[P3_UB_XY], mutex_ids=[16])

    tt_m6_l64 = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                            valid_shape=[-1, -1])
    tt_m6_lc128 = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                              fractal=1024)
    tt_m6_vhs = pl.TileType(shape=[HS, CHUNK_C], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec,
                            pad=pl.TilePad.zero, valid_shape=[-1, -1])
    tt_m6_vhs33 = pl.TileType(shape=[HS + 1, CHUNK_C], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Vec,
                              pad=pl.TilePad.zero, valid_shape=[-1, -1])
    tt_m6_nz = pl.TileType(shape=[HS + 1, CHUNK_C], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Vec, layout=pl.NZ)
    x_ub_g = pl.make_tile_group(type=tt_m6_vhs, addrs=[P3_UB_X], mutex_ids=[15])
    xy_ub_g = pl.make_tile_group(type=tt_m6_vhs33, addrs=[P3_UB_XY], mutex_ids=[16])
    tt_m6_akk_ub = pl.TileType(shape=[HS, CHUNK_C], dtype=pl.DT_BF16,
                                target_memory=pl.MemorySpace.Vec,
                                valid_shape=[-1, -1])
    m6_akk_ub_g = pl.make_tile_group(type=tt_m6_akk_ub,
                                     addrs=[0x3E000, 0x3F000], mutex_ids=[24, 29])

    snz_ub_g = pl.make_tile_group(type=tt_m6_nz, addrs=[P3_UB_XY], mutex_ids=[16])
    y_l1_g = pl.make_tile_group(type=tt_m6_l64, addrs=[P3_L1_Y0, P3_L1_Y1],
                                mutex_ids=[0, 1])
    _t71_l1_fp16 = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                               target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                               fractal=512, compact=0)
    t71_n_db_g = pl.make_tile_group(type=_t71_l1_fp16,
                                    addrs=[0x76000, 0x78000],
                                    mutex_ids=[24, 25])
    t71_eye = pl.make_tile(_t71_l1_fp16, addr=0x7A000)
    t71_zero = pl.make_tile(_t71_l1_fp16, addr=0x7C000)
    t71_scratch_g = pl.make_tile_group(type=_t71_l1_fp16, addrs=172032, mutex_ids=[26])
    t71_p_g = pl.make_tile_group(type=_t71_l1_fp16, addrs=180224, mutex_ids=[27])
    t71_tmp_g = pl.make_tile_group(type=_t71_l1_fp16, addrs=188416, mutex_ids=[28])
    _t71_l1_fp16_d = pl.TileType(shape=[16, 16], dtype=pl.DT_FP16,
                                 target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                                 fractal=512, compact=0)
    t71_tmpd0_g = pl.make_tile_group(type=_t71_l1_fp16_d, addrs=188416, mutex_ids=[28])
    t71_tmpd1_g = pl.make_tile_group(type=_t71_l1_fp16_d, addrs=188416 + 0xA00, mutex_ids=[29])
    t71_tmpd2_g = pl.make_tile_group(type=_t71_l1_fp16_d, addrs=188416 + 0x1400, mutex_ids=[30])
    t71_tmpd3_g = pl.make_tile_group(type=_t71_l1_fp16_d, addrs=188416 + 0x1E00, mutex_ids=[31])
    _t71_l1_bf16 = pl.TileType(shape=[64, 64], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                               fractal=512, compact=0)
    t71_inv_bf16_g = pl.make_tile_group(type=_t71_l1_bf16, addrs=147456, mutex_ids=[29])
    t71_eye_bf16 = pl.make_tile(_t71_l1_bf16, addr=0x7E000)

    _t71_a = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ,
                         fractal=512, compact=0)
    _t71_b = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                         fractal=512, compact=0)
    t71_a_g = pl.make_tile_group(type=_t71_a, addrs=[0x5000, 0x7000], mutex_ids=[18, 19])
    t71_b_g = pl.make_tile_group(type=_t71_b, addrs=[0xC000, 0xE000], mutex_ids=[20, 21])
    _t71_ad = pl.TileType(shape=[16, 16], dtype=pl.DT_FP16,
                          target_memory=pl.MemorySpace.Left, layout=pl.NZ,
                          fractal=512, compact=0)
    _t71_bd = pl.TileType(shape=[16, 16], dtype=pl.DT_FP16,
                          target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                          fractal=512, compact=0)
    t71_ad0_g = pl.make_tile_group(type=_t71_ad, addrs=[0x5000, 0x7000], mutex_ids=[18, 19])
    t71_ad1_g = pl.make_tile_group(type=_t71_ad, addrs=[0x5A00, 0x7A00], mutex_ids=[0, 1])
    t71_ad2_g = pl.make_tile_group(type=_t71_ad, addrs=[0x6400, 0x8400], mutex_ids=[2, 3])
    t71_ad3_g = pl.make_tile_group(type=_t71_ad, addrs=[0x6E00, 0x8E00], mutex_ids=[4, 5])
    t71_a10_g = pl.make_tile_group(type=_t71_ad, addrs=[0x5200, 0x7200], mutex_ids=[18, 19])
    t71_a32_g = pl.make_tile_group(type=_t71_ad, addrs=[0x6600, 0x8600], mutex_ids=[18, 19])
    t71_bd0_g = pl.make_tile_group(type=_t71_bd, addrs=[0xC000, 0xE000], mutex_ids=[20, 21])
    t71_bd1_g = pl.make_tile_group(type=_t71_bd, addrs=[0xCA00, 0xEA00], mutex_ids=[6, 7])
    t71_bd2_g = pl.make_tile_group(type=_t71_bd, addrs=[0xD400, 0xF400], mutex_ids=[8, 9])
    t71_bd3_g = pl.make_tile_group(type=_t71_bd, addrs=[0xDE00, 0xFE00], mutex_ids=[10, 11])
    t71_b10_g = pl.make_tile_group(type=_t71_bd, addrs=[0xC800, 0xE800], mutex_ids=[20, 21])
    t71_b32_g = pl.make_tile_group(type=_t71_bd, addrs=[0xDC00, 0xFC00], mutex_ids=[20, 21])
    _t71_bt = pl.TileType(shape=[16, 32], dtype=pl.DT_FP16,
                          target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                          fractal=512, compact=0)
    t71_bt0_g = pl.make_tile_group(type=_t71_bt, addrs=[0xC000, 0xE000], mutex_ids=[20, 21])
    t71_bt1_g = pl.make_tile_group(type=_t71_bt, addrs=[0xC800, 0xE800], mutex_ids=[20, 21])
    t71_bl0_g = pl.make_tile_group(type=_t71_bt, addrs=[0xD000, 0xF000], mutex_ids=[20, 21])
    t71_bl1_g = pl.make_tile_group(type=_t71_bt, addrs=[0xD800, 0xF800], mutex_ids=[20, 21])

    _t71_c = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         fractal=1024, compact=0)
    t71_c_g = pl.make_tile_group(type=_t71_c, addrs=[0x9000, 0x11000], mutex_ids=[22, 23])
    _t71_c16 = pl.TileType(shape=[16, 16], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                           fractal=1024, compact=0)

    t71_auxd0_g = pl.make_tile_group(type=_t71_c16, addrs=[0xD000, 0x15000], mutex_ids=[24, 25])
    t71_auxd1_g = pl.make_tile_group(type=_t71_c16, addrs=[0xE400, 0x16400], mutex_ids=[26, 27])
    t71_auxd2_g = pl.make_tile_group(type=_t71_c16, addrs=[0xF800, 0x17800], mutex_ids=[28, 29])
    t71_auxd3_g = pl.make_tile_group(type=_t71_c16, addrs=[0x10C00, 0x18C00], mutex_ids=[30, 31])
    t71_aux_g = pl.make_tile_group(type=_t71_c, addrs=[0xD000, 0x15000], mutex_ids=[24, 25])

    _t71_a_bf16 = pl.TileType(shape=[64, 64], dtype=pl.DT_BF16,
                              target_memory=pl.MemorySpace.Left, layout=pl.NZ,
                              fractal=512, compact=0)
    _t71_b_bf16 = pl.TileType(shape=[64, 64], dtype=pl.DT_BF16,
                              target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                              fractal=512, compact=0)
    t71_a_bf16_g = pl.make_tile_group(type=_t71_a_bf16, addrs=[0x5000, 0x7000], mutex_ids=[18, 19])
    t71_b_bf16_g = pl.make_tile_group(type=_t71_b_bf16, addrs=[0xC000, 0xE000], mutex_ids=[20, 21])

    _t71_st_nd_h = pl.TileType(shape=[32, 64], dtype=pl.DT_FP16,
                               target_memory=pl.MemorySpace.Vec,
                               valid_shape=[-1, -1], compact=0)
    _t71_st_nz_h = pl.TileType(shape=[32, 64], dtype=pl.DT_FP16,
                               target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                               fractal=512, valid_shape=[-1, -1], compact=1)
    _t71_st_nd_b = pl.TileType(shape=[32, 64], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Vec,
                               valid_shape=[-1, -1], compact=0)
    _t71_st_nz_b = pl.TileType(shape=[32, 64], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                               fractal=512, valid_shape=[-1, -1], compact=1)
    t71_nd_h_g = pl.make_tile_group(type=_t71_st_nd_h, addrs=[49152], mutex_ids=[16])
    t71_nz_h_g = pl.make_tile_group(type=_t71_st_nz_h, addrs=[53248], mutex_ids=[16])
    t71_nd_b_g = pl.make_tile_group(type=_t71_st_nd_b, addrs=[49152], mutex_ids=[16])
    t71_nz_b_g = pl.make_tile_group(type=_t71_st_nz_b, addrs=[53248], mutex_ids=[16])
    tt_m6_l64 = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                            valid_shape=[-1, -1])
    tt_m6_l128 = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat)
    tt_m6_la = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Left, layout=pl.NZ)
    tt_m6_lb128 = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                              target_memory=pl.MemorySpace.Right, layout=pl.ZN)
    m6_la_g = pl.make_tile_group(type=tt_m6_la, addrs=[LA0], mutex_ids=[18])
    m6_inv_l1_g = pl.make_tile_group(type=tt_m6_l64, addrs=[P3_L1_X], mutex_ids=[31])
    m6_op_l1_g = pl.make_tile_group(type=tt_m6_l128, addrs=[L6_OP], mutex_ids=[19])

    tt_m6_hd64 = pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat)
    tt_m6_la_hd64 = pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_BF16,
                                target_memory=pl.MemorySpace.Left, layout=pl.NZ)
    tt_m6_lb64 = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Right, layout=pl.ZN)
    tt_m6_lc_ut = pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                              fractal=1024)
    m6_vbt_l1_g = pl.make_tile_group(type=tt_m6_hd64, addrs=[L6_OP2],
                                     mutex_ids=[24])
    m6_la_vbt_g = pl.make_tile_group(type=tt_m6_la_hd64, addrs=[L6_LA_VBT],
                                     mutex_ids=[25])
    tt_m6_l64_zn = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Mat, layout=pl.ZN,
                               valid_shape=[-1, -1])
    m6_invt_zn_g = pl.make_tile_group(type=tt_m6_l64_zn, addrs=[P3_L1_X],
                                      mutex_ids=[31])
    m6_lb_invt_g = pl.make_tile_group(type=tt_m6_lb64, addrs=[L6_LB_OP2],
                                      mutex_ids=[25])

    m6_lb128_g = pl.make_tile_group(type=tt_m6_lb128, addrs=[L6_LB_OP], mutex_ids=[20])
    m6_w_lc_g = pl.make_tile_group(type=tt_m6_lc128, addrs=[L6_LC_W], mutex_ids=[21])
    m6_u_lc_g = pl.make_tile_group(type=tt_m6_lc_ut, addrs=[L6_LC_U], mutex_ids=[22])

    str1 = HEAD_D * (1 - layout_code) + seq_len * HEAD_D * layout_code
    str_qk = (num_key_heads * HEAD_D) * (1 - layout_code) + HEAD_D * layout_code
    str_vv = (num_value_heads * HEAD_D) * (1 - layout_code) + HEAD_D * layout_code
    q_v = pl.make_tensor(q, [batch_size, num_key_heads, seq_len, HEAD_D], [q_bs, q_ns, q_ss, 1])
    k_v = pl.make_tensor(k, [batch_size, num_key_heads, seq_len, HEAD_D], [k_bs, k_ns, k_ss, 1])
    g_v = pl.make_tensor(g, [batch_size, num_value_heads, seq_len, HEAD_D], [g_bs, g_ns, g_ss, 1])

    ws_nk_bytes = batch_size * num_key_heads * seq_len * HEAD_D * 2
    qn_o = pl.make_tensor(workspace, [batch_size, num_key_heads, seq_len, HEAD_D],
                          [num_key_heads * seq_len * HEAD_D, str1, str_qk, 1], dtype=pl.DT_BF16)
    kn_o = pl.make_tensor(workspace + ws_nk_bytes, [batch_size, num_key_heads, seq_len, HEAD_D],
                          [num_key_heads * seq_len * HEAD_D, str1, str_qk, 1], dtype=pl.DT_BF16)
    qd_o = pl.make_tensor(
        qd_out,
        [batch_size, num_value_heads, seq_len, HEAD_D],
        [num_value_heads * seq_len * HEAD_D, str1, str_vv, 1],
    )
    kr_o = pl.make_tensor(
        kr_out,
        [batch_size, num_value_heads, seq_len, HEAD_D],
        [num_value_heads * seq_len * HEAD_D, str1, str_vv, 1],
    )
    v_v = pl.make_tensor(v, [batch_size, num_value_heads, seq_len, HEAD_D], [v_bs, v_ns, v_ss, 1])
    w_o = pl.make_tensor(
        w_out,
        [batch_size, num_value_heads, seq_len, HEAD_D],
        [num_value_heads * seq_len * HEAD_D, str1, str_vv, 1],
    )
    upre_o = pl.make_tensor(upre_out, [batch_size, num_value_heads, HEAD_D, seq_len],
                            [num_value_heads * HEAD_D * seq_len, HEAD_D * seq_len, seq_len, 1])
    gam_o = pl.make_tensor(gam_out, [batch_size, num_value_heads, nc, HEAD_D],
                           [num_value_heads * nc * HEAD_D, nc * HEAD_D, HEAD_D, 1])
    mqk_o = pl.make_tensor(mqk_out, [batch_size, num_value_heads, nc, CHUNK_C, CHUNK_C],
                           [num_value_heads * nc * CHUNK_C * CHUNK_C, nc * CHUNK_C * CHUNK_C,
                            CHUNK_C * CHUNK_C, CHUNK_C, 1])
    q_c_o = pl.make_tensor(q_c_out, [batch_size, nc, num_key_heads, CHUNK_C, HEAD_D],
                           [nc * num_key_heads * CHUNK_C * HEAD_D, num_key_heads * CHUNK_C * HEAD_D,
                            CHUNK_C * HEAD_D, HEAD_D, 1])
    k_c_o = pl.make_tensor(k_c_out, [batch_size, nc, num_key_heads, CHUNK_C, HEAD_D],
                           [nc * num_key_heads * CHUNK_C * HEAD_D, num_key_heads * CHUNK_C * HEAD_D,
                            CHUNK_C * HEAD_D, HEAD_D, 1])
    v_c_o = pl.make_tensor(v_c_out, [batch_size, nc, num_value_heads, CHUNK_C, HEAD_D],
                           [nc * num_value_heads * CHUNK_C * HEAD_D, num_value_heads * CHUNK_C * HEAD_D,
                            CHUNK_C * HEAD_D, HEAD_D, 1])
    gk_c_o = pl.make_tensor(gk_c_out, [batch_size, nc, num_value_heads, CHUNK_C, HEAD_D],
                            [nc * num_value_heads * CHUNK_C * HEAD_D, num_value_heads * CHUNK_C * HEAD_D,
                             CHUNK_C * HEAD_D, HEAD_D, 1])
    beta_c_o = pl.make_tensor(beta_c_out, [batch_size, nc, num_value_heads, CHUNK_C],
                              [nc * num_value_heads * CHUNK_C, num_value_heads * CHUNK_C, CHUNK_C, 1])
    akk_o = pl.make_tensor(akk_out, [batch_size, nc, num_value_heads, CHUNK_C, CHUNK_C],
                           [nc * num_value_heads * CHUNK_C * CHUNK_C, num_value_heads * CHUNK_C * CHUNK_C,
                            CHUNK_C * CHUNK_C, CHUNK_C, 1])
    v_beta_bf16_o = pl.make_tensor(workspace + ws_nk_bytes * 2, [batch_size, num_value_heads, seq_len, HEAD_D],
                                   [num_value_heads * seq_len * HEAD_D, str1, str_vv, 1], dtype=pl.DT_BF16)
    v_beta_lo_o = pl.make_tensor(workspace + ws_nk_bytes * 2 + batch_size * num_value_heads * seq_len * HEAD_D * 2,
                                [batch_size, num_value_heads, seq_len, HEAD_D],
                                [num_value_heads * seq_len * HEAD_D, str1, str_vv, 1], dtype=pl.DT_BF16)
    a_view = pl.make_tensor(a_log, [num_value_heads, 1], [1, 1])
    lb_view = pl.make_tensor(lb_t, [1, 1], [1, 1])
    scale_view = pl.make_tensor(scale_t, [1, 1], [1, 1])

    own_tasks = (batch_size * num_key_heads - core_id + num_cores - 1) // num_cores
    own_work = own_tasks * nc * nq
    if reuse_local:
        own_tasks = (batch_size * num_key_heads * nc - core_id + num_cores - 1) // num_cores
        own_work = own_tasks * nq

    with pl.section_cube():
        l_pre = lower_l1_g.next()
        pl.load(l_pre, lower_tril, [0, 0])
    with pl.section_vector():
        sl_m_pre = m5_mask_g.next()
        pl.load(sl_m_pre, sl_mask, [0, 0])
        w_first = 0
        if own_work > 0:
            c_f = 0
            bh_f = core_id * nq
            if reuse_local:
                c_f = core_id % nc
                bh_f = (core_id // nc) * nq
            b_f = bh_f // num_value_heads
            hv_f = bh_f % num_value_heads
            hk_f = hv_f // nq
            m1_prefetch_v(q_v, k_v, g_v, b_f, hk_f, hv_f,
                          c_f * CHUNK_C, x_b_g, xk_g, xg_g)

    pw_valid = 0
    pc = 0
    pb = 0
    phv = 0
    ps0 = 0
    p_gsi = 0
    for work in pl.range(0, own_work):
        c = work % nc
        bh = (core_id + (work // (nc * nq)) * num_cores) * nq + (work // nc) % nq
        if reuse_local:
            flat = core_id + (work // nq) * num_cores
            c = flat % nc
            bh = (flat // nc) * nq + work % nq
        b = bh // num_value_heads
        hv = bh % num_value_heads
        hk = hv // nq
        s0 = c * CHUNK_C
        w_seq = work
        gate_slot_idx = w_seq % 2
        if pw_valid:
            with pl.section_cube():
                m4_kkt_mqk_blocks(m5_kkt_g, m5_mqk_g,
                                  p_gsi,
                                  lk0_g, lk1_g, lk2_g, lk3_g,
                                  lq0_g, lq1_g, lq2_g, lq3_g,
                                  rz0_g, rz1_g, rz2_g, rz3_g,
                                  m4_la_g, m4_b_g, m4_c_g)
        with pl.section_vector():
            if reuse_local and nq == 1:
                m1_gate_norm_v_cached(qn_o, kn_o, a_view, lb_view,
                               dt_bias, gate_l1, gate_slot_idx, b, hv, hk, s0,
                               1 if w_seq >= 2 else 0,
                               x_b_g, xk_g, xg_g, qn_h_g, kn_h_g, gate_f_g,
                               gate_nz_g, a_g, d_g, lb_g,
                               q_c_o, k_c_o, c, 1 if save_cache and hv % nq == 0 else 0, rstd_out)
            else:
                m1_gate_norm_v(qn_o, kn_o, a_view, lb_view,
                               dt_bias, gate_l1, gate_slot_idx, b, hv, hk, s0,
                               1 if w_seq >= 2 else 0,
                               x_b_g, xk_g, xg_g, qn_h_g, kn_h_g, gate_f_g,
                               gate_nz_g, a_g, d_g, lb_g,
                               q_c_o, k_c_o, c, 1 if save_cache and hv % nq == 0 else 0, rstd_out)
        if pw_valid:
            with pl.section_vector():
                if reuse_local:
                    m5_finish_v_local(beta_view, v_v, kn_h_g[p_gsi],
                                mqk_o, v_beta_bf16_o, v_beta_lo_o, v_c_o, beta_c_o,
                                layout_code, m5_mask_g, cs_g, sl_mask,
                                1 if w_seq == 1 else 0,
                                pc, pb, phv, ps0, num_value_heads, save_cache,
                                y_l1_g[0], xy_ub_g, snz_ub_g, m5_kd_nz_g,
                                m6_op_l1_g[0],
                                m5_beta_raw_b_g, m5_beta_blk_f_g, m5_v_b_g, beta_cache_g,
                                m5_kkt_g, m5_mqk_g, m5_mqk_b_g,
                                m5_v_operand_g, m5_kd_operand_g,
                                t71_n_db_g[p_gsi], t71_eye, t71_zero, t71_eye_bf16,
                                t71_nd_h_g[0], t71_nz_h_g[0], t71_nd_b_g[0], t71_nz_b_g[0],
                                x_ub_g, eye64)
                else:
                    m5_finish_v(beta_view, v_v, kn_h_g[p_gsi],
                                mqk_o, v_beta_bf16_o, v_beta_lo_o, v_c_o, beta_c_o,
                                layout_code, m5_mask_g, cs_g, sl_mask,
                                1 if w_seq == 1 else 0,
                                pc, pb, phv, ps0, num_value_heads, save_cache,
                                y_l1_g[0], xy_ub_g, snz_ub_g, m5_kd_nz_g,
                                m6_op_l1_g[0],
                                m5_beta_raw_b_g, m5_beta_blk_f_g, m5_v_b_g, beta_cache_g,
                                m5_kkt_g, m5_mqk_g, m5_mqk_b_g,
                                m5_v_operand_g, m5_kd_operand_g,
                                t71_n_db_g[p_gsi], t71_eye, t71_zero, t71_eye_bf16,
                                t71_nd_h_g[0], t71_nz_h_g[0], t71_nd_b_g[0], t71_nz_b_g[0],
                                x_ub_g, eye64)
        with pl.section_cube():
            if reuse_local and nq == 1:
                m2_cumsum_c_cached(gate_l1, cs_g, gate_slot_idx,
                            pw_valid, lower_l1_g, lower_l0a_g, g_l0b_g, acc_g, save_cache)
            else:
                m2_cumsum_c(gate_l1, cs_g, gate_slot_idx,
                            pw_valid, lower_l1_g, lower_l0a_g, g_l0b_g, acc_g, save_cache)
        with pl.section_vector():
            if reuse_local:
                if reuse_local and nq == 1:
                    m3_qd_kr_gam_v_cached(qn_o, kn_o, qd_o, kr_o, gam_o, cs_g, scale_view,
                                   c, b, hv, hk, s0, gate_slot_idx, num_value_heads, nc,
                                   qn_h_g, kn_h_g, m3_out_b_g, m3_csl_g,
                                   m3_sc_g,
                                   m4_kn_b_g, m4_qn_b_g,
                                   m4_opscr_b16_g, m4_ref_g,
                                   lk0_g, lk1_g, lk2_g, lk3_g,
                                   lq0_g, lq1_g, lq2_g, lq3_g,
                                   rw0_g, rw1_g, rw2_g, rw3_g,
                                   opnz_g, q_c_o, k_c_o, gk_c_o, save_cache)
                else:
                    m3_qd_kr_gam_v_local(qn_o, kn_o, qd_o, kr_o, gam_o, cs_g, scale_view,
                                   c, b, hv, hk, s0, gate_slot_idx, num_value_heads, nc,
                                   qn_h_g, kn_h_g, m3_out_b_g, m3_csl_g,
                                   m3_sc_g,
                                   m4_kn_b_g, m4_qn_b_g,
                                   m4_opscr_b16_g, m4_ref_g,
                                   lk0_g, lk1_g, lk2_g, lk3_g,
                                   lq0_g, lq1_g, lq2_g, lq3_g,
                                   rw0_g, rw1_g, rw2_g, rw3_g,
                                   opnz_g, gk_c_o, save_cache)
            else:
                m3_qd_kr_gam_v(qn_o, kn_o, qd_o, kr_o, gam_o, cs_g, scale_view,
                               c, b, hv, hk, s0, gate_slot_idx, num_value_heads, nc,
                               qn_h_g, kn_h_g, m3_out_b_g, m3_csl_g,
                               m3_sc_g,
                               m4_kn_b_g, m4_qn_b_g,
                               m4_opscr_b16_g, m4_ref_g,
                               lk0_g, lk1_g, lk2_g, lk3_g,
                               lq0_g, lq1_g, lq2_g, lq3_g,
                               rw0_g, rw1_g, rw2_g, rw3_g,
                               opnz_g, gk_c_o, save_cache)

        if pw_valid:
            m6_neumann_cube(
                t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g,
                t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g,
                t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g,
                t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g,
                t71_bl0_g, t71_bl1_g, t71_zero, t71_eye,
                t71_aux_g, t71_auxd0_g, t71_auxd1_g, t71_auxd2_g,
                t71_auxd3_g, t71_a_bf16_g, t71_b_bf16_g, t71_eye_bf16,
                v_beta_bf16_o, v_beta_lo_o, w_o, upre_o, t71_c_g,
                t71_n_db_g, p_gsi, pb, phv,
                ps0, t71_scratch_g, t71_tmpd0_g, t71_tmpd1_g,
                t71_tmpd2_g, t71_tmpd3_g, t71_p_g, t71_tmp_g,
                t71_inv_bf16_g, x_ub_g, m6_inv_l1_g, m6_akk_ub_g, m6_lb128_g,
                m6_op_l1_g, m6_la_g, m6_w_lc_g, m6_vbt_l1_g,
                m6_la_vbt_g, m6_lb_invt_g, m6_invt_zn_g, m6_u_lc_g,
                akk_o, save_cache)

        nw = pl.min(work + 1, own_work - 1)
        c_n = nw % nc
        bh_n = (core_id + (nw // (nc * nq)) * num_cores) * nq + (nw // nc) % nq
        if reuse_local:
            flat_n = core_id + (nw // nq) * num_cores
            c_n = flat_n % nc
            bh_n = (flat_n // nc) * nq + nw % nq
        with pl.section_vector():
            m1_prefetch_v(q_v, k_v, g_v, bh_n // num_value_heads, (bh_n % num_value_heads) // nq,
                          bh_n % num_value_heads, c_n * CHUNK_C, x_b_g, xk_g, xg_g)
        pw_valid = 1
        pc = c
        pb = b
        phv = hv
        ps0 = s0
        p_gsi = gate_slot_idx

    n_works = own_work
    if pw_valid:
        with pl.section_cube():
            m4_kkt_mqk_blocks(m5_kkt_g, m5_mqk_g,
                              p_gsi,
                              lk0_g, lk1_g, lk2_g, lk3_g,
                              lq0_g, lq1_g, lq2_g, lq3_g,
                              rz0_g, rz1_g, rz2_g, rz3_g,
                              m4_la_g, m4_b_g, m4_c_g)
        with pl.section_vector():
            if reuse_local:
                m5_finish_v_local(beta_view, v_v, kn_h_g[p_gsi],
                            mqk_o, v_beta_bf16_o, v_beta_lo_o, v_c_o, beta_c_o,
                            layout_code, m5_mask_g, cs_g, sl_mask,
                            1 if n_works == 1 else 0,
                            pc, pb, phv, ps0, num_value_heads, save_cache,
                            y_l1_g[0], xy_ub_g, snz_ub_g, m5_kd_nz_g,
                            m6_op_l1_g[0],
                            m5_beta_raw_b_g, m5_beta_blk_f_g, m5_v_b_g, beta_cache_g,
                            m5_kkt_g, m5_mqk_g, m5_mqk_b_g,
                            m5_v_operand_g, m5_kd_operand_g,
                            t71_n_db_g[p_gsi], t71_eye, t71_zero, t71_eye_bf16,
                            t71_nd_h_g[0], t71_nz_h_g[0], t71_nd_b_g[0], t71_nz_b_g[0],
                            x_ub_g, eye64)
            else:
                m5_finish_v(beta_view, v_v, kn_h_g[p_gsi],
                            mqk_o, v_beta_bf16_o, v_beta_lo_o, v_c_o, beta_c_o,
                            layout_code, m5_mask_g, cs_g, sl_mask,
                            1 if n_works == 1 else 0,
                            pc, pb, phv, ps0, num_value_heads, save_cache,
                            y_l1_g[0], xy_ub_g, snz_ub_g, m5_kd_nz_g,
                            m6_op_l1_g[0],
                            m5_beta_raw_b_g, m5_beta_blk_f_g, m5_v_b_g, beta_cache_g,
                            m5_kkt_g, m5_mqk_g, m5_mqk_b_g,
                            m5_v_operand_g, m5_kd_operand_g,
                            t71_n_db_g[p_gsi], t71_eye, t71_zero, t71_eye_bf16,
                            t71_nd_h_g[0], t71_nz_h_g[0], t71_nd_b_g[0], t71_nz_b_g[0],
                            x_ub_g, eye64)
    if n_works >= 1:
        with pl.section_vector():
            pass
            for _ in pl.range(0, pl.min(n_works, 2)):
                pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=3)
    if pw_valid:
        m6_neumann_cube(
            t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g,
            t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g,
            t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g,
            t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g,
            t71_bl0_g, t71_bl1_g, t71_zero, t71_eye,
            t71_aux_g, t71_auxd0_g, t71_auxd1_g, t71_auxd2_g,
            t71_auxd3_g, t71_a_bf16_g, t71_b_bf16_g, t71_eye_bf16,
            v_beta_bf16_o, v_beta_lo_o, w_o, upre_o, t71_c_g,
            t71_n_db_g, p_gsi, pb, phv,
            ps0, t71_scratch_g, t71_tmpd0_g, t71_tmpd1_g,
            t71_tmpd2_g, t71_tmpd3_g, t71_p_g, t71_tmp_g,
            t71_inv_bf16_g, x_ub_g, m6_inv_l1_g, m6_akk_ub_g, m6_lb128_g,
            m6_op_l1_g, m6_la_g, m6_w_lc_g, m6_vbt_l1_g,
            m6_la_vbt_g, m6_lb_invt_g, m6_invt_zn_g, m6_u_lc_g,
            akk_o, save_cache)

    if reuse_local and own_work > 0:
        with pl.section_cube():
            # The last M3/M5 releases have no following M2 consumer.
            # Retire them before MIX sync_all reuses event IDs 11-14.
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=7)
            if save_cache:
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=_M3_GK_DONE,
                    sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
            pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=_M5_CS_DONE)

    # Drain this owner before reusing its UB/L1 for the summary chain.
    with pl.section_vector():
        pl.system.bar_all()
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=13)
    with pl.section_cube():
        pl.system.bar_all()
        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=13)
        pl.system.bar_all()
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=12)
    with pl.section_vector():
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=12)
        pl.system.bar_all()
    if reuse_local:
        # Every launched AIC/AIV reaches the same barrier, including idle owners.
        with pl.section_cube():
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        with pl.section_vector():
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
    if reuse_local:
        cp_local_chain(kr_out, gam_out, w_out, upre_out, initial_state, summary,
            local_h, local_v_new, layout_code, has_initial, initial_bs, initial_ns, LocalStateRows)
    else:
        cp_forward_chain(qd_out, kr_out, gam_out, mqk_out, w_out, upre_out,
            initial_state, summary, qd_out, initial_state, local_h, local_v_new,
            layout_code, 0, first_continues, has_initial, reuse_local, 1, nq, initial_bs, initial_ns, 0, 0, 0, 0, 0)


# New CP forward implementation; appended to the frozen working copy by build_forward.py.


@pl.vector_function
def cp_constants_vf(lower, mask, eye, lb, sc, lower_bound, scale):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    x = vf.arange(0, dtype=pl.DT_INT32)
    pi = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
    zero = vf.full(0.0, dtype=pl.DT_FP32)
    one = vf.full(1.0, dtype=pl.DT_FP32)
    for r in pl.range(0, 64):
        le = vf.le(x, r, pi)
        lt = vf.lt(x, r, pi)
        eq = vf.eq(x, r, pi)
        a = vf.select(one, zero, le)
        b = vf.select(one, zero, lt)
        e = vf.select(one, zero, eq)
        vf.store_align(lower + r * 64, a, p)
        vf.store_align(mask + r * 128, b, p)
        vf.store_align(mask + r * 128 + 64, a, p)
        vf.store_align(eye + r * 64, e, p)
    lbr = vf.full(lower_bound, dtype=pl.DT_FP32)
    scr = vf.full(scale, dtype=pl.DT_FP32)
    vf.store_align(lb, lbr, p)
    vf.store_align(sc, scr, p)


@pl.vector_function
def cp_state_init_vf(state, row_offset, identity):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pi = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
    x0 = vf.arange(0, dtype=pl.DT_INT32)
    x1 = vf.arange(64, dtype=pl.DT_INT32)
    z = vf.full(0.0, dtype=pl.DT_FP32)
    one = vf.full(1.0, dtype=pl.DT_FP32)
    for r in pl.range(0, 64):
        if identity:
            e0 = vf.eq(x0, r + row_offset, pi)
            e1 = vf.eq(x1, r + row_offset, pi)
            a = vf.select(one, z, e0)
            b = vf.select(one, z, e1)
        else:
            a = z
            b = z
        vf.store_align(state + r * 128, a, p)
        vf.store_align(state + r * 128 + 64, b, p)


@pl.vector_function
def cp_state_update_vf(state, delta, gamma):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    g0 = vf.load_align(gamma, 0)
    g1 = vf.load_align(gamma, 64)
    for r in pl.range(0, 64):
        a0 = vf.load_align(state, r * 128)
        a1 = vf.load_align(state, r * 128 + 64)
        d0 = vf.load_align(delta, r * 128)
        d1 = vf.load_align(delta, r * 128 + 64)
        s0 = vf.mul(a0, g0, p)
        s1 = vf.mul(a1, g1, p)
        v0 = vf.add(s0, d0, p)
        v1 = vf.add(s1, d1, p)
        vf.store_align(state + r * 128, v0, p)
        vf.store_align(state + r * 128 + 64, v1, p)


@pl.vector_function
def cp_state_add_vf(state, prod, constant):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 128):
        a = vf.load_align(prod, r * 64)
        b = vf.load_align(constant, r * 64)
        s = vf.add(a, b, p)
        vf.store_align(state + r * 64, s, p)


@pl.vector_function
def cp_local_zero_vf(state, rows):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, dtype=pl.DT_FP32)
    for row in pl.range(0, rows):
        vf.store_align(state + row * 128, zero, p)
        vf.store_align(state + row * 128 + 64, zero, p)


@pl.vector_function
def cp_local_update_vf(state, delta, gamma, rows):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    g0 = vf.load_align(gamma, 0)
    g1 = vf.load_align(gamma, 64)
    for row in pl.range(0, rows):
        a0 = vf.load_align(state, row * 128)
        a1 = vf.load_align(state, row * 128 + 64)
        d0 = vf.load_align(delta, row * 128)
        d1 = vf.load_align(delta, row * 128 + 64)
        s0 = vf.mul(a0, g0, p)
        s1 = vf.mul(a1, g1, p)
        z0 = vf.add(s0, d0, p)
        z1 = vf.add(s1, d1, p)
        vf.store_align(state + row * 128, z0, p)
        vf.store_align(state + row * 128 + 64, z1, p)


def cp_local_chain(kr, gam, w, upre, initial, summary, h, v_new,
                   layout_code, has_initial, initial_bs, initial_ns, value_rows):
    """Split only independent V rows; K reductions and chunk order stay unchanged."""
    core = pl.get_block_idx() // pl.get_subblock_num()
    cores = pl.get_block_num()
    sub = pl.get_subblock_idx()
    bsz = kr.shape[0]
    seq_len = kr.shape[1] * (1 - layout_code) + kr.shape[2] * layout_code
    nh = kr.shape[2] * (1 - layout_code) + kr.shape[1] * layout_code
    nc = seq_len // 64
    panels = 128 // value_rows
    half = value_rows // 2
    hn = 128 * (1 - layout_code) + seq_len * 128 * layout_code
    sn = nh * 128 * (1 - layout_code) + 128 * layout_code
    kv = pl.make_tensor(kr, [bsz, nh, seq_len, 128], [nh * seq_len * 128, hn, sn, 1])
    wv = pl.make_tensor(w, [bsz, nh, seq_len, 128], [nh * seq_len * 128, hn, sn, 1])
    iv = pl.make_tensor(initial, [initial.shape[0], nh, 128, 128], [initial_bs, initial_ns, 128, 1])
    st = pl.make_tile_group(type=pl.TileType(shape=[half, 128], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec), addrs=[0], mutex_ids=[0])
    de = pl.make_tile_group(type=pl.TileType(shape=[half, 128], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec), addrs=[0x8000], mutex_ids=[1])
    sb = pl.make_tile_group(type=pl.TileType(shape=[half, 128], dtype=pl.DT_BF16,
        target_memory=pl.MemorySpace.Vec), addrs=[0x10000], mutex_ids=[2])
    sz = pl.make_tile_group(type=pl.TileType(shape=[half, 128], dtype=pl.DT_BF16,
        target_memory=pl.MemorySpace.Vec, layout=pl.NZ), addrs=[0x14000], mutex_ids=[3])
    ud = pl.make_tile_group(type=pl.TileType(shape=[half, 64], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec), addrs=[0x18000], mutex_ids=[4])
    up = pl.make_tile_group(type=pl.TileType(shape=[half, 64], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec), addrs=[0x1C000], mutex_ids=[5])
    ub = pl.make_tile_group(type=pl.TileType(shape=[half, 64], dtype=pl.DT_BF16,
        target_memory=pl.MemorySpace.Vec), addrs=[0x20000], mutex_ids=[6])
    uz = pl.make_tile_group(type=pl.TileType(shape=[half, 64], dtype=pl.DT_BF16,
        target_memory=pl.MemorySpace.Vec, layout=pl.NZ), addrs=[0x22000], mutex_ids=[7])
    ga = pl.make_tile_group(type=pl.TileType(shape=[1, 128], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec), addrs=[0x24000], mutex_ids=[8])
    sl = pl.make_tile_group(type=pl.TileType(shape=[value_rows, 128], dtype=pl.DT_BF16,
        target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=[0], mutex_ids=[0])
    wl = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_BF16,
        target_memory=pl.MemorySpace.Mat), addrs=[0x8000], mutex_ids=[1])
    kl = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16,
        target_memory=pl.MemorySpace.Mat), addrs=[0xC000], mutex_ids=[2])
    ul = pl.make_tile_group(type=pl.TileType(shape=[value_rows, 64], dtype=pl.DT_BF16,
        target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=[0x10000], mutex_ids=[3])
    la = pl.make_tile_group(type=pl.TileType(shape=[value_rows, 128], dtype=pl.DT_BF16,
        target_memory=pl.MemorySpace.Left, layout=pl.NZ), addrs=[0], mutex_ids=[13])
    lua = pl.make_tile_group(type=pl.TileType(shape=[value_rows, 64], dtype=pl.DT_BF16,
        target_memory=pl.MemorySpace.Left, layout=pl.NZ), addrs=[0], mutex_ids=[13])
    rb = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_BF16,
        target_memory=pl.MemorySpace.Right, layout=pl.ZN), addrs=[0], mutex_ids=[16])
    rkb = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16,
        target_memory=pl.MemorySpace.Right, layout=pl.ZN), addrs=[0], mutex_ids=[16])
    uc = pl.make_tile_group(type=pl.TileType(shape=[value_rows, 64], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024), addrs=[0], mutex_ids=[18])
    dc = pl.make_tile_group(type=pl.TileType(shape=[value_rows, 128], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024), addrs=[0x10000], mutex_ids=[19])
    for task in pl.range(core, bsz * nh * panels, cores):
        bh = task // panels
        panel = task % panels
        b = bh // nh
        hv = bh % nh
        row = panel * value_rows + sub * half
        with pl.section_vector():
            if has_initial:
                pl.load(st.current(), iv, [b, hv, row, 0])
            else:
                cp_local_zero_vf(st.current(), half)
            pl.cast(sb.current(), st.current(), mode=pl.RoundMode.CAST_ROUND)
            pl.store(h, sb.current(), [b, 0, hv, row, 0])
        for c in pl.range(0, nc):
            with pl.section_vector():
                pl.cast(sb.current(), st.current(), mode=pl.RoundMode.CAST_ROUND)
                pl.move(sz.current(), sb.current())
                pl.insert(sl.current(), sz.current(), [sub * half, 0])
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=13)
            with pl.section_cube():
                pl.load(wl.current(), wv, [b, hv, c * 64, 0], order=[3, 2])
                pl.load(kl.current(), kv, [b, hv, c * 64, 0])
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=13)
                pl.move(la.current(), sl.current())
                pl.move(rb.current(), wl.current())
                pl.matmul(uc.current(), la.current(), rb.current())
                pl.move(ud.current(), uc.current(), acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=12)
            with pl.section_vector():
                pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=12)
                pl.load(up.current(), upre, [b, hv, row, c * 64])
                m7_ut_add_upre_bf16_vf(ud.current(), up.current(), ub.current(), 0, value_rows)
                pl.store(v_new, ub.current(), [b, c, hv, row, 0], order=[3, 4])
                pl.move(uz.current(), ub.current())
                pl.insert(ul.current(), uz.current(), [sub * half, 0])
                pl.load(ga.current(), gam, [b, hv, c, 0])
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=13)
            with pl.section_cube():
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=13)
                pl.move(lua.current(), ul.current())
                pl.move(rkb.current(), kl.current())
                pl.matmul(dc.current(), lua.current(), rkb.current())
                pl.move(de.current(), dc.current(), acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=12)
            with pl.section_vector():
                pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=12)
                cp_local_update_vf(st.current(), de.current(), ga.current(), half)
                pl.cast(sb.current(), st.current(), mode=pl.RoundMode.CAST_ROUND)
                pl.store(h, sb.current(), [b, c + 1, hv, row, 0])
        with pl.section_vector():
            pl.store(summary, st.current(), [b, hv, 0, row, 0])
            # E is the final FP32 state; T is exactly zero at a sequence start.
            cp_local_zero_vf(st.current(), half)
            pl.store(summary, st.current(), [b, hv, 1, row, 0])
            pl.system.bar_all()


def cp_forward_chain(
    qd,
    kr,
    gam,
    mqk,
    w,
    upre,
    initial_state,
    summary,
    out,
    final_state,
    h,
    v_new,
    layout_code,
    cp_rank,
    first_continues,
    has_initial,
    save_cache,
    summary_mode,
    ratio,
    initial_bs,
    initial_ns,
    out_bs,
    out_ns,
    out_ss,
    final_bs,
    final_ns,
):
    """Inline device region: one owner per batch/value-head, two AIV row halves.

    summary_mode computes E and T after the owner's preprocessing has drained.
    Otherwise it scans preceding packets and executes the real forward chain.
    Both callers are single-launch JITs. No host-side arithmetic is involved.
    """
    core = pl.get_block_idx() // pl.get_subblock_num()
    cores = pl.get_block_num()
    sub = pl.get_subblock_idx()
    batch_size = qd.shape[0]
    seq_len = qd.shape[1] * (1 - layout_code) + qd.shape[2] * layout_code
    num_heads = qd.shape[2] * (1 - layout_code) + qd.shape[1] * layout_code
    nc = seq_len // 64
    hn = 128 * (1 - layout_code) + seq_len * 128 * layout_code
    sn = num_heads * 128 * (1 - layout_code) + 128 * layout_code
    qv = pl.make_tensor(qd, [batch_size, num_heads, seq_len, 128], [num_heads * seq_len * 128, hn, sn, 1])
    kv = pl.make_tensor(kr, [batch_size, num_heads, seq_len, 128], [num_heads * seq_len * 128, hn, sn, 1])
    wv = pl.make_tensor(w, [batch_size, num_heads, seq_len, 128], [num_heads * seq_len * 128, hn, sn, 1])
    ov = pl.make_tensor(out, [batch_size, num_heads, seq_len, 128], [out_bs, out_ns, out_ss, 1])
    iv = pl.make_tensor(initial_state, [initial_state.shape[0], num_heads, 128, 128], [initial_bs, initial_ns, 128, 1])
    fv = pl.make_tensor(final_state, [final_state.shape[0], num_heads, 128, 128], [final_bs, final_ns, 128, 1])
    # UB high-water 0x35000 (< 256 KiB), half state [64 rows,128 columns].
    st = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec), addrs=[0], mutex_ids=[0])
    de = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec), addrs=[0x8000], mutex_ids=[1])
    sb = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Vec), addrs=[0x10000], mutex_ids=[2])
    sz = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
        addrs=[0x14000],
        mutex_ids=[3],
    )
    ud = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec), addrs=[0x18000], mutex_ids=[4])
    up = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec), addrs=[0x1C000], mutex_ids=[5])
    ub = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Vec), addrs=[0x20000], mutex_ids=[6])
    uz = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
        addrs=[0x22000],
        mutex_ids=[7],
    )
    ga = pl.make_tile_group(type=pl.TileType(shape=[1, 128], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec), addrs=[0x24000], mutex_ids=[8])
    tf = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec), addrs=[0x25000], mutex_ids=[9])
    tb = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Vec), addrs=[0x2D000], mutex_ids=[10])
    tz = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
        addrs=[0x31000],
        mutex_ids=[11],
    )
    # L1 views share the same backing for transpose, with the same mutex.
    sl = pl.make_tile_group(type=pl.TileType(
        shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=[0], mutex_ids=[0])
    slt = pl.make_tile_group(type=pl.TileType(
        shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN), addrs=[0], mutex_ids=[0])
    wl = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Mat), addrs=[0x8000], mutex_ids=[1])
    kl = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Mat), addrs=[0xC000], mutex_ids=[2])
    ul = pl.make_tile_group(
        type=pl.TileType(shape=[128, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[0x10000],
        mutex_ids=[3],
    )
    ult = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x10000],
        mutex_ids=[3],
    )
    ql = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Mat), addrs=[0x14000], mutex_ids=[4])
    ml = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Mat), addrs=[0x18000], mutex_ids=[5])
    tl = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[0x20000],
        mutex_ids=[6],
    )
    la = pl.make_tile_group(type=pl.TileType(
        shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left, layout=pl.NZ), addrs=[0], mutex_ids=[13])
    lua = pl.make_tile_group(type=pl.TileType(
        shape=[128, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left, layout=pl.NZ), addrs=[0], mutex_ids=[13])
    lqa = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=[0x8000],
        mutex_ids=[14],
    )
    lma = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=[0xC000],
        mutex_ids=[15],
    )
    rb = pl.make_tile_group(type=pl.TileType(
        shape=[128, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right, layout=pl.ZN), addrs=[0], mutex_ids=[16])
    rkb = pl.make_tile_group(type=pl.TileType(
        shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right, layout=pl.ZN), addrs=[0], mutex_ids=[16])
    rtb = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=[0],
        mutex_ids=[16],
    )
    rsb = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=[0x8000],
        mutex_ids=[17],
    )
    rub = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=[0x8000],
        mutex_ids=[17],
    )
    uc = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=[0],
        mutex_ids=[18],
    )
    dc = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=[0x10000],
        mutex_ids=[19],
    )
    oc = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Acc), addrs=[0x20000], mutex_ids=[20])
    group_count = (batch_size * num_heads // ratio - core + cores - 1) // cores
    for owned in pl.range(0, group_count * ratio):
        task = (core + (owned // ratio) * cores) * ratio + owned % ratio
        b = task // num_heads
        hv = task % num_heads
        for chain in pl.range(0, 1 + summary_mode * first_continues):
            with pl.section_vector():
                if summary_mode and chain == 1:
                    cp_state_init_vf(st.current(), sub * 64, 1)
                elif not first_continues and has_initial:
                    pl.load(st.current(), iv, [b, hv, sub * 64, 0])
                else:
                    cp_state_init_vf(st.current(), sub * 64, 0)
            if not summary_mode and first_continues:
                for rank in pl.range(0, cp_rank):
                    with pl.section_vector():
                        pl.cast(sb.current(), st.current(), mode=pl.RoundMode.CAST_ROUND)
                        pl.move(sz.current(), sb.current())
                        pl.insert(sl.current(), sz.current(), [sub * 64, 0])
                        pl.load(tf.current(), summary, [rank * batch_size + b, hv, 1, sub * 64, 0])
                        pl.cast(tb.current(), tf.current(), mode=pl.RoundMode.CAST_ROUND)
                        pl.move(tz.current(), tb.current())
                        pl.insert(tl.current(), tz.current(), [sub * 64, 0])
                        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=13)
                    with pl.section_cube():
                        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=13)
                        pl.move(la.current(), sl.current())
                        pl.move(rtb.current(), tl.current())
                        pl.matmul(dc.current(), la.current(), rtb.current())
                        pl.move(de.current(), dc.current(), acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=12)
                    with pl.section_vector():
                        pl.load(tf.current(), summary, [rank * batch_size + b, hv, 0, sub * 64, 0])
                        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=12)
                        cp_state_add_vf(st.current(), de.current(), tf.current())
            with pl.section_vector():
                if save_cache and (not summary_mode or chain == 0):
                    pl.cast(sb.current(), st.current(), mode=pl.RoundMode.CAST_ROUND)
                    pl.store(h, sb.current(), [b, 0, hv, sub * 64, 0])
            for c in pl.range(0, nc):
                with pl.section_vector():
                    pl.cast(sb.current(), st.current(), mode=pl.RoundMode.CAST_ROUND)
                    pl.move(sz.current(), sb.current())
                    pl.insert(sl.current(), sz.current(), [sub * 64, 0])
                    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=13)
                with pl.section_cube():
                    pl.load(wl.current(), wv, [b, hv, c * 64, 0], order=[3, 2])
                    pl.load(kl.current(), kv, [b, hv, c * 64, 0])
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=13)
                    pl.move(la.current(), sl.current())
                    pl.move(rb.current(), wl.current())
                    pl.matmul(uc.current(), la.current(), rb.current())
                    pl.move(ud.current(), uc.current(), acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                    if not summary_mode:
                        pl.load(ql.current(), qv, [b, hv, c * 64, 0])
                        pl.move(lqa.current(), ql.current())
                        pl.move(rsb.current(), slt.current())
                        pl.matmul(oc.current(), lqa.current(), rsb.current(), phase=pl.AccPhase.Partial)
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=12)
                with pl.section_vector():
                    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=12)
                    if summary_mode and chain == 1:
                        pl.cast(ub.current(), ud.current(), mode=pl.RoundMode.CAST_ROUND)
                    else:
                        pl.load(up.current(), upre, [b, hv, sub * 64, c * 64])
                        m7_ut_add_upre_bf16_vf(ud.current(), up.current(), ub.current(), 0, 128)
                    if save_cache and (not summary_mode or chain == 0):
                        pl.store(v_new, ub.current(), [b, c, hv, sub * 64, 0], order=[3, 4])
                    pl.move(uz.current(), ub.current())
                    pl.insert(ul.current(), uz.current(), [sub * 64, 0])
                    pl.load(ga.current(), gam, [b, hv, c, 0])
                    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=13)
                with pl.section_cube():
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=13)
                    pl.move(lua.current(), ul.current())
                    pl.move(rkb.current(), kl.current())
                    pl.matmul(dc.current(), lua.current(), rkb.current())
                    if not summary_mode:
                        pl.load(ml.current(), mqk, [b, hv, c, 0, 0])
                        pl.move(lma.current(), ml.current())
                        pl.move(rub.current(), ult.current())
                        pl.matmul_acc(oc.current(), oc.current(), lma.current(), rub.current(), phase=pl.AccPhase.Final)
                        pl.store(ov, oc.current(), [b, hv, c * 64, 0], order=[2, 3], phase=pl.STPhase.Final)
                    pl.move(de.current(), dc.current(), acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=12)
                with pl.section_vector():
                    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=12)
                    cp_state_update_vf(st.current(), de.current(), ga.current())
                    if save_cache and (not summary_mode or chain == 0):
                        pl.cast(sb.current(), st.current(), mode=pl.RoundMode.CAST_ROUND)
                        pl.store(h, sb.current(), [b, c + 1, hv, sub * 64, 0])
            with pl.section_vector():
                if summary_mode:
                    if chain == 1 and not first_continues:
                        cp_state_init_vf(st.current(), sub * 64, 0)
                    pl.store(summary, st.current(), [b, hv, chain, sub * 64, 0])
                else:
                    pl.store(fv, st.current(), [b, hv, sub * 64, 0])
                pl.system.bar_all()

        if summary_mode and not first_continues:
            with pl.section_vector():
                cp_state_init_vf(st.current(), sub * 64, 0)
                pl.store(summary, st.current(), [b, hv, 1, sub * 64, 0])
                pl.system.bar_all()


@pl.jit(auto_mutex=True)
def flash_kda_cp_stage2_kernel(
    qd: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_BF16],
    kr: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_BF16],
    gam: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_FP32],
    mqk: pl.Tensor[[pl.DYNAMIC] * 5, pl.DT_BF16],
    w: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_BF16],
    upre: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_FP32],
    initial_state: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_FP32],
    summary: pl.Tensor[[pl.DYNAMIC] * 5, pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_BF16],
    final_state: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_FP32],
    h: pl.Tensor[[pl.DYNAMIC] * 5, pl.DT_BF16],
    v_new: pl.Tensor[[pl.DYNAMIC] * 5, pl.DT_BF16],
    layout_code: pl.DT_INT64, cp_rank: pl.DT_INT64,
    first_continues: pl.DT_INT64, has_initial: pl.DT_INT64, save_cache: pl.DT_INT64,
    ratio: pl.DT_INT64,
    initial_bs: pl.DT_INT64,
    initial_ns: pl.DT_INT64,
    out_bs: pl.DT_INT64,
    out_ns: pl.DT_INT64,
    out_ss: pl.DT_INT64,
    final_bs: pl.DT_INT64,
    final_ns: pl.DT_INT64,
):
    cp_forward_chain(
        qd,
        kr,
        gam,
        mqk,
        w,
        upre,
        initial_state,
        summary,
        out,
        final_state,
        h,
        v_new,
        layout_code,
        cp_rank,
        first_continues,
        has_initial,
        save_cache,
        0,
        ratio,
        initial_bs,
        initial_ns,
        out_bs,
        out_ns,
        out_ss,
        final_bs,
        final_ns,
    )


def _cp_check_row_major_view(x, name):
    """Allow slices of row-major storage; reject broadcasts/overlap/transposes."""
    strides = x.stride()
    if strides[-1] != 1 or any(s <= 0 for s in strides):
        raise ValueError(f'{name} needs positive strides and a contiguous last axis')
    for d in range(x.ndim - 1):
        if x.shape[d] > 1 and strides[d] < strides[d + 1] * x.shape[d + 1]:
            raise ValueError(f'{name} must be a non-overlapping row-major slice')


def _cp_stride4(x, layout):
    st = x.stride()
    if layout == 'TND':
        return st[0] * x.shape[0], st[1], st[0]
    return (st[0], st[2], st[1]) if layout == 'BSND' else (st[0], st[1], st[2])


def _cp_input_strides(q, k, g, v, beta, initial, layout):
    result = tuple(s for x in (q, k, g, v) for s in _cp_stride4(x, layout))
    bs = beta.stride()
    if layout == 'TND':
        bs = (bs[0] * beta.shape[0], *bs)
    return (*result, *bs, initial.stride(0), initial.stride(1))


def _cp_output_strides(initial, out, final, layout):
    return (initial.stride(0), initial.stride(1), *_cp_stride4(out, layout),
            final.stride(0), final.stride(1))


def _cp_check_bucket(meta, num_key_heads, num_value_heads):
    gk = int(meta.get('global_Nk', num_key_heads))
    gv = int(meta.get('global_Nv', num_value_heads))
    ks = int(meta.get('hk_start', 0))
    vs = int(meta.get('hv_start', 0))
    if gk < 1 or gv < 1 or gv % gk or gv // gk != num_value_heads // num_key_heads:
        raise ValueError('bucket must preserve the global GVA ratio')
    if (
        ks < 0
        or vs < 0
        or ks + num_key_heads > gk
        or vs + num_value_heads > gv
        or vs != ks * (num_value_heads // num_key_heads)
    ):
        raise ValueError('bucket must contain complete contiguous key/value-head groups')


def _cp_bound_output(value, shape, dtype, device, name):
    if value is None:
        return torch.empty(shape, dtype=dtype, device=device)
    if tuple(value.shape) != tuple(shape) or value.dtype != dtype or value.device != device:
        raise ValueError(f'{name} binding shape/dtype/device mismatch')
    _cp_check_row_major_view(value, name)
    if name == 'final_state_out' and tuple(value.stride()[-2:]) != (128, 1):
        raise ValueError('final_state_out inner [V,K] matrix must be contiguous')
    return value


def _cp_check_output_aliases(out, final, prepared):
    # Conservative byte-span check: bindings must not alias live stage inputs.
    def span(x):
        lo = x.data_ptr()
        return lo, lo + (1 + sum((n - 1) * s for n, s in zip(x.shape, x.stride()))) * x.element_size()
    live = [x for x in prepared.values() if isinstance(x, torch.Tensor)]
    for dst, inputs in ((out, live + [final]), (final, live)):
        a, b = span(dst)
        for src in inputs:
            c, d = span(src)
            if a < d and c < b:
                raise ValueError('output bindings must not alias prepared tensors or each other')


def _cp_validate(q, k, v, g, beta, layout, cp_meta, lower_bound, allow_neg_eigval):
    if allow_neg_eigval:
        raise ValueError('CP v2 requires allow_neg_eigval=False (beta_scale=1)')
    if layout not in ('BSND', 'BNSD'):
        raise ValueError('layout must be BSND, BNSD or TND')
    if q.ndim != 4 or k.shape != q.shape or v.shape != g.shape:
        raise ValueError('invalid q/k/v/g dimensions')
    batch_size, seq_len, num_key_heads, head_dim = (
        q.shape if layout == 'BSND' else (q.shape[0], q.shape[2], q.shape[1], q.shape[3])
    )
    num_value_heads = v.shape[2] if layout == 'BSND' else v.shape[1]
    if (
        batch_size < 1
        or num_key_heads < 1
        or num_value_heads < 1
        or head_dim != 128
        or v.shape[-1] != 128
        or num_value_heads % num_key_heads
    ):
        raise ValueError('D=128 and Nv divisible by Nk are required')
    expected_v = (
        (batch_size, seq_len, num_value_heads, 128)
        if layout == 'BSND'
        else (batch_size, num_value_heads, seq_len, 128)
    )
    if tuple(v.shape) != expected_v:
        raise ValueError('v/g batch and sequence dimensions must match q/k')
    if seq_len < 1 or seq_len > 131072:
        raise ValueError('dense S must be in [1,131072]')
    expected_beta = (
        (batch_size, seq_len, num_value_heads) if layout == 'BSND' else (batch_size, num_value_heads, seq_len)
    )
    if tuple(beta.shape) != expected_beta:
        raise ValueError('beta shape mismatch')
    for x in (q, k, v, g, beta):
        if x.dtype != torch.bfloat16 or x.device != q.device:
            raise ValueError('raw inputs must be BF16 on the same device')
        _cp_check_row_major_view(x, 'raw input')
    _cp_check_bucket(cp_meta, num_key_heads, num_value_heads)
    if not -5 <= float(lower_bound) <= 0:
        raise ValueError('lower_bound must be in [-5,0]')
    rank, size = int(cp_meta['cp_rank']), int(cp_meta['cp_size'])
    if not 0 <= rank < size:
        raise ValueError('invalid CP rank/size')
    if cp_meta.get('last_continues', rank < size - 1) and seq_len % 64:
        raise ValueError('internal CP boundary must be chunk aligned')
    return batch_size, seq_len, num_key_heads, num_value_heads, rank, size


@pl.jit(auto_mutex=True)
def flash_kda_cp_cached_stage2_kernel(
    qd: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_BF16],
    mqk: pl.Tensor[[pl.DYNAMIC] * 5, pl.DT_BF16],
    h: pl.Tensor[[pl.DYNAMIC] * 5, pl.DT_BF16],
    v_new: pl.Tensor[[pl.DYNAMIC] * 5, pl.DT_BF16],
    summary: pl.Tensor[[pl.DYNAMIC] * 5, pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_BF16],
    final_state: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_FP32],
    layout_code: pl.DT_INT64,
    out_bs: pl.DT_INT64, out_ns: pl.DT_INT64, out_ss: pl.DT_INT64,
    final_bs: pl.DT_INT64, final_ns: pl.DT_INT64,
):
    """CP=1: consume native BF16 state caches produced by Stage1's E chain."""
    batch_size = qd.shape[0]
    seq_len = qd.shape[1] * (1 - layout_code) + qd.shape[2] * layout_code
    num_heads = qd.shape[2] * (1 - layout_code) + qd.shape[1] * layout_code
    nc = seq_len // 64
    hn = 128 * (1 - layout_code) + seq_len * 128 * layout_code
    sn = num_heads * 128 * (1 - layout_code) + 128 * layout_code
    qv = pl.make_tensor(qd, [batch_size, num_heads, seq_len, 128],
                        [num_heads * seq_len * 128, hn, sn, 1])
    ov = pl.make_tensor(out, [batch_size, num_heads, seq_len, 128], [out_bs, out_ns, out_ss, 1])
    fv = pl.make_tensor(final_state, [batch_size, num_heads, 128, 128], [final_bs, final_ns, 128, 1])
    qg = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16,
        target_memory=pl.MemorySpace.Mat), addrs=[0, 0x4000], mutex_ids=[0, 1])
    hg = pl.make_tile_group(type=pl.TileType(shape=[128, 128], dtype=pl.DT_BF16,
        target_memory=pl.MemorySpace.Mat), addrs=[0x8000, 0x10000], mutex_ids=[2, 3])
    mg = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16,
        target_memory=pl.MemorySpace.Mat), addrs=[0x18000, 0x1A000], mutex_ids=[4, 5])
    ug = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16,
        target_memory=pl.MemorySpace.Mat), addrs=[0x1C000, 0x20000], mutex_ids=[6, 7])
    qa = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16,
        target_memory=pl.MemorySpace.Left, layout=pl.NZ), addrs=[0, 0x4000], mutex_ids=[8, 9])
    ma = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16,
        target_memory=pl.MemorySpace.Left, layout=pl.NZ), addrs=[0x8000, 0xA000], mutex_ids=[10, 11])
    hb = pl.make_tile_group(type=pl.TileType(shape=[128, 128], dtype=pl.DT_BF16,
        target_memory=pl.MemorySpace.Right, layout=pl.ZN), addrs=[0, 0x8000], mutex_ids=[12, 13])
    # hb/ub are typed aliases of the same two L0B slots. Both cursors advance
    # once per task; H is consumed before U overwrites that slot. Distinct
    # ping/pong slots use different mutex IDs, while aliases share each ID.
    ub = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16,
        target_memory=pl.MemorySpace.Right, layout=pl.ZN), addrs=[0, 0x8000], mutex_ids=[12, 13])
    cg = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Acc), addrs=[0, 0x8000], mutex_ids=[16, 17])
    fg = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec), addrs=[0, 0x8000], mutex_ids=[0, 1])
    with pl.section_cube():
        core = pl.get_block_idx()
        cores = pl.get_block_num()
        for task in pl.range(core, batch_size * num_heads * nc, cores):
            bh = task // nc
            c = task % nc
            b = bh // num_heads
            hv = bh % num_heads
            qt = qg.next()
            ht = hg.next()
            mt = mg.next()
            ut = ug.next()
            at = qa.next()
            bt = hb.next()
            am = ma.next()
            bu = ub.next()
            ct = cg.next()
            pl.load(qt, qv, [b, hv, c * 64, 0])
            pl.load(ht, h, [b, c, hv, 0, 0], order=[4, 3])
            pl.load(mt, mqk, [b, hv, c, 0, 0])
            pl.load(ut, v_new, [b, c, hv, 0, 0], order=[4, 3])
            pl.move(at, qt)
            pl.move(bt, ht)
            pl.matmul(ct, at, bt, phase=pl.AccPhase.Partial)
            pl.move(am, mt)
            pl.move(bu, ut)
            pl.matmul_acc(ct, ct, am, bu, phase=pl.AccPhase.Final)
            pl.store(ov, ct, [b, hv, c * 64, 0], order=[2, 3], phase=pl.STPhase.Final)
    with pl.section_vector():
        core = pl.get_block_idx() // pl.get_subblock_num()
        cores = pl.get_block_num()
        sub = pl.get_subblock_idx()
        for task in pl.range(core, batch_size * num_heads, cores):
            b = task // num_heads
            hv = task % num_heads
            ft = fg.next()
            pl.load(ft, summary, [b, hv, 0, sub * 64, 0])
            pl.store(fv, ft, [b, hv, sub * 64, 0])


def flash_kda_stage1_wrapper(q, k, v, g, beta, scale, a_log, dt_bias, lower_bound,
        sequence_initial_state=None, *, cp_meta, layout_qkv='BSND',
        is_training=True, num_cores=None, allow_neg_eigval=False):
    if layout_qkv == 'TND':
        return _cp_stage1_packed(q, k, v, g, beta, scale, a_log, dt_bias, lower_bound,
            sequence_initial_state, cp_meta=cp_meta, is_training=is_training,
            num_cores=num_cores, allow_neg_eigval=allow_neg_eigval)
    batch_size, seq_len, num_key_heads, num_value_heads, rank, size = _cp_validate(
        q, k, v, g, beta, layout_qkv, cp_meta, lower_bound, allow_neg_eigval
    )
    if a_log.shape != (num_value_heads,) or dt_bias.shape != (num_value_heads, 128):
        raise ValueError('parameter shape mismatch')
    for p in (a_log, dt_bias):
        if p.dtype != torch.float32 or p.device != q.device or not p.is_contiguous():
            raise ValueError('parameters must be contiguous FP32 on input device')
    if sequence_initial_state is not None:
        if (
            sequence_initial_state.shape != (batch_size, num_value_heads, 128, 128)
            or sequence_initial_state.dtype != torch.float32
            or sequence_initial_state.device != q.device
        ):
            raise ValueError('initial_state must be FP32 [B,Nv,V,K]')
        _cp_check_row_major_view(sequence_initial_state, 'initial_state')
        if tuple(sequence_initial_state.stride()[-2:]) != (128, 1):
            raise ValueError('initial_state inner [V,K] matrix must be contiguous')
    cores_available, _ = _get_platform_config()
    cores = min(cores_available if num_cores is None else int(num_cores), batch_size * num_key_heads)
    if cores < 1 or cores > cores_available:
        raise ValueError('invalid num_cores')
    nc = (seq_len + 63) // 64
    padded_seq_len = nc * 64
    if is_training and size == 1 and not cp_meta.get('first_continues', rank > 0) and seq_len == padded_seq_len:
        cores = min(cores_available if num_cores is None else int(num_cores), batch_size * num_key_heads * nc)
    if cores < 1 or cores > cores_available:
        raise ValueError('invalid num_cores')
    device = q.device

    def alloc(shape, dtype):
        return torch.empty(shape, dtype=dtype, device=device)
    cshape = (
        (batch_size, padded_seq_len, num_value_heads, 128)
        if layout_qkv == 'BSND'
        else (batch_size, num_value_heads, padded_seq_len, 128)
    )
    prepared = {name: alloc(cshape, torch.bfloat16) for name in ('qd', 'kr', 'w')}
    prepared['gam'] = alloc((batch_size, num_value_heads, nc, 128), torch.float32)
    prepared['mqk'] = alloc((batch_size, num_value_heads, nc, 64, 64), torch.bfloat16)
    prepared['upre'] = alloc((batch_size, num_value_heads, 128, padded_seq_len), torch.float32)
    cache_shapes = {'q_c': ((batch_size, nc, num_key_heads, 64, 128), torch.bfloat16),
        'k_c': ((batch_size, nc, num_key_heads, 64, 128), torch.bfloat16),
        'v_c': ((batch_size, nc, num_value_heads, 64, 128), torch.bfloat16),
        'gk_c': ((batch_size, nc, num_value_heads, 64, 128), torch.float32),
        'beta_c': ((batch_size, nc, num_value_heads, 64), torch.float32),
        'akk': ((batch_size, nc, num_value_heads, 64, 64), torch.bfloat16)}
    launch_cache = {n: alloc(sh if is_training else (1,) * len(sh), dt) for n, (sh, dt) in cache_shapes.items()}
    prepared.update({n: t if is_training else None for n, t in launch_cache.items()})
    initial = sequence_initial_state if sequence_initial_state is not None else alloc((1, 1, 1, 1), torch.float32)
    first = bool(cp_meta.get('first_continues', rank > 0))
    reuse_local = bool(is_training and size == 1 and not first and seq_len == padded_seq_len)
    local_h = alloc((batch_size, nc + 1, num_value_heads, 128, 128) if reuse_local else (1, 1, 1, 1, 1), torch.bfloat16)
    local_v_new = alloc((batch_size, nc, num_value_heads, 128, 64) if reuse_local else (1, 1, 1, 1, 1), torch.bfloat16)
    if reuse_local:
        prepared['_local_h'] = local_h
        prepared['_local_v_new'] = local_v_new
    meta = dict(
        local_cache_reuse=reuse_local,
        cache_version='flash_kda_cp_train_v3_bf16_state',
        summary_version='flash_kda_cp_affine_mm16_v1',
        layout_qkv=layout_qkv,
        prepared_layout=layout_qkv,
        B=batch_size,
        S=seq_len,
        Nk=num_key_heads,
        Nv=num_value_heads,
        Hk=num_key_heads,
        Hv=num_value_heads,
        K=128,
        V=128,
        C=64,
        NC=nc,
        P=padded_seq_len,
        M=batch_size,
        J=nc,
        ratio=num_value_heads // num_key_heads,
        global_Nk=int(cp_meta.get('global_Nk', num_key_heads)),
        global_Nv=int(cp_meta.get('global_Nv', num_value_heads)),
        hk_start=int(cp_meta.get('hk_start', 0)),
        hv_start=int(cp_meta.get('hv_start', 0)),
        scale=float(scale),
        lower_bound=float(lower_bound),
        l2_eps=1e-6,
        beta_scale=1,
        safe_gate=True,
        allow_neg_eigval=False,
        state_v_first=True,
        is_training=bool(is_training),
        has_initial_state=sequence_initial_state is not None,
    )
    prepared.update(meta=meta, cp_meta=dict(cp_meta), sequence_initial_state=sequence_initial_state)
    summary = alloc((batch_size, num_value_heads, 2, 128, 128), torch.float32)
    workspace = alloc(
        (
            batch_size * padded_seq_len * num_key_heads * 128 * 4
            + batch_size * padded_seq_len * num_value_heads * 128 * 4,
        ),
        torch.uint8,
    )
    const_workspace = alloc((cores * 65536,), torch.uint8)
    qk_rstd = alloc(
        (
            (batch_size, nc, num_key_heads, 2, 64)
            if is_training and seq_len == padded_seq_len
            else (1, 1, 1, 1, 1)
        ),
        torch.float32,
    )
    if is_training and seq_len == padded_seq_len:
        prepared['qk_rstd'] = qk_rstd
    kernel = flash_kda_cp_stage1_kernel if seq_len == padded_seq_len else flash_kda_cp_stage1_tail_kernel
    local_rows = 128
    if reuse_local:
        while local_rows > 32 and batch_size * num_value_heads * (128 // local_rows) < cores:
            local_rows //= 2
    launch = kernel[None, cores, {"LocalStateRows": local_rows}] if seq_len == padded_seq_len else kernel[None, cores]
    launch(
        q, k, g, beta, v, a_log, dt_bias, float(lower_bound), float(scale), 0 if layout_qkv == 'BSND' else 1,
        prepared['qd'], prepared['kr'], prepared['gam'], prepared['mqk'], prepared['w'], prepared['upre'],
        *[launch_cache[n] for n in ('q_c', 'k_c', 'v_c', 'gk_c', 'beta_c', 'akk')],
        int(is_training), workspace, const_workspace, initial, summary, int(
            first), int(sequence_initial_state is not None),
        *_cp_input_strides(q, k, g, v, beta, initial, layout_qkv), *([qk_rstd, local_h, local_v_new, int(reuse_local)] if seq_len == padded_seq_len else []))
    return prepared, summary


def flash_kda_stage2_wrapper(prepared, all_fwd_summary, *, cp_meta, num_cores=None,
                            out=None, final_state_out=None):
    m = prepared['meta']
    if m['layout_qkv'] == 'TND':
        return _cp_stage2_packed(prepared, all_fwd_summary, cp_meta=cp_meta, num_cores=num_cores,
                                 out=out, final_state_out=final_state_out)
    batch_size, num_heads, seq_len, nc = m['B'], m['Hv'], m['P'], m['NC']
    rank, size = int(cp_meta['cp_rank']), int(cp_meta['cp_size'])
    if prepared['cp_meta'] != cp_meta:
        raise ValueError('cp_meta must remain unchanged between stages')
    qd = prepared['qd']
    if (
        all_fwd_summary.shape != (size * batch_size, num_heads, 2, 128, 128)
        or all_fwd_summary.dtype != torch.float32
        or not all_fwd_summary.is_contiguous()
        or all_fwd_summary.device != qd.device
    ):
        raise ValueError('gathered summary shape/dtype/device mismatch')

    def alloc(shape, dtype):
        return torch.empty(shape, dtype=dtype, device=qd.device)
    initial = prepared['sequence_initial_state']
    if initial is None:
        initial = alloc((1, 1, 1, 1), torch.float32)
    oshape = (batch_size, m['S'], num_heads, 128) if m['layout_qkv'] == 'BSND' else (batch_size, num_heads, m['S'], 128)
    out = _cp_bound_output(out, oshape, torch.bfloat16, qd.device, 'out')
    final = _cp_bound_output(
        final_state_out, (batch_size, num_heads, 128, 128), torch.float32, qd.device, 'final_state_out'
    )
    _cp_check_output_aliases(out, final, prepared)
    train = m['is_training']
    reuse_local = bool(m.get('local_cache_reuse', False))
    h = prepared['_local_h'] if reuse_local else alloc((batch_size, nc + 1, num_heads, 128, 128) if train else (1, 1, 1, 1, 1), torch.bfloat16)
    v_new = prepared['_local_v_new'] if reuse_local else alloc((batch_size, nc, num_heads, 128, 64) if train else (1, 1, 1, 1, 1), torch.bfloat16)
    available, _ = _get_platform_config()
    cores = min(available if num_cores is None else int(num_cores), batch_size * m['Hk'] * (nc if reuse_local else 1))
    if cores < 1 or cores > available:
        raise ValueError('invalid num_cores')
    if reuse_local:
        strides = _cp_output_strides(initial, out, final, m['layout_qkv'])
        flash_kda_cp_cached_stage2_kernel[None, cores](
            prepared['qd'], prepared['mqk'], h, v_new, all_fwd_summary, out, final,
            0 if m['prepared_layout'] == 'BSND' else 1, *strides[2:])
    else:
        kernel = flash_kda_cp_stage2_kernel if m['S'] == m['P'] else flash_kda_cp_stage2_tail_kernel
        kernel[None, cores](
            *[prepared[n] for n in ('qd', 'kr', 'gam', 'mqk', 'w', 'upre')], initial, all_fwd_summary,
            out, final, h, v_new, 0 if m['prepared_layout'] == 'BSND' else 1, rank,
            int(cp_meta.get('first_continues', rank > 0)), int(m['has_initial_state']), int(train), m['ratio'],
            *_cp_output_strides(initial, out, final, m['layout_qkv']))
    cache = None
    if train:
        cache = {n: prepared[n] for n in ('q_c', 'k_c', 'v_c', 'gk_c', 'beta_c', 'akk', 'mqk', 'w', 'kr', 'gam')}
        cache.update(h=h, v_new=v_new, meta=dict(m), cp_meta=prepared['cp_meta'])
        if 'qk_rstd' in prepared:
            cache['qk_rstd'] = prepared['qk_rstd']
    return out, final, cache


def cp_forward_chain_tnd(
    qd,
    kr,
    gam,
    mqk,
    w,
    upre,
    initial_state,
    summary,
    out,
    final_state,
    h,
    v_new,
    layout_code,
    cp_rank,
    first_continues,
    has_initial,
    save_cache,
    summary_mode,
    ratio,
    cu,
    chunk_offsets,
    h_offsets,
    initial_bs,
    initial_ns,
    out_bs,
    out_ns,
    out_ss,
    final_bs,
    final_ns,
):
    """Inline device region: one owner per batch/value-head, two AIV row halves.

    summary_mode computes E and T after the owner's preprocessing has drained.
    Otherwise it scans preceding packets and executes the real forward chain.
    Both callers are single-launch JITs. No host-side arithmetic is involved.
    """
    core = pl.get_block_idx() // pl.get_subblock_num()
    cores = pl.get_block_num()
    sub = pl.get_subblock_idx()
    batch_size = qd.shape[0]
    seq_len = qd.shape[1] * (1 - layout_code) + qd.shape[2] * layout_code
    num_heads = qd.shape[2] * (1 - layout_code) + qd.shape[1] * layout_code
    nc = seq_len // 64
    num_sequences = cu.shape[0] - 1
    token_count = out.shape[1]
    hn = 128 * (1 - layout_code) + seq_len * 128 * layout_code
    sn = num_heads * 128 * (1 - layout_code) + 128 * layout_code
    qv = pl.make_tensor(qd, [batch_size, num_heads, seq_len, 128], [num_heads * seq_len * 128, hn, sn, 1])
    kv = pl.make_tensor(kr, [batch_size, num_heads, seq_len, 128], [num_heads * seq_len * 128, hn, sn, 1])
    wv = pl.make_tensor(w, [batch_size, num_heads, seq_len, 128], [num_heads * seq_len * 128, hn, sn, 1])
    ov = pl.make_tensor(out, [1, num_heads, token_count, 128], [out_bs, out_ns, out_ss, 1])
    iv = pl.make_tensor(initial_state, [initial_state.shape[0], num_heads, 128, 128], [initial_bs, initial_ns, 128, 1])
    fv = pl.make_tensor(final_state, [final_state.shape[0], num_heads, 128, 128], [final_bs, final_ns, 128, 1])
    # UB high-water 0x35000 (< 256 KiB), half state [64 rows,128 columns].
    st = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec), addrs=[0], mutex_ids=[0])
    de = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec), addrs=[0x8000], mutex_ids=[1])
    sb = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Vec), addrs=[0x10000], mutex_ids=[2])
    sz = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
        addrs=[0x14000],
        mutex_ids=[3],
    )
    ud = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec), addrs=[0x18000], mutex_ids=[4])
    up = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec), addrs=[0x1C000], mutex_ids=[5])
    ub = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Vec), addrs=[0x20000], mutex_ids=[6])
    uz = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
        addrs=[0x22000],
        mutex_ids=[7],
    )
    ga = pl.make_tile_group(type=pl.TileType(shape=[1, 128], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec), addrs=[0x24000], mutex_ids=[8])
    tf = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec), addrs=[0x25000], mutex_ids=[9])
    tb = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Vec), addrs=[0x2D000], mutex_ids=[10])
    tz = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
        addrs=[0x31000],
        mutex_ids=[11],
    )
    # L1 views share the same backing for transpose, with the same mutex.
    sl = pl.make_tile_group(type=pl.TileType(
        shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=[0], mutex_ids=[0])
    slt = pl.make_tile_group(type=pl.TileType(
        shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN), addrs=[0], mutex_ids=[0])
    wl = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Mat), addrs=[0x8000], mutex_ids=[1])
    kl = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Mat), addrs=[0xC000], mutex_ids=[2])
    ul = pl.make_tile_group(
        type=pl.TileType(shape=[128, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[0x10000],
        mutex_ids=[3],
    )
    ult = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x10000],
        mutex_ids=[3],
    )
    ql = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Mat), addrs=[0x14000], mutex_ids=[4])
    ml = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Mat), addrs=[0x18000], mutex_ids=[5])
    tl = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[0x20000],
        mutex_ids=[6],
    )
    la = pl.make_tile_group(type=pl.TileType(
        shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left, layout=pl.NZ), addrs=[0], mutex_ids=[13])
    lua = pl.make_tile_group(type=pl.TileType(
        shape=[128, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left, layout=pl.NZ), addrs=[0], mutex_ids=[13])
    lqa = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=[0x8000],
        mutex_ids=[14],
    )
    lma = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=[0xC000],
        mutex_ids=[15],
    )
    rb = pl.make_tile_group(type=pl.TileType(
        shape=[128, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right, layout=pl.ZN), addrs=[0], mutex_ids=[16])
    rkb = pl.make_tile_group(type=pl.TileType(
        shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right, layout=pl.ZN), addrs=[0], mutex_ids=[16])
    rtb = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=[0],
        mutex_ids=[16],
    )
    rsb = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=[0x8000],
        mutex_ids=[17],
    )
    rub = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=[0x8000],
        mutex_ids=[17],
    )
    uc = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=[0],
        mutex_ids=[18],
    )
    dc = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=[0x10000],
        mutex_ids=[19],
    )
    oc = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Acc), addrs=[0x20000], mutex_ids=[20])
    tail_f = pl.make_tile_group(
        type=pl.TileType(
            shape=[32, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
        ),
        addrs=[0x35000],
        mutex_ids=[12],
    )
    tail_b = pl.make_tile_group(
        type=pl.TileType(
            shape=[32, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
        ),
        addrs=[0x39000],
        mutex_ids=[13],
    )
    group_count = (batch_size * num_heads // ratio - core + cores - 1) // cores
    for owned in pl.range(0, group_count * ratio):
        task = (core + (owned // ratio) * cores) * ratio + owned % ratio
        b = task // num_heads
        hv = task % num_heads
        for seq in pl.range(num_sequences - 1 if summary_mode else 0, num_sequences):
            first_chunk = chunk_offsets[seq]
            end_chunk = chunk_offsets[seq + 1]
            continues = 1 if first_continues and seq == 0 else 0
            for chain in pl.range(0, 1 + summary_mode):
                with pl.section_vector():
                    if summary_mode and chain == 1:
                        cp_state_init_vf(st.current(), sub * 64, 1)
                    elif not continues and has_initial:
                        pl.load(st.current(), iv, [seq, hv, sub * 64, 0])
                    else:
                        cp_state_init_vf(st.current(), sub * 64, 0)
                if not summary_mode and continues:
                    for rank in pl.range(0, cp_rank):
                        with pl.section_vector():
                            pl.cast(sb.current(), st.current(), mode=pl.RoundMode.CAST_ROUND)
                            pl.move(sz.current(), sb.current())
                            pl.insert(sl.current(), sz.current(), [sub * 64, 0])
                            pl.load(tf.current(), summary, [rank * batch_size + b, hv, 1, sub * 64, 0])
                            pl.cast(tb.current(), tf.current(), mode=pl.RoundMode.CAST_ROUND)
                            pl.move(tz.current(), tb.current())
                            pl.insert(tl.current(), tz.current(), [sub * 64, 0])
                            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=13)
                        with pl.section_cube():
                            pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=13)
                            pl.move(la.current(), sl.current())
                            pl.move(rtb.current(), tl.current())
                            pl.matmul(dc.current(), la.current(), rtb.current())
                            pl.move(de.current(), dc.current(), acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=12)
                        with pl.section_vector():
                            pl.load(tf.current(), summary, [rank * batch_size + b, hv, 0, sub * 64, 0])
                            pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=12)
                            cp_state_add_vf(st.current(), de.current(), tf.current())
                with pl.section_vector():
                    if not summary_mode and save_cache:
                        pl.cast(sb.current(), st.current(), mode=pl.RoundMode.CAST_ROUND)
                        pl.store(h, sb.current(), [0, h_offsets[seq], hv, sub * 64, 0])
                for c in pl.range(first_chunk, end_chunk):
                    raw_start = cu[seq] + (c - first_chunk) * 64
                    valid = pl.min(64, cu[seq + 1] - raw_start)
                    with pl.section_vector():
                        pl.cast(sb.current(), st.current(), mode=pl.RoundMode.CAST_ROUND)
                        pl.move(sz.current(), sb.current())
                        pl.insert(sl.current(), sz.current(), [sub * 64, 0])
                        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=13)
                    with pl.section_cube():
                        pl.load(wl.current(), wv, [b, hv, c * 64, 0], order=[3, 2])
                        pl.load(kl.current(), kv, [b, hv, c * 64, 0])
                        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=13)
                        pl.move(la.current(), sl.current())
                        pl.move(rb.current(), wl.current())
                        pl.matmul(uc.current(), la.current(), rb.current())
                        pl.move(ud.current(), uc.current(), acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                        if not summary_mode:
                            pl.load(ql.current(), qv, [b, hv, c * 64, 0])
                            pl.move(lqa.current(), ql.current())
                            pl.move(rsb.current(), slt.current())
                            pl.matmul(oc.current(), lqa.current(), rsb.current(), phase=pl.AccPhase.Partial)
                        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=12)
                    with pl.section_vector():
                        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=12)
                        if summary_mode and chain == 1:
                            pl.cast(ub.current(), ud.current(), mode=pl.RoundMode.CAST_ROUND)
                        else:
                            pl.load(up.current(), upre, [b, hv, sub * 64, c * 64])
                            m7_ut_add_upre_bf16_vf(ud.current(), up.current(), ub.current(), 0, 128)
                        if not summary_mode and save_cache:
                            pl.store(v_new, ub.current(), [b, c, hv, sub * 64, 0], order=[3, 4])
                        pl.move(uz.current(), ub.current())
                        pl.insert(ul.current(), uz.current(), [sub * 64, 0])
                        pl.load(ga.current(), gam, [b, hv, c, 0])
                        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=13)
                    with pl.section_cube():
                        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=13)
                        pl.move(lua.current(), ul.current())
                        pl.move(rkb.current(), kl.current())
                        pl.matmul(dc.current(), lua.current(), rkb.current())
                        if not summary_mode:
                            pl.load(ml.current(), mqk, [b, hv, c, 0, 0])
                            pl.move(lma.current(), ml.current())
                            pl.move(rub.current(), ult.current())
                            pl.matmul_acc(oc.current(), oc.current(), lma.current(),
                                          rub.current(), phase=pl.AccPhase.Final)
                            if valid == 64:
                                pl.store(ov, oc.current(), [0, hv, raw_start, 0], order=[2, 3], phase=pl.STPhase.Final)
                            else:
                                pl.set_validshape(tail_f.current(), [32, 128])
                                pl.move(tail_f.current(), oc.current(),
                                        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM, phase=pl.STPhase.Final)
                                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=10)
                        pl.move(de.current(), dc.current(), acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=12)
                    with pl.section_vector():
                        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=12)
                        if not summary_mode and valid < 64:
                            pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=10)
                            rows = pl.max(0, pl.min(32, valid - sub * 32))
                            if rows > 0:
                                pl.set_validshape(tail_f.current(), [rows, 128])
                                pl.set_validshape(tail_b.current(), [rows, 128])
                                pl.cast(tail_b.current(), tail_f.current(), mode=pl.RoundMode.CAST_ROUND)
                                pl.store(ov, tail_b.current(), [0, hv, raw_start + sub * 32, 0], order=[2, 3])
                        cp_state_update_vf(st.current(), de.current(), ga.current())
                        if not summary_mode and save_cache:
                            pl.cast(sb.current(), st.current(), mode=pl.RoundMode.CAST_ROUND)
                            pl.store(h, sb.current(), [0, h_offsets[seq] + c - first_chunk + 1, hv, sub * 64, 0])
                with pl.section_vector():
                    if summary_mode:
                        if chain == 1 and not continues:
                            cp_state_init_vf(st.current(), sub * 64, 0)
                        pl.store(summary, st.current(), [b, hv, chain, sub * 64, 0])
                    else:
                        pl.store(fv, st.current(), [seq, hv, sub * 64, 0])
                    pl.system.bar_all()


@pl.jit(auto_mutex=True)
def flash_kda_cp_stage1_tnd_kernel(
    chunk_starts: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64],
    chunk_lengths: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64],
    # 输入
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    g: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    beta: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    a_log: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32],
    dt_bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    lower_bound: pl.DT_FP32,
    scale: pl.DT_FP32,
    layout_code: pl.DT_INT64,
    # 输出
    qd_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    kr_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    gam_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    mqk_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    w_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    upre_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    q_c_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    k_c_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    v_c_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    gk_c_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    beta_c_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    akk_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    save_cache: pl.DT_INT64,
    # workspace
    workspace: pl.Ptr[pl.DT_UINT8],
    const_workspace: pl.Ptr[pl.DT_UINT8],
    initial_state: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_FP32],
    summary: pl.Tensor[[pl.DYNAMIC] * 5, pl.DT_FP32],
    first_continues: pl.DT_INT64, has_initial: pl.DT_INT64,
    cu: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64],
    chunk_offsets: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64],
    h_offsets: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64],
    q_bs: pl.DT_INT64,
    q_ns: pl.DT_INT64,
    q_ss: pl.DT_INT64,
    k_bs: pl.DT_INT64,
    k_ns: pl.DT_INT64,
    k_ss: pl.DT_INT64,
    g_bs: pl.DT_INT64,
    g_ns: pl.DT_INT64,
    g_ss: pl.DT_INT64,
    v_bs: pl.DT_INT64,
    v_ns: pl.DT_INT64,
    v_ss: pl.DT_INT64,
    beta_bs: pl.DT_INT64,
    beta_s1: pl.DT_INT64,
    beta_s2: pl.DT_INT64,
    initial_bs: pl.DT_INT64,
    initial_ns: pl.DT_INT64,
):
    cc_cid = pl.get_block_idx() // pl.get_subblock_num()
    cc_base = const_workspace + cc_cid * 65536
    lower_tril = pl.make_tensor(cc_base, [64, 64], [64, 1], dtype=pl.DT_FP32)
    sl_mask = pl.make_tensor(cc_base + 16384, [64, 128], [128, 1], dtype=pl.DT_BF16)
    eye64 = pl.make_tensor(cc_base + 32768, [64, 64], [64, 1], dtype=pl.DT_FP32)
    lb_t = pl.make_tensor(cc_base + 49152, [1, 64], [64, 1], dtype=pl.DT_FP32)
    scale_t = pl.make_tensor(cc_base + 49408, [1, 64], [64, 1], dtype=pl.DT_FP32)
    cc_a = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec), addrs=[0], mutex_ids=[0])
    cc_b = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec), addrs=[0x4000], mutex_ids=[1])
    cc_bb = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16,
        target_memory=pl.MemorySpace.Vec), addrs=[0xC000], mutex_ids=[2])
    cc_e = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec), addrs=[0x10000], mutex_ids=[3])
    cc_l = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec), addrs=[0x14000], mutex_ids=[4])
    cc_s = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec), addrs=[0x14100], mutex_ids=[5])
    with pl.section_vector():
        if pl.get_subblock_idx() == 0:
            cp_constants_vf(cc_a.current(), cc_b.current(), cc_e.current(), cc_l.current(),
                            cc_s.current(), lower_bound, scale)
            pl.cast(cc_bb.current(), cc_b.current(), mode=pl.RoundMode.CAST_ROUND)
            pl.store(lower_tril, cc_a.current(), [0, 0])
            pl.store(sl_mask, cc_bb.current(), [0, 0])
            pl.store(eye64, cc_e.current(), [0, 0])
            pl.store(lb_t, cc_l.current(), [0, 0])
            pl.store(scale_t, cc_s.current(), [0, 0])
        pl.system.bar_all()
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=13)
    with pl.section_cube():
        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=13)
        pl.system.bar_all()
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=12)
    with pl.section_vector():
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=12)
    beta_view = pl.make_tensor(beta, [beta.shape[0], beta.shape[1], beta.shape[2]], [beta_bs, beta_s1, beta_s2])
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    num_cores = pl.get_block_num()

    batch_size = g.shape[0]
    token_count = g.shape[1]
    seq_len = qd_out.shape[1]
    num_key_heads = q.shape[2] * (1 - layout_code) + q.shape[1] * layout_code
    num_value_heads = g.shape[2] * (1 - layout_code) + g.shape[1] * layout_code
    nq = num_value_heads // num_key_heads
    nc = seq_len // CHUNK_C
    total = batch_size * num_value_heads
    tt_b = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                       target_memory=pl.MemorySpace.Vec)
    tt_f32 = pl.TileType(shape=[HS, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec)
    tt_b32 = pl.TileType(shape=[HS, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec)
    tt_c64h = pl.TileType(shape=[HS, CHUNK_C], dtype=pl.DT_FP32,
                          target_memory=pl.MemorySpace.Vec)
    tt_c64bh = pl.TileType(shape=[HS, CHUNK_C], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Vec)
    tt_a = pl.TileType(shape=[CHUNK_C, 1], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Vec,
                       valid_shape=[-1, -1])
    tt_d = pl.TileType(shape=[1, HEAD_D], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Vec)
    tt_c64 = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec)
    tt_c64b = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                          target_memory=pl.MemorySpace.Vec)

    gate_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[L1_G0, L1_G1], mutex_ids=[11, 12],
        fwd_ids=_GATE_FWD_IDS, bwd_ids=_GATE_BWD_IDS,
    )

    cs_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[VA_CS0], mutex_ids=[16],
        fwd_ids=(4,), bwd_ids=(6,),
    )

    tt_tnd_input = pl.TileType(shape=[HS, HEAD_D], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    x_b_g = pl.make_tile_group(type=tt_tnd_input, addrs=[VA_X_B], mutex_ids=[0])
    qn_h_g = pl.make_tile_group(
        type=tt_b32, addrs=[0x34800, 0x36800], mutex_ids=[10, 13])
    kn_h_g = pl.make_tile_group(
        type=tt_b32, addrs=[0x38800, 0x3A800], mutex_ids=[14, 15])
    xk_g = pl.make_tile_group(type=tt_tnd_input, addrs=[0x21000], mutex_ids=[17])

    xg_g = pl.make_tile_group(type=tt_tnd_input, addrs=[0x2000], mutex_ids=[18])
    gate_f_g = pl.make_tile_group(type=tt_f32, addrs=[VA_GATE_F], mutex_ids=[5])
    gate_nz_g = pl.make_tile_group(
        type=pl.TileType(shape=[HS, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                         valid_shape=[-1, -1], compact=1),
        addrs=[VA_GATE_NZ], mutex_ids=[6])
    a_g = pl.make_tile_group(type=tt_a, addrs=[VA_A_LOG], mutex_ids=[8])
    d_g = pl.make_tile_group(type=tt_d, addrs=[VA_DT_B], mutex_ids=[9])
    lb_g = pl.make_tile_group(type=tt_a, addrs=[VA_LB], mutex_ids=[7])

    m3_out_b_g = pl.make_tile_group(type=tt_b32, addrs=[VA_QN_B], mutex_ids=[3])
    m3_csl_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[0x20800], mutex_ids=[21])
    m3_sc_g = pl.make_tile_group(type=tt_a, addrs=[VA_SC], mutex_ids=[22])

    m4_kn_b_g = pl.make_tile_group(type=tt_b, addrs=[VA_KN_B], mutex_ids=[4])
    m4_qn_b_g = pl.make_tile_group(type=tt_b, addrs=[VA_QN_B], mutex_ids=[3])
    m4_opscr_b16_g = pl.make_tile_group(
        type=pl.TileType(shape=[ROW_BLOCK, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[VA_M4_SCR_B16, VA_M4_SCR_B16 + 0x1000], mutex_ids=[6, 19])
    m4_ref_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[0x20A00], mutex_ids=[30])

    _tt_op16 = pl.TileType(shape=[ROW_BLOCK, HEAD_D], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                           valid_shape=[-1, -1], compact=1)

    lk0_g = pl.make_tile_group(type=_tt_op16, addrs=[0x14000, 0x15000], mutex_ids=[31, 27])
    lk1_g = pl.make_tile_group(type=_tt_op16, addrs=[0x16000, 0x17000], mutex_ids=[28, 29])
    lk2_g = pl.make_tile_group(type=_tt_op16, addrs=[0x18000, 0x19000], mutex_ids=[30, 31])
    lk3_g = pl.make_tile_group(type=_tt_op16, addrs=[0x1A000, 0x1B000], mutex_ids=[27, 28])
    lq0_g = pl.make_tile_group(type=_tt_op16, addrs=[0x1C000, 0x1D000], mutex_ids=[29, 30])
    lq1_g = pl.make_tile_group(type=_tt_op16, addrs=[0x1E000, 0x1F000], mutex_ids=[31, 27])
    lq2_g = pl.make_tile_group(type=_tt_op16, addrs=[0x20000, 0x21000], mutex_ids=[28, 29])
    lq3_g = pl.make_tile_group(type=_tt_op16, addrs=[0x22000, 0x23000], mutex_ids=[30, 31])
    _tt_rw = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                         valid_shape=[-1, -1], compact=1)
    _tt_rz64 = pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Mat, layout=pl.ZN,
                           valid_shape=[-1, -1], compact=1)
    rw0_g = pl.make_tile_group(type=_tt_rw, addrs=[0x34000, 0x56000], mutex_ids=[27, 28])
    rw1_g = pl.make_tile_group(type=_tt_rw, addrs=[0x38000, 0x5A000], mutex_ids=[29, 30])
    rw2_g = pl.make_tile_group(type=_tt_rw, addrs=[0x3C000, 0x5E000], mutex_ids=[31, 27])
    rw3_g = pl.make_tile_group(type=_tt_rw, addrs=[0x40000, 0x62000], mutex_ids=[28, 29])
    rz0_g = pl.make_tile_group(type=_tt_rz64, addrs=[0x34000, 0x56000], mutex_ids=[27, 28])
    rz1_g = pl.make_tile_group(type=_tt_rz64, addrs=[0x38000, 0x5A000], mutex_ids=[29, 30])
    rz2_g = pl.make_tile_group(type=_tt_rz64, addrs=[0x3C000, 0x5E000], mutex_ids=[31, 27])
    rz3_g = pl.make_tile_group(type=_tt_rz64, addrs=[0x40000, 0x62000], mutex_ids=[28, 29])

    # kn_h_g[1] ends at 0x3C800; the lagged M5 still needs its last rows.
    opnz_g = pl.make_tile_group(
        type=pl.TileType(shape=[ROW_BLOCK, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                         valid_shape=[-1, -1], compact=1),
        addrs=[0x3C800],
        mutex_ids=[2])
    m4_la_g = pl.make_tile_group(
        type=pl.TileType(shape=[ROW_BLOCK, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=[0x4000, 0xD000], mutex_ids=[13, 23])
    m4_b_g = pl.make_tile_group(
        type=pl.TileType(shape=[HEAD_D, ROW_BLOCK], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=[0x8000, 0x9000, 0xA000, 0xB000],
        mutex_ids=[14, 16, 17, 18])
    m4_c_g = pl.make_tile_group(
        type=pl.TileType(shape=[ROW_BLOCK, ROW_BLOCK], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         fractal=1024),
        addrs=[0x8000, 0x8400, 0x8800, 0x8C00],
        mutex_ids=[15, 19, 20, 21])

    lower_l1_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[L1_L], mutex_ids=[10])
    lower_l0a_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=[LA0], mutex_ids=[13])
    g_l0b_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=[RA0], mutex_ids=[14])
    acc_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         fractal=1024),
        addrs=[CA0], mutex_ids=[15])

    m5_mask_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[0x28000], mutex_ids=[26])
    m5_kkt_g = pl.make_tile_group(
        type=tt_c64h, addrs=[VA_M5_KKT], mutex_ids=[1])
    m5_mqk_g = pl.make_tile_group(
        type=tt_c64h, addrs=[VA_M5_MQK], mutex_ids=[2])
    m5_mqk_b_g = pl.make_tile_group(
        type=tt_c64bh, addrs=[0xE100], mutex_ids=[2])

    m5_beta_raw_b_g = pl.make_tile_group(
        type=tt_c64b, addrs=[0x12000], mutex_ids=[2])
    m5_beta_blk_f_g = pl.make_tile_group(
        type=tt_c64, addrs=[VA_M5_BETA_F], mutex_ids=[3])
    m5_v_b_g = pl.make_tile_group(
        type=tt_tnd_input, addrs=[VA_M5_VB], mutex_ids=[4])
    m5_v_operand_g = pl.make_tile_group(
        type=tt_b32, addrs=[VA_M5_VF], mutex_ids=[6])
    m5_kd_operand_g = pl.make_tile_group(
        type=tt_b32, addrs=[VA_M5_KNF], mutex_ids=[5])

    beta_cache_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, HS], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[0x3D800, 0x3DC00], mutex_ids=[23, 25])

    m5_kd_nz_g = pl.make_tile_group(
        type=pl.TileType(shape=[HS, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                         valid_shape=[-1, -1], compact=1),
        addrs=[P3_UB_XY], mutex_ids=[16])

    tt_m6_l64 = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                            valid_shape=[-1, -1])
    tt_m6_lc128 = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                              fractal=1024)
    tt_m6_vhs = pl.TileType(shape=[HS, CHUNK_C], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec,
                            pad=pl.TilePad.zero, valid_shape=[-1, -1])
    tt_m6_vhs33 = pl.TileType(shape=[HS + 1, CHUNK_C], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Vec,
                              pad=pl.TilePad.zero, valid_shape=[-1, -1])
    tt_m6_nz = pl.TileType(shape=[HS + 1, CHUNK_C], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Vec, layout=pl.NZ)
    x_ub_g = pl.make_tile_group(type=tt_m6_vhs, addrs=[P3_UB_X], mutex_ids=[15])
    xy_ub_g = pl.make_tile_group(type=tt_m6_vhs33, addrs=[P3_UB_XY], mutex_ids=[16])
    tt_m6_akk_ub = pl.TileType(shape=[HS, CHUNK_C], dtype=pl.DT_BF16,
                                target_memory=pl.MemorySpace.Vec,
                                valid_shape=[-1, -1])
    m6_akk_ub_g = pl.make_tile_group(type=tt_m6_akk_ub,
                                     addrs=[0x3E000, 0x3F000], mutex_ids=[24, 29])

    snz_ub_g = pl.make_tile_group(type=tt_m6_nz, addrs=[P3_UB_XY], mutex_ids=[16])
    y_l1_g = pl.make_tile_group(type=tt_m6_l64, addrs=[P3_L1_Y0, P3_L1_Y1],
                                mutex_ids=[0, 1])
    _t71_l1_fp16 = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                               target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                               fractal=512, compact=0)
    t71_n_db_g = pl.make_tile_group(type=_t71_l1_fp16,
                                    addrs=[0x76000, 0x78000],
                                    mutex_ids=[24, 25])
    t71_eye = pl.make_tile(_t71_l1_fp16, addr=0x7A000)
    t71_zero = pl.make_tile(_t71_l1_fp16, addr=0x7C000)
    t71_scratch_g = pl.make_tile_group(type=_t71_l1_fp16, addrs=172032, mutex_ids=[26])
    t71_p_g = pl.make_tile_group(type=_t71_l1_fp16, addrs=180224, mutex_ids=[27])
    t71_tmp_g = pl.make_tile_group(type=_t71_l1_fp16, addrs=188416, mutex_ids=[28])
    _t71_l1_fp16_d = pl.TileType(shape=[16, 16], dtype=pl.DT_FP16,
                                 target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                                 fractal=512, compact=0)
    t71_tmpd0_g = pl.make_tile_group(type=_t71_l1_fp16_d, addrs=188416, mutex_ids=[28])
    t71_tmpd1_g = pl.make_tile_group(type=_t71_l1_fp16_d, addrs=188416 + 0xA00, mutex_ids=[29])
    t71_tmpd2_g = pl.make_tile_group(type=_t71_l1_fp16_d, addrs=188416 + 0x1400, mutex_ids=[30])
    t71_tmpd3_g = pl.make_tile_group(type=_t71_l1_fp16_d, addrs=188416 + 0x1E00, mutex_ids=[31])
    _t71_l1_bf16 = pl.TileType(shape=[64, 64], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                               fractal=512, compact=0)
    t71_inv_bf16_g = pl.make_tile_group(type=_t71_l1_bf16, addrs=147456, mutex_ids=[29])
    t71_eye_bf16 = pl.make_tile(_t71_l1_bf16, addr=0x7E000)

    _t71_a = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ,
                         fractal=512, compact=0)
    _t71_b = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                         fractal=512, compact=0)
    t71_a_g = pl.make_tile_group(type=_t71_a, addrs=[0x5000, 0x7000], mutex_ids=[18, 19])
    t71_b_g = pl.make_tile_group(type=_t71_b, addrs=[0xC000, 0xE000], mutex_ids=[20, 21])
    _t71_ad = pl.TileType(shape=[16, 16], dtype=pl.DT_FP16,
                          target_memory=pl.MemorySpace.Left, layout=pl.NZ,
                          fractal=512, compact=0)
    _t71_bd = pl.TileType(shape=[16, 16], dtype=pl.DT_FP16,
                          target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                          fractal=512, compact=0)
    t71_ad0_g = pl.make_tile_group(type=_t71_ad, addrs=[0x5000, 0x7000], mutex_ids=[18, 19])
    t71_ad1_g = pl.make_tile_group(type=_t71_ad, addrs=[0x5A00, 0x7A00], mutex_ids=[0, 1])
    t71_ad2_g = pl.make_tile_group(type=_t71_ad, addrs=[0x6400, 0x8400], mutex_ids=[2, 3])
    t71_ad3_g = pl.make_tile_group(type=_t71_ad, addrs=[0x6E00, 0x8E00], mutex_ids=[4, 5])
    t71_a10_g = pl.make_tile_group(type=_t71_ad, addrs=[0x5200, 0x7200], mutex_ids=[18, 19])
    t71_a32_g = pl.make_tile_group(type=_t71_ad, addrs=[0x6600, 0x8600], mutex_ids=[18, 19])
    t71_bd0_g = pl.make_tile_group(type=_t71_bd, addrs=[0xC000, 0xE000], mutex_ids=[20, 21])
    t71_bd1_g = pl.make_tile_group(type=_t71_bd, addrs=[0xCA00, 0xEA00], mutex_ids=[6, 7])
    t71_bd2_g = pl.make_tile_group(type=_t71_bd, addrs=[0xD400, 0xF400], mutex_ids=[8, 9])
    t71_bd3_g = pl.make_tile_group(type=_t71_bd, addrs=[0xDE00, 0xFE00], mutex_ids=[10, 11])
    t71_b10_g = pl.make_tile_group(type=_t71_bd, addrs=[0xC800, 0xE800], mutex_ids=[20, 21])
    t71_b32_g = pl.make_tile_group(type=_t71_bd, addrs=[0xDC00, 0xFC00], mutex_ids=[20, 21])
    _t71_bt = pl.TileType(shape=[16, 32], dtype=pl.DT_FP16,
                          target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                          fractal=512, compact=0)
    t71_bt0_g = pl.make_tile_group(type=_t71_bt, addrs=[0xC000, 0xE000], mutex_ids=[20, 21])
    t71_bt1_g = pl.make_tile_group(type=_t71_bt, addrs=[0xC800, 0xE800], mutex_ids=[20, 21])
    t71_bl0_g = pl.make_tile_group(type=_t71_bt, addrs=[0xD000, 0xF000], mutex_ids=[20, 21])
    t71_bl1_g = pl.make_tile_group(type=_t71_bt, addrs=[0xD800, 0xF800], mutex_ids=[20, 21])

    _t71_c = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         fractal=1024, compact=0)
    t71_c_g = pl.make_tile_group(type=_t71_c, addrs=[0x9000, 0x11000], mutex_ids=[22, 23])
    _t71_c16 = pl.TileType(shape=[16, 16], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                           fractal=1024, compact=0)

    t71_auxd0_g = pl.make_tile_group(type=_t71_c16, addrs=[0xD000, 0x15000], mutex_ids=[24, 25])
    t71_auxd1_g = pl.make_tile_group(type=_t71_c16, addrs=[0xE400, 0x16400], mutex_ids=[26, 27])
    t71_auxd2_g = pl.make_tile_group(type=_t71_c16, addrs=[0xF800, 0x17800], mutex_ids=[28, 29])
    t71_auxd3_g = pl.make_tile_group(type=_t71_c16, addrs=[0x10C00, 0x18C00], mutex_ids=[30, 31])
    t71_aux_g = pl.make_tile_group(type=_t71_c, addrs=[0xD000, 0x15000], mutex_ids=[24, 25])

    _t71_a_bf16 = pl.TileType(shape=[64, 64], dtype=pl.DT_BF16,
                              target_memory=pl.MemorySpace.Left, layout=pl.NZ,
                              fractal=512, compact=0)
    _t71_b_bf16 = pl.TileType(shape=[64, 64], dtype=pl.DT_BF16,
                              target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                              fractal=512, compact=0)
    t71_a_bf16_g = pl.make_tile_group(type=_t71_a_bf16, addrs=[0x5000, 0x7000], mutex_ids=[18, 19])
    t71_b_bf16_g = pl.make_tile_group(type=_t71_b_bf16, addrs=[0xC000, 0xE000], mutex_ids=[20, 21])

    _t71_st_nd_h = pl.TileType(shape=[32, 64], dtype=pl.DT_FP16,
                               target_memory=pl.MemorySpace.Vec,
                               valid_shape=[-1, -1], compact=0)
    _t71_st_nz_h = pl.TileType(shape=[32, 64], dtype=pl.DT_FP16,
                               target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                               fractal=512, valid_shape=[-1, -1], compact=1)
    _t71_st_nd_b = pl.TileType(shape=[32, 64], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Vec,
                               valid_shape=[-1, -1], compact=0)
    _t71_st_nz_b = pl.TileType(shape=[32, 64], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                               fractal=512, valid_shape=[-1, -1], compact=1)
    t71_nd_h_g = pl.make_tile_group(type=_t71_st_nd_h, addrs=[49152], mutex_ids=[16])
    t71_nz_h_g = pl.make_tile_group(type=_t71_st_nz_h, addrs=[53248], mutex_ids=[16])
    t71_nd_b_g = pl.make_tile_group(type=_t71_st_nd_b, addrs=[49152], mutex_ids=[16])
    t71_nz_b_g = pl.make_tile_group(type=_t71_st_nz_b, addrs=[53248], mutex_ids=[16])
    tt_m6_l64 = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                            valid_shape=[-1, -1])
    tt_m6_l128 = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat)
    tt_m6_la = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Left, layout=pl.NZ)
    tt_m6_lb128 = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                              target_memory=pl.MemorySpace.Right, layout=pl.ZN)
    m6_la_g = pl.make_tile_group(type=tt_m6_la, addrs=[LA0], mutex_ids=[18])
    m6_inv_l1_g = pl.make_tile_group(type=tt_m6_l64, addrs=[P3_L1_X], mutex_ids=[31])
    m6_op_l1_g = pl.make_tile_group(type=tt_m6_l128, addrs=[L6_OP], mutex_ids=[19])

    tt_m6_hd64 = pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat)
    tt_m6_la_hd64 = pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_BF16,
                                target_memory=pl.MemorySpace.Left, layout=pl.NZ)
    tt_m6_lb64 = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Right, layout=pl.ZN)
    tt_m6_lc_ut = pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                              fractal=1024)
    m6_vbt_l1_g = pl.make_tile_group(type=tt_m6_hd64, addrs=[L6_OP2],
                                     mutex_ids=[24])
    m6_la_vbt_g = pl.make_tile_group(type=tt_m6_la_hd64, addrs=[L6_LA_VBT],
                                     mutex_ids=[25])
    tt_m6_l64_zn = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Mat, layout=pl.ZN,
                               valid_shape=[-1, -1])
    m6_invt_zn_g = pl.make_tile_group(type=tt_m6_l64_zn, addrs=[P3_L1_X],
                                      mutex_ids=[31])
    m6_lb_invt_g = pl.make_tile_group(type=tt_m6_lb64, addrs=[L6_LB_OP2],
                                      mutex_ids=[25])

    m6_lb128_g = pl.make_tile_group(type=tt_m6_lb128, addrs=[L6_LB_OP], mutex_ids=[20])
    m6_w_lc_g = pl.make_tile_group(type=tt_m6_lc128, addrs=[L6_LC_W], mutex_ids=[21])
    m6_u_lc_g = pl.make_tile_group(type=tt_m6_lc_ut, addrs=[L6_LC_U], mutex_ids=[22])

    str1 = HEAD_D * (1 - layout_code) + seq_len * HEAD_D * layout_code
    str_qk = (num_key_heads * HEAD_D) * (1 - layout_code) + HEAD_D * layout_code
    str_vv = (num_value_heads * HEAD_D) * (1 - layout_code) + HEAD_D * layout_code
    q_v = pl.make_tensor(q, [batch_size, num_key_heads, token_count, HEAD_D], [q_bs, q_ns, q_ss, 1])
    k_v = pl.make_tensor(k, [batch_size, num_key_heads, token_count, HEAD_D], [k_bs, k_ns, k_ss, 1])
    g_v = pl.make_tensor(g, [batch_size, num_value_heads, token_count, HEAD_D], [g_bs, g_ns, g_ss, 1])

    ws_nk_bytes = batch_size * num_key_heads * seq_len * HEAD_D * 2
    qn_o = pl.make_tensor(workspace, [batch_size, num_key_heads, seq_len, HEAD_D],
                          [num_key_heads * seq_len * HEAD_D, str1, str_qk, 1], dtype=pl.DT_BF16)
    kn_o = pl.make_tensor(workspace + ws_nk_bytes, [batch_size, num_key_heads, seq_len, HEAD_D],
                          [num_key_heads * seq_len * HEAD_D, str1, str_qk, 1], dtype=pl.DT_BF16)
    qd_o = pl.make_tensor(
        qd_out,
        [batch_size, num_value_heads, seq_len, HEAD_D],
        [num_value_heads * seq_len * HEAD_D, str1, str_vv, 1],
    )
    kr_o = pl.make_tensor(
        kr_out,
        [batch_size, num_value_heads, seq_len, HEAD_D],
        [num_value_heads * seq_len * HEAD_D, str1, str_vv, 1],
    )
    v_v = pl.make_tensor(v, [batch_size, num_value_heads, token_count, HEAD_D], [v_bs, v_ns, v_ss, 1])
    w_o = pl.make_tensor(
        w_out,
        [batch_size, num_value_heads, seq_len, HEAD_D],
        [num_value_heads * seq_len * HEAD_D, str1, str_vv, 1],
    )
    upre_o = pl.make_tensor(upre_out, [batch_size, num_value_heads, HEAD_D, seq_len],
                            [num_value_heads * HEAD_D * seq_len, HEAD_D * seq_len, seq_len, 1])
    gam_o = pl.make_tensor(gam_out, [batch_size, num_value_heads, nc, HEAD_D],
                           [num_value_heads * nc * HEAD_D, nc * HEAD_D, HEAD_D, 1])
    mqk_o = pl.make_tensor(mqk_out, [batch_size, num_value_heads, nc, CHUNK_C, CHUNK_C],
                           [num_value_heads * nc * CHUNK_C * CHUNK_C, nc * CHUNK_C * CHUNK_C,
                            CHUNK_C * CHUNK_C, CHUNK_C, 1])
    q_c_o = pl.make_tensor(q_c_out, [batch_size, nc, num_key_heads, CHUNK_C, HEAD_D],
                           [nc * num_key_heads * CHUNK_C * HEAD_D, num_key_heads * CHUNK_C * HEAD_D,
                            CHUNK_C * HEAD_D, HEAD_D, 1])
    k_c_o = pl.make_tensor(k_c_out, [batch_size, nc, num_key_heads, CHUNK_C, HEAD_D],
                           [nc * num_key_heads * CHUNK_C * HEAD_D, num_key_heads * CHUNK_C * HEAD_D,
                            CHUNK_C * HEAD_D, HEAD_D, 1])
    v_c_o = pl.make_tensor(v_c_out, [batch_size, nc, num_value_heads, CHUNK_C, HEAD_D],
                           [nc * num_value_heads * CHUNK_C * HEAD_D, num_value_heads * CHUNK_C * HEAD_D,
                            CHUNK_C * HEAD_D, HEAD_D, 1])
    gk_c_o = pl.make_tensor(gk_c_out, [batch_size, nc, num_value_heads, CHUNK_C, HEAD_D],
                            [nc * num_value_heads * CHUNK_C * HEAD_D, num_value_heads * CHUNK_C * HEAD_D,
                             CHUNK_C * HEAD_D, HEAD_D, 1])
    beta_c_o = pl.make_tensor(beta_c_out, [batch_size, nc, num_value_heads, CHUNK_C],
                              [nc * num_value_heads * CHUNK_C, num_value_heads * CHUNK_C, CHUNK_C, 1])
    akk_o = pl.make_tensor(akk_out, [batch_size, nc, num_value_heads, CHUNK_C, CHUNK_C],
                           [nc * num_value_heads * CHUNK_C * CHUNK_C, num_value_heads * CHUNK_C * CHUNK_C,
                            CHUNK_C * CHUNK_C, CHUNK_C, 1])
    v_beta_bf16_o = pl.make_tensor(workspace + ws_nk_bytes * 2, [batch_size, num_value_heads, seq_len, HEAD_D],
                                   [num_value_heads * seq_len * HEAD_D, str1, str_vv, 1], dtype=pl.DT_BF16)
    v_beta_lo_o = pl.make_tensor(workspace + ws_nk_bytes * 2 + batch_size * num_value_heads * seq_len * HEAD_D * 2,
                                [batch_size, num_value_heads, seq_len, HEAD_D],
                                [num_value_heads * seq_len * HEAD_D, str1, str_vv, 1], dtype=pl.DT_BF16)
    a_view = pl.make_tensor(a_log, [num_value_heads, 1], [1, 1])
    lb_view = pl.make_tensor(lb_t, [1, 1], [1, 1])
    scale_view = pl.make_tensor(scale_t, [1, 1], [1, 1])

    own_tasks = (batch_size * num_key_heads - core_id + num_cores - 1) // num_cores
    own_work = own_tasks * nc * nq
    with pl.section_cube():
        l_pre = lower_l1_g.next()
        pl.load(l_pre, lower_tril, [0, 0])
    with pl.section_vector():
        sl_m_pre = m5_mask_g.next()
        pl.load(sl_m_pre, sl_mask, [0, 0])
        w_first = 0
        if own_work > 0:
            c_f = 0
            bh_f = core_id * nq
            b_f = bh_f // num_value_heads
            hv_f = bh_f % num_value_heads
            hk_f = hv_f // nq
            tnd_m1_prefetch_v(q_v, k_v, g_v, b_f, hk_f, hv_f,
                          chunk_starts[c_f], chunk_lengths[c_f], x_b_g, xk_g, xg_g)

    pw_valid = 0
    pc = 0
    pb = 0
    phv = 0
    ps0 = 0
    p_gsi = 0
    p_src = 0
    p_valid = 64
    for work in pl.range(0, own_work):
        c = work % nc
        bh = (core_id + (work // (nc * nq)) * num_cores) * nq + (work // nc) % nq
        b = bh // num_value_heads
        hv = bh % num_value_heads
        hk = hv // nq
        s0 = c * CHUNK_C
        src = chunk_starts[c]
        valid = chunk_lengths[c]
        w_seq = work
        gate_slot_idx = w_seq % 2
        if pw_valid:
            with pl.section_cube():
                m4_kkt_mqk_blocks(m5_kkt_g, m5_mqk_g,
                                  p_gsi,
                                  lk0_g, lk1_g, lk2_g, lk3_g,
                                  lq0_g, lq1_g, lq2_g, lq3_g,
                                  rz0_g, rz1_g, rz2_g, rz3_g,
                                  m4_la_g, m4_b_g, m4_c_g)
        with pl.section_vector():
            tnd_m1_gate_norm_v(qn_o, kn_o, a_view, lb_view,
                           dt_bias, gate_l1, gate_slot_idx, b, hv, hk, s0,
                           1 if w_seq >= 2 else 0,
                           x_b_g, xk_g, xg_g, qn_h_g, kn_h_g, gate_f_g,
                           gate_nz_g, a_g, d_g, lb_g,
                           q_c_o, k_c_o, c, 1 if save_cache and hv % nq == 0 else 0, valid)
        if pw_valid:
            with pl.section_vector():
                tnd_m5_finish_v(beta_view, v_v, kn_h_g[p_gsi],
                            mqk_o, v_beta_bf16_o, v_beta_lo_o, v_c_o, beta_c_o,
                            layout_code, m5_mask_g, cs_g, sl_mask,
                            1 if w_seq == 1 else 0,
                            pc, pb, phv, ps0, num_value_heads, save_cache, p_src, p_valid,
                            y_l1_g[0], xy_ub_g, snz_ub_g, m5_kd_nz_g,
                            m6_op_l1_g[0],
                            m5_beta_raw_b_g, m5_beta_blk_f_g, m5_v_b_g, beta_cache_g,
                            m5_kkt_g, m5_mqk_g, m5_mqk_b_g,
                            m5_v_operand_g, m5_kd_operand_g,
                            t71_n_db_g[p_gsi], t71_eye, t71_zero, t71_eye_bf16,
                            t71_nd_h_g[0], t71_nz_h_g[0], t71_nd_b_g[0], t71_nz_b_g[0],
                            x_ub_g, eye64)
        with pl.section_cube():
            m2_cumsum_c(gate_l1, cs_g, gate_slot_idx,
                        pw_valid, lower_l1_g, lower_l0a_g, g_l0b_g, acc_g, save_cache)
        with pl.section_vector():
            m3_qd_kr_gam_v(qn_o, kn_o, qd_o, kr_o, gam_o, cs_g, scale_view,
                           c, b, hv, hk, s0, gate_slot_idx, num_value_heads, nc,
                           qn_h_g, kn_h_g, m3_out_b_g, m3_csl_g,
                           m3_sc_g,
                           m4_kn_b_g, m4_qn_b_g,
                           m4_opscr_b16_g, m4_ref_g,
                           lk0_g, lk1_g, lk2_g, lk3_g,
                           lq0_g, lq1_g, lq2_g, lq3_g,
                           rw0_g, rw1_g, rw2_g, rw3_g,
                           opnz_g, gk_c_o, save_cache)

        if pw_valid:
            m6_neumann_cube(
                t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g,
                t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g,
                t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g,
                t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g,
                t71_bl0_g, t71_bl1_g, t71_zero, t71_eye,
                t71_aux_g, t71_auxd0_g, t71_auxd1_g, t71_auxd2_g,
                t71_auxd3_g, t71_a_bf16_g, t71_b_bf16_g, t71_eye_bf16,
                v_beta_bf16_o, v_beta_lo_o, w_o, upre_o, t71_c_g,
                t71_n_db_g, p_gsi, pb, phv,
                ps0, t71_scratch_g, t71_tmpd0_g, t71_tmpd1_g,
                t71_tmpd2_g, t71_tmpd3_g, t71_p_g, t71_tmp_g,
                t71_inv_bf16_g, x_ub_g, m6_inv_l1_g, m6_akk_ub_g, m6_lb128_g,
                m6_op_l1_g, m6_la_g, m6_w_lc_g, m6_vbt_l1_g,
                m6_la_vbt_g, m6_lb_invt_g, m6_invt_zn_g, m6_u_lc_g,
                akk_o, save_cache)

        nw = pl.min(work + 1, own_work - 1)
        c_n = nw % nc
        bh_n = (core_id + (nw // (nc * nq)) * num_cores) * nq + (nw // nc) % nq
        with pl.section_vector():
            tnd_m1_prefetch_v(q_v, k_v, g_v, bh_n // num_value_heads, (bh_n % num_value_heads) // nq,
                          bh_n % num_value_heads, chunk_starts[c_n], chunk_lengths[c_n], x_b_g, xk_g, xg_g)
        pw_valid = 1
        pc = c
        pb = b
        phv = hv
        ps0 = s0
        p_gsi = gate_slot_idx
        p_src = src
        p_valid = valid

    n_works = own_work
    if pw_valid:
        with pl.section_cube():
            m4_kkt_mqk_blocks(m5_kkt_g, m5_mqk_g,
                              p_gsi,
                              lk0_g, lk1_g, lk2_g, lk3_g,
                              lq0_g, lq1_g, lq2_g, lq3_g,
                              rz0_g, rz1_g, rz2_g, rz3_g,
                              m4_la_g, m4_b_g, m4_c_g)
        with pl.section_vector():
            tnd_m5_finish_v(beta_view, v_v, kn_h_g[p_gsi],
                        mqk_o, v_beta_bf16_o, v_beta_lo_o, v_c_o, beta_c_o,
                        layout_code, m5_mask_g, cs_g, sl_mask,
                        1 if n_works == 1 else 0,
                        pc, pb, phv, ps0, num_value_heads, save_cache, p_src, p_valid,
                        y_l1_g[0], xy_ub_g, snz_ub_g, m5_kd_nz_g,
                        m6_op_l1_g[0],
                        m5_beta_raw_b_g, m5_beta_blk_f_g, m5_v_b_g, beta_cache_g,
                        m5_kkt_g, m5_mqk_g, m5_mqk_b_g,
                        m5_v_operand_g, m5_kd_operand_g,
                        t71_n_db_g[p_gsi], t71_eye, t71_zero, t71_eye_bf16,
                        t71_nd_h_g[0], t71_nz_h_g[0], t71_nd_b_g[0], t71_nz_b_g[0],
                        x_ub_g, eye64)
    if n_works >= 1:
        with pl.section_vector():
            pass
            for _ in pl.range(0, pl.min(n_works, 2)):
                pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=3)
    if pw_valid:
        m6_neumann_cube(
            t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g,
            t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g,
            t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g,
            t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g,
            t71_bl0_g, t71_bl1_g, t71_zero, t71_eye,
            t71_aux_g, t71_auxd0_g, t71_auxd1_g, t71_auxd2_g,
            t71_auxd3_g, t71_a_bf16_g, t71_b_bf16_g, t71_eye_bf16,
            v_beta_bf16_o, v_beta_lo_o, w_o, upre_o, t71_c_g,
            t71_n_db_g, p_gsi, pb, phv,
            ps0, t71_scratch_g, t71_tmpd0_g, t71_tmpd1_g,
            t71_tmpd2_g, t71_tmpd3_g, t71_p_g, t71_tmp_g,
            t71_inv_bf16_g, x_ub_g, m6_inv_l1_g, m6_akk_ub_g, m6_lb128_g,
            m6_op_l1_g, m6_la_g, m6_w_lc_g, m6_vbt_l1_g,
            m6_la_vbt_g, m6_lb_invt_g, m6_invt_zn_g, m6_u_lc_g,
            akk_o, save_cache)

    with pl.section_vector():
        pl.system.bar_all()
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=13)
    with pl.section_cube():
        pl.system.bar_all()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=13)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=12)
    with pl.section_vector():
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=12)
        pl.system.bar_all()
    cp_forward_chain_tnd(
        qd_out,
        kr_out,
        gam_out,
        mqk_out,
        w_out,
        upre_out,
        initial_state,
        summary,
        qd_out,
        initial_state,
        gk_c_out,
        v_c_out,
        layout_code,
        0,
        first_continues,
        has_initial,
        0,
        1,
        nq,
        cu,
        chunk_offsets,
        h_offsets,
        initial_bs,
        initial_ns,
        0,
        0,
        0,
        0,
        0,
    )


@pl.jit(auto_mutex=True)
def flash_kda_cp_stage2_tnd_kernel(
    qd: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_BF16],
    kr: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_BF16],
    gam: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_FP32],
    mqk: pl.Tensor[[pl.DYNAMIC] * 5, pl.DT_BF16],
    w: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_BF16],
    upre: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_FP32],
    initial_state: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_FP32],
    summary: pl.Tensor[[pl.DYNAMIC] * 5, pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_BF16],
    final_state: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_FP32],
    h: pl.Tensor[[pl.DYNAMIC] * 5, pl.DT_BF16],
    v_new: pl.Tensor[[pl.DYNAMIC] * 5, pl.DT_BF16],
    layout_code: pl.DT_INT64, cp_rank: pl.DT_INT64,
    first_continues: pl.DT_INT64, has_initial: pl.DT_INT64, save_cache: pl.DT_INT64,
    ratio: pl.DT_INT64,
    cu: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64],
    chunk_offsets: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64],
    h_offsets: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64],
    initial_bs: pl.DT_INT64,
    initial_ns: pl.DT_INT64,
    out_bs: pl.DT_INT64,
    out_ns: pl.DT_INT64,
    out_ss: pl.DT_INT64,
    final_bs: pl.DT_INT64,
    final_ns: pl.DT_INT64,
):
    cp_forward_chain_tnd(
        qd,
        kr,
        gam,
        mqk,
        w,
        upre,
        initial_state,
        summary,
        out,
        final_state,
        h,
        v_new,
        layout_code,
        cp_rank,
        first_continues,
        has_initial,
        save_cache,
        0,
        ratio,
        cu,
        chunk_offsets,
        h_offsets,
        initial_bs,
        initial_ns,
        out_bs,
        out_ns,
        out_ss,
        final_bs,
        final_ns,
    )


def cp_dense_load_half(tile, tensor, batch, head, src, valid):
    sub = pl.get_subblock_idx()
    rows = pl.max(0, pl.min(HS, valid - sub * HS))
    # LOAD leaves the remaining physical rows untouched; initialise them first.
    pl.set_validshape(tile, [HS, HEAD_D])
    if rows < HS:
        tnd_zero_bf16_vf(tile, HS * HEAD_D)
    if rows > 0:
        pl.set_validshape(tile, [rows, HEAD_D])
        pl.load(tile, tensor, [batch, head, src + sub * HS, 0], order=[2, 3])
    pl.set_validshape(tile, [HS, HEAD_D])


def cp_dense_prefetch_v(q_v, k_v, g_v, b, hk, hv, src, valid, x_b_g, xk_g, xg_g):
    xq = x_b_g.next()
    xk = xk_g.next()
    xg = xg_g.next()
    cp_dense_load_half(xq, q_v, b, hk, src, valid)
    cp_dense_load_half(xk, k_v, b, hk, src, valid)
    cp_dense_load_half(xg, g_v, b, hv, src, valid)


def cp_dense_finish_v(beta, v_v, kn_slot,
                mqk_o, v_beta_bf16_o, v_beta_lo_o, v_c_o, beta_c_o,
                layout_code, m5_mask_g, cs_g, sl_mask, reload_mask,
                c, b, hv, s0, num_value_heads, save_cache, src, valid,
                y0_l1, snz_stg_g, snz_nz_g, kd_nz_g,
                kd_l1,
                m5_beta_raw_b_g, m5_beta_blk_f_g, m5_v_b_g, beta_cache_g,
                m5_kkt_g, m5_mqk_g, m5_mqk_b_g, m5_v_operand_g, m5_kd_operand_g,
                t71_n, t71_eye, t71_zero, t71_eye_bf16,
                t71_nd_h, t71_nz_h, t71_nd_b, t71_nz_b, x_ub_g, eye64):
    """v12.11: M5 as a plain function called from the merged loop (lagged)."""
    h0 = (hv // 64) * 64
    hv_local = hv - h0
    off_base = hv_local * (1 - layout_code) + hv_local * 64 * layout_code
    step = 64 * (1 - layout_code) + 1 * layout_code
    brow_b = src * (1 - layout_code) + h0 * layout_code
    bcol_b = h0 * (1 - layout_code) + src * layout_code
    n_rows = valid * (1 - layout_code) + pl.min(64, num_value_heads - h0) * layout_code
    n_cols = pl.min(64, num_value_heads - h0) * (1 - layout_code) + valid * layout_code
    sub_m5 = pl.get_subblock_idx()
    r0 = sub_m5 * HS
    if reload_mask:
        pl.set_validshape(m5_mask_g.current(), [CHUNK_C, HEAD_D])
        pl.load(m5_mask_g.current(), sl_mask, [0, 0])
    mask_b = m5_mask_g.current()
    bb_raw = m5_beta_raw_b_g.next()
    pl.set_validshape(bb_raw, [64, 64])
    if valid < CHUNK_C:
        tnd_zero_bf16_vf(bb_raw, 64 * 64)
    pl.set_validshape(bb_raw, [n_rows, n_cols])
    pl.load(bb_raw, beta, [b, brow_b, bcol_b], order=[1, 2])
    blk_f = m5_beta_blk_f_g.next()
    pl.set_validshape(bb_raw, [64, n_cols * (1 - layout_code) + 64 * layout_code])
    pl.set_validshape(blk_f, [64, n_cols * (1 - layout_code) + 64 * layout_code])
    pl.cast(blk_f, bb_raw)
    vb = m5_v_b_g.next()
    cp_dense_load_half(vb, v_v, b, hv, src, valid)
    if save_cache:
        pl.store(v_c_o, vb, [b, c, hv, r0, 0], order=[3, 4])

    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=11)
    cs_slot = cs_g[0]
    kb = kn_slot
    vf_t = m5_v_operand_g.next()
    kf = m5_kd_operand_g.next()
    beta_cache_b = beta_cache_g.next()
    tnd_m5_vkn_vf(vb, kb, cs_slot, blk_f, vf_t, kf,
              beta_cache_b, off_base, step, r0, save_cache, valid)
    if save_cache:
        pl.store(beta_c_o, beta_cache_b, [b, c, hv, r0], order=[2, 3])
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=_M5_CS_DONE)
    pl.store(v_beta_bf16_o, vf_t, [b, hv, s0 + r0, 0], order=[2, 3])
    pl.store(v_beta_lo_o, vb, [b, hv, s0 + r0, 0], order=[2, 3])

    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=9)
    kt = m5_kkt_g[0]
    mq = m5_mqk_g[0]
    mqb = m5_mqk_b_g.next()
    tnd_m5_tmqk_vf(kt, mask_b, mq, blk_f, mq, mqb, off_base, step, r0, valid)
    pl.set_validshape(mq, [HS, CHUNK_C])
    nd2nz_sg(mq, snz_stg_g.current(), HS)
    pl.set_validshape(snz_nz_g.current(), [HS, CHUNK_C])
    pl.insert(y0_l1, snz_nz_g.current(), [r0, 0])
    pl.store(mqk_o, mqb, [b, hv, c, r0, 0])

    t71_prepare_neumann_v(m5_mqk_g, t71_n, t71_eye, t71_zero,
                          t71_eye_bf16, t71_nd_h, t71_nz_h,
                          t71_nd_b, t71_nz_b, x_ub_g, eye64, reload_mask)
    kd_nz = kd_nz_g.next()
    pl.set_validshape(kd_nz, [HS, HEAD_D])
    pl.move(kd_nz, kf)
    pl.insert(kd_l1, kd_nz, [r0, 0])

    # _M5M6_EVENT had no consumer in the current M6 chain; do not emit a
    # per-work cross-core token that can accumulate and exhaust the event queue.
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=_M6_UPRE_EVENT)


def cp_forward_chain_tail(
    qd,
    kr,
    gam,
    mqk,
    w,
    upre,
    initial_state,
    summary,
    out,
    final_state,
    h,
    v_new,
    layout_code,
    cp_rank,
    first_continues,
    has_initial,
    save_cache,
    summary_mode,
    ratio,
    initial_bs,
    initial_ns,
    out_bs,
    out_ns,
    out_ss,
    final_bs,
    final_ns,
):
    """Inline device region: one owner per batch/value-head, two AIV row halves.

    summary_mode computes E and T after the owner's preprocessing has drained.
    Otherwise it scans preceding packets and executes the real forward chain.
    Both callers are single-launch JITs. No host-side arithmetic is involved.
    """
    core = pl.get_block_idx() // pl.get_subblock_num()
    cores = pl.get_block_num()
    sub = pl.get_subblock_idx()
    batch_size = qd.shape[0]
    seq_len = qd.shape[1] * (1 - layout_code) + qd.shape[2] * layout_code
    num_heads = qd.shape[2] * (1 - layout_code) + qd.shape[1] * layout_code
    nc = seq_len // 64
    token_count = out.shape[1] * (1 - layout_code) + out.shape[2] * layout_code
    ohn = 128 * (1 - layout_code) + token_count * 128 * layout_code
    hn = 128 * (1 - layout_code) + seq_len * 128 * layout_code
    sn = num_heads * 128 * (1 - layout_code) + 128 * layout_code
    qv = pl.make_tensor(qd, [batch_size, num_heads, seq_len, 128], [num_heads * seq_len * 128, hn, sn, 1])
    kv = pl.make_tensor(kr, [batch_size, num_heads, seq_len, 128], [num_heads * seq_len * 128, hn, sn, 1])
    wv = pl.make_tensor(w, [batch_size, num_heads, seq_len, 128], [num_heads * seq_len * 128, hn, sn, 1])
    ov = pl.make_tensor(out, [batch_size, num_heads, token_count, 128], [out_bs, out_ns, out_ss, 1])
    iv = pl.make_tensor(initial_state, [initial_state.shape[0], num_heads, 128, 128], [initial_bs, initial_ns, 128, 1])
    fv = pl.make_tensor(final_state, [final_state.shape[0], num_heads, 128, 128], [final_bs, final_ns, 128, 1])
    # UB high-water 0x35000 (< 256 KiB), half state [64 rows,128 columns].
    st = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec), addrs=[0], mutex_ids=[0])
    de = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec), addrs=[0x8000], mutex_ids=[1])
    sb = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Vec), addrs=[0x10000], mutex_ids=[2])
    sz = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
        addrs=[0x14000],
        mutex_ids=[3],
    )
    ud = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec), addrs=[0x18000], mutex_ids=[4])
    up = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec), addrs=[0x1C000], mutex_ids=[5])
    ub = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Vec), addrs=[0x20000], mutex_ids=[6])
    uz = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
        addrs=[0x22000],
        mutex_ids=[7],
    )
    ga = pl.make_tile_group(type=pl.TileType(shape=[1, 128], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec), addrs=[0x24000], mutex_ids=[8])
    tf = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec), addrs=[0x25000], mutex_ids=[9])
    tb = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Vec), addrs=[0x2D000], mutex_ids=[10])
    tz = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
        addrs=[0x31000],
        mutex_ids=[11],
    )
    # L1 views share the same backing for transpose, with the same mutex.
    sl = pl.make_tile_group(type=pl.TileType(
        shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=[0], mutex_ids=[0])
    slt = pl.make_tile_group(type=pl.TileType(
        shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN), addrs=[0], mutex_ids=[0])
    wl = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Mat), addrs=[0x8000], mutex_ids=[1])
    kl = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Mat), addrs=[0xC000], mutex_ids=[2])
    ul = pl.make_tile_group(
        type=pl.TileType(shape=[128, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[0x10000],
        mutex_ids=[3],
    )
    ult = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x10000],
        mutex_ids=[3],
    )
    ql = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Mat), addrs=[0x14000], mutex_ids=[4])
    ml = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Mat), addrs=[0x18000], mutex_ids=[5])
    tl = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[0x20000],
        mutex_ids=[6],
    )
    la = pl.make_tile_group(type=pl.TileType(
        shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left, layout=pl.NZ), addrs=[0], mutex_ids=[13])
    lua = pl.make_tile_group(type=pl.TileType(
        shape=[128, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left, layout=pl.NZ), addrs=[0], mutex_ids=[13])
    lqa = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=[0x8000],
        mutex_ids=[14],
    )
    lma = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=[0xC000],
        mutex_ids=[15],
    )
    rb = pl.make_tile_group(type=pl.TileType(
        shape=[128, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right, layout=pl.ZN), addrs=[0], mutex_ids=[16])
    rkb = pl.make_tile_group(type=pl.TileType(
        shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right, layout=pl.ZN), addrs=[0], mutex_ids=[16])
    rtb = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=[0],
        mutex_ids=[16],
    )
    rsb = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=[0x8000],
        mutex_ids=[17],
    )
    rub = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=[0x8000],
        mutex_ids=[17],
    )
    uc = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=[0],
        mutex_ids=[18],
    )
    dc = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=[0x10000],
        mutex_ids=[19],
    )
    oc = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Acc), addrs=[0x20000], mutex_ids=[20])
    tail_f = pl.make_tile_group(
        type=pl.TileType(
            shape=[32, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
        ),
        addrs=[0x35000],
        mutex_ids=[12],
    )
    tail_b = pl.make_tile_group(
        type=pl.TileType(
            shape=[32, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
        ),
        addrs=[0x39000],
        mutex_ids=[13],
    )
    group_count = (batch_size * num_heads // ratio - core + cores - 1) // cores
    for owned in pl.range(0, group_count * ratio):
        task = (core + (owned // ratio) * cores) * ratio + owned % ratio
        b = task // num_heads
        hv = task % num_heads
        for chain in pl.range(0, 1 + summary_mode):
            with pl.section_vector():
                if summary_mode and chain == 1:
                    cp_state_init_vf(st.current(), sub * 64, 1)
                elif not first_continues and has_initial:
                    pl.load(st.current(), iv, [b, hv, sub * 64, 0])
                else:
                    cp_state_init_vf(st.current(), sub * 64, 0)
            if not summary_mode and first_continues:
                for rank in pl.range(0, cp_rank):
                    with pl.section_vector():
                        pl.cast(sb.current(), st.current(), mode=pl.RoundMode.CAST_ROUND)
                        pl.move(sz.current(), sb.current())
                        pl.insert(sl.current(), sz.current(), [sub * 64, 0])
                        pl.load(tf.current(), summary, [rank * batch_size + b, hv, 1, sub * 64, 0])
                        pl.cast(tb.current(), tf.current(), mode=pl.RoundMode.CAST_ROUND)
                        pl.move(tz.current(), tb.current())
                        pl.insert(tl.current(), tz.current(), [sub * 64, 0])
                        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=13)
                    with pl.section_cube():
                        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=13)
                        pl.move(la.current(), sl.current())
                        pl.move(rtb.current(), tl.current())
                        pl.matmul(dc.current(), la.current(), rtb.current())
                        pl.move(de.current(), dc.current(), acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=12)
                    with pl.section_vector():
                        pl.load(tf.current(), summary, [rank * batch_size + b, hv, 0, sub * 64, 0])
                        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=12)
                        cp_state_add_vf(st.current(), de.current(), tf.current())
            with pl.section_vector():
                if not summary_mode and save_cache:
                    pl.cast(sb.current(), st.current(), mode=pl.RoundMode.CAST_ROUND)
                    pl.store(h, sb.current(), [b, 0, hv, sub * 64, 0])
            for c in pl.range(0, nc):
                valid = pl.min(64, token_count - c * 64)
                with pl.section_vector():
                    pl.cast(sb.current(), st.current(), mode=pl.RoundMode.CAST_ROUND)
                    pl.move(sz.current(), sb.current())
                    pl.insert(sl.current(), sz.current(), [sub * 64, 0])
                    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=13)
                with pl.section_cube():
                    pl.load(wl.current(), wv, [b, hv, c * 64, 0], order=[3, 2])
                    pl.load(kl.current(), kv, [b, hv, c * 64, 0])
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=13)
                    pl.move(la.current(), sl.current())
                    pl.move(rb.current(), wl.current())
                    pl.matmul(uc.current(), la.current(), rb.current())
                    pl.move(ud.current(), uc.current(), acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                    if not summary_mode:
                        pl.load(ql.current(), qv, [b, hv, c * 64, 0])
                        pl.move(lqa.current(), ql.current())
                        pl.move(rsb.current(), slt.current())
                        pl.matmul(oc.current(), lqa.current(), rsb.current(), phase=pl.AccPhase.Partial)
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=12)
                with pl.section_vector():
                    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=12)
                    if summary_mode and chain == 1:
                        pl.cast(ub.current(), ud.current(), mode=pl.RoundMode.CAST_ROUND)
                    else:
                        pl.load(up.current(), upre, [b, hv, sub * 64, c * 64])
                        m7_ut_add_upre_bf16_vf(ud.current(), up.current(), ub.current(), 0, 128)
                    if not summary_mode and save_cache:
                        pl.store(v_new, ub.current(), [b, c, hv, sub * 64, 0], order=[3, 4])
                    pl.move(uz.current(), ub.current())
                    pl.insert(ul.current(), uz.current(), [sub * 64, 0])
                    pl.load(ga.current(), gam, [b, hv, c, 0])
                    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=13)
                with pl.section_cube():
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=13)
                    pl.move(lua.current(), ul.current())
                    pl.move(rkb.current(), kl.current())
                    pl.matmul(dc.current(), lua.current(), rkb.current())
                    if not summary_mode:
                        pl.load(ml.current(), mqk, [b, hv, c, 0, 0])
                        pl.move(lma.current(), ml.current())
                        pl.move(rub.current(), ult.current())
                        pl.matmul_acc(oc.current(), oc.current(), lma.current(), rub.current(), phase=pl.AccPhase.Final)
                        if valid == 64:
                            pl.store(ov, oc.current(), [b, hv, c * 64, 0], order=[2, 3], phase=pl.STPhase.Final)
                        else:
                            pl.set_validshape(tail_f.current(), [32, 128])
                            pl.move(tail_f.current(), oc.current(),
                                    acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM, phase=pl.STPhase.Final)
                            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=10)
                    pl.move(de.current(), dc.current(), acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=12)
                with pl.section_vector():
                    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=12)
                    if not summary_mode and valid < 64:
                        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=10)
                        rows = pl.max(0, pl.min(32, valid - sub * 32))
                        if rows > 0:
                            pl.set_validshape(tail_f.current(), [rows, 128])
                            pl.set_validshape(tail_b.current(), [rows, 128])
                            pl.cast(tail_b.current(), tail_f.current(), mode=pl.RoundMode.CAST_ROUND)
                            pl.store(ov, tail_b.current(), [b, hv, c * 64 + sub * 32, 0], order=[2, 3])
                    cp_state_update_vf(st.current(), de.current(), ga.current())
                    if not summary_mode and save_cache:
                        pl.cast(sb.current(), st.current(), mode=pl.RoundMode.CAST_ROUND)
                        pl.store(h, sb.current(), [b, c + 1, hv, sub * 64, 0])
            with pl.section_vector():
                if summary_mode:
                    if chain == 1 and not first_continues:
                        cp_state_init_vf(st.current(), sub * 64, 0)
                    pl.store(summary, st.current(), [b, hv, chain, sub * 64, 0])
                else:
                    pl.store(fv, st.current(), [b, hv, sub * 64, 0])
                pl.system.bar_all()


@pl.jit(auto_mutex=True)
def flash_kda_cp_stage1_tail_kernel(
    # 输入
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    g: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    beta: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    a_log: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32],
    dt_bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    lower_bound: pl.DT_FP32,
    scale: pl.DT_FP32,
    layout_code: pl.DT_INT64,
    # 输出
    qd_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    kr_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    gam_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    mqk_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    w_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    upre_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    q_c_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    k_c_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    v_c_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    gk_c_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    beta_c_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    akk_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    save_cache: pl.DT_INT64,
    # workspace
    workspace: pl.Ptr[pl.DT_UINT8],
    const_workspace: pl.Ptr[pl.DT_UINT8],
    initial_state: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    summary: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    first_continues: pl.DT_INT64,
    has_initial: pl.DT_INT64,
    q_bs: pl.DT_INT64,
    q_ns: pl.DT_INT64,
    q_ss: pl.DT_INT64,
    k_bs: pl.DT_INT64,
    k_ns: pl.DT_INT64,
    k_ss: pl.DT_INT64,
    g_bs: pl.DT_INT64,
    g_ns: pl.DT_INT64,
    g_ss: pl.DT_INT64,
    v_bs: pl.DT_INT64,
    v_ns: pl.DT_INT64,
    v_ss: pl.DT_INT64,
    beta_bs: pl.DT_INT64,
    beta_s1: pl.DT_INT64,
    beta_s2: pl.DT_INT64,
    initial_bs: pl.DT_INT64,
    initial_ns: pl.DT_INT64,
):
    cc_cid = pl.get_block_idx() // pl.get_subblock_num()
    cc_base = const_workspace + cc_cid * 65536
    lower_tril = pl.make_tensor(cc_base, [64, 64], [64, 1], dtype=pl.DT_FP32)
    sl_mask = pl.make_tensor(cc_base + 16384, [64, 128], [128, 1], dtype=pl.DT_BF16)
    eye64 = pl.make_tensor(cc_base + 32768, [64, 64], [64, 1], dtype=pl.DT_FP32)
    lb_t = pl.make_tensor(cc_base + 49152, [1, 64], [64, 1], dtype=pl.DT_FP32)
    scale_t = pl.make_tensor(cc_base + 49408, [1, 64], [64, 1], dtype=pl.DT_FP32)
    cc_a = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec), addrs=[0], mutex_ids=[0])
    cc_b = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec), addrs=[0x4000], mutex_ids=[1])
    cc_bb = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16,
        target_memory=pl.MemorySpace.Vec), addrs=[0xC000], mutex_ids=[2])
    cc_e = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec), addrs=[0x10000], mutex_ids=[3])
    cc_l = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec), addrs=[0x14000], mutex_ids=[4])
    cc_s = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec), addrs=[0x14100], mutex_ids=[5])
    with pl.section_vector():
        if pl.get_subblock_idx() == 0:
            cp_constants_vf(cc_a.current(), cc_b.current(), cc_e.current(), cc_l.current(),
                            cc_s.current(), lower_bound, scale)
            pl.cast(cc_bb.current(), cc_b.current(), mode=pl.RoundMode.CAST_ROUND)
            pl.store(lower_tril, cc_a.current(), [0, 0])
            pl.store(sl_mask, cc_bb.current(), [0, 0])
            pl.store(eye64, cc_e.current(), [0, 0])
            pl.store(lb_t, cc_l.current(), [0, 0])
            pl.store(scale_t, cc_s.current(), [0, 0])
        pl.system.bar_all()
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=13)
    with pl.section_cube():
        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=13)
        pl.system.bar_all()
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=12)
    with pl.section_vector():
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=12)
    beta_view = pl.make_tensor(beta, [beta.shape[0], beta.shape[1], beta.shape[2]], [beta_bs, beta_s1, beta_s2])
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    num_cores = pl.get_block_num()

    batch_size = g.shape[0]
    token_count = g.shape[1] * (1 - layout_code) + g.shape[2] * layout_code
    seq_len = qd_out.shape[1] * (1 - layout_code) + qd_out.shape[2] * layout_code
    num_key_heads = q.shape[2] * (1 - layout_code) + q.shape[1] * layout_code
    num_value_heads = g.shape[2] * (1 - layout_code) + g.shape[1] * layout_code
    nq = num_value_heads // num_key_heads
    nc = seq_len // CHUNK_C
    total = batch_size * num_value_heads
    tt_b = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                       target_memory=pl.MemorySpace.Vec)
    tt_f32 = pl.TileType(shape=[HS, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec)
    tt_b32 = pl.TileType(shape=[HS, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec)
    tt_c64h = pl.TileType(shape=[HS, CHUNK_C], dtype=pl.DT_FP32,
                          target_memory=pl.MemorySpace.Vec)
    tt_c64bh = pl.TileType(shape=[HS, CHUNK_C], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Vec)
    tt_a = pl.TileType(shape=[CHUNK_C, 1], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Vec,
                       valid_shape=[-1, -1])
    tt_d = pl.TileType(shape=[1, HEAD_D], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Vec)
    tt_c64 = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec)
    tt_c64b = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                          target_memory=pl.MemorySpace.Vec)

    gate_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[L1_G0, L1_G1], mutex_ids=[11, 12],
        fwd_ids=_GATE_FWD_IDS, bwd_ids=_GATE_BWD_IDS,
    )

    cs_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[VA_CS0], mutex_ids=[16],
        fwd_ids=(4,), bwd_ids=(6,),
    )

    x_b_g = pl.make_tile_group(type=tt_b32, addrs=[VA_X_B], mutex_ids=[0])
    qn_h_g = pl.make_tile_group(
        type=tt_b32, addrs=[0x34800, 0x36800], mutex_ids=[10, 13])
    kn_h_g = pl.make_tile_group(
        type=tt_b32, addrs=[0x38800, 0x3A800], mutex_ids=[14, 15])
    xk_g = pl.make_tile_group(type=tt_b32, addrs=[0x21000], mutex_ids=[17])

    xg_g = pl.make_tile_group(type=tt_b32, addrs=[0x2000], mutex_ids=[18])
    gate_f_g = pl.make_tile_group(type=tt_f32, addrs=[VA_GATE_F], mutex_ids=[5])
    gate_nz_g = pl.make_tile_group(
        type=pl.TileType(shape=[HS, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                         valid_shape=[-1, -1], compact=1),
        addrs=[VA_GATE_NZ], mutex_ids=[6])
    a_g = pl.make_tile_group(type=tt_a, addrs=[VA_A_LOG], mutex_ids=[8])
    d_g = pl.make_tile_group(type=tt_d, addrs=[VA_DT_B], mutex_ids=[9])
    lb_g = pl.make_tile_group(type=tt_a, addrs=[VA_LB], mutex_ids=[7])

    m3_out_b_g = pl.make_tile_group(type=tt_b32, addrs=[VA_QN_B], mutex_ids=[3])
    m3_csl_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[0x20800], mutex_ids=[21])
    m3_sc_g = pl.make_tile_group(type=tt_a, addrs=[VA_SC], mutex_ids=[22])

    m4_kn_b_g = pl.make_tile_group(type=tt_b, addrs=[VA_KN_B], mutex_ids=[4])
    m4_qn_b_g = pl.make_tile_group(type=tt_b, addrs=[VA_QN_B], mutex_ids=[3])
    m4_opscr_b16_g = pl.make_tile_group(
        type=pl.TileType(shape=[ROW_BLOCK, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[VA_M4_SCR_B16, VA_M4_SCR_B16 + 0x1000], mutex_ids=[6, 19])
    m4_ref_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[0x20A00], mutex_ids=[30])

    _tt_op16 = pl.TileType(shape=[ROW_BLOCK, HEAD_D], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                           valid_shape=[-1, -1], compact=1)

    lk0_g = pl.make_tile_group(type=_tt_op16, addrs=[0x14000, 0x15000], mutex_ids=[31, 27])
    lk1_g = pl.make_tile_group(type=_tt_op16, addrs=[0x16000, 0x17000], mutex_ids=[28, 29])
    lk2_g = pl.make_tile_group(type=_tt_op16, addrs=[0x18000, 0x19000], mutex_ids=[30, 31])
    lk3_g = pl.make_tile_group(type=_tt_op16, addrs=[0x1A000, 0x1B000], mutex_ids=[27, 28])
    lq0_g = pl.make_tile_group(type=_tt_op16, addrs=[0x1C000, 0x1D000], mutex_ids=[29, 30])
    lq1_g = pl.make_tile_group(type=_tt_op16, addrs=[0x1E000, 0x1F000], mutex_ids=[31, 27])
    lq2_g = pl.make_tile_group(type=_tt_op16, addrs=[0x20000, 0x21000], mutex_ids=[28, 29])
    lq3_g = pl.make_tile_group(type=_tt_op16, addrs=[0x22000, 0x23000], mutex_ids=[30, 31])
    _tt_rw = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                         valid_shape=[-1, -1], compact=1)
    _tt_rz64 = pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Mat, layout=pl.ZN,
                           valid_shape=[-1, -1], compact=1)
    rw0_g = pl.make_tile_group(type=_tt_rw, addrs=[0x34000, 0x56000], mutex_ids=[27, 28])
    rw1_g = pl.make_tile_group(type=_tt_rw, addrs=[0x38000, 0x5A000], mutex_ids=[29, 30])
    rw2_g = pl.make_tile_group(type=_tt_rw, addrs=[0x3C000, 0x5E000], mutex_ids=[31, 27])
    rw3_g = pl.make_tile_group(type=_tt_rw, addrs=[0x40000, 0x62000], mutex_ids=[28, 29])
    rz0_g = pl.make_tile_group(type=_tt_rz64, addrs=[0x34000, 0x56000], mutex_ids=[27, 28])
    rz1_g = pl.make_tile_group(type=_tt_rz64, addrs=[0x38000, 0x5A000], mutex_ids=[29, 30])
    rz2_g = pl.make_tile_group(type=_tt_rz64, addrs=[0x3C000, 0x5E000], mutex_ids=[31, 27])
    rz3_g = pl.make_tile_group(type=_tt_rz64, addrs=[0x40000, 0x62000], mutex_ids=[28, 29])

    # kn_h_g[1] ends at 0x3C800; the lagged M5 still needs its last rows.
    opnz_g = pl.make_tile_group(
        type=pl.TileType(shape=[ROW_BLOCK, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                         valid_shape=[-1, -1], compact=1),
        addrs=[0x3C800],
        mutex_ids=[2])
    m4_la_g = pl.make_tile_group(
        type=pl.TileType(shape=[ROW_BLOCK, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=[0x4000, 0xD000], mutex_ids=[13, 23])
    m4_b_g = pl.make_tile_group(
        type=pl.TileType(shape=[HEAD_D, ROW_BLOCK], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=[0x8000, 0x9000, 0xA000, 0xB000],
        mutex_ids=[14, 16, 17, 18])
    m4_c_g = pl.make_tile_group(
        type=pl.TileType(shape=[ROW_BLOCK, ROW_BLOCK], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         fractal=1024),
        addrs=[0x8000, 0x8400, 0x8800, 0x8C00],
        mutex_ids=[15, 19, 20, 21])

    lower_l1_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[L1_L], mutex_ids=[10])
    lower_l0a_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=[LA0], mutex_ids=[13])
    g_l0b_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=[RA0], mutex_ids=[14])
    acc_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         fractal=1024),
        addrs=[CA0], mutex_ids=[15])

    m5_mask_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[0x28000], mutex_ids=[26])
    m5_kkt_g = pl.make_tile_group(
        type=tt_c64h, addrs=[VA_M5_KKT], mutex_ids=[1])
    m5_mqk_g = pl.make_tile_group(
        type=tt_c64h, addrs=[VA_M5_MQK], mutex_ids=[2])
    m5_mqk_b_g = pl.make_tile_group(
        type=tt_c64bh, addrs=[0xE100], mutex_ids=[2])

    m5_beta_raw_b_g = pl.make_tile_group(
        type=tt_c64b, addrs=[0x12000], mutex_ids=[2])
    m5_beta_blk_f_g = pl.make_tile_group(
        type=tt_c64, addrs=[VA_M5_BETA_F], mutex_ids=[3])
    m5_v_b_g = pl.make_tile_group(
        type=tt_b32, addrs=[VA_M5_VB], mutex_ids=[4])
    m5_v_operand_g = pl.make_tile_group(
        type=tt_b32, addrs=[VA_M5_VF], mutex_ids=[6])
    m5_kd_operand_g = pl.make_tile_group(
        type=tt_b32, addrs=[VA_M5_KNF], mutex_ids=[5])

    beta_cache_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, HS], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[0x3D800, 0x3DC00], mutex_ids=[23, 25])

    m5_kd_nz_g = pl.make_tile_group(
        type=pl.TileType(shape=[HS, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                         valid_shape=[-1, -1], compact=1),
        addrs=[P3_UB_XY], mutex_ids=[16])

    tt_m6_l64 = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                            valid_shape=[-1, -1])
    tt_m6_lc128 = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                              fractal=1024)
    tt_m6_vhs = pl.TileType(shape=[HS, CHUNK_C], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec,
                            pad=pl.TilePad.zero, valid_shape=[-1, -1])
    tt_m6_vhs33 = pl.TileType(shape=[HS + 1, CHUNK_C], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Vec,
                              pad=pl.TilePad.zero, valid_shape=[-1, -1])
    tt_m6_nz = pl.TileType(shape=[HS + 1, CHUNK_C], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Vec, layout=pl.NZ)
    x_ub_g = pl.make_tile_group(type=tt_m6_vhs, addrs=[P3_UB_X], mutex_ids=[15])
    xy_ub_g = pl.make_tile_group(type=tt_m6_vhs33, addrs=[P3_UB_XY], mutex_ids=[16])
    tt_m6_akk_ub = pl.TileType(shape=[HS, CHUNK_C], dtype=pl.DT_BF16,
                                target_memory=pl.MemorySpace.Vec,
                                valid_shape=[-1, -1])
    m6_akk_ub_g = pl.make_tile_group(type=tt_m6_akk_ub,
                                     addrs=[0x3E000, 0x3F000], mutex_ids=[24, 29])

    snz_ub_g = pl.make_tile_group(type=tt_m6_nz, addrs=[P3_UB_XY], mutex_ids=[16])
    y_l1_g = pl.make_tile_group(type=tt_m6_l64, addrs=[P3_L1_Y0, P3_L1_Y1],
                                mutex_ids=[0, 1])
    _t71_l1_fp16 = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                               target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                               fractal=512, compact=0)
    t71_n_db_g = pl.make_tile_group(type=_t71_l1_fp16,
                                    addrs=[0x76000, 0x78000],
                                    mutex_ids=[24, 25])
    t71_eye = pl.make_tile(_t71_l1_fp16, addr=0x7A000)
    t71_zero = pl.make_tile(_t71_l1_fp16, addr=0x7C000)
    t71_scratch_g = pl.make_tile_group(type=_t71_l1_fp16, addrs=172032, mutex_ids=[26])
    t71_p_g = pl.make_tile_group(type=_t71_l1_fp16, addrs=180224, mutex_ids=[27])
    t71_tmp_g = pl.make_tile_group(type=_t71_l1_fp16, addrs=188416, mutex_ids=[28])
    _t71_l1_fp16_d = pl.TileType(shape=[16, 16], dtype=pl.DT_FP16,
                                 target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                                 fractal=512, compact=0)
    t71_tmpd0_g = pl.make_tile_group(type=_t71_l1_fp16_d, addrs=188416, mutex_ids=[28])
    t71_tmpd1_g = pl.make_tile_group(type=_t71_l1_fp16_d, addrs=188416 + 0xA00, mutex_ids=[29])
    t71_tmpd2_g = pl.make_tile_group(type=_t71_l1_fp16_d, addrs=188416 + 0x1400, mutex_ids=[30])
    t71_tmpd3_g = pl.make_tile_group(type=_t71_l1_fp16_d, addrs=188416 + 0x1E00, mutex_ids=[31])
    _t71_l1_bf16 = pl.TileType(shape=[64, 64], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                               fractal=512, compact=0)
    t71_inv_bf16_g = pl.make_tile_group(type=_t71_l1_bf16, addrs=147456, mutex_ids=[29])
    t71_eye_bf16 = pl.make_tile(_t71_l1_bf16, addr=0x7E000)

    _t71_a = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ,
                         fractal=512, compact=0)
    _t71_b = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                         fractal=512, compact=0)
    t71_a_g = pl.make_tile_group(type=_t71_a, addrs=[0x5000, 0x7000], mutex_ids=[18, 19])
    t71_b_g = pl.make_tile_group(type=_t71_b, addrs=[0xC000, 0xE000], mutex_ids=[20, 21])
    _t71_ad = pl.TileType(shape=[16, 16], dtype=pl.DT_FP16,
                          target_memory=pl.MemorySpace.Left, layout=pl.NZ,
                          fractal=512, compact=0)
    _t71_bd = pl.TileType(shape=[16, 16], dtype=pl.DT_FP16,
                          target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                          fractal=512, compact=0)
    t71_ad0_g = pl.make_tile_group(type=_t71_ad, addrs=[0x5000, 0x7000], mutex_ids=[18, 19])
    t71_ad1_g = pl.make_tile_group(type=_t71_ad, addrs=[0x5A00, 0x7A00], mutex_ids=[0, 1])
    t71_ad2_g = pl.make_tile_group(type=_t71_ad, addrs=[0x6400, 0x8400], mutex_ids=[2, 3])
    t71_ad3_g = pl.make_tile_group(type=_t71_ad, addrs=[0x6E00, 0x8E00], mutex_ids=[4, 5])
    t71_a10_g = pl.make_tile_group(type=_t71_ad, addrs=[0x5200, 0x7200], mutex_ids=[18, 19])
    t71_a32_g = pl.make_tile_group(type=_t71_ad, addrs=[0x6600, 0x8600], mutex_ids=[18, 19])
    t71_bd0_g = pl.make_tile_group(type=_t71_bd, addrs=[0xC000, 0xE000], mutex_ids=[20, 21])
    t71_bd1_g = pl.make_tile_group(type=_t71_bd, addrs=[0xCA00, 0xEA00], mutex_ids=[6, 7])
    t71_bd2_g = pl.make_tile_group(type=_t71_bd, addrs=[0xD400, 0xF400], mutex_ids=[8, 9])
    t71_bd3_g = pl.make_tile_group(type=_t71_bd, addrs=[0xDE00, 0xFE00], mutex_ids=[10, 11])
    t71_b10_g = pl.make_tile_group(type=_t71_bd, addrs=[0xC800, 0xE800], mutex_ids=[20, 21])
    t71_b32_g = pl.make_tile_group(type=_t71_bd, addrs=[0xDC00, 0xFC00], mutex_ids=[20, 21])
    _t71_bt = pl.TileType(shape=[16, 32], dtype=pl.DT_FP16,
                          target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                          fractal=512, compact=0)
    t71_bt0_g = pl.make_tile_group(type=_t71_bt, addrs=[0xC000, 0xE000], mutex_ids=[20, 21])
    t71_bt1_g = pl.make_tile_group(type=_t71_bt, addrs=[0xC800, 0xE800], mutex_ids=[20, 21])
    t71_bl0_g = pl.make_tile_group(type=_t71_bt, addrs=[0xD000, 0xF000], mutex_ids=[20, 21])
    t71_bl1_g = pl.make_tile_group(type=_t71_bt, addrs=[0xD800, 0xF800], mutex_ids=[20, 21])

    _t71_c = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         fractal=1024, compact=0)
    t71_c_g = pl.make_tile_group(type=_t71_c, addrs=[0x9000, 0x11000], mutex_ids=[22, 23])
    _t71_c16 = pl.TileType(shape=[16, 16], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                           fractal=1024, compact=0)

    t71_auxd0_g = pl.make_tile_group(type=_t71_c16, addrs=[0xD000, 0x15000], mutex_ids=[24, 25])
    t71_auxd1_g = pl.make_tile_group(type=_t71_c16, addrs=[0xE400, 0x16400], mutex_ids=[26, 27])
    t71_auxd2_g = pl.make_tile_group(type=_t71_c16, addrs=[0xF800, 0x17800], mutex_ids=[28, 29])
    t71_auxd3_g = pl.make_tile_group(type=_t71_c16, addrs=[0x10C00, 0x18C00], mutex_ids=[30, 31])
    t71_aux_g = pl.make_tile_group(type=_t71_c, addrs=[0xD000, 0x15000], mutex_ids=[24, 25])

    _t71_a_bf16 = pl.TileType(shape=[64, 64], dtype=pl.DT_BF16,
                              target_memory=pl.MemorySpace.Left, layout=pl.NZ,
                              fractal=512, compact=0)
    _t71_b_bf16 = pl.TileType(shape=[64, 64], dtype=pl.DT_BF16,
                              target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                              fractal=512, compact=0)
    t71_a_bf16_g = pl.make_tile_group(type=_t71_a_bf16, addrs=[0x5000, 0x7000], mutex_ids=[18, 19])
    t71_b_bf16_g = pl.make_tile_group(type=_t71_b_bf16, addrs=[0xC000, 0xE000], mutex_ids=[20, 21])

    _t71_st_nd_h = pl.TileType(shape=[32, 64], dtype=pl.DT_FP16,
                               target_memory=pl.MemorySpace.Vec,
                               valid_shape=[-1, -1], compact=0)
    _t71_st_nz_h = pl.TileType(shape=[32, 64], dtype=pl.DT_FP16,
                               target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                               fractal=512, valid_shape=[-1, -1], compact=1)
    _t71_st_nd_b = pl.TileType(shape=[32, 64], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Vec,
                               valid_shape=[-1, -1], compact=0)
    _t71_st_nz_b = pl.TileType(shape=[32, 64], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                               fractal=512, valid_shape=[-1, -1], compact=1)
    t71_nd_h_g = pl.make_tile_group(type=_t71_st_nd_h, addrs=[49152], mutex_ids=[16])
    t71_nz_h_g = pl.make_tile_group(type=_t71_st_nz_h, addrs=[53248], mutex_ids=[16])
    t71_nd_b_g = pl.make_tile_group(type=_t71_st_nd_b, addrs=[49152], mutex_ids=[16])
    t71_nz_b_g = pl.make_tile_group(type=_t71_st_nz_b, addrs=[53248], mutex_ids=[16])
    tt_m6_l64 = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                            valid_shape=[-1, -1])
    tt_m6_l128 = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat)
    tt_m6_la = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Left, layout=pl.NZ)
    tt_m6_lb128 = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                              target_memory=pl.MemorySpace.Right, layout=pl.ZN)
    m6_la_g = pl.make_tile_group(type=tt_m6_la, addrs=[LA0], mutex_ids=[18])
    m6_inv_l1_g = pl.make_tile_group(type=tt_m6_l64, addrs=[P3_L1_X], mutex_ids=[31])
    m6_op_l1_g = pl.make_tile_group(type=tt_m6_l128, addrs=[L6_OP], mutex_ids=[19])

    tt_m6_hd64 = pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat)
    tt_m6_la_hd64 = pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_BF16,
                                target_memory=pl.MemorySpace.Left, layout=pl.NZ)
    tt_m6_lb64 = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Right, layout=pl.ZN)
    tt_m6_lc_ut = pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                              fractal=1024)
    m6_vbt_l1_g = pl.make_tile_group(type=tt_m6_hd64, addrs=[L6_OP2],
                                     mutex_ids=[24])
    m6_la_vbt_g = pl.make_tile_group(type=tt_m6_la_hd64, addrs=[L6_LA_VBT],
                                     mutex_ids=[25])
    tt_m6_l64_zn = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Mat, layout=pl.ZN,
                               valid_shape=[-1, -1])
    m6_invt_zn_g = pl.make_tile_group(type=tt_m6_l64_zn, addrs=[P3_L1_X],
                                      mutex_ids=[31])
    m6_lb_invt_g = pl.make_tile_group(type=tt_m6_lb64, addrs=[L6_LB_OP2],
                                      mutex_ids=[25])

    m6_lb128_g = pl.make_tile_group(type=tt_m6_lb128, addrs=[L6_LB_OP], mutex_ids=[20])
    m6_w_lc_g = pl.make_tile_group(type=tt_m6_lc128, addrs=[L6_LC_W], mutex_ids=[21])
    m6_u_lc_g = pl.make_tile_group(type=tt_m6_lc_ut, addrs=[L6_LC_U], mutex_ids=[22])

    str1 = HEAD_D * (1 - layout_code) + seq_len * HEAD_D * layout_code
    str_qk = (num_key_heads * HEAD_D) * (1 - layout_code) + HEAD_D * layout_code
    str_vv = (num_value_heads * HEAD_D) * (1 - layout_code) + HEAD_D * layout_code
    q_v = pl.make_tensor(q, [batch_size, num_key_heads, token_count, HEAD_D], [q_bs, q_ns, q_ss, 1])
    k_v = pl.make_tensor(k, [batch_size, num_key_heads, token_count, HEAD_D], [k_bs, k_ns, k_ss, 1])
    g_v = pl.make_tensor(g, [batch_size, num_value_heads, token_count, HEAD_D], [g_bs, g_ns, g_ss, 1])

    ws_nk_bytes = batch_size * num_key_heads * seq_len * HEAD_D * 2
    qn_o = pl.make_tensor(workspace, [batch_size, num_key_heads, seq_len, HEAD_D],
                          [num_key_heads * seq_len * HEAD_D, str1, str_qk, 1], dtype=pl.DT_BF16)
    kn_o = pl.make_tensor(workspace + ws_nk_bytes, [batch_size, num_key_heads, seq_len, HEAD_D],
                          [num_key_heads * seq_len * HEAD_D, str1, str_qk, 1], dtype=pl.DT_BF16)
    qd_o = pl.make_tensor(
        qd_out,
        [batch_size, num_value_heads, seq_len, HEAD_D],
        [num_value_heads * seq_len * HEAD_D, str1, str_vv, 1],
    )
    kr_o = pl.make_tensor(
        kr_out,
        [batch_size, num_value_heads, seq_len, HEAD_D],
        [num_value_heads * seq_len * HEAD_D, str1, str_vv, 1],
    )
    v_v = pl.make_tensor(v, [batch_size, num_value_heads, token_count, HEAD_D], [v_bs, v_ns, v_ss, 1])
    w_o = pl.make_tensor(
        w_out,
        [batch_size, num_value_heads, seq_len, HEAD_D],
        [num_value_heads * seq_len * HEAD_D, str1, str_vv, 1],
    )
    upre_o = pl.make_tensor(upre_out, [batch_size, num_value_heads, HEAD_D, seq_len],
                            [num_value_heads * HEAD_D * seq_len, HEAD_D * seq_len, seq_len, 1])
    gam_o = pl.make_tensor(gam_out, [batch_size, num_value_heads, nc, HEAD_D],
                           [num_value_heads * nc * HEAD_D, nc * HEAD_D, HEAD_D, 1])
    mqk_o = pl.make_tensor(mqk_out, [batch_size, num_value_heads, nc, CHUNK_C, CHUNK_C],
                           [num_value_heads * nc * CHUNK_C * CHUNK_C, nc * CHUNK_C * CHUNK_C,
                            CHUNK_C * CHUNK_C, CHUNK_C, 1])
    q_c_o = pl.make_tensor(q_c_out, [batch_size, nc, num_key_heads, CHUNK_C, HEAD_D],
                           [nc * num_key_heads * CHUNK_C * HEAD_D, num_key_heads * CHUNK_C * HEAD_D,
                            CHUNK_C * HEAD_D, HEAD_D, 1])
    k_c_o = pl.make_tensor(k_c_out, [batch_size, nc, num_key_heads, CHUNK_C, HEAD_D],
                           [nc * num_key_heads * CHUNK_C * HEAD_D, num_key_heads * CHUNK_C * HEAD_D,
                            CHUNK_C * HEAD_D, HEAD_D, 1])
    v_c_o = pl.make_tensor(v_c_out, [batch_size, nc, num_value_heads, CHUNK_C, HEAD_D],
                           [nc * num_value_heads * CHUNK_C * HEAD_D, num_value_heads * CHUNK_C * HEAD_D,
                            CHUNK_C * HEAD_D, HEAD_D, 1])
    gk_c_o = pl.make_tensor(gk_c_out, [batch_size, nc, num_value_heads, CHUNK_C, HEAD_D],
                            [nc * num_value_heads * CHUNK_C * HEAD_D, num_value_heads * CHUNK_C * HEAD_D,
                             CHUNK_C * HEAD_D, HEAD_D, 1])
    beta_c_o = pl.make_tensor(beta_c_out, [batch_size, nc, num_value_heads, CHUNK_C],
                              [nc * num_value_heads * CHUNK_C, num_value_heads * CHUNK_C, CHUNK_C, 1])
    akk_o = pl.make_tensor(akk_out, [batch_size, nc, num_value_heads, CHUNK_C, CHUNK_C],
                           [nc * num_value_heads * CHUNK_C * CHUNK_C, num_value_heads * CHUNK_C * CHUNK_C,
                            CHUNK_C * CHUNK_C, CHUNK_C, 1])
    v_beta_bf16_o = pl.make_tensor(workspace + ws_nk_bytes * 2, [batch_size, num_value_heads, seq_len, HEAD_D],
                                   [num_value_heads * seq_len * HEAD_D, str1, str_vv, 1], dtype=pl.DT_BF16)
    v_beta_lo_o = pl.make_tensor(workspace + ws_nk_bytes * 2 + batch_size * num_value_heads * seq_len * HEAD_D * 2,
                                [batch_size, num_value_heads, seq_len, HEAD_D],
                                [num_value_heads * seq_len * HEAD_D, str1, str_vv, 1], dtype=pl.DT_BF16)
    a_view = pl.make_tensor(a_log, [num_value_heads, 1], [1, 1])
    lb_view = pl.make_tensor(lb_t, [1, 1], [1, 1])
    scale_view = pl.make_tensor(scale_t, [1, 1], [1, 1])

    own_tasks = (batch_size * num_key_heads - core_id + num_cores - 1) // num_cores
    own_work = own_tasks * nc * nq

    with pl.section_cube():
        l_pre = lower_l1_g.next()
        pl.load(l_pre, lower_tril, [0, 0])
    with pl.section_vector():
        sl_m_pre = m5_mask_g.next()
        pl.load(sl_m_pre, sl_mask, [0, 0])
        w_first = 0
        if own_work > 0:
            c_f = 0
            bh_f = core_id * nq
            b_f = bh_f // num_value_heads
            hv_f = bh_f % num_value_heads
            hk_f = hv_f // nq
            cp_dense_prefetch_v(q_v, k_v, g_v, b_f, hk_f, hv_f,
                          c_f * CHUNK_C, pl.min(64, token_count - c_f * 64), x_b_g, xk_g, xg_g)

    pw_valid = 0
    pc = 0
    pb = 0
    phv = 0
    ps0 = 0
    p_gsi = 0
    for work in pl.range(0, own_work):
        c = work % nc
        bh = (core_id + (work // (nc * nq)) * num_cores) * nq + (work // nc) % nq
        b = bh // num_value_heads
        hv = bh % num_value_heads
        hk = hv // nq
        s0 = c * CHUNK_C
        w_seq = work
        gate_slot_idx = w_seq % 2
        if pw_valid:
            with pl.section_cube():
                m4_kkt_mqk_blocks(m5_kkt_g, m5_mqk_g,
                                  p_gsi,
                                  lk0_g, lk1_g, lk2_g, lk3_g,
                                  lq0_g, lq1_g, lq2_g, lq3_g,
                                  rz0_g, rz1_g, rz2_g, rz3_g,
                                  m4_la_g, m4_b_g, m4_c_g)
        with pl.section_vector():
            tnd_m1_gate_norm_v(qn_o, kn_o, a_view, lb_view,
                           dt_bias, gate_l1, gate_slot_idx, b, hv, hk, s0,
                           1 if w_seq >= 2 else 0,
                           x_b_g, xk_g, xg_g, qn_h_g, kn_h_g, gate_f_g,
                           gate_nz_g, a_g, d_g, lb_g,
                           q_c_o, k_c_o, c, 1 if save_cache and hv % nq == 0 else 0, pl.min(64, token_count - s0))
        if pw_valid:
            with pl.section_vector():
                cp_dense_finish_v(beta_view, v_v, kn_h_g[p_gsi],
                            mqk_o, v_beta_bf16_o, v_beta_lo_o, v_c_o, beta_c_o,
                            layout_code, m5_mask_g, cs_g, sl_mask,
                            1 if w_seq == 1 else 0,
                            pc, pb, phv, ps0, num_value_heads, save_cache, ps0, pl.min(64, token_count - ps0),
                            y_l1_g[0], xy_ub_g, snz_ub_g, m5_kd_nz_g,
                            m6_op_l1_g[0],
                            m5_beta_raw_b_g, m5_beta_blk_f_g, m5_v_b_g, beta_cache_g,
                            m5_kkt_g, m5_mqk_g, m5_mqk_b_g,
                            m5_v_operand_g, m5_kd_operand_g,
                            t71_n_db_g[p_gsi], t71_eye, t71_zero, t71_eye_bf16,
                            t71_nd_h_g[0], t71_nz_h_g[0], t71_nd_b_g[0], t71_nz_b_g[0],
                            x_ub_g, eye64)
        with pl.section_cube():
            m2_cumsum_c(gate_l1, cs_g, gate_slot_idx,
                        pw_valid, lower_l1_g, lower_l0a_g, g_l0b_g, acc_g, save_cache)
        with pl.section_vector():
            m3_qd_kr_gam_v(qn_o, kn_o, qd_o, kr_o, gam_o, cs_g, scale_view,
                           c, b, hv, hk, s0, gate_slot_idx, num_value_heads, nc,
                           qn_h_g, kn_h_g, m3_out_b_g, m3_csl_g,
                           m3_sc_g,
                           m4_kn_b_g, m4_qn_b_g,
                           m4_opscr_b16_g, m4_ref_g,
                           lk0_g, lk1_g, lk2_g, lk3_g,
                           lq0_g, lq1_g, lq2_g, lq3_g,
                           rw0_g, rw1_g, rw2_g, rw3_g,
                           opnz_g, gk_c_o, save_cache)

        if pw_valid:
            m6_neumann_cube(
                t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g,
                t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g,
                t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g,
                t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g,
                t71_bl0_g, t71_bl1_g, t71_zero, t71_eye,
                t71_aux_g, t71_auxd0_g, t71_auxd1_g, t71_auxd2_g,
                t71_auxd3_g, t71_a_bf16_g, t71_b_bf16_g, t71_eye_bf16,
                v_beta_bf16_o, v_beta_lo_o, w_o, upre_o, t71_c_g,
                t71_n_db_g, p_gsi, pb, phv,
                ps0, t71_scratch_g, t71_tmpd0_g, t71_tmpd1_g,
                t71_tmpd2_g, t71_tmpd3_g, t71_p_g, t71_tmp_g,
                t71_inv_bf16_g, x_ub_g, m6_inv_l1_g, m6_akk_ub_g, m6_lb128_g,
                m6_op_l1_g, m6_la_g, m6_w_lc_g, m6_vbt_l1_g,
                m6_la_vbt_g, m6_lb_invt_g, m6_invt_zn_g, m6_u_lc_g,
                akk_o, save_cache)

        nw = pl.min(work + 1, own_work - 1)
        c_n = nw % nc
        bh_n = (core_id + (nw // (nc * nq)) * num_cores) * nq + (nw // nc) % nq
        with pl.section_vector():
            cp_dense_prefetch_v(q_v, k_v, g_v, bh_n // num_value_heads, (bh_n % num_value_heads) // nq,
                          bh_n % num_value_heads, c_n * CHUNK_C, pl.min(64, token_count - c_n * 64), x_b_g, xk_g, xg_g)
        pw_valid = 1
        pc = c
        pb = b
        phv = hv
        ps0 = s0
        p_gsi = gate_slot_idx

    n_works = own_work
    if pw_valid:
        with pl.section_cube():
            m4_kkt_mqk_blocks(m5_kkt_g, m5_mqk_g,
                              p_gsi,
                              lk0_g, lk1_g, lk2_g, lk3_g,
                              lq0_g, lq1_g, lq2_g, lq3_g,
                              rz0_g, rz1_g, rz2_g, rz3_g,
                              m4_la_g, m4_b_g, m4_c_g)
        with pl.section_vector():
            cp_dense_finish_v(beta_view, v_v, kn_h_g[p_gsi],
                        mqk_o, v_beta_bf16_o, v_beta_lo_o, v_c_o, beta_c_o,
                        layout_code, m5_mask_g, cs_g, sl_mask,
                        1 if n_works == 1 else 0,
                        pc, pb, phv, ps0, num_value_heads, save_cache, ps0, pl.min(64, token_count - ps0),
                        y_l1_g[0], xy_ub_g, snz_ub_g, m5_kd_nz_g,
                        m6_op_l1_g[0],
                        m5_beta_raw_b_g, m5_beta_blk_f_g, m5_v_b_g, beta_cache_g,
                        m5_kkt_g, m5_mqk_g, m5_mqk_b_g,
                        m5_v_operand_g, m5_kd_operand_g,
                        t71_n_db_g[p_gsi], t71_eye, t71_zero, t71_eye_bf16,
                        t71_nd_h_g[0], t71_nz_h_g[0], t71_nd_b_g[0], t71_nz_b_g[0],
                        x_ub_g, eye64)
    if n_works >= 1:
        with pl.section_vector():
            pass
            for _ in pl.range(0, pl.min(n_works, 2)):
                pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=3)
    if pw_valid:
        m6_neumann_cube(
            t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g,
            t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g,
            t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g,
            t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g,
            t71_bl0_g, t71_bl1_g, t71_zero, t71_eye,
            t71_aux_g, t71_auxd0_g, t71_auxd1_g, t71_auxd2_g,
            t71_auxd3_g, t71_a_bf16_g, t71_b_bf16_g, t71_eye_bf16,
            v_beta_bf16_o, v_beta_lo_o, w_o, upre_o, t71_c_g,
            t71_n_db_g, p_gsi, pb, phv,
            ps0, t71_scratch_g, t71_tmpd0_g, t71_tmpd1_g,
            t71_tmpd2_g, t71_tmpd3_g, t71_p_g, t71_tmp_g,
            t71_inv_bf16_g, x_ub_g, m6_inv_l1_g, m6_akk_ub_g, m6_lb128_g,
            m6_op_l1_g, m6_la_g, m6_w_lc_g, m6_vbt_l1_g,
            m6_la_vbt_g, m6_lb_invt_g, m6_invt_zn_g, m6_u_lc_g,
            akk_o, save_cache)

    # Drain this owner only before aliasing preprocessor UB/L1 for summary.
    with pl.section_vector():
        pl.system.bar_all()
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=13)
    with pl.section_cube():
        pl.system.bar_all()
        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=13)
        pl.system.bar_all()
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=12)
    with pl.section_vector():
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=12)
        pl.system.bar_all()
    cp_forward_chain(qd_out, kr_out, gam_out, mqk_out, w_out, upre_out,
        initial_state, summary, qd_out, initial_state, gk_c_out, v_c_out,
        layout_code, 0, first_continues, has_initial, 0, 1, nq, initial_bs, initial_ns, 0, 0, 0, 0, 0)


@pl.jit(auto_mutex=True)
def flash_kda_cp_stage2_tail_kernel(
    qd: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_BF16],
    kr: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_BF16],
    gam: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_FP32],
    mqk: pl.Tensor[[pl.DYNAMIC] * 5, pl.DT_BF16],
    w: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_BF16],
    upre: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_FP32],
    initial_state: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_FP32],
    summary: pl.Tensor[[pl.DYNAMIC] * 5, pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_BF16],
    final_state: pl.Tensor[[pl.DYNAMIC] * 4, pl.DT_FP32],
    h: pl.Tensor[[pl.DYNAMIC] * 5, pl.DT_BF16],
    v_new: pl.Tensor[[pl.DYNAMIC] * 5, pl.DT_BF16],
    layout_code: pl.DT_INT64, cp_rank: pl.DT_INT64,
    first_continues: pl.DT_INT64, has_initial: pl.DT_INT64, save_cache: pl.DT_INT64,
    ratio: pl.DT_INT64,
    initial_bs: pl.DT_INT64,
    initial_ns: pl.DT_INT64,
    out_bs: pl.DT_INT64,
    out_ns: pl.DT_INT64,
    out_ss: pl.DT_INT64,
    final_bs: pl.DT_INT64,
    final_ns: pl.DT_INT64,
):
    cp_forward_chain_tail(
        qd,
        kr,
        gam,
        mqk,
        w,
        upre,
        initial_state,
        summary,
        out,
        final_state,
        h,
        v_new,
        layout_code,
        cp_rank,
        first_continues,
        has_initial,
        save_cache,
        0,
        ratio,
        initial_bs,
        initial_ns,
        out_bs,
        out_ns,
        out_ss,
        final_bs,
        final_ns,
    )


# Device kernels are generated by build_forward_tnd.py; wrappers launch one each.


def _cp_stage1_packed(q, k, v, g, beta, scale, a_log, dt_bias, lower_bound, initial,
                     *, cp_meta, is_training, num_cores, allow_neg_eigval):
    if allow_neg_eigval:
        raise ValueError('CP v2 requires beta_scale=1')
    if q.ndim != 3 or q.shape != k.shape or v.shape != g.shape:
        raise ValueError('TND inputs must be rank three')
    token_count, num_key_heads, head_dim = q.shape
    num_value_heads = v.shape[1]
    if (
        token_count < 1
        or num_key_heads < 1
        or num_value_heads < 1
        or head_dim != 128
        or v.shape != (token_count, num_value_heads, 128)
        or beta.shape != (token_count, num_value_heads)
        or num_value_heads % num_key_heads
    ):
        raise ValueError('TND shape mismatch')
    for x in (q, k, v, g, beta):
        if x.dtype != torch.bfloat16 or x.device != q.device:
            raise ValueError('TND inputs must be BF16')
        _cp_check_row_major_view(x, 'TND raw input')
    _cp_check_bucket(cp_meta, num_key_heads, num_value_heads)
    lengths = tuple(cp_meta['sequence_lengths'])
    if not lengths or any(int(x) <= 0 for x in lengths) or sum(lengths) != token_count:
        raise ValueError('nonempty sequence_lengths must sum to tokens')
    num_sequences = len(lengths)
    num_chunks = sum((x + 63) // 64 for x in lengths)
    padded_seq_len = 64 * num_chunks
    if padded_seq_len > 131072:
        raise ValueError('local padded P exceeds 131072')
    if not -5 <= float(lower_bound) <= 0:
        raise ValueError('lower_bound outside [-5,0]')
    rank, size = int(cp_meta['cp_rank']), int(cp_meta['cp_size'])
    if not 0 <= rank < size or 'first_continues' not in cp_meta or 'last_continues' not in cp_meta:
        raise ValueError('packed CP rank/boundary flags are required')
    if a_log.shape != (num_value_heads,) or dt_bias.shape != (num_value_heads, 128):
        raise ValueError('parameter shape mismatch')
    for x in (a_log, dt_bias):
        if x.dtype != torch.float32 or x.device != q.device or not x.is_contiguous():
            raise ValueError('parameters must be contiguous FP32')
    if initial is not None and (
        initial.shape != (num_sequences, num_value_heads, 128, 128)
        or initial.dtype != torch.float32
        or initial.device != q.device
    ):
        raise ValueError('packed initial_state must be [M,Hv,V,K] FP32')
    if initial is not None:
        _cp_check_row_major_view(initial, 'initial_state')
        if tuple(initial.stride()[-2:]) != (128, 1):
            raise ValueError('initial_state inner matrix must be contiguous')
    metadata_shapes = {
        'cu_seqlens': (num_sequences + 1,),
        'chunk_starts': (num_chunks,),
        'chunk_lengths': (num_chunks,),
        'chunk_offsets': (num_sequences + 1,),
        'h_offsets': (num_sequences + 1,),
        'chunk_indices': (num_chunks, 2),
    }
    for n, sh in metadata_shapes.items():
        x = cp_meta[n]
        if tuple(x.shape) != sh or x.dtype != torch.int64 or x.device != q.device or not x.is_contiguous():
            raise ValueError(f'{n} must be contiguous device int64 {sh}')
    available, _ = _get_platform_config()
    cores = min(available if num_cores is None else int(num_cores), num_key_heads)
    if cores < 1 or cores > available:
        raise ValueError('invalid num_cores')

    def alloc(sh, dt):
        return torch.empty(sh, dtype=dt, device=q.device)

    prepared = {n: alloc((1, padded_seq_len, num_value_heads, 128), torch.bfloat16) for n in ('qd', 'kr', 'w')}
    prepared.update(
        gam=alloc((1, num_value_heads, num_chunks, 128), torch.float32),
        mqk=alloc((1, num_value_heads, num_chunks, 64, 64), torch.bfloat16),
        upre=alloc((1, num_value_heads, 128, padded_seq_len), torch.float32),
    )
    shapes = {
        'q_c': ((1, num_chunks, num_key_heads, 64, 128), torch.bfloat16),
        'k_c': ((1, num_chunks, num_key_heads, 64, 128), torch.bfloat16),
        'v_c': ((1, num_chunks, num_value_heads, 64, 128), torch.bfloat16),
        'gk_c': ((1, num_chunks, num_value_heads, 64, 128), torch.float32),
        'beta_c': ((1, num_chunks, num_value_heads, 64), torch.float32),
        'akk': ((1, num_chunks, num_value_heads, 64, 64), torch.bfloat16),
    }
    caches = {n: alloc(sh if is_training else (1,) * len(sh), dt) for n, (sh, dt) in shapes.items()}
    prepared.update({n: x if is_training else None for n, x in caches.items()})
    meta = dict(
        cache_version='flash_kda_cp_train_v3_bf16_state',
        summary_version='flash_kda_cp_affine_mm16_v1',
        layout_qkv='TND',
        prepared_layout='BSND',
        B=1,
        S=token_count,
        Nk=num_key_heads,
        Nv=num_value_heads,
        Hk=num_key_heads,
        Hv=num_value_heads,
        K=128,
        V=128,
        C=64,
        NC=num_chunks,
        P=padded_seq_len,
        M=num_sequences,
        J=num_chunks,
        ratio=num_value_heads // num_key_heads,
        global_Nk=int(cp_meta.get('global_Nk', num_key_heads)),
        global_Nv=int(cp_meta.get('global_Nv', num_value_heads)),
        hk_start=int(cp_meta.get('hk_start', 0)),
        hv_start=int(cp_meta.get('hv_start', 0)),
        scale=float(scale),
        lower_bound=float(lower_bound),
        l2_eps=1e-6,
        beta_scale=1,
        safe_gate=True,
        allow_neg_eigval=False,
        state_v_first=True,
        is_training=bool(is_training),
        has_initial_state=initial is not None,
    )
    prepared.update(meta=meta, cp_meta=dict(cp_meta), sequence_initial_state=initial)
    initial_arg = initial if initial is not None else alloc((1, 1, 1, 1), torch.float32)
    summary = alloc((1, num_value_heads, 2, 128, 128), torch.float32)
    workspace = alloc(
        (padded_seq_len * num_key_heads * 128 * 4 + padded_seq_len * num_value_heads * 128 * 4,), torch.uint8
    )
    const = alloc((cores * 65536,), torch.uint8)
    # unsqueeze is a metadata-only view; no conversion/padding kernel.
    flash_kda_cp_stage1_tnd_kernel[None, cores](
        cp_meta['chunk_starts'], cp_meta['chunk_lengths'], q.unsqueeze(0), k.unsqueeze(0),
        g.unsqueeze(0), beta.unsqueeze(0), v.unsqueeze(0), a_log, dt_bias, float(lower_bound), float(scale), 0,
        *[prepared[n] for n in ('qd', 'kr', 'gam', 'mqk', 'w', 'upre')],
        *[caches[n] for n in ('q_c', 'k_c', 'v_c', 'gk_c', 'beta_c', 'akk')],
        int(is_training), workspace, const, initial_arg, summary, int(
            cp_meta['first_continues']), int(initial is not None),
        cp_meta['cu_seqlens'], cp_meta['chunk_offsets'], cp_meta['h_offsets'],
        *_cp_input_strides(q, k, g, v, beta, initial_arg, 'TND'))
    return prepared, summary


def _cp_stage2_packed(prepared, all_summary, *, cp_meta, num_cores, out=None, final_state_out=None):
    m = prepared['meta']
    token_count, num_heads, num_sequences, num_chunks = m['S'], m['Hv'], m['M'], m['J']
    rank, size = int(cp_meta['cp_rank']), int(cp_meta['cp_size'])
    if cp_meta is not prepared['cp_meta']:
        for n in ('cp_rank', 'cp_size', 'first_continues', 'last_continues', 'sequence_lengths'):
            if cp_meta[n] != prepared['cp_meta'][n]:
                raise ValueError('CP metadata changed')
        for n in ('cu_seqlens', 'chunk_offsets', 'h_offsets', 'chunk_starts', 'chunk_lengths'):
            if cp_meta[n] is not prepared['cp_meta'][n]:
                raise ValueError('metadata tensor references changed')
    qd = prepared['qd']
    device = qd.device
    if (
        all_summary.shape != (size, num_heads, 2, 128, 128)
        or all_summary.dtype != torch.float32
        or all_summary.device != device
        or not all_summary.is_contiguous()
    ):
        raise ValueError('gathered summary mismatch')

    def alloc(sh, dt):
        return torch.empty(sh, dtype=dt, device=device)

    initial = prepared['sequence_initial_state']
    if initial is None:
        initial = alloc((1, 1, 1, 1), torch.float32)
    out = _cp_bound_output(out, (token_count, num_heads, 128), torch.bfloat16, device, 'out')
    final = _cp_bound_output(
        final_state_out, (num_sequences, num_heads, 128, 128), torch.float32, device, 'final_state_out'
    )
    _cp_check_output_aliases(out, final, prepared)
    train = m['is_training']
    h = alloc((1, num_chunks + num_sequences, num_heads, 128, 128) if train else (1, 1, 1, 1, 1), torch.bfloat16)
    vn = alloc((1, num_chunks, num_heads, 128, 64) if train else (1, 1, 1, 1, 1), torch.bfloat16)
    available, _ = _get_platform_config()
    cores = min(available if num_cores is None else int(num_cores), m['Hk'])
    if cores < 1 or cores > available:
        raise ValueError('invalid num_cores')
    flash_kda_cp_stage2_tnd_kernel[None, cores](
        *[prepared[n] for n in ('qd', 'kr', 'gam', 'mqk', 'w', 'upre')], initial, all_summary,
        out.unsqueeze(0), final, h, vn, 0, rank, int(cp_meta['first_continues']), int(m['has_initial_state']),
        int(train), m['ratio'], cp_meta['cu_seqlens'], cp_meta['chunk_offsets'], cp_meta['h_offsets'],
        *_cp_output_strides(initial, out, final, 'TND'))
    cache = None
    if train:
        cache = {n: prepared[n] for n in ('q_c', 'k_c', 'v_c', 'gk_c', 'beta_c', 'akk', 'mqk', 'w', 'kr', 'gam')}
        cache.update(h=h, v_new=vn, meta=dict(m), cp_meta=prepared['cp_meta'])
    return out, final, cache
