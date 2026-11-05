# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""mega_recurrent_kda_replayssm — one fused mixed AIC/AIV ``@pl.jit`` kernel, 1 launch.

Per-stage descriptions live with their module helper below (``_module1_cube`` ..
``_module4_cube``); the host wrapper only validates arguments and launches exactly one
kernel, with N / m-tile geometry folded through ``GeometryKey`` (DESIGN.md §5.5).

File map (top to bottom)
    geometry / ``GeometryKey``   host defaults and compile-time N specialisation
    Cube tile helpers            ``_mm_head`` .. ``_aux_gemm*`` (M1 panels, M4 output)
    M2 VF helpers + consumers    Conv1D taps, SiLU, L2 norm, panel drain
    M3 VF helpers                recurrence, decay activation, RMSNorm + gate
    M3 slot loop                 ``_m3_loop_prefetch`` (KEY_PREFETCH=1) / ``_m3_loop_inline``
    loop-A schedulers            ``_loopa_n384`` / ``_loopa_n128`` task tables
    module wrappers              ``_module1_cube`` / ``_module2_vector`` /
                                 ``_module3_vector`` / ``_module4_cube`` (+ ``_make_panels``)
    kernel                       the single ``@pl.jit`` MIX entry (~30 lines, calls only)
    host wrapper                 validation, workspace, exactly one launch
"""


from __future__ import annotations

import math
import os
import sys

import torch

import torch_npu  # noqa: F401

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.tilingkey import TilingKeyField

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OP_DIR = os.path.dirname(_THIS_DIR)
_ASSETS_DIR = os.path.join(_THIS_DIR, "assets")
for _p in (_OP_DIR, _THIS_DIR, _ASSETS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Geometry / design constants (DESIGN.md §3.1)
# ---------------------------------------------------------------------------
H, D, S, CONV_K = 7168, 128, 8, 4
#: Host-side default head count (tests/golden only); the kernel takes N from
#: the ``GeometryKey.KEY_NUM_HEADS`` tiling key and the wrapper infers it from the weights.
DEFAULT_HEADS = 6
BATCH_TILE = 16                 # frontend tile granularity reference
#: m-tile row ladder: the host picks the largest rung dividing ``rows = B*S``.
M_TILE_LADDER = (128, 112, 96, 80, 64, 48, 32)
TILE = 128                      # BM = BN = BK = 128
KC = 256                        # L1 K-panel depth
KS = 128                        # L0A / L0B K step
KC_STEPS = H // KC              # 28
KS_STEPS = KC // KS             # 2
K_STEPS = H // TILE             # 56 = H // BK
PANEL_HEADS = 3                 # 3 heads packed per N384 task
PANEL_N = PANEL_HEADS * D       # 384
# N-dependent constants (KEY_NUM_HEADS / KEY_PROJ_WIDTH / KEY_QKV_WIDTH /
# KEY_QKV_PANELS / KEY_GATE_PANELS / KEY_PANEL_TASKS / KEY_N128_TASKS) are
# compile-time tiling-key fields of ``GeometryKey``; one kernel serves N = 6/12/24
# (TP16/8/4).
PANEL_DEPTH = 2                 # cross-core UB slots per head position
READY_IDS = (0, 1, 2, 3, 4, 5)      # Cube -> Vector, indexed h*2+slot
RELEASE_IDS = (6, 7, 8, 9, 10, 14)  # Vector -> Cube (skip sync_all's 11..13)
MAX_BATCH_GROUPS = 32
DH = D // 2                     # FP32 VF register width (lanes) for the [*,D] tiles
NORM_EPS = 1e-6                 # Q/K L2-norm epsilon (module_interfaces / golden)
PUB_VEC = 3                     # backend step 1: AIV->AIC, INTRA_BLOCK
PUB_ACK = 4                     # backend step 2: AIC->AIV, INTRA_BLOCK


class GeometryKey:
    """Compile-time head-count specialisation (``N = 96/TP``, TP = 16/8/4).

    Every N-dependent constant is a separate field so the parser folds it into the
    compiled variant (one tiling-key field per N-dependent constant);
    the N-dependent extents are declared ``pl.STATIC`` and folded from the launch.
    """

    KEY_NUM_HEADS = TilingKeyField(bits=2, values=(6, 12, 24))
    #: Projection width ``N * D`` (gate / decay-A / beta-output columns).
    KEY_PROJ_WIDTH = TilingKeyField(bits=2, values=(768, 1536, 3072))
    #: QKV projection width ``3 * N * D``.
    KEY_QKV_WIDTH = TilingKeyField(bits=2, values=(2304, 4608, 9216))
    #: QKV N384 panels per group (``KEY_QKV_WIDTH / PANEL_N``).
    KEY_QKV_PANELS = TilingKeyField(bits=2, values=(6, 12, 24))
    #: Gate N384 panels per group (``KEY_PROJ_WIDTH / PANEL_N``).
    KEY_GATE_PANELS = TilingKeyField(bits=2, values=(2, 4, 8))
    #: Projection panels per group (``KEY_QKV_PANELS + KEY_GATE_PANELS``): loop-A panel tasks.
    KEY_PANEL_TASKS = TilingKeyField(bits=2, values=(8, 16, 32))
    #: N128-tier projection tasks per group (``4 * KEY_NUM_HEADS``) when N384 is not used.
    KEY_N128_TASKS = TilingKeyField(bits=2, values=(24, 48, 96))
    #: Padded beta-logit width (16 for N <= 16, 32 for N = 24).
    KEY_BETA_WIDTH = TilingKeyField(bits=1, values=(16, 32))
    #: m-tile rows: the largest rung of ``M_TILE_LADDER`` dividing ``rows``.
    KEY_M_TILE_ROWS = TilingKeyField(bits=3, values=(32, 48, 64, 80, 96, 112, 128))
    #: Half m-tile rows (per-AIV panel half).
    KEY_M_TILE_ROWS_HALF = TilingKeyField(bits=3, values=(16, 24, 32, 40, 48, 56, 64))
    #: Batches packed into one m-tile (``KEY_M_TILE_ROWS / S``).
    KEY_BATCH_PER_GROUP = TilingKeyField(bits=3, values=(4, 6, 8, 10, 12, 14, 16))
    #: Batches per AIV subblock (``KEY_BATCH_PER_GROUP / 2``).
    KEY_BATCH_PER_SUBBLOCK = TilingKeyField(bits=3, values=(2, 3, 4, 5, 6, 7, 8))
    #: K-split of the β / decay-A aux projections (1 = single task, 8 = split).
    KEY_K_SPLIT = TilingKeyField(bits=1, values=(1, 8))
    #: M3 cross-item prefetch (0 = inline loads, 1 = prefetch the next item).
    KEY_PREFETCH = TilingKeyField(bits=1, values=(0, 1))
    #: M3 slot granularity: items per worker per slot (see ``_M3_K_TABLE``).
    KEY_M3_K = TilingKeyField(bits=2, values=(1, 2, 4, 8))


# ===========================================================================
# Cube helpers (M1 projection panels, M4 output projection)
# ===========================================================================
def _mm_head(la, l0b, l0c_g, wz_s, kc, ks, h):
    """One head's L0B sub-window + K-accumulated MMAD into ``l0c_g``."""
    lb = l0b.next()
    pl.move(lb, wz_s, [ks * KS, h * TILE])
    if kc == 0 and ks == 0:
        pl.matmul(l0c_g.current(), la, lb)
    else:
        pl.matmul_acc(l0c_g.current(), l0c_g.current(), la, lb)


def _handoff(panel_g, l0c_g, slot, h):
    """Panel-level Cube -> AIV handoff for one head position (FIX pipe)."""
    event = h * PANEL_DEPTH + slot
    pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=RELEASE_IDS[event])
    pl.set_validshape(l0c_g.current(), [KEY_M_TILE_ROWS, TILE])
    pl.set_validshape(panel_g[slot], [KEY_M_TILE_ROWS_HALF, TILE])
    pl.move(panel_g[slot], l0c_g.current(), acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=READY_IDS[event])


def _n384_task(w, w_off, m_group, task_ord, x2, x_l1, w_l1, w_zn, l0a, l0b,
               l0c0, l0c1, l0c2, panel_h0, panel_h1, panel_h2):
    """One N384 panel task: KC/KS K-tiling (x and weight reuse) + 3 head handoffs."""
    slot = task_ord % PANEL_DEPTH
    for kc in pl.range(KC_STEPS):
        s = kc % 2
        pl.load(x_l1[s], x2, [m_group * KEY_M_TILE_ROWS, kc * KC])
        pl.load(w_l1[s], w, [w_off, kc * KC])
        for ks in pl.range(KS_STEPS):
            la = l0a.next()
            pl.move(la, x_l1[s], [0, ks * KS])
            _mm_head(la, l0b, l0c0, w_zn[s], kc, ks, 0)
            _mm_head(la, l0b, l0c1, w_zn[s], kc, ks, 1)
            _mm_head(la, l0b, l0c2, w_zn[s], kc, ks, 2)
    _handoff(panel_h0, l0c0, slot, 0)
    _handoff(panel_h1, l0c1, slot, 1)
    _handoff(panel_h2, l0c2, slot, 2)


def _n128_task(w, w_off, m_group, task_ord, x2, x_l1, w_l1, w_zn, l0a, l0b,
               l0c0, panel_h0):
    """N128 degenerate tier: one n-tile (one head) per task, no x reuse.

    Same event protocol / slot expression / Section structure as the N384 tier;
    only ``PANEL_HEADS`` (1 vs 3) and the weight panel width differ.
    """
    slot = task_ord % PANEL_DEPTH
    for kc in pl.range(KC_STEPS):
        s = kc % 2
        pl.load(x_l1[s], x2, [m_group * KEY_M_TILE_ROWS, kc * KC])
        pl.load(w_l1[s], w, [w_off, kc * KC])
        for ks in pl.range(KS_STEPS):
            la = l0a.next()
            pl.move(la, x_l1[s], [0, ks * KS])
            lb = l0b.next()
            pl.move(lb, w_zn[s], [ks * KS, 0])
            if kc == 0 and ks == 0:
                pl.matmul(l0c0.current(), la, lb)
            else:
                pl.matmul_acc(l0c0.current(), l0c0.current(), la, lb)
    _handoff(panel_h0, l0c0, slot, 0)


def _aux_gemm(a, w, out, mt, nt, k_tiles, bn, xa, wn, wz, l0a_aux, l0b_aux, acc):
    for kk in pl.range(k_tiles):
        am = xa.next()
        pl.load(am, a, [mt * KEY_M_TILE_ROWS, kk * TILE])
        wq = wn.next()
        pl.load(wq, w, [nt * bn, kk * TILE])
        al = l0a_aux.next()
        pl.move(al, am)
        bl = l0b_aux.next()
        pl.move(bl, wz.next())
        if kk == 0:
            pl.matmul(acc.current(), al, bl)
        else:
            pl.matmul_acc(acc.current(), acc.current(), al, bl)
    pl.set_validshape(acc.current(), [KEY_M_TILE_ROWS, bn])
    pl.store(out, acc.current(), [mt * KEY_M_TILE_ROWS, nt * bn])


def _aux_gemm_resident(a, w_res_zn, out, mt, nt, k_tiles, bn, xa, l0a, l0b, acc):
    """Same as ``_aux_gemm`` but the right operand is a resident L1 panel.

    ``w_res_zn`` is the ZN alias of the whole ``[256, KEY_PROJ_WIDTH]`` output-weight shard
    loaded once per AIC; each K step only slices it into L0.
    """
    for kk in pl.range(k_tiles):
        am = xa.next()
        pl.load(am, a, [mt * KEY_M_TILE_ROWS, kk * TILE])
        al = l0a.next()
        pl.move(al, am)
        bl = l0b.next()
        pl.move(bl, w_res_zn.current(), [kk * TILE, 0])
        if kk == 0:
            pl.matmul(acc.current(), al, bl)
        else:
            pl.matmul_acc(acc.current(), acc.current(), al, bl)
    pl.set_validshape(acc.current(), [KEY_M_TILE_ROWS, bn])
    pl.store(out, acc.current(), [mt * KEY_M_TILE_ROWS, nt * bn])


def _aux_gemm_head(a, w, nt, head, init, xa, wn, wz, l0a, l0b, acc):
    """One head's K tile of the output projection into a persistent accumulator."""
    am = xa.next()
    pl.load(am, a, [0, head * TILE])
    wq = wn.next()
    pl.load(wq, w, [nt * 256, head * TILE])
    al = l0a.next()
    pl.move(al, am)
    bl = l0b.next()
    pl.move(bl, wz.next())
    if init:
        pl.matmul(acc.current(), al, bl)
    else:
        pl.matmul_acc(acc.current(), acc.current(), al, bl)


def _aux_gemm_head_res(a, w_res_zn, head, init, xa, l0a, l0b, acc):
    """One head's K tile from the resident L1 Wout shard into ``acc``."""
    am = xa.next()
    pl.load(am, a, [0, head * TILE])
    al = l0a.next()
    pl.move(al, am)
    bl = l0b.next()
    pl.move(bl, w_res_zn.current(), [head * TILE, 0])
    if init:
        pl.matmul(acc.current(), al, bl)
    else:
        pl.matmul_acc(acc.current(), acc.current(), al, bl)


# ===========================================================================
# Module-2 vector helpers (DESIGN.md §1: VF implementations)
# ===========================================================================
@pl.vector_function
def conv_tap_vf(seq, filt, out):
    """Raw K=4 causal Conv1D taps (pre-SiLU) for the ``S`` tokens, register-resident.

    ``seq`` is ``[S+K-1, D]`` = ``[state_hist(3), raw(8)]``; ``filt`` is ``[4, D]``.
    Accumulation order matches the golden: ``(p0+p1)+(p2+p3)``.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    f0a = vf.load_align(filt, 0)
    f0b = vf.load_align(filt, DH)
    f1a = vf.load_align(filt, D)
    f1b = vf.load_align(filt, D + DH)
    f2a = vf.load_align(filt, 2 * D)
    f2b = vf.load_align(filt, 2 * D + DH)
    f3a = vf.load_align(filt, 3 * D)
    f3b = vf.load_align(filt, 3 * D + DH)
    for t in pl.range(S):
        a0a = vf.load_align(seq, t * D)
        a0b = vf.load_align(seq, t * D + DH)
        a1a = vf.load_align(seq, (t + 1) * D)
        a1b = vf.load_align(seq, (t + 1) * D + DH)
        a2a = vf.load_align(seq, (t + 2) * D)
        a2b = vf.load_align(seq, (t + 2) * D + DH)
        a3a = vf.load_align(seq, (t + 3) * D)
        a3b = vf.load_align(seq, (t + 3) * D + DH)
        p0a = vf.mul(a0a, f0a, preg)
        p0b = vf.mul(a0b, f0b, preg)
        p1a = vf.mul(a1a, f1a, preg)
        p1b = vf.mul(a1b, f1b, preg)
        q0a = vf.add(p0a, p1a, preg)
        q0b = vf.add(p0b, p1b, preg)
        p2a = vf.mul(a2a, f2a, preg)
        p2b = vf.mul(a2b, f2b, preg)
        p3a = vf.mul(a3a, f3a, preg)
        p3b = vf.mul(a3b, f3b, preg)
        q1a = vf.add(p2a, p3a, preg)
        q1b = vf.add(p2b, p3b, preg)
        ca = vf.add(q0a, q1a, preg)
        cb = vf.add(q0b, q1b, preg)
        vf.store_align(out, ca, preg, t * D)
        vf.store_align(out, cb, preg, t * D + DH)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def silu_vf(x):
    """SiLU over a flattened FP32 vector tile, in place (register-resident)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    for r in pl.range((S * D) // DH):
        off = r * DH
        v = vf.load_align(x, off)
        n = vf.neg(v, preg)
        e = vf.exp(n, preg)
        dd = vf.adds(e, 1.0, preg)
        iv = vf.div(one, dd, preg)
        o = vf.mul(v, iv, preg)
        vf.store_align(x, o, preg, off)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def l2norm_vf(x):
    """Row-wise (D-axis) L2 normalisation, eps 1e-6, in place over the FP32 tile."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for m in pl.range(S):
        base = m * D
        s = vf.full(0.0, preg, dtype=pl.DT_FP32)
        for r in pl.range(D // DH):
            reg = vf.load_align(x, base + r * DH)
            sq = vf.mul(reg, reg, preg)
            part = vf.reduce_sum(sq, preg)
            s = vf.add(s, part, preg)
        s = vf.full(s, preg)
        s = vf.adds(s, NORM_EPS, preg)
        s = vf.sqrt(s, preg)
        for r in pl.range(D // DH):
            reg = vf.load_align(x, base + r * DH)
            o = vf.div(reg, s, preg)
            vf.store_align(x, o, preg, base + r * DH)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


# ===========================================================================
# Module-2 consumer helpers (panel drain)
# ===========================================================================
def _load_filter(fb, cw, layout, blk, qkv):
    """Load the canonical ``[CONV_K, D]`` tap filter for channel block ``blk``.

    ``conv1d_weight`` arrives as a raw ``pl.Ptr`` + ``conv_layout`` (README.md 参数说明); the
    in-kernel view for each public layout is built here (no host canonicalisation).
    """
    if layout == 0:
        cw0 = pl.make_tensor(cw, [3 * KEY_NUM_HEADS * CONV_K, D], [D, 1])
        pl.load(fb.current(), cw0, [blk * CONV_K, 0])
    elif layout == 1:
        cw1 = pl.make_tensor(cw, [CONV_K, 3 * KEY_NUM_HEADS, D], [3 * KEY_NUM_HEADS * D, D, 1])
        pl.load(fb.current(), cw1, [0, blk, 0], order=[0, 2])
    else:
        cw2 = pl.make_tensor(cw, [3, CONV_K, D], [CONV_K * D, D, 1])
        pl.load(fb.current(), cw2, [qkv, 0, 0], order=[1, 2])


def _consume_channel(h, slot, qkv, head, m_group, sub, cw, cw_layout,
                     conv_state, conv_indices, conv_accepted, panel_g,
                     ws_qkv, replay_k2,
                     pf, fb, filt, h3b, h3, r8b, r8, seq, sc, ob, sv8, bh):
    """Module 2 for one ``(qkv, head)`` panel: 8 batches end to end."""
    event = h * PANEL_DEPTH + slot
    blk = qkv * KEY_NUM_HEADS + head
    ch0 = blk * D
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=READY_IDS[event])
    _load_filter(fb, cw, cw_layout, blk, qkv)
    pl.cast(filt.current(), fb.current())
    for lb in pl.range(KEY_BATCH_PER_SUBBLOCK):
        b = m_group * KEY_BATCH_PER_GROUP + sub * KEY_BATCH_PER_SUBBLOCK + lb
        row = b * S
        idx = pl.getval(conv_indices, b)
        accepted = pl.getval(conv_accepted, b)
        off = pl.max(accepted - 1, 0)
        pl.move(pf.current(), panel_g[slot], [lb * S, 0])   # V-pipe read of the panel
        pl.cast(r8b.current(), pf.current())                # BF16 panel (design contract)
        if idx != 0:
            pl.set_validshape(h3b.current(), [CONV_K - 1, D])
            pl.load(h3b.current(), conv_state, [idx, off, ch0])
            pl.cast(h3.current(), h3b.current())
        else:
            pl.expands(h3.current(), 0.0)
            pl.expands(r8b.current(), 0.0)
        pl.cast(r8.current(), r8b.current())
        pl.insert(seq.current(), h3.current(), [0, 0])
        pl.insert(seq.current(), r8.current(), [CONV_K - 1, 0])
        conv_tap_vf(seq.current(), filt.current(), sc.current())
        silu_vf(sc.current())
        # BF16 boundary: the golden's 'convolved' is bf16, widened again before the norm.
        pl.cast(ob.current(), sc.current())
        pl.cast(sc.current(), ob.current())
        if qkv < 2:
            l2norm_vf(sc.current())
        pl.cast(ob.current(), sc.current())
        pl.store(ws_qkv, ob.current(), [row, ch0])
        if qkv == 1:
            pl.cast(bh.current(), sc.current())
            pl.store(replay_k2, bh.current(), [row, head * D])
        if idx != 0:
            pl.set_validshape(sv8.current(), [CONV_K - 2, D])
            pl.load(sv8.current(), conv_state, [idx, off + 1, ch0])
            pl.store(conv_state, sv8.current(), [idx, 0, ch0])
            pl.set_validshape(sv8.current(), [S, D])
            pl.move(sv8.current(), r8b.current())           # round-BF16 raw rows
            pl.store(conv_state, sv8.current(), [idx, CONV_K - 2, ch0])
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=RELEASE_IDS[event])


def _consume_gate(h, slot, head, m_group, sub, panel_g, ws_gate, pf, ob):
    """Module 2 pass-through for one gate panel: cast the panel to BF16 and store."""
    event = h * PANEL_DEPTH + slot
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=READY_IDS[event])
    for lb in pl.range(KEY_BATCH_PER_SUBBLOCK):
        b = m_group * KEY_BATCH_PER_GROUP + sub * KEY_BATCH_PER_SUBBLOCK + lb
        row = b * S
        pl.move(pf.current(), panel_g[slot], [lb * S, 0])
        pl.cast(ob.current(), pf.current())
        pl.store(ws_gate, ob.current(), [row, head * D])
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=RELEASE_IDS[event])


def _m2_loop(block_id, num_cores, n_groups, nd_count, is_n384, sub, cw, cw_layout,
             conv_state, conv_indices, conv_accepted, panel_h0, panel_h1, panel_h2,
             ws_qkv, ws_gate, replay_k2,
             pf, fb, filt, h3b, h3, r8b, r8, seq, sc, ob, sv8, bh):
    """Module-2 consumer grid-stride: only the panel tasks (same ``wi`` table as M1)."""
    for wi in pl.range(block_id, n_groups * nd_count, num_cores):
        nd = wi // n_groups
        m_group = wi % n_groups
        task_ord = (wi - block_id) // num_cores
        slot = task_ord % PANEL_DEPTH
        if is_n384:
            # N384 tier: `KEY_QKV_PANELS` qkv panels then `KEY_GATE_PANELS` gate panels per group.
            if nd < KEY_QKV_PANELS:
                qkv = nd // KEY_GATE_PANELS
                head0 = (nd % KEY_GATE_PANELS) * PANEL_HEADS
                _consume_channel(0, slot, qkv, head0 + 0, m_group, sub, cw, cw_layout,
                                 conv_state, conv_indices, conv_accepted, panel_h0,
                                 ws_qkv, replay_k2,
                                 pf, fb, filt, h3b, h3, r8b, r8, seq, sc, ob, sv8, bh)
                _consume_channel(1, slot, qkv, head0 + 1, m_group, sub, cw, cw_layout,
                                 conv_state, conv_indices, conv_accepted, panel_h1,
                                 ws_qkv, replay_k2,
                                 pf, fb, filt, h3b, h3, r8b, r8, seq, sc, ob, sv8, bh)
                _consume_channel(2, slot, qkv, head0 + 2, m_group, sub, cw, cw_layout,
                                 conv_state, conv_indices, conv_accepted, panel_h2,
                                 ws_qkv, replay_k2,
                                 pf, fb, filt, h3b, h3, r8b, r8, seq, sc, ob, sv8, bh)
            else:
                head0 = (nd - KEY_QKV_PANELS) * PANEL_HEADS
                _consume_gate(0, slot, head0 + 0, m_group, sub, panel_h0, ws_gate, pf, ob)
                _consume_gate(1, slot, head0 + 1, m_group, sub, panel_h1, ws_gate, pf, ob)
                _consume_gate(2, slot, head0 + 2, m_group, sub, panel_h2, ws_gate, pf, ob)
        elif nd < 3 * KEY_NUM_HEADS:
            # N128 tier: one head per task, 3*KEY_NUM_HEADS qkv panels then KEY_NUM_HEADS gate panels.
            _consume_channel(0, slot, nd // KEY_NUM_HEADS, nd % KEY_NUM_HEADS, m_group, sub, cw, cw_layout,
                             conv_state, conv_indices, conv_accepted, panel_h0,
                             ws_qkv, replay_k2,
                             pf, fb, filt, h3b, h3, r8b, r8, seq, sc, ob, sv8, bh)
        else:
            _consume_gate(0, slot, nd - 3 * KEY_NUM_HEADS, m_group, sub, panel_h0, ws_gate, pf, ob)


# ===========================================================================
# Module-3 recurrence VF helpers (delta-rule step, single and paired token)
# ===========================================================================
@pl.vector_function
def rec_token_vf2(state, decay_t, key_t, query_t, value_t, o_t, kv_row, delta_row,
                  tb: pl.DT_INT64, beta_logit: pl.DT_FP32, scale: pl.DT_FP32):
    """One delta-rule step for token ``tb // D`` read from ``[S, D]`` row tiles.

    ``beta = sigmoid(beta_logit)``; ``o`` is written into ``o_t`` at row ``tb``.
    ``state`` is the register-resident FP32 ``[V, K]`` state.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    m1 = vf.update_mask(1, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    d0 = vf.load_align(decay_t, tb)
    k0 = vf.load_align(key_t, tb)
    d1 = vf.load_align(decay_t, tb + DH)
    k1 = vf.load_align(key_t, tb + DH)
    dk0 = vf.mul(d0, k0, preg)
    dk1 = vf.mul(d1, k1, preg)
    for dv in pl.range(D):
        st0 = vf.load_align(state, dv * D)
        st1 = vf.load_align(state, dv * D + DH)
        sk0 = vf.mul(st0, dk0, preg)
        sk1 = vf.mul(st1, dk1, preg)
        s = vf.add(sk0, sk1, preg)
        kv = vf.reduce_sum(s, preg)
        vf.store_align(kv_row, kv, m1, dv, dist=pl.StoreDist.FIRST_ELEMENT_B32)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    bl = vf.full(beta_logit, preg, dtype=pl.DT_FP32)
    nb = vf.neg(bl, preg)
    eb = vf.exp(nb, preg)
    db = vf.adds(eb, 1.0, preg)
    beta = vf.div(one, db, preg)
    v0 = vf.load_align(value_t, tb)
    kb0 = vf.load_align(kv_row, 0)
    r0 = vf.sub(v0, kb0, preg)
    r0 = vf.mul(r0, beta, preg)
    vf.store_align(delta_row, r0, preg, 0)
    v1 = vf.load_align(value_t, tb + DH)
    kb1 = vf.load_align(kv_row, DH)
    r1 = vf.sub(v1, kb1, preg)
    r1 = vf.mul(r1, beta, preg)
    vf.store_align(delta_row, r1, preg, DH)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    d0 = vf.load_align(decay_t, tb)
    k0 = vf.load_align(key_t, tb)
    q0 = vf.load_align(query_t, tb)
    d1 = vf.load_align(decay_t, tb + DH)
    k1 = vf.load_align(key_t, tb + DH)
    q1 = vf.load_align(query_t, tb + DH)
    for dv in pl.range(D):
        dl = vf.load_align(delta_row, dv, dist=pl.LoadDist.BRC_B32)
        st0 = vf.load_align(state, dv * D)
        st1 = vf.load_align(state, dv * D + DH)
        a0 = vf.mul(st0, d0, preg)
        b0 = vf.mul(dl, k0, preg)
        sn0 = vf.add(a0, b0, preg)
        a1 = vf.mul(st1, d1, preg)
        b1 = vf.mul(dl, k1, preg)
        sn1 = vf.add(a1, b1, preg)
        vf.store_align(state, sn0, preg, dv * D)
        vf.store_align(state, sn1, preg, dv * D + DH)
        o0 = vf.mul(sn0, q0, preg)
        o1 = vf.mul(sn1, q1, preg)
        oo = vf.add(o0, o1, preg)
        o_sum = vf.reduce_sum(oo, preg)
        o_sum = vf.muls(o_sum, scale, preg)
        vf.store_align(o_t, o_sum, m1, tb + dv, dist=pl.StoreDist.FIRST_ELEMENT_B32)


@pl.vector_function
def rec_pair_vf(state, decay_t, key_t, query_t, value_t, o_t,
                kv0, kv1, op0, op1, delta0, delta1, sc,
                tb: pl.DT_INT64, beta0: pl.DT_FP32, beta1: pl.DT_FP32, scale: pl.DT_FP32):
    """Two consecutive delta-rule steps sharing one state load/store per pass."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    m1 = vf.update_mask(1, dtype=pl.DT_FP32)
    d0a = vf.load_align(decay_t, tb)
    d0b = vf.load_align(decay_t, tb + DH)
    d1a = vf.load_align(decay_t, tb + D)
    d1b = vf.load_align(decay_t, tb + D + DH)
    k0a = vf.load_align(key_t, tb)
    k0b = vf.load_align(key_t, tb + DH)
    k1a = vf.load_align(key_t, tb + D)
    k1b = vf.load_align(key_t, tb + D + DH)
    q0a = vf.load_align(query_t, tb)
    q0b = vf.load_align(query_t, tb + DH)
    q1a = vf.load_align(query_t, tb + D)
    q1b = vf.load_align(query_t, tb + D + DH)
    dk0a = vf.mul(d0a, k0a, preg)
    dk0b = vf.mul(d0b, k0b, preg)
    d1k1a = vf.mul(d1a, k1a, preg)
    d1k1b = vf.mul(d1b, k1b, preg)
    ddk1a = vf.mul(d0a, d1k1a, preg)
    ddk1b = vf.mul(d0b, d1k1b, preg)
    dq0a = vf.mul(d0a, q0a, preg)
    dq0b = vf.mul(d0b, q0b, preg)
    d1q1a = vf.mul(d1a, q1a, preg)
    d1q1b = vf.mul(d1b, q1b, preg)
    ddq1a = vf.mul(d0a, d1q1a, preg)
    ddq1b = vf.mul(d0b, d1q1b, preg)
    dda = vf.mul(d0a, d1a, preg)
    ddb = vf.mul(d0b, d1b, preg)
    kda = vf.mul(k0a, d1a, preg)
    kdb = vf.mul(k0b, d1b, preg)
    tk0 = vf.mul(k0a, d1k1a, preg)
    tk1 = vf.mul(k0b, d1k1b, preg)
    tks = vf.add(tk0, tk1, preg)
    sk = vf.reduce_sum(tks, preg)
    tq0a = vf.mul(k0a, q0a, preg)
    tq0b = vf.mul(k0b, q0b, preg)
    tq0 = vf.add(tq0a, tq0b, preg)
    sq0 = vf.reduce_sum(tq0, preg)
    tq1a = vf.mul(kda, q1a, preg)
    tq1b = vf.mul(kdb, q1b, preg)
    tq1 = vf.add(tq1a, tq1b, preg)
    sq1 = vf.reduce_sum(tq1, preg)
    tq2a = vf.mul(k1a, q1a, preg)
    tq2b = vf.mul(k1b, q1b, preg)
    tq2 = vf.add(tq2a, tq2b, preg)
    sq2 = vf.reduce_sum(tq2, preg)
    vf.store_align(sc, sk, m1, 0, dist=pl.StoreDist.FIRST_ELEMENT_B32)
    vf.store_align(sc, sq0, m1, 1, dist=pl.StoreDist.FIRST_ELEMENT_B32)
    vf.store_align(sc, sq1, m1, 2, dist=pl.StoreDist.FIRST_ELEMENT_B32)
    vf.store_align(sc, sq2, m1, 3, dist=pl.StoreDist.FIRST_ELEMENT_B32)
    for dv in pl.range(D):
        st0 = vf.load_align(state, dv * D)
        st1 = vf.load_align(state, dv * D + DH)
        p0 = vf.mul(st0, dk0a, preg)
        p1 = vf.mul(st1, dk0b, preg)
        sa = vf.add(p0, p1, preg)
        kva = vf.reduce_sum(sa, preg)
        vf.store_align(kv0, kva, m1, dv, dist=pl.StoreDist.FIRST_ELEMENT_B32)
        p2 = vf.mul(st0, ddk1a, preg)
        p3 = vf.mul(st1, ddk1b, preg)
        sb = vf.add(p2, p3, preg)
        kvb = vf.reduce_sum(sb, preg)
        vf.store_align(kv1, kvb, m1, dv, dist=pl.StoreDist.FIRST_ELEMENT_B32)
        p4 = vf.mul(st0, dq0a, preg)
        p5 = vf.mul(st1, dq0b, preg)
        sc0 = vf.add(p4, p5, preg)
        oa = vf.reduce_sum(sc0, preg)
        vf.store_align(op0, oa, m1, dv, dist=pl.StoreDist.FIRST_ELEMENT_B32)
        p6 = vf.mul(st0, ddq1a, preg)
        p7 = vf.mul(st1, ddq1b, preg)
        sd = vf.add(p6, p7, preg)
        ob = vf.reduce_sum(sd, preg)
        vf.store_align(op1, ob, m1, dv, dist=pl.StoreDist.FIRST_ELEMENT_B32)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    skv = vf.load_align(sc, 0, dist=pl.LoadDist.BRC_B32)
    sq0v = vf.load_align(sc, 1, dist=pl.LoadDist.BRC_B32)
    sq1v = vf.load_align(sc, 2, dist=pl.LoadDist.BRC_B32)
    sq2v = vf.load_align(sc, 3, dist=pl.LoadDist.BRC_B32)
    b0 = vf.full(beta0, preg, dtype=pl.DT_FP32)
    b1 = vf.full(beta1, preg, dtype=pl.DT_FP32)
    v0a = vf.load_align(value_t, tb)
    v0b = vf.load_align(value_t, tb + DH)
    v1a = vf.load_align(value_t, tb + D)
    v1b = vf.load_align(value_t, tb + D + DH)
    kv0a = vf.load_align(kv0, 0)
    kv0b = vf.load_align(kv0, DH)
    kv1a = vf.load_align(kv1, 0)
    kv1b = vf.load_align(kv1, DH)
    op0a = vf.load_align(op0, 0)
    op0b = vf.load_align(op0, DH)
    op1a = vf.load_align(op1, 0)
    op1b = vf.load_align(op1, DH)
    r0a = vf.sub(v0a, kv0a, preg)
    r0b = vf.sub(v0b, kv0b, preg)
    dlt0a = vf.mul(r0a, b0, preg)
    dlt0b = vf.mul(r0b, b0, preg)
    c0a = vf.mul(skv, dlt0a, preg)
    c0b = vf.mul(skv, dlt0b, preg)
    r1a = vf.sub(v1a, kv1a, preg)
    r1b = vf.sub(v1b, kv1b, preg)
    r1ca = vf.sub(r1a, c0a, preg)
    r1cb = vf.sub(r1b, c0b, preg)
    dlt1a = vf.mul(r1ca, b1, preg)
    dlt1b = vf.mul(r1cb, b1, preg)
    vf.store_align(delta0, dlt0a, preg, 0)
    vf.store_align(delta0, dlt0b, preg, DH)
    vf.store_align(delta1, dlt1a, preg, 0)
    vf.store_align(delta1, dlt1b, preg, DH)
    o0ta = vf.mul(dlt0a, sq0v, preg)
    o0tb = vf.mul(dlt0b, sq0v, preg)
    o0ua = vf.add(op0a, o0ta, preg)
    o0ub = vf.add(op0b, o0tb, preg)
    o0a = vf.muls(o0ua, scale, preg)
    o0b = vf.muls(o0ub, scale, preg)
    o1ta = vf.mul(dlt0a, sq1v, preg)
    o1tb = vf.mul(dlt0b, sq1v, preg)
    o1ua = vf.add(op1a, o1ta, preg)
    o1ub = vf.add(op1b, o1tb, preg)
    o1va = vf.mul(dlt1a, sq2v, preg)
    o1vb = vf.mul(dlt1b, sq2v, preg)
    o1wa = vf.add(o1ua, o1va, preg)
    o1wb = vf.add(o1ub, o1vb, preg)
    o1a = vf.muls(o1wa, scale, preg)
    o1b = vf.muls(o1wb, scale, preg)
    vf.store_align(o_t, o0a, preg, tb)
    vf.store_align(o_t, o0b, preg, tb + DH)
    vf.store_align(o_t, o1a, preg, tb + D)
    vf.store_align(o_t, o1b, preg, tb + D + DH)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    for dv in pl.range(D):
        dl0 = vf.load_align(delta0, dv, dist=pl.LoadDist.BRC_B32)
        dl1 = vf.load_align(delta1, dv, dist=pl.LoadDist.BRC_B32)
        st0 = vf.load_align(state, dv * D)
        st1 = vf.load_align(state, dv * D + DH)
        q0a = vf.mul(st0, dda, preg)
        q0b = vf.mul(dl0, kda, preg)
        q0c = vf.mul(dl1, k1a, preg)
        q0d = vf.add(q0a, q0b, preg)
        sn0 = vf.add(q0d, q0c, preg)
        q1a = vf.mul(st1, ddb, preg)
        q1b = vf.mul(dl0, kdb, preg)
        q1c = vf.mul(dl1, k1b, preg)
        q1d = vf.add(q1a, q1b, preg)
        sn1 = vf.add(q1d, q1c, preg)
        vf.store_align(state, sn0, preg, dv * D)
        vf.store_align(state, sn1, preg, dv * D + DH)


# ===========================================================================
# Module-3 activation VF helpers (decay activation, RMSNorm + gate, DESIGN.md §1)
# ===========================================================================
@pl.vector_function
def exp_tile_vf(x):
    """In-place elementwise ``exp`` over a flattened FP32 tile (for ``exp(a_log[head])``)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(D // DH):
        v = vf.load_align(x, r * DH)
        e = vf.exp(v, preg)
        vf.store_align(x, e, preg, r * DH)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def decay_vf(d8, dtb, alpha, lower_bound: pl.DT_FP32):
    """``decay = exp(lower_bound * sigmoid(exp(a_log)*(raw_decay + dt_bias)))`` over ``[S,D]``.

    DESIGN.md §1 M3-a.  ``alpha`` holds ``exp(a_log[head])`` (constant over D), ``dtb`` the
    per-head ``dt_bias`` row.  ``sigmoid(u) = 1/(1+exp(-u))`` so ``exp`` never sees a
    positive argument (numerical-safety table, DESIGN.md §1).
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    for m in pl.range(S):
        base = m * D
        for r in pl.range(D // DH):
            off = base + r * DH
            x = vf.load_align(d8, off)
            t = vf.load_align(dtb, r * DH)
            a = vf.load_align(alpha, r * DH)
            s = vf.add(x, t, preg)
            s = vf.mul(s, a, preg)
            s = vf.neg(s, preg)
            e = vf.exp(s, preg)
            dd = vf.adds(e, 1.0, preg)
            iv = vf.div(one, dd, preg)
            s = vf.muls(iv, lower_bound, preg)
            s = vf.exp(s, preg)
            vf.store_align(d8, s, preg, off)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def rmsnorm_gate_vf(o8, gm, sq, r81, rms_eps: pl.DT_FP32, inv_d: pl.DT_FP32):
    """``o8 = rmsnorm(o8)*gamma*sigmoid(sq)`` in place (DESIGN \u00a71 M3-d).

    Row sum-of-squares is broadcast with ``vf.store_align(FIRST_ELEMENT_B32)`` +
    ``vf.load_align(dist=BRC_B32)``.  The
    VST->VLD barriers are hoisted out of the per-row loops (all ``S`` row sums are
    staged into ``r81`` first, then one barrier, then one barrier before the final
    elementwise multiply) so the barrier cost is paid once per pass instead of once
    per row.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    m1 = vf.update_mask(1, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    # ---- pass 1: stage every row's sum-of-squares into r81[m] ----
    for m in pl.range(S):
        base = m * D
        s = vf.full(0.0, preg, dtype=pl.DT_FP32)
        for r in pl.range(D // DH):
            reg = vf.load_align(o8, base + r * DH)
            q = vf.mul(reg, reg, preg)
            part = vf.reduce_sum(q, preg)
            s = vf.add(s, part, preg)
        vf.store_align(r81, s, m1, m, dist=pl.StoreDist.FIRST_ELEMENT_B32)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    # ---- pass 2: rmsnorm * gamma into o8, sigmoid(gate) into sq ----
    for m in pl.range(S):
        base = m * D
        sd = vf.load_align(r81, m, dist=pl.LoadDist.BRC_B32)
        sd = vf.muls(sd, inv_d, preg)
        sd = vf.adds(sd, rms_eps, preg)
        sd = vf.sqrt(sd, preg)
        for r in pl.range(D // DH):
            off = base + r * DH
            reg = vf.load_align(o8, off)
            n = vf.div(reg, sd, preg)
            g = vf.load_align(gm, r * DH)
            n = vf.mul(n, g, preg)
            vf.store_align(o8, n, preg, off)
            gv = vf.load_align(sq, off)
            gn = vf.neg(gv, preg)
            ge = vf.exp(gn, preg)
            gd = vf.adds(ge, 1.0, preg)
            gi = vf.div(one, gd, preg)
            vf.store_align(sq, gi, preg, off)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    # ---- pass 3: elementwise multiply ----
    for r in pl.range((S * D) // DH):
        off = r * DH
        a = vf.load_align(o8, off)
        b = vf.load_align(sq, off)
        o = vf.mul(a, b, preg)
        vf.store_align(o8, o, preg, off)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


def _m3_load_item(b, head, ws_qkv, ws_raw_decay, rec_state2,
                  stage, q8, k8, v8, d8, st):
    """Stage one ``(b, head)`` item: raw q/k/v/decay tiles plus its ``D×D`` checkpoint."""
    row = b * S
    pl.load(stage.current(), ws_qkv, [row, head * D])
    pl.cast(q8.current(), stage.current())
    pl.load(stage.current(), ws_qkv, [row, KEY_PROJ_WIDTH + head * D])
    pl.cast(k8.current(), stage.current())
    pl.load(stage.current(), ws_qkv, [row, 2 * KEY_PROJ_WIDTH + head * D])
    pl.cast(v8.current(), stage.current())
    pl.load(stage.current(), ws_raw_decay, [row, head * D])
    pl.cast(d8.current(), stage.current())
    pl.load(st.current(), rec_state2, [(b * KEY_NUM_HEADS + head) * D, 0])


# ---------------------------------------------------------------------------
# M3 slot protocol, shared by both KEY_PREFETCH variants
#
# ``item = (slot * k_items + _kk) * n_workers + worker``; per slot the AIV runs the
# recurrence, then takes the AIV half of the 3-step protocol:
#   step 1  ``set PUB_VEC`` (MTE3, INTRA_BLOCK) after the last ``ws_gated`` store
#   step 2  ``wait PUB_ACK`` (V, INTRA_BLOCK)
#   step 3  the Cube-side ``sync_all(AIC_ONLY)`` hardware barrier; it is not taken
#           here, which is why M4 overlaps the next slot.
# Publish/ack runs on every slot even when ``item >= total_items``, so both sides
# execute exactly ``slot_count`` rounds.
# ---------------------------------------------------------------------------
def _m3_loop_prefetch(worker, n_workers, k_items, slot_count, total_items, ws_qkv, ws_gate,
                      ws_raw_decay, ws_beta_logit, rec_state2, a_log, dt_bias, gm, ws_gated,
                      replay_u2, replay_decay2, st, t8, t8p, q8, k8, v8, o8, sq, d8, uh,
                      b18, b18f, kvr, dr, kvr1, op0, op1, dr1, scs, r81,
                      alpha, dtb, scale, lower_bound, rms_eps, inv_d,
                      head_major, batch_size):
    """``KEY_PREFETCH=1`` variant: prefetch the next item while the trailing RMSNorm/gate runs.

    The current item's q/k/v/decay are already in ``q8``/``k8``/``v8``/``d8``; the
    prologue stages the worker's first item and each item body stages ``item + n_workers``.
    See the protocol note above.
    """
    # ---- prologue: stage this worker's first item before entering the slot loop ----
    first_on = pl.min(1, pl.max(0, total_items - worker))
    for _f in pl.range(first_on):
        if head_major:
            fb = worker % batch_size
            fh = worker // batch_size
        else:
            fb = worker // KEY_NUM_HEADS
            fh = worker % KEY_NUM_HEADS
        _m3_load_item(fb, fh, ws_qkv, ws_raw_decay, rec_state2, t8p, q8, k8, v8, d8, st)

    for slot in pl.range(slot_count):
        for _kk in pl.range(k_items):
            item = (slot * k_items + _kk) * n_workers + worker
            on = pl.min(1, pl.max(0, total_items - item))
            for _live in pl.range(on):
                if head_major:
                    b = item % batch_size
                    head = item // batch_size
                else:
                    b = item // KEY_NUM_HEADS
                    head = item % KEY_NUM_HEADS
                row = b * S
                col = head * D
                pl.load(dtb.current(), dt_bias, [head, 0])
                pl.expands(alpha.current(), pl.getval(a_log, head))
                exp_tile_vf(alpha.current())
                decay_vf(d8.current(), dtb.current(), alpha.current(), lower_bound)
                pl.store(replay_decay2, d8.current(), [row, col])
                pl.load(b18.current(), ws_beta_logit, [row, 0])
                pl.cast(b18f.current(), b18.current())
                # the gate is only consumed by the trailing RMSNorm; load it early
                pl.load(t8.current(), ws_gate, [row, col])
                pl.cast(sq.current(), t8.current())
                for t2 in pl.range(S // 2):
                    t = t2 * 2
                    blogit0 = pl.getval(b18f.current(), t * KEY_BETA_WIDTH + head)
                    blogit1 = pl.getval(b18f.current(), (t + 1) * KEY_BETA_WIDTH + head)
                    rec_pair_vf(st.current(), d8.current(), k8.current(), q8.current(),
                                v8.current(), o8.current(), kvr.current(), kvr1.current(),
                                op0.current(), op1.current(), dr.current(), dr1.current(),
                                scs.current(), t * D, blogit0, blogit1, scale)
                    pl.cast(uh.current(), dr.current())
                    pl.store(replay_u2, uh.current(), [row + t, head * D])
                    pl.cast(uh.current(), dr1.current())
                    pl.store(replay_u2, uh.current(), [row + t + 1, head * D])
                # ---- prefetch the next item while the trailing RMSNorm/gate runs ----
                nxt = item + n_workers
                nxt_on = pl.min(1, pl.max(0, total_items - nxt))
                for _nx in pl.range(nxt_on):
                    if head_major:
                        nb = nxt % batch_size
                        nh = nxt // batch_size
                    else:
                        nb = nxt // KEY_NUM_HEADS
                        nh = nxt % KEY_NUM_HEADS
                    _m3_load_item(nb, nh, ws_qkv, ws_raw_decay, rec_state2,
                                  t8p, q8, k8, v8, d8, st)
                rmsnorm_gate_vf(o8.current(), gm.current(), sq.current(), r81.current(),
                                rms_eps, inv_d)
                pl.cast(t8.current(), o8.current())
                pl.store(ws_gated, t8.current(), [row, col])
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=PUB_VEC,
                                 sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PUB_ACK,
                                  sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)


def _m3_loop_inline(worker, n_workers, k_items, slot_count, total_items, ws_qkv, ws_gate,
                    ws_raw_decay, ws_beta_logit, rec_state2, a_log, dt_bias, gm, ws_gated,
                    replay_u2, replay_decay2, st, t8, q8, k8, v8, o8, sq, d8, uh,
                    b18, b18f, kvr, dr, r81,
                    alpha, dtb, scale, lower_bound, rms_eps, inv_d):
    """``KEY_PREFETCH=0`` variant: per-item inline loads in batch-major item order.

    q/k/v/decay are staged through ``t8`` immediately before the recurrence and the gate
    is loaded after it.  See the protocol note above.
    """
    for slot in pl.range(slot_count):
        for _kk in pl.range(k_items):
            item = (slot * k_items + _kk) * n_workers + worker
            on = pl.min(1, pl.max(0, total_items - item))
            for _live in pl.range(on):
                b = item // KEY_NUM_HEADS
                head = item % KEY_NUM_HEADS
                row = b * S
                col = head * D
                _m3_load_item(b, head, ws_qkv, ws_raw_decay, rec_state2,
                              t8, q8, k8, v8, d8, st)
                pl.load(dtb.current(), dt_bias, [head, 0])
                pl.expands(alpha.current(), pl.getval(a_log, head))
                exp_tile_vf(alpha.current())
                decay_vf(d8.current(), dtb.current(), alpha.current(), lower_bound)
                pl.store(replay_decay2, d8.current(), [row, col])
                pl.load(b18.current(), ws_beta_logit, [row, 0])
                pl.cast(b18f.current(), b18.current())
                for t in pl.range(S):
                    blogit = pl.getval(b18f.current(), t * KEY_BETA_WIDTH + head)
                    rec_token_vf2(st.current(), d8.current(), k8.current(), q8.current(),
                                  v8.current(), o8.current(), kvr.current(), dr.current(),
                                  t * D, blogit, scale)
                    pl.cast(uh.current(), dr.current())
                    pl.store(replay_u2, uh.current(), [row + t, head * D])
                pl.load(t8.current(), ws_gate, [row, col])
                pl.cast(sq.current(), t8.current())
                rmsnorm_gate_vf(o8.current(), gm.current(), sq.current(), r81.current(),
                                rms_eps, inv_d)
                pl.cast(t8.current(), o8.current())
                pl.store(ws_gated, t8.current(), [row, col])
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=PUB_VEC,
                                 sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PUB_ACK,
                                  sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)


def _aux_beta(x2, wbeta, ws_beta_logit, m_group, x_b, w_beta, w_beta_zn, l0a, l0b_beta, l0c_beta):
    """β = X @ W_beta^T (BN = ``KEY_BETA_WIDTH``), the second-to-last loop-A task kind."""
    _aux_gemm(x2, wbeta, ws_beta_logit, m_group, 0, K_STEPS, KEY_BETA_WIDTH,
              x_b, w_beta, w_beta_zn, l0a, l0b_beta, l0c_beta)


def _aux_fa(x2, wa, ws_decay_a, m_group, x_b, w_fa, w_fa_zn, l0a, l0b, l0c0):
    """decay-A = X @ W_a^T (BN = ``TILE``), the last loop-A task kind."""
    _aux_gemm(x2, wa, ws_decay_a, m_group, 0, K_STEPS, TILE,
              x_b, w_fa, w_fa_zn, l0a, l0b, l0c0)


def _aux_split(a, w, out, mt, part, k_tiles, bn, xa, wn, wz, l0a, l0b, acc):
    """One K-slice of an aux projection, atomically accumulated into ``out``."""
    per = k_tiles // KEY_K_SPLIT
    for kk in pl.range(part * per, (part + 1) * per):
        am = xa.next()
        pl.load(am, a, [mt * KEY_M_TILE_ROWS, kk * TILE])
        wq = wn.next()
        pl.load(wq, w, [0, kk * TILE])
        al = l0a.next()
        pl.move(al, am)
        bl = l0b.next()
        pl.move(bl, wz.next())
        if kk == part * per:
            pl.matmul(acc.current(), al, bl)
        else:
            pl.matmul_acc(acc.current(), acc.current(), al, bl)
    pl.set_validshape(acc.current(), [KEY_M_TILE_ROWS, bn])
    pl.store(out, acc.current(), [mt * KEY_M_TILE_ROWS, 0], atomic=pl.AtomicType.AtomicAdd)


def _aux_task(aux, m_group, x2, wbeta, wa, ws_beta_logit, ws_decay_a,
              x_b, w_beta, w_beta_zn, w_fa, w_fa_zn, l0a, l0b, l0b_beta, l0c_beta, l0c0):
    """Dispatch aux task ``aux`` (β first, then decay-A) for the ``KEY_K_SPLIT`` granularity.

    ``KEY_K_SPLIT == 1`` runs each projection as one whole-K task; ``KEY_K_SPLIT == 8`` splits it into
    eight K-slices that atomically accumulate into the same workspace tile.
    """
    if KEY_K_SPLIT == 1:
        if aux == 0:
            _aux_beta(x2, wbeta, ws_beta_logit, m_group, x_b, w_beta, w_beta_zn,
                      l0a, l0b_beta, l0c_beta)
        else:
            _aux_fa(x2, wa, ws_decay_a, m_group, x_b, w_fa, w_fa_zn, l0a, l0b, l0c0)
    elif aux < KEY_K_SPLIT:
        _aux_split(x2, wbeta, ws_beta_logit, m_group, aux, K_STEPS, KEY_BETA_WIDTH,
                   x_b, w_beta, w_beta_zn, l0a, l0b_beta, l0c_beta)
    else:
        _aux_split(x2, wa, ws_decay_a, m_group, aux - KEY_K_SPLIT, K_STEPS, TILE,
                   x_b, w_fa, w_fa_zn, l0a, l0b, l0c0)


def _loopa_n384(core_id, num_cores, n_groups, x2, wqkv, wgate, wbeta, wa,
                ws_beta_logit, ws_decay_a, x_l1, w_l1, w_zn, l0a, l0b,
                l0c0, l0c1, l0c2, panel_h0, panel_h1, panel_h2,
                x_b, w_beta, w_beta_zn, w_fa, w_fa_zn, l0b_beta, l0c_beta):
    """N384 tier loop A: ``KEY_QKV_PANELS`` qkv + ``KEY_GATE_PANELS`` gate panels + β + Fa.

    One grid-stride over ``KEY_PANEL_TASKS + 2*KEY_K_SPLIT`` tasks per group.
    """
    for wi in pl.range(core_id, n_groups * (KEY_PANEL_TASKS + 2 * KEY_K_SPLIT), num_cores):
        nd = wi // n_groups
        m_group = wi % n_groups
        task_ord = (wi - core_id) // num_cores
        if nd < KEY_QKV_PANELS:
            _n384_task(wqkv, nd * PANEL_N, m_group, task_ord, x2,
                       x_l1, w_l1, w_zn, l0a, l0b, l0c0, l0c1, l0c2,
                       panel_h0, panel_h1, panel_h2)
        elif nd < KEY_QKV_PANELS + KEY_GATE_PANELS:
            _n384_task(wgate, (nd - KEY_QKV_PANELS) * PANEL_N, m_group, task_ord, x2,
                       x_l1, w_l1, w_zn, l0a, l0b, l0c0, l0c1, l0c2,
                       panel_h0, panel_h1, panel_h2)
        else:
            _aux_task(nd - (KEY_QKV_PANELS + KEY_GATE_PANELS), m_group, x2, wbeta, wa,
                      ws_beta_logit, ws_decay_a, x_b, w_beta, w_beta_zn,
                      w_fa, w_fa_zn, l0a, l0b, l0b_beta, l0c_beta, l0c0)


def _loopa_n128(core_id, num_cores, n_groups, x2, wqkv, wgate, wbeta, wa,
                ws_beta_logit, ws_decay_a, x_l1, w_l1n, w_znn, l0a, l0b,
                l0c0, panel_h0, x_b, w_beta, w_beta_zn, w_fa, w_fa_zn,
                l0b_beta, l0c_beta):
    """N128 tier loop A: one head per task (``3*KEY_NUM_HEADS`` qkv + ``KEY_NUM_HEADS`` gate) + β + Fa.

    One grid-stride over ``4*KEY_NUM_HEADS + 2*KEY_K_SPLIT`` tasks per group.
    """
    for wi in pl.range(core_id, n_groups * (4 * KEY_NUM_HEADS + 2 * KEY_K_SPLIT), num_cores):
        nd = wi // n_groups
        m_group = wi % n_groups
        task_ord = (wi - core_id) // num_cores
        if nd < 3 * KEY_NUM_HEADS:
            _n128_task(wqkv, nd * TILE, m_group, task_ord, x2,
                       x_l1, w_l1n, w_znn, l0a, l0b, l0c0, panel_h0)
        elif nd < 4 * KEY_NUM_HEADS:
            _n128_task(wgate, (nd - 3 * KEY_NUM_HEADS) * TILE, m_group, task_ord, x2,
                       x_l1, w_l1n, w_znn, l0a, l0b, l0c0, panel_h0)
        else:
            _aux_task(nd - 4 * KEY_NUM_HEADS, m_group, x2, wbeta, wa,
                      ws_beta_logit, ws_decay_a, x_b, w_beta, w_beta_zn,
                      w_fa, w_fa_zn, l0a, l0b, l0b_beta, l0c_beta, l0c0)


def _prepost_release():
    """Vector pre-authorises all 6 cross-core panel slots (initial writable state)."""
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=RELEASE_IDS[0])
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=RELEASE_IDS[1])
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=RELEASE_IDS[2])
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=RELEASE_IDS[3])
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=RELEASE_IDS[4])
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=RELEASE_IDS[5])


def _make_panels():
    """Cross-core shared panel TileGroups, declared outside both sections (as FA does)."""
    panel_h0 = pl.make_tile_group(
        type=pl.TileType(shape=[KEY_M_TILE_ROWS_HALF, TILE], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=0x00000, mutex_ids=[11, 12])
    panel_h1 = pl.make_tile_group(
        type=pl.TileType(shape=[KEY_M_TILE_ROWS_HALF, TILE], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=0x10000, mutex_ids=[13, 14])
    panel_h2 = pl.make_tile_group(
        type=pl.TileType(shape=[KEY_M_TILE_ROWS_HALF, TILE], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=0x20000, mutex_ids=[15, 16])
    return panel_h0, panel_h1, panel_h2


def _module1_cube(core_id, num_cores, n_groups, use_n384, x2, wqkv, wgate, wbeta, wa, wb,
                  ws_beta_logit, ws_decay_a, ws_raw_decay, panel_h0, panel_h1, panel_h2):
    """Module 1 — projection frontend (Cube, ``loop A``).

    Flat grid-stride over ``n_groups * (KEY_PANEL_TASKS + 2*KEY_K_SPLIT)`` tasks: qkv/gate N384 panels
    (3 heads each) plus β and decay-A (split into ``KEY_K_SPLIT`` K-slices, BF16 atomic add).
    Panel-level Cube→AIV handoff (``DualModeSplitM`` + INTRA_BLOCK READY/RELEASE) and
    a 6 ``(head, slot)`` drain; ``sync_all(AIC_ONLY)`` + decay-B is the only serial
    AIC step.
    """
    # ---- phase A tiles (DESIGN.md §3.2): L1 fully double-buffered ----
    x_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[KEY_M_TILE_ROWS, KC], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x00000, mutex_ids=[0, 1])
    w_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[PANEL_N, KC], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x20000, mutex_ids=[2, 3])
    w_zn = pl.make_tile_group(
        type=pl.TileType(shape=[KC, PANEL_N], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x20000, mutex_ids=[2, 3])
    # N128 degenerate tier: one-head weight panel, same L1 region/mutex.
    w_l1n = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, KC], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x20000, mutex_ids=[2, 3])
    w_znn = pl.make_tile_group(
        type=pl.TileType(shape=[KC, TILE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x20000, mutex_ids=[2, 3])
    l0a = pl.make_tile_group(
        type=pl.TileType(shape=[KEY_M_TILE_ROWS, KS], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0, mutex_ids=[4, 5])
    l0b = pl.make_tile_group(
        type=pl.TileType(shape=[KS, TILE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0, mutex_ids=[6, 7])
    l0c0 = pl.make_tile_group(
        type=pl.TileType(shape=[KEY_M_TILE_ROWS, TILE], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
        addrs=0x0, mutex_ids=[8])
    l0c1 = pl.make_tile_group(
        type=pl.TileType(shape=[KEY_M_TILE_ROWS, TILE], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
        addrs=0x10000, mutex_ids=[9])
    l0c2 = pl.make_tile_group(
        type=pl.TileType(shape=[KEY_M_TILE_ROWS, TILE], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
        addrs=0x20000, mutex_ids=[10])

    # ---- aux tiles (DESIGN.md §3.2 aliases: the note permits reusing the
    # x_l1/w_l1 double-slot regions with the same mutex ids, so the aux loops use
    # the same two slots -> the phase-A 0x80000 peak is unchanged) ----
    x_b = pl.make_tile_group(
        type=pl.TileType(shape=[KEY_M_TILE_ROWS, TILE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[0x00000, 0x10000], mutex_ids=[0, 1])
    w_beta = pl.make_tile_group(
        type=pl.TileType(shape=[KEY_BETA_WIDTH, TILE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[0x20000, 0x50000], mutex_ids=[2, 3])
    w_beta_zn = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, KEY_BETA_WIDTH], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x20000, 0x50000], mutex_ids=[2, 3])
    w_fa = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[0x20000, 0x50000], mutex_ids=[2, 3])
    w_fa_zn = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x20000, 0x50000], mutex_ids=[2, 3])
    o_da_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[KEY_M_TILE_ROWS, TILE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[0x00000, 0x10000], mutex_ids=[0, 1])
    w_fb = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[0x20000, 0x50000], mutex_ids=[2, 3])
    w_fb_zn = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x20000, 0x50000], mutex_ids=[2, 3])
    l0b_beta = pl.make_tile_group(
        type=pl.TileType(shape=[KS, KEY_BETA_WIDTH], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=[0x0, 0x8000], mutex_ids=[6, 7])
    l0c_beta = pl.make_tile_group(
        type=pl.TileType(shape=[KEY_M_TILE_ROWS, KEY_BETA_WIDTH], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
        addrs=0x0, mutex_ids=[8])

    # ---- loop A: ONE grid-stride covering qkv/gate panels + β + Fa (DESIGN.md §1) ----
    if use_n384:
        _loopa_n384(core_id, num_cores, n_groups, x2, wqkv, wgate, wbeta, wa,
                    ws_beta_logit, ws_decay_a, x_l1, w_l1, w_zn, l0a, l0b,
                    l0c0, l0c1, l0c2, panel_h0, panel_h1, panel_h2,
                    x_b, w_beta, w_beta_zn, w_fa, w_fa_zn, l0b_beta, l0c_beta)
    else:
        _loopa_n128(core_id, num_cores, n_groups, x2, wqkv, wgate, wbeta, wa,
                    ws_beta_logit, ws_decay_a, x_l1, w_l1n, w_znn, l0a, l0b,
                    l0c0, panel_h0, x_b, w_beta, w_beta_zn, w_fa, w_fa_zn,
                    l0b_beta, l0c_beta)

    # ---- drain: confirm the last batch of all 6 (head, slot) pairs was consumed ----
    for h in pl.range(PANEL_HEADS):
        for s in pl.range(PANEL_DEPTH):
            pl.system.wait_cross_core(
                pipe=pl.PipeType.FIX, event_id=RELEASE_IDS[h * PANEL_DEPTH + s])

    # ---- the single remaining AIC_ONLY step, then loop B2 (decay-B) ----
    pl.system.sync_all(core_type=pl.SyncCoreType.AIC_ONLY)

    for wi in pl.range(core_id, n_groups * KEY_NUM_HEADS, num_cores):
        _aux_gemm(ws_decay_a, wb, ws_raw_decay, wi // KEY_NUM_HEADS, wi % KEY_NUM_HEADS, D // TILE, TILE,
                  o_da_l1, w_fb, w_fb_zn, l0a, l0b, l0c0)


def _m4_slot_sync():
    """Wait for one M3 slot, acknowledge it, then take the all-AIC barrier."""
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=PUB_VEC,
                              sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PUB_ACK,
                             sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
    pl.system.sync_all(core_type=pl.SyncCoreType.AIC_ONLY)


def _m4_ready_range(slot, slot_cap, total_items, chunk_items):
    """Chunk range whose M3 items are published after ``slot``'s barrier."""
    completed = pl.min((slot + 1) * slot_cap, total_items)
    ready_before = pl.min(slot * slot_cap, total_items) // chunk_items
    return ready_before, completed // chunk_items


def _module4_cube(core_id, num_cores, n_groups, batch, ws_gated, wout, out):
    """Module 4 — output projection (Cube).

    Per slot after the 3-step handshake (wait ``PUB_VEC`` → set ``PUB_ACK`` →
    ``sync_all(AIC_ONLY)``): chunked ``out = ws_gated @ W_out^T`` over whole m-tiles,
    ``BN = 128`` for ``num_cores <= 8`` else 256, FP32 L0C accumulation, BF16 ``[B, S, H]``
    stored through a ``make_tensor([B*S, H], [H, 1])`` view of the public rank-3 ``out``.
    ``W_out`` is FRACTAL_NZ ``[H, KEY_PROJ_WIDTH]``, loaded NZ→NZ and re-read through its ZN alias as
    the transposed right operand.
    """
    # ---- phase 3 backend: M4 — output projection, flat grid-stride over (m_group, n_tile) ----
    out2 = pl.make_tensor(out, [batch * S, H], [H, 1])
    a_o = pl.make_tile_group(
        type=pl.TileType(shape=[KEY_M_TILE_ROWS, TILE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x30000, mutex_ids=[17, 18])
    b_nz_a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x40000, mutex_ids=[19, 20])
    b_zn_a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x40000, mutex_ids=[19, 20])
    a_l0_a = pl.make_tile_group(
        type=pl.TileType(shape=[KEY_M_TILE_ROWS, TILE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0, mutex_ids=[21, 22])
    b_l0_a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0, mutex_ids=[23, 24])
    acc_a = pl.make_tile_group(
        type=pl.TileType(shape=[KEY_M_TILE_ROWS, TILE], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
        addrs=0x0, mutex_ids=[25])
    a_o_b = pl.make_tile_group(
        type=pl.TileType(shape=[KEY_M_TILE_ROWS, TILE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x30000, mutex_ids=[17, 18])
    b_nz_b = pl.make_tile_group(
        type=pl.TileType(shape=[256, TILE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x40000, mutex_ids=[19, 20])
    b_zn_b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, 256], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x40000, mutex_ids=[19, 20])
    a_l0_b = pl.make_tile_group(
        type=pl.TileType(shape=[KEY_M_TILE_ROWS, TILE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0, mutex_ids=[26, 27])
    b_l0_b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, 256], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0, mutex_ids=[28])
    acc_b = pl.make_tile_group(
        type=pl.TileType(shape=[KEY_M_TILE_ROWS, 256], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
        addrs=0x20000, mutex_ids=[29])
    # Resident output-weight panel for N=6 at >=28 AIC: each AIC keeps its
    # 256-column Wout shard (256 x KEY_PROJ_WIDTH BF16 = 384 KiB at KEY_PROJ_WIDTH=768) in L1 and
    # reuses it across every M4 chunk instead of re-reading it from L2.
    wout_res = pl.make_tile_group(
        type=pl.TileType(shape=[256, 6 * TILE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x00000, mutex_ids=[30])
    wout_res_zn = pl.make_tile_group(
        type=pl.TileType(shape=[6 * TILE, 256], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x00000, mutex_ids=[30])
    a_o_res = pl.make_tile_group(
        type=pl.TileType(shape=[KEY_M_TILE_ROWS, TILE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[0x60000, 0x68000], mutex_ids=[17, 18])
    # ---- phase 3 backend: M4 — slot loop; step 3 = hardware all-AIC barrier ----
    # step 1: wait PUB_VEC(3)/INTRA_BLOCK   step 2: set PUB_ACK(4)/INTRA_BLOCK
    # step 3: sync_all(AIC_ONLY) = the all-AIC hardware barrier; it makes the
    #         grid-wide `completed` ledger true.  M4 projects only after it.
    n_workers = num_cores * 2
    total_items = batch * KEY_NUM_HEADS
    chunk_items = KEY_BATCH_PER_GROUP * KEY_NUM_HEADS
    slot_cap = KEY_M3_K * n_workers
    slot_count = (total_items + slot_cap - 1) // slot_cap
    if num_cores <= 8:
        for slot in pl.range(slot_count):
            _m4_slot_sync()
            ready_before, ready_after = _m4_ready_range(
                slot, slot_cap, total_items, chunk_items)
            for chunk in pl.range(ready_before, ready_after):
                for nt in pl.range(core_id, H // 128, num_cores):
                    _aux_gemm(ws_gated, wout, out2, chunk, nt, KEY_PROJ_WIDTH // TILE, 128,
                              a_o, b_nz_a, b_zn_a, a_l0_a, b_l0_a, acc_a)
    else:
        use_res_wout = (KEY_NUM_HEADS == 6) and (num_cores >= 28)
        if use_res_wout:
            if core_id < H // 256:
                pl.load(wout_res.current(), wout, [core_id * 256, 0])
        one_group = (batch // KEY_BATCH_PER_GROUP == 1)
        if one_group:
            # One m-tile == the whole batch: publish head ranges after every M3 slot
            # and accumulate the output projection in L0C across slots, so M4 overlaps
            # the next slot's recurrence instead of waiting for the full tile.
            for slot in pl.range(slot_count):
                _m4_slot_sync()
                completed_before = pl.min(slot * slot_cap, total_items)
                completed = pl.min((slot + 1) * slot_cap, total_items)
                head_lo = completed_before // batch
                head_hi = completed // batch
                for nt in pl.range(core_id, H // 256, num_cores):
                    for h in pl.range(head_lo, head_hi):
                        if use_res_wout:
                            _aux_gemm_head_res(ws_gated, wout_res_zn, h, h == 0,
                                               a_o_res, a_l0_b, b_l0_b, acc_b)
                        else:
                            _aux_gemm_head(ws_gated, wout, nt, h, h == 0,
                                           a_o_b, b_nz_b, b_zn_b, a_l0_b, b_l0_b, acc_b)
                    if head_hi == KEY_NUM_HEADS:
                        pl.set_validshape(acc_b.current(), [KEY_M_TILE_ROWS, 256])
                        pl.store(out2, acc_b.current(), [0, nt * 256])
        else:
            for slot in pl.range(slot_count):
                _m4_slot_sync()
                ready_before, ready_after = _m4_ready_range(
                    slot, slot_cap, total_items, chunk_items)
                for chunk in pl.range(ready_before, ready_after):
                    if use_res_wout:
                        for nt in pl.range(core_id, H // 256, num_cores):
                            _aux_gemm_resident(ws_gated, wout_res_zn, out2, chunk, nt,
                                               KEY_PROJ_WIDTH // TILE, 256, a_o_res,
                                               a_l0_b, b_l0_b, acc_b)
                    else:
                        for nt in pl.range(core_id, H // 256, num_cores):
                            _aux_gemm(ws_gated, wout, out2, chunk, nt, KEY_PROJ_WIDTH // TILE, 256,
                                      a_o_b, b_nz_b, b_zn_b, a_l0_b, b_l0_b, acc_b)


def _zero_aux_workspaces(block_id, num_cores, sub, ws_beta_logit, ws_decay_a):
    """Zero the split-K aux buffers from inside the kernel (AIV grid-stride).

    M1 accumulates ``ws_beta_logit`` / ``ws_decay_a`` with ``AtomicAdd``, so both must
    start at zero.  Doing it here instead of with a host-side ``.zero_()`` avoids an
    extra small-op launch per call.  The caller must publish this pass with a
    grid-wide ``sync_all(MIX)`` before ``_module1_cube`` runs loop A: the K-slices of
    one output tile are spread over different AIC blocks, so a per-block sync is not
    enough.

    ``rows = B * S`` is always a multiple of ``S``, so the ``[S, *]`` tiles cover the
    whole buffer without a tail path.
    """
    zd = pl.make_tile_group(
        type=pl.TileType(shape=[S, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x38000, mutex_ids="auto", depth=1)
    zb = pl.make_tile_group(
        type=pl.TileType(shape=[S, KEY_BETA_WIDTH], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=0x38800, mutex_ids="auto", depth=1)
    subblocks = pl.get_subblock_num()
    worker = block_id * subblocks + sub
    workers = num_cores * subblocks
    rows = ws_decay_a.shape[0]          # = B * S, always a multiple of S
    for r in pl.range(worker * S, rows, workers * S):
        pl.expands(zd.current(), 0.0)
        pl.store(ws_decay_a, zd.current(), [r, 0])
        pl.expands(zb.current(), 0.0)
        pl.store(ws_beta_logit, zb.current(), [r, 0])


def _module2_vector(block_id, num_cores, n_groups, sub, use_n384,
                    conv_w, conv_layout, conv_state, conv_indices, conv_accepted,
                    panel_h0, panel_h1, panel_h2, ws_qkv, ws_gate, replay_k):
    """Module 2 — conv/norm consume (Vector).

    Same panel table, grid-stride: causal Conv1D(K=4)+SiLU, Q/K L2 norm (eps 1e-6),
    ``ws_qkv`` / ``ws_gate`` / ``replay_k``, in-place pooled ``conv_state`` slide.
    """
    rk2 = pl.make_tensor(replay_k, [ws_qkv.shape[0], KEY_NUM_HEADS * D], [KEY_NUM_HEADS * D, 1])

    pf = pl.make_tile_group(
        type=pl.TileType(shape=[S, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x30000, mutex_ids=[20])
    fb = pl.make_tile_group(
        type=pl.TileType(shape=[CONV_K, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x31000, mutex_ids=[21])
    filt = pl.make_tile_group(
        type=pl.TileType(shape=[CONV_K, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x31400, mutex_ids=[22])
    h3b = pl.make_tile_group(
        type=pl.TileType(shape=[CONV_K - 1, D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=0x31C00, mutex_ids=[23])
    h3 = pl.make_tile_group(
        type=pl.TileType(shape=[CONV_K - 1, D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=0x31F00, mutex_ids=[24])
    r8b = pl.make_tile_group(
        type=pl.TileType(shape=[S, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x32500, mutex_ids=[25])
    r8 = pl.make_tile_group(
        type=pl.TileType(shape=[S, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x32D00, mutex_ids=[26])
    seq = pl.make_tile_group(
        type=pl.TileType(shape=[S + CONV_K, D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=0x33D00, mutex_ids=[27])
    sc = pl.make_tile_group(
        type=pl.TileType(shape=[S, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x35500, mutex_ids=[28])
    ob = pl.make_tile_group(
        type=pl.TileType(shape=[S, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x36500, mutex_ids=[29])
    sv8 = pl.make_tile_group(
        type=pl.TileType(shape=[S, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x36D00, mutex_ids=[30])
    bh = pl.make_tile_group(
        type=pl.TileType(shape=[S, D], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x37500, mutex_ids=[31])

    _prepost_release()

    if use_n384:
        _m2_loop(block_id, num_cores, n_groups, KEY_PANEL_TASKS, 1, sub, conv_w, conv_layout,
                 conv_state, conv_indices, conv_accepted, panel_h0, panel_h1, panel_h2,
                 ws_qkv, ws_gate, rk2,
                 pf, fb, filt, h3b, h3, r8b, r8, seq, sc, ob, sv8, bh)
    else:
        _m2_loop(block_id, num_cores, n_groups, KEY_N128_TASKS, 0, sub, conv_w, conv_layout,
                 conv_state, conv_indices, conv_accepted, panel_h0, panel_h1, panel_h2,
                 ws_qkv, ws_gate, rk2,
                 pf, fb, filt, h3b, h3, r8b, r8, seq, sc, ob, sv8, bh)


def _module3_vector(num_cores, n_groups, ws_qkv, ws_gate, ws_raw_decay,
                    ws_beta_logit, rec_state, a_log, dt_bias, gamma, ws_gated,
                    replay_u, replay_decay, scale, lower_bound, rms_eps):
    """Module 3 — decay/beta recurrent gate (Vector).

    After ``sync_all(MIX)``, one ``(batch, head)`` item per AIV in ``(slot, k, worker)``
    order: decay activation, beta + delta-rule recurrence (``rec_pair_vf`` for the default
    two-token pass, ``rec_token_vf2`` for the inline variant), RMSNorm + sigmoid gate →
    ``ws_gated`` / ``replay_u`` / ``replay_decay``.
    """
    # ---- Module 3 tiles: reuse the panel region (freed by barrier #1, DESIGN.md §3.2) ----
    st = pl.make_tile_group(
        type=pl.TileType(shape=[D, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x00000, mutex_ids=[0])
    q8 = pl.make_tile_group(
        type=pl.TileType(shape=[S, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x10000, mutex_ids=[1])
    k8 = pl.make_tile_group(
        type=pl.TileType(shape=[S, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x11000, mutex_ids=[2])
    v8 = pl.make_tile_group(
        type=pl.TileType(shape=[S, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x12000, mutex_ids=[3])
    o8 = pl.make_tile_group(
        type=pl.TileType(shape=[S, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x13000, mutex_ids=[4])
    sq = pl.make_tile_group(
        type=pl.TileType(shape=[S, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x14000, mutex_ids=[5])
    d8 = pl.make_tile_group(
        type=pl.TileType(shape=[S, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x15000, mutex_ids=[6])
    t8 = pl.make_tile_group(
        type=pl.TileType(shape=[S, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x16000, mutex_ids=[7])
    t8p = pl.make_tile_group(
        type=pl.TileType(shape=[S, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x18600, mutex_ids=[20])
    kvr1 = pl.make_tile_group(
        type=pl.TileType(shape=[1, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x18E00, mutex_ids=[21])
    op0 = pl.make_tile_group(
        type=pl.TileType(shape=[1, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x19000, mutex_ids=[22])
    op1 = pl.make_tile_group(
        type=pl.TileType(shape=[1, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x19200, mutex_ids=[23])
    dr1 = pl.make_tile_group(
        type=pl.TileType(shape=[1, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x19400, mutex_ids=[24])
    scs = pl.make_tile_group(
        type=pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x19600, mutex_ids=[25])
    uh = pl.make_tile_group(
        type=pl.TileType(shape=[1, D], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x17000, mutex_ids=[8])
    b18 = pl.make_tile_group(
        type=pl.TileType(shape=[S, KEY_BETA_WIDTH], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x17200, mutex_ids=[9])
    b18f = pl.make_tile_group(
        type=pl.TileType(shape=[S, KEY_BETA_WIDTH], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x17400, mutex_ids=[10])
    kvr = pl.make_tile_group(
        type=pl.TileType(shape=[1, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x17800, mutex_ids=[17])
    dr = pl.make_tile_group(
        type=pl.TileType(shape=[1, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x17A00, mutex_ids=[18])
    alpha = pl.make_tile_group(
        type=pl.TileType(shape=[1, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x17C00, mutex_ids=[19])
    dtb = pl.make_tile_group(
        type=pl.TileType(shape=[1, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x17E00, mutex_ids=[11])
    r81 = pl.make_tile_group(
        type=pl.TileType(shape=[S, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                         layout=pl.DN),
        addrs=0x18000, mutex_ids=[12])
    gm = pl.make_tile_group(
        type=pl.TileType(shape=[1, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x18200, mutex_ids=[13])

    gm1 = pl.make_tensor(gamma, [1, D], [D, 1])
    ru2 = pl.make_tensor(replay_u, [ws_qkv.shape[0], KEY_NUM_HEADS * D], [KEY_NUM_HEADS * D, 1])
    rd2 = pl.make_tensor(replay_decay, [ws_qkv.shape[0], KEY_NUM_HEADS * D], [KEY_NUM_HEADS * D, 1])
    rs2 = pl.make_tensor(rec_state, [ws_qkv.shape[0] // S * KEY_NUM_HEADS * D, D], [D, 1])
    pl.load(gm.current(), gm1, [0, 0])

    m3_workers = num_cores * pl.get_subblock_num()
    m3_total = ws_qkv.shape[0] // S * KEY_NUM_HEADS
    m3_cap = KEY_M3_K * m3_workers
    m3_slot_count = (m3_total + m3_cap - 1) // m3_cap
    if KEY_PREFETCH == 1:
        _m3_loop_prefetch(pl.get_block_idx(), m3_workers, KEY_M3_K, m3_slot_count, m3_total,
                          ws_qkv, ws_gate, ws_raw_decay, ws_beta_logit,
                          rs2, a_log, dt_bias, gm, ws_gated, ru2, rd2,
                          st, t8, t8p, q8, k8, v8, o8, sq, d8, uh, b18, b18f, kvr, dr,
                          kvr1, op0, op1, dr1, scs, r81,
                          alpha, dtb, scale, lower_bound, rms_eps, 1.0 / D,
                          (n_groups == 1) and (num_cores > 8), ws_qkv.shape[0] // S)
    else:
        _m3_loop_inline(pl.get_block_idx(), m3_workers, KEY_M3_K, m3_slot_count, m3_total,
                        ws_qkv, ws_gate, ws_raw_decay, ws_beta_logit,
                        rs2, a_log, dt_bias, gm, ws_gated, ru2, rd2,
                        st, t8, q8, k8, v8, o8, sq, d8, uh, b18, b18f, kvr, dr, r81,
                        alpha, dtb, scale, lower_bound, rms_eps, 1.0 / D)


# ===========================================================================
# Single fused MIX kernel (Module 1 + 2 + 3 + 4)
# ===========================================================================
# Tensor names keep the descriptive local spelling (``wgate`` / ``wout``,
# ``conv_w`` / ``conv_state``, ``a_log`` / ``dt_bias`` / ``rec_state``,
# ``lower_bound`` / ``rms_eps``); only ``gamma`` keeps that spelling from the
# reference kernel.
# PyPTO-specific extras: the ``ws_*`` workspaces and ``conv_layout``; the slot
# granularity is carried by the ``KEY_M3_K`` tiling key.
@pl.jit(auto_mutex=True, tiling_key=GeometryKey, compile_timeout=1200)
def mega_recurrent_kda_replayssm_kernel(
    x: pl.Tensor[[pl.DYNAMIC, S, H], pl.DT_BF16],
    wqkv: pl.Tensor[[pl.STATIC, H], pl.DT_BF16, pl.NZ],
    wa: pl.Tensor[[D, H], pl.DT_BF16, pl.NZ],
    wb: pl.Tensor[[pl.STATIC, D], pl.DT_BF16, pl.NZ],
    wbeta: pl.Tensor[[pl.STATIC, H], pl.DT_BF16, pl.NZ],
    wgate: pl.Tensor[[pl.STATIC, H], pl.DT_BF16, pl.NZ],
    wout: pl.Tensor[[H, pl.STATIC], pl.DT_BF16, pl.NZ],
    ws_beta_logit: pl.Tensor[[pl.DYNAMIC, pl.STATIC], pl.DT_BF16],
    ws_decay_a: pl.Tensor[[pl.DYNAMIC, D], pl.DT_BF16],
    ws_raw_decay: pl.Tensor[[pl.DYNAMIC, pl.STATIC], pl.DT_BF16],
    conv_w: pl.Ptr[pl.DT_BF16],
    conv_layout: pl.DT_INT32,
    conv_state: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.STATIC], pl.DT_BF16],
    conv_indices: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    conv_accepted: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    ws_qkv: pl.Tensor[[pl.DYNAMIC, pl.STATIC], pl.DT_BF16],
    ws_gate: pl.Tensor[[pl.DYNAMIC, pl.STATIC], pl.DT_BF16],
    replay_k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    ws_gated: pl.Tensor[[pl.DYNAMIC, pl.STATIC], pl.DT_BF16],
    rec_state: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    a_log: pl.Tensor[[pl.STATIC], pl.DT_FP32],
    dt_bias: pl.Tensor[[pl.STATIC, D], pl.DT_FP32],
    gamma: pl.Tensor[[D], pl.DT_FP32],
    replay_u: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    replay_decay: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    scale: pl.DT_FP32,
    lower_bound: pl.DT_FP32,
    rms_eps: pl.DT_FP32,
    out: pl.Tensor[[pl.DYNAMIC, S, H], pl.DT_BF16],
):
    """Module 1 + Module 2 + Module 3 + Module 4 in one launch.

    Two phase ``sync_all(MIX)`` barriers separate M1/M2 from M4/M3; when split-K is
    enabled one extra init barrier publishes the in-kernel zeroing of the aux
    workspaces (see ``_zero_aux_workspaces``).
    """
    panel_h0, panel_h1, panel_h2 = _make_panels()

    with pl.section_cube():
        core_id = pl.get_block_idx() // pl.get_subblock_num()
        num_cores = pl.get_block_num()
        n_groups = x.shape[0] // KEY_BATCH_PER_GROUP
        # Occupancy predicate (DESIGN.md 5.4): N384 only when it fills >= 3/4 of the grid.
        use_n384 = (n_groups * KEY_PANEL_TASKS * 4) >= (num_cores * 3)
        x2 = pl.make_tensor(x, [x.shape[0] * S, H], [H, 1])
        # Split-K aux buffers are zeroed by the AIV side; wait for that pass before
        # loop A's first AtomicAdd (see ``_zero_aux_workspaces``).
        if KEY_K_SPLIT > 1:
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        _module1_cube(core_id, num_cores, n_groups, use_n384, x2, wqkv, wgate, wbeta, wa, wb,
                      ws_beta_logit, ws_decay_a, ws_raw_decay, panel_h0, panel_h1, panel_h2)

        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)   # phase barrier: M1/M2 -> M4/M3
        _module4_cube(core_id, num_cores, n_groups, x.shape[0], ws_gated, wout, out)

    with pl.section_vector():
        block_id = pl.get_block_idx() // pl.get_subblock_num()
        num_cores = pl.get_block_num()
        sub = pl.get_subblock_idx()
        n_groups = ws_qkv.shape[0] // (KEY_BATCH_PER_GROUP * S)
        use_n384 = (n_groups * KEY_PANEL_TASKS * 4) >= (num_cores * 3)
        if KEY_K_SPLIT > 1:
            # In-kernel zero init (no host-side small op); M1 waits on this barrier
            # before its first AtomicAdd.
            _zero_aux_workspaces(block_id, num_cores, sub, ws_beta_logit, ws_decay_a)
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        _module2_vector(block_id, num_cores, n_groups, sub, use_n384,
                        conv_w, conv_layout, conv_state, conv_indices, conv_accepted,
                        panel_h0, panel_h1, panel_h2, ws_qkv, ws_gate, replay_k)

        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)   # phase barrier: M1/M2 -> M4/M3
        _module3_vector(num_cores, n_groups, ws_qkv, ws_gate, ws_raw_decay,
                        ws_beta_logit, rec_state, a_log, dt_bias, gamma, ws_gated,
                        replay_u, replay_decay, scale, lower_bound, rms_eps)


# ===========================================================================
# Host wrapper: validation + torch.empty + exactly one launch
# ===========================================================================
_NZ = getattr(torch_npu.Format, "FRACTAL_NZ", None)
_GRID_TIERS = (8, 16, 24, 28, 32)


def select_grid():
    """Largest supported AIC tier within the platform budget."""
    try:
        from pypto_pro.runtime.platform import get_platform_info
        info = get_platform_info()
        cube = int(info.cube_core_num)
        vector = int(info.vector_core_num)
    except Exception:  # noqa: BLE001
        cube, vector = 28, 56
    if cube <= 0 or vector < 2 * cube:
        cube, vector = 28, 56
    budget = min(cube, vector // 2)
    supported = [c for c in _GRID_TIERS if c <= budget]
    if not supported:
        raise ValueError(f"current stream exposes {budget} AIC cores; at least 8 are required")
    return max(supported)


#: ``m3_k`` 查表：M3 每个 slot 中每个 worker 处理的 item 数（发布粒度）。
#:
#: 作用：``m3_k`` 决定 M3→M4 的槽位大小 ``slot_cap = m3_k * workers``，也就是两次
#: 跨核 ``set/wait_cross_core`` + ``sync_all`` 之间发布多少递推工作。取值越大，握手
#: 与屏障越少，但 M4 要等更久才能凑齐一个 chunk（``KEY_BATCH_PER_GROUP * KEY_NUM_HEADS`` 个 item = 一个 m-tile），
#: M3 尾段暴露越多；取值越小，消费者启动越早、与下一个 slot 重叠越好，但握手越密。
#: 因此它不是单调调优参数，需按 (核数, N, 组数) 扫描；表内取值为 DESIGN.md §5.2 的实测结论。
#:
#: key 分桶：head = 6 / 12（N=12 与 N=24 共用 12），
#: batch = 0(≤32) / 1(≤64) / 2(>64)，core = 0(≤8 AIC) / 1(16/24/28 AIC) / 2(32 AIC)。
#: 该查表只在 ``_tiling_key`` 里调用一次，结果写入 ``KEY_M3_K`` 编译期字段；
#: kernel 不再接收 ``m3_k`` 标量参数。
_M3_K_TABLE = {
    # N=6：输出投影 K 只有 768、chunk 多；>8 AIC 时细 slot 更划算，32 AIC 用 1 最稳。
    (6, 0, 0): 1, (6, 0, 1): 1, (6, 0, 2): 1,
    (6, 1, 0): 2, (6, 1, 1): 2, (6, 1, 2): 1,
    (6, 2, 0): 4, (6, 2, 1): 2, (6, 2, 2): 1,
    # N=12/24：K 更长、chunk 少；8 AIC 只有 16 个 AIV，slot 太细会把 M4 就绪块切碎。
    (12, 0, 0): 8, (12, 0, 1): 1, (12, 0, 2): 1,
    (12, 1, 0): 8, (12, 1, 1): 2, (12, 1, 2): 2,
    (12, 2, 0): 8, (12, 2, 1): 4, (12, 2, 2): 4,
}


def _m3_k(num_heads, batch, num_cores):
    """Pick ``m3_k`` (items per worker per M3 slot) from ``_M3_K_TABLE``."""
    head_bucket = 6 if num_heads == 6 else 12
    batch_bucket = 0 if batch <= 32 else (1 if batch <= 64 else 2)
    core_bucket = 0 if num_cores <= 8 else (1 if num_cores < 32 else 2)
    return _M3_K_TABLE[(head_bucket, batch_bucket, core_bucket)]


def _conv_layout(shape):
    """Derive the conv-layout integer from the public ``conv1d_weight`` shape."""
    if len(shape) == 4:
        if shape[0] == 3 and shape[2] == CONV_K and shape[3] == D:
            return 0
    elif len(shape) == 2:
        if shape[0] == CONV_K and shape[1] % (3 * D) == 0:
            return 1
    elif tuple(shape) == (3, CONV_K, D):
        return 2
    raise ValueError(f"conv1d_weight must be [3,N,4,D] / [4,3*P] / [3,4,D], got {shape}")


def _m_tile_rows(b):
    rows = int(b) * S
    for mr in M_TILE_LADDER:
        if rows % mr == 0:
            return mr
    raise ValueError(
        f"B={b} (rows={rows}) is not divisible by any m-tile rung {M_TILE_LADDER}")


def _infer_heads(qkv_w, wbeta, wgate, ow, wb):
    """Recover ``N = 96/TP`` from the weight shapes; returns ``(n, p, p3)``."""
    n_beta = int(wbeta.shape[0])
    n_qkv = int(qkv_w.shape[0]) // (3 * D)
    n = n_beta if n_beta in (6, 12, 24) else n_qkv
    if n not in (6, 12, 24):
        raise ValueError(
            f"unsupported head count: N = 96/TP must be one of 6/12/24; "
            f"qkv_projection_weight {tuple(qkv_w.shape)} implies N={n_qkv}, "
            f"beta_projection_weight implies N={n_beta}")
    p, p3 = n * D, 3 * n * D
    shapes = (("qkv_projection_weight", qkv_w, (p3, H)),
              ("decay_projection_b_weight", wb, (p, D)),
              ("output_gate_projection_weight", wgate, (p, H)),
              ("output_projection_weight", ow, (H, p)))
    for name, t, shape in shapes:
        if tuple(t.shape) != shape:
            raise ValueError(f"{name} must be {shape} for N={n}, got {tuple(t.shape)}")
    if n_qkv != n:
        raise ValueError(
            f"qkv_projection_weight {tuple(qkv_w.shape)} implies N={n_qkv}, "
            f"beta_projection_weight implies N={n_beta}")
    return n, p, p3


def _tiling_key(num_heads, proj_width, qkv_width, batch, num_cores):
    """Derive the ``GeometryKey`` tiling-key dict for one launch.

    This helper is the single source of truth for **compile-time specialisation**:
    every value that ends up in a ``TilingKeyField`` is derived here, so the launch
    dict cannot drift from the ``GeometryKey`` schema.  The only inputs are the
    launch geometry (``num_heads`` / ``proj_width`` / ``qkv_width`` / ``batch``) and
    the grid size (``num_cores``).  Derived here rather than in the wrapper:

    - ``m_tile_rows`` / ``batch_per_group``: m-tile rung and the batches packed into
      one tile (``_m_tile_rows`` enforces the m-tile ladder contract);
    - ``k_split``: K-split of the beta / decay-A aux projections (fixed at 8);
    - ``beta_width``: padded beta-logit width, ``16`` for ``N <= 16`` else ``32``;
    - ``enable_prefetch``: ``0`` (inline M3 loads) only on ``<= 8`` AIC with ``N = 6``
      and more than one batch group, where the prefetch variant measured slower;
      ``1`` everywhere else.
    - ``m3_k``: M3 slot granularity (items per worker per slot).  It is part of the
      key, so the kernel needs no extra scalar argument.

    The wrapper keeps validation, workspace allocation and grid selection; when it
    still needs a value that the key owns (``beta_width`` for the beta workspace) it
    reads it back from the returned dict instead of recomputing the formula.
    """
    m_tile_rows = _m_tile_rows(batch)
    batch_per_group = m_tile_rows // S
    num_groups = batch // batch_per_group
    k_split = 8
    beta_width = 16 if num_heads <= 16 else 32
    enable_prefetch = 0 if (num_cores <= 8 and num_heads == 6 and num_groups > 1) else 1
    return {"KEY_NUM_HEADS": num_heads, "KEY_PROJ_WIDTH": proj_width, "KEY_QKV_WIDTH": qkv_width,
            "KEY_QKV_PANELS": qkv_width // PANEL_N, "KEY_GATE_PANELS": proj_width // PANEL_N,
            "KEY_PANEL_TASKS": qkv_width // PANEL_N + proj_width // PANEL_N,
            "KEY_N128_TASKS": 4 * num_heads, "KEY_BETA_WIDTH": beta_width,
            "KEY_M_TILE_ROWS": m_tile_rows, "KEY_M_TILE_ROWS_HALF": m_tile_rows // 2,
            "KEY_BATCH_PER_GROUP": batch_per_group, "KEY_BATCH_PER_SUBBLOCK": batch_per_group // 2,
            "KEY_K_SPLIT": k_split, "KEY_PREFETCH": enable_prefetch,
            "KEY_M3_K": _m3_k(num_heads, batch, num_cores)}


def _validate_hidden(hidden_states):
    if hidden_states.dim() != 3 or tuple(hidden_states.shape[1:]) != (S, H):
        raise ValueError(f"hidden_states must be [B,{S},{H}], got {tuple(hidden_states.shape)}")
    b = int(hidden_states.shape[0])
    if b < 4 or b > BATCH_TILE * MAX_BATCH_GROUPS:
        raise ValueError(
            f"B must be in [4, {BATCH_TILE * MAX_BATCH_GROUPS}]; got B={b}")
    _m_tile_rows(b)
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError(f"hidden_states must be bfloat16, got {hidden_states.dtype}")
    if not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous (wrapper does not reshape)")
    return b


def _validate_weights(qkv_w, wa, wb, wbeta, wgate, ow):
    n, p, p3 = _infer_heads(qkv_w, wbeta, wgate, ow, wb)
    for name, t, shape in (("qkv_projection_weight", qkv_w, (p3, H)),
                           ("decay_projection_a_weight", wa, (D, H)),
                           ("decay_projection_b_weight", wb, (p, D)),
                           ("output_gate_projection_weight", wgate, (p, H)),
                           ("output_projection_weight", ow, (H, p))):
        if tuple(t.shape) != shape:
            raise ValueError(f"{name} must be {shape}, got {tuple(t.shape)}")
        if t.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be bfloat16, got {t.dtype}")
        if _NZ is not None and torch_npu.get_npu_format(t) != _NZ:
            raise ValueError(f"{name} must be FRACTAL_NZ (no host format cast)")
    if tuple(wbeta.shape) != (n, H):
        raise ValueError(f"beta_projection_weight must be [{n},{H}], got {tuple(wbeta.shape)}")
    if wbeta.dtype != torch.bfloat16:
        raise ValueError(f"beta_projection_weight must be bfloat16, got {wbeta.dtype}")
    if _NZ is not None and torch_npu.get_npu_format(wbeta) != _NZ:
        raise ValueError("beta_projection_weight must be FRACTAL_NZ (no host pad)")
    return n, p, p3


def _validate_norm_conv(onw, cw, cs, p3):
    if onw.dim() != 1 or int(onw.shape[0]) != D or onw.dtype != torch.float32:
        raise ValueError(
            f"output_norm_weight must be float32 [{D}], got {tuple(onw.shape)} {onw.dtype}")
    if cw.dtype != torch.bfloat16:
        raise ValueError(f"conv1d_weight must be bfloat16, got {cw.dtype}")
    _conv_layout(tuple(cw.shape))
    if cs.dim() != 3 or int(cs.shape[2]) != p3:
        raise ValueError(f"conv_state must be [C,L,{p3}], got {tuple(cs.shape)}")
    if int(cs.shape[1]) < S + 2:
        raise ValueError(
            f"conv_state state_length must be >= S+2 = {S + 2}, got {int(cs.shape[1])}")
    if cs.dtype != torch.bfloat16:
        raise ValueError(f"conv_state must be bfloat16, got {cs.dtype}")


def _validate_state_tensors(rs, ru, rk, rd, b, n):
    if tuple(rs.shape) != (b, n, D, D):
        raise ValueError(f"recurrent_state must be [{b},{n},{D},{D}], got {tuple(rs.shape)}")
    if rs.dtype != torch.float32:
        raise ValueError(f"recurrent_state must be float32, got {rs.dtype}")
    for name, t, dt in (("replay_u", ru, torch.float16), ("replay_k", rk, torch.float16),
                        ("replay_decay", rd, torch.float32)):
        if tuple(t.shape) != (b, S, n, D) or t.dtype != dt:
            raise ValueError(
                f"{name} must be [{b},{S},{n},{D}] {dt}, got {tuple(t.shape)} {t.dtype}")


def _validate_index_tensors(ci, ca, a_log, dt_bias, b, n):
    if tuple(ci.shape) != (b,) or tuple(ca.shape) != (b,):
        raise ValueError("conv_state_indices / conv_num_accepted_tokens must be [B]")
    if ci.dtype != torch.int32 or ca.dtype != torch.int32:
        raise ValueError("conv_state_indices / conv_num_accepted_tokens must be int32")
    if tuple(a_log.shape) != (n,) or a_log.dtype != torch.float32:
        raise ValueError(f"a_log must be float32 [{n}], got {tuple(a_log.shape)} {a_log.dtype}")
    if tuple(dt_bias.shape) != (n, D) or dt_bias.dtype != torch.float32:
        raise ValueError(
            f"dt_bias must be float32 [{n},{D}], got {tuple(dt_bias.shape)} {dt_bias.dtype}")


def _validate_scalars(scale, lower_bound, rms_norm_eps):
    # Scalar args are bound as builtin Python floats by the launcher, so require the
    # exact type here instead of silently coercing numpy scalars.
    if not isinstance(scale, float) or scale.__class__ is not float or not math.isfinite(scale):
        raise ValueError(
            f"scale must be a finite Python float, got {scale.__class__.__name__} {scale!r}")
    if not isinstance(lower_bound, float) or lower_bound.__class__ is not float:
        raise ValueError(
            f"lower_bound must be a Python float in [-5,0], got "
            f"{lower_bound.__class__.__name__} {lower_bound!r}")
    if not -5.0 <= lower_bound <= 0.0:
        raise ValueError(
            f"lower_bound must be a Python float in [-5,0], got "
            f"{lower_bound.__class__.__name__} {lower_bound!r}")
    if not isinstance(rms_norm_eps, float) or rms_norm_eps.__class__ is not float:
        raise ValueError(
            f"rms_norm_eps must be a Python float > 0, got "
            f"{rms_norm_eps.__class__.__name__} {rms_norm_eps!r}")
    if not rms_norm_eps > 0.0:
        raise ValueError(
            f"rms_norm_eps must be a Python float > 0, got "
            f"{rms_norm_eps.__class__.__name__} {rms_norm_eps!r}")


def _validate_args(hidden_states, qkv_w, wa, wb, wbeta, wgate, onw, ow, cw, cs, rs,
                   ru, rk, rd, ci, ca, a_log, dt_bias, scale, lower_bound, rms_norm_eps):
    """Cheap host-side contract checks only (no tensor transform)."""
    b = _validate_hidden(hidden_states)
    n, _p, p3 = _validate_weights(qkv_w, wa, wb, wbeta, wgate, ow)
    _validate_norm_conv(onw, cw, cs, p3)
    _validate_state_tensors(rs, ru, rk, rd, b, n)
    _validate_index_tensors(ci, ca, a_log, dt_bias, b, n)
    _validate_scalars(scale, lower_bound, rms_norm_eps)


def mega_recurrent_kda_replayssm_wrapper(
    hidden_states, qkv_projection_weight, decay_projection_a_weight,
    decay_projection_b_weight, beta_projection_weight, output_gate_projection_weight,
    output_norm_weight, output_projection_weight, conv1d_weight, conv_state,
    recurrent_state, replay_u, replay_k, replay_decay, conv_state_indices,
    conv_num_accepted_tokens, a_log, dt_bias, scale, lower_bound, rms_norm_eps=1e-5,
):
    """Public entry: validate, allocate the workspace, launch one MIX kernel.

    Returns ``out`` ``[B, S, H]`` (bf16).  ``conv_state``, ``replay_k``, ``replay_u``
    and ``replay_decay`` are updated in place; ``recurrent_state`` is read-only.
    """
    _validate_args(hidden_states, qkv_projection_weight, decay_projection_a_weight,
                   decay_projection_b_weight, beta_projection_weight,
                   output_gate_projection_weight, output_norm_weight,
                   output_projection_weight, conv1d_weight, conv_state, recurrent_state,
                   replay_u, replay_k, replay_decay, conv_state_indices,
                   conv_num_accepted_tokens, a_log, dt_bias, scale, lower_bound, rms_norm_eps)
    dev = hidden_states.device
    batch = int(hidden_states.shape[0])
    token_rows = batch * S
    conv_layout = _conv_layout(tuple(conv1d_weight.shape))
    num_heads, proj_width, qkv_width = _infer_heads(
        qkv_projection_weight, beta_projection_weight,
        output_gate_projection_weight, output_projection_weight,
        decay_projection_b_weight)
    # Compile-time specialisation is derived in ``_tiling_key`` (single source of
    # truth for the GeometryKey schema).  The host still needs a few of those values
    # for allocation / initialisation, so read them back from the returned key.
    num_cores = select_grid()
    tiling_key = _tiling_key(num_heads, proj_width, qkv_width, batch, num_cores)
    beta_width = tiling_key["KEY_BETA_WIDTH"]

    def ws(shape, dtype):
        return torch.empty(shape, dtype=dtype, device=dev)

    ws_beta_logit = ws((token_rows, beta_width), torch.bfloat16)
    ws_decay_a = ws((token_rows, D), torch.bfloat16)
    ws_raw_decay = ws((token_rows, proj_width), torch.bfloat16)
    ws_qkv = ws((token_rows, qkv_width), torch.bfloat16)
    ws_gate = ws((token_rows, proj_width), torch.bfloat16)
    ws_gated = ws((token_rows, proj_width), torch.bfloat16)
    out = ws((batch, S, H), torch.bfloat16)

    mega_recurrent_kda_replayssm_kernel[None, num_cores, tiling_key](
        hidden_states, qkv_projection_weight, decay_projection_a_weight,
        decay_projection_b_weight, beta_projection_weight, output_gate_projection_weight,
        output_projection_weight,
        ws_beta_logit, ws_decay_a, ws_raw_decay,
        conv1d_weight, conv_layout, conv_state, conv_state_indices,
        conv_num_accepted_tokens, ws_qkv, ws_gate, replay_k,
        ws_gated, recurrent_state, a_log, dt_bias, output_norm_weight,
        replay_u, replay_decay,
        scale, lower_bound, rms_norm_eps, out)
    return out


