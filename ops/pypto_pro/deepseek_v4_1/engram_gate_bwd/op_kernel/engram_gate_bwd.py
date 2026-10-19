# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
"""engram_gate_bwd PyPTO-Pro kernel and host wrapper (single dynamic-axis kernel).

One module-level ``@pl.jit(auto_mutex=True)`` kernel serves any ``t/hc/dim``
(dynamic axes).  ``engram_gate_bwd`` is the analytic adjoint of the sibling
``engram_gate`` forward; like ``torch.autograd`` it consumes the tensors the
forward saved -- ``rstd_x``, ``rstd_k``, ``raw_dot`` and ``gate`` -- and returns
``grad_x``, ``grad_key``, ``grad_value`` and ``grad_weight``.

Work partitioning: core ``c`` owns head ``c % hc`` for a contiguous slab of
tokens, so the four heads of a token land on four neighbouring cores and every
core sees the same mix of real and image tokens.  A ``(token, head)`` work item
is one VF call that leaves ``grad_x``/``grad_key`` in the UB tiles it read and
accumulates ``grad_weight`` in UB; all arithmetic is inside the kernel, the host
only validates, allocates and launches once.
"""

import math

import torch

import pypto
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import get_platform_info

# ---------------------------------------------------------------- constants
HC = 4                      # hc_mult
STEP = 128                  # elements per register-level conversion step
MAXC = 7168                 # max supported dim (fixed UB row width)
MAXV = MAXC // HC           # max supported v-slice width
NPART = 16                  # grad_weight partial rows == num_vec_cores // HC
S42_LEN = 256               # scratch tile width: 4 * HC saved rows + scalar slots

# The gate threshold is a compile-time literal inside the VF, so the wrapper can
# only accept this value.
_CLAMP_VALUE = 1e-06


def _addr_table():
    """UB byte offsets; every entry is 256 B aligned (>= 32 B required)."""
    a = {}
    off = 0
    # Every per-token tile gets two slots: the next work item's MTE2 is in flight
    # while the current one is being computed.
    for name, nbytes in (
        ("go_b", 2 * MAXC * 2),      # bf16 [1, MAXC] x2  grad_out -> grad_x
        ("v_b", 2 * MAXC * 2),       # bf16 [1, MAXC] x2  value -> grad_value slice
        ("x_b", 2 * MAXC * 2),       # bf16 [1, MAXC] x2  x
        ("k_b", 2 * MAXC * 2),       # bf16 [1, MAXC] x2  key   -> grad_key
        ("og0", 2 * MAXV * 2),       # bf16 [1, MAXV] x2  dy quarter, head h1
        ("og1", 2 * MAXV * 2),       # bf16 [1, MAXV] x2  dy quarter, head h2
        ("og2", 2 * MAXV * 2),       # bf16 [1, MAXV] x2  dy quarter, head h3
        ("w", MAXC * 4),             # fp32 [1, MAXC]     resident weight row
        ("gw", MAXC * 4),            # fp32 [1, MAXC]     resident grad_weight
        ("zb", MAXC * 2),            # bf16 [1, MAXC] zero source for image tokens
        ("s42", 2 * S42_LEN * 4),    # fp32 [1, 256] x2   forward-saved statistics
    ):
        a[name] = off
        off += nbytes
    return a, off


VA, UB_USED = _addr_table()
if UB_USED > 253952:
    raise RuntimeError(f"UB budget exceeded: {UB_USED} B > 253952 B")


# ---------------------------------------------------------------- VF stages
@pl.vector_function
def _vf_clear(tile, n_dual: pl.DT_INT64):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    z = vf.full(0.0, preg, dtype=pl.DT_FP32)
    for s in pl.range(0, n_dual):
        vf.store_align(tile + s * STEP, z, z, preg, dist=pl.StoreDist.INTLV_B32)


@pl.vector_function
def _vf_zero_b16(tile, n_dual: pl.DT_INT64):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    z = vf.full(0.0, preg, dtype=pl.DT_BF16)
    for s in pl.range(0, n_dual):
        vf.store_align(tile + s * 128, z, preg, dist=pl.StoreDist.NORM_B16)


@pl.vector_function
def engram_gate_bwd_vf(go_b, v_b, x_b, k_b, w32, gw32, s42, og0, og1, og2,
                       head: pl.DT_INT64, vs: pl.DT_INT64,
                       n_step: pl.DT_INT64, n_step_v: pl.DT_INT64,
                       dim: pl.DT_INT64):
    """One (token, head) work item end to end.

    All stages live in one VF call, so `beta`/`ndx`/`ndk` and the four
    `gate_o*active` values stay in registers instead of being published to and
    reloaded from `s42`, and a work item costs one VF call instead of three.
    The stage order is forced by the maths: `beta` depends on the whole `dgate`
    reduction, so the reduction must complete before any grad_x/grad_k write.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)

    # ---- stage 2a: everything that depends only on `stats` ----
    raw_dot = vf.load_align(s42, head, dist=pl.LoadDist.BRC_B32)
    gate_h = vf.load_align(s42, HC + head, dist=pl.LoadDist.BRC_B32)
    ix = vf.load_align(s42, 2 * HC + head, dist=pl.LoadDist.BRC_B32)
    ik = vf.load_align(s42, 3 * HC + head, dist=pl.LoadDist.BRC_B32)

    # 1/sqrt(dim) and -1/dim are derived here, in the kernel, from the one runtime
    # axis.  Image tokens never reach this path, so the gate weight is always 1.
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    active = vf.full(1.0, preg, dtype=pl.DT_FP32)
    dimf = vf.full(dim, preg, dtype=pl.DT_FP32)
    sqdim = vf.sqrt(dimf, preg)
    sv = vf.div(one, sqdim, preg)
    neg_one = vf.full(-1.0, preg, dtype=pl.DT_FP32)
    nih = vf.div(neg_one, dimf, preg)
    cv = vf.full(_CLAMP_VALUE, preg, dtype=pl.DT_FP32)
    ds = vf.mul(ix, ik, preg)
    ds = vf.mul(ds, sv, preg)
    ard = vf.abs(raw_dot, preg)
    sc_ = vf.mul(ard, ds, preg)
    ag = vf.ge(sc_, cv, preg)
    q = vf.div(ds, ard, preg)
    sq = vf.sqrt(q, preg)
    sgs = vf.muls(sq, 0.5, preg)
    gg = vf.mul(gate_h, gate_h, preg)
    gg = vf.sub(gate_h, gg, preg)

    # ---- stage 1: dgate = sum_d go*v  (two accumulators, unrolled x2) ----
    acc0 = vf.full(0.0, preg, dtype=pl.DT_FP32)
    acc1 = vf.full(0.0, preg, dtype=pl.DT_FP32)
    n_pair = n_step // 2
    for s in pl.range(0, n_pair):
        off = s * 2 * STEP
        gb = vf.load_align(go_b, off)
        vb = vf.load_align(v_b, off)
        ge = vf.astype(gb, preg16, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP32)
        gd = vf.astype(gb, preg16, layout=pl.CastLayout.ONE, dtype=pl.DT_FP32)
        ve = vf.astype(vb, preg16, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP32)
        vo = vf.astype(vb, preg16, layout=pl.CastLayout.ONE, dtype=pl.DT_FP32)
        acc0 = vf.mul_add_dst(ge, ve, preg)
        acc0 = vf.mul_add_dst(gd, vo, preg)
        off = s * 2 * STEP + STEP
        gb = vf.load_align(go_b, off)
        vb = vf.load_align(v_b, off)
        ge = vf.astype(gb, preg16, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP32)
        gd = vf.astype(gb, preg16, layout=pl.CastLayout.ONE, dtype=pl.DT_FP32)
        ve = vf.astype(vb, preg16, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP32)
        vo = vf.astype(vb, preg16, layout=pl.CastLayout.ONE, dtype=pl.DT_FP32)
        acc1 = vf.mul_add_dst(ge, ve, preg)
        acc1 = vf.mul_add_dst(gd, vo, preg)
    for s in pl.range(n_pair * 2, n_step):
        off = s * STEP
        gb = vf.load_align(go_b, off)
        vb = vf.load_align(v_b, off)
        ge = vf.astype(gb, preg16, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP32)
        gd = vf.astype(gb, preg16, layout=pl.CastLayout.ONE, dtype=pl.DT_FP32)
        ve = vf.astype(vb, preg16, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP32)
        vo = vf.astype(vb, preg16, layout=pl.CastLayout.ONE, dtype=pl.DT_FP32)
        acc0 = vf.mul_add_dst(ge, ve, preg)
        acc0 = vf.mul_add_dst(gd, vo, preg)
    acct = vf.add(acc0, acc1, preg)
    dg = vf.reduce_sum(acct, preg)
    dgb = vf.full(dg, preg)

    # ---- stage 2b: the dgb-dependent tail of the scalar chain ----
    t1 = vf.mul(gg, sgs, ag)
    t2 = vf.mul(dgb, t1, preg)
    beta = vf.mul(t2, active, preg)
    rdh = vf.mul(beta, raw_dot, preg)
    c = vf.mul(rdh, nih, preg)
    ixx = vf.mul(ix, ix, preg)
    ikk = vf.mul(ik, ik, preg)
    ndx = vf.mul(c, ixx, preg)
    ndk = vf.mul(c, ikk, preg)

    # grad_v needs the owned head's gate plus the three others'; the two runtime
    # indices let this stay in registers with no runtime selection chain.
    ga_h = vf.mul(gate_h, active, preg)
    h1 = (head + 1) % HC
    h2 = (head + 2) % HC
    h3 = (head + 3) % HC
    g_1 = vf.load_align(s42, HC + h1, dist=pl.LoadDist.BRC_B32)
    g_2 = vf.load_align(s42, HC + h2, dist=pl.LoadDist.BRC_B32)
    g_3 = vf.load_align(s42, HC + h3, dist=pl.LoadDist.BRC_B32)
    ga1 = vf.mul(g_1, active, preg)
    ga2 = vf.mul(g_2, active, preg)
    ga3 = vf.mul(g_3, active, preg)


    # ---- stage 3: grad_v over the owned quarter, written over v_b. ----
    # Runs BEFORE stage 4: stage 4 overwrites go_b with grad_x in place, and the
    # owned quarter of grad_v is read straight out of go_b while it still holds dy. ----
    for s in pl.range(0, n_step_v):
        off = s * STEP
        own = vf.load_align(go_b, vs + off)
        b1 = vf.load_align(og0, off)
        b2 = vf.load_align(og1, off)
        b3 = vf.load_align(og2, off)
        o0e = vf.astype(own, preg16, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP32)
        d1e = vf.astype(b1, preg16, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP32)
        d2e = vf.astype(b2, preg16, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP32)
        d3e = vf.astype(b3, preg16, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP32)
        acc = vf.mul(ga_h, o0e, preg)
        acc = vf.mul_add_dst(ga1, d1e, preg)
        acc = vf.mul_add_dst(ga2, d2e, preg)
        acc = vf.mul_add_dst(ga3, d3e, preg)
        ae = vf.astype(acc, preg, layout=pl.CastLayout.ZERO, dtype=pl.DT_BF16)

        o0o = vf.astype(own, preg16, layout=pl.CastLayout.ONE, dtype=pl.DT_FP32)
        d1o = vf.astype(b1, preg16, layout=pl.CastLayout.ONE, dtype=pl.DT_FP32)
        d2o = vf.astype(b2, preg16, layout=pl.CastLayout.ONE, dtype=pl.DT_FP32)
        d3o = vf.astype(b3, preg16, layout=pl.CastLayout.ONE, dtype=pl.DT_FP32)
        aco = vf.mul(ga_h, o0o, preg)
        aco = vf.mul_add_dst(ga1, d1o, preg)
        aco = vf.mul_add_dst(ga2, d2o, preg)
        aco = vf.mul_add_dst(ga3, d3o, preg)
        ao = vf.astype(aco, preg, layout=pl.CastLayout.ONE, dtype=pl.DT_BF16)

        comb = vf.xor(ae, ao, preg16)
        vf.store_align(v_b + off, comb, preg16, dist=pl.StoreDist.NORM_B16)

    # ---- stage 4: grad_x / grad_k / grad_w over the full row ----
    for s in pl.range(0, n_step):
        off = s * STEP
        gb = vf.load_align(go_b, off)
        xb = vf.load_align(x_b, off)
        kb = vf.load_align(k_b, off)
        we, wo = vf.load_align(w32, off, dist=pl.LoadDist.DINTLV_B32)
        gwe, gwo = vf.load_align(gw32, off, dist=pl.LoadDist.DINTLV_B32)

        ge = vf.astype(gb, preg16, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP32)
        xe = vf.astype(xb, preg16, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP32)
        ke = vf.astype(kb, preg16, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP32)
        bxe = vf.mul(beta, xe, preg)
        bke = vf.mul(beta, ke, preg)
        gxe = vf.mul(we, bke, preg)
        gxe = vf.mul_add_dst(ndx, xe, preg)
        gxe = vf.add(ge, gxe, preg)
        gke = vf.mul(we, bxe, preg)
        gke = vf.mul_add_dst(ndk, ke, preg)
        nwe = vf.mul(bxe, ke, preg)
        nwe = vf.add(gwe, nwe, preg)

        gd = vf.astype(gb, preg16, layout=pl.CastLayout.ONE, dtype=pl.DT_FP32)
        xo = vf.astype(xb, preg16, layout=pl.CastLayout.ONE, dtype=pl.DT_FP32)
        ko = vf.astype(kb, preg16, layout=pl.CastLayout.ONE, dtype=pl.DT_FP32)
        bxo = vf.mul(beta, xo, preg)
        bko = vf.mul(beta, ko, preg)
        gxo = vf.mul(wo, bko, preg)
        gxo = vf.mul_add_dst(ndx, xo, preg)
        gxo = vf.add(gd, gxo, preg)
        gko = vf.mul(wo, bxo, preg)
        gko = vf.mul_add_dst(ndk, ko, preg)
        nwo = vf.mul(bxo, ko, preg)
        nwo = vf.add(gwo, nwo, preg)

        ae = vf.astype(gxe, preg, layout=pl.CastLayout.ZERO, dtype=pl.DT_BF16)
        ao = vf.astype(gxo, preg, layout=pl.CastLayout.ONE, dtype=pl.DT_BF16)
        comb = vf.xor(ae, ao, preg16)
        vf.store_align(go_b + off, comb, preg16, dist=pl.StoreDist.NORM_B16)

        ae = vf.astype(gke, preg, layout=pl.CastLayout.ZERO, dtype=pl.DT_BF16)
        ao = vf.astype(gko, preg, layout=pl.CastLayout.ONE, dtype=pl.DT_BF16)
        comb = vf.xor(ae, ao, preg16)
        vf.store_align(k_b + off, comb, preg16, dist=pl.StoreDist.NORM_B16)

        vf.store_align(gw32 + off, nwe, nwo, preg, dist=pl.StoreDist.INTLV_B32)


# ---------------------------------------------------------------- kernel
@pl.jit(auto_mutex=True)
def engram_gate_bwd_kernel(
    grad_out: pl.Tensor[[pl.DYNAMIC, HC, pl.DYNAMIC], pl.DT_BF16],
    x: pl.Tensor[[pl.DYNAMIC, HC, pl.DYNAMIC], pl.DT_BF16],
    key: pl.Tensor[[pl.DYNAMIC, HC, pl.DYNAMIC], pl.DT_BF16],
    value: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    weight: pl.Tensor[[HC, pl.DYNAMIC], pl.DT_FP32],
    saved: pl.Tensor[[pl.DYNAMIC, 4 * HC], pl.DT_FP32],
    image_mask: pl.Tensor[[pl.DYNAMIC], pl.DT_BOOL],
    grad_x: pl.Tensor[[pl.DYNAMIC, HC, pl.DYNAMIC], pl.DT_BF16],
    grad_key: pl.Tensor[[pl.DYNAMIC, HC, pl.DYNAMIC], pl.DT_BF16],
    grad_value: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    grad_w_partial: pl.Tensor[[NPART, HC, pl.DYNAMIC], pl.DT_FP32],
    dim: pl.DT_INT64,
):
    t_b16 = pl.TileType(shape=[1, MAXC], dtype=pl.DT_BF16,
                        target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    t_f32 = pl.TileType(shape=[1, MAXC], dtype=pl.DT_FP32,
                        target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    t_s = pl.TileType(shape=[1, S42_LEN], dtype=pl.DT_FP32,
                      target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    t_vb16 = pl.TileType(shape=[1, MAXV], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])

    g_go = pl.make_tile_group(type=t_b16, addrs=[VA["go_b"], VA["go_b"] + MAXC * 2],
                              mutex_ids=[0, 1])
    g_vb = pl.make_tile_group(type=t_b16, addrs=[VA["v_b"], VA["v_b"] + MAXC * 2],
                              mutex_ids=[2, 3])
    g_xb = pl.make_tile_group(type=t_b16, addrs=[VA["x_b"], VA["x_b"] + MAXC * 2],
                              mutex_ids=[4, 5])
    g_kb = pl.make_tile_group(type=t_b16, addrs=[VA["k_b"], VA["k_b"] + MAXC * 2],
                              mutex_ids=[6, 7])
    g_o0 = pl.make_tile_group(type=t_vb16, addrs=[VA["og0"], VA["og0"] + MAXV * 2],
                              mutex_ids=[8, 9])
    g_o1 = pl.make_tile_group(type=t_vb16, addrs=[VA["og1"], VA["og1"] + MAXV * 2],
                              mutex_ids=[14, 15])
    g_o2 = pl.make_tile_group(type=t_vb16, addrs=[VA["og2"], VA["og2"] + MAXV * 2],
                              mutex_ids=[16, 17])
    g_w = pl.make_tile_group(type=t_f32, addrs=[VA["w"]], mutex_ids=[10])
    g_gw = pl.make_tile_group(type=t_f32, addrs=[VA["gw"]], mutex_ids=[11])
    g_s = pl.make_tile_group(type=t_s, addrs=[VA["s42"], VA["s42"] + S42_LEN * 4],
                             mutex_ids=[12, 13])
    g_z = pl.make_tile_group(type=t_b16, addrs=[VA["zb"]], mutex_ids=[14])

    with pl.section_vector():
        ncore = pl.get_block_num()
        cid = pl.get_block_idx()
        ng = ncore // HC
        head = cid % HC
        grp = cid // HC
        ntok = grad_out.shape[0]
        nv = dim // HC
        vs = head * nv
        nstep = dim // STEP
        nstepv = nv // STEP
        h1 = (head + 1) % HC
        h2 = (head + 2) % HC
        h3 = (head + 3) % HC


        wt = g_w.next()
        pl.set_validshape(wt, [1, dim])
        pl.load(wt, weight, [head, 0])

        gw = g_gw.next()
        pl.set_validshape(gw, [1, dim])
        _vf_clear(gw, nstep)

        zt = g_z.next()
        pl.set_validshape(zt, [1, dim])
        _vf_zero_b16(zt, dim // 128)

        # Depth-2 slot rotation with no explicit pipe barrier in the loop body: the
        # framework's auto_mutex inserts the MTE2 -> V -> MTE3 ordering per slot, so
        # slot i's compute overlaps slot i+1's copies.  Two work items per iteration
        # and every copy issued before any compute, so the second item's MTE2 stays
        # in flight during the first item's vector work.
        # Each item branches on the image-token flag: an image token only copies
        # grad_out into grad_x and zeroes its grad_key / grad_value rows, so it does
        # not pay for work the forward contract discards.
        #
        # Balanced CONTIGUOUS ownership: core `grp` owns one contiguous slab of
        # tokens.  A strided mapping (t = grp + i*ng) hands every core a single
        # token residue class, so with an image-token mask some cores would get
        # only cheap image tokens and others only full work; the kernel then runs
        # at the speed of the slowest core and the mask buys nothing.  Cores
        # 0..nr-1 take nq+1 tokens, the rest take nq, so coverage stays exact.
        nq = ntok // ng
        nr = ntok - nq * ng
        base = grp * nq + nr
        nitems = nq
        if grp < nr:
            base = grp * (nq + 1)
            nitems = nq + 1
        n_pair = nitems // 2

        for i in pl.range(0, n_pair):
            ta = base + i * 2
            tb = ta + 1
            fa = pl.getval(image_mask, ta)
            fb = pl.getval(image_mask, tb)

            # Bind every tile handle unconditionally, guard only the loads: the
            # DSL parser needs a single static definition per name, and an image
            # token should not pay for tiles whose results the contract discards.
            sga = g_go[0]
            sgb = g_go[1]
            sv_a = g_vb[0]
            sv_b = g_vb[1]
            sxa = g_xb[0]
            sxb = g_xb[1]
            ska = g_kb[0]
            skb = g_kb[1]
            o0a = g_o0[0]
            o0b = g_o0[1]
            o1a = g_o1[0]
            o1b = g_o1[1]
            o2a = g_o2[0]
            o2b = g_o2[1]
            s42a = g_s[0]
            s42b = g_s[1]

            pl.set_validshape(sga, [1, dim])
            pl.load(sga, grad_out, [ta, head, 0])
            pl.set_validshape(sgb, [1, dim])
            pl.load(sgb, grad_out, [tb, head, 0])

            if fa == 0:
                pl.set_validshape(s42a, [1, 4 * HC])
                pl.load(s42a, saved, [ta, 0])
                pl.set_validshape(sv_a, [1, dim])
                pl.load(sv_a, value, [ta, 0])
                pl.set_validshape(sxa, [1, dim])
                pl.load(sxa, x, [ta, head, 0])
                pl.set_validshape(ska, [1, dim])
                pl.load(ska, key, [ta, head, 0])
                pl.set_validshape(o0a, [1, nv])
                pl.load(o0a, grad_out, [ta, h1, vs])
                pl.set_validshape(o1a, [1, nv])
                pl.load(o1a, grad_out, [ta, h2, vs])
                pl.set_validshape(o2a, [1, nv])
                pl.load(o2a, grad_out, [ta, h3, vs])
            if fb == 0:
                pl.set_validshape(s42b, [1, 4 * HC])
                pl.load(s42b, saved, [tb, 0])
                pl.set_validshape(sv_b, [1, dim])
                pl.load(sv_b, value, [tb, 0])
                pl.set_validshape(sxb, [1, dim])
                pl.load(sxb, x, [tb, head, 0])
                pl.set_validshape(skb, [1, dim])
                pl.load(skb, key, [tb, head, 0])
                pl.set_validshape(o0b, [1, nv])
                pl.load(o0b, grad_out, [tb, h1, vs])
                pl.set_validshape(o1b, [1, nv])
                pl.load(o1b, grad_out, [tb, h2, vs])
                pl.set_validshape(o2b, [1, nv])
                pl.load(o2b, grad_out, [tb, h3, vs])

            if fa == 0:
                engram_gate_bwd_vf(sga, sv_a, sxa, ska, wt, gw, s42a, o0a, o1a, o2a,
                          head, vs, nstep, nstepv, dim)
                pl.store(grad_x, sga, [ta, head, 0])
                pl.store(grad_key, ska, [ta, head, 0])
                pl.set_validshape(sv_a, [1, nv])
                pl.store(grad_value, sv_a, [ta, vs])
            else:
                pl.set_validshape(zt, [1, dim])
                pl.store(grad_x, sga, [ta, head, 0])
                pl.store(grad_key, zt, [ta, head, 0])
                pl.set_validshape(zt, [1, nv])
                pl.store(grad_value, zt, [ta, vs])
            if fb == 0:
                engram_gate_bwd_vf(sgb, sv_b, sxb, skb, wt, gw, s42b, o0b, o1b, o2b,
                          head, vs, nstep, nstepv, dim)
                pl.store(grad_x, sgb, [tb, head, 0])
                pl.store(grad_key, skb, [tb, head, 0])
                pl.set_validshape(sv_b, [1, nv])
                pl.store(grad_value, sv_b, [tb, vs])
            else:
                pl.set_validshape(zt, [1, dim])
                pl.store(grad_x, sgb, [tb, head, 0])
                pl.store(grad_key, zt, [tb, head, 0])
                pl.set_validshape(zt, [1, nv])
                pl.store(grad_value, zt, [tb, vs])

        for i in pl.range(n_pair * 2, nitems):
            t = base + i
            f = pl.getval(image_mask, t)
            sg = g_go[0]
            s42 = g_s[0]
            sv_ = g_vb[0]
            sx = g_xb[0]
            sk = g_kb[0]
            o0 = g_o0[0]
            o1 = g_o1[0]
            o2 = g_o2[0]
            pl.set_validshape(sg, [1, dim])
            pl.load(sg, grad_out, [t, head, 0])
            if f == 0:
                pl.set_validshape(s42, [1, 4 * HC])
                pl.load(s42, saved, [t, 0])
                pl.set_validshape(sv_, [1, dim])
                pl.load(sv_, value, [t, 0])
                pl.set_validshape(sx, [1, dim])
                pl.load(sx, x, [t, head, 0])
                pl.set_validshape(sk, [1, dim])
                pl.load(sk, key, [t, head, 0])
                pl.set_validshape(o0, [1, nv])
                pl.load(o0, grad_out, [t, h1, vs])
                pl.set_validshape(o1, [1, nv])
                pl.load(o1, grad_out, [t, h2, vs])
                pl.set_validshape(o2, [1, nv])
                pl.load(o2, grad_out, [t, h3, vs])
                engram_gate_bwd_vf(sg, sv_, sx, sk, wt, gw, s42, o0, o1, o2,
                          head, vs, nstep, nstepv, dim)
                pl.store(grad_x, sg, [t, head, 0])
                pl.store(grad_key, sk, [t, head, 0])
                pl.set_validshape(sv_, [1, nv])
                pl.store(grad_value, sv_, [t, vs])
            else:
                pl.set_validshape(zt, [1, dim])
                pl.store(grad_x, sg, [t, head, 0])
                pl.store(grad_key, zt, [t, head, 0])
                pl.set_validshape(zt, [1, nv])
                pl.store(grad_value, zt, [t, vs])

        pl.system.bar_all()
        gws = g_gw.next()
        pl.set_validshape(gws, [1, dim])
        pl.store(grad_w_partial, gws, [grp, head, 0])


# ---------------------------------------------------------------- host wrapper
def _vector_core_count():
    """Vector core count from the PyPTO-Pro platform info (not hard-coded)."""
    info = get_platform_info()
    count = int(info.vector_core_num)
    if count <= 0:
        raise RuntimeError(
            "cannot determine the NPU vector core count from the PyPTO-Pro platform info")
    return count


def _launch_width():
    """Cores to launch: the platform vector-core count, clamped to what this
    operator can use and rounded down so the per-core token stride stays a
    multiple of hc."""
    n = min(_vector_core_count(), NPART * HC)
    return n - n % HC


def _pack_stats(raw_dot, gate, rstd_x, rstd_k):
    """Pack the four forward-saved rows into one contiguous ``[t, 4 * hc]`` buffer."""
    return torch.cat([raw_dot.float(), gate.float(), rstd_x.float(),
                      rstd_k.float()], dim=1).contiguous()


def _alloc_outputs(x, key, value, weight):
    """Allocate the four outputs with the dtypes and shapes the kernel writes."""
    grad_w_partial = torch.empty((NPART, HC, weight.shape[-1]),
                                 dtype=torch.float32, device=x.device)
    return (torch.empty_like(x), torch.empty_like(key), torch.empty_like(value),
            grad_w_partial)


def engram_gate_bwd_wrapper(grad_out, x, key, value, weight,
                            rstd_x, rstd_k, raw_dot, gate, image_mask,
                            clamp_value=_CLAMP_VALUE):
    """Public wrapper: validate, allocate the outputs and launch the kernel once.

    ``rstd_x``/``rstd_k``/``raw_dot``/``gate`` are the per-``(t, hc)`` FP32
    tensors the forward ``engram_gate`` saves for the backward, with
    ``raw_dot = sum_d x * weight * key``.  The host packs those four rows into
    one contiguous ``[t, 4 * hc]`` buffer -- a layout step, no arithmetic -- and
    combines the per-core ``grad_weight`` partials once at the end, because the
    framework offers no cross-core atomic add.
    """
    t, hc, dim = x.shape
    if hc != HC:
        raise ValueError(f"this kernel bakes hc={HC}, got hc={hc}")
    if list(grad_out.shape) != [t, hc, dim] or list(key.shape) != [t, hc, dim]:
        raise ValueError("grad_out/key shapes do not match x")
    if list(weight.shape) != [hc, dim]:
        raise ValueError("weight shape does not match x")
    if list(value.shape) != [t, dim] or list(image_mask.shape) != [t]:
        raise ValueError("value/image_mask shapes do not match x")
    for name, tensor in (("rstd_x", rstd_x), ("rstd_k", rstd_k),
                         ("raw_dot", raw_dot), ("gate", gate)):
        if list(tensor.shape) != [t, hc]:
            raise ValueError(f"{name} must be [{t}, {hc}]")
    if dim > MAXC:
        raise ValueError(f"dim {dim} exceeds the supported row width {MAXC}")
    if dim % (HC * STEP) != 0:
        raise ValueError(
            f"dim {dim} must be a multiple of hc * STEP = {HC * STEP}: the row is "
            "scanned in STEP-element steps and the grad_value quarter in hc of them")
    if not math.isclose(clamp_value, _CLAMP_VALUE):
        raise ValueError(
            f"this kernel bakes clamp_value={_CLAMP_VALUE}, got {clamp_value}")
    for name, tensor in (("grad_out", grad_out), ("x", x), ("key", key),
                         ("value", value), ("weight", weight),
                         ("image_mask", image_mask)):
        if not tensor.is_contiguous():
            raise ValueError(
                f"{name} must be contiguous: the kernel addresses rows from the "
                "shape and does not read strides, so a view would be read wrong")

    saved = _pack_stats(raw_dot, gate, rstd_x, rstd_k)
    grad_x, grad_key, grad_value, grad_w_partial = _alloc_outputs(x, key, value, weight)

    block_dim = _launch_width()
    with pypto.options(pass_options={"enable_slice": False}):
        engram_gate_bwd_kernel[None, block_dim](
            grad_out, x, key, value, weight, saved, image_mask,
            grad_x, grad_key, grad_value, grad_w_partial,
            dim,
        )
    return grad_x, grad_key, grad_value, grad_w_partial.sum(0)
