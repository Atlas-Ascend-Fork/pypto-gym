# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
"""mhc_pre_fuse PyPTO-Pro kernel and host wrapper.

One ``@pl.jit(auto_mutex=True)`` kernel definition covers the fused mHC
pre-pass (m = 4 residual mixes):

    RMSNorm -> split mixes -> Sinkhorn -> pre-apply mix

and returns the three results of that pass (``post_mix``, ``comb_mix``,
``layer_input``).  ``T`` and ``H`` are compile-time specialisations of that
single definition (they fix the UB layout and the hand-unrolled pre-apply
pipeline); the host side validates shapes, allocates the three outputs and
launches once.

Layout conventions (all vector tiles live in UB)
------------------------------------------------
* one fp32 vector register holds 64 lanes; a ``[tb, 64]`` fp32 tile therefore
  has ``row == token`` and ``lane == mix index``.
* ``mix``  : lane 0..23 = mixes[t, 0..23]  (valid [tb, 24])
* ``pre``  : lane 0..3  = pre[t, m]
* ``post`` : lane 0..3  = post[t, m]
* ``p0..p3``: lane c    = comb[t, r, c]    (valid [tb, 4]); lanes 4..63 carry
  values that never reach the output.
* ``res``  : bf16 ``[4*tb, h_blk]``, row ``4*t + m`` is token ``t`` / mix ``m``.
"""

from dataclasses import dataclass

import torch
import torch_npu  # noqa: F401

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import get_platform_info

MHC = 4
MHC2 = 16
MHC3 = 24
TB_DEFAULT = 8          # token block used whenever it divides T
PIPE_DEPTH = 5          # residual chunks kept in flight, also the unroll depth
_VR = 64                # fp32 lanes per vector register
# Unified buffer of one vector core, from the CANN platform config for this
# SoC (``Ascend950PR_9589.ini``, ``[VectorCoreSpec] ub_size``).
_UB_BYTES = 253952
_STREAM_BASE = 0xC000   # first UB offset used by the residual streaming tiles
OP_NAME = "mhc_pre_fuse"


@dataclass(frozen=True)
class Knobs:
    """Scalar switches of the mHC pre pass, named after the operator contract."""

    rms_eps: float = 1e-6
    mhc_pre_eps: float = 1e-6
    mhc_sinkhorn_eps: float = 1e-6
    mhc_post_mult_value: float = 1.0
    sinkhorn_repeat: int = 10


_DEFAULT_KNOBS = Knobs()


@dataclass(frozen=True)
class _Spec:
    """Everything one kernel instance is specialised on."""

    t: int
    hidden: int
    ncore: int
    knobs: Knobs = _DEFAULT_KNOBS


def _vector_core_count():
    """Vector core count from the PyPTO-Pro platform info (not hard-coded)."""
    info = get_platform_info()
    count = int(info.vector_core_num)
    if count <= 0:
        raise RuntimeError(
            "cannot determine the NPU vector core count from the PyPTO-Pro platform info"
        )
    return count


# ---------------------------------------------------------------------------
# vector functions
# ---------------------------------------------------------------------------
@pl.vector_function
def vf_rms_sq(mix, sqv, inv, eps, ntb):
    """mix[t, k] *= 1/sqrt(sq[t] * inv + eps); rms is computed in-register."""
    pall = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    one = vf.full(1.0, pall, dtype=pl.DT_FP32)
    for t in pl.range(0, ntb, 1):
        s = vf.load_align(sqv, t, dist=pl.LoadDist.BRC_B32)
        s = vf.muls(s, inv, pall)
        s = vf.adds(s, eps, pall)
        s = vf.sqrt(s, pall)
        r = vf.div(one, s, pall)
        v = vf.load_align(mix, t * _VR)
        v = vf.mul(v, r, pall)
        vf.store_align(mix, v, pall, t * _VR)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def vf_pre(dst, mix, sct, bst, eps, ntb):
    """The pre-branch gate of every token.

    ``sct`` slot 0 holds ``mhc_scale[0]`` and ``bst`` the 24 base values; both
    are broadcast to all 64 lanes, so lane ``m`` ends up with the pre mix ``m``
    of the current token.  The sigmoid is evaluated as ``1 / (1 + exp(-x))``
    and the branch constant is added after it.
    """
    pall = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    v4 = pall
    zero = vf.full(0.0, pall, dtype=pl.DT_FP32)
    one = vf.full(1.0, pall, dtype=pl.DT_FP32)
    s = vf.load_align(sct, 0, dist=pl.LoadDist.BRC_B32)
    bv = vf.load_align(bst, 0)
    for t in pl.range(0, ntb, 1):
        x = vf.load_align(mix, t * _VR + 0)
        v = vf.mul(x, s, pall)
        v = vf.add(v, bv, pall)
        e = vf.exp_sub(zero, v, pall)
        d = vf.add(e, one, pall)
        g = vf.div(one, d, pall)
        g = vf.adds(g, eps, v4)
        vf.store_align(dst, g, v4, t * _VR)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def vf_post(dst, mix, sct, bst, mult, ntb):
    """The post-branch gate of every token, scaled by ``mult``.

    The post branch reads mixes 4..7, so the lane index starts at ``MHC``
    instead of 0 while the scale slot moves on to ``mhc_scale[1]``.  The base
    values are gathered with the same shift.  The product with the branch
    multiplier replaces the additive constant of the pre branch.
    """
    pall = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    v4 = pall
    zero = vf.full(0.0, pall, dtype=pl.DT_FP32)
    one = vf.full(1.0, pall, dtype=pl.DT_FP32)
    pi = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    idx = vf.arange(0, dtype=pl.DT_UINT32)
    s = vf.load_align(sct, 1, dist=pl.LoadDist.BRC_B32)
    biv = vf.adds(idx, MHC, pi)
    bv = vf.gather(bst, biv, pall)
    for t in pl.range(0, ntb, 1):
        ix = vf.adds(idx, t * _VR + MHC, pi)
        x = vf.gather(mix, ix, pall)
        v = vf.mul(x, s, pall)
        v = vf.add(v, bv, pall)
        e = vf.exp_sub(zero, v, pall)
        d = vf.add(e, one, pall)
        g = vf.div(one, d, pall)
        g = vf.muls(g, mult, v4)
        vf.store_align(dst, g, v4, t * _VR)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def vf_comb(dst, mix, sct, bst, r, ntb):
    """One row of the comb matrix, before Sinkhorn.

    Row ``r`` reads mixes ``8 + 4r .. 8 + 4r + 3``, so lanes 0..3 of the
    destination carry ``comb[t, r, 0..3]``.  The vector store fills the whole
    register, therefore the lanes above 3 hold gathered neighbours; they are
    never read back as matrix elements.
    """
    pall = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pi = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    idx = vf.arange(0, dtype=pl.DT_UINT32)
    soff = MHC * 2 + r * MHC
    s = vf.load_align(sct, 2, dist=pl.LoadDist.BRC_B32)
    biv = vf.adds(idx, soff, pi)
    bv = vf.gather(bst, biv, pall)
    for t in pl.range(0, ntb, 1):
        ix = vf.adds(idx, t * _VR + soff, pi)
        x = vf.gather(mix, ix, pall)
        v = vf.mul(x, s, pall)
        v = vf.add(v, bv, pall)
        vf.store_align(dst, v, pall, t * _VR)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def vf_sinkhorn_step(srt, eps, first, ntb):
    """One row-then-column normalisation pass over every token's 4x4 matrix.

    ``srt`` keeps the four rows of a token in four consecutive registers, so a
    row sum only has to read the four lanes ``ro + 0 .. ro + 3``; the resulting
    divisor is the same on every lane, and the pairwise tree fixes the FP32
    rounding order.

    ``first`` selects the contract's opening pass: it divides by the plain row
    sum and adds eps afterwards, while every later pass adds eps to the row sum
    first.  Both end with the same column normalisation, which is why the
    column sums and the row sums do not carry the same residual.
    """
    pall = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for t in pl.range(0, ntb, 1):
        ro = t * 4 * _VR
        a = vf.load_align(srt, ro)
        b = vf.load_align(srt, ro + _VR)
        c = vf.load_align(srt, ro + 2 * _VR)
        d = vf.load_align(srt, ro + 3 * _VR)
        z0 = vf.load_align(srt, ro + 0, dist=pl.LoadDist.BRC_B32)
        z1 = vf.load_align(srt, ro + 1, dist=pl.LoadDist.BRC_B32)
        z2 = vf.load_align(srt, ro + 2, dist=pl.LoadDist.BRC_B32)
        z3 = vf.load_align(srt, ro + 3, dist=pl.LoadDist.BRC_B32)
        u01 = vf.add(z0, z1, pall)
        u23 = vf.add(z2, z3, pall)
        s_a = vf.add(u01, u23, pall)
        z0 = vf.load_align(srt, ro + _VR + 0, dist=pl.LoadDist.BRC_B32)
        z1 = vf.load_align(srt, ro + _VR + 1, dist=pl.LoadDist.BRC_B32)
        z2 = vf.load_align(srt, ro + _VR + 2, dist=pl.LoadDist.BRC_B32)
        z3 = vf.load_align(srt, ro + _VR + 3, dist=pl.LoadDist.BRC_B32)
        u01 = vf.add(z0, z1, pall)
        u23 = vf.add(z2, z3, pall)
        s_b = vf.add(u01, u23, pall)
        z0 = vf.load_align(srt, ro + 2 * _VR + 0, dist=pl.LoadDist.BRC_B32)
        z1 = vf.load_align(srt, ro + 2 * _VR + 1, dist=pl.LoadDist.BRC_B32)
        z2 = vf.load_align(srt, ro + 2 * _VR + 2, dist=pl.LoadDist.BRC_B32)
        z3 = vf.load_align(srt, ro + 2 * _VR + 3, dist=pl.LoadDist.BRC_B32)
        u01 = vf.add(z0, z1, pall)
        u23 = vf.add(z2, z3, pall)
        s_c = vf.add(u01, u23, pall)
        z0 = vf.load_align(srt, ro + 3 * _VR + 0, dist=pl.LoadDist.BRC_B32)
        z1 = vf.load_align(srt, ro + 3 * _VR + 1, dist=pl.LoadDist.BRC_B32)
        z2 = vf.load_align(srt, ro + 3 * _VR + 2, dist=pl.LoadDist.BRC_B32)
        z3 = vf.load_align(srt, ro + 3 * _VR + 3, dist=pl.LoadDist.BRC_B32)
        u01 = vf.add(z0, z1, pall)
        u23 = vf.add(z2, z3, pall)
        s_d = vf.add(u01, u23, pall)
        if first:
            a = vf.div(a, s_a, pall)
            b = vf.div(b, s_b, pall)
            c = vf.div(c, s_c, pall)
            d = vf.div(d, s_d, pall)
            a = vf.adds(a, eps, pall)
            b = vf.adds(b, eps, pall)
            c = vf.adds(c, eps, pall)
            d = vf.adds(d, eps, pall)
        else:
            s_a = vf.adds(s_a, eps, pall)
            s_b = vf.adds(s_b, eps, pall)
            s_c = vf.adds(s_c, eps, pall)
            s_d = vf.adds(s_d, eps, pall)
            a = vf.div(a, s_a, pall)
            b = vf.div(b, s_b, pall)
            c = vf.div(c, s_c, pall)
            d = vf.div(d, s_d, pall)
        cs = vf.add(a, b, pall)
        cs = vf.add(cs, c, pall)
        cs = vf.add(cs, d, pall)
        cs = vf.adds(cs, eps, pall)
        a = vf.div(a, cs, pall)
        b = vf.div(b, cs, pall)
        c = vf.div(c, cs, pall)
        d = vf.div(d, cs, pall)
        vf.store_align(srt, a, pall, ro)
        vf.store_align(srt, b, pall, ro + _VR)
        vf.store_align(srt, c, pall, ro + 2 * _VR)
        vf.store_align(srt, d, pall, ro + 3 * _VR)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def vf_sinkhorn(pa, pb, pc, pd, srt, eps, extra, ntb):
    """Alternating row / column normalisation of the 4x4 comb matrix.

    Each row is first shifted down by its own maximum before ``exp``, the
    stabilisation the operator definition prescribes.  The shift is constant
    along a row, so it cancels again in the row normalisation and leaves the
    result unchanged.

    ``srt`` is a ``[4*tb, 64]`` scratch tile; slots 4t..4t+3 hold the four
    matrix rows of token t, one row per 64-lane register.  The opening pass and
    the ``extra`` follow-up passes then run over that scratch tile.
    """
    pall = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    row4 = vf.create_mask(pattern=pl.MaskPattern.VL4, dtype=pl.DT_FP32)
    pi = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    zidx = vf.full(0, pi, dtype=pl.DT_UINT32)   # every lane reads lane 0

    # ---- exp(comb - rowmax) into the scratch buffer ----
    for t in pl.range(0, ntb, 1):
        a = vf.load_align(pa, t * _VR)
        b = vf.load_align(pb, t * _VR)
        c = vf.load_align(pc, t * _VR)
        d = vf.load_align(pd, t * _VR)
        rm = vf.reduce_max(a, row4)
        rs = vf.gather(rm, zidx)
        a = vf.exp_sub(a, rs, pall)
        rm = vf.reduce_max(b, row4)
        rs = vf.gather(rm, zidx)
        b = vf.exp_sub(b, rs, pall)
        rm = vf.reduce_max(c, row4)
        rs = vf.gather(rm, zidx)
        c = vf.exp_sub(c, rs, pall)
        rm = vf.reduce_max(d, row4)
        rs = vf.gather(rm, zidx)
        d = vf.exp_sub(d, rs, pall)
        vf.store_align(srt, a, pall, (t * 4 + 0) * _VR)
        vf.store_align(srt, b, pall, (t * 4 + 1) * _VR)
        vf.store_align(srt, c, pall, (t * 4 + 2) * _VR)
        vf.store_align(srt, d, pall, (t * 4 + 3) * _VR)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)

    # the opening pass carries no eps inside the row normalisation
    vf_sinkhorn_step(srt, eps, True, ntb)
    for _it in pl.range(0, extra, 1):
        vf_sinkhorn_step(srt, eps, False, ntb)

    # ---- move the result back to pa..pd ----
    for t in pl.range(0, ntb, 1):
        a = vf.load_align(srt, (t * 4 + 0) * _VR)
        b = vf.load_align(srt, (t * 4 + 1) * _VR)
        c = vf.load_align(srt, (t * 4 + 2) * _VR)
        d = vf.load_align(srt, (t * 4 + 3) * _VR)
        vf.store_align(pa, a, pall, t * _VR)
        vf.store_align(pb, b, pall, t * _VR)
        vf.store_align(pc, c, pall, t * _VR)
        vf.store_align(pd, d, pall, t * _VR)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def vf_apply(dst, x, pre, nchunk, ncol, ntb):
    """Weighted sum of the four residual mixes, written back as bf16.

    ``x`` holds the bf16 residual stream (row ``4t + m`` per mix) and ``pre``
    the four gate values of token ``t``.  The products are accumulated in FP32
    with the same order the contract prescribes, then rounded to bf16 once.
    """
    pall = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for t in pl.range(0, ntb, 1):
        p0 = vf.load_align(pre, t * _VR + 0, dist=pl.LoadDist.BRC_B32)
        p1 = vf.load_align(pre, t * _VR + 1, dist=pl.LoadDist.BRC_B32)
        p2 = vf.load_align(pre, t * _VR + 2, dist=pl.LoadDist.BRC_B32)
        p3 = vf.load_align(pre, t * _VR + 3, dist=pl.LoadDist.BRC_B32)
        for cc in pl.range(0, nchunk, 1):
            col = cc * _VR
            x0 = vf.load_align(x, (t * 4 + 0) * ncol + col,
                               dist=pl.LoadDist.UNPK_B16, dtype=pl.DT_BF16)
            x1 = vf.load_align(x, (t * 4 + 1) * ncol + col,
                               dist=pl.LoadDist.UNPK_B16, dtype=pl.DT_BF16)
            x2 = vf.load_align(x, (t * 4 + 2) * ncol + col,
                               dist=pl.LoadDist.UNPK_B16, dtype=pl.DT_BF16)
            x3 = vf.load_align(x, (t * 4 + 3) * ncol + col,
                               dist=pl.LoadDist.UNPK_B16, dtype=pl.DT_BF16)
            a0 = vf.astype(x0, pall, dtype=pl.DT_FP32)
            a1 = vf.astype(x1, pall, dtype=pl.DT_FP32)
            a2 = vf.astype(x2, pall, dtype=pl.DT_FP32)
            a3 = vf.astype(x3, pall, dtype=pl.DT_FP32)
            acc = vf.mul(a0, p0, pall)
            acc = vf.mul_add_dst(a1, p1, pall)
            acc = vf.mul_add_dst(a2, p2, pall)
            acc = vf.mul_add_dst(a3, p3, pall)
            y = vf.astype(acc, pall, dtype=pl.DT_BF16)
            vf.store_align(dst, y, pall, t * ncol + col, dist=pl.StoreDist.PACK_B32)
    # vector stores visible to MTE3, vector loads retired before MTE2 reloads
    vf.mem_bar(mode=pl.MemBarMode.VST_LD)
    vf.mem_bar(mode=pl.MemBarMode.VLD_ST)


# ---------------------------------------------------------------------------
# kernel definition
# ---------------------------------------------------------------------------
def _pick_h_blk(hidden, tb):
    """Widest hidden chunk that works for this shape.

    Candidates are the multiples of 64 listed below, widest first.  A chunk is
    accepted when it divides ``hidden``, when at least ``PIPE_DEPTH`` chunks
    remain (the residual pipeline below has that many slots), and when the
    streaming tiles still fit into UB next to the resident buffers.  The widest
    combination that can reach the check (``c = 512`` with ``tb = 8``) lands
    exactly on ``_UB_BYTES``.
    """
    for c in (512, 256, 192, 128, 64):
        if hidden % c:
            continue
        if hidden // c < PIPE_DEPTH:
            continue
        if _STREAM_BASE + PIPE_DEPTH * (tb * 4 * c * 2) + PIPE_DEPTH * (tb * c * 2) > _UB_BYTES:
            continue
        return c
    raise ValueError(
        f"hidden={hidden} not supported: it must be a multiple of 64 with "
        f"hidden / 64 >= {PIPE_DEPTH}, so that the residual stream has at "
        f"least {PIPE_DEPTH} chunks"
    )


def _pick_tb(t):
    """Token block: TB_DEFAULT whenever it divides T (at T=512 that gives one
    block per vector core); otherwise the largest divisor among 4 / 2 / 1, so
    that no token is ever dropped."""
    for c in (TB_DEFAULT, 4, 2):
        if t % c == 0:
            return c
    return 1


def _make_kernel(spec):
    """Instantiate the single kernel definition for one ``_Spec``."""
    t, hidden, ncore = spec.t, spec.hidden, spec.ncore
    rms_eps = spec.knobs.rms_eps
    mhc_pre_eps = spec.knobs.mhc_pre_eps
    mhc_sinkhorn_eps = spec.knobs.mhc_sinkhorn_eps
    mhc_post_mult_value = spec.knobs.mhc_post_mult_value
    sinkhorn_repeat = spec.knobs.sinkhorn_repeat
    tb = _pick_tb(t)
    h_blk = _pick_h_blk(hidden, tb)
    n_t = t // tb
    n_ch = hidden // h_blk
    n_chunk = h_blk // _VR
    inv_group = 1.0 / float(MHC * hidden)
    # The unrolled residual pipeline advances its prefetch by PIPE_DEPTH slots,
    # so the main loop below is written for that depth; _pick_h_blk guarantees
    # n_ch >= PIPE_DEPTH.
    if n_ch < PIPE_DEPTH:
        raise ValueError(
            f"hidden={hidden} with h_blk={h_blk} leaves only {n_ch} chunks, but "
            f"the residual pipeline needs {PIPE_DEPTH}"
        )
    xsz = tb * MHC * h_blk * 2
    osz = tb * h_blk * 2
    # plain python lists: the DSL parser rejects comprehensions in-kernel
    xg_addrs = []
    xg_mtx = []
    ob_addrs = []
    ob_mtx = []
    for _i in range(PIPE_DEPTH):
        xg_addrs.append(_STREAM_BASE + _i * xsz)
        xg_mtx.append(11 + _i)
        ob_addrs.append(_STREAM_BASE + PIPE_DEPTH * xsz + _i * osz)
        ob_mtx.append(11 + PIPE_DEPTH + _i)
    n_round = (n_ch - PIPE_DEPTH) // PIPE_DEPTH
    base_slot = n_round * PIPE_DEPTH
    r_rem = n_ch - base_slot - PIPE_DEPTH
    if not 0 <= r_rem < PIPE_DEPTH:
        raise ValueError(f"residual tail length {r_rem} is out of range")

    @pl.jit(auto_mutex=True, name=OP_NAME)
    def mhc_pre_fuse_kernel(
        mul: pl.Tensor[[t, MHC3], pl.DT_FP32],
        sq: pl.Tensor[[t], pl.DT_FP32],
        scale: pl.Tensor[[3], pl.DT_FP32],
        base: pl.Tensor[[MHC3], pl.DT_FP32],
        res: pl.Tensor[[t * MHC, hidden], pl.DT_BF16],
        post_mix: pl.Tensor[[t, MHC], pl.DT_FP32],
        comb_mix: pl.Tensor[[t, MHC2], pl.DT_FP32],
        layer_input: pl.Tensor[[t, hidden], pl.DT_BF16],
    ):
        tt_16_64 = pl.TileType(shape=[tb, _VR], dtype=pl.DT_FP32,
                               target_memory=pl.MemorySpace.Vec)
        tt_1_64 = pl.TileType(shape=[1, _VR], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Vec)
        tt_64_64 = pl.TileType(shape=[tb * MHC, _VR], dtype=pl.DT_FP32,
                               target_memory=pl.MemorySpace.Vec)
        tt_x = pl.TileType(shape=[tb * MHC, h_blk], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Vec)
        tt_ob = pl.TileType(shape=[tb, h_blk], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Vec)

        with pl.section_vector():
            mix = pl.make_tile_group(type=tt_16_64, addrs=0x0000, mutex_ids=[0]).current()
            pre_t = pl.make_tile_group(type=tt_16_64, addrs=0x1000, mutex_ids=[1]).current()
            post_t = pl.make_tile_group(type=tt_16_64, addrs=0x2000, mutex_ids=[2]).current()
            p0 = pl.make_tile_group(type=tt_16_64, addrs=0x3000, mutex_ids=[3]).current()
            p1 = pl.make_tile_group(type=tt_16_64, addrs=0x4000, mutex_ids=[4]).current()
            p2 = pl.make_tile_group(type=tt_16_64, addrs=0x5000, mutex_ids=[5]).current()
            p3 = pl.make_tile_group(type=tt_16_64, addrs=0x6000, mutex_ids=[6]).current()
            sqv = pl.make_tile_group(type=tt_1_64, addrs=0x7000, mutex_ids=[7]).current()
            sct = pl.make_tile_group(type=tt_1_64, addrs=0x7100, mutex_ids=[8]).current()
            bst = pl.make_tile_group(type=tt_1_64, addrs=0x7200, mutex_ids=[9]).current()
            srt = pl.make_tile_group(type=tt_64_64, addrs=0x8000, mutex_ids=[10]).current()
            # Resident buffers occupy the low 40 KiB (srt ends at 0xA000); the
            # streaming tiles start at _STREAM_BASE so the two never overlap.
            xg = pl.make_tile_group(type=tt_x, addrs=xg_addrs,
                                    mutex_ids=xg_mtx, depth=PIPE_DEPTH)
            o_b = pl.make_tile_group(type=tt_ob, addrs=ob_addrs,
                                     mutex_ids=ob_mtx, depth=PIPE_DEPTH)

            pl.set_validshape(mix, [tb, MHC3])
            pl.set_validshape(pre_t, [tb, MHC])
            pl.set_validshape(post_t, [tb, MHC])
            pl.set_validshape(p0, [tb, MHC])
            pl.set_validshape(p1, [tb, MHC])
            pl.set_validshape(p2, [tb, MHC])
            pl.set_validshape(p3, [tb, MHC])
            pl.set_validshape(sqv, [1, tb])
            pl.set_validshape(sct, [1, 3])
            pl.set_validshape(bst, [1, MHC3])

            core = pl.get_block_idx()

            pl.load(sct, scale, [0])
            pl.load(bst, base, [0])
            pl.system.bar_all()

            for blk in pl.range(core, n_t, ncore):
                t0 = blk * tb

                # Open the residual stream *before* the prologue: the small
                # mul/sq loads retire first, then MTE2 keeps draining chunks
                # 0..2 while the vector pipe runs rms / pre / post / comb /
                # sinkhorn.  This is what hides the per-block prologue.
                pl.load(mix, mul, [t0, 0])
                pl.load(sqv, sq, [t0])
                pl.load(xg[0], res, [t0 * MHC, 0 * h_blk])
                pl.load(xg[1], res, [t0 * MHC, 1 * h_blk])
                pl.load(xg[2], res, [t0 * MHC, 2 * h_blk])
                pl.load(xg[3], res, [t0 * MHC, 3 * h_blk])
                pl.load(xg[4], res, [t0 * MHC, 4 * h_blk])

                vf_rms_sq(mix, sqv, inv_group, rms_eps, tb)
                vf_pre(pre_t, mix, sct, bst, mhc_pre_eps, tb)

                vf_post(post_t, mix, sct, bst, mhc_post_mult_value, tb)
                vf_comb(p0, mix, sct, bst, 0, tb)
                vf_comb(p1, mix, sct, bst, 1, tb)
                vf_comb(p2, mix, sct, bst, 2, tb)
                vf_comb(p3, mix, sct, bst, 3, tb)

                vf_sinkhorn(p0, p1, p2, p3, srt, mhc_sinkhorn_eps, sinkhorn_repeat - 1, tb)
                pl.system.bar_all()

                pl.store(post_mix, post_t, [t0, 0])
                pl.store(comb_mix, p0, [t0, 0 * MHC])
                pl.store(comb_mix, p1, [t0, 1 * MHC])
                pl.store(comb_mix, p2, [t0, 2 * MHC])
                pl.store(comb_mix, p3, [t0, 3 * MHC])

                # PIPE_DEPTH-deep software pipeline
                for r in pl.range(0, n_round, 1):
                    h = r * PIPE_DEPTH
                    vf_apply(o_b[0], xg[0], pre_t, n_chunk, h_blk, tb)
                    pl.store(layer_input, o_b[0], [t0, (h + 0) * h_blk])
                    pl.load(xg[0], res, [t0 * MHC, (h + PIPE_DEPTH + 0) * h_blk])
                    vf_apply(o_b[1], xg[1], pre_t, n_chunk, h_blk, tb)
                    pl.store(layer_input, o_b[1], [t0, (h + 1) * h_blk])
                    pl.load(xg[1], res, [t0 * MHC, (h + PIPE_DEPTH + 1) * h_blk])
                    vf_apply(o_b[2], xg[2], pre_t, n_chunk, h_blk, tb)
                    pl.store(layer_input, o_b[2], [t0, (h + 2) * h_blk])
                    pl.load(xg[2], res, [t0 * MHC, (h + PIPE_DEPTH + 2) * h_blk])
                    vf_apply(o_b[3], xg[3], pre_t, n_chunk, h_blk, tb)
                    pl.store(layer_input, o_b[3], [t0, (h + 3) * h_blk])
                    pl.load(xg[3], res, [t0 * MHC, (h + PIPE_DEPTH + 3) * h_blk])
                    vf_apply(o_b[4], xg[4], pre_t, n_chunk, h_blk, tb)
                    pl.store(layer_input, o_b[4], [t0, (h + 4) * h_blk])
                    pl.load(xg[4], res, [t0 * MHC, (h + PIPE_DEPTH + 4) * h_blk])
                vf_apply(o_b[0], xg[0], pre_t, n_chunk, h_blk, tb)
                pl.store(layer_input, o_b[0], [t0, (base_slot + 0) * h_blk])
                if r_rem >= 1:
                    pl.load(xg[0], res, [t0 * MHC, (base_slot + PIPE_DEPTH + 0) * h_blk])
                vf_apply(o_b[1], xg[1], pre_t, n_chunk, h_blk, tb)
                pl.store(layer_input, o_b[1], [t0, (base_slot + 1) * h_blk])
                if r_rem >= 2:
                    pl.load(xg[1], res, [t0 * MHC, (base_slot + PIPE_DEPTH + 1) * h_blk])
                vf_apply(o_b[2], xg[2], pre_t, n_chunk, h_blk, tb)
                pl.store(layer_input, o_b[2], [t0, (base_slot + 2) * h_blk])
                if r_rem >= 3:
                    pl.load(xg[2], res, [t0 * MHC, (base_slot + PIPE_DEPTH + 2) * h_blk])
                vf_apply(o_b[3], xg[3], pre_t, n_chunk, h_blk, tb)
                pl.store(layer_input, o_b[3], [t0, (base_slot + 3) * h_blk])
                if r_rem >= 4:
                    pl.load(xg[3], res, [t0 * MHC, (base_slot + PIPE_DEPTH + 3) * h_blk])
                vf_apply(o_b[4], xg[4], pre_t, n_chunk, h_blk, tb)
                pl.store(layer_input, o_b[4], [t0, (base_slot + 4) * h_blk])
                if r_rem >= 1:
                    vf_apply(o_b[0], xg[0], pre_t, n_chunk, h_blk, tb)
                    pl.store(layer_input, o_b[0], [t0, (base_slot + PIPE_DEPTH + 0) * h_blk])
                if r_rem >= 2:
                    vf_apply(o_b[1], xg[1], pre_t, n_chunk, h_blk, tb)
                    pl.store(layer_input, o_b[1], [t0, (base_slot + PIPE_DEPTH + 1) * h_blk])
                if r_rem >= 3:
                    vf_apply(o_b[2], xg[2], pre_t, n_chunk, h_blk, tb)
                    pl.store(layer_input, o_b[2], [t0, (base_slot + PIPE_DEPTH + 2) * h_blk])
                if r_rem >= 4:
                    vf_apply(o_b[3], xg[3], pre_t, n_chunk, h_blk, tb)
                    pl.store(layer_input, o_b[3], [t0, (base_slot + PIPE_DEPTH + 3) * h_blk])

            # one drain for the whole kernel instead of one per token block
            pl.system.bar_all()

    return mhc_pre_fuse_kernel


_KERNELS = {}


def _kernel_for(spec):
    """Compile the kernel once per ``_Spec`` and cache it."""
    kernel = _KERNELS.get(spec)
    if kernel is None:
        kernel = _make_kernel(spec)
        _KERNELS[spec] = kernel
    return kernel


def mhc_pre_fuse_wrapper(gemm_out_mul, gemm_out_sqrsum, mhc_scale, mhc_base,
                         residual, knobs=_DEFAULT_KNOBS):
    """Public wrapper: validate, allocate the three outputs and launch once.

    Args:
        gemm_out_mul: [T, 24] float32.  A leading split axis is accepted and
            its slot 0 is used, because the Ascend contract reduces the split-K
            partials inside the preceding GEMM (n_splits == 1).
        gemm_out_sqrsum: [T] float32 (a leading split axis is accepted too).
        mhc_scale: [3] float32.
        mhc_base: [24] float32.
        residual: [T, 4, hidden] bfloat16.
        knobs: the five scalar switches of the pass, see ``Knobs``.

    Returns:
        (post_mix, comb_mix, layer_input): [T, 4] float32, [T, 16] float32,
        [T, hidden] bfloat16.
    """
    if gemm_out_mul.dim() == 3:
        gemm_out_mul = gemm_out_mul[0]
    if gemm_out_sqrsum.dim() == 2:
        gemm_out_sqrsum = gemm_out_sqrsum[0]
    t, mhc, hidden = residual.shape
    if mhc != MHC:
        raise ValueError(f"residual mix axis is {mhc}, expected {MHC}")
    if list(gemm_out_mul.shape) != [t, MHC3] or list(gemm_out_sqrsum.shape) != [t]:
        raise ValueError("gemm_out_mul / gemm_out_sqrsum shapes do not match residual")
    if list(mhc_scale.shape) != [3] or list(mhc_base.shape) != [MHC3]:
        raise ValueError("mhc_scale / mhc_base shapes do not match the mHC contract")
    if knobs.sinkhorn_repeat < 1:
        raise ValueError(f"sinkhorn_repeat must be >= 1, got {knobs.sinkhorn_repeat}")

    post_mix = torch.empty((t, MHC), dtype=torch.float32, device=residual.device)
    comb_mix = torch.empty((t, MHC2), dtype=torch.float32, device=residual.device)
    layer_input = torch.empty((t, hidden), dtype=torch.bfloat16, device=residual.device)

    ncore = _vector_core_count()
    kernel = _kernel_for(_Spec(t=t, hidden=hidden, ncore=ncore, knobs=knobs))
    kernel[None, ncore](
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual.view(t * MHC, hidden),
        post_mix,
        comb_mix,
        layer_input,
    )
    return post_mix, comb_mix, layer_input
