# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""mega-mHC P2 delivery kernel; arithmetic and synchronization from the candidate.

One binary entry, eight legal specializations: seven batch16 widths and batch8 H7168.
Workspace packs Q, inverse RMS and range flags at 512-byte-aligned boundaries.
"""

import dataclasses
import math
import torch
import pypto
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf
from pypto_pro.runtime.tilingkey import TilingKeyField

try:  # JIT/package import
    from .mega_mhc_common import (
        F32,
        B16,
        CHUNK,
        SLOW_BATCH,
        SLOW_CHUNK,
        _slow_dot_first,
        _slow_dot_accum,
        _slow_dot_last,
        _slow_finish_pair,
        _slow_dot_last_finish_single,
        range_init,
        range_panel,
        range_record,
        range_merge,
        pack_fn_bf2_128,
        _partial_record,
        _prefetch_produce,
        _overlap_project,
        _norm_chunk_rows,
    )
except ImportError:  # binary codegen loads this file as a standalone module
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from mega_mhc_common import (
        F32,
        B16,
        CHUNK,
        SLOW_BATCH,
        SLOW_CHUNK,
        _slow_dot_first,
        _slow_dot_accum,
        _slow_dot_last,
        _slow_finish_pair,
        _slow_dot_last_finish_single,
        range_init,
        range_panel,
        range_record,
        range_merge,
        pack_fn_bf2_128,
        _partial_record,
        _prefetch_produce,
        _overlap_project,
        _norm_chunk_rows,
    )


class MegaMhcKey:
    Width = TilingKeyField(bits=3, values=[1024, 2048, 3072, 4096, 5120, 6144, 7168])
    BlockJobs = TilingKeyField(bits=2, values=[7, 8, 27])

    @classmethod
    def is_valid(cls, key):
        width, jobs = key
        return jobs == (7 if width == 7168 else 8) or (width == 7168 and jobs == 27)


@dataclasses.dataclass
class MegaMhcTiling:
    tokens: int
    rows: int
    inv_offset: int
    flags_offset: int
    repeat: int
    hc_eps: float
    pre_eps: float
    post_scale: float
    sink_eps: float
    norm_eps: float


BATCH = 16


BATCH8 = 8


SLOW_SLOTS = 3


SLOW_BANKS = 2


SLOW_LEAD = 2


SLOW_PREFETCH = 0


@pl.vector_function
def _slow_post_stream(r, rf, x, u, post, comb, prev, stat, partial, offset, invh, inv4h, hc_eps, norm_eps):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=B16)
    pu = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    c00 = vf.load_align(comb, 0, dist=pl.LoadDist.BRC_B32)
    c01 = vf.load_align(comb, 1, dist=pl.LoadDist.BRC_B32)
    c02 = vf.load_align(comb, 2, dist=pl.LoadDist.BRC_B32)
    c03 = vf.load_align(comb, 3, dist=pl.LoadDist.BRC_B32)
    c10 = vf.load_align(comb, 4, dist=pl.LoadDist.BRC_B32)
    c11 = vf.load_align(comb, 5, dist=pl.LoadDist.BRC_B32)
    c12 = vf.load_align(comb, 6, dist=pl.LoadDist.BRC_B32)
    c13 = vf.load_align(comb, 7, dist=pl.LoadDist.BRC_B32)
    c20 = vf.load_align(comb, 8, dist=pl.LoadDist.BRC_B32)
    c21 = vf.load_align(comb, 9, dist=pl.LoadDist.BRC_B32)
    c22 = vf.load_align(comb, 10, dist=pl.LoadDist.BRC_B32)
    c23 = vf.load_align(comb, 11, dist=pl.LoadDist.BRC_B32)
    c30 = vf.load_align(comb, 12, dist=pl.LoadDist.BRC_B32)
    c31 = vf.load_align(comb, 13, dist=pl.LoadDist.BRC_B32)
    c32 = vf.load_align(comb, 14, dist=pl.LoadDist.BRC_B32)
    c33 = vf.load_align(comb, 15, dist=pl.LoadDist.BRC_B32)
    squares = vf.full(0.0, p, dtype=F32)
    usquares = vf.full(0.0, p, dtype=F32)
    if offset != 0:
        squares = vf.load_align(partial, [0, 0])
        usquares = vf.load_align(partial, [1, 0])
    for col in pl.range(0, SLOW_CHUNK, 64):
        xb = vf.load_align(x, col, dist=pl.LoadDist.UNPK_B16)
        b0 = vf.load_align(r, [0, col], dist=pl.LoadDist.UNPK_B16)
        b1 = vf.load_align(r, [1, col], dist=pl.LoadDist.UNPK_B16)
        b2 = vf.load_align(r, [2, col], dist=pl.LoadDist.UNPK_B16)
        b3 = vf.load_align(r, [3, col], dist=pl.LoadDist.UNPK_B16)
        xv = vf.astype(xb, p, dtype=F32)
        r0 = vf.astype(b0, p, dtype=F32)
        r1 = vf.astype(b1, p, dtype=F32)
        r2 = vf.astype(b2, p, dtype=F32)
        r3 = vf.astype(b3, p, dtype=F32)
        z0 = vf.mul(r0, c00, p)
        z0 = vf.mul_add_dst(r1, c10, p)
        z0 = vf.mul_add_dst(r2, c20, p)
        z0 = vf.mul_add_dst(r3, c30, p)
        pv = vf.load_align(post, 0, dist=pl.LoadDist.BRC_B32)
        q = vf.mul(xv, pv, p)
        z0 = vf.add(z0, q, p)
        z1 = vf.mul(r0, c01, p)
        z1 = vf.mul_add_dst(r1, c11, p)
        z1 = vf.mul_add_dst(r2, c21, p)
        z1 = vf.mul_add_dst(r3, c31, p)
        pv = vf.load_align(post, 1, dist=pl.LoadDist.BRC_B32)
        q = vf.mul(xv, pv, p)
        z1 = vf.add(z1, q, p)
        z2 = vf.mul(r0, c02, p)
        z2 = vf.mul_add_dst(r1, c12, p)
        z2 = vf.mul_add_dst(r2, c22, p)
        z2 = vf.mul_add_dst(r3, c32, p)
        pv = vf.load_align(post, 2, dist=pl.LoadDist.BRC_B32)
        q = vf.mul(xv, pv, p)
        z2 = vf.add(z2, q, p)
        z3 = vf.mul(r0, c03, p)
        z3 = vf.mul_add_dst(r1, c13, p)
        z3 = vf.mul_add_dst(r2, c23, p)
        z3 = vf.mul_add_dst(r3, c33, p)
        pv = vf.load_align(post, 3, dist=pl.LoadDist.BRC_B32)
        q = vf.mul(xv, pv, p)
        z3 = vf.add(z3, q, p)
        b0 = vf.astype(z0, p16, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        b1 = vf.astype(z1, p16, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        b2 = vf.astype(z2, p16, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        b3 = vf.astype(z3, p16, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        z0 = vf.astype(b0, p, dtype=F32)
        z1 = vf.astype(b1, p, dtype=F32)
        z2 = vf.astype(b2, p, dtype=F32)
        z3 = vf.astype(b3, p, dtype=F32)
        v0 = vf.bit_cast(b0, dtype=pl.DT_UINT32)
        v1 = vf.bit_cast(b1, dtype=pl.DT_UINT32)
        v2 = vf.bit_cast(b2, dtype=pl.DT_UINT32)
        v3 = vf.bit_cast(b3, dtype=pl.DT_UINT32)
        vf.store_align(r, v0, pu, [0, col], dist=pl.StoreDist.PACK)
        vf.store_align(r, v1, pu, [1, col], dist=pl.StoreDist.PACK)
        vf.store_align(r, v2, pu, [2, col], dist=pl.StoreDist.PACK)
        vf.store_align(r, v3, pu, [3, col], dist=pl.StoreDist.PACK)
        vf.store_align(rf, z0, p, [0, col])
        vf.store_align(rf, z1, p, [1, col])
        vf.store_align(rf, z2, p, [2, col])
        vf.store_align(rf, z3, p, [3, col])
        squares = vf.mul_add_dst(z0, z0, p)
        squares = vf.mul_add_dst(z1, z1, p)
        squares = vf.mul_add_dst(z2, z2, p)
        squares = vf.mul_add_dst(z3, z3, p)
        av = vf.load_align(prev, 0, dist=pl.LoadDist.BRC_B32)
        uv = vf.mul(z0, av, p)
        av = vf.load_align(prev, 1, dist=pl.LoadDist.BRC_B32)
        q = vf.mul(z1, av, p)
        uv = vf.add(uv, q, p)
        av = vf.load_align(prev, 2, dist=pl.LoadDist.BRC_B32)
        q = vf.mul(z2, av, p)
        uv = vf.add(uv, q, p)
        av = vf.load_align(prev, 3, dist=pl.LoadDist.BRC_B32)
        q = vf.mul(z3, av, p)
        uv = vf.add(uv, q, p)
        ub = vf.astype(uv, p16, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        ur = vf.astype(ub, p, dtype=F32)
        usquares = vf.mul_add_dst(ur, ur, p)
        packed = vf.bit_cast(ub, dtype=pl.DT_UINT32)
        vf.store_align(u, packed, pu, col, dist=pl.StoreDist.PACK)
    vf.store_align(partial, squares, p, [0, 0])
    vf.store_align(partial, usquares, p, [1, 0])
    if offset + SLOW_CHUNK == Width:
        ss = vf.reduce_sum(squares, p)
        mean = vf.muls(ss, inv4h, p)
        mean = vf.adds(mean, hc_eps, p)
        denom = vf.sqrt(mean, p)
        one = vf.full(1.0, p, dtype=F32)
        inv = vf.div(one, denom, p)
        vf.store_align(stat, inv, p, 0, dist=pl.StoreDist.FIRST_ELEMENT_B32)
        us = vf.reduce_sum(usquares, p)
        um = vf.muls(us, invh, p)
        um = vf.adds(um, norm_eps, p)
        ud = vf.sqrt(um, p)
        ui = vf.div(one, ud, p)
        vf.store_align(stat, ui, p, 1, dist=pl.StoreDist.FIRST_ELEMENT_B32)


@pl.vector_function
def _slow_norm_all(u, gamma, stats, count):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
    ph = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=B16)
    pu = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    for local in pl.range(count):
        inv = vf.load_align(stats, local * 8 + 1, dist=pl.LoadDist.BRC_B32)
        for col in pl.range(0, Width, 64):
            ub = vf.load_align(u, local * Width + col, dist=pl.LoadDist.UNPK_B16)
            gb = vf.load_align(gamma, col, dist=pl.LoadDist.UNPK_B16)
            uv = vf.astype(ub, p, dtype=F32)
            gv = vf.astype(gb, p, dtype=F32)
            y = vf.mul(uv, inv, p)
            y = vf.mul(y, gv, p)
            yb = vf.astype(y, ph, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
            packed = vf.bit_cast(yb, dtype=pl.DT_UINT32)
            vf.store_align(u, packed, pu, local * Width + col, dist=pl.StoreDist.PACK)


@pl.pipeline.stage
def _overlap_stash(job, shard, tokens, qwork, scoreg, partialg, recordg, batch):
    scores = scoreg.next()
    record = recordg.current()
    partial = partialg[job % 2]
    start = job * (2 * batch) + pl.get_subblock_idx() * batch
    count = pl.max(0, pl.min(batch, tokens - start))
    if count > 0:
        _partial_record(scores, partial, record, count)
        pl.set_validshape(record, [count, 32])
        pl.store(qwork, record, [start % (2 * batch * BlockJobs), shard, 0], order=[0, 2])


@pl.vector_function
def _merge_finish_quad_registers(
    records,
    scales,
    bases,
    newpre,
    newpost,
    newcomb,
    invout,
    count,
    invh,
    inv4h,
    hc_eps,
    norm_eps,
    pre_eps,
    post_scale,
    repeat,
    eps,
):
    pall = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
    pu = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    p1 = vf.update_mask(1, dtype=F32)
    p32 = vf.update_mask(32, dtype=F32)
    ids = vf.arange(0, dtype=pl.DT_UINT32)
    zero_id = vf.full(0, pu, dtype=pl.DT_UINT32)
    lowmask = vf.full(31, pu, dtype=pl.DT_UINT32)
    low = vf.and_(ids, lowmask, pu)
    high_index = vf.adds(low, 32, pu)
    zero = vf.full(0.0, pall, dtype=F32)
    i24 = vf.full(24, pu, dtype=pl.DT_UINT32)
    i25 = vf.full(25, pu, dtype=pl.DT_UINT32)
    mask15 = vf.full(15, pu, dtype=pl.DT_UINT32)
    comb_local = vf.and_(ids, mask15, pu)
    comb_qidx = vf.adds(comb_local, 8, pu)
    mask7 = vf.full(7, pu, dtype=pl.DT_UINT32)
    pp_local = vf.and_(ids, mask7, pu)
    qc = vf.full(0.0, pall, dtype=F32)
    qp = vf.full(0.0, pall, dtype=F32)
    ss = vf.full(0.0, pall, dtype=F32)
    su = vf.full(0.0, pall, dtype=F32)
    for token in pl.range(4):
        total = vf.full(0.0, pall, dtype=F32)
        if token < count:
            for pair_id in pl.range(Width // (2 * CHUNK)):
                value = vf.load_align(records, [token, pair_id * 64])
                total = vf.add(total, value, pall)
        upper = vf.gather(total, high_index)
        total = vf.add(total, upper, p32)
        cq = vf.gather(total, comb_qidx)
        pq = vf.gather(total, pp_local)
        tss = vf.gather(total, i24)
        tsu = vf.gather(total, i25)
        tss = vf.select(tss, zero, p1)
        tsu = vf.select(tsu, zero, p1)
        tss = vf.reduce_sum(tss, pall)
        tsu = vf.reduce_sum(tsu, pall)
        tss = vf.gather(tss, zero_id)
        tsu = vf.gather(tsu, zero_id)
        threshold_c = vf.full(token * 16, pu, dtype=pl.DT_UINT32)
        threshold_p = vf.full(token * 8, pu, dtype=pl.DT_UINT32)
        older_c = vf.lt(ids, threshold_c, pu)
        older_p = vf.lt(ids, threshold_p, pu)
        qc = vf.select(qc, cq, older_c)
        qp = vf.select(qp, pq, older_p)
        ss = vf.select(ss, tss, older_c)
        su = vf.select(su, tsu, older_c)
    inv = vf.muls(ss, inv4h, pall)
    inv = vf.adds(inv, hc_eps, pall)
    inv = vf.sqrt(inv, pall)
    one = vf.full(1.0, pall, dtype=F32)
    inv = vf.div(one, inv, pall)
    ur = vf.muls(su, invh, pall)
    ur = vf.adds(ur, norm_eps, pall)
    ur = vf.sqrt(ur, pall)
    ur = vf.div(one, ur, pall)
    p16 = vf.update_mask(16, dtype=F32)
    for inv_token in pl.range(count):
        ur_index = vf.full(inv_token * 16, pu, dtype=pl.DT_UINT32)
        ur_token = vf.gather(ur, ur_index)
        vf.store_align(invout, ur_token, p16, [inv_token, 0])
    rawbase = vf.load_align(bases, 0)
    basep = vf.gather(rawbase, pp_local)
    basec = vf.gather(rawbase, comb_qidx)
    mask31 = vf.full(31, pu, dtype=pl.DT_UINT32)
    pp_id = vf.and_(ids, mask31, pu)
    pp_inv_index = vf.shift_right(pp_id, 3, pu)
    pp_inv_index = vf.shift_left(pp_inv_index, 4, pu)
    pinv = vf.gather(inv, pp_inv_index)
    pcount_pp = vf.update_mask(count * 8, dtype=F32)
    c4 = vf.full(4, pu, dtype=pl.DT_UINT32)
    pre_lanes = vf.lt(pp_local, c4, pu)
    s0 = vf.load_align(scales, 0, dist=pl.LoadDist.BRC_B32)
    s1 = vf.load_align(scales, 1, dist=pl.LoadDist.BRC_B32)
    scale_pp = vf.select(s0, s1, pre_lanes)
    ppvalue = vf.mul(qp, pinv, pcount_pp)
    ppvalue = vf.mul(ppvalue, scale_pp, pcount_pp)
    ppvalue = vf.add(ppvalue, basep, pcount_pp)
    neg = vf.muls(ppvalue, -1.0, pcount_pp)
    expv = vf.exp(neg, pcount_pp)
    den = vf.adds(expv, 1.0, pcount_pp)
    sig = vf.div(one, den, pcount_pp)
    pstore_pp = vf.lt(pp_local, c4, pcount_pp)
    pre = vf.adds(sig, pre_eps, pall)
    vf.store_align(newpre, pre, pstore_pp, 0)
    post_index = vf.adds(pp_id, 4, pu)
    post = vf.gather(sig, post_index)
    post = vf.muls(post, post_scale, pall)
    vf.store_align(newpost, post, pstore_pp, 0)
    active = vf.update_mask(count * 16, dtype=F32)
    s2 = vf.load_align(scales, 2, dist=pl.LoadDist.BRC_B32)
    v = vf.mul(qc, inv, active)
    v = vf.mul(v, s2, active)
    v = vf.add(v, basec, active)
    mask60 = vf.full(60, pu, dtype=pl.DT_UINT32)
    row0 = vf.and_(ids, mask60, pu)
    row1 = vf.adds(row0, 1, pu)
    row2 = vf.adds(row0, 2, pu)
    row3 = vf.adds(row0, 3, pu)
    a0 = vf.gather(v, row0)
    a1 = vf.gather(v, row1)
    a2 = vf.gather(v, row2)
    a3 = vf.gather(v, row3)
    mx = vf.max(a0, a1, active)
    mx = vf.max(mx, a2, active)
    mx = vf.max(mx, a3, active)
    v = vf.sub(v, mx, active)
    v = vf.exp(v, active)
    a0 = vf.gather(v, row0)
    a1 = vf.gather(v, row1)
    a2 = vf.gather(v, row2)
    a3 = vf.gather(v, row3)
    rs = vf.add(a0, a1, active)
    rs = vf.add(rs, a2, active)
    rs = vf.add(rs, a3, active)
    v = vf.div(v, rs, active)
    v = vf.adds(v, eps, active)
    mask3 = vf.full(3, pu, dtype=pl.DT_UINT32)
    cols = vf.and_(ids, mask3, pu)
    mask48 = vf.full(48, pu, dtype=pl.DT_UINT32)
    tokenbase = vf.and_(ids, mask48, pu)
    col0 = vf.add(tokenbase, cols, pu)
    col1 = vf.adds(col0, 4, pu)
    col2 = vf.adds(col0, 8, pu)
    col3 = vf.adds(col0, 12, pu)
    a0 = vf.gather(v, col0)
    a1 = vf.gather(v, col1)
    a2 = vf.gather(v, col2)
    a3 = vf.gather(v, col3)
    cs = vf.add(a0, a1, active)
    cs = vf.add(cs, a2, active)
    cs = vf.add(cs, a3, active)
    cs = vf.adds(cs, eps, active)
    v = vf.div(v, cs, active)
    for _ in pl.range(repeat - 1):
        a0 = vf.gather(v, row0)
        a1 = vf.gather(v, row1)
        a2 = vf.gather(v, row2)
        a3 = vf.gather(v, row3)
        rs = vf.add(a0, a1, active)
        rs = vf.add(rs, a2, active)
        rs = vf.add(rs, a3, active)
        rs = vf.adds(rs, eps, active)
        v = vf.div(v, rs, active)
        a0 = vf.gather(v, col0)
        a1 = vf.gather(v, col1)
        a2 = vf.gather(v, col2)
        a3 = vf.gather(v, col3)
        cs = vf.add(a0, a1, active)
        cs = vf.add(cs, a2, active)
        cs = vf.add(cs, a3, active)
        cs = vf.adds(cs, eps, active)
        v = vf.div(v, cs, active)
    output_row = vf.shift_right(ids, 3, pu)
    output_index = vf.shift_left(output_row, 2, pu)
    output_index = vf.add(output_index, cols, pu)
    comb_first = vf.gather(v, output_index)
    output_index_high = vf.adds(output_index, 32, pu)
    comb_second = vf.gather(v, output_index_high)
    physical_token = vf.shift_right(ids, 5, pu)
    token_count = vf.full(count, pu, dtype=pl.DT_UINT32)
    first_tokens = vf.lt(physical_token, token_count, pu)
    cols8 = vf.and_(ids, mask7, pu)
    first_store = vf.lt(cols8, c4, first_tokens)
    physical_token_high = vf.adds(physical_token, 2, pu)
    second_tokens = vf.lt(physical_token_high, token_count, pu)
    second_store = vf.lt(cols8, c4, second_tokens)
    vf.store_align(newcomb, comb_first, first_store, [0, 0])
    vf.store_align(newcomb, comb_second, second_store, [8, 0])


@pl.jit(arch="3510", auto_mutex=True, tiling_key=MegaMhcKey, name="mega_mhc_kernel")
def mega_mhc_kernel(
    x_ptr: pl.Ptr[pl.DT_UINT8],
    residual_ptr: pl.Ptr[pl.DT_UINT8],
    prev_ptr: pl.Ptr[pl.DT_UINT8],
    post_ptr: pl.Ptr[pl.DT_UINT8],
    comb_ptr: pl.Ptr[pl.DT_UINT8],
    fn_ptr: pl.Ptr[pl.DT_UINT8],
    scales_ptr: pl.Ptr[pl.DT_UINT8],
    bases_ptr: pl.Ptr[pl.DT_UINT8],
    gamma_ptr: pl.Ptr[pl.DT_UINT8],
    rout_ptr: pl.Ptr[pl.DT_UINT8],
    pout_ptr: pl.Ptr[pl.DT_UINT8],
    cout_ptr: pl.Ptr[pl.DT_UINT8],
    yout_ptr: pl.Ptr[pl.DT_UINT8],
    vout_ptr: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: MegaMhcTiling,
):
    tokens = tiling.tokens
    rows = tiling.rows
    hc_eps = tiling.hc_eps
    pre_eps = tiling.pre_eps
    post_scale = tiling.post_scale
    sink_eps = tiling.sink_eps
    repeat = tiling.repeat
    norm_eps = tiling.norm_eps
    x = pl.make_tensor(x_ptr, [tokens, Width], [Width, 1], dtype=B16)
    residual = pl.make_tensor(residual_ptr, [tokens * 4, Width], [Width, 1], dtype=B16)
    prev = pl.make_tensor(prev_ptr, [tokens, 4], [4, 1], dtype=F32)
    post = pl.make_tensor(post_ptr, [tokens, 4], [4, 1], dtype=F32)
    comb = pl.make_tensor(comb_ptr, [tokens, 16], [16, 1], dtype=F32)
    fn = pl.make_tensor(fn_ptr, [24, 4 * Width], [4 * Width, 1], dtype=F32)
    scales = pl.make_tensor(scales_ptr, [1, 3], [3, 1], dtype=F32)
    bases = pl.make_tensor(bases_ptr, [1, 24], [24, 1], dtype=F32)
    gamma = pl.make_tensor(gamma_ptr, [1, Width], [Width, 1], dtype=B16)
    rout = pl.make_tensor(rout_ptr, [tokens * 4, Width], [Width, 1], dtype=B16)
    pout = pl.make_tensor(pout_ptr, [tokens, 4], [4, 1], dtype=F32)
    cout = pl.make_tensor(cout_ptr, [tokens * 4, 4], [4, 1], dtype=F32)
    yout = pl.make_tensor(yout_ptr, [tokens, Width], [Width, 1], dtype=B16)
    vout = pl.make_tensor(vout_ptr, [tokens, 4], [4, 1], dtype=F32)
    qwork = pl.make_tensor(workspace, [rows, Width // 256, 32], [Width // 256 * 32, 32, 1], dtype=F32)
    qflat = pl.make_tensor(workspace, [rows, Width // 256 * 32], [Width // 256 * 32, 1], dtype=F32)
    invwork = pl.make_tensor(pl.addptr(workspace, tiling.inv_offset), [rows, 16], [16, 1], dtype=F32)
    fwork = pl.make_tensor(
        pl.addptr(workspace, tiling.flags_offset), [Width // 256 + 1, 16], [16, 1], dtype=pl.DT_INT32
    )
    if BlockJobs != 27:
        a0g = pl.make_tile_group(
            type=pl.TileType(shape=[32, 256], dtype=B16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0,
            mutex_ids='auto',
            depth=2,
        )
        a1g = pl.make_tile_group(
            type=pl.TileType(shape=[32, 256], dtype=B16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=32768,
            mutex_ids='auto',
            depth=2,
        )
        a2g = pl.make_tile_group(
            type=pl.TileType(shape=[32, 256], dtype=B16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=65536,
            mutex_ids='auto',
            depth=2,
        )
        a3g = pl.make_tile_group(
            type=pl.TileType(shape=[32, 256], dtype=B16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=98304,
            mutex_ids='auto',
            depth=2,
        )
        b0hg = pl.make_tile_group(
            type=pl.TileType(shape=[32, CHUNK], dtype=B16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=131072,
            mutex_ids='auto',
            depth=1,
        )
        b0lg = pl.make_tile_group(
            type=pl.TileType(shape=[32, CHUNK], dtype=B16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=147456,
            mutex_ids='auto',
            depth=1,
        )
        b1hg = pl.make_tile_group(
            type=pl.TileType(shape=[32, CHUNK], dtype=B16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=163840,
            mutex_ids='auto',
            depth=1,
        )
        b1lg = pl.make_tile_group(
            type=pl.TileType(shape=[32, CHUNK], dtype=B16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=180224,
            mutex_ids='auto',
            depth=1,
        )
        b2hg = pl.make_tile_group(
            type=pl.TileType(shape=[32, CHUNK], dtype=B16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=196608,
            mutex_ids='auto',
            depth=1,
        )
        b2lg = pl.make_tile_group(
            type=pl.TileType(shape=[32, CHUNK], dtype=B16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=212992,
            mutex_ids='auto',
            depth=1,
        )
        b3hg = pl.make_tile_group(
            type=pl.TileType(shape=[32, CHUNK], dtype=B16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=229376,
            mutex_ids='auto',
            depth=1,
        )
        b3lg = pl.make_tile_group(
            type=pl.TileType(shape=[32, CHUNK], dtype=B16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=245760,
            mutex_ids='auto',
            depth=1,
        )
        la = pl.make_tile_group(
            type=pl.TileType(shape=[32, 256], dtype=B16, target_memory=pl.MemorySpace.Left, compact=1),
            addrs=0,
            mutex_ids='auto',
            depth=2,
        )
        lb = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK, 32], dtype=B16, target_memory=pl.MemorySpace.Right, compact=1),
            addrs=0,
            mutex_ids='auto',
            depth=2,
        )
        accg = pl.make_tile_group(
            type=pl.TileType(shape=[32, 32], dtype=F32, target_memory=pl.MemorySpace.Acc, compact=1),
            addrs=0,
            mutex_ids='auto',
            depth=1,
        )
        scoreg = pl.make_tile_group(
            type=pl.TileType(shape=[16, 32], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=0,
            mutex_ids='auto',
            depth=1,
        )
        ug = pl.make_tile_group(
            type=pl.TileType(shape=[16, 256], dtype=B16, target_memory=pl.MemorySpace.Vec),
            addrs=[183296, 191488, 199680, 207872, 216064, 224256, 232448, 240640, 2048],
            mutex_ids=None,
            depth=9,
        )
        partialg = pl.make_tile_group(
            type=pl.TileType(shape=[32, 64], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=10240,
            mutex_ids='auto',
            depth=2,
        )
        rg = pl.make_tile_group(
            type=pl.TileType(shape=[64, 256], dtype=B16, target_memory=pl.MemorySpace.Vec),
            addrs=26624,
            mutex_ids='auto',
            depth=2,
        )
        xg = pl.make_tile_group(
            type=pl.TileType(shape=[16, 256], dtype=B16, target_memory=pl.MemorySpace.Vec),
            addrs=92160,
            mutex_ids='auto',
            depth=2,
        )
        gz00 = pl.make_tile_group(
            type=pl.TileType(shape=[17, 128], dtype=B16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=0),
            addrs=108544,
            mutex_ids='auto',
            depth=1,
        )
        gz01 = pl.make_tile_group(
            type=pl.TileType(shape=[17, 128], dtype=B16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=0),
            addrs=112896,
            mutex_ids='auto',
            depth=1,
        )
        gz10 = pl.make_tile_group(
            type=pl.TileType(shape=[17, 128], dtype=B16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=0),
            addrs=117248,
            mutex_ids='auto',
            depth=1,
        )
        gz11 = pl.make_tile_group(
            type=pl.TileType(shape=[17, 128], dtype=B16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=0),
            addrs=121600,
            mutex_ids='auto',
            depth=1,
        )
        gz20 = pl.make_tile_group(
            type=pl.TileType(shape=[17, 128], dtype=B16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=0),
            addrs=125952,
            mutex_ids='auto',
            depth=1,
        )
        gz21 = pl.make_tile_group(
            type=pl.TileType(shape=[17, 128], dtype=B16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=0),
            addrs=130304,
            mutex_ids='auto',
            depth=1,
        )
        gz30 = pl.make_tile_group(
            type=pl.TileType(shape=[17, 128], dtype=B16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=0),
            addrs=134656,
            mutex_ids='auto',
            depth=1,
        )
        gz31 = pl.make_tile_group(
            type=pl.TileType(shape=[17, 128], dtype=B16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=0),
            addrs=139008,
            mutex_ids='auto',
            depth=1,
        )
        fwg = pl.make_tile_group(
            type=pl.TileType(shape=[16, 256], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=143360,
            mutex_ids='auto',
            depth=1,
        )
        h0g = pl.make_tile_group(
            type=pl.TileType(shape=[17, 128], dtype=B16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=0),
            addrs=159744,
            mutex_ids='auto',
            depth=1,
        )
        h1g = pl.make_tile_group(
            type=pl.TileType(shape=[17, 128], dtype=B16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=0),
            addrs=164096,
            mutex_ids='auto',
            depth=1,
        )
        l0g = pl.make_tile_group(
            type=pl.TileType(shape=[17, 128], dtype=B16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=0),
            addrs=168448,
            mutex_ids='auto',
            depth=1,
        )
        l1g = pl.make_tile_group(
            type=pl.TileType(shape=[17, 128], dtype=B16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=0),
            addrs=172800,
            mutex_ids='auto',
            depth=1,
        )
        pg = pl.make_tile_group(
            type=pl.TileType(shape=[16, 8], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=177152,
            mutex_ids='auto',
            depth=2,
        )
        cg = pl.make_tile_group(
            type=pl.TileType(shape=[16, 16], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=178176,
            mutex_ids='auto',
            depth=2,
        )
        vg = pl.make_tile_group(
            type=pl.TileType(shape=[16, 8], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=180224,
            mutex_ids='auto',
            depth=2,
        )
        recordg = pl.make_tile_group(
            type=pl.TileType(shape=[16, 32], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=181248,
            mutex_ids='auto',
            depth=1,
        )
        records = pl.make_tile(
            pl.TileType(shape=[4, Width // CHUNK * 32], dtype=F32, target_memory=pl.MemorySpace.Vec), addr=108544
        )
        scores = pl.make_tile(pl.TileType(shape=[4, 64], dtype=F32, target_memory=pl.MemorySpace.Vec), addr=122880)
        partial = pl.make_tile(pl.TileType(shape=[8, 64], dtype=F32, target_memory=pl.MemorySpace.Vec), addr=123904)
        s = pl.make_tile(pl.TileType(shape=[1, 8], dtype=F32, target_memory=pl.MemorySpace.Vec), addr=152064)
        b = pl.make_tile(pl.TileType(shape=[1, 64], dtype=F32, target_memory=pl.MemorySpace.Vec), addr=152320)
        newpre = pl.make_tile(pl.TileType(shape=[4, 8], dtype=F32, target_memory=pl.MemorySpace.Vec), addr=125952)
        newpost = pl.make_tile(pl.TileType(shape=[4, 8], dtype=F32, target_memory=pl.MemorySpace.Vec), addr=126208)
        newcomb = pl.make_tile(pl.TileType(shape=[16, 8], dtype=F32, target_memory=pl.MemorySpace.Vec), addr=126464)
        gamma_chunk = pl.make_tile(
            pl.TileType(shape=[1, CHUNK], dtype=B16, target_memory=pl.MemorySpace.Vec), addr=153344
        )
        invout = pl.make_tile(pl.TileType(shape=[4, 16], dtype=F32, target_memory=pl.MemorySpace.Vec), addr=126976)
        invg = pl.make_tile_group(
            type=pl.TileType(shape=[BATCH, 16], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=154112,
            mutex_ids=None,
            depth=BlockJobs,
        )
        shard = pl.get_block_idx() // pl.get_subblock_num()
        det_statsg = pl.make_tile_group(
            type=pl.TileType(shape=[3, 64], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=0,
            mutex_ids=None,
            depth=1,
        )
        det_scaleg = pl.make_tile_group(
            type=pl.TileType(shape=[1, 64], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=1024,
            mutex_ids=None,
            depth=1,
        )
        det_recordg = pl.make_tile_group(
            type=pl.TileType(shape=[1, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
            addrs=1280,
            mutex_ids=None,
            depth=1,
        )
        det_allg = pl.make_tile_group(
            type=pl.TileType(shape=[Width // CHUNK, 16], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
            addrs=2048,
            mutex_ids=None,
            depth=1,
        )
        det_globalg = pl.make_tile_group(
            type=pl.TileType(shape=[1, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
            addrs=4096,
            mutex_ids=None,
            depth=1,
        )
        slow_wg = pl.make_tile_group(
            type=pl.TileType(shape=[24, SLOW_CHUNK], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=0,
            mutex_ids=None,
            depth=SLOW_SLOTS,
        )
        slow_rf_parent = pl.make_tile_group(
            type=pl.TileType(
                shape=[SLOW_BATCH * SLOW_BANKS * 4, SLOW_CHUNK], dtype=F32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS,
            mutex_ids=None,
            depth=1,
        )
        slow_rfg = pl.make_tile_group(
            type=pl.TileType(shape=[4, SLOW_CHUNK], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS,
            mutex_ids=None,
            depth=SLOW_BATCH * SLOW_BANKS,
        )
        slow_rbg_parent = pl.make_tile_group(
            type=pl.TileType(shape=[4 * SLOW_BATCH, SLOW_CHUNK], dtype=B16, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS + 16 * SLOW_CHUNK * SLOW_BATCH * SLOW_BANKS,
            mutex_ids=None,
            depth=1,
        )
        slow_rbg = pl.make_tile_group(
            type=pl.TileType(shape=[4, SLOW_CHUNK], dtype=B16, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS + 16 * SLOW_CHUNK * SLOW_BATCH * SLOW_BANKS,
            mutex_ids=None,
            depth=SLOW_BATCH,
        )
        slow_xg_parent = pl.make_tile_group(
            type=pl.TileType(shape=[SLOW_BATCH, SLOW_CHUNK], dtype=B16, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS + (16 * SLOW_BANKS + 8) * SLOW_CHUNK * SLOW_BATCH,
            mutex_ids=None,
            depth=1,
        )
        slow_xg = pl.make_tile_group(
            type=pl.TileType(shape=[1, SLOW_CHUNK], dtype=B16, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS + (16 * SLOW_BANKS + 8) * SLOW_CHUNK * SLOW_BATCH,
            mutex_ids=None,
            depth=SLOW_BATCH,
        )
        slow_up = pl.make_tile_group(
            type=pl.TileType(shape=[SLOW_BATCH, Width], dtype=B16, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH,
            mutex_ids=None,
            depth=1,
        )
        slow_ug = pl.make_tile_group(
            type=pl.TileType(shape=[1, SLOW_CHUNK], dtype=B16, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH,
            mutex_ids=None,
            depth=SLOW_BATCH * Width // SLOW_CHUNK,
        )
        slow_gg = pl.make_tile_group(
            type=pl.TileType(shape=[1, Width], dtype=B16, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * SLOW_BATCH,
            mutex_ids=None,
            depth=1,
        )
        slow_ag = pl.make_tile_group(
            type=pl.TileType(shape=[SLOW_BATCH * 24, 64], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1),
            mutex_ids=None,
            depth=1,
        )
        slow_pg_parent = pl.make_tile_group(
            type=pl.TileType(shape=[SLOW_BATCH, 8], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 6144 * SLOW_BATCH,
            mutex_ids=None,
            depth=1,
        )
        slow_pg = pl.make_tile_group(
            type=pl.TileType(shape=[1, 8], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 6144 * SLOW_BATCH,
            mutex_ids=None,
            depth=SLOW_BATCH,
        )
        slow_cg_parent = pl.make_tile_group(
            type=pl.TileType(shape=[SLOW_BATCH, 16], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 6176 * SLOW_BATCH,
            mutex_ids=None,
            depth=1,
        )
        slow_cg = pl.make_tile_group(
            type=pl.TileType(shape=[1, 16], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 6176 * SLOW_BATCH,
            mutex_ids=None,
            depth=SLOW_BATCH,
        )
        slow_vg_parent = pl.make_tile_group(
            type=pl.TileType(shape=[SLOW_BATCH, 8], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 6240 * SLOW_BATCH,
            mutex_ids=None,
            depth=1,
        )
        slow_vg = pl.make_tile_group(
            type=pl.TileType(shape=[1, 8], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 6240 * SLOW_BATCH,
            mutex_ids=None,
            depth=SLOW_BATCH,
        )
        slow_st_parent = pl.make_tile_group(
            type=pl.TileType(shape=[SLOW_BATCH, 8], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 6272 * SLOW_BATCH,
            mutex_ids=None,
            depth=1,
        )
        slow_stg = pl.make_tile_group(
            type=pl.TileType(shape=[1, 8], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 6272 * SLOW_BATCH,
            mutex_ids=None,
            depth=SLOW_BATCH,
        )
        slow_rsg_parent = pl.make_tile_group(
            type=pl.TileType(shape=[2 * SLOW_BATCH, 64], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 6304 * SLOW_BATCH,
            mutex_ids=None,
            depth=1,
        )
        slow_rsg = pl.make_tile_group(
            type=pl.TileType(shape=[2, 64], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 6304 * SLOW_BATCH,
            mutex_ids=None,
            depth=SLOW_BATCH,
        )
        slow_qg = pl.make_tile_group(
            type=pl.TileType(shape=[1, SLOW_BATCH * 64], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 6816 * SLOW_BATCH,
            mutex_ids=None,
            depth=1,
        )
        slow_sg = pl.make_tile_group(
            type=pl.TileType(shape=[1, 8], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 7072 * SLOW_BATCH,
            mutex_ids=None,
            depth=1,
        )
        slow_bg = pl.make_tile_group(
            type=pl.TileType(shape=[1, 64], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 7072 * SLOW_BATCH
            + 32,
            mutex_ids=None,
            depth=1,
        )
        slow_cog = pl.make_tile_group(
            type=pl.TileType(shape=[SLOW_BATCH * 4, 8], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 7072 * SLOW_BATCH
            + 512,
            mutex_ids=None,
            depth=1,
        )
        slow_pongp = pl.make_tile_group(
            type=pl.TileType(shape=[SLOW_BATCH, 8], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 7072 * SLOW_BATCH
            + 512
            + 128 * SLOW_BATCH,
            mutex_ids=None,
            depth=1,
        )
        slow_pongv = pl.make_tile_group(
            type=pl.TileType(shape=[SLOW_BATCH, 8], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 7072 * SLOW_BATCH
            + 512
            + 160 * SLOW_BATCH,
            mutex_ids=None,
            depth=1,
        )
        with pl.section_vector():
            det_scale = det_scaleg.current()
            det_state = det_statsg.current()
            pl.set_validshape(det_scale, [1, 3])
            pl.load(det_scale, scales, [0, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            range_init(det_scale, det_state)
            pl.system.bar_all()
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
        with pl.section_vector():
            sub = pl.get_subblock_idx()
            weight_rows = pl.min(16, 24 - sub * 16)
            fw = fwg.current()
            pl.set_validshape(fw, [weight_rows, CHUNK])
            pl.load(fw, fn, [sub * 16, 0 * Width + shard * CHUNK])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            range_panel(fw, det_state, weight_rows)
            pl.system.bar_all()
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
            hi0 = h0g.current()
            hi1 = h1g.current()
            lo0 = l0g.current()
            lo1 = l1g.current()
            pack_fn_bf2_128(fw, hi0, lo0, weight_rows, 0)
            pack_fn_bf2_128(fw, hi1, lo1, weight_rows, 128)
            hi = b0hg.current()
            lo = b0lg.current()
            pl.set_validshape(hi0, [16, 128])
            pl.set_validshape(hi1, [16, 128])
            pl.set_validshape(lo0, [16, 128])
            pl.set_validshape(lo1, [16, 128])
            pl.insert(hi, hi0, [sub * 16, 0])
            pl.insert(hi, hi1, [sub * 16, 128])
            pl.insert(lo, lo0, [sub * 16, 0])
            pl.insert(lo, lo1, [sub * 16, 128])
            fw = fwg.current()
            pl.set_validshape(fw, [weight_rows, CHUNK])
            pl.load(fw, fn, [sub * 16, 1 * Width + shard * CHUNK])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            range_panel(fw, det_state, weight_rows)
            pl.system.bar_all()
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
            hi0 = h0g.current()
            hi1 = h1g.current()
            lo0 = l0g.current()
            lo1 = l1g.current()
            pack_fn_bf2_128(fw, hi0, lo0, weight_rows, 0)
            pack_fn_bf2_128(fw, hi1, lo1, weight_rows, 128)
            hi = b1hg.current()
            lo = b1lg.current()
            pl.set_validshape(hi0, [16, 128])
            pl.set_validshape(hi1, [16, 128])
            pl.set_validshape(lo0, [16, 128])
            pl.set_validshape(lo1, [16, 128])
            pl.insert(hi, hi0, [sub * 16, 0])
            pl.insert(hi, hi1, [sub * 16, 128])
            pl.insert(lo, lo0, [sub * 16, 0])
            pl.insert(lo, lo1, [sub * 16, 128])
            fw = fwg.current()
            pl.set_validshape(fw, [weight_rows, CHUNK])
            pl.load(fw, fn, [sub * 16, 2 * Width + shard * CHUNK])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            range_panel(fw, det_state, weight_rows)
            pl.system.bar_all()
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
            hi0 = h0g.current()
            hi1 = h1g.current()
            lo0 = l0g.current()
            lo1 = l1g.current()
            pack_fn_bf2_128(fw, hi0, lo0, weight_rows, 0)
            pack_fn_bf2_128(fw, hi1, lo1, weight_rows, 128)
            hi = b2hg.current()
            lo = b2lg.current()
            pl.set_validshape(hi0, [16, 128])
            pl.set_validshape(hi1, [16, 128])
            pl.set_validshape(lo0, [16, 128])
            pl.set_validshape(lo1, [16, 128])
            pl.insert(hi, hi0, [sub * 16, 0])
            pl.insert(hi, hi1, [sub * 16, 128])
            pl.insert(lo, lo0, [sub * 16, 0])
            pl.insert(lo, lo1, [sub * 16, 128])
            fw = fwg.current()
            pl.set_validshape(fw, [weight_rows, CHUNK])
            pl.load(fw, fn, [sub * 16, 3 * Width + shard * CHUNK])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            range_panel(fw, det_state, weight_rows)
            pl.system.bar_all()
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
            hi0 = h0g.current()
            hi1 = h1g.current()
            lo0 = l0g.current()
            lo1 = l1g.current()
            pack_fn_bf2_128(fw, hi0, lo0, weight_rows, 0)
            pack_fn_bf2_128(fw, hi1, lo1, weight_rows, 128)
            hi = b3hg.current()
            lo = b3lg.current()
            pl.set_validshape(hi0, [16, 128])
            pl.set_validshape(hi1, [16, 128])
            pl.set_validshape(lo0, [16, 128])
            pl.set_validshape(lo1, [16, 128])
            pl.insert(hi, hi0, [sub * 16, 0])
            pl.insert(hi, hi1, [sub * 16, 128])
            pl.insert(lo, lo0, [sub * 16, 0])
            pl.insert(lo, lo1, [sub * 16, 128])
            det_record = det_recordg.current()
            range_record(det_state, det_record)
            pl.system.bar_all()
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
            pl.set_validshape(det_record, [1, 8])
            pl.store(fwork, det_record, [shard, sub * 8])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.S, event_id=2)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.S, event_id=2)
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=6)
        with pl.section_cube():
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=6)
        with pl.section_cube():
            pl.system.bar_all()
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        with pl.section_vector():
            pl.system.bar_all()
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        with pl.section_vector():
            if pl.get_block_idx() == 0:
                det_all = det_allg.current()
                det_global = det_globalg.current()
                pl.load(det_all, fwork, [0, 0])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                range_merge(det_all, det_global, Width // CHUNK)
                pl.system.bar_all()
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.set_validshape(det_global, [1, 16])
                pl.store(fwork, det_global, [Width // CHUNK, 0])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.S, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.S, event_id=2)
        with pl.section_cube():
            pl.system.bar_all()
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        with pl.section_vector():
            pl.system.bar_all()
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        pl.system.dcci(fwork, [Width // CHUNK, 0], cache_line=pl.CacheLine.SINGLE_CACHE_LINE)
        use_slow = fwork[Width // CHUNK, 0]
        if use_slow == 0:
            with pl.section_vector():
                pl.load(gamma_chunk, gamma, [0, shard * CHUNK])
                pl.set_validshape(s, [1, 3])
                pl.set_validshape(b, [1, 24])
                pl.load(s, scales, [0, 0])
                pl.load(b, bases, [0, 0])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=3)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=3)
            with pl.section_vector():
                prime_sub = pl.get_subblock_idx()
                prime_start = prime_sub * BATCH
                prime_count = pl.max(0, pl.min(BATCH, tokens - prime_start))
                prime_rb = rg[0]
                prime_xb = xg[0]
                prime_pp = pg[0]
                prime_cc = cg[0]
                prime_vv = vg[0]
                pl.set_validshape(prime_rb, [prime_count * 4, CHUNK])
                pl.set_validshape(prime_xb, [prime_count, CHUNK])
                pl.set_validshape(prime_pp, [prime_count, 4])
                pl.set_validshape(prime_cc, [prime_count, 16])
                pl.set_validshape(prime_vv, [prime_count, 4])
                if prime_count > 0:
                    pl.load(prime_rb, residual, [prime_start * 4, shard * CHUNK])
                    pl.load(prime_xb, x, [prime_start, shard * CHUNK])
                    pl.load(prime_pp, post, [prime_start, 0])
                    pl.load(prime_cc, comb, [prime_start, 0])
                    pl.load(prime_vv, prev, [prime_start, 0])
            fast_jobs = (tokens + 2 * BATCH - 1) // (2 * BATCH)
            for block in pl.range((tokens + (32 * BlockJobs) - 1) // (32 * BlockJobs)):
                first_job = block * BlockJobs
                local_jobs = pl.min(BlockJobs, fast_jobs - first_job)
                for beat in pl.range(local_jobs + 1):
                    job = first_job + beat
                    with pl.section_vector():
                        if beat < local_jobs:
                            if block > 0:
                                old_start = (block - 1) * (32 * BlockJobs) + beat * 32 + pl.get_subblock_idx() * BATCH
                                old_uu = ug[(first_job - BlockJobs + beat) % 9]
                                old_inverse = invg[beat]
                                pl.set_validshape(old_uu, [BATCH, CHUNK])
                                _norm_chunk_rows(old_uu, gamma_chunk, old_inverse, BATCH)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=beat)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=beat)
                                pl.store(yout, old_uu, [old_start, shard * CHUNK])
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=beat)
                                if beat >= 9 - BlockJobs:
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE3,
                                        wait_pipe=pl.PipeType.V,
                                        event_id=beat - (9 - BlockJobs),
                                    )
                            if beat >= 2:
                                pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=2 + job % 2)
                            _prefetch_produce(
                                job,
                                shard,
                                tokens,
                                x,
                                residual,
                                prev,
                                post,
                                comb,
                                rout,
                                rg,
                                xg,
                                pg,
                                cg,
                                vg,
                                ug,
                                partialg,
                                a0g,
                                a1g,
                                a2g,
                                a3g,
                                gz00,
                                gz01,
                                gz10,
                                gz11,
                                gz20,
                                gz21,
                                gz30,
                                gz31,
                                BATCH,
                                9,
                            )
                            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=job % 2)
                        if beat > 0:
                            pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=4)
                            _overlap_stash(job - 1, shard, tokens, qwork, scoreg, partialg, recordg, BATCH)
                            if job < fast_jobs:
                                pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=5)
                    with pl.section_cube():
                        if beat < local_jobs:
                            pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=job % 2)
                            if beat > 0:
                                pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=5)
                            _overlap_project(
                                job,
                                a0g,
                                a1g,
                                a2g,
                                a3g,
                                b0hg,
                                b0lg,
                                b1hg,
                                b1lg,
                                b2hg,
                                b2lg,
                                b3hg,
                                b3lg,
                                la,
                                lb,
                                accg,
                                scoreg,
                                BATCH,
                            )
                            pl.system.set_cross_core(pipe=pl.PipeType.MTE1, event_id=2 + job % 2)
                            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=4)
                with pl.section_vector():
                    if local_jobs > 0:
                        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=2 + (first_job + local_jobs - 1) % 2)
                    if local_jobs > 1:
                        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=2 + (first_job + local_jobs - 2) % 2)
                    if block > 0:
                        for pending_old_job in pl.range(pl.max(0, local_jobs - (9 - BlockJobs)), local_jobs):
                            pl.system.sync_dst(
                                set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=pending_old_job
                            )
                        for old_tail_job in pl.range(local_jobs, BlockJobs):
                            tail_start = (
                                (block - 1) * (32 * BlockJobs) + old_tail_job * 32 + pl.get_subblock_idx() * BATCH
                            )
                            tail_uu = ug[(first_job - BlockJobs + old_tail_job) % 9]
                            tail_inverse = invg[old_tail_job]
                            pl.set_validshape(tail_uu, [BATCH, CHUNK])
                            _norm_chunk_rows(tail_uu, gamma_chunk, tail_inverse, BATCH)
                            pl.system.sync_src(
                                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=old_tail_job
                            )
                            pl.system.sync_dst(
                                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=old_tail_job
                            )
                            pl.store(yout, tail_uu, [tail_start, shard * CHUNK])
                            pl.system.sync_src(
                                set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=old_tail_job
                            )
                            pl.system.sync_dst(
                                set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=old_tail_job
                            )
                    pl.system.bar_all()
                    pl.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)
                with pl.section_cube():
                    if local_jobs > 0:
                        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=5)
                    pl.system.bar_all()
                with pl.section_vector():
                    worker = pl.get_block_idx()
                    workers = 2 * (Width // CHUNK)
                    valid_quads = pl.min((8 * BlockJobs), (tokens - block * (32 * BlockJobs) + 3) // 4)
                    first_quad = (worker + workers - (block * (8 * BlockJobs)) % workers) % workers
                    unused_u = ug[0]
                    for local_quad in pl.range(first_quad, valid_quads, workers):
                        token = block * (32 * BlockJobs) + local_quad * 4
                        count = pl.min(4, tokens - token)
                        pl.set_validshape(records, [count, Width // CHUNK * 32])
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=3)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=3)
                        pl.load(records, qflat, [local_quad * 4, 0])
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=3)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=3)
                        _merge_finish_quad_registers(
                            records,
                            s,
                            b,
                            newpre,
                            newpost,
                            newcomb,
                            invout,
                            count,
                            1.0 / Width,
                            1.0 / (4 * Width),
                            hc_eps,
                            norm_eps,
                            pre_eps,
                            post_scale,
                            repeat,
                            sink_eps,
                        )
                        pl.set_validshape(newpre, [count, 4])
                        pl.set_validshape(newpost, [count, 4])
                        pl.set_validshape(newcomb, [count * 4, 4])
                        pl.set_validshape(invout, [count, 16])
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=3)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=3)
                        pl.store(vout, newpre, [token, 0])
                        pl.store(pout, newpost, [token, 0])
                        pl.store(cout, newcomb, [token * 4, 0])
                        pl.store(invwork, invout, [local_quad * 4, 0])
                    pl.system.bar_all()
                    pl.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)
                    sub = pl.get_subblock_idx()
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=3)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=3)
                    for inv_job in pl.range(local_jobs):
                        inv_start = (first_job + inv_job) * 32 + sub * BATCH
                        inv_count = pl.max(0, pl.min(BATCH, tokens - inv_start))
                        if inv_count > 0:
                            inverse = invg[inv_job]
                            pl.set_validshape(inverse, [inv_count, 16])
                            pl.load(inverse, invwork, [inv_job * 32 + sub * BATCH, 0])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=3)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=3)
                    if first_job + local_jobs == fast_jobs:
                        for local_job in pl.range(local_jobs):
                            start = (first_job + local_job) * 32 + sub * BATCH
                            count = pl.max(0, pl.min(BATCH, tokens - start))
                            if count > 0:
                                uu = ug[(first_job + local_job) % 9]
                                inverse = invg[local_job]
                                pl.set_validshape(uu, [count, CHUNK])
                                _norm_chunk_rows(uu, gamma_chunk, inverse, count)
                                pl.system.sync_src(
                                    set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=local_job
                                )
                                pl.system.sync_dst(
                                    set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=local_job
                                )
                                pl.store(yout, uu, [start, shard * CHUNK])
                        pl.system.bar_all()
                        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=5)
        else:
            with pl.section_vector():
                slow_s = slow_sg.current()
                slow_b = slow_bg.current()
                slow_g = slow_gg.current()
                slow_a = slow_ag.current()
                slow_q = slow_qg.current()
                pl.set_validshape(slow_s, [1, 3])
                pl.system.bar_all()
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.load(slow_s, scales, [0, 0])
                pl.system.bar_all()
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.set_validshape(slow_b, [1, 24])
                pl.system.bar_all()
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.load(slow_b, bases, [0, 0])
                pl.system.bar_all()
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.system.bar_all()
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.load(slow_g, gamma, [0, 0])
                pl.system.bar_all()
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                slow_epoch = 0
                slow_slot = 0
                for slow_job in pl.range(
                    pl.get_block_idx(), (tokens + SLOW_BATCH - 1) // SLOW_BATCH, 2 * (Width // CHUNK)
                ):
                    slow_job_count = pl.min(SLOW_BATCH, tokens - slow_job * SLOW_BATCH)
                    if slow_epoch == 0:
                        slow_initial_count = pl.min(SLOW_BATCH, tokens - slow_job * SLOW_BATCH)
                        slow_initial_r = slow_rbg_parent.current()
                        slow_initial_x = slow_xg_parent.current()
                        slow_initial_p = slow_pg_parent.current()
                        slow_initial_c = slow_cg_parent.current()
                        slow_initial_v = slow_vg_parent.current()
                        pl.set_validshape(slow_initial_r, [slow_initial_count * 4, SLOW_CHUNK])
                        pl.set_validshape(slow_initial_x, [slow_initial_count, SLOW_CHUNK])
                        pl.set_validshape(slow_initial_p, [slow_initial_count, 4])
                        pl.set_validshape(slow_initial_c, [slow_initial_count, 16])
                        pl.set_validshape(slow_initial_v, [slow_initial_count, 4])
                        pl.system.bar_all()
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.load(slow_initial_r, residual, [slow_job * SLOW_BATCH * 4, 0])
                        pl.system.bar_all()
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.bar_all()
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.load(slow_initial_x, x, [slow_job * SLOW_BATCH, 0])
                        pl.system.bar_all()
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.bar_all()
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.load(slow_initial_p, post, [slow_job * SLOW_BATCH, 0])
                        pl.system.bar_all()
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.bar_all()
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.load(slow_initial_c, comb, [slow_job * SLOW_BATCH, 0])
                        pl.system.bar_all()
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.bar_all()
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.load(slow_initial_v, prev, [slow_job * SLOW_BATCH, 0])
                        pl.system.bar_all()
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        for slow_ahead in pl.range(SLOW_SLOTS):
                            slow_fw = slow_wg[slow_ahead]
                            pl.system.bar_all()
                            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                            pl.load(slow_fw, fn, [0, slow_ahead % 4 * Width + slow_ahead // 4 * SLOW_CHUNK])
                            pl.system.bar_all()
                            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    if SLOW_BANKS == 2:
                        for slow_local in pl.range(SLOW_BATCH):
                            slow_token = slow_job * SLOW_BATCH + slow_local
                            if slow_token < tokens:
                                slow_rb = slow_rbg[slow_local]
                                slow_rf = slow_rfg[0 % SLOW_BANKS * SLOW_BATCH + slow_local]
                                slow_ux = slow_xg[slow_local]
                                slow_u = slow_ug[slow_local * (Width // SLOW_CHUNK) + 0]
                                slow_p = slow_pg[slow_local]
                                slow_c = slow_cg[slow_local]
                                slow_v = slow_vg[slow_local]
                                slow_st = slow_stg[slow_local]
                                slow_rs = slow_rsg[slow_local]
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                _slow_post_stream(
                                    slow_rb,
                                    slow_rf,
                                    slow_ux,
                                    slow_u,
                                    slow_p,
                                    slow_c,
                                    slow_v,
                                    slow_st,
                                    slow_rs,
                                    0 * SLOW_CHUNK,
                                    1.0 / Width,
                                    1.0 / (4 * Width),
                                    hc_eps,
                                    norm_eps,
                                )
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                if SLOW_BATCH == 1:
                                    pl.system.bar_all()
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    pl.store(rout, slow_rb, [slow_token * 4, 0 * SLOW_CHUNK])
                                    pl.system.bar_all()
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                        if SLOW_BATCH > 1:
                            slow_initial_store = slow_rbg_parent.current()
                            pl.set_validshape(slow_initial_store, [slow_job_count * 4, SLOW_CHUNK])
                            pl.system.bar_all()
                            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                            pl.store(rout, slow_initial_store, [slow_job * SLOW_BATCH * 4, 0 * SLOW_CHUNK])
                            pl.system.bar_all()
                            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    for slow_band in pl.range(Width // SLOW_CHUNK):
                        if SLOW_BANKS == 1:
                            for slow_local in pl.range(SLOW_BATCH):
                                slow_token = slow_job * SLOW_BATCH + slow_local
                                if slow_token < tokens:
                                    slow_rb = slow_rbg[slow_local]
                                    slow_rf = slow_rfg[slow_band % SLOW_BANKS * SLOW_BATCH + slow_local]
                                    slow_ux = slow_xg[slow_local]
                                    slow_u = slow_ug[slow_local * (Width // SLOW_CHUNK) + slow_band]
                                    slow_p = slow_pg[slow_local]
                                    slow_c = slow_cg[slow_local]
                                    slow_v = slow_vg[slow_local]
                                    slow_st = slow_stg[slow_local]
                                    slow_rs = slow_rsg[slow_local]
                                    pl.system.bar_all()
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    _slow_post_stream(
                                        slow_rb,
                                        slow_rf,
                                        slow_ux,
                                        slow_u,
                                        slow_p,
                                        slow_c,
                                        slow_v,
                                        slow_st,
                                        slow_rs,
                                        slow_band * SLOW_CHUNK,
                                        1.0 / Width,
                                        1.0 / (4 * Width),
                                        hc_eps,
                                        norm_eps,
                                    )
                                    pl.system.bar_all()
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    if SLOW_BATCH == 1:
                                        pl.system.bar_all()
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                        )
                                        pl.store(rout, slow_rb, [slow_token * 4, slow_band * SLOW_CHUNK])
                                        pl.system.bar_all()
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                        )
                            if SLOW_BATCH > 1:
                                slow_current_store = slow_rbg_parent.current()
                                pl.set_validshape(slow_current_store, [slow_job_count * 4, SLOW_CHUNK])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.store(rout, slow_current_store, [slow_job * SLOW_BATCH * 4, slow_band * SLOW_CHUNK])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        slow_r0 = slow_rfg[slow_band % SLOW_BANKS * SLOW_BATCH]
                        slow_r1 = slow_rfg[slow_band % SLOW_BANKS * SLOW_BATCH + pl.min(1, SLOW_BATCH - 1)]
                        slow_r2 = slow_rfg[slow_band % SLOW_BANKS * SLOW_BATCH + pl.min(2, SLOW_BATCH - 1)]
                        slow_r3 = slow_rfg[slow_band % SLOW_BANKS * SLOW_BATCH + pl.min(3, SLOW_BATCH - 1)]
                        for slow_part in pl.range(4):
                            slow_step = slow_band * 4 + slow_part
                            slow_route = slow_part
                            slow_w = slow_wg[slow_slot]
                            if slow_band == 0 and slow_route == 0:
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                _slow_dot_first(slow_r0, slow_r1, slow_r2, slow_r3, slow_w, slow_a, slow_q, slow_route)
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                            elif slow_band + 1 == Width // SLOW_CHUNK and slow_route == 3:
                                if SLOW_BATCH == 1:
                                    slow_fused_stat = slow_st_parent.current()
                                    slow_fused_prev = slow_pongv.current()
                                    slow_fused_post = slow_pongp.current()
                                    slow_fused_comb = slow_cog.current()
                                    pl.system.bar_all()
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    _slow_dot_last_finish_single(
                                        slow_r0,
                                        slow_r1,
                                        slow_r2,
                                        slow_r3,
                                        slow_w,
                                        slow_a,
                                        slow_q,
                                        slow_route,
                                        slow_fused_stat,
                                        slow_s,
                                        slow_b,
                                        slow_fused_prev,
                                        slow_fused_post,
                                        slow_fused_comb,
                                        pre_eps,
                                        post_scale,
                                        repeat,
                                        sink_eps,
                                        1,
                                    )
                                    pl.system.bar_all()
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                else:
                                    pl.system.bar_all()
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    _slow_dot_last(
                                        slow_r0, slow_r1, slow_r2, slow_r3, slow_w, slow_a, slow_q, slow_route
                                    )
                                    pl.system.bar_all()
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                            else:
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                _slow_dot_accum(slow_r0, slow_r1, slow_r2, slow_r3, slow_w, slow_a, slow_q, slow_route)
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                            if slow_part == SLOW_PREFETCH and slow_band + 1 < Width // SLOW_CHUNK:
                                slow_hidden_count = pl.min(SLOW_BATCH, tokens - slow_job * SLOW_BATCH)
                                slow_hn_r = slow_rbg_parent.current()
                                slow_hn_x = slow_xg_parent.current()
                                pl.set_validshape(slow_hn_r, [slow_hidden_count * 4, SLOW_CHUNK])
                                pl.set_validshape(slow_hn_x, [slow_hidden_count, SLOW_CHUNK])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.load(slow_hn_r, residual, [slow_job * SLOW_BATCH * 4, (slow_band + 1) * SLOW_CHUNK])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.load(slow_hn_x, x, [slow_job * SLOW_BATCH, (slow_band + 1) * SLOW_CHUNK])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                            if (
                                slow_step + SLOW_SLOTS < 4 * (Width // SLOW_CHUNK)
                                or slow_job + 2 * (Width // CHUNK) < (tokens + SLOW_BATCH - 1) // SLOW_BATCH
                            ):
                                slow_wf = slow_wg[slow_slot]
                                slow_future = slow_step + SLOW_SLOTS
                                if slow_future >= 4 * (Width // SLOW_CHUNK):
                                    slow_future = slow_future - 4 * (Width // SLOW_CHUNK)
                                slow_fband = slow_future // 4
                                slow_froute = slow_future % 4
                                slow_fout = 0
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.load(slow_wf, fn, [slow_fout, slow_froute * Width + slow_fband * SLOW_CHUNK])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                            if (
                                slow_band + 1 == Width // SLOW_CHUNK
                                and slow_part == 0
                                and (slow_job + 2 * (Width // CHUNK) < (tokens + SLOW_BATCH - 1) // SLOW_BATCH)
                            ):
                                slow_next_job = slow_job + 2 * (Width // CHUNK)
                                slow_next_count = pl.min(SLOW_BATCH, tokens - slow_next_job * SLOW_BATCH)
                                slow_next_r = slow_rbg_parent.current()
                                slow_next_x = slow_xg_parent.current()
                                slow_next_p = slow_pg_parent.current()
                                slow_next_c = slow_cg_parent.current()
                                slow_next_v = slow_vg_parent.current()
                                pl.set_validshape(slow_next_r, [slow_next_count * 4, SLOW_CHUNK])
                                pl.set_validshape(slow_next_x, [slow_next_count, SLOW_CHUNK])
                                pl.set_validshape(slow_next_p, [slow_next_count, 4])
                                pl.set_validshape(slow_next_c, [slow_next_count, 16])
                                pl.set_validshape(slow_next_v, [slow_next_count, 4])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.load(slow_next_r, residual, [slow_next_job * SLOW_BATCH * 4, 0])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.load(slow_next_x, x, [slow_next_job * SLOW_BATCH, 0])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.load(slow_next_p, post, [slow_next_job * SLOW_BATCH, 0])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.load(slow_next_c, comb, [slow_next_job * SLOW_BATCH, 0])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.load(slow_next_v, prev, [slow_next_job * SLOW_BATCH, 0])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                            if SLOW_BANKS == 2 and slow_part == 4 - SLOW_LEAD and (slow_band + 1 < Width // SLOW_CHUNK):
                                for slow_local in pl.range(SLOW_BATCH):
                                    slow_token = slow_job * SLOW_BATCH + slow_local
                                    if slow_token < tokens:
                                        slow_rb = slow_rbg[slow_local]
                                        slow_rf = slow_rfg[(slow_band + 1) % SLOW_BANKS * SLOW_BATCH + slow_local]
                                        slow_ux = slow_xg[slow_local]
                                        slow_u = slow_ug[slow_local * (Width // SLOW_CHUNK) + (slow_band + 1)]
                                        slow_p = slow_pg[slow_local]
                                        slow_c = slow_cg[slow_local]
                                        slow_v = slow_vg[slow_local]
                                        slow_st = slow_stg[slow_local]
                                        slow_rs = slow_rsg[slow_local]
                                        pl.system.bar_all()
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                        )
                                        _slow_post_stream(
                                            slow_rb,
                                            slow_rf,
                                            slow_ux,
                                            slow_u,
                                            slow_p,
                                            slow_c,
                                            slow_v,
                                            slow_st,
                                            slow_rs,
                                            (slow_band + 1) * SLOW_CHUNK,
                                            1.0 / Width,
                                            1.0 / (4 * Width),
                                            hc_eps,
                                            norm_eps,
                                        )
                                        pl.system.bar_all()
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                        )
                                        if SLOW_BATCH == 1:
                                            pl.system.bar_all()
                                            pl.system.sync_src(
                                                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0
                                            )
                                            pl.system.sync_dst(
                                                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0
                                            )
                                            pl.system.sync_src(
                                                set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
                                            )
                                            pl.system.sync_dst(
                                                set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
                                            )
                                            pl.system.sync_src(
                                                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
                                            )
                                            pl.system.sync_dst(
                                                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
                                            )
                                            pl.system.sync_src(
                                                set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1
                                            )
                                            pl.system.sync_dst(
                                                set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1
                                            )
                                            pl.system.sync_src(
                                                set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                            )
                                            pl.system.sync_dst(
                                                set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                            )
                                            pl.system.sync_src(
                                                set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                            )
                                            pl.system.sync_dst(
                                                set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                            )
                                            pl.store(rout, slow_rb, [slow_token * 4, (slow_band + 1) * SLOW_CHUNK])
                                            pl.system.bar_all()
                                            pl.system.sync_src(
                                                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0
                                            )
                                            pl.system.sync_dst(
                                                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0
                                            )
                                            pl.system.sync_src(
                                                set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
                                            )
                                            pl.system.sync_dst(
                                                set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
                                            )
                                            pl.system.sync_src(
                                                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
                                            )
                                            pl.system.sync_dst(
                                                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
                                            )
                                            pl.system.sync_src(
                                                set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1
                                            )
                                            pl.system.sync_dst(
                                                set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1
                                            )
                                            pl.system.sync_src(
                                                set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                            )
                                            pl.system.sync_dst(
                                                set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                            )
                                            pl.system.sync_src(
                                                set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                            )
                                            pl.system.sync_dst(
                                                set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                            )
                                if SLOW_BATCH > 1:
                                    slow_next_store = slow_rbg_parent.current()
                                    pl.set_validshape(slow_next_store, [slow_job_count * 4, SLOW_CHUNK])
                                    pl.system.bar_all()
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    pl.store(
                                        rout, slow_next_store, [slow_job * SLOW_BATCH * 4, (slow_band + 1) * SLOW_CHUNK]
                                    )
                                    pl.system.bar_all()
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                            slow_norm_start = (Width // SLOW_CHUNK - 1) * 4
                            if SLOW_BANKS == 2 and Width // SLOW_CHUNK > 1:
                                slow_norm_start = slow_norm_start - SLOW_LEAD
                            if slow_step == slow_norm_start:
                                slow_ua = slow_up.current()
                                slow_sa = slow_st_parent.current()
                                slow_count = pl.min(SLOW_BATCH, tokens - slow_job * SLOW_BATCH)
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                _slow_norm_all(slow_ua, slow_g, slow_sa, slow_count)
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.set_validshape(slow_ua, [slow_count, Width])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.store(yout, slow_ua, [slow_job * SLOW_BATCH, 0])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                            slow_slot = slow_slot + 1
                            if slow_slot == SLOW_SLOTS:
                                slow_slot = 0
                    slow_finish_count = pl.min(SLOW_BATCH, tokens - slow_job * SLOW_BATCH)
                    slow_os = slow_st_parent.current()
                    slow_ov = slow_pongv.current()
                    slow_op = slow_pongp.current()
                    slow_oc = slow_cog.current()
                    if SLOW_BATCH == 2:
                        pl.system.bar_all()
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        _slow_finish_pair(
                            slow_q,
                            slow_os,
                            slow_s,
                            slow_b,
                            slow_ov,
                            slow_op,
                            slow_oc,
                            pre_eps,
                            post_scale,
                            repeat,
                            sink_eps,
                            slow_finish_count,
                        )
                        pl.system.bar_all()
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.set_validshape(slow_ov, [slow_finish_count, 4])
                    pl.set_validshape(slow_op, [slow_finish_count, 4])
                    pl.set_validshape(slow_oc, [slow_finish_count * 4, 4])
                    pl.system.bar_all()
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.store(vout, slow_ov, [slow_job * SLOW_BATCH, 0])
                    pl.system.bar_all()
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.system.bar_all()
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.store(pout, slow_op, [slow_job * SLOW_BATCH, 0])
                    pl.system.bar_all()
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.system.bar_all()
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.store(cout, slow_oc, [slow_job * SLOW_BATCH * 4, 0])
                    pl.system.bar_all()
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    slow_epoch = slow_epoch + 1

    else:
        a0g = pl.make_tile_group(
            type=pl.TileType(shape=[16, 256], dtype=B16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0,
            mutex_ids='auto',
            depth=2,
        )
        a1g = pl.make_tile_group(
            type=pl.TileType(shape=[16, 256], dtype=B16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=32768,
            mutex_ids='auto',
            depth=2,
        )
        a2g = pl.make_tile_group(
            type=pl.TileType(shape=[16, 256], dtype=B16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=65536,
            mutex_ids='auto',
            depth=2,
        )
        a3g = pl.make_tile_group(
            type=pl.TileType(shape=[16, 256], dtype=B16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=98304,
            mutex_ids='auto',
            depth=2,
        )
        b0hg = pl.make_tile_group(
            type=pl.TileType(shape=[32, CHUNK], dtype=B16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=131072,
            mutex_ids='auto',
            depth=1,
        )
        b0lg = pl.make_tile_group(
            type=pl.TileType(shape=[32, CHUNK], dtype=B16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=147456,
            mutex_ids='auto',
            depth=1,
        )
        b1hg = pl.make_tile_group(
            type=pl.TileType(shape=[32, CHUNK], dtype=B16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=163840,
            mutex_ids='auto',
            depth=1,
        )
        b1lg = pl.make_tile_group(
            type=pl.TileType(shape=[32, CHUNK], dtype=B16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=180224,
            mutex_ids='auto',
            depth=1,
        )
        b2hg = pl.make_tile_group(
            type=pl.TileType(shape=[32, CHUNK], dtype=B16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=196608,
            mutex_ids='auto',
            depth=1,
        )
        b2lg = pl.make_tile_group(
            type=pl.TileType(shape=[32, CHUNK], dtype=B16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=212992,
            mutex_ids='auto',
            depth=1,
        )
        b3hg = pl.make_tile_group(
            type=pl.TileType(shape=[32, CHUNK], dtype=B16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=229376,
            mutex_ids='auto',
            depth=1,
        )
        b3lg = pl.make_tile_group(
            type=pl.TileType(shape=[32, CHUNK], dtype=B16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=245760,
            mutex_ids='auto',
            depth=1,
        )
        la = pl.make_tile_group(
            type=pl.TileType(shape=[16, 256], dtype=B16, target_memory=pl.MemorySpace.Left, compact=1),
            addrs=0,
            mutex_ids='auto',
            depth=2,
        )
        lb = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK, 32], dtype=B16, target_memory=pl.MemorySpace.Right, compact=1),
            addrs=0,
            mutex_ids='auto',
            depth=2,
        )
        accg = pl.make_tile_group(
            type=pl.TileType(shape=[16, 32], dtype=F32, target_memory=pl.MemorySpace.Acc, compact=1),
            addrs=0,
            mutex_ids='auto',
            depth=1,
        )
        scoreg = pl.make_tile_group(
            type=pl.TileType(shape=[8, 32], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=0,
            mutex_ids='auto',
            depth=1,
        )
        ug = pl.make_tile_group(
            type=pl.TileType(shape=[8, 256], dtype=B16, target_memory=pl.MemorySpace.Vec),
            addrs=[
                183296,
                187392,
                191488,
                195584,
                199680,
                203776,
                207872,
                211968,
                216064,
                220160,
                224256,
                228352,
                232448,
                236544,
                240640,
                244736,
                2048,
                6144,
                18432,
                22528,
                59392,
                63488,
                67584,
                71680,
                75776,
                79872,
                83968,
                88064,
                100352,
                104448,
                127232,
                131328,
                135424,
            ],
            mutex_ids=None,
            depth=33,
        )
        partialg = pl.make_tile_group(
            type=pl.TileType(shape=[16, 64], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=10240,
            mutex_ids='auto',
            depth=2,
        )
        rg = pl.make_tile_group(
            type=pl.TileType(shape=[32, 256], dtype=B16, target_memory=pl.MemorySpace.Vec),
            addrs=26624,
            mutex_ids='auto',
            depth=2,
        )
        xg = pl.make_tile_group(
            type=pl.TileType(shape=[8, 256], dtype=B16, target_memory=pl.MemorySpace.Vec),
            addrs=92160,
            mutex_ids='auto',
            depth=2,
        )
        gz00 = pl.make_tile_group(
            type=pl.TileType(shape=[9, 128], dtype=B16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=0),
            addrs=108544,
            mutex_ids='auto',
            depth=1,
        )
        gz01 = pl.make_tile_group(
            type=pl.TileType(shape=[9, 128], dtype=B16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=0),
            addrs=110848,
            mutex_ids='auto',
            depth=1,
        )
        gz10 = pl.make_tile_group(
            type=pl.TileType(shape=[9, 128], dtype=B16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=0),
            addrs=113152,
            mutex_ids='auto',
            depth=1,
        )
        gz11 = pl.make_tile_group(
            type=pl.TileType(shape=[9, 128], dtype=B16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=0),
            addrs=115456,
            mutex_ids='auto',
            depth=1,
        )
        gz20 = pl.make_tile_group(
            type=pl.TileType(shape=[9, 128], dtype=B16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=0),
            addrs=117760,
            mutex_ids='auto',
            depth=1,
        )
        gz21 = pl.make_tile_group(
            type=pl.TileType(shape=[9, 128], dtype=B16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=0),
            addrs=120064,
            mutex_ids='auto',
            depth=1,
        )
        gz30 = pl.make_tile_group(
            type=pl.TileType(shape=[9, 128], dtype=B16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=0),
            addrs=122368,
            mutex_ids='auto',
            depth=1,
        )
        gz31 = pl.make_tile_group(
            type=pl.TileType(shape=[9, 128], dtype=B16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=0),
            addrs=124672,
            mutex_ids='auto',
            depth=1,
        )
        fwg = pl.make_tile_group(
            type=pl.TileType(shape=[16, 256], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=143360,
            mutex_ids='auto',
            depth=1,
        )
        h0g = pl.make_tile_group(
            type=pl.TileType(shape=[17, 128], dtype=B16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=0),
            addrs=159744,
            mutex_ids='auto',
            depth=1,
        )
        h1g = pl.make_tile_group(
            type=pl.TileType(shape=[17, 128], dtype=B16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=0),
            addrs=164096,
            mutex_ids='auto',
            depth=1,
        )
        l0g = pl.make_tile_group(
            type=pl.TileType(shape=[17, 128], dtype=B16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=0),
            addrs=168448,
            mutex_ids='auto',
            depth=1,
        )
        l1g = pl.make_tile_group(
            type=pl.TileType(shape=[17, 128], dtype=B16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=0),
            addrs=172800,
            mutex_ids='auto',
            depth=1,
        )
        pg = pl.make_tile_group(
            type=pl.TileType(shape=[16, 8], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=177152,
            mutex_ids='auto',
            depth=2,
        )
        cg = pl.make_tile_group(
            type=pl.TileType(shape=[16, 16], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=178176,
            mutex_ids='auto',
            depth=2,
        )
        vg = pl.make_tile_group(
            type=pl.TileType(shape=[16, 8], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=180224,
            mutex_ids='auto',
            depth=2,
        )
        recordg = pl.make_tile_group(
            type=pl.TileType(shape=[8, 32], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=181248,
            mutex_ids='auto',
            depth=1,
        )
        records = pl.make_tile(
            pl.TileType(shape=[4, Width // CHUNK * 32], dtype=F32, target_memory=pl.MemorySpace.Vec), addr=108544
        )
        scores = pl.make_tile(pl.TileType(shape=[4, 64], dtype=F32, target_memory=pl.MemorySpace.Vec), addr=122880)
        partial = pl.make_tile(pl.TileType(shape=[8, 64], dtype=F32, target_memory=pl.MemorySpace.Vec), addr=123904)
        s = pl.make_tile(pl.TileType(shape=[1, 8], dtype=F32, target_memory=pl.MemorySpace.Vec), addr=152064)
        b = pl.make_tile(pl.TileType(shape=[1, 64], dtype=F32, target_memory=pl.MemorySpace.Vec), addr=152320)
        newpre = pl.make_tile(pl.TileType(shape=[4, 8], dtype=F32, target_memory=pl.MemorySpace.Vec), addr=125952)
        newpost = pl.make_tile(pl.TileType(shape=[4, 8], dtype=F32, target_memory=pl.MemorySpace.Vec), addr=126208)
        newcomb = pl.make_tile(pl.TileType(shape=[16, 8], dtype=F32, target_memory=pl.MemorySpace.Vec), addr=126464)
        gamma_chunk = pl.make_tile(
            pl.TileType(shape=[1, CHUNK], dtype=B16, target_memory=pl.MemorySpace.Vec), addr=153344
        )
        invout = pl.make_tile(pl.TileType(shape=[4, 16], dtype=F32, target_memory=pl.MemorySpace.Vec), addr=126976)
        invg = pl.make_tile_group(
            type=pl.TileType(shape=[BATCH8, 16], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=154112,
            mutex_ids=None,
            depth=BlockJobs,
        )
        shard = pl.get_block_idx() // pl.get_subblock_num()
        det_statsg = pl.make_tile_group(
            type=pl.TileType(shape=[3, 64], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=0,
            mutex_ids=None,
            depth=1,
        )
        det_scaleg = pl.make_tile_group(
            type=pl.TileType(shape=[1, 64], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=1024,
            mutex_ids=None,
            depth=1,
        )
        det_recordg = pl.make_tile_group(
            type=pl.TileType(shape=[1, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
            addrs=1280,
            mutex_ids=None,
            depth=1,
        )
        det_allg = pl.make_tile_group(
            type=pl.TileType(shape=[Width // CHUNK, 16], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
            addrs=2048,
            mutex_ids=None,
            depth=1,
        )
        det_globalg = pl.make_tile_group(
            type=pl.TileType(shape=[1, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
            addrs=4096,
            mutex_ids=None,
            depth=1,
        )
        slow_wg = pl.make_tile_group(
            type=pl.TileType(shape=[24, SLOW_CHUNK], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=0,
            mutex_ids=None,
            depth=SLOW_SLOTS,
        )
        slow_rf_parent = pl.make_tile_group(
            type=pl.TileType(
                shape=[SLOW_BATCH * SLOW_BANKS * 4, SLOW_CHUNK], dtype=F32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS,
            mutex_ids=None,
            depth=1,
        )
        slow_rfg = pl.make_tile_group(
            type=pl.TileType(shape=[4, SLOW_CHUNK], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS,
            mutex_ids=None,
            depth=SLOW_BATCH * SLOW_BANKS,
        )
        slow_rbg_parent = pl.make_tile_group(
            type=pl.TileType(shape=[4 * SLOW_BATCH, SLOW_CHUNK], dtype=B16, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS + 16 * SLOW_CHUNK * SLOW_BATCH * SLOW_BANKS,
            mutex_ids=None,
            depth=1,
        )
        slow_rbg = pl.make_tile_group(
            type=pl.TileType(shape=[4, SLOW_CHUNK], dtype=B16, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS + 16 * SLOW_CHUNK * SLOW_BATCH * SLOW_BANKS,
            mutex_ids=None,
            depth=SLOW_BATCH,
        )
        slow_xg_parent = pl.make_tile_group(
            type=pl.TileType(shape=[SLOW_BATCH, SLOW_CHUNK], dtype=B16, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS + (16 * SLOW_BANKS + 8) * SLOW_CHUNK * SLOW_BATCH,
            mutex_ids=None,
            depth=1,
        )
        slow_xg = pl.make_tile_group(
            type=pl.TileType(shape=[1, SLOW_CHUNK], dtype=B16, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS + (16 * SLOW_BANKS + 8) * SLOW_CHUNK * SLOW_BATCH,
            mutex_ids=None,
            depth=SLOW_BATCH,
        )
        slow_up = pl.make_tile_group(
            type=pl.TileType(shape=[SLOW_BATCH, Width], dtype=B16, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH,
            mutex_ids=None,
            depth=1,
        )
        slow_ug = pl.make_tile_group(
            type=pl.TileType(shape=[1, SLOW_CHUNK], dtype=B16, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH,
            mutex_ids=None,
            depth=SLOW_BATCH * Width // SLOW_CHUNK,
        )
        slow_gg = pl.make_tile_group(
            type=pl.TileType(shape=[1, Width], dtype=B16, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * SLOW_BATCH,
            mutex_ids=None,
            depth=1,
        )
        slow_ag = pl.make_tile_group(
            type=pl.TileType(shape=[SLOW_BATCH * 24, 64], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1),
            mutex_ids=None,
            depth=1,
        )
        slow_pg_parent = pl.make_tile_group(
            type=pl.TileType(shape=[SLOW_BATCH, 8], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 6144 * SLOW_BATCH,
            mutex_ids=None,
            depth=1,
        )
        slow_pg = pl.make_tile_group(
            type=pl.TileType(shape=[1, 8], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 6144 * SLOW_BATCH,
            mutex_ids=None,
            depth=SLOW_BATCH,
        )
        slow_cg_parent = pl.make_tile_group(
            type=pl.TileType(shape=[SLOW_BATCH, 16], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 6176 * SLOW_BATCH,
            mutex_ids=None,
            depth=1,
        )
        slow_cg = pl.make_tile_group(
            type=pl.TileType(shape=[1, 16], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 6176 * SLOW_BATCH,
            mutex_ids=None,
            depth=SLOW_BATCH,
        )
        slow_vg_parent = pl.make_tile_group(
            type=pl.TileType(shape=[SLOW_BATCH, 8], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 6240 * SLOW_BATCH,
            mutex_ids=None,
            depth=1,
        )
        slow_vg = pl.make_tile_group(
            type=pl.TileType(shape=[1, 8], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 6240 * SLOW_BATCH,
            mutex_ids=None,
            depth=SLOW_BATCH,
        )
        slow_st_parent = pl.make_tile_group(
            type=pl.TileType(shape=[SLOW_BATCH, 8], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 6272 * SLOW_BATCH,
            mutex_ids=None,
            depth=1,
        )
        slow_stg = pl.make_tile_group(
            type=pl.TileType(shape=[1, 8], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 6272 * SLOW_BATCH,
            mutex_ids=None,
            depth=SLOW_BATCH,
        )
        slow_rsg_parent = pl.make_tile_group(
            type=pl.TileType(shape=[2 * SLOW_BATCH, 64], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 6304 * SLOW_BATCH,
            mutex_ids=None,
            depth=1,
        )
        slow_rsg = pl.make_tile_group(
            type=pl.TileType(shape=[2, 64], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 6304 * SLOW_BATCH,
            mutex_ids=None,
            depth=SLOW_BATCH,
        )
        slow_qg = pl.make_tile_group(
            type=pl.TileType(shape=[1, SLOW_BATCH * 64], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 6816 * SLOW_BATCH,
            mutex_ids=None,
            depth=1,
        )
        slow_sg = pl.make_tile_group(
            type=pl.TileType(shape=[1, 8], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 7072 * SLOW_BATCH,
            mutex_ids=None,
            depth=1,
        )
        slow_bg = pl.make_tile_group(
            type=pl.TileType(shape=[1, 64], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 7072 * SLOW_BATCH
            + 32,
            mutex_ids=None,
            depth=1,
        )
        slow_cog = pl.make_tile_group(
            type=pl.TileType(shape=[SLOW_BATCH * 4, 8], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 7072 * SLOW_BATCH
            + 512,
            mutex_ids=None,
            depth=1,
        )
        slow_pongp = pl.make_tile_group(
            type=pl.TileType(shape=[SLOW_BATCH, 8], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 7072 * SLOW_BATCH
            + 512
            + 128 * SLOW_BATCH,
            mutex_ids=None,
            depth=1,
        )
        slow_pongv = pl.make_tile_group(
            type=pl.TileType(shape=[SLOW_BATCH, 8], dtype=F32, target_memory=pl.MemorySpace.Vec),
            addrs=96 * SLOW_CHUNK * SLOW_SLOTS
            + (16 * SLOW_BANKS + 10) * SLOW_CHUNK * SLOW_BATCH
            + 2 * Width * (SLOW_BATCH + 1)
            + 7072 * SLOW_BATCH
            + 512
            + 160 * SLOW_BATCH,
            mutex_ids=None,
            depth=1,
        )
        with pl.section_vector():
            det_scale = det_scaleg.current()
            det_state = det_statsg.current()
            pl.set_validshape(det_scale, [1, 3])
            pl.load(det_scale, scales, [0, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            range_init(det_scale, det_state)
            pl.system.bar_all()
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
        with pl.section_vector():
            sub = pl.get_subblock_idx()
            weight_rows = pl.min(16, 24 - sub * 16)
            fw = fwg.current()
            pl.set_validshape(fw, [weight_rows, CHUNK])
            pl.load(fw, fn, [sub * 16, 0 * Width + shard * CHUNK])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            range_panel(fw, det_state, weight_rows)
            pl.system.bar_all()
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
            hi0 = h0g.current()
            hi1 = h1g.current()
            lo0 = l0g.current()
            lo1 = l1g.current()
            pack_fn_bf2_128(fw, hi0, lo0, weight_rows, 0)
            pack_fn_bf2_128(fw, hi1, lo1, weight_rows, 128)
            hi = b0hg.current()
            lo = b0lg.current()
            pl.set_validshape(hi0, [16, 128])
            pl.set_validshape(hi1, [16, 128])
            pl.set_validshape(lo0, [16, 128])
            pl.set_validshape(lo1, [16, 128])
            pl.insert(hi, hi0, [sub * 16, 0])
            pl.insert(hi, hi1, [sub * 16, 128])
            pl.insert(lo, lo0, [sub * 16, 0])
            pl.insert(lo, lo1, [sub * 16, 128])
            fw = fwg.current()
            pl.set_validshape(fw, [weight_rows, CHUNK])
            pl.load(fw, fn, [sub * 16, 1 * Width + shard * CHUNK])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            range_panel(fw, det_state, weight_rows)
            pl.system.bar_all()
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
            hi0 = h0g.current()
            hi1 = h1g.current()
            lo0 = l0g.current()
            lo1 = l1g.current()
            pack_fn_bf2_128(fw, hi0, lo0, weight_rows, 0)
            pack_fn_bf2_128(fw, hi1, lo1, weight_rows, 128)
            hi = b1hg.current()
            lo = b1lg.current()
            pl.set_validshape(hi0, [16, 128])
            pl.set_validshape(hi1, [16, 128])
            pl.set_validshape(lo0, [16, 128])
            pl.set_validshape(lo1, [16, 128])
            pl.insert(hi, hi0, [sub * 16, 0])
            pl.insert(hi, hi1, [sub * 16, 128])
            pl.insert(lo, lo0, [sub * 16, 0])
            pl.insert(lo, lo1, [sub * 16, 128])
            fw = fwg.current()
            pl.set_validshape(fw, [weight_rows, CHUNK])
            pl.load(fw, fn, [sub * 16, 2 * Width + shard * CHUNK])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            range_panel(fw, det_state, weight_rows)
            pl.system.bar_all()
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
            hi0 = h0g.current()
            hi1 = h1g.current()
            lo0 = l0g.current()
            lo1 = l1g.current()
            pack_fn_bf2_128(fw, hi0, lo0, weight_rows, 0)
            pack_fn_bf2_128(fw, hi1, lo1, weight_rows, 128)
            hi = b2hg.current()
            lo = b2lg.current()
            pl.set_validshape(hi0, [16, 128])
            pl.set_validshape(hi1, [16, 128])
            pl.set_validshape(lo0, [16, 128])
            pl.set_validshape(lo1, [16, 128])
            pl.insert(hi, hi0, [sub * 16, 0])
            pl.insert(hi, hi1, [sub * 16, 128])
            pl.insert(lo, lo0, [sub * 16, 0])
            pl.insert(lo, lo1, [sub * 16, 128])
            fw = fwg.current()
            pl.set_validshape(fw, [weight_rows, CHUNK])
            pl.load(fw, fn, [sub * 16, 3 * Width + shard * CHUNK])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            range_panel(fw, det_state, weight_rows)
            pl.system.bar_all()
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
            hi0 = h0g.current()
            hi1 = h1g.current()
            lo0 = l0g.current()
            lo1 = l1g.current()
            pack_fn_bf2_128(fw, hi0, lo0, weight_rows, 0)
            pack_fn_bf2_128(fw, hi1, lo1, weight_rows, 128)
            hi = b3hg.current()
            lo = b3lg.current()
            pl.set_validshape(hi0, [16, 128])
            pl.set_validshape(hi1, [16, 128])
            pl.set_validshape(lo0, [16, 128])
            pl.set_validshape(lo1, [16, 128])
            pl.insert(hi, hi0, [sub * 16, 0])
            pl.insert(hi, hi1, [sub * 16, 128])
            pl.insert(lo, lo0, [sub * 16, 0])
            pl.insert(lo, lo1, [sub * 16, 128])
            det_record = det_recordg.current()
            range_record(det_state, det_record)
            pl.system.bar_all()
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
            pl.set_validshape(det_record, [1, 8])
            pl.store(fwork, det_record, [shard, sub * 8])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.S, event_id=2)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.S, event_id=2)
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=6)
        with pl.section_cube():
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=6)
        with pl.section_cube():
            pl.system.bar_all()
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        with pl.section_vector():
            pl.system.bar_all()
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        with pl.section_vector():
            if pl.get_block_idx() == 0:
                det_all = det_allg.current()
                det_global = det_globalg.current()
                pl.load(det_all, fwork, [0, 0])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                range_merge(det_all, det_global, Width // CHUNK)
                pl.system.bar_all()
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.set_validshape(det_global, [1, 16])
                pl.store(fwork, det_global, [Width // CHUNK, 0])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.S, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.S, event_id=2)
        with pl.section_cube():
            pl.system.bar_all()
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        with pl.section_vector():
            pl.system.bar_all()
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        pl.system.dcci(fwork, [Width // CHUNK, 0], cache_line=pl.CacheLine.SINGLE_CACHE_LINE)
        use_slow = fwork[Width // CHUNK, 0]
        if use_slow == 0:
            with pl.section_vector():
                pl.load(gamma_chunk, gamma, [0, shard * CHUNK])
                pl.set_validshape(s, [1, 3])
                pl.set_validshape(b, [1, 24])
                pl.load(s, scales, [0, 0])
                pl.load(b, bases, [0, 0])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=3)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=3)
            with pl.section_vector():
                prime_sub = pl.get_subblock_idx()
                prime_start = prime_sub * BATCH8
                prime_count = pl.max(0, pl.min(BATCH8, tokens - prime_start))
                prime_rb = rg[0]
                prime_xb = xg[0]
                prime_pp = pg[0]
                prime_cc = cg[0]
                prime_vv = vg[0]
                pl.set_validshape(prime_rb, [prime_count * 4, CHUNK])
                pl.set_validshape(prime_xb, [prime_count, CHUNK])
                pl.set_validshape(prime_pp, [prime_count, 4])
                pl.set_validshape(prime_cc, [prime_count, 16])
                pl.set_validshape(prime_vv, [prime_count, 4])
                if prime_count > 0:
                    pl.load(prime_rb, residual, [prime_start * 4, shard * CHUNK])
                    pl.load(prime_xb, x, [prime_start, shard * CHUNK])
                    pl.load(prime_pp, post, [prime_start, 0])
                    pl.load(prime_cc, comb, [prime_start, 0])
                    pl.load(prime_vv, prev, [prime_start, 0])
            fast_jobs = (tokens + 2 * BATCH8 - 1) // (2 * BATCH8)
            for block in pl.range((tokens + 2 * BATCH8 * BlockJobs - 1) // (2 * BATCH8 * BlockJobs)):
                first_job = block * BlockJobs
                local_jobs = pl.min(BlockJobs, fast_jobs - first_job)
                for beat in pl.range(local_jobs + 1):
                    job = first_job + beat
                    with pl.section_vector():
                        if beat < local_jobs:
                            if block > 0:
                                old_start = (
                                    (block - 1) * (2 * BATCH8 * BlockJobs)
                                    + beat * (2 * BATCH8)
                                    + pl.get_subblock_idx() * BATCH8
                                )
                                old_uu = ug[(first_job - BlockJobs + beat) % 33]
                                old_inverse = invg[beat]
                                pl.set_validshape(old_uu, [BATCH8, CHUNK])
                                _norm_chunk_rows(old_uu, gamma_chunk, old_inverse, BATCH8)
                                pl.system.sync_src(
                                    set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=beat % 8
                                )
                                pl.system.sync_dst(
                                    set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=beat % 8
                                )
                                pl.store(yout, old_uu, [old_start, shard * CHUNK])
                                pl.system.sync_src(
                                    set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=beat % 8
                                )
                                if beat >= 33 - BlockJobs:
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE3,
                                        wait_pipe=pl.PipeType.V,
                                        event_id=(beat - (33 - BlockJobs)) % 8,
                                    )
                            if beat >= 2:
                                pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=2 + job % 2)
                            _prefetch_produce(
                                job,
                                shard,
                                tokens,
                                x,
                                residual,
                                prev,
                                post,
                                comb,
                                rout,
                                rg,
                                xg,
                                pg,
                                cg,
                                vg,
                                ug,
                                partialg,
                                a0g,
                                a1g,
                                a2g,
                                a3g,
                                gz00,
                                gz01,
                                gz10,
                                gz11,
                                gz20,
                                gz21,
                                gz30,
                                gz31,
                                BATCH8,
                                33,
                            )
                            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=job % 2)
                        if beat > 0:
                            pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=4)
                            _overlap_stash(job - 1, shard, tokens, qwork, scoreg, partialg, recordg, BATCH8)
                            if job < fast_jobs:
                                pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=5)
                    with pl.section_cube():
                        if beat < local_jobs:
                            pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=job % 2)
                            if beat > 0:
                                pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=5)
                            _overlap_project(
                                job,
                                a0g,
                                a1g,
                                a2g,
                                a3g,
                                b0hg,
                                b0lg,
                                b1hg,
                                b1lg,
                                b2hg,
                                b2lg,
                                b3hg,
                                b3lg,
                                la,
                                lb,
                                accg,
                                scoreg,
                                BATCH8,
                            )
                            pl.system.set_cross_core(pipe=pl.PipeType.MTE1, event_id=2 + job % 2)
                            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=4)
                with pl.section_vector():
                    if local_jobs > 0:
                        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=2 + (first_job + local_jobs - 1) % 2)
                    if local_jobs > 1:
                        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=2 + (first_job + local_jobs - 2) % 2)
                    if block > 0:
                        for pending_old_job in pl.range(pl.max(0, local_jobs - (33 - BlockJobs)), local_jobs):
                            pl.system.sync_dst(
                                set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=pending_old_job % 8
                            )
                        for old_tail_job in pl.range(local_jobs, BlockJobs):
                            tail_start = (
                                (block - 1) * (2 * BATCH8 * BlockJobs)
                                + old_tail_job * (2 * BATCH8)
                                + pl.get_subblock_idx() * BATCH8
                            )
                            tail_uu = ug[(first_job - BlockJobs + old_tail_job) % 33]
                            tail_inverse = invg[old_tail_job]
                            pl.set_validshape(tail_uu, [BATCH8, CHUNK])
                            _norm_chunk_rows(tail_uu, gamma_chunk, tail_inverse, BATCH8)
                            pl.system.sync_src(
                                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=old_tail_job % 8
                            )
                            pl.system.sync_dst(
                                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=old_tail_job % 8
                            )
                            pl.store(yout, tail_uu, [tail_start, shard * CHUNK])
                            pl.system.sync_src(
                                set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=old_tail_job % 8
                            )
                            pl.system.sync_dst(
                                set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=old_tail_job % 8
                            )
                    pl.system.bar_all()
                    pl.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)
                with pl.section_cube():
                    if local_jobs > 0:
                        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=5)
                    pl.system.bar_all()
                with pl.section_vector():
                    worker = pl.get_block_idx()
                    workers = 2 * (Width // CHUNK)
                    valid_quads = pl.min(BATCH8 // 2 * BlockJobs, (tokens - block * (2 * BATCH8 * BlockJobs) + 3) // 4)
                    first_quad = (worker + workers - block * (BATCH8 // 2 * BlockJobs) % workers) % workers
                    unused_u = ug[0]
                    for local_quad in pl.range(first_quad, valid_quads, workers):
                        token = block * (2 * BATCH8 * BlockJobs) + local_quad * 4
                        count = pl.min(4, tokens - token)
                        pl.set_validshape(records, [count, Width // CHUNK * 32])
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=3)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=3)
                        pl.load(records, qflat, [local_quad * 4, 0])
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=3)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=3)
                        _merge_finish_quad_registers(
                            records,
                            s,
                            b,
                            newpre,
                            newpost,
                            newcomb,
                            invout,
                            count,
                            1.0 / Width,
                            1.0 / (4 * Width),
                            hc_eps,
                            norm_eps,
                            pre_eps,
                            post_scale,
                            repeat,
                            sink_eps,
                        )
                        pl.set_validshape(newpre, [count, 4])
                        pl.set_validshape(newpost, [count, 4])
                        pl.set_validshape(newcomb, [count * 4, 4])
                        pl.set_validshape(invout, [count, 16])
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=3)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=3)
                        pl.store(vout, newpre, [token, 0])
                        pl.store(pout, newpost, [token, 0])
                        pl.store(cout, newcomb, [token * 4, 0])
                        pl.store(invwork, invout, [local_quad * 4, 0])
                    pl.system.bar_all()
                    pl.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)
                    sub = pl.get_subblock_idx()
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=3)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=3)
                    for inv_job in pl.range(local_jobs):
                        inv_start = (first_job + inv_job) * (2 * BATCH8) + sub * BATCH8
                        inv_count = pl.max(0, pl.min(BATCH8, tokens - inv_start))
                        if inv_count > 0:
                            inverse = invg[inv_job]
                            pl.set_validshape(inverse, [inv_count, 16])
                            pl.load(inverse, invwork, [inv_job * (2 * BATCH8) + sub * BATCH8, 0])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=3)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=3)
                    if first_job + local_jobs == fast_jobs:
                        for local_job in pl.range(local_jobs):
                            if local_job >= 8:
                                pl.system.sync_dst(
                                    set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=(local_job - 8) % 8
                                )
                            start = (first_job + local_job) * (2 * BATCH8) + sub * BATCH8
                            count = pl.max(0, pl.min(BATCH8, tokens - start))
                            uu = ug[(first_job + local_job) % 33]
                            inverse = invg[local_job]
                            if count > 0:
                                pl.set_validshape(uu, [count, CHUNK])
                                _norm_chunk_rows(uu, gamma_chunk, inverse, count)
                            pl.system.sync_src(
                                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=local_job % 8
                            )
                            pl.system.sync_dst(
                                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=local_job % 8
                            )
                            if count > 0:
                                pl.store(yout, uu, [start, shard * CHUNK])
                            pl.system.sync_src(
                                set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=local_job % 8
                            )
                        for pending_current_job in pl.range(pl.max(0, local_jobs - 8), local_jobs):
                            pl.system.sync_dst(
                                set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=pending_current_job % 8
                            )
                        pl.system.bar_all()
                        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=5)
        else:
            with pl.section_vector():
                slow_s = slow_sg.current()
                slow_b = slow_bg.current()
                slow_g = slow_gg.current()
                slow_a = slow_ag.current()
                slow_q = slow_qg.current()
                pl.set_validshape(slow_s, [1, 3])
                pl.system.bar_all()
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.load(slow_s, scales, [0, 0])
                pl.system.bar_all()
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.set_validshape(slow_b, [1, 24])
                pl.system.bar_all()
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.load(slow_b, bases, [0, 0])
                pl.system.bar_all()
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.system.bar_all()
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.load(slow_g, gamma, [0, 0])
                pl.system.bar_all()
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                slow_epoch = 0
                slow_slot = 0
                for slow_job in pl.range(
                    pl.get_block_idx(), (tokens + SLOW_BATCH - 1) // SLOW_BATCH, 2 * (Width // CHUNK)
                ):
                    slow_job_count = pl.min(SLOW_BATCH, tokens - slow_job * SLOW_BATCH)
                    if slow_epoch == 0:
                        slow_initial_count = pl.min(SLOW_BATCH, tokens - slow_job * SLOW_BATCH)
                        slow_initial_r = slow_rbg_parent.current()
                        slow_initial_x = slow_xg_parent.current()
                        slow_initial_p = slow_pg_parent.current()
                        slow_initial_c = slow_cg_parent.current()
                        slow_initial_v = slow_vg_parent.current()
                        pl.set_validshape(slow_initial_r, [slow_initial_count * 4, SLOW_CHUNK])
                        pl.set_validshape(slow_initial_x, [slow_initial_count, SLOW_CHUNK])
                        pl.set_validshape(slow_initial_p, [slow_initial_count, 4])
                        pl.set_validshape(slow_initial_c, [slow_initial_count, 16])
                        pl.set_validshape(slow_initial_v, [slow_initial_count, 4])
                        pl.system.bar_all()
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.load(slow_initial_r, residual, [slow_job * SLOW_BATCH * 4, 0])
                        pl.system.bar_all()
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.bar_all()
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.load(slow_initial_x, x, [slow_job * SLOW_BATCH, 0])
                        pl.system.bar_all()
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.bar_all()
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.load(slow_initial_p, post, [slow_job * SLOW_BATCH, 0])
                        pl.system.bar_all()
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.bar_all()
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.load(slow_initial_c, comb, [slow_job * SLOW_BATCH, 0])
                        pl.system.bar_all()
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.bar_all()
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.load(slow_initial_v, prev, [slow_job * SLOW_BATCH, 0])
                        pl.system.bar_all()
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        for slow_ahead in pl.range(SLOW_SLOTS):
                            slow_fw = slow_wg[slow_ahead]
                            pl.system.bar_all()
                            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                            pl.load(slow_fw, fn, [0, slow_ahead % 4 * Width + slow_ahead // 4 * SLOW_CHUNK])
                            pl.system.bar_all()
                            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    if SLOW_BANKS == 2:
                        for slow_local in pl.range(SLOW_BATCH):
                            slow_token = slow_job * SLOW_BATCH + slow_local
                            if slow_token < tokens:
                                slow_rb = slow_rbg[slow_local]
                                slow_rf = slow_rfg[0 % SLOW_BANKS * SLOW_BATCH + slow_local]
                                slow_ux = slow_xg[slow_local]
                                slow_u = slow_ug[slow_local * (Width // SLOW_CHUNK) + 0]
                                slow_p = slow_pg[slow_local]
                                slow_c = slow_cg[slow_local]
                                slow_v = slow_vg[slow_local]
                                slow_st = slow_stg[slow_local]
                                slow_rs = slow_rsg[slow_local]
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                _slow_post_stream(
                                    slow_rb,
                                    slow_rf,
                                    slow_ux,
                                    slow_u,
                                    slow_p,
                                    slow_c,
                                    slow_v,
                                    slow_st,
                                    slow_rs,
                                    0 * SLOW_CHUNK,
                                    1.0 / Width,
                                    1.0 / (4 * Width),
                                    hc_eps,
                                    norm_eps,
                                )
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                if SLOW_BATCH == 1:
                                    pl.system.bar_all()
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    pl.store(rout, slow_rb, [slow_token * 4, 0 * SLOW_CHUNK])
                                    pl.system.bar_all()
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                        if SLOW_BATCH > 1:
                            slow_initial_store = slow_rbg_parent.current()
                            pl.set_validshape(slow_initial_store, [slow_job_count * 4, SLOW_CHUNK])
                            pl.system.bar_all()
                            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                            pl.store(rout, slow_initial_store, [slow_job * SLOW_BATCH * 4, 0 * SLOW_CHUNK])
                            pl.system.bar_all()
                            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    for slow_band in pl.range(Width // SLOW_CHUNK):
                        if SLOW_BANKS == 1:
                            for slow_local in pl.range(SLOW_BATCH):
                                slow_token = slow_job * SLOW_BATCH + slow_local
                                if slow_token < tokens:
                                    slow_rb = slow_rbg[slow_local]
                                    slow_rf = slow_rfg[slow_band % SLOW_BANKS * SLOW_BATCH + slow_local]
                                    slow_ux = slow_xg[slow_local]
                                    slow_u = slow_ug[slow_local * (Width // SLOW_CHUNK) + slow_band]
                                    slow_p = slow_pg[slow_local]
                                    slow_c = slow_cg[slow_local]
                                    slow_v = slow_vg[slow_local]
                                    slow_st = slow_stg[slow_local]
                                    slow_rs = slow_rsg[slow_local]
                                    pl.system.bar_all()
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    _slow_post_stream(
                                        slow_rb,
                                        slow_rf,
                                        slow_ux,
                                        slow_u,
                                        slow_p,
                                        slow_c,
                                        slow_v,
                                        slow_st,
                                        slow_rs,
                                        slow_band * SLOW_CHUNK,
                                        1.0 / Width,
                                        1.0 / (4 * Width),
                                        hc_eps,
                                        norm_eps,
                                    )
                                    pl.system.bar_all()
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    if SLOW_BATCH == 1:
                                        pl.system.bar_all()
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                        )
                                        pl.store(rout, slow_rb, [slow_token * 4, slow_band * SLOW_CHUNK])
                                        pl.system.bar_all()
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                        )
                            if SLOW_BATCH > 1:
                                slow_current_store = slow_rbg_parent.current()
                                pl.set_validshape(slow_current_store, [slow_job_count * 4, SLOW_CHUNK])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.store(rout, slow_current_store, [slow_job * SLOW_BATCH * 4, slow_band * SLOW_CHUNK])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        slow_r0 = slow_rfg[slow_band % SLOW_BANKS * SLOW_BATCH]
                        slow_r1 = slow_rfg[slow_band % SLOW_BANKS * SLOW_BATCH + pl.min(1, SLOW_BATCH - 1)]
                        slow_r2 = slow_rfg[slow_band % SLOW_BANKS * SLOW_BATCH + pl.min(2, SLOW_BATCH - 1)]
                        slow_r3 = slow_rfg[slow_band % SLOW_BANKS * SLOW_BATCH + pl.min(3, SLOW_BATCH - 1)]
                        for slow_part in pl.range(4):
                            slow_step = slow_band * 4 + slow_part
                            slow_route = slow_part
                            slow_w = slow_wg[slow_slot]
                            if slow_band == 0 and slow_route == 0:
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                _slow_dot_first(slow_r0, slow_r1, slow_r2, slow_r3, slow_w, slow_a, slow_q, slow_route)
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                            elif slow_band + 1 == Width // SLOW_CHUNK and slow_route == 3:
                                if SLOW_BATCH == 1:
                                    slow_fused_stat = slow_st_parent.current()
                                    slow_fused_prev = slow_pongv.current()
                                    slow_fused_post = slow_pongp.current()
                                    slow_fused_comb = slow_cog.current()
                                    pl.system.bar_all()
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    _slow_dot_last_finish_single(
                                        slow_r0,
                                        slow_r1,
                                        slow_r2,
                                        slow_r3,
                                        slow_w,
                                        slow_a,
                                        slow_q,
                                        slow_route,
                                        slow_fused_stat,
                                        slow_s,
                                        slow_b,
                                        slow_fused_prev,
                                        slow_fused_post,
                                        slow_fused_comb,
                                        pre_eps,
                                        post_scale,
                                        repeat,
                                        sink_eps,
                                        1,
                                    )
                                    pl.system.bar_all()
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                else:
                                    pl.system.bar_all()
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    _slow_dot_last(
                                        slow_r0, slow_r1, slow_r2, slow_r3, slow_w, slow_a, slow_q, slow_route
                                    )
                                    pl.system.bar_all()
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                            else:
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                _slow_dot_accum(slow_r0, slow_r1, slow_r2, slow_r3, slow_w, slow_a, slow_q, slow_route)
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                            if slow_part == SLOW_PREFETCH and slow_band + 1 < Width // SLOW_CHUNK:
                                slow_hidden_count = pl.min(SLOW_BATCH, tokens - slow_job * SLOW_BATCH)
                                slow_hn_r = slow_rbg_parent.current()
                                slow_hn_x = slow_xg_parent.current()
                                pl.set_validshape(slow_hn_r, [slow_hidden_count * 4, SLOW_CHUNK])
                                pl.set_validshape(slow_hn_x, [slow_hidden_count, SLOW_CHUNK])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.load(slow_hn_r, residual, [slow_job * SLOW_BATCH * 4, (slow_band + 1) * SLOW_CHUNK])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.load(slow_hn_x, x, [slow_job * SLOW_BATCH, (slow_band + 1) * SLOW_CHUNK])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                            if (
                                slow_step + SLOW_SLOTS < 4 * (Width // SLOW_CHUNK)
                                or slow_job + 2 * (Width // CHUNK) < (tokens + SLOW_BATCH - 1) // SLOW_BATCH
                            ):
                                slow_wf = slow_wg[slow_slot]
                                slow_future = slow_step + SLOW_SLOTS
                                if slow_future >= 4 * (Width // SLOW_CHUNK):
                                    slow_future = slow_future - 4 * (Width // SLOW_CHUNK)
                                slow_fband = slow_future // 4
                                slow_froute = slow_future % 4
                                slow_fout = 0
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.load(slow_wf, fn, [slow_fout, slow_froute * Width + slow_fband * SLOW_CHUNK])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                            if (
                                slow_band + 1 == Width // SLOW_CHUNK
                                and slow_part == 0
                                and (slow_job + 2 * (Width // CHUNK) < (tokens + SLOW_BATCH - 1) // SLOW_BATCH)
                            ):
                                slow_next_job = slow_job + 2 * (Width // CHUNK)
                                slow_next_count = pl.min(SLOW_BATCH, tokens - slow_next_job * SLOW_BATCH)
                                slow_next_r = slow_rbg_parent.current()
                                slow_next_x = slow_xg_parent.current()
                                slow_next_p = slow_pg_parent.current()
                                slow_next_c = slow_cg_parent.current()
                                slow_next_v = slow_vg_parent.current()
                                pl.set_validshape(slow_next_r, [slow_next_count * 4, SLOW_CHUNK])
                                pl.set_validshape(slow_next_x, [slow_next_count, SLOW_CHUNK])
                                pl.set_validshape(slow_next_p, [slow_next_count, 4])
                                pl.set_validshape(slow_next_c, [slow_next_count, 16])
                                pl.set_validshape(slow_next_v, [slow_next_count, 4])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.load(slow_next_r, residual, [slow_next_job * SLOW_BATCH * 4, 0])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.load(slow_next_x, x, [slow_next_job * SLOW_BATCH, 0])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.load(slow_next_p, post, [slow_next_job * SLOW_BATCH, 0])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.load(slow_next_c, comb, [slow_next_job * SLOW_BATCH, 0])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.load(slow_next_v, prev, [slow_next_job * SLOW_BATCH, 0])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                            if SLOW_BANKS == 2 and slow_part == 4 - SLOW_LEAD and (slow_band + 1 < Width // SLOW_CHUNK):
                                for slow_local in pl.range(SLOW_BATCH):
                                    slow_token = slow_job * SLOW_BATCH + slow_local
                                    if slow_token < tokens:
                                        slow_rb = slow_rbg[slow_local]
                                        slow_rf = slow_rfg[(slow_band + 1) % SLOW_BANKS * SLOW_BATCH + slow_local]
                                        slow_ux = slow_xg[slow_local]
                                        slow_u = slow_ug[slow_local * (Width // SLOW_CHUNK) + (slow_band + 1)]
                                        slow_p = slow_pg[slow_local]
                                        slow_c = slow_cg[slow_local]
                                        slow_v = slow_vg[slow_local]
                                        slow_st = slow_stg[slow_local]
                                        slow_rs = slow_rsg[slow_local]
                                        pl.system.bar_all()
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                        )
                                        _slow_post_stream(
                                            slow_rb,
                                            slow_rf,
                                            slow_ux,
                                            slow_u,
                                            slow_p,
                                            slow_c,
                                            slow_v,
                                            slow_st,
                                            slow_rs,
                                            (slow_band + 1) * SLOW_CHUNK,
                                            1.0 / Width,
                                            1.0 / (4 * Width),
                                            hc_eps,
                                            norm_eps,
                                        )
                                        pl.system.bar_all()
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                        )
                                        pl.system.sync_src(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                        )
                                        pl.system.sync_dst(
                                            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                        )
                                        if SLOW_BATCH == 1:
                                            pl.system.bar_all()
                                            pl.system.sync_src(
                                                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0
                                            )
                                            pl.system.sync_dst(
                                                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0
                                            )
                                            pl.system.sync_src(
                                                set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
                                            )
                                            pl.system.sync_dst(
                                                set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
                                            )
                                            pl.system.sync_src(
                                                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
                                            )
                                            pl.system.sync_dst(
                                                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
                                            )
                                            pl.system.sync_src(
                                                set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1
                                            )
                                            pl.system.sync_dst(
                                                set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1
                                            )
                                            pl.system.sync_src(
                                                set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                            )
                                            pl.system.sync_dst(
                                                set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                            )
                                            pl.system.sync_src(
                                                set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                            )
                                            pl.system.sync_dst(
                                                set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                            )
                                            pl.store(rout, slow_rb, [slow_token * 4, (slow_band + 1) * SLOW_CHUNK])
                                            pl.system.bar_all()
                                            pl.system.sync_src(
                                                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0
                                            )
                                            pl.system.sync_dst(
                                                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0
                                            )
                                            pl.system.sync_src(
                                                set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
                                            )
                                            pl.system.sync_dst(
                                                set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
                                            )
                                            pl.system.sync_src(
                                                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
                                            )
                                            pl.system.sync_dst(
                                                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
                                            )
                                            pl.system.sync_src(
                                                set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1
                                            )
                                            pl.system.sync_dst(
                                                set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1
                                            )
                                            pl.system.sync_src(
                                                set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                            )
                                            pl.system.sync_dst(
                                                set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                            )
                                            pl.system.sync_src(
                                                set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                            )
                                            pl.system.sync_dst(
                                                set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                            )
                                if SLOW_BATCH > 1:
                                    slow_next_store = slow_rbg_parent.current()
                                    pl.set_validshape(slow_next_store, [slow_job_count * 4, SLOW_CHUNK])
                                    pl.system.bar_all()
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    pl.store(
                                        rout, slow_next_store, [slow_job * SLOW_BATCH * 4, (slow_band + 1) * SLOW_CHUNK]
                                    )
                                    pl.system.bar_all()
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2
                                    )
                                    pl.system.sync_src(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                                    pl.system.sync_dst(
                                        set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2
                                    )
                            slow_norm_start = (Width // SLOW_CHUNK - 1) * 4
                            if SLOW_BANKS == 2 and Width // SLOW_CHUNK > 1:
                                slow_norm_start = slow_norm_start - SLOW_LEAD
                            if slow_step == slow_norm_start:
                                slow_ua = slow_up.current()
                                slow_sa = slow_st_parent.current()
                                slow_count = pl.min(SLOW_BATCH, tokens - slow_job * SLOW_BATCH)
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                _slow_norm_all(slow_ua, slow_g, slow_sa, slow_count)
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.set_validshape(slow_ua, [slow_count, Width])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.store(yout, slow_ua, [slow_job * SLOW_BATCH, 0])
                                pl.system.bar_all()
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                            slow_slot = slow_slot + 1
                            if slow_slot == SLOW_SLOTS:
                                slow_slot = 0
                    slow_finish_count = pl.min(SLOW_BATCH, tokens - slow_job * SLOW_BATCH)
                    slow_os = slow_st_parent.current()
                    slow_ov = slow_pongv.current()
                    slow_op = slow_pongp.current()
                    slow_oc = slow_cog.current()
                    if SLOW_BATCH == 2:
                        pl.system.bar_all()
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        _slow_finish_pair(
                            slow_q,
                            slow_os,
                            slow_s,
                            slow_b,
                            slow_ov,
                            slow_op,
                            slow_oc,
                            pre_eps,
                            post_scale,
                            repeat,
                            sink_eps,
                            slow_finish_count,
                        )
                        pl.system.bar_all()
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.set_validshape(slow_ov, [slow_finish_count, 4])
                    pl.set_validshape(slow_op, [slow_finish_count, 4])
                    pl.set_validshape(slow_oc, [slow_finish_count * 4, 4])
                    pl.system.bar_all()
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.store(vout, slow_ov, [slow_job * SLOW_BATCH, 0])
                    pl.system.bar_all()
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.system.bar_all()
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.store(pout, slow_op, [slow_job * SLOW_BATCH, 0])
                    pl.system.bar_all()
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.system.bar_all()
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.store(cout, slow_oc, [slow_job * SLOW_BATCH * 4, 0])
                    pl.system.bar_all()
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    slow_epoch = slow_epoch + 1


# Host wrapper for development; .run consumers use aclnnMegaMhc directly.
def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_nonnegative(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def select_config(tokens, hidden, *, batch=None, chunk=256, slots=2, workers=None):
    if not _is_int(tokens) or tokens < 0:
        raise ValueError('T must be a nonnegative integer')
    if not _is_int(hidden) or hidden not in range(1024, 7169, 1024):
        raise ValueError('H must be a multiple of 1024 in [1024,7168]')
    if batch is not None and (not _is_int(batch) or batch != 16):
        raise ValueError('public batch must be None or 16')
    if not _is_int(chunk) or chunk != 256:
        raise ValueError('chunk=256 and slots=2 are required')
    if not _is_int(slots) or slots != 2:
        raise ValueError('chunk=256 and slots=2 are required')
    shards = hidden // 256
    if workers is not None and (not _is_int(workers) or workers != shards):
        raise ValueError('workers must be H//256')
    b8 = batch is None and hidden == 7168 and tokens >= 4096
    actual_batch = 8 if b8 else 16
    jobs = 27 if b8 else (7 if hidden == 7168 else 8)
    rows = min(tokens, 2 * actual_batch * jobs)

    def align(value):
        return (value + 511) // 512 * 512

    inv_offset = align(rows * shards * 32 * 4)
    flags_offset = inv_offset + align(rows * 16 * 4)
    return dict(
        batch=actual_batch,
        J=jobs,
        block_jobs=jobs,
        workers=shards,
        workspace_rows=rows,
        block_tokens=2 * actual_batch * jobs,
        u_slots=33 if b8 else 9,
        chunk=256,
        slots=2,
        inv_offset=inv_offset,
        flags_offset=flags_offset,
        workspace_bytes=max(1024, flags_offset + align((shards + 1) * 16 * 4)),
        tiling_key=dict(Width=hidden, BlockJobs=jobs),
    )


def allocate_outputs(x):
    t, h = x.shape
    return (
        torch.empty((t, 4, h), dtype=torch.bfloat16, device=x.device),
        torch.empty((t, 4, 1), dtype=torch.float32, device=x.device),
        torch.empty((t, 4, 4), dtype=torch.float32, device=x.device),
        torch.empty_like(x),
        torch.empty((t, 4), dtype=torch.float32, device=x.device),
    )


def allocate_workspace(x, *, batch=None):
    cfg = select_config(*x.shape, batch=batch)
    return torch.empty(cfg['workspace_bytes'], dtype=torch.uint8, device=x.device)


def _check_tensor(tensor, shape, dtype, device):
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
        raise ValueError(f'expected dense {shape} {dtype} on {device}')
    if tensor.dtype != dtype or tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f'expected dense {shape} {dtype} on {device}')


def _check_output_aliases(inputs, outputs):
    occupied = [(t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in inputs]
    for t in outputs:
        a, b = t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()
        if a != b:
            if any(c != d and a < d and c < b for c, d in occupied):
                raise ValueError('writable tensors must not overlap inputs or each other')
            occupied.append((a, b))


def mega_mhc(
    x,
    residual,
    previous_mix,
    post_mix,
    comb_res_mix,
    fn,
    mix_scales,
    mix_bases,
    rmsnorm_weight,
    *,
    hc_norm_eps=2e-5,
    hc_pre_eps=3e-4,
    hc_post_scale=1.25,
    sinkhorn_eps=2e-6,
    num_sinkhorn_iters=10,
    rmsnorm_eps=7e-6,
    workspace=None,
    batch=None,
    chunk=256,
    slots=2,
    workers=None,
):
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError('x must be a rank-2 BF16 NPU tensor')
    if x.dtype != torch.bfloat16 or x.device.type != 'npu':
        raise ValueError('x must be a rank-2 BF16 NPU tensor')
    scalars = (hc_norm_eps, hc_pre_eps, hc_post_scale, sinkhorn_eps, rmsnorm_eps)
    if not all(_is_finite_nonnegative(value) for value in scalars):
        raise ValueError('eps and post scale must be finite nonnegative scalars')
    if hc_norm_eps <= 0 or rmsnorm_eps <= 0:
        raise ValueError('the two RMS eps values must be positive')
    if type(num_sinkhorn_iters) is not int or not 1 <= num_sinkhorn_iters <= 65535:
        raise ValueError('num_sinkhorn_iters must be an integer in [1,65535]')
    t, h = x.shape
    cfg = select_config(t, h, batch=batch, chunk=chunk, slots=slots, workers=workers)
    inputs = (x, residual, previous_mix, post_mix, comb_res_mix, fn, mix_scales, mix_bases, rmsnorm_weight)
    specs = (
        ((t, h), torch.bfloat16),
        ((t, 4, h), torch.bfloat16),
        ((t, 4), torch.float32),
        ((t, 4, 1), torch.float32),
        ((t, 4, 4), torch.float32),
        ((24, 4 * h), torch.float32),
        ((3,), torch.float32),
        ((24,), torch.float32),
        ((h,), torch.bfloat16),
    )
    for value, (shape, dtype) in zip(inputs, specs):
        _check_tensor(value, shape, dtype, x.device)
    outputs = allocate_outputs(x)
    out_specs = (
        ((t, 4, h), torch.bfloat16),
        ((t, 4, 1), torch.float32),
        ((t, 4, 4), torch.float32),
        ((t, h), torch.bfloat16),
        ((t, 4), torch.float32),
    )
    for value, (shape, dtype) in zip(outputs, out_specs):
        _check_tensor(value, shape, dtype, x.device)
    workspace = allocate_workspace(x, batch=batch) if workspace is None else workspace
    _check_tensor(workspace, (cfg['workspace_bytes'],), torch.uint8, x.device)
    _check_output_aliases(inputs, outputs + (workspace,))
    if t == 0:
        return outputs
    tiling = MegaMhcTiling(
        t,
        cfg['workspace_rows'],
        cfg['inv_offset'],
        cfg['flags_offset'],
        num_sinkhorn_iters,
        hc_norm_eps,
        hc_pre_eps,
        hc_post_scale,
        sinkhorn_eps,
        rmsnorm_eps,
    )
    with pypto.options(pass_options={'enable_slice': False}):
        mega_mhc_kernel[None, cfg['workers'], cfg['tiling_key']](*inputs, *outputs, workspace, tiling)
    return outputs
