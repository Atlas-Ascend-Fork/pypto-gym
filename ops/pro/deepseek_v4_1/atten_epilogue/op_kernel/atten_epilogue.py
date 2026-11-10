# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F821
"""Inverse RoPE + two-stage OCP MXFP8 attention output projection (PyPTO-Pro).

For an attention output ``o[T, 64, 512]`` (bf16) the fused kernel applies the inverse
RoPE of the head layout, regroups and quantizes the activation to OCP MXFP8, and runs
two block-scaled GEMMs against the pre-quantized ND weights ``woa`` and ``wob``,
writing a contiguous bf16 ``[T, 5120]`` projection.

I/O contract: ``o`` is bf16 ``[T, 64, 512]``; ``woa`` is FP8E4M3 ``[8, 1024, 4096]``
with E8M0 block scales ``descale_woa [8, 1024, 64, 2]``; ``wob`` is FP8E4M3
``[5120, 8192]`` with ``descale_wob [5120, 128, 2]``; ``rope_sin``/``rope_cos`` are
fp32 ``[T, 64]``.

All four phases belong to one native task. AQ/YQ GM payloads carry data across
Vector RoPE/quantization and the two Cube GEMMs. Slab-ready notifications and the
core-grid barriers order producers and consumers while permitting overlap. Tiling
constants and the capacity budgets behind them are described on ``EpilogueKey`` and in
the README.

Delivery form (binary / aclnn): the kernel takes raw ``pl.Ptr`` I/O plus one trailing
``EpilogueTiling`` struct, exactly the layout ``op_host/atten_epilogue_def.cpp``
registers (inputs, then outputs, then workspace, then tiling).  The four GM payloads
(``aq/asc/yq/ys``) are no longer caller-allocated tensors: they are carved out of the
single framework workspace whose size the host tiling reports
(``WORKSPACE_BYTES_PER_ROW * padded``), so the op needs no user-visible scratch.
"""

import dataclasses
import threading
from collections import OrderedDict

import pypto
import pypto_pro.language as pl
import torch
import torch_npu
from pypto_pro.language import Vf as vf
from pypto_pro.runtime.platform import get_platform_info
from pypto_pro.runtime.tilingkey import TilingKeyField

FP8, E8 = pl.DT_FP8E4M3FN, pl.DT_FP8E8M0


Q_ROWS = 64
R_ROWS = 8

# Fixed (T-independent) problem dimensions of the fused projection.
HEADS = 64
HEAD_DIM = 512
GROUPS = 8
GROUP_N = 1024
GEMM1_K = 4096
OUT_FEATURES = 5120
GEMM2_K = 8192
# T is padded up to this multiple: the M1 row split and the C->V half-tile hand-off
# both need an even 16-row granularity.
PAD_MULTIPLE = 16

# GM workspace layout (bytes), one flat buffer carved by pl.addptr in the kernel and
# sized by the host tiling.  Offsets are multiples of 512 because `padded` is a
# multiple of 16, so every payload stays 512-byte aligned.
AQ_BYTES_PER_ROW = GROUPS * GEMM1_K  # 32768
ASC_BYTES_PER_ROW = GROUPS * GEMM1_K // 32  # 1024
YQ_BYTES_PER_ROW = GEMM2_K  # 8192
YS_BYTES_PER_ROW = GEMM2_K // 32  # 256
ASC_OFFSET_PER_ROW = AQ_BYTES_PER_ROW
YQ_OFFSET_PER_ROW = ASC_OFFSET_PER_ROW + ASC_BYTES_PER_ROW
YS_OFFSET_PER_ROW = YQ_OFFSET_PER_ROW + YQ_BYTES_PER_ROW
WORKSPACE_BYTES_PER_ROW = YS_OFFSET_PER_ROW + YS_BYTES_PER_ROW  # 42240


@dataclasses.dataclass
class EpilogueTiling:
    """Runtime tiling struct (packed C-struct, last kernel argument).

    Field NAMES, ORDER and TYPES must match ``EpilogueTilingData`` in
    op_host/atten_epilogue_tiling.h exactly (PyPTO lowers ``int`` to
    int64_t); cmake/scripts/tiling_layout_check.py enforces this at configure
    time and a layout check program re-verifies it before the tiling compile.
    """

    t: int  # effective token count (rows written to `out`)
    padded: int  # t rounded up to PAD_MULTIPLE (rows of the GM payloads)


def workspace_bytes(t):
    """Flat GM workspace size in bytes for `t` tokens."""
    return WORKSPACE_BYTES_PER_ROW * (
        (t + PAD_MULTIPLE - 1) // PAD_MULTIPLE * PAD_MULTIPLE
    )


class EpilogueKey:
    """Finite native profiles selected by T and the actual Cube core count.

    Algo: 0 generic, 1 small-M, 2 paired tail, 3 medium, 4 large/pipelined.
    MCAP bounds the physical small-M accumulator; T remains runtime data.
    """

    BM = TilingKeyField(bits=2, values=[64, 128, 256])
    BK = TilingKeyField(bits=1, values=[256, 512])
    VSlots = TilingKeyField(bits=1, values=[1, 2])
    K0 = TilingKeyField(bits=1, values=[128, 256])
    BN1 = TilingKeyField(bits=2, values=[128, 160, 256])
    BN = TilingKeyField(bits=1, values=[160, 192])
    Algo = TilingKeyField(bits=3, values=[0, 1, 2, 3, 4])
    MCAP = TilingKeyField(bits=2, values=[16, 32, 48, 64])

    PROFILES = (
        (64, 512, 1, 256, 256, 160, 1, 16),
        (64, 512, 2, 256, 128, 192, 1, 16),
        (64, 512, 1, 256, 256, 160, 1, 32),
        (64, 512, 2, 256, 128, 192, 1, 32),
        (64, 512, 1, 256, 256, 160, 1, 48),
        (64, 512, 2, 256, 128, 192, 1, 48),
        (64, 512, 1, 256, 256, 160, 1, 64),
        (64, 512, 2, 256, 128, 192, 1, 64),
        (128, 256, 1, 128, 256, 160, 3, 16),
        (256, 256, 1, 128, 256, 160, 3, 16),
        (256, 256, 1, 128, 256, 160, 2, 16),
        (256, 256, 1, 128, 256, 160, 4, 16),
        (256, 256, 1, 128, 256, 160, 0, 16),
        (128, 256, 2, 128, 160, 192, 4, 16),
        (256, 256, 1, 128, 160, 192, 4, 16),
        (256, 256, 1, 128, 256, 192, 4, 16),
        (64, 512, 2, 256, 160, 192, 0, 16),
        (128, 256, 2, 256, 160, 192, 0, 16),
        (256, 256, 1, 256, 160, 192, 0, 16),
    )

    def is_valid(self, key):
        """Accept only profiles reachable through the public capacity selector."""
        return key in self.PROFILES


@pl.vector_function
def _apply_inverse_rope(x, cos, sin):
    """Rotate one 512-wide head slice by the cached cos/sin tables (inverse RoPE), in place."""
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    i = vf.arange(0, dtype=pl.DT_UINT32)
    one = vf.full(1, dtype=pl.DT_UINT32)
    swapped = vf.xor(i, one, p)
    parity = vf.and_(i, one, p)
    odd = vf.eq(parity, one, p)
    c = vf.load_align(cos, 0)
    s = vf.load_align(sin, 0)
    ns = vf.muls(s, -1.0, p)
    signed_s = vf.select(ns, s, odd)
    for h in pl.range(8):
        b = vf.load_align(x, h * 512 + 448, dist=pl.LoadDist.UNPK_B16)
        f = vf.astype(b, p, dtype=pl.DT_FP32)
        partner = vf.gather(f, swapped)
        a = vf.mul(f, c, p)
        v = vf.mul(partner, signed_s, p)
        r = vf.add(a, v, p)
        bf = vf.astype(
            r,
            p,
            dtype=pl.DT_BF16,
            round_mode=pl.VFRoundMode.CAST_RINT,
            saturate=pl.SaturateMode.OFF,
        )
        vf.store_align(x, bf, p, h * 512 + 448, dist=pl.StoreDist.PACK_B32)


@pl.vector_function
def _compute_mx_block_scales(x, s, reciprocal, count: pl.DT_INT64):
    # Two deinterleaved b16 registers cover 256 values. A 16-lane block maximum
    # therefore covers 32 consecutive input values. Scale bytes use even b8
    # positions after narrowing; squeeze emits one byte for each 32-value group.
    """Per-32-value OCP MX block scale (E8M0) plus the reciprocal used by the quantizer."""
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    p8 = vf.create_mask(pattern=pl.MaskPattern.VL16, dtype=pl.DT_UINT8)
    short16 = vf.create_mask(pattern=pl.MaskPattern.VL16, dtype=pl.DT_UINT16)
    em = vf.full(0x7F80, dtype=pl.DT_UINT16)
    eight = vf.full(8, dtype=pl.DT_UINT16)
    c262 = vf.full(262, dtype=pl.DT_UINT16)
    zero = vf.full(0, dtype=pl.DT_UINT16)
    idx = vf.arange(0, dtype=pl.DT_UINT8)
    one = vf.full(1, dtype=pl.DT_UINT8)
    parity = vf.and_(idx, one, p8)
    choose = vf.eq(parity, 0, p8)
    vf.clear_spr()
    ar = vf.unalign_reg_for_store()
    for chunk in pl.range(count // 256):
        x0, x1 = vf.load_align(x, chunk * 256, dist=pl.LoadDist.DINTLV_B16)
        bits0 = vf.bit_cast(x0, dtype=pl.DT_UINT16)
        bits1 = vf.bit_cast(x1, dtype=pl.DT_UINT16)
        a0 = vf.and_(bits0, em, p16)
        a1 = vf.and_(bits1, em, p16)
        both = vf.max(a0, a1, p16)
        gm = vf.reduce_max(both, p16, datablock=True)
        exponent = vf.shift_right(gm, 7, p16)
        e = vf.maxs(exponent, 8, p16)
        sb = vf.sub(e, eight, p16)
        sb8 = vf.astype(sb, p16, dtype=pl.DT_UINT8)
        dense = vf.squeeze(sb8, choose, gather_mode=pl.SqueezeMode.STORE_REG)
        vf.squeeze_store_unalign(s, dense, ar)
        reverse = vf.sub(c262, e, p16)
        rev = vf.shift_left(reverse, 7, p16)
        tiny = vf.eq(e, eight, p16)
        rev = vf.select(zero, rev, tiny)
        dup, _unused = vf.interleave(rev, rev)
        vf.store_align(reciprocal, dup, short16, chunk * 16)
    vf.squeeze_store_unalign_post(s, ar)


@pl.vector_function
def _quantize_to_fp8_payload(x, q, reciprocal, even, odd, count: pl.DT_INT64):
    """Quantize bf16 values to FP8E4M3 with the reciprocals produced by _compute_mx_block_scales."""
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    p32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    for chunk in pl.range(count // 256):
        for half in pl.range(2):
            off = chunk * 256 + half * 128
            xv = vf.load_align(x, off)
            rs = vf.load_align(
                reciprocal, chunk * 16 + half * 8, dist=pl.LoadDist.E2B_B16
            )
            reciprocal_bf = vf.bit_cast(rs, dtype=pl.DT_BF16)
            y = vf.mul(xv, reciprocal_bf, p16)
            yz = vf.astype(y, p16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
            yo = vf.astype(y, p16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
            # Explicit finite E4M3 saturation also covers values rounding past 448.
            qz = vf.astype(yz, p32, dtype=FP8, saturate=pl.SaturateMode.ON)
            qo = vf.astype(yo, p32, dtype=FP8, saturate=pl.SaturateMode.ON)
            vf.store_align(
                even, qz, p32, chunk * 128 + half * 64, dist=pl.StoreDist.PACK4_B32
            )
            vf.store_align(
                odd, qo, p32, chunk * 128 + half * 64, dist=pl.StoreDist.PACK4_B32
            )
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    for chunk in pl.range(count // 256):
        for half in pl.range(2):
            a = vf.load_align(even, chunk * 128 + half * 64)
            b = vf.load_align(odd, chunk * 128 + half * 64)
            vf.store_align(
                q + chunk * 256 + half * 128, a, b, p8, dist=pl.StoreDist.INTLV_B8
            )


@pl.vector_function
def _zero_fill_quantized_tile(q, s, count: pl.DT_INT64):
    """Emit zero payload and zero scales for an empty (fully masked) tile."""
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    zero = vf.full(0, dtype=pl.DT_UINT8)
    for i in pl.range(count // 256):
        vf.store_align(q, zero, p, i * 256)
    ps = vf.update_mask(count // 32, dtype=pl.DT_UINT8)
    vf.store_align(s, zero, ps, 0)


@pl.vector_function
def _cast_accumulator_to_bf16(src, bf, count: pl.DT_INT64, offset: pl.DT_INT64):
    """Cast the fp32 GEMM1 accumulator slab to bf16 for the MX quantizer."""
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for i in pl.range(count // 64):
        f = vf.load_align(src, offset + i * 64)
        b = vf.astype(
            f,
            p,
            dtype=pl.DT_BF16,
            round_mode=pl.VFRoundMode.CAST_RINT,
            saturate=pl.SaturateMode.OFF,
        )
        vf.store_align(bf, b, p, i * 64, dist=pl.StoreDist.PACK_B32)


@pl.vector_function
def _pack_mx_scale_bytes(src, dst, rows: pl.DT_INT64, cols: pl.DT_INT64):
    """Pack per-32-value scale bytes into the 2-byte E8M0 layout GEMM2 reads."""
    i = vf.arange(0, dtype=pl.DT_UINT8)
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    valid = vf.update_mask(cols, dtype=pl.DT_UINT8)
    for r in pl.range(rows):
        v = vf.load_align(src, (r * cols // 32) * 32)
        index = vf.adds(i, r * cols % 32, p)
        packed = vf.gather(v, index)
        vf.store_align(dst, packed, valid, r * 32)


@pl.vector_function
def _apply_inverse_rope_slab(x, cos, sin, rows: pl.DT_INT64):
    """Inverse RoPE for a row slab (partial tile) of the activation."""
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    i = vf.arange(0, dtype=pl.DT_UINT32)
    one = vf.full(1, dtype=pl.DT_UINT32)
    swapped = vf.xor(i, one, p)
    parity = vf.and_(i, one, p)
    odd = vf.eq(parity, one, p)
    for row in pl.range(rows):
        c = vf.load_align(cos, row * 64)
        s = vf.load_align(sin, row * 64)
        ns = vf.muls(s, -1.0, p)
        signed_s = vf.select(ns, s, odd)
        for h in pl.range(4):
            b = vf.load_align(x, row * 2048 + h * 512 + 448, dist=pl.LoadDist.UNPK_B16)
            f = vf.astype(b, p, dtype=pl.DT_FP32)
            partner = vf.gather(f, swapped)
            a = vf.mul(f, c, p)
            v = vf.mul(partner, signed_s, p)
            r = vf.add(a, v, p)
            bf = vf.astype(
                r,
                p,
                dtype=pl.DT_BF16,
                round_mode=pl.VFRoundMode.CAST_RINT,
                saturate=pl.SaturateMode.OFF,
            )
            vf.store_align(
                x, bf, p, row * 2048 + h * 512 + 448, dist=pl.StoreDist.PACK_B32
            )


@pl.vector_function
def _zero_fill_quantized_tail(q, s, offset: pl.DT_INT64, count: pl.DT_INT64):
    """Zero-fill the tail rows of a quantized tile."""
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    zero = vf.full(0, dtype=pl.DT_UINT8)
    for i in pl.range(count // 256):
        vf.store_align(q, zero, p, offset + i * 256)
    ps = vf.update_mask(count // 32, dtype=pl.DT_UINT8)
    vf.store_align(s, zero, ps, offset // 32)


@pl.vector_function
def _zero_fill_quantized_batch(q, s, count: pl.DT_INT64):
    """Zero-fill a batch of quantized tiles in one pass."""
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    zero = vf.full(0, dtype=pl.DT_UINT8)
    for i in pl.range(count // 256):
        vf.store_align(q, zero, p, i * 256)
    for block in pl.range(count // 8192):
        vf.store_align(s, zero, p, block * 256)
    remainder = (count // 32) % 256
    if remainder > 0:
        ps = vf.update_mask(remainder, dtype=pl.DT_UINT8)
        vf.store_align(s, zero, ps, (count // 8192) * 256)


@pl.vector_function
def _zero_fill_quantized_batch_tail(q, s, offset: pl.DT_INT64, count: pl.DT_INT64):
    """Zero-fill the tail rows of a batch of quantized tiles."""
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    zero = vf.full(0, dtype=pl.DT_UINT8)
    for i in pl.range(count // 256):
        vf.store_align(q, zero, p, offset + i * 256)
    for block in pl.range(count // 8192):
        vf.store_align(s, zero, p, offset // 32 + block * 256)
    remainder = (count // 32) % 256
    if remainder > 0:
        ps = vf.update_mask(remainder, dtype=pl.DT_UINT8)
        vf.store_align(s, zero, ps, offset // 32 + (count // 8192) * 256)


@pl.pipeline.stage
def _gemm1_tile(
    work,
    t,
    padded,
    a,
    sa,
    b,
    sb,
    wb,
    wbg,
    ag,
    bg,
    b_transposed,
    asg,
    bsg,
    al,
    als,
    bl,
    bls,
    accg,
    shared,
):
    """GEMM1 tile (group, m, n): acc = a[group, m] @ woa[group, n]^T, drained to the C->V ring."""
    nm = (t + BM - 1) // BM
    nn = (1024 + BN1 - 1) // BN1
    if BN1 == 256:
        group = (work // nn) % 8
        mi = work // (8 * nn) * BM
    else:
        group = work // (nm * nn)
        mi = work // nn % nm * BM
    ni = work % nn * BN1
    mv = pl.min(BM, padded - mi)
    nv = pl.min(BN1, 1024 - ni)
    acc = accg.next()
    ast = asg.current()
    bst = bsg.current()
    pl.set_validshape(acc, [mv, nv])
    pl.set_validshape(ast, [mv, 64 if BN1 == 256 else 128])
    pl.set_validshape(ag, [mv, BK])
    pl.set_validshape(al, [mv, K0])
    pl.set_validshape(als, [mv, K0 // 32])
    pl.set_validshape(bg, [nv, BK])
    pl.set_validshape(b_transposed, [BK, nv])
    pl.set_validshape(bst, [128, nv])
    pl.set_validshape(bl, [K0, nv])
    pl.set_validshape(bls, [K0 // 32, nv])
    if BN1 != 256:
        pl.load(ast, sa, [group * padded + mi, 0, 0])
    if work != pl.get_block_idx():
        pl.load(bst, sb, [group * 1024 + ni, 0, 0], order=[1, 0])
    for ki in pl.range(0, 4096, BK):
        if BN1 == 256 and ki % 2048 == 0:
            # All 32 AICs have a first tile; later tiles reuse immutable slabs.
            if work == pl.get_block_idx():
                if ki == 0:
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=3)
                else:
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=4)
            pl.load(ast, sa, [group * padded + mi, ki // 64, 0])
        at = ag.next()
        bt = bg[(ki // BK) % 2]
        pl.load(at, a, [group * padded + mi, ki])
        if work != pl.get_block_idx() or ki >= (2 if BN1 == 256 else 1) * BK:
            pl.load(bt, b, [group * 1024 + ni, ki])
        tr = b_transposed[(ki // BK) % 2]
        _mx_matmul_strip(
            acc,
            at,
            tr,
            ast,
            bst,
            al,
            als,
            bl,
            bls,
            ki,
            4096,
            ki % 2048 if BN1 == 256 else ki,
        )
    target = shared.next()
    pl.set_validshape(target, [mv // 2, nv])
    pl.move(
        target,
        acc,
        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
        phase=pl.STPhase.Final,
    )
    if work + pl.get_block_num() >= 8 * nm * nn:
        second_nn = (5120 + BN - 1) // BN
        core = pl.get_block_idx()
        if core < nm * second_nn:
            first = wbg[0]
            col = core % second_nn * BN
            pl.set_validshape(first, [pl.min(BN, 5120 - col), BK])
            pl.load(first, wb, [col, 0])


def _prefetch_gemm1_weights(b, sb, bg, bsg, t):
    """Load GEMM1's first B tile (and its scales) into L1 before the task loop."""
    work = pl.get_block_idx()
    nm = (t + BM - 1) // BM
    nn = (1024 + BN1 - 1) // BN1
    if work < 8 * nm * nn:
        if BN1 == 256:
            group = (work // nn) % 8
        else:
            group = work // (nm * nn)
        col = work % nn * BN1
        nv = pl.min(BN1, 1024 - col)
        first = bg[0]
        second = bg[1]
        scales = bsg.current()
        pl.set_validshape(first, [nv, BK])
        pl.set_validshape(second, [nv, BK])
        pl.set_validshape(scales, [128, nv])
        pl.load(first, b, [group * 1024 + col, 0])
        if BN1 == 256:
            pl.load(second, b, [group * 1024 + col, BK])
        pl.load(scales, sb, [group * 1024 + col, 0, 0], order=[1, 0])


@pl.pipeline.stage
def _quantize_gemm1_output(
    work, t, padded, yq, ys, shared, bfg, qg, esg, rg, eg, og, spg
):
    """M3: quantize one tile of GEMM1's output into the yq/ys payload GEMM2 consumes."""
    nm = (t + BM - 1) // BM
    nn = (1024 + BN1 - 1) // BN1
    if BN1 == 256:
        group = (work // nn) % 8
        mi = work // (8 * nn) * BM
    else:
        group = work // (nm * nn)
        mi = work // nn % nm * BM
    ni = work % nn * BN1
    nv = pl.min(BN1, 1024 - ni)
    half = pl.min(BM, padded - mi) // 2
    row = mi + pl.get_subblock_idx() * half
    tile = shared.next()
    bf = bfg.current()
    q = qg.current()
    es = esg.current()
    reciprocal = rg.current()
    even = eg.current()
    odd = og.current()
    scales = spg.current()
    if BN1 == 160 or BM <= 128:
        _cast_accumulator_to_bf16(tile, bf, half * BN1, 0)
        _compute_mx_block_scales(bf, es, reciprocal, half * BN1)
        _quantize_to_fp8_payload(bf, q, reciprocal, even, odd, half * BN1)
        _pack_mx_scale_bytes(es, scales, half, BN1 // 32)
        pl.set_validshape(q, [half, nv])
        pl.set_validshape(scales, [half, nv // 32])
        pl.store(yq, q, [row, group * 1024 + ni])
        pl.store(ys, scales, [row, group * 32 + ni // 32])
    else:
        for start in pl.range(0, half, Q_ROWS):
            rows = pl.min(Q_ROWS, half - start)
            _cast_accumulator_to_bf16(tile, bf, rows * BN1, start * BN1)
            _compute_mx_block_scales(bf, es, reciprocal, rows * BN1)
            _quantize_to_fp8_payload(bf, q, reciprocal, even, odd, rows * BN1)
            _pack_mx_scale_bytes(es, scales, rows, BN1 // 32)
            pl.set_validshape(q, [rows, nv])
            pl.set_validshape(scales, [rows, nv // 32])
            pl.store(yq, q, [row + start, group * 1024 + ni])
            pl.store(ys, scales, [row + start, group * 32 + ni // 32])


def _rope_quantize_slabbed(o_ptr, cos_ptr, sin_ptr, aq_ptr, asc_ptr, t, padded):
    """M1: inverse RoPE, regroup and MX quantization of the activation (slab-tiled variant)."""
    r_rows = R_ROWS if BM == 256 else 4
    source = pl.make_tensor(o_ptr, [t, 32768], dtype=pl.DT_BF16)
    cosine = pl.make_tensor(cos_ptr, [t, 64], dtype=pl.DT_FP32)
    sine = pl.make_tensor(sin_ptr, [t, 64], dtype=pl.DT_FP32)
    aq = pl.make_tensor(aq_ptr, [8 * padded, 4096], dtype=pl.DT_UINT8)
    asc = pl.make_tensor(asc_ptr, [8 * padded, 128], dtype=pl.DT_UINT8)
    # Rope storage uses 13120*r_rows+2560 bytes. Its stores finish before
    # the second slab notification permits Cube to write the shared UB view.
    cos_addr = 8192 * r_rows
    sin_addr = cos_addr + 256 * r_rows
    q_addr = sin_addr + 256 * r_rows
    es_addr = q_addr + (r_rows + 1) * 2048
    recip_addr = es_addr + r_rows * 64
    even_addr = recip_addr + r_rows * 256
    odd_addr = even_addr + r_rows * 1024 + 256
    xg = pl.make_tile_group(
        type=pl.TileType(
            shape=[r_rows, 2048], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[0, r_rows * 4096],
        mutex_ids=[0, 1],
    )
    cg = pl.make_tile_group(
        type=pl.TileType(
            shape=[r_rows, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[cos_addr],
        mutex_ids=[2],
    )
    sg = pl.make_tile_group(
        type=pl.TileType(
            shape=[r_rows, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[sin_addr],
        mutex_ids=[3],
    )
    qg = pl.make_tile_group(
        type=pl.TileType(
            shape=[r_rows + 1, 2048],
            dtype=pl.DT_UINT8,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[q_addr],
        mutex_ids=[4],
    )
    esg = pl.make_tile_group(
        type=pl.TileType(
            shape=[r_rows, 64], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[es_addr],
        mutex_ids=[5],
    )
    rg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, r_rows * 128],
            dtype=pl.DT_UINT16,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[recip_addr],
        mutex_ids=[6],
    )
    eg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, r_rows * 1024 + 256], dtype=FP8, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[even_addr],
        mutex_ids=[7],
    )
    og = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, r_rows * 1024 + 256], dtype=FP8, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[odd_addr],
        mutex_ids=[8],
    )
    with pl.section_vector():
        worker = pl.get_block_idx()
        group = worker % 8
        lane = worker // 8
        lanes = pl.get_block_num() * pl.get_subblock_num() // 8
        batch_rows = r_rows
        if BM == 256:
            rows_per_lane = (padded + lanes - 1) // lanes
            batches = (rows_per_lane + r_rows - 1) // r_rows
            batch_rows = (rows_per_lane + batches - 1) // batches
        for slab in pl.range(2):
            for row in pl.range(lane * batch_rows, padded, lanes * batch_rows):
                rows = pl.min(batch_rows, padded - row)
                x = xg.next()
                q = qg.current()
                es = esg.current()
                reciprocal = rg.current()
                even = eg.current()
                odd = og.current()
                if row < t:
                    valid = pl.min(rows, t - row)
                    c = cg.current()
                    s = sg.current()
                    pl.set_validshape(x, [valid, 2048])
                    pl.set_validshape(c, [valid, 64])
                    pl.set_validshape(s, [valid, 64])
                    pl.load(x, source, [row, group * 4096 + slab * 2048])
                    pl.load(c, cosine, [row, 0])
                    pl.load(s, sine, [row, 0])
                    _apply_inverse_rope_slab(x, c, s, valid)
                    _compute_mx_block_scales(x, es, reciprocal, valid * 2048)
                    _quantize_to_fp8_payload(x, q, reciprocal, even, odd, valid * 2048)
                    if valid < rows:
                        if BM == 256:
                            _zero_fill_quantized_batch_tail(
                                q, es, valid * 2048, (rows - valid) * 2048
                            )
                        else:
                            _zero_fill_quantized_tail(
                                q, es, valid * 2048, (rows - valid) * 2048
                            )
                else:
                    if BM == 256:
                        _zero_fill_quantized_batch(q, es, rows * 2048)
                    else:
                        _zero_fill_quantized_tile(q, es, rows * 2048)
                pl.set_validshape(q, [rows, 2048])
                pl.set_validshape(es, [rows, 64])
                pl.store(aq, q, [group * padded + row, slab * 2048])
                pl.store(asc, es, [group * padded + row, slab * 64])
            # Each immutable slab is globally complete before both AIV subcores
            # publish it to their AIC. MTE3 orders publication after all stores.
            if slab == 0:
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=5,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=5,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=3)
            else:
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=6,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=6,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=4)


def _rope_quantize_whole(o_ptr, cos_ptr, sin_ptr, aq_ptr, asc_ptr, t, padded):
    """M1: inverse RoPE, regroup and MX quantization of the activation (whole-range variant)."""
    source = pl.make_tensor(o_ptr, [t * 8, 4096], dtype=pl.DT_BF16)
    cosine = pl.make_tensor(cos_ptr, [t, 64], dtype=pl.DT_FP32)
    sine = pl.make_tensor(sin_ptr, [t, 64], dtype=pl.DT_FP32)
    aq = pl.make_tensor(aq_ptr, [8 * padded, 4096], dtype=pl.DT_UINT8)
    asc = pl.make_tensor(asc_ptr, [8 * padded, 128], dtype=pl.DT_UINT8)
    # The activation workspace occupies Vec bytes [0, 26752).
    xg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 4096], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[0, 8192],
        mutex_ids=[0, 1],
    )
    cg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[16384],
        mutex_ids=[2],
    )
    sg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[16640],
        mutex_ids=[3],
    )
    qg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 4608], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[16896],
        mutex_ids=[4],
    )
    esg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 128], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[21504],
        mutex_ids=[5],
    )
    rg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 256], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[21632],
        mutex_ids=[6],
    )
    eg = pl.make_tile_group(
        type=pl.TileType(shape=[1, 2304], dtype=FP8, target_memory=pl.MemorySpace.Vec),
        addrs=[22144],
        mutex_ids=[7],
    )
    og = pl.make_tile_group(
        type=pl.TileType(shape=[1, 2304], dtype=FP8, target_memory=pl.MemorySpace.Vec),
        addrs=[24448],
        mutex_ids=[8],
    )
    with pl.section_vector():
        for r in pl.range(
            pl.get_block_idx(), padded * 8, pl.get_block_num() * pl.get_subblock_num()
        ):
            x = xg.next()
            q = qg.current()
            es = esg.current()
            reciprocal = rg.current()
            even = eg.current()
            odd = og.current()
            if r // 8 < t:
                c = cg.current()
                s = sg.current()
                pl.load(x, source, [r, 0])
                pl.load(c, cosine, [r // 8, 0])
                pl.load(s, sine, [r // 8, 0])
                _apply_inverse_rope(x, c, s)
                _compute_mx_block_scales(x, es, reciprocal, 4096)
                _quantize_to_fp8_payload(x, q, reciprocal, even, odd, 4096)
            else:
                _zero_fill_quantized_tile(q, es, 4096)
            pl.set_validshape(q, [1, 4096])
            dest = (r % 8) * padded + r // 8
            pl.store(aq, q, [dest, 0])
            pl.store(asc, es, [dest, 0])


def _gemm2_asym_pair(yq_ptr, ys_ptr, wb_ptr, sb_ptr, out_ptr, t, padded, ag0):
    """Host dispatch guarantees 257<=T<=384, BM256/BK256/K0128/N160.

    Original M0=256 and M1=T-256 geometry, K order and N160 ownership stay
    unchanged. B and ScaleB remain read-only in each Right slot for both M
    consumers before overwrite. L1=417792B with two L1 slots per M panel;
    Left=49152B, Right=40960B, ScaleLeft=3072B, ScaleRight=2560B.
    Disjoint Acc0[256,160] and Acc1[128,160] total 245760B. Each Final
    matmul is followed by its own Final store. G1 is unchanged.
    """
    k = 8192
    n = 5120
    a = pl.make_tensor(yq_ptr, [padded, k], dtype=FP8)
    sa = pl.make_tensor(ys_ptr, [padded, 128, 2], dtype=E8)
    b = pl.make_tensor(wb_ptr, [n, k], dtype=FP8)
    sb = pl.make_tensor(sb_ptr, [n, 128, 2], dtype=E8)
    output = pl.make_tensor(out_ptr, [padded, n], dtype=pl.DT_BF16)
    # G1's AG0/1 (IDs0/1, shapes256x256 at0/65536) are fully
    # drained by its Cube bar_all. Reuse that exact group for M0's two
    # L1 slots; this keeps all 32 mutex IDs within range and preserves
    # the original per-M double-buffer distance.
    a1g = pl.make_tile_group(
        type=pl.TileType(shape=[128, 256], dtype=FP8, target_memory=pl.MemorySpace.Mat),
        addrs=[172032, 204800],
        mutex_ids=[15, 16],
    )
    # Keep the original first-B prefetch at address 2*256*256.
    bg = pl.make_tile_group(
        type=pl.TileType(shape=[160, 256], dtype=FP8, target_memory=pl.MemorySpace.Mat),
        addrs=[131072, 237568],
        mutex_ids=[17, 18],
    )
    s0g = pl.make_tile_group(
        type=pl.TileType(
            shape=[256, 256], dtype=E8, target_memory=pl.MemorySpace.Mat, layout=pl.ZZ
        ),
        addrs=[278528],
        mutex_ids=[19],
    )
    s1g = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 256], dtype=E8, target_memory=pl.MemorySpace.Mat, layout=pl.ZZ
        ),
        addrs=[344064],
        mutex_ids=[20],
    )
    bsg = pl.make_tile_group(
        type=pl.TileType(
            shape=[256, 160], dtype=E8, target_memory=pl.MemorySpace.Mat, layout=pl.NN
        ),
        addrs=[376832],
        mutex_ids=[21],
    )
    l0g = pl.make_tile_group(
        type=pl.TileType(
            shape=[256, 128], dtype=FP8, target_memory=pl.MemorySpace.Left, compact=1
        ),
        addrs=[0],
        mutex_ids=[22],
    )
    l1g = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 128], dtype=FP8, target_memory=pl.MemorySpace.Left, compact=1
        ),
        addrs=[32768],
        mutex_ids=[23],
    )
    bl = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 160], dtype=FP8, target_memory=pl.MemorySpace.Right, compact=1
        ),
        addrs=[0, 20480],
        mutex_ids=[24, 25],
    )
    sl0g = pl.make_tile_group(
        type=pl.TileType(
            shape=[256, 4], dtype=E8, target_memory=pl.MemorySpace.ScaleLeft, compact=1
        ),
        addrs=[0],
        mutex_ids=[26],
    )
    sl1g = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 4], dtype=E8, target_memory=pl.MemorySpace.ScaleLeft, compact=1
        ),
        addrs=[2048],
        mutex_ids=[27],
    )
    bls = pl.make_tile_group(
        type=pl.TileType(
            shape=[4, 160], dtype=E8, target_memory=pl.MemorySpace.ScaleRight, compact=1
        ),
        addrs=[0, 1280],
        mutex_ids=[28, 29],
    )
    c0g = pl.make_tile_group(
        type=pl.TileType(
            shape=[256, 160],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=[0],
        mutex_ids=[30],
    )
    c1g = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 160],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=[163840],
        mutex_ids=[31],
    )
    transposed = pl.reinterpret(bg, shape=[256, 160], layout=pl.ZN)
    with pl.section_cube():
        core = pl.get_block_idx()
        ahead = bg[1]
        scales = bsg.current()
        col = core * 160
        pl.load(ahead, b, [col, 256])
        pl.load(scales, sb, [col, 0, 0], order=[1, 0])
        # All AICs consume this once. YQ/YS reads stay after the wait.
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=7)
        for work in pl.range(pl.get_block_idx(), 32, pl.get_block_num()):
            ni = work * 160
            tail = t - 256
            pl.set_validshape(ag0, [256, 256])
            ast0 = s0g.current()
            ast1 = s1g.current()
            bst = bsg.current()
            left0 = l0g.current()
            left1 = l1g.current()
            ls0 = sl0g.current()
            ls1 = sl1g.current()
            acc0 = c0g.current()
            acc1 = c1g.current()
            pl.set_validshape(a1g, [tail, 256])
            pl.set_validshape(ast1, [tail, 256])
            pl.set_validshape(left1, [tail, 128])
            pl.set_validshape(ls1, [tail, 4])
            pl.set_validshape(acc1, [tail, 160])
            pl.load(ast0, sa, [0, 0, 0])
            pl.load(ast1, sa, [256, 0, 0])
            if work != pl.get_block_idx():
                pl.load(bst, sb, [ni, 0, 0], order=[1, 0])
            for ki in pl.range(0, 8192, 256):
                at0 = ag0[(ki // 256) % 2]
                at1 = a1g[(ki // 256) % 2]
                bt = bg[(ki // 256) % 2]
                tr = transposed[(ki // 256) % 2]
                pl.load(at0, a, [0, ki])
                pl.load(at1, a, [256, ki])
                if work != pl.get_block_idx() or ki >= 512:
                    pl.load(bt, b, [ni, ki])
                _gemm2_asym_step(
                    acc0,
                    acc1,
                    at0,
                    at1,
                    tr,
                    ast0,
                    ast1,
                    bst,
                    left0,
                    left1,
                    ls0,
                    ls1,
                    bl,
                    bls,
                    ki,
                    0,
                    output,
                    ni,
                )
                _gemm2_asym_step(
                    acc0,
                    acc1,
                    at0,
                    at1,
                    tr,
                    ast0,
                    ast1,
                    bst,
                    left0,
                    left1,
                    ls0,
                    ls1,
                    bl,
                    bls,
                    ki,
                    128,
                    output,
                    ni,
                )


def _gemm2_asym_step(
    acc0,
    acc1,
    at0,
    at1,
    tr,
    ast0,
    ast1,
    bst,
    left0,
    left1,
    ls0,
    ls1,
    bl,
    bls,
    ki,
    inner,
    output,
    ni,
):
    right = bl[inner // 128]
    right_scale = bls[inner // 128]
    pl.move(right, tr, [inner, 0])
    pl.move(right_scale, bst, [(ki + inner) // 32, 0])
    pl.move(left0, at0, [0, inner])
    pl.move(ls0, ast0, [0, (ki + inner) // 32])
    if ki == 0 and inner == 0:
        pl.matmul_mx(acc0, left0, right, ls0, right_scale, phase=pl.AccPhase.Partial)
    elif ki + 256 == 8192 and inner == 128:
        pl.matmul_mx_acc(
            acc0, acc0, left0, right, ls0, right_scale, phase=pl.AccPhase.Final
        )
        pl.store(output, acc0, [0, ni], phase=pl.STPhase.Final)
    else:
        pl.matmul_mx_acc(
            acc0, acc0, left0, right, ls0, right_scale, phase=pl.AccPhase.Partial
        )
    pl.move(left1, at1, [0, inner])
    pl.move(ls1, ast1, [0, (ki + inner) // 32])
    if ki == 0 and inner == 0:
        pl.matmul_mx(acc1, left1, right, ls1, right_scale, phase=pl.AccPhase.Partial)
    elif ki + 256 == 8192 and inner == 128:
        pl.matmul_mx_acc(
            acc1, acc1, left1, right, ls1, right_scale, phase=pl.AccPhase.Final
        )
        pl.store(output, acc1, [256, ni], phase=pl.STPhase.Final)
    else:
        pl.matmul_mx_acc(
            acc1, acc1, left1, right, ls1, right_scale, phase=pl.AccPhase.Partial
        )


def _gemm2_tile(yq_ptr, ys_ptr, wb_ptr, sb_ptr, out_ptr, t, padded):
    """GEMM2: out[padded, 5120] = yq[padded, 8192] @ wob[5120, 8192]^T in bf16."""
    k = 8192
    n = 5120
    a = pl.make_tensor(yq_ptr, [padded, k], dtype=FP8)
    sa = pl.make_tensor(ys_ptr, [padded, k // 64, 2], dtype=E8)
    b = pl.make_tensor(wb_ptr, [n, k], dtype=FP8)
    sb = pl.make_tensor(sb_ptr, [n, k // 64, 2], dtype=E8)
    output = pl.make_tensor(out_ptr, [padded, n], dtype=pl.DT_BF16)
    # The Cube pipe barrier drains the first projection before address reuse.
    # Continue its Cube mutex numbering; L0 scale planes follow data address >> 4.
    l0_slots = 256 // K0
    cube_id = 6 + 4 * l0_slots + min(2, 65536 // (BM * BN1))
    acc_slots = min(2, 65536 // (BM * BN))
    ag = pl.make_tile_group(
        type=pl.TileType(shape=[BM, BK], dtype=FP8, target_memory=pl.MemorySpace.Mat),
        addrs=[0, BM * BK],
        mutex_ids=[cube_id, cube_id + 1],
    )
    bg = pl.make_tile_group(
        type=pl.TileType(shape=[BN, BK], dtype=FP8, target_memory=pl.MemorySpace.Mat),
        addrs=[2 * BM * BK, (2 * BM + BN) * BK],
        mutex_ids=[cube_id + 2, cube_id + 3],
    )
    asg = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, k // 32],
            dtype=E8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZZ,
        ),
        addrs=[2 * (BM + BN) * BK],
        mutex_ids=[cube_id + 4],
    )
    bsg = pl.make_tile_group(
        type=pl.TileType(
            shape=[k // 32, BN],
            dtype=E8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NN,
        ),
        addrs=[2 * (BM + BN) * BK + BM * (k // 32)],
        mutex_ids=[cube_id + 5],
    )
    al = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, K0], dtype=FP8, target_memory=pl.MemorySpace.Left, compact=1
        ),
        addrs=[0, BM * K0][:l0_slots],
        mutex_ids=[cube_id + 6, cube_id + 7][:l0_slots],
    )
    bl = pl.make_tile_group(
        type=pl.TileType(
            shape=[K0, BN], dtype=FP8, target_memory=pl.MemorySpace.Right, compact=1
        ),
        addrs=[0, BN * K0][:l0_slots],
        mutex_ids=[cube_id + 6 + l0_slots, cube_id + 7 + l0_slots][:l0_slots],
    )
    als = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, K0 // 32],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleLeft,
            compact=1,
        ),
        addrs=[0, BM * K0 // 16][:l0_slots],
        mutex_ids=[cube_id + 6 + 2 * l0_slots, cube_id + 7 + 2 * l0_slots][:l0_slots],
    )
    bls = pl.make_tile_group(
        type=pl.TileType(
            shape=[K0 // 32, BN],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleRight,
            compact=1,
        ),
        addrs=[0, BN * K0 // 16][:l0_slots],
        mutex_ids=[cube_id + 6 + 3 * l0_slots, cube_id + 7 + 3 * l0_slots][:l0_slots],
    )
    accg = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, BN],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=[0, 4 * BM * BN][:acc_slots],
        mutex_ids=[cube_id + 6 + 4 * l0_slots, cube_id + 7 + 4 * l0_slots][:acc_slots],
    )
    b_transposed = pl.reinterpret(bg, shape=[BK, BN], layout=pl.ZN)
    with pl.section_cube():
        nm = (t + BM - 1) // BM
        nn = (n + BN - 1) // BN
        for work in pl.range(pl.get_block_idx(), nm * nn, pl.get_block_num()):
            mi = work // nn % nm * BM
            ni = work % nn * BN
            mv = pl.min(BM, t - mi)
            nv = pl.min(BN, n - ni)
            acc = accg.next()
            ast = asg.current()
            bst = bsg.current()
            pl.set_validshape(acc, [mv, nv])
            pl.set_validshape(ast, [mv, k // 32])
            pl.set_validshape(bst, [k // 32, nv])
            pl.set_validshape(ag, [mv, BK])
            pl.set_validshape(bg, [nv, BK])
            pl.set_validshape(b_transposed, [BK, nv])
            pl.set_validshape(al, [mv, K0])
            pl.set_validshape(als, [mv, K0 // 32])
            pl.set_validshape(bl, [K0, nv])
            pl.set_validshape(bls, [K0 // 32, nv])
            pl.load(ast, sa, [mi, 0, 0])
            pl.load(bst, sb, [ni, 0, 0], order=[1, 0])
            for ki in pl.range(0, k, BK):
                at = ag.next()
                bt = bg[(ki // BK) % 2]
                pl.load(at, a, [mi, ki])
                if work != pl.get_block_idx() or ki != 0:
                    pl.load(bt, b, [ni, ki])
                bt_transposed = b_transposed[(ki // BK) % 2]
                _mx_matmul_strip(
                    acc, at, bt_transposed, ast, bst, al, als, bl, bls, ki, k, ki
                )
            pl.store(output, acc, [mi, ni], phase=pl.STPhase.Final)


def _mx_matmul_strip(acc, at, bt_transposed, ast, bst, al, als, bl, bls, ki, k, sa_ki):
    """Issue the block-scaled (MX) fp8 matmul steps covering one L1 K strip."""
    _mx_matmul_step(acc, at, bt_transposed, ast, bst, al, als, bl, bls, ki, k, sa_ki, 0)
    if BK >= 2 * K0:
        _mx_matmul_step(
            acc, at, bt_transposed, ast, bst, al, als, bl, bls, ki, k, sa_ki, K0
        )
    if BK >= 4 * K0:
        _mx_matmul_step(
            acc, at, bt_transposed, ast, bst, al, als, bl, bls, ki, k, sa_ki, 2 * K0
        )
        _mx_matmul_step(
            acc, at, bt_transposed, ast, bst, al, als, bl, bls, ki, k, sa_ki, 3 * K0
        )
    if BK >= 8 * K0:
        _mx_matmul_step(
            acc, at, bt_transposed, ast, bst, al, als, bl, bls, ki, k, sa_ki, 4 * K0
        )
        _mx_matmul_step(
            acc, at, bt_transposed, ast, bst, al, als, bl, bls, ki, k, sa_ki, 5 * K0
        )
        _mx_matmul_step(
            acc, at, bt_transposed, ast, bst, al, als, bl, bls, ki, k, sa_ki, 6 * K0
        )
        _mx_matmul_step(
            acc, at, bt_transposed, ast, bst, al, als, bl, bls, ki, k, sa_ki, 7 * K0
        )


def _mx_matmul_step(acc, at, bt, ast, bst, al, als, bl, bls, ki, k, sa_ki, inner):
    """One L0 sub-step of the MX matmul: load the A/B strips with their scales, then mac."""
    left = al[(inner // K0) % (256 // K0)]
    left_scale = als[(inner // K0) % (256 // K0)]
    right = bl[(inner // K0) % (256 // K0)]
    right_scale = bls[(inner // K0) % (256 // K0)]
    # Unit flags overlap M and FIX across output tiles.
    pl.move(left, at, [0, inner])
    pl.move(left_scale, ast, [0, (sa_ki + inner) // 32])
    pl.move(right, bt, [inner, 0])
    pl.move(right_scale, bst, [(ki + inner) // 32, 0])
    if inner == 0 and ki == 0:
        pl.matmul_mx(
            acc, left, right, left_scale, right_scale, phase=pl.AccPhase.Partial
        )
    elif inner + K0 == BK and ki + BK == k:
        pl.matmul_mx_acc(
            acc, acc, left, right, left_scale, right_scale, phase=pl.AccPhase.Final
        )
    else:
        pl.matmul_mx_acc(
            acc, acc, left, right, left_scale, right_scale, phase=pl.AccPhase.Partial
        )


def _small_gemm1_nstep(acc, left, left_s, right, right_s, tr, bst, ki, col):
    pl.move(right, tr, [0, col])
    pl.move(right_s, bst, [ki // 32, col])
    if ki == 0:
        pl.matmul_mx(acc, left, right, left_s, right_s, phase=pl.AccPhase.Partial)
    elif ki + 512 == 4096:
        pl.matmul_mx_acc(
            acc, acc, left, right, left_s, right_s, phase=pl.AccPhase.Final
        )
    else:
        pl.matmul_mx_acc(
            acc, acc, left, right, left_s, right_s, phase=pl.AccPhase.Partial
        )


def _small_gemm2_nstep(
    acc, left, left_s, right, right_s, bt, bst, output, mi, ni, ki, ck, col
):
    # Left and ScaleLeft are read-only across the whole N split at this K.
    pl.move(right, bt, [0, col])
    pl.move(right_s, bst, [ki // 32, col])
    if ki == 0:
        pl.matmul_mx(acc, left, right, left_s, right_s, phase=pl.AccPhase.Partial)
    elif ki + ck == 8192:
        pl.matmul_mx_acc(
            acc, acc, left, right, left_s, right_s, phase=pl.AccPhase.Final
        )
        pl.store(output, acc, [mi, ni + col], phase=pl.STPhase.Final)
    else:
        pl.matmul_mx_acc(
            acc, acc, left, right, left_s, right_s, phase=pl.AccPhase.Partial
        )


def _small_gemm2(yq_ptr, ys_ptr, wb_ptr, sb_ptr, out_ptr, t, padded):
    ck = 512
    n0 = 64
    nmain = 2
    cube_id = 6 + 4 * (256 // K0) + min(2, 65536 // (BM * BN1))
    a = pl.make_tensor(yq_ptr, [padded, 8192], dtype=FP8)
    sa = pl.make_tensor(ys_ptr, [padded, 128, 2], dtype=E8)
    b = pl.make_tensor(wb_ptr, [5120, 8192], dtype=FP8)
    sb = pl.make_tensor(sb_ptr, [5120, 128, 2], dtype=E8)
    output = pl.make_tensor(out_ptr, [padded, 5120], dtype=pl.DT_BF16)
    ag = pl.make_tile_group(
        type=pl.TileType(shape=[BM, ck], dtype=FP8, target_memory=pl.MemorySpace.Mat),
        addrs=[0, BM * ck],
        mutex_ids=[cube_id, cube_id + 1],
    )
    bg = pl.make_tile_group(
        type=pl.TileType(shape=[160, ck], dtype=FP8, target_memory=pl.MemorySpace.Mat),
        addrs=[2 * BM * ck, (2 * BM + 160) * ck],
        mutex_ids=[cube_id + 2, cube_id + 3],
    )
    asg = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, 256], dtype=E8, target_memory=pl.MemorySpace.Mat, layout=pl.ZZ
        ),
        addrs=[2 * (BM + 160) * ck],
        mutex_ids=[cube_id + 4],
    )
    bsg = pl.make_tile_group(
        type=pl.TileType(
            shape=[256, 160], dtype=E8, target_memory=pl.MemorySpace.Mat, layout=pl.NN
        ),
        addrs=[2 * (BM + 160) * ck + BM * 256],
        mutex_ids=[cube_id + 5],
    )
    al = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, ck], dtype=FP8, target_memory=pl.MemorySpace.Left, compact=1
        ),
        addrs=[0, 32768],
        mutex_ids=[cube_id + 6, cube_id + 7],
    )
    bl = pl.make_tile_group(
        type=pl.TileType(
            shape=[ck, n0], dtype=FP8, target_memory=pl.MemorySpace.Right, compact=1
        ),
        addrs=[0, 32768],
        mutex_ids=[cube_id + 8, cube_id + 9],
    )
    als = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, ck // 32],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleLeft,
            compact=1,
        ),
        addrs=[0, 2048],
        mutex_ids=[cube_id + 10, cube_id + 11],
    )
    bls = pl.make_tile_group(
        type=pl.TileType(
            shape=[ck // 32, n0],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleRight,
            compact=1,
        ),
        addrs=[0, 2048],
        mutex_ids=[cube_id + 12, cube_id + 13],
    )
    accg = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, n0],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=[0, 4 * BM * n0][:nmain],
        mutex_ids=[cube_id + 14, cube_id + 15][:nmain],
    )
    tailg = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, 32],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=[4 * BM * n0 * nmain],
        mutex_ids=[cube_id + 16],
    )
    btr = pl.reinterpret(bg, shape=[ck, 160], layout=pl.ZN)
    pl.set_validshape(btr, [ck, 160])
    with pl.section_cube():
        tail_blg = pl.reinterpret(bl, shape=[ck, 32])
        tail_blsg = pl.reinterpret(bls, shape=[ck // 32, 32])
        pl.set_validshape(bl, [ck, n0])
        pl.set_validshape(bls, [ck // 32, n0])
        pl.set_validshape(tail_blg, [ck, 32])
        pl.set_validshape(tail_blsg, [ck // 32, 32])
        core = pl.get_block_idx()
        first = bg[0]
        second = bg[1]
        bst = bsg.current()
        pl.set_validshape(first, [160, ck])
        pl.load(first, b, [core * 160, 0])
        pl.set_validshape(second, [160, ck])
        pl.set_validshape(bst, [256, 160])
        pl.load(second, b, [core * 160, ck])
        pl.load(bst, sb, [core * 160, 0, 0], order=[1, 0])
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=7)
        mi = 0
        ni = core * 160
        mv = t
        ast = asg.current()
        acc0 = accg[0]
        tail = tailg.current()
        pl.set_validshape(acc0, [mv, n0])
        pl.set_validshape(tail, [mv, 32])
        if BM == 64:
            acc1 = accg[1]
            pl.set_validshape(acc1, [mv, n0])
        pl.set_validshape(ag, [mv, ck])
        pl.set_validshape(ast, [mv, 256])
        pl.set_validshape(al, [mv, ck])
        pl.set_validshape(als, [mv, ck // 32])
        pl.load(ast, sa, [mi, 0, 0])
        for ki in pl.range(0, 8192, ck):
            slot = ki // ck % 2
            at = ag[slot]
            bt = bg[slot]
            trans = btr[slot]
            pl.load(at, a, [mi, ki])
            if ki >= 2 * ck:
                pl.load(bt, b, [ni, ki])
            lslot = ki // ck % 2
            left = al[lslot]
            left_s = als[lslot]
            right0 = bl[lslot]
            right_s0 = bls[lslot]
            right1 = bl[1 - lslot]
            right_s1 = bls[1 - lslot]
            tail_right = tail_blg[lslot]
            tail_right_s = tail_blsg[lslot]
            pl.move(left, at)
            pl.move(left_s, ast, [0, ki // 32])
            _small_gemm2_nstep(
                acc0,
                left,
                left_s,
                right0,
                right_s0,
                trans,
                bst,
                output,
                mi,
                ni,
                ki,
                ck,
                0,
            )
            _small_gemm2_nstep(
                acc1,
                left,
                left_s,
                right1,
                right_s1,
                trans,
                bst,
                output,
                mi,
                ni,
                ki,
                ck,
                64,
            )
            _small_gemm2_nstep(
                tail,
                left,
                left_s,
                tail_right,
                tail_right_s,
                trans,
                bst,
                output,
                mi,
                ni,
                ki,
                ck,
                128,
            )


@pl.pipeline.stage
def _quantize_gemm1_valid_rows(
    work, t, padded, yq, ys, shared, bfg, qg, esg, rg, eg, og, spg
):
    """M3: quantize one tile of GEMM1's output into the yq/ys payload GEMM2 consumes."""
    if Algo == 3:
        group = work // 4
        mi = 0
        ni = work % 4 * 256
        nv = 256
        extent = t
    else:
        nm = (t + BM - 1) // BM
        nn = (1024 + BN1 - 1) // BN1
        if BN1 == 256:
            group = (work // nn) % 8
            mi = work // (8 * nn) * BM
        else:
            group = work // (nm * nn)
            mi = work // nn % nm * BM
        ni = work % nn * BN1
        nv = pl.min(BN1, 1024 - ni)
        extent = pl.min(BM, t - mi)
    split = (extent + 1) // 2
    half = pl.max(0, pl.min(split, extent - pl.get_subblock_idx() * split))
    row = mi + pl.get_subblock_idx() * split
    tile = shared.next()
    bf = bfg.current()
    q = qg.current()
    es = esg.current()
    reciprocal = rg.current()
    even = eg.current()
    odd = og.current()
    scales = spg.current()
    if half > 0:
        if BN1 == 160 or BM <= 128:
            _cast_accumulator_to_bf16(tile, bf, half * BN1, 0)
            _compute_mx_block_scales(bf, es, reciprocal, half * BN1)
            _quantize_to_fp8_payload(bf, q, reciprocal, even, odd, half * BN1)
            _pack_mx_scale_bytes(es, scales, half, BN1 // 32)
            pl.set_validshape(q, [half, nv])
            pl.set_validshape(scales, [half, nv // 32])
            pl.store(yq, q, [row, group * 1024 + ni])
            pl.store(ys, scales, [row, group * 32 + ni // 32])
        else:
            for start in pl.range(0, half, Q_ROWS):
                rows = pl.min(Q_ROWS, half - start)
                _cast_accumulator_to_bf16(tile, bf, rows * BN1, start * BN1)
                _compute_mx_block_scales(bf, es, reciprocal, rows * BN1)
                _quantize_to_fp8_payload(bf, q, reciprocal, even, odd, rows * BN1)
                _pack_mx_scale_bytes(es, scales, rows, BN1 // 32)
                pl.set_validshape(q, [rows, nv])
                pl.set_validshape(scales, [rows, nv // 32])
                pl.store(yq, q, [row + start, group * 1024 + ni])
                pl.store(ys, scales, [row + start, group * 32 + ni // 32])


def _gemm2_paired_tail(yq_ptr, ys_ptr, wb_ptr, sb_ptr, out_ptr, t, padded, ag):
    """GEMM2: out[padded, 5120] = yq[padded, 8192] @ wob[5120, 8192]^T in bf16."""
    k = 8192
    n = 5120
    a = pl.make_tensor(yq_ptr, [padded, k], dtype=FP8)
    sa = pl.make_tensor(ys_ptr, [padded, k // 64, 2], dtype=E8)
    b = pl.make_tensor(wb_ptr, [n, k], dtype=FP8)
    sb = pl.make_tensor(sb_ptr, [n, k // 64, 2], dtype=E8)
    output = pl.make_tensor(out_ptr, [padded, n], dtype=pl.DT_BF16)
    l0_slots = 256 // K0
    cube_id = 6 + 4 * l0_slots + min(2, 65536 // (BM * BN1))
    acc_slots = min(2, 65536 // (BM * BN))
    bslots = 3
    # G1 has drained at the root Cube barrier; its A aliases are now retired.
    a512 = pl.make_tile_group(
        type=pl.TileType(shape=[BM, 512], dtype=FP8, target_memory=pl.MemorySpace.Mat),
        addrs=[0, BM * 512],
        mutex_ids=[0, 1],
    )
    bg = pl.make_tile_group(
        type=pl.TileType(shape=[BN, 512], dtype=FP8, target_memory=pl.MemorySpace.Mat),
        addrs=[
            2 * BM * 512,
            2 * BM * 512 + BN * 512,
            2 * BM * 512 + 2 * BN * 512,
            2 * BM * 512 + 3 * BN * 512,
        ][:bslots],
        mutex_ids=[cube_id, cube_id + 1, cube_id + 2, cube_id + 3][:bslots],
    )
    asg = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, k // 32],
            dtype=E8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZZ,
        ),
        addrs=[2 * BM * 512 + bslots * BN * 512],
        mutex_ids=[cube_id + 4],
    )
    bsg = pl.make_tile_group(
        type=pl.TileType(
            shape=[k // 32, BN],
            dtype=E8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NN,
        ),
        addrs=[2 * BM * 512 + bslots * BN * 512 + BM * (k // 32)],
        mutex_ids=[cube_id + 5],
    )
    al = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, K0], dtype=FP8, target_memory=pl.MemorySpace.Left, compact=1
        ),
        addrs=[0, BM * K0][:l0_slots],
        mutex_ids=[cube_id + 6, cube_id + 7][:l0_slots],
    )
    bl = pl.make_tile_group(
        type=pl.TileType(
            shape=[K0, BN], dtype=FP8, target_memory=pl.MemorySpace.Right, compact=1
        ),
        addrs=[0, BN * K0][:l0_slots],
        mutex_ids=[cube_id + 6 + l0_slots, cube_id + 7 + l0_slots][:l0_slots],
    )
    als = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, K0 // 32],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleLeft,
            compact=1,
        ),
        addrs=[0, BM * K0 // 16][:l0_slots],
        mutex_ids=[cube_id + 6 + 2 * l0_slots, cube_id + 7 + 2 * l0_slots][:l0_slots],
    )
    bls = pl.make_tile_group(
        type=pl.TileType(
            shape=[K0 // 32, BN],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleRight,
            compact=1,
        ),
        addrs=[0, BN * K0 // 16][:l0_slots],
        mutex_ids=[cube_id + 6 + 3 * l0_slots, cube_id + 7 + 3 * l0_slots][:l0_slots],
    )
    accg = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, BN],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=[0, 4 * BM * BN][:acc_slots],
        mutex_ids=[cube_id + 6 + 4 * l0_slots, cube_id + 7 + 4 * l0_slots][:acc_slots],
    )
    b_transposed = pl.reinterpret(bg, shape=[512, BN], layout=pl.ZN)
    with pl.section_cube():
        core = pl.get_block_idx()
        first_col = core * BN
        b0 = bg[0]
        b1 = bg[1]
        pre_scales = bsg.current()
        pl.load(b0, b, [first_col, 0])
        pl.load(b1, b, [first_col, 512])
        pl.load(pre_scales, sb, [first_col, 0, 0], order=[1, 0])
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=7)
        mi = 0
        ni = core * BN
        mv = t
        nv = BN
        acc = accg.next()
        ast = asg.current()
        bst = bsg.current()
        pl.set_validshape(acc, [mv, nv])
        pl.set_validshape(ast, [mv, k // 32])
        pl.set_validshape(bst, [k // 32, nv])
        pl.set_validshape(a512, [mv, 512])
        pl.set_validshape(bg, [nv, 512])
        pl.set_validshape(b_transposed, [512, nv])
        pl.set_validshape(al, [mv, K0])
        pl.set_validshape(als, [mv, K0 // 32])
        pl.set_validshape(bl, [K0, nv])
        pl.set_validshape(bls, [K0 // 32, nv])
        pl.load(ast, sa, [mi, 0, 0])
        first_a = a512[0]
        second_a = a512[1]
        pl.load(first_a, a, [mi, 0])
        pl.load(second_a, a, [mi, 512])
        for window in pl.range(0, k, 512):
            if window + 1024 < k:
                future = bg[(window // 512 + 2) % bslots]
                pl.load(future, b, [ni, window + 1024])
            tr = b_transposed[window // 512 % bslots]
            at = a512[window // 512 % 2]
            _g2_a512_step(acc, at, tr, ast, bst, al, als, bl, bls, window, 0)
            _g2_a512_step(acc, at, tr, ast, bst, al, als, bl, bls, window, 128)
            _g2_a512_step(acc, at, tr, ast, bst, al, als, bl, bls, window, 256)
            _g2_a512_step(acc, at, tr, ast, bst, al, als, bl, bls, window, 384)
            if window + 1024 < k:
                pl.load(at, a, [mi, window + 1024])
        pl.store(output, acc, [mi, ni], phase=pl.STPhase.Final)


def _g2_a512_step(acc, at, bt, ast, bst, al, als, bl, bls, window, inner):
    left = al[(inner // 128) % 2]
    left_scale = als[(inner // 128) % 2]
    right = bl[(inner // 128) % 2]
    right_scale = bls[(inner // 128) % 2]
    pl.move(left, at, [0, inner])
    pl.move(left_scale, ast, [0, (window + inner) // 32])
    pl.move(right, bt, [inner, 0])
    pl.move(right_scale, bst, [(window + inner) // 32, 0])
    if window + inner == 0:
        pl.matmul_mx(
            acc, left, right, left_scale, right_scale, phase=pl.AccPhase.Partial
        )
    elif window + inner + 128 == 8192:
        pl.matmul_mx_acc(
            acc, acc, left, right, left_scale, right_scale, phase=pl.AccPhase.Final
        )
    else:
        pl.matmul_mx_acc(
            acc, acc, left, right, left_scale, right_scale, phase=pl.AccPhase.Partial
        )


def _mx_matmul_offset_strip(acc, at, tr, ast, bst, al, als, bl, bls, ki, k, b_offset):
    _mx_matmul_offset_step(
        acc, at, tr, ast, bst, al, als, bl, bls, ki, k, ki, 0, b_offset
    )
    _mx_matmul_offset_step(
        acc, at, tr, ast, bst, al, als, bl, bls, ki, k, ki, K0, b_offset
    )


def _mx_matmul_offset_step(
    acc, at, bt, ast, bst, al, als, bl, bls, ki, k, sa_ki, inner, b_offset
):
    """One L0 sub-step of the MX matmul: load the A/B strips with their scales, then mac."""
    left = al[(inner // K0) % (256 // K0)]
    left_scale = als[(inner // K0) % (256 // K0)]
    right = bl[(inner // K0) % (256 // K0)]
    right_scale = bls[(inner // K0) % (256 // K0)]
    # Unit flags overlap M and FIX across output tiles.
    pl.move(left, at, [0, inner])
    pl.move(left_scale, ast, [0, (sa_ki + inner) // 32])
    pl.move(right, bt, [b_offset + inner, 0])
    pl.move(right_scale, bst, [(ki + inner) // 32, 0])
    if inner == 0 and ki == 0:
        pl.matmul_mx(
            acc, left, right, left_scale, right_scale, phase=pl.AccPhase.Partial
        )
    elif inner + K0 == BK and ki + BK == k:
        pl.matmul_mx_acc(
            acc, acc, left, right, left_scale, right_scale, phase=pl.AccPhase.Final
        )
    else:
        pl.matmul_mx_acc(
            acc, acc, left, right, left_scale, right_scale, phase=pl.AccPhase.Partial
        )


@pl.pipeline.stage
def _gemm1_prefetch(
    work,
    t,
    padded,
    a,
    sa,
    b,
    sb,
    wb,
    wbg,
    ag,
    bg,
    b_transposed,
    asg,
    bsg,
    al,
    als,
    bl,
    bls,
    accg,
    shared,
):
    """GEMM1 tile (group, m, n): acc = a[group, m] @ woa[group, n]^T, drained to the C->V ring."""
    nm = (t + BM - 1) // BM
    nn = (1024 + BN1 - 1) // BN1
    if BN1 == 256:
        group = (work // nn) % 8
        mi = work // (8 * nn) * BM
    else:
        group = work // (nm * nn)
        mi = work // nn % nm * BM
    ni = work % nn * BN1
    mv = pl.min(BM, t - mi)
    nv = pl.min(BN1, 1024 - ni)
    acc = accg.next()
    ast = asg.current()
    bst = bsg.current()
    pl.set_validshape(acc, [mv, nv])
    pl.set_validshape(ast, [mv, 64 if BN1 == 256 else 128])
    pl.set_validshape(ag, [mv, BK])
    pl.set_validshape(al, [mv, K0])
    pl.set_validshape(als, [mv, K0 // 32])
    pl.set_validshape(bg, [nv, BK])
    pl.set_validshape(b_transposed, [BK, nv])
    pl.set_validshape(bst, [128, nv])
    pl.set_validshape(bl, [K0, nv])
    pl.set_validshape(bls, [K0 // 32, nv])
    if BN1 != 256:
        pl.load(ast, sa, [group * padded + mi, 0, 0])
    # Full-K BS remains immutable at the same group/N across M.
    for ki in pl.range(0, 4096, BK):
        if BN1 == 256 and ki % 2048 == 0:
            # All 32 AICs have a first tile; later tiles reuse immutable slabs.
            if work == pl.get_block_idx():
                if ki == 0:
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=3)
                else:
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=4)
            pl.load(ast, sa, [group * padded + mi, ki // 64, 0])
        at = ag.next()
        bt = bg[(ki // BK) % 2]
        pl.load(at, a, [group * padded + mi, ki])
        if work != pl.get_block_idx() or ki >= (2 if BN1 == 256 else 1) * BK:
            pl.load(bt, b, [group * 1024 + ni, ki])
        tr = b_transposed[(ki // BK) % 2]
        _mx_matmul_strip(
            acc,
            at,
            tr,
            ast,
            bst,
            al,
            als,
            bl,
            bls,
            ki,
            4096,
            ki % 2048 if BN1 == 256 else ki,
        )
    target = shared.next()
    pl.set_validshape(target, [(mv + 1) // 2, nv])
    pl.move(
        target,
        acc,
        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
        phase=pl.STPhase.Final,
    )
    if work + pl.get_block_num() >= 8 * nm * nn:
        second_nn = (5120 + BN - 1) // BN
        core = pl.get_block_idx()
        if core < nm * second_nn:
            first = wbg[0]
            col = core % second_nn * BN
            pl.set_validshape(first, [pl.min(BN, 5120 - col), BK])
            pl.load(first, wb, [col, 0])


def _gemm2_prefetch4(yq_ptr, ys_ptr, wb_ptr, sb_ptr, out_ptr, t, padded, ag):
    """GEMM2: out[padded, 5120] = yq[padded, 8192] @ wob[5120, 8192]^T in bf16."""
    k = 8192
    n = 5120
    a = pl.make_tensor(yq_ptr, [padded, k], dtype=FP8)
    sa = pl.make_tensor(ys_ptr, [padded, k // 64, 2], dtype=E8)
    b = pl.make_tensor(wb_ptr, [n, k], dtype=FP8)
    sb = pl.make_tensor(sb_ptr, [n, k // 64, 2], dtype=E8)
    output = pl.make_tensor(out_ptr, [padded, n], dtype=pl.DT_BF16)
    # The Cube pipe barrier drains the first projection before address reuse.
    # Continue its Cube mutex numbering; L0 scale planes follow data address >> 4.
    l0_slots = 256 // K0
    cube_id = 6 + 4 * l0_slots + min(2, 65536 // (BM * BN1))
    acc_slots = min(2, 65536 // (BM * BN))
    bg = pl.make_tile_group(
        type=pl.TileType(shape=[BN, BK], dtype=FP8, target_memory=pl.MemorySpace.Mat),
        addrs=[
            2 * BM * BK,
            (2 * BM + BN) * BK,
            (2 * BM + 2 * BN) * BK,
            (2 * BM + 3 * BN) * BK,
        ],
        mutex_ids=[cube_id, cube_id + 1, cube_id + 2, cube_id + 3],
    )
    asg = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, k // 32],
            dtype=E8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZZ,
        ),
        addrs=[(2 * BM + 4 * BN) * BK],
        mutex_ids=[cube_id + 4],
    )
    bsg = pl.make_tile_group(
        type=pl.TileType(
            shape=[k // 32, BN],
            dtype=E8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NN,
        ),
        addrs=[(2 * BM + 4 * BN) * BK + BM * (k // 32)],
        mutex_ids=[cube_id + 5],
    )
    al = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, K0], dtype=FP8, target_memory=pl.MemorySpace.Left, compact=1
        ),
        addrs=[0, BM * K0][:l0_slots],
        mutex_ids=[cube_id + 6, cube_id + 7][:l0_slots],
    )
    bl = pl.make_tile_group(
        type=pl.TileType(
            shape=[K0, BN], dtype=FP8, target_memory=pl.MemorySpace.Right, compact=1
        ),
        addrs=[0, BN * K0][:l0_slots],
        mutex_ids=[cube_id + 6 + l0_slots, cube_id + 7 + l0_slots][:l0_slots],
    )
    als = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, K0 // 32],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleLeft,
            compact=1,
        ),
        addrs=[0, BM * K0 // 16][:l0_slots],
        mutex_ids=[cube_id + 6 + 2 * l0_slots, cube_id + 7 + 2 * l0_slots][:l0_slots],
    )
    bls = pl.make_tile_group(
        type=pl.TileType(
            shape=[K0 // 32, BN],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleRight,
            compact=1,
        ),
        addrs=[0, BN * K0 // 16][:l0_slots],
        mutex_ids=[cube_id + 6 + 3 * l0_slots, cube_id + 7 + 3 * l0_slots][:l0_slots],
    )
    accg = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, BN],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=[0, 4 * BM * BN][:acc_slots],
        mutex_ids=[cube_id + 6 + 4 * l0_slots, cube_id + 7 + 4 * l0_slots][:acc_slots],
    )
    b_transposed = pl.reinterpret(bg, shape=[BK, BN], layout=pl.ZN)
    with pl.section_cube():
        # First B0 was prefetched by GEMM1 at this exact physical base.
        # Immutable B1 and all B scales may load before YQ/YS are ready.
        core = pl.get_block_idx()
        first_col = core * BN
        b1 = bg[1]
        pre_scales = bsg.current()
        pl.load(b1, b, [first_col, BK])
        pl.load(pre_scales, sb, [first_col, 0, 0], order=[1, 0])
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=7)
        nm = (t + BM - 1) // BM
        nn = (n + BN - 1) // BN
        for work in pl.range(pl.get_block_idx(), nm * nn, pl.get_block_num()):
            mi = work // nn % nm * BM
            ni = work % nn * BN
            mv = pl.min(BM, t - mi)
            nv = pl.min(BN, n - ni)
            acc = accg.next()
            ast = asg.current()
            bst = bsg.current()
            pl.set_validshape(acc, [mv, nv])
            pl.set_validshape(ast, [mv, k // 32])
            pl.set_validshape(bst, [k // 32, nv])
            pl.set_validshape(ag, [mv, BK])
            pl.set_validshape(bg, [nv, BK])
            pl.set_validshape(b_transposed, [BK, nv])
            pl.set_validshape(al, [mv, K0])
            pl.set_validshape(als, [mv, K0 // 32])
            pl.set_validshape(bl, [K0, nv])
            pl.set_validshape(bls, [K0 // 32, nv])
            pl.load(ast, sa, [mi, 0, 0])
            if work != pl.get_block_idx():
                first0 = bg[0]
                first1 = bg[1]
                pl.load(first0, b, [ni, 0])
                pl.load(first1, b, [ni, BK])
                # Initial full-K BS remains resident: ni is unchanged.
            first_a = ag[0]
            second_a = ag[1]
            pl.load(first_a, a, [mi, 0])
            pl.load(second_a, a, [mi, BK])
            for ki in pl.range(0, k, BK):
                # B leads AY by two strips throughout the steady state.
                # Four physical slots separate future writes from live reads.
                if ki + 2 * BK < k:
                    future = bg[(ki // BK + 2) % 4]
                    pl.load(future, b, [ni, ki + 2 * BK])
                at = ag[(ki // BK) % 2]
                _bt = bg[(ki // BK) % 4]

                bt_transposed = b_transposed[(ki // BK) % 4]
                _mx_matmul_strip(
                    acc, at, bt_transposed, ast, bst, al, als, bl, bls, ki, k, ki
                )
                if ki + 2 * BK < k:
                    # Same physical A slot; MTE2 overwrite follows MTE1 read.
                    pl.load(at, a, [mi, ki + 2 * BK])
            pl.store(output, acc, [mi, ni], phase=pl.STPhase.Final)


def _rope_quantize_pipeline(o_ptr, cos_ptr, sin_ptr, aq_ptr, asc_ptr, t, padded):
    """M1: inverse RoPE, regroup and MX quantization of the activation (slab-tiled variant)."""
    r_rows = 16 if BM == 256 else 4
    source = pl.make_tensor(o_ptr, [t, 32768], dtype=pl.DT_BF16)
    cosine = pl.make_tensor(cos_ptr, [t, 64], dtype=pl.DT_FP32)
    sine = pl.make_tensor(sin_ptr, [t, 64], dtype=pl.DT_FP32)
    aq = pl.make_tensor(aq_ptr, [8 * padded, 4096], dtype=pl.DT_UINT8)
    asc = pl.make_tensor(asc_ptr, [8 * padded, 128], dtype=pl.DT_UINT8)
    # Rope storage uses 13120*r_rows+2560 bytes. Its stores finish before
    # the second slab notification permits Cube to write the shared UB view.
    cos_addr = 8192 * r_rows
    sin_addr = cos_addr + 256 * r_rows
    q_addr = sin_addr + 256 * r_rows
    es_addr = q_addr + (r_rows + 1) * 2048
    recip_addr = es_addr + r_rows * 64
    even_addr = recip_addr + r_rows * 256
    odd_addr = even_addr + r_rows * 1024 + 256
    xg = pl.make_tile_group(
        type=pl.TileType(
            shape=[r_rows, 2048], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[0, r_rows * 4096],
        mutex_ids=[0, 1],
    )
    cg = pl.make_tile_group(
        type=pl.TileType(
            shape=[r_rows, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[cos_addr],
        mutex_ids=[2],
    )
    sg = pl.make_tile_group(
        type=pl.TileType(
            shape=[r_rows, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[sin_addr],
        mutex_ids=[3],
    )
    qg = pl.make_tile_group(
        type=pl.TileType(
            shape=[r_rows + 1, 2048],
            dtype=pl.DT_UINT8,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[q_addr],
        mutex_ids=[4],
    )
    esg = pl.make_tile_group(
        type=pl.TileType(
            shape=[r_rows, 64], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[es_addr],
        mutex_ids=[5],
    )
    rg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, r_rows * 128],
            dtype=pl.DT_UINT16,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[recip_addr],
        mutex_ids=[6],
    )
    eg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, r_rows * 1024 + 256], dtype=FP8, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[even_addr],
        mutex_ids=[7],
    )
    og = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, r_rows * 1024 + 256], dtype=FP8, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[odd_addr],
        mutex_ids=[8],
    )
    with pl.section_vector():
        worker = pl.get_block_idx()
        group = worker % 8
        lane = worker // 8
        lanes = 8 if BN == 160 else pl.get_block_num() * pl.get_subblock_num() // 8
        row_start = t * lane // lanes
        row_end = t * (lane + 1) // lanes
        for slab in pl.range(2):
            for row in pl.range(row_start, row_end, r_rows):
                rows = pl.min(r_rows, row_end - row)
                x = xg.next()
                q = qg.current()
                es = esg.current()
                reciprocal = rg.current()
                even = eg.current()
                odd = og.current()
                valid = rows
                c = cg.current()
                s = sg.current()
                pl.set_validshape(x, [valid, 2048])
                pl.set_validshape(c, [valid, 64])
                pl.set_validshape(s, [valid, 64])
                pl.load(x, source, [row, group * 4096 + slab * 2048])
                pl.load(c, cosine, [row, 0])
                pl.load(s, sine, [row, 0])
                _apply_inverse_rope_slab(x, c, s, valid)
                _compute_mx_block_scales(x, es, reciprocal, valid * 2048)
                _quantize_to_fp8_payload(x, q, reciprocal, even, odd, valid * 2048)
                pl.set_validshape(q, [rows, 2048])
                pl.set_validshape(es, [rows, 64])
                pl.store(aq, q, [group * padded + row, slab * 2048])
                pl.store(asc, es, [group * padded + row, slab * 64])
            # Each immutable slab is globally complete before both AIV subcores
            # publish it to their AIC. MTE3 orders publication after all stores.
            if slab == 0:
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=5,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=5,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=3)
            else:
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=6,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=6,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=4)


@pl.pipeline.stage
def _gemm1_small_valid_rows(
    work,
    t,
    padded,
    a,
    sa,
    b,
    sb,
    wb,
    wbg,
    ag,
    bg,
    b_transposed,
    asg,
    bsg,
    al,
    als,
    bl,
    bls,
    accg,
    whole,
    shared,
):
    group = work // 4
    ni = work % 4 * 256
    ast = asg.current()
    bst = bsg.current()
    # ceil16(valid T)==MCAP preserves every compound Acc fragment's pitch.
    pl.set_validshape(ag, [t, 512])
    pl.set_validshape(ast, [t, 64])
    pl.set_validshape(al, [t, 512])
    pl.set_validshape(als, [t, 16])
    pl.set_validshape(accg, [t, 64])
    acc0 = accg[0]
    acc1 = accg[1]
    acc2 = accg[2]
    acc3 = accg[3]
    right0 = bl[0]
    right1 = bl[1]
    right_s0 = bls[0]
    right_s1 = bls[1]
    for ki in pl.range(0, 4096, 512):
        if ki % 2048 == 0:
            if ki == 0:
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=3)
            else:
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=4)
            pl.load(ast, sa, [group * padded, ki // 64, 0])
        slot = (ki // 512) % 2
        at = ag[slot]
        bt = bg[slot]
        tr = b_transposed[slot]
        left = al[slot]
        left_s = als[slot]
        pl.load(at, a, [group * padded, ki])
        if ki >= 1024:
            pl.load(bt, b, [group * 1024 + ni, ki])
        pl.move(left, at)
        pl.move(left_s, ast, [0, (ki % 2048) // 32])
        _small_gemm1_nstep(acc0, left, left_s, right0, right_s0, tr, bst, ki, 0)
        _small_gemm1_nstep(acc1, left, left_s, right1, right_s1, tr, bst, ki, 64)
        _small_gemm1_nstep(acc2, left, left_s, right0, right_s0, tr, bst, ki, 128)
        _small_gemm1_nstep(acc3, left, left_s, right1, right_s1, tr, bst, ki, 192)
    full = whole.current()
    pl.set_validshape(full, [t, 256])
    target = shared.next()
    pl.set_validshape(target, [(t + 1) // 2, 256])
    pl.move(
        target,
        full,
        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
        phase=pl.STPhase.Final,
    )


@pl.pipeline.stage
def _quantize_small_valid_rows(
    work, t, padded, yq, ys, shared, bfg, qg, esg, rg, eg, og, spg
):
    """M3: quantize one tile of GEMM1's output into the yq/ys payload GEMM2 consumes."""
    group = work // 4
    mi = 0
    ni = work % 4 * 256
    nv = 256
    split = (t + 1) // 2
    half = pl.max(0, pl.min(split, t - pl.get_subblock_idx() * split))
    row = mi + pl.get_subblock_idx() * split
    tile = shared.next()
    bf = bfg.current()
    q = qg.current()
    es = esg.current()
    reciprocal = rg.current()
    even = eg.current()
    odd = og.current()
    scales = spg.current()
    if half > 0:
        if BN1 == 160 or BM <= 128:
            _cast_accumulator_to_bf16(tile, bf, half * BN1, 0)
            _compute_mx_block_scales(bf, es, reciprocal, half * BN1)
            _quantize_to_fp8_payload(bf, q, reciprocal, even, odd, half * BN1)
            _pack_mx_scale_bytes(es, scales, half, BN1 // 32)
            pl.set_validshape(q, [half, nv])
            pl.set_validshape(scales, [half, nv // 32])
            pl.store(yq, q, [row, group * 1024 + ni])
            pl.store(ys, scales, [row, group * 32 + ni // 32])
        else:
            for start in pl.range(0, half, Q_ROWS):
                rows = pl.min(Q_ROWS, half - start)
                _cast_accumulator_to_bf16(tile, bf, rows * BN1, start * BN1)
                _compute_mx_block_scales(bf, es, reciprocal, rows * BN1)
                _quantize_to_fp8_payload(bf, q, reciprocal, even, odd, rows * BN1)
                _pack_mx_scale_bytes(es, scales, rows, BN1 // 32)
                pl.set_validshape(q, [rows, nv])
                pl.set_validshape(scales, [rows, nv // 32])
                pl.store(yq, q, [row + start, group * 1024 + ni])
                pl.store(ys, scales, [row + start, group * 32 + ni // 32])


@pl.pipeline.stage
def _gemm1_large_full_ascale(
    work,
    t,
    padded,
    a,
    sa,
    b,
    sb,
    wb,
    wbg,
    ag,
    bg,
    b_transposed,
    asg,
    bsg,
    al,
    als,
    bl,
    bls,
    accg,
    shared,
):
    """GEMM1 tile (group, m, n): acc = a[group, m] @ woa[group, n]^T, drained to the C->V ring."""
    nm = (t + BM - 1) // BM
    nn = (1024 + BN1 - 1) // BN1
    if BN1 == 256:
        group = (work // nn) % 8
        mi = work // (8 * nn) * BM
    else:
        group = work // (nm * nn)
        mi = work // nn % nm * BM
    ni = work % nn * BN1
    mv = pl.min(BM, t - mi)
    nv = pl.min(BN1, 1024 - ni)
    acc = accg.next()
    ast = asg.current()
    bst = bsg.current()
    pl.set_validshape(acc, [mv, nv])
    pl.set_validshape(ast, [mv, 128])
    pl.set_validshape(ag, [mv, BK])
    pl.set_validshape(al, [mv, K0])
    pl.set_validshape(als, [mv, K0 // 32])
    pl.set_validshape(bg, [nv, BK])
    pl.set_validshape(b_transposed, [BK, nv])
    pl.set_validshape(bst, [128, nv])
    pl.set_validshape(bl, [K0, nv])
    pl.set_validshape(bls, [K0 // 32, nv])
    if work != pl.get_block_idx():
        # Every slab was globally published before this core's first M completed.
        # A single full row-scale load replaces two identical-stride partial loads.
        pl.load(ast, sa, [group * padded + mi, 0, 0])
    for ki in pl.range(0, 4096, BK):
        if work == pl.get_block_idx() and ki % 2048 == 0:
            if ki == 0:
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=3)
                pl.set_validshape(ast, [mv, 64])
            else:
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=4)
                pl.set_validshape(ast, [mv, 128])
            # First half: only published columns are loaded. Second half:
            # all128 columns are now legal, and retain the declared128-column pitch.
            pl.load(ast, sa, [group * padded + mi, 0, 0])
        at = ag.next()
        bt = bg[(ki // BK) % 2]
        pl.load(at, a, [group * padded + mi, ki])
        if work != pl.get_block_idx() or ki >= (2 if BN1 == 256 else 1) * BK:
            pl.load(bt, b, [group * 1024 + ni, ki])
        tr = b_transposed[(ki // BK) % 2]
        _mx_matmul_strip(acc, at, tr, ast, bst, al, als, bl, bls, ki, 4096, ki)
    target = shared.next()
    pl.set_validshape(target, [(mv + 1) // 2, nv])
    pl.move(
        target,
        acc,
        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
        phase=pl.STPhase.Final,
    )
    if work + pl.get_block_num() >= 8 * nm * nn:
        second_nn = (5120 + BN - 1) // BN
        core = pl.get_block_idx()
        if core < nm * second_nn:
            first = wbg[0]
            col = core % second_nn * BN
            pl.set_validshape(first, [pl.min(BN, 5120 - col), BK])
            pl.load(first, wb, [col, 0])


@pl.pipeline.stage
def _gemm1_single_tile(
    work,
    t,
    padded,
    a,
    sa,
    b,
    sb,
    wb,
    wbg,
    ag,
    bg,
    b_transposed,
    asg,
    bsg,
    al,
    als,
    bl,
    bls,
    accg,
    shared,
):
    """GEMM1 tile (group, m, n): acc = a[group, m] @ woa[group, n]^T, drained to the C->V ring."""
    group = work // 4
    mi = 0
    ni = work % 4 * 256
    mv = t
    nv = 256
    acc = accg.next()
    ast = asg.current()
    bst = bsg.current()
    pl.set_validshape(acc, [mv, nv])
    pl.set_validshape(ast, [mv, 64 if BN1 == 256 else 128])
    pl.set_validshape(ag, [mv, BK])
    pl.set_validshape(al, [mv, K0])
    pl.set_validshape(als, [mv, K0 // 32])
    pl.set_validshape(bg, [nv, BK])
    pl.set_validshape(b_transposed, [BK, nv])
    pl.set_validshape(bst, [128, nv])
    pl.set_validshape(bl, [K0, nv])
    pl.set_validshape(bls, [K0 // 32, nv])
    if BN1 != 256:
        pl.load(ast, sa, [group * padded + mi, 0, 0])
    for ki in pl.range(0, 4096, BK):
        if BN1 == 256 and ki % 2048 == 0:
            if ki == 0:
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=3)
            else:
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=4)
            pl.load(ast, sa, [group * padded + mi, ki // 64, 0])
        at = ag.next()
        bt = bg[ki // BK % 2]
        pl.load(at, a, [group * padded + mi, ki])
        if ki >= (2 if BN1 == 256 else 1) * BK:
            pl.load(bt, b, [group * 1024 + ni, ki])
        tr = b_transposed[ki // BK % 2]
        _mx_matmul_strip(
            acc,
            at,
            tr,
            ast,
            bst,
            al,
            als,
            bl,
            bls,
            ki,
            4096,
            ki % 2048 if BN1 == 256 else ki,
        )
    target = shared.next()
    pl.set_validshape(target, [(mv + 1) // 2, nv])
    pl.move(
        target,
        acc,
        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
        phase=pl.STPhase.Final,
    )
    first = wbg[0]
    core = pl.get_block_idx()
    pl.set_validshape(first, [BN, BK])
    pl.load(first, wb, [core * BN, 0])


def _prefetch_gemm1_single_tile(b, sb, bg, bsg, t):
    core = pl.get_block_idx()
    group = core // 4
    col = core % 4 * 256
    first = bg[0]
    second = bg[1]
    scales = bsg.current()
    pl.set_validshape(first, [256, BK])
    pl.set_validshape(second, [256, BK])
    pl.set_validshape(scales, [128, 256])
    pl.load(first, b, [group * 1024 + col, 0])
    pl.load(second, b, [group * 1024 + col, BK])
    pl.load(scales, sb, [group * 1024 + col, 0, 0], order=[1, 0])


def _prefetch_mid_wide_b(b, sb, bg, bsg, t):
    """Load GEMM1's first B tile (and its scales) into L1 before the task loop."""
    work = pl.get_block_idx()
    group = work // 4
    col = work % 4 * 256
    nv = 256
    first = bg[0]
    second = bg[1]
    scales = bsg.current()
    pl.set_validshape(first, [nv, 512])
    pl.set_validshape(second, [nv, 512])
    pl.set_validshape(scales, [128, nv])
    pl.load(first, b, [group * 1024 + col, 0])
    pl.load(second, b, [group * 1024 + col, 512])
    pl.load(scales, sb, [group * 1024 + col, 0, 0], order=[1, 0])


@pl.pipeline.stage
def _gemm1_mid_wide_b(
    work,
    t,
    padded,
    a,
    sa,
    b,
    sb,
    wb,
    wbg,
    ag,
    bg,
    b_transposed,
    asg,
    bsg,
    al,
    als,
    bl,
    bls,
    accg,
    shared,
):
    """GEMM1 tile (group, m, n): acc = a[group, m] @ woa[group, n]^T, drained to the C->V ring."""
    group = work // 4
    mi = 0
    ni = work % 4 * 256
    mv = t
    nv = 256
    acc = accg.next()
    ast = asg.current()
    bst = bsg.current()
    pl.set_validshape(acc, [mv, nv])
    pl.set_validshape(ast, [mv, 64 if BN1 == 256 else 128])
    pl.set_validshape(ag, [mv, BK])
    pl.set_validshape(al, [mv, K0])
    pl.set_validshape(als, [mv, K0 // 32])
    pl.set_validshape(bg, [nv, 512])
    pl.set_validshape(b_transposed, [512, nv])
    pl.set_validshape(bst, [128, nv])
    pl.set_validshape(bl, [K0, nv])
    pl.set_validshape(bls, [K0 // 32, nv])
    if BN1 != 256:
        pl.load(ast, sa, [group * padded + mi, 0, 0])
    for window in pl.range(0, 4096, 512):
        if window % 2048 == 0:
            if window == 0:
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=3)
            else:
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=4)
            pl.load(ast, sa, [group * padded + mi, window // 64, 0])
        first_a = ag[0]
        pl.load(first_a, a, [group * padded + mi, window])
        # Current A was issued first; prefetch the immutable B panel two ahead.
        if window + 1024 < 4096:
            future_b = bg[(window // 512 + 2) % 3]
            pl.load(future_b, b, [group * 1024 + ni, window + 1024])
        tr = b_transposed[window // 512 % 3]
        _mid_wide_b_strip(acc, first_a, tr, ast, bst, al, als, bl, bls, window, 4096, 0)
        second_a = ag[1]
        pl.load(second_a, a, [group * padded + mi, window + BK])
        _mid_wide_b_strip(
            acc, second_a, tr, ast, bst, al, als, bl, bls, window + BK, 4096, BK
        )
    target = shared.next()
    pl.set_validshape(target, [(mv + 1) // 2, nv])
    pl.move(
        target,
        acc,
        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
        phase=pl.STPhase.Final,
    )


def _mid_wide_b_strip(acc, at, tr, ast, bst, al, als, bl, bls, ki, k, b_offset):
    _mx_matmul_offset_step(
        acc, at, tr, ast, bst, al, als, bl, bls, ki, k, ki % 2048, 0, b_offset
    )
    _mx_matmul_offset_step(
        acc, at, tr, ast, bst, al, als, bl, bls, ki, k, ki % 2048, K0, b_offset
    )


def _gemm2_mid_prefill3(yq_ptr, ys_ptr, wb_ptr, sb_ptr, out_ptr, t, padded, ag):
    """GEMM2: out[padded, 5120] = yq[padded, 8192] @ wob[5120, 8192]^T in bf16."""
    k = 8192
    n = 5120
    a = pl.make_tensor(yq_ptr, [padded, k], dtype=FP8)
    sa = pl.make_tensor(ys_ptr, [padded, k // 64, 2], dtype=E8)
    b = pl.make_tensor(wb_ptr, [n, k], dtype=FP8)
    sb = pl.make_tensor(sb_ptr, [n, k // 64, 2], dtype=E8)
    output = pl.make_tensor(out_ptr, [padded, n], dtype=pl.DT_BF16)
    l0_slots = 256 // K0
    cube_id = 6 + 4 * l0_slots + min(2, 65536 // (BM * BN1))
    acc_slots = min(2, 65536 // (BM * BN))
    bg = pl.make_tile_group(
        type=pl.TileType(shape=[BN, BK], dtype=FP8, target_memory=pl.MemorySpace.Mat),
        addrs=[
            2 * BM * BK,
            (2 * BM + BN) * BK,
            (2 * BM + 2 * BN) * BK,
            (2 * BM + 3 * BN) * BK,
        ],
        mutex_ids=[cube_id, cube_id + 1, cube_id + 2, cube_id + 3],
    )
    asg = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, k // 32],
            dtype=E8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZZ,
        ),
        addrs=[(2 * BM + 4 * BN) * BK],
        mutex_ids=[cube_id + 4],
    )
    bsg = pl.make_tile_group(
        type=pl.TileType(
            shape=[k // 32, BN],
            dtype=E8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NN,
        ),
        addrs=[(2 * BM + 4 * BN) * BK + BM * (k // 32)],
        mutex_ids=[cube_id + 5],
    )
    al = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, K0], dtype=FP8, target_memory=pl.MemorySpace.Left, compact=1
        ),
        addrs=[0, BM * K0][:l0_slots],
        mutex_ids=[cube_id + 6, cube_id + 7][:l0_slots],
    )
    bl = pl.make_tile_group(
        type=pl.TileType(
            shape=[K0, BN], dtype=FP8, target_memory=pl.MemorySpace.Right, compact=1
        ),
        addrs=[0, BN * K0][:l0_slots],
        mutex_ids=[cube_id + 6 + l0_slots, cube_id + 7 + l0_slots][:l0_slots],
    )
    als = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, K0 // 32],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleLeft,
            compact=1,
        ),
        addrs=[0, BM * K0 // 16][:l0_slots],
        mutex_ids=[cube_id + 6 + 2 * l0_slots, cube_id + 7 + 2 * l0_slots][:l0_slots],
    )
    bls = pl.make_tile_group(
        type=pl.TileType(
            shape=[K0 // 32, BN],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleRight,
            compact=1,
        ),
        addrs=[0, BN * K0 // 16][:l0_slots],
        mutex_ids=[cube_id + 6 + 3 * l0_slots, cube_id + 7 + 3 * l0_slots][:l0_slots],
    )
    accg = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, BN],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=[0, 4 * BM * BN][:acc_slots],
        mutex_ids=[cube_id + 6 + 4 * l0_slots, cube_id + 7 + 4 * l0_slots][:acc_slots],
    )
    b_transposed = pl.reinterpret(bg, shape=[BK, BN], layout=pl.ZN)
    with pl.section_cube():
        core = pl.get_block_idx()
        first_col = core * BN
        b1 = bg[1]
        pre_scales = bsg.current()
        pl.load(b1, b, [first_col, BK])
        pl.load(pre_scales, sb, [first_col, 0, 0], order=[1, 0])
        b2 = bg[2]
        pl.load(b2, b, [first_col, 2 * BK])
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=7)
        mi = 0
        ni = core * BN
        mv = t
        nv = BN
        acc = accg.next()
        ast = asg.current()
        bst = bsg.current()
        pl.set_validshape(acc, [mv, nv])
        pl.set_validshape(ast, [mv, k // 32])
        pl.set_validshape(bst, [k // 32, nv])
        pl.set_validshape(ag, [mv, BK])
        pl.set_validshape(bg, [nv, BK])
        pl.set_validshape(b_transposed, [BK, nv])
        pl.set_validshape(al, [mv, K0])
        pl.set_validshape(als, [mv, K0 // 32])
        pl.set_validshape(bl, [K0, nv])
        pl.set_validshape(bls, [K0 // 32, nv])
        pl.load(ast, sa, [mi, 0, 0])
        first_a = ag[0]
        second_a = ag[1]
        pl.load(first_a, a, [mi, 0])
        pl.load(second_a, a, [mi, BK])
        for ki in pl.range(0, k, BK):
            if ki + 3 * BK < k:
                future = bg[(ki // BK + 3) % 4]
                pl.load(future, b, [ni, ki + 3 * BK])
            at = ag[ki // BK % 2]
            _bt = bg[ki // BK % 4]
            bt_transposed = b_transposed[ki // BK % 4]
            _mx_matmul_strip(
                acc, at, bt_transposed, ast, bst, al, als, bl, bls, ki, k, ki
            )
            if ki + 2 * BK < k:
                pl.load(at, a, [mi, ki + 2 * BK])
        pl.store(output, acc, [mi, ni], phase=pl.STPhase.Final)


@pl.pipeline.stage
def _gemm1_large_full_ascale_28(
    work,
    t,
    padded,
    a,
    sa,
    b,
    sb,
    wb,
    wbg,
    ag,
    bg,
    b_transposed,
    asg,
    bsg,
    al,
    als,
    bl,
    bls,
    accg,
    shared,
):
    """GEMM1 tile (group, m, n): acc = a[group, m] @ woa[group, n]^T, drained to the C->V ring."""
    nm = (t + BM - 1) // BM
    nn = (1024 + BN1 - 1) // BN1
    if BN1 >= 128:
        group = (work // nn) % 8
        mi = work // (8 * nn) * BM
    else:
        group = work // (nm * nn)
        mi = work // nn % nm * BM
    ni = work % nn * BN1
    mv = pl.min(BM, t - mi)
    nv = pl.min(BN1, 1024 - ni)
    acc = accg.next()
    ast = asg.current()
    bst = bsg.current()
    pl.set_validshape(acc, [mv, nv])
    pl.set_validshape(ast, [mv, 128])
    pl.set_validshape(ag, [mv, BK])
    pl.set_validshape(al, [mv, K0])
    pl.set_validshape(als, [mv, K0 // 32])
    pl.set_validshape(bg, [nv, BK])
    pl.set_validshape(b_transposed, [BK, nv])
    pl.set_validshape(bst, [128, nv])
    pl.set_validshape(bl, [K0, nv])
    pl.set_validshape(bls, [K0 // 32, nv])
    if work != pl.get_block_idx():
        pl.load(bst, sb, [group * 1024 + ni, 0, 0], order=[1, 0])
        # Every slab was globally published before this core's first M completed.
        # A single full row-scale load replaces two identical-stride partial loads.
        pl.load(ast, sa, [group * padded + mi, 0, 0])
    for ki in pl.range(0, 4096, BK):
        if work == pl.get_block_idx() and ki % 2048 == 0:
            if ki == 0:
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=3)
                pl.set_validshape(ast, [mv, 64])
            else:
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=4)
                pl.set_validshape(ast, [mv, 128])
            # First half: only published columns are loaded. Second half:
            # all128 columns are now legal, and retain the declared128-column pitch.
            pl.load(ast, sa, [group * padded + mi, 0, 0])
        at = ag.next()
        bt = bg[(ki // BK) % 2]
        pl.load(at, a, [group * padded + mi, ki])
        if work != pl.get_block_idx() or ki >= (2 if BN1 >= 128 else 1) * BK:
            pl.load(bt, b, [group * 1024 + ni, ki])
        tr = b_transposed[(ki // BK) % 2]
        _mx_matmul_strip(acc, at, tr, ast, bst, al, als, bl, bls, ki, 4096, ki)
    target = shared.next()
    pl.set_validshape(target, [(mv + 1) // 2, nv])
    pl.move(
        target,
        acc,
        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
        phase=pl.STPhase.Final,
    )
    if work + pl.get_block_num() >= 8 * nm * nn:
        second_nn = (5120 + BN - 1) // BN
        core = pl.get_block_idx()
        if core < second_nn:
            first = wbg[0]
            col = core % second_nn * BN
            pl.set_validshape(first, [pl.min(BN, 5120 - col), BK])
            pl.load(first, wb, [col, 0])


def _gemm2_n192_28(yq_ptr, ys_ptr, wb_ptr, sb_ptr, out_ptr, t, padded, ag):
    """GEMM2: out[padded, 5120] = yq[padded, 8192] @ wob[5120, 8192]^T in bf16."""
    k = 8192
    n = 5120
    a = pl.make_tensor(yq_ptr, [padded, k], dtype=FP8)
    sa = pl.make_tensor(ys_ptr, [padded, k // 64, 2], dtype=E8)
    b = pl.make_tensor(wb_ptr, [n, k], dtype=FP8)
    sb = pl.make_tensor(sb_ptr, [n, k // 64, 2], dtype=E8)
    output = pl.make_tensor(out_ptr, [padded, n], dtype=pl.DT_BF16)
    # The Cube pipe barrier drains the first projection before address reuse.
    # Continue its Cube mutex numbering; L0 scale planes follow data address >> 4.
    l0_slots = 256 // K0
    cube_id = 6 + 4 * l0_slots + min(2, 65536 // (BM * BN1))
    acc_slots = min(2, 65536 // (BM * BN))
    bg = pl.make_tile_group(
        type=pl.TileType(shape=[BN, BK], dtype=FP8, target_memory=pl.MemorySpace.Mat),
        addrs=[
            2 * BM * BK,
            (2 * BM + BN) * BK,
            (2 * BM + 2 * BN) * BK,
            (2 * BM + 3 * BN) * BK,
        ],
        mutex_ids=[cube_id, cube_id + 1, cube_id + 2, cube_id + 3],
    )
    asg = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, k // 32],
            dtype=E8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZZ,
        ),
        addrs=[(2 * BM + 4 * BN) * BK],
        mutex_ids=[cube_id + 4],
    )
    bsg = pl.make_tile_group(
        type=pl.TileType(
            shape=[k // 32, BN],
            dtype=E8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NN,
        ),
        addrs=[(2 * BM + 4 * BN) * BK + BM * (k // 32)],
        mutex_ids=[cube_id + 5],
    )
    al = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, K0], dtype=FP8, target_memory=pl.MemorySpace.Left, compact=1
        ),
        addrs=[0, BM * K0][:l0_slots],
        mutex_ids=[cube_id + 6, cube_id + 7][:l0_slots],
    )
    bl = pl.make_tile_group(
        type=pl.TileType(
            shape=[K0, BN], dtype=FP8, target_memory=pl.MemorySpace.Right, compact=1
        ),
        addrs=[0, BN * K0][:l0_slots],
        mutex_ids=[cube_id + 6 + l0_slots, cube_id + 7 + l0_slots][:l0_slots],
    )
    als = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, K0 // 32],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleLeft,
            compact=1,
        ),
        addrs=[0, BM * K0 // 16][:l0_slots],
        mutex_ids=[cube_id + 6 + 2 * l0_slots, cube_id + 7 + 2 * l0_slots][:l0_slots],
    )
    bls = pl.make_tile_group(
        type=pl.TileType(
            shape=[K0 // 32, BN],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleRight,
            compact=1,
        ),
        addrs=[0, BN * K0 // 16][:l0_slots],
        mutex_ids=[cube_id + 6 + 3 * l0_slots, cube_id + 7 + 3 * l0_slots][:l0_slots],
    )
    accg = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, BN],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=[0, 4 * BM * BN][:acc_slots],
        mutex_ids=[cube_id + 6 + 4 * l0_slots, cube_id + 7 + 4 * l0_slots][:acc_slots],
    )
    b_transposed = pl.reinterpret(bg, shape=[BK, BN], layout=pl.ZN)
    with pl.section_cube():
        # First B0 was prefetched by GEMM1 at this exact physical base.
        # Immutable B1 and all B scales may load before YQ/YS are ready.
        core = pl.get_block_idx()
        first_col = core * BN
        b1 = bg[1]
        pre_scales = bsg.current()
        if core < (n + BN - 1) // BN:
            pl.set_validshape(b1, [pl.min(BN, n - first_col), BK])
            pl.set_validshape(pre_scales, [k // 32, pl.min(BN, n - first_col)])
            pl.load(b1, b, [first_col, BK])
            pl.load(pre_scales, sb, [first_col, 0, 0], order=[1, 0])
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=7)
        nm = (t + BM - 1) // BM
        nn = (n + BN - 1) // BN
        for nwork in pl.range(core, nn, pl.get_block_num()):
            for mwork in pl.range(nm):
                work = mwork * nn + nwork
                mi = mwork * BM
                ni = nwork * BN
                mv = pl.min(BM, t - mi)
                nv = pl.min(BN, n - ni)
                acc = accg.next()
                ast = asg.current()
                bst = bsg.current()
                pl.set_validshape(acc, [mv, nv])
                pl.set_validshape(ast, [mv, k // 32])
                pl.set_validshape(bst, [k // 32, nv])
                pl.set_validshape(ag, [mv, BK])
                pl.set_validshape(bg, [nv, BK])
                pl.set_validshape(b_transposed, [BK, nv])
                pl.set_validshape(al, [mv, K0])
                pl.set_validshape(als, [mv, K0 // 32])
                pl.set_validshape(bl, [K0, nv])
                pl.set_validshape(bls, [K0 // 32, nv])
                pl.load(ast, sa, [mi, 0, 0])
                if work != pl.get_block_idx():
                    first0 = bg[0]
                    first1 = bg[1]
                    pl.load(first0, b, [ni, 0])
                    pl.load(first1, b, [ni, BK])
                    # Initial full-K BS remains resident: ni is unchanged.
                first_a = ag[0]
                second_a = ag[1]
                pl.load(first_a, a, [mi, 0])
                pl.load(second_a, a, [mi, BK])
                for ki in pl.range(0, k, BK):
                    # B leads AY by two strips throughout the steady state.
                    # Four physical slots separate future writes from live reads.
                    if ki + 2 * BK < k:
                        future = bg[(ki // BK + 2) % 4]
                        pl.load(future, b, [ni, ki + 2 * BK])
                    at = ag[(ki // BK) % 2]
                    _bt = bg[(ki // BK) % 4]

                    bt_transposed = b_transposed[(ki // BK) % 4]
                    _mx_matmul_strip(
                        acc, at, bt_transposed, ast, bst, al, als, bl, bls, ki, k, ki
                    )
                    if ki + 2 * BK < k:
                        # Same physical A slot; MTE2 overwrite follows MTE1 read.
                        pl.load(at, a, [mi, ki + 2 * BK])
                pl.store(output, acc, [mi, ni], phase=pl.STPhase.Final)


def _prefetch_gemm1_weights_28(b, sb, bg, bsg, t):
    """Load GEMM1's first B tile (and its scales) into L1 before the task loop."""
    work = pl.get_block_idx()
    nm = (t + BM - 1) // BM
    nn = (1024 + BN1 - 1) // BN1
    if work < 8 * nm * nn:
        if BN1 >= 128:
            group = (work // nn) % 8
        else:
            group = work // (nm * nn)
        col = work % nn * BN1
        nv = pl.min(BN1, 1024 - col)
        first = bg[0]
        second = bg[1]
        scales = bsg.current()
        pl.set_validshape(first, [nv, BK])
        pl.set_validshape(second, [nv, BK])
        pl.set_validshape(scales, [128, nv])
        pl.load(first, b, [group * 1024 + col, 0])
        if BN1 >= 128:
            pl.load(second, b, [group * 1024 + col, BK])
        pl.load(scales, sb, [group * 1024 + col, 0, 0], order=[1, 0])


@pl.pipeline.stage
def _gemm1_small_28(
    work,
    t,
    padded,
    a,
    sa,
    b,
    sb,
    wb,
    wbg,
    ag,
    bg,
    b_transposed,
    asg,
    bsg,
    al,
    als,
    bl,
    bls,
    accg,
    whole,
    shared,
):
    group = work // 8
    ni = work % 8 * 128
    ast = asg.current()
    bst = bsg.current()
    if work != pl.get_block_idx():
        pl.load(bst, sb, [group * 1024 + ni, 0, 0], order=[1, 0])
    # ceil16(valid T)==MCAP preserves every compound Acc fragment's pitch.
    pl.set_validshape(ag, [t, 512])
    pl.set_validshape(ast, [t, 64])
    pl.set_validshape(al, [t, 512])
    pl.set_validshape(als, [t, 16])
    pl.set_validshape(accg, [t, 64])
    acc0 = accg[0]
    acc1 = accg[1]
    right0 = bl[0]
    right1 = bl[1]
    right_s0 = bls[0]
    right_s1 = bls[1]
    for ki in pl.range(0, 4096, 512):
        if ki % 2048 == 0:
            if work == pl.get_block_idx():
                if ki == 0:
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=3)
                else:
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=4)
            pl.load(ast, sa, [group * padded, ki // 64, 0])
        slot = (ki // 512) % 2
        at = ag[slot]
        bt = bg[slot]
        tr = b_transposed[slot]
        left = al[slot]
        left_s = als[slot]
        pl.load(at, a, [group * padded, ki])
        if work != pl.get_block_idx() or ki >= 1024:
            pl.load(bt, b, [group * 1024 + ni, ki])
        pl.move(left, at)
        pl.move(left_s, ast, [0, (ki % 2048) // 32])
        _small_gemm1_nstep(acc0, left, left_s, right0, right_s0, tr, bst, ki, 0)
        _small_gemm1_nstep(acc1, left, left_s, right1, right_s1, tr, bst, ki, 64)
    full = whole.current()
    pl.set_validshape(full, [t, 128])
    target = shared.next()
    pl.set_validshape(target, [(t + 1) // 2, 128])
    pl.move(
        target,
        full,
        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
        phase=pl.STPhase.Final,
    )


def _gemm2_small_28(yq_ptr, ys_ptr, wb_ptr, sb_ptr, out_ptr, t, padded):
    ck = 512
    n0 = 64
    nmain = 3
    cube_id = 6 + 4 * (256 // K0) + min(2, 65536 // (BM * BN1))
    a = pl.make_tensor(yq_ptr, [padded, 8192], dtype=FP8)
    sa = pl.make_tensor(ys_ptr, [padded, 128, 2], dtype=E8)
    b = pl.make_tensor(wb_ptr, [5120, 8192], dtype=FP8)
    sb = pl.make_tensor(sb_ptr, [5120, 128, 2], dtype=E8)
    output = pl.make_tensor(out_ptr, [padded, 5120], dtype=pl.DT_BF16)
    ag = pl.make_tile_group(
        type=pl.TileType(shape=[BM, ck], dtype=FP8, target_memory=pl.MemorySpace.Mat),
        addrs=[0, BM * ck],
        mutex_ids=[cube_id, cube_id + 1],
    )
    bg = pl.make_tile_group(
        type=pl.TileType(shape=[BN, ck], dtype=FP8, target_memory=pl.MemorySpace.Mat),
        addrs=[2 * BM * ck, (2 * BM + BN) * ck],
        mutex_ids=[cube_id + 2, cube_id + 3],
    )
    asg = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, 256], dtype=E8, target_memory=pl.MemorySpace.Mat, layout=pl.ZZ
        ),
        addrs=[2 * (BM + BN) * ck],
        mutex_ids=[cube_id + 4],
    )
    bsg = pl.make_tile_group(
        type=pl.TileType(
            shape=[256, BN], dtype=E8, target_memory=pl.MemorySpace.Mat, layout=pl.NN
        ),
        addrs=[2 * (BM + BN) * ck + BM * 256],
        mutex_ids=[cube_id + 5],
    )
    al = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, ck], dtype=FP8, target_memory=pl.MemorySpace.Left, compact=1
        ),
        addrs=[0, 32768],
        mutex_ids=[cube_id + 6, cube_id + 7],
    )
    bl = pl.make_tile_group(
        type=pl.TileType(
            shape=[ck, n0], dtype=FP8, target_memory=pl.MemorySpace.Right, compact=1
        ),
        addrs=[0, 32768],
        mutex_ids=[cube_id + 8, cube_id + 9],
    )
    als = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, ck // 32],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleLeft,
            compact=1,
        ),
        addrs=[0, 2048],
        mutex_ids=[cube_id + 10, cube_id + 11],
    )
    bls = pl.make_tile_group(
        type=pl.TileType(
            shape=[ck // 32, n0],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleRight,
            compact=1,
        ),
        addrs=[0, 2048],
        mutex_ids=[cube_id + 12, cube_id + 13],
    )
    accg = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, n0],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=[0, 4 * BM * n0, 8 * BM * n0][:nmain],
        mutex_ids=[cube_id + 14, cube_id + 15, cube_id + 16][:nmain],
    )
    btr = pl.reinterpret(bg, shape=[ck, BN], layout=pl.ZN)
    pl.set_validshape(btr, [ck, BN])
    with pl.section_cube():
        pl.set_validshape(bl, [ck, n0])
        pl.set_validshape(bls, [ck // 32, n0])
        core = pl.get_block_idx()
        first = bg[0]
        second = bg[1]
        bst = bsg.current()
        nn = (5120 + BN - 1) // BN
        if core < nn:
            nv0 = pl.min(BN, 5120 - core * BN)
            pl.set_validshape(first, [nv0, ck])
            pl.set_validshape(second, [nv0, ck])
            pl.set_validshape(bst, [256, nv0])
            pl.load(first, b, [core * BN, 0])
            pl.load(second, b, [core * BN, ck])
            pl.load(bst, sb, [core * BN, 0, 0], order=[1, 0])
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=7)
        nm = (t + BM - 1) // BM
        for work in pl.range(core, nm * nn, pl.get_block_num()):
            mi = work // nn * BM
            ni = work % nn * BN
            mv = pl.min(BM, t - mi)
            nv = pl.min(BN, 5120 - ni)
            pl.set_validshape(bg, [nv, ck])
            pl.set_validshape(btr, [ck, nv])
            pl.set_validshape(bst, [256, nv])
            acc2 = accg[2]
            pl.set_validshape(acc2, [mv, n0])
            ast = asg.current()
            acc0 = accg[0]
            pl.set_validshape(acc0, [mv, n0])
            if BM == 64:
                acc1 = accg[1]
                pl.set_validshape(acc1, [mv, n0])
            pl.set_validshape(ag, [mv, ck])
            pl.set_validshape(ast, [mv, 256])
            pl.set_validshape(al, [mv, ck])
            pl.set_validshape(als, [mv, ck // 32])
            pl.load(ast, sa, [mi, 0, 0])
            if work != core:
                pl.load(bst, sb, [ni, 0, 0], order=[1, 0])
            for ki in pl.range(0, 8192, ck):
                slot = (ki // ck) % 2
                at = ag[slot]
                bt = bg[slot]
                trans = btr[slot]
                pl.load(at, a, [mi, ki])
                if work != core or ki >= 2 * ck:
                    pl.load(bt, b, [ni, ki])
                lslot = (ki // ck) % 2
                left = al[lslot]
                left_s = als[lslot]
                right0 = bl[lslot]
                right_s0 = bls[lslot]
                right1 = bl[1 - lslot]
                right_s1 = bls[1 - lslot]
                pl.move(left, at)
                pl.move(left_s, ast, [0, ki // 32])
                _small_gemm2_nstep(
                    acc0,
                    left,
                    left_s,
                    right0,
                    right_s0,
                    trans,
                    bst,
                    output,
                    mi,
                    ni,
                    ki,
                    ck,
                    0,
                )
                _small_gemm2_nstep(
                    acc1,
                    left,
                    left_s,
                    right1,
                    right_s1,
                    trans,
                    bst,
                    output,
                    mi,
                    ni,
                    ki,
                    ck,
                    64,
                )
                if nv > 128:
                    _small_gemm2_nstep(
                        acc2,
                        left,
                        left_s,
                        right0,
                        right_s0,
                        trans,
                        bst,
                        output,
                        mi,
                        ni,
                        ki,
                        ck,
                        128,
                    )


@pl.vector_function
def _zero_bf16_pad128(bf, count: pl.DT_INT64):
    # N128 makes count a multiple of128; initialize the one missing half-chunk.
    if count % 256 != 0:
        p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
        z = vf.full(0, dtype=pl.DT_BF16)
        vf.store_align(bf, z, p, count)


@pl.pipeline.stage
def _quantize_small_28(work, t, padded, yq, ys, shared, bfg, qg, esg, rg, eg, og, spg):
    """M3: quantize one tile of GEMM1's output into the yq/ys payload GEMM2 consumes."""
    nm = (t + BM - 1) // BM
    nn = (1024 + BN1 - 1) // BN1
    if BN1 == 256:
        group = (work // nn) % 8
        mi = work // (8 * nn) * BM
    else:
        group = work // (nm * nn)
        mi = work // nn % nm * BM
    ni = work % nn * BN1
    nv = pl.min(BN1, 1024 - ni)
    split = (t + 1) // 2
    half = pl.max(0, pl.min(split, t - pl.get_subblock_idx() * split))
    row = mi + pl.get_subblock_idx() * split
    tile = shared.next()
    bf = bfg.current()
    q = qg.current()
    es = esg.current()
    reciprocal = rg.current()
    even = eg.current()
    odd = og.current()
    scales = spg.current()
    if half > 0:
        if BN1 == 160 or BM <= 128:
            _cast_accumulator_to_bf16(tile, bf, half * BN1, 0)
            _zero_bf16_pad128(bf, half * BN1)
            _compute_mx_block_scales(
                bf, es, reciprocal, ((half * BN1 + 255) // 256) * 256
            )
            _quantize_to_fp8_payload(
                bf, q, reciprocal, even, odd, ((half * BN1 + 255) // 256) * 256
            )
            _pack_mx_scale_bytes(es, scales, half, BN1 // 32)
            pl.set_validshape(q, [half, nv])
            pl.set_validshape(scales, [half, nv // 32])
            pl.store(yq, q, [row, group * 1024 + ni])
            pl.store(ys, scales, [row, group * 32 + ni // 32])
        else:
            for start in pl.range(0, half, Q_ROWS):
                rows = pl.min(Q_ROWS, half - start)
                _cast_accumulator_to_bf16(tile, bf, rows * BN1, start * BN1)
                _zero_bf16_pad128(bf, rows * BN1)
                _compute_mx_block_scales(
                    bf, es, reciprocal, ((rows * BN1 + 255) // 256) * 256
                )
                _quantize_to_fp8_payload(
                    bf, q, reciprocal, even, odd, ((rows * BN1 + 255) // 256) * 256
                )
                _pack_mx_scale_bytes(es, scales, rows, BN1 // 32)
                pl.set_validshape(q, [rows, nv])
                pl.set_validshape(scales, [rows, nv // 32])
                pl.store(yq, q, [row + start, group * 1024 + ni])
                pl.store(ys, scales, [row + start, group * 32 + ni // 32])


@pl.pipeline.stage
def _quantize_n128_28(work, t, padded, yq, ys, shared, bfg, qg, esg, rg, eg, og, spg):
    """M3: quantize one tile of GEMM1's output into the yq/ys payload GEMM2 consumes."""
    nm = (t + BM - 1) // BM
    nn = (1024 + BN1 - 1) // BN1
    if BN1 == 256:
        group = (work // nn) % 8
        mi = work // (8 * nn) * BM
    else:
        group = work // (nm * nn)
        mi = work // nn % nm * BM
    ni = work % nn * BN1
    nv = pl.min(BN1, 1024 - ni)
    extent = pl.min(BM, t - mi)
    split = (extent + 1) // 2
    half = pl.max(0, pl.min(split, extent - pl.get_subblock_idx() * split))
    row = mi + pl.get_subblock_idx() * split
    tile = shared.next()
    bf = bfg.current()
    q = qg.current()
    es = esg.current()
    reciprocal = rg.current()
    even = eg.current()
    odd = og.current()
    scales = spg.current()
    if half > 0:
        if BN1 == 160 or BM <= 128:
            _cast_accumulator_to_bf16(tile, bf, half * BN1, 0)
            _zero_bf16_pad128(bf, half * BN1)
            _compute_mx_block_scales(
                bf, es, reciprocal, ((half * BN1 + 255) // 256) * 256
            )
            _quantize_to_fp8_payload(
                bf, q, reciprocal, even, odd, ((half * BN1 + 255) // 256) * 256
            )
            _pack_mx_scale_bytes(es, scales, half, BN1 // 32)
            pl.set_validshape(q, [half, nv])
            pl.set_validshape(scales, [half, nv // 32])
            pl.store(yq, q, [row, group * 1024 + ni])
            pl.store(ys, scales, [row, group * 32 + ni // 32])
        else:
            for start in pl.range(0, half, Q_ROWS):
                rows = pl.min(Q_ROWS, half - start)
                _cast_accumulator_to_bf16(tile, bf, rows * BN1, start * BN1)
                _zero_bf16_pad128(bf, rows * BN1)
                _compute_mx_block_scales(
                    bf, es, reciprocal, ((rows * BN1 + 255) // 256) * 256
                )
                _quantize_to_fp8_payload(
                    bf, q, reciprocal, even, odd, ((rows * BN1 + 255) // 256) * 256
                )
                _pack_mx_scale_bytes(es, scales, rows, BN1 // 32)
                pl.set_validshape(q, [rows, nv])
                pl.set_validshape(scales, [rows, nv // 32])
                pl.store(yq, q, [row + start, group * 1024 + ni])
                pl.store(ys, scales, [row + start, group * 32 + ni // 32])


@pl.pipeline.stage
def _gemm1_n128_wide_28(
    work,
    t,
    padded,
    a,
    sa,
    b,
    sb,
    wb,
    wbg,
    ag,
    bg,
    b_transposed,
    asg,
    bsg,
    al,
    als,
    bl,
    bls,
    accg,
    shared,
):
    """GEMM1 tile (group, m, n): acc = a[group, m] @ woa[group, n]^T, drained to the C->V ring."""
    nm = (t + BM - 1) // BM
    nn = (1024 + BN1 - 1) // BN1
    if BN1 >= 128:
        group = (work // nn) % 8
        mi = work // (8 * nn) * BM
    else:
        group = work // (nm * nn)
        mi = work // nn % nm * BM
    ni = work % nn * BN1
    mv = pl.min(BM, t - mi)
    nv = pl.min(BN1, 1024 - ni)
    acc = accg.next()
    ast = asg.current()
    bst = bsg.current()
    pl.set_validshape(acc, [mv, nv])
    pl.set_validshape(ast, [mv, 64 if BN1 >= 128 else 128])
    pl.set_validshape(ag, [mv, BK])
    pl.set_validshape(al, [mv, K0])
    pl.set_validshape(als, [mv, K0 // 32])
    pl.set_validshape(bg, [nv, 512])
    pl.set_validshape(b_transposed, [512, nv])
    pl.set_validshape(bst, [128, nv])
    pl.set_validshape(bl, [K0, nv])
    pl.set_validshape(bls, [K0 // 32, nv])
    if BN1 < 128:
        pl.load(ast, sa, [group * padded + mi, 0, 0])
    if work != pl.get_block_idx():
        pl.load(bst, sb, [group * 1024 + ni, 0, 0], order=[1, 0])
    for window in pl.range(0, 4096, 512):
        if window % 2048 == 0:
            # All 32 AICs have a first tile; later tiles reuse immutable slabs.
            if work == pl.get_block_idx():
                if window == 0:
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=3)
                else:
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=4)
            pl.load(ast, sa, [group * padded + mi, window // 64, 0])
        bt = bg[(window // 512) % 2]
        if work != pl.get_block_idx() or window >= 1024:
            pl.load(bt, b, [group * 1024 + ni, window])
        tr = b_transposed[(window // 512) % 2]
        # One B512 acquisition covers two statically emitted A256 consumers.
        first_a = ag[0]
        pl.load(first_a, a, [group * padded + mi, window])
        _mid_wide_b_strip(acc, first_a, tr, ast, bst, al, als, bl, bls, window, 4096, 0)
        second_a = ag[1]
        pl.load(second_a, a, [group * padded + mi, window + BK])
        _mid_wide_b_strip(
            acc, second_a, tr, ast, bst, al, als, bl, bls, window + BK, 4096, BK
        )
    target = shared.next()
    pl.set_validshape(target, [(mv + 1) // 2, nv])
    pl.move(
        target,
        acc,
        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
        phase=pl.STPhase.Final,
    )
    if work + pl.get_block_num() >= 8 * nm * nn:
        core = pl.get_block_idx()
        second_nn = (5120 + BN - 1) // BN
        if core < second_nn:
            first = wbg[0]
            col = core * BN
            pl.set_validshape(first, [pl.min(BN, 5120 - col), BK])
            pl.load(first, wb, [col, 0])


def _prefetch_n128_wide_28(b, sb, bg, bsg, t):
    """Load GEMM1's first B tile (and its scales) into L1 before the task loop."""
    work = pl.get_block_idx()
    nm = (t + BM - 1) // BM
    nn = (1024 + BN1 - 1) // BN1
    if work < 8 * nm * nn:
        if BN1 >= 128:
            group = (work // nn) % 8
        else:
            group = work // (nm * nn)
        col = work % nn * BN1
        nv = pl.min(BN1, 1024 - col)
        first = bg[0]
        second = bg[1]
        scales = bsg.current()
        pl.set_validshape(first, [nv, 512])
        pl.set_validshape(second, [nv, 512])
        pl.set_validshape(scales, [128, nv])
        pl.load(first, b, [group * 1024 + col, 0])
        pl.load(second, b, [group * 1024 + col, 512])
        pl.load(scales, sb, [group * 1024 + col, 0, 0], order=[1, 0])


@pl.vector_function
def _cast_n160_full(src, bf, count: pl.DT_INT64, offset: pl.DT_INT64):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for i in pl.range(count // 64):
        f = vf.load_align(src, offset + i * 64)
        b = vf.astype(
            f,
            p,
            dtype=pl.DT_BF16,
            round_mode=pl.VFRoundMode.CAST_RINT,
            saturate=pl.SaturateMode.OFF,
        )
        vf.store_align(bf, b, p, i * 64, dist=pl.StoreDist.PACK_B32)
    if count % 64 != 0:
        # Last64 overlaps valid preceding values; it never reads invalid shared bytes.
        f = vf.load_align(src, offset + count - 64)
        b = vf.astype(
            f,
            p,
            dtype=pl.DT_BF16,
            round_mode=pl.VFRoundMode.CAST_RINT,
            saturate=pl.SaturateMode.OFF,
        )
        vf.store_align(bf, b, p, count - 64, dist=pl.StoreDist.PACK_B32)


@pl.vector_function
def _cast_n160_roi64(src, bf, rows: pl.DT_INT64, start: pl.DT_INT64):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for row in pl.range(rows):
        f = vf.load_align(src, (start + row) * 160)
        b = vf.astype(
            f,
            p,
            dtype=pl.DT_BF16,
            round_mode=pl.VFRoundMode.CAST_RINT,
            saturate=pl.SaturateMode.OFF,
        )
        vf.store_align(bf, b, p, row * 64, dist=pl.StoreDist.PACK_B32)


@pl.vector_function
def _prepare_quant_tail160(bf, es, even, odd, count: pl.DT_INT64):
    rounded = ((count + 255) // 256) * 256
    z16 = vf.full(0, dtype=pl.DT_BF16)
    for off in pl.range(count, rounded, 128):
        p = vf.update_mask(pl.min(128, rounded - off), dtype=pl.DT_BF16)
        vf.store_align(bf, z16, p, off)
    z8 = vf.full(0, dtype=pl.DT_UINT8)
    for off in pl.range(0, 576, 256):
        p = vf.update_mask(pl.min(256, 576 - off), dtype=pl.DT_UINT8)
        vf.store_align(es, z8, p, off)
    p8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    vf.store_align(even, z8, p8, rounded // 2)
    vf.store_align(odd, z8, p8, rounded // 2)


@pl.pipeline.stage
def _quantize_n160_28(work, t, padded, yq, ys, shared, bfg, qg, esg, rg, eg, og, spg):
    nn = 7
    group = (work // nn) % 8
    mi = work // (8 * nn) * BM
    ni = work % nn * 160
    nv = pl.min(160, 1024 - ni)
    extent = pl.min(BM, t - mi)
    split = (extent + 1) // 2
    half = pl.max(0, pl.min(split, extent - pl.get_subblock_idx() * split))
    row = mi + pl.get_subblock_idx() * split
    tile = shared.next()
    bf = bfg.current()
    q = qg.current()
    q64g = pl.reinterpret(qg, shape=[170, 64])
    q64 = q64g.current()
    es = esg.current()
    reciprocal = rg.current()
    even = eg.current()
    odd = og.current()
    scales = spg.current()
    for start in pl.range(0, half, 64):
        rows = pl.min(64, half - start)
        count = rows * nv
        rounded = ((count + 255) // 256) * 256
        if nv == 64:
            _cast_n160_roi64(tile, bf, rows, start)
            _prepare_quant_tail160(bf, es, even, odd, count)
            _compute_mx_block_scales(bf, es, reciprocal, rounded)
            _quantize_to_fp8_payload(bf, q64, reciprocal, even, odd, rounded)
            _pack_mx_scale_bytes(es, scales, rows, 2)
            pl.set_validshape(q64, [rows, 64])
            pl.set_validshape(scales, [rows, 2])
            pl.store(yq, q64, [row + start, group * 1024 + ni])
            pl.store(ys, scales, [row + start, group * 32 + ni // 32])
        else:
            _cast_n160_full(tile, bf, count, start * 160)
            _prepare_quant_tail160(bf, es, even, odd, count)
            _compute_mx_block_scales(bf, es, reciprocal, rounded)
            _quantize_to_fp8_payload(bf, q, reciprocal, even, odd, rounded)
            _pack_mx_scale_bytes(es, scales, rows, 5)
            pl.set_validshape(q, [rows, 160])
            pl.set_validshape(scales, [rows, 5])
            pl.store(yq, q, [row + start, group * 1024 + ni])
            pl.store(ys, scales, [row + start, group * 32 + ni // 32])


@pl.jit(
    arch="a5",
    auto_mutex=True,
    tiling_key=EpilogueKey,
    compile_timeout=300,
    pipeline=pl.pipeline.PipelineConfig(preload=0),
)
def atten_epilogue_kernel(
    o_ptr: pl.Ptr[pl.DT_UINT8],
    wa_ptr: pl.Ptr[pl.DT_UINT8],
    wb_ptr: pl.Ptr[pl.DT_UINT8],
    sa_ptr: pl.Ptr[pl.DT_UINT8],
    sb_ptr: pl.Ptr[pl.DT_UINT8],
    sin_ptr: pl.Ptr[pl.DT_UINT8],
    cos_ptr: pl.Ptr[pl.DT_UINT8],
    out_ptr: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: EpilogueTiling,
):
    # Delivery-form ABI: parameter order is exactly the op_host registration
    # (inputs, outputs, workspace, tiling); the second-to-last slot is the
    # framework workspace and the last is the tiling struct.
    t = tiling.t
    padded = tiling.padded
    # The four GM payloads are slices of the single workspace buffer, in the same
    # order and with the same sizes the host tiling budgets for.
    aq_ptr = workspace
    asc_ptr = pl.addptr(workspace, ASC_OFFSET_PER_ROW * padded)
    yq_ptr = pl.addptr(workspace, YQ_OFFSET_PER_ROW * padded)
    ys_ptr = pl.addptr(workspace, YS_OFFSET_PER_ROW * padded)
    if Algo == 1:
        with pl.section_vector():
            _saved_ctrl_policy = pl.get_ctrl_spr(60, 60)
            pl.set_ctrl_spr(60, 60, 0)
        if BN1 >= 128:
            _rope_quantize_pipeline(o_ptr, cos_ptr, sin_ptr, aq_ptr, asc_ptr, t, padded)
        else:
            _rope_quantize_whole(o_ptr, cos_ptr, sin_ptr, aq_ptr, asc_ptr, t, padded)
        a = pl.make_tensor(aq_ptr, [8 * padded, 4096], dtype=FP8)
        sa = pl.make_tensor(asc_ptr, [8 * padded, 64, 2], dtype=E8)
        b = pl.make_tensor(wa_ptr, [8192, 4096], dtype=FP8)
        sb = pl.make_tensor(sa_ptr, [8192, 64, 2], dtype=E8)
        wb = pl.make_tensor(wb_ptr, [5120, 8192], dtype=FP8)
        yq = pl.make_tensor(yq_ptr, [padded, 8192], dtype=pl.DT_UINT8)
        ys = pl.make_tensor(ys_ptr, [padded, 256], dtype=pl.DT_UINT8)
        # Byte offsets preserve the L1 prefetch alias and the C->V buffer lifetime.
        # Vec mutex IDs 0..8 belong to inverse RoPE; these groups start at ID 9.
        l0_slots = 256 // K0
        acc_id = 6 + 4 * l0_slots
        acc_slots = min(2, 65536 // (BM * BN1))
        q_rows = Q_ROWS if BN1 >= 128 else BM // 2
        scale_cols = 64 if BN1 >= 128 else 128
        vec_bf = 32768 + 2 * BM * BN1 * VSlots
        vec_q = vec_bf + 2 * q_rows * BN1
        vec_es = vec_q + (q_rows + (512 + BN1 - 1) // BN1) * BN1
        vec_recip = vec_es + ((q_rows * BN1 // 32 + 31) // 32) * 32 + 256
        vec_even = vec_recip + q_rows * BN1 // 8
        vec_odd = vec_even + q_rows * BN1 // 2 + 256
        vec_scale = vec_odd + q_rows * BN1 // 2 + 256
        ag = pl.make_tile_group(
            type=pl.TileType(
                shape=[MCAP, 512], dtype=FP8, target_memory=pl.MemorySpace.Mat
            ),
            addrs=[0, MCAP * 512],
            mutex_ids=[0, 1],
        )
        bg = pl.make_tile_group(
            type=pl.TileType(
                shape=[BN1, 512], dtype=FP8, target_memory=pl.MemorySpace.Mat
            ),
            addrs=[2 * MCAP * 512, (2 * MCAP + BN1) * 512],
            mutex_ids=[2, 3],
        )
        asg = pl.make_tile_group(
            type=pl.TileType(
                shape=[MCAP, 64],
                dtype=E8,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZZ,
            ),
            addrs=[2 * (MCAP + BN1) * 512],
            mutex_ids=[4],
        )
        bsg = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, BN1],
                dtype=E8,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NN,
            ),
            addrs=[2 * (MCAP + BN1) * 512 + MCAP * 64],
            mutex_ids=[5],
        )
        al = pl.make_tile_group(
            type=pl.TileType(
                shape=[MCAP, 512],
                dtype=FP8,
                target_memory=pl.MemorySpace.Left,
                compact=1,
            ),
            addrs=[0, 32768],
            mutex_ids=[6, 7],
        )
        bl = pl.make_tile_group(
            type=pl.TileType(
                shape=[512, 64],
                dtype=FP8,
                target_memory=pl.MemorySpace.Right,
                compact=1,
            ),
            addrs=[0, 32768],
            mutex_ids=[8, 9],
        )
        als = pl.make_tile_group(
            type=pl.TileType(
                shape=[MCAP, 16],
                dtype=E8,
                target_memory=pl.MemorySpace.ScaleLeft,
                compact=1,
            ),
            addrs=[0, 2048],
            mutex_ids=[10, 11],
        )
        bls = pl.make_tile_group(
            type=pl.TileType(
                shape=[16, 64],
                dtype=E8,
                target_memory=pl.MemorySpace.ScaleRight,
                compact=1,
            ),
            addrs=[0, 2048],
            mutex_ids=[12, 13],
        )
        accg = pl.make_tile_group(
            type=pl.TileType(
                shape=[MCAP, 64],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                compact=1,
            ),
            addrs=[0, 256 * MCAP, 512 * MCAP, 768 * MCAP],
            mutex_ids=[14, 15, 16, 17],
        )
        if BN1 == 128:
            whole = pl.make_tile_group(
                type=pl.TileType(
                    shape=[MCAP, 128],
                    dtype=pl.DT_FP32,
                    target_memory=pl.MemorySpace.Acc,
                    compact=1,
                ),
                addrs=[0],
                mutex_ids=[[14, 15]],
            )
        else:
            whole = pl.make_tile_group(
                type=pl.TileType(
                    shape=[MCAP, 256],
                    dtype=pl.DT_FP32,
                    target_memory=pl.MemorySpace.Acc,
                    compact=1,
                ),
                addrs=[0],
                mutex_ids=[[14, 15, 16, 17]],
            )
        # Cross-core IDs are literal lists selected by the compile-time slot count.
        if VSlots == 2:
            shared = pl.make_tile_group(
                type=pl.TileType(
                    shape=[BM // 2, BN1],
                    dtype=pl.DT_FP32,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=[32768, 32768 + 2 * BM * BN1],
                mutex_ids=[9, 10],
                fwd_ids=[0, 1],
                bwd_ids=[0, 1],
            )
        else:
            shared = pl.make_tile_group(
                type=pl.TileType(
                    shape=[BM // 2, BN1],
                    dtype=pl.DT_FP32,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=[32768],
                mutex_ids=[9],
                fwd_ids=[0],
                bwd_ids=[0],
            )
        bfg = pl.make_tile_group(
            type=pl.TileType(
                shape=[q_rows, BN1], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[vec_bf],
            mutex_ids=[9 + VSlots],
        )
        qg = pl.make_tile_group(
            type=pl.TileType(
                shape=[q_rows + (512 + BN1 - 1) // BN1, BN1],
                dtype=pl.DT_UINT8,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_q],
            mutex_ids=[10 + VSlots],
        )
        esg = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, (q_rows * BN1 // 32 + 31) // 32 * 32 + 256],
                dtype=pl.DT_UINT8,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_es],
            mutex_ids=[11 + VSlots],
        )
        rg = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, q_rows * BN1 // 16],
                dtype=pl.DT_UINT16,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_recip],
            mutex_ids=[12 + VSlots],
        )
        eg = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, q_rows * BN1 // 2 + 256],
                dtype=FP8,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_even],
            mutex_ids=[13 + VSlots],
        )
        og = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, q_rows * BN1 // 2 + 256],
                dtype=FP8,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_odd],
            mutex_ids=[14 + VSlots],
        )
        spg = pl.make_tile_group(
            type=pl.TileType(
                shape=[q_rows, 32], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[vec_scale],
            mutex_ids=[15 + VSlots],
        )
        b_transposed = pl.reinterpret(bg, shape=[BK, BN1], layout=pl.ZN)
        # Prefetched data must use GEMM2's physical NZ row pitch.
        wb_prefetch = pl.reinterpret(bg, shape=[BN1, BK])
        with pl.section_cube():
            if BN == 192:
                _prefetch_gemm1_weights_28(b, sb, bg, bsg, t)
            else:
                _prefetch_gemm1_weights(b, sb, bg, bsg, t)
            pl.system.bar_all()
            if BN1 < 128:
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        if BN1 < 128:
            with pl.section_vector():
                pl.system.bar_all()
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        nm = (t + BM - 1) // BM
        if BN == 160:
            work = pl.get_block_idx() // pl.get_subblock_num()
            for _handoff_once in pl.range(1):
                with pl.section_cube():
                    if BN == 192:
                        _gemm1_small_28(
                            work,
                            t,
                            padded,
                            a,
                            sa,
                            b,
                            sb,
                            wb,
                            wb_prefetch,
                            ag,
                            bg,
                            b_transposed,
                            asg,
                            bsg,
                            al,
                            als,
                            bl,
                            bls,
                            accg,
                            whole,
                            shared,
                        )
                    else:
                        _gemm1_small_valid_rows(
                            work,
                            t,
                            padded,
                            a,
                            sa,
                            b,
                            sb,
                            wb,
                            wb_prefetch,
                            ag,
                            bg,
                            b_transposed,
                            asg,
                            bsg,
                            al,
                            als,
                            bl,
                            bls,
                            accg,
                            whole,
                            shared,
                        )
                with pl.section_vector():
                    if BN1 == 128:
                        _quantize_small_28(
                            work,
                            t,
                            padded,
                            yq,
                            ys,
                            shared,
                            bfg,
                            qg,
                            esg,
                            rg,
                            eg,
                            og,
                            spg,
                        )
                    else:
                        _quantize_small_valid_rows(
                            work,
                            t,
                            padded,
                            yq,
                            ys,
                            shared,
                            bfg,
                            qg,
                            esg,
                            rg,
                            eg,
                            og,
                            spg,
                        )
        else:
            for work in pl.range(
                pl.get_block_idx() // pl.get_subblock_num(),
                8 * nm * ((1024 + BN1 - 1) // BN1),
                pl.get_block_num(),
            ):
                with pl.section_cube():
                    if BN == 192:
                        _gemm1_small_28(
                            work,
                            t,
                            padded,
                            a,
                            sa,
                            b,
                            sb,
                            wb,
                            wb_prefetch,
                            ag,
                            bg,
                            b_transposed,
                            asg,
                            bsg,
                            al,
                            als,
                            bl,
                            bls,
                            accg,
                            whole,
                            shared,
                        )
                    else:
                        _gemm1_small_valid_rows(
                            work,
                            t,
                            padded,
                            a,
                            sa,
                            b,
                            sb,
                            wb,
                            wb_prefetch,
                            ag,
                            bg,
                            b_transposed,
                            asg,
                            bsg,
                            al,
                            als,
                            bl,
                            bls,
                            accg,
                            whole,
                            shared,
                        )
                with pl.section_vector():
                    if BN1 == 128:
                        _quantize_small_28(
                            work,
                            t,
                            padded,
                            yq,
                            ys,
                            shared,
                            bfg,
                            qg,
                            esg,
                            rg,
                            eg,
                            og,
                            spg,
                        )
                    else:
                        _quantize_small_valid_rows(
                            work,
                            t,
                            padded,
                            yq,
                            ys,
                            shared,
                            bfg,
                            qg,
                            esg,
                            rg,
                            eg,
                            og,
                            spg,
                        )
        with pl.section_cube():
            pl.system.bar_all()
            if BN1 >= 128:
                pass  # G2 consumes Y7 after immutable B preload.
            else:
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        with pl.section_vector():
            if BN1 >= 128:
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=8,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=8,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=7)
            else:
                pl.system.bar_all()
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        if BN == 192:
            _gemm2_small_28(yq_ptr, ys_ptr, wb_ptr, sb_ptr, out_ptr, t, padded)
        else:
            _small_gemm2(yq_ptr, ys_ptr, wb_ptr, sb_ptr, out_ptr, t, padded)
        with pl.section_vector():
            pl.system.bar_all()
            pl.set_ctrl_spr(60, 60, _saved_ctrl_policy)
    elif Algo == 2:
        with pl.section_vector():
            _saved_ctrl_policy = pl.get_ctrl_spr(60, 60)
            pl.set_ctrl_spr(60, 60, 0)
        if BN1 == 256:
            _rope_quantize_slabbed(o_ptr, cos_ptr, sin_ptr, aq_ptr, asc_ptr, t, padded)
        else:
            _rope_quantize_whole(o_ptr, cos_ptr, sin_ptr, aq_ptr, asc_ptr, t, padded)
        a = pl.make_tensor(aq_ptr, [8 * padded, 4096], dtype=FP8)
        sa = pl.make_tensor(asc_ptr, [8 * padded, 64, 2], dtype=E8)
        b = pl.make_tensor(wa_ptr, [8192, 4096], dtype=FP8)
        sb = pl.make_tensor(sa_ptr, [8192, 64, 2], dtype=E8)
        wb = pl.make_tensor(wb_ptr, [5120, 8192], dtype=FP8)
        yq = pl.make_tensor(yq_ptr, [padded, 8192], dtype=pl.DT_UINT8)
        ys = pl.make_tensor(ys_ptr, [padded, 256], dtype=pl.DT_UINT8)
        # Byte offsets preserve the L1 prefetch alias and the C->V buffer lifetime.
        # Vec mutex IDs 0..8 belong to inverse RoPE; these groups start at ID 9.
        l0_slots = 256 // K0
        acc_id = 6 + 4 * l0_slots
        acc_slots = min(2, 65536 // (BM * BN1))
        q_rows = Q_ROWS if BN1 == 256 else BM // 2
        scale_cols = 64 if BN1 == 256 else 128
        vec_bf = 32768 + 2 * BM * BN1 * VSlots
        vec_q = vec_bf + 2 * q_rows * BN1
        vec_es = vec_q + (q_rows + (512 + BN1 - 1) // BN1) * BN1
        vec_recip = vec_es + ((q_rows * BN1 // 32 + 31) // 32) * 32 + 256
        vec_even = vec_recip + q_rows * BN1 // 8
        vec_odd = vec_even + q_rows * BN1 // 2 + 256
        vec_scale = vec_odd + q_rows * BN1 // 2 + 256
        ag = pl.make_tile_group(
            type=pl.TileType(
                shape=[BM, BK], dtype=FP8, target_memory=pl.MemorySpace.Mat
            ),
            addrs=[0, BM * BK],
            mutex_ids=[0, 1],
        )
        bg = pl.make_tile_group(
            type=pl.TileType(
                shape=[max(BN1, BN), BK], dtype=FP8, target_memory=pl.MemorySpace.Mat
            ),
            addrs=[2 * BM * BK, (2 * BM + max(BN1, BN)) * BK],
            mutex_ids=[2, 3],
        )
        asg = pl.make_tile_group(
            type=pl.TileType(
                shape=[BM, scale_cols],
                dtype=E8,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZZ,
            ),
            addrs=[2 * (BM + max(BN1, BN)) * BK],
            mutex_ids=[4],
        )
        bsg = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, BN1],
                dtype=E8,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NN,
            ),
            addrs=[2 * (BM + max(BN1, BN)) * BK + BM * scale_cols],
            mutex_ids=[5],
        )
        al = pl.make_tile_group(
            type=pl.TileType(
                shape=[BM, K0], dtype=FP8, target_memory=pl.MemorySpace.Left, compact=1
            ),
            addrs=[0, BM * K0][:l0_slots],
            mutex_ids=[6, 7][:l0_slots],
        )
        bl = pl.make_tile_group(
            type=pl.TileType(
                shape=[K0, BN1],
                dtype=FP8,
                target_memory=pl.MemorySpace.Right,
                compact=1,
            ),
            addrs=[0, BN1 * K0][:l0_slots],
            mutex_ids=[6 + l0_slots, 7 + l0_slots][:l0_slots],
        )
        als = pl.make_tile_group(
            type=pl.TileType(
                shape=[BM, K0 // 32],
                dtype=E8,
                target_memory=pl.MemorySpace.ScaleLeft,
                compact=1,
            ),
            addrs=[0, BM * K0 // 16][:l0_slots],
            mutex_ids=[6 + 2 * l0_slots, 7 + 2 * l0_slots][:l0_slots],
        )
        bls = pl.make_tile_group(
            type=pl.TileType(
                shape=[K0 // 32, BN1],
                dtype=E8,
                target_memory=pl.MemorySpace.ScaleRight,
                compact=1,
            ),
            addrs=[0, BN1 * K0 // 16][:l0_slots],
            mutex_ids=[6 + 3 * l0_slots, 7 + 3 * l0_slots][:l0_slots],
        )
        accg = pl.make_tile_group(
            type=pl.TileType(
                shape=[BM, BN1],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                compact=1,
            ),
            addrs=[0, 4 * BM * BN1][:acc_slots],
            mutex_ids=[acc_id, acc_id + 1][:acc_slots],
        )
        # Cross-core IDs are literal lists selected by the compile-time slot count.
        if VSlots == 2:
            shared = pl.make_tile_group(
                type=pl.TileType(
                    shape=[BM // 2, BN1],
                    dtype=pl.DT_FP32,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=[32768, 32768 + 2 * BM * BN1],
                mutex_ids=[9, 10],
                fwd_ids=[0, 1],
                bwd_ids=[0, 1],
            )
        else:
            shared = pl.make_tile_group(
                type=pl.TileType(
                    shape=[BM // 2, BN1],
                    dtype=pl.DT_FP32,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=[32768],
                mutex_ids=[9],
                fwd_ids=[0],
                bwd_ids=[0],
            )
        bfg = pl.make_tile_group(
            type=pl.TileType(
                shape=[q_rows, BN1], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[vec_bf],
            mutex_ids=[9 + VSlots],
        )
        qg = pl.make_tile_group(
            type=pl.TileType(
                shape=[q_rows + (512 + BN1 - 1) // BN1, BN1],
                dtype=pl.DT_UINT8,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_q],
            mutex_ids=[10 + VSlots],
        )
        esg = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, (q_rows * BN1 // 32 + 31) // 32 * 32 + 256],
                dtype=pl.DT_UINT8,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_es],
            mutex_ids=[11 + VSlots],
        )
        rg = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, q_rows * BN1 // 16],
                dtype=pl.DT_UINT16,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_recip],
            mutex_ids=[12 + VSlots],
        )
        eg = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, q_rows * BN1 // 2 + 256],
                dtype=FP8,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_even],
            mutex_ids=[13 + VSlots],
        )
        og = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, q_rows * BN1 // 2 + 256],
                dtype=FP8,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_odd],
            mutex_ids=[14 + VSlots],
        )
        spg = pl.make_tile_group(
            type=pl.TileType(
                shape=[q_rows, 32], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[vec_scale],
            mutex_ids=[15 + VSlots],
        )
        b_transposed = pl.reinterpret(bg, shape=[BK, max(BN1, BN)], layout=pl.ZN)
        # Prefetched data must use GEMM2's physical NZ row pitch.
        wb_prefetch = pl.reinterpret(bg, shape=[BN, BK])
        with pl.section_cube():
            _prefetch_gemm1_weights(b, sb, bg, bsg, t)
            pl.system.bar_all()
            if BN1 != 256:
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        if BN1 != 256:
            with pl.section_vector():
                pl.system.bar_all()
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        nm = (t + BM - 1) // BM
        for work in pl.range(
            pl.get_block_idx() // pl.get_subblock_num(),
            8 * nm * ((1024 + BN1 - 1) // BN1),
            pl.get_block_num(),
        ):
            with pl.section_cube():
                _gemm1_tile(
                    work,
                    t,
                    padded,
                    a,
                    sa,
                    b,
                    sb,
                    wb,
                    wb_prefetch,
                    ag,
                    bg,
                    b_transposed,
                    asg,
                    bsg,
                    al,
                    als,
                    bl,
                    bls,
                    accg,
                    shared,
                )
            with pl.section_vector():
                _quantize_gemm1_output(
                    work, t, padded, yq, ys, shared, bfg, qg, esg, rg, eg, og, spg
                )
        with pl.section_cube():
            pl.system.bar_all()
            # The region's reader consumes event7 inside G2 after independent
            # Wb prefetches. The Cube barrier still drains all G1 aliases.
        with pl.section_vector():
            if BN1 == 256:
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=8,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=8,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=7)
            else:
                pl.system.bar_all()
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        _gemm2_asym_pair(yq_ptr, ys_ptr, wb_ptr, sb_ptr, out_ptr, t, padded, ag)
        with pl.section_vector():
            pl.system.bar_all()
            pl.set_ctrl_spr(60, 60, _saved_ctrl_policy)
    elif Algo == 3:
        with pl.section_vector():
            _saved_ctrl_policy = pl.get_ctrl_spr(60, 60)
            pl.set_ctrl_spr(60, 60, 0)
        if BN1 == 256:
            _rope_quantize_pipeline(o_ptr, cos_ptr, sin_ptr, aq_ptr, asc_ptr, t, padded)
        else:
            _rope_quantize_whole(o_ptr, cos_ptr, sin_ptr, aq_ptr, asc_ptr, t, padded)
        a = pl.make_tensor(aq_ptr, [8 * padded, 4096], dtype=FP8)
        sa = pl.make_tensor(asc_ptr, [8 * padded, 64, 2], dtype=E8)
        b = pl.make_tensor(wa_ptr, [8192, 4096], dtype=FP8)
        sb = pl.make_tensor(sa_ptr, [8192, 64, 2], dtype=E8)
        wb = pl.make_tensor(wb_ptr, [5120, 8192], dtype=FP8)
        yq = pl.make_tensor(yq_ptr, [padded, 8192], dtype=pl.DT_UINT8)
        ys = pl.make_tensor(ys_ptr, [padded, 256], dtype=pl.DT_UINT8)
        # Byte offsets preserve the L1 prefetch alias and the C->V buffer lifetime.
        # Vec mutex IDs 0..8 belong to inverse RoPE; these groups start at ID 9.
        l0_slots = 256 // K0
        acc_id = 6 + 4 * l0_slots
        acc_slots = min(2, 65536 // (BM * BN1))
        q_rows = Q_ROWS if BN1 == 256 else BM // 2
        scale_cols = 64 if BN1 == 256 else 128
        vec_bf = 32768 + 2 * BM * BN1 * VSlots
        vec_q = vec_bf + 2 * q_rows * BN1
        vec_es = vec_q + (q_rows + (512 + BN1 - 1) // BN1) * BN1
        vec_recip = vec_es + ((q_rows * BN1 // 32 + 31) // 32) * 32 + 256
        vec_even = vec_recip + q_rows * BN1 // 8
        vec_odd = vec_even + q_rows * BN1 // 2 + 256
        vec_scale = vec_odd + q_rows * BN1 // 2 + 256
        g1_bk = 512 if BM <= 128 else BK
        g1_bslots = 3 if BM <= 128 else 2
        ag = pl.make_tile_group(
            type=pl.TileType(
                shape=[BM, BK], dtype=FP8, target_memory=pl.MemorySpace.Mat
            ),
            addrs=[0, BM * BK],
            mutex_ids=[0, 1],
        )
        bg = pl.make_tile_group(
            type=pl.TileType(
                shape=[max(BN1, BN), g1_bk], dtype=FP8, target_memory=pl.MemorySpace.Mat
            ),
            addrs=[
                2 * BM * BK,
                2 * BM * BK + max(BN1, BN) * g1_bk,
                2 * BM * BK + 2 * max(BN1, BN) * g1_bk,
            ][:g1_bslots],
            # ID16 is retired before G2 reuses it, at the root Cube bar_all.
            mutex_ids=[2, 3, 16][:g1_bslots],
        )
        asg = pl.make_tile_group(
            type=pl.TileType(
                shape=[BM, scale_cols],
                dtype=E8,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZZ,
            ),
            addrs=[(2 * BM * BK + g1_bslots * max(BN1, BN) * g1_bk)],
            mutex_ids=[4],
        )
        bsg = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, BN1],
                dtype=E8,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NN,
            ),
            addrs=[(2 * BM * BK + g1_bslots * max(BN1, BN) * g1_bk) + BM * scale_cols],
            mutex_ids=[5],
        )
        al = pl.make_tile_group(
            type=pl.TileType(
                shape=[BM, K0], dtype=FP8, target_memory=pl.MemorySpace.Left, compact=1
            ),
            addrs=[0, BM * K0][:l0_slots],
            mutex_ids=[6, 7][:l0_slots],
        )
        bl = pl.make_tile_group(
            type=pl.TileType(
                shape=[K0, BN1],
                dtype=FP8,
                target_memory=pl.MemorySpace.Right,
                compact=1,
            ),
            addrs=[0, BN1 * K0][:l0_slots],
            mutex_ids=[6 + l0_slots, 7 + l0_slots][:l0_slots],
        )
        als = pl.make_tile_group(
            type=pl.TileType(
                shape=[BM, K0 // 32],
                dtype=E8,
                target_memory=pl.MemorySpace.ScaleLeft,
                compact=1,
            ),
            addrs=[0, BM * K0 // 16][:l0_slots],
            mutex_ids=[6 + 2 * l0_slots, 7 + 2 * l0_slots][:l0_slots],
        )
        bls = pl.make_tile_group(
            type=pl.TileType(
                shape=[K0 // 32, BN1],
                dtype=E8,
                target_memory=pl.MemorySpace.ScaleRight,
                compact=1,
            ),
            addrs=[0, BN1 * K0 // 16][:l0_slots],
            mutex_ids=[6 + 3 * l0_slots, 7 + 3 * l0_slots][:l0_slots],
        )
        accg = pl.make_tile_group(
            type=pl.TileType(
                shape=[BM, BN1],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                compact=1,
            ),
            addrs=[0, 4 * BM * BN1][:acc_slots],
            mutex_ids=[acc_id, acc_id + 1][:acc_slots],
        )
        # Cross-core IDs are literal lists selected by the compile-time slot count.
        if VSlots == 2:
            shared = pl.make_tile_group(
                type=pl.TileType(
                    shape=[BM // 2, BN1],
                    dtype=pl.DT_FP32,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=[32768, 32768 + 2 * BM * BN1],
                mutex_ids=[9, 10],
                fwd_ids=[0, 1],
                bwd_ids=[0, 1],
            )
        else:
            shared = pl.make_tile_group(
                type=pl.TileType(
                    shape=[BM // 2, BN1],
                    dtype=pl.DT_FP32,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=[32768],
                mutex_ids=[9],
                fwd_ids=[0],
                bwd_ids=[0],
            )
        bfg = pl.make_tile_group(
            type=pl.TileType(
                shape=[q_rows, BN1], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[vec_bf],
            mutex_ids=[9 + VSlots],
        )
        qg = pl.make_tile_group(
            type=pl.TileType(
                shape=[q_rows + (512 + BN1 - 1) // BN1, BN1],
                dtype=pl.DT_UINT8,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_q],
            mutex_ids=[10 + VSlots],
        )
        esg = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, (q_rows * BN1 // 32 + 31) // 32 * 32 + 256],
                dtype=pl.DT_UINT8,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_es],
            mutex_ids=[11 + VSlots],
        )
        rg = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, q_rows * BN1 // 16],
                dtype=pl.DT_UINT16,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_recip],
            mutex_ids=[12 + VSlots],
        )
        eg = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, q_rows * BN1 // 2 + 256],
                dtype=FP8,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_even],
            mutex_ids=[13 + VSlots],
        )
        og = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, q_rows * BN1 // 2 + 256],
                dtype=FP8,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_odd],
            mutex_ids=[14 + VSlots],
        )
        spg = pl.make_tile_group(
            type=pl.TileType(
                shape=[q_rows, 32], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[vec_scale],
            mutex_ids=[15 + VSlots],
        )
        b_transposed = pl.reinterpret(bg, shape=[g1_bk, max(BN1, BN)], layout=pl.ZN)
        # Prefetched data must use GEMM2's physical NZ row pitch.
        wb_prefetch = pl.reinterpret(bg, shape=[BN, BK])
        with pl.section_cube():
            if BM <= 128:
                _prefetch_mid_wide_b(b, sb, bg, bsg, t)
            else:
                _prefetch_gemm1_single_tile(b, sb, bg, bsg, t)
            pl.system.bar_all()
            if BN1 != 256:
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        if BN1 != 256:
            with pl.section_vector():
                pl.system.bar_all()
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        work = pl.get_block_idx() // pl.get_subblock_num()
        for _handoff_once in pl.range(1):
            with pl.section_cube():
                if BM <= 128:
                    _gemm1_mid_wide_b(
                        work,
                        t,
                        padded,
                        a,
                        sa,
                        b,
                        sb,
                        wb,
                        wb_prefetch,
                        ag,
                        bg,
                        b_transposed,
                        asg,
                        bsg,
                        al,
                        als,
                        bl,
                        bls,
                        accg,
                        shared,
                    )
                else:
                    _gemm1_single_tile(
                        work,
                        t,
                        padded,
                        a,
                        sa,
                        b,
                        sb,
                        wb,
                        wb_prefetch,
                        ag,
                        bg,
                        b_transposed,
                        asg,
                        bsg,
                        al,
                        als,
                        bl,
                        bls,
                        accg,
                        shared,
                    )
            with pl.section_vector():
                _quantize_gemm1_valid_rows(
                    work, t, padded, yq, ys, shared, bfg, qg, esg, rg, eg, og, spg
                )
        with pl.section_cube():
            pl.system.bar_all()
            if BN1 == 256:
                pass  # New G2 consumes Y7 after B1/BS preload.
            else:
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        with pl.section_vector():
            if BN1 == 256:
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=8,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=8,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=7)
            else:
                pl.system.bar_all()
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        if BM <= 128:
            _gemm2_paired_tail(yq_ptr, ys_ptr, wb_ptr, sb_ptr, out_ptr, t, padded, ag)
        else:
            _gemm2_mid_prefill3(yq_ptr, ys_ptr, wb_ptr, sb_ptr, out_ptr, t, padded, ag)
        with pl.section_vector():
            pl.system.bar_all()
            pl.set_ctrl_spr(60, 60, _saved_ctrl_policy)
    elif Algo == 4:
        with pl.section_vector():
            _saved_ctrl_policy = pl.get_ctrl_spr(60, 60)
            pl.set_ctrl_spr(60, 60, 0)
        if BN1 >= 128:
            _rope_quantize_pipeline(o_ptr, cos_ptr, sin_ptr, aq_ptr, asc_ptr, t, padded)
        else:
            _rope_quantize_whole(o_ptr, cos_ptr, sin_ptr, aq_ptr, asc_ptr, t, padded)
        a = pl.make_tensor(aq_ptr, [8 * padded, 4096], dtype=FP8)
        sa = pl.make_tensor(asc_ptr, [8 * padded, 64, 2], dtype=E8)
        b = pl.make_tensor(wa_ptr, [8192, 4096], dtype=FP8)
        sb = pl.make_tensor(sa_ptr, [8192, 64, 2], dtype=E8)
        wb = pl.make_tensor(wb_ptr, [5120, 8192], dtype=FP8)
        yq = pl.make_tensor(yq_ptr, [padded, 8192], dtype=pl.DT_UINT8)
        ys = pl.make_tensor(ys_ptr, [padded, 256], dtype=pl.DT_UINT8)
        # Byte offsets preserve the L1 prefetch alias and the C->V buffer lifetime.
        # Vec mutex IDs 0..8 belong to inverse RoPE; these groups start at ID 9.
        l0_slots = 256 // K0
        acc_id = 6 + 4 * l0_slots
        acc_slots = min(2, 65536 // (BM * BN1))
        q_rows = Q_ROWS if BN1 >= 128 else BM // 2
        scale_cols = 128
        vec_bf = 32768 + 2 * BM * BN1 * VSlots
        vec_q = vec_bf + 2 * q_rows * BN1
        vec_es = vec_q + (q_rows + (512 + BN1 - 1) // BN1) * BN1
        vec_recip = vec_es + ((q_rows * BN1 // 32 + 31) // 32) * 32 + 256
        vec_even = vec_recip + q_rows * BN1 // 8
        vec_odd = vec_even + q_rows * BN1 // 2 + 256
        vec_scale = vec_odd + q_rows * BN1 // 2 + 256
        ag = pl.make_tile_group(
            type=pl.TileType(
                shape=[BM, BK], dtype=FP8, target_memory=pl.MemorySpace.Mat
            ),
            addrs=[0, BM * BK],
            mutex_ids=[0, 1],
        )
        g1_bk = 512 if BN1 == 128 else BK
        bg = pl.make_tile_group(
            type=pl.TileType(
                shape=[max(BN1, BN), g1_bk], dtype=FP8, target_memory=pl.MemorySpace.Mat
            ),
            addrs=[2 * BM * BK, 2 * BM * BK + max(BN1, BN) * g1_bk],
            mutex_ids=[2, 3],
        )
        asg = pl.make_tile_group(
            type=pl.TileType(
                shape=[BM, scale_cols],
                dtype=E8,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZZ,
            ),
            addrs=[2 * BM * BK + 2 * max(BN1, BN) * g1_bk],
            mutex_ids=[4],
        )
        bsg = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, BN1],
                dtype=E8,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NN,
            ),
            addrs=[2 * BM * BK + 2 * max(BN1, BN) * g1_bk + BM * scale_cols],
            mutex_ids=[5],
        )
        al = pl.make_tile_group(
            type=pl.TileType(
                shape=[BM, K0], dtype=FP8, target_memory=pl.MemorySpace.Left, compact=1
            ),
            addrs=[0, BM * K0][:l0_slots],
            mutex_ids=[6, 7][:l0_slots],
        )
        bl = pl.make_tile_group(
            type=pl.TileType(
                shape=[K0, BN1],
                dtype=FP8,
                target_memory=pl.MemorySpace.Right,
                compact=1,
            ),
            addrs=[0, BN1 * K0][:l0_slots],
            mutex_ids=[6 + l0_slots, 7 + l0_slots][:l0_slots],
        )
        als = pl.make_tile_group(
            type=pl.TileType(
                shape=[BM, K0 // 32],
                dtype=E8,
                target_memory=pl.MemorySpace.ScaleLeft,
                compact=1,
            ),
            addrs=[0, BM * K0 // 16][:l0_slots],
            mutex_ids=[6 + 2 * l0_slots, 7 + 2 * l0_slots][:l0_slots],
        )
        bls = pl.make_tile_group(
            type=pl.TileType(
                shape=[K0 // 32, BN1],
                dtype=E8,
                target_memory=pl.MemorySpace.ScaleRight,
                compact=1,
            ),
            addrs=[0, BN1 * K0 // 16][:l0_slots],
            mutex_ids=[6 + 3 * l0_slots, 7 + 3 * l0_slots][:l0_slots],
        )
        accg = pl.make_tile_group(
            type=pl.TileType(
                shape=[BM, BN1],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                compact=1,
            ),
            addrs=[0, 4 * BM * BN1][:acc_slots],
            mutex_ids=[acc_id, acc_id + 1][:acc_slots],
        )
        # Cross-core IDs are literal lists selected by the compile-time slot count.
        if VSlots == 2:
            shared = pl.make_tile_group(
                type=pl.TileType(
                    shape=[BM // 2, BN1],
                    dtype=pl.DT_FP32,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=[32768, 32768 + 2 * BM * BN1],
                mutex_ids=[9, 10],
                fwd_ids=[0, 1],
                bwd_ids=[0, 1],
            )
        else:
            shared = pl.make_tile_group(
                type=pl.TileType(
                    shape=[BM // 2, BN1],
                    dtype=pl.DT_FP32,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=[32768],
                mutex_ids=[9],
                fwd_ids=[0],
                bwd_ids=[0],
            )
        bfg = pl.make_tile_group(
            type=pl.TileType(
                shape=[q_rows, BN1], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[vec_bf],
            mutex_ids=[9 + VSlots],
        )
        qg = pl.make_tile_group(
            type=pl.TileType(
                shape=[q_rows + (512 + BN1 - 1) // BN1, BN1],
                dtype=pl.DT_UINT8,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_q],
            mutex_ids=[10 + VSlots],
        )
        esg = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, (q_rows * BN1 // 32 + 31) // 32 * 32 + 256],
                dtype=pl.DT_UINT8,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_es],
            mutex_ids=[11 + VSlots],
        )
        rg = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, q_rows * BN1 // 16],
                dtype=pl.DT_UINT16,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_recip],
            mutex_ids=[12 + VSlots],
        )
        eg = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, q_rows * BN1 // 2 + 256],
                dtype=FP8,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_even],
            mutex_ids=[13 + VSlots],
        )
        og = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, q_rows * BN1 // 2 + 256],
                dtype=FP8,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_odd],
            mutex_ids=[14 + VSlots],
        )
        spg = pl.make_tile_group(
            type=pl.TileType(
                shape=[q_rows, 32], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[vec_scale],
            mutex_ids=[15 + VSlots],
        )
        b_transposed = pl.reinterpret(bg, shape=[g1_bk, max(BN1, BN)], layout=pl.ZN)
        # Prefetched data must use GEMM2's physical NZ row pitch.
        wb_prefetch = pl.reinterpret(bg, shape=[BN, BK])
        with pl.section_cube():
            if BN == 192:
                if BN1 == 128:
                    _prefetch_n128_wide_28(b, sb, bg, bsg, t)
                else:
                    _prefetch_gemm1_weights_28(b, sb, bg, bsg, t)
            else:
                _prefetch_gemm1_weights(b, sb, bg, bsg, t)
            pl.system.bar_all()
            if BN1 < 128:
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        if BN1 < 128:
            with pl.section_vector():
                pl.system.bar_all()
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        nm = (t + BM - 1) // BM
        for work in pl.range(
            pl.get_block_idx() // pl.get_subblock_num(),
            8 * nm * ((1024 + BN1 - 1) // BN1),
            pl.get_block_num(),
        ):
            with pl.section_cube():
                if BN == 192:
                    if BN1 == 128:
                        _gemm1_n128_wide_28(
                            work,
                            t,
                            padded,
                            a,
                            sa,
                            b,
                            sb,
                            wb,
                            wb_prefetch,
                            ag,
                            bg,
                            b_transposed,
                            asg,
                            bsg,
                            al,
                            als,
                            bl,
                            bls,
                            accg,
                            shared,
                        )
                    else:
                        _gemm1_large_full_ascale_28(
                            work,
                            t,
                            padded,
                            a,
                            sa,
                            b,
                            sb,
                            wb,
                            wb_prefetch,
                            ag,
                            bg,
                            b_transposed,
                            asg,
                            bsg,
                            al,
                            als,
                            bl,
                            bls,
                            accg,
                            shared,
                        )
                else:
                    _gemm1_large_full_ascale(
                        work,
                        t,
                        padded,
                        a,
                        sa,
                        b,
                        sb,
                        wb,
                        wb_prefetch,
                        ag,
                        bg,
                        b_transposed,
                        asg,
                        bsg,
                        al,
                        als,
                        bl,
                        bls,
                        accg,
                        shared,
                    )
            with pl.section_vector():
                if BN1 == 128:
                    _quantize_n128_28(
                        work, t, padded, yq, ys, shared, bfg, qg, esg, rg, eg, og, spg
                    )
                elif BN1 == 160:
                    _quantize_n160_28(
                        work, t, padded, yq, ys, shared, bfg, qg, esg, rg, eg, og, spg
                    )
                else:
                    _quantize_gemm1_valid_rows(
                        work, t, padded, yq, ys, shared, bfg, qg, esg, rg, eg, og, spg
                    )
        with pl.section_cube():
            pl.system.bar_all()
            if BN1 >= 128:
                pass  # New G2 consumes Y7 after B1/BS preload.
            else:
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        with pl.section_vector():
            if BN1 >= 128:
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=8,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=8,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=7)
            else:
                pl.system.bar_all()
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        if BN == 192:
            _gemm2_n192_28(yq_ptr, ys_ptr, wb_ptr, sb_ptr, out_ptr, t, padded, ag)
        else:
            _gemm2_prefetch4(yq_ptr, ys_ptr, wb_ptr, sb_ptr, out_ptr, t, padded, ag)
        with pl.section_vector():
            pl.system.bar_all()
            pl.set_ctrl_spr(60, 60, _saved_ctrl_policy)
    else:
        with pl.section_vector():
            _saved_ctrl_policy = pl.get_ctrl_spr(60, 60)
            pl.set_ctrl_spr(60, 60, 0)
        if BN1 == 256:
            _rope_quantize_slabbed(o_ptr, cos_ptr, sin_ptr, aq_ptr, asc_ptr, t, padded)
        else:
            _rope_quantize_whole(o_ptr, cos_ptr, sin_ptr, aq_ptr, asc_ptr, t, padded)
        a = pl.make_tensor(aq_ptr, [8 * padded, 4096], dtype=FP8)
        sa = pl.make_tensor(asc_ptr, [8 * padded, 64, 2], dtype=E8)
        b = pl.make_tensor(wa_ptr, [8192, 4096], dtype=FP8)
        sb = pl.make_tensor(sa_ptr, [8192, 64, 2], dtype=E8)
        wb = pl.make_tensor(wb_ptr, [5120, 8192], dtype=FP8)
        yq = pl.make_tensor(yq_ptr, [padded, 8192], dtype=pl.DT_UINT8)
        ys = pl.make_tensor(ys_ptr, [padded, 256], dtype=pl.DT_UINT8)
        # Byte offsets preserve the L1 prefetch alias and the C->V buffer lifetime.
        # Vec mutex IDs 0..8 belong to inverse RoPE; these groups start at ID 9.
        l0_slots = 256 // K0
        acc_id = 6 + 4 * l0_slots
        acc_slots = min(2, 65536 // (BM * BN1))
        q_rows = Q_ROWS if BN1 == 256 else BM // 2
        scale_cols = 64 if BN1 == 256 else 128
        vec_bf = 32768 + 2 * BM * BN1 * VSlots
        vec_q = vec_bf + 2 * q_rows * BN1
        vec_es = vec_q + (q_rows + (512 + BN1 - 1) // BN1) * BN1
        vec_recip = vec_es + ((q_rows * BN1 // 32 + 31) // 32) * 32 + 256
        vec_even = vec_recip + q_rows * BN1 // 8
        vec_odd = vec_even + q_rows * BN1 // 2 + 256
        vec_scale = vec_odd + q_rows * BN1 // 2 + 256
        ag = pl.make_tile_group(
            type=pl.TileType(
                shape=[BM, BK], dtype=FP8, target_memory=pl.MemorySpace.Mat
            ),
            addrs=[0, BM * BK],
            mutex_ids=[0, 1],
        )
        bg = pl.make_tile_group(
            type=pl.TileType(
                shape=[max(BN1, BN), BK], dtype=FP8, target_memory=pl.MemorySpace.Mat
            ),
            addrs=[2 * BM * BK, (2 * BM + max(BN1, BN)) * BK],
            mutex_ids=[2, 3],
        )
        asg = pl.make_tile_group(
            type=pl.TileType(
                shape=[BM, scale_cols],
                dtype=E8,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZZ,
            ),
            addrs=[2 * (BM + max(BN1, BN)) * BK],
            mutex_ids=[4],
        )
        bsg = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, BN1],
                dtype=E8,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NN,
            ),
            addrs=[2 * (BM + max(BN1, BN)) * BK + BM * scale_cols],
            mutex_ids=[5],
        )
        al = pl.make_tile_group(
            type=pl.TileType(
                shape=[BM, K0], dtype=FP8, target_memory=pl.MemorySpace.Left, compact=1
            ),
            addrs=[0, BM * K0][:l0_slots],
            mutex_ids=[6, 7][:l0_slots],
        )
        bl = pl.make_tile_group(
            type=pl.TileType(
                shape=[K0, BN1],
                dtype=FP8,
                target_memory=pl.MemorySpace.Right,
                compact=1,
            ),
            addrs=[0, BN1 * K0][:l0_slots],
            mutex_ids=[6 + l0_slots, 7 + l0_slots][:l0_slots],
        )
        als = pl.make_tile_group(
            type=pl.TileType(
                shape=[BM, K0 // 32],
                dtype=E8,
                target_memory=pl.MemorySpace.ScaleLeft,
                compact=1,
            ),
            addrs=[0, BM * K0 // 16][:l0_slots],
            mutex_ids=[6 + 2 * l0_slots, 7 + 2 * l0_slots][:l0_slots],
        )
        bls = pl.make_tile_group(
            type=pl.TileType(
                shape=[K0 // 32, BN1],
                dtype=E8,
                target_memory=pl.MemorySpace.ScaleRight,
                compact=1,
            ),
            addrs=[0, BN1 * K0 // 16][:l0_slots],
            mutex_ids=[6 + 3 * l0_slots, 7 + 3 * l0_slots][:l0_slots],
        )
        accg = pl.make_tile_group(
            type=pl.TileType(
                shape=[BM, BN1],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                compact=1,
            ),
            addrs=[0, 4 * BM * BN1][:acc_slots],
            mutex_ids=[acc_id, acc_id + 1][:acc_slots],
        )
        # Cross-core IDs are literal lists selected by the compile-time slot count.
        if VSlots == 2:
            shared = pl.make_tile_group(
                type=pl.TileType(
                    shape=[BM // 2, BN1],
                    dtype=pl.DT_FP32,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=[32768, 32768 + 2 * BM * BN1],
                mutex_ids=[9, 10],
                fwd_ids=[0, 1],
                bwd_ids=[0, 1],
            )
        else:
            shared = pl.make_tile_group(
                type=pl.TileType(
                    shape=[BM // 2, BN1],
                    dtype=pl.DT_FP32,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=[32768],
                mutex_ids=[9],
                fwd_ids=[0],
                bwd_ids=[0],
            )
        bfg = pl.make_tile_group(
            type=pl.TileType(
                shape=[q_rows, BN1], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[vec_bf],
            mutex_ids=[9 + VSlots],
        )
        qg = pl.make_tile_group(
            type=pl.TileType(
                shape=[q_rows + (512 + BN1 - 1) // BN1, BN1],
                dtype=pl.DT_UINT8,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_q],
            mutex_ids=[10 + VSlots],
        )
        esg = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, (q_rows * BN1 // 32 + 31) // 32 * 32 + 256],
                dtype=pl.DT_UINT8,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_es],
            mutex_ids=[11 + VSlots],
        )
        rg = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, q_rows * BN1 // 16],
                dtype=pl.DT_UINT16,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_recip],
            mutex_ids=[12 + VSlots],
        )
        eg = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, q_rows * BN1 // 2 + 256],
                dtype=FP8,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_even],
            mutex_ids=[13 + VSlots],
        )
        og = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, q_rows * BN1 // 2 + 256],
                dtype=FP8,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[vec_odd],
            mutex_ids=[14 + VSlots],
        )
        spg = pl.make_tile_group(
            type=pl.TileType(
                shape=[q_rows, 32], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[vec_scale],
            mutex_ids=[15 + VSlots],
        )
        b_transposed = pl.reinterpret(bg, shape=[BK, max(BN1, BN)], layout=pl.ZN)
        # Prefetched data must use GEMM2's physical NZ row pitch.
        wb_prefetch = pl.reinterpret(bg, shape=[BN, BK])
        with pl.section_cube():
            _prefetch_gemm1_weights(b, sb, bg, bsg, t)
            pl.system.bar_all()
            if BN1 != 256:
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        if BN1 != 256:
            with pl.section_vector():
                pl.system.bar_all()
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        nm = (t + BM - 1) // BM
        for work in pl.range(
            pl.get_block_idx() // pl.get_subblock_num(),
            8 * nm * ((1024 + BN1 - 1) // BN1),
            pl.get_block_num(),
        ):
            with pl.section_cube():
                _gemm1_tile(
                    work,
                    t,
                    padded,
                    a,
                    sa,
                    b,
                    sb,
                    wb,
                    wb_prefetch,
                    ag,
                    bg,
                    b_transposed,
                    asg,
                    bsg,
                    al,
                    als,
                    bl,
                    bls,
                    accg,
                    shared,
                )
            with pl.section_vector():
                _quantize_gemm1_output(
                    work, t, padded, yq, ys, shared, bfg, qg, esg, rg, eg, og, spg
                )
        with pl.section_cube():
            pl.system.bar_all()
            if BN1 == 256:
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=7)
            else:
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        with pl.section_vector():
            if BN1 == 256:
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=8,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=8,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=7)
            else:
                pl.system.bar_all()
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        _gemm2_tile(yq_ptr, ys_ptr, wb_ptr, sb_ptr, out_ptr, t, padded)
        with pl.section_vector():
            pl.system.bar_all()
            pl.set_ctrl_spr(60, 60, _saved_ctrl_policy)


WORKSPACE_CACHE = OrderedDict()
_LOCK = threading.RLock()


def scratch_workspace(o, padded):
    """Return the cached flat GM workspace (aq/asc/yq/ys) for this device/stream/padded.

    The built-in (aclnn) path never calls this: there the host tiling reports
    ``workspace_bytes(t)`` and the framework hands the same flat buffer to the
    kernel as its second-to-last argument.  This is the jit-path equivalent.
    """
    key = (o.device, torch.npu.current_stream(o.device).npu_stream, padded)
    capture = torch.npu.is_current_stream_capturing()
    with _LOCK:
        if not capture and key in WORKSPACE_CACHE:
            WORKSPACE_CACHE.move_to_end(key)
            return WORKSPACE_CACHE[key]
        result = torch.empty(
            WORKSPACE_BYTES_PER_ROW * padded, dtype=torch.uint8, device=o.device
        )
        if not capture:
            WORKSPACE_CACHE[key] = result
            if len(WORKSPACE_CACHE) > 8:
                WORKSPACE_CACHE.popitem(last=False)
        return result


def _capacity_key_for(t, cube_cores):
    """Select the capacity profile for T given the number of cube cores in use."""
    wide = cube_cores == 32
    bm, bk, slots = (
        (64, 512, 2) if t <= 64 else ((128, 256, 2) if t <= 128 else (256, 256, 1))
    )
    return {
        "BM": bm,
        "BK": bk,
        "VSlots": slots,
        "K0": 128 if wide and bm >= 128 else 256,
        "BN1": 256 if wide else 160,
        "BN": 160 if wide else 192,
    }


def tiling_key_for(t, cube_cores):
    key = _capacity_key_for(t, cube_cores)
    algo = 0
    mcap = 16
    if cube_cores == 32:
        if t <= 64:
            algo = 1
            mcap = (t + 15) // 16 * 16
        elif t <= 256:
            algo = 3
        elif t <= 384:
            algo = 2
        elif t <= 1024:
            algo = 4
    if cube_cores == 28 and t <= 64:
        key = {"BM": 64, "BK": 512, "VSlots": 2, "K0": 256, "BN1": 128, "BN": 192}
        algo = 1
        mcap = (t + 15) // 16 * 16
    if cube_cores == 28 and 64 < t <= 1024:
        key = {
            "BM": 128 if t <= 128 else 256,
            "BK": 256,
            "VSlots": 2 if t <= 128 else 1,
            "K0": 128,
            "BN1": 160 if t <= 256 else 256,
            "BN": 192,
        }
        algo = 4
    # On32AIC, T<=128 gives32 G1 tiles: each core publishes only one C->V tile.
    # A second handoff buffer cannot overlap another output tile in this regime.
    if cube_cores == 32 and t <= 128:
        key["VSlots"] = 1
    return dict(key, Algo=algo, MCAP=mcap)


def launch_kernel(
    o, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos, workspace, out
):
    """Launch the fused kernel once, with the profile chosen for (T, cube cores)."""
    t = o.shape[0]
    padded = (t + PAD_MULTIPLE - 1) // PAD_MULTIPLE * PAD_MULTIPLE
    info = get_platform_info()
    tiling = EpilogueTiling(t=t, padded=padded)
    args = (
        o,
        woa,
        wob,
        descale_woa,
        descale_wob,
        rope_sin,
        rope_cos,
        out,
        workspace,
        tiling,
    )
    with pypto.options(pass_options={"enable_slice": False}):
        atten_epilogue_kernel[
            None, info.cube_core_num, tiling_key_for(t, info.cube_core_num)
        ](*args)


def atten_epilogue(o, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos, out):
    """Write the BF16 [T,5120] projection of the attention output into `out`.

    Signature mirrors the built-in aclnn entry (op_host registration order:
    inputs, then the caller-allocated output), so the same test runs unchanged
    in jit and aclnn mode.

    Args:
        o: bf16 ``[T, 64, 512]`` attention output (head-major, 64 heads x 512 dim).
        woa: FP8E4M3 ``[8, 1024, 4096]`` GEMM1 weights with E8M0 scales ``descale_woa``.
        wob: FP8E4M3 ``[5120, 8192]`` GEMM2 weights with E8M0 scales ``descale_wob``.
        descale_woa: E8M0 ``[8, 1024, 64, 2]`` block scales of ``woa``.
        descale_wob: E8M0 ``[5120, 128, 2]`` block scales of ``wob``.
        rope_sin: fp32 ``[T, 64]`` sine table (with ``rope_cos``) of the inverse RoPE.
        rope_cos: fp32 ``[T, 64]`` cosine table of the inverse RoPE.
        out: bf16 ``[T, 5120]`` contiguous caller-allocated output (written in place).

    Returns:
        ``out`` (the same tensor that was passed in).
    """
    if (
        not isinstance(o, torch.Tensor)
        or o.ndim != 3
        or tuple(o.shape[1:]) != (HEADS, HEAD_DIM)
    ):
        raise ValueError(f"o must have shape [T,{HEADS},{HEAD_DIM}]")
    t = o.shape[0]
    if t < 1:
        raise ValueError("T must be positive")
    if o.device.type not in ("npu", "privateuseone"):
        raise ValueError("inputs must be on an NPU")
    for name, tensor, shape, dtype in (
        ("o", o, (t, HEADS, HEAD_DIM), torch.bfloat16),
        ("woa", woa, (GROUPS, GROUP_N, GEMM1_K), torch.float8_e4m3fn),
        ("wob", wob, (OUT_FEATURES, GEMM2_K), torch.float8_e4m3fn),
        (
            "descale_woa",
            descale_woa,
            (GROUPS, GROUP_N, GEMM1_K // 64, 2),
            torch.float8_e8m0fnu,
        ),
        (
            "descale_wob",
            descale_wob,
            (OUT_FEATURES, GEMM2_K // 64, 2),
            torch.float8_e8m0fnu,
        ),
        ("rope_sin", rope_sin, (t, HEADS), torch.float32),
        ("rope_cos", rope_cos, (t, HEADS), torch.float32),
        ("out", out, (t, OUT_FEATURES), torch.bfloat16),
    ):
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}")
        if tensor.device != o.device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous on {o.device}")
        if torch_npu.get_npu_format(tensor) not in (0, 2):
            raise ValueError(f"{name} must use ND storage")
    padded = (t + PAD_MULTIPLE - 1) // PAD_MULTIPLE * PAD_MULTIPLE
    with torch.npu.device(o.device):
        workspace = scratch_workspace(o, padded)
        launch_kernel(
            o, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos, workspace, out
        )
    return out
