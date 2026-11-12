# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# PyPTO injects the eight TilingKey fields into native helper scopes.
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

__all__ = [
    "AQ_BYTES_PER_ROW",
    "ASC_BYTES_PER_ROW",
    "ASC_OFFSET_PER_ROW",
    "GEMM1_K",
    "GEMM2_K",
    "GROUPS",
    "GROUP_N",
    "HEADS",
    "HEAD_DIM",
    "OUT_FEATURES",
    "PAD_MULTIPLE",
    "WORKSPACE_BYTES_PER_ROW",
    "YQ_BYTES_PER_ROW",
    "YQ_OFFSET_PER_ROW",
    "YS_BYTES_PER_ROW",
    "YS_OFFSET_PER_ROW",
    "EpilogueKey",
    "EpilogueTiling",
    "atten_epilogue",
    "atten_epilogue_kernel",
    "launch_kernel",
    "scratch_workspace",
    "tiling_key_for",
    "vf",
    "workspace_bytes",
]

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

# Contract

HEADS = 64

HEAD_DIM = 512

GROUPS = 8

GROUP_N = 1024

GEMM1_K = 4096

OUT_FEATURES = 5120

GEMM2_K = 8192

PAD_MULTIPLE = 16

AQ_BYTES_PER_ROW = GROUPS * GEMM1_K  # 32768

ASC_BYTES_PER_ROW = GROUPS * GEMM1_K // 32  # 1024

YQ_BYTES_PER_ROW = GEMM2_K  # 8192

YS_BYTES_PER_ROW = GEMM2_K // 32  # 256

ASC_OFFSET_PER_ROW = AQ_BYTES_PER_ROW

YQ_OFFSET_PER_ROW = ASC_OFFSET_PER_ROW + ASC_BYTES_PER_ROW

YS_OFFSET_PER_ROW = YQ_OFFSET_PER_ROW + YQ_BYTES_PER_ROW

WORKSPACE_BYTES_PER_ROW = YS_OFFSET_PER_ROW + YS_BYTES_PER_ROW  # 42240

# Tiling


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


# Native

FP8, E8 = pl.DT_FP8E4M3FN, pl.DT_FP8E8M0

Q_ROWS = 64

R_ROWS = 8


def _wait_rope_slab(k_offset):
    """Wait for the same first/second immutable activation-slab notification."""
    if k_offset == 0:
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=3)
    else:
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=4)


def _gemm1_tile_coordinates(work, t):
    """Gemm1 tile coordinates."""
    nm = (t + BM - 1) // BM
    nn = (1024 + BN1 - 1) // BN1
    if BN1 == 256:
        group = (work // nn) % 8
        mi = work // (8 * nn) * BM
    else:
        group = work // (nm * nn)
        mi = work // nn % nm * BM
    ni = work % nn * BN1
    return pl.make_tuple(nm=nm, nn=nn, group=group, mi=mi, ni=ni)


def _gemm1_large_coordinates(work, t):
    """Gemm1 large coordinates."""
    nm = (t + BM - 1) // BM
    nn = (1024 + BN1 - 1) // BN1
    group = (work // nn) % 8
    mi = work // (8 * nn) * BM
    ni = work % nn * BN1
    mv = pl.min(BM, t - mi)
    nv = pl.min(BN1, 1024 - ni)
    return pl.make_tuple(nm=nm, nn=nn, group=group, mi=mi, ni=ni, mv=mv, nv=nv)


def _vector_grid_barrier(event):
    """Publish completed Vector stores across the core grid."""
    pl.system.set_cross_core(
        pipe=pl.PipeType.MTE3,
        event_id=event,
        sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
    )
    pl.system.wait_cross_core(
        pipe=pl.PipeType.MTE3,
        event_id=event,
        sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
    )


def _drain_gemm1(acc, shared, rows, columns):
    """Drain the accumulator using the profile's exact half-tile extent."""
    target = shared.next()
    pl.set_validshape(target, [rows, columns])
    pl.move(
        target,
        acc,
        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
        phase=pl.STPhase.Final,
    )


def _quantization_addresses(q_rows):
    """Quantization addresses."""
    vec_bf = 32768 + 2 * BM * BN1 * VSlots
    vec_q = vec_bf + 2 * q_rows * BN1
    vec_es = vec_q + (q_rows + (512 + BN1 - 1) // BN1) * BN1
    vec_recip = vec_es + ((q_rows * BN1 // 32 + 31) // 32) * 32 + 256
    vec_even = vec_recip + q_rows * BN1 // 8
    vec_odd = vec_even + q_rows * BN1 // 2 + 256
    vec_scale = vec_odd + q_rows * BN1 // 2 + 256
    return pl.make_tuple(
        vec_bf=vec_bf,
        vec_q=vec_q,
        vec_es=vec_es,
        vec_recip=vec_recip,
        vec_even=vec_even,
        vec_odd=vec_odd,
        vec_scale=vec_scale,
    )


def _quantization_groups(
    q_rows, vec_bf, vec_q, vec_es, vec_recip, vec_even, vec_odd, vec_scale
):
    """Quantization groups."""
    bfg = _tile_group(
        [q_rows, BN1], pl.DT_BF16, pl.MemorySpace.Vec, [vec_bf], [9 + VSlots]
    )
    qg = _tile_group(
        [q_rows + (512 + BN1 - 1) // BN1, BN1],
        pl.DT_UINT8,
        pl.MemorySpace.Vec,
        [vec_q],
        [10 + VSlots],
    )
    esg = _tile_group(
        [1, (q_rows * BN1 // 32 + 31) // 32 * 32 + 256],
        pl.DT_UINT8,
        pl.MemorySpace.Vec,
        [vec_es],
        [11 + VSlots],
    )
    rg = _tile_group(
        [1, q_rows * BN1 // 16],
        pl.DT_UINT16,
        pl.MemorySpace.Vec,
        [vec_recip],
        [12 + VSlots],
    )
    eg = _tile_group(
        [1, q_rows * BN1 // 2 + 256], FP8, pl.MemorySpace.Vec, [vec_even], [13 + VSlots]
    )
    og = _tile_group(
        [1, q_rows * BN1 // 2 + 256], FP8, pl.MemorySpace.Vec, [vec_odd], [14 + VSlots]
    )
    spg = _tile_group(
        [q_rows, 32], pl.DT_UINT8, pl.MemorySpace.Vec, [vec_scale], [15 + VSlots]
    )
    return pl.make_tuple(bfg=bfg, qg=qg, esg=esg, rg=rg, eg=eg, og=og, spg=spg)


def _rope_slab_storage(r_rows):
    """Rope slab storage."""
    cos_addr = 8192 * r_rows
    sin_addr = cos_addr + 256 * r_rows
    q_addr = sin_addr + 256 * r_rows
    es_addr = q_addr + (r_rows + 1) * 2048
    recip_addr = es_addr + r_rows * 64
    even_addr = recip_addr + r_rows * 256
    odd_addr = even_addr + r_rows * 1024 + 256
    xg = _tile_group(
        [r_rows, 2048], pl.DT_BF16, pl.MemorySpace.Vec, [0, r_rows * 4096], [0, 1]
    )
    cg = _tile_group([r_rows, 64], pl.DT_FP32, pl.MemorySpace.Vec, [cos_addr], [2])
    sg = _tile_group([r_rows, 64], pl.DT_FP32, pl.MemorySpace.Vec, [sin_addr], [3])
    qg = _tile_group([r_rows + 1, 2048], pl.DT_UINT8, pl.MemorySpace.Vec, [q_addr], [4])
    esg = _tile_group([r_rows, 64], pl.DT_UINT8, pl.MemorySpace.Vec, [es_addr], [5])
    rg = _tile_group(
        [1, r_rows * 128], pl.DT_UINT16, pl.MemorySpace.Vec, [recip_addr], [6]
    )
    eg = _tile_group(
        [1, r_rows * 1024 + 256], FP8, pl.MemorySpace.Vec, [even_addr], [7]
    )
    og = _tile_group([1, r_rows * 1024 + 256], FP8, pl.MemorySpace.Vec, [odd_addr], [8])
    return pl.make_tuple(xg=xg, cg=cg, sg=sg, qg=qg, esg=esg, rg=rg, eg=eg, og=og)


def _quantization_tiles(buffers):
    """Quantization tiles."""
    bf = buffers.bfg.current()
    q = buffers.qg.current()
    es = buffers.esg.current()
    reciprocal = buffers.rg.current()
    even = buffers.eg.current()
    odd = buffers.og.current()
    scales = buffers.spg.current()
    return pl.make_tuple(
        bf=bf, q=q, es=es, reciprocal=reciprocal, even=even, odd=odd, scales=scales
    )


def _tile_group(shape, dtype, memory, addrs, mutexes):
    """Allocate a native rotating tile group without changing its layout or IDs."""
    return pl.make_tile_group(
        type=pl.TileType(shape=shape, dtype=dtype, target_memory=memory),
        addrs=addrs,
        mutex_ids=mutexes,
    )


def _matrix_group(shape, dtype, memory, addrs, mutexes, layout):
    """Allocate matrix storage with its explicit hardware layout."""
    return pl.make_tile_group(
        type=pl.TileType(shape=shape, dtype=dtype, target_memory=memory, layout=layout),
        addrs=addrs,
        mutex_ids=mutexes,
    )


def _gemm1_small_storage():
    """Allocate the existing capacity-bounded small-M Cube storage."""
    ag = _tile_group([MCAP, 512], FP8, pl.MemorySpace.Mat, [0, MCAP * 512], [0, 1])
    bg = _tile_group(
        [BN1, 512],
        FP8,
        pl.MemorySpace.Mat,
        [2 * MCAP * 512, (2 * MCAP + BN1) * 512],
        [2, 3],
    )
    asg = _matrix_group(
        [MCAP, 64], E8, pl.MemorySpace.Mat, [2 * (MCAP + BN1) * 512], [4], pl.ZZ
    )
    bsg = _matrix_group(
        [128, BN1],
        E8,
        pl.MemorySpace.Mat,
        [2 * (MCAP + BN1) * 512 + MCAP * 64],
        [5],
        pl.NN,
    )
    al = _tile_group([MCAP, 512], FP8, pl.MemorySpace.Left, [0, 32768], [6, 7])
    bl = _tile_group([512, 64], FP8, pl.MemorySpace.Right, [0, 32768], [8, 9])
    als = _tile_group([MCAP, 16], E8, pl.MemorySpace.ScaleLeft, [0, 2048], [10, 11])
    bls = _tile_group([16, 64], E8, pl.MemorySpace.ScaleRight, [0, 2048], [12, 13])
    return pl.make_tuple(ag=ag, bg=bg, asg=asg, bsg=bsg, al=al, als=als, bl=bl, bls=bls)


def _gemm1_l1_groups(scale_columns, b_width, b_slots):
    """Use each profile's original two/three B slots and scale-plane addresses."""
    ag = _tile_group([BM, BK], FP8, pl.MemorySpace.Mat, [0, BM * BK], [0, 1])
    bg = _tile_group(
        [max(BN1, BN), b_width],
        FP8,
        pl.MemorySpace.Mat,
        [
            2 * BM * BK,
            2 * BM * BK + max(BN1, BN) * b_width,
            2 * BM * BK + 2 * max(BN1, BN) * b_width,
        ][:b_slots],
        [2, 3, 16][:b_slots],
    )
    scale_base = 2 * BM * BK + b_slots * max(BN1, BN) * b_width
    asg = _matrix_group(
        [BM, scale_columns], E8, pl.MemorySpace.Mat, [scale_base], [4], pl.ZZ
    )
    bsg = _matrix_group(
        [128, BN1],
        E8,
        pl.MemorySpace.Mat,
        [scale_base + BM * scale_columns],
        [5],
        pl.NN,
    )
    return pl.make_tuple(ag=ag, bg=bg, asg=asg, bsg=bsg)


def _gemm1_l0_groups(slots):
    """Retain the exact L0 data/scale addresses and Cube mutex numbering."""
    al = _tile_group(
        [BM, K0], FP8, pl.MemorySpace.Left, [0, BM * K0][:slots], [6, 7][:slots]
    )
    bl = _tile_group(
        [K0, BN1],
        FP8,
        pl.MemorySpace.Right,
        [0, BN1 * K0][:slots],
        [6 + slots, 7 + slots][:slots],
    )
    als = _tile_group(
        [BM, K0 // 32],
        E8,
        pl.MemorySpace.ScaleLeft,
        [0, BM * K0 // 16][:slots],
        [6 + 2 * slots, 7 + 2 * slots][:slots],
    )
    bls = _tile_group(
        [K0 // 32, BN1],
        E8,
        pl.MemorySpace.ScaleRight,
        [0, BN1 * K0 // 16][:slots],
        [6 + 3 * slots, 7 + 3 * slots][:slots],
    )
    return pl.make_tuple(al=al, als=als, bl=bl, bls=bls)


def _finish_gemm1():
    """Keep the generic MIX barrier and optimized YQ publication protocols."""
    with pl.section_cube():
        pl.system.bar_all()
        if Algo == 0:
            if BN1 == 256:
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=7)
            else:
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
    with pl.section_vector():
        if Algo == 0 and BN1 != 256:
            pl.system.bar_all()
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        else:
            _vector_grid_barrier(8)
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=7)


def _publish_rope_slab(slab):
    """Publish each immutable slab after its original global Vector barrier."""
    if slab == 0:
        _vector_grid_barrier(5)
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=3)
    else:
        _vector_grid_barrier(6)
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=4)


# Vector


@pl.vector_function
def _cast_bf16_store(src, bf, predicate, source_offset, target_offset):
    """Convert one register with the shared exact BF16 rounding mode."""
    f = vf.load_align(src, source_offset)
    b = vf.astype(
        f,
        predicate,
        dtype=pl.DT_BF16,
        round_mode=pl.VFRoundMode.CAST_RINT,
        saturate=pl.SaturateMode.OFF,
    )
    vf.store_align(bf, b, predicate, target_offset, dist=pl.StoreDist.PACK_B32)


@pl.vector_function
def _rotate_rope_store(x, predicate, swapped, cosine, signed_sine, offset):
    """Apply the exact inverse rotation to one register, then store BF16."""
    b = vf.load_align(x, offset, dist=pl.LoadDist.UNPK_B16)
    f = vf.astype(b, predicate, dtype=pl.DT_FP32)
    partner = vf.gather(f, swapped)
    a = vf.mul(f, cosine, predicate)
    v = vf.mul(partner, signed_sine, predicate)
    r = vf.add(a, v, predicate)
    bf = vf.astype(
        r,
        predicate,
        dtype=pl.DT_BF16,
        round_mode=pl.VFRoundMode.CAST_RINT,
        saturate=pl.SaturateMode.OFF,
    )
    vf.store_align(x, bf, predicate, offset, dist=pl.StoreDist.PACK_B32)


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
        _rotate_rope_store(x, p, swapped, c, signed_s, h * 512 + 448)


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
        _cast_bf16_store(src, bf, p, offset + i * 64, i * 64)


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
            _rotate_rope_store(x, p, swapped, c, signed_s, row * 2048 + h * 512 + 448)


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


def _quantize_gemm1_output(work, t, padded, destinations, buffers, shared):
    """M3: quantize one tile of GEMM1's output into the yq/ys payload GEMM2 consumes."""
    _, _, group, mi, ni = _gemm1_tile_coordinates(work, t)
    nv = pl.min(BN1, 1024 - ni)
    half = pl.min(BM, padded - mi) // 2
    row = mi + pl.get_subblock_idx() * half
    tile = shared.next()
    quant = _quantization_tiles(buffers)
    _quantize_projection_tile(
        tile, quant, destinations, row, group * 1024 + ni, half, nv
    )


def _rope_quantize_whole(o_ptr, cos_ptr, sin_ptr, aq_ptr, asc_ptr, t, padded):
    """M1: inverse RoPE, regroup and MX quantization of the activation (whole-range variant)."""
    source = pl.make_tensor(o_ptr, [t * 8, 4096], dtype=pl.DT_BF16)
    cosine = pl.make_tensor(cos_ptr, [t, 64], dtype=pl.DT_FP32)
    sine = pl.make_tensor(sin_ptr, [t, 64], dtype=pl.DT_FP32)
    aq = pl.make_tensor(aq_ptr, [8 * padded, 4096], dtype=pl.DT_UINT8)
    asc = pl.make_tensor(asc_ptr, [8 * padded, 128], dtype=pl.DT_UINT8)
    # The activation workspace occupies Vec bytes [0, 26752).
    xg = _tile_group([1, 4096], pl.DT_BF16, pl.MemorySpace.Vec, [0, 8192], [0, 1])
    cg = _tile_group([1, 64], pl.DT_FP32, pl.MemorySpace.Vec, [16384], [2])
    sg = _tile_group([1, 64], pl.DT_FP32, pl.MemorySpace.Vec, [16640], [3])
    qg = _tile_group([1, 4608], pl.DT_UINT8, pl.MemorySpace.Vec, [16896], [4])
    esg = _tile_group([1, 128], pl.DT_UINT8, pl.MemorySpace.Vec, [21504], [5])
    rg = _tile_group([1, 256], pl.DT_UINT16, pl.MemorySpace.Vec, [21632], [6])
    eg = _tile_group([1, 2304], FP8, pl.MemorySpace.Vec, [22144], [7])
    og = _tile_group([1, 2304], FP8, pl.MemorySpace.Vec, [24448], [8])
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


def _quantize_gemm1_valid_rows(work, t, destinations, buffers, shared):
    """M3: quantize one tile of GEMM1's output into the yq/ys payload GEMM2 consumes."""
    if Algo == 3:
        group = work // 4
        mi = 0
        ni = work % 4 * 256
        nv = 256
        extent = t
    else:
        nn = (1024 + BN1 - 1) // BN1
        group = (work // nn) % 8
        mi = work // (8 * nn) * BM
        ni = work % nn * BN1
        nv = pl.min(BN1, 1024 - ni)
        extent = pl.min(BM, t - mi)
    split = (extent + 1) // 2
    half = pl.max(0, pl.min(split, extent - pl.get_subblock_idx() * split))
    row = mi + pl.get_subblock_idx() * split
    tile = shared.next()
    quant = _quantization_tiles(buffers)
    if half > 0:
        _quantize_projection_tile(
            tile, quant, destinations, row, group * 1024 + ni, half, nv
        )


def _quantize_small_valid_rows(work, t, destinations, buffers, shared):
    """M3: quantize one tile of GEMM1's output into the yq/ys payload GEMM2 consumes."""
    group = work // 4
    mi = 0
    ni = work % 4 * 256
    nv = 256
    split = (t + 1) // 2
    half = pl.max(0, pl.min(split, t - pl.get_subblock_idx() * split))
    row = mi + pl.get_subblock_idx() * split
    tile = shared.next()
    quant = _quantization_tiles(buffers)
    if half > 0:
        _quantize_projection_rows(
            tile, quant, destinations, row, group * 1024 + ni, half, nv, 0
        )


@pl.vector_function
def _zero_bf16_pad128(bf, count: pl.DT_INT64):
    # N128 makes count a multiple of128; initialize the one missing half-chunk.
    if count % 256 != 0:
        p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
        z = vf.full(0, dtype=pl.DT_BF16)
        vf.store_align(bf, z, p, count)


def _quantize_small_28(work, t, destinations, buffers, shared):
    """M3: quantize one tile of GEMM1's output into the yq/ys payload GEMM2 consumes."""
    nm = (t + BM - 1) // BM
    nn = (1024 + BN1 - 1) // BN1
    group = work // (nm * nn)
    mi = work // nn % nm * BM
    ni = work % nn * BN1
    nv = pl.min(BN1, 1024 - ni)
    split = (t + 1) // 2
    half = pl.max(0, pl.min(split, t - pl.get_subblock_idx() * split))
    row = mi + pl.get_subblock_idx() * split
    tile = shared.next()
    quant = _quantization_tiles(buffers)
    if half > 0:
        _cast_accumulator_to_bf16(tile, quant.bf, half * BN1, 0)
        _zero_bf16_pad128(quant.bf, half * BN1)
        _compute_mx_block_scales(
            quant.bf, quant.es, quant.reciprocal, ((half * BN1 + 255) // 256) * 256
        )
        _quantize_to_fp8_payload(
            quant.bf,
            quant.q,
            quant.reciprocal,
            quant.even,
            quant.odd,
            ((half * BN1 + 255) // 256) * 256,
        )
        _pack_mx_scale_bytes(quant.es, quant.scales, half, BN1 // 32)
        pl.set_validshape(quant.q, [half, nv])
        pl.set_validshape(quant.scales, [half, nv // 32])
        pl.store(destinations.yq, quant.q, [row, group * 1024 + ni])
        pl.store(destinations.ys, quant.scales, [row, group * 32 + ni // 32])


@pl.vector_function
def _cast_n160_full(src, bf, count: pl.DT_INT64, offset: pl.DT_INT64):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for i in pl.range(count // 64):
        _cast_bf16_store(src, bf, p, offset + i * 64, i * 64)
    if count % 64 != 0:
        # Last64 overlaps valid preceding values; it never reads invalid shared bytes.
        _cast_bf16_store(src, bf, p, offset + count - 64, count - 64)


@pl.vector_function
def _cast_n160_roi64(src, bf, rows: pl.DT_INT64, start: pl.DT_INT64):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for row in pl.range(rows):
        _cast_bf16_store(src, bf, p, (start + row) * 160, row * 64)


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


def _quantize_n160_28(work, t, destinations, buffers, shared):
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
    bf = buffers.bfg.current()
    q = buffers.qg.current()
    q64g = pl.reinterpret(buffers.qg, shape=[170, 64])
    q64 = q64g.current()
    es = buffers.esg.current()
    reciprocal = buffers.rg.current()
    even = buffers.eg.current()
    odd = buffers.og.current()
    scales = buffers.spg.current()
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
            pl.store(destinations.yq, q64, [row + start, group * 1024 + ni])
            pl.store(destinations.ys, scales, [row + start, group * 32 + ni // 32])
        else:
            _cast_n160_full(tile, bf, count, start * 160)
            _prepare_quant_tail160(bf, es, even, odd, count)
            _compute_mx_block_scales(bf, es, reciprocal, rounded)
            _quantize_to_fp8_payload(bf, q, reciprocal, even, odd, rounded)
            _pack_mx_scale_bytes(es, scales, rows, 5)
            pl.set_validshape(q, [rows, 160])
            pl.set_validshape(scales, [rows, 5])
            pl.store(destinations.yq, q, [row + start, group * 1024 + ni])
            pl.store(destinations.ys, scales, [row + start, group * 32 + ni // 32])


def _prepare_activation(o_ptr, cos_ptr, sin_ptr, aq_ptr, asc_ptr, t, padded):
    """Retain each profile's original RoPE/quantization publication scheme."""
    if Algo == 1 or Algo == 3 or Algo == 4:
        _rope_quantize_tiled(
            o_ptr,
            cos_ptr,
            sin_ptr,
            aq_ptr,
            asc_ptr,
            t,
            padded,
            16 if BM == 256 else 4,
            0,
        )
    elif Algo == 2 or BN1 == 256:
        _rope_quantize_tiled(
            o_ptr,
            cos_ptr,
            sin_ptr,
            aq_ptr,
            asc_ptr,
            t,
            padded,
            R_ROWS if BM == 256 else 4,
            1,
        )
    else:
        _rope_quantize_whole(o_ptr, cos_ptr, sin_ptr, aq_ptr, asc_ptr, t, padded)


@pl.pipeline.stage
def _quantize_profile(work, t, padded, destinations, buffers, shared):
    """The Vector actor paired with every Cube work item."""
    if Algo == 1:
        if BN == 160:
            _quantize_small_valid_rows(work, t, destinations, buffers, shared)
        else:
            _quantize_small_28(work, t, destinations, buffers, shared)
    elif Algo == 3:
        _quantize_gemm1_valid_rows(work, t, destinations, buffers, shared)
    elif Algo == 4:
        if BN1 == 160:
            _quantize_n160_28(work, t, destinations, buffers, shared)
        else:
            _quantize_gemm1_valid_rows(work, t, destinations, buffers, shared)
    else:
        _quantize_gemm1_output(work, t, padded, destinations, buffers, shared)


def _quantize_projection_rows(
    tile, quant, destinations, row, column, rows, valid_columns, source_offset
):
    """Convert exactly the same rows and preserve the valid GM writeback extent."""
    _cast_accumulator_to_bf16(tile, quant.bf, rows * BN1, source_offset)
    _compute_mx_block_scales(quant.bf, quant.es, quant.reciprocal, rows * BN1)
    _quantize_to_fp8_payload(
        quant.bf, quant.q, quant.reciprocal, quant.even, quant.odd, rows * BN1
    )
    _pack_mx_scale_bytes(quant.es, quant.scales, rows, BN1 // 32)
    pl.set_validshape(quant.q, [rows, valid_columns])
    pl.set_validshape(quant.scales, [rows, valid_columns // 32])
    pl.store(destinations.yq, quant.q, [row, column])
    pl.store(destinations.ys, quant.scales, [row, column // 32])


def _rope_quantize_valid_batch(views, buffers, quant, position, valid):
    """Load and rotate exactly the valid rows of one 2048-column slab."""
    c = buffers.cg.current()
    s = buffers.sg.current()
    pl.set_validshape(quant.x, [valid, 2048])
    pl.set_validshape(c, [valid, 64])
    pl.set_validshape(s, [valid, 64])
    pl.load(
        quant.x,
        views.source,
        [position.row, position.group * 4096 + position.slab * 2048],
    )
    pl.load(c, views.cosine, [position.row, 0])
    pl.load(s, views.sine, [position.row, 0])
    _apply_inverse_rope_slab(quant.x, c, s, valid)
    _compute_mx_block_scales(quant.x, quant.es, quant.reciprocal, valid * 2048)
    _quantize_to_fp8_payload(
        quant.x, quant.q, quant.reciprocal, quant.even, quant.odd, valid * 2048
    )


def _rope_quantize_batch(views, buffers, position, t, padded, padded_rows):
    """Quantize one scheduled batch, preserving its original zero-fill policy."""
    x = buffers.xg.next()
    q = buffers.qg.current()
    es = buffers.esg.current()
    reciprocal = buffers.rg.current()
    even = buffers.eg.current()
    odd = buffers.og.current()
    quant = pl.make_tuple(x=x, q=q, es=es, reciprocal=reciprocal, even=even, odd=odd)
    if padded_rows:
        if position.row < t:
            valid = pl.min(position.rows, t - position.row)
            _rope_quantize_valid_batch(views, buffers, quant, position, valid)
            if valid < position.rows:
                _zero_rope_tail(q, es, valid, position.rows)
        else:
            _zero_rope_batch(q, es, position.rows)
    else:
        _rope_quantize_valid_batch(views, buffers, quant, position, position.rows)
    pl.set_validshape(q, [position.rows, 2048])
    pl.set_validshape(es, [position.rows, 64])
    pl.store(
        views.aq, q, [position.group * padded + position.row, position.slab * 2048]
    )
    pl.store(
        views.asc, es, [position.group * padded + position.row, position.slab * 64]
    )


def _quantize_projection_tile(
    tile, quant, destinations, row, column, rows, valid_columns
):
    """Use each profile's original whole-half or 64-row conversion granularity."""
    if BN1 == 160 or BM <= 128:
        _quantize_projection_rows(
            tile, quant, destinations, row, column, rows, valid_columns, 0
        )
    else:
        for start in pl.range(0, rows, Q_ROWS):
            count = pl.min(Q_ROWS, rows - start)
            _quantize_projection_rows(
                tile,
                quant,
                destinations,
                row + start,
                column,
                count,
                valid_columns,
                start * BN1,
            )


def _rope_quantize_tiled(
    o_ptr, cos_ptr, sin_ptr, aq_ptr, asc_ptr, t, padded, r_rows, padded_rows
):
    """Retain the original row partition and zero-fill policy of each tiled path."""
    source = pl.make_tensor(o_ptr, [t, 32768], dtype=pl.DT_BF16)
    cosine = pl.make_tensor(cos_ptr, [t, 64], dtype=pl.DT_FP32)
    sine = pl.make_tensor(sin_ptr, [t, 64], dtype=pl.DT_FP32)
    aq = pl.make_tensor(aq_ptr, [8 * padded, 4096], dtype=pl.DT_UINT8)
    asc = pl.make_tensor(asc_ptr, [8 * padded, 128], dtype=pl.DT_UINT8)
    buffers = _rope_slab_storage(r_rows)
    views = pl.make_tuple(source=source, cosine=cosine, sine=sine, aq=aq, asc=asc)
    with pl.section_vector():
        worker = pl.get_block_idx()
        group = worker % 8
        lane = worker // 8
        if padded_rows:
            lanes = pl.get_block_num() * pl.get_subblock_num() // 8
            batch_size = r_rows
            if BM == 256:
                per_lane = (padded + lanes - 1) // lanes
                batches = (per_lane + r_rows - 1) // r_rows
                batch_size = (per_lane + batches - 1) // batches
            first = lane * batch_size
            last = padded
            stride = lanes * batch_size
        else:
            lanes = 8 if BN == 160 else pl.get_block_num() * pl.get_subblock_num() // 8
            first = t * lane // lanes
            last = t * (lane + 1) // lanes
            batch_size = r_rows
            stride = r_rows
        for slab in pl.range(2):
            for row in pl.range(first, last, stride):
                rows = pl.min(batch_size, last - row)
                _rope_quantize_batch(
                    views,
                    buffers,
                    pl.make_tuple(row=row, group=group, slab=slab, rows=rows),
                    t,
                    padded,
                    padded_rows,
                )
            _publish_rope_slab(slab)


@pl.vector_function
def _zero_fill_quantized_tail(q, s, offset: pl.DT_INT64, count: pl.DT_INT64):
    """Zero-fill the tail rows of a quantized tile."""
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    zero = vf.full(0, dtype=pl.DT_UINT8)
    for i in pl.range(count // 256):
        vf.store_align(q, zero, p, offset + i * 256)
    ps = vf.update_mask(count // 32, dtype=pl.DT_UINT8)
    vf.store_align(s, zero, ps, offset // 32)


def _zero_rope_tail(q, scales, valid, rows):
    """Use the upstream whole-tile or slab-tail zero-fill granularity."""
    if BM == 256:
        _zero_fill_quantized_batch_tail(q, scales, valid * 2048, (rows - valid) * 2048)
    else:
        _zero_fill_quantized_tail(q, scales, valid * 2048, (rows - valid) * 2048)


def _zero_rope_batch(q, scales, rows):
    """Preserve upstream's small and large zero-only batch implementations."""
    if BM == 256:
        _zero_fill_quantized_batch(q, scales, rows * 2048)
    else:
        _zero_fill_quantized_tile(q, scales, rows * 2048)


# Matmul


def _accumulate_mx(acc, left, right, left_scale, right_scale, first, last):
    """Keep the native Partial/Final accumulator lifetime unchanged."""
    if first:
        pl.matmul_mx(
            acc, left, right, left_scale, right_scale, phase=pl.AccPhase.Partial
        )
    elif last:
        pl.matmul_mx_acc(
            acc, acc, left, right, left_scale, right_scale, phase=pl.AccPhase.Final
        )
    else:
        pl.matmul_mx_acc(
            acc, acc, left, right, left_scale, right_scale, phase=pl.AccPhase.Partial
        )


def _gemm2_asym_consumer(acc, operands, buffers, ki, inner, output, row, column):
    """Consume a shared Right strip, then drain this M panel's own accumulator."""
    pl.move(buffers.a, operands.a, [0, inner])
    pl.move(buffers.scale_a, operands.scale_a, [0, (ki + inner) // 32])
    if ki == 0 and inner == 0:
        pl.matmul_mx(
            acc,
            buffers.a,
            operands.b,
            buffers.scale_a,
            operands.scale_b,
            phase=pl.AccPhase.Partial,
        )
    elif ki + 256 == 8192 and inner == 128:
        pl.matmul_mx_acc(
            acc,
            acc,
            buffers.a,
            operands.b,
            buffers.scale_a,
            operands.scale_b,
            phase=pl.AccPhase.Final,
        )
        pl.store(output, acc, [row, column], phase=pl.STPhase.Final)
    else:
        pl.matmul_mx_acc(
            acc,
            acc,
            buffers.a,
            operands.b,
            buffers.scale_a,
            operands.scale_b,
            phase=pl.AccPhase.Partial,
        )


def _gemm2_asym_step(accumulators, operands, buffers, ki, inner, output, ni):
    right = buffers.bl[inner // 128]
    right_scale = buffers.bls[inner // 128]
    pl.move(right, operands.tr, [inner, 0])
    pl.move(right_scale, operands.bst, [(ki + inner) // 32, 0])
    _gemm2_asym_consumer(
        accumulators.acc0,
        pl.make_tuple(
            a=operands.at0, scale_a=operands.ast0, b=right, scale_b=right_scale
        ),
        pl.make_tuple(a=buffers.left0, scale_a=buffers.ls0),
        ki,
        inner,
        output,
        0,
        ni,
    )
    _gemm2_asym_consumer(
        accumulators.acc1,
        pl.make_tuple(
            a=operands.at1, scale_a=operands.ast1, b=right, scale_b=right_scale
        ),
        pl.make_tuple(a=buffers.left1, scale_a=buffers.ls1),
        ki,
        inner,
        output,
        256,
        ni,
    )


def _mx_matmul_strip(acc, at, bt, scale_a, scale_b, buffers, ki, k, scale_offset):
    """Issue one L1 strip through the shared, named L0 buffer contract."""
    operands = pl.make_tuple(a=at, b=bt, scale_a=scale_a, scale_b=scale_b)
    l0 = pl.make_tuple(
        a=buffers.al, scale_a=buffers.als, b=buffers.bl, scale_b=buffers.bls
    )
    _mx_matmul_step(acc, operands, l0, ki, k, scale_offset, 0)
    if BK >= 2 * K0:
        _mx_matmul_step(acc, operands, l0, ki, k, scale_offset, K0)


def _mx_matmul_step(acc, operands, buffers, ki, k, sa_ki, inner):
    """One L0 sub-step of the MX matmul: load the A/B strips with their scales, then mac."""
    at = operands.a
    bt = operands.b
    ast = operands.scale_a
    bst = operands.scale_b
    al = buffers.a
    als = buffers.scale_a
    bl = buffers.b
    bls = buffers.scale_b
    left = al[(inner // K0) % (256 // K0)]
    left_scale = als[(inner // K0) % (256 // K0)]
    right = bl[(inner // K0) % (256 // K0)]
    right_scale = bls[(inner // K0) % (256 // K0)]
    # Unit flags overlap M and FIX across output tiles.
    pl.move(left, at, [0, inner])
    pl.move(left_scale, ast, [0, (sa_ki + inner) // 32])
    pl.move(right, bt, [inner, 0])
    pl.move(right_scale, bst, [(ki + inner) // 32, 0])
    _accumulate_mx(
        acc,
        left,
        right,
        left_scale,
        right_scale,
        inner == 0 and ki == 0,
        inner + K0 == BK and ki + BK == k,
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


def _small_gemm2_nstep(acc, operands, buffers, output, mi, ni, ki, ck, col):
    pl.move(buffers.right, operands.bt, [0, col])
    pl.move(buffers.right_s, operands.bst, [ki // 32, col])
    if ki == 0:
        pl.matmul_mx(
            acc,
            operands.left,
            buffers.right,
            operands.left_s,
            buffers.right_s,
            phase=pl.AccPhase.Partial,
        )
    elif ki + ck == 8192:
        pl.matmul_mx_acc(
            acc,
            acc,
            operands.left,
            buffers.right,
            operands.left_s,
            buffers.right_s,
            phase=pl.AccPhase.Final,
        )
        pl.store(output, acc, [mi, ni + col], phase=pl.STPhase.Final)
    else:
        pl.matmul_mx_acc(
            acc,
            acc,
            operands.left,
            buffers.right,
            operands.left_s,
            buffers.right_s,
            phase=pl.AccPhase.Partial,
        )


def _g2_a512_step(acc, operands, buffers, window, inner):
    left = buffers.al[(inner // 128) % 2]
    left_scale = buffers.als[(inner // 128) % 2]
    right = buffers.bl[(inner // 128) % 2]
    right_scale = buffers.bls[(inner // 128) % 2]
    pl.move(left, operands.at, [0, inner])
    pl.move(left_scale, operands.ast, [0, (window + inner) // 32])
    pl.move(right, operands.bt, [inner, 0])
    pl.move(right_scale, operands.bst, [(window + inner) // 32, 0])
    _accumulate_mx(
        acc,
        left,
        right,
        left_scale,
        right_scale,
        window + inner == 0,
        window + inner + 128 == 8192,
    )


def _mx_matmul_offset_step(acc, operands, buffers, ki, k, sa_ki, inner, b_offset):
    """One L0 sub-step of the MX matmul: load the A/B strips with their scales, then mac."""
    left = buffers.al[(inner // K0) % (256 // K0)]
    left_scale = buffers.als[(inner // K0) % (256 // K0)]
    right = buffers.bl[(inner // K0) % (256 // K0)]
    right_scale = buffers.bls[(inner // K0) % (256 // K0)]
    # Unit flags overlap M and FIX across output tiles.
    pl.move(left, operands.at, [0, inner])
    pl.move(left_scale, operands.ast, [0, (sa_ki + inner) // 32])
    pl.move(right, operands.bt, [b_offset + inner, 0])
    pl.move(right_scale, operands.bst, [(ki + inner) // 32, 0])
    _accumulate_mx(
        acc,
        left,
        right,
        left_scale,
        right_scale,
        inner == 0 and ki == 0,
        inner + K0 == BK and ki + BK == k,
    )


def _mid_wide_b_strip(acc, operands, buffers, ki, k, b_offset):
    _mx_matmul_offset_step(
        acc,
        pl.make_tuple(
            at=operands.at, bt=operands.tr, ast=operands.ast, bst=operands.bst
        ),
        pl.make_tuple(al=buffers.al, als=buffers.als, bl=buffers.bl, bls=buffers.bls),
        ki,
        k,
        ki % 2048,
        0,
        b_offset,
    )
    _mx_matmul_offset_step(
        acc,
        pl.make_tuple(
            at=operands.at, bt=operands.tr, ast=operands.ast, bst=operands.bst
        ),
        pl.make_tuple(al=buffers.al, als=buffers.als, bl=buffers.bl, bls=buffers.bls),
        ki,
        k,
        ki % 2048,
        K0,
        b_offset,
    )


def _small_gemm2_pair(
    accumulators, operands, right, output, row, column, k, strip_width
):
    """Consume the same Left panel for the first two 64-column output fragments."""
    _small_gemm2_nstep(
        accumulators.first,
        operands,
        pl.make_tuple(right=right.first, right_s=right.first_scale),
        output,
        row,
        column,
        k,
        strip_width,
        0,
    )
    _small_gemm2_nstep(
        accumulators.second,
        operands,
        pl.make_tuple(right=right.second, right_s=right.second_scale),
        output,
        row,
        column,
        k,
        strip_width,
        64,
    )


# Gemm2 storage


def _gemm2_l0_groups(cube_id, l0_slots):
    """Gemm2 l0 groups."""
    al = _tile_group(
        [BM, K0],
        FP8,
        pl.MemorySpace.Left,
        [0, BM * K0][:l0_slots],
        [cube_id + 6, cube_id + 7][:l0_slots],
    )
    bl = _tile_group(
        [K0, BN],
        FP8,
        pl.MemorySpace.Right,
        [0, BN * K0][:l0_slots],
        [cube_id + 6 + l0_slots, cube_id + 7 + l0_slots][:l0_slots],
    )
    als = _tile_group(
        [BM, K0 // 32],
        E8,
        pl.MemorySpace.ScaleLeft,
        [0, BM * K0 // 16][:l0_slots],
        [cube_id + 6 + 2 * l0_slots, cube_id + 7 + 2 * l0_slots][:l0_slots],
    )
    bls = _tile_group(
        [K0 // 32, BN],
        E8,
        pl.MemorySpace.ScaleRight,
        [0, BN * K0 // 16][:l0_slots],
        [cube_id + 6 + 3 * l0_slots, cube_id + 7 + 3 * l0_slots][:l0_slots],
    )
    return pl.make_tuple(al=al, bl=bl, als=als, bls=bls)


def _small_gemm2_l0_groups(cube_id, ck, n0):
    """Small gemm2 l0 groups."""
    al = _tile_group(
        [BM, ck], FP8, pl.MemorySpace.Left, [0, 32768], [cube_id + 6, cube_id + 7]
    )
    bl = _tile_group(
        [ck, n0], FP8, pl.MemorySpace.Right, [0, 32768], [cube_id + 8, cube_id + 9]
    )
    als = _tile_group(
        [BM, ck // 32],
        E8,
        pl.MemorySpace.ScaleLeft,
        [0, 2048],
        [cube_id + 10, cube_id + 11],
    )
    bls = _tile_group(
        [ck // 32, n0],
        E8,
        pl.MemorySpace.ScaleRight,
        [0, 2048],
        [cube_id + 12, cube_id + 13],
    )
    return pl.make_tuple(al=al, bl=bl, als=als, bls=bls)


def _gemm2_asym_pair_storage():
    """Gemm2 asym pair storage."""
    a1g = _tile_group([128, 256], FP8, pl.MemorySpace.Mat, [172032, 204800], [15, 16])
    # Keep the original first-B prefetch at address 2*256*256.
    bg = _tile_group([160, 256], FP8, pl.MemorySpace.Mat, [131072, 237568], [17, 18])
    s0g = _matrix_group([256, 256], E8, pl.MemorySpace.Mat, [278528], [19], pl.ZZ)
    s1g = _matrix_group([128, 256], E8, pl.MemorySpace.Mat, [344064], [20], pl.ZZ)
    bsg = _matrix_group([256, 160], E8, pl.MemorySpace.Mat, [376832], [21], pl.NN)
    l0g = _tile_group([256, 128], FP8, pl.MemorySpace.Left, [0], [22])
    l1g = _tile_group([128, 128], FP8, pl.MemorySpace.Left, [32768], [23])
    bl = _tile_group([128, 160], FP8, pl.MemorySpace.Right, [0, 20480], [24, 25])
    sl0g = _tile_group([256, 4], E8, pl.MemorySpace.ScaleLeft, [0], [26])
    sl1g = _tile_group([128, 4], E8, pl.MemorySpace.ScaleLeft, [2048], [27])
    bls = _tile_group([4, 160], E8, pl.MemorySpace.ScaleRight, [0, 1280], [28, 29])
    c0g = _tile_group([256, 160], pl.DT_FP32, pl.MemorySpace.Acc, [0], [30])
    c1g = _tile_group([128, 160], pl.DT_FP32, pl.MemorySpace.Acc, [163840], [31])
    transposed = pl.reinterpret(bg, shape=[256, 160], layout=pl.ZN)
    return pl.make_tuple(
        a1g=a1g,
        bg=bg,
        bl=bl,
        bls=bls,
        bsg=bsg,
        c0g=c0g,
        c1g=c1g,
        l0g=l0g,
        l1g=l1g,
        s0g=s0g,
        s1g=s1g,
        sl0g=sl0g,
        sl1g=sl1g,
        transposed=transposed,
    )


def _gemm2_l1_groups(cube_id, k, width, slots, b_id_offset):
    """Allocate the original B ring and full-K scales at their exact addresses."""
    bg = _tile_group(
        [BN, width],
        FP8,
        pl.MemorySpace.Mat,
        [
            2 * BM * width,
            (2 * BM + BN) * width,
            (2 * BM + 2 * BN) * width,
            (2 * BM + 3 * BN) * width,
        ][:slots],
        [
            cube_id + b_id_offset,
            cube_id + b_id_offset + 1,
            cube_id + b_id_offset + 2,
            cube_id + b_id_offset + 3,
        ][:slots],
    )
    scale_base = (2 * BM + slots * BN) * width
    asg = _matrix_group(
        [BM, k // 32], E8, pl.MemorySpace.Mat, [scale_base], [cube_id + 4], pl.ZZ
    )
    bsg = _matrix_group(
        [k // 32, BN],
        E8,
        pl.MemorySpace.Mat,
        [scale_base + BM * (k // 32)],
        [cube_id + 5],
        pl.NN,
    )
    return pl.make_tuple(bg=bg, asg=asg, bsg=bsg)


def _gemm2_storage(cube_id, k, l0_slots, acc_slots, width, b_slots, b_id_offset):
    """Share the common G2 buffer contract; geometry remains profile-controlled."""
    bg, asg, bsg = _gemm2_l1_groups(cube_id, k, width, b_slots, b_id_offset)
    al, bl, als, bls = _gemm2_l0_groups(cube_id, l0_slots)
    accg = _tile_group(
        [BM, BN],
        pl.DT_FP32,
        pl.MemorySpace.Acc,
        [0, 4 * BM * BN][:acc_slots],
        [cube_id + 6 + 4 * l0_slots, cube_id + 7 + 4 * l0_slots][:acc_slots],
    )
    b_transposed = pl.reinterpret(bg, shape=[width, BN], layout=pl.ZN)
    return pl.make_tuple(
        accg=accg,
        al=al,
        als=als,
        asg=asg,
        b_transposed=b_transposed,
        bg=bg,
        bl=bl,
        bls=bls,
        bsg=bsg,
    )


def _small_gemm2_storage(cube_id, ck, n0, nmain, columns):
    """Keep small-M 160/192-column geometry and compound Acc fragments."""
    ag = _tile_group(
        [BM, ck], FP8, pl.MemorySpace.Mat, [0, BM * ck], [cube_id, cube_id + 1]
    )
    bg = _tile_group(
        [columns, ck],
        FP8,
        pl.MemorySpace.Mat,
        [2 * BM * ck, (2 * BM + columns) * ck],
        [cube_id + 2, cube_id + 3],
    )
    asg = _matrix_group(
        [BM, 256],
        E8,
        pl.MemorySpace.Mat,
        [2 * (BM + columns) * ck],
        [cube_id + 4],
        pl.ZZ,
    )
    bsg = _matrix_group(
        [256, columns],
        E8,
        pl.MemorySpace.Mat,
        [2 * (BM + columns) * ck + BM * 256],
        [cube_id + 5],
        pl.NN,
    )
    al, bl, als, bls = _small_gemm2_l0_groups(cube_id, ck, n0)
    accg = _tile_group(
        [BM, n0],
        pl.DT_FP32,
        pl.MemorySpace.Acc,
        [0, 4 * BM * n0, 8 * BM * n0][:nmain],
        [cube_id + 14, cube_id + 15, cube_id + 16][:nmain],
    )
    btr = pl.reinterpret(bg, shape=[ck, columns], layout=pl.ZN)
    pl.set_validshape(btr, [ck, columns])
    return pl.make_tuple(
        accg=accg,
        ag=ag,
        al=al,
        als=als,
        asg=asg,
        bg=bg,
        bl=bl,
        bls=bls,
        bsg=bsg,
        btr=btr,
    )


# Gemm2 steps


def _gemm2_asym_pair_kstep(state, current):
    """Gemm2 asym pair kstep."""
    at0 = state.ag[(current.ki // 256) % 2]
    at1 = state.buffers.a1g[(current.ki // 256) % 2]
    bt = state.buffers.bg[(current.ki // 256) % 2]
    tr = state.buffers.transposed[(current.ki // 256) % 2]
    pl.load(at0, state.inputs.a, [0, current.ki])
    pl.load(at1, state.inputs.a, [256, current.ki])
    if current.work != pl.get_block_idx() or current.ki >= 512:
        pl.load(bt, state.inputs.b, [current.ni, current.ki])
    strip_accumulators = pl.make_tuple(acc0=current.acc0, acc1=current.acc1)
    strip_operands = pl.make_tuple(
        at0=at0, at1=at1, tr=tr, ast0=current.ast0, ast1=current.ast1, bst=current.bst
    )
    strip_buffers = pl.make_tuple(
        left0=current.left0,
        left1=current.left1,
        ls0=current.ls0,
        ls1=current.ls1,
        bl=state.buffers.bl,
        bls=state.buffers.bls,
    )
    _gemm2_asym_step(
        strip_accumulators,
        strip_operands,
        strip_buffers,
        current.ki,
        0,
        state.inputs.output,
        current.ni,
    )
    _gemm2_asym_step(
        strip_accumulators,
        strip_operands,
        strip_buffers,
        current.ki,
        128,
        state.inputs.output,
        current.ni,
    )


def _gemm2_paired_tail_kstep(state, current):
    """Gemm2 paired tail kstep."""
    if current.window + 1024 < GEMM2_K:
        future = state.buffers.bg[(current.window // 512 + 2) % state.bslots]
        pl.load(future, state.inputs.b, [current.ni, current.window + 1024])
    tr = state.buffers.b_transposed[current.window // 512 % state.bslots]
    at = state.ag[current.window // 512 % 2]
    strip_operands = pl.make_tuple(at=at, bt=tr, ast=current.ast, bst=current.bst)
    strip_buffers = pl.make_tuple(
        al=state.buffers.al,
        als=state.buffers.als,
        bl=state.buffers.bl,
        bls=state.buffers.bls,
    )
    _g2_a512_step(current.acc, strip_operands, strip_buffers, current.window, 0)
    _g2_a512_step(current.acc, strip_operands, strip_buffers, current.window, 128)
    _g2_a512_step(current.acc, strip_operands, strip_buffers, current.window, 256)
    _g2_a512_step(current.acc, strip_operands, strip_buffers, current.window, 384)
    if current.window + 1024 < GEMM2_K:
        pl.load(at, state.inputs.a, [current.mi, current.window + 1024])


def _gemm2_prefetch_strip(state, current, distance):
    """Advance B by its profile's exact distance; A stays two strips ahead."""
    if current.ki + distance * BK < GEMM2_K:
        future = state.buffers.bg[(current.ki // BK + distance) % 4]
        pl.load(future, state.inputs.b, [current.ni, current.ki + distance * BK])
    at = state.ag[(current.ki // BK) % 2]
    bt = state.buffers.b_transposed[(current.ki // BK) % 4]
    _gemm2_issue_strip(state, current, at, bt)
    if current.ki + 2 * BK < GEMM2_K:
        pl.load(at, state.inputs.a, [current.mi, current.ki + 2 * BK])


def _gemm2_small_28_kstep(state, current):
    """Gemm2 small 28 kstep."""
    slot = (current.ki // state.ck) % 2
    at = state.ag[slot]
    bt = state.buffers.bg[slot]
    trans = state.buffers.btr[slot]
    pl.load(at, state.inputs.a, [current.mi, current.ki])
    if current.work != current.core or current.ki >= 2 * state.ck:
        pl.load(bt, state.inputs.b, [current.ni, current.ki])
    lslot = (current.ki // state.ck) % 2
    _small_gemm2_pair_step(state, current, at, trans, lslot)


def _gemm2_tile_kstep(state, current):
    """Gemm2 tile kstep."""
    at = state.ag.next()
    bt = state.buffers.bg[(current.ki // BK) % 2]
    pl.load(at, state.inputs.a, [current.mi, current.ki])
    if current.work != pl.get_block_idx() or current.ki != 0:
        pl.load(bt, state.inputs.b, [current.ni, current.ki])
    bt_transposed = state.buffers.b_transposed[(current.ki // BK) % 2]
    _gemm2_issue_strip(state, current, at, bt_transposed)


def _small_gemm2_kstep(state, current):
    """Small gemm2 kstep."""
    slot = current.ki // state.ck % 2
    at = state.ag[slot]
    bt = state.buffers.bg[slot]
    trans = state.buffers.btr[slot]
    pl.load(at, state.inputs.a, [current.mi, current.ki])
    if current.ki >= 2 * state.ck:
        pl.load(bt, state.inputs.b, [current.ni, current.ki])
    lslot = current.ki // state.ck % 2
    _small_gemm2_pair_step(state, current, at, trans, lslot)


def _gemm2_asym_pair_work_tile(state, current):
    """Gemm2 asym pair work tile."""
    ni = current.work * 160
    tail = state.t - 256
    pl.set_validshape(state.ag, [256, 256])
    ast0 = state.buffers.s0g.current()
    ast1 = state.buffers.s1g.current()
    bst = state.buffers.bsg.current()
    left0 = state.buffers.l0g.current()
    left1 = state.buffers.l1g.current()
    ls0 = state.buffers.sl0g.current()
    ls1 = state.buffers.sl1g.current()
    acc0 = state.buffers.c0g.current()
    acc1 = state.buffers.c1g.current()
    pl.set_validshape(state.buffers.a1g, [tail, 256])
    pl.set_validshape(ast1, [tail, 256])
    pl.set_validshape(left1, [tail, 128])
    pl.set_validshape(ls1, [tail, 4])
    pl.set_validshape(acc1, [tail, 160])
    pl.load(ast0, state.inputs.sa, [0, 0, 0])
    pl.load(ast1, state.inputs.sa, [256, 0, 0])
    if current.work != pl.get_block_idx():
        pl.load(bst, state.inputs.sb, [ni, 0, 0], order=[1, 0])
    for ki in pl.range(0, 8192, 256):
        _gemm2_asym_pair_kstep(
            state,
            pl.make_tuple(
                ki=ki,
                work=current.work,
                ni=ni,
                acc0=acc0,
                acc1=acc1,
                ast0=ast0,
                ast1=ast1,
                bst=bst,
                left0=left0,
                left1=left1,
                ls0=ls0,
                ls1=ls1,
            ),
        )


def _gemm2_n192_28_work_tile(state, current):
    """Preserve this profile's M/N work ownership."""
    work = current.mwork * current.nn + current.nwork
    mi = current.mwork * BM
    ni = current.nwork * BN
    mv = pl.min(BM, state.t - mi)
    nv = pl.min(BN, OUT_FEATURES - ni)
    _gemm2_compute_tile(state, work, mi, ni, mv, nv, 2)


def _gemm2_prefetch4_work_tile(state, current):
    """Preserve this profile's M/N work ownership."""
    work = current.work
    mi = work // current.nn % current.nm * BM
    ni = work % current.nn * BN
    mv = pl.min(BM, state.t - mi)
    nv = pl.min(BN, OUT_FEATURES - ni)
    _gemm2_compute_tile(state, work, mi, ni, mv, nv, 2)


def _gemm2_small_28_work_tile(state, current):
    """Gemm2 small 28 work tile."""
    mi = current.work // current.nn * BM
    ni = current.work % current.nn * BN
    mv = pl.min(BM, state.t - mi)
    nv = pl.min(BN, 5120 - ni)
    pl.set_validshape(state.buffers.bg, [nv, state.ck])
    pl.set_validshape(state.buffers.btr, [state.ck, nv])
    pl.set_validshape(current.bst, [256, nv])
    acc2 = state.buffers.accg[2]
    pl.set_validshape(acc2, [mv, state.n0])
    ast = state.buffers.asg.current()
    acc0 = state.buffers.accg[0]
    pl.set_validshape(acc0, [mv, state.n0])
    acc1 = state.buffers.accg[1]
    pl.set_validshape(acc1, [mv, state.n0])
    pl.set_validshape(state.ag, [mv, state.ck])
    pl.set_validshape(ast, [mv, 256])
    pl.set_validshape(state.buffers.al, [mv, state.ck])
    pl.set_validshape(state.buffers.als, [mv, state.ck // 32])
    pl.load(ast, state.inputs.sa, [mi, 0, 0])
    if current.work != current.core:
        pl.load(current.bst, state.inputs.sb, [ni, 0, 0], order=[1, 0])
    for ki in pl.range(0, 8192, state.ck):
        _gemm2_small_28_kstep(
            state,
            pl.make_tuple(
                ki=ki,
                mi=mi,
                work=current.work,
                core=current.core,
                ni=ni,
                ast=ast,
                bst=current.bst,
                acc0=acc0,
                acc1=acc1,
                nv=nv,
                acc2=acc2,
            ),
        )


def _gemm2_tile_work_tile(state, current):
    """Preserve this profile's M/N work ownership."""
    work = current.work
    mi = work // current.nn % current.nm * BM
    ni = work % current.nn * BN
    mv = pl.min(BM, state.t - mi)
    nv = pl.min(BN, OUT_FEATURES - ni)
    _gemm2_compute_tile(state, work, mi, ni, mv, nv, 0)


def _gemm2_issue_strip(state, current, at, bt):
    """Share the MX issue operation across the G2 memory strategies."""
    _mx_matmul_strip(
        current.acc,
        at,
        bt,
        current.ast,
        current.bst,
        state.buffers,
        current.ki,
        GEMM2_K,
        current.ki,
    )


def _configure_gemm2_tile(acc, scale_a, scale_b, state, row, rows, columns):
    """Set the original valid extents and load the selected A-scale row."""
    pl.set_validshape(acc, [rows, columns])
    pl.set_validshape(scale_a, [rows, GEMM2_K // 32])
    pl.set_validshape(scale_b, [GEMM2_K // 32, columns])
    pl.set_validshape(state.ag, [rows, BK])
    pl.set_validshape(state.buffers.bg, [columns, BK])
    pl.set_validshape(state.buffers.b_transposed, [BK, columns])
    pl.set_validshape(state.buffers.al, [rows, K0])
    pl.set_validshape(state.buffers.als, [rows, K0 // 32])
    pl.set_validshape(state.buffers.bl, [K0, columns])
    pl.set_validshape(state.buffers.bls, [K0 // 32, columns])
    pl.load(scale_a, state.inputs.sa, [row, 0, 0])


def _gemm2_compute_tile(state, work, mi, ni, rows, columns, mode):
    """Keep generic, two-ahead and three-ahead memory issue order distinct."""
    acc = state.buffers.accg.next()
    ast = state.buffers.asg.current()
    bst = state.buffers.bsg.current()
    _configure_gemm2_tile(acc, ast, bst, state, mi, rows, columns)
    if mode == 0:
        pl.load(bst, state.inputs.sb, [ni, 0, 0], order=[1, 0])
    else:
        if mode == 2 and work != pl.get_block_idx():
            first0 = state.buffers.bg[0]
            first1 = state.buffers.bg[1]
            pl.load(first0, state.inputs.b, [ni, 0])
            pl.load(first1, state.inputs.b, [ni, BK])
        first_a = state.ag[0]
        second_a = state.ag[1]
        pl.load(first_a, state.inputs.a, [mi, 0])
        pl.load(second_a, state.inputs.a, [mi, BK])
    for ki in pl.range(0, GEMM2_K, BK):
        current = pl.make_tuple(
            acc=acc, ast=ast, bst=bst, mi=mi, ni=ni, ki=ki, work=work
        )
        if mode == 0:
            _gemm2_tile_kstep(state, current)
        else:
            _gemm2_prefetch_strip(state, current, mode)
    pl.store(state.inputs.output, acc, [mi, ni], phase=pl.STPhase.Final)


def _small_gemm2_pair_step(state, current, at, transposed, slot):
    """Move one Left panel, then consume it for the first two N fragments."""
    left = state.buffers.al[slot]
    left_s = state.buffers.als[slot]
    right0 = state.buffers.bl[slot]
    right_s0 = state.buffers.bls[slot]
    right1 = state.buffers.bl[1 - slot]
    right_s1 = state.buffers.bls[1 - slot]
    if BN == 160:
        tail_right = current.tail_blg[slot]
        tail_scale = current.tail_blsg[slot]
    else:
        tail_right = right0
        tail_scale = right_s0
    pl.move(left, at)
    pl.move(left_s, current.ast, [0, current.ki // 32])
    operands = pl.make_tuple(left=left, left_s=left_s, bt=transposed, bst=current.bst)
    _small_gemm2_pair(
        pl.make_tuple(first=current.acc0, second=current.acc1),
        operands,
        pl.make_tuple(
            first=right0, first_scale=right_s0, second=right1, second_scale=right_s1
        ),
        state.inputs.output,
        current.mi,
        current.ni,
        current.ki,
        state.ck,
    )
    _small_gemm2_last_fragment(state, current, operands, tail_right, tail_scale)


def _small_gemm2_last_fragment(state, current, operands, right, scale):
    """Keep all three consumers in the same strip lifetime; do not return live tiles."""
    if BN == 160:
        _small_gemm2_nstep(
            current.tail,
            operands,
            pl.make_tuple(right=right, right_s=scale),
            state.inputs.output,
            current.mi,
            current.ni,
            current.ki,
            state.ck,
            128,
        )
    elif current.nv > 128:
        _small_gemm2_nstep(
            current.acc2,
            operands,
            pl.make_tuple(right=right, right_s=scale),
            state.inputs.output,
            current.mi,
            current.ni,
            current.ki,
            state.ck,
            128,
        )


# Gemm1


def _configure_gemm1_tile(
    acc, scale_a, scale_b, buffers, rows, columns, scale_columns, b_width
):
    """Set valid extents without changing a profile's physical row pitches."""
    pl.set_validshape(acc, [rows, columns])
    pl.set_validshape(scale_a, [rows, scale_columns])
    pl.set_validshape(buffers.ag, [rows, BK])
    pl.set_validshape(buffers.al, [rows, K0])
    pl.set_validshape(buffers.als, [rows, K0 // 32])
    pl.set_validshape(buffers.bg, [columns, b_width])
    pl.set_validshape(buffers.b_transposed, [b_width, columns])
    pl.set_validshape(scale_b, [128, columns])
    pl.set_validshape(buffers.bl, [K0, columns])
    pl.set_validshape(buffers.bls, [K0 // 32, columns])


def _prefetch_gemm2_after_gemm1(inputs, buffers, nm):
    """Prefetch gemm2 after gemm1."""
    second_nn = (5120 + BN - 1) // BN
    core = pl.get_block_idx()
    if core < nm * second_nn:
        first = buffers.wbg[0]
        col = core % second_nn * BN
        pl.set_validshape(first, [pl.min(BN, 5120 - col), BK])
        pl.load(first, inputs.wb, [col, 0])


def _gemm1_tile(work, t, padded, inputs, buffers, accg, shared):
    """GEMM1 tile (group, m, n): acc = a[group, m] @ woa[group, n]^T, drained to the C->V ring."""
    nm, nn, group, mi, ni = _gemm1_tile_coordinates(work, t)
    mv = pl.min(BM, padded - mi)
    nv = pl.min(BN1, 1024 - ni)
    acc = accg.next()
    ast = buffers.asg.current()
    bst = buffers.bsg.current()
    _configure_gemm1_tile(acc, ast, bst, buffers, mv, nv, 64 if BN1 == 256 else 128, BK)
    if BN1 != 256:
        pl.load(ast, inputs.sa, [group * padded + mi, 0, 0])
    if work != pl.get_block_idx():
        pl.load(bst, inputs.sb, [group * 1024 + ni, 0, 0], order=[1, 0])
    strip = pl.make_tuple(
        scale_a=ast, scale_b=bst, a_row=group * padded + mi, b_row=group * 1024 + ni
    )
    for ki in pl.range(0, 4096, BK):
        if BN1 == 256 and ki % 2048 == 0:
            # All 32 AICs have a first tile; later tiles reuse immutable slabs.
            if work == pl.get_block_idx():
                _wait_rope_slab(ki)
            pl.load(ast, inputs.sa, [group * padded + mi, ki // 64, 0])
        _gemm1_strip(
            acc, inputs, buffers, strip, work, ki, ki % 2048 if BN1 == 256 else ki, 0
        )
    _drain_gemm1(acc, shared, mv // 2, nv)
    if work + pl.get_block_num() >= 8 * nm * nn:
        _prefetch_gemm2_after_gemm1(inputs, buffers, nm)


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


def _gemm1_small_valid_rows(work, t, padded, inputs, buffers, accg, whole, shared):
    group = work // 4
    ni = work % 4 * 256
    ast = buffers.asg.current()
    bst = buffers.bsg.current()
    # ceil16(valid T)==MCAP preserves every compound Acc fragment's pitch.
    pl.set_validshape(buffers.ag, [t, 512])
    pl.set_validshape(ast, [t, 64])
    pl.set_validshape(buffers.al, [t, 512])
    pl.set_validshape(buffers.als, [t, 16])
    pl.set_validshape(accg, [t, 64])
    acc0 = accg[0]
    acc1 = accg[1]
    acc2 = accg[2]
    acc3 = accg[3]
    right0 = buffers.bl[0]
    right1 = buffers.bl[1]
    right_s0 = buffers.bls[0]
    right_s1 = buffers.bls[1]
    for ki in pl.range(0, 4096, 512):
        if ki % 2048 == 0:
            _wait_rope_slab(ki)
            pl.load(ast, inputs.sa, [group * padded, ki // 64, 0])
        slot = (ki // 512) % 2
        at = buffers.ag[slot]
        bt = buffers.bg[slot]
        tr = buffers.b_transposed[slot]
        left = buffers.al[slot]
        left_s = buffers.als[slot]
        pl.load(at, inputs.a, [group * padded, ki])
        if ki >= 1024:
            pl.load(bt, inputs.b, [group * 1024 + ni, ki])
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


def _gemm1_large_full_ascale(work, t, padded, inputs, buffers, accg, shared):
    """GEMM1 tile (group, m, n): acc = a[group, m] @ woa[group, n]^T, drained to the C->V ring."""
    nm, nn, group, mi, ni, mv, nv = _gemm1_large_coordinates(work, t)
    acc = accg.next()
    ast = buffers.asg.current()
    bst = buffers.bsg.current()
    _configure_gemm1_tile(acc, ast, bst, buffers, mv, nv, 128, BK)
    if work != pl.get_block_idx():
        # Every slab was globally published before this core's first M completed.
        # A single full row-scale load replaces two identical-stride partial loads.
        pl.load(ast, inputs.sa, [group * padded + mi, 0, 0])
    strip = pl.make_tuple(
        scale_a=ast, scale_b=bst, a_row=group * padded + mi, b_row=group * 1024 + ni
    )
    for ki in pl.range(0, 4096, BK):
        _load_gemm1_full_scales(work, ki, ast, inputs.sa, strip.a_row, mv)
        _gemm1_strip(acc, inputs, buffers, strip, work, ki, ki, 0)
    _drain_gemm1(acc, shared, (mv + 1) // 2, nv)
    if work + pl.get_block_num() >= 8 * nm * nn:
        _prefetch_gemm2_after_gemm1(inputs, buffers, nm)


def _gemm1_single_tile(work, t, padded, inputs, buffers, accg, shared):
    """GEMM1 tile (group, m, n): acc = a[group, m] @ woa[group, n]^T, drained to the C->V ring."""
    group = work // 4
    mi = 0
    ni = work % 4 * 256
    mv = t
    nv = 256
    acc = accg.next()
    ast = buffers.asg.current()
    bst = buffers.bsg.current()
    _configure_gemm1_tile(acc, ast, bst, buffers, mv, nv, 64 if BN1 == 256 else 128, BK)
    strip = pl.make_tuple(
        scale_a=ast, scale_b=bst, a_row=group * padded + mi, b_row=group * 1024 + ni
    )
    for ki in pl.range(0, 4096, BK):
        if BN1 == 256 and ki % 2048 == 0:
            _wait_rope_slab(ki)
            pl.load(ast, inputs.sa, [group * padded + mi, ki // 64, 0])
        _gemm1_strip(
            acc, inputs, buffers, strip, work, ki, ki % 2048 if BN1 == 256 else ki, 1
        )
    _drain_gemm1(acc, shared, (mv + 1) // 2, nv)
    first = buffers.wbg[0]
    core = pl.get_block_idx()
    pl.set_validshape(first, [BN, BK])
    pl.load(first, inputs.wb, [core * BN, 0])


def _prefetch_gemm1_single_tile(b, sb, bg, bsg):
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


def _prefetch_mid_wide_b(b, sb, bg, bsg):
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


def _gemm1_mid_wide_b(work, t, padded, inputs, buffers, accg, shared):
    """GEMM1 tile (group, m, n): acc = a[group, m] @ woa[group, n]^T, drained to the C->V ring."""
    group = work // 4
    mi = 0
    ni = work % 4 * 256
    mv = t
    nv = 256
    acc = accg.next()
    ast = buffers.asg.current()
    bst = buffers.bsg.current()
    _configure_gemm1_tile(
        acc, ast, bst, buffers, mv, nv, 64 if BN1 == 256 else 128, 512
    )
    for window in pl.range(0, 4096, 512):
        if window % 2048 == 0:
            if window == 0:
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=3)
            else:
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=4)
            pl.load(ast, inputs.sa, [group * padded + mi, window // 64, 0])
        first_a = buffers.ag[0]
        pl.load(first_a, inputs.a, [group * padded + mi, window])
        # Current A was issued first; prefetch the immutable B panel two ahead.
        if window + 1024 < 4096:
            future_b = buffers.bg[(window // 512 + 2) % 3]
            pl.load(future_b, inputs.b, [group * 1024 + ni, window + 1024])
        tr = buffers.b_transposed[window // 512 % 3]
        _mid_wide_b_strip(
            acc,
            pl.make_tuple(at=first_a, tr=tr, ast=ast, bst=bst),
            pl.make_tuple(
                al=buffers.al, als=buffers.als, bl=buffers.bl, bls=buffers.bls
            ),
            window,
            4096,
            0,
        )
        second_a = buffers.ag[1]
        pl.load(second_a, inputs.a, [group * padded + mi, window + BK])
        _mid_wide_b_strip(
            acc,
            pl.make_tuple(at=second_a, tr=tr, ast=ast, bst=bst),
            pl.make_tuple(
                al=buffers.al, als=buffers.als, bl=buffers.bl, bls=buffers.bls
            ),
            window + BK,
            4096,
            BK,
        )
    _drain_gemm1(acc, shared, (mv + 1) // 2, nv)


def _gemm1_large_full_ascale_28(work, t, padded, inputs, buffers, accg, shared):
    """GEMM1 tile (group, m, n): acc = a[group, m] @ woa[group, n]^T, drained to the C->V ring."""
    nm, nn, group, mi, ni, mv, nv = _gemm1_large_coordinates(work, t)
    acc = accg.next()
    ast = buffers.asg.current()
    bst = buffers.bsg.current()
    _configure_gemm1_tile(acc, ast, bst, buffers, mv, nv, 128, BK)
    if work != pl.get_block_idx():
        pl.load(bst, inputs.sb, [group * 1024 + ni, 0, 0], order=[1, 0])
        # Every slab was globally published before this core's first M completed.
        # A single full row-scale load replaces two identical-stride partial loads.
        pl.load(ast, inputs.sa, [group * padded + mi, 0, 0])
    strip = pl.make_tuple(
        scale_a=ast, scale_b=bst, a_row=group * padded + mi, b_row=group * 1024 + ni
    )
    for ki in pl.range(0, 4096, BK):
        _load_gemm1_full_scales(work, ki, ast, inputs.sa, strip.a_row, mv)
        _gemm1_strip(acc, inputs, buffers, strip, work, ki, ki, 2)
    _drain_gemm1(acc, shared, (mv + 1) // 2, nv)
    if work + pl.get_block_num() >= 8 * nm * nn:
        second_nn = (5120 + BN - 1) // BN
        core = pl.get_block_idx()
        if core < second_nn:
            first = buffers.wbg[0]
            col = core % second_nn * BN
            pl.set_validshape(first, [pl.min(BN, 5120 - col), BK])
            pl.load(first, inputs.wb, [col, 0])


def _prefetch_gemm1_weights_28(b, sb, bg, bsg, t):
    """Load GEMM1's first B tile (and its scales) into L1 before the task loop."""
    work = pl.get_block_idx()
    nm = (t + BM - 1) // BM
    nn = (1024 + BN1 - 1) // BN1
    if work < 8 * nm * nn:
        group = (work // nn) % 8
        col = work % nn * BN1
        nv = pl.min(BN1, 1024 - col)
        first = bg[0]
        second = bg[1]
        scales = bsg.current()
        pl.set_validshape(first, [nv, BK])
        pl.set_validshape(second, [nv, BK])
        pl.set_validshape(scales, [128, nv])
        pl.load(first, b, [group * 1024 + col, 0])
        pl.load(second, b, [group * 1024 + col, BK])
        pl.load(scales, sb, [group * 1024 + col, 0, 0], order=[1, 0])


def _gemm1_small_28(work, t, padded, inputs, buffers, accg, whole, shared):
    group = work // 8
    ni = work % 8 * 128
    ast = buffers.asg.current()
    bst = buffers.bsg.current()
    if work != pl.get_block_idx():
        pl.load(bst, inputs.sb, [group * 1024 + ni, 0, 0], order=[1, 0])
    # ceil16(valid T)==MCAP preserves every compound Acc fragment's pitch.
    pl.set_validshape(buffers.ag, [t, 512])
    pl.set_validshape(ast, [t, 64])
    pl.set_validshape(buffers.al, [t, 512])
    pl.set_validshape(buffers.als, [t, 16])
    pl.set_validshape(accg, [t, 64])
    acc0 = accg[0]
    acc1 = accg[1]
    right0 = buffers.bl[0]
    right1 = buffers.bl[1]
    right_s0 = buffers.bls[0]
    right_s1 = buffers.bls[1]
    for ki in pl.range(0, 4096, 512):
        if ki % 2048 == 0:
            if work == pl.get_block_idx():
                _wait_rope_slab(ki)
            pl.load(ast, inputs.sa, [group * padded, ki // 64, 0])
        slot = (ki // 512) % 2
        at = buffers.ag[slot]
        bt = buffers.bg[slot]
        tr = buffers.b_transposed[slot]
        left = buffers.al[slot]
        left_s = buffers.als[slot]
        pl.load(at, inputs.a, [group * padded, ki])
        if work != pl.get_block_idx() or ki >= 1024:
            pl.load(bt, inputs.b, [group * 1024 + ni, ki])
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


def _prefetch_gemm1_profile(inputs, buffers, t):
    """Issue the original immutable-weight prefetch for the selected profile."""
    if Algo == 3:
        if BM <= 128:
            _prefetch_mid_wide_b(inputs.b, inputs.sb, buffers.bg, buffers.bsg)
        else:
            _prefetch_gemm1_single_tile(inputs.b, inputs.sb, buffers.bg, buffers.bsg)
    elif (Algo == 1 or Algo == 4) and BN == 192:
        _prefetch_gemm1_weights_28(inputs.b, inputs.sb, buffers.bg, buffers.bsg, t)
    else:
        _prefetch_gemm1_weights(inputs.b, inputs.sb, buffers.bg, buffers.bsg, t)


@pl.pipeline.stage
def _gemm1_profile(work, t, padded, inputs, buffers, accg, whole, shared):
    """The Cube actor; preserve each profile's original tile implementation."""
    if Algo == 1:
        if BN == 160:
            _gemm1_small_valid_rows(
                work, t, padded, inputs, buffers, accg, whole, shared
            )
        else:
            _gemm1_small_28(work, t, padded, inputs, buffers, accg, whole, shared)
    elif Algo == 3:
        if BM <= 128:
            _gemm1_mid_wide_b(work, t, padded, inputs, buffers, accg, shared)
        else:
            _gemm1_single_tile(work, t, padded, inputs, buffers, accg, shared)
    elif Algo == 4:
        if BN == 192:
            _gemm1_large_full_ascale_28(work, t, padded, inputs, buffers, accg, shared)
        else:
            _gemm1_large_full_ascale(work, t, padded, inputs, buffers, accg, shared)
    else:
        _gemm1_tile(work, t, padded, inputs, buffers, accg, shared)


def _gemm1_strip(acc, inputs, buffers, strip, work, ki, scale_offset, reload_policy):
    """Load the current G1 strip, then issue its unchanged MX operation."""
    at = buffers.ag.next()
    bt = buffers.bg[(ki // BK) % 2]
    pl.load(at, inputs.a, [strip.a_row, ki])
    if reload_policy == 1:
        if ki >= (2 if BN1 == 256 else 1) * BK:
            pl.load(bt, inputs.b, [strip.b_row, ki])
    elif reload_policy == 2:
        if work != pl.get_block_idx() or ki >= 2 * BK:
            pl.load(bt, inputs.b, [strip.b_row, ki])
    else:
        if work != pl.get_block_idx() or ki >= (2 if BN1 == 256 else 1) * BK:
            pl.load(bt, inputs.b, [strip.b_row, ki])
    tr = buffers.b_transposed[(ki // BK) % 2]
    _mx_matmul_strip(
        acc, at, tr, strip.scale_a, strip.scale_b, buffers, ki, 4096, scale_offset
    )


def _load_gemm1_full_scales(work, ki, scale, source, row, rows):
    """Keep the first-tile partial/full scale-row publication protocol."""
    if work == pl.get_block_idx() and ki % 2048 == 0:
        if ki == 0:
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=3)
            pl.set_validshape(scale, [rows, 64])
        else:
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=4)
            pl.set_validshape(scale, [rows, 128])
        pl.load(scale, source, [row, 0, 0])


# Gemm2


def _gemm2_asym_pair(inputs, t, padded, ag0):
    """Host dispatch guarantees 257<=T<=384, BM256/BK256/K0128/N160.

    Original M0=256 and M1=T-256 geometry, K order and N160 ownership stay
    unchanged. B and ScaleB remain read-only in each Right slot for both M
    consumers before overwrite. L1=417792B with two L1 slots per M panel;
    Left=49152B, Right=40960B, ScaleLeft=3072B, ScaleRight=2560B.
    Disjoint Acc0[256,160] and Acc1[128,160] total 245760B. Each Final
    matmul is followed by its own Final store. G1 is unchanged.
    """

    # G1's AG0/1 (IDs0/1, shapes256x256 at0/65536) are fully
    # drained by its Cube bar_all. Reuse that exact group for M0's two
    # L1 slots; this keeps all 32 mutex IDs within range and preserves
    # the original per-M double-buffer distance.
    storage = _gemm2_asym_pair_storage()
    with pl.section_cube():
        _gemm2_asym_pair_cube(
            pl.make_tuple(inputs=inputs, buffers=storage, t=t, ag=ag0)
        )


def _gemm2_tile(inputs, t, padded):
    """Run generic GEMM2 tiles with the continued Cube mutex namespace."""
    k = 8192

    # The Cube pipe barrier drains the first projection before address reuse.
    # Continue its Cube mutex numbering; L0 scale planes follow data address >> 4.
    l0_slots = 256 // K0
    cube_id = 6 + 4 * l0_slots + min(2, 65536 // (BM * BN1))
    acc_slots = min(2, 65536 // (BM * BN))
    ag = _tile_group(
        [BM, BK], FP8, pl.MemorySpace.Mat, [0, BM * BK], [cube_id, cube_id + 1]
    )
    storage = _gemm2_storage(cube_id, k, l0_slots, acc_slots, BK, 2, 2)
    with pl.section_cube():
        _gemm2_tile_cube(pl.make_tuple(inputs=inputs, buffers=storage, t=t, ag=ag))


def _small_gemm2(inputs, t, padded):
    ck = 512
    n0 = 64
    nmain = 2
    cube_id = 6 + 4 * (256 // K0) + min(2, 65536 // (BM * BN1))

    storage = _small_gemm2_storage(cube_id, ck, n0, nmain, 160)
    tailg = _tile_group(
        [BM, 32], pl.DT_FP32, pl.MemorySpace.Acc, [4 * BM * n0 * nmain], [cube_id + 16]
    )
    with pl.section_cube():
        _small_gemm2_cube(
            pl.make_tuple(
                inputs=inputs,
                buffers=storage,
                ck=ck,
                n0=n0,
                t=t,
                ag=storage.ag,
                tailg=tailg,
            )
        )


def _gemm2_paired_tail(inputs, t, padded):
    """Project paired output-tail tiles with the existing profile layout."""
    k = 8192

    l0_slots = 256 // K0
    cube_id = 6 + 4 * l0_slots + min(2, 65536 // (BM * BN1))
    acc_slots = min(2, 65536 // (BM * BN))
    bslots = 3
    # G1 has drained at the root Cube barrier; its A aliases are now retired.
    a512 = _tile_group([BM, 512], FP8, pl.MemorySpace.Mat, [0, BM * 512], [0, 1])
    storage = _gemm2_storage(cube_id, k, l0_slots, acc_slots, 512, bslots, 0)
    with pl.section_cube():
        _gemm2_paired_tail_cube(
            pl.make_tuple(inputs=inputs, buffers=storage, t=t, bslots=bslots, ag=a512)
        )


def _gemm2_prefetch4(inputs, t, padded, ag):
    """Use the four-slot B ring with two-strip weight lookahead."""
    k = 8192

    # The Cube pipe barrier drains the first projection before address reuse.
    # Continue its Cube mutex numbering; L0 scale planes follow data address >> 4.
    l0_slots = 256 // K0
    cube_id = 6 + 4 * l0_slots + min(2, 65536 // (BM * BN1))
    acc_slots = min(2, 65536 // (BM * BN))
    storage = _gemm2_storage(cube_id, k, l0_slots, acc_slots, BK, 4, 0)
    with pl.section_cube():
        # First B0 was prefetched by GEMM1 at this exact physical base.
        # Immutable B1 and all B scales may load before YQ/YS are ready.
        _gemm2_prefetch4_cube(pl.make_tuple(inputs=inputs, buffers=storage, t=t, ag=ag))


def _gemm2_mid_prefill3(inputs, t, padded, ag):
    """Prefill three medium-M weight strips while retaining the A-buffer aliases."""
    k = 8192

    l0_slots = 256 // K0
    cube_id = 6 + 4 * l0_slots + min(2, 65536 // (BM * BN1))
    acc_slots = min(2, 65536 // (BM * BN))
    storage = _gemm2_storage(cube_id, k, l0_slots, acc_slots, BK, 4, 0)
    with pl.section_cube():
        _gemm2_mid_prefill3_cube(
            pl.make_tuple(inputs=inputs, buffers=storage, t=t, ag=ag)
        )


def _gemm2_n192_28(inputs, t, padded, ag):
    """Run the 28-core 192-column projection and crop its final output tile."""
    k = 8192

    # The Cube pipe barrier drains the first projection before address reuse.
    # Continue its Cube mutex numbering; L0 scale planes follow data address >> 4.
    l0_slots = 256 // K0
    cube_id = 6 + 4 * l0_slots + min(2, 65536 // (BM * BN1))
    acc_slots = min(2, 65536 // (BM * BN))
    storage = _gemm2_storage(cube_id, k, l0_slots, acc_slots, BK, 4, 0)
    with pl.section_cube():
        # First B0 was prefetched by GEMM1 at this exact physical base.
        # Immutable B1 and all B scales may load before YQ/YS are ready.
        _gemm2_n192_28_cube(pl.make_tuple(inputs=inputs, buffers=storage, t=t, ag=ag))


def _gemm2_small_28(inputs, t, padded):
    ck = 512
    n0 = 64
    nmain = 3
    cube_id = 6 + 4 * (256 // K0) + min(2, 65536 // (BM * BN1))

    storage = _small_gemm2_storage(cube_id, ck, n0, nmain, 192)
    with pl.section_cube():
        _gemm2_small_28_cube(
            pl.make_tuple(
                inputs=inputs, buffers=storage, ck=ck, n0=n0, t=t, ag=storage.ag
            )
        )


def _gemm2_asym_pair_cube(state):
    """Gemm2 asym pair cube."""
    core = pl.get_block_idx()
    ahead = state.buffers.bg[1]
    scales = state.buffers.bsg.current()
    col = core * 160
    pl.load(ahead, state.inputs.b, [col, 256])
    pl.load(scales, state.inputs.sb, [col, 0, 0], order=[1, 0])
    # All AICs consume this once. YQ/YS reads stay after the wait.
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=7)
    for work in pl.range(pl.get_block_idx(), 32, pl.get_block_num()):
        _gemm2_asym_pair_work_tile(state, pl.make_tuple(work=work))


def _gemm2_mid_prefill3_cube(state):
    """Gemm2 mid prefill3 cube."""
    core = pl.get_block_idx()
    first_col = core * BN
    b1 = state.buffers.bg[1]
    pre_scales = state.buffers.bsg.current()
    pl.load(b1, state.inputs.b, [first_col, BK])
    pl.load(pre_scales, state.inputs.sb, [first_col, 0, 0], order=[1, 0])
    b2 = state.buffers.bg[2]
    pl.load(b2, state.inputs.b, [first_col, 2 * BK])
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=7)
    mi = 0
    ni = core * BN
    mv = state.t
    nv = BN
    _gemm2_compute_tile(state, core, mi, ni, mv, nv, 3)


def _gemm2_n192_28_cube(state):
    """Gemm2 n192 28 cube."""
    core = pl.get_block_idx()
    first_col = core * BN
    b1 = state.buffers.bg[1]
    pre_scales = state.buffers.bsg.current()
    if core < (OUT_FEATURES + BN - 1) // BN:
        pl.set_validshape(b1, [pl.min(BN, OUT_FEATURES - first_col), BK])
        pl.set_validshape(
            pre_scales, [GEMM2_K // 32, pl.min(BN, OUT_FEATURES - first_col)]
        )
        pl.load(b1, state.inputs.b, [first_col, BK])
        pl.load(pre_scales, state.inputs.sb, [first_col, 0, 0], order=[1, 0])
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=7)
    nm = (state.t + BM - 1) // BM
    nn = (OUT_FEATURES + BN - 1) // BN
    for nwork in pl.range(core, nn, pl.get_block_num()):
        for mwork in pl.range(nm):
            _gemm2_n192_28_work_tile(
                state, pl.make_tuple(mwork=mwork, nn=nn, nwork=nwork)
            )


def _gemm2_paired_tail_cube(state):
    """Gemm2 paired tail cube."""
    core = pl.get_block_idx()
    first_col = core * BN
    b0 = state.buffers.bg[0]
    b1 = state.buffers.bg[1]
    pre_scales = state.buffers.bsg.current()
    pl.load(b0, state.inputs.b, [first_col, 0])
    pl.load(b1, state.inputs.b, [first_col, 512])
    pl.load(pre_scales, state.inputs.sb, [first_col, 0, 0], order=[1, 0])
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=7)
    mi = 0
    ni = core * BN
    mv = state.t
    nv = BN
    acc = state.buffers.accg.next()
    ast = state.buffers.asg.current()
    bst = state.buffers.bsg.current()
    pl.set_validshape(acc, [mv, nv])
    pl.set_validshape(ast, [mv, GEMM2_K // 32])
    pl.set_validshape(bst, [GEMM2_K // 32, nv])
    pl.set_validshape(state.ag, [mv, 512])
    pl.set_validshape(state.buffers.bg, [nv, 512])
    pl.set_validshape(state.buffers.b_transposed, [512, nv])
    pl.set_validshape(state.buffers.al, [mv, K0])
    pl.set_validshape(state.buffers.als, [mv, K0 // 32])
    pl.set_validshape(state.buffers.bl, [K0, nv])
    pl.set_validshape(state.buffers.bls, [K0 // 32, nv])
    pl.load(ast, state.inputs.sa, [mi, 0, 0])
    first_a = state.ag[0]
    second_a = state.ag[1]
    pl.load(first_a, state.inputs.a, [mi, 0])
    pl.load(second_a, state.inputs.a, [mi, 512])
    for window in pl.range(0, GEMM2_K, 512):
        _gemm2_paired_tail_kstep(
            state, pl.make_tuple(window=window, ni=ni, ast=ast, bst=bst, acc=acc, mi=mi)
        )
    pl.store(state.inputs.output, acc, [mi, ni], phase=pl.STPhase.Final)


def _gemm2_prefetch4_cube(state):
    """Gemm2 prefetch4 cube."""
    core = pl.get_block_idx()
    first_col = core * BN
    b1 = state.buffers.bg[1]
    pre_scales = state.buffers.bsg.current()
    pl.load(b1, state.inputs.b, [first_col, BK])
    pl.load(pre_scales, state.inputs.sb, [first_col, 0, 0], order=[1, 0])
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=7)
    nm = (state.t + BM - 1) // BM
    nn = (OUT_FEATURES + BN - 1) // BN
    for work in pl.range(pl.get_block_idx(), nm * nn, pl.get_block_num()):
        _gemm2_prefetch4_work_tile(state, pl.make_tuple(work=work, nn=nn, nm=nm))


def _gemm2_small_28_cube(state):
    """Gemm2 small 28 cube."""
    pl.set_validshape(state.buffers.bl, [state.ck, state.n0])
    pl.set_validshape(state.buffers.bls, [state.ck // 32, state.n0])
    core = pl.get_block_idx()
    first = state.buffers.bg[0]
    second = state.buffers.bg[1]
    bst = state.buffers.bsg.current()
    nn = (5120 + BN - 1) // BN
    if core < nn:
        nv0 = pl.min(BN, 5120 - core * BN)
        pl.set_validshape(first, [nv0, state.ck])
        pl.set_validshape(second, [nv0, state.ck])
        pl.set_validshape(bst, [256, nv0])
        pl.load(first, state.inputs.b, [core * BN, 0])
        pl.load(second, state.inputs.b, [core * BN, state.ck])
        pl.load(bst, state.inputs.sb, [core * BN, 0, 0], order=[1, 0])
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=7)
    nm = (state.t + BM - 1) // BM
    for work in pl.range(core, nm * nn, pl.get_block_num()):
        _gemm2_small_28_work_tile(
            state, pl.make_tuple(work=work, nn=nn, bst=bst, core=core)
        )


def _gemm2_tile_cube(state):
    """Gemm2 tile cube."""
    nm = (state.t + BM - 1) // BM
    nn = (OUT_FEATURES + BN - 1) // BN
    for work in pl.range(pl.get_block_idx(), nm * nn, pl.get_block_num()):
        _gemm2_tile_work_tile(state, pl.make_tuple(work=work, nn=nn, nm=nm))


def _small_gemm2_cube(state):
    """Small gemm2 cube."""
    tail_blg = pl.reinterpret(state.buffers.bl, shape=[state.ck, 32])
    tail_blsg = pl.reinterpret(state.buffers.bls, shape=[state.ck // 32, 32])
    pl.set_validshape(state.buffers.bl, [state.ck, state.n0])
    pl.set_validshape(state.buffers.bls, [state.ck // 32, state.n0])
    pl.set_validshape(tail_blg, [state.ck, 32])
    pl.set_validshape(tail_blsg, [state.ck // 32, 32])
    core = pl.get_block_idx()
    first = state.buffers.bg[0]
    second = state.buffers.bg[1]
    bst = state.buffers.bsg.current()
    pl.set_validshape(first, [160, state.ck])
    pl.load(first, state.inputs.b, [core * 160, 0])
    pl.set_validshape(second, [160, state.ck])
    pl.set_validshape(bst, [256, 160])
    pl.load(second, state.inputs.b, [core * 160, state.ck])
    pl.load(bst, state.inputs.sb, [core * 160, 0, 0], order=[1, 0])
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=7)
    mi = 0
    ni = core * 160
    mv = state.t
    ast = state.buffers.asg.current()
    acc0 = state.buffers.accg[0]
    tail = state.tailg.current()
    pl.set_validshape(acc0, [mv, state.n0])
    pl.set_validshape(tail, [mv, 32])
    acc1 = state.buffers.accg[1]
    pl.set_validshape(acc1, [mv, state.n0])
    pl.set_validshape(state.ag, [mv, state.ck])
    pl.set_validshape(ast, [mv, 256])
    pl.set_validshape(state.buffers.al, [mv, state.ck])
    pl.set_validshape(state.buffers.als, [mv, state.ck // 32])
    pl.load(ast, state.inputs.sa, [mi, 0, 0])
    for ki in pl.range(0, 8192, state.ck):
        _small_gemm2_kstep(
            state,
            pl.make_tuple(
                ki=ki,
                mi=mi,
                ni=ni,
                tail_blg=tail_blg,
                tail_blsg=tail_blsg,
                ast=ast,
                bst=bst,
                acc0=acc0,
                acc1=acc1,
                tail=tail,
            ),
        )


def _run_second_projection(yq_ptr, ys_ptr, wb_ptr, sb_ptr, out_ptr, t, padded, ag):
    """Select the existing G2 implementation without changing its tiling."""
    a = pl.make_tensor(yq_ptr, [padded, 8192], dtype=FP8)
    sa = pl.make_tensor(ys_ptr, [padded, 128, 2], dtype=E8)
    b = pl.make_tensor(wb_ptr, [5120, 8192], dtype=FP8)
    sb = pl.make_tensor(sb_ptr, [5120, 128, 2], dtype=E8)
    output = pl.make_tensor(out_ptr, [padded, 5120], dtype=pl.DT_BF16)
    inputs = pl.make_tuple(a=a, sa=sa, b=b, sb=sb, output=output)
    if Algo == 1:
        if BN == 192:
            _gemm2_small_28(inputs, t, padded)
        else:
            _small_gemm2(inputs, t, padded)
    elif Algo == 2:
        _gemm2_asym_pair(inputs, t, padded, ag)
    elif Algo == 3:
        if BM <= 128:
            _gemm2_paired_tail(inputs, t, padded)
        else:
            _gemm2_mid_prefill3(inputs, t, padded, ag)
    elif Algo == 4:
        if BN == 192:
            _gemm2_n192_28(inputs, t, padded, ag)
        else:
            _gemm2_prefetch4(inputs, t, padded, ag)
    else:
        _gemm2_tile(inputs, t, padded)


# Native entry and host API


def workspace_bytes(t):
    """Flat GM workspace size in bytes for `t` tokens."""
    return WORKSPACE_BYTES_PER_ROW * (
        (t + PAD_MULTIPLE - 1) // PAD_MULTIPLE * PAD_MULTIPLE
    )


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
    # Preserve the 9-pointer + packed 16-byte runtime tiling ABI.
    t = tiling.t
    padded = tiling.padded
    aq_ptr = workspace
    asc_ptr = pl.addptr(workspace, ASC_OFFSET_PER_ROW * padded)
    yq_ptr = pl.addptr(workspace, YQ_OFFSET_PER_ROW * padded)
    ys_ptr = pl.addptr(workspace, YS_OFFSET_PER_ROW * padded)
    with pl.section_vector():
        saved_ctrl_policy = pl.get_ctrl_spr(60, 60)
        pl.set_ctrl_spr(60, 60, 0)
    _prepare_activation(o_ptr, cos_ptr, sin_ptr, aq_ptr, asc_ptr, t, padded)
    # Tensor views stay with their raw pointers; no returned tensor aliases.
    a = pl.make_tensor(aq_ptr, [8 * padded, 4096], dtype=FP8)
    sa = pl.make_tensor(asc_ptr, [8 * padded, 64, 2], dtype=E8)
    b = pl.make_tensor(wa_ptr, [8192, 4096], dtype=FP8)
    sb = pl.make_tensor(sa_ptr, [8192, 64, 2], dtype=E8)
    wb = pl.make_tensor(wb_ptr, [5120, 8192], dtype=FP8)
    yq = pl.make_tensor(yq_ptr, [padded, 8192], dtype=pl.DT_UINT8)
    ys = pl.make_tensor(ys_ptr, [padded, 256], dtype=pl.DT_UINT8)
    q_rows = Q_ROWS if Algo == 1 or Algo == 4 or BN1 == 256 else BM // 2
    vec_bf, vec_q, vec_es, vec_recip, vec_even, vec_odd, vec_scale = (
        _quantization_addresses(q_rows)
    )
    if Algo == 1:
        ag, bg, asg, bsg, al, als, bl, bls = _gemm1_small_storage()
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
        b_transposed = pl.reinterpret(bg, shape=[BK, BN1], layout=pl.ZN)
    else:
        l0_slots = 256 // K0
        acc_id = 6 + 4 * l0_slots
        acc_slots = min(2, 65536 // (BM * BN1))
        scale_cols = 128 if Algo == 4 or BN1 != 256 else 64
        # Only Algo3/BM128 uses the three-slot 512-column B prefetch panel.
        g1_bk = 512 if Algo == 3 and BM <= 128 else BK
        g1_bslots = 3 if Algo == 3 and BM <= 128 else 2
        ag, bg, asg, bsg = _gemm1_l1_groups(scale_cols, g1_bk, g1_bslots)
        al, als, bl, bls = _gemm1_l0_groups(l0_slots)
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
        b_transposed = pl.reinterpret(bg, shape=[g1_bk, max(BN1, BN)], layout=pl.ZN)
    # The pipeline scanner requires direct Acc and cross-core declarations.
    if VSlots == 2:
        shared = pl.make_tile_group(
            type=pl.TileType(
                shape=[BM // 2, BN1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[32768, 32768 + 2 * BM * BN1],
            mutex_ids=[9, 10],
            fwd_ids=[0, 1],
            bwd_ids=[0, 1],
        )
    else:
        shared = pl.make_tile_group(
            type=pl.TileType(
                shape=[BM // 2, BN1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[32768],
            mutex_ids=[9],
            fwd_ids=[0],
            bwd_ids=[0],
        )
    bfg, qg, esg, rg, eg, og, spg = _quantization_groups(
        q_rows, vec_bf, vec_q, vec_es, vec_recip, vec_even, vec_odd, vec_scale
    )
    if Algo == 1:
        wb_prefetch = bg
    else:
        wb_prefetch = pl.reinterpret(bg, shape=[BN, BK])
    inputs = pl.make_tuple(a=a, sa=sa, b=b, sb=sb, wb=wb)
    buffers = pl.make_tuple(
        wbg=wb_prefetch,
        ag=ag,
        bg=bg,
        b_transposed=b_transposed,
        asg=asg,
        bsg=bsg,
        al=al,
        als=als,
        bl=bl,
        bls=bls,
    )
    destinations = pl.make_tuple(yq=yq, ys=ys)
    quantization = pl.make_tuple(bfg=bfg, qg=qg, esg=esg, rg=rg, eg=eg, og=og, spg=spg)
    with pl.section_cube():
        _prefetch_gemm1_profile(inputs, buffers, t)
        pl.system.bar_all()
        if Algo == 0 and BN1 != 256:
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
    if Algo == 0 and BN1 != 256:
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
            if Algo == 1:
                _gemm1_profile(work, t, padded, inputs, buffers, accg, whole, shared)
            else:
                _gemm1_profile(work, t, padded, inputs, buffers, accg, accg, shared)
        with pl.section_vector():
            _quantize_profile(work, t, padded, destinations, quantization, shared)
    _finish_gemm1()
    _run_second_projection(yq_ptr, ys_ptr, wb_ptr, sb_ptr, out_ptr, t, padded, ag)
    with pl.section_vector():
        pl.system.bar_all()
        pl.set_ctrl_spr(60, 60, saved_ctrl_policy)


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
