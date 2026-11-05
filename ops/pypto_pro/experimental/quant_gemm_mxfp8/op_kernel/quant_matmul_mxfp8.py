#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F821

"""A5 MXFP8 matmul (rank-2).

Computes

    out = BF16( dequant(x1) @ dequant(x2) + bias )

where x1/x2 are FP8 (E5M2 or E4M3FN, both operands the same type) and every
group of 32 consecutive K elements shares one E8M0 scale (byte u means
2**(u-127)). Scales are carried packed as [..., K/64, 2], the last axis holding
the two 32-element groups inside one 64-element window. bias is optional FP32
[N], broadcast over M.

Supported shapes: M and N are arbitrary positive integers (M/N tail tiles
included), K is a multiple of BK = 128, and inputs are rank-2 NN. x2 may be ND
or WeightNZ (format 29, E4M3FN only); x1, both scales, bias and the output are
always ND. The output is BF16 and is the only device allocation, except that the
split-K profile below also allocates an FP32 workspace.

Three execution profiles cover the shapes this module was built for, selected by
(M,N,K); each has its own tile residency plan, launch geometry and L1 layout:

  * Profile 0 -- (96,1792,5120), split-K. 7 N tiles x 4 K splits of 1280 = 28
    useful AICs in a 32-block mixed launch. Each split writes its FP32 partials
    to a GM workspace; one all-AIC FIX completion barrier then lets 56 AIVs
    reduce the four partials strictly left-to-right in FP32 and round to BF16
    (RNE) once. Two L1 data buffers.
  * Profile 1 -- (96,32768,1280), A fully resident. All five A K-panels
    (K/KL1 = 1280/256) and the whole A scale window stay in L1 while each AIC
    walks four N tiles; 32 blocks, four rounds.
  * Profile 2 -- (96,5120,8192), sliding window. A and B stream through four L1
    slots and the scales refresh every 4096 K elements; 20 blocks, one N tile
    each.

Any other shape takes the shape-general kernel, which uses Profile 2's tile
geometry and L1 address table with no split-K: pure Cube, no workspace, no vector
epilogue. It narrows M/N tail tiles with pl.set_validshape, shrinks a partial
trailing K window with pl.min, and walks an arbitrary tile count with a
grid-stride loop.

Accumulation order is part of the numerical contract: the first MX MMAD of an
output tile initializes the FP32 accumulator -- consuming the FP32 BiasTable
operand directly when bias is present, i.e. init_with_zero=false -- the remaining
steps accumulate in ascending K, the last carries AccPhase.Final, and the tile is
stored with STPhase.Final as BF16 (RNE). When K == BK the first step is also the
last, so it carries Final. Under split-K, bias is applied on split 0 only and the
four FP32 partials are reduced strictly left-to-right with a single final
rounding. Within one L0 iteration the order is A data + A scale, the
L1->BiasTable move, then B data + B scale. The cube section runs under
pl.system.set_mm_layout_transform(enabled=True) (column-first fixpipe L0C drain)
and restores False after the final store.

Both x2 loads pass l2_cache_mode=B_LOAD_L2_HINT so the weight stream does not allocate
in L2; no other tensor is hinted. Profile 0's vector epilogue calls
pl.system.icache_preload(2) after the cross-core wait and inside the active-AIV
guard, so only the 56 participating AIVs preload.

Synchronization is automatic (auto_mutex=True), except for Profile 0's cross-core
handover, which needs one FIX completion barrier covering all 32 launched AICs
(including the four idle ones) before any AIV reads another AIC's workspace. No
runtime tiling calculation, platform query, SDK ini read or resource-plan
construction is performed.
"""

import torch
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf
from pypto_pro.runtime.tilingkey import TilingKeyField

M = 96
SHAPES = ((96, 1792, 5120), (96, 32768, 1280), (96, 5120, 8192))

# Both x2 loads pass TLoadL2Hint.NotAllocKeep, i.e. the weight stream is not
# allocated in L2. x2 is the dominant traffic and is streamed once per output
# tile: it is 9.2 / 41.9 / 41.9 MB of the 10.3 / 49.8 / 45.1 MB logical totals
# of the three profiles, against 128 MiB of L2, so letting it allocate would
# displace the tensors that are actually reused across tiles. Only x2 is
# hinted -- x1, both scales and bias are small and reused, and the output store
# keeps the default policy. l2_cache_mode is a compile-time lowering attribute
# (it becomes a TLOAD<TLoadL2Hint::...> template argument), so one value
# applies to every tile, N tail tiles included.
B_LOAD_L2_HINT = pl.TLoadL2Hint.NotAllocKeep

# Profile 0's vector epilogue preloads 2 instruction-cache lines before starting
# the partial-sum reduction. The call sits after the cross-core wait and inside
# the active-AIV guard, so only the 56 participating AIVs issue it. Profiles 1/2
# and the shape-general kernel have no vector section and issue no preload.
ICACHE_PREFETCH_LEN = 2

# Tile geometry of the shape-general schedule (Profile 3). It reuses Profile 2's
# L1/L0 address table, so these match Profile 2's kl1/scale_kl1 exactly.
G_BM = M            # 96; the tile height, also Profile 2's a_l1 row count
G_BN = 256
G_BK = 128
G_KL1 = 256
G_SCALE_KL1 = 4096



class MxGemmKey:
    # Finite user contract. Each profile changes L1 placement, residency,
    # launch geometry, and/or split-K arithmetic; it is not a tuning case ID.
    # 0/1/2 are the three measured shapes, each with its own schedule;
    # 3 is the shape-general schedule that serves every other shape.
    Profile = TilingKeyField(bits=2, values=[0, 1, 2, 3])
    InputFormat = TilingKeyField(bits=1, values=[0, 1])
    HasBias = TilingKeyField(bits=1, values=[0, 1])
    WeightNz = TilingKeyField(bits=1, values=[0, 1])


@pl.vector_function
def _add_partial_in_place(src, partial_index: pl.DT_INT64):
    """One Add pass of the split-K reduction: add partial `partial_index` into
    the first FP32 UB partial, in place. Called three times (indices 1, 2, 3) so
    the four partials combine strictly left-to-right as ((p0+p1)+p2)+p3."""
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for off in pl.range(0, 3072, 64):
        previous = vf.load_align(src, [0, off])
        partial = vf.load_align(src, [partial_index, off])
        total = vf.add(previous, partial, mask)
        vf.store_align(src, total, mask, [0, off])


@pl.vector_function
def _cast_first_partial_in_place(src, dst):
    """RINT cast of the reduced FP32 sum to BF16, at the SAME UB base address.

    Forward narrowing writes bytes [2*off, 2*(off+64)), whereas its input
    reads [4*off, 4*(off+64)). Stores never overlap a later iteration's
    input. VLD_VST orders loads before an overlapping narrowing store,
    including loads from earlier iterations; no in-VF RAW edge is needed.
    """
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for off in pl.range(0, 3072, 64):
        total = vf.load_align(src, [0, off])
        bf = vf.astype(total, mask, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_RINT)
        vf.mem_bar(mode=pl.MemBarMode.VLD_VST)
        # Conversion leaves one BF16 in each b32 container. Compact exactly
        # 64 BF16 values; an ordinary full-width b16 store is not equivalent.
        vf.store_align(dst, bf, mask, off, dist=pl.StoreDist.PACK_B32)


@pl.jit(arch="a5", auto_mutex=True, tiling_key=MxGemmKey)
def _quant_matmul_mxfp8_kernel(
    x1_ptr: pl.Ptr[pl.DT_UINT8],
    x2_ptr: pl.Ptr[pl.DT_UINT8],
    x2_nz: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP8E4M3FN, pl.NZ],
    sa_ptr: pl.Ptr[pl.DT_UINT8],
    sb_ptr: pl.Ptr[pl.DT_UINT8],
    bias_ptr: pl.Ptr[pl.DT_UINT8],
    workspace_ptr: pl.Ptr[pl.DT_UINT8],
    out_ptr: pl.Ptr[pl.DT_UINT8],
    m_dim: pl.DT_INT64,
    n_dim: pl.DT_INT64,
    k_dim: pl.DT_INT64,
):
    # One kernel, four compile-time schedules selected by the Profile tiling-key
    # field, so each compiled variant traces exactly one branch:
    #
    #   Profile 0  (96,1792,5120)   split-K: 7 N tiles x 4 K splits, FP32
    #                               workspace, all-AIC FIX barrier, then a
    #                               56-AIV vector reduction. MIX kernel.
    #   Profile 1  (96,32768,1280)  A-resident: a_l1 depth=5, sa_l1 depth=1,
    #                               four N tiles per worker.
    #   Profile 2  (96,5120,8192)   sliding window: 20 workers, one N tile each.
    #   Profile 3  every other shape, see the branch below.
    #
    # Profiles 0/1 hard-code their shape into the schedule (split count, panel
    # depths), so neither is reusable for arbitrary shapes; Profile 3 therefore
    # keeps Profile 2's tile geometry and L1/L0 address table -- G_BM/G_BN/G_BK
    # and G_KL1/G_SCALE_KL1 -- with no split-K: pure Cube, no workspace, no
    # cross-core barrier and no vector section. K is reduced by one worker end to
    # end, so a shape whose N is small relative to K gets less parallelism than
    # Profile 0's split-K schedule would give it.
    #
    # Profile 3 handles arbitrary shapes with:
    #   * M/N tails: pl.set_validshape narrows the L1, L0, scale-plane, bias and
    #     Acc tiles. Cube accepts a partial M and/or N down to one 16x16 fractal.
    #   * K tails: a real scale-window loop with scale_len/data_len =
    #     pl.min(window, remaining), so K need not equal G_SCALE_KL1 and need not
    #     be a multiple of G_KL1.
    #   * arbitrary tile counts: a grid-stride loop over m_tiles*n_tiles, so the
    #     launch geometry need not be exactly one full wave.
    #
    # m_dim/n_dim/k_dim are runtime values read only by Profile 3; workspace_ptr
    # is read only by Profile 0. Shapes never enter the tiling key, so one
    # compiled variant per key serves every shape of its schedule.
    if Profile == 3:
        if InputFormat == 0:
            fp8 = pl.DT_FP8E5M2
        else:
            fp8 = pl.DT_FP8E4M3FN
        # Shapes arrive as runtime scalars rather than being read off a tensor, so the
        # pointer-based ABI is kept and both E8M0 scale carriers (native and uint8 +
        # override) still work through make_tensor's dtype argument.
        x1 = pl.make_tensor(x1_ptr, [m_dim, k_dim], dtype=fp8)
        x2 = pl.make_tensor(x2_ptr, [k_dim, n_dim], dtype=fp8)
        sa = pl.make_tensor(sa_ptr, [m_dim, k_dim // 64, 2], dtype=pl.DT_FP8E8M0)
        sb = pl.make_tensor(sb_ptr, [k_dim // 64, n_dim, 2], dtype=pl.DT_FP8E8M0)
        out = pl.make_tensor(out_ptr, [m_dim, n_dim], dtype=pl.DT_BF16)
        if HasBias == 1:
            bias = pl.make_tensor(bias_ptr, [1, n_dim], dtype=pl.DT_FP32)

        # Profile 2's address table verbatim; only valid_shape becomes dynamic so tail
        # blocks can narrow M and/or N.
        a_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[G_BM, G_KL1], dtype=fp8, target_memory=pl.MemorySpace.Mat,
                             valid_shape=[-1, -1]),
            addrs=[0, 262144, 24576, 286720], mutex_ids="auto", depth=4)
        b_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[G_KL1, G_BN], dtype=fp8, target_memory=pl.MemorySpace.Mat,
                             valid_shape=[-1, -1]),
            addrs=[49152, 311296, 114688, 376832], mutex_ids="auto", depth=4)
        sa_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[G_BM, G_SCALE_KL1 // 32], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.Mat, layout=pl.ZZ,
                             valid_shape=[-1, -1]),
            addrs=[180224, 442368], mutex_ids="auto", depth=2)
        sb_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[G_SCALE_KL1 // 32, G_BN], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NN,
                             valid_shape=[-1, -1]),
            addrs=[192512, 454656], mutex_ids="auto", depth=2)
        a_l0 = pl.make_tile_group(
            type=pl.TileType(shape=[G_BM, G_BK], dtype=fp8, target_memory=pl.MemorySpace.Left,
                             layout=pl.NZ, compact=1, valid_shape=[-1, -1]),
            addrs=[0, 32768], mutex_ids="auto", depth=2)
        b_l0 = pl.make_tile_group(
            type=pl.TileType(shape=[G_BK, G_BN], dtype=fp8, target_memory=pl.MemorySpace.Right,
                             layout=pl.ZN, compact=1, valid_shape=[-1, -1]),
            addrs=[0, 32768], mutex_ids="auto", depth=2)
        plane_a = pl.make_tile_group(
            type=pl.TileType(shape=[G_BM, G_BK // 32], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.ScaleLeft, layout=pl.ZZ,
                             compact=1, valid_shape=[-1, -1]),
            addrs=[0, 2048], mutex_ids="auto", depth=2)
        plane_b = pl.make_tile_group(
            type=pl.TileType(shape=[G_BK // 32, G_BN], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.ScaleRight, layout=pl.NN,
                             compact=1, valid_shape=[-1, -1]),
            addrs=[0, 2048], mutex_ids="auto", depth=2)
        acc_group = pl.make_tile_group(
            type=pl.TileType(shape=[G_BM, G_BN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                             layout=pl.NZ, fractal=1024, compact=1, valid_shape=[-1, -1]),
            addrs=[0, 131072], mutex_ids="auto", depth=2)
        if HasBias == 1:
            bias_l1 = pl.make_tile_group(
                type=pl.TileType(shape=[1, G_BN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat,
                                 valid_shape=[-1, -1]),
                addrs=[225280, 487424], mutex_ids="auto", depth=2)
            bias_bt = pl.make_tile_group(
                type=pl.TileType(shape=[1, G_BN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias,
                                 valid_shape=[-1, -1]),
                addrs=[0, 1024], mutex_ids="auto", depth=2)

        m_tiles = (m_dim + G_BM - 1) // G_BM
        n_tiles = (n_dim + G_BN - 1) // G_BN
        total_tiles = m_tiles * n_tiles

        with pl.section_cube():
            pl.system.set_mm_layout_transform(enabled=True)
            core_id = pl.get_block_idx()
            num_cores = pl.get_block_num()
            for tile_id in pl.range(core_id, total_tiles, num_cores):
                m_offset = (tile_id // n_tiles) * G_BM
                n_offset = (tile_id % n_tiles) * G_BN
                vm = pl.min(G_BM, m_dim - m_offset)
                vn = pl.min(G_BN, n_dim - n_offset)
                acc = acc_group.next()
                pl.set_validshape(acc, [vm, vn])
                if HasBias == 1:
                    bias_mat = bias_l1.next()
                    bt = bias_bt.next()
                    pl.set_validshape(bias_mat, [1, vn])
                    pl.set_validshape(bt, [1, vn])
                for scale_k in pl.range(0, k_dim, G_SCALE_KL1):
                    scale_len = pl.min(G_SCALE_KL1, k_dim - scale_k)
                    ascale = sa_l1.next()
                    bscale = sb_l1.next()
                    pl.set_validshape(ascale, [vm, scale_len // 32])
                    pl.set_validshape(bscale, [scale_len // 32, vn])
                    pl.load(ascale, sa, [m_offset, scale_k // 64, 0])
                    pl.load(bscale, sb, [scale_k // 64, n_offset, 0])
                    for data_k in pl.range(scale_k, scale_k + scale_len, G_KL1):
                        data_len = pl.min(G_KL1, k_dim - data_k)
                        al1 = a_l1.next()
                        bl1 = b_l1.next()
                        pl.set_validshape(al1, [vm, data_len])
                        pl.set_validshape(bl1, [data_len, vn])
                        pl.load(al1, x1, [m_offset, data_k])
                        if HasBias == 1 and data_k == 0:
                            pl.load(bias_mat, bias, [0, n_offset])
                        if WeightNz == 1:
                            pl.load(bl1, x2_nz, [data_k, n_offset], l2_cache_mode=B_LOAD_L2_HINT)
                        else:
                            pl.load(bl1, x2, [data_k, n_offset], l2_cache_mode=B_LOAD_L2_HINT)
                        for l0_k in pl.range(0, data_len, G_BK):
                            al0 = a_l0.next()
                            pal0 = plane_a.next()
                            bl0 = b_l0.next()
                            pbl0 = plane_b.next()
                            pl.set_validshape(al0, [vm, G_BK])
                            pl.set_validshape(pal0, [vm, G_BK // 32])
                            pl.set_validshape(bl0, [G_BK, vn])
                            pl.set_validshape(pbl0, [G_BK // 32, vn])
                            # E8M0 column inside the current scale window.
                            s_off = (data_k - scale_k + l0_k) // 32
                            pl.move(al0, al1, [0, l0_k])
                            pl.move(pal0, ascale, [0, s_off])
                            if HasBias == 1 and data_k == 0 and l0_k == 0:
                                pl.move(bt, bias_mat)
                            pl.move(bl0, bl1, [l0_k, 0])
                            pl.move(pbl0, bscale, [s_off, 0])
                            # data_k is ABSOLUTE, so the absolute K offset is data_k+l0_k.
                            k_offset = data_k + l0_k
                            # First and last are tested NESTED, not as an if/elif chain:
                            # when K == G_BK the first step is also the last and must
                            # carry Final, otherwise the accumulator is never finalized
                            # and the STPhase.Final store waits forever (aicore timeout).
                            last_step = k_offset + G_BK == k_dim
                            if k_offset == 0:
                                if last_step:
                                    if HasBias == 1:
                                        pl.matmul_mx(acc, al0, bl0, pal0, pbl0, bt,
                                                     phase=pl.AccPhase.Final)
                                    else:
                                        pl.matmul_mx(acc, al0, bl0, pal0, pbl0,
                                                     phase=pl.AccPhase.Final)
                                else:
                                    if HasBias == 1:
                                        pl.matmul_mx(acc, al0, bl0, pal0, pbl0, bt,
                                                     phase=pl.AccPhase.Partial)
                                    else:
                                        pl.matmul_mx(acc, al0, bl0, pal0, pbl0,
                                                     phase=pl.AccPhase.Partial)
                            else:
                                if last_step:
                                    pl.matmul_mx_acc(acc, acc, al0, bl0, pal0, pbl0,
                                                     phase=pl.AccPhase.Final)
                                else:
                                    pl.matmul_mx_acc(acc, acc, al0, bl0, pal0, pbl0,
                                                     phase=pl.AccPhase.Partial)
                pl.store(out, acc, [m_offset, n_offset], phase=pl.STPhase.Final)
            pl.system.set_mm_layout_transform(enabled=False)
    else:
        if InputFormat == 0:
            fp8 = pl.DT_FP8E5M2
        else:
            fp8 = pl.DT_FP8E4M3FN
        if Profile == 0:
            n, k, bk, kl1, scale_kl1 = 1792, 5120, 128, 512, 1024
            active, rounds, local_k = 28, 1, 1280
        elif Profile == 1:
            n, k, bk, kl1, scale_kl1 = 32768, 1280, 128, 256, 1280
            active, rounds, local_k = 32, 4, 1280
        else:
            n, k, bk, kl1, scale_kl1 = 5120, 8192, 128, 256, 4096
            active, rounds, local_k = 20, 1, 8192

        x1 = pl.make_tensor(x1_ptr, [M, k], dtype=fp8)
        x2 = pl.make_tensor(x2_ptr, [k, n], dtype=fp8)
        sa = pl.make_tensor(sa_ptr, [M, k // 64, 2], dtype=pl.DT_FP8E8M0)
        sb = pl.make_tensor(sb_ptr, [k // 64, n, 2], dtype=pl.DT_FP8E8M0)
        out = pl.make_tensor(out_ptr, [M, n], dtype=pl.DT_BF16)
        if HasBias == 1:
            bias = pl.make_tensor(bias_ptr, [1, n], dtype=pl.DT_FP32)

        # Exact source half-L1 placement. Full-A NZ panels are consecutive along
        # K: [K/32,M/16,16,32], so five [96,256] panels form [96,1280] NZ.
        if Profile == 0:
            a_l1 = pl.make_tile_group(
                type=pl.TileType(shape=[M, kl1], dtype=fp8,
                                 target_memory=pl.MemorySpace.Mat, compact=1),
                addrs=[0, 262144], mutex_ids="auto", depth=2)
            b_l1 = pl.make_tile_group(
                type=pl.TileType(shape=[kl1, 256], dtype=fp8,
                                 target_memory=pl.MemorySpace.Mat, compact=1),
                addrs=[49152, 311296], mutex_ids="auto", depth=2)
            sa_l1 = pl.make_tile_group(
                type=pl.TileType(shape=[M, 32], dtype=pl.DT_FP8E8M0,
                                 target_memory=pl.MemorySpace.Mat, layout=pl.ZZ, compact=1),
                addrs=[180224, 442368], mutex_ids="auto", depth=2)
            sb_l1 = pl.make_tile_group(
                type=pl.TileType(shape=[32, 256], dtype=pl.DT_FP8E8M0,
                                 target_memory=pl.MemorySpace.Mat, layout=pl.NN, compact=1),
                addrs=[183296, 445440], mutex_ids="auto", depth=2)
            if HasBias == 1:
                bias_l1 = pl.make_tile_group(
                    type=pl.TileType(shape=[1, 256], dtype=pl.DT_FP32,
                                     target_memory=pl.MemorySpace.Mat),
                    addrs=[191488, 453632], mutex_ids="auto", depth=2)
        elif Profile == 1:
            bias_bytes = HasBias * 1024
            a_l1 = pl.make_tile_group(
                type=pl.TileType(shape=[M, kl1], dtype=fp8, target_memory=pl.MemorySpace.Mat),
                addrs=[141312 + bias_bytes, 165888 + bias_bytes, 190464 + bias_bytes,
                       215040 + bias_bytes, 239616 + bias_bytes], mutex_ids="auto", depth=5)
            b_l1 = pl.make_tile_group(
                type=pl.TileType(shape=[kl1, 256], dtype=fp8, target_memory=pl.MemorySpace.Mat),
                addrs=[0, 268032 + bias_bytes, 65536, 333568 + bias_bytes], mutex_ids="auto", depth=4)
            sa_l1 = pl.make_tile_group(
                type=pl.TileType(shape=[M, 40], dtype=pl.DT_FP8E8M0,
                                 target_memory=pl.MemorySpace.Mat, layout=pl.ZZ),
                addrs=264192 + bias_bytes, mutex_ids="auto", depth=1)
            sb_l1 = pl.make_tile_group(
                type=pl.TileType(shape=[40, 256], dtype=pl.DT_FP8E8M0,
                                 target_memory=pl.MemorySpace.Mat, layout=pl.NN),
                addrs=[131072, 399104 + bias_bytes], mutex_ids="auto", depth=2)
            if HasBias == 1:
                bias_l1 = pl.make_tile_group(
                    type=pl.TileType(shape=[1, 256], dtype=pl.DT_FP32,
                                     target_memory=pl.MemorySpace.Mat),
                    addrs=[141312, 410368], mutex_ids="auto", depth=2)
        else:
            a_l1 = pl.make_tile_group(
                type=pl.TileType(shape=[M, kl1], dtype=fp8, target_memory=pl.MemorySpace.Mat),
                addrs=[0, 262144, 24576, 286720], mutex_ids="auto", depth=4)
            b_l1 = pl.make_tile_group(
                type=pl.TileType(shape=[kl1, 256], dtype=fp8, target_memory=pl.MemorySpace.Mat),
                addrs=[49152, 311296, 114688, 376832], mutex_ids="auto", depth=4)
            sa_l1 = pl.make_tile_group(
                type=pl.TileType(shape=[M, 128], dtype=pl.DT_FP8E8M0,
                                 target_memory=pl.MemorySpace.Mat, layout=pl.ZZ),
                addrs=[180224, 442368], mutex_ids="auto", depth=2)
            sb_l1 = pl.make_tile_group(
                type=pl.TileType(shape=[128, 256], dtype=pl.DT_FP8E8M0,
                                 target_memory=pl.MemorySpace.Mat, layout=pl.NN),
                addrs=[192512, 454656], mutex_ids="auto", depth=2)
            if HasBias == 1:
                bias_l1 = pl.make_tile_group(
                    type=pl.TileType(shape=[1, 256], dtype=pl.DT_FP32,
                                     target_memory=pl.MemorySpace.Mat),
                    addrs=[225280, 487424], mutex_ids="auto", depth=2)

        a_l0 = pl.make_tile_group(
            type=pl.TileType(shape=[M, bk], dtype=fp8, target_memory=pl.MemorySpace.Left,
                             layout=pl.NZ, compact=1),
            addrs=[0, 32768], mutex_ids="auto", depth=2)
        b_l0 = pl.make_tile_group(
            type=pl.TileType(shape=[bk, 256], dtype=fp8, target_memory=pl.MemorySpace.Right,
                             layout=pl.ZN, compact=1),
            addrs=[0, 32768], mutex_ids="auto", depth=2)
        plane_a = pl.make_tile_group(
            type=pl.TileType(shape=[M, bk // 32], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.ScaleLeft, layout=pl.ZZ, compact=1),
            addrs=[0, 2048], mutex_ids="auto", depth=2)
        plane_b = pl.make_tile_group(
            type=pl.TileType(shape=[bk // 32, 256], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.ScaleRight, layout=pl.NN, compact=1),
            addrs=[0, 2048], mutex_ids="auto", depth=2)
        acc_group = pl.make_tile_group(
            type=pl.TileType(shape=[M, 256], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024, compact=1),
            addrs=[0, 131072], mutex_ids="auto", depth=2)
        if HasBias == 1:
            bias_bt = pl.make_tile_group(
                type=pl.TileType(shape=[1, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias),
                addrs=[0, 1024], mutex_ids="auto", depth=2)

        with pl.section_cube():
            pl.system.set_mm_layout_transform(enabled=True)
            worker = pl.get_block_idx()
            if worker < active:
                if Profile == 0:
                    n_idx = worker // 4
                    k_start = (worker % 4) * 1280
                    use_bias = HasBias == 1 and worker % 4 == 0
                    workspace = pl.make_tensor(workspace_ptr, [7, 4, M, 256], dtype=pl.DT_FP32)
                else:
                    n_idx = worker
                    k_start = 0
                    use_bias = HasBias == 1
                for output_iter in pl.range(rounds):
                    n_offset = (n_idx + output_iter * 32) * 256
                    acc = acc_group.next()
                    if HasBias == 1:
                        bias_mat = bias_l1.next()
                        bt = bias_bt.next()
                    for scale_k in pl.range(0, local_k, scale_kl1):
                        scale_len = pl.min(scale_kl1, local_k - scale_k)
                        bscale = sb_l1.next()
                        if Profile == 1:
                            ascale = sa_l1.current()
                            # CopyFullLoadScalesToL1: B first, then A only once.
                            pl.load(bscale, sb, [(k_start + scale_k) // 64, n_offset, 0])
                            if output_iter == 0:
                                pl.load(ascale, sa, [0, 0, 0])
                        else:
                            ascale = sa_l1.next()
                            if Profile == 0:
                                pl.set_validshape(ascale, [M, scale_len // 32])
                                pl.set_validshape(bscale, [scale_len // 32, 256])
                            pl.load(ascale, sa, [0, (k_start + scale_k) // 64, 0])
                            pl.load(bscale, sb, [(k_start + scale_k) // 64, n_offset, 0])
                        for data_k in pl.range(scale_k, scale_k + scale_len, kl1):
                            data_len = pl.min(kl1, local_k - data_k)
                            al1 = a_l1.next()
                            bl1 = b_l1.next()
                            if Profile == 0:
                                # Narrow the descriptors so the last panel of each K
                                # split covers only its valid K columns: a split is
                                # local_k = 1280 long, i.e. five kl1 = 256 panels, and
                                # the trailing one may be shorter.
                                pl.set_validshape(al1, [M, data_len])
                                pl.set_validshape(bl1, [data_len, 256])
                            if Profile != 1 or output_iter == 0:
                                pl.load(al1, x1, [0, k_start + data_k])
                            if HasBias == 1:
                                if use_bias and data_k == 0:
                                    pl.load(bias_mat, bias, [0, n_offset])
                            if WeightNz == 1:
                                # Native GM NZ -> L1 NZ, including compact K tail.
                                pl.load(bl1, x2_nz, [k_start + data_k, n_offset],
                                        l2_cache_mode=B_LOAD_L2_HINT)
                            else:
                                pl.load(bl1, x2, [k_start + data_k, n_offset],
                                        l2_cache_mode=B_LOAD_L2_HINT)
                            for l0_k in pl.range(0, data_len, bk):
                                al0 = a_l0.next()
                                pal0 = plane_a.next()
                                bl0 = b_l0.next()
                                pbl0 = plane_b.next()
                                if Profile == 1:
                                    scale_a_offset = (data_k + l0_k) // 32
                                else:
                                    scale_a_offset = (data_k - scale_k + l0_k) // 32
                                pl.move(al0, al1, [0, l0_k])
                                pl.move(pal0, ascale, [0, scale_a_offset])
                                if HasBias == 1:
                                    if use_bias and data_k == 0 and l0_k == 0:
                                        # Order within the first L0 iteration:
                                        # A/scaleA, then the L1->BiasTable move, then
                                        # B/scaleB. Under split-K only split 0 carries
                                        # bias, so `use_bias` gates it.
                                        pl.move(bt, bias_mat)
                                pl.move(bl0, bl1, [l0_k, 0])
                                pl.move(pbl0, bscale, [(data_k - scale_k + l0_k) // 32, 0])
                                if data_k == 0 and l0_k == 0:
                                    # The first MX MMAD initializes the FP32
                                    # accumulator, and with bias it consumes the FP32
                                    # BiasTable in the same instruction
                                    # (init_with_zero=false), so no separate
                                    # accumulator-preset MMAD is needed.
                                    if use_bias:
                                        pl.matmul_mx(acc, al0, bl0, pal0, pbl0, bt,
                                                     phase=pl.AccPhase.Partial)
                                    else:
                                        pl.matmul_mx(acc, al0, bl0, pal0, pbl0,
                                                     phase=pl.AccPhase.Partial)
                                elif data_k + l0_k + bk == local_k:
                                    pl.matmul_mx_acc(acc, acc, al0, bl0, pal0, pbl0,
                                                     phase=pl.AccPhase.Final)
                                else:
                                    pl.matmul_mx_acc(acc, acc, al0, bl0, pal0, pbl0,
                                                     phase=pl.AccPhase.Partial)
                    if Profile == 0:
                        pl.store(workspace, acc, [n_idx, worker % 4, 0, 0], phase=pl.STPhase.Final)
                    else:
                        pl.store(out, acc, [0, n_offset], phase=pl.STPhase.Final)
            pl.system.set_mm_layout_transform(enabled=False)
            if Profile == 0:
                # Every one of the 32 launched AICs participates, including the
                # four idle AICs. Pair-only handover would permit premature reads
                # of partials produced by other blocks.
                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0,
                                         sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK)
                pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=0,
                                          sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK)
                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=1)

        if Profile == 0:
            with pl.section_vector():
                # The cross-core wait comes first, so inactive AIVs can return before
                # any preload or GM binding happens.
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=1)
                vector_idx = pl.get_block_idx()
                if vector_idx < 56:
                    partials = pl.make_tensor(workspace_ptr, [7, 4, M * 256], dtype=pl.DT_FP32)
                    # Only the 56 active AIVs preload, after the wait and the GM
                    # binding and before the index calculation and the GM read.
                    pl.system.icache_preload(ICACHE_PREFETCH_LEN)
                    src_group = pl.make_tile_group(
                        type=pl.TileType(shape=[4, 3072], dtype=pl.DT_FP32,
                                         target_memory=pl.MemorySpace.Vec),
                        addrs=0, mutex_ids="auto", depth=1)
                    src = src_group.current()
                    dst = pl.reinterpret(src, dtype=pl.DT_BF16, shape=[12, 256])
                    row = (vector_idx % 8) * 12
                    n_tile = vector_idx // 8
                    pl.load(src, partials, [n_tile, 0, row * 256])
                    # Match the three whole-buffer Add passes before Cast.
                    # Separate VF calls expose each UB read/write boundary to
                    # auto_mutex, rather than racing stores with later VF loads.
                    _add_partial_in_place(src, 1)
                    _add_partial_in_place(src, 2)
                    _add_partial_in_place(src, 3)
                    _cast_first_partial_in_place(src, dst)
                    pl.store(out, dst, [row, n_tile * 256])


def quant_matmul_mxfp8(
    x1, x2, scale, *, offset=None, pertoken_scale=None, bias=None,
    output_dtype=None, x1_dtype=None, x2_dtype=None,
    pertoken_scale_dtype=None, scale_dtype=None, group_sizes=None, y_scale=None,
):
    """MXFP8 matmul (rank-2), with optional FP32 [N] bias.

    Supported shapes: ``M`` and ``N`` are arbitrary positive integers (M/N tail
    tiles included) and ``K`` must be a multiple of ``G_BK`` (128). WeightNZ
    ``x2`` additionally needs ``N`` to be a multiple of the 32-column fractal.
    Over the whole supported domain the output satisfies the independent CPU FP64
    golden within the test tolerances.

    The three shapes listed in ``SHAPES`` have a dedicated per-shape schedule
    (split-K, fully A-resident, sliding window), each with its own L1 residency
    plan and launch geometry; every other shape takes the single shape-general
    schedule. Both are described in the module docstring, and the choice is
    transparent to the caller.
    """
    for name, value in (("offset", offset), ("x1_dtype", x1_dtype),
                        ("x2_dtype", x2_dtype), ("y_scale", y_scale)):
        if value is not None:
            raise NotImplementedError(f"{name} is not supported; pass None")
    if output_dtype != torch.bfloat16:
        raise ValueError("explicit output_dtype=torch.bfloat16 is required")
    if group_sizes is not None and list(group_sizes) != [1, 1, 32]:
        raise ValueError("MX group_sizes must be [1,1,32] or None")
    if x1.ndim != 2 or x2.ndim != 2:
        raise ValueError("only rank-2 NN QuantMatmul inputs are supported")
    m, k = (int(v) for v in x1.shape)
    n = int(x2.shape[1])
    if tuple(int(v) for v in x2.shape) != (k, n):
        raise ValueError(f"x2.shape must be (K,N)={(k, n)}, got {tuple(x2.shape)}")
    if min(m, n, k) < 1:
        raise ValueError(f"M/N/K must all be >= 1; got {(m, n, k)}")
    if x1.dtype not in (torch.float8_e5m2, torch.float8_e4m3fn) or x2.dtype != x1.dtype:
        raise TypeError("x1/x2 must have equal E5M2 or E4M3FN dtype")
    if pertoken_scale is None or scale is None:
        raise ValueError("both MX scales are required")
    if tuple(pertoken_scale.shape) != (m, k // 64, 2) or tuple(scale.shape) != (k // 64, n, 2):
        raise ValueError("MX scale shapes must be [M,K/64,2] and [K/64,N,2]")
    native = getattr(torch, "float8_e8m0fnu", None)
    for name, tensor, override in (("pertoken_scale", pertoken_scale, pertoken_scale_dtype),
                                   ("scale", scale, scale_dtype)):
        if override is None:
            if native is None or tensor.dtype != native:
                raise TypeError(f"{name} must be native E8M0 without a dtype override")
        else:
            import torch_npu
            carrier_dtype = getattr(torch_npu, "float8_e8m0fnu", None)
            if not ((native is not None and override == native)
                    or (carrier_dtype is not None and override == carrier_dtype)):
                raise TypeError(f"{name}_dtype must specify FLOAT8_E8M0")
            if tensor.dtype != torch.uint8 and (native is None or tensor.dtype != native):
                raise TypeError(f"{name} must carry native E8M0 or uint8 scale bits")
    if bias is not None and (bias.dtype != torch.float32 or tuple(bias.shape) != (n,)):
        raise ValueError("bias must be None or FP32 [N]")
    tensors = (x1, x2, pertoken_scale, scale) + ((bias,) if bias is not None else ())
    if x1.device.type != "npu" or any(t.device != x1.device for t in tensors):
        raise ValueError("all inputs must be on the same NPU")
    import torch_npu
    formats = tuple(torch_npu.get_npu_format(t) for t in tensors)
    weight_nz_input = formats[1] == 29
    if any(fmt not in (0, 2) for i, fmt in enumerate(formats) if i != 1):
        raise ValueError("x1, pertoken_scale, scale and bias must be ND; ScaleNZ is not supported")
    if weight_nz_input:
        if x1.dtype != torch.float8_e4m3fn:
            raise TypeError("WeightNZ supports only equal FP8_E4M3FN x1/x2")
        if x2.storage_offset() != 0:
            raise ValueError("WeightNZ x2 must use unshifted physical storage")
    elif formats[1] not in (0, 2):
        raise ValueError("x2 must be ND or FRACTAL_NZ (29)")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("only NN logical strides are supported; no transposed views")
    # One kernel serves the whole supported domain; the Profile tiling-key field
    # selects the schedule at compile time, so a compiled variant contains only
    # its own branch. The three measured shapes keep their per-shape schedules
    # (split-K / A-resident / sliding window); every other shape takes Profile 3,
    # which walks Profile 2's tiles over the shape with M/N tail narrowing, a
    # real K-window loop and a grid-stride tile loop.
    if k % G_BK != 0:
        raise ValueError(f"K must be a multiple of BK={G_BK}; got K={k}")
    if weight_nz_input and n % 32 != 0:
        raise ValueError(
            "WeightNZ packs x2 in 32-column fractals, so N must be a multiple "
            f"of 32; got N={n}")
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x1.device)
    key = {"InputFormat": 0 if x1.dtype == torch.float8_e5m2 else 1,
           "HasBias": int(bias is not None), "WeightNz": int(weight_nz_input)}
    if (m, n, k) in SHAPES:
        profile = SHAPES.index((m, n, k))
        workers = 20 if profile == 2 else 32
        workspace = torch.empty((7, 4, M, 256), dtype=torch.float32,
                                device=x1.device) if profile == 0 else None
    else:
        profile = 3
        tiles = -(-m // G_BM) * -(-n // G_BN)
        workers = max(1, min(tiles, 32))
        workspace = None
    key["Profile"] = profile
    # The no-bias and no-workspace specializations never read those pointers, so
    # a live input pointer is passed as an unused ABI placeholder instead of
    # allocating one.
    _quant_matmul_mxfp8_kernel[None, workers, key](
        x1, x2, x2 if weight_nz_input else None, pertoken_scale, scale,
        bias if bias is not None else x1,
        workspace if workspace is not None else x1, out, m, n, k,
    )
    # Launch is asynchronous. Bias rides on the first MX MMAD, so a biased call
    # allocates no accumulator-preset buffer; only the split-K profile needs an
    # internal GM workspace, and that is the only buffer needing a keepalive
    # attribute on the returned tensor.
    if workspace is not None:
        out._mxfp8_keepalive = workspace
    return out
