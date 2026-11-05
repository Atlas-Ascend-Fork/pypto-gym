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

"""A5 MXFP8 batch matmul.

Computes, per batch element,

    out[b] = BF16( dequant(x1[b]) @ dequant(x2[b]) + bias )

where x1/x2 are FP8 (E5M2 or E4M3FN, both operands the same type) and every
group of 32 consecutive K elements shares one E8M0 scale (byte u means
2**(u-127)). Scales are carried packed as [..., K/64, 2], the last axis holding
the two 32-element groups inside one 64-element window.

Supported shapes: BATCH, M and N are arbitrary positive integers (M/N tail
tiles included); K is a multiple of BK = 128. x2 may be ND or WeightNZ
(format 29, E4M3FN only); x1, both scales, bias and the output are always ND,
and only NN logical strides are accepted. bias is optional FP32, either [N]
(broadcast over B and M) or [B,1,N] (per batch, broadcast over M). Output is
BF16 and is the only device allocation.

Tile geometry and on-chip addresses are compile-time constants:
BM/BN/BK = 96/256/128, KL1 = 256, SCALE_KL1 = 4096, four L1 data slots, two
full-window scale slots, two L0 slots and two Acc slots. SCALE_KL1 spans 4096 K
elements, so the scale-window loop runs ceil(K/4096) times and each window holds
a full-window E8M0 slice; inside a window, data is staged in KL1 = 256 chunks
and each chunk feeds two BK = 128 MX products. The L1 data/scale/bias offsets
split L1 into two halves (A slots at +0/+262144 and +24576/+286720, B slots at
+49152/+311296 and +114688/+376832, scales at 180224/442368 and 192512/454656,
bias at 225280/487424); L0 data uses half of L0A/L0B, the MX scale planes sit at
data_address >> 4, and Acc uses half of L0C.

One output tile is [BM, BN] = [96, 256]. The tile count is
BATCH * ceil(M/96) * ceil(N/256); the launch uses min(count, 32) Cube workers
and each worker walks its own tiles with a grid-stride loop, so counts above 32
take several waves. Tail tiles narrow M and/or N with pl.set_validshape, and a
partial trailing K window narrows the same way.

Accumulation order is part of the numerical contract: the first MX MMAD of an
output tile initializes the FP32 accumulator -- consuming the FP32 BiasTable
operand directly when bias is present, i.e. init_with_zero=false -- the
remaining steps accumulate in ascending K, the last step carries
AccPhase.Final, and the tile is stored with STPhase.Final as BF16 (RNE). When
K == BK the first step is also the last, so it carries Final. GM->L1 load order
within an iteration is scale, A, bias (first iteration only), B; the L1->Bias
move sits after the A/scaleA extraction and before the B/scaleB extraction. The
cube section runs under pl.system.set_mm_layout_transform(enabled=True)
(column-first fixpipe L0C drain) and restores False after the final store.

Both x2 loads pass l2_cache_mode=B_LOAD_L2_HINT so the weight stream does not
allocate in L2; no other tensor is hinted.

Synchronization is automatic (auto_mutex=True): dependencies are derived from
TileGroup mutex ownership rather than explicit event ids. No runtime tiling
calculation, platform query, SDK ini read or resource-plan construction is
performed.
"""

import torch
import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField

BM, BN, BK = 96, 256, 128
# KL1 = 256 is the GM->L1 data staging granularity: one L1 data panel holds
# [BM, 256] of x1 and [256, BN] of x2, and feeds two BK = 128 MX products.
# SCALE_KL1 = 4096 is the E8M0 scale window: one scale slot holds a
# [BM, 128] (A) or [128, BN] (B) window covering 4096 K elements, so the window
# loop runs ceil(K/4096) times. A wide window is what lets the two scale slots
# stay resident across all KL1 chunks of a tile; K = 4096, the shape the
# performance cases use, needs exactly one window.
KL1, SCALE_KL1 = 256, 4096

# Both x2 loads pass TLoadL2Hint.NotAllocKeep, i.e. the weight stream is not
# allocated in L2. x2 is the dominant traffic and is streamed once per output
# tile: at B/M/N/K = 8/96/1024/4096 it is 33.55 MB of the 39.45 MB logical
# total, against 128 MiB of L2, so letting it allocate would displace the
# tensors that are actually reused across tiles. Only x2 is hinted -- x1, both
# scales and bias are small and reused, and the output store keeps the default
# policy. l2_cache_mode is a compile-time lowering attribute (it becomes a
# TLOAD<TLoadL2Hint::...> template argument), so one value applies to every
# tile, N tail tiles included.
B_LOAD_L2_HINT = pl.TLoadL2Hint.NotAllocKeep


class MxKey:
    # Only the FP8 type, the presence of bias and the x2 storage format select
    # different device code; no shape value enters the key.
    InputFormat = TilingKeyField(bits=1, values=[0, 1])
    HasBias = TilingKeyField(bits=1, values=[0, 1])
    WeightNz = TilingKeyField(bits=1, values=[0, 1])


# ---------------------------------------------------------------------------
# The single device kernel: arbitrary BATCH / M / N, with K a multiple of BK.
#
# Tile geometry and the L1/L0 address table are compile-time constants
# (BM/BN/BK, KL1/SCALE_KL1 and the immediate offsets below); only the walk over
# the shape is dynamic:
#
#   * M/N tails: pl.set_validshape narrows the L1, L0, scale-plane, bias and Acc
#     tiles. Cube accepts a partial M and/or N down to a single 16x16 fractal,
#     so no host-side padding is needed.
#   * K tails: a real scale-window loop with scale_len/data_len =
#     pl.min(window, remaining), so K need not equal SCALE_KL1 and need not be a
#     multiple of KL1.
#   * arbitrary tile counts: a grid-stride loop over BATCH*m_tiles*n_tiles, so
#     the launch geometry need not be exactly one full wave.
#
# Shapes are runtime values rather than tiling-key fields, so one compiled
# variant per key serves every shape. When the tile count is <= 32 each worker
# takes exactly one tile and the grid-stride loop runs a single wave, which is
# the shape the performance cases use.
# ---------------------------------------------------------------------------
@pl.jit(arch="a5", auto_mutex=True, tiling_key=MxKey)
def _quant_batch_matmul_mxfp8_kernel(
    x1_ptr: pl.Ptr[pl.DT_UINT8],
    x2_ptr: pl.Ptr[pl.DT_UINT8],
    x2_nz: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP8E4M3FN, pl.NZ],
    x1_scale: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
    x2_scale: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
    bias_ptr: pl.Ptr[pl.DT_UINT8],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    bias_per_batch: pl.DT_INT64,
):
    if InputFormat == 0:
        fp8 = pl.DT_FP8E5M2
    else:
        fp8 = pl.DT_FP8E4M3FN
    # Shapes are read from tensors that are always present (x2_nz is None on the
    # ND path, so it must not be queried). K is recovered from the packed scale
    # window count: x1_scale is [BATCH, M, K/64, 2].
    batch = out.shape[0]
    m = out.shape[1]
    n = out.shape[2]
    k = x1_scale.shape[2] * 64
    x1 = pl.make_tensor(x1_ptr, [batch, m, k], dtype=fp8)
    x2 = pl.make_tensor(x2_ptr, [batch, k, n], dtype=fp8)
    if HasBias == 1:
        bias = pl.make_tensor(bias_ptr, [1 + (batch - 1) * bias_per_batch, n],
                              dtype=pl.DT_FP32)

    # L1 layout, derived from the tile sizes. One slot of each buffer is
    # aOne = BM*KL1 = 24576, bOne = KL1*BN = 65536, scaleAOne = 96*128 = 12288,
    # scaleBOne = 256*128 = 32768 and biasOne = BN*4 = 1024 bytes. L1 is used as
    # two halves 262144 bytes apart, each holding two data slots plus one copy of
    # every scale/bias slot:
    #   first half   A 0..24575, 24576..49151 | B 49152..114687, 114688..180223
    #                scaleA 180224, scaleB 192512, bias 225280
    #   second half  the same five buffers at +262144
    # so the four A slots are 0/24576/262144/286720, the four B slots are
    # 49152/114688/311296/376832, and scaleA/scaleB/bias are the depth-2 pairs
    # 180224+442368, 192512+454656 and 225280+487424. Every TileType carries
    # valid_shape=[-1, -1] so a tail tile can narrow M and/or N in place.
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[BM, KL1], dtype=fp8, target_memory=pl.MemorySpace.Mat,
                         valid_shape=[-1, -1]),
        addrs=[0, 262144, 24576, 286720], mutex_ids="auto", depth=4,
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[KL1, BN], dtype=fp8, target_memory=pl.MemorySpace.Mat,
                         valid_shape=[-1, -1]),
        addrs=[49152, 311296, 114688, 376832], mutex_ids="auto", depth=4,
    )
    # Each scale slot holds one SCALE_KL1-wide E8M0 window, so the window loop
    # runs ceil(K/SCALE_KL1) times and the two slots rotate across windows. When
    # K <= SCALE_KL1 -- the performance cases use K = 4096 -- there is a single
    # window, it is loaded once per output tile and the cursor stays on slot 0.
    sa_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[BM, SCALE_KL1 // 32], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZZ,
                         valid_shape=[-1, -1]),
        addrs=[180224, 442368], mutex_ids="auto", depth=2,
    )
    sb_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[SCALE_KL1 // 32, BN], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NN,
                         valid_shape=[-1, -1]),
        addrs=[192512, 454656], mutex_ids="auto", depth=2,
    )
    a_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[BM, BK], dtype=fp8, target_memory=pl.MemorySpace.Left,
                         layout=pl.NZ, compact=1, valid_shape=[-1, -1]),
        addrs=[0, 32768], mutex_ids="auto", depth=2,
    )
    b_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[BK, BN], dtype=fp8, target_memory=pl.MemorySpace.Right,
                         layout=pl.ZN, compact=1, valid_shape=[-1, -1]),
        addrs=[0, 32768], mutex_ids="auto", depth=2,
    )
    plane_a = pl.make_tile_group(
        type=pl.TileType(shape=[BM, BK // 32], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.ScaleLeft, layout=pl.ZZ,
                         compact=1, valid_shape=[-1, -1]),
        addrs=[0, 2048], mutex_ids="auto", depth=2,
    )
    plane_b = pl.make_tile_group(
        type=pl.TileType(shape=[BK // 32, BN], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.ScaleRight, layout=pl.NN,
                         compact=1, valid_shape=[-1, -1]),
        addrs=[0, 2048], mutex_ids="auto", depth=2,
    )
    accumulator = pl.make_tile_group(
        type=pl.TileType(shape=[BM, BN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         layout=pl.NZ, fractal=1024, compact=1, valid_shape=[-1, -1]),
        addrs=[0, 131072], mutex_ids="auto", depth=2,
    )
    if HasBias == 1:
        bias_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[1, BN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat,
                             valid_shape=[-1, -1]),
            addrs=[225280, 487424], mutex_ids="auto", depth=2,
        )
        bias_bt = pl.make_tile_group(
            type=pl.TileType(shape=[1, BN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias,
                             valid_shape=[-1, -1]),
            addrs=[0, 1024], mutex_ids="auto", depth=2,
        )

    m_tiles = (m + BM - 1) // BM
    n_tiles = (n + BN - 1) // BN
    total_tiles = batch * m_tiles * n_tiles

    with pl.section_cube():
        # The fixpipe drains L0C column-first (N direction) for the whole cube
        # section; restored to False after the final store.
        pl.system.set_mm_layout_transform(enabled=True)
        core_id = pl.get_block_idx()
        num_cores = pl.get_block_num()
        # Grid-stride: one output tile per iteration, any number of waves. Tiles
        # are decoded batch-major over the M blocks and then the N blocks, so a
        # worker's tiles are contiguous in N. With total_tiles <= num_cores the
        # loop body runs once per worker and each owns exactly one [BM, BN] tile;
        # at BATCH/M/N/K = 8/96/1024/4096 that is 8 * 1 * 4 = 32 tiles over 32
        # workers, i.e. batch-major over the four N blocks.
        for tile_id in pl.range(core_id, total_tiles, num_cores):
            batch_idx = tile_id // (m_tiles * n_tiles)
            rem = tile_id % (m_tiles * n_tiles)
            m_offset = (rem // n_tiles) * BM
            n_offset = (rem % n_tiles) * BN
            vm = pl.min(BM, m - m_offset)
            vn = pl.min(BN, n - n_offset)
            acc = accumulator.next()
            pl.set_validshape(acc, [vm, vn])
            if HasBias == 1:
                bias_mat = bias_l1.next()
                bias_table = bias_bt.next()
                pl.set_validshape(bias_mat, [1, vn])
                pl.set_validshape(bias_table, [1, vn])
            # Per output tile and per scale window: load the window's scales
            # once, then for each KL1 chunk load A, the bias (first chunk of the
            # first window only) and B.
            for scale_k in pl.range(0, k, SCALE_KL1):
                scale_len = pl.min(SCALE_KL1, k - scale_k)
                sal1 = sa_l1.next()
                sbl1 = sb_l1.next()
                pl.set_validshape(sal1, [vm, scale_len // 32])
                pl.set_validshape(sbl1, [scale_len // 32, vn])
                pl.load(sal1, x1_scale, [batch_idx, m_offset, scale_k // 64, 0])
                pl.load(sbl1, x2_scale, [batch_idx, scale_k // 64, n_offset, 0])
                for data_k in pl.range(scale_k, scale_k + scale_len, KL1):
                    data_len = pl.min(KL1, k - data_k)
                    al1 = a_l1.next()
                    bl1 = b_l1.next()
                    pl.set_validshape(al1, [vm, data_len])
                    pl.set_validshape(bl1, [data_len, vn])
                    pl.load(al1, x1, [batch_idx, m_offset, data_k])
                    if HasBias == 1 and scale_k == 0 and data_k == 0:
                        pl.load(bias_mat, bias, [batch_idx * bias_per_batch, n_offset])
                    if WeightNz == 1:
                        # Native GM NZ -> L1 NZ copy, addressed by logical K/N
                        # offsets; no numeric conversion and no staging buffer.
                        pl.load(bl1, x2_nz, [batch_idx, data_k, n_offset],
                                l2_cache_mode=B_LOAD_L2_HINT)
                    else:
                        pl.load(bl1, x2, [batch_idx, data_k, n_offset],
                                l2_cache_mode=B_LOAD_L2_HINT)
                    for l0_k in pl.range(0, data_len, BK):
                        al0 = a_l0.next()
                        pal0 = plane_a.next()
                        bl0 = b_l0.next()
                        pbl0 = plane_b.next()
                        pl.set_validshape(al0, [vm, BK])
                        pl.set_validshape(pal0, [vm, BK // 32])
                        pl.set_validshape(bl0, [BK, vn])
                        pl.set_validshape(pbl0, [BK // 32, vn])
                        # L1->L0 extraction offsets are in elements of the L1
                        # tile. The E8M0 planes are addressed independently of the
                        # data: one scale covers 32 K values, so the plane
                        # column/row index is the group index inside the current
                        # scale window, i.e. (data_k - scale_k + l0_k) // 32.
                        s_off = (data_k - scale_k + l0_k) // 32
                        pl.move(al0, al1, [0, l0_k])
                        pl.move(pal0, sal1, [0, s_off])
                        # Order within the first L0 iteration: A data + A scale,
                        # then the L1->BiasTable move, then B data + B scale, so
                        # the BiasTable is in place before the first MX MMAD. The
                        # move touches neither L0A nor L0B, so it has no ordering
                        # constraint against the extractions -- only against that
                        # first MMAD.
                        if HasBias == 1:
                            if scale_k == 0 and data_k == 0 and l0_k == 0:
                                pl.move(bias_table, bias_mat)
                        pl.move(bl0, bl1, [l0_k, 0])
                        pl.move(pbl0, sbl1, [s_off, 0])
                        # data_k is ABSOLUTE (the loop starts at scale_k), so the
                        # absolute K offset of this BK step is data_k + l0_k, while
                        # the E8M0 column inside the current scale window is s_off.
                        k_offset = data_k + l0_k
                        # "First step" and "last step" are tested as nested
                        # conditions, not as an if/elif chain, because K == BK
                        # makes them the same step: that step must carry Final,
                        # otherwise the accumulator is never finalized and the
                        # STPhase.Final store has nothing to wait for. The dynamic
                        # tests are kept outside the compile-time HasBias split.
                        last_step = k_offset + BK == k
                        # The first MX MMAD initializes the FP32 accumulator. With
                        # bias it consumes the FP32 BiasTable in the same
                        # instruction (init_with_zero=false), so no separate
                        # accumulator-preset MMAD is needed; matmul_mx_acc has no
                        # bias operand, hence bias only ever rides on this one.
                        if k_offset == 0:
                            if last_step:
                                if HasBias == 1:
                                    pl.matmul_mx(acc, al0, bl0, pal0, pbl0, bias_table,
                                                 phase=pl.AccPhase.Final)
                                else:
                                    pl.matmul_mx(acc, al0, bl0, pal0, pbl0,
                                                 phase=pl.AccPhase.Final)
                            else:
                                if HasBias == 1:
                                    pl.matmul_mx(acc, al0, bl0, pal0, pbl0, bias_table,
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
            # L0C -> GM, BF16 RNE, one store per output tile. STPhase.Final waits
            # for the AccPhase.Final accumulation above, which is why a K == BK
            # tile must still emit Final on its only step.
            pl.store(out, acc, [batch_idx, m_offset, n_offset], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)


def quant_batch_matmul_mxfp8(
    x1, x2, scale, *, offset=None, pertoken_scale=None, bias=None,
    output_dtype=None, x1_dtype=None, x2_dtype=None,
    pertoken_scale_dtype=None, scale_dtype=None, group_sizes=None, y_scale=None,
):
    """MXFP8 batch matmul.

    Supported shapes: ``BATCH``, ``M`` and ``N`` are arbitrary positive integers
    (M/N tail tiles included) and ``K`` must be a multiple of ``BK`` (128).
    WeightNZ ``x2`` additionally needs ``N`` to be a multiple of the 32-column
    fractal. Over the whole supported domain the output satisfies the independent
    CPU FP64 golden within the test tolerances.

    ``output_dtype`` must be explicitly ``torch.bfloat16``; INT8 output is not
    supported. ``group_sizes`` accepts ``[1,1,32]`` or ``None``, the latter
    inferring the fixed MX groups ``[1,1,32]``. Scale dtype overrides accept
    native E8M0 or torch_npu's E8M0 identifier for raw uint8 carriers.
    Unsupported optional inputs raise rather than being silently ignored.
    """
    for name, value in (("offset", offset), ("x1_dtype", x1_dtype),
                        ("x2_dtype", x2_dtype), ("y_scale", y_scale)):
        if value is not None:
            raise NotImplementedError(f"{name} is not supported for these MXFP8 cases; pass None")
    if output_dtype != torch.bfloat16:
        raise ValueError("only output_dtype=torch.bfloat16 is supported; it must be explicit")
    if group_sizes is not None:
        group_sizes_ok = (isinstance(group_sizes, (list, tuple)) and len(group_sizes) == 3
                          and all(type(v) is int for v in group_sizes)
                          and tuple(group_sizes) == (1, 1, 32))
        if not group_sizes_ok:
            raise ValueError("group_sizes must be [1,1,32] or None (infer MX groups)")
    if pertoken_scale is None:
        raise ValueError("pertoken_scale is required for these MXFP8 cases")
    tensors = (x1, x2, pertoken_scale, scale) + ((bias,) if bias is not None else ())
    # Derive the problem size from x1/x2, then require every other tensor to
    # agree with it.
    if x1.ndim != 3:
        raise ValueError(f"x1 must be rank-3 [BATCH,M,K]; got rank {x1.ndim}")
    if x2.ndim != 3:
        raise ValueError(f"x2 must be rank-3 [BATCH,K,N]; got rank {x2.ndim}")
    batch, m, k = (int(v) for v in x1.shape)
    batch2, k2, n = (int(v) for v in x2.shape)
    if (batch2, k2) != (batch, k):
        raise ValueError(f"x2.shape[:2] must equal (BATCH,K)={(batch, k)}, got {(batch2, k2)}")
    if min(batch, m, n, k) < 1:
        raise ValueError(f"BATCH/M/N/K must all be >= 1; got {(batch, m, n, k)}")
    expected_shapes = ((batch, m, k), (batch, k, n),
                       (batch, m, k // 64, 2), (batch, k // 64, n, 2))
    for name, tensor, shape in zip(("x1", "x2", "pertoken_scale", "scale"),
                                   tensors[:4], expected_shapes):
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name}.shape must be {shape}, got {tuple(tensor.shape)}")
    if x1.dtype not in (torch.float8_e5m2, torch.float8_e4m3fn) or x2.dtype != x1.dtype:
        raise TypeError("x1/x2 must have the same E5M2 or E4M3FN dtype")
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    for name, tensor, override in (("pertoken_scale", pertoken_scale, pertoken_scale_dtype),
                                   ("scale", scale, scale_dtype)):
        if override is None:
            if e8m0 is None or tensor.dtype != e8m0:
                raise TypeError(f"{name} must be native E8M0 when its dtype override is omitted")
        else:
            import torch_npu
            npu_e8m0 = getattr(torch_npu, "float8_e8m0fnu", None)
            if not ((e8m0 is not None and override == e8m0)
                    or (npu_e8m0 is not None and override == npu_e8m0)):
                raise TypeError(f"{name}_dtype must specify FLOAT8_E8M0")
            if tensor.dtype != torch.uint8 and (e8m0 is None or tensor.dtype != e8m0):
                raise TypeError(f"{name} must carry E8M0 bits as native E8M0 or uint8")
    if bias is not None and (bias.dtype != torch.float32
                             or tuple(bias.shape) not in ((n,), (batch, 1, n))):
        raise ValueError(f"bias must be None, FP32 [{n}] or FP32 [{batch},1,{n}]")
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

    # One device path serves the whole supported domain. The kernel walks
    # batch*ceil(M/BM)*ceil(N/BN) output tiles with a grid-stride loop, narrows
    # M/N tail tiles with pl.set_validshape and contracts a partial trailing K
    # window with pl.min, so no shape value reaches the tiling key.
    if k % BK != 0:
        raise ValueError(f"K must be a multiple of BK={BK}; got K={k}")
    if weight_nz_input and n % 32 != 0:
        raise ValueError(
            "WeightNZ packs x2 in 32-column fractals, so N must be a multiple "
            f"of 32; got N={n}")

    out = torch.empty((batch, m, n), dtype=torch.bfloat16, device=x1.device)
    bias_per_batch = int(bias is not None and bias.ndim == 3)
    # The no-bias specialization never reads the bias pointer, so a live input
    # pointer is passed as an unused ABI placeholder instead of allocating one.
    bias_arg = bias.view(-1, n) if bias is not None else x1
    key = {"InputFormat": 0 if x1.dtype == torch.float8_e5m2 else 1,
           "HasBias": int(bias is not None), "WeightNz": int(weight_nz_input)}
    args = (x1, x2, x2 if weight_nz_input else None,
            pertoken_scale, scale, bias_arg, out, bias_per_batch)
    tiles = batch * -(-m // BM) * -(-n // BN)
    _quant_batch_matmul_mxfp8_kernel[None, max(1, min(tiles, 32)), key](*args)
    # Bias rides on the first MX MMAD, so a biased call allocates no internal GM
    # buffer: the output tensor is the only device allocation, and one public call
    # emits exactly one device task.
    return out
