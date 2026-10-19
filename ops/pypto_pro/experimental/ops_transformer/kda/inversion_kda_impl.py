# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""inversion_kda — cube-based Neumann series inversion (block-sparse, on-chip).

Computes (I + L)^{-1} for each [128, 128] strictly lower triangular chunk
using a full-matrix Neumann series with repeated doubling, accelerated by
128x128 cube matmul:

    Y = L                                      # NOT -L: sign is absorbed at round 0
    X = I
    for round in range(NUM_ROUNDS):            # log2(CS) = 7 for CS=128
        if round + 1 < NUM_ROUNDS:             # last round: Y_sq is dead (= L^CS = 0)
            Y_sq = Y @ Y                        # cube matmul (block-sparse: BBxBB tiles,
                                               #   skip zero blocks via I>=K>=J coord)
        XY = X @ Y                             # cube matmul
        X  = (X - XY) if round == 0 else (X + XY)   # vec sub/add
        Y  = Y_sq                              # (skipped on last round)

Sign note: (I+L)^-1 = sum (-L)^i.  With M=-L, M^(2^k)=L^(2^k) for k>=1 (even
power), so only the M^1 term is negative.  We therefore keep Y=L (no up-front
negate) and let round 0 subtract; every later round adds.  Result is identical
to the M=-L formulation.

After NUM_ROUNDS rounds: X = sum_{i=0}^{CS-1} (-L)^i = (I + L)^{-1}.  This works
because L is strictly lower triangular (nilpotent: L^128 = 0), so the series
terminates exactly.

Everything is ON-CHIP (no GM workspaces except ws_ysq): X (partial sum) lives
resident in UB, Y and the X-mirror live resident in L1.  Round 0 loads both
matmul operands straight GM->L1 (identity into x_l1, L=A into y_l1; the GM->L1
load does ND->NZ on the fly, so no Vec seed and no separate Init section —
X0=I is seeded into UB inside round 0's Vec phase).  Rounds >=1 feed the
previous round's results back to L1 via an L0C->UB (fix) drain + a
UB->UB(NZ)->L1(insert) 3-hop, keeping everything fp32.  L0C (acc) is
double-buffered so mm2's matmul overlaps mm1's drain.

Cross-core sync (set_cross_core / wait_cross_core) coordinates the Cube<->Vec
pipe transitions.  Events: e0 y_l1-ready, e3 x_l1-ready (Vec->Cube, round>=1),
e1 Y@Y-drained, e2 X@Y-drained (Cube->Vec).

Performance: 13 cube matmuls (7 rounds x 2, minus the dead Y@Y on the last
round) in a single kernel launch.
"""

import torch
import pypto_pro.language as pl

from .kda_common import C, DEVICE as _DEVICE

CS = C
CS_HALF = CS // 2   # matmul results are drained splitM: each Vec subblock owns [CS_HALF, CS]
NUM_ROUNDS = 7

# ---- L1 (Mat) addresses ---- (full [CS, CS] operands, reassembled from both subblocks)
L1_A = 0x00000   # [CS, CS] fp32 NZ  2x64 KB — y_l1 (double-buffered: 0x0 and 0x10000)
L1_B = 0x20000   # [CS, CS] fp32 NZ  64 KB  — x_l1 (moved up to clear y_l1's 2nd buffer)

# ---- UB (Vec) addresses ---- (WHOLE 128x128 tiles; footprint = 3*64 = 192 KB < 248)

UB_X   = 0x00000   # x_ub    [CS_HALF, CS] fp32  32 KB  — resident X half (subblock-private)
UB_XY  = 0x08000   # xy_ub   [CS_HALF, CS] fp32  32 KB  — X@Y row-band half (per subblock)
# (0x08000 region free)
UB_NZ  = 0x10000   # stage   [CS_HALF, CS] fp32  32 KB  — NZ staging for X->x_l1 insert

BLOCK_STRIDE_ND = CS >> 1 | 0x1
REPEAT_STRIDE_ND = 1


@pl.vector_function
def nd2nz_sg(tile, out, m):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for i in pl.range(m):
        inreg = vf.load_align(tile, i * CS)
        vf.store_align(out, inreg, preg_all,
                block_stride=BLOCK_STRIDE_ND, repeat_stride=REPEAT_STRIDE_ND,
                data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)


@pl.vector_function
def nd2nz(tile, out, m):
    out_unroll = out + BLOCK_STRIDE_ND * (CS >> 1)
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for i in pl.range(m):
        inreg = vf.load_align(tile, i * CS)
        inreg_unroll = vf.load_align(tile, i * CS + 64)
        vf.store_align(out, inreg, preg_all,
                block_stride=BLOCK_STRIDE_ND, repeat_stride=REPEAT_STRIDE_ND,
                data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
        vf.store_align(out_unroll, inreg_unroll, preg_all,
                block_stride=BLOCK_STRIDE_ND, repeat_stride=REPEAT_STRIDE_ND,
                data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)


@pl.jit(auto_mutex=True)
def inversion_kda_cube_kernel(
    A: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    identity: pl.Tensor[[CS, CS], pl.DT_FP32],
    A_inv: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    ws_ysq: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    cu_seqlens: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
):
    """Fused Neumann series inversion in a single kernel.

    All fp32, all ON-CHIP: X (partial sum) resident in UB; Y and the X-mirror
    resident in L1. Round 0 loads both operands straight GM->L1 (identity->x_l1,
    L=A->y_l1) and seeds X0=I into UB; rounds >=1 feed results back L0C->UB (fix)
    then UB->UB(NZ)->L1(insert). Neumann uses Y=L with a round-0 subtract (see
    module docstring). L0C double-buffered for mm1-drain / mm2-matmul overlap.

    Inputs:
        A:          [B, H, T, CS]  fp32   strictly lower triangular L (BNTD, GM)
        identity:   [CS, CS]       fp32   +I constant (GM), seeds X0 = I / x_l1
        cu_seqlens: [num_seqs+1]   int32  cumulative sequence lengths (GM)

    Output:
        A_inv:  [B, H, T, CS]  fp32   (I + L)^{-1} (BNTD, GM)
    """
    H = A.shape[1]
    num_seqs = cu_seqlens.shape[0] - 1
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()

    # ── Cube tile groups ──
    # y_l1 / x_l1: resident L1 operands (Y and the X mirror), refreshed each round on-chip.
    # y_l1: double-buffered (2) so round k+1's GM->L1 load overlaps round k's matmul
    # reads of the previous buffer. fp32 [CS,CS]=64KB x2 = 128KB (L1=512KB, fits).
    y_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CS, CS], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                         valid_shape=[-1, -1]),
        addrs=L1_A, mutex_ids=[0, 1])
    x_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CS, CS], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=L1_B, mutex_ids=[2])
    # ── 64x64 block tiles: BOTH matmuls (Y@Y and X@Y) are block-sparse, so the old
    # 128x128 a_l0a/b_l0b/acc are gone.  These own L0 outright — no aliasing, no stomp.
    BB = 32                # L0 block base tile
    NB = CS // BB          # blocks per side (2 for BB=64)
    a_blk = pl.make_tile_group(   # Left/L0A: 64x64 fp32 = 16KB x2 buffers = 32KB (L0A 64KB)
        type=pl.TileType(shape=[BB, BB], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ, compact=1),
        addrs=0x0, mutex_ids=[3, 4, 5, 6])
    b_blk = pl.make_tile_group(   # Right/L0B
        type=pl.TileType(shape=[BB, BB], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN, compact=1),
        addrs=0x0, mutex_ids=[7, 8, 9, 10])
    # acc_blk: one L0C buffer per output block (NBLK = NB*(NB+1)/2) so a matmul section
    # never reuses a buffer whose store(FIX) is still draining. acc_blk owns all of L0C now.
    acc_blk = pl.make_tile_group(  # Acc/L0C: BBxBB fp32; NBLK buffers
        type=pl.TileType(shape=[BB, BB], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         fractal=1024, compact=1),
        addrs=0x0, mutex_ids=(11, 12, 13, 14))

    # ── Vec tile groups ── (splitM half-tiles: each Vec subblock owns [CS_HALF, CS])
    # x_ub: resident X half (partial sum), updated in place across rounds (Vec-only).
    # xy_ub: Acc→UB splitM drain of X@Y. (Y@Y no longer drains to UB — goes L0C->GM.)
    # stage_nz: one NZ staging tile, reused for both UB→L1 inserts (Y then X) each round.
    x_ub = pl.make_tile_group(   # [CS_HALF, CS] subblock-private resident X row-band
        type=pl.TileType(shape=[CS_HALF, CS], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         pad=pl.TilePad.zero, valid_shape=[-1, -1]),
        addrs=UB_X, mutex_ids=[15])
    xy_ub = pl.make_tile_group(  # [CS_HALF, CS] per-subblock X@Y row-band (col slices)
        type=pl.TileType(shape=[CS_HALF, CS], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         pad=pl.TilePad.zero, valid_shape=[-1, -1]),
        addrs=UB_XY, mutex_ids=[16])
    # stage_nz: NZ staging, reused sequentially for the Y and X UB->L1 inserts each round.
    stage_nz = pl.make_tile_group(  # [CS_HALF, CS] NZ staging for X half -> x_l1 insert
        type=pl.TileType(shape=[CS_HALF + 1, CS], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
        addrs=UB_NZ, mutex_ids=[17])

    # ── Compute num_chunks by scanning cu_seqlens ──
    num_chunks = 0
    for seq_idx in pl.range(0, num_seqs):
        seq_start = pl.getval(cu_seqlens, seq_idx)
        seq_end = pl.getval(cu_seqlens, seq_idx + 1)
        seq_len = seq_end - seq_start
        seq_num_chunks = (seq_len + CS - 1) // CS
        num_chunks = num_chunks + seq_num_chunks

    total_work = num_chunks * H

    # ── Work loop ──
    for work_id in pl.range(core_id, total_work, num_cores):
        chunk_id = work_id // H
        head_id = work_id % H

        # ── Scan cu_seqlens for row_start and valid_size ──
        accumulated_chunks = 0
        row_start = 0
        valid_size = CS
        found = 0
        for seq_idx in pl.range(0, num_seqs):
            if found == 0:
                seq_start = pl.getval(cu_seqlens, seq_idx)
                seq_end = pl.getval(cu_seqlens, seq_idx + 1)
                seq_len = seq_end - seq_start
                seq_num_chunks = (seq_len + CS - 1) // CS
                if chunk_id < accumulated_chunks + seq_num_chunks:
                    local_chunk = chunk_id - accumulated_chunks
                    row_start = seq_start + local_chunk * CS
                    remaining = seq_end - row_start
                    valid_size = pl.min(remaining, CS)
                    found = 1
                accumulated_chunks = accumulated_chunks + seq_num_chunks
        with pl.section_vector():
            pl.expands(xy_ub.current(), 0.0)

        # ================================================================
        #  Neumann rounds: Cube (Y_sq, XY) → Vec (X+XY, Y=Y_sq)
        #  X0 = I is seeded inside round-0's Vec section (below), not a separate Init.
        # ================================================================
        for round in pl.range(NUM_ROUNDS):
            # Last round: Y_sq = Y@Y = M^CS = 0 and the Y=Y_sq copy are both dead
            # (Y is never read after this round — only X is stored out), so skip them.
            is_last = (round + 1 == NUM_ROUNDS)

            # ── Cube: Y_sq = Y @ Y (skipped on last round),  XY = X @ Y ──
            with pl.section_cube():
                y = y_l1.next()   # rotate to the other L1 buffer this round (DB)
                base = core_id * CS   # this core's row band in the GM workspaces

                if round == 0:
                    pl.set_validshape(y, [valid_size, CS])
                    pl.load(y, A, [0, head_id, row_start, 0], order=[2, 3])
                    # pl.set_validshape(y, [CS, CS])
                else:
                    pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.set_validshape(y, [valid_size, CS])
                    pl.load(y, ws_ysq, [core_id * CS, 0])
                # Y_sq = Y @ Y (skipped on last round), block-sparse -> ws_ysq (GM).
                if not is_last:
                    # Y_sq = Y @ Y, block-sparse over an NBxNB grid of BBxBB blocks.
                    # Y is (block-)lower-triangular, so output block (I,J) is nonzero only
                    # for I>=J, and its K-sum runs J..I (Y[I,K]!=0 => I>=K; Y[K,J]!=0 =>
                    # K>=J). Predicate I>=K>=J. Skipped upper blocks (I<J) are never stored;
                    # ws_ysq is host-zero-init so they read back as 0 next round.
                    # Narrow y's valid view so the offset-move extracts exactly a BBxBB
                    # sub-block (offset=physical position, validshape=read extent).
                    pl.set_validshape(y, [BB, BB])
                    # Wait previous round's X@Y FIX stores before reusing acc_blk (round>0).
                    for bI in pl.range(NB):
                        for bJ in pl.range(bI + 1):            # lower-tri output blocks
                            acj = acc_blk.next()
                            # K runs bJ..bI. First term (bK==bJ) writes the acc via matmul;
                            # later terms accumulate. Last term (bK==bI) is the Final phase.
                            for bK in pl.range(bJ, bI + 1):
                                aL = a_blk.next()
                                bR = b_blk.next()
                                pl.move(aL, y, [bI * BB, bK * BB])  # Y[I,K]
                                pl.move(bR, y, [bK * BB, bJ * BB])  # Y[K,J]
                                # phase must be an inline literal in the matmul call.
                                # first term (bK==bJ): matmul; else matmul_acc.
                                # last term (bK==bI): Final; else Partial.
                                if bK == bJ and bK == bI:      # single-term block (diagonal)
                                    pl.matmul(acj, aL, bR, phase=pl.AccPhase.Final)
                                elif bK == bJ:                 # first of several
                                    pl.matmul(acj, aL, bR, phase=pl.AccPhase.Partial)
                                elif bK == bI:                 # last of several
                                    pl.matmul_acc(acj, acj, aL, bR, phase=pl.AccPhase.Final)
                                else:                          # middle
                                    pl.matmul_acc(acj, acj, aL, bR, phase=pl.AccPhase.Partial)
                            pl.store(ws_ysq, acj, [base + bI * BB, bJ * BB], phase=pl.STPhase.Final)
                    pl.set_validshape(y, [CS, CS])   # restore full view (round-0 A-load / DB rotation need it)
                    # One FIX->MTE2 sync after all block stores, before next round's load.
                    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.MTE2, event_id=0)

                # XY = X @ Y, block-sparse (same I>=K>=J coordinate-skip as Y@Y).
                # lhs blocks from x_l1 (unit-lower-tri), rhs blocks from y (strict-lower).
                # Output blocks moved into xy_ub UB quadrants; upper skipped (D01=0).
                if round > 0:
                    # Rounds >=1: x_l1 holds X from the previous Vec round (via e3).
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=3)
                    # Wait previous round's Vec zeroing of xy_ub (e5): skipped band blocks
                    # must be 0 before this round's drain writes only the fresh band blocks.
                    pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=5)
                    xtile = x_l1.current()
                    pl.set_validshape(xtile, [BB, BB])   # narrow for offset sub-block moves
                    pl.set_validshape(y, [BB, BB])
                    for bI in pl.range(NB):
                        for bJ in pl.range(bI + 1):            # lower-tri output blocks
                            # Band-mask: XY_k has band i-j >= 2**round. A block whose whole
                            # i-j range is < 2**round is exactly zero -> skip (trace-time:
                            # round/bI/bJ are Python ints). Skipped blocks are re-zeroed in
                            # Vec each round (see xy_ub expands), so the reused buffer stays
                            # correct as the band recedes.
                            if (bI - bJ) * BB + BB - 1 < (1 << round):
                                continue
                            acj = acc_blk.next()
                            for bK in pl.range(bJ, bI + 1):    # K = J..I
                                aL = a_blk.next()
                                bR = b_blk.next()
                                pl.move(aL, xtile, [bI * BB, bK * BB])  # X[I,K]
                                pl.move(bR, y, [bK * BB, bJ * BB])      # Y[K,J]
                                # No AccPhase: this X@Y block drains to UB (move to xy_ub slice),
                                # not L0C->GM, so phase must not be set. K-accumulation is still
                                # done by matmul (first term) then matmul_acc (rest).
                                if bK == bJ:                   # first K term: write acc
                                    pl.matmul(acj, aL, bR)
                                else:                          # subsequent K terms: accumulate
                                    pl.matmul_acc(acj, acj, aL, bR)
                            # Drain this 64x64 block to the owning subblock's xy_ub half.
                            # Row-block bI selects the subblock: bI==0 -> Vec0 (sub0), bI==1 ->
                            # Vec1 (sub1). Within that subblock's [CS_HALF, CS] half, the column
                            # slice picks block-column bJ. (bI/bJ are trace-time ints.)
                            xy_tile = xy_ub.current()
                            pl.set_validshape(xy_tile, [BB, BB])
                            if bI < NB // 2:
                                pl.move(xy_tile[bI * BB:(bI + 1) * BB, bJ * BB:(bJ + 1) * BB],
                                        acj, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                            else:
                                pl.move(xy_tile[(bI - NB // 2) * BB:(bI - NB // 2 + 1) * BB, bJ * BB:(bJ + 1) * BB],
                                        acj, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec1)
                    pl.set_validshape(xtile, [CS, CS])   # restore source views
                    pl.set_validshape(y, [CS, CS])
                    # X@Y drains (FIX) done; signal Vec (which reads xy_ub) and gate next
                    # round's acc_blk reuse.
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=2)

            # ── Vec (half-tile, both subblocks): X = X -/+ XY on each row-band ──
            # xy_ub is each subblock's [CS_HALF, CS] row-band (Cube routed D via Vec0/Vec1).
            # Both subblocks work: sub0 owns rows [0:64], sub1 owns rows [64:128].
            with pl.section_vector():
                sub_index = pl.get_subblock_idx()
                off = sub_index * CS_HALF
                x = x_ub.current()
                snz = stage_nz.current()
                if round == 0:
                    # Round 0: X0 = I; XY = X0@Y0 = I@L = L (dead matmul eliminated — Cube
                    # skips round-0 X@Y). Load L's row-half from A straight into xy_ub, then
                    # X1 = I - L. L is strictly-lower, so Vec0's D01 region (cols [BB:] of
                    # rows [0:BB]) = strict-upper of L = 0, preserving the xy_ub zero invariant.
                    pl.set_validshape(x, [CS_HALF, CS])
                    pl.load(x, identity, [off, 0])
                    sub_valid0 = pl.min(pl.max(valid_size - off, 0), CS_HALF)
                    pl.set_validshape(xy_ub.current(), [sub_valid0, CS])
                    pl.load(xy_ub.current(), A, [0, head_id, row_start + off, 0], order=[2, 3])
                    pl.set_validshape(xy_ub.current(), [CS_HALF, CS])
                    pl.sub(x, x, xy_ub.current())    # X1 = I - L (this half)
                else:
                    # X_new = X + XY. Cube drained xy_ub on FIX (cross-core e2); wait e2 on V.
                    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=2)
                    pl.add(x, x, xy_ub.current())

                if is_last:
                    # X = (I+L)^{-1}. Write this subblock's row-half to A_inv (clamp ragged).
                    sub_valid = pl.min(pl.max(valid_size - off, 0), CS_HALF)
                    pl.set_validshape(x, [sub_valid, CS])
                    pl.store(A_inv, x, [0, head_id, row_start + off, 0], order=[2, 3])
                else:
                    # x_l1 <- X half: ND->NZ then insert at row off for next round's mm2.
                    pl.set_validshape(x, [CS_HALF, CS])
                    # pl.move(snz, x)                            # UB ND -> UB NZ
                    if CS > 64:
                        nd2nz(x, snz, CS_HALF)                            # UB ND -> UB NZ
                    else:
                        nd2nz_sg(x, snz, CS_HALF)
                    pl.set_validshape(snz, [CS_HALF, CS])
                    pl.insert(x_l1.current(), snz, [off, 0])   # UB half -> L1 at row off
                    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=3)
                    # Re-zero xy_ub for next round: the band-mask skip set grows each round,
                    # so previously-drained (now-zero) blocks must be cleared in the reused
                    # buffer. Signal Cube (e5) so next round's drain waits for this.
                    pl.expands(xy_ub.current(), 0.0)
                    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=5)


def _compute_num_chunks(cu_seqlens, T, cs):
    if cu_seqlens is None:
        return (T + cs - 1) // cs
    cu = cu_seqlens.tolist() if hasattr(cu_seqlens, "tolist") else cu_seqlens
    return sum((cu[i + 1] - cu[i] + cs - 1) // cs for i in range(len(cu) - 1))


def inversion_kda_cube(
    A: torch.Tensor,
    cu_seqlens=None,
    num_cores: int | None = None,
) -> torch.Tensor:
    """Compute (I + L)^{-1} using fused block-sparse Neumann series kernel.

    Args:
        A:           [B, H, T, CS]  fp16/fp32   strictly lower triangular L (BNTD)
        cu_seqlens:  list[int] | None   cumulative sequence lengths
        num_cores:   int | None  None → auto-detect from device

    Returns:
        A_inv:  [B, H, T, CS]  fp32   (I + L)^{-1} (BNTD)
    """
    orig_device = A.device
    A = A.to(_DEVICE)
    if num_cores is None:
        num_cores = torch.npu.get_device_properties(0).cube_core_num
    B, H, T, _ = A.shape

    num_chunks = _compute_num_chunks(cu_seqlens, T, CS)
    total_work = num_chunks * H
    nc = min(num_cores, total_work)

    identity = torch.zeros(CS, CS, device=_DEVICE, dtype=torch.float32)
    identity.fill_diagonal_(1.0)

    A_inv = torch.zeros(B, H, T, CS, device=_DEVICE, dtype=torch.float32)

    # ws_ysq MUST be zero-initialized: block-sparse Y@Y skips upper-triangle blocks
    # (never stored), and round>=1 reads them back expecting 0.
    ws_ysq = torch.zeros(nc * CS, CS, device=_DEVICE, dtype=torch.float32)

    if cu_seqlens is None:
        cu_seqlens_t = torch.tensor([0, T], dtype=torch.int32, device=_DEVICE)
    else:
        cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int32, device=_DEVICE)

    inversion_kda_cube_kernel[None, nc](
        A.to(torch.float32), identity, A_inv, ws_ysq, cu_seqlens_t
    )
    torch.npu.synchronize()

    return A_inv.to(orig_device)