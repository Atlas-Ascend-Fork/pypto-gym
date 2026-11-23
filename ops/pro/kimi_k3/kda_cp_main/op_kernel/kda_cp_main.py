#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F821

"""kda_cp_main — FlashKDA CP main operator (PyPTO-Pro, Ascend 950PR / dav3510).

Single `@pl.jit` mixed kernel implementing the chunked state recurrence

    delta_c = u_c + w_c @ S_{c-1}^T
    O_c     = qg_c @ S_{c-1}^T + M_{qk,c} @ delta_c
    S_c     = S_{c-1} * decay_c (dk-column broadcast) + delta_c^T @ kg_c

with CHUNK=64, D=128, dv split x2 (item = (batch, head, dv_idx), 64 items
strided over cores). The 2-HOP RING (DESIGN.md 8.22): delta is produced
entirely on the cube — w@S^T plus the I@u identity-matmul trick replaces the
vector add — then materialized in L1 via a FIX Acc->Mat move (bf16 in flight,
matching the golden's single rounding) and re-read through its ZN twin. The
cube hands only T = delta^T @ kg to the vector for the state update, so the
serial ring is

    cube: T(c) = delta(c)^T @ kg(c)  ->  vec: S(c) = S(c-1)*dec + T(c)

two cross-core hops per chunk instead of four. Numerics are bit-equivalent
to the golden path (delta is bf16-rounded exactly once, before SD).
Tail chunks zero-fill w/u L1 tails explicitly (loads do not zero-fill,
probe_pad.py); delta's zero tail makes kg's clipped tail harmless.
Cube-side FIX->L1->MTE1 handoffs are auto_mutex-ordered (probe_fix2mte.py,
DESIGN.md 8.8 forbids cube->cube events).

Validated on-target semantics (probe_semantics.py / probe_*.py):
  * wait_cross_core blocks only its own pipe: waits sit on the consumer pipe.
  * NZ->NZ inserts take natural row offsets ([row, 0]); the ZN-destination
    spelling is avoided.
  * GM loads do NOT zero-fill clipped tails: tail rows are zeroed explicitly.
  * AIC-side INTRA_BLOCK waits cover both AIV sub-blocks.
  * pl.cast to an NZ-layout destination compiles but produces WRONG bytes
    (65% data err, DESIGN.md 8.5); the ND->NZ UB move is mandatory.
  * vf.astype CastLayout splits element INDEX parity; interleave() restores
    natural order (probe_vf_widen.py, DESIGN.md 8.17).
  * Acc->Vec dual-mode deliveries are cheap <=8KB/AIV and slow at 16KB/AIV
    (DESIGN.md 8.19); the O tail rides a SplitM delivery, full chunks store
    directly from L0C.
"""

from dataclasses import dataclass

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf
import torch

CHUNK = 64
D = 128
HALF_DK = 64
DVH = 64          # dv-split slice width: state rows per item
SUB_ROWS = 32

# ---- L1 (512KB) ----
L1_W = 0x00000       # [64,128] bf16 x4 (chunk parity)
L1_QG = 0x10000
L1_KG = 0x20000
L1_MQK = 0x30000     # [64,64] x4
L1_S = 0x38000       # item S half [64,128] NZ + ZN twin, chunk-parity 2 x 16KB
L1_U = 0x40000       # u half [64,64] x4 (chunk parity)
L1_DELTA = 0x48000   # delta [64,64] NZ + ZN twin, chunk-parity 2 x 8KB
L1_I = 0x4C000       # identity [64,64] (I@u trick), loaded once per task

# ---- L0A (64KB) ----
LA_W = 0x00000       # w [64,128] 16KB
LA_Q = 0x04000       # qg [64,128] 16KB
LA_MQK = 0x08000     # mqk [64,64] 8KB
LA_DT = 0x0A000      # delta^T twin [64,64] 8KB
LA_I = 0x0C000       # identity [64,64] 8KB (task-resident)
# ---- L0B (64KB) ----
LB_S = 0x00000       # S^T twin [128,64] ZN 16KB
LB_KG = 0x04000      # kg [64,128] ZN 16KB
LB_U = 0x08000       # u [64,64] ZN 8KB (I@u B operand)
LB_D = 0x0A000       # delta [64,64] ZN 8KB (mqk@delta B operand)
# ---- L0C (256KB) ----
CA_D = 0x00000       # delta acc [64,64] f32
CA_O = 0x04000       # O acc [64,64] f32
CA_T = 0x08000       # T = delta^T@kg [64,128] f32

# ---- UB (248KB per AIV) ----
UB_DEC = 0x00000     # [1,64] f32 decay row (parity bank A)
UB_DEC_B = 0x00200
UB_ST0 = 0x01000     # item state half [64,64] f32 (task odd/even slots)
UB_ST1 = 0x05000
UB_SD = 0x09000      # T [64,64] f32 SplitN target x2 (chunk parity)
UB_WS = 0x12000      # O-tail delivery [32,64] f32 x2 (8KB banks)
UB_UND = 0x16000     # O-tail cast [32,64] bf16 x2 (4KB banks)
UB_SND = 0x18000     # [64,64] bf16 snapshot (natural)
UB_SNZ = 0x1A000     # [64,64] bf16 NZ staging

# ---- events (INTRA_BLOCK) ----
# 2-hop ring: EV_SD carries T to the vector state update, EV_S carries the
# published S back. EV_WS exists only for tail-chunk O delivery.
EV_SD = ((2, 10), (6, 14))
EV_S = ((3, 11), (7, 15))
EV_WS = ((0, 8), (4, 12))


@pl.vector_function
def kda_state_fuse_vf(s_t, dec_t, d_t, out_f, snap_b):
    """S = S*dec + T in one V pass; also emit the bf16 snapshot rows (m7 recipe)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    g = vf.load_align(dec_t, 0)
    for j2 in pl.range(0, DVH // 2):
        off = j2 * (2 * HALF_DK)
        sv0 = vf.load_align(s_t, off)
        sv1 = vf.load_align(s_t, off + HALF_DK)
        dv0 = vf.load_align(d_t, off)
        dv1 = vf.load_align(d_t, off + HALF_DK)
        t0 = sv0
        t1 = sv1
        t0 = vf.mul_dst_add(g, dv0, preg)
        t1 = vf.mul_dst_add(g, dv1, preg)
        vf.store_align(out_f + off, t0, preg)
        vf.store_align(out_f + off + HALF_DK, t1, preg)
        b0 = vf.astype(t0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(t1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        p0, _y0 = vf.de_interleave(b0, b1)
        vf.store_align(snap_b + off, p0, pregbf)


@dataclass
class KdaCpTiling:
    """Runtime shape/stride description of the rank-erased operands."""

    batch: int
    nv: int
    storage_batch: int
    storage_len: int
    chunks_cap: int
    stride_b: int
    stride_n: int
    stride_s: int
    out_stride_b: int
    out_stride_n: int
    out_stride_s: int
    is_packed: int
    ncore: int = 1


@pl.jit(auto_mutex=True)
def kda_cp_main_kernel(
    qg_ptr: pl.Ptr[pl.DT_BF16],
    kg_ptr: pl.Ptr[pl.DT_BF16],
    u_ptr: pl.Ptr[pl.DT_BF16],
    w_ptr: pl.Ptr[pl.DT_BF16],
    decay_ptr: pl.Ptr[pl.DT_FP32],
    mqk_ptr: pl.Ptr[pl.DT_BF16],
    state_in_ptr: pl.Ptr[pl.DT_FP32],
    out_ptr: pl.Ptr[pl.DT_BF16],
    state_out_ptr: pl.Ptr[pl.DT_FP32],
    eye_ptr: pl.Ptr[pl.DT_BF16],
    cu_ptr: pl.Ptr[pl.DT_INT32],
    tiling: KdaCpTiling,
):
    # Rank-erased views; token tensors share one [B, Nv, L, D] logical shape.
    qg_t = pl.make_tensor(qg_ptr, [tiling.storage_batch, tiling.nv, tiling.storage_len, D],
                          stride=(tiling.stride_b, tiling.stride_n, tiling.stride_s, 1),
                          dtype=pl.DT_BF16)
    kg_t = pl.make_tensor(kg_ptr, [tiling.storage_batch, tiling.nv, tiling.storage_len, D],
                          stride=(tiling.stride_b, tiling.stride_n, tiling.stride_s, 1),
                          dtype=pl.DT_BF16)
    u_t = pl.make_tensor(u_ptr, [tiling.storage_batch, tiling.nv, tiling.storage_len, D],
                         stride=(tiling.stride_b, tiling.stride_n, tiling.stride_s, 1),
                         dtype=pl.DT_BF16)
    w_t = pl.make_tensor(w_ptr, [tiling.storage_batch, tiling.nv, tiling.storage_len, D],
                         stride=(tiling.stride_b, tiling.stride_n, tiling.stride_s, 1),
                         dtype=pl.DT_BF16)
    out_t = pl.make_tensor(out_ptr, [tiling.storage_batch, tiling.nv, tiling.storage_len, D],
                           stride=(tiling.out_stride_b, tiling.out_stride_n,
                                   tiling.out_stride_s, 1),
                           dtype=pl.DT_BF16)
    decay_t = pl.make_tensor(decay_ptr, [tiling.batch, tiling.nv, tiling.chunks_cap, D],
                             dtype=pl.DT_FP32)
    mqk_t = pl.make_tensor(mqk_ptr, [tiling.batch, tiling.nv, tiling.chunks_cap, CHUNK, CHUNK],
                           dtype=pl.DT_BF16)
    sin_t = pl.make_tensor(state_in_ptr, [tiling.batch, tiling.nv, D, D], dtype=pl.DT_FP32)
    sout_t = pl.make_tensor(state_out_ptr, [tiling.batch, tiling.nv, D, D], dtype=pl.DT_FP32)
    eye_t = pl.make_tensor(eye_ptr, [CHUNK, CHUNK], dtype=pl.DT_BF16)
    cu_t = pl.make_tensor(cu_ptr, [tiling.batch + 1], dtype=pl.DT_INT32)

    # ---- L1 tile groups ----
    w_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat,
                         layout=pl.NZ),
        addrs=[L1_W + 0x4000 * i for i in range(4)], mutex_ids=[0, 0, 0, 0])
    qg_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat,
                         layout=pl.NZ),
        addrs=[L1_QG + 0x4000 * i for i in range(4)], mutex_ids=[1, 1, 1, 1])
    kg_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat,
                         layout=pl.NZ),
        addrs=[L1_KG + 0x4000 * i for i in range(4)], mutex_ids=[2, 2, 2, 2])
    mqk_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, CHUNK], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[L1_MQK + 0x2000 * i for i in range(4)], mutex_ids=[3, 3, 3, 3])
    s_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[DVH, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat,
                         layout=pl.NZ),
        addrs=[L1_S, L1_S + 0x4000], mutex_ids=[4, 4])
    s_l1_t = pl.make_tile_group(
        type=pl.TileType(shape=[D, DVH], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat,
                         layout=pl.ZN),
        addrs=[L1_S, L1_S + 0x4000], mutex_ids=[4, 4])
    u_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, DVH], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat,
                         layout=pl.NZ),
        addrs=[L1_U + 0x2000 * i for i in range(4)], mutex_ids=[5, 5, 5, 5])
    delta_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, DVH], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat,
                         layout=pl.NZ),
        addrs=[L1_DELTA, L1_DELTA + 0x2000], mutex_ids=[6, 6])
    delta_l1_t = pl.make_tile_group(
        type=pl.TileType(shape=[DVH, CHUNK], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat,
                         layout=pl.ZN),
        addrs=[L1_DELTA, L1_DELTA + 0x2000], mutex_ids=[6, 6])
    i_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, CHUNK], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat,
                         layout=pl.NZ),
        addrs=[L1_I], mutex_ids=[7])

    # ---- L0 staging (every slot independent, no aliases) ----
    la_w = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left,
                         layout=pl.NZ),
        addrs=[LA_W], mutex_ids=[9])
    la_q = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left,
                         layout=pl.NZ),
        addrs=[LA_Q], mutex_ids=[10])
    la_mqk = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, CHUNK], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=[LA_MQK], mutex_ids=[11])
    la_dt = pl.make_tile_group(
        type=pl.TileType(shape=[DVH, CHUNK], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left,
                         layout=pl.NZ),
        addrs=[LA_DT], mutex_ids=[12])
    la_i = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, CHUNK], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=[LA_I], mutex_ids=[13])
    lb_s = pl.make_tile_group(
        type=pl.TileType(shape=[D, DVH], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right,
                         layout=pl.ZN),
        addrs=[LB_S], mutex_ids=[14])
    lb_kg = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right,
                         layout=pl.ZN),
        addrs=[LB_KG], mutex_ids=[15])
    lb_u = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, DVH], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right,
                         layout=pl.ZN),
        addrs=[LB_U], mutex_ids=[16])
    lb_d = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, DVH], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right,
                         layout=pl.ZN),
        addrs=[LB_D], mutex_ids=[17])

    # ---- accumulators ----
    acc_d = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, DVH], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         layout=pl.NZ, fractal=1024),
        addrs=[CA_D], mutex_ids=[18])
    acc_o = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, DVH], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         layout=pl.NZ, fractal=1024),
        addrs=[CA_O], mutex_ids=[19])
    acc_t = pl.make_tile_group(
        type=pl.TileType(shape=[DVH, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         layout=pl.NZ, fractal=1024),
        addrs=[CA_T], mutex_ids=[20])

    # ---- UB tile groups (per AIV) ----
    dec_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, HALF_DK], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=[UB_DEC, UB_DEC_B], mutex_ids=[24, 24])  # parity-banked
    s_state = pl.make_tile_group(
        type=pl.TileType(shape=[DVH, HALF_DK], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=[UB_ST0, UB_ST1], mutex_ids=[19, 19])
    sd_ub = pl.make_tile_group(
        type=pl.TileType(shape=[DVH, HALF_DK], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=[UB_SD, UB_SD + 0x4000], mutex_ids=[22, 23])
    ws_ub = pl.make_tile_group(
        type=pl.TileType(shape=[SUB_ROWS, DVH], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=[UB_WS, UB_WS + 0x2000], mutex_ids=[20, 21])
    u_nd = pl.make_tile_group(
        type=pl.TileType(shape=[SUB_ROWS, DVH], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=[UB_UND, UB_UND + 0x1000], mutex_ids=[25, 26])
    s_nd = pl.make_tile_group(
        type=pl.TileType(shape=[DVH, HALF_DK], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=[UB_SND], mutex_ids=[28])
    s_nz = pl.make_tile_group(
        type=pl.TileType(shape=[DVH, HALF_DK], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec,
                         layout=pl.NZ),
        addrs=[UB_SNZ], mutex_ids=[30])

    core_id = pl.get_block_idx() // pl.get_subblock_num()
    num_cores = pl.get_block_num()
    sub_id = pl.get_subblock_idx()
    row0 = SUB_ROWS * sub_id
    col0 = HALF_DK * sub_id

    total_items = tiling.batch * tiling.nv * 2

    with pl.section_cube():
        pl.load(i_l1[0], eye_t, [0, 0])
        pl.move(la_i.current(), i_l1[0])
        for item_a in pl.range(core_id, total_items, num_cores):
            head_item = item_a // 2
            dvc = (item_a % 2) * DVH
            seq_a = head_item // tiling.nv
            head_a = head_item % tiling.nv
            t0_a = 0
            end_a = tiling.storage_len
            if tiling.is_packed == 1:
                t0_a = pl.getval(cu_t, seq_a)
                end_a = pl.getval(cu_t, seq_a + 1)
            n_a = (end_a - t0_a + CHUNK - 1) // CHUNK
            b_a = 0 if tiling.is_packed == 1 else seq_a
            n_max = n_a
            # prologue: chunk-0 operands (identity loaded once per core)
            v0 = pl.min(CHUNK, end_a - t0_a)
            if v0 != CHUNK:
                pl.expands(w_l1[0], 0.0)
                pl.expands(u_l1[0], 0.0)
            pl.set_validshape(w_l1[0], [v0, D])
            pl.load(w_l1[0], w_t, [b_a, head_a, t0_a, 0], order=[2, 3])
            pl.set_validshape(w_l1[0], [CHUNK, D])
            pl.set_validshape(qg_l1[0], [v0, D])
            pl.load(qg_l1[0], qg_t, [b_a, head_a, t0_a, 0], order=[2, 3])
            pl.set_validshape(qg_l1[0], [CHUNK, D])
            pl.set_validshape(kg_l1[0], [v0, D])
            pl.load(kg_l1[0], kg_t, [b_a, head_a, t0_a, 0], order=[2, 3])
            pl.set_validshape(kg_l1[0], [CHUNK, D])
            pl.load(mqk_l1[0], mqk_t, [seq_a, head_a, 0, 0, 0], order=[3, 4])
            pl.set_validshape(u_l1[0], [v0, DVH])
            pl.load(u_l1[0], u_t, [b_a, head_a, t0_a, dvc], order=[2, 3])
            pl.set_validshape(u_l1[0], [CHUNK, DVH])
            for c in pl.range(0, n_max):
                par = c % 2
                va = pl.min(CHUNK, end_a - (t0_a + c * CHUNK))
                sa = 0 if va != CHUNK else par
                if c < n_a:
                    # stage all chunk-c operands before the EV_S wait: the
                    # MTE1 copies drain while the cube blocks on the wait
                    pl.move(la_w.current(), w_l1[sa])
                    pl.move(la_q.current(), qg_l1[sa])
                    pl.move(lb_kg.current(), kg_l1[sa])
                    pl.move(la_mqk.current(), mqk_l1[sa])
                    pl.move(lb_u.current(), u_l1[sa])
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=EV_S[0][par])
                    pl.move(lb_s.current(), s_l1_t[par])
                    # delta chain (cube-only; I@u replaces the vector add)
                    pl.matmul(acc_d.current(), la_w.current(), lb_s.current())
                    pl.matmul_acc(acc_d.current(), acc_d.current(), la_i.current(),
                                  lb_u.current())
                    pl.move(delta_l1[par], acc_d.current())
                    pl.move(la_dt.current(), delta_l1_t[par])
                    # O branch
                    pl.matmul(acc_o.current(), la_q.current(), lb_s.current())
                    pl.move(lb_d.current(), delta_l1[par])
                    pl.matmul_acc(acc_o.current(), acc_o.current(), la_mqk.current(),
                                  lb_d.current())
                    valid_a = pl.min(CHUNK, end_a - (t0_a + c * CHUNK))
                    # All chunks deliver O through the UB drain: the cube ring
                    # keeps only the FIX->UB move; the vector side casts and
                    # stores O to GM while the cube runs the next chunk.
                    pl.move(ws_ub[par], acc_o.current(),
                            acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=EV_WS[0][par])
                    # T = delta^T @ kg, delivered to the vector ring segment
                    pl.matmul(acc_t.current(), la_dt.current(), lb_kg.current())
                    pl.move(sd_ub[par], acc_t.current(),
                            acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=EV_SD[0][par])
                    # prefetch chunk c+1 operands
                    if c + 1 < n_a:
                        t_next = t0_a + (c + 1) * CHUNK
                        v_next = pl.min(CHUNK, end_a - t_next)
                        sa2 = 0 if v_next != CHUNK else (c + 1) % 2
                        if v_next != CHUNK:
                            pl.expands(w_l1[sa2], 0.0)
                            pl.expands(u_l1[sa2], 0.0)
                        pl.set_validshape(w_l1[sa2], [v_next, D])
                        pl.load(w_l1[sa2], w_t, [b_a, head_a, t_next, 0], order=[2, 3])
                        pl.set_validshape(w_l1[sa2], [CHUNK, D])
                        pl.set_validshape(qg_l1[sa2], [v_next, D])
                        pl.load(qg_l1[sa2], qg_t, [b_a, head_a, t_next, 0], order=[2, 3])
                        pl.set_validshape(qg_l1[sa2], [CHUNK, D])
                        pl.set_validshape(kg_l1[sa2], [v_next, D])
                        pl.load(kg_l1[sa2], kg_t, [b_a, head_a, t_next, 0], order=[2, 3])
                        pl.set_validshape(kg_l1[sa2], [CHUNK, D])
                        pl.load(mqk_l1[sa2], mqk_t, [seq_a, head_a, c + 1, 0, 0], order=[3, 4])
                        pl.set_validshape(u_l1[sa2], [v_next, DVH])
                        pl.load(u_l1[sa2], u_t, [b_a, head_a, t_next, dvc], order=[2, 3])
                        pl.set_validshape(u_l1[sa2], [CHUNK, DVH])
    with pl.section_vector():
        for item_a in pl.range(core_id, total_items, num_cores):
            task_ord = item_a // tiling.ncore
            st_slot = task_ord % 2
            head_item = item_a // 2
            dvc = (item_a % 2) * DVH
            seq_a = head_item // tiling.nv
            head_a = head_item % tiling.nv
            t0_a = 0
            end_a = tiling.storage_len
            if tiling.is_packed == 1:
                t0_a = pl.getval(cu_t, seq_a)
                end_a = pl.getval(cu_t, seq_a + 1)
            n_a = (end_a - t0_a + CHUNK - 1) // CHUNK
            b_a = 0 if tiling.is_packed == 1 else seq_a
            n_max = n_a

            if item_a == core_id:
                # first item on this core: cold prologue (later items were
                # prefetched by their predecessor's last chunk)
                pl.load(s_state[st_slot], sin_t, [seq_a, head_a, dvc, col0], order=[2, 3])
                pl.cast(s_nd.current(), s_state[st_slot])
                pl.move(s_nz.current(), s_nd.current())
                pl.insert(s_l1[0], s_nz.current(), [0, col0])
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=EV_S[0][0])

            for c in pl.range(0, n_max):
                par = c % 2
                npar = (c + 1) % 2
                if c < n_a:
                    t_base = t0_a + c * CHUNK
                    valid_a = pl.min(CHUNK, end_a - t_base)
                    vr_a = valid_a - row0
                    vr_a = pl.max(0, pl.min(SUB_ROWS, vr_a))
                    pl.load(dec_ub[par], decay_t, [seq_a, head_a, c, col0], order=[2, 3])
                    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=EV_SD[0][par])
                    kda_state_fuse_vf(s_state[st_slot], dec_ub[par], sd_ub[par],
                                      s_state[st_slot], s_nd.current())
                    if c + 1 < n_a:
                        pl.move(s_nz.current(), s_nd.current())
                        pl.insert(s_l1[npar], s_nz.current(), [0, col0])
                        pl.system.set_cross_core(pipe=pl.PipeType.MTE3,
                                                 event_id=EV_S[0][npar])
                    else:
                        # last chunk: prefetch the next item's initial state
                        nxt = item_a + num_cores
                        if nxt < total_items:
                            nts = (task_ord + 1) % 2
                            n_head = nxt // 2
                            n_dvc = (nxt % 2) * DVH
                            pl.load(s_state[nts], sin_t,
                                    [n_head // tiling.nv, n_head % tiling.nv, n_dvc, col0],
                                    order=[2, 3])
                            pl.cast(s_nd.current(), s_state[nts])
                            pl.move(s_nz.current(), s_nd.current())
                            pl.insert(s_l1[0], s_nz.current(), [0, col0])
                            pl.system.set_cross_core(pipe=pl.PipeType.MTE3,
                                                     event_id=EV_S[0][0])
                    # O drain (all chunks; full-chunk rows use SUB_ROWS width)
                    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=EV_WS[0][par])
                    if vr_a > 0:
                        pl.cast(u_nd[par], ws_ub[par])
                        pl.set_validshape(u_nd[par], [vr_a, DVH])
                        pl.store(out_t, u_nd[par], [b_a, head_a, t_base + row0, dvc],
                                 order=[2, 3])
                        pl.set_validshape(u_nd[par], [SUB_ROWS, DVH])
            pl.store(sout_t, s_state[st_slot], [seq_a, head_a, dvc, col0], order=[2, 3])


def _require_tensor_meta(tensors, ref):
    """Every input must be a contiguous torch.Tensor on the reference device."""
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError("all inputs must be torch.Tensor")
        if not t.is_contiguous():
            raise ValueError("all inputs must be contiguous")
        if t.device != ref.device:
            raise ValueError("all inputs must be on the same device")


def _validate_tnd_layout(qg, cu_seqlens, batch, nv):
    """Validate the packed TND layout; return (storage_len, storage_batch, chunks_cap)."""
    storage_len, qg_nv = int(qg.shape[0]), int(qg.shape[1])
    if qg_nv != nv:
        raise ValueError("token tensor head count must match state")
    if cu_seqlens is None or cu_seqlens.dtype != torch.int32 or not cu_seqlens.is_contiguous():
        raise ValueError("TND requires a contiguous int32 cu_seqlens")
    if tuple(cu_seqlens.shape) != (batch + 1,):
        raise ValueError(f"cu_seqlens must have {batch + 1} entries")
    if cu_seqlens.device != qg.device:
        raise ValueError("cu_seqlens must be on the same device")
    if int(cu_seqlens[0].item()) != 0 or int(cu_seqlens[-1].item()) != storage_len:
        raise ValueError("cu_seqlens must start at 0 and end at the token count")
    return storage_len, 1, (storage_len + CHUNK - 1) // CHUNK + batch - 1


def _validate_dense_layout(qg, batch, nv, layout):
    """Validate the BNSD/BSND dense layout; return (storage_len, storage_batch, chunks_cap)."""
    if int(qg.shape[0]) != batch or int(qg.shape[1]) != nv:
        raise ValueError(f"{layout} token tensors must start with [{batch}, {nv}]")
    storage_len = int(qg.shape[2])
    return storage_len, batch, (storage_len + CHUNK - 1) // CHUNK + batch - 1


def _validate_operand_dtypes(decay, mqk, state, qg, kg, u, w, batch, nv, chunks_cap):
    """Every operand must match its declared dtype and [T, D]-derived shape."""
    for name, t, shape, dt in (
        ("decay", decay, (batch, nv, chunks_cap, D), torch.float32),
        ("mqk", mqk, (batch, nv, chunks_cap, CHUNK, CHUNK), torch.bfloat16),
        ("local_initial_state", state, (batch, nv, D, D), torch.float32),
    ):
        if t.dtype != dt or tuple(t.shape) != shape:
            raise ValueError(f"{name} must be {shape} {str(dt).split('.')[-1]}")
    for name, t in (("qg", qg), ("kg", kg), ("u", u), ("w", w)):
        if t.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be bfloat16")
        if t.shape != qg.shape:
            raise ValueError(f"{name} must have the same shape as qg")


def _validate_inputs(qg, kg, u, w, decay, mqk, state, layout, cu_seqlens):
    """Validate every public input; return (batch, nv, storage_len, chunks_cap)."""
    if layout not in ("BNSD", "BSND", "TND"):
        raise ValueError("layout_qkv must be BNSD, BSND, or TND")
    _require_tensor_meta((qg, kg, u, w, decay, mqk, state), qg)
    if state.dim() != 4 or tuple(state.shape[2:]) != (D, D):
        raise ValueError(f"state tensors must be [B, Nv, {D}, {D}]")
    batch, nv = int(state.shape[0]), int(state.shape[1])
    if qg.dim() != (3 if layout == "TND" else 4):
        raise ValueError(f"{layout} token tensors must be rank-{3 if layout == 'TND' else 4}")
    if layout == "TND":
        storage_len, storage_batch, chunks_cap = _validate_tnd_layout(qg, cu_seqlens, batch, nv)
    else:
        storage_len, storage_batch, chunks_cap = _validate_dense_layout(qg, batch, nv, layout)
    if storage_len <= 0:
        raise ValueError("token count must be positive")
    _validate_operand_dtypes(decay, mqk, state, qg, kg, u, w, batch, nv, chunks_cap)
    if qg.device.type != "npu":
        raise ValueError("flash_kda_cp_main requires NPU tensors")
    return batch, nv, storage_len, chunks_cap, storage_batch


_EYE_CACHE = {}


def _chunk_identity(device):
    eye = _EYE_CACHE.get(device)
    if eye is None:
        eye = torch.eye(CHUNK, dtype=torch.bfloat16, device=device)
        _EYE_CACHE[device] = eye
    return eye


def flash_kda_cp_main(transition, state_ext, decay, qg, kg, u, w, mqk,
                      local_initial_state, layout_qkv, metadata=None, cu_seqlens=None):
    """Run the FlashKDA CP recurrence; returns (out, final_state)."""
    batch, nv, storage_len, chunks_cap, storage_batch = _validate_inputs(
        qg, kg, u, w, decay, mqk, local_initial_state, layout_qkv, cu_seqlens)
    if layout_qkv == "TND":
        out = torch.empty((storage_len, nv, D), dtype=torch.bfloat16, device=qg.device)
        stride_b, stride_n, stride_s = nv * D * storage_len, D, nv * D
        out_sb, out_sn, out_ss = nv * D * storage_len, D, nv * D
        is_packed = 1
    else:
        # Token inputs use BNSD storage for both BNSD and BSND requests (the
        # reference contract: "BSND changes only the output layout").
        stride_b, stride_n, stride_s = nv * storage_len * D, storage_len * D, D
        if layout_qkv == "BNSD":
            out = torch.empty((batch, nv, storage_len, D), dtype=torch.bfloat16, device=qg.device)
            out_sb, out_sn, out_ss = nv * storage_len * D, storage_len * D, D
        else:
            out = torch.empty((batch, storage_len, nv, D), dtype=torch.bfloat16, device=qg.device)
            out_sb, out_sn, out_ss = storage_len * nv * D, D, nv * D
        is_packed = 0
    final_state = torch.empty_like(local_initial_state)
    total_items = batch * nv * 2
    cores = int(torch.npu.get_device_properties(qg.device).vector_core_num // 2)
    blocks = max(1, min(cores, total_items))
    tiling = KdaCpTiling(
        batch=batch, nv=nv, storage_batch=storage_batch, storage_len=storage_len,
        chunks_cap=chunks_cap, stride_b=stride_b, stride_n=stride_n, stride_s=stride_s,
        out_stride_b=out_sb, out_stride_n=out_sn, out_stride_s=out_ss,
        is_packed=is_packed, ncore=blocks,
    )
    eye = _chunk_identity(qg.device)
    kda_cp_main_kernel[None, blocks](
        qg, kg, u, w, decay, mqk, local_initial_state, out, final_state,
        eye, cu_seqlens, tiling,
    )
    return out, final_state
