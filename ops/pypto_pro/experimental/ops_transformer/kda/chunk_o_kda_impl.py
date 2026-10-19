# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""chunk_o_kda operator — fused single-kernel output pass for KDA.

Implements Stage 6 of the KDA (Kernelized Decay Attention) pipeline.

Math (per chunk of size C, per head):
    q_eff = q * exp(g_cs)                 # [C, K]
    k_eff = k * exp(-g_cs)                # [C, K]
    inter = q_eff @ S                     # [C, V]   S = s_snapshots[ci, head]
    Aqk   = tril(q_eff @ k_eff^T, k=0)    # [C, C]   inclusive lower triangle
    o     = inter + Aqk @ v_corr          # [C, V]

Fused-kernel data flow (5 phases, alternating Vec/Cube via cross-core sync):
    Phase 1 (Vec):  q_eff, k_eff, vcorr_f, s_f → per-core GM workspace
    Phase 2 (Cube): inter = q_eff @ S;  aqk_raw = q_eff @ k_eff^T → workspace
    Phase 3 (Vec):  aqk = aqk_raw * mask → workspace
    Phase 4 (Cube): tmp = aqk @ v_corr → workspace
    Phase 5 (Vec):  o = inter + tmp; cast fp16; store to GM output
"""

import torch
import pypto_pro.language as pl

from .kda_common import C, K, V, HC, DEVICE as _DEVICE, build_chunk_tables


# ---- L1 (Mat) addresses ----
# Phase 2 uses 4 operands simultaneously (q16/S for 2-1, q32/k for 2-2):
L1_Q16 = 0x00000          # q_eff [C, K] fp16 NZ  (Phase 2-1)  32KB
L1_S   = 0x08000          # S     [K, V] fp16 NZ  (Phase 2-1)  32KB
L1_Q   = 0x10000          # q_eff [C, K] fp32 NZ  (Phase 2-2)  64KB
L1_K   = 0x20000          # k_eff [C, K] fp32 ZN  (Phase 2-2)  64KB
# Phase 4 (L1 独立分配, L1 共 512KB, 无需复用):
L1_AQK = 0x30000          # aqk    [C, C] fp16 NZ  32KB
L1_VC  = 0x38000          # v_corr [C, V] fp16 NZ  32KB

# ---- UB (Vec) addresses ---- (total 248KB, sequential non-overlapping)
UB_H = 0x00000            # fp16 load buffer [HC,K]    16 KB
UB_G = 0x04000            # g_cs fp32      [HC,K]      32 KB
UB_F = 0x0C000            # fp32 work      [HC,K]      32 KB
UB_E = 0x14000            # fp32 exp       [HC,K]      32 KB
UB_Q16 = 0x1C000          # q_eff fp16 cast [HC,K]     16 KB
UB_S = 0x20000            # S fp16         [K/2,V]     16 KB
UB_MASK = 0x24000         # mask [HC,C] fp32 (phase 3) 32 KB
UB_OH   = 0x2C000         # [HC,V] fp16 (phase 3/5)    16 KB
UB_INTER = 0x30000        # inter [HC,V] fp32 (phase 2-1 Acc→UB) 32 KB
# Total: 16+32+32+32+16+16+32+16+32 = 224 KB < 248 KB


def _mat_nz(shape, dtype):
    return pl.TileType(shape=shape, dtype=dtype,
                       target_memory=pl.MemorySpace.Mat, layout=pl.NZ)


def _mat_zn(shape, dtype):
    return pl.TileType(shape=shape, dtype=dtype,
                       target_memory=pl.MemorySpace.Mat, layout=pl.ZN)


def _left(shape, dtype):
    return pl.TileType(shape=shape, dtype=dtype,
                       target_memory=pl.MemorySpace.Left, layout=pl.NZ)


def _right(shape, dtype):
    return pl.TileType(shape=shape, dtype=dtype,
                       target_memory=pl.MemorySpace.Right, layout=pl.ZN)


def _acc(shape):
    return pl.TileType(shape=shape, dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Acc,
                       layout=pl.NZ, fractal=1024)


def _vec(shape, dtype):
    return pl.TileType(shape=shape, dtype=dtype,
                       target_memory=pl.MemorySpace.Vec)


@pl.jit(auto_mutex=True)
def chunk_o_kda_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    g: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    vcorr: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    s_snap: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    o: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    ws_q16: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    ws_v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    ws_s: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    ws_aqk: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    num_heads: pl.DT_INT32,
    seq_len: pl.DT_INT64,
    cu_seqlens: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    num_seqs: pl.DT_INT32,
    chunk_tbase: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    chunk_valid: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
):
    """Fused chunk_o_kda: 5 phases in one kernel via Cube<->Vec cross-core sync."""

    # -- Cube L1 tile groups (Phase 2-1: inter = q_eff @ S, fp16 inputs) --
    # NOTE: q16_l0a/q_l0a share Left addr 0x0 and s_l0b/k_l0b share Right addr 0x0.
    # They MUST share mutex_id (3 / 4) so auto_mutex serializes matmul 2-1's L0
    # read against 2-2's L0 write (no cross-core sync chain exists between them).
    q16_l1 = pl.make_tile_group(type=_mat_nz([C, K], pl.DT_FP16), addrs=L1_Q16, mutex_ids=[0])
    s_l1 = pl.make_tile_group(type=_mat_nz([K, V], pl.DT_FP16), addrs=L1_S, mutex_ids=[1])
    q16_l0a = pl.make_tile_group(type=_left([C, K], pl.DT_FP16), addrs=0x0, mutex_ids=[3])
    s_l0b = pl.make_tile_group(type=_right([K, V], pl.DT_FP16), addrs=0x0, mutex_ids=[4])
    inter_acc = pl.make_tile_group(type=_acc([C, V]), addrs=0x0, mutex_ids=[5])

    # -- Cube L1 tile groups (Phase 2-2: aqk_raw = q_eff @ k_eff^T, fp32 inputs) --
    q_l1 = pl.make_tile_group(type=_mat_nz([C, K], pl.DT_FP32), addrs=L1_Q, mutex_ids=[16])
    k_l1 = pl.make_tile_group(type=_mat_zn([K, C], pl.DT_FP32), addrs=L1_K, mutex_ids=[2])
    q_l0a = pl.make_tile_group(type=_left([C, K], pl.DT_FP32), addrs=0x0, mutex_ids=[3])
    k_l0b = pl.make_tile_group(type=_right([K, C], pl.DT_FP32), addrs=0x0, mutex_ids=[4])
    aqk_acc = pl.make_tile_group(type=_acc([C, C]), addrs=0x10000, mutex_ids=[6])

    # -- Vec UB: inter (Phase 2-1 Acc→UB target, consumed by Phase 5) --
    inter_ub_fp32_grp = pl.make_tile_group(type=_vec([HC, V], pl.DT_FP32), addrs=UB_INTER, mutex_ids=[23])

    # -- Cube L1 tile groups (Phase 4: tmp = aqk @ v_corr, fp16; L1 独立分配) --
    # NOTE: aqk_l0a/v_l0b share L0 addr (0x0, hardware space) with Phase 2 operands —
    # L0 mutex_ids MUST match (3/4). L1 tiles are independently allocated (512KB budget).
    aqk_l1 = pl.make_tile_group(type=_mat_nz([C, C], pl.DT_FP16), addrs=L1_AQK, mutex_ids=[7])
    v_l1 = pl.make_tile_group(type=_mat_nz([C, V], pl.DT_FP16), addrs=L1_VC, mutex_ids=[8])
    aqk_l0a = pl.make_tile_group(type=_left([C, C], pl.DT_FP16), addrs=0x0, mutex_ids=[3])
    v_l0b = pl.make_tile_group(type=_right([C, V], pl.DT_FP16), addrs=0x0, mutex_ids=[4])
    tmp_acc = pl.make_tile_group(type=_acc([C, V]), addrs=0x0, mutex_ids=[5])

    # -- Vec UB tile groups (Phase 1: gate, HalfC per sub-block) --
    q_ub_fp16_grp = pl.make_tile_group(
        type=pl.TileType(shape=[HC, K], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec,
                         pad=pl.TilePad.zero, valid_shape=[-1, -1]),
        addrs=UB_H, mutex_ids=[12])
    g_ub_fp32_grp = pl.make_tile_group(
        type=pl.TileType(shape=[HC, K], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         pad=pl.TilePad.zero, valid_shape=[-1, -1]),
        addrs=UB_G, mutex_ids=[13])
    q_ub_fp32_grp = pl.make_tile_group(type=_vec([HC, K], pl.DT_FP32), addrs=UB_F, mutex_ids=[14])
    gExp_ub_fp32_grp = pl.make_tile_group(type=_vec([HC, K], pl.DT_FP32), addrs=UB_E, mutex_ids=[15])
    # dedicated buffer for q_eff fp16 cast (avoid WAR with q_ub_fp16_grp reload of k)
    q16_ub_fp16_grp = pl.make_tile_group(
        type=pl.TileType(shape=[HC, K], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec,
                         pad=pl.TilePad.zero, valid_shape=[-1, -1]),
        addrs=UB_Q16, mutex_ids=[22])
    s_ub_fp16_grp = pl.make_tile_group(
        type=pl.TileType(shape=[K // 2, V], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_S, mutex_ids=[21])

    # -- Vec UB tile groups (Phase 3: mask; HalfC per sub-block, UB reused) --
    ub_mask = pl.make_tile_group(type=_vec([HC, C], pl.DT_FP32), addrs=UB_MASK, mutex_ids=[12])
    ub_oh = pl.make_tile_group(
        type=pl.TileType(shape=[HC, V], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec,
                         pad=pl.TilePad.zero, valid_shape=[-1, -1]),
        addrs=UB_OH, mutex_ids=[20])

    core_id = pl.get_block_idx() // pl.get_subblock_num()
    num_cores = pl.get_block_num()
    sub_id = pl.get_subblock_idx()

    if num_seqs == 1:
        total_chunks = (seq_len + C - 1) // C
    else:
        total_chunks = chunk_tbase.shape[0]
    total_work = total_chunks * num_heads
    row = sub_id * HC

    for work_id in pl.range(core_id, total_work, num_cores):
        chunk_id = work_id // num_heads
        head_id = work_id % num_heads

        global_chunk_id = chunk_id
        if num_seqs == 1:
            t_base = chunk_id * C
            valid_size = pl.min(seq_len - t_base, C)
        else:
            t_base = pl.getval(chunk_tbase, chunk_id)
            valid_size = pl.getval(chunk_valid, chunk_id)

        # ================================================================
        #  Phase 1 (Vec): gate q/k, cast v/s (both sub-blocks, HalfC each)
        # ================================================================
        valid_rows = pl.max(0, pl.min(HC, valid_size - row))
        t_offset = t_base + row

        with pl.section_vector():
            q_fp16 = q_ub_fp16_grp.current()
            g_fp32 = g_ub_fp32_grp.current()
            q_fp32 = q_ub_fp32_grp.current()
            g_exp_fp32 = gExp_ub_fp32_grp.current()
            q16_fp16 = q16_ub_fp16_grp.current()
            s_fp16 = s_ub_fp16_grp.current()

            pl.set_validshape(g_fp32, [valid_rows, K])
            pl.load(g_fp32, g, [0, head_id, t_offset, 0], order=[2, 3])
            pl.set_validshape(g_fp32, [HC, K])
            pl.exp(g_exp_fp32, g_fp32)

            pl.set_validshape(q_fp16, [valid_rows, K])
            pl.load(q_fp16, q, [0, t_offset, head_id, 0], order=[1, 3])
            pl.set_validshape(q_fp16, [HC, K])
            pl.cast(q_fp32, q_fp16, mode=pl.RoundMode.CAST_NONE)
            pl.mul(q_fp32, q_fp32, g_exp_fp32)
            # q_eff cast to fp16 (feeds Phase 2-1 inter = q_eff @ S)
            pl.set_validshape(q16_fp16, [valid_rows, K])
            pl.cast(q16_fp16, q_fp32, mode=pl.RoundMode.CAST_ROUND)
            pl.store(ws_q16, q16_fp16, [work_id * C + row, 0])
            pl.set_validshape(q16_fp16, [HC, K])

            # S stored directly as fp16 (source is fp16, no cast needed)
            pl.load(s_fp16, s_snap, [head_id, global_chunk_id, sub_id * (K // 2), 0], order=[2, 3])
            pl.store(ws_s, s_fp16, [work_id * K + sub_id * (K // 2), 0])
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
            
            
            # v_corr stored directly as fp16 (source is fp16, no cast needed).
            # Zero-fill invalid rows first: Phase 4 matmul reads full [C, V] and
            # mask-zeroed aqk entries multiply these rows; uninitialised fp16
            # (torch.empty) can hold NaN bit patterns → 0 * NaN = NaN.
            pl.set_validshape(q_fp16, [HC, V])
            pl.expands(q_fp16, 0.0)
            pl.set_validshape(q_fp16, [valid_rows, V])
            pl.load(q_fp16, vcorr, [0, head_id, t_offset, 0], order=[2, 3])
            pl.set_validshape(q_fp16, [HC, V])
            pl.store(ws_v, q_fp16, [work_id * C + row, 0])
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=4)

        
        # ================================================================
        #  Phase 5 (Vec): o = inter + tmp; cast fp16; store (both sub-blocks)
        # ================================================================
            pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=3)
            pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=5)

            bi = inter_ub_fp32_grp.current()
            bt = g_ub_fp32_grp.current()
            bo = ub_oh.current()

            valid_rows = pl.max(0, pl.min(HC, valid_size - row))
            pl.add(bi, bi, bt)
            pl.set_validshape(bo, [HC, V])
            pl.cast(bo, bi, mode=pl.RoundMode.CAST_ROUND)
            pl.set_validshape(bo, [valid_rows, V])
            pl.store(o, bo, [0, t_base + row, head_id, 0], order=[1, 3])
            pl.system.bar_all()
            

        # ================================================================
        #  Phase 2 (Cube): inter = q_eff @ S;  aqk_raw = q_eff @ k_eff^T
        # ================================================================
        with pl.section_cube():
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)

            # Phase 2-1: inter = q_eff @ S  (fp16 inputs)
            cur_q16_l1 = q16_l1.current()
            cur_s_l1 = s_l1.current()
            cur_q16_l0a = q16_l0a.current()
            cur_s_l0b = s_l0b.current()
            cur_inter_acc = inter_acc.current()

            pl.load(cur_q16_l1, ws_q16, [work_id * C, 0])
            pl.load(cur_s_l1, ws_s, [work_id * K, 0])
            pl.move(cur_q16_l0a, cur_q16_l1)
            pl.move(cur_s_l0b, cur_s_l1)
            pl.matmul(cur_inter_acc, cur_q16_l0a, cur_s_l0b)
            pl.move(inter_ub_fp32_grp.current(), cur_inter_acc,
                    acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=5)

        # ================================================================
        #  Phase 4 (Cube): tmp = aqk @ v_corr
        # ================================================================
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=4)

            cur_aqk_l1 = aqk_l1.current()
            cur_v_l1 = v_l1.current()
            cur_aqk_l0a = aqk_l0a.current()
            cur_v_l0b = v_l0b.current()
            cur_tmp_acc = tmp_acc.current()

            pl.load(cur_aqk_l1, ws_aqk, [work_id * C, 0])
            pl.load(cur_v_l1, ws_v, [work_id * C, 0])
            pl.move(cur_aqk_l0a, cur_aqk_l1)
            pl.move(cur_v_l0b, cur_v_l1)
            pl.matmul(cur_tmp_acc, cur_aqk_l0a, cur_v_l0b)
            pl.move(g_ub_fp32_grp.current(), cur_tmp_acc,
                    acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=3)
            pl.system.bar_all()        


def run_chunk_o_kda(q, v_corr, s_snapshots, g_cs, o, ws_aqk,
                    num_chunks, num_cores, cu_seqlens_list=None):
    """Run the fused chunk_o_kda kernel on NPU.

    Args:
        q:             [1, T, HV, K]  fp16   queries (BSND)
        v_corr:        [1, HV, T, V]  fp16   corrected values (BNTD)
        s_snapshots:   [HV, n_chunks, K, V]  fp16   state snapshots (HNCV)
        g_cs:          [1, HV, T, K]  fp32   cumulative gate sum (BNTD)
        o:             [1, T, HV, V]  fp16   output (BSND)
        ws_aqk:        [nc*C, C] fp16  aqk (masked) produced by gate_kkt
        num_chunks:    int
        num_cores:     int
        cu_seqlens_list: list[int] | None
    """
    q = q.to(_DEVICE)
    v_corr = v_corr.to(_DEVICE)
    s_snapshots = s_snapshots.to(_DEVICE)
    g_cs = g_cs.to(_DEVICE)
    ws_aqk = ws_aqk.to(_DEVICE)
    o = o.to(_DEVICE)

    n_cores = num_cores
    T = q.shape[1]
    HV = q.shape[2]

    cu_seqlens_tensor, chunk_tbase_t, chunk_valid_t, _ = build_chunk_tables(T, cu_seqlens_list, C, _DEVICE)
    num_seqs = 1 if cu_seqlens_list is None else len(cu_seqlens_list) - 1

    tw_tot = num_chunks * HV
    ws_q16 = torch.empty(tw_tot * C, K, device=_DEVICE, dtype=torch.float16)
    ws_v   = torch.empty(tw_tot * C, V, device=_DEVICE, dtype=torch.float16)
    ws_s   = torch.empty(tw_tot * K, V, device=_DEVICE, dtype=torch.float16)

    chunk_o_kda_kernel[None, n_cores](
        q, g_cs, v_corr, s_snapshots, o,
        ws_q16, ws_v, ws_s, ws_aqk,
        HV, T, cu_seqlens_tensor, num_seqs,
        chunk_tbase_t, chunk_valid_t,
    )
    torch.npu.synchronize()
