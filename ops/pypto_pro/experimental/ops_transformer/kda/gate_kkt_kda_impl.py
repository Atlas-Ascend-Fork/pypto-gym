# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""gate_kkt_kda — fused gate_cumsum + kkt_kda with Phase 3 M-split.

Phase 3 (the most Vec-intensive phase with 2x exp + 2x mul) is split across
two Vec sub-blocks by row range: sub_id=0 → rows [0,HC), sub_id=1 → [HC,C).
Each sub-block processes HC=64 rows instead of C=128, halving the per-sub-block
Vec workload for exp/mul.

Phases 1, 3, 5 run on the AIV (single section_vector); Phases 2, 4 run on the
AIC (single section_cube), mirroring chunk_h_kda's section layout.

Pipeline:
    Vec1:  g fp16 → ws_g (fp16)                       (sub-block split)
    Cub2:  g_cs = L @ g (fp16 inputs, fp32 acc)  → acc→UB (gk), GM g_cs
    Vec3:  per sub-block [HC,K]: A = k*exp(g_cs) → ws_a, B → ws_b
    Cub4:  L_full = A @ B^T (fp32 inputs)        → acc→UB (l5)
    Vec5:  beta scale + mask + cast → L_out

L0C→UB uses pl.move(acc_to_vec_mode=DualModeSplitM), so no GM workspace
round-trip for g_cs / L_full.
"""

import torch
import pypto_pro.language as pl

from .kda_common import C, K, HC, DEVICE as _DEVICE, build_chunk_tables


# ---- UB (Vec) addresses ----
UB_GH  = 0x00000   # [HC, K] fp16  16 KB  (Phase 1 g)
UB_KH  = 0x04000   # [HC, K] fp16  16 KB  (Phase 3 k)
UB_KF  = 0x08000   # [HC, K] fp32  32 KB  (Phase 3 k cast / kg_a)
UB_G   = 0x10000   # [HC, K] fp32  32 KB  (Phase 2 gk / g_cs band)
UB_E   = 0x18000   # [HC, K] fp32  32 KB  (Phase 3 exp / row_dec)
UB_GC  = 0x20000   # [HC, K] fp32  32 KB  (Phase 3 g_cs 分块; Phase 5 复用 L5)
UB_KI  = 0x28000   # [HC, K] fp32  32 KB  (Phase 3 ki 分块; Phase 5 复用 T5)
UB_GP  = 0x30000   # [1, K]  fp32  512 B  (Phase 3 pivot)
UB_MASK  = 0x30200  # [HC, C] fp32 32 KB (Phase 5 mask)
UB_BETA5 = 0x38200  # [1, HC] fp16  128 B
UB_BF5   = 0x38280  # [1, HC] fp32  256 B
UB_L16_5 = 0x38400  # [HC, C] fp16  16 KB

CLAMP = 80.0   # pivot 分解 exp 输入对称 clamp

# ---- L1 (Mat) addresses ----
L1_L = 0x00000   # [C, C]  fp16 NZ  32 KB  Phase 2
L1_G = 0x08000   # [C, K]  fp16 NZ  32 KB  Phase 2
L1_A = 0x10000   # [HC, K] fp32 NZ  32 KB  Phase 4 band kg_a
L1_B = 0x20000   # [K, C]  fp32 ZN  64 KB  Phase 4 band ki_a


@pl.jit(auto_mutex=True)
def gate_kkt_kda_kernel(
    g: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    beta: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    L_mat: pl.Tensor[[C, C], pl.DT_FP16],
    mask: pl.Tensor[[C, C], pl.DT_FP32],
    mask_incl: pl.Tensor[[C, C], pl.DT_FP32],
    g_cs: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    L_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    ws_g: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    ws_a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    ws_ki: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    ws_q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    ws_aqk: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    cu_seqlens: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    chunk_tbase: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    chunk_valid: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
):
    """Fused gate_cumsum + kkt_kda with Phase 3 M-split."""
    T = g.shape[1]
    HV_dim = g.shape[2]
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    sub_id = pl.get_subblock_idx()
    ro = sub_id * HC

    num_seqs = cu_seqlens.shape[0] - 1
    if num_seqs == 1:
        num_chunks = (T + C - 1) // C
    else:
        num_chunks = chunk_tbase.shape[0]
    total_work = num_chunks * HV_dim

    # ── Phase 1 tiles ──
    t_gh = pl.make_tile_group(
        type=pl.TileType(shape=[HC, K], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec,
                         pad=pl.TilePad.zero, valid_shape=[-1, -1]),
        addrs=UB_GH, mutex_ids=[0])

    # ── Phase 2 tiles (fp16 in, fp32 acc) ──
    L_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=L1_L, mutex_ids=[2])
    g_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[C, K], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=L1_G, mutex_ids=[3])
    L_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0, mutex_ids=[4])
    g_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[C, K], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0, mutex_ids=[5])
    acc_g = pl.make_tile_group(
        type=pl.TileType(shape=[C, K], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         fractal=1024),
        addrs=0x0, mutex_ids=[6])

    # ── Phase 3 tiles ──
    t_kh = pl.make_tile_group(
        type=pl.TileType(shape=[HC, K], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec,
                         pad=pl.TilePad.zero, valid_shape=[-1, -1]),
        addrs=UB_KH, mutex_ids=[7])
    t_kf = pl.make_tile_group(
        type=pl.TileType(shape=[HC, K], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_KF, mutex_ids=[8])
    t_gk = pl.make_tile_group(
        type=pl.TileType(shape=[HC, K], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         pad=pl.TilePad.zero, valid_shape=[-1, -1]),
        addrs=UB_G, mutex_ids=[9])
    t_e = pl.make_tile_group(
        type=pl.TileType(shape=[HC, K], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_E, mutex_ids=[10])
    t_gp = pl.make_tile_group(
        type=pl.TileType(shape=[1, K], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_GP, mutex_ids=[22])
    t_gc = pl.make_tile_group(
        type=pl.TileType(shape=[HC, K], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_GC, mutex_ids=[23])
    t_ki = pl.make_tile_group(
        type=pl.TileType(shape=[HC, K], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_KI, mutex_ids=[24])

    # ── Phase 4 tiles (fp32 in, fp32 acc) ──
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[C, K], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=L1_A, mutex_ids=[11])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=L1_B, mutex_ids=[12])
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[C, K], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0, mutex_ids=[4])
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[K, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0, mutex_ids=[5])
    acc_k = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         fractal=1024),
        addrs=0x0, mutex_ids=[6])
    acc_k2 = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         fractal=1024),
        addrs=0x10000, mutex_ids=[29])

    # ── Phase 5 tiles ──
    t_l5 = pl.make_tile_group(
        type=pl.TileType(shape=[HC, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_GC, mutex_ids=[23])
    t_t5 = pl.make_tile_group(
        type=pl.TileType(shape=[HC, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_KI, mutex_ids=[24])
    t_beta5 = pl.make_tile_group(
        type=pl.TileType(shape=[1, HC], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec,
                         pad=pl.TilePad.zero, valid_shape=[-1, -1]),
        addrs=UB_BETA5, mutex_ids=[18])
    t_bf5 = pl.make_tile_group(
        type=pl.TileType(shape=[1, HC], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_BF5, mutex_ids=[19])
    t_l16_5 = pl.make_tile_group(
        type=pl.TileType(shape=[HC, C], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec,
                         pad=pl.TilePad.zero, valid_shape=[-1, -1]),
        addrs=UB_L16_5, mutex_ids=[20])
    t_bf5_col = pl.make_tile_group(
        type=pl.TileType(shape=[HC, 1], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, layout=pl.DN),
        addrs=UB_BF5, mutex_ids=[19])
    t_mask5 = pl.make_tile_group(
        type=pl.TileType(shape=[HC, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_MASK, mutex_ids=[25])
    t_l5aqk = pl.make_tile_group(
        type=pl.TileType(shape=[HC, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_GH, mutex_ids=[0])
    t_l5b = pl.make_tile_group(
        type=pl.TileType(shape=[HC, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_KF, mutex_ids=[8])
    t_l5aqk_b = pl.make_tile_group(
        type=pl.TileType(shape=[HC, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_E, mutex_ids=[10])

    # ── Tile references ──
    gh = t_gh.current()
    cur_L = L_l1.current()
    cur_g = g_l1.current()
    al_g = L_l0a.current()
    br_g = g_l0b.current()
    ac_g = acc_g.current()

    kh = t_kh.current()
    kf = t_kf.current()
    gk = t_gk.current()
    e = t_e.current()
    gp = t_gp.current()
    gc = t_gc.current()
    ki = t_ki.current()
    cur_a = a_l1.current()
    cur_b = b_l1.current()
    al_k = a_l0a.current()
    br_k = b_l0b.current()
    ac_k = acc_k.current()
    ac_k2 = acc_k2.current()

    lv5 = t_l5.current()
    lv5aqk = t_l5aqk.current()
    lv5b = t_l5b.current()
    lv5aqk_b = t_l5aqk_b.current()
    tv5 = t_t5.current()
    bv5 = t_beta5.current()
    bf5 = t_bf5.current()
    bf5_col = t_bf5_col.current()
    l16_5 = t_l16_5.current()

    # ── Work loop ──
    pl.system.bar_all()
    for work_id in pl.range(core_id, total_work, num_cores):
        # pl.system.bar_all()
        chunk_id = work_id // HV_dim
        head_id = work_id % HV_dim

        if num_seqs == 1:
            t_base = chunk_id * C
            valid_size = pl.min(T - t_base, C)
        else:
            t_base = pl.getval(chunk_tbase, chunk_id)
            valid_size = pl.getval(chunk_valid, chunk_id)

        valid_rows = pl.max(0, pl.min(HC, valid_size - ro))
        valid_rows2 = pl.max(0, pl.min(HC, valid_size))
        valid_rows3 = pl.max(0, pl.min(HC, valid_size - HC))

        # ================================================================
        #  section_vector:  Vec1 (g→ws_g) → Vec3 (A/B→ws_a/ws_b) → Vec5 (→L_out)
        # ================================================================
        with pl.section_vector():
            # ---- Phase 1 (Vec): g fp16 → ws_g (fp16)  (split, HC per sub-block) ----
            if valid_size < C:
                pl.set_validshape(gh, [HC, K])
                pl.expands(gh, 0.0)
                pl.store(ws_g, gh, [core_id * C + ro, 0])
                pl.set_validshape(e, [HC, K])
                pl.expands(e, 0.0)
                pl.store(ws_a, e, [core_id * C + ro, 0])
            pl.set_validshape(gh, [valid_rows, K])
            pl.load(gh, g, [0, t_base + ro, head_id, 0], order=[1, 3])
            pl.set_validshape(gh, [HC, K])
            pl.set_validshape(gh, [valid_rows, K])
            pl.store(ws_g, gh, [core_id * C + ro, 0])
            pl.set_validshape(gh, [HC, K])

            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=0)

            # ---- Phase 3 (Vec): 分带局部 pivot 分解 (每 band 自己的中间行 pivot) ----
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=1)

            # pivot = gk 的 band 中间行 (本地 UB, Phase 2 DMS 搬入)
            pl.system.bar_all()
            pl.move(gp, gk, [HC // 2, 0])   # [1,K]

            # 行因子: e = gk - gp (gk = g_cs_band), clamp ±80
            pl.set_validshape(kh, [valid_rows, K])
            pl.load(kh, k, [0, t_base + ro, head_id, 0], order=[1, 3])
            pl.set_validshape(kh, [HC, K])
            pl.cast(kf, kh, mode=pl.RoundMode.CAST_NONE)
            pl.set_validshape(e, [HC, K])
            pl.expand_sub(e, gk, gp, dim=1)
            pl.minimum(e, e, CLAMP)
            pl.maximum(e, e, -CLAMP)
            pl.exp(ki, e)                       # ki = exp(e) 行因子 exp

            # qg_a = q * exp(e)  半段
            pl.set_validshape(kh, [valid_rows, K])
            pl.load(kh, q, [0, t_base + ro, head_id, 0], order=[1, 3])
            pl.set_validshape(kh, [HC, K])
            pl.cast(gc, kh, mode=pl.RoundMode.CAST_NONE)
            pl.mul(gc, gc, ki)
            pl.set_validshape(e, [HC, K])
            pl.expands(e, 0.0)
            pl.store(ws_q, e, [core_id * 2 * C + HC, 0])
            pl.store(ws_q, e, [core_id * 2 * C + C, 0])
            if valid_rows < HC:
                pl.store(ws_q, e, [core_id * 2 * C + C + HC, 0])
            pl.set_validshape(gc, [valid_rows, K])
            pl.store(ws_q, gc, [core_id * 2 * C + sub_id * C + ro, 0])
            pl.set_validshape(gc, [HC, K])

            # kg_a = k * exp(e)
            pl.mul(ki, kf, ki)
            pl.set_validshape(e, [HC, K])
            pl.expands(e, 0.0)
            pl.store(ws_a, e, [core_id * 2 * C + HC, 0])
            pl.store(ws_a, e, [core_id * 2 * C + C, 0])
            if valid_rows < HC:
                pl.store(ws_a, e, [core_id * 2 * C + C + HC, 0])
            pl.set_validshape(ki, [valid_rows, K])
            pl.store(ws_a, ki, [core_id * 2 * C + sub_id * C + ro, 0])
            pl.set_validshape(ki, [HC, K])

            # 列因子 ki_a = k_full * exp(gp - g_cs_full)  前半 [0:64] (读 g_cs 全列)
            pl.set_validshape(gc, [valid_rows2, K])
            pl.load(gc, g_cs, [0, head_id, t_base, 0], order=[2, 3])
            pl.set_validshape(gc, [HC, K])
            pl.set_validshape(kh, [valid_rows2, K])
            pl.load(kh, k, [0, t_base + 0, head_id, 0], order=[1, 3])
            pl.set_validshape(kh, [HC, K])
            pl.cast(kf, kh, mode=pl.RoundMode.CAST_NONE)
            pl.expand_sub(ki, gp, gc, dim=1)
            pl.minimum(ki, ki, CLAMP)
            pl.maximum(ki, ki, -CLAMP)
            pl.exp(ki, ki)
            pl.mul(ki, kf, ki)
            if valid_rows2 < HC:
                pl.set_validshape(e, [HC, K])
                pl.expands(e, 0.0)
                pl.store(ws_ki, e, [core_id * 2 * C + sub_id * C, 0])
            pl.set_validshape(ki, [valid_rows2, K])
            pl.store(ws_ki, ki, [core_id * 2 * C + sub_id * C, 0])
            pl.set_validshape(ki, [HC, K])

            # 列因子后半 [64:128]
            pl.set_validshape(gc, [valid_rows3, K])
            pl.load(gc, g_cs, [0, head_id, t_base + HC, 0], order=[2, 3])
            pl.set_validshape(gc, [HC, K])
            pl.set_validshape(kh, [valid_rows3, K])
            pl.load(kh, k, [0, t_base + HC, head_id, 0], order=[1, 3])
            pl.set_validshape(kh, [HC, K])
            pl.cast(kf, kh, mode=pl.RoundMode.CAST_NONE)
            pl.expand_sub(ki, gp, gc, dim=1)
            pl.minimum(ki, ki, CLAMP)
            pl.maximum(ki, ki, -CLAMP)
            pl.exp(ki, ki)
            pl.mul(ki, kf, ki)
            if valid_rows3 < HC:
                pl.set_validshape(e, [HC, K])
                pl.expands(e, 0.0)
                pl.store(ws_ki, e, [core_id * 2 * C + sub_id * C + HC, 0])
            pl.set_validshape(ki, [valid_rows3, K])
            pl.store(ws_ki, ki, [core_id * 2 * C + sub_id * C + HC, 0])
            pl.set_validshape(ki, [HC, K])

            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)
# ---- Phase 5 (Vec): beta scale + mask + cast → L_out; aqk → ws_aqk  (split) ----
            pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=3)

            if valid_size < C:
                pl.expands(bv5, 0.0)
            pl.set_validshape(bv5, [1, valid_rows])
            pl.load(bv5, beta, [0, head_id, t_base + ro])
            pl.set_validshape(bv5, [1, HC])
            pl.cast(bf5, bv5, mode=pl.RoundMode.CAST_NONE)

            mask5 = t_mask5.current()
            pl.set_validshape(mask5, [HC, C])
            pl.load(mask5, mask, [ro, 0])
            # L_out = beta * (lv5 + lv5b) * mask   (DualModeSplitM: 每 sub 持自己半段)
            pl.expand_mul(lv5b, lv5b, bf5_col, dim=0)
            pl.expand_mul(tv5, lv5, bf5_col, dim=0)
            pl.add(tv5, tv5, lv5b)
            pl.mul(tv5, tv5, mask5)
            pl.set_validshape(l16_5, [HC, C])
            pl.cast(l16_5, tv5, mode=pl.RoundMode.CAST_ROUND)
            pl.set_validshape(l16_5, [valid_rows, C])
            pl.store(L_out, l16_5, [0, head_id, t_base + ro, 0], order=[2, 3])

            # aqk = (lv5aqk + lv5aqk_b) * mask_incl → fp16 → ws_aqk
            pl.set_validshape(mask5, [HC, C])
            pl.load(mask5, mask_incl, [ro, 0])
            pl.add(tv5, lv5aqk, lv5aqk_b)
            pl.mul(tv5, tv5, mask5)
            pl.set_validshape(l16_5, [HC, C])
            pl.cast(l16_5, tv5, mode=pl.RoundMode.CAST_ROUND)
            pl.set_validshape(l16_5, [valid_rows, C])
            pl.store(ws_aqk, l16_5, [work_id * C + ro, 0])

        # ================================================================
        #  section_cube:  Cub2 (L@g→g_cs) → Cub4 (A@B^T → L_full)
        # ================================================================
        with pl.section_cube():
            # ---- Phase 2 (Cube): g_cs = L @ g  (fp16 inputs, fp32 acc) ----
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)

            pl.load(cur_L, L_mat, [0, 0])
            pl.load(cur_g, ws_g, [core_id * C, 0])
            pl.move(al_g, cur_L)
            pl.move(br_g, cur_g)
            pl.matmul(ac_g, al_g, br_g)
            pl.set_validshape(ac_g, [C, K])
            pl.move(gk, ac_g, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
            pl.set_validshape(ac_g, [valid_size, K])
            pl.store(g_cs, ac_g, [0, head_id, t_base, 0], order=[2, 3])

            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=1)
            pl.system.bar_all()

            # ---- Phase 4 (Cube): L_full = A @ B^T → UB l5 (fp32 inputs) ----
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=2)

            # band0 L_full = kg_a0 @ ki_a0^T
            pl.load(cur_a, ws_a, [core_id * 2 * C, 0])
            pl.load(cur_b, ws_ki, [core_id * 2 * C, 0], order=[1, 0])
            pl.move(al_k, cur_a)
            pl.move(br_k, cur_b)
            pl.matmul(ac_k, al_k, br_k)
            pl.set_validshape(ac_k, [C, C])
            pl.move(lv5, ac_k, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
            # band1 L_full = kg_a1 @ ki_a1^T
            pl.load(cur_a, ws_a, [core_id * 2 * C + C, 0])
            pl.load(cur_b, ws_ki, [core_id * 2 * C + C, 0], order=[1, 0])
            pl.move(al_k, cur_a)
            pl.move(br_k, cur_b)
            pl.matmul(ac_k2, al_k, br_k)
            pl.set_validshape(ac_k2, [C, C])
            pl.move(lv5b, ac_k2, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
            # band0 aqk = qg_a0 @ ki_a0^T  (共享 ki_a0)
            pl.load(cur_a, ws_q, [core_id * 2 * C, 0])
            pl.load(cur_b, ws_ki, [core_id * 2 * C, 0], order=[1, 0])
            pl.move(al_k, cur_a)
            pl.move(br_k, cur_b)
            pl.matmul(ac_k, al_k, br_k)
            pl.set_validshape(ac_k, [C, C])
            pl.move(lv5aqk, ac_k, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
            # band1 aqk = qg_a1 @ ki_a1^T  (共享 ki_a1)
            pl.load(cur_a, ws_q, [core_id * 2 * C + C, 0])
            pl.load(cur_b, ws_ki, [core_id * 2 * C + C, 0], order=[1, 0])
            pl.move(al_k, cur_a)
            pl.move(br_k, cur_b)
            pl.matmul(ac_k2, al_k, br_k)
            pl.set_validshape(ac_k2, [C, C])
            pl.move(lv5aqk_b, ac_k2, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)

            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=3)


def run_gate_kkt_kda(g, q, k, beta, L_mat, mask, mask_incl, g_cs, L_out,
                     num_cores, cu_seqlens=None, ws_aqk=None):
    """Launch the fused gate_cumsum + kkt_kda kernel.

    Args:
        g:       [B, T, HV, K]  fp16   raw gate values (BSND)
        q:       [B, T, HV, K]  fp16   queries (BSND)
        k:       [B, T, HV, K]  fp16   keys (BSND)
        beta:    [B, HV, T]     fp16   beta head-major (BNTD)
        L_mat:   [C, C]         fp32   lower-triangular ones (constant, cast to fp16 internally)
        mask:    [C, C]         fp32   strict lower-tri mask (L_out)
        mask_incl: [C, C]       fp32   inclusive lower-tri mask (aqk)
        g_cs:    [B, HV, T, K]  fp32   cumulative gate sum output (BNTD)
        L_out:   [B, HV, T, C]  fp16   output (zeroed, BNTD)
        num_cores: int          number of AI cores to launch
        cu_seqlens: list[int] | None   cumulative sequence lengths
        ws_aqk:  Tensor | None  [nc*C, C] fp16  aqk output workspace (allocated if None)
    Returns:
        ws_aqk:  Tensor         [nc*C, C] fp16  aqk output (shared with chunk_o)
    """
    B, T, HV_dim, Kd = g.shape

    g = g.to(_DEVICE)
    q = q.to(_DEVICE)
    k = k.to(_DEVICE)
    beta = beta.to(_DEVICE)
    L_mat = L_mat.to(_DEVICE).half()
    mask = mask.to(_DEVICE)
    mask_incl = mask_incl.to(_DEVICE)
    g_cs = g_cs.to(_DEVICE)
    L_out = L_out.to(_DEVICE)

    cu_seqlens_t, chunk_tbase_t, chunk_valid_t, _ = build_chunk_tables(T, cu_seqlens, C, _DEVICE)
    num_chunks = (T + C - 1) // C if cu_seqlens is None else chunk_tbase_t.shape[0]

    nc = num_cores
    ws_g  = torch.empty(nc * C, Kd, device=_DEVICE, dtype=torch.float16)
    ws_a  = torch.empty(nc * 2 * C, Kd, device=_DEVICE, dtype=torch.float32)
    ws_ki = torch.empty(nc * 2 * C, Kd, device=_DEVICE, dtype=torch.float32)
    ws_q  = torch.empty(nc * 2 * C, Kd, device=_DEVICE, dtype=torch.float32)
    if ws_aqk is None:
        ws_aqk = torch.zeros(num_chunks * HV_dim * C, C, device=_DEVICE, dtype=torch.float16)

    gate_kkt_kda_kernel[None, nc](
        g, q, k, beta, L_mat, mask, mask_incl, g_cs, L_out,
        ws_g, ws_a, ws_ki, ws_q, ws_aqk,
        cu_seqlens_t, chunk_tbase_t, chunk_valid_t,
    )
    torch.npu.synchronize()
    return ws_aqk
