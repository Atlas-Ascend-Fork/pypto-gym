# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""wy_kda operator — WY representation for KDA chunk recurrence.

Implements Stage 4 of the KDA (Kernelized Decay Attention) pipeline:
computes the two auxiliary tensors U and W per chunk that decouple the
within-chunk solve from the cross-chunk recurrence.

Math (per chunk of size C, per head):
    V_scaled = v * beta              # [C, V]  (row-scale by beta)
    K_eff    = k * exp(g_cs) * beta  # [C, K]  (per-dim gate + row-scale)
    u = A_inv @ V_scaled             # [C, V]
    w = A_inv @ K_eff                # [C, K]

All computation is inside a single fused kernel with three phases:
  Phase 1 (Vec):  V_scaled = v*beta, K_eff = k*exp(g_cs)*beta → GM workspace
  Phase 2 (Cube): u = A_inv @ V_scaled, w = A_inv @ K_eff      → GM workspace
  Phase 3 (Vec):  cast u, w to fp16                            → u_out, w_out

Beta row-scaling uses expand_mul(dim=0) with a [C,1] beta column tile.

Supports variable-length packed sequences via cu_seqlens (tail chunks).
"""

import torch
import pypto_pro.language as pl

from .kda_common import C, K, V, HC, DEVICE as _DEVICE, build_chunk_tables


TILE_N = 128

# ---- UB (Vec) addresses — HC per sub-block（sub_id=0 → rows [0,HC), sub_id=1 → [HC,C)）----
UB_VH   = 0x00000   # [HC, V] fp16  16 KB
UB_VF   = 0x04000   # [HC, V] fp32  32 KB
UB_TV   = 0x0C000   # [HC, V] fp32  32 KB  (expand_mul result)
UB_G    = 0x14000   # [HC, K] fp32  32 KB  (g_cs + exp)

UB_BETA = 0x1C000   # [1, HC] fp16  128 B
UB_BF   = 0x1C080   # [1, HC] fp32  256 B  (beta fp32 + [HC,1] col view)

# ---- L1 (Mat) addresses ----
L1_A   = 0x00000   # [C, C]  fp16 NZ  32 KB  — A_inv (loaded once)
L1_X   = 0x08000   # [C, N]  fp16 NZ  32 KB  — V_scaled / K_eff


@pl.jit(auto_mutex=True)
def wy_kda_kernel(
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    g_cs: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    beta: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    A_inv: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    u_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    w_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    ws_vs: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    ws_ke: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    cu_seqlens: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    chunk_tbase: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    chunk_valid: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
):
    """Fused wy_kda: exp + beta + matmul + cast in one kernel.

    Inputs (all on GM):
        k:           [B, T, HV, K]  fp16   (BSND)
        v:           [B, T, HV, V]  fp16   (BSND)
        g_cs:        [B, HV, T, K]  fp32   (BNTD)
        beta:        [B, HV, T]     fp16   beta head-major (3D, BNTD)
        A_inv:       [B, HV, T, C]  fp32   (BNTD)
        ws_vs:       [n_cores * C, V]  fp16  per-core workspace for V_scaled
        ws_ke:       [n_cores * C, K]  fp16  per-core workspace for K_eff
        cu_seqlens:  [num_seqs+1]   int32  cumulative sequence lengths
        chunk_tbase: [num_chunks]   int32  pre-computed t_base per chunk
        chunk_valid: [num_chunks]   int32  pre-computed valid_size per chunk

    Outputs:
        u_out:   [B, HV, T, V]  fp32   (BNTD)
        w_out:   [B, HV, T, K]  fp32   (BNTD)
    """
    T = k.shape[1]
    HV_dim = k.shape[2]
    num_seqs = cu_seqlens.shape[0] - 1
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    sub_id = pl.get_subblock_idx()
    ro = sub_id * HC

    if num_seqs == 1:
        num_chunks = (T + C - 1) // C
    else:
        num_chunks = chunk_tbase.shape[0]
    total_work = num_chunks * HV_dim

    # ── Tile groups ──

    # Vec tiles (HC per sub-block)
    t_vh = pl.make_tile_group(
        type=pl.TileType(shape=[HC, V], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_VH, mutex_ids=[0])
    t_vf = pl.make_tile_group(
        type=pl.TileType(shape=[HC, V], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_VF, mutex_ids=[1])
    t_tv = pl.make_tile_group(
        type=pl.TileType(shape=[HC, V], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_TV, mutex_ids=[2])

    t_g = pl.make_tile_group(
        type=pl.TileType(shape=[HC, K], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_G, mutex_ids=[4])
    t_beta = pl.make_tile_group(
        type=pl.TileType(shape=[1, HC], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_BETA, mutex_ids=[6])
    t_bf = pl.make_tile_group(
        type=pl.TileType(shape=[1, HC], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_BF, mutex_ids=[7])
    t_bf_col = pl.make_tile_group(
        type=pl.TileType(shape=[HC, 1], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, layout=pl.DN),
        addrs=UB_BF, mutex_ids=[7])

    # Cube tiles
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=L1_A, mutex_ids=[9])
    x_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[C, TILE_N], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=L1_X, mutex_ids=[10])
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0, mutex_ids=[11])
    x_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[C, TILE_N], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0, mutex_ids=[12])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[C, TILE_N], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         fractal=1024),
        addrs=0x0, mutex_ids=[13])

    # ── Common tile references ──
    vh = t_vh.current()
    vf = t_vf.current()
    tv = t_tv.current()

    g  = t_g.current()
    bv = t_beta.current()
    bf = t_bf.current()
    bf_col = t_bf_col.current()
    cur_a = a_l1.current()
    cur_x = x_l1.current()
    al = a_l0a.current()
    xr = x_l0b.current()
    ac = acc.current()

    # ── Work loop ──
    for work_id in pl.range(core_id, total_work, num_cores):
        chunk_id = work_id // HV_dim
        head_id = work_id % HV_dim

        if num_seqs == 1:
            t_base = chunk_id * C
            valid_size = pl.min(T - t_base, C)
        else:
            t_base = pl.getval(chunk_tbase, chunk_id)
            valid_size = pl.getval(chunk_valid, chunk_id)

        # ================================================================
        #  Vec 1: beta + V_scaled → GM, signal Cube  (HC per sub-block)
        # ================================================================
        valid_rows = pl.max(0, pl.min(HC, valid_size - ro))
        with pl.section_vector():
            pl.set_validshape(bv, [1, valid_rows])
            pl.load(bv, beta, [0, head_id, t_base + ro], order=[0, 2])
            pl.set_validshape(bv, [1, HC])
            pl.cast(bf, bv, mode=pl.RoundMode.CAST_NONE)

            pl.set_validshape(vh, [valid_rows, V])
            pl.load(vh, v, [0, t_base + ro, head_id, 0], order=[1, 3])
            pl.set_validshape(vh, [HC, V])
            pl.cast(vf, vh, mode=pl.RoundMode.CAST_NONE)
            pl.expand_mul(tv, vf, bf_col, dim=0)
            pl.set_validshape(vh, [HC, V])
            pl.expands(vh, 0.0)
            pl.set_validshape(vh, [valid_rows, V])
            pl.cast(vh, tv, mode=pl.RoundMode.CAST_ROUND)
            pl.set_validshape(vh, [HC, V])
            pl.store(ws_vs, vh, [core_id * C + ro, 0])

            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=0)

        # ================================================================
        #  Vec 2: K_eff = k*exp(g_cs)*beta → GM  (HC per sub-block)
        # ================================================================
            pl.set_validshape(vh, [valid_rows, K])
            pl.load(vh, k, [0, t_base + ro, head_id, 0], order=[1, 3])
            pl.set_validshape(vh, [HC, K])
            pl.cast(vf, vh, mode=pl.RoundMode.CAST_NONE)
            pl.set_validshape(g, [valid_rows, K])
            pl.load(g, g_cs, [0, head_id, t_base + ro, 0], order=[2, 3])
            pl.set_validshape(g, [HC, K])
            pl.exp(g, g)
            pl.mul(vf, vf, g)
            pl.expand_mul(tv, vf, bf_col, dim=0)
            pl.set_validshape(vh, [HC, K])
            pl.expands(vh, 0.0)
            pl.set_validshape(vh, [valid_rows, K])
            pl.cast(vh, tv, mode=pl.RoundMode.CAST_ROUND)
            pl.set_validshape(vh, [HC, K])
            pl.store(ws_ke, vh, [core_id * C + ro, 0])

            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)
            
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=3)       

        # ================================================================
        #  Cube 1: load A_inv ONCE, u = A_inv @ V_scaled
        # ================================================================
        with pl.section_cube():
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)

            pl.load(cur_a, A_inv, [0, head_id, t_base, 0], order=[2, 3])
            pl.move(al, cur_a)

            pl.load(cur_x, ws_vs, [core_id * C, 0])
            pl.move(xr, cur_x)
            pl.set_validshape(ac, [C, V])
            pl.matmul(ac, al, xr)
            pl.set_validshape(ac, [valid_size, V])
            pl.store(u_out, ac, [0, head_id, t_base, 0], order=[2, 3])

            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=1)

        # ================================================================
        #  Cube 2: w = A_inv @ K_eff  (A_inv already in L0A)
        # ================================================================
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=2)

            pl.load(cur_x, ws_ke, [core_id * C, 0])
            pl.move(xr, cur_x)
            pl.set_validshape(ac, [C, K])
            pl.matmul(ac, al, xr)
            pl.set_validshape(ac, [valid_size, K])
            pl.store(w_out, ac, [0, head_id, t_base, 0], order=[2, 3])

            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=3)


def wy_kda_block(
    k: torch.Tensor,
    v: torch.Tensor,
    g_cs: torch.Tensor,
    beta_sig: torch.Tensor,
    A_inv: torch.Tensor,
    chunk_size: int = C,
    cu_seqlens=None,
    num_cores: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute wy_kda using fused block kernel.

    Args:
        k:        [B, T, HV, K]  fp16   (BSND)
        v:        [B, T, HV, V]  fp16   (BSND)
        g_cs:     [B, HV, T, K]  fp32   (BNTD)
        beta_sig: [B, T, HV]     fp16   (BSND, permuted internally to BNTD)
        A_inv:    [B, HV, T, C]  fp32   (BNTD)
        chunk_size: int
        cu_seqlens: list | None  cumulative sequence lengths
        num_cores: int | None  None → auto-detect from device cube_core_num

    Returns:
        u: [B, HV, T, V] fp32   (BNTD)
        w: [B, HV, T, K] fp32   (BNTD)
    """
    orig_device = k.device
    k = k.to(_DEVICE)
    v = v.to(_DEVICE)
    g_cs = g_cs.to(_DEVICE)
    beta_sig = beta_sig.to(_DEVICE)
    A_inv = A_inv.to(_DEVICE).half()

    B, T, HV_dim, Kd = k.shape
    Vd = v.shape[-1]

    cu_seqlens_t, chunk_tbase_t, chunk_valid_t, num_chunks = build_chunk_tables(T, cu_seqlens, chunk_size, _DEVICE)

    total_work = num_chunks * HV_dim
    if num_cores is None:
        num_cores = torch.npu.get_device_properties(0).cube_core_num
    nc = min(num_cores, total_work)

    beta_t = beta_sig.permute(0, 2, 1).contiguous()

    ws_vs = torch.empty(nc * C, Vd, device=_DEVICE, dtype=torch.float16)
    ws_ke = torch.empty(nc * C, Kd, device=_DEVICE, dtype=torch.float16)

    u_out = torch.zeros(B, HV_dim, T, Vd, device=_DEVICE, dtype=torch.float32)
    w_out = torch.zeros(B, HV_dim, T, Kd, device=_DEVICE, dtype=torch.float32)

    wy_kda_kernel[None, nc](
        k, v, g_cs, beta_t, A_inv,
        u_out, w_out,
        ws_vs, ws_ke,
        cu_seqlens_t, chunk_tbase_t, chunk_valid_t,
    )
    torch.npu.synchronize()

    return u_out.to(orig_device), w_out.to(orig_device)
