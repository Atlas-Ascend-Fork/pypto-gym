# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F821
"""Delivery Cube projection for IndexerPrologueK (ACLNN ABI)."""

import dataclasses

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField

BASE_M = 16
BASE_N = 64
BASE_N_WIDE = 128
BASE_K = 128
NZ_ALIGN = 16


class ProjectionKey:
    NMode = TilingKeyField(bits=2, values=[0, 1, 2])
    KMode = TilingKeyField(bits=1, values=[0, 1])


@dataclasses.dataclass
class IndexerPrologueKProjectionTiling:
    t: int
    head_dim: int
    dim: int


@pl.jit(arch="3510", auto_mutex=True, compile_timeout=200,
        tiling_key=ProjectionKey, name="IndexerPrologueKProjection")
def indexer_prologue_k_projection_kernel(
    latent_ptr: pl.Ptr[pl.DT_UINT8],
    wk_ptr: pl.Ptr[pl.DT_UINT8],
    projected_ptr: pl.Ptr[pl.DT_UINT8],
    workspace_ptr: pl.Ptr[pl.DT_UINT8],
    tiling: IndexerPrologueKProjectionTiling,
):
    """Cube-only latent x wk projection into the projected GM tensor."""
    t = tiling.t
    head_dim = tiling.head_dim
    dim = tiling.dim

    latent = pl.make_tensor(latent_ptr, [t, head_dim], dtype=pl.DT_BF16)
    wk = pl.make_tensor(wk_ptr, [dim, head_dim], dtype=pl.DT_BF16)
    projected = pl.make_tensor(projected_ptr, [t, dim], dtype=pl.DT_BF16)

    if NMode == 0:
        base_n = BASE_N_WIDE
    elif NMode == 1:
        base_n = BASE_N
    else:
        base_n = NZ_ALIGN
    if KMode == 0:
        base_k = BASE_K
    else:
        base_k = NZ_ALIGN
    n_tiles = (dim + base_n - 1) // base_n
    k_tiles = (head_dim + base_k - 1) // base_k

    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx()
    m_tiles = (t + BASE_M - 1) // BASE_M
    total_tiles = m_tiles * n_tiles
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[BASE_M, base_k], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat),
        addrs=0, mutex_ids=[0, 1])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[base_k, base_n], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x10000, mutex_ids=[2, 3])
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[BASE_M, base_k], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Left, compact=1),
        addrs=0, mutex_ids=[4, 5])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[base_k, base_n], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Right, compact=1),
        addrs=0, mutex_ids=[6, 7])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[BASE_M, base_n], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, compact=1),
        addrs=0, mutex_ids=[8])
    with pl.section_cube():
        for tile_idx in pl.range(core_id, total_tiles, num_cores):
            m_idx = tile_idx // n_tiles
            n_idx = tile_idx % n_tiles
            valid_m = pl.min(BASE_M, t - m_idx * BASE_M)
            acc_tile = acc.next()
            pl.set_validshape(acc_tile, [valid_m, base_n])
            for k_idx in pl.range(0, k_tiles):
                valid_k = pl.min(base_k, head_dim - k_idx * base_k)
                a_tile = a_l1.next()
                b_tile = b_l1.next()
                pl.set_validshape(a_tile, [valid_m, valid_k])
                pl.load(a_tile, latent, [m_idx * BASE_M, k_idx * base_k])
                pl.set_validshape(b_tile, [valid_k, base_n])
                pl.load(b_tile, wk, [n_idx * base_n, k_idx * base_k], order=[1, 0])
                left_tile = a_left.next()
                right_tile = b_right.next()
                pl.set_validshape(left_tile, [valid_m, valid_k])
                pl.move(left_tile, a_tile)
                pl.set_validshape(right_tile, [valid_k, base_n])
                pl.move(right_tile, b_tile)
                if k_idx == 0:
                    pl.matmul(acc_tile, left_tile, right_tile)
                else:
                    pl.matmul_acc(acc_tile, acc_tile, left_tile, right_tile)
            pl.store(projected, acc_tile, [m_idx * BASE_M, n_idx * base_n])
        pl.system.sync_all(core_type=pl.SyncCoreType.AIC_ONLY)
