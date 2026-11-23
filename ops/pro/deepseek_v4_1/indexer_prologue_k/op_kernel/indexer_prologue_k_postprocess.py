# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F821
"""Delivery Vector postprocess for IndexerPrologueK (ACLNN ABI)."""

import dataclasses

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField

FP4_GROUP_SIZE = 32
E2M1_VALUES_PER_BYTE = 2


class PostKey:
    DimCap = TilingKeyField(bits=2, values=[512, 1024, 2048])


@dataclasses.dataclass
class IndexerPrologueKPostprocessTiling:
    t: int
    dim: int
    dr: int
    storage_mode: int
    combined_block_size: int
    write_scale_cache: int
    cache_blocks: int
    cache_slots: int
    cache_width: int
    scale_boxes: int
    epsilon: float


def _make_postprocess_row(threads):
    @pl.vector_function(mode="simt", max_threads=threads)
    def postprocess_row(proj, gamma, norm, pub, red, cos_t, sin_t, packed, scale,
                        epsilon, dim, tail_start):
        t = pl.simt.linear_thread_idx()
        lane = pl.simt.lane_id()
        if t == 0:
            red[0, 0] = 0.0
        pl.simt.syncthreads()
        x = pl.simt.cast(proj[0, t], pl.DT_FP32)
        partial = pl.simt.warp_reduce_add(x * x)
        if lane == 0:
            if t < dim:
                pl.simt.atomic_add(red, [0, 0], partial)
        pl.simt.syncthreads()
        if t == 0:
            red[1, 0] = pl.simt.rsqrt(red[0, 0] * (1.0 / dim) + epsilon)
        pl.simt.syncthreads()
        inverse = red[1, 0]
        if t < dim:
            v = x * inverse * gamma[0, t]
            norm[0, t] = pl.simt.cast(v, pl.DT_BF16, mode=pl.RoundMode.CAST_RINT)
        pl.simt.syncthreads()
        if t < dim:
            if t < tail_start:
                pub[0, t] = norm[0, t]
            else:
                local = t - tail_start
                ce = cos_t[0, local]
                se = sin_t[0, local]
                if local % 2 == 0:
                    a = pl.simt.cast(norm[0, t], pl.DT_FP32)
                    b = pl.simt.cast(norm[0, t + 1], pl.DT_FP32)
                    r = a * ce - b * se
                else:
                    b = pl.simt.cast(norm[0, t], pl.DT_FP32)
                    a = pl.simt.cast(norm[0, t - 1], pl.DT_FP32)
                    r = b * ce + a * se
                pub[0, t] = pl.simt.cast(r, pl.DT_BF16, mode=pl.RoundMode.CAST_RINT)
            val = pl.simt.cast(pub[0, t], pl.DT_FP32)
            group_amax = pl.simt.warp_reduce_max(pl.simt.abs(val))
            safe_amax = pl.simt.max(group_amax, 6.0 * (2.0 ** -126))
            raw_scale = safe_amax * (1.0 / 6.0)
            bits = pl.simt.bitcast(raw_scale, pl.DT_UINT32)
            exponent = pl.simt.cast((bits >> 23) & 0xFF, pl.DT_UINT32)
            if (bits & 0x7FFFFF) != 0:
                exponent = pl.simt.cast(exponent + 1, pl.DT_UINT32)
            inv_scale = pl.simt.bitcast(pl.simt.cast((254 - exponent) << 23, pl.DT_UINT32), pl.DT_FP32)
            n = pl.simt.abs(val) * inv_scale
            code = 7
            if n <= 5.0:
                code = 6
            if n <= 3.5:
                code = 5
            if n <= 2.5:
                code = 4
            if n <= 1.75:
                code = 3
            if n <= 1.25:
                code = 2
            if n <= 0.75:
                code = 1
            if n <= 0.25:
                code = 0
            if val < 0.0:
                code = code + 8
            odd_code = pl.simt.warp_shfl_xor(code, 1)
            if lane % 2 == 0:
                packed[0, t // 2] = pl.simt.cast(code | (odd_code << 4), pl.DT_UINT8)
            if lane == 0:
                scale[0, t // FP4_GROUP_SIZE] = pl.simt.cast(exponent, pl.DT_UINT8)
    return postprocess_row


_postprocess_row_512 = _make_postprocess_row(512)
_postprocess_row_1024 = _make_postprocess_row(1024)
_postprocess_row_2048 = _make_postprocess_row(2048)


@pl.jit(arch="3510", auto_mutex=True, compile_timeout=200,
        tiling_key=PostKey, name="IndexerPrologueKPostprocess")
def indexer_prologue_k_postprocess_kernel(
    projected_ptr: pl.Ptr[pl.DT_UINT8],
    gamma_ptr: pl.Ptr[pl.DT_UINT8],
    rope_sin_ptr: pl.Ptr[pl.DT_UINT8],
    rope_cos_ptr: pl.Ptr[pl.DT_UINT8],
    cache_index_ptr: pl.Ptr[pl.DT_UINT8],
    cache_in_ptr: pl.Ptr[pl.DT_UINT8],
    scale_in_ptr: pl.Ptr[pl.DT_UINT8],
    cache_out_ptr: pl.Ptr[pl.DT_UINT8],
    scale_out_ptr: pl.Ptr[pl.DT_UINT8],
    workspace_ptr: pl.Ptr[pl.DT_UINT8],
    tiling: IndexerPrologueKPostprocessTiling,
):
    """Vector-only RMSNorm / RoPE / MXFP4 scatter from projected GM rows."""
    t = tiling.t
    dim = tiling.dim
    rd = tiling.dr
    storage_mode = tiling.storage_mode
    group_size = tiling.combined_block_size
    write_scale_cache = tiling.write_scale_cache
    cache_blocks = tiling.cache_blocks
    cache_slots = tiling.cache_slots
    cache_width = tiling.cache_width
    scale_boxes = tiling.scale_boxes
    epsilon = tiling.epsilon
    write_actual_scale = storage_mode == 0 and write_scale_cache != 0
    value_bytes = dim // E2M1_VALUES_PER_BYTE
    scale_bytes = dim // FP4_GROUP_SIZE
    tail_start = dim - rd
    raw_block_size = cache_slots * group_size

    projected = pl.make_tensor(projected_ptr, [t, dim], dtype=pl.DT_BF16)
    gamma = pl.make_tensor(gamma_ptr, [1, dim], dtype=pl.DT_FP32)
    rope_sin = pl.make_tensor(rope_sin_ptr, [t, rd], dtype=pl.DT_FP32)
    rope_cos = pl.make_tensor(rope_cos_ptr, [t, rd], dtype=pl.DT_FP32)
    cache_index = pl.make_tensor(cache_index_ptr, [t], dtype=pl.DT_INT64)
    cache2 = pl.make_tensor(cache_out_ptr, [cache_blocks * cache_slots, cache_width],
                            dtype=pl.DT_UINT8)
    scale2 = pl.make_tensor(scale_out_ptr, [cache_blocks * cache_slots, scale_boxes],
                            dtype=pl.DT_UINT8)

    with pl.section_vector():
        if DimCap == 512:
            proj_addr = 0x00000
            norm_addr = 0x01000
            pub_addr = 0x02000
            gamma_addr = 0x03000
            red_addr = 0x04000
            cos_addr = 0x05000
            sin_addr = 0x06000
            packed_addr = 0x07000
            scale_addr = 0x07100
        elif DimCap == 1024:
            proj_addr = 0x00000
            norm_addr = 0x00800
            pub_addr = 0x01000
            gamma_addr = 0x01800
            red_addr = 0x02800
            cos_addr = 0x03000
            sin_addr = 0x04000
            packed_addr = 0x05000
            scale_addr = 0x05200
        else:
            proj_addr = 0x00000
            norm_addr = 0x01000
            pub_addr = 0x02000
            gamma_addr = 0x03000
            red_addr = 0x05000
            cos_addr = 0x05080
            sin_addr = 0x07080
            packed_addr = 0x09080
            scale_addr = 0x09480
        proj_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, DimCap], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
            addrs=[proj_addr], mutex_ids=[0])
        norm_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, DimCap], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
            addrs=[norm_addr], mutex_ids=[1])
        pub_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, DimCap], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
            addrs=[pub_addr], mutex_ids=[2])
        gamma_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, DimCap], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=[gamma_addr], mutex_ids=[3])
        red_g = pl.make_tile_group(
            type=pl.TileType(shape=[8, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=[red_addr], mutex_ids=[4])
        cos_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, DimCap], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=[cos_addr], mutex_ids=[5])
        sin_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, DimCap], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=[sin_addr], mutex_ids=[6])
        packed_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, DimCap // 2], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
            addrs=[packed_addr], mutex_ids=[7])
        scale_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, ((DimCap // FP4_GROUP_SIZE) + 31) // 32 * 32],
                             dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
            addrs=[scale_addr], mutex_ids=[8])
        gamma_tile = gamma_g.current()
        pl.set_validshape(gamma_tile, [1, dim])
        pl.load(gamma_tile, gamma, [0, 0])
        block_id = pl.get_block_idx()
        num_blocks = pl.get_block_num()
        next_cr = pl.getval(cache_index, 0)
        for row in pl.range(0, t):
            cr = next_cr
            if row + 1 < t:
                next_cr = pl.getval(cache_index, row + 1)
            if cr != -1:
                owner_key = 0
                dest0 = 0
                dest1 = 0
                data_col = 0
                scale_col = 0
                if storage_mode == 0:
                    owner_key = cr
                    dest0 = cr // cache_slots
                    dest1 = cr % cache_slots
                else:
                    block_id_c = cr // raw_block_size
                    slot = cr % raw_block_size
                    owner_key = block_id_c * cache_slots + slot // group_size
                    dest0 = block_id_c
                    dest1 = slot // group_size
                    data_col = (slot % group_size) * value_bytes
                    scale_col = group_size * value_bytes + (slot % group_size) * scale_bytes
                if owner_key % num_blocks == block_id:
                    proj_tile = proj_g.next()
                    norm_tile = norm_g.next()
                    pub_tile = pub_g.next()
                    red_tile = red_g.current()
                    cos_tile = cos_g.current()
                    sin_tile = sin_g.current()
                    packed_tile = packed_g.current()
                    scale_tile = scale_g.current()
                    pl.set_validshape(proj_tile, [1, dim])
                    pl.load(proj_tile, projected, [row, 0])
                    pl.set_validshape(cos_tile, [1, rd])
                    pl.load(cos_tile, rope_cos, [row, 0])
                    pl.set_validshape(sin_tile, [1, rd])
                    pl.load(sin_tile, rope_sin, [row, 0])
                    pl.set_validshape(packed_tile, [1, value_bytes])
                    pl.set_validshape(scale_tile, [1, scale_bytes])
                    if DimCap == 512:
                        _postprocess_row_512[512](
                            proj_tile, gamma_tile, norm_tile, pub_tile, red_tile,
                            cos_tile, sin_tile, packed_tile, scale_tile, epsilon, dim, tail_start)
                    elif DimCap == 1024:
                        _postprocess_row_1024[1024](
                            proj_tile, gamma_tile, norm_tile, pub_tile, red_tile,
                            cos_tile, sin_tile, packed_tile, scale_tile, epsilon, dim, tail_start)
                    else:
                        _postprocess_row_2048[2048](
                            proj_tile, gamma_tile, norm_tile, pub_tile, red_tile,
                            cos_tile, sin_tile, packed_tile, scale_tile, epsilon, dim, tail_start)
                    dest_row = dest0 * cache_slots + dest1
                    if storage_mode == 0:
                        pl.store(cache2, packed_tile, [dest_row, 0])
                        if write_actual_scale:
                            pl.store(scale2, scale_tile, [dest_row, 0])
                    else:
                        pl.store(cache2, packed_tile, [dest_row, data_col])
                        pl.store(cache2, scale_tile, [dest_row, scale_col])
