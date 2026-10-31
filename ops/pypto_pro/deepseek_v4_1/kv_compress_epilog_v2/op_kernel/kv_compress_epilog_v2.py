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

"""Block-scaled FP8/FP4 quantization with in-place cache updates.

Each vector worker loads a complete BF16 row and writes one contiguous
record containing quantized data, BF16 scales and any 2D padding. The output
byte tile and scale view share storage and synchronization. FP8 uses group
FP32 division. FP4 uses BF16 scaling followed by E2M1 conversion and packing;
group-16 rows with at least 16 groups process eight groups per register batch.
The input allocation includes a 256-byte guard for full-register loads;
global-memory transfers and active computation use the actual row length.
Layout-specific addressing and padding are selected through the tiling key.
"""
# ruff: noqa: F821
from typing import NamedTuple

import torch
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf
from pypto_pro.runtime.tilingkey import TilingKeyField


class KvCompressKey:
    Kind = TilingKeyField(bits=2, values=[0, 1, 2])
    Group = TilingKeyField(bits=6, values=[16, 32])
    RoundScale = TilingKeyField(bits=1, values=[0, 1])
    Slot64 = TilingKeyField(bits=1, values=[0, 1])
    Width = TilingKeyField(bits=14, values=[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])

    LargeScale = TilingKeyField(bits=1, values=[0, 1])
    Layout4 = TilingKeyField(bits=1, values=[0, 1])

    @classmethod
    def is_valid(cls, key):
        return ((key[0] == 2 or key[1] == 32) and
                (key[5] == 0 or (key[0] == 2 and key[1] == 16)))


@pl.vector_function
def _zero_output(out, byte_count):
    full_mask = vf.update_mask(256, dtype=pl.DT_UINT8)
    zero = vf.full(0, full_mask, dtype=pl.DT_UINT8)
    for col in pl.range(0, byte_count, 256):
        active = pl.min(256, byte_count - col)
        mask = vf.update_mask(active, dtype=pl.DT_UINT8)
        vf.store_align(out, zero, mask, [0, col])


@pl.vector_function
def _quantize_fp4_row(src, data, scale, dim, scale_offset):
    # Group starts are 32B aligned in the dense BF16 private input tile.
    # Compute each group scale, then stage its BF16-rounded product in
    # place. One VST->VLD edge precedes 128-element cast/pack beats. Every
    # PACK4_B32 destination begins at chunk/2 (a multiple of 64 bytes).
    for group in pl.range(dim // Group):
        pm16 = vf.update_mask(Group, dtype=pl.DT_BF16)
        pm1h = vf.update_mask(1, dtype=pl.DT_UINT16)
        x16 = vf.load_align(src, [0, group * Group])
        xbits = vf.bit_cast(x16, dtype=pl.DT_UINT16)
        expmask16 = vf.full(0x7F80, pm16, dtype=pl.DT_UINT16)
        exponents = vf.and_(xbits, expmask16, pm16)
        max_result = vf.reduce_max(exponents, pm16)
        max_exp = vf.full(max_result, pm16)
        is_finite = vf.ne(max_exp, 0x7F80, pm16)
        is_nonzero = vf.ne(max_exp, 0, pm16)
        clamped_exp = vf.maxs(max_exp, 0x0100, pm16)
        bias4 = vf.full(0x0100, pm16, dtype=pl.DT_UINT16)
        shared_exp = vf.sub(clamped_exp, bias4, pm16)
        nan16 = vf.full(0x7F81, pm16, dtype=pl.DT_UINT16)
        zero16 = vf.full(0, pm16, dtype=pl.DT_UINT16)
        finite_scale = vf.select(shared_exp, nan16, is_finite)
        scale_bits = vf.select(finite_scale, zero16, is_nonzero)
        scale16 = vf.bit_cast(scale_bits, dtype=pl.DT_BF16)
        vf.store_align(scale, scale16, pm1h, [0, scale_offset + group], dist=pl.StoreDist.FIRST_ELEMENT_B16)
        inv_bias = vf.full(0x7F00, pm16, dtype=pl.DT_UINT16)
        inverse_raw = vf.sub(inv_bias, shared_exp, pm16)
        inverse_finite = vf.select(inverse_raw, nan16, is_finite)
        inverse_nonzero = vf.select(inverse_finite, zero16, is_nonzero)
        is_special = vf.eq(shared_exp, 0x7F00, pm16)
        special16 = vf.full(0x0040, pm16, dtype=pl.DT_UINT16)
        inverse_bits = vf.select(special16, inverse_nonzero, is_special)
        inverse16 = vf.bit_cast(inverse_bits, dtype=pl.DT_BF16)
        scaled16 = vf.mul(x16, inverse16, pm16)
        vf.store_align(src, scaled16, pm16, [0, group * Group])
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    for chunk in pl.range(0, dim, 128):
        active = pl.min(128, dim - chunk)
        pmcast = vf.update_mask(active, dtype=pl.DT_BF16)
        scaled_chunk = vf.load_align(src, [0, chunk])
        q4 = vf.astype(scaled_chunk, pmcast, dtype=pl.DT_FP4E2M1,
                       layout=pl.CastLayout.ZERO,
                       round_mode=pl.VFRoundMode.CAST_RINT)
        q4bytes = vf.bit_cast(q4, dtype=pl.DT_INT8)
        vf.store_align(data, q4bytes, pmcast, [0, chunk // 2],
                       dist=pl.StoreDist.PACK4_B32)


@pl.vector_function
def _quantize_fp4_large(src, data, scale, dim, scale_offset):
    # One 128-BF16 chunk contains eight independent mathematical groups.
    # Data-block reduction packs the eight UINT16 exponent maxima in low lanes.
    # Only the valid group prefix is consumed; register-gather indices for all
    # 128 physical lanes are clamped to that defined prefix before gathering.
    # Scale stores are exact 2B FIRST_ELEMENT operations (no aligned-short
    # store at an unaligned scale start). Data starts at chunk/2, a 64B multiple.
    # No UB input writeback, no VST->VLD round trip, and no shared register state
    # crosses chunk iterations. Data and scale stores touch disjoint byte ranges.
    all16 = vf.update_mask(128, dtype=pl.DT_UINT16)
    pm1h = vf.update_mask(1, dtype=pl.DT_UINT16)
    exponent_mask = vf.full(0x7F80, all16, dtype=pl.DT_UINT16)
    bias4 = vf.full(0x0100, all16, dtype=pl.DT_UINT16)
    nan16 = vf.full(0x7F81, all16, dtype=pl.DT_UINT16)
    zero16 = vf.full(0, all16, dtype=pl.DT_UINT16)
    inv_bias = vf.full(0x7F00, all16, dtype=pl.DT_UINT16)
    special16 = vf.full(0x0040, all16, dtype=pl.DT_UINT16)
    lanes = vf.arange(0, dtype=pl.DT_UINT16)
    raw_indices = vf.shift_right(lanes, 4, all16)
    for chunk in pl.range(0, dim, 128):
        active = pl.min(128, dim - chunk)
        group_count = active // 16
        pmdata = vf.update_mask(active, dtype=pl.DT_BF16)
        pmgroup = vf.update_mask(group_count, dtype=pl.DT_UINT16)
        x16 = vf.load_align(src, [0, chunk])
        xbits = vf.bit_cast(x16, dtype=pl.DT_UINT16)
        exponents = vf.and_(xbits, exponent_mask, pmdata)
        max_exp = vf.reduce_max(exponents, pmdata, datablock=True)
        is_finite = vf.ne(max_exp, 0x7F80, pmgroup)
        is_nonzero = vf.ne(max_exp, 0, pmgroup)
        clamped_exp = vf.maxs(max_exp, 0x0100, pmgroup)
        shared_exp = vf.sub(clamped_exp, bias4, pmgroup)
        finite_scale = vf.select(shared_exp, nan16, is_finite)
        scale_bits = vf.select(finite_scale, zero16, is_nonzero)
        inverse_raw = vf.sub(inv_bias, shared_exp, pmgroup)
        inverse_finite = vf.select(inverse_raw, nan16, is_finite)
        inverse_nonzero = vf.select(inverse_finite, zero16, is_nonzero)
        is_special = vf.eq(shared_exp, 0x7F00, pmgroup)
        inverse_bits = vf.select(special16, inverse_nonzero, is_special)
        safe_indices = vf.mins(raw_indices, group_count - 1, all16)
        inverse_broadcast_bits = vf.gather(inverse_bits, safe_indices)
        inverse16 = vf.bit_cast(inverse_broadcast_bits, dtype=pl.DT_BF16)
        scaled16 = vf.mul(x16, inverse16, pmdata)
        q4 = vf.astype(scaled16, pmdata, dtype=pl.DT_FP4E2M1,
                       layout=pl.CastLayout.ZERO,
                       round_mode=pl.VFRoundMode.CAST_RINT)
        q4bytes = vf.bit_cast(q4, dtype=pl.DT_INT8)
        vf.store_align(data, q4bytes, pmdata, [0, chunk // 2],
                       dist=pl.StoreDist.PACK4_B32)
        for g in pl.range(group_count):
            scale_index = vf.full(g, all16, dtype=pl.DT_UINT16)
            picked_scale_bits = vf.gather(scale_bits, scale_index)
            picked_scale = vf.bit_cast(picked_scale_bits, dtype=pl.DT_BF16)
            vf.store_align(scale, picked_scale, pm1h,
                           [0, scale_offset + chunk // 16 + g],
                           dist=pl.StoreDist.FIRST_ELEMENT_B16)



@pl.vector_function
def _quantize_row(src, data, scale, dim, scale_offset):
    for group in pl.range(dim // 32):
        # UNPK places each BF16 in the low half of one b32 container; pm32
        # selects the same physical predicate positions for widening/compute.
        pm32 = vf.update_mask(32, dtype=pl.DT_FP32)
        pm1 = vf.update_mask(1, dtype=pl.DT_FP32)
        low = vf.load_align(src, [0, group * 32], dist=pl.LoadDist.UNPK)
        x32 = vf.astype(low, pm32, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
        ax = vf.abs(x32, pm32)
        amax = vf.reduce_max(ax, pm32)
        floor_max = vf.maxs(amax, 1.0e-4, pm1)
        if Kind == 0:
            unrounded = vf.muls(floor_max, 0.002232142857142857, pm1)
        else:
            unrounded = vf.muls(floor_max, 0.00001743861607142857, pm1)
        if RoundScale == 1:
            raw = vf.bit_cast(unrounded, dtype=pl.DT_UINT32)
            emask = vf.full(0x7F800000, pm1, dtype=pl.DT_UINT32)
            mmask = vf.full(0x007FFFFF, pm1, dtype=pl.DT_UINT32)
            exp = vf.and_(raw, emask, pm1)
            mantissa = vf.and_(raw, mmask, pm1)
            nz = vf.ne(mantissa, 0, pm1)
            increment = vf.full(0x00800000, pm1, dtype=pl.DT_UINT32)
            zero32 = vf.full(0, pm1, dtype=pl.DT_UINT32)
            rounded_increment = vf.select(increment, zero32, nz)
            exp_rounded = vf.add(exp, rounded_increment, pm1)
            scale32 = vf.bit_cast(exp_rounded, dtype=pl.DT_FP32)
        else:
            scale32 = vf.move(unrounded, pm1)
        scale_broadcast = vf.full(scale32, pm32)
        divided = vf.div(x32, scale_broadcast, pm32)
        if Kind == 0:
            bounded_low = vf.maxs(divided, -448.0, pm32)
            bounded = vf.mins(bounded_low, 448.0, pm32)
            quantized = vf.astype(bounded, pm32, dtype=pl.DT_FP8E4M3FN,
                                  layout=pl.CastLayout.ZERO,
                                  round_mode=pl.VFRoundMode.CAST_RINT,
                                  saturate=pl.SaturateMode.ON)
        else:
            bounded_low = vf.maxs(divided, -57344.0, pm32)
            bounded = vf.mins(bounded_low, 57344.0, pm32)
            quantized = vf.astype(bounded, pm32, dtype=pl.DT_FP8E5M2,
                                  layout=pl.CastLayout.ZERO,
                                  round_mode=pl.VFRoundMode.CAST_RINT,
                                  saturate=pl.SaturateMode.ON)
        qbytes = vf.bit_cast(quantized, dtype=pl.DT_UINT8)
        vf.store_align(data, qbytes, pm32, [0, group * 32], dist=pl.StoreDist.PACK4_B32)
        sb16 = vf.astype(scale32, pm1, dtype=pl.DT_BF16,
                         layout=pl.CastLayout.ZERO,
                         round_mode=pl.VFRoundMode.CAST_RINT,
                         saturate=pl.SaturateMode.OFF)
        vf.store_align(scale, sb16, pm1, [0, scale_offset + group], dist=pl.StoreDist.FIRST_ELEMENT_B16)


def _cache_record_base(slot, cache_cols, block_size, block_stride, bundle):
    if Layout4 == 1:
        base = (slot // block_size) * block_stride + (slot % block_size) * bundle
    else:
        base = slot * cache_cols
    return base


def _quantize_cache_record(inp, output, x, row, dim, output_capacity,
                           output_bytes, data_cols, write_cols, bundle):
    # The BF16 view inherits output's allocation and mutex ID.
    # Both parameters name exactly the same bytes; no independent
    # overlapping group or arena-to-arena copy is created.
    output_b16 = pl.reinterpret(output, dtype=pl.DT_BF16,
                                shape=[1, output_capacity // 2])
    pl.set_validshape(inp, [1, dim])
    pl.set_validshape(output, [1, output_bytes])
    pl.set_validshape(output_b16, [1, output_bytes // 2])
    pl.load(inp, x, [row, 0])
    if Layout4 == 0:
        if write_cols > bundle:
            _zero_output(output, write_cols)
    if Kind < 2:
        _quantize_row(inp, output, output_b16, dim, data_cols // 2)
    else:
        if LargeScale == 1:
            # At least two full 128-element batches amortize the
            # grouped-statistics setup. Scales use the full
            # inherited view and the actual BF16-word offset.
            _quantize_fp4_large(inp, output, output_b16, dim, data_cols // 2)
        else:
            _quantize_fp4_row(inp, output, output_b16, dim, data_cols // 2)


@pl.jit(auto_mutex=True, tiling_key=KvCompressKey, name="KvCompressEpilogV2")
def kv_compress_epilog_v2_kernel(
        cache_ptr: pl.Ptr[pl.DT_UINT8],
        x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
        slot_ptr: pl.Ptr[pl.DT_UINT8],
        tokens: pl.DT_INT64, dim: pl.DT_INT64,
        cache_rows: pl.DT_INT64, cache_cols: pl.DT_INT64,
        block_size: pl.DT_INT64, block_stride: pl.DT_INT64,
        storage_bytes: pl.DT_INT64):
    cache = pl.make_tensor(cache_ptr, [storage_bytes], dtype=pl.DT_UINT8)
    if Slot64 == 1:
        slots = pl.make_tensor(slot_ptr, [tokens], dtype=pl.DT_INT64)
    else:
        slots = pl.make_tensor(slot_ptr, [tokens], dtype=pl.DT_INT32)
    if Kind < 2:
        saved_overflow_mode = pl.get_ctrl_spr(60, 60)
        saved_float_mode = pl.get_ctrl_spr(48, 48)
        pl.set_ctrl_spr(48, 48, 1)
        pl.set_ctrl_spr(60, 60, 1)
    with pl.section_vector():
        if Kind < 2:
            output_capacity = ((Width + 2 * (Width // 32) + 31) // 32) * 32
            data_cols = dim
        else:
            output_capacity = ((Width // 2 + 2 * (Width // Group) + 31) // 32) * 32
            data_cols = dim // 2
        input_capacity = Width + 128
        ig = pl.make_tile_group(type=pl.TileType(shape=[1, input_capacity], dtype=pl.DT_BF16,
                    target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
                    addrs=[0], mutex_ids=[0])
        og = pl.make_tile_group(type=pl.TileType(shape=[1, output_capacity], dtype=pl.DT_UINT8,
                    target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
                    addrs=[2 * input_capacity], mutex_ids=[1])
        bundle = data_cols + 2 * (dim // Group)
        write_cols = ((bundle + 31) // 32) * 32
        if Layout4 == 1:
            output_bytes = bundle
        else:
            output_bytes = write_cols
        for row in pl.range(pl.get_block_idx(), tokens, pl.get_block_num()):
            slot = pl.getval(slots, row)
            if slot >= 0 and slot < cache_rows:
                base = _cache_record_base(slot, cache_cols, block_size, block_stride, bundle)
                inp = ig.next()
                output = og.next()
                _quantize_cache_record(inp, output, x, row, dim, output_capacity,
                                       output_bytes, data_cols, write_cols, bundle)
                pl.store(cache, output, [base])

    if Kind < 2:
        pl.system.bar_all()
        pl.set_ctrl_spr(48, 48, saved_float_mode)
        pl.set_ctrl_spr(60, 60, saved_overflow_mode)


class _CacheGeometry(NamedTuple):
    rows: int
    cols: int
    block_size: int
    block_stride: int
    layout4: int


class _LaunchMetadata(NamedTuple):
    kind: int
    tokens: int
    dim: int
    cache_rows: int
    cache_cols: int
    block_size: int
    block_stride: int
    storage_bytes: int
    layout4: int


def _validate_input(x, group):
    if x.dtype != torch.bfloat16 or x.ndim != 2 or not x.is_contiguous():
        raise ValueError('x must be contiguous BF16 [T,D]')
    t, d = x.shape
    if t <= 0 or d <= 0 or d > 8192:
        raise ValueError('invalid T, D or quant_group_size')
    if group not in (16, 32) or d % group:
        raise ValueError('invalid T, D or quant_group_size')
    return t, d


def _validate_attributes(group, mode, x_scale):
    if mode not in (2, 4) or (mode == 2 and group != 32):
        raise ValueError('unsupported quantization attributes')
    if x_scale != 1.0:
        raise ValueError('unsupported quantization attributes')


def _validate_slots(slot_mapping, tokens):
    if (slot_mapping.shape != (tokens,) or slot_mapping.dtype not in (torch.int32, torch.int64)
            or not slot_mapping.is_contiguous()):
        raise ValueError('slot_mapping must be contiguous INT32/INT64 [T]')


def _cache_kind(cache, mode):
    if mode == 2:
        if cache.dtype == torch.float8_e4m3fn:
            return 0
        if cache.dtype == torch.float8_e5m2:
            return 1
        raise ValueError('mode2 requires FP8 E4M3FN/E5M2 cache')
    if cache.dtype != torch.uint8:
        raise ValueError('mode4 requires UINT8 cache')
    return 2


def _cache_geometry(cache, bundle, block_stride):
    if cache.ndim == 2:
        n, c = cache.shape
        if n <= 0 or c < ((bundle + 31) // 32) * 32:
            raise ValueError('invalid 2D cache geometry')
        return _CacheGeometry(n, c, 1, c, 0)
    if cache.ndim == 4:
        b, bs, one, c = cache.shape
        if one != 1 or b <= 0 or bs <= 0:
            raise ValueError('invalid 4D cache geometry')
        if c < bundle:
            raise ValueError('invalid 4D cache geometry')
        stride = int(block_stride) if block_stride else bs * c
        if stride < bs * c or stride % 32:
            raise ValueError('invalid block_stride')
        return _CacheGeometry(b * bs, c, bs, stride, 1)
    raise ValueError('cache must be 2D or 4D')


def _validate_storage(cache, geometry, bundle):
    storage_bytes = cache.untyped_storage().nbytes() - cache.storage_offset() * cache.element_size()
    needed = (cache.shape[0] - 1) * geometry.block_stride + geometry.block_size * bundle
    if not geometry.layout4:
        needed = geometry.rows * geometry.cols
    if storage_bytes < needed:
        raise ValueError('cache backing storage is too small for physical writes')
    return storage_bytes


def _validate(cache, x, slot_mapping, group, mode, x_scale, block_stride):
    t, d = _validate_input(x, group)
    _validate_attributes(group, mode, x_scale)
    _validate_slots(slot_mapping, t)
    if not (cache.device == x.device == slot_mapping.device) or x.device.type != 'npu':
        raise ValueError('all tensors must share one NPU device')
    kind = _cache_kind(cache, mode)
    data = d if mode == 2 else d // 2
    bundle = data + 2 * (d // group)
    geometry = _cache_geometry(cache, bundle, block_stride)
    storage_bytes = _validate_storage(cache, geometry, bundle)
    return _LaunchMetadata(kind=kind, tokens=t, dim=d, cache_rows=geometry.rows,
                           cache_cols=geometry.cols, block_size=geometry.block_size,
                           block_stride=geometry.block_stride, storage_bytes=storage_bytes,
                           layout4=geometry.layout4)


def prepare_launch(cache, x, slot_mapping, quant_group_size=32, quant_mode=2,
                   round_scale=True, x_scale=1.0, block_stride=0, workers=None):
    """Validate/prepare once; returned zero-argument callback performs only launch."""
    meta = _validate(cache, x, slot_mapping, quant_group_size, quant_mode, x_scale, block_stride)
    key = {'Kind': meta.kind, 'Group': quant_group_size, 'RoundScale': int(bool(round_scale)),
           'Slot64': int(slot_mapping.dtype == torch.int64),
           'Width': max(32 if meta.kind < 2 else 128, 1 << (meta.dim - 1).bit_length()),
           'LargeScale': int(meta.kind == 2 and quant_group_size == 16 and meta.dim // quant_group_size >= 16),
           'Layout4': meta.layout4}
    worker_count = min(meta.tokens, 40) if workers is None else int(workers)
    if not 1 <= worker_count <= 80:
        raise ValueError('workers must be in 1..80')

    def launch():
        kv_compress_epilog_v2_kernel[None, worker_count, key](
            cache, x, slot_mapping, meta.tokens, meta.dim, meta.cache_rows, meta.cache_cols,
            meta.block_size, meta.block_stride, meta.storage_bytes)
    launch.key = key
    launch.workers = worker_count
    return launch


def kv_compress_epilog_v2_wrapper(cache, x, slot_mapping, quant_group_size=32, quant_mode=2,
                         round_scale=True, x_scale=1.0, block_stride=0):
    """Quantize BF16 rows and update the supplied cache in place."""
    prepare_launch(cache, x, slot_mapping, quant_group_size, quant_mode,
                   round_scale, x_scale, block_stride)()
    return cache
