# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CompressorV2 computation, mixed-core entry and public JIT wrapper."""
import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pypto
import pypto_pro.language as pl
import torch
from pypto_pro.language import Vf as vf
from pypto_pro.runtime.platform import get_platform_info
from pypto_pro.runtime.tilingkey import TilingKeyField


class CompressorKey:
    Stage = TilingKeyField(bits=2, values=[0, 1, 2])
    Rows = TilingKeyField(bits=3, values=[16, 32, 64, 96, 128, 256])
    Ratio = TilingKeyField(bits=3, values=[2, 4, 8, 16, 32, 64, 128])
    Fuse = TilingKeyField(bits=1, values=[0, 1])
    Cols = TilingKeyField(bits=1, values=[64, 128])
    Direct = TilingKeyField(bits=1, values=[0, 1])
    Parts = TilingKeyField(bits=2, values=[1, 2, 3, 4])
    Dim = TilingKeyField(bits=1, values=[128, 512])

    def is_valid(self, key_tuple):
        stage, rows, ratio, fuse, cols, direct, parts, dim = key_tuple
        if dim not in self.Dim.values:
            return False
        if stage == 0:
            return direct == 0 and ((parts == 1 and (rows != 256 or (cols == 128 and fuse == 1))) or
                (parts > 1 and cols == 64 and fuse == 1) or
                (parts == 2 and rows == 128 and fuse == 1 and cols == 128))
        return (stage == 2 and fuse == 1 and parts == 1 and rows >= 2 * ratio and
                (rows, cols) in ((256, 128), (128, 128), (64, 64), (32, 64)) and
                (rows != 256 or direct == 1))


class CompressorVectorKey:
    Stage = TilingKeyField(bits=2, values=[0, 1, 2])
    Rows = TilingKeyField(bits=3, values=[1, 16, 32, 64, 96, 128, 256])
    Ratio = TilingKeyField(bits=3, values=[2, 4, 8, 16, 32, 64, 128])
    Fuse = TilingKeyField(bits=1, values=[0, 1])
    Cols = TilingKeyField(bits=1, values=[64, 128])
    Direct = TilingKeyField(bits=1, values=[0, 1])
    Parts = TilingKeyField(bits=2, values=[1, 2, 3, 4])
    Dim = TilingKeyField(bits=1, values=[128, 512])
    Mode = TilingKeyField(bits=4, values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 20, 21, 37, 44, 52, 53])

    def is_valid(self, key_tuple):
        stage, rows, _ratio, fuse, cols, direct, _parts, dim, mode = key_tuple
        if dim not in self.Dim.values or mode not in self.Mode.values:
            return False
        return (stage == 1 and (rows == 16 or (rows == 1 and mode in (52, 53))) and
                fuse == 0 and cols == 64 and direct == 0)


@pl.vector_function
def _weighted_group(kv, score, out, ratio, half, groups, cols, scale, rescale):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pu = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    scale_register = vf.full(scale, p, dtype=pl.DT_FP32)
    rescale_register = vf.full(rescale, p, dtype=pl.DT_FP32)
    for group in pl.range(groups):
        base = group * ratio
        for col in pl.range(0, cols, 64):
            maximum = vf.load_align(score, [base, col])
            for row in pl.range(1, ratio):
                value = vf.load_align(score, [base + row, col])
                maximum = vf.max(maximum, value, p)
            numerator = vf.full(0.0, dtype=pl.DT_FP32)
            denominator = vf.full(0.0, dtype=pl.DT_FP32)
            for row in pl.range(ratio):
                value = vf.load_align(kv, [base + row, col])
                value = vf.mul(value, scale_register, p)
                logit = vf.load_align(score, [base + row, col])
                shifted = vf.sub(logit, maximum, p)
                weight = vf.exp(shifted, p)
                product = vf.mul(value, weight, p)
                numerator = vf.add(numerator, product, p)
                denominator = vf.add(denominator, weight, p)
            result = vf.div(numerator, denominator, p)
            result = vf.mul(result, rescale_register, p)
            if half:
                p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
                narrowed = vf.astype(result, p16, dtype=pl.DT_FP16, round_mode=pl.VFRoundMode.CAST_RINT)
            else:
                p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
                narrowed = vf.astype(result, p16, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            packed = vf.bit_cast(narrowed, dtype=pl.DT_UINT32)
            vf.store_align(out, packed, pu, [group, col], dist=pl.StoreDist.PACK)


@pl.vector_function
def _copy_projection_rows(kv, score, dst_kv, dst_score, begin, count, cols):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for row in pl.range(count):
        for col in pl.range(0, cols, 64):
            k = vf.load_align(kv, [begin + row, col])
            g = vf.load_align(score, [begin + row, col])
            vf.store_align(dst_kv, k, p, [row, col])
            vf.store_align(dst_score, g, p, [row, col])


@pl.vector_function
def _add_projection_2(destination, partial1, rows):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for row in pl.range(rows):
        current = vf.load_align(destination, [row, 0])
        value = vf.load_align(partial1, [row, 0])
        current = vf.add(current, value, p)
        vf.store_align(destination, current, p, [row, 0])


@pl.vector_function
def _add_projection_3(destination, partial1, partial2, rows):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for row in pl.range(rows):
        current = vf.load_align(destination, [row, 0])
        value1 = vf.load_align(partial1, [row, 0])
        value2 = vf.load_align(partial2, [row, 0])
        current = vf.add(current, value1, p)
        current = vf.add(current, value2, p)
        vf.store_align(destination, current, p, [row, 0])


@pl.vector_function
def _add_projection_4(destination, partial1, partial2, partial3, rows):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for row in pl.range(rows):
        current = vf.load_align(destination, [row, 0])
        value1 = vf.load_align(partial1, [row, 0])
        value2 = vf.load_align(partial2, [row, 0])
        value3 = vf.load_align(partial3, [row, 0])
        current = vf.add(current, value1, p)
        current = vf.add(current, value2, p)
        current = vf.add(current, value3, p)
        vf.store_align(destination, current, p, [row, 0])


@pl.vector_function
def _insert_rows(destination, source, offset, rows):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for row in pl.range(rows):
        value = vf.load_align(source, [row, 0])
        vf.store_align(destination, value, p, [offset + row, 0])


def _load_projection(destination, partial_group, projected, parts, kind, token, col, rows):
    pl.load(destination, projected, [0, kind, token, col])
    if parts > 1:
        partial1 = partial_group.next()
        pl.set_validshape(partial1, [rows, 64])
        pl.load(partial1, projected, [1, kind, token, col])
        if parts == 2:
            _add_projection_2(destination, partial1, rows)
        elif parts == 3:
            partial2 = partial_group.next()
            pl.set_validshape(partial2, [rows, 64])
            pl.load(partial2, projected, [2, kind, token, col])
            _add_projection_3(destination, partial1, partial2, rows)
        else:
            partial2 = partial_group.next()
            pl.set_validshape(partial2, [rows, 64])
            pl.load(partial2, projected, [2, kind, token, col])
            partial3 = partial_group.next()
            pl.set_validshape(partial3, [rows, 64])
            pl.load(partial3, projected, [3, kind, token, col])
            _add_projection_4(destination, partial1, partial2, partial3, rows)


@dataclass
class CompressorV2Tiling:
    tokens: int
    hidden: int
    dim: int
    batches: int
    sequence: int
    block_size: int
    cache_stride: int
    block_count: int
    out_rows: int
    packed_layout: int
    has_used: int
    has_start: int


class _Inputs(NamedTuple):
    x: torch.Tensor
    wkv: torch.Tensor
    wgate: torch.Tensor
    cache: torch.Tensor
    ratio: int
    table: torch.Tensor
    cu: torch.Tensor
    used: torch.Tensor
    starts: torch.Tensor


class _DeviceTensors(NamedTuple):
    x: torch.Tensor
    wkv: torch.Tensor
    wgate: torch.Tensor
    cache: torch.Tensor
    table: torch.Tensor
    cu: torch.Tensor
    used: torch.Tensor
    starts: torch.Tensor
    projected: torch.Tensor
    history: torch.Tensor
    out: torch.Tensor


def _vector_group(rows, cols, address, mutex_ids, depth):
    return pl.make_tile_group(
        type=pl.TileType(shape=[rows, cols], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=address,
        mutex_ids=mutex_ids,
        depth=depth,
    )


def _matrix_groups(left_type, right_type, accumulator_type):
    left = pl.make_tile_group(type=left_type, addrs=0, mutex_ids=[6, 7], depth=2)
    right = pl.make_tile_group(type=right_type, addrs=0, mutex_ids=[8, 9], depth=2)
    accumulator = pl.make_tile_group(type=accumulator_type, addrs=0, mutex_ids=[10, 11], depth=2)
    return pl.make_tuple(left=left, right=right, accumulator=accumulator)


def _matrix_operands(a, b, groups, shape, k):
    left = groups.left.next()
    right = groups.right.next()
    pl.set_validshape(left, [shape.valid, shape.step])
    pl.move(left, a, offset=[0, k])
    pl.move(right, b, offset=[k, 0])
    return pl.make_tuple(left=left, right=right)


def _matrix_multiply(accumulator, left, right, initialize, k):
    if initialize and k == 0:
        pl.matmul(accumulator, left, right)
    else:
        pl.matmul_acc(accumulator, accumulator, left, right)


def _matrix_panel(accumulator, a, b, groups, shape, initialize):
    pl.set_validshape(accumulator, [shape.valid, shape.cols])
    for k in pl.range(0, shape.panel, shape.step):
        operands = _matrix_operands(a, b, groups, shape, k)
        _matrix_multiply(accumulator, operands.left, operands.right, initialize, k)


def _matrix_pair(accumulator, accumulator_gate, a, b, g, groups, shape, initialize):
    pl.set_validshape(accumulator, [shape.valid, shape.cols])
    pl.set_validshape(accumulator_gate, [shape.valid, shape.cols])
    for k in pl.range(0, shape.panel, shape.step):
        operands = _matrix_operands(a, b, groups, shape, k)
        _matrix_multiply(accumulator, operands.left, operands.right, initialize, k)
        gate_right = groups.right.next()
        pl.move(gate_right, g, offset=[k, 0])
        _matrix_multiply(accumulator_gate, operands.left, gate_right, initialize, k)


def _projection_buffers(dtype):
    step = 128
    panel = 512
    if Rows == 256:
        step = 64
    if Parts > 1 and Rows <= 64 and Cols == 64:
        step = 256
    if Fuse or Rows == 256:
        panel = 256
    activation = pl.make_tile_group(
        type=pl.TileType(shape=[Rows, panel], dtype=dtype, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=0,
        mutex_ids=[0, 1],
        depth=2,
    )
    weight = pl.make_tile_group(
        type=pl.TileType(shape=[Cols, panel], dtype=dtype, target_memory=pl.MemorySpace.Mat),
        addrs=Rows * panel * 4,
        mutex_ids=[2, 3],
        depth=2,
    )
    transposed = pl.reinterpret(weight, shape=[panel, Cols], layout=pl.ZN)
    gate = weight
    gate_transposed = transposed
    if Fuse:
        gate = pl.make_tile_group(
            type=pl.TileType(shape=[Cols, panel], dtype=dtype, target_memory=pl.MemorySpace.Mat),
            addrs=(Rows + Cols) * panel * 4,
            mutex_ids=[4, 5],
            depth=2,
        )
        gate_transposed = pl.reinterpret(gate, shape=[panel, Cols], layout=pl.ZN)
    left_type = pl.TileType(shape=[Rows, step], dtype=dtype, target_memory=pl.MemorySpace.Left, compact=1)
    right_type = pl.TileType(shape=[step, Cols], dtype=dtype, target_memory=pl.MemorySpace.Right)
    accumulator_type = pl.TileType(shape=[Rows, Cols], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, compact=1)
    registers = _matrix_groups(left_type, right_type, accumulator_type)
    return pl.make_tuple(
        activation=activation,
        weight=weight,
        transposed=transposed,
        gate=gate,
        gate_transposed=gate_transposed,
        registers=registers,
        step=step,
        panel=panel,
    )


def _project_tile(work, data, shape, buffers):
    mtiles = (shape.tokens + Rows - 1) // Rows
    ntiles = shape.dim // Cols
    kind = 0
    part = 0
    tile_work = work
    if Parts > 1:
        part = work // (mtiles * ntiles)
        tile_work = work % (mtiles * ntiles)
    elif Fuse == 0:
        kind = work // (mtiles * ntiles)
        tile_work = work % (mtiles * ntiles)
    row = tile_work // ntiles * Rows
    col = tile_work % ntiles * Cols
    valid = pl.min(Rows, shape.tokens - row)
    accumulator = buffers.registers.accumulator.next()
    if Fuse:
        accumulator_gate = buffers.registers.accumulator.next()
    h_begin = 0
    h_end = shape.hidden
    if Parts == 2:
        h_begin = (shape.hidden // 2) * part
        h_end = (shape.hidden // 2) * (part + 1)
    elif Parts > 2:
        h_begin = (shape.hidden // buffers.panel * part // Parts) * buffers.panel
        h_end = (shape.hidden // buffers.panel * (part + 1) // Parts) * buffers.panel
    matrix_shape = pl.make_tuple(valid=valid, cols=Cols, panel=buffers.panel, step=buffers.step)
    for h in pl.range(h_begin, h_end, buffers.panel):
        a = buffers.activation.next()
        bmath = buffers.transposed.next()
        b = buffers.weight.current()
        pl.set_validshape(a, [valid, buffers.panel])
        pl.load(a, data.x, [row, h])
        if kind == 0:
            pl.load(b, data.wkv, [col, h])
        else:
            pl.load(b, data.wgate, [col, h])
        if Fuse:
            bmath_gate = buffers.gate_transposed.next()
            b_gate = buffers.gate.current()
            pl.load(b_gate, data.wgate, [col, h])
        if Fuse:
            _matrix_pair(
                accumulator, accumulator_gate, a, bmath, bmath_gate, buffers.registers, matrix_shape, h == h_begin
            )
        else:
            _matrix_panel(accumulator, a, bmath, buffers.registers, matrix_shape, h == h_begin)
    pl.store(data.projected, accumulator, [part, kind, row, col])
    if Fuse:
        pl.store(data.projected, accumulator_gate, [part, 1, row, col])


def _project_stage(data, shape, dtype):
    buffers = _projection_buffers(dtype)
    mtiles = (shape.tokens + Rows - 1) // Rows
    ntiles = shape.dim // Cols
    for work in pl.range(pl.get_block_idx(), (2 - Fuse) * Parts * mtiles * ntiles, pl.get_block_num()):
        _project_tile(work, data, shape, buffers)


def _snapshot_stage(data, shape):
    history_group = _vector_group(Ratio, 64, 0, [12], 1)
    cols = 2 * shape.dim // 64
    for work in pl.range(pl.get_block_idx(), shape.batches * cols, pl.get_block_num() * pl.get_subblock_num()):
        batch = work // cols
        col = work % cols * 64
        start = 0
        if shape.has_start:
            start = pl.getval(data.starts, batch)
        prefix = start % Ratio
        # No complete group can read history when even the total token
        # count plus this prefix is smaller than Ratio.
        if prefix > 0 and prefix + shape.tokens >= Ratio:
            block = pl.getval(data.table, batch)
            offset = 0
            while offset < prefix:
                slot = (start - prefix + offset) % shape.block_size
                count = pl.min(prefix - offset, shape.block_size - slot)
                tile = history_group.next()
                pl.set_validshape(tile, [count, 64])
                pl.load(tile, data.cache, [block, slot, col])
                pl.store(data.history, tile, [batch, offset, col])
                offset = offset + count


def _compression_buffers(dtype):
    # A full BSH group needs at most Ratio rows in each FP32 staging tile.
    full_group = (Mode & 8) != 0 and (Mode & 32) == 0 and Ratio <= 64
    if Rows == 1:
        rows = Ratio
        tile_bytes = rows * 64 * 4
        kv = _vector_group(rows, 64, 0, [0], 1)
        score = _vector_group(rows, 64, tile_bytes, [1], 1)
        temporary = _vector_group(rows, 64, 2 * tile_bytes, [2, 7], 2)
        partial_base = 4 * tile_bytes
        output_rows = 1
    elif full_group:
        rows = Ratio
        tile_bytes = rows * 64 * 4
        kv = _vector_group(rows, 64, 0, [0, 7], 2)
        score = _vector_group(rows, 64, 2 * tile_bytes, [1, 8], 2)
        temporary = _vector_group(rows, 64, 4 * tile_bytes, [2, 10], 2)
        partial_base = 6 * tile_bytes
        output_rows = 1
    else:
        rows = 128
        tile_bytes = 32768
        output_rows = 128 // Ratio
        if Parts <= 3 and (Mode & 16) == 0:
            kv = _vector_group(rows, 64, 0, [0, 7], 2)
            score = _vector_group(rows, 64, 65536, [1, 8], 2)
            temporary = _vector_group(rows, 64, 131072, [2], 1)
            partial_base = 163840
        else:
            kv = _vector_group(rows, 64, 0, [0], 1)
            score = _vector_group(rows, 64, 32768, [1], 1)
            temporary = _vector_group(rows, 64, 65536, [2, 7], 2)
            partial_base = 131072
    partial_slots = 1
    if Parts > 2:
        partial_slots = Parts - 1
    if Parts == 4:
        partial = _vector_group(rows, 64, partial_base, [3, 4, 5], 3)
    elif Parts == 3:
        partial = _vector_group(rows, 64, partial_base, [3, 4], 2)
    else:
        partial = _vector_group(rows, 64, partial_base, [3], 1)
    output = pl.make_tile_group(
        type=pl.TileType(shape=[output_rows, 64], dtype=dtype, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=partial_base + tile_bytes * partial_slots,
        mutex_ids=[6, 9],
        depth=2,
    )
    return pl.make_tuple(kv=kv, score=score, temporary=temporary, partial=partial, output=output)


def _compression_lanes(shape):
    ntiles = shape.dim // 64
    batch_begin = 0
    batch_step = 1
    lane_begin = pl.get_block_idx()
    lane_step = pl.get_block_num()
    if (shape.packed_layout == 0 and shape.sequence <= 128) or Mode == 21:
        batch_begin = pl.get_block_idx() // ntiles
        batch_step = pl.get_block_num() // ntiles
        lane_begin = pl.get_block_idx() % ntiles
        lane_step = ntiles
    return pl.make_tuple(batch_begin=batch_begin, batch_step=batch_step, begin=lane_begin, step=lane_step)


def _sequence_info(data, shape, batch):
    begin = batch * shape.sequence
    length = shape.sequence
    if Rows == 1:
        begin = 0
        length = 1
    elif shape.packed_layout:
        begin = pl.getval(data.cu, batch)
        end = pl.getval(data.cu, batch + 1)
        length = end - begin
    if shape.has_used:
        length = pl.getval(data.used, batch)
    start = 0
    if shape.has_start:
        start = pl.getval(data.starts, batch)
    prefix = start % Ratio
    groups = (prefix + length) // Ratio
    if length == 0:
        groups = 0
    cache_slot = 0
    cache_reuse = 0
    if Mode & 8:
        cache_slot = start % shape.block_size
        if prefix == 0 and cache_slot + Ratio <= shape.block_size:
            cache_reuse = 1
    return pl.make_tuple(batch=batch, begin=begin, length=length, start=start, prefix=prefix, groups=groups,
                         cache_slot=cache_slot, cache_reuse=cache_reuse)


def _read_group_tiles(tiles, data, shape, sequence, buffers):
    if tiles.group == 0 and sequence.prefix > 0:
        temp = buffers.temporary.next()
        pl.set_validshape(temp, [sequence.prefix, 64])
        pl.load(temp, data.history, [sequence.batch, 0, tiles.col])
        _insert_rows(tiles.kv, temp, 0, sequence.prefix)
        pl.load(temp, data.history, [sequence.batch, 0, shape.dim + tiles.col])
        _insert_rows(tiles.score, temp, 0, sequence.prefix)
        count = tiles.valid_groups * Ratio - sequence.prefix
        pl.set_validshape(temp, [count, 64])
        _load_projection(temp, buffers.partial, data.projected, Parts, 0, sequence.begin, tiles.col, count)
        _insert_rows(tiles.kv, temp, sequence.prefix, count)
        _load_projection(temp, buffers.partial, data.projected, Parts, 1, sequence.begin, tiles.col, count)
        _insert_rows(tiles.score, temp, sequence.prefix, count)
    else:
        token = sequence.begin + tiles.group * Ratio - sequence.prefix
        count = tiles.valid_groups * Ratio
        _load_projection(tiles.kv, buffers.partial, data.projected, Parts, 0, token, tiles.col, count)
        _load_projection(tiles.score, buffers.partial, data.projected, Parts, 1, token, tiles.col, count)


def _compress_sequence(data, shape, sequence, buffers, lanes, output_base):
    ntiles = shape.dim // 64
    groups_per_tile = 128 // Ratio
    if Ratio == 64:
        groups_per_tile = 1
    group_blocks = (sequence.groups + groups_per_tile - 1) // groups_per_tile
    for work in pl.range(lanes.begin, group_blocks * ntiles, lanes.step):
        group = work // ntiles * groups_per_tile
        col = work % ntiles * 64
        valid_groups = pl.min(groups_per_tile, sequence.groups - group)
        kv = buffers.kv.next()
        score = buffers.score.next()
        pl.set_validshape(kv, [valid_groups * Ratio, 64])
        pl.set_validshape(score, [valid_groups * Ratio, 64])
        tiles = pl.make_tuple(kv=kv, score=score, group=group, col=col, valid_groups=valid_groups)
        _read_group_tiles(tiles, data, shape, sequence, buffers)
        result = buffers.output.next()
        pl.set_validshape(result, [valid_groups, 64])
        _weighted_group(kv, score, result, Ratio, io_dtype == pl.DT_FP16, valid_groups, 64, 1.0 / Ratio, Ratio * 1.0)
        pl.store(data.out, result, [output_base + group, col])
        if sequence.cache_reuse != 0:
            block = pl.getval(data.table, sequence.batch)
            pl.store(data.cache, kv, [block, sequence.cache_slot, col])
            pl.store(data.cache, score, [block, sequence.cache_slot, shape.dim + col])


def _write_cache_rows(data, shape, sequence, buffers, work, block):
    ntiles = shape.dim // 64
    kind = work // ntiles
    col = work % ntiles * 64
    offset = pl.max(0, sequence.length - shape.block_size)
    if Rows == 1:
        slot = sequence.start
        if (shape.block_size & (shape.block_size - 1)) == 0:
            slot = slot & (shape.block_size - 1)
        else:
            slot = slot % shape.block_size
        temp = buffers.temporary.next()
        pl.set_validshape(temp, [1, 64])
        _load_projection(temp, buffers.partial, data.projected, Parts, kind, 0, col, 1)
        pl.store(data.cache, temp, [block, slot, kind * shape.dim + col])
    else:
        while offset < sequence.length:
            slot = sequence.start + offset
            if (shape.block_size & (shape.block_size - 1)) == 0:
                slot = slot & (shape.block_size - 1)
            else:
                slot = slot % shape.block_size
            count = pl.min(128, pl.min(sequence.length - offset, shape.block_size - slot))
            temp = buffers.temporary.next()
            pl.set_validshape(temp, [count, 64])
            _load_projection(temp, buffers.partial, data.projected, Parts, kind, sequence.begin + offset, col, count)
            pl.store(data.cache, temp, [block, slot, kind * shape.dim + col])
            offset = offset + count


def _write_sequence_cache(data, shape, sequence, buffers, lanes):
    if sequence.cache_reuse == 0:
        ntiles = shape.dim // 64
        groups_per_tile = 128 // Ratio
        if Ratio == 64:
            groups_per_tile = 1
        group_blocks = (sequence.groups + groups_per_tile - 1) // groups_per_tile
        block = pl.getval(data.table, sequence.batch)
        cache_lane = lanes.begin
        # Keep the compile-time layout specialization separate from the runtime lane check.
        if (shape.packed_layout or shape.sequence > 128) and Mode != 21:
            if (group_blocks + 2) * ntiles <= pl.get_block_num():
                cache_lane = pl.get_block_idx() - group_blocks * ntiles
        if cache_lane >= 0:
            for work in pl.range(cache_lane, 2 * ntiles, lanes.step):
                _write_cache_rows(data, shape, sequence, buffers, work, block)


def _compress_batch(data, shape, batch, buffers, lanes, output_base):
    sequence = _sequence_info(data, shape, batch)
    if shape.packed_layout == 0:
        output_base = batch * ((shape.sequence + Ratio - 1) // Ratio)
    _compress_sequence(data, shape, sequence, buffers, lanes, output_base)
    _write_sequence_cache(data, shape, sequence, buffers, lanes)
    return sequence.groups


def _packed_group_count(data, shape, begin_batch, end_batch):
    output_base = 0
    for previous in pl.range(begin_batch, end_batch):
        begin = pl.getval(data.cu, previous)
        end = pl.getval(data.cu, previous + 1)
        start = pl.getval(data.starts, previous)
        groups = (start % Ratio + end - begin) // Ratio
        if end == begin:
            groups = 0
        output_base = output_base + groups
    return output_base


def _compress_batches(data, shape, buffers, lanes):
    output_base = 0
    if Mode == 21:
        output_base = _packed_group_count(data, shape, 0, lanes.batch_begin)
    for batch in pl.range(lanes.batch_begin, shape.batches, lanes.batch_step):
        groups = _compress_batch(data, shape, batch, buffers, lanes, output_base)
        if shape.packed_layout:
            output_base = output_base + groups
        if Mode == 21:
            next_batch = batch + lanes.batch_step
            if next_batch < shape.batches:
                skipped = _packed_group_count(data, shape, batch + 1, next_batch)
                output_base = output_base + skipped


def _compression_stage(data, shape, dtype):
    buffers = _compression_buffers(dtype)
    lanes = _compression_lanes(shape)
    if Mode & 32:
        if lanes.batch_begin == 0:
            _compress_batch(data, shape, 0, buffers, lanes, 0)
    else:
        _compress_batches(data, shape, buffers, lanes)


def _fused_cube_buffers(dtype):
    step = 128
    if Rows == 256:
        step = 64
    activation = pl.make_tile_group(
        type=pl.TileType(shape=[Rows, 256], dtype=dtype, target_memory=pl.MemorySpace.Mat),
        addrs=0,
        mutex_ids=[0, 1],
        depth=2,
    )
    weight = pl.make_tile_group(
        type=pl.TileType(shape=[Cols, 256], dtype=dtype, target_memory=pl.MemorySpace.Mat),
        addrs=Rows * 1024,
        mutex_ids=[2, 3],
        depth=2,
    )
    gate = pl.make_tile_group(
        type=pl.TileType(shape=[Cols, 256], dtype=dtype, target_memory=pl.MemorySpace.Mat),
        addrs=(Rows + Cols) * 1024,
        mutex_ids=[4, 5],
        depth=2,
    )
    weight_transposed = pl.reinterpret(weight, shape=[256, Cols], layout=pl.ZN)
    gate_transposed = pl.reinterpret(gate, shape=[256, Cols], layout=pl.ZN)
    left_type = pl.TileType(shape=[Rows, step], dtype=dtype, target_memory=pl.MemorySpace.Left)
    right_type = pl.TileType(shape=[step, Cols], dtype=dtype, target_memory=pl.MemorySpace.Right)
    accumulator_type = pl.TileType(shape=[Rows, Cols], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc)
    registers = _matrix_groups(left_type, right_type, accumulator_type)
    return pl.make_tuple(
        activation=activation,
        weight=weight,
        gate=gate,
        weight_transposed=weight_transposed,
        gate_transposed=gate_transposed,
        registers=registers,
        step=step,
    )


def _fused_vector_buffers(dtype):
    kv = pl.make_tile_group(
        type=pl.TileType(shape=[Rows // 2, Cols], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0,
        mutex_ids=[12],
        depth=1,
    )
    score = pl.make_tile_group(
        type=pl.TileType(shape=[Rows // 2, Cols], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=Rows * Cols * 2,
        mutex_ids=[13],
        depth=1,
    )
    output = pl.make_tile_group(
        type=pl.TileType(shape=[Rows // (2 * Ratio), Cols], dtype=dtype, target_memory=pl.MemorySpace.Vec),
        addrs=Rows * Cols * 4,
        mutex_ids=[14],
        depth=1,
    )
    cache_kv = kv
    cache_score = score
    if Direct == 0:
        cache_kv = _vector_group(Rows // 2, Cols, Rows * Cols * 4 + Rows * Cols // Ratio, [15], 1)
        cache_score = _vector_group(Rows // 2, Cols, Rows * Cols * 6 + Rows * Cols // Ratio, [16], 1)
    return pl.make_tuple(kv=kv, score=score, output=output, cache_kv=cache_kv, cache_score=cache_score)


def _fused_project(work, data, shape, cube, shared, iteration):
    row = work // (shape.dim // Cols) * Rows
    col = work % (shape.dim // Cols) * Cols
    accumulator = cube.registers.accumulator.next()
    accumulator_gate = cube.registers.accumulator.next()
    matrix_shape = pl.make_tuple(valid=Rows, cols=Cols, panel=256, step=cube.step)
    for h in pl.range(0, shape.hidden, 256):
        a = cube.activation.next()
        bm = cube.weight_transposed.next()
        b = cube.weight.current()
        gm = cube.gate_transposed.next()
        g = cube.gate.current()
        pl.load(a, data.x, [row, h])
        pl.load(b, data.wkv, [col, h])
        pl.load(g, data.wgate, [col, h])
        _matrix_pair(accumulator, accumulator_gate, a, bm, gm, cube.registers, matrix_shape, h == 0)
    kv = shared.kv.next()
    score = shared.score.next()
    if iteration > 0:
        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=1)
    pl.move(kv, accumulator, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    pl.move(score, accumulator_gate, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)


def _fused_compress(work, data, shape, shared):
    kv = shared.kv.next()
    score = shared.score.next()
    result = shared.output.next()
    half_rows = Rows // 2
    row = work // (shape.dim // Cols) * Rows + pl.get_subblock_idx() * half_rows
    col = work % (shape.dim // Cols) * Cols
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=0)
    _weighted_group(
        kv, score, result, Ratio, io_dtype == pl.DT_FP16, half_rows // Ratio, Cols, 1.0 / Ratio, Ratio * 1.0
    )
    pl.store(data.out, result, [row // Ratio, col])
    offset = pl.max(0, shape.tokens - shape.block_size - row)
    block = pl.getval(data.table, 0)
    if Direct:
        if offset == 0:
            slot = row % shape.block_size
            pl.store(data.cache, kv, [block, slot, col])
            pl.store(data.cache, score, [block, slot, shape.dim + col])
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=1)
    else:
        while offset < half_rows:
            slot = (row + offset) % shape.block_size
            count = pl.min(half_rows - offset, shape.block_size - slot)
            cache_kv = shared.cache_kv.next()
            cache_score = shared.cache_score.next()
            _copy_projection_rows(kv, score, cache_kv, cache_score, offset, count, Cols)
            pl.set_validshape(cache_kv, [count, Cols])
            pl.set_validshape(cache_score, [count, Cols])
            pl.store(data.cache, cache_kv, [block, slot, col])
            pl.store(data.cache, cache_score, [block, slot, shape.dim + col])
            offset = offset + count
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=1)


def _fused_stage(data, shape, dtype):
    cube = _fused_cube_buffers(dtype)
    shared = _fused_vector_buffers(dtype)
    core = pl.get_block_idx() // pl.get_subblock_num()
    core_count = pl.get_block_num()
    iteration = 0
    for work in pl.range(core, shape.tokens // Rows * (shape.dim // Cols), core_count):
        with pl.section_cube():
            _fused_project(work, data, shape, cube, shared, iteration)
        with pl.section_vector():
            _fused_compress(work, data, shape, shared)
        iteration = iteration + 1
    with pl.section_cube():
        if iteration > 0:
            pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=1)


def _project_snapshot(x_ptr, kv_ptr, gate_ptr, cache_ptr, table_ptr, start_ptr, projected_ptr, history_ptr, shape):
    dtype = io_dtype
    x = pl.make_tensor(x_ptr, [shape.tokens, shape.hidden], dtype=dtype)
    wkv = pl.make_tensor(kv_ptr, [Dim, shape.hidden], dtype=dtype)
    wgate = pl.make_tensor(gate_ptr, [Dim, shape.hidden], dtype=dtype)
    cache = pl.make_tensor(
        cache_ptr,
        [shape.block_count, shape.block_size, 2 * Dim],
        [shape.cache_stride, 2 * Dim, 1],
        dtype=pl.DT_FP32,
    )
    table = pl.make_tensor(table_ptr, [shape.batches], dtype=pl.DT_INT32)
    starts = pl.make_tensor(start_ptr, [shape.batches], dtype=pl.DT_INT32)
    projected = pl.make_tensor(projected_ptr, [Parts, 2, shape.tokens, Dim], dtype=pl.DT_FP32)
    history = pl.make_tensor(history_ptr, [shape.batches, Ratio, 2 * Dim], dtype=pl.DT_FP32)
    data = pl.make_tuple(
        x=x, wkv=wkv, wgate=wgate, cache=cache, table=table, starts=starts, projected=projected, history=history
    )
    config = pl.make_tuple(
        tokens=shape.tokens,
        hidden=shape.hidden,
        dim=Dim,
        batches=shape.batches,
        block_size=shape.block_size,
        has_start=shape.has_start,
    )
    with pl.section_cube():
        _project_stage(data, config, dtype)
    with pl.section_vector():
        _snapshot_stage(data, config)


def compress_cache(cache_ptr, table_ptr, cu_ptr, used_ptr, start_ptr, projected_ptr, history_ptr, out_ptr, shape):
    dtype = io_dtype
    batches = 1 if Mode & 32 else shape.batches
    cache = pl.make_tensor(
        cache_ptr,
        [shape.block_count, shape.block_size, 2 * Dim],
        [shape.cache_stride, 2 * Dim, 1],
        dtype=pl.DT_FP32,
    )
    table = pl.make_tensor(table_ptr, [batches], dtype=pl.DT_INT32)
    cu = pl.make_tensor(cu_ptr, [batches + 1], dtype=pl.DT_INT32)
    used = pl.make_tensor(used_ptr, [batches], dtype=pl.DT_INT32)
    starts = pl.make_tensor(start_ptr, [batches], dtype=pl.DT_INT32)
    tokens = 1 if Rows == 1 else shape.tokens
    projected = pl.make_tensor(projected_ptr, [Parts, 2, tokens, Dim], dtype=pl.DT_FP32)
    history = pl.make_tensor(history_ptr, [batches, Ratio, 2 * Dim], dtype=pl.DT_FP32)
    out = pl.make_tensor(out_ptr, [shape.out_rows, Dim], dtype=dtype)
    data = pl.make_tuple(
        cache=cache, table=table, cu=cu, used=used, starts=starts, projected=projected, history=history, out=out
    )
    sequence = 1 if Rows == 1 else shape.sequence
    if Mode & 8:
        sequence = Ratio
    config = pl.make_tuple(
        dim=Dim,
        batches=batches,
        sequence=sequence,
        block_size=shape.block_size,
        packed_layout=Mode & 1,
        has_used=(Mode >> 1) & 1,
        has_start=(Mode >> 2) & 1,
    )
    with pl.section_vector():
        _compression_stage(data, config, dtype)


def _compress_onchip(x_ptr, kv_ptr, gate_ptr, cache_ptr, table_ptr, out_ptr, shape):
    dtype = io_dtype
    cache_dims = [shape.block_count, shape.block_size, 2 * Dim]
    cache_strides = [shape.cache_stride, 2 * Dim, 1]
    data = pl.make_tuple(
        x=pl.make_tensor(x_ptr, [shape.tokens, shape.hidden], dtype=dtype),
        wkv=pl.make_tensor(kv_ptr, [Dim, shape.hidden], dtype=dtype),
        wgate=pl.make_tensor(gate_ptr, [Dim, shape.hidden], dtype=dtype),
        cache=pl.make_tensor(cache_ptr, cache_dims, cache_strides, dtype=pl.DT_FP32),
        table=pl.make_tensor(table_ptr, [shape.batches], dtype=pl.DT_INT32),
        out=pl.make_tensor(out_ptr, [shape.out_rows, Dim], dtype=dtype),
    )
    config = pl.make_tuple(tokens=shape.tokens, hidden=shape.hidden, dim=Dim, block_size=shape.block_size)
    _fused_stage(data, config, dtype)


# CANN ABI: ten inputs, four outputs, workspace and tiling must remain separate parameters.
@pl.jit(arch="3510", auto_mutex=True, tiling_key=CompressorKey,
        datatype={"x": "io_dtype", "wkv": "io_dtype", "wgate": "io_dtype", "out": "io_dtype"})
def compressor_v2(
    x: pl.Ptr[pl.DT_UINT8], wkv: pl.Ptr[pl.DT_UINT8], wgate: pl.Ptr[pl.DT_UINT8],
    cache_in_ptr: pl.Ptr[pl.DT_FP32],
    table_ptr: pl.Ptr[pl.DT_INT32], cu_ptr: pl.Ptr[pl.DT_INT32],
    used_ptr: pl.Ptr[pl.DT_INT32], start_ptr: pl.Ptr[pl.DT_INT32],
    projected_in_ptr: pl.Ptr[pl.DT_FP32], history_in_ptr: pl.Ptr[pl.DT_FP32],
    out: pl.Ptr[pl.DT_UINT8], cache_out_ptr: pl.Ptr[pl.DT_FP32],
    projected_out_ptr: pl.Ptr[pl.DT_FP32], history_out_ptr: pl.Ptr[pl.DT_FP32],
    workspace: pl.Ptr[pl.DT_UINT8], tiling: CompressorV2Tiling,
):
    if Stage == 0:
        _project_snapshot(x, wkv, wgate, cache_in_ptr, table_ptr, start_ptr,
                          projected_out_ptr, history_out_ptr, tiling)
    else:
        _compress_onchip(x, wkv, wgate, cache_out_ptr, table_ptr, out, tiling)


def _validate_activation(x):
    if x.ndim not in (2, 3) or x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("x must be a rank-2/rank-3 FP16 or BF16 tensor")
    hidden = x.shape[-1]
    if hidden < 1024 or hidden > 10240 or hidden % 512:
        raise ValueError("H must be in [1024, 10240] and divisible by 512")
    return hidden


def _validate_weights(inputs, hidden):
    dim = inputs.wkv.shape[0]
    if dim not in (128, 512) or tuple(inputs.wkv.shape) != (dim, hidden) or inputs.wgate.shape != inputs.wkv.shape:
        raise ValueError("weights must have shape [D,H], D=128 or 512")
    if inputs.wkv.dtype != inputs.x.dtype or inputs.wgate.dtype != inputs.x.dtype:
        raise ValueError("activation and weight dtypes must match")
    if inputs.ratio not in CompressorKey.Ratio.values:
        raise ValueError("cmp_ratio must be one of 2,4,8,16,32,64,128")
    return dim


def _validate_cache(cache, dim):
    if cache.ndim != 3 or cache.dtype != torch.float32 or cache.shape[2] != 2 * dim:
        raise ValueError("state_cache must be FP32 [block_count,block_size,2*D]")
    if cache.shape[0] < 1 or not 1 <= cache.shape[1] <= 1024:
        raise ValueError("state_cache needs positive block_count and block_size in [1,1024]")
    if cache.stride(2) != 1 or cache.stride(1) != 2 * dim or cache.stride(0) < cache.shape[1] * 2 * dim:
        raise ValueError("only the cache block stride may be noncontiguous")


def _batch_count(inputs):
    if inputs.x.ndim == 2:
        if inputs.cu is None or inputs.cu.ndim != 1 or inputs.cu.numel() < 1:
            raise ValueError("TH layout requires cu_seqlens[B+1]")
        return inputs.cu.numel() - 1
    if inputs.cu is not None:
        raise ValueError("BSH layout requires cu_seqlens=None")
    return inputs.x.shape[0]


def _validate_metadata(inputs, batches):
    metadata = (
        ("state_block_table", inputs.table, batches),
        ("cu_seqlens", inputs.cu, batches + 1),
        ("seqused", inputs.used, batches),
        ("start_pos", inputs.starts, batches),
    )
    for name, tensor, size in metadata:
        if tensor is not None and (tensor.dtype != torch.int32 or tuple(tensor.shape) != (size,)):
            raise ValueError(f"{name} must be int32 [{size}]")
    if inputs.table is None:
        raise ValueError("state_block_table is required by the upstream kernel")


def _validate_device(inputs):
    if inputs.x.device.type != "npu" or inputs.cache.device != inputs.x.device:
        raise ValueError("inputs must be contiguous on one NPU, except the cache block stride")
    tensors = (inputs.x, inputs.wkv, inputs.wgate, inputs.table, inputs.cu, inputs.used, inputs.starts)
    for tensor in tensors:
        if tensor is not None and (tensor.device != inputs.x.device or not tensor.is_contiguous()):
            raise ValueError("inputs must be contiguous on one NPU, except the cache block stride")


def _validate(inputs):
    hidden = _validate_activation(inputs.x)
    dim = _validate_weights(inputs, hidden)
    _validate_cache(inputs.cache, dim)
    batches = _batch_count(inputs)
    _validate_metadata(inputs, batches)
    _validate_device(inputs)
    return batches, hidden, dim


def _row_capacity(tokens, dim, cols, cores):
    parallel_rows = max(1, cores // (2 * (dim // cols)))
    ideal = (tokens + parallel_rows - 1) // parallel_rows
    return next((rows for rows in (16, 32, 64, 96, 128) if rows >= ideal), 128)


def _column_capacity(tokens):
    return 64 if tokens <= 128 else 128


def _single_fused_panel(route):
    return (route["Parts"], route["Rows"], route["Fuse"], route["Cols"]) == (1, 128, 1, 128)


def _refine_panel_key(route, tokens, dim, cores, hidden):
    if _single_fused_panel(route) and hidden % 512 == 0:
        grid = ((tokens + 127) // 128) * (dim // 128)
        current_panel_work = ((grid + cores - 1) // cores) * (hidden // 256)
        split_panel_work = ((2 * grid + cores - 1) // cores) * (hidden // 512)
        if split_panel_work * 8 <= current_panel_work * 7:
            route["Parts"] = 2
    if _single_fused_panel(route) and tokens % 256 == 0:
        narrow_grid = (tokens // 128) * (dim // 128)
        wide_grid = (tokens // 256) * (dim // 128)
        if 2 * ((wide_grid + cores - 1) // cores) <= (narrow_grid + cores - 1) // cores:
            route["Rows"] = 256
    return route


def _cube_key(tokens, dim, dtype, ratio, cores, hidden):
    cols = _column_capacity(tokens)
    rows = _row_capacity(tokens, dim, cols, cores)
    fuse = int(((tokens + rows - 1) // rows) * (dim // cols) >= cores)
    route = {
        "Stage": 0, "Half": int(dtype == torch.float16), "Rows": rows, "Ratio": ratio,
        "Fuse": fuse, "Cols": cols, "Direct": 0, "Parts": 1, "Dim": dim,
    }
    split_grid = max(1, ((tokens + 255) // 256) * (dim // 64))
    parts = min(4, cores // split_grid)
    if parts >= 2 and hidden >= 1024 * parts:
        current_grid = (2 - fuse) * ((tokens + rows - 1) // rows) * (dim // cols)
        current_cost = ((current_grid + cores - 1) // cores) * (min(tokens, rows) + (1 + fuse) * cols) * hidden
        split_cost = ((split_grid * parts + cores - 1) // cores) * (min(tokens, 256) + 128)
        split_cost *= (hidden + parts - 1) // parts
        if split_cost * 5 < current_cost * 4:
            route.update(Rows=256, Cols=64, Fuse=1, Parts=parts)
    if 1 <= tokens <= 128 and route["Parts"] > 1:
        route["Rows"] = next(rows for rows in (16, 32, 64, 96, 128) if rows >= tokens)
    if 128 < tokens <= 256 and route["Parts"] > 1:
        grid = ((tokens + 127) // 128) * (dim // route["Cols"]) * route["Parts"]
        if grid <= cores:
            route["Rows"] = 128
    return _refine_panel_key(route, tokens, dim, cores, hidden)


def _onchip_cost(choice, tokens, dim, cores):
    rows, cols = choice
    rounds = ((tokens // rows) * (dim // cols) + cores - 1) // cores
    return rounds * (rows + 2 * cols + rows * cols // 64)


def _onchip_geometry(tokens, dim, ratio, cores, block_size):
    if tokens < 32:
        return None
    choices = [(256, 128), (128, 128), (64, 64), (32, 64)]
    valid = []
    for rows, cols in choices:
        if rows < 2 * ratio or tokens % rows:
            continue
        if rows == 256 and block_size % (rows // 2):
            continue
        valid.append((rows, cols))
    if not valid:
        return None
    return min(valid, key=lambda choice: _onchip_cost(choice, tokens, dim, cores))


def _launch_keys(x, batches, tokens, dim, ratio, seqused, start_pos, cores, block_size):
    if batches == 1 and seqused is None and start_pos is None:
        geometry = _onchip_geometry(tokens, dim, ratio, cores, block_size)
        if geometry is not None:
            rows, cols = geometry
            return [
                {
                    "Stage": 2,
                    "Half": int(x.dtype == torch.float16),
                    "Rows": rows,
                    "Ratio": ratio,
                    "Fuse": 1,
                    "Cols": cols,
                    "Direct": int(block_size % (rows // 2) == 0),
                    "Parts": 1,
                    "Dim": dim,
                }
            ]
    cube = _cube_key(tokens, dim, x.dtype, ratio, cores, x.shape[-1])
    mode = int(len(x.shape) == 2) | (int(seqused is not None) << 1) | (int(start_pos is not None) << 2)
    if len(x.shape) == 3 and seqused is None and x.shape[1] == ratio:
        mode |= 8
    if mode in (4, 5) and tokens <= (64 if ratio == 64 else 128):
        mode |= 16
    if batches == 1 and mode in (5, 12, 20, 21):
        mode |= 32
    vector_rows = 1 if batches == 1 and tokens == 1 and mode in (52, 53) else 16
    return [cube, dict(cube, Stage=1, Rows=vector_rows, Fuse=0, Cols=64, Mode=mode)]


def _make_shape(inputs, batches, hidden, dim, out):
    return CompressorV2Tiling(
        tokens=inputs.x.numel() // hidden,
        hidden=hidden,
        dim=dim,
        batches=batches,
        sequence=inputs.x.shape[1] if inputs.x.ndim == 3 else 0,
        block_size=inputs.cache.shape[1],
        cache_stride=inputs.cache.stride(0),
        block_count=inputs.cache.shape[0],
        out_rows=out.numel() // dim,
        packed_layout=int(inputs.x.ndim == 2),
        has_used=int(inputs.used is not None),
        has_start=int(inputs.starts is not None),
    )


def _device_tensors(inputs, projected, history, out):
    return _DeviceTensors(
        inputs.x,
        inputs.wkv,
        inputs.wgate,
        inputs.cache,
        inputs.table,
        inputs.table if inputs.cu is None else inputs.cu,
        inputs.table if inputs.used is None else inputs.used,
        inputs.table if inputs.starts is None else inputs.starts,
        projected,
        history,
        out,
    )


def _vector_entry():
    """Load the sibling entry without colliding with the original JIT module."""
    path = Path(__file__).with_name("compressor_v2_vector.py")
    name = "_compressor_v2_vector_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].compressor_v2_vector


def _launch_stage(route, workers, shape, tensors):
    x, wkv, wgate, cache, table, cu, used, starts, projected, history, out = tensors
    kernel = _vector_entry() if route["Stage"] == 1 else compressor_v2
    dtype = pl.DT_FP16 if x.dtype == torch.float16 else pl.DT_BF16
    datatype = {"x": dtype, "wkv": dtype, "wgate": dtype, "out": dtype}
    key = {name: value for name, value in route.items() if name != "Half"}
    kernel[None, workers, key, datatype](
        x, wkv, wgate, cache, table, cu, used, starts, projected, history,
        out, cache, projected, history, None, shape,
    )


def _launch_workers(routes, shape, cube_cores, vector_cores):
    """Remove idle blocks only when each remaining owner retains the same work."""
    ncols = shape.dim // 64
    vector_workers = vector_cores
    short_sequence = shape.packed_layout == 0 and shape.sequence <= 128
    batch_mode = routes[-1]["Stage"] == 1 and routes[-1].get("Mode", 0) == 21
    if short_sequence or batch_mode:
        vector_workers = min(vector_workers // ncols * ncols, shape.batches * ncols)
    counts = []
    for route in routes:
        workers = cube_cores
        if route["Stage"] == 0:
            projection = ((shape.tokens + route["Rows"] - 1) // route["Rows"]) * (shape.dim // route["Cols"])
            projection *= (2 - route["Fuse"]) * route["Parts"]
            snapshot = shape.batches * (2 * shape.dim // 64)
            workers = min(cube_cores, max(projection, (snapshot + 1) // 2))
        elif route["Stage"] == 1:
            workers = vector_workers
        counts.append(workers)
    return counts


def compressor(
    x, wkv, wgate, state_cache, cmp_ratio=4, *, state_block_table=None, cu_seqlens=None, seqused=None, start_pos=None
):
    """Return compressed KV and update state_cache in place; unused output rows are unspecified."""
    inputs = _Inputs(x, wkv, wgate, state_cache, cmp_ratio, state_block_table, cu_seqlens, seqused, start_pos)
    batches, hidden, dim = _validate(inputs)
    tokens = x.numel() // hidden
    sequence = x.shape[1] if x.ndim == 3 else 0
    out_shape = (
        (batches, (sequence + cmp_ratio - 1) // cmp_ratio, dim)
        if x.ndim == 3
        else (min(tokens, tokens // cmp_ratio + batches), dim)
    )
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    if tokens == 0:
        return out
    props = get_platform_info()
    routes = _launch_keys(
        x, batches, tokens, dim, cmp_ratio, seqused, start_pos, int(props.cube_core_num), state_cache.shape[1]
    )
    projected, history = state_cache, state_cache
    if len(routes) == 2:
        projected = torch.empty((routes[0]["Parts"], 2, tokens, dim), device=x.device, dtype=torch.float32)
        history = torch.empty((batches, cmp_ratio, 2 * dim), device=x.device, dtype=torch.float32)
    shape = _make_shape(inputs, batches, hidden, dim, out)
    tensors = _device_tensors(inputs, projected, history, out)
    counts = _launch_workers(routes, shape, int(props.cube_core_num), int(props.vector_core_num))
    for route, workers in zip(routes, counts):
        _launch_stage(route, workers, shape, tensors)
    return out
