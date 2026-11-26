# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SIMT n-gram hashing with a common JIT and offline launch ABI."""
from dataclasses import dataclass

import torch
import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField

TILE = 128
SMALL_GRAMS = 16
SMALL_TABLES = 32
SMALL_COLS = (SMALL_GRAMS - 1) * SMALL_TABLES
# For two-layer, eight-table decode, keys 5 and 6 specialize a masked
# single token. Keys 3 and 4 retain their original row-tile meaning.
SINGLE_TOKEN_KEY_OFFSET = 2
PREFIX_GRAMS = 64
PREFIX_TABLES = 8
PREFIX_COLS = (PREFIX_GRAMS - 1) * PREFIX_TABLES


class EngramHashKey:
    HasMask = TilingKeyField(bits=1, values=[0, 1])
    Geometry = TilingKeyField(bits=3, values=[0, 3, 4, 5, 6])


@dataclass
class EngramHashTiling:
    tokens: int
    layers: int
    ngram: int
    tables: int


def _bucket(h, divisor, offset):
    d = pl.simt.cast(divisor, pl.DT_INT64)
    r = h % d
    if r < 0:
        r = r + d
    return pl.simt.cast(r, pl.DT_INT32) + offset


def _hash_row(ids, mult, vocab, offs, out, tid, ngram):
    h = pl.simt.cast(ids[tid, 0], pl.DT_INT64) * mult[0]
    for i in pl.range(1, ngram):
        h = h ^ (pl.simt.cast(ids[tid, i], pl.DT_INT64) * mult[i])
        for table in pl.range(0, 8):
            col = (i - 1) * 8 + table
            out[tid, col] = _bucket(h, vocab[col], offs[col])


@pl.vector_function(mode="simt", max_threads=TILE)
def hash_tile(ids, mult, vocab, offs, mask, out, valid, ngram, has_mask):
    tid = pl.simt.linear_thread_idx()
    if tid < valid:
        masked = False
        if has_mask:
            masked = mask[tid] != 0
        if masked:
            for col in pl.range(0, (ngram - 1) * 8):
                out[tid, col] = -1
        else:
            _hash_row(ids, mult, vocab, offs, out, tid, ngram)


@pl.vector_function(mode="simt", max_threads=TILE)
def hash_token_tables(ids, mult, vocab, offs, mask, out, first, count, grams, tables):
    lane = pl.simt.linear_thread_idx()
    if lane < count:
        flat = first + lane
        row = flat // tables
        table = flat % tables
        width = (grams - 1) * tables
        masked = False
        if HasMask:
            masked = mask[row] != 0
        if masked:
            for i in pl.range(1, grams):
                out[row * width + (i - 1) * tables + table] = -1
        else:
            h = pl.simt.cast(ids[row * grams], pl.DT_INT64) * mult[0]
            for i in pl.range(1, grams):
                h = h ^ (pl.simt.cast(ids[row * grams + i], pl.DT_INT64) * mult[i])
                col = (i - 1) * tables + table
                out[row * width + col] = _bucket(h, vocab[col], offs[col])


@pl.vector_function(mode="simt", max_threads=TILE)
def emit_tables(vocab, offsets, output, count, h, masked):
    table = pl.simt.linear_thread_idx()
    if table < count:
        if masked:
            output[table] = -1
        else:
            d = pl.simt.cast(vocab[table], pl.DT_INT64)
            r = h % d
            if r < 0:
                r = r + d
            output[table] = pl.simt.cast(r, pl.DT_INT32) + offsets[table]


def _write_prefixes(x, m, buffers, shape, row, layer, h, masked):
    vocab, offs, out = buffers
    t, g, n, cols = shape
    for i in pl.range(1, g):
        h = h ^ (pl.astype(x[row, i], pl.DT_INT64) * m[layer, i])
        for start in pl.range(0, n, TILE):
            col = (i - 1) * n + start
            v = pl.make_tensor(pl.addptr(vocab, layer * cols + col), [TILE])
            o = pl.make_tensor(pl.addptr(offs, layer * cols + col), [TILE])
            y = pl.make_tensor(pl.addptr(out, (layer * t + row) * cols + col), [TILE])
            emit_tables[TILE](v, o, y, pl.min(TILE, n - start), h, masked)


def _hash_general(ids, mult, buffers, mask, shape):
    t, l, g, n, cols = shape
    x = pl.make_tensor(ids, [t, g])
    m = pl.make_tensor(mult, [l, g])
    image = pl.make_tensor(mask, [t])
    for row in pl.range(pl.get_block_idx(), t, pl.get_block_num()):
        masked = 0
        if HasMask:
            masked = image[row]
        for layer in pl.range(0, l):
            h = pl.astype(x[row, 0], pl.DT_INT64) * m[layer, 0]
            _write_prefixes(x, m, buffers, (t, g, n, cols), row, layer, h, masked)


def _hash_compact(ids, mult, vocab, offs, mask, out, shape):
    t, l, g, n, cols = shape
    # One SIMT lane owns one (token, table) for all n-gram prefixes.
    # Static views bound a tile; count and the runtime strides limit every access
    # to the actual input/output extent, including the final partial tile.
    cells = t * n
    tiles = (cells + TILE - 1) // TILE
    for job in pl.range(pl.get_block_idx(), tiles * l, pl.get_block_num()):
        layer = job // tiles
        start = (job % tiles) * TILE
        row = start // n
        x = pl.make_tensor(pl.addptr(ids, row * g), [TILE * SMALL_GRAMS])
        m = pl.make_tensor(pl.addptr(mult, layer * g), [SMALL_GRAMS])
        v = pl.make_tensor(pl.addptr(vocab, layer * cols), [SMALL_COLS])
        o = pl.make_tensor(pl.addptr(offs, layer * cols), [SMALL_COLS])
        image = pl.make_tensor(pl.addptr(mask, row * HasMask), [TILE])
        y = pl.make_tensor(pl.addptr(out, (layer * t + row) * cols), [TILE * SMALL_COLS])
        hash_token_tables[TILE](x, m, v, o, image, y, start % n, pl.min(TILE, cells - start), g, n)


def _hash_tiles(ids, mult, vocab, offs, mask, out, t, l, cols):
    # SIMT accepts statically shaped views. Limit the last tile to live lanes.
    for tile in pl.range(pl.get_block_idx(), (t + TILE - 1) // TILE, pl.get_block_num()):
        row = tile * TILE
        valid = pl.min(TILE, t - row)
        x = pl.make_tensor(pl.addptr(ids, row * Geometry), [TILE, Geometry])
        image = pl.make_tensor(pl.addptr(mask, row * HasMask), [TILE])
        for layer in pl.range(0, l):
            m = pl.make_tensor(pl.addptr(mult, layer * Geometry), [Geometry])
            v = pl.make_tensor(pl.addptr(vocab, layer * cols), [(Geometry - 1) * 8])
            o = pl.make_tensor(pl.addptr(offs, layer * cols), [(Geometry - 1) * 8])
            y = pl.make_tensor(pl.addptr(out, (layer * t + row) * cols), [TILE, (Geometry - 1) * 8])
            hash_tile[TILE](x, m, v, o, image, y, valid, Geometry, HasMask)


def _prefix_hash(ids, mult, start, prefix):
    h = pl.simt.cast(ids[0], pl.DT_INT64) * mult[start]
    for i in pl.range(1, prefix + 1):
        h = h ^ (pl.simt.cast(ids[i], pl.DT_INT64) * mult[start + i])
    return h


@pl.vector_function(mode="simt", max_threads=TILE)
def _single_token(ids, mult, vocab, offs, mask, out):
    lane = pl.simt.linear_thread_idx()
    g = Geometry - SINGLE_TOKEN_KEY_OFFSET
    width = (g - 1) * 8
    if lane < 2 * width:
        if mask[0] != 0:
            out[lane] = -1
        else:
            layer = lane // width
            prefix = (lane % width) // 8 + 1
            h = _prefix_hash(ids, mult, layer * g, prefix)
            out[lane] = _bucket(h, vocab[lane], offs[lane])


@pl.vector_function(mode="simt", max_threads=TILE)
def _hash_prefix_group(ids, mult, vocab, offs, mask, out, first, width, tables):
    col = first + pl.simt.linear_thread_idx()
    if col < width:
        masked = False
        if HasMask:
            masked = mask[0] != 0
        if masked:
            out[col] = -1
        else:
            prefix = col // tables + 1
            h = _prefix_hash(ids, mult, 0, prefix)
            out[col] = _bucket(h, vocab[col], offs[col])


def _hash_short_rows(ids, mult, vocab, offs, mask, out, shape):
    # With few tables, distribute independent prefix outputs across lanes.
    tokens, layers, grams, tables, width = shape
    for job in pl.range(pl.get_block_idx(), tokens * layers, pl.get_block_num()):
        layer = job // tokens
        row = job % tokens
        x = pl.make_tensor(pl.addptr(ids, row * grams), [PREFIX_GRAMS])
        m = pl.make_tensor(pl.addptr(mult, layer * grams), [PREFIX_GRAMS])
        v = pl.make_tensor(pl.addptr(vocab, layer * width), [PREFIX_COLS])
        o = pl.make_tensor(pl.addptr(offs, layer * width), [PREFIX_COLS])
        image = pl.make_tensor(pl.addptr(mask, row * HasMask), [1])
        y = pl.make_tensor(pl.addptr(out, job * width), [PREFIX_COLS])
        for first in pl.range(0, width, TILE):
            _hash_prefix_group[TILE](x, m, v, o, image, y, first, width, tables)


@pl.jit(arch="3510", auto_mutex=True, tiling_key=EngramHashKey)
def engram_hash_kernel(ids: pl.Ptr[pl.DT_INT32], mult: pl.Ptr[pl.DT_INT64],
                       vocab: pl.Ptr[pl.DT_INT32], offs: pl.Ptr[pl.DT_INT32],
                       mask: pl.Ptr[pl.DT_UINT8], out: pl.Ptr[pl.DT_INT32],
                       workspace: pl.Ptr[pl.DT_UINT8], tiling: EngramHashTiling):
    with pl.section_vector():
        if Geometry == 5 or Geometry == 6:
            # Static views cover both supported geometries. Only the live
            # 2 * (g - 1) * 8 output lanes access memory.
            x = pl.make_tensor(ids, [4])
            m = pl.make_tensor(mult, [8])
            v = pl.make_tensor(vocab, [48])
            o = pl.make_tensor(offs, [48])
            image = pl.make_tensor(mask, [1])
            y = pl.make_tensor(out, [48])
            _single_token[TILE](x, m, v, o, image, y)
        else:
            t = tiling.tokens
            l = tiling.layers
            g = tiling.ngram
            n = tiling.tables
            cols = (g - 1) * n
            if Geometry != 0:
                _hash_tiles(ids, mult, vocab, offs, mask, out, t, l, cols)
            elif g <= SMALL_GRAMS and n <= SMALL_TABLES:
                _hash_compact(ids, mult, vocab, offs, mask, out, (t, l, g, n, cols))
            elif g > SMALL_GRAMS and g <= PREFIX_GRAMS and n <= PREFIX_TABLES and t <= TILE:
                _hash_short_rows(ids, mult, vocab, offs, mask, out, (t, l, g, n, cols))
            else:
                # Keep the full input contract for other n-gram and table dimensions.
                _hash_general(ids, mult, (vocab, offs, out), mask, (t, l, g, n, cols))


def make_offsets(vocab_sizes):
    """由 vocab_sizes 求每层的 exclusive prefix-sum（与参考实现一致，int32）。"""
    rows = []
    for layer in range(vocab_sizes.shape[0]):
        flat = vocab_sizes[layer].reshape(-1)
        rows.append(torch.cat([torch.zeros(1, dtype=torch.int32, device=flat.device),
                               flat[:-1].cumsum(0, dtype=torch.int32)]))
    return torch.stack(rows, dim=0)


def _check_nd(name, tensor):
    import torch_npu

    if tensor.device.type != "npu" or torch_npu.get_npu_format(tensor) != 2:
        raise ValueError(f"engram_hash: {name} must use ND storage on NPU")


def _input_dimensions(ngram_token_ids, multipliers, vocab_sizes, offsets):
    for name, tensor, rank in (("ngram_token_ids", ngram_token_ids, 2),
                               ("multipliers", multipliers, 2),
                               ("vocab_sizes", vocab_sizes, 3),
                               ("offsets", offsets, 2)):
        if tensor.dim() != rank:
            raise ValueError(f"engram_hash: {name} must be {rank}-D, got shape {tuple(tensor.shape)}")
    devices = {t.device for t in (ngram_token_ids, multipliers, vocab_sizes, offsets)}
    if len(devices) != 1:
        raise ValueError(f"engram_hash: all inputs must share one device, got {sorted(map(str, devices))}")

    num_tokens, max_ngram = ngram_token_ids.shape
    num_layers = multipliers.shape[0]
    tables = vocab_sizes.shape[2]
    num_cols = (max_ngram - 1) * tables

    expected = (
        ("ngram_token_ids", ngram_token_ids, torch.int32, (num_tokens, max_ngram)),
        ("multipliers", multipliers, torch.int64, (num_layers, max_ngram)),
        ("vocab_sizes", vocab_sizes, torch.int32, (num_layers, max_ngram - 1, tables)),
        ("offsets", offsets, torch.int32, (num_layers, num_cols)),
    )
    for name, tensor, want_dtype, want_shape in expected:
        if tensor.dtype != want_dtype:
            raise ValueError(f"engram_hash: {name} must be {want_dtype}, got {tensor.dtype}")
        if tuple(tensor.shape) != want_shape:
            raise ValueError(f"engram_hash: {name} must be {want_shape}, got {tuple(tensor.shape)}")
        _check_nd(name, tensor)
    if num_layers <= 0 or max_ngram < 2 or tables <= 0:
        raise ValueError("engram_hash: require num_layers > 0, max_ngram >= 2, tables > 0")

    return num_tokens, max_ngram, num_layers, tables


def engram_hash(
    ngram_token_ids: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
    image_token_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Engram hash indices。

    Args:
        ngram_token_ids: ``[num_tokens, max_ngram]`` int32。
        multipliers: ``[num_layers, max_ngram]`` int64。
        vocab_sizes: ``[num_layers, max_ngram - 1, tables]`` int32。
        offsets: ``[num_layers, (max_ngram - 1) * tables]`` int32，
            由 :func:`make_offsets` 生成。
        image_token_mask: 可选，``[num_tokens]`` bool/uint8，True 表示 image token，
            其整行输出为 -1。None 等价于全 False。

    Returns:
        ``[num_layers, num_tokens, (max_ngram - 1) * tables]`` int32。

    所有张量必须在同一 NPU 设备上。dtype 不符会 fail closed（不做隐式转换，
    因为 ids/vocab/offsets 的整型宽度决定计算结果）。
    """
    # 先查 rank，再按 rank 取维度；混设备的张量必须在 launch 前拦掉。
    num_tokens, max_ngram, num_layers, tables = _input_dimensions(ngram_token_ids, multipliers, vocab_sizes, offsets)
    num_cols = (max_ngram - 1) * tables

    if image_token_mask is None:
        mask = None
    else:
        if image_token_mask.shape != (num_tokens,):
            raise ValueError(
                f"engram_hash: image_token_mask must be [{num_tokens}], got {tuple(image_token_mask.shape)}")
        if image_token_mask.device != ngram_token_ids.device:
            raise ValueError(
                f"engram_hash: image_token_mask must be on {ngram_token_ids.device}, "
                f"got {image_token_mask.device}")
        if image_token_mask.dtype not in (torch.bool, torch.uint8):
            raise ValueError("engram_hash: image_token_mask must be bool or uint8")
        _check_nd("image_token_mask", image_token_mask)
        mask = image_token_mask.to(torch.uint8).reshape(num_tokens, 1)

    output = torch.empty((num_layers, num_tokens, num_cols), dtype=torch.int32,
                         device=ngram_token_ids.device)
    if num_tokens == 0:
        return output

    tiling = EngramHashTiling(num_tokens, num_layers, max_ngram, tables)
    has_mask = int(mask is not None)
    cores = torch.npu.get_device_properties(output.device.index).vector_core_num
    row_tiles = (num_tokens + TILE - 1) // TILE
    # Short inputs leave most vector cores idle with a row-only split.
    # Distribute their tables across cores even for the standard geometry.
    standard_tables = tables == 8 and max_ngram in (3, 4)
    use_row_tiles = standard_tables and row_tiles * 4 >= cores
    geometry = max_ngram if use_row_tiles else 0
    if has_mask and num_tokens == 1 and num_layers == 2 and standard_tables:
        geometry = max_ngram + SINGLE_TOKEN_KEY_OFFSET
        jobs = 1
    elif use_row_tiles:
        jobs = row_tiles
    elif max_ngram <= SMALL_GRAMS and tables <= SMALL_TABLES:
        jobs = num_layers * ((num_tokens * tables + TILE - 1) // TILE)
    else:
        jobs = num_tokens
    grid = min(jobs, cores)
    workspace = torch.empty(1, dtype=torch.uint8, device=output.device)
    engram_hash_kernel[None, grid, {"HasMask": has_mask, "Geometry": geometry}](
        ngram_token_ids.contiguous(), multipliers.contiguous(),
        vocab_sizes.contiguous(), offsets.contiguous(),
        mask.contiguous() if has_mask else workspace, output, workspace, tiling,
    )
    return output
