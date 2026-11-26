# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F821
"""Offline ABI of the verified QLI pipeline: inputs, outputs, workspace, tiling.

Sequence reads and task ownership remain in the separate AICPU entry. Only
compile-time storage capacities are bucketed; valid shapes, masks and stores
retain the original runtime bounds. All mathematical and synchronization
primitives are inherited from quant_lightning_indexer.
"""
import importlib.util
import sys
from dataclasses import astuple, dataclass
from functools import lru_cache
from math import gcd
from pathlib import Path
from typing import NamedTuple, Optional

import torch
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf
from pypto_pro.runtime.tilingkey import TilingKeyField
F4, E8, BF, F32 = pl.DT_FP4E2M1, pl.DT_FP8E8M0, pl.DT_BF16, pl.DT_FP32
I32, U16, U32 = pl.DT_INT32, pl.DT_UINT16, pl.DT_UINT32


class QliKey:
    Operation = TilingKeyField(bits=1, values=[0, 1])
    TileN = TilingKeyField(bits=2, values=[16, 32, 64, 128])
    SelectWidth = TilingKeyField(bits=4, values=[0, 256, 512, 1024, 1536, 2048, 4096, 8192, 16384])
    ShardCapacity = TilingKeyField(bits=2, values=[4096, 8192, 16384, 9472])
    ReturnValues = TilingKeyField(bits=1, values=[0, 1])
    FixedTopK = TilingKeyField(bits=1, values=[0, 512])
    MaskMode = TilingKeyField(bits=1, values=[0, 3])
    RatioMode = TilingKeyField(bits=1, values=[0, 2])

    @classmethod
    def is_valid(cls, key):
        op, n, select, shard, _, fixed, mask, ratio = key
        if mask == 0 and ratio != 0:
            return False
        if op:
            return n == 128 and select == 0 and fixed == 0
        return shard == 4096 and (select != 0 or fixed == 0)


@dataclass
class QliTiling:
    rows: int
    batches: int
    max_q: int
    max_k: int
    page_size: int
    k_rows: int
    s_rows: int
    k_page_stride: int
    ks_page_stride: int
    k_stride: int
    ks_stride: int
    table_cols: int
    cube: int
    vector: int
    plan_elems: int
    capacity: int
    flags: int
    cmp_ratio: int
    ws_scores: int
    ws_index_map: int
    ws_partial: int
    score_cols: int
    aux_cols: int
    topk: int


def _vector_row_bounds(plan, rows):
    """Read this AIV's postprocessing row range, assigned by AICPU."""
    record = 1 + 3 * rows + 2 * pl.get_block_idx()
    return pl.getval(plan, record), pl.getval(plan, record + 1)


def _row_visible(plan, row, cmp_ratio):
    """Resolve the causal boundary on AI Core from AICPU's sequence operands."""
    record = 1 + 3 * row
    remaining = pl.getval(plan, record)
    length = pl.getval(plan, record + 1)
    remainder = pl.getval(plan, record + 2)
    ratio = cmp_ratio
    if RatioMode == 2:
        ratio = 2
    visible = length
    if remaining <= 0:
        visible = 0
    elif MaskMode == 3:
        visible = pl.min(
            length, pl.max(0, (ratio * length + remainder - remaining + 1) // ratio)
        )
    return visible


# ============================================================================
# Selection
# ============================================================================

# Native (VF) form. BF16 radix selection; the VF threshold primitives are
# adapted from the PyPTO lightning_indexer ST sample.
# Fused short-row selection uses two BF16 radix histograms. Long rows use
# bounded-memory radix scans inside the same AI Core invocation.


@pl.vector_function
def _selection_keys_from_scores(scores, keys, size, valid_size):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U16)
    sign = vf.full(0x8000, p, dtype=U16)
    allbits = vf.full(0xFFFF, p, dtype=U16)
    zero = vf.full(0, p, dtype=U16)
    for c in pl.range(0, size, 128):
        value = vf.load_align(scores, [0, c])
        bits = vf.bit_cast(value, dtype=U16)
        negative = vf.ge(bits, sign, p)
        flip = vf.select(allbits, sign, negative)
        key = vf.xor(bits, flip, p)
        active = vf.update_mask(pl.max(0, pl.min(128, valid_size - c)), dtype=U16)
        safe_key = vf.select(key, zero, active)
        vf.store_align(keys, safe_key, p, [0, c])


@pl.vector_function
def _selection_emit_ordered(scores, out_idx, out_val, topk, count, offset, write_values):
    """All keys fit: preserve logical order and pad without selection."""
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U32)
    inf_bits = vf.full(0xFF80, p, dtype=U32)
    minus_one = vf.full(-1, p, dtype=I32)
    for c in pl.range(0, topk, 64):
        active = vf.update_mask(pl.min(64, topk - c), dtype=U32)
        present = vf.update_mask(pl.max(0, pl.min(64, count - c)), dtype=U32)
        raw = vf.load_align(
            scores,
            pl.min(c, (count - 1) // 64 * 64),
            dist=pl.LoadDist.UNPK_B16,
            dtype=U32,
        )
        valid = vf.ne(raw, inf_bits, present)
        index = vf.arange(offset + c, dtype=I32)
        oi = vf.select(index, minus_one, valid)
        ov = vf.select(raw, inf_bits, valid)
        vf.store_align(out_idx, oi, active, c)
        if write_values:
            vf.store_align(out_val, ov, active, c, dist=pl.StoreDist.PACK_B32)


@pl.vector_function
def _selection_emit_selection(
    indices16,
    scores,
    out_idx,
    out_val,
    topk,
    count,
    begin,
    mapping,
    mapped,
    offset,
    write_values,
):
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U16)
    p32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U32)
    for c in pl.range(0, topk, 64):
        active = vf.update_mask(pl.min(64, topk - c), dtype=U32)
        valid = vf.update_mask(pl.max(0, pl.min(64, count - c)), dtype=U32)
        raw = vf.load_align(
            indices16,
            [0, pl.min(c, pl.max(0, count - 1) // 64 * 64)],
            dist=pl.LoadDist.UNPK_B16,
            dtype=U32,
        )
        zero = vf.full(0, p32, dtype=U32)
        idx = vf.select(raw, zero, valid)
        value = vf.gather(scores, idx, active)
        # Gather with b32 indices yields bf16 payload in low halfwords.
        index = vf.adds(idx, begin, active)
        negone = vf.full(-1, p32, dtype=I32)
        si = vf.bit_cast(index, dtype=I32)
        if mapped:
            si = vf.gather(mapping, idx, active)
        shifted = vf.adds(si, offset, active)
        bits = vf.bit_cast(value, dtype=U32)
        invalid_bits = vf.full(0xFF80, p32, dtype=U32)
        finite = vf.ne(bits, invalid_bits, valid)
        oi = vf.select(shifted, negone, finite)
        vf.store_align(out_idx, oi, active, [0, c])
        ov = vf.select(bits, invalid_bits, finite)
        if write_values:
            vf.store_align(out_val, ov, active, [0, c], dist=pl.StoreDist.PACK_B32)


@pl.vector_function
def _selection_histograms_high_vf(
    hist_high_tile, score_u16_tile, hist_loop: pl.DT_INT64, base
):
    """Histogram high byte (BYTE_1, MSB); chistv2 accumulates into dst reg."""
    preg_b8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    preg_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    preg_b32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)

    cout0 = vf.full(0, preg_b16, dtype=pl.DT_UINT16)
    cout1 = vf.full(0, preg_b16, dtype=pl.DT_UINT16)

    for i in pl.range(0, hist_loop):
        vreg_low, vreg_high = vf.load_align(
            score_u16_tile,
            base + i * 256,
            dist=pl.LoadDist.DINTLV_B8,
            dtype=pl.DT_UINT8,
        )
        cout0 = vf.histograms(
            vreg_high,
            preg_b8,
            bin_type=pl.BinType.BIN0,
            hist_type=pl.HistType.ACCUMULATE,
        )
        cout1 = vf.histograms(
            vreg_high,
            preg_b8,
            bin_type=pl.BinType.BIN1,
            hist_type=pl.HistType.ACCUMULATE,
        )

    c0_even = vf.astype(cout0, preg_b16, layout=pl.CastLayout.ZERO, dtype=pl.DT_UINT32)
    c0_odd = vf.astype(cout0, preg_b16, layout=pl.CastLayout.ONE, dtype=pl.DT_UINT32)
    c1_even = vf.astype(cout1, preg_b16, layout=pl.CastLayout.ZERO, dtype=pl.DT_UINT32)
    c1_odd = vf.astype(cout1, preg_b16, layout=pl.CastLayout.ONE, dtype=pl.DT_UINT32)
    vf.store_align(
        hist_high_tile, c0_even, c0_odd, preg_b32, dist=pl.StoreDist.INTLV_B32
    )
    vf.store_align(
        hist_high_tile + 128, c1_even, c1_odd, preg_b32, dist=pl.StoreDist.INTLV_B32
    )
    vf.mem_bar()


@pl.vector_function
def _selection_find_high_bin_and_histograms_low(
    hist_low_tile, idx_high_tile, hist_high_tile, score_u16_tile,
    sqz_idx_buf, nk_value_tile, bottom_k, hist_loop, base,
):
    p8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U16)
    p32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U32)
    target = vf.full(bottom_k, p32, dtype=U32)
    absent = vf.full(256, p32, dtype=U32)
    best = vf.full(256, p32, dtype=U32)
    for i in pl.range(4):
        idx = vf.arange(i * 64, dtype=U32)
        counts = vf.load_align(hist_high_tile, i * 64)
        match = vf.ge(counts, target, p32)
        chosen = vf.select(idx, absent, match)
        best = vf.min(best, chosen, p32)
    minimum = vf.reduce_min(best, p32)
    high = vf.full(minimum, p32)
    prev = vf.maxs(high, 1, p32)
    one = vf.full(1, p32, dtype=U32)
    prev = vf.sub(prev, one, p32)
    counts = vf.gather(hist_high_tile, prev, p32)
    zero = vf.full(0, p32, dtype=U32)
    has_prev = vf.gt(high, zero, p32)
    preceding = vf.select(counts, zero, has_prev)
    next_k = vf.sub(target, preceding, p32)
    repeated = vf.muls(high, 0x01010101, p32)
    vf.store_align(idx_high_tile, repeated, p32, 0)
    vf.store_align(nk_value_tile, next_k, p32, 0)
    high8 = vf.bit_cast(repeated, dtype=pl.DT_UINT8)
    count0 = vf.full(0, p16, dtype=U16)
    count1 = vf.full(0, p16, dtype=U16)
    for i in pl.range(hist_loop):
        low, hi = vf.load_align(score_u16_tile, base + i * 256,
                                dist=pl.LoadDist.DINTLV_B8, dtype=pl.DT_UINT8)
        match = vf.eq(hi, high8, p8)
        count0 = vf.histograms(low, match, bin_type=pl.BinType.BIN0,
                               hist_type=pl.HistType.ACCUMULATE)
        count1 = vf.histograms(low, match, bin_type=pl.BinType.BIN1,
                               hist_type=pl.HistType.ACCUMULATE)
    e0 = vf.astype(count0, p16, layout=pl.CastLayout.ZERO, dtype=U32)
    o0 = vf.astype(count0, p16, layout=pl.CastLayout.ONE, dtype=U32)
    e1 = vf.astype(count1, p16, layout=pl.CastLayout.ZERO, dtype=U32)
    o1 = vf.astype(count1, p16, layout=pl.CastLayout.ONE, dtype=U32)
    vf.store_align(hist_low_tile, e0, o0, p32, dist=pl.StoreDist.INTLV_B32)
    vf.store_align(hist_low_tile + 128, e1, o1, p32, dist=pl.StoreDist.INTLV_B32)


@pl.vector_function
def _selection_find_kth_vf(
    kth_value_tile, hist_low_tile, idx_high_tile, sqz_idx_buf, nk_value_tile
):
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U16)
    p32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U32)
    target = vf.load_align(nk_value_tile, 0)
    absent = vf.full(256, p32, dtype=U32)
    best = vf.full(256, p32, dtype=U32)
    for i in pl.range(4):
        idx = vf.arange(i * 64, dtype=U32)
        counts = vf.load_align(hist_low_tile, i * 64)
        match = vf.ge(counts, target, p32)
        chosen = vf.select(idx, absent, match)
        best = vf.min(best, chosen, p32)
    minimum = vf.reduce_min(best, p32)
    low = vf.full(minimum, p32)
    high = vf.load_align(idx_high_tile, 0)
    mask = vf.full(255, p32, dtype=U32)
    high = vf.and_(high, mask, p32)
    high = vf.shift_left(high, 8, p32)
    key = vf.or_(high, low, p32)
    upper = vf.shift_left(key, 16, p32)
    both = vf.or_(key, upper, p32)
    key16 = vf.bit_cast(both, dtype=U16)
    vf.store_align(kth_value_tile, key16, p16, 0)


@pl.vector_function
def _selection_collect_indices_gt_eq_vf(
    output_idx_tile,
    score_u16_tile,
    kth_value_tile,
    vf_loop: pl.DT_INT64,
    base,
):
    """Collect indices where value > kth (GT), then value == kth (EQ).

    GT indices come first, EQ indices follow. The downstream pl.store
    with set_validshape(topk) truncates to exactly topk entries.
    Output indices are DT_INT32 (matching reference implementation).
    """
    preg_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)

    vf.clear_spr()
    align_idx = vf.unalign_reg_for_store()

    kth_value = vf.load_align(
        kth_value_tile, 0, dist=pl.LoadDist.BRC_B16, dtype=pl.DT_UINT16
    )

    # Pass 1: collect GT indices
    for i in pl.range(0, vf_loop):
        idx_c = vf.arange(i * 128, dtype=pl.DT_INT16)
        vreg_input = vf.load_align(score_u16_tile, base + i * 128)
        pout_gt = vf.gt(vreg_input, kth_value, preg_b16)
        sqz_idx_out = vf.squeeze(idx_c, pout_gt, gather_mode=pl.SqueezeMode.STORE_REG)
        vf.squeeze_store_unalign(output_idx_tile, sqz_idx_out, align_idx)

    # Pass 2: collect EQ indices (appends after GT)
    for i in pl.range(0, vf_loop):
        idx_c = vf.arange(i * 128, dtype=pl.DT_INT16)
        vreg_input = vf.load_align(score_u16_tile, base + i * 128)
        pout_eq = vf.eq(vreg_input, kth_value, preg_b16)
        sqz_idx_out = vf.squeeze(idx_c, pout_eq, gather_mode=pl.SqueezeMode.STORE_REG)
        vf.squeeze_store_unalign(output_idx_tile, sqz_idx_out, align_idx)

    vf.squeeze_store_unalign_post(output_idx_tile, align_idx)


# ============================================================================
# Cache Select
# ============================================================================

# Native (VF) form. Select a BF16 sortable-key cache without GM scores.


@pl.vector_function
def _cache_select_clear_cache(cache, hist, width, written):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U16)
    z = vf.full(0, p, dtype=U16)
    for r in pl.range(3):
        for i in pl.range(written, width, 128):
            vf.store_align(cache, z, p, r * width + i)
    initial = vf.full(width - written, p, dtype=U16)
    for i in pl.range(0, 3 * 256, 128):
        vf.store_align(hist, initial, p, i)


@pl.vector_function
def _cache_select_widen_histogram(src, dst, row):
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U16)
    p32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U32)
    lo = vf.load_align(src, row * 256)
    hi = vf.load_align(src, row * 256 + 128)
    e0 = vf.astype(lo, p16, layout=pl.CastLayout.ZERO, dtype=U32)
    o0 = vf.astype(lo, p16, layout=pl.CastLayout.ONE, dtype=U32)
    e1 = vf.astype(hi, p16, layout=pl.CastLayout.ZERO, dtype=U32)
    o1 = vf.astype(hi, p16, layout=pl.CastLayout.ONE, dtype=U32)
    vf.store_align(dst, e0, o0, p32, dist=pl.StoreDist.INTLV_B32)
    vf.store_align(dst + 128, e1, o1, p32, dist=pl.StoreDist.INTLV_B32)


@pl.vector_function
def _cache_select_emit_cached(
    indices, cache, out_idx, out_val, topk, base, begin,
    mapped, mapping, permuted, offset, write_values,
):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U32)
    zero = vf.full(0, p, dtype=U32)
    sign = vf.full(0x8000, p, dtype=U32)
    allbits = vf.full(0xFFFF, p, dtype=U32)
    neginf = vf.full(0xFF80, p, dtype=U32)
    negone = vf.full(-1, p, dtype=I32)
    for c in pl.range(0, topk, 64):
        active = vf.update_mask(pl.min(64, topk - c), dtype=U32)
        idx = vf.load_align(indices, c, dist=pl.LoadDist.UNPK_B16, dtype=U32)
        gather_indices = vf.adds(idx, base, p)
        raw = vf.gather(cache, gather_indices, active)
        key = vf.bit_cast(raw, dtype=U32)
        positive = vf.ge(key, sign, p)
        flip = vf.select(sign, allbits, positive)
        bits = vf.xor(key, flip, p)
        valid = vf.gt(key, zero, active)
        value = vf.select(bits, neginf, valid)
        original = vf.adds(idx, 0, p)
        if permuted:
            lowmask = vf.full(15, p, dtype=U32)
            middlemask = vf.full(112, p, dtype=U32)
            planemask = vf.full(128, p, dtype=U32)
            tilemask = vf.full(0xFFFFFF00, p, dtype=U32)
            low = vf.and_(idx, lowmask, p)
            middle = vf.and_(idx, middlemask, p)
            middle = vf.shift_left(middle, 1, p)
            plane = vf.and_(idx, planemask, p)
            plane = vf.shift_right(plane, 3, p)
            tile = vf.and_(idx, tilemask, p)
            original = vf.or_(low, middle, p)
            original = vf.or_(original, plane, p)
            original = vf.or_(original, tile, p)
        global_idx = vf.adds(original, begin, p)
        signed_idx = vf.bit_cast(global_idx, dtype=I32)
        if mapped:
            signed_idx = vf.gather(mapping, idx, active)
            valid = vf.ne(signed_idx, negone, valid)
        shifted_idx = vf.adds(signed_idx, offset, p)
        oi = vf.select(shifted_idx, negone, valid)
        vf.store_align(out_idx, oi, active, c)
        if write_values:
            vf.store_align(out_val, value, active, c, dist=pl.StoreDist.PACK_B32)


# ============================================================================
# Candidates
# ============================================================================

# Native (VF) form. Eight-token maxima and the mandatory partial block.


CANDIDATES_NEG_INF = float("-inf")
CANDIDATES_POS_INF = float("inf")


@pl.vector_function
def _candidates_block_max(src, dst, active, start, partial):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
    ninf = vf.full(CANDIDATES_NEG_INF, p, dtype=F32)
    pinf = vf.full(CANDIDATES_POS_INF, p, dtype=F32)
    idx = vf.arange(0, dtype=U32)
    eight = vf.full(8, p, dtype=U32)
    shifted = vf.sub(idx, eight, p)
    first = vf.lt(idx, eight, p)
    out_mask = vf.update_mask(16, dtype=F32)
    for c in pl.range(0, active, 128):
        x0 = vf.load_align(src, c, dist=pl.LoadDist.UNPK_B16)
        x1 = vf.load_align(src, c + 64, dist=pl.LoadDist.UNPK_B16)
        f0 = vf.astype(x0, p, dtype=F32, layout=pl.CastLayout.ZERO)
        f1 = vf.astype(x1, p, dtype=F32, layout=pl.CastLayout.ZERO)
        p0 = vf.update_mask(pl.min(64, active - c), dtype=F32)
        p1 = vf.update_mask(pl.max(0, pl.min(64, active - c - 64)), dtype=F32)
        a0 = vf.select(f0, ninf, p0)
        a1 = vf.select(f1, ninf, p1)
        m0 = vf.reduce_max(a0, p, datablock=True)
        m1 = vf.reduce_max(a1, p, datablock=True)
        hi = vf.gather(m1, shifted)
        joined = vf.select(m0, hi, first)
        indices = vf.adds(idx, (start + c) // 8, p)
        si = vf.bit_cast(indices, dtype=I32)
        tail = vf.eq(si, partial, p)
        value = vf.select(pinf, joined, tail)
        bf = vf.astype(value, out_mask, dtype=BF, layout=pl.CastLayout.ZERO)
        bits = vf.bit_cast(bf, dtype=U32)
        vf.store_align(dst, bits, out_mask, c // 8, dist=pl.StoreDist.PACK_B32)


@pl.vector_function
def _candidates_write_count(dst, count):
    p = vf.update_mask(8, dtype=I32)
    value = vf.full(count, p, dtype=I32)
    vf.store_align(dst, value, p, 0)


# ============================================================================
# Streaming Selection
# ============================================================================

# Native (VF) form. Bounded-memory selection for large candidate capacities.
#
# One vector worker owns a row. A 16-bit radix search counts keys at or above
# each midpoint, then emits greater keys followed by ties. All DMA buffers
# are bounded by 8192 elements, independent of the requested output capacity.


@pl.vector_function
def _streaming_selection_count_keys(src, count_tile, active, threshold):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U16)
    zero = vf.full(0, p, dtype=U16)
    one = vf.full(1, p, dtype=U16)
    sign = vf.full(0x8000, p, dtype=U16)
    neg = vf.full(0xFFFF, p, dtype=U16)
    target = vf.full(threshold, p, dtype=U16)
    counts = vf.full(0, p, dtype=U16)
    for c in pl.range(0, active, 128):
        valid = vf.update_mask(pl.min(128, active - c), dtype=U16)
        raw = vf.load_align(src, c)
        bits = vf.bit_cast(raw, dtype=U16)
        negative = vf.ge(bits, sign, p)
        flip = vf.select(neg, sign, negative)
        key = vf.xor(bits, flip, p)
        selected = vf.ge(key, target, valid)
        increment = vf.select(one, zero, selected)
        counts = vf.add(counts, increment, p)
    total = vf.reduce_sum(counts, p)
    vf.store_align(count_tile, total, p, 0)


@pl.vector_function
def _streaming_selection_collect_keys(src, index_tile, count_tile, active, threshold, equal):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U16)
    zero = vf.full(0, p, dtype=U16)
    one = vf.full(1, p, dtype=U16)
    sign = vf.full(0x8000, p, dtype=U16)
    neg = vf.full(0xFFFF, p, dtype=U16)
    target = vf.full(threshold, p, dtype=U16)
    counts = vf.full(0, p, dtype=U16)
    vf.clear_spr()
    ar = vf.unalign_reg_for_store()
    for c in pl.range(0, active, 128):
        valid = vf.update_mask(pl.min(128, active - c), dtype=U16)
        raw = vf.load_align(src, c)
        bits = vf.bit_cast(raw, dtype=U16)
        negative = vf.ge(bits, sign, p)
        flip = vf.select(neg, sign, negative)
        key = vf.xor(bits, flip, p)
        if equal:
            selected = vf.eq(key, target, valid)
        else:
            selected = vf.gt(key, target, valid)
        increment = vf.select(one, zero, selected)
        counts = vf.add(counts, increment, p)
        index = vf.arange(c, dtype=U16)
        compact = vf.squeeze(index, selected, gather_mode=pl.SqueezeMode.STORE_REG)
        vf.squeeze_store_unalign(index_tile, compact, ar)
    vf.squeeze_store_unalign_post(index_tile, ar)
    total = vf.reduce_sum(counts, p)
    vf.store_align(count_tile, total, p, 0)


@pl.vector_function
def _streaming_selection_fill_padding(indices, values, size):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=I32)
    minus_one = vf.full(-1, p, dtype=I32)
    inf_bits = vf.full(0xFF80, p, dtype=pl.DT_UINT32)
    for c in pl.range(0, size, 64):
        active = vf.update_mask(pl.min(64, size - c), dtype=I32)
        vf.store_align(indices, minus_one, active, c)
        vf.store_align(values, inf_bits, active, c, dist=pl.StoreDist.PACK_B32)


STREAMING_SELECTION_CHUNK = 8192


def _streaming_selection_groups():
    """Reuse drained score-stage UB with distinct postprocessing mutex IDs."""
    sg = pl.make_tile_group(
        type=pl.TileType(shape=[1, STREAMING_SELECTION_CHUNK], dtype=BF,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[0], mutex_ids=[19],
    )
    xg = pl.make_tile_group(
        type=pl.TileType(shape=[1, STREAMING_SELECTION_CHUNK + 128], dtype=U16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[2 * STREAMING_SELECTION_CHUNK], mutex_ids=[20],
    )
    cg = pl.make_tile_group(
        type=pl.TileType(shape=[1, 128], dtype=U16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[4 * STREAMING_SELECTION_CHUNK + 256], mutex_ids=[21],
    )
    ig = pl.make_tile_group(
        type=pl.TileType(shape=[1, STREAMING_SELECTION_CHUNK], dtype=I32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[4 * STREAMING_SELECTION_CHUNK + 512], mutex_ids=[22],
    )
    vg = pl.make_tile_group(
        type=pl.TileType(shape=[1, STREAMING_SELECTION_CHUNK], dtype=BF,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[8 * STREAMING_SELECTION_CHUNK + 512], mutex_ids=[23],
    )
    groups = (sg, xg, cg, ig, vg)
    return groups


def _streaming_selection_split(scores, groups, span):
    """Bisect the key space for the value that keeps at most topk keys above it."""
    sg, cg = groups
    row, topk, width = span
    src = sg.current()
    ct = cg.current()
    lo = 127
    if topk < width:
        hi = 65535
        for _ in pl.range(16):
            midpoint = (lo + hi + 1) // 2
            total = pl.const(0, pl.DT_INT64)
            for start in pl.range(0, width, STREAMING_SELECTION_CHUNK):
                active = pl.min(STREAMING_SELECTION_CHUNK, width - start)
                pl.set_validshape(src, [1, active])
                pl.load(src, scores, [row, start])
                _streaming_selection_count_keys(src, ct, active, midpoint)
                total = total + pl.getval(ct, 0)
            if total >= topk:
                lo = midpoint
            else:
                hi = midpoint - 1
    return lo


def _streaming_selection_write(scores, indices, values, groups, span, write_values):
    """Emit the keys above the split value, then pad the row up to topk."""
    sg, xg, cg, ig, vg = groups
    row, offset, topk, width, lo = span
    src = sg.current()
    local = xg.current()
    ct = cg.current()
    oi = ig.current()
    ov = vg.current()
    written = 0
    for phase in pl.range(2):
        for start in pl.range(0, width, STREAMING_SELECTION_CHUNK):
            if written < topk:
                active = pl.min(STREAMING_SELECTION_CHUNK, width - start)
                pl.set_validshape(src, [1, active])
                pl.load(src, scores, [row, start])
                if phase == 0:
                    _streaming_selection_collect_keys(
                        src, local, ct, active, lo, False
                    )
                else:
                    _streaming_selection_collect_keys(
                        src, local, ct, active, lo, True
                    )
                available = pl.getval(ct, 0)
                count = pl.min(available, topk - written)
                if count > 0:
                    _selection_emit_selection(
                        local, src, oi, ov, count, count,
                        start, local, False, offset, write_values,
                    )
                    pl.set_validshape(oi, [1, count])
                    pl.set_validshape(ov, [1, count])
                    pl.store(indices, oi, [row, written])
                    if write_values:
                        pl.store(values, ov, [row, written])
                    written = written + count
    for start in pl.range(written, topk, STREAMING_SELECTION_CHUNK):
        count = pl.min(STREAMING_SELECTION_CHUNK, topk - start)
        _streaming_selection_fill_padding(oi, ov, count)
        pl.set_validshape(oi, [1, count])
        pl.set_validshape(ov, [1, count])
        pl.store(indices, oi, [row, start])
        if write_values:
            pl.store(values, ov, [row, start])


def _generic_finish(
    scores: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], BF],
    plan: pl.Tensor[[pl.DYNAMIC], I32],
    maxima: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], BF],
    offsets: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], I32],
    indices: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], I32],
    values: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], BF],
    candidates: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], I32],
    lengths: pl.Tensor[[pl.DYNAMIC, 1], I32],
    capacity: pl.DT_INT64,
 flags, cmp_ratio,):
    """Complete long-row selection and candidate stages in the score launch.

    The MIX barrier drains all score producers before row owners reuse UB.
    Each AICPU-assigned row stays on one AIV through its complete radix scans,
    block reduction and candidate selection; no host stage loop is needed.
    """
    rows = scores.shape[0]
    width = scores.shape[1]
    topk = indices.shape[2]
    flat_indices = pl.make_tensor(indices, [rows, topk])
    flat_values = pl.make_tensor(values, [rows, topk])
    sg, xg, cg, ig, vg = _streaming_selection_groups()
    with pl.section_cube():
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
    with pl.section_vector():
        pl.system.bar_all()
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        row_begin, row_end = _vector_row_bounds(plan, rows)
        for row in pl.range(row_begin, row_end):
            if not SelectWidth:
                offset = 0
                if (flags & 4):
                    offset = pl.getval(offsets, row)
                lo = _streaming_selection_split(scores, (sg, cg), (row, topk, width))
                _streaming_selection_write(
                    scores, flat_indices, flat_values, (sg, xg, cg, ig, vg),
                    (row, offset, topk, width, lo), ReturnValues,
                )
            if (flags & 16):
                visible = _row_visible(plan, row, cmp_ratio)
                partial = -1
                if visible % 8 != 0:
                    partial = visible // 8
                count = pl.min(capacity, (visible + 7) // 8)
                length = pl.reinterpret(cg.current(), dtype=I32, shape=[1, 8])
                _candidates_write_count(length, count)
                pl.set_validshape(length, [1, 1])
                pl.store(lengths, length, [row, 0])
                for start in pl.range(0, width, STREAMING_SELECTION_CHUNK):
                    active = pl.min(STREAMING_SELECTION_CHUNK, width - start)
                    src = sg.current()
                    out = vg.current()
                    pl.set_validshape(src, [1, active])
                    pl.load(src, scores, [row, start])
                    _candidates_block_max(src, out, active, start, partial)
                    pl.set_validshape(out, [1, (active + 7) // 8])
                    pl.store(maxima, out, [row, start // 8])
                # This row's block maxima must reach GM before the radix scan.
                pl.system.bar_all()
                blocks = maxima.shape[1]
                flat_candidates = pl.make_tensor(candidates, [rows, capacity])
                lo = _streaming_selection_split(maxima, (sg, cg), (row, capacity, blocks))
                _streaming_selection_write(
                    maxima, flat_candidates, flat_values, (sg, xg, cg, ig, vg),
                    (row, 0, capacity, blocks, lo), False,
                )


# ============================================================================
# Empty
# ============================================================================

# Native form. Device-side output fill when every sequence is empty.
#
# This inline body uses Vec IDs 16..18 and scratch after the general path's
# highest address. The two runtime branches therefore have disjoint buffers
# and mutex IDs. No score workspace or Cube/Vector handover is needed here.


def _empty_empty_outputs(plan, indices, values, candidates, lengths, capacity, flags):
    topk = indices.shape[2]
    output_width = 8192
    if FixedTopK:
        topk = FixedTopK
        output_width = FixedTopK
    base = 51072 + 6 * max(256, SelectWidth) + 4608 + 6 * 8192
    ig = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, output_width], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[base],
        mutex_ids=[16],
    )
    vg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, output_width], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[base + 4 * output_width],
        mutex_ids=[17],
    )
    lg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[base + 6 * output_width],
        mutex_ids=[18],
    )
    with pl.section_vector():
        row_begin, row_end = _vector_row_bounds(plan, indices.shape[0])
        for row in pl.range(row_begin, row_end):
            oi = ig.current()
            ov = vg.current()
            _streaming_selection_fill_padding(oi, ov, topk)
            pl.set_validshape(oi, [1, topk])
            pl.set_validshape(ov, [1, topk])
            pl.store(indices, oi, [row, 0, 0])
            if ReturnValues:
                pl.store(values, ov, [row, 0, 0])
            if (flags & 8):
                _streaming_selection_fill_padding(oi, ov, capacity)
                pl.set_validshape(oi, [1, capacity])
                pl.store(candidates, oi, [row, 0, 0])
                count = lg.current()
                _candidates_write_count(count, 0)
                pl.set_validshape(count, [1, 1])
                pl.store(lengths, count, [row, 0])


# ============================================================================
# Generic
# ============================================================================

# Native form (explicit resources and MX moves). MXFP4 QK with the QLI BF16
# sequential head reduction.
#
# Each block owns a six-query group and a contiguous key shard. Q/K carry
# logical [M,128]/[N,128] FP4 values, scales [M,2,2]/[N,2,2]. Page gathers
# happen on Cube. The two AIVs receive three queries each by directed BF16
# FIX copies. At N=128 the two receiving slots use 49152 bytes/AIV, the
# weight/output scratch less than 2 KiB; L1 < 64 KiB, L0A/B < 32 KiB,
# L0C 196608 bytes. Every head FMA rounds to BF16, as the upstream evaluator
# requires. This general path materializes scores. For <=16384 keys a drained
# MIX barrier hands them to row-owned selection in the same native task.
# Candidate blocks reuse those scores and the selection scratch. Larger
# inputs use bounded hierarchical selection. All batch/query/key axes
# are dynamic; only physical tile capacities and layout choices specialize.
# The explicit addresses, mutex IDs and paired L0 moves preserve the former
# lowering, while supporting runtimes without allocation/GEMM sugar.


# Explicit preload=0 handshake: initialize both reverse tokens once before
# the full grid, use FIX->V and V->FIX per slot, and drain both after the grid.
# The slot counter advances across outer items, including partial query groups.


GENERIC_MQ = 6
GENERIC_MM = 192
GENERIC_HALF = 96
GENERIC_VEC_N = 128
GENERIC_SELECTION_BASE = 396 * GENERIC_VEC_N + 384
GENERIC_FORWARD_IDS = [0, 1]
GENERIC_BACKWARD_IDS = [0, 1]


@pl.vector_function
def _generic_reduce_heads(src, weight, dst, row, valid_count):
    p32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=BF)
    zero_bits = vf.full(0, p16, dtype=pl.DT_UINT16)
    zero = vf.bit_cast(zero_bits, dtype=BF)
    inf_bits = vf.full(0xFF80, p16, dtype=pl.DT_UINT16)
    neginf = vf.bit_cast(inf_bits, dtype=BF)
    for c in pl.range(GENERIC_VEC_N // 128):
        acc = vf.full(zero, p16)
        for h in pl.range(32):
            v = vf.load_align(src, [(row * 32 + h), c * 128])
            v = vf.max(v, zero, p16)
            w32 = vf.load_align(weight, [row, h], dist=pl.LoadDist.BRC_B32)
            w16e = vf.astype(w32, p32, dtype=BF, layout=pl.CastLayout.ZERO)
            w16o = vf.astype(w32, p32, dtype=BF, layout=pl.CastLayout.ONE)
            w16 = vf.xor(w16e, w16o, p16)
            acc = vf.mul_add_dst(v, w16, p16)
        valid = vf.update_mask(pl.max(0, pl.min(128, valid_count - c * 128)), dtype=BF)
        value = vf.select(acc, neginf, valid)
        vf.store_align(dst, value, p16, [row, c * 128])


def _stage_query_panel(t, groups, pos, q, qs):
    """Stage the 32-head query panel and its scales once per query group."""
    qg, qsg = groups
    qbase, qvalid = pos
    qt = qg.current()
    qst = qsg.current()
    if t == 0:
        pl.set_validshape(qt, [qvalid * 32, 128])
        pl.set_validshape(qst, [qvalid * 32, 4])
        pl.load(qt, q, [qbase * 32, 0])
        pl.load(qst, qs, [qbase * 32, 0, 0])
    return qt, qst


def _generic_cube_step(
    t, qbase, qvalid, b, kbase, kl, kstart, q, k, qs, ks, table,
    qg, kg, qsg, ksg, la, lb, sa, sb, cg, vg, k_page_stride, ks_page_stride,
 flags, page_size,):
    qt, qst = _stage_query_panel(t, (qg, qsg), (qbase, qvalid), q, qs)
    kt = kg.next()
    kst = ksg.next()
    # One page per step for the first interface probe (N=page_size=128).
    active = pl.min(TileN, pl.max(0, kl - kbase))
    pl.set_validshape(kt, [128, active])
    pl.set_validshape(kst, [4, active])
    physical = kstart + kbase
    physical_s = kstart + kbase
    if flags & 1:
        page = 0
        within_page = 0
        if flags & 64:
            page = kbase // TileN
            within_page = kbase % TileN
        else:
            page = kbase // page_size
            within_page = kbase % page_size
        if flags & 2:
            page = pl.getval(table, b * table.shape[1] + page)
        physical = page * k_page_stride + within_page
        physical_s = page * ks_page_stride + within_page
    vt = vg.next()
    if active > 0:
        pl.load(kt, k, [physical, 0], order=[1, 0])
        pl.load(kst, ks, [physical_s, 0, 0], order=[1, 0])
        ct = cg.next()
        a = la.next()
        a_scale = sa.next()
        b_tile = lb.next()
        b_scale = sb.next()
        pl.move(a, qt)
        pl.move(a_scale, qst)
        pl.move(b_tile, kt)
        pl.move(b_scale, kst)
        pl.matmul_mx(ct, a, b_tile, a_scale, b_scale)
        pl.set_validshape(vt, [GENERIC_HALF, TileN])
        pl.move(vt, ct, [0, 0], acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
        pl.move(
            vt, ct, [GENERIC_HALF, 0], acc_to_vec_mode=pl.AccToVecMode.SingleModeVec1
        )


@pl.vector_function
def _generic_fill_invalid(tile):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    bits = vf.full(0xFF80, mask, dtype=pl.DT_UINT16)
    values = vf.bit_cast(bits, dtype=BF)
    for i in pl.range(0, 3 * GENERIC_VEC_N, 128):
        vf.store_align(tile, values, mask, i)


def _generic_vector_step(plan, weights, scores, groups, span, cmp_ratio):
    wg, og, vg = groups
    t, qbase, qvalid, kbase, kl = span
    vt = vg.next()
    wt = wg.current()
    sub = pl.get_subblock_idx()
    if t == 0:
        pl.set_validshape(wt, [pl.max(0, pl.min(3, qvalid - sub * 3)), 32])
        if qvalid > sub * 3:
            pl.load(wt, weights, [qbase + sub * 3, 0])
    ot = og.next()
    if kbase < kl:
        for r in pl.range(pl.max(0, pl.min(3, qvalid - sub * 3))):
            visible = _row_visible(plan, qbase + sub * 3 + r, cmp_ratio)
            _generic_reduce_heads(
                vt, wt, ot, r, pl.max(0, pl.min(TileN, visible - kbase))
            )
    else:
        _generic_fill_invalid(ot)
    nr = pl.max(0, pl.min(3, qvalid - sub * 3))
    if nr > 0:
        pl.set_validshape(ot, [nr, pl.min(TileN, scores.shape[1] - kbase)])
        pl.store(scores, ot, [qbase + sub * 3, kbase])


def _generic_score_generic(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], F4],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], F4],
    weights: pl.Tensor[[pl.DYNAMIC, 32], F32],
    qs: pl.Tensor[[pl.DYNAMIC, 2, 2], E8],
    ks: pl.Tensor[[pl.DYNAMIC, 2, 2], E8],
    plan: pl.Tensor[[pl.DYNAMIC], I32],
    table: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], I32],
    scores: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], BF],
    max_q: pl.DT_INT64,
    batches: pl.DT_INT64,
    offsets: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], I32],
    final_idx: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], I32],
    final_val: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], BF],
    candidates: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], I32],
    lengths: pl.Tensor[[pl.DYNAMIC, 1], I32],
    k_page_stride: pl.DT_INT64,
    ks_page_stride: pl.DT_INT64,
    k_stride: pl.DT_INT64,
    ks_stride: pl.DT_INT64,
    capacity: pl.DT_INT64,
    vector_workers: pl.DT_INT64,
 flags, page_size, cmp_ratio,):
    topk = final_idx.shape[2]
    output_width = 8192
    if FixedTopK:
        topk = FixedTopK
        output_width = FixedTopK
    block_width = (SelectWidth + 2047) // 2048 * 256
    selection_capacity = max(256, SelectWidth)
    rows = scores.shape[0]
    max_k = scores.shape[1]
    key_view = pl.make_tensor(k, [k.shape[0], 128], [k_stride * 2, 1])
    scale_view = pl.make_tensor(ks, [ks.shape[0], 2, 2], [ks_stride, 2, 1])
    qg = pl.make_tile_group(
        type=pl.TileType(
            shape=[GENERIC_MM, 128],
            dtype=F4,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
        ),
        addrs=[0],
        mutex_ids=[0],
    )
    kg = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, TileN],
            dtype=F4,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZN,
        ),
        addrs=[12288, 12288 + 64 * TileN],
        mutex_ids=[1, 2],
    )
    qsg = pl.make_tile_group(
        type=pl.TileType(
            shape=[GENERIC_MM, 4],
            dtype=E8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZZ,
        ),
        addrs=[12288 + 128 * TileN],
        mutex_ids=[3],
    )
    ksg = pl.make_tile_group(
        type=pl.TileType(
            shape=[4, TileN],
            dtype=E8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NN,
        ),
        addrs=[13056 + 128 * TileN, 13056 + 132 * TileN],
        mutex_ids=[4, 5],
    )
    la = pl.make_tile_group(
        type=pl.TileType(
            shape=[GENERIC_MM, 128],
            dtype=F4,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            compact=1,
        ),
        addrs=[0, 12288],
        mutex_ids=[6, 7],
    )
    lb = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, TileN],
            dtype=F4,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            compact=1,
        ),
        addrs=[0, 64 * TileN],
        mutex_ids=[8, 9],
    )
    sa = pl.make_tile_group(
        type=pl.TileType(
            shape=[GENERIC_MM, 4],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleLeft,
            layout=pl.ZZ,
            compact=1,
        ),
        addrs=[0, 768],
        mutex_ids=[10, 11],
    )
    sb = pl.make_tile_group(
        type=pl.TileType(
            shape=[4, TileN],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleRight,
            layout=pl.NN,
            compact=1,
        ),
        addrs=[0, 4 * TileN],
        mutex_ids=[12, 13],
    )
    cg = pl.make_tile_group(
        type=pl.TileType(
            shape=[GENERIC_MM, TileN],
            dtype=F32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
            compact=1,
        ),
        addrs=[0, GENERIC_MM * TileN * 4],
        mutex_ids=[14, 15],
    )
    vg = pl.make_tile_group(
        type=pl.TileType(
            shape=[GENERIC_HALF, GENERIC_VEC_N],
            dtype=BF,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[0, GENERIC_HALF * GENERIC_VEC_N * 2],
        mutex_ids=[0, 1],
    )
    wg = pl.make_tile_group(
        type=pl.TileType(shape=[3, 32], dtype=F32, target_memory=pl.MemorySpace.Vec),
        addrs=[GENERIC_HALF * GENERIC_VEC_N * 4],
        mutex_ids=[2],
    )
    og = pl.make_tile_group(
        type=pl.TileType(
            shape=[3, GENERIC_VEC_N], dtype=BF, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[
            GENERIC_HALF * GENERIC_VEC_N * 4 + 384,
            GENERIC_HALF * GENERIC_VEC_N * 4 + 384 + 6 * GENERIC_VEC_N,
        ],
        mutex_ids=[3, 4],
    )
    ss = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, selection_capacity],
            dtype=BF,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[GENERIC_SELECTION_BASE],
        mutex_ids=[5],
    )
    sk = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, selection_capacity],
            dtype=pl.DT_UINT16,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[GENERIC_SELECTION_BASE + 2 * selection_capacity],
        mutex_ids=[6],
    )
    sh = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[GENERIC_SELECTION_BASE + 4 * selection_capacity],
        mutex_ids=[7],
    )
    sl = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[GENERIC_SELECTION_BASE + 4 * selection_capacity + 1024],
        mutex_ids=[8],
    )
    sbuf = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[GENERIC_SELECTION_BASE + 4 * selection_capacity + 2048],
        mutex_ids=[9],
    )
    si = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 128], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[GENERIC_SELECTION_BASE + 4 * selection_capacity + 3072],
        mutex_ids=[10],
    )
    sn = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 128], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[GENERIC_SELECTION_BASE + 4 * selection_capacity + 3584],
        mutex_ids=[11],
    )
    st = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 128], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[GENERIC_SELECTION_BASE + 4 * selection_capacity + 4096],
        mutex_ids=[12],
    )
    sx = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, selection_capacity + 128],
            dtype=pl.DT_UINT16,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[GENERIC_SELECTION_BASE + 4 * selection_capacity + 4352],
        mutex_ids=[13],
    )
    so = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, output_width], dtype=I32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[GENERIC_SELECTION_BASE + 6 * selection_capacity + 4608],
        mutex_ids=[14],
    )
    sv = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, output_width], dtype=BF, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[
            GENERIC_SELECTION_BASE + 6 * selection_capacity + 4608 + 4 * output_width
        ],
        mutex_ids=[15],
    )
    core = pl.get_block_idx() // pl.get_subblock_num()
    sync_id = 0
    with pl.section_vector():
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=GENERIC_BACKWARD_IDS[0])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=GENERIC_BACKWARD_IDS[1])
    core_record = 1 + 3 * rows + 2 * vector_workers + 2 * core
    task_begin = pl.getval(plan, core_record)
    task_end = pl.getval(plan, core_record + 1)
    for item in pl.range(task_begin, task_end):
        record = 1 + 3 * rows + 2 * vector_workers + 2 * pl.get_block_num() + 8 * item
        b = pl.getval(plan, record + 5)
        qbase = pl.getval(plan, record)
        qvalid = pl.getval(plan, record + 1)
        kl = pl.getval(plan, record + 2)
        kstart = pl.getval(plan, record + 3)
        key_begin = pl.getval(plan, record + 6)
        tiles = pl.getval(plan, record + 7)
        if kl > 0:
            for t in pl.range(tiles):
                kb = key_begin + t * TileN
                with pl.section_cube():
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.FIX, event_id=GENERIC_BACKWARD_IDS[sync_id % 2]
                    )
                    _generic_cube_step(
                        t, qbase, qvalid, b, kb, kl, kstart,
                        q, key_view, qs, scale_view, table,
                        qg, kg, qsg, ksg, la, lb, sa, sb,
                        cg, vg, k_page_stride,
                        ks_page_stride,
                        flags, page_size)
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX, event_id=GENERIC_FORWARD_IDS[sync_id % 2]
                    )
                with pl.section_vector():
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.V, event_id=GENERIC_FORWARD_IDS[sync_id % 2]
                    )
                    _generic_vector_step(
                        plan, weights, scores, (wg, og, vg),
                        (t, qbase, qvalid, kb, kl),
                     cmp_ratio,)
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.V, event_id=GENERIC_BACKWARD_IDS[sync_id % 2]
                    )
                sync_id = sync_id + 1
        else:
            # An empty sequence needs only -inf score stores. Keep the handover
            # cursor unchanged: no Cube producer or VG consumer runs here.
            with pl.section_vector():
                sub = pl.get_subblock_idx()
                nr = pl.max(0, pl.min(3, qvalid - sub * 3))
                for t in pl.range(tiles):
                    kb = key_begin + t * TileN
                    ot = og.next()
                    _generic_fill_invalid(ot)
                    if nr > 0:
                        pl.set_validshape(ot, [nr, pl.min(TileN, max_k - kb)])
                        pl.store(scores, ot, [qbase + sub * 3, kb])
    with pl.section_cube():
        pl.system.wait_cross_core(
            pipe=pl.PipeType.FIX, event_id=GENERIC_BACKWARD_IDS[0]
        )
        pl.system.wait_cross_core(
            pipe=pl.PipeType.FIX, event_id=GENERIC_BACKWARD_IDS[1]
        )

    if SelectWidth:
        with pl.section_cube():
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        with pl.section_vector():
            pl.system.bar_all()
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
            row_begin, row_end = _vector_row_bounds(plan, rows)
            for row in pl.range(row_begin, row_end):
                src = ss.current()
                keys = sk.current()
                high = sh.current()
                low = sl.current()
                buf = sbuf.current()
                index = si.current()
                nth = sn.current()
                threshold = st.current()
                selected = sx.current()
                oi = so.current()
                ov = sv.current()
                pl.set_validshape(src, [1, max_k])
                pl.load(src, scores, [row, 0])
                offset = 0
                if (flags & 4):
                    offset = pl.getval(offsets, row)
                if max_k <= topk:
                    _selection_emit_ordered(src, oi, ov, topk, max_k, offset, ReturnValues)
                else:
                    _selection_keys_from_scores(src, keys, SelectWidth, max_k)
                    _selection_histograms_high_vf(high, keys, SelectWidth // 256, 0)
                    _selection_find_high_bin_and_histograms_low(
                        low,
                        index,
                        high,
                        keys,
                        buf,
                        nth,
                        SelectWidth - pl.min(topk, max_k) + 1,
                        SelectWidth // 256, 0,
                    )
                    _selection_find_kth_vf(threshold, low, index, buf, nth)
                    _selection_collect_indices_gt_eq_vf(
                        selected, keys, threshold, SelectWidth // 128, 0
                    )
                    _selection_emit_selection(
                        selected,
                        src,
                        oi,
                        ov,
                        topk,
                        pl.min(topk, max_k),
                        0,
                        src,
                        False,
                        offset,
                        ReturnValues,
                    )
                pl.set_validshape(oi, [1, topk])
                pl.set_validshape(ov, [1, topk])
                pl.store(final_idx, oi, [row, 0, 0])
                if ReturnValues:
                    pl.store(final_val, ov, [row, 0, 0])
                if (flags & 8):
                    visible = _row_visible(plan, row, cmp_ratio)
                    partial = -1
                    if visible % 8 != 0:
                        partial = visible // 8
                    row_count = pl.min(capacity, (visible + 7) // 8)
                    count = pl.reinterpret(buf, dtype=I32, shape=[1, 8])
                    _candidates_write_count(count, row_count)
                    pl.set_validshape(count, [1, 1])
                    pl.store(lengths, count, [row, 0])
                    # Each 128-score chunk is fully loaded before its 16
                    # maxima overwrite an already-consumed prefix of src.
                    _candidates_block_max(src, src, max_k, 0, partial)
                    blocks = (max_k + 7) // 8
                    if capacity >= blocks:
                        _selection_emit_ordered(
                            src, oi, ov, capacity, blocks, 0, False
                        )
                    else:
                        _selection_keys_from_scores(src, keys, block_width, blocks)
                        _selection_histograms_high_vf(high, keys, block_width // 256, 0)
                        _selection_find_high_bin_and_histograms_low(
                            low,
                            index,
                            high,
                            keys,
                            buf,
                            nth,
                            block_width - pl.min(capacity, blocks) + 1,
                            block_width // 256, 0,
                        )
                        _selection_find_kth_vf(threshold, low, index, buf, nth)
                        _selection_collect_indices_gt_eq_vf(
                            selected, keys, threshold, block_width // 128, 0
                        )
                        _selection_emit_selection(
                            selected,
                            src,
                            oi,
                            ov,
                            capacity,
                            capacity,
                            0,
                            src,
                            False,
                            0,
                            False,
                        )
                    pl.set_validshape(oi, [1, capacity])
                    pl.store(candidates, oi, [row, 0, 0])


# ============================================================================
# Paged
# ============================================================================

# Native form (L0 operands kept; native VF with explicit cross-core handshakes).
#
# Each Cube worker owns a six-query group and a contiguous key shard. Q/K
# are packed E2M1, with four E8M0 scales per 128 logical elements. Both L0A
# query halves remain live throughout the shard, and each L0B key panel feeds
# two distinct accumulators. This is manual GEMM exception 6: ordinary sugar
# would stage the unchanged operands again. Per-slot auto_mutex is retained.
# QK, weights and every sequential head FMA have the upstream BF16 rounding.
#
# Each AIV owns three queries. For N=256 the Acc view [96,256] -> [192,128]
# permutes 16-column groups, without moving bytes. The consumer reduces the
# matching head rows; cache_select restores logical key indices on emission.
# Partial masks follow that same permutation. The two interleaved UB slots
# use a 256-byte slot stride and a 512-byte row stride, so each slot holds
# 192 rows of 128 dense columns.
# Their accessed byte sets are disjoint. The conservative bounding boxes
# overlap, explaining MemSpaceCollision; full-width accesses are prohibited.
# Cube/VF handshakes protect each slot. A drained MIX barrier separates all
# partial writes from the final merge, after which the UB views can be reused.
#
# Resource spans: the Mat region ends at 198656 bytes, of which 47872 bytes are
# live payload. The L0 operand regions take 12288, 32768, 768 and 2048 bytes, and
# Acc takes 196608 bytes. Vec ends at 108544+8*L bytes for an L-token shard,
# including all slots, cached keys, histograms and selection scratch; dispatch
# enforces L<=19200. The kernel emits only bounded partial Top-K lists to GM and
# merges them in the same native call. No full score matrix is materialized.


# Explicit preload=0 handshake: initialize both reverse tokens once before
# the full grid, use FIX->V and V->FIX per slot, and drain both after the grid.
# The slot counter advances across outer items, including partial query groups.


PAGED_PAGE_SIZE = 128
PAGED_TILE_N = 256
PAGED_TOPK = 512
PAGED_HANDOVER_DEPTH = 2
PAGED_VG_IDS = [24, 25]
PAGED_FORWARD_IDS = [0, 1]
PAGED_BACKWARD_IDS = [2, 3]
PAGED_MQ = 6
PAGED_MM = 192
PAGED_HALF = 96
PAGED_KR = 128
PAGED_KN = 256


@pl.vector_function
def _paged_convert_weights(src, dst, count):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
    for c in pl.range(0, 128, 64):
        active = vf.update_mask(pl.max(0, pl.min(64, count - c)), dtype=F32)
        value = vf.load_align(src, c)
        narrow = vf.astype(value, active, dtype=BF, layout=pl.CastLayout.ZERO)
        bits = vf.bit_cast(narrow, dtype=pl.DT_UINT32)
        vf.store_align(dst, bits, p, c, dist=pl.StoreDist.PACK_B32)


@pl.vector_function
def _paged_reduce_heads(src, weight, dst, hist, position, e0, o0, e1, o1, e2, o2, all_valid):
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=BF)
    zb = vf.full(0, p16, dtype=pl.DT_UINT16)
    zero = vf.bit_cast(zb, dtype=BF)
    sign = vf.full(32768, p16, dtype=pl.DT_UINT16)
    flipneg = vf.full(65535, p16, dtype=pl.DT_UINT16)
    p8 = vf.update_mask(128, dtype=pl.DT_UINT8)
    acc0_0 = vf.full(zero, p16)
    acc0_1 = vf.full(zero, p16)
    acc1_0 = vf.full(zero, p16)
    acc1_1 = vf.full(zero, p16)
    acc2_0 = vf.full(zero, p16)
    acc2_1 = vf.full(zero, p16)
    for h in pl.range(32):
        w0 = vf.load_align(weight, [0, h], dist=pl.LoadDist.BRC_B16)
        qk0_0 = vf.load_align(src, [0 * 32 + h + 0 * PAGED_HALF, 0])
        relu0_0 = vf.max(qk0_0, zero, p16)
        acc0_0 = vf.mul_add_dst(relu0_0, w0, p16)
        qk0_1 = vf.load_align(src, [0 * 32 + h + 1 * PAGED_HALF, 0])
        relu0_1 = vf.max(qk0_1, zero, p16)
        acc0_1 = vf.mul_add_dst(relu0_1, w0, p16)
        w1 = vf.load_align(weight, [1, h], dist=pl.LoadDist.BRC_B16)
        qk1_0 = vf.load_align(src, [1 * 32 + h + 0 * PAGED_HALF, 0])
        relu1_0 = vf.max(qk1_0, zero, p16)
        acc1_0 = vf.mul_add_dst(relu1_0, w1, p16)
        qk1_1 = vf.load_align(src, [1 * 32 + h + 1 * PAGED_HALF, 0])
        relu1_1 = vf.max(qk1_1, zero, p16)
        acc1_1 = vf.mul_add_dst(relu1_1, w1, p16)
        w2 = vf.load_align(weight, [2, h], dist=pl.LoadDist.BRC_B16)
        qk2_0 = vf.load_align(src, [2 * 32 + h + 0 * PAGED_HALF, 0])
        relu2_0 = vf.max(qk2_0, zero, p16)
        acc2_0 = vf.mul_add_dst(relu2_0, w2, p16)
        qk2_1 = vf.load_align(src, [2 * 32 + h + 1 * PAGED_HALF, 0])
        relu2_1 = vf.max(qk2_1, zero, p16)
        acc2_1 = vf.mul_add_dst(relu2_1, w2, p16)
    h0lo = vf.load_align(hist, 0 * 256)
    h0hi = vf.load_align(hist, 0 * 256 + 128)
    bits0_0 = vf.bit_cast(acc0_0, dtype=pl.DT_UINT16)
    negative0_0 = vf.ge(bits0_0, sign, p16)
    flip0_0 = vf.select(flipneg, sign, negative0_0)
    rawkey0_0 = vf.xor(bits0_0, flip0_0, p16)
    if all_valid:
        key0_0 = rawkey0_0
    else:
        mask0_0 = vf.update_mask(e0, dtype=BF)
        key0_0 = vf.select(rawkey0_0, zb, mask0_0)
    vf.store_align(dst, key0_0, p16, [0, position + 0 * 128])
    high0_0 = vf.shift_right(key0_0, 8, p16)
    packed0_0 = vf.pack(high0_0, dtype=pl.DT_UINT8, part=pl.PackPart.LOWER)
    h0lo = vf.histograms(
        packed0_0, p8, bin_type=pl.BinType.BIN0, hist_type=pl.HistType.ACCUMULATE
    )
    h0hi = vf.histograms(
        packed0_0, p8, bin_type=pl.BinType.BIN1, hist_type=pl.HistType.ACCUMULATE
    )
    bits0_1 = vf.bit_cast(acc0_1, dtype=pl.DT_UINT16)
    negative0_1 = vf.ge(bits0_1, sign, p16)
    flip0_1 = vf.select(flipneg, sign, negative0_1)
    rawkey0_1 = vf.xor(bits0_1, flip0_1, p16)
    if all_valid:
        key0_1 = rawkey0_1
    else:
        mask0_1 = vf.update_mask(o0, dtype=BF)
        key0_1 = vf.select(rawkey0_1, zb, mask0_1)
    vf.store_align(dst, key0_1, p16, [0, position + 1 * 128])
    high0_1 = vf.shift_right(key0_1, 8, p16)
    packed0_1 = vf.pack(high0_1, dtype=pl.DT_UINT8, part=pl.PackPart.LOWER)
    h0lo = vf.histograms(
        packed0_1, p8, bin_type=pl.BinType.BIN0, hist_type=pl.HistType.ACCUMULATE
    )
    h0hi = vf.histograms(
        packed0_1, p8, bin_type=pl.BinType.BIN1, hist_type=pl.HistType.ACCUMULATE
    )
    vf.store_align(hist, h0lo, p16, 0 * 256)
    vf.store_align(hist, h0hi, p16, 0 * 256 + 128)
    h1lo = vf.load_align(hist, 1 * 256)
    h1hi = vf.load_align(hist, 1 * 256 + 128)
    bits1_0 = vf.bit_cast(acc1_0, dtype=pl.DT_UINT16)
    negative1_0 = vf.ge(bits1_0, sign, p16)
    flip1_0 = vf.select(flipneg, sign, negative1_0)
    rawkey1_0 = vf.xor(bits1_0, flip1_0, p16)
    if all_valid:
        key1_0 = rawkey1_0
    else:
        mask1_0 = vf.update_mask(e1, dtype=BF)
        key1_0 = vf.select(rawkey1_0, zb, mask1_0)
    vf.store_align(dst, key1_0, p16, [1, position + 0 * 128])
    high1_0 = vf.shift_right(key1_0, 8, p16)
    packed1_0 = vf.pack(high1_0, dtype=pl.DT_UINT8, part=pl.PackPart.LOWER)
    h1lo = vf.histograms(
        packed1_0, p8, bin_type=pl.BinType.BIN0, hist_type=pl.HistType.ACCUMULATE
    )
    h1hi = vf.histograms(
        packed1_0, p8, bin_type=pl.BinType.BIN1, hist_type=pl.HistType.ACCUMULATE
    )
    bits1_1 = vf.bit_cast(acc1_1, dtype=pl.DT_UINT16)
    negative1_1 = vf.ge(bits1_1, sign, p16)
    flip1_1 = vf.select(flipneg, sign, negative1_1)
    rawkey1_1 = vf.xor(bits1_1, flip1_1, p16)
    if all_valid:
        key1_1 = rawkey1_1
    else:
        mask1_1 = vf.update_mask(o1, dtype=BF)
        key1_1 = vf.select(rawkey1_1, zb, mask1_1)
    vf.store_align(dst, key1_1, p16, [1, position + 1 * 128])
    high1_1 = vf.shift_right(key1_1, 8, p16)
    packed1_1 = vf.pack(high1_1, dtype=pl.DT_UINT8, part=pl.PackPart.LOWER)
    h1lo = vf.histograms(
        packed1_1, p8, bin_type=pl.BinType.BIN0, hist_type=pl.HistType.ACCUMULATE
    )
    h1hi = vf.histograms(
        packed1_1, p8, bin_type=pl.BinType.BIN1, hist_type=pl.HistType.ACCUMULATE
    )
    vf.store_align(hist, h1lo, p16, 1 * 256)
    vf.store_align(hist, h1hi, p16, 1 * 256 + 128)
    h2lo = vf.load_align(hist, 2 * 256)
    h2hi = vf.load_align(hist, 2 * 256 + 128)
    bits2_0 = vf.bit_cast(acc2_0, dtype=pl.DT_UINT16)
    negative2_0 = vf.ge(bits2_0, sign, p16)
    flip2_0 = vf.select(flipneg, sign, negative2_0)
    rawkey2_0 = vf.xor(bits2_0, flip2_0, p16)
    if all_valid:
        key2_0 = rawkey2_0
    else:
        mask2_0 = vf.update_mask(e2, dtype=BF)
        key2_0 = vf.select(rawkey2_0, zb, mask2_0)
    vf.store_align(dst, key2_0, p16, [2, position + 0 * 128])
    high2_0 = vf.shift_right(key2_0, 8, p16)
    packed2_0 = vf.pack(high2_0, dtype=pl.DT_UINT8, part=pl.PackPart.LOWER)
    h2lo = vf.histograms(
        packed2_0, p8, bin_type=pl.BinType.BIN0, hist_type=pl.HistType.ACCUMULATE
    )
    h2hi = vf.histograms(
        packed2_0, p8, bin_type=pl.BinType.BIN1, hist_type=pl.HistType.ACCUMULATE
    )
    bits2_1 = vf.bit_cast(acc2_1, dtype=pl.DT_UINT16)
    negative2_1 = vf.ge(bits2_1, sign, p16)
    flip2_1 = vf.select(flipneg, sign, negative2_1)
    rawkey2_1 = vf.xor(bits2_1, flip2_1, p16)
    if all_valid:
        key2_1 = rawkey2_1
    else:
        mask2_1 = vf.update_mask(o2, dtype=BF)
        key2_1 = vf.select(rawkey2_1, zb, mask2_1)
    vf.store_align(dst, key2_1, p16, [2, position + 1 * 128])
    high2_1 = vf.shift_right(key2_1, 8, p16)
    packed2_1 = vf.pack(high2_1, dtype=pl.DT_UINT8, part=pl.PackPart.LOWER)
    h2lo = vf.histograms(
        packed2_1, p8, bin_type=pl.BinType.BIN0, hist_type=pl.HistType.ACCUMULATE
    )
    h2hi = vf.histograms(
        packed2_1, p8, bin_type=pl.BinType.BIN1, hist_type=pl.HistType.ACCUMULATE
    )
    vf.store_align(hist, h2lo, p16, 2 * 256)
    vf.store_align(hist, h2hi, p16, 2 * 256 + 128)


def _paged_cube_step(
    t, qbase, qvalid, b, kbase, kl, q, k, qs, ks,
    table, qg, kg, qsg, ksg, la, lb, sa, sb, cg,
    cgv, vg, kp0, kp1, sp0, sp1, k_page_stride,
    ks_page_stride,
):
    qt, qst = _stage_query_panel(t, (qg, qsg), (qbase, qvalid), q, qs)
    kt = kg.next()
    kst = ksg.next()
    active = pl.min(PAGED_TILE_N, pl.max(0, kl - kbase))
    pl.set_validshape(kt, [128, active])
    pl.set_validshape(kst, [4, active])
    page = 0
    if active > 0:
        page = pl.getval(table, b * table.shape[1] + kbase // PAGED_PAGE_SIZE)
    physical = page * k_page_stride + kbase % PAGED_PAGE_SIZE
    physical_s = page * ks_page_stride + kbase % PAGED_PAGE_SIZE
    p0 = kp0.next()
    p1 = kp1.next()
    s0 = sp0.next()
    s1 = sp1.next()
    page1 = page
    if active > 128:
        page1 = pl.getval(table, b * table.shape[1] + kbase // PAGED_PAGE_SIZE + 1)
    pl.set_validshape(p0, [128, 128])
    pl.set_validshape(p1, [128, 128])
    pl.load(p0, k, [physical, 0], order=[1, 0])
    pl.load(p1, k, [page1 * k_page_stride, 0], order=[1, 0])
    pl.load(s0, ks, [physical_s, 0, 0], order=[1, 0])
    pl.load(s1, ks, [page1 * ks_page_stride, 0, 0], order=[1, 0])
    a0, a1, as0, as1 = la[0], la[1], sa[0], sa[1]
    if t == 0:
        pl.move(a0, qt, [0, 0])
        pl.move(a1, qt, [PAGED_HALF, 0])
        pl.move(as0, qst, [0, 0])
        pl.move(as1, qst, [PAGED_HALF, 0])
    br = lb.next()
    bs = sb.next()
    pl.move(br, kt)
    pl.move(bs, kst)
    ct0 = cg.next()
    pl.matmul_mx(ct0, a0, br, scale_a=as0, scale_b=bs)
    vt = vg.next()
    pl.set_validshape(vt, [PAGED_HALF * 2, PAGED_TILE_N // 2])
    pl.move(vt, cgv[0], acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
    ct1 = cg.next()
    pl.matmul_mx(ct1, a1, br, scale_a=as1, scale_b=bs)
    pl.move(vt, cgv[1], acc_to_vec_mode=pl.AccToVecMode.SingleModeVec1)


def _paged_emit_shard(groups, buffers, out_idx, out_val, span):
    """Select this shard's Top-K per query row and write the partial output.

    ``groups`` are the selection buffers, ``span`` carries the query row base,
    the shard index, the shard width and the valid row count of this subblock.
    """
    hhg, hlg, bg, ig, ng, tg, xg, oig, ovg = groups
    ot, hist = buffers
    row, part, shard_width, nr, key_begin = span
    ht = hhg.current()
    lt = hlg.current()
    bt = bg.current()
    it = ig.current()
    nt = ng.current()
    tt = tg.current()
    xt = xg.current()
    oi = oig.current()
    ov = ovg.current()
    for r in pl.range(nr):
        _cache_select_widen_histogram(hist, ht, r)
        _selection_find_high_bin_and_histograms_low(
            lt, it, ht, ot, bt, nt,
            shard_width - PAGED_TOPK + 1, shard_width // 256, r * shard_width,
        )
        _selection_find_kth_vf(tt, lt, it, bt, nt)
        _selection_collect_indices_gt_eq_vf(
            xt, ot, tt, shard_width // 128, r * shard_width
        )
        _cache_select_emit_cached(
            xt, ot, oi, ov, PAGED_TOPK, r * shard_width,
            key_begin, False, 0, True, 0, True,
        )
        pl.store(out_idx, oi, [row + r, part * PAGED_TOPK])
        pl.store(out_val, ov, [row + r, part * PAGED_TOPK])


def _paged_vector_step(weights, out_idx, out_val, groups, span):
    wg, wbg, og, vg, hg, hhg, hlg, bg, ig, ng, tg, xg, oig, ovg = groups
    t, qbase, qvalid, kbase, part, steps, visible0, visible1, visible2, shard_width, key_begin = span
    vt = vg.next()
    wt = wg.current()
    wbt = wbg.current()
    sub = pl.get_subblock_idx()
    nr = pl.max(0, pl.min(3, qvalid - sub * 3))
    ot = og.current()
    hist = hg.current()
    if t == 0:
        pl.set_validshape(wt, [nr, 32])
        if qvalid > sub * 3:
            pl.load(wt, weights, [qbase + sub * 3, 0])
        _paged_convert_weights(wt, wbt, nr * 32)
        _cache_select_clear_cache(ot, hist, shard_width, steps * PAGED_TILE_N)
    valid0 = pl.max(0, pl.min(PAGED_TILE_N, visible0 - kbase))
    valid1 = pl.max(0, pl.min(PAGED_TILE_N, visible1 - kbase))
    valid2 = pl.max(0, pl.min(PAGED_TILE_N, visible2 - kbase))
    e0 = valid0 // 32 * 16 + pl.min(16, valid0 % 32)
    o0 = valid0 // 32 * 16 + pl.max(0, valid0 % 32 - 16)
    e1 = valid1 // 32 * 16 + pl.min(16, valid1 % 32)
    o1 = valid1 // 32 * 16 + pl.max(0, valid1 % 32 - 16)
    e2 = valid2 // 32 * 16 + pl.min(16, valid2 % 32)
    o2 = valid2 // 32 * 16 + pl.max(0, valid2 % 32 - 16)
    position = t * PAGED_TILE_N
    if kbase + PAGED_TILE_N <= pl.min(visible0, pl.min(visible1, visible2)):
        _paged_reduce_heads(
            vt, wbt, ot, hist, position, e0, o0, e1, o1, e2, o2, True
        )
    else:
        _paged_reduce_heads(
            vt, wbt, ot, hist, position, e0, o0, e1, o1, e2, o2, False
        )
    if t == steps - 1:
        _paged_emit_shard(
            (hhg, hlg, bg, ig, ng, tg, xg, oig, ovg), (ot, hist),
            out_idx, out_val, (qbase + sub * 3, part, shard_width, nr, key_begin),
        )


def _paged_score_paged(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], F4],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], F4],
    weights: pl.Tensor[[pl.DYNAMIC, 32], F32],
    qs: pl.Tensor[[pl.DYNAMIC, 2, 2], E8],
    ks: pl.Tensor[[pl.DYNAMIC, 2, 2], E8],
    plan: pl.Tensor[[pl.DYNAMIC], I32],
    table: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], I32],
    out_idx: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], I32],
    out_val: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], BF],
    final_idx: pl.Tensor[[pl.DYNAMIC, PAGED_TOPK], I32],
    final_val: pl.Tensor[[pl.DYNAMIC, PAGED_TOPK], BF],
    offsets: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], I32],
    max_q: pl.DT_INT64,
    batches: pl.DT_INT64,
    k_page_stride: pl.DT_INT64,
    ks_page_stride: pl.DT_INT64,
    vector_workers: pl.DT_INT64,
 flags, cmp_ratio,):
    """Fused PA128 score and Top-K, with explicit packed-FP4 storage.

    qg is packed FP4 [192,128], exactly 192*128/2=12288 Mat bytes at [0,12288).
    qsg uses [12288,13056); these accessed byte sets are disjoint. A checker
    rounding each four-bit element to one byte overestimates qg as [0,24576).
    Explicit addresses preserve the packed layout; no full-byte Q access occurs.
    """
    # AICPU supplies the shard boundaries. The 9472-element bucket is the
    # 128K geometry with 14 partial Top-K lists per query row.
    shard_width = ShardCapacity
    parts = 14 if ShardCapacity == 9472 else 8
    merge_width = parts * PAGED_TOPK
    rows = weights.shape[0]
    ub = PAGED_HALF * PAGED_TILE_N * 2 * PAGED_HANDOVER_DEPTH
    # Running UB offsets for this stage, in allocation order. Each value is
    # the byte where its buffer starts, and each increment is a byte size.
    at_wg = ub + 256
    at_wbg = at_wg + 512
    at_og = at_wbg + 256
    at_hg = at_og + 3 * shard_width * 2
    at_hhg = at_hg + 1536
    at_hlg = at_hhg + 1024
    at_bg = at_hlg + 1024
    at_ig = at_bg + 1024
    at_ng = at_ig + 512
    at_tg = at_ng + 512
    at_xg = at_tg + 256
    at_oig = at_xg + (shard_width + 128) * 2
    at_ovg = at_oig + PAGED_TOPK * 4
    qg = pl.make_tile_group(
        type=pl.TileType(
            shape=[PAGED_MM, 128],
            dtype=F4,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
        ),
        addrs=[0],
        mutex_ids=[0],
    )
    qsg = pl.make_tile_group(
        type=pl.TileType(
            shape=[PAGED_MM, 4],
            dtype=E8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZZ,
        ),
        addrs=[12288],
        mutex_ids=[1],
    )
    la = pl.make_tile_group(
        type=pl.TileType(
            shape=[PAGED_HALF, 128],
            dtype=F4,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            compact=1,
        ),
        addrs=[0, 6144],
        mutex_ids=[2, 3],
    )
    lb = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, PAGED_TILE_N],
            dtype=F4,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            compact=1,
        ),
        addrs=[0, 16384],
        mutex_ids=[4, 5],
    )
    sa = pl.make_tile_group(
        type=pl.TileType(
            shape=[PAGED_HALF, 4],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleLeft,
            layout=pl.ZZ,
            compact=1,
        ),
        addrs=[0, 384],
        mutex_ids=[6, 7],
    )
    sb = pl.make_tile_group(
        type=pl.TileType(
            shape=[4, PAGED_TILE_N],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleRight,
            layout=pl.NN,
            compact=1,
        ),
        addrs=[0, 1024],
        mutex_ids=[8, 9],
    )
    cg = pl.make_tile_group(
        type=pl.TileType(
            shape=[PAGED_HALF, PAGED_TILE_N],
            dtype=F32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
            compact=1,
        ),
        addrs=[0, PAGED_HALF * PAGED_TILE_N * 4],
        mutex_ids=[10, 11],
    )
    cgv = pl.make_tile_group(
        type=pl.TileType(
            shape=[PAGED_HALF * 2, PAGED_TILE_N // 2],
            dtype=F32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
            compact=1,
        ),
        addrs=[0, PAGED_HALF * PAGED_TILE_N * 4],
        mutex_ids=[10, 11],
    )
    vg = pl.make_tile_group(
        type=pl.TileType(
            shape=[PAGED_HALF * 2, PAGED_TILE_N],
            dtype=BF,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[0, PAGED_TILE_N],
        depth=PAGED_HANDOVER_DEPTH,
        mutex_ids=PAGED_VG_IDS,
    )
    wg = pl.make_tile_group(
        type=pl.TileType(shape=[4, 32], dtype=F32, target_memory=pl.MemorySpace.Vec),
        addrs=[at_wg],
        mutex_ids=[26],
    )
    wbg = pl.make_tile_group(
        type=pl.TileType(shape=[4, 32], dtype=BF, target_memory=pl.MemorySpace.Vec),
        addrs=[at_wbg],
        mutex_ids=[27],
    )
    og = pl.make_tile_group(
        type=pl.TileType(
            shape=[3, shard_width],
            dtype=pl.DT_UINT16,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[at_og],
        mutex_ids=[0],
    )
    hg = pl.make_tile_group(
        type=pl.TileType(
            shape=[3, 256], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[at_hg],
        mutex_ids=[1],
    )
    hhg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[at_hhg],
        mutex_ids=[2],
    )
    hlg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[at_hlg],
        mutex_ids=[3],
    )
    bg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[at_bg],
        mutex_ids=[4],
    )
    ig = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 128], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[at_ig],
        mutex_ids=[5],
    )
    ng = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 128], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[at_ng],
        mutex_ids=[6],
    )
    tg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 128], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[at_tg],
        mutex_ids=[7],
    )
    xg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, shard_width + 128],
            dtype=pl.DT_UINT16,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[at_xg],
        mutex_ids=[8],
    )
    oig = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, PAGED_TOPK], dtype=I32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[at_oig],
        mutex_ids=[9],
    )
    ovg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, PAGED_TOPK], dtype=BF, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[at_ovg],
        mutex_ids=[10],
    )
    kg = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, PAGED_TILE_N],
            dtype=pl.DT_FP4E2M1,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZN,
        ),
        addrs=[65536, 131072],
        mutex_ids=[20, 21],
    )
    kp0 = pl.make_tile_group(
        type=pl.TileType(
            shape=[PAGED_KR, PAGED_KN],
            dtype=pl.DT_FP4E2M1,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZN,
        ),
        addrs=[65536, 131072],
        mutex_ids=[20, 21],
    )
    kp1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[PAGED_KR, PAGED_KN],
            dtype=pl.DT_FP4E2M1,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZN,
        ),
        addrs=[69632, 135168],
        mutex_ids=[20, 21],
    )
    ksg = pl.make_tile_group(
        type=pl.TileType(
            shape=[4, PAGED_TILE_N],
            dtype=pl.DT_FP8E8M0,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NN,
        ),
        addrs=[196608, 196608 + 4 * PAGED_TILE_N],
        mutex_ids=[22, 23],
    )
    sp0 = pl.make_tile_group(
        type=pl.TileType(
            shape=[4, 128],
            dtype=pl.DT_FP8E8M0,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NN,
        ),
        addrs=[196608, 197632],
        mutex_ids=[22, 23],
    )
    sp1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[4, 128],
            dtype=pl.DT_FP8E8M0,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NN,
        ),
        addrs=[197120, 198144],
        mutex_ids=[22, 23],
    )
    core = pl.get_block_idx() // pl.get_subblock_num()
    sync_id = 0
    with pl.section_vector():
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PAGED_BACKWARD_IDS[0])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PAGED_BACKWARD_IDS[1])
    core_record = 1 + 3 * rows + 2 * vector_workers + 2 * core
    task_begin = pl.getval(plan, core_record)
    task_end = pl.getval(plan, core_record + 1)
    for item in pl.range(task_begin, task_end):
        record = 1 + 3 * rows + 2 * vector_workers + 2 * pl.get_block_num() + 8 * item
        shard = pl.getval(plan, record + 4)
        b = pl.getval(plan, record + 5)
        qbase = pl.getval(plan, record)
        qvalid = pl.getval(plan, record + 1)
        kl = pl.getval(plan, record + 2)
        sub = pl.get_subblock_idx()
        visible0 = 0
        visible1 = 0
        visible2 = 0
        if qvalid > sub * 3:
            visible0 = _row_visible(plan, qbase + sub * 3, cmp_ratio)
        if qvalid > sub * 3 + 1:
            visible1 = _row_visible(plan, qbase + sub * 3 + 1, cmp_ratio)
        if qvalid > sub * 3 + 2:
            visible2 = _row_visible(plan, qbase + sub * 3 + 2, cmp_ratio)
        key_begin = pl.getval(plan, record + 6)
        steps = pl.getval(plan, record + 7)
        for t in pl.range(steps):
            kb = key_begin + t * PAGED_TILE_N
            with pl.section_cube():
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.FIX, event_id=PAGED_BACKWARD_IDS[sync_id % 2]
                )
                _paged_cube_step(
                    t, qbase, qvalid, b, kb, kl, q, k, qs,
                    ks, table, qg, kg, qsg, ksg, la, lb, sa,
                    sb, cg, cgv, vg, kp0, kp1, sp0, sp1,
                    k_page_stride, ks_page_stride
                )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.FIX, event_id=PAGED_FORWARD_IDS[sync_id % 2]
                )
            with pl.section_vector():
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.V, event_id=PAGED_FORWARD_IDS[sync_id % 2]
                )
                _paged_vector_step(
                    weights, out_idx, out_val,
                    (wg, wbg, og, vg, hg, hhg, hlg, bg, ig,
                     ng, tg, xg, oig, ovg),
                    (t, qbase, qvalid, kb, shard, steps,
                     visible0, visible1, visible2, shard_width, key_begin),
                )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.V, event_id=PAGED_BACKWARD_IDS[sync_id % 2]
                )
            sync_id = sync_id + 1
    with pl.section_cube():
        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=PAGED_BACKWARD_IDS[0])
        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=PAGED_BACKWARD_IDS[1])
    with pl.section_cube():
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
    with pl.section_vector():
        pl.system.bar_all()
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        cache = og.current()
        sk = pl.reinterpret(cache, shape=[1, merge_width])
        val = pl.reinterpret(cache, dtype=BF, shape=[1, merge_width])
        shared = vg[0]
        idx = pl.reinterpret(shared, dtype=I32, shape=[1, merge_width])
        ht = hhg.current()
        lt = hlg.current()
        bt = bg.current()
        it = ig.current()
        nt = ng.current()
        tt = tg.current()
        xt = xg.current()
        oi = oig.current()
        ov = ovg.current()
        row_begin, row_end = _vector_row_bounds(plan, rows)
        for row in pl.range(row_begin, row_end):
            pl.load(val, out_val, [row, 0])
            pl.load(idx, out_idx, [row, 0])
            _selection_keys_from_scores(val, sk, merge_width, merge_width)
            _selection_histograms_high_vf(ht, sk, merge_width // 256, 0)
            _selection_find_high_bin_and_histograms_low(
                lt, it, ht, sk, bt, nt, merge_width - PAGED_TOPK + 1, merge_width // 256, 0
            )
            _selection_find_kth_vf(tt, lt, it, bt, nt)
            _selection_collect_indices_gt_eq_vf(xt, sk, tt, merge_width // 128, 0)
            row_offset = 0
            if (flags & 4):
                row_offset = pl.getval(offsets, row)
            _cache_select_emit_cached(
                xt,
                sk,
                oi,
                ov,
                PAGED_TOPK,
                0,
                0,
                True,
                idx,
                False,
                row_offset,
                ReturnValues,
            )
            pl.store(final_idx, oi, [row, 0])
            if ReturnValues:
                pl.store(final_val, ov, [row, 0])


# ============================================================================
# Entry

@pl.jit(arch="3510", auto_mutex=True, tiling_key=QliKey)
def gym_quant_lightning_indexer_kernel(
    q: pl.Ptr[pl.DT_UINT8], k: pl.Ptr[pl.DT_UINT8], w: pl.Ptr[pl.DT_UINT8],
    qs: pl.Ptr[pl.DT_UINT8], ks: pl.Ptr[pl.DT_UINT8],
    cu_q: pl.Ptr[pl.DT_UINT8], cu_k: pl.Ptr[pl.DT_UINT8],
    used_q: pl.Ptr[pl.DT_UINT8], used_k: pl.Ptr[pl.DT_UINT8],
    residual: pl.Ptr[pl.DT_UINT8], table: pl.Ptr[pl.DT_UINT8], offset: pl.Ptr[pl.DT_UINT8],
    plan: pl.Ptr[pl.DT_UINT8], indices: pl.Ptr[pl.DT_UINT8], values: pl.Ptr[pl.DT_UINT8],
    candidates: pl.Ptr[pl.DT_UINT8], lengths: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8], tiling: QliTiling,
):
    rows = tiling.rows
    batches = tiling.batches
    max_q = tiling.max_q
    max_k = tiling.max_k
    page_size = tiling.page_size
    k_rows = tiling.k_rows
    s_rows = tiling.s_rows
    k_page_stride = tiling.k_page_stride
    ks_page_stride = tiling.ks_page_stride
    k_stride = tiling.k_stride
    ks_stride = tiling.ks_stride
    table_cols = tiling.table_cols
    cube = tiling.cube
    vector_workers = tiling.vector
    plan_elems = tiling.plan_elems
    capacity = tiling.capacity
    flags = tiling.flags
    cmp_ratio = tiling.cmp_ratio
    topk = tiling.topk
    q = pl.make_tensor(pl.make_ptr(q, F4), [rows * 32, 128], [128, 1], dtype=F4)
    k = pl.make_tensor(pl.make_ptr(k, F4), [k_rows, 128], [k_stride * 2, 1], dtype=F4)
    weights = pl.make_tensor(pl.make_ptr(w, F32), [rows, 32], [32, 1], dtype=F32)
    qs = pl.make_tensor(pl.make_ptr(qs, E8), [rows * 32, 2, 2], [4, 2, 1], dtype=E8)
    ks = pl.make_tensor(pl.make_ptr(ks, E8), [s_rows, 2, 2], [ks_stride, 2, 1], dtype=E8)
    plan = pl.make_tensor(pl.make_ptr(plan, I32), [plan_elems], [1], dtype=I32)
    table = pl.make_tensor(pl.make_ptr(table, I32), [batches, table_cols], [table_cols, 1], dtype=I32)
    offsets = pl.make_tensor(pl.make_ptr(offset, I32), [rows, 1], [1, 1], dtype=I32)
    indices = pl.make_tensor(pl.make_ptr(indices, I32), [rows, 1, topk], [topk, topk, 1], dtype=I32)
    values = pl.make_tensor(pl.make_ptr(values, BF), [rows, 1, topk], [topk, topk, 1], dtype=BF)
    candidates = pl.make_tensor(
        pl.make_ptr(candidates, I32),
        [rows, 1, pl.max(1, capacity)], [pl.max(1, capacity), pl.max(1, capacity), 1], dtype=I32)
    lengths = pl.make_tensor(pl.make_ptr(lengths, I32), [rows, 1], [1, 1], dtype=I32)
    scores = pl.make_tensor(
        pl.make_ptr(pl.addptr(workspace, tiling.ws_scores), BF),
        [rows, tiling.score_cols], [tiling.score_cols, 1], dtype=BF)
    index_map = pl.make_tensor(
        pl.make_ptr(pl.addptr(workspace, tiling.ws_index_map), I32),
        [rows, tiling.score_cols], [tiling.score_cols, 1], dtype=I32)
    partial_values = pl.make_tensor(
        pl.make_ptr(pl.addptr(workspace, tiling.ws_partial), BF),
        [rows, tiling.aux_cols], [tiling.aux_cols, 1], dtype=BF)
    if Operation == 0:
        all_empty = pl.const(0, pl.DT_INT64)
        if flags & 32:
            all_empty = pl.getval(plan, 0)
        if all_empty:
            _empty_empty_outputs(plan, indices, values, candidates, lengths, capacity, flags)
        else:
            _generic_score_generic(
                q, k, weights, qs, ks, plan, table, scores, max_q, batches,
                offsets, indices, values, candidates, lengths, k_page_stride,
                ks_page_stride, k_stride, ks_stride, capacity, vector_workers,
             flags, page_size, cmp_ratio,)
            if not SelectWidth or (flags & 16):
                _generic_finish(
                    scores, plan, partial_values, offsets, indices, values,
                    candidates, lengths, capacity,
                 flags, cmp_ratio,)
    else:
        flat_indices = pl.make_tensor(indices, [indices.shape[0], 512])
        flat_values = pl.make_tensor(values, [values.shape[0], 512])
        _paged_score_paged(
            q, k, weights, qs, ks, plan, table, index_map, partial_values,
            flat_indices, flat_values, offsets, max_q, batches,
            k_page_stride, ks_page_stride, vector_workers,
         flags, cmp_ratio,)


@lru_cache(maxsize=1)
def _load_metadata_kernel():
    path = Path(__file__).resolve().parents[1] / "op_kernel_aicpu" / "quant_lightning_indexer_metadata.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.quant_lightning_indexer_metadata


def _require_plan_tensor(name, tensor, shape, device):
    """Reject an optional int32 plan tensor that is mis-shaped or strided."""
    if tensor is None:
        return
    meta = (tuple(tensor.shape), tensor.dtype, tensor.device)
    if meta != (shape, torch.int32, device) or not tensor.is_contiguous():
        raise ValueError(f"invalid {name}")


def _require_metadata(metadata, device):
    """metadata is a fixed 1024-entry int32 scratch tensor when supplied."""
    if metadata is None:
        return
    meta = (metadata.shape, metadata.dtype, metadata.device)
    if meta != ((1024,), torch.int32, device) or not metadata.is_contiguous():
        raise ValueError("metadata must be contiguous int32[1024] on the query device")


def _require_page_strides(k, descale_k, page):
    """Both page-major buffers must keep every page row whole and disjoint."""
    if k.stride(0) < page * 64 or k.stride(0) % 64:
        raise ValueError("page strides must preserve complete, non-overlapping rows")
    if descale_k.stride(0) < page * 4 or descale_k.stride(0) % 4:
        raise ValueError("page strides must preserve complete, non-overlapping rows")


def _require_block_table(block_table, batches, device):
    """block_table maps every batch row onto physical pages when supplied."""
    if block_table is None:
        return
    if block_table.ndim != 2 or block_table.shape[1] <= 0:
        raise ValueError("invalid block_table")
    valid_shape_dtype = block_table.shape[0] == batches and block_table.dtype == torch.int32
    if not valid_shape_dtype or block_table.device != device or not block_table.is_contiguous():
        raise ValueError("invalid block_table")


class _Request(NamedTuple):
    """Every tensor argument of one call, packed so the host helpers stay narrow."""

    q: torch.Tensor
    k: torch.Tensor
    w: torch.Tensor
    descale_q: torch.Tensor
    descale_k: torch.Tensor
    cu_q: Optional[torch.Tensor]
    cu_k: Optional[torch.Tensor]
    used_q: Optional[torch.Tensor]
    used_k: Optional[torch.Tensor]
    residual: Optional[torch.Tensor]
    table: Optional[torch.Tensor]
    offset: Optional[torch.Tensor]
    metadata: Optional[torch.Tensor]


class _Modes(NamedTuple):
    """The scalar arguments that pick a kernel body and bound the shapes."""

    topk: int
    mask_mode: int
    cmp_ratio: int
    candidates: int
    block_size: int
    quant_mode: int
    layout_q: str
    layout_k: str
    return_value: bool
    max_seqlen_q: int


class _KeyView(NamedTuple):
    """Key storage viewed as dense rows, plus the strides the kernel needs."""

    kv: torch.Tensor
    sv: torch.Tensor
    max_k: int
    page: int
    kp: int
    ksp: int
    k_stride: int
    s_stride: int
    tile_n: int
    paged: bool


class _Tiling(NamedTuple):
    """Static capacities, core counts and the fused compute specialization."""

    rows: int
    batches: int
    max_q: int
    parts: int
    split: int
    shard_width: int
    select_width: int
    fused: int
    count: int
    fast: bool
    cube: int
    vector: int
    tile_n: int
    page: int
    topk: int
    return_value: bool
    allow_empty: bool


class _Stage(NamedTuple):
    """The tensors one native launch reads and writes."""

    plan: torch.Tensor
    table: torch.Tensor
    dummy: torch.Tensor
    indices: torch.Tensor
    values: torch.Tensor
    candidates: torch.Tensor
    lengths: torch.Tensor
    offsets: torch.Tensor
    candidate_arg: torch.Tensor
    length_arg: torch.Tensor


def _check_modes(modes):
    """The scalar contract: one quantization mode, one query layout, valid sizes."""
    if (
        modes.quant_mode != 1
        or modes.layout_q != "TND"
        or modes.layout_k not in ("TND", "PA_BBND")
    ):
        raise ValueError("expected quant_mode=1, Q=TND and K=TND/PA_BBND")
    if (
        not 1 <= modes.topk <= 8192
        or modes.mask_mode not in (0, 3)
        or not 1 <= modes.cmp_ratio <= 128
    ):
        raise ValueError("invalid topk, mask_mode or cmp_ratio")
    if modes.candidates == -1:
        if modes.block_size != -1:
            raise ValueError("disabled candidates require block_size=-1")
    elif modes.candidates <= 0 or modes.block_size != 8:
        raise ValueError("candidates require positive capacity and block_size=8")


def _check_storage(request):
    """Every input buffer is dense and lives on the query device."""
    import torch_npu

    device = request.q.device
    # The AICPU and AI Core entries read raw pointers with linear strides.
    # torch.is_contiguous() also accepts logically contiguous FRACTAL_NZ.
    for name, tensor in zip(request._fields, request):
        if tensor is None:
            continue
        if tensor.device != device or torch_npu.get_npu_format(tensor) not in (0, 2, 30):
            raise ValueError(f"{name} requires ND, NCHW or NCDHW storage on the query device")
    for name, tensor in [
        ("q", request.q),
        ("w", request.w),
        ("descale_q", request.descale_q),
    ]:
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous on the query device")
    for name, tensor in [("k", request.k), ("descale_k", request.descale_k)]:
        if tensor.dtype != torch.uint8:
            raise ValueError(f"{name} must use uint8 on the query device")


def _plan_tensors(request, rows, batches):
    """The optional plan tensors are contiguous int32 arrays of a fixed shape."""
    device = request.q.device
    for name, tensor, shape in [
        ("cu_seqlens_q", request.cu_q, (batches + 1,)),
        ("cu_seqlens_k", request.cu_k, (batches + 1,)),
        ("seqused_q", request.used_q, (batches,)),
        ("seqused_k", request.used_k, (batches,)),
        ("cmp_residual_k", request.residual, (batches,)),
        ("output_idx_offset", request.offset, (rows, 1)),
    ]:
        _require_plan_tensor(name, tensor, shape, device)
    _require_metadata(request.metadata, device)


def _check_request(request, modes):
    """Reject anything the kernel body cannot represent, before allocating."""
    _check_modes(modes)
    if (
        request.q.ndim != 3
        or tuple(request.q.shape[1:]) != (32, 64)
        or request.q.dtype != torch.uint8
    ):
        raise ValueError("q must be packed uint8 [T,32,64]")
    rows = request.q.shape[0]
    if request.q.device.type != "npu" or request.k.ndim == 0 or request.k.shape[0] <= 0:
        raise ValueError("NPU tensors and nonempty key storage are required")
    batches = request.cu_q.numel() - 1 if request.cu_q is not None else 1
    if rows <= 0 or batches <= 0:
        raise ValueError("positive query and batch dimensions are required")
    if tuple(request.w.shape) != (rows, 32) or request.w.dtype != torch.float32:
        raise ValueError("w must be float32 [T,32]")
    if (
        tuple(request.descale_q.shape) != (rows, 32, 2, 2)
        or request.descale_q.dtype != torch.uint8
    ):
        raise ValueError("descale_q must be uint8 [T,32,2,2]")
    _check_storage(request)
    _plan_tensors(request, rows, batches)
    if (modes.mask_mode == 3 and modes.cmp_ratio != 1) != (
        request.residual is not None
    ):
        raise ValueError(
            "compression residual is required exactly for compressed causal mode"
        )
    return rows, batches


def _paged_key_view(request, batches):
    """View page-major storage as dense key rows; only the page axis may pad."""
    k, scales = request.k, request.descale_k
    if (
        k.ndim != 4
        or tuple(k.shape[2:]) != (1, 64)
        or tuple(scales.shape) != (k.shape[0], k.shape[1], 1, 2, 2)
    ):
        raise ValueError("invalid PA key or scale shape")
    page = k.shape[1]
    if not 0 < page <= 1024 or page % 16:
        raise ValueError("page size must be a multiple of 16, at most 1024")
    if k.stride()[1:] != (64, 64, 1) or scales.stride()[1:] != (4, 4, 2, 1):
        raise ValueError("only the page axis may have storage padding")
    _require_page_strides(k, scales, page)
    _require_block_table(request.table, batches, request.q.device)
    if request.table is None and batches != 1:
        raise ValueError("multiple PA batches require block_table")
    kp = k.stride(0) // 64
    ksp = scales.stride(0) // 4
    rows = (k.shape[0] - 1) * kp + page
    srows = (k.shape[0] - 1) * ksp + page
    width = request.table.shape[1] if request.table is not None else k.shape[0]
    return _KeyView(
        kv=k.as_strided((rows, 64), (64, 1)),
        sv=scales.as_strided((srows, 2, 2), (4, 2, 1)),
        max_k=width * page,
        page=page,
        kp=kp,
        ksp=ksp,
        k_stride=64,
        s_stride=4,
        tile_n=gcd(page, 128),
        paged=True,
    )


def _flat_key_view(request, batches):
    """View token-major storage as dense key rows; only the token axis may pad."""
    k, scales = request.k, request.descale_k
    if (
        k.ndim != 3
        or tuple(k.shape[1:]) != (1, 64)
        or tuple(scales.shape) != (k.shape[0], 1, 2, 2)
    ):
        raise ValueError("invalid TND key or scale shape")
    if request.table is not None or (batches > 1 and request.cu_k is None):
        raise ValueError("TND uses K prefixes, without a page table")
    if k.stride()[1:] != (64, 1) or scales.stride()[1:] != (4, 2, 1):
        raise ValueError("only the token axis may have storage padding")
    k_stride, s_stride = k.stride(0), scales.stride(0)
    if k_stride < 64 or s_stride < 4:
        raise ValueError("token rows must not overlap")
    rows = k.shape[0]
    return _KeyView(
        kv=k.as_strided((rows, 64), (k_stride, 1)),
        sv=scales.as_strided((rows, 2, 2), (s_stride, 2, 1)),
        max_k=rows,
        page=128,
        kp=0,
        ksp=0,
        k_stride=k_stride,
        s_stride=s_stride,
        tile_n=128,
        paged=False,
    )


def _key_view(request, batches, modes):
    """Resolve the key storage of the requested layout."""
    if modes.layout_k == "PA_BBND":
        return _paged_key_view(request, batches)
    return _flat_key_view(request, batches)


def _tiling(request, view, modes, rows, batches):
    """Resolve static capacities, core counts and the fused tiling key."""
    max_q = rows if modes.max_seqlen_q <= 0 else modes.max_seqlen_q
    split = 8
    shard_tiles = (view.max_k + 256 * split - 1) // (256 * split)
    shard_width = shard_tiles * 256
    parts = ((view.max_k + 255) // 256 + shard_tiles - 1) // shard_tiles
    fast = (
        view.paged
        and view.page == 128
        and 8192 <= view.max_k <= 131072
        and view.max_k % 256 == 0
        and modes.topk == 512
        and modes.candidates == -1
        and request.table is not None
        and request.cu_q is not None
        and request.used_q is not None
        and request.used_k is not None
        and parts * 512 <= shard_width
    )
    select_width = (
        (view.max_k + 255) // 256 * 256 if not fast and view.max_k <= 16384 else 0
    )
    count = modes.candidates
    fused = count if select_width and 0 < count <= 8192 else 0
    properties = torch.npu.get_device_properties(request.q.device.index)
    groups = batches * ((max_q + 5) // 6)
    if fast and view.max_k == 131072 and properties.cube_core_num == 28:
        # Use the finer compiled geometry when it cuts the longest worker's
        # tile count by at least 20%, leaving room for the wider final merge.
        coarse = ((groups * split + 27) // 28) * shard_tiles
        fine_tiles = (view.max_k + 14 * 256 - 1) // (14 * 256)
        fine = ((groups * 14 + 27) // 28) * fine_tiles
        if fine * 5 <= coarse * 4:
            split = parts = 14
            shard_width = fine_tiles * 256
    # Bound the launch grid by static score/postprocessing capacities. AICPU
    # still owns every actual sequence boundary and per-core task assignment.
    plan_split = parts if fast else min(7, (view.max_k + view.tile_n - 1) // view.tile_n)
    task_capacity = groups * plan_split
    vectors_per_cube = properties.vector_core_num // properties.cube_core_num
    cube = min(properties.cube_core_num,
               max(task_capacity, (rows + vectors_per_cube - 1) // vectors_per_cube))
    tiling = _Tiling(
        rows=rows, batches=batches, max_q=max_q, parts=parts, split=split,
        shard_width=shard_width, select_width=select_width, fused=fused,
        count=count, fast=fast, cube=cube,
        vector=cube * vectors_per_cube, tile_n=128 if fast else view.tile_n,
        page=128 if fast else view.page, topk=modes.topk,
        return_value=modes.return_value, allow_empty=bool(select_width) and (count < 0 or fused > 0),
    )
    return tiling


def _stage_tensors(request, view, tiling, plan):
    """Allocate the outputs and the scratch the native entry reads and writes."""
    device = request.q.device
    count = tiling.count
    dummy = torch.empty((1,), dtype=torch.int32, device=device)
    indices = torch.empty(
        (tiling.rows, 1, tiling.topk), dtype=torch.int32, device=device
    )
    values = torch.empty(indices.shape, dtype=torch.bfloat16, device=device)
    candidates = torch.empty(
        (tiling.rows, 1, count) if count > 0 else (0,),
        dtype=torch.int32,
        device=device,
    )
    lengths = torch.empty(
        (tiling.rows, 1) if count > 0 else (0,), dtype=torch.int32, device=device
    )
    return _Stage(
        plan=plan,
        table=request.table if request.table is not None else dummy.view(1, 1),
        dummy=dummy,
        indices=indices,
        values=values,
        candidates=candidates,
        lengths=lengths,
        offsets=request.offset if request.offset is not None else dummy.view(1, 1),
        candidate_arg=candidates if count > 0 else dummy.view(1, 1, 1),
        length_arg=lengths if count > 0 else dummy.view(1, 1),
    )


def _flags(request, view, tiling):
    return (int(view.paged and not tiling.fast)
            | (int(request.table is not None) << 1)
            | (int(request.offset is not None) << 2)
            | (int(tiling.fused > 0) << 3)
            | (int(tiling.count > 0 and not tiling.fused) << 4)
            | (int(tiling.allow_empty) << 5)
            | (int(view.page == tiling.tile_n) << 6))


def _offline_score_key(modes, tiling):
    select = next((v for v in (256, 512, 1024, 1536, 2048, 4096, 8192, 16384)
                   if v >= tiling.select_width), 0) if tiling.select_width else 0
    shard = (tiling.shard_width if tiling.parts == 14 else
             next((v for v in (4096, 8192, 16384) if v >= tiling.shard_width), 16384))
    return dict(Operation=int(tiling.fast), TileN=tiling.tile_n,
                SelectWidth=select, ShardCapacity=shard if tiling.fast else 4096,
                ReturnValues=int(modes.return_value),
                FixedTopK=512 if tiling.select_width and modes.topk == 512 and modes.candidates <= 512 else 0,
                MaskMode=modes.mask_mode,
                RatioMode=2 if modes.mask_mode == 3 and modes.cmp_ratio == 2 else 0)


def _native_device_plan(request, view, tiling):
    """Ask the offline AICPU source to derive the packed schedule."""
    metadata_kernel = _load_metadata_kernel()

    split = (
        tiling.parts if tiling.fast
        else min(7, (view.max_k + view.tile_n - 1) // view.tile_n)
    )
    items = tiling.batches * ((tiling.max_q + 5) // 6) * split
    plan = torch.empty(
        (1 + 3 * tiling.rows + 2 * tiling.vector + 2 * tiling.cube + 8 * items,),
        dtype=torch.int32, device=request.q.device
    )
    dummy = torch.empty((1,), dtype=torch.int32, device=request.q.device)
    flags = (
        int(tiling.fast) | (int(request.cu_q is not None) << 1)
        | (int(request.cu_k is not None) << 2) | (int(request.used_q is not None) << 3)
        | (int(request.used_k is not None) << 4) | (int(request.residual is not None) << 5)
        | (int(tiling.allow_empty) << 6)
    )
    schedule = (split | ((256 if tiling.fast else view.tile_n) << 8)
                | (tiling.cube << 24) | (tiling.vector << 32) | (flags << 40))
    metadata_kernel(
        request.cu_q if request.cu_q is not None else dummy,
        request.cu_k if request.cu_k is not None else dummy,
        request.used_q if request.used_q is not None else dummy,
        request.used_k if request.used_k is not None else dummy,
        request.residual if request.residual is not None else dummy,
        plan, tiling.rows, tiling.max_q, view.max_k, schedule,
    )
    return plan


def _workspace_and_tiling(request, view, modes, tiling, stage):
    def align(size):
        return (size + 511) // 512 * 512

    cols = tiling.parts * 512 if tiling.fast else view.max_k
    index_off = align(tiling.rows * cols * 2)
    aux_off = index_off + (align(tiling.rows * cols * 4) if tiling.fast else 512)
    aux_cols = cols if tiling.fast else (view.max_k + 7) // 8
    end = aux_off + (align(tiling.rows * aux_cols * 2) if tiling.count > 0 and not tiling.fused else 512)
    if tiling.fast:
        aux_off = 0
    workspace = torch.empty((end,), dtype=torch.uint8, device=request.q.device)
    td = QliTiling(tiling.rows, tiling.batches, tiling.max_q, view.max_k,
        view.page, view.kv.shape[0], view.sv.shape[0], view.kp, view.ksp,
        view.k_stride, view.s_stride, stage.table.shape[1], tiling.cube, tiling.vector,
        stage.plan.numel(), tiling.count, _flags(request, view, tiling), modes.cmp_ratio,
        0, index_off, aux_off, cols, aux_cols, modes.topk)
    # The offline tiling ABI is a sequence of int64 fields in declaration order.
    # Pass the device buffer directly so a warm JIT launch need not re-inspect the schema.
    packed_tiling = torch.tensor(astuple(td), dtype=torch.int64, device="cpu").view(torch.uint8).to(request.q.device)
    return workspace, packed_tiling


def _launch_native_jit(request, view, modes, tiling, stage):
    """Exercise the offline kernel with its packed workspace and tiling ABI."""

    workspace, packed_tiling = _workspace_and_tiling(request, view, modes, tiling, stage)
    dummy = stage.dummy
    score_key = _offline_score_key(modes, tiling)
    gym_quant_lightning_indexer_kernel[None, tiling.cube, score_key](
        request.q, request.k, request.w, request.descale_q, request.descale_k,
        request.cu_q if request.cu_q is not None else dummy,
        request.cu_k if request.cu_k is not None else dummy,
        request.used_q if request.used_q is not None else dummy,
        request.used_k if request.used_k is not None else dummy,
        request.residual if request.residual is not None else dummy,
        stage.table, stage.offsets, stage.plan,
        stage.indices, stage.values, stage.candidate_arg, stage.length_arg, workspace, packed_tiling)
    return (stage.indices, stage.values if modes.return_value else stage.values.reshape(-1)[:0],
            stage.candidates, stage.lengths)


def quant_lightning_indexer(
    q, k, w, descale_q, descale_k,
    cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None, seqused_k=None,
    cmp_residual_k=None, block_table=None, output_idx_offset=None, metadata=None,
    *, topk, quant_mode, max_seqlen_q=-1, mask_mode=0, cmp_ratio=1,
    layout_q="TND", layout_k="TND", return_value=False,
    candidate_topk_blocks=-1, candidate_block_size=-1,
):
    """Build the device plan and launch the JIT compute kernel on the query device."""

    request = _Request(
        q=q, k=k, w=w, descale_q=descale_q, descale_k=descale_k,
        cu_q=cu_seqlens_q, cu_k=cu_seqlens_k, used_q=seqused_q,
        used_k=seqused_k, residual=cmp_residual_k, table=block_table,
        offset=output_idx_offset, metadata=metadata,
    )
    modes = _Modes(
        topk=topk, mask_mode=mask_mode, cmp_ratio=cmp_ratio,
        candidates=candidate_topk_blocks, block_size=candidate_block_size,
        quant_mode=quant_mode, layout_q=layout_q, layout_k=layout_k,
        return_value=return_value, max_seqlen_q=max_seqlen_q,
    )
    rows, batches = _check_request(request, modes)
    view = _key_view(request, batches, modes)
    tiling = _tiling(request, view, modes, rows, batches)
    with torch.npu.device(q.device):
        plan = _native_device_plan(request, view, tiling)
        stage = _stage_tensors(request, view, tiling, plan)
        return _launch_native_jit(request, view, modes, tiling, stage)
