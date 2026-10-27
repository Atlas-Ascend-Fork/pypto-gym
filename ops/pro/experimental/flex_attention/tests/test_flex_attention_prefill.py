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

"""Prefill FlashAttention for the 950 PR hybrid-mask operator.

Targets ``attention_with_kvcache_prefill_bf16_hybrid_mask``: bf16, GQA, right-
down aligned, and driven by the operator's own ``mm_prefix_range`` spans. It
keeps the NBuffer + auto_mutex Cube/Vector pipeline of test_flex_attention.py
but **regenerates the mask from scratch** rather than reusing that kernel's
per-row ``maskr`` / hole encoding. That sibling file remains the general flex
mask (arbitrary holes, sliding window); this one implements the operator.

The mask contract comes straight from the operator's signature -- the only mask
input is ``mm_prefix_range [batch, max_spans, 2]``, so everything else is
derived on device. For query row r at absolute position
``q_abs = skv - sq + r``::

    bound = q_abs + 1                                  # RIGHT_DOWN causal
    for each valid span [s, e]:                        # -1 marks padding
        if s <= q_abs <= e:  bound = max(bound, e + 1) # bidirectional in span
    visible keys = [0, min(bound, skv))

Two properties of that rule shape the whole design:

  * **It is a pure prefix.** ``bound`` only grows and the lower edge is pinned
    at 0, so the visible KV blocks of a Q tile are always ``0..kv_loop-1`` with
    no gaps. There is nothing to skip in the middle, hence no block-sparse
    bitmaps -- the general kernel needs those only because a sliding window
    punches holes.
  * **It is monotone in the row.** For q1 < q2, a span covering q1 either also
    covers q2 or ends before it (in which case q2 + 1 already exceeds e + 1).
    So a Q tile is fully characterized by two scalar bounds: bound(first row)
    decides which blocks are fully visible, bound(last row) decides how many
    blocks to walk. Both are computed by ``scalar_bound`` on device.

So the host ships no mask metadata at all: no maskr, no holes, no tile_range,
no sparse bitmaps. ``compute_row_bounds_vf`` expands the spans into per-row
bounds in one 64-lane register and ``decode_prefix_mask`` turns those into the
element mask with a single compare.

Also relative to the fp16 kernel:

  * **bf16**: q/k/v/o and the L1/L0/P tiles are DT_BF16. Accumulation stays
    fp32 (Acc tiles, softmax); only the matmul operand type and the P/output
    casts change. bf16 has an 8-bit mantissa vs fp16's 11, so the tolerance is
    ~8x looser (see ATOL_BF16).
  * **GQA**: with ``group = n_head_q // n_head_kv``, q head ``n_idx`` reads kv
    head ``n_idx // group``. group == 1 is MHA.

K/V are paged exactly as the operator pages them: the caches are
``[num_blocks, block_size, n_head_kv, dim]`` (NHD -- the inner three axes are
tokens-in-page, kv heads, head dim) addressed through ``block_ids``, with
``block_size == TKV`` so one KV tile is one page.

Sequence metadata follows the operator verbatim, and both halves are strict:
``cu_seqlens_q`` is ``[B+1]`` and **its first element must be 0**, because
batch b's Q rows start at ``cu_seqlens_q[b]``; ``seqlens_kvcache`` is ``[B]``
*absolute* per-request KV lengths, not a prefix sum.

Usage:
    python3 python/tests/st/pypto_pro/frontend/fa/test_flex_attention_tx.py
"""

import itertools
import logging
import math
import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# ================================================================
#  Configuration
# ================================================================
QK_PRELOAD = 3
FIFO_SIZE = QK_PRELOAD + 1

# ================================================================
#  Tile dimensions and constants
# ================================================================
TS = 128
TKV = 128
TD = 128
TS_HALF = TS // 2
NEG_INF = -1e9
SCALE = 1.0 / math.sqrt(TD)

# Span budget of the 950 PR hybrid mask (HYBRID_MAX_SPANS in the operator).
# mm_prefix_range is always padded out to this width so the kernel's span loop
# is a statically unrolled, fixed-length walk over inert -1 padding slots.
HYBRID_MAX_SPANS = 16

# UB row width, in uint32, used to stage one span per row. Only the first two
# columns hold data (start, end); the row is padded out to 32 bytes because a
# Vec tile cannot have rows narrower than a block -- a [max_spans, 2] tile fails
# to instantiate outright. Widening the row keeps mm_prefix_range in GM at the
# operator's exact [B, max_spans, 2] shape instead of reshaping it to suit us.
SPAN_ROW = 8

# PagedAttention page size. One KV tile maps to exactly one page, which is what
# lets the page table be consulted once per (Q tile, KV block) and keeps the
# load itself unchanged. A block_size != TKV would make a tile span or share
# pages; the operator only validates 128.
# Default page size for the test helpers. The kernel itself reads the real one
# from k.shape[1] and supports any multiple of TKV: a page then holds
# PAGE_TILES = block_size / TKV KV tiles, so tile ki sits in page ki//PAGE_TILES
# at token offset (ki%PAGE_TILES)*TKV. Only block_size < TKV is unsupported --
# a tile would then span several pages that need not be physically adjacent,
# so it would take multiple loads stitched together in L1.
BLOCK_SIZE = TKV

# bf16 keeps 8 mantissa bits against fp16's 11, so the same FA flow lands ~8x
# coarser. The reference models the kernel's bf16 P quantization, so what is
# left is the final fp32->bf16 output cast plus matmul accumulation order.
ATOL_BF16 = 8e-3

BLOCK_STRIDE_ND = TS >> 1 | 0x1
REPEAT_STRIDE_ND = 1
FLOAT_REP_SIZE = 64  # elements per fp32 register
D_LOOPS = TD // FLOAT_REP_SIZE
TAIL_D = TD % FLOAT_REP_SIZE
REDUCE_SIZE = 1

# Buffer sizes (bytes). 16-bit operands: identical footprint for fp16 and bf16.
Q_B16 = TS * TD * 2
KT_B16 = TD * TKV * 2
V_B16 = TKV * TD * 2
P_B16 = TS * TKV * 2
QK_HALF_F32 = TS * TKV * 4
PV_HALF_F32 = TS * TD * 4

# VEC buffer sizes
VB4_KV = TS_HALF * TKV * 4
VB2_KV = TS_HALF * TKV * 2
VB4 = TS_HALF * TD * 4
VB2 = TS_HALF * TD * 2
VB6 = (TS_HALF + 1) * TKV * 2
VB_RED = TS_HALF * 1 * 4
VB_MASK = TS_HALF * TKV
VB_SPANS = HYBRID_MAX_SPANS * SPAN_ROW * 4

# ================================================================
#  Buffer addresses
# ================================================================
# MAT (512KB) - L1 buffers
MA0_Q = 0
MA1_K = Q_B16 * 2
MA2_P = MA1_K + KT_B16 * 2
MA3_V = MA2_P + P_B16 * 3

# L0A/L0B/L0C addresses
LA0 = 0
LA1 = P_B16
RA0 = 0
RA1 = KT_B16
CA0 = 0
CA1 = QK_HALF_F32

# VEC (248KB) addresses
VA0 = 0
VA1 = VA0 + VB4_KV * 2
VA_GMAX0 = VA1 + VB6 * 2
VA_GMAX1 = VA_GMAX0 + VB_RED
VA_GMAX2 = VA_GMAX1 + VB_RED
VA_GSUM0 = VA_GMAX2 + VB_RED
VA_GSUM1 = VA_GSUM0 + VB_RED
VA_GSUM2 = VA_GSUM1 + VB_RED
VA_EXPMAX0 = VA_GSUM2 + VB_RED
VA_EXPMAX1 = VA_EXPMAX0 + VB_RED
VA_EXPMAX2 = VA_EXPMAX1 + VB_RED
VA7 = VA_EXPMAX2 + VB_RED
VA8 = VA7 + VB4
VA9 = VA8 + VB4 * 2
VA10 = VA9 + VB2
VA11 = VA10 + 8192 * 2
VA12 = VA11 + VB_RED
VA13 = VA12 + VB_RED
# bound_db: 2 x [1, TS_HALF] uint32 -- the per-row bounds built on device.
VA14 = VA13 + VB_RED * 2
# spans_db: 2 x [HYBRID_MAX_SPANS, SPAN_ROW] uint32.
VA15 = VA14 + VB_SPANS * 2
assert VA15 <= 248 * 1024

# Cross-core shared buffer IDs
QK_READY_FORWARD_IDS = (0, 1)
QK_READY_BARKWARD_IDS = (2, 3)
P_READY_FORWARD_IDS = (4, 5, 6)
PV_READY_FORWARD_IDS = (7, 8)
PV_READY_BARKWARD_IDS = (9, 10)


@pl.vector_function
def process_vec1_nd_no_update_vf_unalign64(input_tile, dst_tile, max_tile, max_tile_st, sum_tile, mask_tile,
                                           s1_size, s2_size):
    preg_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(s2_size, dtype=pl.DT_FP32)
    preg_all_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x = vf.muls(vreg_x, SCALE, preg_tail)

        preg_mask = vf.load_align(mask_tile, m * TKV, dist=pl.LoadDist.DS)
        vreg_x = vf.select(vreg_min, vreg_x, preg_mask)
        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vreg_max = vf.reduce_max(vreg_x, preg_tail, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(max_tile, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(max_tile, ureg_max, 0, post_update=True)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)

    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(max_tile_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2 = vf.load_align(input_tile, i * TKV)
        vreg_exp_even = vf.exp_sub(vreg_x_2, vreg_max_2, preg_tail)

        vreg_exp_sum = vf.reduce_sum(vreg_exp_even, preg_tail, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(sum_tile, vreg_exp_sum, ureg_exp_sum, 1, post_update=True)

        vreg_exp_even_b16 = vf.astype(vreg_exp_even, preg_all_b16, layout=pl.CastLayout.ZERO, dtype=pl.DT_BF16)
        vreg_dst_even_b16, vreg_dst_odd_b16 = vf.de_interleave(vreg_exp_even_b16, vreg_exp_even_b16)
        vf.store_align(dst_tile, vreg_dst_even_b16, preg_all_b16,
                       block_stride=BLOCK_STRIDE_ND, repeat_stride=REPEAT_STRIDE_ND,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
    vf.store_unalign_post(sum_tile, ureg_exp_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_no_update_vf_unalign(input_tile, dst_tile, max_tile, max_tile_st, sum_tile, mask_tile,
                                         s1_size, s2_size):
    preg_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_mask_unroll = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(s2_size - 64, dtype=pl.DT_FP32)
    preg_all_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x_unroll = vf.load_align(input_tile, m * TKV + 64)

        vreg_x = vf.muls(vreg_x, SCALE, preg_all)
        vreg_x_unroll = vf.muls(vreg_x_unroll, SCALE, preg_tail)

        preg_mask = vf.load_align(mask_tile, m * TKV, dist=pl.LoadDist.DS)
        preg_mask_unroll = vf.load_align(mask_tile, m * TKV + 64, dist=pl.LoadDist.DS)
        vreg_x = vf.select(vreg_min, vreg_x, preg_mask)
        vreg_x_unroll = vf.select(vreg_min, vreg_x_unroll, preg_mask_unroll)
        vreg_x_unroll = vf.select(vreg_x_unroll, vreg_min, preg_tail)
        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vf.store_align(input_tile + m * TKV + 64, vreg_x_unroll, preg_all)
        vreg_max_tmp = vf.max(vreg_x, vreg_x_unroll, preg_all)
        vreg_max = vf.reduce_max(vreg_max_tmp, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(max_tile, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(max_tile, ureg_max, 0, post_update=True)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)

    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(max_tile_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2, vreg_x_unroll_2 = vf.load_align(input_tile, i * TKV, dist=pl.LoadDist.DINTLV_B32)
        vreg_exp_even = vf.exp_sub(vreg_x_2, vreg_max_2, preg_all)
        vreg_exp_odd = vf.exp_sub(vreg_x_unroll_2, vreg_max_2, preg_all)

        vreg_exp_sum = vf.add(vreg_exp_even, vreg_exp_odd, preg_all)
        vreg_exp_sum = vf.reduce_sum(vreg_exp_sum, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(sum_tile, vreg_exp_sum, ureg_exp_sum, 1, post_update=True)

        vreg_exp_even_b16 = vf.astype(vreg_exp_even, preg_all, layout=pl.CastLayout.ZERO, dtype=pl.DT_BF16)
        vreg_exp_odd_b16 = vf.astype(vreg_exp_odd, preg_all, layout=pl.CastLayout.ONE, dtype=pl.DT_BF16)
        vreg_exp_b16 = vf.or_(vreg_exp_even_b16, vreg_exp_odd_b16, preg_all_b16)
        vf.store_align(dst_tile, vreg_exp_b16, preg_all_b16,
                       block_stride=BLOCK_STRIDE_ND, repeat_stride=REPEAT_STRIDE_ND,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
    vf.store_unalign_post(sum_tile, ureg_exp_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_no_update_vf(input_tile, dst_tile, max_tile, max_tile_st, sum_tile, mask_tile, s1_size):
    preg_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_mask_unroll = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_all_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x_unroll = vf.load_align(input_tile, m * TKV + 64)

        vreg_x = vf.muls(vreg_x, SCALE, preg_all)
        vreg_x_unroll = vf.muls(vreg_x_unroll, SCALE, preg_all)

        preg_mask = vf.load_align(mask_tile, m * TKV, dist=pl.LoadDist.DS)
        preg_mask_unroll = vf.load_align(mask_tile, m * TKV + 64, dist=pl.LoadDist.DS)
        vreg_x = vf.select(vreg_min, vreg_x, preg_mask)
        vreg_x_unroll = vf.select(vreg_min, vreg_x_unroll, preg_mask_unroll)
        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vf.store_align(input_tile + m * TKV + 64, vreg_x_unroll, preg_all)
        vreg_max_tmp = vf.max(vreg_x, vreg_x_unroll, preg_all)
        vreg_max = vf.reduce_max(vreg_max_tmp, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(max_tile, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(max_tile, ureg_max, 0, post_update=True)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)

    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(max_tile_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2, vreg_x_unroll_2 = vf.load_align(input_tile, i * TKV, dist=pl.LoadDist.DINTLV_B32)
        vreg_exp_even = vf.exp_sub(vreg_x_2, vreg_max_2, preg_all)
        vreg_exp_odd = vf.exp_sub(vreg_x_unroll_2, vreg_max_2, preg_all)

        vreg_exp_sum = vf.add(vreg_exp_even, vreg_exp_odd, preg_all)
        vreg_exp_sum = vf.reduce_sum(vreg_exp_sum, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(sum_tile, vreg_exp_sum, ureg_exp_sum, 1, post_update=True)

        vreg_exp_even_b16 = vf.astype(vreg_exp_even, preg_all, layout=pl.CastLayout.ZERO, dtype=pl.DT_BF16)
        vreg_exp_odd_b16 = vf.astype(vreg_exp_odd, preg_all, layout=pl.CastLayout.ONE, dtype=pl.DT_BF16)
        vreg_exp_b16 = vf.or_(vreg_exp_even_b16, vreg_exp_odd_b16, preg_all_b16)
        vf.store_align(dst_tile, vreg_exp_b16, preg_all_b16,
                       block_stride=BLOCK_STRIDE_ND, repeat_stride=REPEAT_STRIDE_ND,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
    vf.store_unalign_post(sum_tile, ureg_exp_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_update_vf_unalign64(input_tile, dst_tile, max_tile, mask_tile,
                                        tmp_max, tmp_max_st, tmp_exp_sum, s1_size, s2_size):
    preg_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(s2_size, dtype=pl.DT_FP32)
    preg_all_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x = vf.muls(vreg_x, SCALE, preg_tail)

        preg_mask = vf.load_align(mask_tile, m * TKV, dist=pl.LoadDist.DS)
        vreg_x = vf.select(vreg_min, vreg_x, preg_mask)
        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vreg_max = vf.reduce_max(vreg_x, preg_tail, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_max, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(tmp_max, ureg_max, 0, post_update=True)
    vreg_in_max = vf.load_align(max_tile, 0)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    vreg_max_new = vf.load_align(tmp_max_st, 0)
    vreg_max_new = vf.max(vreg_in_max, vreg_max_new, preg_all)
    vf.store_align(tmp_max_st, vreg_max_new, preg_all)

    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(tmp_max_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2 = vf.load_align(input_tile, i * TKV)
        vreg_exp = vf.exp_sub(vreg_x_2, vreg_max_2, preg_tail)

        vreg_exp_sum = vf.reduce_sum(vreg_exp, preg_tail, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_exp_sum, vreg_exp_sum, ureg_exp_sum, 1, post_update=True)

        vreg_exp_even_b16 = vf.astype(vreg_exp, preg_all, layout=pl.CastLayout.ZERO, dtype=pl.DT_BF16)
        vreg_dst_even_b16, vreg_dst_odd_b16 = vf.de_interleave(vreg_exp_even_b16, vreg_exp_even_b16)
        vf.store_align(dst_tile, vreg_dst_even_b16, preg_all_b16,
                       block_stride=BLOCK_STRIDE_ND, repeat_stride=REPEAT_STRIDE_ND,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
    vf.store_unalign_post(tmp_exp_sum, ureg_exp_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_update_vf_unalign(input_tile, dst_tile, max_tile, mask_tile,
                                      tmp_max, tmp_max_st, tmp_exp_sum, s1_size, s2_size):
    preg_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_mask_unroll = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(s2_size - 64, dtype=pl.DT_FP32)
    preg_all_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x_unroll = vf.load_align(input_tile, m * TKV + 64)

        vreg_x = vf.muls(vreg_x, SCALE, preg_all)
        vreg_x_unroll = vf.muls(vreg_x_unroll, SCALE, preg_tail)

        preg_mask = vf.load_align(mask_tile, m * TKV, dist=pl.LoadDist.DS)
        preg_mask_unroll = vf.load_align(mask_tile, m * TKV + 64, dist=pl.LoadDist.DS)
        vreg_x = vf.select(vreg_min, vreg_x, preg_mask)
        vreg_x_unroll = vf.select(vreg_min, vreg_x_unroll, preg_mask_unroll)
        vreg_x_unroll = vf.select(vreg_x_unroll, vreg_min, preg_tail)
        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vf.store_align(input_tile + m * TKV + 64, vreg_x_unroll, preg_all)
        vreg_max_tmp = vf.max(vreg_x, vreg_x_unroll, preg_all)
        vreg_max = vf.reduce_max(vreg_max_tmp, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_max, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(tmp_max, ureg_max, 0, post_update=True)
    vreg_in_max = vf.load_align(max_tile, 0)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    vreg_max_new = vf.load_align(tmp_max_st, 0)
    vreg_max_new = vf.max(vreg_in_max, vreg_max_new, preg_all)
    vf.store_align(tmp_max_st, vreg_max_new, preg_all)

    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(tmp_max_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2, vreg_x_unroll_2 = vf.load_align(input_tile, i * TKV, dist=pl.LoadDist.DINTLV_B32)
        vreg_exp_even = vf.exp_sub(vreg_x_2, vreg_max_2, preg_all)
        vreg_exp_odd = vf.exp_sub(vreg_x_unroll_2, vreg_max_2, preg_all)

        vreg_exp_sum = vf.add(vreg_exp_even, vreg_exp_odd, preg_all)
        vreg_exp_sum = vf.reduce_sum(vreg_exp_sum, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_exp_sum, vreg_exp_sum, ureg_exp_sum, 1, post_update=True)

        vreg_exp_even_b16 = vf.astype(vreg_exp_even, preg_all, layout=pl.CastLayout.ZERO, dtype=pl.DT_BF16)
        vreg_exp_odd_b16 = vf.astype(vreg_exp_odd, preg_all, layout=pl.CastLayout.ONE, dtype=pl.DT_BF16)
        vreg_exp_b16 = vf.or_(vreg_exp_even_b16, vreg_exp_odd_b16, preg_all_b16)
        vf.store_align(dst_tile, vreg_exp_b16, preg_all_b16,
                       block_stride=BLOCK_STRIDE_ND, repeat_stride=REPEAT_STRIDE_ND,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
    vf.store_unalign_post(tmp_exp_sum, ureg_exp_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_update_vf(input_tile, dst_tile, max_tile, mask_tile,
                              tmp_max, tmp_max_st, tmp_exp_sum, s1_size):
    preg_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_mask_unroll = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_all_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x_unroll = vf.load_align(input_tile, m * TKV + 64)

        vreg_x = vf.muls(vreg_x, SCALE, preg_all)
        vreg_x_unroll = vf.muls(vreg_x_unroll, SCALE, preg_all)

        preg_mask = vf.load_align(mask_tile, m * TKV, dist=pl.LoadDist.DS)
        preg_mask_unroll = vf.load_align(mask_tile, m * TKV + 64, dist=pl.LoadDist.DS)
        vreg_x = vf.select(vreg_min, vreg_x, preg_mask)
        vreg_x_unroll = vf.select(vreg_min, vreg_x_unroll, preg_mask_unroll)
        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vf.store_align(input_tile + m * TKV + 64, vreg_x_unroll, preg_all)
        vreg_max_tmp = vf.max(vreg_x, vreg_x_unroll, preg_all)
        vreg_max = vf.reduce_max(vreg_max_tmp, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_max, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(tmp_max, ureg_max, 0, post_update=True)
    vreg_in_max = vf.load_align(max_tile, 0)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    vreg_max_new = vf.load_align(tmp_max_st, 0)
    vreg_max_new = vf.max(vreg_in_max, vreg_max_new, preg_all)
    vf.store_align(tmp_max_st, vreg_max_new, preg_all)

    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(tmp_max_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2, vreg_x_unroll_2 = vf.load_align(input_tile, i * TKV, dist=pl.LoadDist.DINTLV_B32)
        vreg_exp_even = vf.exp_sub(vreg_x_2, vreg_max_2, preg_all)
        vreg_exp_odd = vf.exp_sub(vreg_x_unroll_2, vreg_max_2, preg_all)

        vreg_exp_sum = vf.add(vreg_exp_even, vreg_exp_odd, preg_all)
        vreg_exp_sum = vf.reduce_sum(vreg_exp_sum, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_exp_sum, vreg_exp_sum, ureg_exp_sum, 1, post_update=True)

        vreg_exp_even_b16 = vf.astype(vreg_exp_even, preg_all, layout=pl.CastLayout.ZERO, dtype=pl.DT_BF16)
        vreg_exp_odd_b16 = vf.astype(vreg_exp_odd, preg_all, layout=pl.CastLayout.ONE, dtype=pl.DT_BF16)
        vreg_exp_b16 = vf.or_(vreg_exp_even_b16, vreg_exp_odd_b16, preg_all_b16)
        vf.store_align(dst_tile, vreg_exp_b16, preg_all_b16,
                       block_stride=BLOCK_STRIDE_ND, repeat_stride=REPEAT_STRIDE_ND,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
    vf.store_unalign_post(tmp_exp_sum, ureg_exp_sum, 0, post_update=True)


@pl.vector_function
def update_exp_sum(exp_diff, max_tile, tmp_max, sum_tile, tmp_sum):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)

    vreg_max = vf.load_align(max_tile, 0)
    vreg_max_tmp = vf.load_align(tmp_max, 0)
    vreg_exp_max = vf.exp_sub(vreg_max, vreg_max_tmp, preg_all)
    vf.store_align(exp_diff, vreg_exp_max, preg_all)
    vf.store_align(max_tile, vreg_max_tmp, preg_all)

    vreg_sum = vf.load_align(sum_tile, 0)
    vreg_sum_tmp = vf.load_align(tmp_sum, 0)
    vreg_exp_update = vf.mul(vreg_sum, vreg_exp_max, preg_all)
    vreg_exp_update = vf.add(vreg_exp_update, vreg_sum_tmp, preg_all)
    vf.store_align(sum_tile, vreg_exp_update, preg_all)


@pl.vector_function
def flash_update_basic_vf(dst_tile, cur_tile, pre_tile, exp_max_tile, s1_size, has_tail):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(TAIL_D, dtype=pl.DT_FP32)
    for i in pl.range(0, s1_size):
        vreg_exp_max = vf.load_align(exp_max_tile, i * REDUCE_SIZE, dist=pl.LoadDist.BRC_B32)
        for j in pl.range(0, D_LOOPS):
            vreg_input_pre = vf.load_align(pre_tile, i * TD + j * FLOAT_REP_SIZE)
            vreg_input_cur = vf.load_align(cur_tile, i * TD + j * FLOAT_REP_SIZE)
            vreg_mul = vf.mul(vreg_exp_max, vreg_input_pre, preg_all)
            vreg_add = vf.add(vreg_mul, vreg_input_cur, preg_all)
            vf.store_align(dst_tile + (i * TD + j * FLOAT_REP_SIZE), vreg_add, preg_all)
        for _ in pl.range(0, has_tail):
            vreg_input_pre = vf.load_align(pre_tile, i * TD + D_LOOPS * FLOAT_REP_SIZE)
            vreg_input_cur = vf.load_align(cur_tile, i * TD + D_LOOPS * FLOAT_REP_SIZE)
            vreg_mul = vf.mul(vreg_exp_max, vreg_input_pre, preg_tail)
            vreg_add = vf.add(vreg_mul, vreg_input_cur, preg_tail)
            vf.store_align(dst_tile + (i * TD + D_LOOPS * FLOAT_REP_SIZE), vreg_add, preg_tail)


@pl.vector_function
def flash_update_last_basic_vf(dst_tile, cur_tile, pre_tile, exp_max_tile, exp_sum_tile, s1_size, has_tail):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(TAIL_D, dtype=pl.DT_FP32)
    for i in pl.range(0, s1_size):
        vreg_exp_max = vf.load_align(exp_max_tile, i * REDUCE_SIZE, dist=pl.LoadDist.BRC_B32)
        vreg_exp_sum = vf.load_align(exp_sum_tile, i * REDUCE_SIZE, dist=pl.LoadDist.BRC_B32)
        for j in pl.range(0, D_LOOPS):
            vreg_input_pre = vf.load_align(pre_tile, i * TD + j * FLOAT_REP_SIZE)
            vreg_input_cur = vf.load_align(cur_tile, i * TD + j * FLOAT_REP_SIZE)
            vreg_mul = vf.mul(vreg_exp_max, vreg_input_pre, preg_all)
            vreg_add = vf.add(vreg_mul, vreg_input_cur, preg_all)
            vreg_div = vf.div(vreg_add, vreg_exp_sum, preg_all)
            vf.store_align(dst_tile + (i * TD + j * FLOAT_REP_SIZE), vreg_div, preg_all)
        for _ in pl.range(0, has_tail):
            vreg_input_pre = vf.load_align(pre_tile, i * TD + D_LOOPS * FLOAT_REP_SIZE)
            vreg_input_cur = vf.load_align(cur_tile, i * TD + D_LOOPS * FLOAT_REP_SIZE)
            vreg_mul = vf.mul(vreg_exp_max, vreg_input_pre, preg_tail)
            vreg_add = vf.add(vreg_mul, vreg_input_cur, preg_tail)
            vreg_div = vf.div(vreg_add, vreg_exp_sum, preg_tail)
            vf.store_align(dst_tile + (i * TD + D_LOOPS * FLOAT_REP_SIZE), vreg_div, preg_tail)


@pl.vector_function
def last_div_vf(dst_tile, cur_tile, exp_sum_tile, s1_size, has_tail):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(TAIL_D, dtype=pl.DT_FP32)
    for i in pl.range(0, s1_size):
        vreg_exp_sum = vf.load_align(exp_sum_tile, i * REDUCE_SIZE, dist=pl.LoadDist.BRC_B32)
        for j in pl.range(0, D_LOOPS):
            vreg_input_cur = vf.load_align(cur_tile, i * TD + j * FLOAT_REP_SIZE)
            vreg_div = vf.div(vreg_input_cur, vreg_exp_sum, preg_all)
            vf.store_align(dst_tile + (i * TD + j * FLOAT_REP_SIZE), vreg_div, preg_all)
        for _ in pl.range(0, has_tail):
            vreg_input_cur = vf.load_align(cur_tile, i * TD + D_LOOPS * FLOAT_REP_SIZE)
            vreg_div = vf.div(vreg_input_cur, vreg_exp_sum, preg_tail)
            vf.store_align(dst_tile + (i * TD + D_LOOPS * FLOAT_REP_SIZE), vreg_div, preg_tail)


@pl.vector_function
def compute_row_bounds_vf(bound_tile, spans_tile, q_abs0, skv, span_num):
    """Generate this sub-tile's per-row visibility bound straight from the spans.

    All TS_HALF rows are done in one 64-lane register: lane i holds the absolute
    query position ``q_abs = q_abs0 + i`` and ends up holding

        bound = min(skv, max(q_abs + 1, max{e + 1 : span [s,e] covers q_abs}))

    which is the operator's documented rule. Row r then sees KV columns
    ``[0, bound)`` -- a pure prefix, so no hole channel is involved anywhere.

    ``spans_tile`` holds this request's mm_prefix_range staged one span per row,
    rows padded to SPAN_ROW; span h is therefore at ``h * SPAN_ROW`` (start) and
    ``+ 1`` (end). VF code cannot reach GM itself -- ``pl.getval`` inside a VF
    section fails to compile with "Unsupported Inst must be hoisted" -- so the
    caller stages it and this reads each endpoint with a broadcast load.

    Padding spans (``[-1, -1]``) need no branch: the tile is uint32, so ``s``
    reads back as 0xFFFFFFFF and ``q_abs >= s`` is false for every real
    position, leaving the slot inert.
    """
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_hit = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    preg_le = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)

    vreg_qabs = vf.arange(q_abs0, dtype=pl.DT_UINT32)
    vreg_bound = vf.adds(vreg_qabs, 1, preg_all)

    for h in pl.range(0, span_num):
        vreg_s = vf.load_align(spans_tile, h * SPAN_ROW, dist=pl.LoadDist.BRC_B32)
        vreg_e = vf.load_align(spans_tile, h * SPAN_ROW + 1, dist=pl.LoadDist.BRC_B32)
        preg_hit = vf.ge(vreg_qabs, vreg_s, preg_all)
        preg_le = vf.le(vreg_qabs, vreg_e, preg_all)
        preg_hit = vf.and_(preg_hit, preg_le, preg_all)
        vreg_e1 = vf.adds(vreg_e, 1, preg_all)
        vreg_big = vf.max(vreg_bound, vreg_e1, preg_all)
        vreg_bound = vf.select(vreg_big, vreg_bound, preg_hit)

    vreg_bound = vf.mins(vreg_bound, skv, preg_all)
    vf.store_align(bound_tile, vreg_bound, preg_all)


@pl.vector_function
def decode_prefix_mask(mask, bound, col, s1_size):
    """Expand per-row bounds into the [TS_HALF, TKV] uint8 element mask.

    Visible iff ``c < bound[row]``. That single compare is the whole mask: the
    hybrid mask is a prefix, so there is no hole term to AND in.
    """
    index = vf.arange(col, dtype=pl.DT_UINT32)
    index_unroll = vf.arange(col + 64, dtype=pl.DT_UINT32)
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_all_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    merge_bit = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    merge_unroll_bit = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    row_reg = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    temp_reg = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    vreg_zero = vf.full(0, dtype=pl.DT_UINT16)
    vreg_one = vf.full(1, dtype=pl.DT_UINT16)
    for m in pl.range(0, s1_size):
        bound_reg = vf.load_align(bound, m, dist=pl.LoadDist.BRC_B32)
        merge_bit = vf.lt(index, bound_reg, preg_all)
        merge_unroll_bit = vf.lt(index_unroll, bound_reg, preg_all)
        row_reg, temp_reg = vf.de_interleave(merge_bit, merge_unroll_bit, dtype=pl.DT_UINT16)
        mask_b16 = vf.select(vreg_zero, vreg_one, row_reg)
        vf.store_align(mask + (m * TKV), mask_b16, preg_all_b16, dist=pl.StoreDist.PACK)


def compute_qk(ctx, ki, sq_off, q, k, cur_q_slot, k_l1_db, left_db, right_db, acc_db, qk_vec_db, task_id,
               left2, right2, acc2):
    # --- compute_qk inlined ---
    # Paged KV: ctx carries the resolved page and the tile's token offset
    # inside it, so a page may hold several KV tiles.
    cur_k_slot = k_l1_db.next()
    qk_left = left_db.next()
    qk_right = right_db.next()
    qk_acc = acc_db.next()
    left2.next()
    right2.next()
    acc2.next()

    if ctx.loop_count == 0:
        pl.set_validshape(cur_q_slot, [ctx.s1_size, TD])
        pl.load(cur_q_slot, q, [sq_off, ctx.n_idx, 0], order=[0, 2])
    pl.set_validshape(cur_k_slot, [TD, ctx.s2_size])
    # GQA: q head ctx.n_idx shares kv head ctx.kv_n_idx == n_idx // group.
    pl.load(cur_k_slot, k, [ctx.kv_page, ctx.kv_slot, ctx.kv_n_idx, 0], order=[3, 1])

    pl.set_validshape(qk_left, [ctx.s1_size, TD])
    pl.move(qk_left, cur_q_slot)
    pl.set_validshape(qk_right, [TD, ctx.s2_size])
    pl.move(qk_right, cur_k_slot)
    pl.set_validshape(qk_acc, [ctx.s1_size, ctx.s2_size])
    pl.matmul(qk_acc, qk_left, qk_right)

    qk_slot = qk_vec_db.next()
    pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_BARKWARD_IDS[task_id % 2])
    pl.set_validshape(qk_slot, [(ctx.s1_size + 1) // 2, ctx.s2_size])
    pl.set_validshape(qk_acc, [(ctx.s1_size + 1) // 2 * 2, (ctx.s2_size + 7) // 8 * 8])
    pl.move(qk_slot, qk_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_FORWARD_IDS[task_id % 2])


def compute_pv(ctx, v_l1_db, p_mat_db, left_db, right_db, acc_db, pv_vec_db, v, left2, right2, acc2):
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=P_READY_FORWARD_IDS[ctx.task_id % 3])
    cur_v_slot = v_l1_db.next()
    # # current() advances p_mat_db cursor to stay in sync with Vec,
    cur_p_slot = p_mat_db.next()
    pv_left = left_db.next()
    pv_right = right_db.next()
    pv_acc = acc_db.next()
    left2.next()
    right2.next()
    acc2.next()

    pl.set_validshape(cur_v_slot, [ctx.s2_size, TD])
    pl.load(cur_v_slot, v, [ctx.kv_page, ctx.kv_slot, ctx.kv_n_idx, 0], order=[1, 3])
    pl.set_validshape(cur_p_slot, [ctx.s1_size, ctx.s2_size])
    pl.set_validshape(pv_left, [ctx.s1_size, ctx.s2_size])
    pl.move(pv_left, cur_p_slot)
    pl.set_validshape(pv_right, [ctx.s2_size, TD])
    pl.move(pv_right, cur_v_slot)
    pl.set_validshape(pv_acc, [ctx.s1_size, TD])
    pl.matmul(pv_acc, pv_left, pv_right)

    pv_slot = pv_vec_db.next()
    pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_BARKWARD_IDS[ctx.task_id % 2])
    pl.set_validshape(pv_slot, [(ctx.s1_size + 1) // 2, TD])
    pl.set_validshape(pv_acc, [(ctx.s1_size + 1) // 2 * 2, TD])
    pl.move(pv_slot, pv_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_FORWARD_IDS[ctx.task_id % 2])


def compute_p(ctx_p, sub_id, qk_vec_db, tile_nz_db, tmp_max, tmp_sum,
              global_max, global_sum, exp_corr_db, p_mat_db, mask_db,
              bound_db, spans_db, spans_gm, zero_mask_gm):
    """Softmax on KQ tile -> P. Includes cross-core sync."""
    p_eid = ctx_p.task_id % 2
    q_idx_p = ctx_p.q_count % 3

    qk_slot = qk_vec_db.next()
    tile_nz = tile_nz_db.next()
    mask_buf = mask_db.next()
    bound_buf = bound_db.next()
    spans_buf = spans_db.next()
    gmax_p = global_max[q_idx_p]
    gsum_p = global_sum[q_idx_p]
    exp_diff = exp_corr_db[ctx_p.task_id % 3]
    row_offset = ctx_p.first_s1 * sub_id
    col = ctx_p.ki * TKV

    if ctx_p.mask_bit == 0:
        # Fully-visible block: every row's bound already covers it, so the mask
        # would decode to all-visible. Copy a fixed [TS_HALF, TKV] all-zero mask
        # in from GM (0 == visible) instead of spending the vector unit on it.
        pl.set_validshape(mask_buf, [TS_HALF, TKV])
        pl.load(mask_buf, zero_mask_gm, [0, 0])
    else:
        # Generate the mask from this request's spans. sub_id splits the Q tile
        # into two row halves, so this subblock's first row sits at
        # q_abs_tile + row_offset in absolute (KV-cache) coordinates.
        # Valid region is [span_num, 2]; the tile itself is SPAN_ROW wide, so
        # the load fills the first two columns of each row and leaves the pad.
        pl.set_validshape(spans_buf, [HYBRID_MAX_SPANS, 2])
        pl.load(spans_buf, spans_gm, [ctx_p.b_idx, 0, 0], order=[1, 2])
        compute_row_bounds_vf(bound_buf, spans_buf, ctx_p.q_abs0 + row_offset,
                              ctx_p.s2_total, HYBRID_MAX_SPANS)
        decode_prefix_mask(mask_buf, bound_buf, col, ctx_p.half_s1)

    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_FORWARD_IDS[p_eid])
    if ctx_p.loop_count == 0:
        if ctx_p.s2_size == 128:
            process_vec1_nd_no_update_vf(qk_slot, tile_nz, gmax_p, gmax_p, gsum_p, mask_buf, ctx_p.half_s1)
        elif ctx_p.s2_size <= 64:
            process_vec1_nd_no_update_vf_unalign64(qk_slot, tile_nz, gmax_p, gmax_p, gsum_p, mask_buf,
                                                   ctx_p.half_s1, ctx_p.s2_size)
        else:
            process_vec1_nd_no_update_vf_unalign(qk_slot, tile_nz, gmax_p, gmax_p, gsum_p, mask_buf,
                                         ctx_p.half_s1, ctx_p.s2_size)
    else:
        if ctx_p.s2_size == 128:
            process_vec1_nd_update_vf(qk_slot, tile_nz, gmax_p, mask_buf, tmp_max, tmp_max, tmp_sum, ctx_p.half_s1)
        elif ctx_p.s2_size <= 64:
            process_vec1_nd_update_vf_unalign64(qk_slot, tile_nz, gmax_p, mask_buf, tmp_max, tmp_max, tmp_sum,
                                                ctx_p.half_s1, ctx_p.s2_size)
        else:
            process_vec1_nd_update_vf_unalign(qk_slot, tile_nz, gmax_p, mask_buf, tmp_max, tmp_max, tmp_sum,
                                              ctx_p.half_s1, ctx_p.s2_size)
        update_exp_sum(exp_diff, gmax_p, tmp_max, gsum_p, tmp_sum)
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BARKWARD_IDS[p_eid])

    cur_p_slot = p_mat_db.next()
    pl.set_validshape(tile_nz, [ctx_p.half_s1, ctx_p.s2_size])
    pl.insert(cur_p_slot, tile_nz, [row_offset, 0])
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY_FORWARD_IDS[ctx_p.task_id % 3])


def compute_gu(ctx_gu, pv_vec_db, exp_corr_db, global_sum_buf, running_o, o_bf16, o):
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_FORWARD_IDS[ctx_gu.task_id % 2])

    sub_id = pl.get_subblock_idx()
    row_offset = ctx_gu.first_s1 * sub_id
    pv_slot = pv_vec_db.next()
    gsum_gu = global_sum_buf[ctx_gu.q_count % 3]
    exp_corr_gu = exp_corr_db[ctx_gu.task_id % 3]
    pl.set_validshape(running_o, [ctx_gu.half_s1, TD])
    pl.set_validshape(pv_slot, [ctx_gu.half_s1, TD])
    has_tail = 0
    if TAIL_D != 0:
        has_tail = 1
    if ctx_gu.loop_count == 0:
        pl.move(running_o, pv_slot)
    else:
        if ctx_gu.ki < ctx_gu.kv_loop - 1:
            flash_update_basic_vf(running_o, pv_slot, running_o, exp_corr_gu, ctx_gu.half_s1, has_tail)
        else:
            flash_update_last_basic_vf(running_o, pv_slot, running_o, exp_corr_gu, gsum_gu, ctx_gu.half_s1, has_tail)
            pl.set_validshape(o_bf16, [ctx_gu.half_s1, TD])
            pl.cast(o_bf16, running_o, mode=pl.RoundMode.CAST_ROUND)
            pl.store(o, o_bf16, [ctx_gu.q_off + row_offset, ctx_gu.n_idx, 0], order=[0, 2])

    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[ctx_gu.task_id % 2])
    if ctx_gu.ki == ctx_gu.kv_loop - 1:
        if ctx_gu.loop_count == 0:
            last_div_vf(running_o, running_o, gsum_gu, ctx_gu.half_s1, has_tail)
            pl.set_validshape(o_bf16, [ctx_gu.half_s1, TD])
            pl.cast(o_bf16, running_o, mode=pl.RoundMode.CAST_ROUND)
            pl.store(o, o_bf16, [ctx_gu.q_off + row_offset, ctx_gu.n_idx, 0], order=[0, 2])


def scalar_bound(spans_gm, b_idx, span_num, q_abs):
    """Scalar form of the operator's bound rule for one absolute query position.

    Traced inline (both the Cube and the Vector section need it to size their KV
    loop). ``bound`` is non-decreasing in ``q_abs``: for q1 < q2 a span covering
    q1 either also covers q2, or ends before it, in which case q2 + 1 already
    exceeds that span's e + 1. That monotonicity is what lets a whole Q tile be
    characterized by just its first and last row -- see the caller.
    """
    bound = q_abs + 1
    for h in pl.range(0, span_num):
        s = pl.getval(spans_gm, (b_idx * span_num + h) * 2)
        e = pl.getval(spans_gm, (b_idx * span_num + h) * 2 + 1)
        if s >= 0:                      # -1 marks a padding slot
            if s <= q_abs:
                if q_abs <= e:
                    bound = pl.max(bound, e + 1)
    return bound


# ================================================================
#  Kernel with NBuffer + auto_mutex
# ================================================================
@pl.jit(arch="a5", auto_mutex=True, compile_timeout=200)
def flex_attention_bf16(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    # PagedAttention NHD, exactly as the operator declares them:
    # [num_blocks, block_size, n_head_kv, dim].
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    o: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    # The four below are the operator's own inputs, in its order.
    #   cu_seqlens_q    [B+1] prefix sum, first element MUST be 0. Batch b's Q
    #                   rows begin at cu_seqlens_q[b] and the kernel indexes
    #                   q/o there directly, so a nonzero head shifts every
    #                   request. Built by make_cu_seqlens_q(), which enforces it.
    #   seqlens_kvcache [B] *absolute* per-request KV lengths, not a prefix sum.
    #   mm_prefix_range [B, max_spans, 2] exactly as the operator defines it,
    #                   passed through untouched. It is read as scalars rather
    #                   than reshaped to suit the UB tile -- see SPAN_ROW.
    cu_seqlens_q: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    block_ids: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    seqlens_kvcache: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    mm_prefix_range: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 2], pl.DT_INT32],
    # Kernel-internal extras the operator handles inside its tiling function.
    zero_mask: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT8],
    # [cores, 4] = (work_start, n_items, split_mode, two_c). See
    # build_work_ranges for what the two split modes are and why.
    work_ranges: pl.Tensor[[pl.DYNAMIC, 4], pl.DT_INT32],
):
    n_dim = q.shape[1]
    # GQA: n_dim query heads share kv_n_dim key/value heads. group == 1 is MHA.
    kv_n_dim = k.shape[2]
    group = n_dim // kv_n_dim
    batch = seqlens_kvcache.shape[0]
    span_num = mm_prefix_range.shape[1]
    max_blocks = block_ids.shape[1]
    # KV tiles per page; 1 when block_size == TKV.
    page_tiles = k.shape[1] // TKV
    core_id = pl.get_block_idx() // pl.get_subblock_num()

    # ========== Cross-core shared buffers (UBNBuffer for double-buffer) ==========
    # P MAT - Vector insert, Cube PV read
    p_mat_db = pl.make_tile_group(
        type=pl.TileType(shape=[TS, TKV], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat),
        addrs=MA2_P, mutex_ids=[14, 15, 16])

    # qk_vec UB - Cube store from ACC, Vector softmax (double-buffer for FIFO)
    qk_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TS_HALF, TKV], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1],
                         compact=1, pad=pl.TilePad.min),
        addrs=VA0, mutex_ids=[17, 18])

    # pv_vec UB - Cube store from ACC, Vector GU (double-buffer for FIFO)
    pv_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=VA8, mutex_ids=[19, 20])

    # =================== CUBE SECTION ===================
    with pl.section_cube():
        # Cube-only buffers (independent buf_id space: 0-11)
        q_l1_db = pl.make_tile_group(
            type=pl.TileType(shape=[TS, TD], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1], compact=1),
            addrs=MA0_Q, mutex_ids=[0, 1])
        k_l1_db = pl.make_tile_group(
            type=pl.TileType(shape=[TD, TKV], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat,
                             layout=pl.ZN, valid_shape=[-1, -1], compact=1),
            addrs=MA1_K, mutex_ids=[2, 3])
        v_l1_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TD], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1], compact=1),
            addrs=MA3_V, mutex_ids=[4, 5])

        left_db = pl.make_tile_group(
            type=pl.TileType(shape=[TS, TD], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1], compact=1),
            addrs=[0, 32768], mutex_ids=[6, 7])
        right_db = pl.make_tile_group(
            type=pl.TileType(shape=[TD, TKV], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1], compact=1),
            addrs=[0, 32768], mutex_ids=[8, 9])
        acc_db = pl.make_tile_group(
            type=pl.TileType(shape=[TS, TKV], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1], compact=1),
            addrs=[0, 65536, 131072, 196608], mutex_ids=[10, 11, 12, 13])
        left_db2 = pl.make_tile_group(
            type=pl.TileType(shape=[TS, TKV], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1], compact=1),
            addrs=[0, 32768], mutex_ids=[6, 7])
        right_db2 = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TD], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1], compact=1),
            addrs=[0, 32768], mutex_ids=[8, 9])
        acc_db2 = pl.make_tile_group(
            type=pl.TileType(shape=[TS, TD], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1], compact=1),
            addrs=[0, 65536, 131072, 196608], mutex_ids=[10, 11, 12, 13])

        work_start = pl.getval(work_ranges, core_id * 4)
        n_items = pl.getval(work_ranges, core_id * 4 + 1)
        split_mode = pl.getval(work_ranges, core_id * 4 + 2)
        two_c = pl.getval(work_ranges, core_id * 4 + 3)
        task_id = 0
        b_idx = 0
        actual_s1 = 0
        actual_s2 = 0
        s1o_acc = 0
        ctx_arr = pl.struct_array(4, "CubeCtx", n_idx=0, kv_n_idx=0, qi=0, ki=0,
                                  task_id=0, kv_page=0, kv_slot=0, s1_size=0, s2_size=0, loop_count=0)
        s1o_size = 0 # tmp-val
        for idx in pl.range(batch):
            actual_s1 = pl.getval(cu_seqlens_q, idx + 1) - pl.getval(cu_seqlens_q, idx)
            actual_s2 = pl.getval(seqlens_kvcache, idx)
            s1o_size = s1o_size + (actual_s1 + TS - 1) // TS * n_dim
            if (work_start >= s1o_size):
                s1o_acc = s1o_size
                b_idx = b_idx + 1
                continue
            break
        for i in pl.range(0, n_items):
            # Two core-split modes. Contiguous (mode 0) hands each core one
            # [start, start+n_items) run; strided-mirrored (mode 1) hands it
            # indices t*2C+c and t*2C+2C-1-c. Mode 1 is what keeps L2 warm once
            # B*N exceeds the core count -- see build_work_ranges.
            work_id = work_start + i
            if split_mode == 1:
                work_id = (i // 2) * two_c + core_id
                if i % 2 == 1:
                    work_id = (i // 2) * two_c + two_c - 1 - core_id
            for _ in pl.range(b_idx, batch):
                actual_s1 = pl.getval(cu_seqlens_q, b_idx + 1) - pl.getval(cu_seqlens_q, b_idx)
                actual_s2 = pl.getval(seqlens_kvcache, b_idx)
                s1o_size = s1o_acc + (actual_s1 + TS - 1) // TS * n_dim
                if (work_id >= s1o_size):
                    s1o_acc = s1o_size
                    b_idx = b_idx + 1
                    continue
                break
            cur_b_s1o = (actual_s1 + TS - 1) // TS
            s1o_size = work_id - s1o_acc
            n_idx = s1o_size // cur_b_s1o
            s1_idx = s1o_size % cur_b_s1o
            s1_size = pl.min(TS, actual_s1 - s1_idx * TS)

            # cu_seqlens_q[b] is already this batch's Q offset -- no accumulator.
            sq_off = pl.getval(cu_seqlens_q, b_idx) + s1_idx * TS
            cur_q_slot = q_l1_db.next()
            # Right-down alignment: this Q tile's rows occupy absolute KV
            # positions [q_abs0, q_abs0 + s1_size). bound is monotone in the row,
            # so the tile's last row gives the furthest KV column anyone needs.
            q_abs0 = actual_s2 - actual_s1 + s1_idx * TS
            kv_end = pl.min(scalar_bound(mm_prefix_range, b_idx, span_num, q_abs0 + s1_size - 1),
                            actual_s2)
            kv_loop = (kv_end + TKV - 1) // TKV
            drain = 0
            if i == n_items - 1:
                drain = 2
            for ki in pl.range(0, kv_loop + drain):
                if ki < kv_loop:
                    # A prefix mask leaves no gaps: blocks 0..kv_loop-1 are all
                    # computed, so there is no per-block skip bitmap to consult.
                    ctx_curr = ctx_arr[task_id % 4]
                    ctx_curr.task_id = task_id
                    ctx_curr.n_idx = n_idx
                    ctx_curr.kv_n_idx = n_idx // group
                    ctx_curr.ki = ki
                    # Resolve page and in-page offset here, in the producer:
                    # compute_pv runs two pipeline steps later and must see the
                    # values for its own ki.
                    ctx_curr.kv_page = pl.getval(block_ids,
                                                 b_idx * max_blocks + ki // page_tiles)
                    ctx_curr.kv_slot = (ki % page_tiles) * TKV
                    ctx_curr.s1_size = s1_size
                    ctx_curr.s2_size = pl.min(TKV, kv_end - ki * TKV)
                    ctx_curr.loop_count = ki  # ki == 0 -> load Q

                    # ========== compute_qk (current step) ==========
                    compute_qk(ctx_curr, ki, sq_off, q, k, cur_q_slot, k_l1_db,
                               left_db, right_db, acc_db, qk_vec_db, task_id,
                               left_db2, right_db2, acc_db2)

                # ========== compute_pv (delayed 1 step: uses ctx from task_id-1) ==========
                if task_id > 1:
                    ctx_pre2 = ctx_arr[(task_id + 2) % 4]
                    compute_pv(ctx_pre2, v_l1_db, p_mat_db, left_db2, right_db2, acc_db2, pv_vec_db, v,
                               left_db, right_db, acc_db)
                task_id = task_id + 1

    # =================== VECTOR SECTION ===================
    with pl.section_vector():
        tile_nz_g = pl.make_tile_group(
            type=pl.TileType(shape=[65, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec,
                             valid_shape=[-1, -1], layout=pl.NZ),
            addrs=VA1, mutex_ids=[0, 1])

        running_o = pl.make_tile(
            pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addr=VA7, size=VB4)

        o_bf16_g = pl.make_tile_group(
            type=pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
            addrs=VA9, mutex_ids=[2])
        o_bf16 = o_bf16_g.next()

        mask_db = pl.make_tile_group(
            type=pl.TileType(shape=[TS_HALF, TKV], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
            addrs=VA10, mutex_ids=[3, 4])

        # Per-row visibility bounds, produced on device by compute_row_bounds_vf.
        bound_db = pl.make_tile_group(
            type=pl.TileType(shape=[1, TS_HALF], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
            addrs=VA13, mutex_ids=[5, 6])

        # This request's mm_prefix_range, one span per row. Rows are widened to
        # SPAN_ROW purely to clear the block-width floor; only columns 0/1 are
        # ever written or read.
        spans_db = pl.make_tile_group(
            type=pl.TileType(shape=[HYBRID_MAX_SPANS, SPAN_ROW], dtype=pl.DT_UINT32,
                             target_memory=pl.MemorySpace.Vec, compact=1),
            addrs=VA14, mutex_ids=[7, 8])

        # Double-buffered global state (per Q tile) -use tile tuples for dynamic
        # indexing by q_count % 2, since StructArray ctx references need runtime index.
        red_type = pl.TileType(shape=[TS_HALF, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN)
        gmax_0 = pl.make_tile(red_type, addr=VA_GMAX0, size=VB_RED)
        gmax_1 = pl.make_tile(red_type, addr=VA_GMAX1, size=VB_RED)
        gmax_2 = pl.make_tile(red_type, addr=VA_GMAX2, size=VB_RED)
        global_max = (gmax_0, gmax_1, gmax_2)

        gsum_0 = pl.make_tile(red_type, addr=VA_GSUM0, size=VB_RED)
        gsum_1 = pl.make_tile(red_type, addr=VA_GSUM1, size=VB_RED)
        gsum_2 = pl.make_tile(red_type, addr=VA_GSUM2, size=VB_RED)
        global_sum = (gsum_0, gsum_1, gsum_2)

        tmp_max = pl.make_tile(red_type, addr=VA11, size=VB_RED)
        tmp_sum = pl.make_tile(red_type, addr=VA12, size=VB_RED)

        # FIFO exp_corr -use NBuffer with current() auto-rotate
        exp_max0 = pl.make_tile(red_type, addr=VA_EXPMAX0, size=VB_RED)
        exp_max1 = pl.make_tile(red_type, addr=VA_EXPMAX1, size=VB_RED)
        exp_max2 = pl.make_tile(red_type, addr=VA_EXPMAX2, size=VB_RED)
        exp_corr_db = (exp_max0, exp_max1, exp_max2)

        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BARKWARD_IDS[0])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BARKWARD_IDS[1])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[0])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[1])

        work_start = pl.getval(work_ranges, core_id * 4)
        n_items = pl.getval(work_ranges, core_id * 4 + 1)
        split_mode = pl.getval(work_ranges, core_id * 4 + 2)
        two_c = pl.getval(work_ranges, core_id * 4 + 3)
        sub_id = pl.get_subblock_idx()

        task_id = 0
        q_count = 0
        b_idx = 0
        actual_s1 = 0
        actual_s2 = 0
        s1o_acc = 0

        # StructArray(3) for pipeline context tracking (same as original)
        ctx_arr = pl.struct_array(4, "VecCtx", b_idx=0, n_idx=0, s1_idx=0, q_off=0, ki=0,
                                  q_count=0, sub_id=0, task_id=0, kv_loop=0, half_s1=0, first_s1=0,
                                  s1_size=0, s2_size=0, s2_total=0, q_abs0=0,
                                  mask_bit=0, loop_count=0)

        # calc start
        s1o_size = 0 # tmp-val
        for idx in pl.range(batch):
            actual_s1 = pl.getval(cu_seqlens_q, idx + 1) - pl.getval(cu_seqlens_q, idx)
            actual_s2 = pl.getval(seqlens_kvcache, idx)
            s1o_size = s1o_size + (actual_s1 + TS - 1) // TS * n_dim
            if (work_start >= s1o_size):
                s1o_acc = s1o_size
                b_idx = b_idx + 1
                continue
            break
        for i in pl.range(0, n_items):
            # Two core-split modes. Contiguous (mode 0) hands each core one
            # [start, start+n_items) run; strided-mirrored (mode 1) hands it
            # indices t*2C+c and t*2C+2C-1-c. Mode 1 is what keeps L2 warm once
            # B*N exceeds the core count -- see build_work_ranges.
            work_id = work_start + i
            if split_mode == 1:
                work_id = (i // 2) * two_c + core_id
                if i % 2 == 1:
                    work_id = (i // 2) * two_c + two_c - 1 - core_id
            for _ in pl.range(b_idx, batch):
                actual_s1 = pl.getval(cu_seqlens_q, b_idx + 1) - pl.getval(cu_seqlens_q, b_idx)
                actual_s2 = pl.getval(seqlens_kvcache, b_idx)
                s1o_size = s1o_acc + (actual_s1 + TS - 1) // TS * n_dim
                if (work_id >= s1o_size):
                    s1o_acc = s1o_size
                    b_idx = b_idx + 1
                    continue
                break
            cur_b_s1o = (actual_s1 + TS - 1) // TS
            s1o_size = work_id - s1o_acc
            n_idx = s1o_size // cur_b_s1o
            s1_idx = s1o_size % cur_b_s1o
            s1_size = pl.min(TS, actual_s1 - s1_idx * TS)

            q_off = pl.getval(cu_seqlens_q, b_idx) + s1_idx * TS
            # Same two scalar bounds the Cube section derives. bound is monotone
            # in the row, so bound(first row) is the tile's minimum and
            # bound(last row) its maximum -- together they decide both how many
            # KV blocks to walk and which of them need a per-element mask.
            q_abs0 = actual_s2 - actual_s1 + s1_idx * TS
            bound_lo = scalar_bound(mm_prefix_range, b_idx, span_num, q_abs0)
            kv_end = pl.min(scalar_bound(mm_prefix_range, b_idx, span_num, q_abs0 + s1_size - 1),
                            actual_s2)
            kv_loop = (kv_end + TKV - 1) // TKV

            drain = 0
            if i == n_items - 1:
                drain = 3
            for ki in pl.range(0, kv_loop + drain):
                if ki < kv_loop:
                    # ===== producer: build this task's context =====
                    ctx_curr = ctx_arr[task_id % 4]
                    ctx_curr.b_idx = b_idx
                    ctx_curr.n_idx = n_idx
                    ctx_curr.s1_idx = s1_idx
                    ctx_curr.q_off = q_off
                    ctx_curr.task_id = task_id
                    ctx_curr.ki = ki
                    ctx_curr.kv_loop = kv_loop
                    ctx_curr.q_count = q_count
                    ctx_curr.s1_size = s1_size
                    ctx_curr.s2_size = pl.min(TKV, kv_end - ki * TKV)
                    ctx_curr.s2_total = actual_s2
                    ctx_curr.q_abs0 = q_abs0
                    half_s1 = (ctx_curr.s1_size + 1) // 2
                    first_s1 = half_s1
                    if sub_id == 1:
                        half_s1 = ctx_curr.s1_size - half_s1
                    ctx_curr.first_s1 = first_s1
                    ctx_curr.half_s1 = half_s1
                    # Fully visible iff even the tile's *least* generous row
                    # covers the block's last column -> no decode needed.
                    ctx_curr.mask_bit = 1
                    if (ki + 1) * TKV <= bound_lo:
                        ctx_curr.mask_bit = 0
                    ctx_curr.loop_count = ki

                # ===== compute_p (consumer, lag 1: real steps + 1st drain step) =====
                if task_id > 0:
                    if ki <= kv_loop:
                        ctx_p = ctx_arr[(task_id + 3) % 4]
                        compute_p(ctx_p, sub_id, qk_vec_db, tile_nz_g, tmp_max, tmp_sum,
                            global_max, global_sum, exp_corr_db, p_mat_db, mask_db,
                            bound_db, spans_db, mm_prefix_range, zero_mask)

                # ===== compute_gu (consumer, lag 3: every step incl. all drains) =====
                if task_id > 2:
                    ctx_gu = ctx_arr[(task_id + 1) % 4]
                    compute_gu(ctx_gu, pv_vec_db, exp_corr_db, global_sum, running_o, o_bf16, o)

                task_id = task_id + 1
            q_count = q_count + 1


def flash_attention_ref_tnd(q_tnd, k_tnd, v_tnd, seq_q_list, seq_kv_list, d, masks):
    """Reference attention operating on TND tensors, per-batch.

    ``masks[bi]`` is a bool tensor of shape [sq, skv] where True == masked
    (the cell is invisible and gets NEG_INF before softmax). It is the single
    source of truth shared with the kernel: both come from the same spans via
    the bound rule (host: hybrid_bounds, device: compute_row_bounds_vf).

    This mirrors the *on-device flash-attention compute flow* rather than a
    single fused fp32 softmax, so the reference stays faithful to what the
    kernel actually computes:
      * KV is consumed in TKV-wide blocks with an online (running max/sum)
        softmax, exactly like the Cube/Vector pipeline;
      * the P (probability) tile is cast to bf16 *before* the P@V matmul, as the
        kernel does (``vf.astype(..., DT_BF16)`` in ``process_vec1_*``), so P@V
        runs in bf16 with fp32 accumulation;
      * the running denominator uses the fp32 exp sum (the kernel reduces the
        sum from the fp32 exp, before the bf16 cast).
    Modelling the bf16-P quantization is what lets the reference agree with the
    kernel to ~bf16 ulp even for sliding-window masks, where the attention peak
    lands in a later block and earlier blocks' bf16 P is downscaled by exp_corr.

    GQA: q head ``ni`` reads kv head ``ni // (n_head_q // n_head_kv)``, matching
    the kernel's ``ctx.kv_n_idx``.
    """
    q_tnd = q_tnd.cpu()
    k_tnd = k_tnd.cpu()
    v_tnd = v_tnd.cpu()
    seq_q_list = seq_q_list.cpu() if torch.is_tensor(seq_q_list) else seq_q_list
    seq_kv_list = seq_kv_list.cpu() if torch.is_tensor(seq_kv_list) else seq_kv_list
    scale_val = 1.0 / math.sqrt(d)
    n = q_tnd.shape[1]
    group = n // k_tnd.shape[1]
    o_tnd = torch.zeros_like(q_tnd)
    q_off = 0
    kv_off = 0
    for bi, sq_val in enumerate(seq_q_list):
        sq = int(sq_val)
        skv = int(seq_kv_list[bi])
        mask = masks[bi].cpu()
        for ni in range(n):
            kvi = ni // group
            qi = q_tnd[q_off:q_off + sq, ni, :].float()      # [sq, d]
            ki = k_tnd[kv_off:kv_off + skv, kvi, :].float()  # [skv, d]
            vi = v_tnd[kv_off:kv_off + skv, kvi, :].float()  # [skv, d]
            running_m = torch.full((sq, 1), float(NEG_INF))
            running_l = torch.zeros((sq, 1))
            acc = torch.zeros((sq, d))
            for c0 in range(0, skv, TKV):
                c1 = min(c0 + TKV, skv)
                qk = torch.matmul(qi, ki[c0:c1].T) * scale_val          # [sq, blk] fp32
                qk = qk.masked_fill(mask[:, c0:c1], NEG_INF)
                blk_max = qk.max(dim=1, keepdim=True).values            # [sq, 1]
                new_m = torch.maximum(running_m, blk_max)
                corr = torch.exp(running_m - new_m)                     # [sq, 1]
                p = torch.exp(qk - new_m)                               # [sq, blk] fp32
                p16 = p.bfloat16().float()                              # bf16 P (FA flow)
                acc = acc * corr + torch.matmul(p16, vi[c0:c1])
                running_l = running_l * corr + p.sum(dim=1, keepdim=True)
                running_m = new_m
            o_tnd[q_off:q_off + sq, ni, :] = (acc / running_l).bfloat16()
        q_off += sq
        kv_off += skv
    return o_tnd


def flash_attention_ref_fp32(q_tnd, k_tnd, v_tnd, seq_q_list, seq_kv_list, d, masks):
    """Naive fused fp32 attention -- the reference the operator grades against.

    Deliberately *not* shaped like the kernel: one fp32 softmax over the whole
    row, no KV blocking, no bf16 P quantization. flash_attention_ref_tnd mirrors
    the kernel's flow (which is what lets it agree to ~1 ulp); this one is
    independent, so it also cross-checks that reference rather than sharing its
    assumptions. The doc's criterion against it is cosine similarity >= 0.99.
    """
    q_tnd, k_tnd, v_tnd = q_tnd.cpu(), k_tnd.cpu(), v_tnd.cpu()
    scale_val = 1.0 / math.sqrt(d)
    n = q_tnd.shape[1]
    group = n // k_tnd.shape[1]
    o_tnd = torch.zeros_like(q_tnd)
    q_off = kv_off = 0
    for bi, sq_val in enumerate(seq_q_list):
        sq, skv = int(sq_val), int(seq_kv_list[bi])
        mask = masks[bi].cpu()
        for ni in range(n):
            kvi = ni // group
            qi = q_tnd[q_off:q_off + sq, ni, :].float()
            ki = k_tnd[kv_off:kv_off + skv, kvi, :].float()
            vi = v_tnd[kv_off:kv_off + skv, kvi, :].float()
            sc = torch.matmul(qi, ki.T) * scale_val
            sc = sc.masked_fill(mask, float("-inf"))
            o_tnd[q_off:q_off + sq, ni, :] = torch.matmul(torch.softmax(sc, dim=-1), vi).bfloat16()
        q_off += sq
        kv_off += skv
    return o_tnd


def cosine_similarity(a, b):
    """Whole-tensor cosine similarity, the doc's accuracy metric.

    Accumulated in float64 deliberately. At these sizes (millions of elements)
    an fp32 dot product and fp32 norms drift by more than the deviation being
    measured: with fp32 reductions cos(x, x) came back as 0.9969 instead of 1.0,
    which would have swamped the real ~1e-6 signal and reported every case as
    far worse than it is.
    """
    x, y = a.double().flatten(), b.double().flatten()
    return float(torch.dot(x, y) / (x.norm() * y.norm()))


def hybrid_bounds(sq, skv, spans):
    """Per-row visibility bound -- the host mirror of the on-device rule.

    This is the *only* mask quantity the host computes, and it exists purely to
    drive the reference and the core-balancing cost model; nothing derived here
    is shipped to the kernel, which reads mm_prefix_range and rebuilds the mask
    itself. Query row r sits at absolute position ``q_abs = skv - sq + r`` and
    sees KV columns ``[0, bound[r])``.
    """
    q_abs = torch.arange(sq, dtype=torch.int64) + (skv - sq)
    bound = q_abs + 1
    for s, e in spans:
        hit = (q_abs >= s) & (q_abs <= e)
        bound = torch.where(hit, torch.clamp(bound, min=e + 1), bound)
    return torch.clamp(bound, max=skv)


def normalize_spans(mm_prefix_range, b):
    """Validate mm_prefix_range and return (device tensor, per-batch span list).

    Accepts what the operator accepts: ``None`` (pure causal), ``[max_spans, 2]``
    broadcast to every request, or ``[b, max_spans, 2]``; ``-1`` marks padding.
    The device tensor keeps the operator's own ``[b, max_spans, 2]`` layout --
    only padded out to the full span budget, so the kernel's span loop is a
    fixed statically unrolled length and padding slots stay inert (read
    unsigned, ``s = -1`` never matches a real position). The [2, max_spans]
    transpose the UB staging tile needs is done by the load, not here.
    """
    dev = torch.full((b, HYBRID_MAX_SPANS, 2), -1, dtype=torch.int32)
    if mm_prefix_range is None:
        return dev, [[] for _ in range(b)]

    t = torch.as_tensor(mm_prefix_range, dtype=torch.int32)
    if t.dim() == 2:                                    # [max_spans, 2] -> broadcast
        t = t.unsqueeze(0).expand(b, -1, -1)
    assert t.shape[0] == b and t.shape[2] == 2, "mm_prefix_range must be [b, max_spans, 2]"
    assert t.shape[1] <= HYBRID_MAX_SPANS, \
        "at most %d spans per request" % HYBRID_MAX_SPANS

    dev[:, :t.shape[1], :] = t
    per_batch = [[(int(s), int(e)) for s, e in t[bi].tolist() if int(s) >= 0] for bi in range(b)]
    return dev, per_batch


def dense_mask_from_bounds(bound, skv):
    """Dense [sq, skv] reference mask (True == masked) from per-row bounds."""
    cols = torch.arange(skv, dtype=torch.int64)
    return ~(cols[None, :] < bound[:, None])



def build_work_ranges(seq_q_list, seq_kv_list, spans_per_batch, n, num_cores):
    """Assign work to cores; returns int32 [num_cores, 4] =
    (work_start, n_items, split_mode, two_c).

    A work item is one (batch, q head, Q tile) triple, enumerated with the Q
    tile innermost: ``work_id = n_idx * blk_m + s1_idx`` within a batch.

    **Mode 1, strided-mirrored** -- used once ``B * n_head_q > num_cores``.
    Core c takes ``t*2C + c`` and ``t*2C + 2C-1-c`` for t = 0, 1, ...
    (C = num_cores). Two things fall out of that:

      * *L2 locality*, which is the point. Contiguous ranges put the cores on
        ``num_cores`` different (batch, head) pairs at any instant, so the KV
        they collectively stream is the entire cache -- 128 MB at HY3-VL b=4
        sq=8192, far past what L2 holds, and the profile shows it: aic_mte2
        pegged at 0.947 with cube utilisation down to 88.8%. Striding puts all
        cores on *one* (batch, head) at a time, and since Q tile i reads KV
        blocks 0..i their ranges nest, so the live set is ~1.8 MB.
      * *Balance for free*. Under a causal mask a Q tile's cost grows with its
        index, so pairing a forward index with its mirror cancels the ramp --
        no host cost model, no binary search. Measured imbalance 1.00-1.03x,
        matching what the cost-balanced contiguous split achieves.

    **Mode 0, cost-balanced contiguous** -- kept for ``B * n_head_q <=
    num_cores``. There the mirrored pairing has too few items to average over
    (b=1 sq=512 lands at 1.23x) and the shapes are small enough that L2 was
    never the constraint.

    The kernel walks a plain counter ``i`` and maps it to a work_id, so both
    modes share one loop body. The mapping keeps work_id increasing in i --
    ``fwd_0 < rev_0 < fwd_1 < rev_1 < ...`` -- which the kernel's batch
    accumulator relies on, and validity is a prefix of that order, so n_items
    alone bounds the loop with no per-item guard.
    """
    b = len(seq_q_list)
    blk_m = [(sq + TS - 1) // TS for sq in seq_q_list]
    total_work = sum(m * n for m in blk_m)
    out = torch.zeros((num_cores, 4), dtype=torch.int32)

    # FLEX_SPLIT_MODE forces a mode for A/B measurement; unset picks by B*N.
    forced = os.environ.get("FLEX_SPLIT_MODE")
    strided = (b * n > num_cores) if forced is None else (forced == "1")
    if strided:
        two_c = 2 * num_cores
        for c in range(num_cores):
            fwd = len(range(c, total_work, two_c))
            rev = len(range(two_c - 1 - c, total_work, two_c))
            out[c] = torch.tensor([0, fwd + rev, 1, two_c], dtype=torch.int32)
        return out

    # ---- mode 0: split contiguously so each core's summed cost is ~equal ----
    # Cost of a work item is the number of KV blocks it walks, read straight off
    # the per-row bounds (the same expression the kernel evaluates on device).
    costs = []
    for bi in range(b):
        bound = hybrid_bounds(seq_q_list[bi], seq_kv_list[bi], spans_per_batch[bi])
        tile_cost = [(int(bound[min((s1 + 1) * TS, seq_q_list[bi]) - 1]) + TKV - 1) // TKV
                     for s1 in range(blk_m[bi])]
        for _n in range(n):
            costs.extend(tile_cost)

    def _num_parts(cap):
        parts, cur = 1, 0
        for c in costs:
            if c > cap:
                return total_work + 1                 # infeasible cap
            if cur + c > cap:
                parts, cur = parts + 1, 0
            cur += c
        return parts

    lo, hi = (max(costs) if costs else 0), (sum(costs) or 1)
    while lo < hi:                                    # smallest feasible max part
        mid = (lo + hi) // 2
        if _num_parts(mid) <= num_cores:
            hi = mid
        else:
            lo = mid + 1

    core, start, cur, i = 0, 0, 0, 0
    while i < total_work and core < num_cores:
        if cur + costs[i] > lo and cur > 0:
            out[core] = torch.tensor([start, i - start, 0, 0], dtype=torch.int32)
            core, start, cur = core + 1, i, 0
            continue
        cur += costs[i]
        i += 1
    if core < num_cores:
        out[core] = torch.tensor([start, total_work - start, 0, 0], dtype=torch.int32)
        core += 1
    for c in range(core, num_cores):
        out[c] = torch.tensor([total_work, 0, 0, 0], dtype=torch.int32)
    return out


def hybrid_mask(mm_prefix_range):
    """Mask spec: the operator's mm_prefix_range, verbatim."""
    return mm_prefix_range


@pytest.mark.soc("950")
def test_hybrid_bounds_match_doc_semantics():
    """Independently re-derive the bound rule the whole pipeline is built on.

    ``hybrid_bounds`` feeds the reference AND the core-cost model, and the kernel
    reimplements the same rule twice (scalar_bound for loop sizing,
    compute_row_bounds_vf for the element mask). A bug in the shared host
    formula would cancel out in the device test, so pin it here against a
    literal transcription of the operator's pseudocode.
    """
    for sq, skv, spans in (
        (512, 512, [(64, 128), (256, 320)]),
        (256, 512, [(300, 400)]),                   # right-down: spans in cache coords
        (256, 256, [(0, 255)]),                     # whole sequence bidirectional
        (512, 512, [(i * 32, i * 32 + 15) for i in range(HYBRID_MAX_SPANS)]),
        (128, 512, []),                             # no spans -> pure causal
    ):
        got = hybrid_bounds(sq, skv, spans)
        for r in range(sq):
            q_abs = (skv - sq) + r
            bound = q_abs + 1                       # doc: RIGHT_DOWN causal default
            for s, e in spans:
                if s <= q_abs <= e:                 # doc: bidirectional inside a span
                    bound = max(bound, e + 1)
            assert int(got[r]) == min(bound, skv), \
                "bound[%d] wrong (sq=%d skv=%d spans=%s)" % (r, sq, skv, spans)
        if spans:
            causal = hybrid_bounds(sq, skv, [])
            assert bool((got > causal).any()), "spans widened nothing -- case not discriminating"
            assert bool((got >= causal).all()), "spans must never narrow visibility"


@pytest.mark.soc("950")
def test_spans_padding_is_inert():
    """``-1`` padding slots must not change the mask, at any padding width."""
    sq = skv = 256
    real = [(32, 96), (160, 200)]
    base = hybrid_bounds(sq, skv, real)
    for pad_to in (2, 3, 8, HYBRID_MAX_SPANS):
        dev, per_batch = normalize_spans([[list(s) for s in real]], 1)
        assert dev.shape == (1, HYBRID_MAX_SPANS, 2)
        assert per_batch[0] == real, "padding slots leaked into the span list"
        assert torch.equal(hybrid_bounds(sq, skv, per_batch[0]), base)
    # None => no spans at all => pure right-down causal.
    dev, per_batch = normalize_spans(None, 3)
    assert per_batch == [[], [], []]
    assert int(dev.max()) == -1, "None must produce an all-padding tensor"
    assert torch.equal(hybrid_bounds(sq, skv, []),
                       torch.arange(sq, dtype=torch.int64) + (skv - sq) + 1)



def make_cu_seqlens_q(seq_q_list, device=None):
    """Build the operator's cu_seqlens_q: [B+1] int32 prefix sum, leading 0.

    The leading 0 is a hard requirement, not a convention: the kernel reads
    batch b's Q base straight out of cu_seqlens_q[b], so a nonzero first element
    would slide every request forward. Constructing it in one place keeps that
    invariant out of each caller's hands.
    """
    cu = [0] + list(itertools.accumulate(seq_q_list))
    assert cu[0] == 0, "cu_seqlens_q[0] must be 0"
    t = torch.tensor(cu, dtype=torch.int32)
    return t if device is None else t.to(device)


def build_paged_kv(k_tnd, v_tnd, seq_kv_list, block_size=BLOCK_SIZE):
    """Scatter contiguous TND K/V into a PagedAttention NHD cache + page table.

    Returns ``(kcache, vcache, block_ids)`` with the caches shaped
    ``[num_blocks, block_size, n_head_kv, dim]`` -- the operator's layout, where
    the inner three axes are N(tokens in the page) H(kv heads) D(head dim) --
    and ``block_ids`` shaped ``[b, max_blocks]``.

    Two deliberate choices make this discriminating rather than decorative:
    pages are handed out in *shuffled* order, and the pool is over-allocated and
    pre-filled with random data. A kernel that ignored the page table and walked
    the cache linearly, or that read the wrong page, would pick up that garbage
    instead of silently landing on the right tokens.
    """
    assert block_size % TKV == 0, \
        "block_size must be a multiple of TKV=%d (a smaller page would split a "\
        "KV tile across pages that need not be adjacent)" % TKV
    nkv, d = k_tnd.shape[1], k_tnd.shape[2]
    blocks_per = [(s + block_size - 1) // block_size for s in seq_kv_list]
    max_blocks = max(blocks_per)
    pool = sum(blocks_per) * 2 + 3                      # unused pages stay garbage
    kcache = torch.rand((pool, block_size, nkv, d), dtype=k_tnd.dtype)
    vcache = torch.rand((pool, block_size, nkv, d), dtype=v_tnd.dtype)
    block_ids = torch.zeros((len(seq_kv_list), max_blocks), dtype=torch.int32)

    perm = torch.randperm(pool)[:sum(blocks_per)]
    kv_off, page_i = 0, 0
    for bi, skv in enumerate(seq_kv_list):
        for j in range(blocks_per[bi]):
            page = int(perm[page_i])
            page_i += 1
            block_ids[bi, j] = page
            lo, hi = j * block_size, min((j + 1) * block_size, skv)
            kcache[page, :hi - lo] = k_tnd[kv_off + lo:kv_off + hi]
            vcache[page, :hi - lo] = v_tnd[kv_off + lo:kv_off + hi]
        kv_off += skv
    return kcache, vcache, block_ids


def _run_case(device, seq_q_list, seq_kv_list, n, nkv, mm_prefix_range, num_cores, atol):
    """Build inputs, launch the kernel, compare against the FA-flow reference.

    Returns (status, max_abs_diff, mask_desc, ctx) where ctx carries the CPU
    tensors so a caller can apply a second, independent accuracy criterion.
    """
    b, tq, tkv = len(seq_q_list), sum(seq_q_list), sum(seq_kv_list)
    assert n % nkv == 0, "n_head_q must be divisible by n_head_kv"
    for sq, skv in zip(seq_q_list, seq_kv_list):
        assert skv >= sq, "right-down alignment requires skv >= sq"

    q = torch.rand((tq, n, TD), device="cpu", dtype=torch.bfloat16)
    k = torch.rand((tkv, nkv, TD), device="cpu", dtype=torch.bfloat16)
    v = torch.rand((tkv, nkv, TD), device="cpu", dtype=torch.bfloat16)
    o = torch.zeros((tq, n, TD), device="cpu", dtype=torch.bfloat16)
    zero_mask = torch.zeros((TS_HALF, TKV), device="cpu", dtype=torch.uint8)

    # The only mask input the kernel gets is mm_prefix_range; the reference
    # masks are derived host-side from the same spans via the bound rule.
    spans_dev, spans_per_batch = normalize_spans(mm_prefix_range, b)
    masks = [dense_mask_from_bounds(hybrid_bounds(seq_q_list[bi], seq_kv_list[bi],
                                                  spans_per_batch[bi]), seq_kv_list[bi])
             for bi in range(b)]
    mask_desc = "causal" if mm_prefix_range is None else \
        "spans=%s" % [len(sp) for sp in spans_per_batch]

    # Page the KV cache. The reference keeps using the contiguous k/v, so the
    # comparison also checks that the kernel gathers the right pages back.
    kcache, vcache, block_ids = build_paged_kv(k, v, seq_kv_list)

    total_work = sum((x + TS - 1) // TS for x in seq_q_list) * n
    # Cost-balanced contiguous partition (see build_work_ranges): splits so
    # each core's summed KV-block cost is ~equal instead of its tile count.
    work_ranges = build_work_ranges(seq_q_list, seq_kv_list, spans_per_batch, n, num_cores)
    ################h2d###############
    # Operator formats, verbatim. seqlens_kvcache is [B] absolute lengths, NOT
    # a prefix sum; cu_seqlens_q is [B+1] with the mandatory leading 0.
    cu_seqlens_q = make_cu_seqlens_q(seq_q_list, device)
    seqlens_kvcache = torch.tensor(list(seq_kv_list), device=device, dtype=torch.int32)
    # Enforced at the launch boundary, so every device case re-checks it and a
    # future refactor that builds the tensor some other way cannot slip past.
    assert cu_seqlens_q.numel() == b + 1 and int(cu_seqlens_q[0]) == 0, \
        "cu_seqlens_q must be [B+1] with a leading 0"
    o = o.to(device)
    zero_mask = zero_mask.to(device)
    spans_dev = spans_dev.to(device)
    block_ids = block_ids.to(device)
    work_ranges = work_ranges.to(device)
    # The caches go in with the operator's own 4-D shape, untouched.
    kc = kcache.to(device)
    vc = vcache.to(device)
    flex_attention_bf16[None, min(num_cores, total_work)](
        q.to(device), kc, vc, o, cu_seqlens_q, block_ids, seqlens_kvcache,
        spans_dev, zero_mask, work_ranges)
    torch.npu.synchronize()

    o_ref = flash_attention_ref_tnd(q, k, v, seq_q_list, seq_kv_list, TD, masks)
    o = o.cpu()
    # Difference in fp32: a bf16 subtract would round the residual away.
    diff = (o.float() - o_ref.float()).abs().max().item()
    status = "PASS"
    try:
        # Guard against the all-zero regression (host `o` never written back).
        assert bool(o.abs().max().item() != 0.0), "kernel output is all-zero"
        torch.testing.assert_close(o, o_ref, rtol=atol, atol=atol)
    except AssertionError:
        status = "FAIL"
    return status, diff, mask_desc, dict(q=q, k=k, v=v, o=o, masks=masks)


@pytest.mark.soc("950")
def test_cu_seqlens_q_format():
    """Pin the cu_seqlens_q contract: [B+1] int32, prefix sum, leading 0.

    The leading 0 is required, so there is deliberately no device case for a
    shifted prefix sum: under this contract ``cu_seqlens_q[b]`` and "accumulate
    the lengths yourself" are the same function, and blessing a nonzero head
    would lock in behaviour the operator never promises. What can still go wrong
    is the caller building the tensor wrong, which is what this checks --
    make_cu_seqlens_q is the single place every case goes through.
    """
    for seq_q_list in ([512], [128, 256, 200], [1], [128] * 8):
        cu = make_cu_seqlens_q(seq_q_list)
        assert cu.dtype == torch.int32
        assert cu.numel() == len(seq_q_list) + 1, "cu_seqlens_q must be [B+1]"
        assert int(cu[0]) == 0, "cu_seqlens_q[0] must be 0"
        assert int(cu[-1]) == sum(seq_q_list), "last element must be total Q rows"
        lens = cu[1:] - cu[:-1]
        assert torch.equal(lens, torch.tensor(seq_q_list, dtype=torch.int32)), \
            "consecutive differences must be the per-request Q lengths"
        assert bool((lens >= 0).all()), "must be non-decreasing"


# Reference configuration from the operator doc, section 6: batch 1,
# sq = skv = 512, GQA 32 -> 4, head_dim 128, block_size 128, max_spans 3.
DOC_SQ, DOC_HQ, DOC_HKV = 512, 32, 4
DOC_COS_SIM = 0.99          # doc section 7 accuracy bar


def doc_spans(sq, skv):
    """The image spans the doc's minimal example builds, in cache coordinates."""
    start = skv - sq
    return [[start + sq // 8, start + sq // 4],
            [start + sq // 2, start + 5 * sq // 8],
            [-1, -1]]


@pytest.mark.soc("950")
def test_doc_precision_cases():
    """The operator's own accuracy matrix (doc section 7), at its own shapes.

    The main case list keeps head counts tiny so it stays fast, which means the
    documented configuration -- GQA 32 -> 4 at sq = skv = 512 with the section 6
    spans -- is never actually exercised as written. These five cases mirror the
    operator's test names one-for-one and run at that configuration.

    They are also graded by the *doc's* criterion rather than ours: cosine
    similarity >= 0.99 against a naive fused fp32 reference. That reference
    shares no structure with the blocked FA-flow one used elsewhere, so passing
    both means the kernel agrees with two independently derived goldens.
    """
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(42)
    # Guard the metric before trusting it: a tensor against itself must be
    # exactly 1. This catches reduction precision loss, which at these tensor
    # sizes is larger than the deviation the metric is supposed to resolve.
    probe = torch.rand(4_000_000, dtype=torch.float32)
    assert abs(cosine_similarity(probe, probe) - 1.0) < 1e-12, \
        "cosine_similarity is not exact for identical inputs"
    sq, hq, hkv = DOC_SQ, DOC_HQ, DOC_HKV
    cases = [
        # test_bf16_prefill_hybrid: causal + image spans.
        ("hybrid", [sq], [sq], [doc_spans(sq, sq)]),
        # test_bf16_prefill_hybrid_none_is_causal: None degenerates to causal.
        ("hybrid_none_is_causal", [sq], [sq], None),
        # test_bf16_prefill_hybrid_perbatch: per-request independent,
        # non-overlapping spans (heterogeneous multimodal batch).
        ("hybrid_perbatch", [sq] * 3, [sq] * 3,
         [[[32, 96], [160, 220], [-1, -1]],
          [[8, 40], [300, 380], [-1, -1]],
          [[128, 200], [400, 448], [-1, -1]]]),
        # test_bf16_prefill_hybrid_mixed_empty: some requests carry image spans,
        # some are all -1 (pure causal) in the same batch.
        ("hybrid_mixed_empty", [sq] * 3, [sq] * 3,
         [[[64, 192], [-1, -1], [-1, -1]],
          [[-1, -1], [-1, -1], [-1, -1]],
          [[256, 400], [-1, -1], [-1, -1]]]),
        # test_bf16_prefill_causal: the pure-causal control. The operator reaches
        # it by passing an explicit 2048x2048 mask tensor to its causal kernel;
        # here the same semantics are mm_prefix_range=None, so this differs from
        # hybrid_none_is_causal only in intent.
        ("causal", [sq], [sq], None),
    ]
    failures = []
    for name, seq_q_list, seq_kv_list, spans in cases:
        status, diff, _, ctx = _run_case(device, seq_q_list, seq_kv_list, hq, hkv,
                                         spans, 28, ATOL_BF16)
        o_fp32 = flash_attention_ref_fp32(ctx["q"], ctx["k"], ctx["v"],
                                          seq_q_list, seq_kv_list, TD, ctx["masks"])
        cos = cosine_similarity(ctx["o"], o_fp32)
        ok = status == "PASS" and cos >= DOC_COS_SIM
        # 1 - cos carries the signal; cos itself prints as 1.000000 at this accuracy.
        logging.info("  [%s] %-24s b=%d sq=%d nq=%d nkv=%d  1-cos=%.2e (bar %.0e)  max|diff|=%.5f",
                     "PASS" if ok else "FAIL", name, len(seq_q_list), sq, hq, hkv,
                     1.0 - cos, 1.0 - DOC_COS_SIM, diff)
        if not ok:
            failures.append((name, cos, diff, status))
    if failures:
        raise AssertionError("%d/%d doc precision cases failed: %s"
                             % (len(failures), len(cases), failures))


@pytest.mark.soc("950")
def test_paged_block_sizes():
    """The page size is free to be any multiple of TKV.

    The kernel reads it from ``k.shape[1]``: a page holds
    ``PAGE_TILES = block_size / TKV`` KV tiles, so tile ki lives in page
    ``ki // PAGE_TILES`` at token offset ``(ki % PAGE_TILES) * TKV``. With
    block_size == TKV both collapse to the 1:1 case, which is the only one the
    rest of the suite exercises -- hence this case. Every page size must give
    bit-identical output, since paging only moves where the same bytes live.
    """
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(7)
    seq_q_list, seq_kv_list, n, nkv = [512, 256], [512, 256], 8, 2
    spans = [[[64, 192], [-1, -1]], [[32, 100], [-1, -1]]]
    b, tq, tkv = len(seq_q_list), sum(seq_q_list), sum(seq_kv_list)
    q = torch.rand((tq, n, TD), device="cpu", dtype=torch.bfloat16)
    k = torch.rand((tkv, nkv, TD), device="cpu", dtype=torch.bfloat16)
    v = torch.rand((tkv, nkv, TD), device="cpu", dtype=torch.bfloat16)
    spans_dev, sper = normalize_spans(spans, b)
    masks = [dense_mask_from_bounds(hybrid_bounds(seq_q_list[i], seq_kv_list[i], sper[i]),
                                    seq_kv_list[i]) for i in range(b)]
    o_ref = flash_attention_ref_tnd(q, k, v, seq_q_list, seq_kv_list, TD, masks)

    outs = {}
    for block_size in (TKV, 2 * TKV, 4 * TKV):
        kcache, vcache, block_ids = build_paged_kv(k, v, seq_kv_list, block_size)
        assert kcache.shape[1] == block_size
        work_ranges = build_work_ranges(seq_q_list, seq_kv_list, sper, n, 28)
        o = torch.zeros((tq, n, TD), dtype=torch.bfloat16, device=device)
        total_work = sum((x + TS - 1) // TS for x in seq_q_list) * n
        flex_attention_bf16[None, min(28, total_work)](
            q.to(device), kcache.to(device), vcache.to(device), o,
            make_cu_seqlens_q(seq_q_list, device), block_ids.to(device),
            torch.tensor(list(seq_kv_list), device=device, dtype=torch.int32),
            spans_dev.to(device),
            torch.zeros((TS_HALF, TKV), device=device, dtype=torch.uint8),
            work_ranges.to(device))
        torch.npu.synchronize()
        outs[block_size] = o.cpu()
        diff = (outs[block_size].float() - o_ref.float()).abs().max().item()
        logging.info("  [%s] block_size=%d pages/req=%s max|diff|=%.5f",
                     "PASS" if diff <= ATOL_BF16 else "FAIL", block_size,
                     [(s + block_size - 1) // block_size for s in seq_kv_list], diff)
        torch.testing.assert_close(outs[block_size], o_ref, rtol=ATOL_BF16, atol=ATOL_BF16)
    for bsz, out in outs.items():
        assert torch.equal(out, outs[TKV]), \
            "block_size=%d differs from %d bit-for-bit" % (bsz, TKV)


@pytest.mark.soc("950")
def test_flex_attention_bf16_model_shapes():
    """The head counts and sequence lengths the operator actually ships with.

    The main case list keeps head counts tiny to stay fast, which leaves the
    real shapes -- HY-image-3.0 (32/8), HY-image-3.5 (32/4), HY3-VL (64/8) --
    and multi-KV-block Q tiles untested. Those stress work_ranges partitioning
    across many more work_ids and drive kv_loop well past a handful of blocks.
    """
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(42)
    cfgs = [
        ([512], [512], 32, 4, None),                                 # HY-image-3.0
        ([512], [512], 32, 8, None),                                 # HY-image-3.0
        ([512], [512], 32, 4, hybrid_mask([[[64, 192], [-1, -1]]])),  # HY-image-3.5
        ([512], [512], 64, 8,                                        # HY3-VL
         hybrid_mask([[[64, 192], [300, 400], [-1, -1]]])),
        ([1024, 512, 512, 256], [1024, 512, 512, 256], 32, 4,        # batch=4 heterogeneous
         hybrid_mask([[[32, 200]], [[-1, -1]], [[64, 300]], [[10, 100]]])),
        ([2048], [2048], 4, 1,                                       # kv_loop=16 blocks
         hybrid_mask([[[256, 512], [1024, 1280], [-1, -1]]])),
        ([1024, 512], [4096, 2048], 4, 1,                            # right-down, deep cache
         hybrid_mask([[[100, 900]], [[50, 400]]])),
        ([8192, 8192, 8192, 8192], [8192, 8192, 8192, 8192], 64, 8,                            # right-down, deep cache
         None),
    ]
    failures = []
    for cfg in cfgs:
        seq_q_list, seq_kv_list, n, nkv, spans = cfg
        status, diff, mask_desc, _ = _run_case(device, seq_q_list, seq_kv_list, n, nkv,
                                               spans, 28, ATOL_BF16)
        logging.info("  [%s] seq_q=%s seq_kv=%s nq=%d nkv=%d mask=%s max|diff|=%.5f",
                     status, seq_q_list, seq_kv_list, n, nkv, mask_desc, diff)
        if status == "FAIL":
            failures.append((cfg, diff))
    if failures:
        raise AssertionError("%d/%d model-shape cases failed: %s" % (len(failures), len(cfgs), failures))


@pytest.mark.soc("950")
def test_flex_attention_bf16_gqa():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(42)

    # Each cfg targets a specific code path of the kernel. The three softmax
    # branches in compute_p() are selected by the KV-tile width s2_size:
    #   s2_size == 128            -> process_vec1_nd_*_vf            (aligned)
    #   s2_size <= 64             -> process_vec1_nd_*_vf_unalign64
    #   64 < s2_size < 128        -> process_vec1_nd_*_vf_unalign
    # s2_size of a Q tile's last KV block == kv_end % 128, and kv_end is the
    # tile's largest bound -- so the spans themselves steer which branch runs.
    #
    # cfg = (seq_q_list, seq_kv_list, n_head_q, n_head_kv, d, mm_prefix_range, cores, atol)
    # n_head_q == n_head_kv is MHA (group 1); the GQA cases cover group 2/4/8,
    # including the 32->4 and 64->8 ratios the 950 PR operator uses. skv > sq is
    # prefill against an existing KV cache, which the operator mandates.
    failures = []
    test_cfgs = [
        # ---- pure causal (mm_prefix_range=None), covering the tiling paths ----
        ([512], [512], 1, 1, TD, None, 28, ATOL_BF16),  # baseline: s2=128 aligned path
        ([200], [200], 1, 1, TD, None, 28, ATOL_BF16),  # q-tail(72) + kv tail s2=72 -> unalign
        ([160], [160], 1, 1, TD, None, 28, ATOL_BF16),  # kv tail s2=32 -> unalign64 path
        ([100], [100], 1, 1, TD, None, 28, ATOL_BF16),  # single sub-128 block, s2=100 -> unalign
        ([50], [50], 1, 1, TD, None, 28, ATOL_BF16),  # single tiny block, s2=50 -> unalign64
        ([256], [256], 2, 2, TD, None, 28, ATOL_BF16),  # multi-head MHA n=2
        ([128, 256, 200], [128, 256, 200], 1, 1, TD, None, 28, ATOL_BF16),  # multi-batch varlen TND
        # ---- GQA ----
        ([256], [256], 4, 2, TD, None, 28, ATOL_BF16),  # GQA group=2
        ([256], [256], 8, 2, TD, None, 28, ATOL_BF16),  # GQA group=4 (HY-image-3.5 32->4)
        ([512], [512], 8, 1, TD, None, 28, ATOL_BF16),  # GQA group=8 (HY3-VL 64->8)
        ([200], [200], 4, 2, TD, None, 28, ATOL_BF16),  # GQA + q/kv tail -> unalign path
        ([128, 256, 200], [128, 256, 200], 8, 2, TD, None, 28, ATOL_BF16),  # GQA + varlen
        # ---- right-down alignment: prefill against an existing KV cache (skv > sq) ----
        ([128], [512], 1, 1, TD, None, 28, ATOL_BF16),  # 1 Q tile over 4 KV blocks
        ([256], [640], 1, 1, TD, None, 28, ATOL_BF16),  # kv tail s2=128 exact
        ([200], [456], 1, 1, TD, None, 28, ATOL_BF16),  # q tail + kv tail s2=72
        ([256], [512], 8, 2, TD, None, 28, ATOL_BF16),  # right-down + GQA group=4
        ([128, 200], [512, 456], 4, 2, TD, None, 28, ATOL_BF16),  # right-down varlen + GQA
        # ---- hybrid mask, mirroring the operator's own test matrix ----
        # test_bf16_prefill_hybrid: causal + image spans.
        ([512], [512], 1, 1, TD, hybrid_mask([[[64, 128], [256, 320], [-1, -1]]]), 28, ATOL_BF16),
        # Right-down hybrid: spans live in absolute (cache) coordinates, skv > sq.
        ([256], [512], 1, 1, TD, hybrid_mask([[[300, 400], [-1, -1], [-1, -1]]]), 28, ATOL_BF16),
        # test_bf16_prefill_hybrid_perbatch: per-request independent spans.
        ([256, 256], [256, 256], 1, 1, TD,
         hybrid_mask([[[32, 96], [160, 200]], [[8, 40], [130, 250]]]), 28, ATOL_BF16),
        # test_bf16_prefill_hybrid_mixed_empty: some requests have spans, some all -1.
        ([256, 256, 128], [256, 256, 128], 1, 1, TD,
         hybrid_mask([[[32, 200], [-1, -1]], [[-1, -1], [-1, -1]], [[16, 100], [-1, -1]]]), 28, ATOL_BF16),
        # A span spilling past the Q tile boundary, plus GQA.
        ([512], [512], 8, 2, TD, hybrid_mask([[[100, 300], [400, 500], [-1, -1]]]), 28, ATOL_BF16),
        # Full HYBRID_MAX_SPANS budget on one request.
        ([512], [512], 1, 1, TD,
         hybrid_mask([[[i * 32, i * 32 + 15] for i in range(HYBRID_MAX_SPANS)]]), 28, ATOL_BF16),
        # Spans shared across the batch via the [max_spans, 2] broadcast form.
        ([256, 256], [256, 256], 2, 1, TD, hybrid_mask([[40, 90], [150, 220]]), 28, ATOL_BF16),
        # Span reaching the very end -> last Q tile sees the whole KV range.
        ([256], [256], 1, 1, TD, hybrid_mask([[[0, 255], [-1, -1]]]), 28, ATOL_BF16),
        # q/kv tail together with spans -> unalign path under a widened bound.
        ([200], [200], 4, 2, TD, hybrid_mask([[[20, 150], [-1, -1]]]), 28, ATOL_BF16),
    ]
    for test_cfg in test_cfgs:
        seq_q_list, seq_kv_list, n, nkv, d, mm_prefix_range, num_cores, atol = test_cfg
        status, diff, mask_desc, _ = _run_case(device, seq_q_list, seq_kv_list, n, nkv,
                                               mm_prefix_range, num_cores, atol)
        if status == "FAIL":
            failures.append((test_cfg, diff))
        logging.info("  [%s] b=%d seq_q=%s seq_kv=%s nq=%d nkv=%d mask=%s atol=%.0e max|diff|=%.5f",
                     status, len(seq_q_list), seq_q_list, seq_kv_list, n, nkv, mask_desc, atol, diff)

    if failures:
        msg = "\n".join("  cfg=%s max|diff|=%.5f" % (c, d) for c, d in failures)
        raise AssertionError("%d/%d precision cases failed:\n%s" % (len(failures), len(test_cfgs), msg))
    logging.info("  PASS")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.npu.set_device(ST_DEVICE)
    logging.info("FlexAttention bf16 + GQA (QK_PRELOAD=%s, FIFO=%s)", QK_PRELOAD, FIFO_SIZE)
    logging.info("%s", '=' * 60)
    test_flex_attention_bf16_model_shapes()
    logging.info("\nAll tests passed!")
