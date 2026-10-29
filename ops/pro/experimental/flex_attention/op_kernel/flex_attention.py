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

"""FlashAttention (flex-attention) kernel with NBuffer + auto_mutex, driven by
a compile_mask.md mask_info contract (bitmask interval schedule only).

Refactored from test_fa_perf_tkv_preload.py to use:
  - NBuffer with current() auto-rotate cursor (no manual buf_idx)
  - auto_mutex=True for automatic Mutex synchronization
  - Cross-core event_id synchronization preserved (QK_READY, P_READY, PV_READY)

mask_info consumption:
  - descriptor (maskr/mask_start/mask_len, mask_num) -> in-kernel mask bitmap generation
    (decode_mask prefix-minus-gaps)
  - schedule -> tile-level compute decisions: bitmask interval (block_range
    + block_compute/block_mask bitmaps)

FP16/BF16 polymorphism: q/k/v/o are raw ``pl.Ptr[pl.DT_UINT8]`` params typed
via the ``datatype`` schema (io_dtype injected const).

Usage:
    python3 python/tests/st/pypto_pro/frontend/fa/test_flex_attention.py
"""

import dataclasses
import math

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField
import torch

# ================================================================
#  Configuration
# ================================================================
QK_PRELOAD = 3
FIFO_SIZE = QK_PRELOAD + 1

# Layout variant switch feeding the compile-time TilingKey IsDN field:
# False → ND (Q@K^T), True → DN (K@Q). Default ND because DN is slower.
# Flip to True to rebuild the DN variant (ND/DN cannot be mixed in one process).
IS_DN = False


# ================================================================
#  Tile dimensions and constants
# ================================================================
TS = 128
TKV = 128
TD = 128
TS_HALF = TS // 2
NEG_INF = -1e9
# Tail-row sentinel: strictly below NEG_INF so exp(tail - max) underflows to 0
# even when every valid row is masked to NEG_INF (fully-invisible tile). With
# NEG_INF itself, exp(NEG_INF - NEG_INF) = exp(0) = 1 would inflate gsum by
# (TKV - s2_size) and shrink the output by the same factor.
NEG_INF_TAIL = -1e10
SCALE = 1.0 / math.sqrt(TD)

# Descriptor slot capacity per decode pass (mask_start/mask_len channels).
# Rows whose gap count exceeds CHUNK are handled by the HoleChunk=1 tiling
# variant: the visibility bitmap is decoded in ceil(mask_num / CHUNK) passes
# (base pass + refine passes, see decode_mask_refine), so mask_num itself has
# no kernel-side upper bound -- only the host sanity limit MASK_NUM_LIMIT.
CHUNK = 16
# Host-side sanity cap on mask_num (the chunked decoder is unbounded; this
# only guards against runaway GM descriptor allocations).
MASK_NUM_LIMIT = 256

# Tiling struct (packed C-struct, passed as the last kernel argument, xattention
# OpTiling style). work_starts of core i yields [starts[i], starts[i+1]): the
# work_id ranges are contiguous, so only starts + a trailing total_work sentinel
# are needed (2N -> N+1 ints). Capacity 64 = 63 cores + sentinel.
MAX_TILING_CORES = 63


# ================================================================
#  TilingKey: single compile-time variant (required by binary delivery)
# ================================================================
class FlexAttnTilingKey:
    """Single-schedule kernel (mask_info interval/bitmask only).

    PyPTO binary delivery requires a tiling_key schema (pypto_compile.py:
    generate_binary_headers rejects kernels without one); the single value is
    folded away by the parser and shipped via SetTilingKey(0).
    """
    Schedule = TilingKeyField(bits=1, values=[0])
    Layout = TilingKeyField(bits=1, values=[0, 1])
    IsDN = TilingKeyField(bits=1, values=[0, 1])  # 0: ND (Q@K^T), 1: DN (K@Q)
    # 0: mask_num <= CHUNK, single-pass base decode.  1: mask_num > CHUNK,
    # ceil(mask_num / CHUNK)-pass decode (base + refine).  Compile-time so the
    # base binary carries no chunk/refine code at all.
    HoleChunk = TilingKeyField(bits=1, values=[0, 1])


@dataclasses.dataclass
class FlexAttnTiling:
    mask_num: int
    n_dim: int
    kv_heads: int
    batch: int                 # B: number of batch items
    r_f: int                   # max_block_m: Q-tile count of the max-batch
    w_f: int                   # max_block_n: sparse bitmap word count
    s1: int                    # bshd (Layout=1): uniform Q seq length; varlen: 0
    s2: int                    # bshd (Layout=1): uniform KV seq length; varlen: 0
    d: int                     # head dim (runtime; UB tiles stay allocated at TD=128)
    scale: float               # softmax scale 1/sqrt(d) (x_attention_v2 precedent)
    work_starts: "int[64]"     # per-core start + sentinel (work_starts[cores] = total_work)

BLOCK_STRIDE_ND = TS >> 1 | 0x1
REPEAT_STRIDE_ND = 1
BLOCK_STRIDE_DN = TKV >> 1 | 0x1  # 65
REPEAT_STRIDE_DN = 1
FLOAT_REP_SIZE = 64  # elements per fp32 register
# Head-dim generalization: d comes from tiling.d at runtime; the update VFs
# branch on d <= 64 (s1-unrolled single chunk) vs d > 64 (d-unrolled two
# chunks) with per-lane masks -- no D_LOOPS/TAIL_D constants or runtime
# derivatives remain.
REDUCE_SIZE = 1

# Buffer sizes (bytes)
Q_F16 = TS * TD * 2
KT_F16 = TD * TKV * 2
V_F16 = TKV * TD * 2
P_F16 = TS * TKV * 2
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

# ================================================================
#  Buffer addresses
# ================================================================
# MAT (512KB) - L1 buffers
MA0_Q = 0
MA1_K = Q_F16 * 2
MA2_P = MA1_K + KT_F16 * 2
MA3_V = MA2_P + P_F16 * 3

# L0A/L0B/L0C addresses
LA0 = 0
LA1 = P_F16
RA0 = 0
RA1 = KT_F16
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
VA14 = VA13 + VB_RED * 2
# mask_start/mask_len double buffers hold CHUNK descriptor slot channels each.
VA15 = VA14 + VB_RED * 2 * CHUNK
VA16 = VA15 + VB_RED * 2 * CHUNK
assert VA16 <= 248 * 1024

# Cross-core shared buffer IDs
QK_READY_FORWARD_IDS = (0, 1)
QK_READY_BARKWARD_IDS = (2, 3)
P_READY_FORWARD_IDS = (4, 5, 6)
PV_READY_FORWARD_IDS = (7, 8)
PV_READY_BARKWARD_IDS = (9, 10)


@pl.vector_function
def process_vec1_nd_no_update_vf_unalign64(input_tile, dst_tile, max_tile, max_tile_st, sum_tile, mask_tile,
                                           s1_size, s2_size, scale):
    preg_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(s2_size, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x = vf.muls(vreg_x, scale, preg_tail)

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

        vreg_exp_even_f16 = vf.astype(vreg_exp_even, preg_all_f16, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_dst_even_f16, vreg_dst_odd_f16 = vf.de_interleave(vreg_exp_even_f16, vreg_exp_even_f16)
        vf.store_align(dst_tile, vreg_dst_even_f16, preg_all_f16,
                       block_stride=BLOCK_STRIDE_ND, repeat_stride=REPEAT_STRIDE_ND,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
    vf.store_unalign_post(sum_tile, ureg_exp_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_no_update_vf_unalign(input_tile, dst_tile, max_tile, max_tile_st, sum_tile, mask_tile,
                                         s1_size, s2_size, scale):
    preg_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_mask_unroll = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(s2_size - 64, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x_unroll = vf.load_align(input_tile, m * TKV + 64)

        vreg_x = vf.muls(vreg_x, scale, preg_all)
        vreg_x_unroll = vf.muls(vreg_x_unroll, scale, preg_tail)

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

        vreg_exp_even_f16 = vf.astype(vreg_exp_even, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_exp_odd_f16 = vf.astype(vreg_exp_odd, preg_all, layout=pl.CastLayout.ONE, dtype=io_dtype)
        vreg_exp_f16 = vf.or_(vreg_exp_even_f16, vreg_exp_odd_f16, preg_all_f16)
        vf.store_align(dst_tile, vreg_exp_f16, preg_all_f16,
                       block_stride=BLOCK_STRIDE_ND, repeat_stride=REPEAT_STRIDE_ND,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
    vf.store_unalign_post(sum_tile, ureg_exp_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_no_update_vf(input_tile, dst_tile, max_tile, max_tile_st, sum_tile, mask_tile, s1_size, scale):
    preg_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_mask_unroll = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x_unroll = vf.load_align(input_tile, m * TKV + 64)

        vreg_x = vf.muls(vreg_x, scale, preg_all)
        vreg_x_unroll = vf.muls(vreg_x_unroll, scale, preg_all)

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

        vreg_exp_even_f16 = vf.astype(vreg_exp_even, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_exp_odd_f16 = vf.astype(vreg_exp_odd, preg_all, layout=pl.CastLayout.ONE, dtype=io_dtype)
        vreg_exp_f16 = vf.or_(vreg_exp_even_f16, vreg_exp_odd_f16, preg_all_f16)
        vf.store_align(dst_tile, vreg_exp_f16, preg_all_f16,
                       block_stride=BLOCK_STRIDE_ND, repeat_stride=REPEAT_STRIDE_ND,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
    vf.store_unalign_post(sum_tile, ureg_exp_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_update_vf_unalign64(input_tile, dst_tile, max_tile, mask_tile,
                                        tmp_max, tmp_max_st, tmp_exp_sum, s1_size, s2_size, scale):
    preg_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(s2_size, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x = vf.muls(vreg_x, scale, preg_tail)

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

        vreg_exp_even_f16 = vf.astype(vreg_exp, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_dst_even_f16, vreg_dst_odd_f16 = vf.de_interleave(vreg_exp_even_f16, vreg_exp_even_f16)
        vf.store_align(dst_tile, vreg_dst_even_f16, preg_all_f16,
                       block_stride=BLOCK_STRIDE_ND, repeat_stride=REPEAT_STRIDE_ND,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
    vf.store_unalign_post(tmp_exp_sum, ureg_exp_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_update_vf_unalign(input_tile, dst_tile, max_tile, mask_tile,
                                      tmp_max, tmp_max_st, tmp_exp_sum, s1_size, s2_size, scale):
    preg_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_mask_unroll = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(s2_size - 64, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x_unroll = vf.load_align(input_tile, m * TKV + 64)

        vreg_x = vf.muls(vreg_x, scale, preg_all)
        vreg_x_unroll = vf.muls(vreg_x_unroll, scale, preg_tail)

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

        vreg_exp_even_f16 = vf.astype(vreg_exp_even, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_exp_odd_f16 = vf.astype(vreg_exp_odd, preg_all, layout=pl.CastLayout.ONE, dtype=io_dtype)
        vreg_exp_f16 = vf.or_(vreg_exp_even_f16, vreg_exp_odd_f16, preg_all_f16)
        vf.store_align(dst_tile, vreg_exp_f16, preg_all_f16,
                       block_stride=BLOCK_STRIDE_ND, repeat_stride=REPEAT_STRIDE_ND,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
    vf.store_unalign_post(tmp_exp_sum, ureg_exp_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_update_vf(input_tile, dst_tile, max_tile, mask_tile,
                              tmp_max, tmp_max_st, tmp_exp_sum, s1_size, scale):
    preg_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_mask_unroll = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x_unroll = vf.load_align(input_tile, m * TKV + 64)

        vreg_x = vf.muls(vreg_x, scale, preg_all)
        vreg_x_unroll = vf.muls(vreg_x_unroll, scale, preg_all)

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

        vreg_exp_even_f16 = vf.astype(vreg_exp_even, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_exp_odd_f16 = vf.astype(vreg_exp_odd, preg_all, layout=pl.CastLayout.ONE, dtype=io_dtype)
        vreg_exp_f16 = vf.or_(vreg_exp_even_f16, vreg_exp_odd_f16, preg_all_f16)
        vf.store_align(dst_tile, vreg_exp_f16, preg_all_f16,
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
def flash_update_basic_vf_s1(dst_tile, cur_tile, pre_tile, exp_max_tile, s1_size, d):
    # Non-final flash-update (fp32), d <= 64 group: one 64-lane fp32 chunk per
    # row, compute/store masked to the d valid lanes (no garbage round-trip).
    # Unrolled along s1, two rows per iteration; when s1_size is odd the last
    # pair's second row is a phantom row -- in-allocation garbage read/write
    # clipped by o_f16's validshape downstream (same stale-data semantics as
    # the column direction).
    preg_d = vf.update_mask(d, dtype=pl.DT_FP32)
    for i2 in pl.range(0, (s1_size + 1) // 2):
        i = i2 * 2
        vreg_exp_max_a = vf.load_align(exp_max_tile, i * REDUCE_SIZE, dist=pl.LoadDist.BRC_B32)
        vreg_pre_a = vf.load_align(pre_tile, i * TD)
        vreg_cur_a = vf.load_align(cur_tile, i * TD)
        vreg_mul_a = vf.mul(vreg_exp_max_a, vreg_pre_a, preg_d)
        vreg_add_a = vf.add(vreg_mul_a, vreg_cur_a, preg_d)
        vf.store_align(dst_tile + i * TD, vreg_add_a, preg_d)
        vreg_exp_max_b = vf.load_align(exp_max_tile, (i + 1) * REDUCE_SIZE, dist=pl.LoadDist.BRC_B32)
        vreg_pre_b = vf.load_align(pre_tile, (i + 1) * TD)
        vreg_cur_b = vf.load_align(cur_tile, (i + 1) * TD)
        vreg_mul_b = vf.mul(vreg_exp_max_b, vreg_pre_b, preg_d)
        vreg_add_b = vf.add(vreg_mul_b, vreg_cur_b, preg_d)
        vf.store_align(dst_tile + (i + 1) * TD, vreg_add_b, preg_d)


@pl.vector_function
def flash_update_basic_vf_d(dst_tile, cur_tile, pre_tile, exp_max_tile, s1_size, d):
    # Non-final flash-update (fp32), d > 64 group: unrolled along d -- chunk0
    # (cols 0-63) is fully valid, chunk1 (cols 64-127) holds d-64 valid lanes;
    # per-chunk masks, no garbage compute.
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(d - 64, dtype=pl.DT_FP32)
    for i in pl.range(0, s1_size):
        vreg_exp_max = vf.load_align(exp_max_tile, i * REDUCE_SIZE, dist=pl.LoadDist.BRC_B32)
        vreg_pre0 = vf.load_align(pre_tile, i * TD)
        vreg_cur0 = vf.load_align(cur_tile, i * TD)
        vreg_mul0 = vf.mul(vreg_exp_max, vreg_pre0, preg_all)
        vreg_add0 = vf.add(vreg_mul0, vreg_cur0, preg_all)
        vf.store_align(dst_tile + i * TD, vreg_add0, preg_all)
        vreg_pre1 = vf.load_align(pre_tile, i * TD + FLOAT_REP_SIZE)
        vreg_cur1 = vf.load_align(cur_tile, i * TD + FLOAT_REP_SIZE)
        vreg_mul1 = vf.mul(vreg_exp_max, vreg_pre1, preg_tail)
        vreg_add1 = vf.add(vreg_mul1, vreg_cur1, preg_tail)
        vf.store_align(dst_tile + i * TD + FLOAT_REP_SIZE, vreg_add1, preg_tail)


@pl.vector_function
def flash_update_last_basic_vf(dst_tile, cur_tile, pre_tile, exp_max_tile, exp_sum_tile, s1_size, d):
    # Fused final flash-update: fp32 (pre*corr + cur) / gsum rounded to the IO
    # dtype in-register (CAST_ROUND, same rounding as the retired standalone
    # pl.cast pass), written straight into the f16/bf16 O tile. The row is
    # loaded once with DINTLV_B32 (even/odd 32-bit unit de-interleave, same
    # chain as the ND softmax paths): astype layout ZERO lands the even-column
    # results in each lane's low halfword and ONE the odd-column results in
    # the high halfword, so or_ merges them back into sequential column order
    # for a single NORM store. Compute/store are masked to the d valid columns
    # (even register ceil(d/2) lanes, odd register d/2 lanes, f16 store d
    # lanes) -- no garbage compute. cur/pre stay fp32.
    preg_even = vf.update_mask((d + 1) // 2, dtype=pl.DT_FP32)
    preg_odd = vf.update_mask(d // 2, dtype=pl.DT_FP32)
    preg_out_d = vf.update_mask(d, dtype=io_dtype)
    for i in pl.range(0, s1_size):
        vreg_exp_max = vf.load_align(exp_max_tile, i * REDUCE_SIZE, dist=pl.LoadDist.BRC_B32)
        vreg_exp_sum = vf.load_align(exp_sum_tile, i * REDUCE_SIZE, dist=pl.LoadDist.BRC_B32)
        vreg_pre_e, vreg_pre_o = vf.load_align(pre_tile, i * TD, dist=pl.LoadDist.DINTLV_B32)
        vreg_cur_e, vreg_cur_o = vf.load_align(cur_tile, i * TD, dist=pl.LoadDist.DINTLV_B32)
        vreg_mul_e = vf.mul(vreg_exp_max, vreg_pre_e, preg_even)
        vreg_add_e = vf.add(vreg_mul_e, vreg_cur_e, preg_even)
        vreg_div_e = vf.div(vreg_add_e, vreg_exp_sum, preg_even)
        vreg_a16 = vf.astype(vreg_div_e, preg_out_d, layout=pl.CastLayout.ZERO, dtype=io_dtype,
                             round_mode=pl.VFRoundMode.CAST_ROUND)
        vreg_mul_o = vf.mul(vreg_exp_max, vreg_pre_o, preg_odd)
        vreg_add_o = vf.add(vreg_mul_o, vreg_cur_o, preg_odd)
        vreg_div_o = vf.div(vreg_add_o, vreg_exp_sum, preg_odd)
        vreg_b16 = vf.astype(vreg_div_o, preg_out_d, layout=pl.CastLayout.ONE, dtype=io_dtype,
                             round_mode=pl.VFRoundMode.CAST_ROUND)
        vreg_full = vf.or_(vreg_a16, vreg_b16, preg_out_d)
        vf.store_align(dst_tile + i * TD, vreg_full, preg_out_d)


@pl.vector_function
def last_div_vf(dst_tile, cur_tile, exp_sum_tile, s1_size, d):
    # Fused final divide: fp32 cur / gsum rounded to the IO dtype in-register
    # (CAST_ROUND, same rounding as the retired standalone pl.cast pass) and
    # stored straight into the f16/bf16 O tile. Row loaded with DINTLV_B32
    # (even/odd 32-bit unit de-interleave, same chain as the ND softmax
    # paths): astype ZERO/ONE place even/odd columns in each lane's low/high
    # halfword, or_ merges them back into sequential column order for a
    # single NORM store. Compute/store masked to the d valid columns (even
    # register ceil(d/2) lanes, odd register d/2 lanes, f16 store d lanes).
    preg_even = vf.update_mask((d + 1) // 2, dtype=pl.DT_FP32)
    preg_odd = vf.update_mask(d // 2, dtype=pl.DT_FP32)
    preg_out_d = vf.update_mask(d, dtype=io_dtype)
    for i in pl.range(0, s1_size):
        vreg_exp_sum = vf.load_align(exp_sum_tile, i * REDUCE_SIZE, dist=pl.LoadDist.BRC_B32)
        vreg_cur_e, vreg_cur_o = vf.load_align(cur_tile, i * TD, dist=pl.LoadDist.DINTLV_B32)
        vreg_div_e = vf.div(vreg_cur_e, vreg_exp_sum, preg_even)
        vreg_a16 = vf.astype(vreg_div_e, preg_out_d, layout=pl.CastLayout.ZERO, dtype=io_dtype,
                             round_mode=pl.VFRoundMode.CAST_ROUND)
        vreg_div_o = vf.div(vreg_cur_o, vreg_exp_sum, preg_odd)
        vreg_b16 = vf.astype(vreg_div_o, preg_out_d, layout=pl.CastLayout.ONE, dtype=io_dtype,
                             round_mode=pl.VFRoundMode.CAST_ROUND)
        vreg_full = vf.or_(vreg_a16, vreg_b16, preg_out_d)
        vf.store_align(dst_tile + i * TD, vreg_full, preg_out_d)


@pl.vector_function
def decode_mask(mask, maskr, mask_start, mask_len, col, s1_size, mask_num):
    """Generate the [TS_HALF, TKV] visibility bitmap from the descriptor.

    mask_info interval contract (maskgen-aligned): V(row) = [0, maskr) minus the
    masked gaps \u22c3_h [mask_start_h, mask_start_h + mask_len_h). Gap h, row m lives at
    offset h * TS_HALF + m in the packed buffers; mask_len == 0 marks an unused
    slot (a no-op).
    """
    index = vf.arange(col, dtype=pl.DT_INT32)
    index_unroll = vf.arange(col + 64, dtype=pl.DT_INT32)
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_all_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    merge_bit = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    merge_unroll_bit = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    row_reg = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    temp_reg = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    vreg_zero = vf.full(0, dtype=pl.DT_UINT16)
    vreg_one = vf.full(1, dtype=pl.DT_UINT16)
    for m in pl.range(0, s1_size):
        # Start from the right bound: visible where c < maskr.
        maskr_reg = vf.load_align(maskr, m, dist=pl.LoadDist.BRC_B32, dtype=pl.DT_INT32)
        merge_bit = vf.lt(index, maskr_reg, preg_all)
        merge_unroll_bit = vf.lt(index_unroll, maskr_reg, preg_all)
        # Remove each masked gap [start, start + len): AND NOT (c in gap).
        for h in pl.range(0, mask_num):
            start_reg = vf.load_align(mask_start, h * TS_HALF + m, dist=pl.LoadDist.BRC_B32, dtype=pl.DT_INT32)
            len_reg = vf.load_align(mask_len, h * TS_HALF + m, dist=pl.LoadDist.BRC_B32, dtype=pl.DT_INT32)
            hi = vf.add(start_reg, len_reg, preg_all)
            lo_bit = vf.ge(index, start_reg, preg_all)
            hi_bit = vf.lt(index, hi, preg_all)
            in_gap = vf.and_(lo_bit, hi_bit, preg_all)
            out_bit = vf.not_(in_gap, preg_all)
            merge_bit = vf.and_(merge_bit, out_bit, preg_all)
            lo_unroll_bit = vf.ge(index_unroll, start_reg, preg_all)
            hi_unroll_bit = vf.lt(index_unroll, hi, preg_all)
            in_gap_unroll = vf.and_(lo_unroll_bit, hi_unroll_bit, preg_all)
            out_unroll_bit = vf.not_(in_gap_unroll, preg_all)
            merge_unroll_bit = vf.and_(merge_unroll_bit, out_unroll_bit, preg_all)
        row_reg, temp_reg = vf.de_interleave(merge_bit, merge_unroll_bit, dtype=pl.DT_UINT16)
        mask_b16 = vf.select(vreg_zero, vreg_one, row_reg)
        vf.store_align(mask + (m * TKV), mask_b16, preg_all_b16, dist=pl.StoreDist.PACK)


@pl.vector_function
def decode_mask_refine(mask, mask_start, mask_len, col, s1_size, hole_cnt):
    """
    HoleChunk=1 refine pass: AND one loaded chunk of gap planes into the
    already-decoded [s1_size, TKV] bitmap (HoleChunk>0 passes only, holes
    [ck*CHUNK, ck*CHUNK+hole_cnt)).  The stored bitmap is re-read as a DS
    predicate (1 == masked, softmax's readback contract), merged in the
    masked domain (masked | in_gap), and re-packed.  Same row/interleave/
    PACK contract as decode_mask.
    """
    index = vf.arange(col, dtype=pl.DT_INT32)
    index_unroll = vf.arange(col + 64, dtype=pl.DT_INT32)
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_all_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    masked_bit = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    masked_unroll_bit = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    row_reg = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    temp_reg = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    vreg_one = vf.full(1, dtype=pl.DT_UINT16)
    vreg_zero = vf.full(0, dtype=pl.DT_UINT16)
    for m in pl.range(0, s1_size):
        # Stored bitmap as predicate: bit set == masked (1 byte per column).
        masked_bit = vf.load_align(mask, m * TKV, dist=pl.LoadDist.DS)
        masked_unroll_bit = vf.load_align(mask, m * TKV + 64, dist=pl.LoadDist.DS)
        # Remove this chunk's gaps: masked |= in_gap.
        for h in pl.range(0, hole_cnt):
            start_reg = vf.load_align(mask_start, h * TS_HALF + m, dist=pl.LoadDist.BRC_B32, dtype=pl.DT_INT32)
            len_reg = vf.load_align(mask_len, h * TS_HALF + m, dist=pl.LoadDist.BRC_B32, dtype=pl.DT_INT32)
            hi = vf.add(start_reg, len_reg, preg_all)
            lo_bit = vf.ge(index, start_reg, preg_all)
            hi_bit = vf.lt(index, hi, preg_all)
            in_gap = vf.and_(lo_bit, hi_bit, preg_all)
            masked_bit = vf.or_(masked_bit, in_gap, preg_all)
            lo_unroll_bit = vf.ge(index_unroll, start_reg, preg_all)
            hi_unroll_bit = vf.lt(index_unroll, hi, preg_all)
            in_gap_unroll = vf.and_(lo_unroll_bit, hi_unroll_bit, preg_all)
            masked_unroll_bit = vf.or_(masked_unroll_bit, in_gap_unroll, preg_all)
        row_reg, temp_reg = vf.de_interleave(masked_bit, masked_unroll_bit, dtype=pl.DT_UINT16)
        # select(a, b, preg) = preg ? a : b; masked lanes store 1.
        mask_b16 = vf.select(vreg_one, vreg_zero, row_reg)
        vf.store_align(mask + (m * TKV), mask_b16, preg_all_b16, dist=pl.StoreDist.PACK)


@pl.vector_function
def fill_zero_mask(mask_tile, loop_count, stride):
    """Write loop_count all-zero rows: row i at offset i*stride (elements).
    Handles both ND (loop_count=TS_HALF, stride=TKV) and DN
    (loop_count=TKV, stride=TS_HALF) layouts.
    """
    preg_all_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    vreg_zero = vf.full(0, dtype=pl.DT_UINT16)
    for i in pl.range(0, loop_count):
        vf.store_align(mask_tile + (i * stride), vreg_zero, preg_all_b16, dist=pl.StoreDist.PACK)


# ================================================================
#  DN (K@Q) 版本 VF 函数：mask 列循环 + online softmax
#  数据布局：每核 [s2_size, TS_HALF]，行=KV 位置，列=Q 位置
# ================================================================
@pl.vector_function
def decode_mask_dn_vf(mask, maskr, mask_start, mask_len, col, s1_size, s2_size, mask_num):
    """Generate an [s2_size, s1_size] uint8 bitmap (DN layout: row = KV position).

    Column-wise generation: the loop variable c (KV position) is broadcast as a
    scalar and compared against the per-Q maskr/start/len vectors, which are
    pre-loaded once (no per-row BRC loads).  Output row c holds the visibility
    of all Q positions at KV position c, byte-per-Q (0 == visible).
    """
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_all_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    row_reg = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    vreg_zero = vf.full(0, dtype=pl.DT_UINT16)
    vreg_one = vf.full(1, dtype=pl.DT_UINT16)
    maskr_vec = vf.load_align(maskr, 0, dtype=pl.DT_INT32)
    for c in pl.range(0, s2_size):
        c_bcast = vf.full(col + c, dtype=pl.DT_INT32)
        merge_bit = vf.lt(c_bcast, maskr_vec, preg_all)
        for h in pl.range(0, mask_num):
            start_vec = vf.load_align(mask_start, h * s1_size, dtype=pl.DT_INT32)
            len_vec = vf.load_align(mask_len, h * s1_size, dtype=pl.DT_INT32)
            hi = vf.add(start_vec, len_vec, preg_all)
            lo_bit = vf.ge(c_bcast, start_vec, preg_all)
            hi_bit = vf.lt(c_bcast, hi, preg_all)
            in_gap = vf.and_(lo_bit, hi_bit, preg_all)
            out_bit = vf.not_(in_gap, preg_all)
            merge_bit = vf.and_(merge_bit, out_bit, preg_all)
        row_reg, _ = vf.de_interleave(merge_bit, merge_bit, dtype=pl.DT_UINT16)
        mask_b16 = vf.select(vreg_zero, vreg_one, row_reg)
        vf.store_align(mask + (c * s1_size), mask_b16, preg_all_b16, dist=pl.StoreDist.PACK)
    # Tail rows (s2_size..TKV) are written visible (0) so the softmax select
    # keeps the tail QK at NEG_INF_TAIL: exp(tail - max) underflows to 0 even
    # when max == NEG_INF (fully-invisible tile). Leftover NBuffer data with
    # bit=1 (invisible) would instead swap the tail QK back to NEG_INF and
    # contribute exp(0)=1 to gsum.
    for c in pl.range(s2_size, TKV):
        vf.store_align(mask + (c * s1_size), vreg_zero, preg_all_b16, dist=pl.StoreDist.PACK)


@pl.vector_function
def decode_mask_dn_refine(mask, mask_start, mask_len, col, s1_size, s2_size, hole_cnt):
    """
    HoleChunk=1 refine pass on the DN bitmap (row = KV position): DS-read
    each stored row as a masked-domain predicate, OR in this chunk's gaps,
    re-pack.  Rows [0, s2_size) only -- the tail rows written visible by the
    base pass are never touched.
    """
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_all_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    masked_bit = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    vreg_one = vf.full(1, dtype=pl.DT_UINT16)
    vreg_zero = vf.full(0, dtype=pl.DT_UINT16)
    for c in pl.range(0, s2_size):
        c_bcast = vf.full(col + c, dtype=pl.DT_INT32)
        masked_bit = vf.load_align(mask, c * s1_size, dist=pl.LoadDist.DS)
        for h in pl.range(0, hole_cnt):
            start_vec = vf.load_align(mask_start, h * s1_size, dtype=pl.DT_INT32)
            len_vec = vf.load_align(mask_len, h * s1_size, dtype=pl.DT_INT32)
            hi = vf.add(start_vec, len_vec, preg_all)
            lo_bit = vf.ge(c_bcast, start_vec, preg_all)
            hi_bit = vf.lt(c_bcast, hi, preg_all)
            in_gap = vf.and_(lo_bit, hi_bit, preg_all)
            masked_bit = vf.or_(masked_bit, in_gap, preg_all)
        row_reg, _ = vf.de_interleave(masked_bit, masked_bit, dtype=pl.DT_UINT16)
        mask_b16 = vf.select(vreg_one, vreg_zero, row_reg)
        vf.store_align(mask + (c * s1_size), mask_b16, preg_all_b16, dist=pl.StoreDist.PACK)


@pl.vector_function
def softmax_dn_no_update_vf(input_tile, dst_tile, max_tile, sum_tile, mask_tile, s1_size, s2_size, scale):
    """Online softmax on a [s2_size, s1_size] QK tile (DN layout).

    max/sum are 64-lane register accumulators (one lane per Q position); they
    never round-trip through UB between the two passes, so no per-row BRC max
    load (PLDI) is emitted.
    """
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)
    vreg_min = vf.full(NEG_INF, dtype=pl.DT_FP32)
    vreg_tail = vf.full(NEG_INF_TAIL, dtype=pl.DT_FP32)

    # Zero the UB tail rows (s2_size..TKV) so the fixed 32x4-row loop is safe.
    for i in pl.range(s2_size, TKV):
        vf.store_align(input_tile + (i * s1_size), vreg_tail, preg_all)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)

    # ---- pass 1: per-Q max accumulation (register) ----
    vreg_max = vf.full(NEG_INF, dtype=pl.DT_FP32)
    preg_tmp0 = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    preg_tmp1 = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    preg_tmp2 = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    preg_tmp3 = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    for kv4 in pl.range(0, TKV // 4):
        src0 = vf.load_align(input_tile, kv4 * s1_size)
        src1 = vf.load_align(input_tile, (TKV // 4 + kv4) * s1_size)
        src2 = vf.load_align(input_tile, (TKV // 2 + kv4) * s1_size)
        src3 = vf.load_align(input_tile, (3 * TKV // 4 + kv4) * s1_size)
        src0 = vf.muls(src0, scale, preg_all)
        src1 = vf.muls(src1, scale, preg_all)
        src2 = vf.muls(src2, scale, preg_all)
        src3 = vf.muls(src3, scale, preg_all)
        preg_tmp0 = vf.load_align(mask_tile, kv4 * s1_size, dist=pl.LoadDist.DS)
        preg_tmp1 = vf.load_align(mask_tile, (TKV // 4 + kv4) * s1_size, dist=pl.LoadDist.DS)
        preg_tmp2 = vf.load_align(mask_tile, (TKV // 2 + kv4) * s1_size, dist=pl.LoadDist.DS)
        preg_tmp3 = vf.load_align(mask_tile, (3 * TKV // 4 + kv4) * s1_size, dist=pl.LoadDist.DS)
        src0 = vf.select(vreg_min, src0, preg_tmp0)
        src1 = vf.select(vreg_min, src1, preg_tmp1)
        src2 = vf.select(vreg_min, src2, preg_tmp2)
        src3 = vf.select(vreg_min, src3, preg_tmp3)
        vf.store_align(input_tile + kv4 * s1_size, src0, preg_all)
        vf.store_align(input_tile + (TKV // 4 + kv4) * s1_size, src1, preg_all)
        vf.store_align(input_tile + (TKV // 2 + kv4) * s1_size, src2, preg_all)
        vf.store_align(input_tile + (3 * TKV // 4 + kv4) * s1_size, src3, preg_all)
        vreg_max = vf.max(vreg_max, src0, preg_all)
        vreg_max = vf.max(vreg_max, src1, preg_all)
        vreg_max = vf.max(vreg_max, src2, preg_all)
        vreg_max = vf.max(vreg_max, src3, preg_all)
    vf.store_align(max_tile, vreg_max, preg_all)

    # ---- pass 2: exp + per-Q sum (register max, no BRC) ----
    vreg_sum = vf.full(0, dtype=pl.DT_FP32)
    for kv4 in pl.range(0, TKV // 4):
        src0 = vf.load_align(input_tile, kv4 * s1_size)
        src1 = vf.load_align(input_tile, (TKV // 4 + kv4) * s1_size)
        src2 = vf.load_align(input_tile, (TKV // 2 + kv4) * s1_size)
        src3 = vf.load_align(input_tile, (3 * TKV // 4 + kv4) * s1_size)
        exp0 = vf.exp_sub(src0, vreg_max, preg_all)
        exp1 = vf.exp_sub(src1, vreg_max, preg_all)
        exp2 = vf.exp_sub(src2, vreg_max, preg_all)
        exp3 = vf.exp_sub(src3, vreg_max, preg_all)
        vreg_sum = vf.add(vreg_sum, exp0, preg_all)
        vreg_sum = vf.add(vreg_sum, exp1, preg_all)
        vreg_sum = vf.add(vreg_sum, exp2, preg_all)
        vreg_sum = vf.add(vreg_sum, exp3, preg_all)

        e0_f16 = vf.astype(exp0, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        e2_f16 = vf.astype(exp2, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        pack_even, _ = vf.de_interleave(e0_f16, e2_f16, dtype=io_dtype)
        e1_f16 = vf.astype(exp1, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        e3_f16 = vf.astype(exp3, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        pack_odd, _ = vf.de_interleave(e1_f16, e3_f16, dtype=io_dtype)
        vf.store_align(dst_tile, pack_even, preg_all_f16,
                       block_stride=BLOCK_STRIDE_DN, repeat_stride=REPEAT_STRIDE_DN,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
        vf.store_align(dst_tile + 512, pack_odd, preg_all_f16,
                       block_stride=BLOCK_STRIDE_DN, repeat_stride=REPEAT_STRIDE_DN,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
    vf.store_align(sum_tile, vreg_sum, preg_all)


@pl.vector_function
def softmax_dn_update_vf(input_tile, dst_tile, max_tile, sum_tile, exp_corr_tile, mask_tile, s1_size, s2_size, scale):
    """Online softmax (DN) for non-first KV blocks: merge this tile's max/sum
    with the running global max/sum (one old-max/sum UB load per tile, not per
    row), emitting the exp correction factor for the PV flash-update.
    """
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)
    vreg_min = vf.full(NEG_INF, dtype=pl.DT_FP32)
    vreg_tail = vf.full(NEG_INF_TAIL, dtype=pl.DT_FP32)

    for i in pl.range(s2_size, TKV):
        vf.store_align(input_tile + (i * s1_size), vreg_tail, preg_all)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)

    # ---- pass 1: per-Q max accumulation ----
    vreg_max = vf.full(NEG_INF, dtype=pl.DT_FP32)
    preg_tmp0 = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    preg_tmp1 = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    preg_tmp2 = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    preg_tmp3 = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    for kv4 in pl.range(0, TKV // 4):
        src0 = vf.load_align(input_tile, kv4 * s1_size)
        src1 = vf.load_align(input_tile, (TKV // 4 + kv4) * s1_size)
        src2 = vf.load_align(input_tile, (TKV // 2 + kv4) * s1_size)
        src3 = vf.load_align(input_tile, (3 * TKV // 4 + kv4) * s1_size)
        src0 = vf.muls(src0, scale, preg_all)
        src1 = vf.muls(src1, scale, preg_all)
        src2 = vf.muls(src2, scale, preg_all)
        src3 = vf.muls(src3, scale, preg_all)
        preg_tmp0 = vf.load_align(mask_tile, kv4 * s1_size, dist=pl.LoadDist.DS)
        preg_tmp1 = vf.load_align(mask_tile, (TKV // 4 + kv4) * s1_size, dist=pl.LoadDist.DS)
        preg_tmp2 = vf.load_align(mask_tile, (TKV // 2 + kv4) * s1_size, dist=pl.LoadDist.DS)
        preg_tmp3 = vf.load_align(mask_tile, (3 * TKV // 4 + kv4) * s1_size, dist=pl.LoadDist.DS)
        src0 = vf.select(vreg_min, src0, preg_tmp0)
        src1 = vf.select(vreg_min, src1, preg_tmp1)
        src2 = vf.select(vreg_min, src2, preg_tmp2)
        src3 = vf.select(vreg_min, src3, preg_tmp3)
        vf.store_align(input_tile + kv4 * s1_size, src0, preg_all)
        vf.store_align(input_tile + (TKV // 4 + kv4) * s1_size, src1, preg_all)
        vf.store_align(input_tile + (TKV // 2 + kv4) * s1_size, src2, preg_all)
        vf.store_align(input_tile + (3 * TKV // 4 + kv4) * s1_size, src3, preg_all)
        vreg_max = vf.max(vreg_max, src0, preg_all)
        vreg_max = vf.max(vreg_max, src1, preg_all)
        vreg_max = vf.max(vreg_max, src2, preg_all)
        vreg_max = vf.max(vreg_max, src3, preg_all)

    # ---- merge with running global max/sum (once per tile) ----
    vreg_old_max = vf.load_align(max_tile, 0)
    vreg_old_sum = vf.load_align(sum_tile, 0)
    vreg_max_new = vf.max(vreg_old_max, vreg_max, preg_all)
    vreg_corr = vf.exp_sub(vreg_old_max, vreg_max_new, preg_all)
    vf.store_align(max_tile, vreg_max_new, preg_all)
    vf.store_align(exp_corr_tile, vreg_corr, preg_all)

    # ---- pass 2: exp (register max) + sum with correction ----
    vreg_sum = vf.full(0, dtype=pl.DT_FP32)
    for kv4 in pl.range(0, TKV // 4):
        src0 = vf.load_align(input_tile, kv4 * s1_size)
        src1 = vf.load_align(input_tile, (TKV // 4 + kv4) * s1_size)
        src2 = vf.load_align(input_tile, (TKV // 2 + kv4) * s1_size)
        src3 = vf.load_align(input_tile, (3 * TKV // 4 + kv4) * s1_size)
        exp0 = vf.exp_sub(src0, vreg_max_new, preg_all)
        exp1 = vf.exp_sub(src1, vreg_max_new, preg_all)
        exp2 = vf.exp_sub(src2, vreg_max_new, preg_all)
        exp3 = vf.exp_sub(src3, vreg_max_new, preg_all)
        vreg_sum = vf.add(vreg_sum, exp0, preg_all)
        vreg_sum = vf.add(vreg_sum, exp1, preg_all)
        vreg_sum = vf.add(vreg_sum, exp2, preg_all)
        vreg_sum = vf.add(vreg_sum, exp3, preg_all)

        e0_f16 = vf.astype(exp0, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        e2_f16 = vf.astype(exp2, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        pack_even, _ = vf.de_interleave(e0_f16, e2_f16, dtype=io_dtype)
        e1_f16 = vf.astype(exp1, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        e3_f16 = vf.astype(exp3, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        pack_odd, _ = vf.de_interleave(e1_f16, e3_f16, dtype=io_dtype)
        vf.store_align(dst_tile, pack_even, preg_all_f16,
                       block_stride=BLOCK_STRIDE_DN, repeat_stride=REPEAT_STRIDE_DN,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
        vf.store_align(dst_tile + 512, pack_odd, preg_all_f16,
                       block_stride=BLOCK_STRIDE_DN, repeat_stride=REPEAT_STRIDE_DN,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
    vreg_corr_mul = vf.mul(vreg_corr, vreg_old_sum, preg_all)
    vreg_sum_new = vf.add(vreg_corr_mul, vreg_sum, preg_all)
    vf.store_align(sum_tile, vreg_sum_new, preg_all)


@pl.vector_function
def softmax_dn_no_update_vf_tail64(input_tile, dst_tile, max_tile, sum_tile, mask_tile, s1_size, s2_size, scale):
    """DN softmax for tail blocks (s2_size <= 64). Fixed 64-row loop, 256 offset.

    Padding rows (s2_size..63) get QK=NEG_INF_TAIL + mask=visible(0) so
    exp(pad - max) ≈ 0 regardless of the mask state.  The NZ tile layout is
    identical to the full-size version, but pack_odd is written at +256 (16×16)
    instead of +512 (32×16).
    """
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)
    preg_all_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    vreg_min = vf.full(NEG_INF, dtype=pl.DT_FP32)
    vreg_tail = vf.full(NEG_INF_TAIL, dtype=pl.DT_FP32)
    vreg_zero_b16 = vf.full(0, dtype=pl.DT_UINT16)

    # Padding rows: QK minimal + mask visible(0) → select keeps QK minimal.
    for i in pl.range(s2_size, 64):
        vf.store_align(input_tile + (i * s1_size), vreg_tail, preg_all)
        vf.store_align(mask_tile + (i * s1_size), vreg_zero_b16, preg_all_b16, dist=pl.StoreDist.PACK)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)

    # ---- pass 1: per-Q max accumulation ----
    vreg_max = vf.full(NEG_INF, dtype=pl.DT_FP32)
    preg_tmp0 = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    preg_tmp1 = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    preg_tmp2 = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    preg_tmp3 = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    for kv4 in pl.range(0, 16):
        src0 = vf.load_align(input_tile, (kv4 + 0 * 16) * s1_size)
        src1 = vf.load_align(input_tile, (kv4 + 1 * 16) * s1_size)
        src2 = vf.load_align(input_tile, (kv4 + 2 * 16) * s1_size)
        src3 = vf.load_align(input_tile, (kv4 + 3 * 16) * s1_size)
        src0 = vf.muls(src0, scale, preg_all)
        src1 = vf.muls(src1, scale, preg_all)
        src2 = vf.muls(src2, scale, preg_all)
        src3 = vf.muls(src3, scale, preg_all)
        preg_tmp0 = vf.load_align(mask_tile, (kv4 + 0 * 16) * s1_size, dist=pl.LoadDist.DS)
        preg_tmp1 = vf.load_align(mask_tile, (kv4 + 1 * 16) * s1_size, dist=pl.LoadDist.DS)
        preg_tmp2 = vf.load_align(mask_tile, (kv4 + 2 * 16) * s1_size, dist=pl.LoadDist.DS)
        preg_tmp3 = vf.load_align(mask_tile, (kv4 + 3 * 16) * s1_size, dist=pl.LoadDist.DS)
        src0 = vf.select(vreg_min, src0, preg_tmp0)
        src1 = vf.select(vreg_min, src1, preg_tmp1)
        src2 = vf.select(vreg_min, src2, preg_tmp2)
        src3 = vf.select(vreg_min, src3, preg_tmp3)
        vf.store_align(input_tile + (kv4 + 0 * 16) * s1_size, src0, preg_all)
        vf.store_align(input_tile + (kv4 + 1 * 16) * s1_size, src1, preg_all)
        vf.store_align(input_tile + (kv4 + 2 * 16) * s1_size, src2, preg_all)
        vf.store_align(input_tile + (kv4 + 3 * 16) * s1_size, src3, preg_all)
        vreg_max = vf.max(vreg_max, src0, preg_all)
        vreg_max = vf.max(vreg_max, src1, preg_all)
        vreg_max = vf.max(vreg_max, src2, preg_all)
        vreg_max = vf.max(vreg_max, src3, preg_all)
    vf.store_align(max_tile, vreg_max, preg_all)

    # ---- pass 2: exp + per-Q sum ----
    vreg_sum = vf.full(0, dtype=pl.DT_FP32)
    for kv4 in pl.range(0, 16):
        src0 = vf.load_align(input_tile, (kv4 + 0 * 16) * s1_size)
        src1 = vf.load_align(input_tile, (kv4 + 1 * 16) * s1_size)
        src2 = vf.load_align(input_tile, (kv4 + 2 * 16) * s1_size)
        src3 = vf.load_align(input_tile, (kv4 + 3 * 16) * s1_size)
        exp0 = vf.exp_sub(src0, vreg_max, preg_all)
        exp1 = vf.exp_sub(src1, vreg_max, preg_all)
        exp2 = vf.exp_sub(src2, vreg_max, preg_all)
        exp3 = vf.exp_sub(src3, vreg_max, preg_all)
        vreg_sum = vf.add(vreg_sum, exp0, preg_all)
        vreg_sum = vf.add(vreg_sum, exp1, preg_all)
        vreg_sum = vf.add(vreg_sum, exp2, preg_all)
        vreg_sum = vf.add(vreg_sum, exp3, preg_all)

        e0_f16 = vf.astype(exp0, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        e2_f16 = vf.astype(exp2, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        pack_even, _ = vf.de_interleave(e0_f16, e2_f16, dtype=io_dtype)
        e1_f16 = vf.astype(exp1, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        e3_f16 = vf.astype(exp3, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        pack_odd, _ = vf.de_interleave(e1_f16, e3_f16, dtype=io_dtype)
        vf.store_align(dst_tile, pack_even, preg_all_f16,
                       block_stride=BLOCK_STRIDE_DN, repeat_stride=REPEAT_STRIDE_DN,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
        vf.store_align(dst_tile + 256, pack_odd, preg_all_f16,
                       block_stride=BLOCK_STRIDE_DN, repeat_stride=REPEAT_STRIDE_DN,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
    vf.store_align(sum_tile, vreg_sum, preg_all)


@pl.vector_function
def softmax_dn_update_vf_tail64(input_tile, dst_tile, max_tile, sum_tile, exp_corr_tile, mask_tile,
                                s1_size, s2_size, scale):
    """DN softmax (non-first KV block) for tail blocks, fixed 64-row loop."""
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)
    preg_all_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    vreg_min = vf.full(NEG_INF, dtype=pl.DT_FP32)
    vreg_tail = vf.full(NEG_INF_TAIL, dtype=pl.DT_FP32)
    vreg_zero_b16 = vf.full(0, dtype=pl.DT_UINT16)

    for i in pl.range(s2_size, 64):
        vf.store_align(input_tile + (i * s1_size), vreg_tail, preg_all)
        vf.store_align(mask_tile + (i * s1_size), vreg_zero_b16, preg_all_b16, dist=pl.StoreDist.PACK)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)

    # ---- pass 1: per-Q max accumulation ----
    vreg_max = vf.full(NEG_INF, dtype=pl.DT_FP32)
    preg_tmp0 = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    preg_tmp1 = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    preg_tmp2 = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    preg_tmp3 = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    for kv4 in pl.range(0, 16):
        src0 = vf.load_align(input_tile, (kv4 + 0 * 16) * s1_size)
        src1 = vf.load_align(input_tile, (kv4 + 1 * 16) * s1_size)
        src2 = vf.load_align(input_tile, (kv4 + 2 * 16) * s1_size)
        src3 = vf.load_align(input_tile, (kv4 + 3 * 16) * s1_size)
        src0 = vf.muls(src0, scale, preg_all)
        src1 = vf.muls(src1, scale, preg_all)
        src2 = vf.muls(src2, scale, preg_all)
        src3 = vf.muls(src3, scale, preg_all)
        preg_tmp0 = vf.load_align(mask_tile, (kv4 + 0 * 16) * s1_size, dist=pl.LoadDist.DS)
        preg_tmp1 = vf.load_align(mask_tile, (kv4 + 1 * 16) * s1_size, dist=pl.LoadDist.DS)
        preg_tmp2 = vf.load_align(mask_tile, (kv4 + 2 * 16) * s1_size, dist=pl.LoadDist.DS)
        preg_tmp3 = vf.load_align(mask_tile, (kv4 + 3 * 16) * s1_size, dist=pl.LoadDist.DS)
        src0 = vf.select(vreg_min, src0, preg_tmp0)
        src1 = vf.select(vreg_min, src1, preg_tmp1)
        src2 = vf.select(vreg_min, src2, preg_tmp2)
        src3 = vf.select(vreg_min, src3, preg_tmp3)
        vf.store_align(input_tile + (kv4 + 0 * 16) * s1_size, src0, preg_all)
        vf.store_align(input_tile + (kv4 + 1 * 16) * s1_size, src1, preg_all)
        vf.store_align(input_tile + (kv4 + 2 * 16) * s1_size, src2, preg_all)
        vf.store_align(input_tile + (kv4 + 3 * 16) * s1_size, src3, preg_all)
        vreg_max = vf.max(vreg_max, src0, preg_all)
        vreg_max = vf.max(vreg_max, src1, preg_all)
        vreg_max = vf.max(vreg_max, src2, preg_all)
        vreg_max = vf.max(vreg_max, src3, preg_all)

    # ---- merge with running global max/sum ----
    vreg_old_max = vf.load_align(max_tile, 0)
    vreg_old_sum = vf.load_align(sum_tile, 0)
    vreg_max_new = vf.max(vreg_old_max, vreg_max, preg_all)
    vreg_corr = vf.exp_sub(vreg_old_max, vreg_max_new, preg_all)
    vf.store_align(max_tile, vreg_max_new, preg_all)
    vf.store_align(exp_corr_tile, vreg_corr, preg_all)

    # ---- pass 2: exp + sum with correction ----
    vreg_sum = vf.full(0, dtype=pl.DT_FP32)
    for kv4 in pl.range(0, 16):
        src0 = vf.load_align(input_tile, (kv4 + 0 * 16) * s1_size)
        src1 = vf.load_align(input_tile, (kv4 + 1 * 16) * s1_size)
        src2 = vf.load_align(input_tile, (kv4 + 2 * 16) * s1_size)
        src3 = vf.load_align(input_tile, (kv4 + 3 * 16) * s1_size)
        exp0 = vf.exp_sub(src0, vreg_max_new, preg_all)
        exp1 = vf.exp_sub(src1, vreg_max_new, preg_all)
        exp2 = vf.exp_sub(src2, vreg_max_new, preg_all)
        exp3 = vf.exp_sub(src3, vreg_max_new, preg_all)
        vreg_sum = vf.add(vreg_sum, exp0, preg_all)
        vreg_sum = vf.add(vreg_sum, exp1, preg_all)
        vreg_sum = vf.add(vreg_sum, exp2, preg_all)
        vreg_sum = vf.add(vreg_sum, exp3, preg_all)

        e0_f16 = vf.astype(exp0, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        e2_f16 = vf.astype(exp2, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        pack_even, _ = vf.de_interleave(e0_f16, e2_f16, dtype=io_dtype)
        e1_f16 = vf.astype(exp1, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        e3_f16 = vf.astype(exp3, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        pack_odd, _ = vf.de_interleave(e1_f16, e3_f16, dtype=io_dtype)
        vf.store_align(dst_tile, pack_even, preg_all_f16,
                       block_stride=BLOCK_STRIDE_DN, repeat_stride=REPEAT_STRIDE_DN,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
        vf.store_align(dst_tile + 256, pack_odd, preg_all_f16,
                       block_stride=BLOCK_STRIDE_DN, repeat_stride=REPEAT_STRIDE_DN,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
    vreg_corr_mul = vf.mul(vreg_corr, vreg_old_sum, preg_all)
    vreg_sum_new = vf.add(vreg_corr_mul, vreg_sum, preg_all)
    vf.store_align(sum_tile, vreg_sum_new, preg_all)


def compute_qk(ctx, ki, sq_off, q, k, cur_q_slot, k_l1_db, left_db, right_db, acc_db, qk_vec_db, task_id,
               buf_a, buf_b, buf_c, d):
    # --- compute_qk inlined ---
    skv_off = ctx.s2_size_acc + ctx.ki * TKV
    cur_k_slot = k_l1_db.next()
    qk_left = left_db[buf_a]
    qk_right = right_db[buf_b]
    qk_acc = acc_db[buf_c]

    if ctx.loop_count == 0:
        pl.set_validshape(cur_q_slot, [ctx.s1_size, d])
        pl.load(cur_q_slot, q, [sq_off, ctx.q_n, 0], order=[0, 2])
    pl.set_validshape(cur_k_slot, [d, ctx.s2_size])
    pl.load(cur_k_slot, k, [skv_off, ctx.k_n, 0], order=[2, 0])

    pl.set_validshape(qk_left, [ctx.s1_size, d])
    pl.move(qk_left, cur_q_slot)
    pl.set_validshape(qk_right, [d, ctx.s2_size])
    pl.move(qk_right, cur_k_slot)
    pl.set_validshape(qk_acc, [ctx.s1_size, ctx.s2_size])
    pl.matmul(qk_acc, qk_left, qk_right)

    qk_slot = qk_vec_db.next()
    pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_BARKWARD_IDS[task_id % 2])
    pl.set_validshape(qk_slot, [(ctx.s1_size + 1) // 2, ctx.s2_size])
    pl.set_validshape(qk_acc, [(ctx.s1_size + 1) // 2 * 2, (ctx.s2_size + 7) // 8 * 8])
    pl.move(qk_slot, qk_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_FORWARD_IDS[task_id % 2])


def compute_pv(ctx, v_l1_db, p_mat_db, left_db, right_db, acc_db, v, pv_dst, buf_a, buf_b, buf_c, d):
    sv_off = ctx.s2_size_acc + ctx.ki * TKV
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=P_READY_FORWARD_IDS[ctx.task_id % 3])
    cur_v_slot = v_l1_db.next()
    # # current() advances p_mat_db cursor to stay in sync with Vec,
    cur_p_slot = p_mat_db.next()
    pv_left = left_db[buf_a]
    pv_right = right_db[buf_b]
    pv_acc = acc_db[buf_c]

    pl.set_validshape(cur_v_slot, [ctx.s2_size, d])
    pl.load(cur_v_slot, v, [sv_off, ctx.k_n, 0], order=[0, 2])
    pl.set_validshape(cur_p_slot, [ctx.s1_size, ctx.s2_size])
    pl.set_validshape(pv_left, [ctx.s1_size, ctx.s2_size])
    pl.move(pv_left, cur_p_slot)
    pl.set_validshape(pv_right, [ctx.s2_size, d])
    pl.move(pv_right, cur_v_slot)
    pl.set_validshape(pv_acc, [ctx.s1_size, d])
    pl.matmul(pv_acc, pv_left, pv_right)

    # pv_dst is the next pv_vec_db slot (caller-resolved). Block-0's PV stays
    # in this slot and is consumed by the next GU via previous() (deferred
    # init -- no running_o move on the Vector side).
    pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_BARKWARD_IDS[ctx.task_id % 2])
    pl.set_validshape(pv_dst, [(ctx.s1_size + 1) // 2, d])
    pl.set_validshape(pv_acc, [(ctx.s1_size + 1) // 2 * 2, d])
    pl.move(pv_dst, pv_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_FORWARD_IDS[ctx.task_id % 2])


def compute_p(ctx_p, sub_id, qk_vec_db, tile_nz_db, tmp_max, tmp_sum,
              global_max, global_sum, exp_corr_db, p_mat_db, mask_db,
              maskr_db, mask_start_db, mask_len_db, maskr_gm, mask_start_gm, mask_len_gm,
              scale):
    """Softmax on KQ tile -> P. Includes cross-core sync."""
    p_eid = ctx_p.task_id % 2
    q_idx_p = ctx_p.q_count % 3

    qk_slot = qk_vec_db.next()
    tile_nz = tile_nz_db.next()
    mask_buf = mask_db.next()
    maskr_buf = maskr_db.next()
    mask_start_buf = mask_start_db.next()
    mask_len_buf = mask_len_db.next()
    gmax_p = global_max[q_idx_p]
    gsum_p = global_sum[q_idx_p]
    exp_diff = exp_corr_db[ctx_p.task_id % 3]
    row_offset = ctx_p.first_s1 * sub_id
    row = ctx_p.s1_idx * TS
    col = ctx_p.ki * TKV

    if ctx_p.mask_bit == 0:
        # Fully-visible block: no per-element decode. Synthesize an all-zero
        # [TS_HALF, TKV] mask (0 == visible) directly on the vector unit --
        # the legacy path loaded a host-allocated GM zero buffer; that GM
        # tensor is now retired in favour of in-UB synthesis (no GM traffic).
        pl.set_validshape(mask_buf, [TS_HALF, TKV])
        fill_zero_mask(mask_buf, TS_HALF, TKV)
    else:
        # sub_id splits the Q tile's rows into two halves; each subblock must read
        # its own half's per-row descriptor slots, so offset the GM row by row_offset.
        mask_row = row + row_offset
        pl.set_validshape(maskr_buf, [1, ctx_p.half_s1])
        pl.load(maskr_buf, maskr_gm, [ctx_p.b_idx, mask_row])
        if HoleChunk == 0:  # noqa: F821
            # Single pass: all mask_num <= CHUNK gap planes resident at once.
            if ctx_p.mask_num > 0:
                # mask_start/mask_len GM inputs are None when mask_num == 0 (contract conditional).
                pl.set_validshape(mask_start_buf, [ctx_p.mask_num, ctx_p.half_s1])
                pl.load(mask_start_buf, mask_start_gm, [ctx_p.b_idx, 0, mask_row], order=[1, 2])
                pl.set_validshape(mask_len_buf, [ctx_p.mask_num, ctx_p.half_s1])
                pl.load(mask_len_buf, mask_len_gm, [ctx_p.b_idx, 0, mask_row], order=[1, 2])
            decode_mask(mask_buf, maskr_buf, mask_start_buf, mask_len_buf, col, ctx_p.half_s1, ctx_p.mask_num)
        else:
            # Chunked decode (mask_num > CHUNK): base pass + refine passes.
            # Host guarantees mask_num > CHUNK here, so every chunk is non-empty.
            n_chunks = (ctx_p.mask_num + CHUNK - 1) // CHUNK
            for ck in pl.range(0, n_chunks):
                holes_ck = pl.min(CHUNK, ctx_p.mask_num - ck * CHUNK)
                pl.set_validshape(mask_start_buf, [holes_ck, ctx_p.half_s1])
                pl.load(mask_start_buf, mask_start_gm, [ctx_p.b_idx, ck * CHUNK, mask_row], order=[1, 2])
                pl.set_validshape(mask_len_buf, [holes_ck, ctx_p.half_s1])
                pl.load(mask_len_buf, mask_len_gm, [ctx_p.b_idx, ck * CHUNK, mask_row], order=[1, 2])
                if ck == 0:
                    decode_mask(mask_buf, maskr_buf, mask_start_buf, mask_len_buf, col, ctx_p.half_s1, holes_ck)
                else:
                    decode_mask_refine(mask_buf, mask_start_buf, mask_len_buf, col, ctx_p.half_s1, holes_ck)

    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_FORWARD_IDS[p_eid])
    if ctx_p.loop_count == 0:
        if ctx_p.s2_size == 128:
            process_vec1_nd_no_update_vf(qk_slot, tile_nz, gmax_p, gmax_p, gsum_p, mask_buf, ctx_p.half_s1, scale)
        elif ctx_p.s2_size <= 64:
            process_vec1_nd_no_update_vf_unalign64(qk_slot, tile_nz, gmax_p, gmax_p, gsum_p, mask_buf,
                                                   ctx_p.half_s1, ctx_p.s2_size, scale)
        else:
            process_vec1_nd_no_update_vf_unalign(qk_slot, tile_nz, gmax_p, gmax_p, gsum_p, mask_buf,
                                         ctx_p.half_s1, ctx_p.s2_size, scale)
    else:
        if ctx_p.s2_size == 128:
            process_vec1_nd_update_vf(qk_slot, tile_nz, gmax_p, mask_buf, tmp_max, tmp_max, tmp_sum,
                                      ctx_p.half_s1, scale)
        elif ctx_p.s2_size <= 64:
            process_vec1_nd_update_vf_unalign64(qk_slot, tile_nz, gmax_p, mask_buf, tmp_max, tmp_max, tmp_sum,
                                                ctx_p.half_s1, ctx_p.s2_size, scale)
        else:
            process_vec1_nd_update_vf_unalign(qk_slot, tile_nz, gmax_p, mask_buf, tmp_max, tmp_max, tmp_sum,
                                              ctx_p.half_s1, ctx_p.s2_size, scale)
        update_exp_sum(exp_diff, gmax_p, tmp_max, gsum_p, tmp_sum)
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BARKWARD_IDS[p_eid])

    cur_p_slot = p_mat_db.next()
    pl.set_validshape(tile_nz, [ctx_p.half_s1, ctx_p.s2_size])
    pl.insert(cur_p_slot, tile_nz, [row_offset, 0])
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY_FORWARD_IDS[ctx_p.task_id % 3])


def compute_gu(ctx_gu, pv_vec_db, exp_corr_db, global_max_buf, global_sum_buf, running_o, o_f16, o,
               tensor_smax, tensor_ssum, stats_max_buf, stats_sum_buf, d):
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_FORWARD_IDS[ctx_gu.task_id % 2])

    sub_id = pl.get_subblock_idx()
    row_offset = ctx_gu.first_s1 * sub_id
    gmax_gu = global_max_buf[ctx_gu.q_count % 3]
    gsum_gu = global_sum_buf[ctx_gu.q_count % 3]
    stats_max = stats_max_buf[ctx_gu.q_count % 3]
    stats_sum = stats_sum_buf[ctx_gu.q_count % 3]
    exp_corr_gu = exp_corr_db[ctx_gu.task_id % 3]
    pv_slot = pv_vec_db.next()
    pl.set_validshape(running_o, [ctx_gu.half_s1, d])
    pl.set_validshape(pv_slot, [ctx_gu.half_s1, d])
    if ctx_gu.loop_count == 0:
        # Deferred init: block-0 PV stays in pv_slot. A single-block tile
        # finalizes straight from it; otherwise the next GU consumes it via
        # previous() as flash_update 'pre'. No BARKWARD fire here -- TMOV(k+2)
        # reuses this slot only after that GU releases it (bounded 1-task wait).
        if ctx_gu.ki == ctx_gu.kv_loop - 1:
            last_div_vf(o_f16, pv_slot, gsum_gu, ctx_gu.half_s1, d)
            pl.set_validshape(o_f16, [ctx_gu.half_s1, d])
            pl.store(o, o_f16, [ctx_gu.q_off + row_offset, ctx_gu.n_idx, 0], order=[0, 2])
            pl.set_validshape(stats_max, [1, ctx_gu.half_s1])
            pl.store(tensor_smax, stats_max, [ctx_gu.n_idx, ctx_gu.q_off + row_offset], order=[0, 1])
            pl.set_validshape(stats_sum, [1, ctx_gu.half_s1])
            pl.store(tensor_ssum, stats_sum, [ctx_gu.n_idx, ctx_gu.q_off + row_offset], order=[0, 1])
            pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[ctx_gu.task_id % 2])
    elif ctx_gu.loop_count == 1:
        # First accumulation: running_o's first write -- PV0 x corr + PV1
        # (math-identical to the retired init move + in-place update).
        pv_prev = pv_vec_db.previous()
        if ctx_gu.ki < ctx_gu.kv_loop - 1:
            if d <= 64:
                flash_update_basic_vf_s1(running_o, pv_slot, pv_prev, exp_corr_gu, ctx_gu.half_s1, d)
            else:
                flash_update_basic_vf_d(running_o, pv_slot, pv_prev, exp_corr_gu, ctx_gu.half_s1, d)
        else:
            flash_update_last_basic_vf(o_f16, pv_slot, pv_prev, exp_corr_gu, gsum_gu, ctx_gu.half_s1, d)
            pl.set_validshape(o_f16, [ctx_gu.half_s1, d])
            pl.store(o, o_f16, [ctx_gu.q_off + row_offset, ctx_gu.n_idx, 0], order=[0, 2])
            pl.set_validshape(stats_max, [1, ctx_gu.half_s1])
            pl.store(tensor_smax, stats_max, [ctx_gu.n_idx, ctx_gu.q_off + row_offset], order=[0, 1])
            pl.set_validshape(stats_sum, [1, ctx_gu.half_s1])
            pl.store(tensor_ssum, stats_sum, [ctx_gu.n_idx, ctx_gu.q_off + row_offset], order=[0, 1])
        # Release this block's slot AND the deferred block-0 slot (read above).
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[ctx_gu.task_id % 2])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[(ctx_gu.task_id - 1) % 2])
    else:
        if ctx_gu.ki < ctx_gu.kv_loop - 1:
            if d <= 64:
                flash_update_basic_vf_s1(running_o, pv_slot, running_o, exp_corr_gu, ctx_gu.half_s1, d)
            else:
                flash_update_basic_vf_d(running_o, pv_slot, running_o, exp_corr_gu, ctx_gu.half_s1, d)
        else:
            flash_update_last_basic_vf(o_f16, pv_slot, running_o, exp_corr_gu, gsum_gu, ctx_gu.half_s1, d)
            pl.set_validshape(o_f16, [ctx_gu.half_s1, d])
            pl.store(o, o_f16, [ctx_gu.q_off + row_offset, ctx_gu.n_idx, 0], order=[0, 2])
            pl.set_validshape(stats_max, [1, ctx_gu.half_s1])
            pl.store(tensor_smax, stats_max, [ctx_gu.n_idx, ctx_gu.q_off + row_offset], order=[0, 1])
            pl.set_validshape(stats_sum, [1, ctx_gu.half_s1])
            pl.store(tensor_ssum, stats_sum, [ctx_gu.n_idx, ctx_gu.q_off + row_offset], order=[0, 1])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[ctx_gu.task_id % 2])


def compute_gu_dn(ctx_gu, sub_id, pv_vec_db, exp_corr_db, global_max_buf, global_sum_buf, running_o, o_f16, o,
                  tensor_smax, tensor_ssum, stats_max_buf, stats_sum_buf, d):
    """DN flash-update + output store, 64-row (TS_HALF) aligned per subblock.

    The DN PV matmul produces [TS, d] split by DualModeSplitM into [TS_HALF, d]
    per subblock, so running_o / gsum / exp_corr lanes are all Q-column aligned
    (Q 64*sub_id + i). Unlike compute_gu (ND), the row count is always TS_HALF,
    never half_s1. O store rows beyond the real s1_size are clipped.
    """
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_FORWARD_IDS[ctx_gu.task_id % 2])

    row_offset = TS_HALF * sub_id
    gmax_gu = global_max_buf[ctx_gu.q_count % 3]
    gsum_gu = global_sum_buf[ctx_gu.q_count % 3]
    stats_max = stats_max_buf[ctx_gu.q_count % 3]
    stats_sum = stats_sum_buf[ctx_gu.q_count % 3]
    exp_corr_gu = exp_corr_db[ctx_gu.task_id % 3]
    pv_slot = pv_vec_db.next()
    pl.set_validshape(running_o, [TS_HALF, d])
    pl.set_validshape(pv_slot, [TS_HALF, d])
    if ctx_gu.loop_count == 0:
        # Deferred init: block-0 PV stays in pv_slot. Single-block tile
        # finalizes straight from it; otherwise the next GU consumes it via
        # previous() as flash_update 'pre'. No BARKWARD fire here -- TMOV(k+2)
        # reuses this slot only after that GU releases it (bounded 1-task wait).
        if ctx_gu.ki == ctx_gu.kv_loop - 1:
            last_div_vf(o_f16, pv_slot, gsum_gu, TS_HALF, d)
            if ctx_gu.s1_size > row_offset:
                valid_rows = pl.min(TS_HALF, ctx_gu.s1_size - row_offset)
                pl.set_validshape(o_f16, [valid_rows, d])
                pl.store(o, o_f16, [ctx_gu.q_off + row_offset, ctx_gu.n_idx, 0], order=[0, 2])
                pl.set_validshape(stats_max, [1, valid_rows])
                pl.store(tensor_smax, stats_max, [ctx_gu.n_idx, ctx_gu.q_off + row_offset], order=[0, 1])
                pl.set_validshape(stats_sum, [1, valid_rows])
                pl.store(tensor_ssum, stats_sum, [ctx_gu.n_idx, ctx_gu.q_off + row_offset], order=[0, 1])
            pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[ctx_gu.task_id % 2])
    elif ctx_gu.loop_count == 1:
        # First accumulation: running_o's first write -- PV0 x corr + PV1
        # (math-identical to the retired init move + in-place update).
        pv_prev = pv_vec_db.previous()
        if ctx_gu.ki < ctx_gu.kv_loop - 1:
            if d <= 64:
                flash_update_basic_vf_s1(running_o, pv_slot, pv_prev, exp_corr_gu, TS_HALF, d)
            else:
                flash_update_basic_vf_d(running_o, pv_slot, pv_prev, exp_corr_gu, TS_HALF, d)
        else:
            flash_update_last_basic_vf(o_f16, pv_slot, pv_prev, exp_corr_gu, gsum_gu, TS_HALF, d)
            if ctx_gu.s1_size > row_offset:
                valid_rows = pl.min(TS_HALF, ctx_gu.s1_size - row_offset)
                pl.set_validshape(o_f16, [valid_rows, d])
                pl.store(o, o_f16, [ctx_gu.q_off + row_offset, ctx_gu.n_idx, 0], order=[0, 2])
                pl.set_validshape(stats_max, [1, valid_rows])
                pl.store(tensor_smax, stats_max, [ctx_gu.n_idx, ctx_gu.q_off + row_offset], order=[0, 1])
                pl.set_validshape(stats_sum, [1, valid_rows])
                pl.store(tensor_ssum, stats_sum, [ctx_gu.n_idx, ctx_gu.q_off + row_offset], order=[0, 1])
        # Release this block's slot AND the deferred block-0 slot (read above).
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[ctx_gu.task_id % 2])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[(ctx_gu.task_id - 1) % 2])
    else:
        if ctx_gu.ki < ctx_gu.kv_loop - 1:
            if d <= 64:
                flash_update_basic_vf_s1(running_o, pv_slot, running_o, exp_corr_gu, TS_HALF, d)
            else:
                flash_update_basic_vf_d(running_o, pv_slot, running_o, exp_corr_gu, TS_HALF, d)
        else:
            flash_update_last_basic_vf(o_f16, pv_slot, running_o, exp_corr_gu, gsum_gu, TS_HALF, d)
            if ctx_gu.s1_size > row_offset:
                valid_rows = pl.min(TS_HALF, ctx_gu.s1_size - row_offset)
                pl.set_validshape(o_f16, [valid_rows, d])
                pl.store(o, o_f16, [ctx_gu.q_off + row_offset, ctx_gu.n_idx, 0], order=[0, 2])
                pl.set_validshape(stats_max, [1, valid_rows])
                pl.store(tensor_smax, stats_max, [ctx_gu.n_idx, ctx_gu.q_off + row_offset], order=[0, 1])
                pl.set_validshape(stats_sum, [1, valid_rows])
                pl.store(tensor_ssum, stats_sum, [ctx_gu.n_idx, ctx_gu.q_off + row_offset], order=[0, 1])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[ctx_gu.task_id % 2])


# ================================================================
#  DN (K@Q) compute functions
# ================================================================
def compute_qk_dn(ctx, ki, sq_off, q, k, cur_q_slot, k_l1_db, left_db, right_db, acc_db, qk_vec_db, task_id,
                  cur_q_rhs_slot, buf_a, buf_b, buf_c, d):
    """K@Q matmul: K [s2,d] @ Q^T [d,s1] → ACC [s2,s1]; DualModeSplitN → per-core [s2, s1/2].
    K loads into cur_q_slot (q_l1_db ND layout, fixed per Q tile). Q^T uses cur_q_rhs_slot (k_l1_db ZN, loaded once).
    """
    skv_off = ctx.s2_size_acc + ctx.ki * TKV
    qk_left = left_db[buf_a]
    qk_right = right_db[buf_b]
    qk_acc = acc_db[buf_c]

    # K: left matrix into q_l1_db (ND layout), loaded every loop
    pl.set_validshape(cur_q_slot, [ctx.s2_size, d])
    pl.load(cur_q_slot, k, [skv_off, ctx.k_n, 0], order=[0, 2])
    # Q^T: right matrix into cur_q_rhs_slot (k_l1_db ZN layout), loaded once (loop_count==0) and reused
    if ctx.loop_count == 0:
        pl.set_validshape(cur_q_rhs_slot, [d, ctx.s1_size])
        pl.load(cur_q_rhs_slot, q, [sq_off, ctx.q_n, 0], order=[2, 0])

    pl.set_validshape(qk_left, [ctx.s2_size, d])
    pl.move(qk_left, cur_q_slot)
    pl.set_validshape(qk_right, [d, ctx.s1_size])
    pl.move(qk_right, cur_q_rhs_slot)
    pl.set_validshape(qk_acc, [ctx.s2_size, ctx.s1_size])
    pl.matmul(qk_acc, qk_left, qk_right)

    qk_slot = qk_vec_db.next()
    pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_BARKWARD_IDS[task_id % 2])
    pl.set_validshape(qk_slot, [ctx.s2_size, TS_HALF])
    pl.set_validshape(qk_acc, [(ctx.s2_size + 7) // 8 * 8, TS])
    pl.move(qk_slot, qk_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_FORWARD_IDS[task_id % 2])


def compute_pv_dn(ctx, v_l1_db, p_mat_db, left_db, right_db, acc_db, v, pv_dst, buf_a, buf_b, buf_c, d):
    """PV matmul with DN P layout: MAT [s2,s1] → Left [s1,s2] via pl.move 2D transpose."""
    sv_off = ctx.s2_size_acc + ctx.ki * TKV
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=P_READY_FORWARD_IDS[ctx.task_id % 3])
    cur_v_slot = v_l1_db.next()
    cur_p_slot = p_mat_db.next()
    pv_left = left_db[buf_a]
    pv_right = right_db[buf_b]
    pv_acc = acc_db[buf_c]

    pl.set_validshape(cur_v_slot, [ctx.s2_size, d])
    pl.load(cur_v_slot, v, [sv_off, ctx.k_n, 0], order=[0, 2])
    pl.set_validshape(cur_p_slot, [ctx.s2_size, TS])
    pl.set_validshape(pv_left, [TS, ctx.s2_size])
    pl.move(pv_left, cur_p_slot)  # MAT [s2,TS] → Left [TS,s2] via TMOV 2D transpose
    pl.set_validshape(pv_right, [ctx.s2_size, d])
    pl.move(pv_right, cur_v_slot)
    pl.set_validshape(pv_acc, [TS, d])
    pl.matmul(pv_acc, pv_left, pv_right)

    # pv_dst is the next pv_vec_db slot (caller-resolved). Block-0's PV stays
    # in this slot and is consumed by the next GU via previous() (deferred
    # init -- no running_o move on the Vector side).
    pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_BARKWARD_IDS[ctx.task_id % 2])
    pl.set_validshape(pv_dst, [TS_HALF, d])
    pl.set_validshape(pv_acc, [TS, d])
    pl.move(pv_dst, pv_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_FORWARD_IDS[ctx.task_id % 2])


def compute_p_dn(ctx_p, sub_id, qk_vec_db, tile_nz_db, tile_nz_db_1, tmp_max, tmp_sum,
                 global_max, global_sum, exp_corr_db, p_mat_db, mask_db,
                 maskr_db, mask_start_db, mask_len_db, maskr_gm, mask_start_gm, mask_len_gm,
                 scale):
    """DN softmax on KQ tile → P.  Column-wise mask + online softmax, no PLDI.

    P is split into two NZ regions: tile_nz_db (x_exp, even row groups) and
    tile_nz_db_1 (x_exp_1, odd row groups), inserted into Mat in two steps.
    """
    p_eid = ctx_p.task_id % 2
    q_idx_p = ctx_p.q_count % 3

    qk_slot = qk_vec_db.next()
    tile_nz = tile_nz_db.next()
    tile_nz_1 = tile_nz_db_1.next()
    mask_buf = mask_db.next()
    maskr_buf = maskr_db.next()
    mask_start_buf = mask_start_db.next()
    mask_len_buf = mask_len_db.next()
    gmax_p = global_max[q_idx_p]
    gsum_p = global_sum[q_idx_p]
    exp_diff = exp_corr_db[ctx_p.task_id % 3]
    row_offset = TS_HALF * sub_id
    row = ctx_p.s1_idx * TS

    if ctx_p.mask_bit == 0:
        pl.set_validshape(mask_buf, [ctx_p.s2_size, TS_HALF])
        fill_zero_mask(mask_buf, TKV, TS_HALF)
    else:
        mask_row = row + row_offset
        pl.set_validshape(maskr_buf, [1, TS_HALF])
        pl.load(maskr_buf, maskr_gm, [ctx_p.b_idx, mask_row])
        if HoleChunk == 0:  # noqa: F821
            if ctx_p.mask_num > 0:
                pl.set_validshape(mask_start_buf, [ctx_p.mask_num, TS_HALF])
                pl.load(mask_start_buf, mask_start_gm, [ctx_p.b_idx, 0, mask_row], order=[1, 2])
                pl.set_validshape(mask_len_buf, [ctx_p.mask_num, TS_HALF])
                pl.load(mask_len_buf, mask_len_gm, [ctx_p.b_idx, 0, mask_row], order=[1, 2])
            decode_mask_dn_vf(mask_buf, maskr_buf, mask_start_buf, mask_len_buf,
                              ctx_p.ki * TKV, TS_HALF, ctx_p.s2_size, ctx_p.mask_num)
        else:
            # Chunked DN decode (mask_num > CHUNK): base + refine passes.
            n_chunks = (ctx_p.mask_num + CHUNK - 1) // CHUNK
            for ck in pl.range(0, n_chunks):
                holes_ck = pl.min(CHUNK, ctx_p.mask_num - ck * CHUNK)
                pl.set_validshape(mask_start_buf, [holes_ck, TS_HALF])
                pl.load(mask_start_buf, mask_start_gm, [ctx_p.b_idx, ck * CHUNK, mask_row], order=[1, 2])
                pl.set_validshape(mask_len_buf, [holes_ck, TS_HALF])
                pl.load(mask_len_buf, mask_len_gm, [ctx_p.b_idx, ck * CHUNK, mask_row], order=[1, 2])
                if ck == 0:
                    decode_mask_dn_vf(mask_buf, maskr_buf, mask_start_buf, mask_len_buf,
                                      ctx_p.ki * TKV, TS_HALF, ctx_p.s2_size, holes_ck)
                else:
                    decode_mask_dn_refine(mask_buf, mask_start_buf, mask_len_buf,
                                          ctx_p.ki * TKV, TS_HALF, ctx_p.s2_size, holes_ck)

    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_FORWARD_IDS[p_eid])
    if ctx_p.loop_count == 0:
        if ctx_p.s2_size <= 64:
            softmax_dn_no_update_vf_tail64(qk_slot, tile_nz, gmax_p, gsum_p, mask_buf, TS_HALF, ctx_p.s2_size, scale)
        else:
            softmax_dn_no_update_vf(qk_slot, tile_nz, gmax_p, gsum_p, mask_buf, TS_HALF, ctx_p.s2_size, scale)
    else:
        if ctx_p.s2_size <= 64:
            softmax_dn_update_vf_tail64(qk_slot, tile_nz, gmax_p, gsum_p, exp_diff, mask_buf, TS_HALF,
                                        ctx_p.s2_size, scale)
        else:
            softmax_dn_update_vf(qk_slot, tile_nz, gmax_p, gsum_p, exp_diff, mask_buf, TS_HALF, ctx_p.s2_size, scale)
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BARKWARD_IDS[p_eid])

    cur_p_slot = p_mat_db.next()
    # 第 1 次：x_exp（偶数行组）→ Mat [0, row_offset]
    pl.set_validshape(tile_nz, [(ctx_p.s2_size + 63) // 64 * 64 // 2, TS_HALF])
    pl.insert(cur_p_slot, tile_nz, [0, row_offset])
    # 第 2 次：x_exp_1（奇数行组）→ Mat [64, row_offset]
    pl.set_validshape(tile_nz_1, [(ctx_p.s2_size + 63) // 64 * 64 // 2, TS_HALF])
    pl.insert(cur_p_slot, tile_nz_1, [(ctx_p.s2_size + 63) // 64 * 64 // 2, row_offset])
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY_FORWARD_IDS[ctx_p.task_id % 3])


# ================================================================
#  Kernel with NBuffer + auto_mutex
# ================================================================
@pl.jit(arch="a5", auto_mutex=True, compile_timeout=200,
        tiling_key=FlexAttnTilingKey,
        datatype={
            "query": "io_dtype",
            "key": "io_dtype",
            "value": "io_dtype",
            "attn_out": "io_dtype",
        })
def flex_attention_kernel(
    query: pl.Ptr[pl.DT_UINT8],
    key: pl.Ptr[pl.DT_UINT8],
    value: pl.Ptr[pl.DT_UINT8],
    maskr: pl.Ptr[pl.DT_UINT8],
    mask_start: pl.Ptr[pl.DT_UINT8],
    mask_len: pl.Ptr[pl.DT_UINT8],
    block_range: pl.Ptr[pl.DT_UINT8],
    block_compute: pl.Ptr[pl.DT_UINT8],
    block_mask: pl.Ptr[pl.DT_UINT8],
    actual_seq_q: pl.Ptr[pl.DT_UINT8],
    actual_seq_kv: pl.Ptr[pl.DT_UINT8],
    attn_out: pl.Ptr[pl.DT_UINT8],
    softmax_max_out: pl.Ptr[pl.DT_UINT8],
    softmax_sum_out: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: FlexAttnTiling,
):
    # ========== Delivery-form shape derivation (all from OpTiling) ==========
    # raw Ptr I/O carry no shape; reconstruct typed views with OpTiling scalars.
    batch = tiling.batch
    max_block_m = tiling.r_f
    max_block_n = tiling.w_f
    p_f = max_block_m * TS
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    n_dim = tiling.n_dim
    kv_heads = tiling.kv_heads
    mask_num = tiling.mask_num
    d = tiling.d
    scale = tiling.scale
    if Layout == 1:  # noqa: F821
        tq = batch * tiling.s1
        tkv = batch * tiling.s2
    else:
        actual_seq_q_t = pl.make_tensor(actual_seq_q, [batch], [1], dtype=pl.DT_INT64)
        actual_seq_kv_t = pl.make_tensor(actual_seq_kv, [batch], [1], dtype=pl.DT_INT32)
        tq = pl.getval(actual_seq_q_t, batch - 1)
        tkv = pl.getval(actual_seq_kv_t, batch - 1)
    tensor_q = pl.make_tensor(query, [tq, n_dim, d], [n_dim * d, d, 1], dtype=io_dtype)
    tensor_k = pl.make_tensor(key, [tkv, kv_heads, d], [kv_heads * d, d, 1], dtype=io_dtype)
    tensor_v = pl.make_tensor(value, [tkv, kv_heads, d], [kv_heads * d, d, 1], dtype=io_dtype)
    tensor_o = pl.make_tensor(attn_out, [tq, n_dim, d], [n_dim * d, d, 1], dtype=io_dtype)
    tensor_smax = pl.make_tensor(softmax_max_out, [n_dim, tq], [tq, 1], dtype=pl.DT_FP32)
    tensor_ssum = pl.make_tensor(softmax_sum_out, [n_dim, tq], [tq, 1], dtype=pl.DT_FP32)
    # mask_info typed views (absent inputs arrive as 1-elem placeholders that are
    # reconstructed here but never read -- the schedule_type branch gates them).
    maskr_t = pl.make_tensor(maskr, [batch, p_f], [p_f, 1], dtype=pl.DT_INT32)
    mask_start_t = pl.make_tensor(mask_start, [batch, mask_num, p_f], [mask_num * p_f, p_f, 1], dtype=pl.DT_INT32)
    mask_len_t = pl.make_tensor(mask_len, [batch, mask_num, p_f], [mask_num * p_f, p_f, 1], dtype=pl.DT_INT32)
    block_range_t = pl.make_tensor(block_range, [batch, max_block_m], [max_block_m, 1], dtype=pl.DT_INT32)
    block_compute_t = pl.make_tensor(block_compute, [batch, max_block_m, max_block_n],
                                      [max_block_m * max_block_n, max_block_n, 1], dtype=pl.DT_INT64)
    block_mask_t = pl.make_tensor(block_mask, [batch, max_block_m, max_block_n],
                                   [max_block_m * max_block_n, max_block_n, 1], dtype=pl.DT_UINT32)

    # ========== Cross-core shared buffers (UBNBuffer for double-buffer) ==========
    # P MAT - Vector insert, Cube PV read (DN: [TS, TKV] ZN)
    if IsDN == 1:  # noqa: F821
        p_mat_db = pl.make_tile_group(
            type=pl.TileType(shape=[TS, TKV], dtype=io_dtype, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
            addrs=MA2_P, mutex_ids=[14, 15, 16])
    else:
        p_mat_db = pl.make_tile_group(
            type=pl.TileType(shape=[TS, TKV], dtype=io_dtype, target_memory=pl.MemorySpace.Mat),
            addrs=MA2_P, mutex_ids=[14, 15, 16])

    # qk_vec UB - Cube store from ACC, Vector softmax (double-buffer for FIFO)
    # DN: [TKV, TS_HALF] row=KV, col=Q (from DualModeSplitN)
    if IsDN == 1:  # noqa: F821
        qk_vec_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TS_HALF], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1],
                             compact=1, pad=pl.TilePad.min),
            addrs=VA0, mutex_ids=[17, 18])
    else:
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
            type=pl.TileType(shape=[TS, TD], dtype=io_dtype,
                             target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1], compact=1),
            addrs=MA0_Q, mutex_ids=[0, 1])
        k_l1_db = pl.make_tile_group(
            type=pl.TileType(shape=[TD, TKV], dtype=io_dtype, target_memory=pl.MemorySpace.Mat,
                             layout=pl.ZN, valid_shape=[-1, -1], compact=1),
            addrs=MA1_K, mutex_ids=[2, 3])
        v_l1_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TD], dtype=io_dtype,
                             target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1], compact=1),
            addrs=MA3_V, mutex_ids=[4, 5])

        left_db = pl.make_tile_group(
            type=pl.TileType(shape=[TS, TD], dtype=io_dtype,
                             target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1], compact=1),
            addrs=[0, 32768], mutex_ids=[6, 7])
        right_db = pl.make_tile_group(
            type=pl.TileType(shape=[TD, TKV], dtype=io_dtype,
                             target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1], compact=1),
            addrs=[0, 32768], mutex_ids=[8, 9])
        acc_db = pl.make_tile_group(
            type=pl.TileType(shape=[TS, TKV], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1], compact=1),
            addrs=[0, 65536, 131072, 196608], mutex_ids=[10, 11, 12, 13])
        left_db2 = pl.make_tile_group(
            type=pl.TileType(shape=[TS, TKV], dtype=io_dtype,
                             target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1], compact=1),
            addrs=[0, 32768], mutex_ids=[6, 7])
        right_db2 = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TD], dtype=io_dtype,
                             target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1], compact=1),
            addrs=[0, 32768], mutex_ids=[8, 9])
        acc_db2 = pl.make_tile_group(
            type=pl.TileType(shape=[TS, TD], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1], compact=1),
            addrs=[0, 65536, 131072, 196608], mutex_ids=[10, 11, 12, 13])

        work_start = tiling.work_starts[core_id]
        work_end = tiling.work_starts[core_id + 1]
        # Self-clamp: host tiling may ship a shape-derived (non-data) upper bound
        # for work_end when the value-dependency host data is unavailable (the
        # uniform fallback over-estimates total_work for unequal-length batches).
        # The real total_work is re-derived here from actual_seq_q so phantom
        # work_ids (>= real total) are never iterated -> correct + crash-safe.
        total_work = 0
        if Layout == 1:  # noqa: F821
            total_work = batch * ((tiling.s1 + TS - 1) // TS) * n_dim
        else:
            for tw_idx in pl.range(batch):
                if tw_idx == 0:
                    tw_s1 = pl.getval(actual_seq_q_t, 0)
                else:
                    tw_s1 = pl.getval(actual_seq_q_t, tw_idx) - pl.getval(actual_seq_q_t, tw_idx - 1)
                total_work = total_work + (tw_s1 + TS - 1) // TS * n_dim
        work_end = pl.min(work_end, total_work)
        task_id = 0
        b_idx = 0
        s1_size_acc = 0
        s2_size_acc = 0
        actual_s1 = 0
        actual_s2 = 0
        s1o_acc = 0
        # Shared L0A/L0B/L0C buffer counters (explicit, see optimization-experience.md).
        # QK and PV share the same address spaces; one counter per space keeps
        # cursor phases in sync without empty .next() sync calls (which stall MTE1).
        buf_a = 0  # L0A (Left):  left_db / left_db2, depth 2
        buf_b = 0  # L0B (Right): right_db / right_db2, depth 2
        buf_c = 0  # L0C (Acc):   acc_db / acc_db2, depth 4
        ctx_arr = pl.struct_array(4, "CubeCtx", q_n=0, k_n=0, qi=0, ki=0,
                                  task_id=0, s1_size_acc=0, s2_size_acc=0, s1_size=0, s2_size=0, loop_count=0)
        s1o_size = 0 # tmp-val
        for idx in pl.range(batch):
            if Layout == 1:  # noqa: F821
                actual_s1 = tiling.s1
                actual_s2 = tiling.s2
            else:
                if idx == 0:
                    actual_s1 = pl.getval(actual_seq_q_t, idx)
                    actual_s2 = pl.getval(actual_seq_kv_t, idx)
                else:
                    actual_s1 = pl.getval(actual_seq_q_t, idx) - pl.getval(actual_seq_q_t, idx - 1)
                    actual_s2 = pl.getval(actual_seq_kv_t, idx) - pl.getval(actual_seq_kv_t, idx - 1)
            s1o_size = s1o_size + (actual_s1 + TS - 1) // TS * n_dim
            if (work_start >= s1o_size):
                s1o_acc = s1o_size
                s1_size_acc = s1_size_acc + actual_s1
                s2_size_acc = s2_size_acc + actual_s2
                b_idx = b_idx + 1
                continue
            break
        for work_id in pl.range(work_start, work_end):
            for _ in pl.range(b_idx, batch):
                if Layout == 1:  # noqa: F821
                    actual_s1 = tiling.s1
                    actual_s2 = tiling.s2
                else:
                    if b_idx == 0:
                        actual_s1 = pl.getval(actual_seq_q_t, b_idx)
                        actual_s2 = pl.getval(actual_seq_kv_t, b_idx)
                    else:
                        actual_s1 = pl.getval(actual_seq_q_t, b_idx) - pl.getval(actual_seq_q_t, b_idx - 1)
                        actual_s2 = pl.getval(actual_seq_kv_t, b_idx) - pl.getval(actual_seq_kv_t, b_idx - 1)
                s1o_size = s1o_acc + (actual_s1 + TS - 1) // TS * n_dim
                if (work_id >= s1o_size):
                    s1o_acc = s1o_size
                    s1_size_acc = s1_size_acc + actual_s1
                    s2_size_acc = s2_size_acc + actual_s2
                    b_idx = b_idx + 1
                    continue
                break
            cur_b_s1o = (actual_s1 + TS - 1) // TS
            s1o_size = work_id - s1o_acc
            n_idx = s1o_size // cur_b_s1o
            s1_idx = s1o_size % cur_b_s1o
            s1_size = pl.min(TS, actual_s1 - s1_idx * TS)

            sq_off = s1_size_acc + s1_idx * TS
            cur_q_slot = q_l1_db.next()
            if IsDN == 1:  # noqa: F821
                cur_q_rhs_slot = k_l1_db.next()  # Q^T right matrix, allocated once, reused
            kv_loop = pl.getval(block_range_t, b_idx * max_block_m + s1_idx)
            kv_end = pl.min(kv_loop * TKV, actual_s2)
            drain = 0
            if work_id == work_end - 1:
                drain = 2
            q_loop_count = 0  # index of the computed KV block within this Q tile
            for ki in pl.range(0, kv_loop + drain):
                if ki < kv_loop:
                    base_offset = b_idx * max_block_m * max_block_n + s1_idx * max_block_n
                    word_offset = ki // 32
                    work_index = ki % 32
                    compute_bit = pl.getval(block_compute_t, base_offset + word_offset)
                    compute_bit = (compute_bit >> work_index) & 1
                    if compute_bit == 0:
                        continue
                    # Save current context
                    ctx_curr = ctx_arr[task_id % 4]
                    ctx_curr.task_id = task_id
                    ctx_curr.q_n = n_idx
                    ctx_curr.k_n = n_idx * tiling.kv_heads // n_dim
                    ctx_curr.ki = ki
                    ctx_curr.s1_size_acc = s1_size_acc
                    ctx_curr.s2_size_acc = s2_size_acc
                    ctx_curr.s1_size = s1_size
                    ctx_curr.s2_size = pl.min(TKV, kv_end - ki * TKV)
                    ctx_curr.loop_count = q_loop_count  # 0 on first computed block -> load Q
                    q_loop_count = q_loop_count + 1

                    # ========== compute_qk (current step) ==========
                    if IsDN == 1:  # noqa: F821
                        compute_qk_dn(ctx_curr, ki, sq_off, tensor_q, tensor_k, cur_q_slot, k_l1_db,
                                      left_db, right_db, acc_db, qk_vec_db, task_id,
                                      cur_q_rhs_slot, buf_a, buf_b, buf_c, d)
                    else:
                        compute_qk(ctx_curr, ki, sq_off, tensor_q, tensor_k, cur_q_slot, k_l1_db,
                                   left_db, right_db, acc_db, qk_vec_db, task_id,
                                   buf_a, buf_b, buf_c, d)
                    buf_a = (buf_a + 1) % 2
                    buf_b = (buf_b + 1) % 2
                    buf_c = (buf_c + 1) % 4

                # ========== compute_pv (delayed 1 step: uses ctx from task_id-1) ==========
                if task_id > 1:
                    ctx_pre2 = ctx_arr[(task_id + 2) % 4]
                    if IsDN == 1:  # noqa: F821
                        compute_pv_dn(ctx_pre2, v_l1_db, p_mat_db, left_db2, right_db2, acc_db2, tensor_v,
                                      pv_vec_db.next(), buf_a, buf_b, buf_c, d)
                    else:
                        compute_pv(ctx_pre2, v_l1_db, p_mat_db, left_db2, right_db2, acc_db2, tensor_v,
                                   pv_vec_db.next(), buf_a, buf_b, buf_c, d)
                    buf_a = (buf_a + 1) % 2
                    buf_b = (buf_b + 1) % 2
                    buf_c = (buf_c + 1) % 4
                task_id = task_id + 1

    # =================== VECTOR SECTION ===================
    with pl.section_vector():
        tile_nz_g = pl.make_tile_group(
            type=pl.TileType(shape=[65, 128], dtype=io_dtype, target_memory=pl.MemorySpace.Vec,
                                valid_shape=[-1, -1], layout=pl.NZ),
            addrs=VA1, mutex_ids=[0, 1])
        if IsDN == 1:  # noqa: F821
            tile_nz_x_exp_1_db = pl.make_tile_group(
                type=pl.TileType(shape=[65, 128], dtype=io_dtype, target_memory=pl.MemorySpace.Vec,
                                 valid_shape=[-1, -1], layout=pl.NZ),
                addrs=VA1 + TS_HALF * (TS_HALF + 1) * 2, mutex_ids=[0, 1])

        running_o = pl.make_tile(
            pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addr=VA7)

        o_f16_g = pl.make_tile_group(
            type=pl.TileType(shape=[TS_HALF, TD], dtype=io_dtype, target_memory=pl.MemorySpace.Vec),
            addrs=VA9, mutex_ids=[2])
        o_f16 = o_f16_g.next()

        if IsDN == 1:  # noqa: F821
            mask_db = pl.make_tile_group(
                type=pl.TileType(shape=[TKV, TS_HALF], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
                addrs=VA10, mutex_ids=[3, 4])
        else:
            mask_db = pl.make_tile_group(
                type=pl.TileType(shape=[TS_HALF, TKV], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
                addrs=VA10, mutex_ids=[3, 4])

        maskr_db = pl.make_tile_group(
            type=pl.TileType(shape=[1, TS_HALF], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
            addrs=VA13, mutex_ids=[5, 6])

        mask_start_db = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK, TS_HALF], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
            addrs=VA14, mutex_ids=[7, 8])

        mask_len_db = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK, TS_HALF], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
            addrs=VA15, mutex_ids=[9, 10])

        # Double-buffered global state (per Q tile) -use tile tuples for dynamic
        # indexing by q_count % 2, since StructArray ctx references need runtime index.
        red_type = pl.TileType(shape=[TS_HALF, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN)
        gmax_0 = pl.make_tile(red_type, addr=VA_GMAX0)
        gmax_1 = pl.make_tile(red_type, addr=VA_GMAX1)
        gmax_2 = pl.make_tile(red_type, addr=VA_GMAX2)
        global_max = (gmax_0, gmax_1, gmax_2)

        gsum_0 = pl.make_tile(red_type, addr=VA_GSUM0)
        gsum_1 = pl.make_tile(red_type, addr=VA_GSUM1)
        gsum_2 = pl.make_tile(red_type, addr=VA_GSUM2)
        global_sum = (gsum_0, gsum_1, gsum_2)

        tmp_max = pl.make_tile(red_type, addr=VA11)
        tmp_sum = pl.make_tile(red_type, addr=VA12)

        # Stats-output tiles: [1, TS_HALF] plain layout sharing the gmax/gsum
        # addresses.  [64, 1] and [1, 64] have the same memory layout (64
        # contiguous fp32), so the softmax-written gmax values are visible
        # through this tile without any copy.  order=[1,0] maps the 64 columns
        # to consecutive GM rows (s-direction, stride 1).
        stats_type = pl.TileType(shape=[1, TS_HALF], dtype=pl.DT_FP32,
                                 target_memory=pl.MemorySpace.Vec)
        stats_max_t0 = pl.make_tile(stats_type, addr=VA_GMAX0)
        stats_max_t1 = pl.make_tile(stats_type, addr=VA_GMAX1)
        stats_max_t2 = pl.make_tile(stats_type, addr=VA_GMAX2)
        stats_max_buf = (stats_max_t0, stats_max_t1, stats_max_t2)
        stats_sum_t0 = pl.make_tile(stats_type, addr=VA_GSUM0)
        stats_sum_t1 = pl.make_tile(stats_type, addr=VA_GSUM1)
        stats_sum_t2 = pl.make_tile(stats_type, addr=VA_GSUM2)
        stats_sum_buf = (stats_sum_t0, stats_sum_t1, stats_sum_t2)

        # FIFO exp_corr -use NBuffer with current() auto-rotate
        exp_max0 = pl.make_tile(red_type, addr=VA_EXPMAX0)
        exp_max1 = pl.make_tile(red_type, addr=VA_EXPMAX1)
        exp_max2 = pl.make_tile(red_type, addr=VA_EXPMAX2)
        exp_corr_db = (exp_max0, exp_max1, exp_max2)

        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BARKWARD_IDS[0])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BARKWARD_IDS[1])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[0])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[1])

        work_start = tiling.work_starts[core_id]
        work_end = tiling.work_starts[core_id + 1]
        # Self-clamp (see cube section): re-derive real total_work from
        # actual_seq_q so a shape-derived (over-estimating) host work_end never
        # iterates phantom work_ids.
        total_work = 0
        if Layout == 1:  # noqa: F821
            total_work = batch * ((tiling.s1 + TS - 1) // TS) * n_dim
        else:
            for tw_idx in pl.range(batch):
                if tw_idx == 0:
                    tw_s1 = pl.getval(actual_seq_q_t, 0)
                else:
                    tw_s1 = pl.getval(actual_seq_q_t, tw_idx) - pl.getval(actual_seq_q_t, tw_idx - 1)
                total_work = total_work + (tw_s1 + TS - 1) // TS * n_dim
        work_end = pl.min(work_end, total_work)
        sub_id = pl.get_subblock_idx()

        task_id = 0
        q_count = 0
        b_idx = 0
        s1_size_acc = 0
        s2_size_acc = 0
        actual_s1 = 0
        actual_s2 = 0
        s1o_acc = 0

        # StructArray(3) for pipeline context tracking (same as original)
        ctx_arr = pl.struct_array(4, "VecCtx", b_idx=0, n_idx=0, s1_idx=0, q_off=0, ki=0,
                                  q_count=0, sub_id=0, task_id=0, kv_loop=0, half_s1=0, first_s1=0,
                                  s1_size=0, s2_size=0, compute_bit=0, mask_bit=0, loop_count=0, mask_num=0)

        # calc start
        s1o_size = 0 # tmp-val
        for idx in pl.range(batch):
            if Layout == 1:  # noqa: F821
                actual_s1 = tiling.s1
                actual_s2 = tiling.s2
            else:
                if idx == 0:
                    actual_s1 = pl.getval(actual_seq_q_t, idx)
                    actual_s2 = pl.getval(actual_seq_kv_t, idx)
                else:
                    actual_s1 = pl.getval(actual_seq_q_t, idx) - pl.getval(actual_seq_q_t, idx - 1)
                    actual_s2 = pl.getval(actual_seq_kv_t, idx) - pl.getval(actual_seq_kv_t, idx - 1)
            s1o_size = s1o_size + (actual_s1 + TS - 1) // TS * n_dim
            if (work_start >= s1o_size):
                s1o_acc = s1o_size
                s1_size_acc = s1_size_acc + actual_s1
                s2_size_acc = s2_size_acc + actual_s2
                b_idx = b_idx + 1
                continue
            break
        for work_id in pl.range(work_start, work_end):
            for _ in pl.range(b_idx, batch):
                if Layout == 1:  # noqa: F821
                    actual_s1 = tiling.s1
                    actual_s2 = tiling.s2
                else:
                    if b_idx == 0:
                        actual_s1 = pl.getval(actual_seq_q_t, b_idx)
                        actual_s2 = pl.getval(actual_seq_kv_t, b_idx)
                    else:
                        actual_s1 = pl.getval(actual_seq_q_t, b_idx) - pl.getval(actual_seq_q_t, b_idx - 1)
                        actual_s2 = pl.getval(actual_seq_kv_t, b_idx) - pl.getval(actual_seq_kv_t, b_idx - 1)
                s1o_size = s1o_acc + (actual_s1 + TS - 1) // TS * n_dim
                if (work_id >= s1o_size):
                    s1o_acc = s1o_size
                    s1_size_acc = s1_size_acc + actual_s1
                    s2_size_acc = s2_size_acc + actual_s2
                    b_idx = b_idx + 1
                    continue
                break
            cur_b_s1o = (actual_s1 + TS - 1) // TS
            s1o_size = work_id - s1o_acc
            n_idx = s1o_size // cur_b_s1o
            s1_idx = s1o_size % cur_b_s1o
            s1_size = pl.min(TS, actual_s1 - s1_idx * TS)

            sq_acc = 0
            if Layout == 1:  # noqa: F821
                sq_acc = b_idx * tiling.s1
            else:
                if b_idx > 0:
                    sq_acc = pl.getval(actual_seq_q_t, b_idx - 1)
            q_off = sq_acc + s1_idx * TS
            kv_loop = pl.getval(block_range_t, b_idx * max_block_m + s1_idx)
            kv_end = pl.min(kv_loop * TKV, actual_s2)

            drain = 0
            if work_id == work_end - 1:
                drain = 3
            q_loop_count = 0  # reset per Q tile: first computed KV block gets loop_count 0
            for ki in pl.range(0, kv_loop + drain):
                if ki < kv_loop:
                    base_offset = b_idx * max_block_m * max_block_n + s1_idx * max_block_n
                    word_offset = ki // 32
                    work_index = ki % 32
                    compute_bit = pl.getval(block_compute_t, base_offset + word_offset)
                    mask_bit = pl.getval(block_mask_t, base_offset + word_offset)
                    cur_compute = (compute_bit >> work_index) & 1
                    cur_mask = (mask_bit >> work_index) & 1
                    if cur_compute == 0:
                        continue
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
                    half_s1 = (ctx_curr.s1_size + 1) // 2
                    first_s1 = half_s1
                    if sub_id == 1:
                        half_s1 = ctx_curr.s1_size - half_s1
                    ctx_curr.first_s1 = first_s1
                    ctx_curr.half_s1 = half_s1
                    ctx_curr.compute_bit = cur_compute
                    ctx_curr.mask_bit = cur_mask
                    ctx_curr.loop_count = q_loop_count
                    ctx_curr.mask_num = mask_num
                    q_loop_count = q_loop_count + 1

                # ===== compute_p (consumer, lag 1: real steps + 1st drain step) =====
                if task_id > 0:
                    if ki <= kv_loop:
                        ctx_p = ctx_arr[(task_id + 3) % 4]
                        if IsDN == 1:  # noqa: F821
                            compute_p_dn(ctx_p, sub_id, qk_vec_db, tile_nz_g, tile_nz_x_exp_1_db, tmp_max, tmp_sum,
                                global_max, global_sum, exp_corr_db, p_mat_db, mask_db, maskr_db,
                                mask_start_db, mask_len_db, maskr_t, mask_start_t, mask_len_t, scale)
                        else:
                            compute_p(ctx_p, sub_id, qk_vec_db, tile_nz_g, tmp_max, tmp_sum,
                                global_max, global_sum, exp_corr_db, p_mat_db, mask_db, maskr_db,
                                mask_start_db, mask_len_db, maskr_t, mask_start_t, mask_len_t, scale)

                # ===== compute_gu (consumer, lag 3: every step incl. all drains) =====
                if task_id > 2:
                    ctx_gu = ctx_arr[(task_id + 1) % 4]
                    if IsDN == 1:
                        compute_gu_dn(ctx_gu, sub_id, pv_vec_db, exp_corr_db,
                                      global_max, global_sum, running_o, o_f16, tensor_o,
                                      tensor_smax, tensor_ssum, stats_max_buf, stats_sum_buf, d)
                    else:
                        compute_gu(ctx_gu, pv_vec_db, exp_corr_db, global_max, global_sum,
                                   running_o, o_f16, tensor_o, tensor_smax, tensor_ssum,
                                   stats_max_buf, stats_sum_buf, d)

                task_id = task_id + 1
            q_count = q_count + 1


def build_work_ranges(seq_q_list, tile_costs, n, num_cores):
    """Assign a contiguous work_id range to each core, balancing *cost* rather
    than the raw work_id count.

    Each work_id is one (batch, head, Q-tile) triple; the kernel walks them in
    the order  for bi: for n_idx: for s1_idx.  ``tile_costs[bi][s1]`` is the
    number of KV blocks Q-tile ``s1`` of batch ``bi`` actually computes
    (derived by the interface from the mask_info schedule: popcount of
    block_compute), which under a causal / sliding-window mask is highly
    non-uniform. Splitting
    so each core's summed cost is ~equal removes the straggler skew.

    The kernel requires each core's range to be a *contiguous* [start, end)
    (its b_idx / seq accumulators advance monotonically with work_id), so this
    is a "split array into num_cores parts minimizing the largest part-sum"
    problem, solved exactly by binary-searching the max part cost.

    Returns a list of num_cores+1 ints: per-core start + trailing total_work
    sentinel (core i's range is [starts[i], starts[i+1])).
    """
    b = len(seq_q_list)

    # Per-work_id cost, in kernel work_id order (bi outer, then n_idx, then s1_idx).
    costs = []
    for bi in range(b):
        blk_m = (seq_q_list[bi] + TS - 1) // TS
        tile_cost = [tile_costs[bi][s1] for s1 in range(blk_m)]
        for _n in range(n):
            costs.extend(tile_cost)
    total_work = len(costs)

    prefix = [0] * (total_work + 1)
    for i, c in enumerate(costs):
        prefix[i + 1] = prefix[i] + c

    def _num_parts(cap):
        # greedily pack contiguous work_ids into as few cores as possible s.t.
        # no core's cost exceeds cap; returns parts needed (inf if a single
        # work_id already exceeds cap -> cap too small).
        parts, cur, i = 1, 0, 0
        while i < total_work:
            c = costs[i]
            if c > cap:
                return total_work + 1  # infeasible cap
            if cur + c > cap:
                parts += 1
                cur = 0
            cur += c
            i += 1
        return parts

    # Binary-search the smallest feasible max-part cost (min-max objective).
    lo, hi = (max(costs) if costs else 0), (prefix[total_work] or 1)
    while lo < hi:
        mid = (lo + hi) // 2
        if _num_parts(mid) <= num_cores:
            hi = mid
        else:
            lo = mid + 1
    cap = lo

    # Emit contiguous ranges packed under `cap`; ranges are contiguous with no
    # gaps, so only starts are kept plus a trailing total_work sentinel. Cores
    # beyond the last packed one get an empty range [total_work, total_work).
    starts = []
    core, start, cur, i = 0, 0, 0, 0
    while i < total_work and core < num_cores:
        c = costs[i]
        if cur + c > cap and cur > 0:
            starts.append(start)
            core += 1
            start, cur = i, 0
            continue
        cur += c
        i += 1
    starts.append(start)                     # last (possibly empty) packed core
    for _ in range(core + 1, num_cores):
        starts.append(total_work)            # empty cores
    starts.append(total_work)                # sentinel: starts[num_cores]
    return starts


def flex_attention(q, k, v, actual_seq_q, actual_seq_kv,
                   maskr, mask_start, mask_len,
                   block_range, block_compute, block_mask,
                   attn_out, softmax_max, softmax_sum, layout="varlen"):
    """Host interface: NPU tensors in -> mask_info-driven tiling -> kernel.

    Parameter order matches the op_host flex_attention_def.cpp
    operator registration (inputs first, then attributes):
        maskr          [B, Pf] int32 (Required)
        mask_start/mask_len  [B, mask_num, Pf] int32 (mask_num is derived from
                       mask_start.shape[1]; both Required when mask_num > 0)
        block_range    [B, Rf] int32 (kv_loop = value)
        block_compute/block_mask [B, Rf, Wf] (int64/uint32)
        actual_seq_q   [B] int32 cumulative Q lengths (Required for varlen;
                       pass None for bshd, lengths come from shapes)
        actual_seq_kv  [B] int32 cumulative KV lengths (Required for varlen;
                       pass None for bshd)
        attn_out       [tq, n, d] fp16/bf16: caller-allocated output tensor
                       (result is written in place; read it directly after)
        softmax_max    [n, tq] fp32: caller-allocated per-Q global max(S),
                       S = QK*scale + mask (written in place)
        softmax_sum    [n, tq] fp32: caller-allocated per-Q
                       sum(exp(S - max_final)) (written in place)
        mask_num       derived from mask_start.shape[1] (0 when mask_start is
                       None; 0..MASK_NUM_LIMIT; >CHUNK uses the chunked
                       HoleChunk=1 decoder)
        head_dim       d in [8, 128] (tiling.d; UB tiles allocated at 128,
                       runtime validshape/params follow d; softmax scale is
                       1/sqrt(d) via tiling.scale). 16-multiples validated.
        layout         str attr: "varlen" (packed [L, H, D]) or
                       "bshd" ([B, S, H, D]; kernel reads lengths from
                       tiling.s1/s2, actual_seq_q/kv not dereferenced)

    GQA: k/v heads may differ from q heads. kv_heads is derived from k.shape[1];
    kernel reads k/v by k_n = q_n * kv_heads // n_dim.
    """
    from pypto_pro.runtime.platform import get_platform_info  # noqa: E402

    device = q.device
    for name, t in [("q", q), ("k", k), ("v", v)]:
        assert t.is_npu, f"input '{name}' must be an NPU tensor (got {t.device})."
    assert q.dtype in (torch.float16, torch.bfloat16), f"unsupported q dtype {q.dtype}."
    assert q.dtype == k.dtype == v.dtype, "q/k/v must share the same dtype."
    assert layout in ("varlen", "bshd"), f"layout must be 'varlen' or 'bshd', got {layout!r}."

    if layout == "bshd":
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, \
            "bshd layout requires q/k/v of shape [B, S, H, D]."
        b, s1, n, d = q.shape
        b2, s2, kv_heads, d2 = k.shape
        assert b2 == b and v.shape == (b, s2, kv_heads, d2), \
            "k/v shape must be [B, S_kv, H_kv, D] matching q batch."
        seq_q_list = [s1] * b
        seq_kv_list = [s2] * b
        q = q.reshape(b * s1, n, d)
        k = k.reshape(b * s2, kv_heads, d)
        v = v.reshape(b * s2, kv_heads, d)
        layout_id = 1
    else:
        assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3, \
            "varlen layout requires q/k/v of shape [L, H, D]."
        assert actual_seq_q is not None and actual_seq_kv is not None, \
            "varlen layout requires actual_seq_q/actual_seq_kv."
        assert actual_seq_q.dtype == torch.int64, \
            f"actual_seq_q must be int64 (kernel reads DT_INT64), got {actual_seq_q.dtype}."
        assert actual_seq_kv.dtype == torch.int32, \
            f"actual_seq_kv must be int32, got {actual_seq_kv.dtype}."
        aq_cpu = actual_seq_q.cpu().tolist()
        akv_cpu = actual_seq_kv.cpu().tolist()
        seq_q_list = [int(aq_cpu[0])] + [int(aq_cpu[i] - aq_cpu[i - 1]) for i in range(1, len(aq_cpu))]
        seq_kv_list = [int(akv_cpu[0])] + [int(akv_cpu[i] - akv_cpu[i - 1]) for i in range(1, len(akv_cpu))]
        s1 = s2 = 0
        layout_id = 0

    tq, n, d = q.shape
    tkv, kv_heads, d2 = k.shape
    assert d2 == d, f"k head_dim {d2} != q head_dim {d}"
    # Head-dim generalization: any 8 <= d <= 128 runs the same binary (UB tiles
    # stay allocated at TD=128, runtime validshape/params follow tiling.d).
    # 16-multiples are the validated set; other values are supported by the
    # same code path (GM stride + matmul K-dim alignment untested there).
    assert 8 <= d <= TD, f"head_dim {d} out of supported range [8, {TD}]."
    assert n % kv_heads == 0, f"n={n} not divisible by kv_heads={kv_heads}."
    assert v.shape == (tkv, kv_heads, d), \
        f"v shape must be [tkv, kv_heads, d], got {v.shape}."
    assert sum(seq_q_list) == tq, f"sum(seq_q_list) {sum(seq_q_list)} != q rows {tq}."
    assert sum(seq_kv_list) == tkv, f"sum(seq_kv_list) != kv rows {tkv}."
    assert attn_out.shape == (tq, n, d), \
        f"attn_out shape {tuple(attn_out.shape)} != {(tq, n, d)}."
    assert attn_out.dtype == q.dtype, \
        f"attn_out dtype {attn_out.dtype} != q dtype {q.dtype}."
    assert softmax_max.shape == (n, tq), \
        f"softmax_max shape {tuple(softmax_max.shape)} != {(n, tq)}."
    assert softmax_max.dtype == torch.float32, \
        f"softmax_max must be fp32, got {softmax_max.dtype}."
    assert softmax_sum.shape == (n, tq), \
        f"softmax_sum shape {tuple(softmax_sum.shape)} != {(n, tq)}."
    assert softmax_sum.dtype == torch.float32, \
        f"softmax_sum must be fp32, got {softmax_sum.dtype}."

    # ---------- descriptor validation ----------
    b = len(seq_q_list)
    r_f = max((sq + TS - 1) // TS for sq in seq_q_list)
    p_f = r_f * TS
    w_f = (max((skv + TKV - 1) // TKV for skv in seq_kv_list) + 31) // 32

    assert maskr is not None, "maskr is Required."
    assert maskr.shape == (b, p_f), f"maskr shape {tuple(maskr.shape)} != {(b, p_f)}."
    # mask_num is not a separate input: it is derived from the mask_start shape
    # (mirrors the host tiling, which reads mask_start.dim(1)). mask_num == 0
    # <=> mask_start absent.
    if mask_start is not None:
        assert mask_start.dim() == 3, \
            f"mask_start must be rank-3 [B, mask_num, Pf], got rank {mask_start.dim()}."
        assert mask_len is not None and mask_start.shape == mask_len.shape, \
            (f"mask_len shape {None if mask_len is None else tuple(mask_len.shape)} "
             f"must match mask_start {tuple(mask_start.shape)}.")
        mask_num = int(mask_start.shape[1])
        assert mask_start.shape == (b, mask_num, p_f), \
            f"mask_start shape {tuple(mask_start.shape)} != {(b, mask_num, p_f)}."
    else:
        mask_num = 0
        mask_len = None
    # No kernel-side upper bound: mask_num > CHUNK runs the HoleChunk=1
    # chunked decoder.  The sanity limit only guards runaway GM descriptors.
    assert mask_num <= MASK_NUM_LIMIT, \
        f"mask_num {mask_num} > sanity limit {MASK_NUM_LIMIT}."

    # ---------- schedule validation (bitmask only) ----------
    assert block_range is not None, "block_range is Required."
    assert block_compute is not None, "block_compute is Required."
    assert block_mask is not None, "block_mask is Required."
    assert block_range.shape == (b, r_f), f"block_range shape {tuple(block_range.shape)} != {(b, r_f)}."
    assert tuple(block_compute.shape) == (b, r_f, w_f), \
        f"block_compute shape {tuple(block_compute.shape)} != {(b, r_f, w_f)}."
    assert tuple(block_mask.shape) == (b, r_f, w_f), \
        f"block_mask shape {tuple(block_mask.shape)} != {(b, r_f, w_f)}."

    # ---------- device/dtype validation (caller passes device tensors) ----------
    assert block_compute.dtype == torch.int64, \
        f"block_compute must be int64 (kernel reads DT_INT64), got {block_compute.dtype}."
    for name, t in [("maskr", maskr), ("mask_start", mask_start), ("mask_len", mask_len),
                    ("block_range", block_range), ("block_compute", block_compute),
                    ("block_mask", block_mask), ("attn_out", attn_out),
                    ("softmax_max", softmax_max), ("softmax_sum", softmax_sum),
                    ("actual_seq_q", actual_seq_q), ("actual_seq_kv", actual_seq_kv)]:
        if t is not None:
            assert t.device == device, \
                f"input '{name}' must be on {device} (got {t.device})."
    sc_np = block_compute.cpu().numpy() if torch.is_tensor(block_compute) else block_compute
    tile_costs = [[sum(int(sc_np[bi, m, w]).bit_count() for w in range(w_f)) for m in range(r_f)]
                  for bi in range(b)]

    # ---------- work ranges ----------
    total_work = sum((x + TS - 1) // TS for x in seq_q_list) * n
    num_cores = get_platform_info().core_num
    actual_num_cores = min(num_cores, total_work)
    work_starts = build_work_ranges(seq_q_list, tile_costs, n, actual_num_cores)

    # ---------- h2d ----------
    _ph = torch.zeros(1, dtype=torch.uint8, device=device)

    def _or_ph(x):
        return x if x is not None else _ph

    # ---------- launch ----------
    assert len(work_starts) <= MAX_TILING_CORES + 1, \
        f"work_starts {len(work_starts)} exceeds tiling capacity {MAX_TILING_CORES + 1}."
    pl_dtype = pl.DT_FP16 if q.dtype == torch.float16 else pl.DT_BF16
    datatype = {"query": pl_dtype, "key": pl_dtype, "value": pl_dtype, "attn_out": pl_dtype}
    tiling = FlexAttnTiling(
        mask_num=mask_num,
        n_dim=n,
        kv_heads=kv_heads,
        batch=b,
        r_f=r_f,
        w_f=w_f,
        s1=s1,
        s2=s2,
        d=d,
        scale=1.0 / math.sqrt(d),
        work_starts=work_starts + [0] * (64 - len(work_starts)),
    )
    flex_attention_kernel[None, actual_num_cores,
                          {"Schedule": 0, "Layout": layout_id,
                           "IsDN": 1 if IS_DN else 0,
                           "HoleChunk": 1 if mask_num > CHUNK else 0}, datatype](
        q, k, v,
        _or_ph(maskr), _or_ph(mask_start), _or_ph(mask_len),
        block_range,
        block_compute, block_mask,
        actual_seq_q, actual_seq_kv,
        attn_out,
        softmax_max, softmax_sum,
        _ph,  # workspace placeholder (PyPTO routes workspace via SetSysWorkspaceForce)
        tiling)
    torch.npu.synchronize()