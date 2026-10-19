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
"""PyPTO-Pro engram_backward kernel implementation.

Backward of the Engram Gated Memory operator (forward = engram_v4).

Single-kernel design: a vector section (produces grad_value_ws / grad_key_ws
plus grad_hidden_states and the per-subblock grad_γ FP32 workspace) feeds a
cube section (Nest1 grad_emb + Nest2 grad_W_v + Nest3 grad_W_k). BF16
activation inputs are cast to FP32 once at the computation boundary; all RMS /
gate / scaled-dot math runs in FP32 inside VF helpers. Final output stores
are low precision (BF16).

7-step backward (per head m, reversed Step7->Step1):
  Step7  grad_gates[m]=Σ_h(go·value); grad_value=Σ_m(go·gates)
  Step6  linear_bw(grad_value, E, W_v)  -> grad_emb_v, grad_W_v
  Step5  grad_score   = signed_sqrt_gate_bw(grad_gate, score, gate)
  Step4  grad_nKey    = grad_score·(1/√H)·nQuery ; grad_nQuery symmetric
  Step3  (grad_hidden_m, grad_γ_q) = rms_norm_bw(grad_nQuery, hidden, γ_q)
  Step2  (grad_key_m,  grad_γ_k) = rms_norm_bw(grad_nKey, key, γ_k)
  Step1  linear_bw(grad_key_m, E, W_k[m]) -> grad_emb_k, grad_W_k[m]
  Sum:   grad_emb = grad_emb_v + Σ_m grad_emb_k
"""

import logging

import torch

import pypto_pro.language as pl
from pypto_pro.runtime.platform import get_platform_info
from pypto_pro.runtime.tilingkey import TilingKeyField

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# TilingKey: two specialization fields (4x4 = 16 compiled variants).
#   HMode (bits=2): vector sub-tile row count on H + per-H UB layout.
#     0 -> H=1280 (rows: B=4, A=8;  H_CHUNK=1280)   1 -> H=2560 (rows: B=4, A=4; H_CHUNK=2560)
#     2 -> H=2048 (rows: B=4, A=4;  H_CHUNK=2048)    3 -> H=1536 (rows: B=4, A=4; H_CHUNK=1536)


class EngramTilingKey:
    HMode = TilingKeyField(bits=2, values=[0, 1, 2, 3])   # 0->1280 1->2560 2->2048 3->1536
    BSMode = TilingKeyField(bits=2, values=[0, 1, 2, 3])  # V_TILE = 128/64/32/16


# Compile-time constants (mirror forward engram_v4).
# M_H (head count) is DYNAMIC (1-16), derived in kernel body from grad_output.shape[1].
# Cube tiles are fixed 128 (cube_mn == cube_k in kernel); TILE_M/TILE_N references removed.
# Vector worker BS-row tile candidates, selected per launch by BSMode so the
# 64 AIV workers (32 cores x 2) stay filled. V_TILE is a multiple of
# TILE_BS_VEC_A (4/8) so a worker's row window never straddles VF sub-tiles.
V_TILE_BS0 = 128           # BSMode 0: BS >= 8192
V_TILE_BS1 = 64            # BSMode 1: BS >= 4096
V_TILE_BS2 = 32            # BSMode 2: BS >= 2048
V_TILE_BS3 = 16            # BSMode 3: small BS
LANES_FP32 = 64          # VF FP32 register width

# Per-HMode vector-tile split sizes. A four-way `if (HMode==k)` binds
# TILE_BS_VEC / TILE_BS_VEC_A / H_CHUNK to one triple so TileType.shape stays
# compile-time constant.  B-phase rows are fixed 4 (Pass-A: 8 for H=1280, else 4).
TILE_BS_VEC_1280 = 4           # HMode 0 H=1280: vec sub-row tile (B-key/B-query)
TILE_BS_VEC_2560 = 4           # HMode 1 H=2560
TILE_BS_VEC_2048 = 4           # HMode 2 H=2048
TILE_BS_VEC_1536 = 4           # HMode 3 H=1536 
TILE_BS_VEC_A_1280 = 8         # HMode 0
TILE_BS_VEC_A_2560 = 4         # HMode 1
TILE_BS_VEC_A_2048 = 4         # HMode 2
TILE_BS_VEC_A_1536 = 4         # HMode 3
H_CHUNK_1280 = 1280           # HMode 0: one full-H vector tile
H_CHUNK_2560 = 2560           # HMode 1
H_CHUNK_2048 = 2048           # HMode 2
H_CHUNK_1536 = 1536           # HMode 3

GATE_EPS = 1.0e-12       # signed_sqrt_gate denominator guard (aligns golden)
# clamp_value / eps 为 kernel 运行时标量参数 (与 pypto tensor 版本 API 一致),
# wrapper 可选传入, 默认 1e-6.
GAMMA_SLOT_SPLIT = 1

# UB addresses (vector section), two groups:
#  (A) [rows,H_CHUNK] tiles: bytes = rows*H.  CONSTANT (40KB) across HModes 0/1
#      (8*1280==4*2560), so those two keys share the SAME (A) addresses.  HMode 2
#      (4*2048=32KB) and HMode 3 (8*1536=48KB) BREAK the constant product, so their
#      (A) addresses are HMode-specialized too.
#  (B) [rows,64]/[1,H_CHUNK] tiles: column count does NOT scale with H, so doubling
#      rows doubles bytes -> addresses MUST be specialized per HMode (a shared stride
#      would overlap at fewer-column keys -> RMS-bw reads garbage).
#  HMode-dependent addrs use a parser-foldable 4-way select
#      (HMode == 0)*A_1280 + (HMode == 1)*A_2560 + (HMode == 2)*A_2048 + (HMode == 3)*A_1536
#  (HMode is a ConstInt) — the parser cannot see names bound in `if ` blocks,
#  so the address arithmetic is written inline at each make_tile_group call.

# ---- HMode = 0 (H = 1280, B-rows = 8): (A) group ----
VA_GOUT32_1280 = 0x05000  # [8,1280] FP32 grad_output FP32 (40KB)
VA_PTNR16_1280 = 0x0F000  # [8,1280] BF16 partner (20KB)
VA_PTNR32_1280 = 0x14000  # [8,1280] FP32 partner (40KB)
VA_GN32_1280 = 0x28000    # [8,1280] FP32 Pass-B grad_nQuery/grad_nKey (INDEPENDENT) (40KB)
VA_GBUF16_1280 = 0x1E000  # BF16 final vector output (aliases GVAL32)
VA_XCACHE32_1280 = 0x1E000  # RMS-x cache (aliases GVAL32)

# ---- HMode = 1 (H = 2560, B-rows = 4): (A) group (byte-identical to HMode 0) ----
VA_GOUT32_2560 = 0x05000
VA_PTNR16_2560 = 0x0F000
VA_PTNR32_2560 = 0x14000
VA_GN32_2560 = 0x28000
VA_GBUF16_2560 = 0x1E000
VA_XCACHE32_2560 = 0x1E000

# ---- HMode = 2 (H = 2048, B-rows = 4): (A) group (32KB FP32 tiles) ----
VA_GOUT32_2048 = 0x04000  # [4,2048] FP32 (32KB)
VA_PTNR16_2048 = 0x0C000  # [4,2048] BF16 (16KB)
VA_PTNR32_2048 = 0x10000  # [4,2048] FP32 (32KB)
VA_GN32_2048 = 0x20000    # [4,2048] FP32 (32KB)
VA_GBUF16_2048 = 0x18000  # aliases GVAL32
VA_XCACHE32_2048 = 0x18000  # aliases GVAL32

# ---- HMode = 3 (H = 1536, B-rows = 4): (A) group (24KB FP32 tiles) ----
# rows=4 (like 2048/2560) shrinks each big tile to 24KB, so ALL 6 big tiles fit
# fully independently in 154KB -- NO dead-region reuse, NO risky aliasing.  GN32
# is independent; only the baseline-safe aliases apply (GBUF16/XCACHE32==GVAL32).
VA_GOUT32_1536 = 0x03000  # [4,1536] FP32 (24KB)
VA_PTNR16_1536 = 0x09000  # [4,1536] BF16 (12KB)
VA_PTNR32_1536 = 0x0C000  # [4,1536] FP32 (24KB)
VA_GN32_1536 = 0x18000    # [4,1536] FP32 (24KB)
VA_GBUF16_1536 = 0x12000  # aliases GVAL32
VA_XCACHE32_1536 = 0x12000  # aliases GVAL32

# (B) HMode-specialized addresses (module-level ints).
#     [rows,64] FP32 stride: 0x800 (rows=8) / 0x400 (rows=4).
#     [1,H_CHUNK] gamma: BF16 = H*2, FP32 = H*4.

# ---- HMode = 0 (H = 1280, rows = 8) ----
VA_GGATE32_1280 = 0x33000
VA_GSCORE32_1280 = 0x33800
VA_RSQ32_1280 = 0x34000
VA_RINV32_1280 = 0x34800
VA_RMEAN32_1280 = 0x35000
VA_GAMMA16_1280 = 0x35800  # [1,1280] BF16 gamma
VA_GAMMA32_1280 = 0x36200  # [1,1280] FP32 gamma
VA_GGAMQ32_1280 = 0x37600  # [1,1280] FP32 grad_gamma -- query
VA_GGAMK32_1280 = 0x38A00  # [1,1280] FP32 grad_gamma -- key
VA_GGAMQ_ACC32_1280 = 0x39E00  # [1,1280] FP32 RMW acc -- query   (ends 0x3B200)

# ---- HMode = 1 (H = 2560, rows = 4) ----
VA_GGATE32_2560 = 0x32800
VA_GSCORE32_2560 = 0x32C00
VA_RSQ32_2560 = 0x33000
VA_RINV32_2560 = 0x33400
VA_RMEAN32_2560 = 0x33800
VA_GAMMA16_2560 = 0x33C00  # [1,2560] BF16 gamma
VA_GAMMA32_2560 = 0x35000  # [1,2560] FP32 gamma
VA_GGAMQ32_2560 = 0x37800  # [1,2560] FP32 grad_gamma -- query
VA_GGAMK32_2560 = 0x3A000  # [1,2560] FP32 grad_gamma -- key
VA_GGAMQ_ACC32_2560 = 0x3C800  # [1,2560] FP32 RMW acc -- query   (ends 0x3F000)

# ---- HMode = 2 (H = 2048, rows = 4): (B) starts at (A) end 0x28000 ----
VA_GGATE32_2048 = 0x28800
VA_GSCORE32_2048 = 0x28C00
VA_RSQ32_2048 = 0x29000
VA_RINV32_2048 = 0x29400
VA_RMEAN32_2048 = 0x29800
VA_GAMMA16_2048 = 0x29C00  # [1,2048] BF16 gamma (0x1000)
VA_GAMMA32_2048 = 0x2AC00  # [1,2048] FP32 gamma (0x2000)
VA_GGAMQ32_2048 = 0x2CC00  # [1,2048] FP32 grad_gamma -- query
VA_GGAMK32_2048 = 0x2EC00  # [1,2048] FP32 grad_gamma -- key
VA_GGAMQ_ACC32_2048 = 0x30C00  # [1,2048] FP32 RMW acc -- query

# ---- HMode = 3 (H = 1536, rows = 4): (B) fully-independent, after (A) end 0x1E000 ----
# rows=4 makes the (A) group end at 0x1E000, leaving ample room for score/gamma
# tiles at independent addresses (like 1280/2048).  NO dead-region reuse.
VA_GGATE32_1536 = 0x1E800
VA_GSCORE32_1536 = 0x1EC00
VA_RSQ32_1536 = 0x1F000
VA_RINV32_1536 = 0x1F400
VA_RMEAN32_1536 = 0x1F800
VA_GAMMA16_1536 = 0x1FC00   # [1,1536] BF16 3K
VA_GAMMA32_1536 = 0x20800   # [1,1536] FP32 6K
VA_GGAMQ32_1536 = 0x22000   # [1,1536] FP32 6K
VA_GGAMK32_1536 = 0x23800   # [1,1536] FP32 6K
VA_GGAMQ_ACC32_1536 = 0x25000  # [1,1536] FP32 6K  ends 0x26800 (154KB)

# ---- HMode 0 (H=1280, A-rows=8) ----   ---- HMode 1 (H=2560, A-rows=4) same addrs ----
VA_PA_GO16_1280 = 0x00000   # BF16 [8,1280] (20KB)
VA_PA_GO32_1280 = 0x05000   # FP32 [8,1280] (40KB)
VA_PA_VAL32_1280 = 0x0F000  # FP32 [8,1280] (40KB) value cache for Step7b
VA_PA_GV32_1280 = 0x19000   # FP32 [8,1280] (40KB) grad_value accumulator
VA_PA_GV16_1280 = 0x23000   # BF16 [8,1280] (20KB)
VA_PA_GATE32_1280 = 0x28000  # FP32 [8,64] (2KB)
VA_PA_SCORE32_1280 = 0x28800 # FP32 [8,64] (2KB) scores for Step5
VA_PA_GG32_1280 = 0x29000   # FP32 [8,64] (2KB) grad_gate temp
VA_PA_GS32_1280 = 0x29800   # FP32 [8,64] (2KB) grad_score output
# HMode 1 byte-identical (4×2560 == 8×1280)
VA_PA_GO16_2560 = 0x00000
VA_PA_GO32_2560 = 0x05000
VA_PA_VAL32_2560 = 0x0F000
VA_PA_GV32_2560 = 0x19000
VA_PA_GV16_2560 = 0x23000
VA_PA_GATE32_2560 = 0x28000
VA_PA_SCORE32_2560 = 0x28800
VA_PA_GG32_2560 = 0x29000
VA_PA_GS32_2560 = 0x29800
# ---- HMode 2 (H=2048, A-rows=4) ----
VA_PA_GO16_2048 = 0x00000   # BF16 [4,2048] (16KB)
VA_PA_GO32_2048 = 0x04000   # FP32 [4,2048] (32KB)
VA_PA_VAL32_2048 = 0x0C000  # FP32 [4,2048] (32KB)
VA_PA_GV32_2048 = 0x14000   # FP32 [4,2048] (32KB)
VA_PA_GV16_2048 = 0x1C000   # BF16 [4,2048] (16KB)
VA_PA_GATE32_2048 = 0x20000  # FP32 [4,64] (1KB)
VA_PA_SCORE32_2048 = 0x20400
VA_PA_GG32_2048 = 0x20800
VA_PA_GS32_2048 = 0x20C00
# ---- HMode 3 (H=1536, A-rows=4) ----
VA_PA_GO16_1536 = 0x00000   # BF16 [4,1536] (12KB)
VA_PA_GO32_1536 = 0x03000   # FP32 [4,1536] (24KB)
VA_PA_VAL32_1536 = 0x09000  # FP32 [4,1536] (24KB)
VA_PA_GV32_1536 = 0x0F000   # FP32 [4,1536] (24KB)
VA_PA_GV16_1536 = 0x15000   # BF16 [4,1536] (12KB)
VA_PA_GATE32_1536 = 0x18000  # FP32 [4,64] (1KB)
VA_PA_SCORE32_1536 = 0x18400
VA_PA_GG32_1536 = 0x18800
VA_PA_GS32_1536 = 0x18C00

# ---- VAL32: value 复用 cache 地址 (HMode 2/3 被 GAMMA32_K 别名引用) ----
VA_VAL32_2048 = 0x32C00   # [4,2048] FP32 = 32KB
VA_VAL32_1536 = 0x26800   # [4,1536] FP32 = 24KB

# ---- GAMMA32_K: second gamma tile (key_gamma), pre-loaded ONCE per m_head ----
VA_GAMMA32_K_1280 = 0x00000             # [1,1280] FP32 = 5KB
VA_GAMMA32_K_2560 = 0x00000             # [1,2560] FP32 = 10KB
VA_GAMMA32_K_2048 = VA_VAL32_2048       # [1,2048] FP32 = 8KB
VA_GAMMA32_K_1536 = VA_VAL32_1536       # [1,1536] FP32 = 6KB

VA_RMEANQ32_1280 = VA_GGATE32_1280      # 0x33000 [8,64] FP32 = 2KB
VA_RMEANQ32_2560 = VA_GGATE32_2560      # 0x32800 [4,64] FP32 = 1KB
VA_RMEANQ32_2048 = VA_GGATE32_2048      # 0x28800 [4,64] FP32 = 1KB
VA_RMEANQ32_1536 = VA_GGATE32_1536      # 0x1E800 [4,64] FP32 = 1KB

# L1 addresses (cube section)
LA_L1_A_BASE = 0x00000   # a_l1 4-buffer base (slots at +0x0000/+0x8000/+0x10000/+0x18000)
LA_L1_B_BASE = 0x20000   # b_l1 4-buffer base (slots at +0x0000/+0x8000/+0x10000/+0x18000)
LA_L1_SLOT_STRIDE = 0x8000   # 32KB = [128,128] BF16
# L0A/L0B/L0C tile group addresses are written inline (0x0000/0x8000/0x10000).

# Cross-core event id 规划。
EVENT_GV_WAVE_BASE = 0               # grad_value_ws wave-ready pair {0,1}
EVENT_GK_BASE = 2                    # per-head grad_key_ws ready pair {2,3}
EVENT_GK_ACK_BASE = 4                # cube->vec per-head backpressure pair {4,5}
EVENT_GV_ACK_BASE = 6                # cube->vec wave backpressure pair {6,7}


# ═══════════════════════════════════════════════════════════════════
# Layer C: VF helpers — ALL FP32 in/out (NO vf.astype)
# ═══════════════════════════════════════════════════════════════════

@pl.vector_function
def vf_zero_2d(tile, n_rows, n_cols, row_stride):
    """Zero a [n_rows, n_cols] FP32 tile with an explicit row stride."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32
    for m in pl.range(0, n_rows):
        for r in pl.range(0, n_regs):
            vf.store_align(tile + m * row_stride + r * LANES_FP32, zero, preg)


@pl.vector_function
def vf_zero_1d(tile, n_cols):
    """Zero a [1, n_cols] FP32 tile."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32
    for r in pl.range(0, n_regs):
        vf.store_align(tile + r * LANES_FP32, zero, preg)


@pl.vector_function
def vf_grad_value_accum(
    go_f32,            # [TILE_BS_VEC, H_CHUNK] FP32  grad_output for one head
    gate_f32,          # [TILE_BS_VEC, 64]      FP32  gate (scalar per row, lane 0)
    gv_acc,            # [TILE_BS_VEC, H_CHUNK] FP32 in/out  grad_value accumulator
    n_rows, n_cols, row_stride,
):
    """Step7a^{-1}: gv_acc[r,h] += go[r,h] * gate[r]   (gate broadcast over h)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32
    for m in pl.range(0, n_rows):
        gate_reg = vf.load_align(gate_f32, m * 64)
        gate_b = vf.full(gate_reg, preg)
        for r in pl.range(0, n_regs):
            off = m * row_stride + r * LANES_FP32
            g = vf.load_align(go_f32, off)
            a = vf.load_align(gv_acc, off)
            prod = vf.mul(g, gate_b, preg)
            res = vf.add(a, prod, preg)
            vf.store_align(gv_acc + off, res, preg)


@pl.vector_function
def vf_grad_gate_accum(
    go_f32,            # [TILE_BS_VEC, H_CHUNK] FP32  grad_output for this head
    val_f32,           # [TILE_BS_VEC, H_CHUNK] FP32  value (shared)
    gg_acc,            # [TILE_BS_VEC, 64] FP32 in/out  grad_gate accumulator
    n_rows, n_cols, row_stride,
):
    """Step7b^{-1}: gg_acc[r] += Σ_h(go[r,h] * val[r,h]) for this H-chunk."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32
    for m in pl.range(0, n_rows):
        gg = vf.load_align(gg_acc, m * 64)
        for r in pl.range(0, n_regs):
            off = m * row_stride + r * LANES_FP32
            g = vf.load_align(go_f32, off)
            v = vf.load_align(val_f32, off)
            prod = vf.mul(g, v, preg)
            part = vf.reduce_sum(prod, preg, merge_mode=pl.MergeMode.ZEROING)
            gg = vf.add(gg, part, preg)
        vf.store_align(gg_acc + m * 64, gg, preg)


@pl.vector_function
def vf_gate_bw(
    gg_f32,            # [TILE_BS_VEC, 64] FP32  grad_gate (scalar per row)
    score_f32,         # [TILE_BS_VEC, 64] FP32  score (scalar per row)
    gate_f32,          # [TILE_BS_VEC, 64] FP32  gate g (scalar per row)
    gs_out,            # [TILE_BS_VEC, 64] FP32 write  grad_score
    n_rows,
    clamp_value,
):
    """Step5^{-1}: signed_sqrt_gate backward.
       grad_score = grad_gate · g(1−g) · mask / (2·√max(|s|,c) + 1e-12)
       mask = (|s| > c)
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    clamp_reg = vf.full(clamp_value, preg, dtype=pl.DT_FP32)
    eps_reg = vf.full(GATE_EPS, preg, dtype=pl.DT_FP32)
    two_reg = vf.full(2.0, preg, dtype=pl.DT_FP32)
    zero_reg = vf.full(0.0, preg, dtype=pl.DT_FP32)
    one_reg = vf.full(1.0, preg, dtype=pl.DT_FP32)

    for m in pl.range(0, n_rows):
        gg = vf.load_align(gg_f32, m * 64)
        gg_b = vf.full(gg, preg)
        sc = vf.load_align(score_f32, m * 64)
        sc_b = vf.full(sc, preg)
        gt = vf.load_align(gate_f32, m * 64)
        gt_b = vf.full(gt, preg)

        # sigmoid_grad = g - g*g = g(1-g)
        g_sq = vf.mul(gt_b, gt_b, preg)
        sig_grad = vf.sub(gt_b, g_sq, preg)
        # mask = (|s| > clamp) ? 1 : 0
        abs_s = vf.abs(sc_b, preg)
        mask_gt = vf.gt(abs_s, clamp_reg, preg)
        mask = vf.select(one_reg, zero_reg, mask_gt)
        # sqrt_abs = sqrt(max(|s|, clamp))
        clamped = vf.max(abs_s, clamp_reg, preg)
        sqrt_abs = vf.sqrt(clamped, preg)
        # logits_grad = mask / (2*sqrt_abs + 1e-12)
        denom = vf.mul(two_reg, sqrt_abs, preg)
        denom = vf.add(denom, eps_reg, preg)
        logits_grad = vf.div(mask, denom, preg)
        # grad_score = grad_gate · sigmoid_grad · logits_grad
        result = vf.mul(gg_b, sig_grad, preg)
        result = vf.mul(result, logits_grad, preg)
        vf.store_align(gs_out + m * 64, result, preg)


@pl.vector_function
def vf_scaled_dot(
    gs_f32,            # [TILE_BS_VEC, 64]      FP32  grad_score (scalar per row)
    partner_f32,       # [TILE_BS_VEC, H_CHUNK] FP32  the "other" normed vector
    out_f32,           # [TILE_BS_VEC, H_CHUNK] FP32 write  result
    n_rows, n_cols, row_stride, h_value,
):
    """Step4^{-1}: out[r,h] = grad_score[r] · (1/√H) · partner[r,h].
       Used as: partner=normed_key -> grad_nQuery; partner=normed_query -> grad_nKey.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32
    # Do not pass 1/sqrt(H) as a VF scalar: scalar constants are lowered
    # through the BF16 scalar path on this target.  H itself is exact, so
    # construct the FP32 scale in vector registers.
    h_reg = vf.full(h_value, preg, dtype=pl.DT_FP32)
    sqrt_h = vf.sqrt(h_reg, preg)
    for m in pl.range(0, n_rows):
        gs = vf.load_align(gs_f32, m * 64)
        gs_b = vf.full(gs, preg)
        gs_scaled = vf.div(gs_b, sqrt_h, preg)
        for r in pl.range(0, n_regs):
            off = m * row_stride + r * LANES_FP32
            p = vf.load_align(partner_f32, off)
            res = vf.mul(gs_scaled, p, preg)
            vf.store_align(out_f32 + off, res, preg)


# ── rms_norm_backward (3-pass; H-split cross-chunk accumulation) ──
@pl.vector_function
def vf_compute_inv_rms(
    x_f32,              # [TILE_BS_VEC, H_CHUNK] FP32  x (key or hidden)
    rms_out,            # [TILE_BS_VEC, 64] FP32 write  inv_rms = 1/sqrt(mean(x²)+eps)
    n_rows, n_cols, row_stride,
    h_value, eps,
):
    """inv_rms[r] = 1 / sqrt(Σx²/H + eps).  H itself is exact, so the scale is
       constructed in vector registers (same trick as vf_scaled_dot)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32
    h_reg = vf.full(h_value, preg, dtype=pl.DT_FP32)
    eps_reg = vf.full(eps, preg, dtype=pl.DT_FP32)
    one_reg = vf.full(1.0, preg, dtype=pl.DT_FP32)
    for m in pl.range(0, n_rows):
        acc = vf.full(0.0, preg, dtype=pl.DT_FP32)
        for r in pl.range(0, n_regs):
            off = m * row_stride + r * LANES_FP32
            x = vf.load_align(x_f32, off)
            sq = vf.mul(x, x, preg)
            part = vf.reduce_sum(sq, preg, merge_mode=pl.MergeMode.ZEROING)
            acc = vf.add(acc, part, preg)
        mean = vf.div(acc, h_reg, preg)
        mean_eps = vf.add(mean, eps_reg, preg)
        sqrt_v = vf.sqrt(mean_eps, preg)
        inv_rms = vf.div(one_reg, sqrt_v, preg)
        vf.store_align(rms_out + m * 64, inv_rms, preg)

@pl.vector_function
def vf_rmsnorm_fwd(x_f32, gam_f32, rms_f32, out_f32, n_rows, n_cols, row_stride):
    """Recompute RMSNorm(x, gamma) in FP32 without materializing a cache.
       rms_f32 stores inv_rms (1/rms), recomputed in-kernel → n = x * inv_rms.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32
    for m in pl.range(0, n_rows):
        rms = vf.load_align(rms_f32, m * 64)
        rms_b = vf.full(rms, preg)
        for r in pl.range(0, n_regs):
            off = m * row_stride + r * LANES_FP32
            x = vf.load_align(x_f32, off)
            gam = vf.load_align(gam_f32, r * LANES_FP32)
            normed = vf.mul(x, rms_b, preg)
            out = vf.mul(normed, gam, preg)
            vf.store_align(out_f32 + off, out, preg)


@pl.vector_function
def vf_rmsbw_rmean(
    gx_f32,            # [TILE_BS_VEC, H_CHUNK] FP32  grad_xhat (= grad_n)
    x_f32,             # [TILE_BS_VEC, H_CHUNK] FP32  x
    gam_f32,           # [1, H_CHUNK] FP32  gamma
    rms_f32,           # [TILE_BS_VEC, 64] FP32  inv_rms per row
    rmean_acc,         # [TILE_BS_VEC, 64] FP32 in/out
    gg_acc,            # [1, H_CHUNK] FP32 in/out  grad_gamma partial
    n_rows, n_cols, row_stride,
):
    """rms_bw Pass 2: rmean[r] += Σ(grad_n·n); gg_acc[h] += Σ grad_xhat·n.
       n = x * inv_rms ; grad_n = grad_xhat · gamma.
       rms_f32 stores inv_rms (1/rms), recomputed in-kernel.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32
    for m in pl.range(0, n_rows):
        rms = vf.load_align(rms_f32, m * 64)
        rms_b = vf.full(rms, preg)
        rmean = vf.load_align(rmean_acc, m * 64)
        for r in pl.range(0, n_regs):
            off = m * row_stride + r * LANES_FP32
            x = vf.load_align(x_f32, off)
            n_reg = vf.mul(x, rms_b, preg)
            gx = vf.load_align(gx_f32, off)
            gam = vf.load_align(gam_f32, r * LANES_FP32)
            grad_n = vf.mul(gx, gam, preg)
            dn = vf.mul(grad_n, n_reg, preg)
            part = vf.reduce_sum(dn, preg, merge_mode=pl.MergeMode.ZEROING)
            rmean = vf.add(rmean, part, preg)
            dg = vf.mul(gx, n_reg, preg)
            prev_gg = vf.load_align(gg_acc, r * LANES_FP32)
            new_gg = vf.add(prev_gg, dg, preg)
            vf.store_align(gg_acc + r * LANES_FP32, new_gg, preg)
        vf.store_align(rmean_acc + m * 64, rmean, preg)


@pl.vector_function
def vf_rmsbw_rmean_finalize(rmean_acc, n_rows, h_value):
    """Finalize rmean by dividing the FP32 sum by the exact integer H."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    h_reg = vf.full(h_value, preg, dtype=pl.DT_FP32)
    for m in pl.range(0, n_rows):
        val = vf.load_align(rmean_acc, m * 64)
        val = vf.div(val, h_reg, preg)
        vf.store_align(rmean_acc + m * 64, val, preg)


@pl.vector_function
def vf_rmsbw_gradx(
    gx_f32,            # [TILE_BS_VEC, H_CHUNK] FP32  grad_xhat
    x_f32,             # [TILE_BS_VEC, H_CHUNK] FP32  x
    gam_f32,           # [1, H_CHUNK] FP32  gamma
    rms_f32,           # [TILE_BS_VEC, 64] FP32  inv_rms (from forward cache)
    rmean_f32,         # [TILE_BS_VEC, 64] FP32  rmean (mean)
    gx_out_f32,        # [TILE_BS_VEC, H_CHUNK] FP32 write  grad_x
    n_rows, n_cols, row_stride,
):
    """rms_bw Pass 3: grad_x = (grad_n − n·rmean) * inv_rms.
       n = x * inv_rms ; grad_n = grad_xhat · gamma.
       rms_f32 stores inv_rms (1/rms).
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32
    for m in pl.range(0, n_rows):
        rms = vf.load_align(rms_f32, m * 64)
        rms_b = vf.full(rms, preg)
        rmean = vf.load_align(rmean_f32, m * 64)
        rmean_b = vf.full(rmean, preg)
        for r in pl.range(0, n_regs):
            off = m * row_stride + r * LANES_FP32
            x = vf.load_align(x_f32, off)
            n_reg = vf.mul(x, rms_b, preg)
            gx = vf.load_align(gx_f32, off)
            gam = vf.load_align(gam_f32, r * LANES_FP32)
            grad_n = vf.mul(gx, gam, preg)
            n_rmean = vf.mul(n_reg, rmean_b, preg)
            diff = vf.sub(grad_n, n_rmean, preg)
            grad_x = vf.mul(diff, rms_b, preg)
            vf.store_align(gx_out_f32 + off, grad_x, preg)


# ═══════════════════════════════════════════════════════════════════
# Layer D: engram_backward_kernel
# ═══════════════════════════════════════════════════════════════════

@pl.jit(auto_mutex=True, tiling_key=EngramTilingKey)
def engram_backward_kernel(
    # Inputs (scores/gates are FP32; other activation inputs are BF16)
    grad_output: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    hidden_states: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    embeddings: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    key_gamma: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    query_gamma: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    scores: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 64], pl.DT_FP32],
    gates: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 64], pl.DT_FP32],
    keys: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    value: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    # host-pre-transposed inputs (plain NZ loads, NO is_transpose / layout=ZN)
    emb_t: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],  # [De, BS] BF16
    wv_t: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],  # [H, De] BF16
    wk_t: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],  # [M_H, H, De] BF16
    # BF16 cube operand workspaces (vector produces FP32, casts to BF16 on store)
    grad_value_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],  # [M, H]
    grad_key_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],  # [M, M_H, H]
    gscore_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 64], pl.DT_FP32],  # [M, M_H, 64] grad_score cache
    # FP32 per-subblock gamma workspace [num_cores*2, M_H, H]
    grad_qgamma_acc: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    grad_kgamma_acc: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    # FP32 grad_emb accumulator [M, De]: Nest1 value + per-head key all atomic-add
    # here in FP32; host casts to BF16 grad_embeddings. MUST be pre-zeroed.
    grad_emb_acc: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],  # [M, De]
    # BF16 outputs (grad_embeddings produced by host from grad_emb_acc)
    grad_hidden_states: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    grad_key_proj_weights: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    grad_value_proj_weights: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    # attrs (runtime scalars, aligned with the pypto tensor-version API)
    clamp_value: pl.DT_FP32,
    eps: pl.DT_FP32,
):
    """Single kernel: vector section (produces grad_value_ws/grad_key_ws +
    grad_hidden + grad_γ) then cube section (consumes ws, produces grad_emb +
    grad_W). BF16 activation inputs are cast once at the computation boundary;
    grad_value_ws and grad_key_ws feed legal FP32 Cube paths directly.
    Final output stores are low precision.
    """
    bs = grad_output.shape[0]
    m_h = grad_output.shape[1]  # head count, dynamic (1-16)
    h = grad_output.shape[2]
    de = embeddings.shape[1]

    # Bind vector tile sizes to compile-time constants via TilingKey. HMode/BSMode
    # are ConstInt -> `if` parses only the taken branch, so these names
    # are usable in TileType.shape and pl.range steps.  Rows fixed 4 (B-phase).
    if HMode == 0:  # H = 1280
        tile_bs_vec = TILE_BS_VEC_1280  # 8
        tile_bs_vec_a = TILE_BS_VEC_A_1280  # 16
        h_chunk_size = H_CHUNK_1280  # 1280
    elif HMode == 1:  # H = 2560
        tile_bs_vec = TILE_BS_VEC_2560  # 4
        tile_bs_vec_a = TILE_BS_VEC_A_2560  # 8
        h_chunk_size = H_CHUNK_2560  # 2560
    elif HMode == 2:  # H = 2048
        tile_bs_vec = TILE_BS_VEC_2048  # 4
        tile_bs_vec_a = TILE_BS_VEC_A_2048  # 8
        h_chunk_size = H_CHUNK_2048  # 2048
    else:  # HMode == 3: H = 1536
        tile_bs_vec = TILE_BS_VEC_1536  # 8
        tile_bs_vec_a = TILE_BS_VEC_A_1536  # 16
        h_chunk_size = H_CHUNK_1536  # 1536

    if BSMode == 0:  # BS >= 8192
        v_tile = V_TILE_BS0  # 128
    elif BSMode == 1:  # BS >= 4096
        v_tile = V_TILE_BS1  # 64
    elif BSMode == 2:  # BS >= 2048
        v_tile = V_TILE_BS2  # 32
    else:  # BSMode == 3: small BS
        v_tile = V_TILE_BS3  # 16
    n_h_chunks = (h + h_chunk_size - 1) // h_chunk_size  # exactly one full-H tile
    cube_k = 128  # cube K tile: BF16 [128,128]=32KB fits L0A/L0B double-buffer
    cube_mn = 128  # cube M/N output tile
    n_kh = (h + cube_k - 1) // cube_k  # K=H slice count (Nest1 reduces over H)
    n_kbs = (bs + cube_k - 1) // cube_k  # K=BS slice count (Nest2/3 reduce over BS)
    n_de = (de + cube_mn - 1) // cube_mn  # De output tile count
    n_h = (h + cube_mn - 1) // cube_mn  # H output tile count (Nest2/3 N=H)
    n_bs = (bs + cube_mn - 1) // cube_mn  # cube BS output tile count (Nest1 M=BS)
    n_bs_v = (bs + v_tile - 1) // v_tile  # vector worker tile count (>=64 fills workers)
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    num_subcores = 2                       # AIV block has 2 subblocks (sub_idx 0/1)
    sub_id = pl.get_subblock_idx()         # 0 or 1 within this AI Core
    total_subblocks = num_cores * num_subcores
    worker_id = core_id * num_subcores + sub_id   # global vector-subblock index
    iters_per_core = (n_bs + num_cores - 1) // num_cores      # per-core BS-tile iterations (cube)
    iters_per_subblock = (n_bs_v + total_subblocks - 1) // total_subblocks  # per-worker
    gv_groups = iters_per_subblock
    gv_iters_per_group = (iters_per_subblock + gv_groups - 1) // gv_groups  # = 1

    # ═══════════════════════════════════════════════════════════════
    # VECTOR SECTION (runs first; produces grad_value_ws / grad_key_ws)
    # ═══════════════════════════════════════════════════════════════
    tt_mv16 = pl.TileType(shape=[tile_bs_vec, h_chunk_size], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    tt_mv32 = pl.TileType(shape=[tile_bs_vec, h_chunk_size], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    tt_score32 = pl.TileType(shape=[tile_bs_vec, 64], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    tt_gamma16 = pl.TileType(shape=[1, h_chunk_size], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    tt_gamma32 = pl.TileType(shape=[1, h_chunk_size], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])

    # Tile capacity is H_CHUNK (1280/2560), but every vector op uses the actual H
    # as its valid width (set_validshape). H <= H_CHUNK, so H is never split.

    # (A) group -- HMode-specialized: HModes 0/1 share addrs (byte-constant), 2/3 differ.
    gout32_grp = pl.make_tile_group(
        type=tt_mv32,
        addrs=((HMode == 0) * VA_GOUT32_1280 + (HMode == 1) * VA_GOUT32_2560 +
               (HMode == 2) * VA_GOUT32_2048 + (HMode == 3) * VA_GOUT32_1536),
        mutex_ids=[1])
    ptnr16_grp = pl.make_tile_group(
        type=tt_mv16,
        addrs=((HMode == 0) * VA_PTNR16_1280 + (HMode == 1) * VA_PTNR16_2560 +
               (HMode == 2) * VA_PTNR16_2048 + (HMode == 3) * VA_PTNR16_1536),
        mutex_ids=[2])
    ptnr32_grp = pl.make_tile_group(
        type=tt_mv32,
        addrs=((HMode == 0) * VA_PTNR32_1280 + (HMode == 1) * VA_PTNR32_2560 +
               (HMode == 2) * VA_PTNR32_2048 + (HMode == 3) * VA_PTNR32_1536),
        mutex_ids=[3])
    gn32_grp = pl.make_tile_group(
        type=tt_mv32,
        addrs=((HMode == 0) * VA_GN32_1280 + (HMode == 1) * VA_GN32_2560 +
               (HMode == 2) * VA_GN32_2048 + (HMode == 3) * VA_GN32_1536),
        mutex_ids=[17])
    gbuf16_grp = pl.make_tile_group(
        type=tt_mv16,
        addrs=((HMode == 0) * VA_GBUF16_1280 + (HMode == 1) * VA_GBUF16_2560 +
               (HMode == 2) * VA_GBUF16_2048 + (HMode == 3) * VA_GBUF16_1536),
        mutex_ids=[4])
    # (B) group -- HMode-specialized [rows,64]/[1,H_CHUNK] addresses.
    gscore_grp = pl.make_tile_group(
        type=tt_score32,
        addrs=((HMode == 0) * VA_GSCORE32_1280 + (HMode == 1) * VA_GSCORE32_2560 +
               (HMode == 2) * VA_GSCORE32_2048 + (HMode == 3) * VA_GSCORE32_1536),
        mutex_ids=[10])
    rsq_grp = pl.make_tile_group(
        type=tt_score32,
        addrs=((HMode == 0) * VA_RSQ32_1280 + (HMode == 1) * VA_RSQ32_2560 +
               (HMode == 2) * VA_RSQ32_2048 + (HMode == 3) * VA_RSQ32_1536),
        mutex_ids=[11])
    rinv_grp = pl.make_tile_group(
        type=tt_score32,
        addrs=((HMode == 0) * VA_RINV32_1280 + (HMode == 1) * VA_RINV32_2560 +
               (HMode == 2) * VA_RINV32_2048 + (HMode == 3) * VA_RINV32_1536),
        mutex_ids=[12])
    rmean_grp = pl.make_tile_group(
        type=tt_score32,
        addrs=((HMode == 0) * VA_RMEAN32_1280 + (HMode == 1) * VA_RMEAN32_2560 +
               (HMode == 2) * VA_RMEAN32_2048 + (HMode == 3) * VA_RMEAN32_1536),
        mutex_ids=[13])
    rmeanq_grp = pl.make_tile_group(
        type=tt_score32,
        addrs=((HMode == 0) * VA_RMEANQ32_1280 + (HMode == 1) * VA_RMEANQ32_2560 +
               (HMode == 2) * VA_RMEANQ32_2048 + (HMode == 3) * VA_RMEANQ32_1536),
        mutex_ids=[6])
    gamma16_grp = pl.make_tile_group(
        type=tt_gamma16,
        addrs=((HMode == 0) * VA_GAMMA16_1280 + (HMode == 1) * VA_GAMMA16_2560 +
               (HMode == 2) * VA_GAMMA16_2048 + (HMode == 3) * VA_GAMMA16_1536),
        mutex_ids=[14])
    gamma32_grp = pl.make_tile_group(
        type=tt_gamma32,
        addrs=((HMode == 0) * VA_GAMMA32_1280 + (HMode == 1) * VA_GAMMA32_2560 +
               (HMode == 2) * VA_GAMMA32_2048 + (HMode == 3) * VA_GAMMA32_1536),
        mutex_ids=[15])
    gamma32_k_grp = pl.make_tile_group(
        type=tt_gamma32,
        addrs=((HMode == 0) * VA_GAMMA32_K_1280 + (HMode == 1) * VA_GAMMA32_K_2560 +
               (HMode == 2) * VA_GAMMA32_K_2048 + (HMode == 3) * VA_GAMMA32_K_1536),
        mutex_ids=[21])
    ggamq_grp = pl.make_tile_group(
        type=tt_gamma32,
        addrs=((HMode == 0) * VA_GGAMQ32_1280 + (HMode == 1) * VA_GGAMQ32_2560 +
               (HMode == 2) * VA_GGAMQ32_2048 + (HMode == 3) * VA_GGAMQ32_1536),
        mutex_ids=[16])
    ggamk_grp = pl.make_tile_group(
        type=tt_gamma32,
        addrs=((HMode == 0) * VA_GGAMK32_1280 + (HMode == 1) * VA_GGAMK32_2560 +
               (HMode == 2) * VA_GGAMK32_2048 + (HMode == 3) * VA_GGAMK32_1536),
        mutex_ids=[18])
    ggamq_acc_grp = pl.make_tile_group(
        type=tt_gamma32,
        addrs=((HMode == 0) * VA_GGAMQ_ACC32_1280 + (HMode == 1) * VA_GGAMQ_ACC32_2560 +
               (HMode == 2) * VA_GGAMQ_ACC32_2048 + (HMode == 3) * VA_GGAMQ_ACC32_1536),
        mutex_ids=[19])
    # RMS-x cache: keys (B-key) / hidden (B-query), loaded once in RMS Pass1,
    # reused in Pass2/Pass3. Aliases grad_value (dead after GV_DONE).
    xcache_grp = pl.make_tile_group(
        type=tt_mv32,
        addrs=((HMode == 0) * VA_XCACHE32_1280 + (HMode == 1) * VA_XCACHE32_2560 +
               (HMode == 2) * VA_XCACHE32_2048 + (HMode == 3) * VA_XCACHE32_1536),
        mutex_ids=[4])

    tt_pa16 = pl.TileType(shape=[tile_bs_vec_a, h_chunk_size], dtype=pl.DT_BF16,
                          target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    tt_pa32 = pl.TileType(shape=[tile_bs_vec_a, h_chunk_size], dtype=pl.DT_FP32,
                          target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    tt_pa_sc = pl.TileType(shape=[tile_bs_vec_a, 64], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    pa_go16_grp = pl.make_tile_group(
        type=tt_pa16,
        addrs=((HMode == 0) * VA_PA_GO16_1280 + (HMode == 1) * VA_PA_GO16_2560 +
               (HMode == 2) * VA_PA_GO16_2048 + (HMode == 3) * VA_PA_GO16_1536),
        mutex_ids=[22])
    pa_go32_grp = pl.make_tile_group(
        type=tt_pa32,
        addrs=((HMode == 0) * VA_PA_GO32_1280 + (HMode == 1) * VA_PA_GO32_2560 +
               (HMode == 2) * VA_PA_GO32_2048 + (HMode == 3) * VA_PA_GO32_1536),
        mutex_ids=[23])
    pa_val32_grp = pl.make_tile_group(
        type=tt_pa32,
        addrs=((HMode == 0) * VA_PA_VAL32_1280 + (HMode == 1) * VA_PA_VAL32_2560 +
               (HMode == 2) * VA_PA_VAL32_2048 + (HMode == 3) * VA_PA_VAL32_1536),
        mutex_ids=[28])
    pa_gv32_grp = pl.make_tile_group(
        type=tt_pa32,
        addrs=((HMode == 0) * VA_PA_GV32_1280 + (HMode == 1) * VA_PA_GV32_2560 +
               (HMode == 2) * VA_PA_GV32_2048 + (HMode == 3) * VA_PA_GV32_1536),
        mutex_ids=[24])
    pa_gv16_grp = pl.make_tile_group(
        type=tt_pa16,
        addrs=((HMode == 0) * VA_PA_GV16_1280 + (HMode == 1) * VA_PA_GV16_2560 +
               (HMode == 2) * VA_PA_GV16_2048 + (HMode == 3) * VA_PA_GV16_1536),
        mutex_ids=[25])
    pa_gate32_grp = pl.make_tile_group(
        type=tt_pa_sc,
        addrs=((HMode == 0) * VA_PA_GATE32_1280 + (HMode == 1) * VA_PA_GATE32_2560 +
               (HMode == 2) * VA_PA_GATE32_2048 + (HMode == 3) * VA_PA_GATE32_1536),
        mutex_ids=[26])
    pa_score32_grp = pl.make_tile_group(
        type=tt_pa_sc,
        addrs=((HMode == 0) * VA_PA_SCORE32_1280 + (HMode == 1) * VA_PA_SCORE32_2560 +
               (HMode == 2) * VA_PA_SCORE32_2048 + (HMode == 3) * VA_PA_SCORE32_1536),
        mutex_ids=[29])
    pa_gg32_grp = pl.make_tile_group(
        type=tt_pa_sc,
        addrs=((HMode == 0) * VA_PA_GG32_1280 + (HMode == 1) * VA_PA_GG32_2560 +
               (HMode == 2) * VA_PA_GG32_2048 + (HMode == 3) * VA_PA_GG32_1536),
        mutex_ids=[30])
    pa_gs32_grp = pl.make_tile_group(
        type=tt_pa_sc,
        addrs=((HMode == 0) * VA_PA_GS32_1280 + (HMode == 1) * VA_PA_GS32_2560 +
               (HMode == 2) * VA_PA_GS32_2048 + (HMode == 3) * VA_PA_GS32_1536),
        mutex_ids=[31])

    with pl.section_vector():
        # ── Phase A waves (wave-major row ownership) ──
        wave_span = num_cores * cube_mn
        n_waves = (bs + wave_span - 1) // wave_span
        share_w = wave_span // total_subblocks
        for wave_idx in pl.range(0, n_waves):
            row_base = wave_idx * wave_span + worker_id * share_w
            my_rows = pl.min(share_w, bs - row_base)
            if my_rows > 0:
                for bs_start in pl.range(0, my_rows, tile_bs_vec_a):
                    valid_bsv = pl.min(tile_bs_vec_a, my_rows - bs_start)
                    bs_off = row_base + bs_start
                    # Pass A+: Step7a (grad_value) + Step7b (grad_gate) + Step5 (grad_score)
                    # grad_output loaded ONCE (shared Step7a+Step7b), gates loaded ONCE
                    for h_chunk in pl.range(0, n_h_chunks):
                        h_off = h_chunk * h
                        # Preload value → pa_val32 (跨 m_head 复用)
                        pa_go16 = pa_go16_grp.current()
                        pl.set_validshape(pa_go16, [valid_bsv, h])
                        pl.load(pa_go16, value, [bs_off, h_off])
                        pa_val32 = pa_val32_grp.current()
                        pl.set_validshape(pa_val32, [valid_bsv, h])
                        pl.cast(pa_val32, pa_go16, mode=pl.RoundMode.CAST_NONE)

                        gv = pa_gv32_grp.current()
                        pl.set_validshape(gv, [valid_bsv, h])
                        vf_zero_2d(gv, valid_bsv, h, h_chunk_size)
                        for m_head in pl.range(0, m_h):
                            pa_go16 = pa_go16_grp.current()
                            pl.set_validshape(pa_go16, [valid_bsv, h])
                            pl.load(pa_go16, grad_output,
                                    [bs_off, m_head, h_off], order=[0, 2])
                            pa_go32 = pa_go32_grp.current()
                            pl.set_validshape(pa_go32, [valid_bsv, h])
                            pl.cast(pa_go32, pa_go16, mode=pl.RoundMode.CAST_NONE)
                            pa_gate32 = pa_gate32_grp.current()
                            pl.set_validshape(pa_gate32, [valid_bsv, 64])
                            pl.load(pa_gate32, gates, [bs_off, m_head, 0], order=[0, 2])
                            # Step7a: grad_value += go·gate
                            vf_grad_value_accum(pa_go32, pa_gate32, gv, valid_bsv, h, h_chunk_size)

                            # Step7b: grad_gate = Σ_h(go·val)
                            pa_gg32 = pa_gg32_grp.current()
                            pl.set_validshape(pa_gg32, [valid_bsv, 64])
                            vf_zero_2d(pa_gg32, valid_bsv, 64, 64)
                            vf_grad_gate_accum(pa_go32, pa_val32, pa_gg32,
                                               valid_bsv, h, h_chunk_size)

                            # Step5: grad_score = gate_bw(grad_gate, score, gate)
                            pa_score32 = pa_score32_grp.current()
                            pl.set_validshape(pa_score32, [valid_bsv, 64])
                            pl.load(pa_score32, scores, [bs_off, m_head, 0], order=[0, 2])
                            pa_gs32 = pa_gs32_grp.current()
                            pl.set_validshape(pa_gs32, [valid_bsv, 64])
                            vf_gate_bw(pa_gg32, pa_score32, pa_gate32, pa_gs32,
                                       valid_bsv, clamp_value)
                            pl.store(gscore_ws, pa_gs32, [bs_off, m_head, 0], order=[0, 2])

                        pa_gv16 = pa_gv16_grp.current()
                        pl.set_validshape(pa_gv16, [valid_bsv, h])
                        pl.cast(pa_gv16, gv, mode=pl.RoundMode.CAST_ROUND)  # FP32 -> BF16
                        pl.store(grad_value_ws, pa_gv16, [bs_off, h_off])

            pl.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)
            # 不变量：每 worker 每波必发（无行也发）；iters_per_core == n_waves
            # 保证 cube 每轮 wait 都有对应 set。复用 parity 槽位前须等 cube
            # 对 wave w-2 的 ACK，否则二进制 flag 合并（陈旧/未就绪读）。
            if wave_idx >= 2:
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=EVENT_GV_ACK_BASE + (wave_idx % 2),
                    sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE3,
                event_id=EVENT_GV_WAVE_BASE + (wave_idx % 2),
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

        # ===== Phase B fused pass: gscore_ws → KEY + QUERY in ONE loop =====
        for m_head in pl.range(0, m_h):
            for tile_idx in pl.range(0, iters_per_subblock):
                bsi = worker_id + tile_idx * total_subblocks
                if bsi < n_bs_v:
                    row_off = bsi * v_tile
                    bs_tile_rows = pl.min(v_tile, bs - row_off)
                    ggam_q = ggamq_grp.current()
                    pl.set_validshape(ggam_q, [1, h])
                    vf_zero_1d(ggam_q, h)
                    ggam_k = ggamk_grp.current()
                    pl.set_validshape(ggam_k, [1, h])
                    vf_zero_1d(ggam_k, h)
                    # Pre-load query_gamma→gamma32, key_gamma→gamma32_k ONCE per m_head.
                    for h_chunk in pl.range(0, n_h_chunks):
                        h_off = h_chunk * h
                        gamma16 = gamma16_grp.current()
                        pl.set_validshape(gamma16, [1, h])
                        gamma32 = gamma32_grp.current()
                        pl.set_validshape(gamma32, [1, h])
                        pl.load(gamma16, query_gamma, [m_head, h_off])
                        pl.cast(gamma32, gamma16, mode=pl.RoundMode.CAST_NONE)
                        gamma32_k = gamma32_k_grp.current()
                        pl.set_validshape(gamma32_k, [1, h])
                        pl.load(gamma16, key_gamma, [m_head, h_off])
                        pl.cast(gamma32_k, gamma16, mode=pl.RoundMode.CAST_NONE)

                    for bs_start in pl.range(0, bs_tile_rows, tile_bs_vec):
                        valid_bsv = pl.min(tile_bs_vec, bs_tile_rows - bs_start)
                        bs_off = row_off + bs_start
                        # ── grad_score loaded ONCE, shared by both chains ──
                        gs = gscore_grp.current()
                        pl.set_validshape(gs, [valid_bsv, 64])
                        pl.load(gs, gscore_ws, [bs_off, m_head, 0], order=[0, 2])
                        rms = rinv_grp.current()
                        pl.set_validshape(rms, [valid_bsv, 64])
                        partner_sq = rsq_grp.current()
                        pl.set_validshape(partner_sq, [valid_bsv, 64])

                        # ══ rmean stage — both chains share one tile pair ══
                        rmean_k = rmean_grp.current()
                        pl.set_validshape(rmean_k, [valid_bsv, 64])
                        vf_zero_2d(rmean_k, valid_bsv, 64, 64)
                        rmean_q = rmeanq_grp.current()
                        pl.set_validshape(rmean_q, [valid_bsv, 64])
                        vf_zero_2d(rmean_q, valid_bsv, 64, 64)
                        for h_chunk in pl.range(0, n_h_chunks):
                            h_off = h_chunk * h
                            ptnr16 = ptnr16_grp.current()
                            pl.set_validshape(ptnr16, [valid_bsv, h])
                            pl.load(ptnr16, keys,
                                    [bs_off, m_head, h_off], order=[0, 2])
                            keys32 = ptnr32_grp.current()
                            pl.set_validshape(keys32, [valid_bsv, h])
                            pl.cast(keys32, ptnr16, mode=pl.RoundMode.CAST_NONE)
                            pl.load(ptnr16, hidden_states,
                                    [bs_off, m_head, h_off], order=[0, 2])
                            hidden32 = xcache_grp.current()
                            pl.set_validshape(hidden32, [valid_bsv, h])
                            pl.cast(hidden32, ptnr16, mode=pl.RoundMode.CAST_NONE)
                            # recompute inv_rms (1/rms) from keys/hidden in FP32
                            vf_compute_inv_rms(keys32, rms, valid_bsv, h,
                                               h_chunk_size, h, eps)
                            vf_compute_inv_rms(hidden32, partner_sq, valid_bsv, h,
                                               h_chunk_size, h, eps)
                            gamma32 = gamma32_grp.current()  # query_gamma
                            gx_k = gn32_grp.current()
                            pl.set_validshape(gx_k, [valid_bsv, h])
                            vf_rmsnorm_fwd(hidden32, gamma32, partner_sq, gx_k,
                                           valid_bsv, h, h_chunk_size)
                            vf_scaled_dot(gs, gx_k, gx_k, valid_bsv, h,
                                          h_chunk_size, h)
                            gamma32_k = gamma32_k_grp.current()  # key_gamma
                            # ggam_q/ggam_k NOT zeroed here — accumulate across bs_starts
                            vf_rmsbw_rmean(gx_k, keys32, gamma32_k, rms, rmean_k,
                                           ggam_k, valid_bsv, h, h_chunk_size)
                            # QUERY: gx_q = gs·norm(keys·γ_k)/√H, resident in gout32
                            # (gn32 keeps the live gx_k for the gradx stage below).
                            gx_q = gout32_grp.current()
                            pl.set_validshape(gx_q, [valid_bsv, h])
                            vf_rmsnorm_fwd(keys32, gamma32_k, rms, gx_q,
                                           valid_bsv, h, h_chunk_size)
                            vf_scaled_dot(gs, gx_q, gx_q, valid_bsv, h,
                                          h_chunk_size, h)
                            vf_rmsbw_rmean(gx_q, hidden32, gamma32, partner_sq,
                                           rmean_q, ggam_q, valid_bsv, h,
                                           h_chunk_size)
                        vf_rmsbw_rmean_finalize(rmean_k, valid_bsv, h)
                        vf_rmsbw_rmean_finalize(rmean_q, valid_bsv, h)

                        # ══ gradx stage — ZERO GM loads: keys32/hidden32/gx_k/gx_q
                        # resident from rmean stage.  BOTH rmsbw_gradx calls run
                        # FIRST (hidden32's last read is the QUERY one), THEN both
                        # BF16 casts go through the single gbuf16 temp (overlays
                        # xcache — dead by then).  ptnr16 stays a PURE load buffer
                        # so the next bs_start's loads prefetch during compute. ══
                        for h_chunk in pl.range(0, n_h_chunks):
                            h_off = h_chunk * h
                            # Re-acquire the resident tiles in THIS loop (same
                            # single-address groups — current() returns the same
                            # tile; the codegen scopes mutex vars per acquisition
                            # point, matching the old passes' pattern).
                            keys32 = ptnr32_grp.current()
                            pl.set_validshape(keys32, [valid_bsv, h])
                            hidden32 = xcache_grp.current()
                            pl.set_validshape(hidden32, [valid_bsv, h])
                            gx_k = gn32_grp.current()
                            pl.set_validshape(gx_k, [valid_bsv, h])
                            gx_q = gout32_grp.current()
                            pl.set_validshape(gx_q, [valid_bsv, h])
                            # KEY: grad_key = rmsbw_gradx(gx_k, keys) → in-place over gx_k
                            gamma32_k = gamma32_k_grp.current()  # key_gamma
                            vf_rmsbw_gradx(gx_k, keys32, gamma32_k, rms, rmean_k,
                                           gx_k, valid_bsv, h, h_chunk_size)
                            # QUERY: grad_hidden = rmsbw_gradx(gx_q, hidden) → in-place over gx_q
                            # (hidden32/xcache dies here — gbuf16 casts may follow)
                            gamma32 = gamma32_grp.current()  # query_gamma
                            vf_rmsbw_gradx(gx_q, hidden32, gamma32, partner_sq,
                                           rmean_q, gx_q, valid_bsv, h, h_chunk_size)
                            gh16 = gbuf16_grp.current()
                            pl.set_validshape(gh16, [valid_bsv, h])
                            pl.cast(gh16, gx_q, mode=pl.RoundMode.CAST_ROUND)
                            pl.store(grad_hidden_states, gh16,
                                     [bs_off, m_head, h_off], order=[0, 2])
                            gk16 = gbuf16_grp.current()
                            pl.set_validshape(gk16, [valid_bsv, h])
                            pl.cast(gk16, gx_k, mode=pl.RoundMode.CAST_ROUND)
                            pl.store(grad_key_ws, gk16,
                                     [bs_off, m_head, h_off], order=[0, 2])
                    # ── RMW gamma accs ONCE per (tile, m_head).  BOTH chains stage
                    # through ggamq_acc_grp (independent address). ──
                    gslot = worker_id * GAMMA_SLOT_SPLIT + (tile_idx % GAMMA_SLOT_SPLIT)
                    gamma_acc = ggamq_acc_grp.current()
                    pl.set_validshape(gamma_acc, [1, h])
                    for h_chunk in pl.range(0, n_h_chunks):
                        h_off = h_chunk * h
                        pl.load(gamma_acc, grad_qgamma_acc,
                                [gslot, m_head, h_off], order=[1, 2])
                        pl.add(gamma_acc, gamma_acc, ggam_q)
                        pl.store(grad_qgamma_acc, gamma_acc,
                                 [gslot, m_head, h_off], order=[1, 2])
                    for h_chunk in pl.range(0, n_h_chunks):
                        h_off = h_chunk * h
                        pl.load(gamma_acc, grad_kgamma_acc,
                                [gslot, m_head, h_off], order=[1, 2])
                        pl.add(gamma_acc, gamma_acc, ggam_k)
                        pl.store(grad_kgamma_acc, gamma_acc,
                                 [gslot, m_head, h_off], order=[1, 2])
            # ===== Per-head GK_DONE: ALL workers finished this head's
            # grad_key_ws (and grad_hidden_states) rows -> Cube may consume
            # head m_head's Nest1-key + Nest3 while vector produces the
            # remaining heads. Event id 排在 GV 分组事件之后。 =====
            pl.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)
            if m_head >= 2:
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=EVENT_GK_ACK_BASE + (m_head % 2),
                    sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE3,
                event_id=EVENT_GK_BASE + (m_head % 2),
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

    # ═══════════════════════════════════════════════════════════════
    # CUBE SECTION (consumes grad_value_ws / grad_key_ws and transposed inputs)
    # Cube tile: BF16 [128,128,128] (A/B/L0A/L0B BF16, L0C accumulator FP32).
    tt_a_l1 = pl.TileType(shape=[cube_mn, cube_k], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                            valid_shape=[-1, -1], compact=1)
    tt_b_l1 = pl.TileType(shape=[cube_k, cube_mn], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                            valid_shape=[-1, -1], compact=1)
    tt_a_l0 = pl.TileType(shape=[cube_mn, cube_k], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Left, layout=pl.NZ,
                            valid_shape=[-1, -1], compact=1)
    tt_b_l0 = pl.TileType(shape=[cube_k, cube_mn], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                            valid_shape=[-1, -1], compact=1)
    tt_acc = pl.TileType(shape=[cube_mn, cube_mn], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Acc, fractal=1024,
                            layout=pl.NZ, valid_shape=[-1, -1], compact=1)

    # L1/L0 double-buffered (overlap GM->L1 load + L1->L0 move with matmul);
    # acc single-buffered (matmul_acc K-chain is a true dependency).
    a_l1_grp = pl.make_tile_group(type=tt_a_l1,
                                  addrs=[LA_L1_A_BASE + 0 * LA_L1_SLOT_STRIDE,
                                         LA_L1_A_BASE + 1 * LA_L1_SLOT_STRIDE,
                                         LA_L1_A_BASE + 2 * LA_L1_SLOT_STRIDE,
                                         LA_L1_A_BASE + 3 * LA_L1_SLOT_STRIDE],
                                  mutex_ids=[10, 11, 27, 28])
    b_l1_grp = pl.make_tile_group(type=tt_b_l1,
                                  addrs=[LA_L1_B_BASE + 0 * LA_L1_SLOT_STRIDE,
                                         LA_L1_B_BASE + 1 * LA_L1_SLOT_STRIDE,
                                         LA_L1_B_BASE + 2 * LA_L1_SLOT_STRIDE,
                                         LA_L1_B_BASE + 3 * LA_L1_SLOT_STRIDE],
                                  mutex_ids=[12, 13, 29, 30])
    a_l0_grp = pl.make_tile_group(type=tt_a_l0, addrs=[0x0000, 0x8000], mutex_ids=[14, 15])
    b_l0_grp = pl.make_tile_group(type=tt_b_l0, addrs=[0x0000, 0x8000], mutex_ids=[16, 17])
    acc_grp = pl.make_tile_group(type=tt_acc, addrs=[0x00000, 0x10000, 0x20000, 0x30000],
                                 mutex_ids=[18, 21, 22, 23])

    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)

        for tile_idx in pl.range(0, iters_per_core):
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE2,
                event_id=EVENT_GV_WAVE_BASE + (tile_idx % 2),
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            pl.system.set_cross_core(
                pipe=pl.PipeType.FIX,
                event_id=EVENT_GV_ACK_BASE + (tile_idx % 2),
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            bsi = core_id + tile_idx * num_cores
            if bsi < n_bs:
                for ni in pl.range(0, n_de):
                    row_off = bsi * cube_mn
                    valid_bs = pl.min(cube_mn, bs - row_off)
                    col_off = ni * cube_mn
                    valid_n = pl.min(cube_mn, de - col_off)
                    ac = acc_grp.next()
                    pl.set_validshape(ac, [valid_bs, valid_n])
                    for k_idx in pl.range(0, n_kh):
                        k_off = k_idx * cube_k
                        valid_k = pl.min(cube_k, h - k_off)
                        left = a_l1_grp.next()
                        right = b_l1_grp.next()
                        pl.set_validshape(left, [valid_bs, valid_k])
                        pl.set_validshape(right, [valid_k, valid_n])
                        pl.load(left, grad_value_ws, [row_off, k_off])
                        pl.load(right, wv_t, [k_off, col_off])
                        al = a_l0_grp.next()
                        br = b_l0_grp.next()
                        pl.set_validshape(al, [valid_bs, valid_k])
                        pl.set_validshape(br, [valid_k, valid_n])
                        pl.move(al, left)
                        pl.move(br, right)
                        if k_idx == 0:
                            pl.matmul(ac, al, br)
                        else:
                            pl.matmul_acc(ac, ac, al, br)
                    # Value contribution: atomic-add into grad_emb_acc (FP32 sum with
                    # per-head key stores below).
                    pl.store(grad_emb_acc, ac, [row_off, col_off],
                             atomic=pl.AtomicType.AtomicAdd)

        # Nest 2: grad_W_v[De, H] = emb_t @ grad_value_ws. Only depends on
        # grad_value_ws -> runs before wait(GK_DONE), overlapping Vector Phase B.
        # NO wave waits here: Nest1's rounds already waited every wave in
        # order (iters_per_core == n_waves), so all grad_value_ws rows this
        # K-loop touches are long since ready.
        total_wv = n_de * n_h
        iters_wv = (total_wv + num_cores - 1) // num_cores
        for wv_idx in pl.range(0, iters_wv):
            flat = core_id + wv_idx * num_cores
            if flat < total_wv:
                de_idx = flat // n_h
                h_idx = flat % n_h
                de_off = de_idx * cube_mn
                h_off = h_idx * cube_mn
                valid_de = pl.min(cube_mn, de - de_off)
                valid_h = pl.min(cube_mn, h - h_off)
                ac = acc_grp.next()
                pl.set_validshape(ac, [valid_de, valid_h])
                for k_idx in pl.range(0, n_kbs):
                    k_off = k_idx * cube_k
                    valid_k = pl.min(cube_k, bs - k_off)
                    left = a_l1_grp.next()
                    right = b_l1_grp.next()
                    pl.set_validshape(left, [valid_de, valid_k])
                    pl.set_validshape(right, [valid_k, valid_h])
                    pl.load(left, emb_t, [de_off, k_off])
                    pl.load(right, grad_value_ws, [k_off, h_off])
                    al = a_l0_grp.next()
                    br = b_l0_grp.next()
                    pl.set_validshape(al, [valid_de, valid_k])
                    pl.set_validshape(br, [valid_k, valid_h])
                    pl.move(al, left)
                    pl.move(br, right)
                    if k_idx == 0:
                        pl.matmul(ac, al, br)
                    else:
                        pl.matmul_acc(ac, ac, al, br)
                pl.store(grad_value_proj_weights, ac, [de_off, h_off])

        # Nest 1 (per-head key) + Nest 3 consumed PER HEAD: each head's GK
        for m_head in pl.range(0, m_h):
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE2,
                                      event_id=EVENT_GK_BASE + (m_head % 2),
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            pl.system.set_cross_core(
                pipe=pl.PipeType.FIX,
                event_id=EVENT_GK_ACK_BASE + (m_head % 2),
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # Nest 1 (per-head key): grad_key_ws[m] @ wk_t[m] -> atomic-add into the
            # same grad_emb_acc slot as the value store above.
            for tile_idx in pl.range(0, iters_per_core):
                bsi = core_id + tile_idx * num_cores
                for n_idx in pl.range(0, n_de):
                    if bsi < n_bs:
                        row_off = bsi * cube_mn
                        valid_bs = pl.min(cube_mn, bs - row_off)
                        col_off = n_idx * cube_mn
                        valid_n = pl.min(cube_mn, de - col_off)
                        ac = acc_grp.next()
                        pl.set_validshape(ac, [valid_bs, valid_n])
                        for k_idx in pl.range(0, n_kh):
                            k_off = k_idx * cube_k
                            valid_k = pl.min(cube_k, h - k_off)
                            left = a_l1_grp.next()
                            right = b_l1_grp.next()
                            pl.set_validshape(left, [valid_bs, valid_k])
                            pl.set_validshape(right, [valid_k, valid_n])
                            pl.load(left, grad_key_ws,
                                    [row_off, m_head, k_off], order=[0, 2])
                            pl.load(right, wk_t,
                                    [m_head, k_off, col_off], order=[1, 2])
                            al = a_l0_grp.next()
                            br = b_l0_grp.next()
                            pl.set_validshape(al, [valid_bs, valid_k])
                            pl.set_validshape(br, [valid_k, valid_n])
                            pl.move(al, left)
                            pl.move(br, right)
                            if k_idx == 0:
                                pl.matmul(ac, al, br)
                            else:
                                pl.matmul_acc(ac, ac, al, br)
                        # Per-head key contribution: atomic-add into the same grad_emb_acc slot.
                        pl.store(grad_emb_acc, ac, [row_off, col_off],
                                 atomic=pl.AtomicType.AtomicAdd)

            total_wk = n_de * n_h
            iters_wk = (total_wk + num_cores - 1) // num_cores
            for wk_idx in pl.range(0, iters_wk):
                flat = core_id + wk_idx * num_cores
                if flat < total_wk:
                    de_idx = flat // n_h
                    h_idx = flat % n_h
                    de_off = de_idx * cube_mn
                    h_off = h_idx * cube_mn
                    valid_de = pl.min(cube_mn, de - de_off)
                    valid_h = pl.min(cube_mn, h - h_off)
                    ac = acc_grp.next()
                    pl.set_validshape(ac, [valid_de, valid_h])
                    for k_idx in pl.range(0, n_kbs):
                        k_off = k_idx * cube_k
                        valid_k = pl.min(cube_k, bs - k_off)
                        left = a_l1_grp.next()
                        right = b_l1_grp.next()
                        pl.set_validshape(left, [valid_de, valid_k])
                        pl.set_validshape(right, [valid_k, valid_h])
                        pl.load(left, emb_t, [de_off, k_off])
                        pl.load(right, grad_key_ws,
                                [k_off, m_head, h_off], order=[0, 2])
                        al = a_l0_grp.next()
                        br = b_l0_grp.next()
                        pl.set_validshape(al, [valid_de, valid_k])
                        pl.set_validshape(br, [valid_k, valid_h])
                        pl.move(al, left)
                        pl.move(br, right)
                        if k_idx == 0:
                            pl.matmul(ac, al, br)
                        else:
                            pl.matmul_acc(ac, ac, al, br)
                    pl.store(grad_key_proj_weights, ac,
                             [m_head, de_off, h_off], order=[1, 2])

        pl.system.set_mm_layout_transform(enabled=False)
        # grad_emb_acc is final (Nest1 value + per-head key atomic-add). Host casts it -> BF16.


# ═══════════════════════════════════════════════════════════════════
# Layer E: Host wrapper
# ═══════════════════════════════════════════════════════════════════

def engram_backward_wrapper(
    grad_output,        # [B, S, M_H, H] BF16
    hidden_states,      # [B, S, M_H, H] BF16
    embeddings,         # [B, S, De]     BF16
    key_proj_weights,   # [M_H, De, H]   BF16  (host-pre-transposed below)
    value_proj_weights, # [De, H]        BF16  (host-pre-transposed below)
    key_gamma,          # [M_H, H]       BF16
    query_gamma,        # [M_H, H]       BF16
    scores,             # [B, S, M_H, 1] FP32
    gates,              # [B, S, M_H, 1] FP32
    keys,               # [B, S, M_H, H] BF16 cache; cast to FP32 in kernel
    value,              # [B, S, H]      BF16 cache; cast to FP32 in kernel
    clamp_value=1e-6,
    eps=1e-6,
):
    """Host wrapper. Mirrors forward's reshape + launch + reshape-back, and
    additionally:
      * host pre-transposes W_v, W_k, E (rule 3: no is_transpose / layout=ZN)
      * pads scores/gates scalar dim 1->64 (PyPTO tile load does not broadcast 1)
      * pre-zeros FP32 grad_γ GM (FP32 tile RMW targets)
      * pre-zeros FP32 grad_emb GM accumulator (cube atomicAdd target);
        grad_W_v / grad_W_k FP32 buffers are single-writer (full-K per core)
        so they need no pre-zeroing; host casts all three to BF16 after launch
      * keeps only scores/gates in FP32 cache form; other cache tensors stay BF16.
      * returns low-precision final gradients; gamma RMW and reduction use an
        FP32 per-core GM workspace, with one vector subblock as the writer.
    """
    if scores.dtype != torch.float32 or gates.dtype != torch.float32:
        raise TypeError(
            "engram_backward_wrapper expects scores and gates to be torch.float32"
        )
    b, s, m_dim, h = grad_output.shape
    bs = b * s
    de = embeddings.shape[-1]
    device = grad_output.device
    dtype = grad_output.dtype

    # ── TilingKey selection: H in {1280, 2560, 2048, 1536} have specialized kernels. ──
    hmode_map = {1280: 0, 2560: 1, 2048: 2, 1536: 3}
    if h not in hmode_map:
        raise ValueError(
            f"engram_backward_wrapper only supports H in {{1280, 2560, 2048, 1536}} "
            f"(TilingKey specialization), got H={h}"
        )
    hmode = hmode_map[h]
    # BSMode: largest V_TILE that fills all 64 vector workers
    # (need n_bs_v = ceil(BS/V_TILE) >= 64; small BS floors at V_TILE=16).
    if bs >= 128 * 64:    # 8192
        v_tile, bsmode = V_TILE_BS0, 0
    elif bs >= 64 * 64:   # 4096
        v_tile, bsmode = V_TILE_BS1, 1
    elif bs >= 32 * 64:   # 2048
        v_tile, bsmode = V_TILE_BS2, 2
    else:
        v_tile, bsmode = V_TILE_BS3, 3
    tiling_key = {"HMode": hmode, "BSMode": bsmode}

    # ── Host reshape: merge B·S -> BS ──
    go_bs = grad_output.reshape(bs, m_dim, h).contiguous()
    hid_bs = hidden_states.reshape(bs, m_dim, h).contiguous()
    emb_bs = embeddings.reshape(bs, de).contiguous()
    keys_bs = keys.reshape(bs, m_dim, h).contiguous()
    val_bs = value.reshape(bs, h).contiguous()
    # Pad scores/gates 1->64 (forward stores score_back/gate_back as [.,.,64]).
    sc_bs = scores.reshape(bs, m_dim, 1).expand(-1, -1, 64).contiguous()
    gt_bs = gates.reshape(bs, m_dim, 1).expand(-1, -1, 64).contiguous()

    # ── Host pre-transpose (rule 3): kernel uses plain NZ loads.  Weights stay
    #    BF16 (cube is BF16) -- no .float() upcast. ──
    emb_t = emb_bs.t().contiguous()                                      # [De, BS] BF16
    wv_t = value_proj_weights.t().contiguous()                          # [H, De] BF16
    wk_t = key_proj_weights.transpose(-1, -2).contiguous()              # [M_H, H, De] BF16

    # ── BF16 cube-operand workspaces (vector produces FP32, casts to BF16 on store) ──
    grad_value_ws = torch.empty((bs, h), dtype=torch.bfloat16, device=device)
    grad_key_ws = torch.empty((bs, m_dim, h), dtype=torch.bfloat16, device=device)

    num_cores = get_platform_info().core_num

    # FP32 per-slot gamma workspace. Each of the 64 workers owns GAMMA_SLOT_SPLIT
    # slots (round-robin on tile_idx) to shorten sequential RMW depth; host sum(dim=0)
    # over all 64*SPLIT slots is a shallow pairwise reduce (approaches torch.sum topo).
    acc_shape = (num_cores * 2 * GAMMA_SLOT_SPLIT, m_dim, h)
    grad_qgamma_acc = torch.zeros(acc_shape, dtype=torch.float32,
                                  device=device)  # B-query split
    grad_kgamma_acc = torch.zeros(acc_shape, dtype=torch.float32,
                                  device=device)  # B-key split

    # FP32 grad_emb accumulator [BS, De]: Nest1 value + per-head key all atomic-add
    # here. MUST be pre-zeroed (atomicAdd accumulates onto the existing value).
    grad_emb_acc = torch.zeros((bs, de), dtype=torch.float32, device=device)

    # ── grad_score cache (B-key writes, B-query reads; avoids recomputing Step7b+Step5) ──
    gscore_ws = torch.empty((bs, m_dim, 64), dtype=torch.float32, device=device)

    # ── Low-precision final outputs; all source calculations stay FP32 ──
    # (grad_embeddings is now produced by host from grad_emb_acc, see below.)
    grad_hidden_states = torch.empty((bs, m_dim, h), dtype=dtype, device=device)
    grad_key_proj_weights = torch.empty((m_dim, de, h), dtype=dtype, device=device)
    grad_value_proj_weights = torch.empty((de, h), dtype=dtype, device=device)

    engram_backward_kernel[None, num_cores, tiling_key](
        go_bs, hid_bs, emb_bs,
        key_gamma, query_gamma,
        sc_bs, gt_bs, keys_bs, val_bs,
        emb_t, wv_t, wk_t,
        grad_value_ws, grad_key_ws, gscore_ws,
        grad_qgamma_acc, grad_kgamma_acc, grad_emb_acc,
        grad_hidden_states, grad_key_proj_weights, grad_value_proj_weights,
        clamp_value, eps,
    )
    torch.npu.synchronize()

    # Cast FP32 grad_emb_acc -> BF16 grad_embeddings on host.
    grad_embeddings = grad_emb_acc.to(dtype)
    grad_kgamma_out = grad_kgamma_acc.sum(dim=0).to(dtype)
    grad_qgamma_out = grad_qgamma_acc.sum(dim=0).to(dtype)

    grad_hidden_states = grad_hidden_states.reshape(b, s, m_dim, h)
    grad_embeddings = grad_embeddings.reshape(b, s, de)
    grad_key_proj_weights = grad_key_proj_weights.reshape(m_dim, de, h)
    return (
        grad_hidden_states,
        grad_embeddings,
        grad_key_proj_weights,
        grad_value_proj_weights,
        grad_kgamma_out,
        grad_qgamma_out,
    )
