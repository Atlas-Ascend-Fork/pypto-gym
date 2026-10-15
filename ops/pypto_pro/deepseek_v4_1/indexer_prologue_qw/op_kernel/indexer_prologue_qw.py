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

"""indexer_prologue_qw：手写 PyPTO-Pro（``@pl.jit``）融合算子。

算子目的
--------
一个 MIX kernel 完成 SPEC 的全部计算，包含两条 GEMM 路径与一个向量收尾：

* **Q 路径**（MXFP8 → RoPE → MXFP4）：``qr @ wqb^T``，E4M3 字节 + 每 32 个 K
  一个 E8M0 scale，FP32 累加；随后只对末尾 ``Dr=64`` 列做 rotate-half RoPE；
  最后按每 32 列一组做 MXFP4 量化（E2M1 最近邻、.5 处远离零取整；E8M0
  向上取整），每字节打包两个 nibble（偶数列在低 4 bit）。
* **W 路径**（BF16 × scale）：``x @ ww^T``（K=``dim``=5120，FP32 累加）再乘
  ``softmax_scale``。

冻结几何：``dim=5120, q_lora=1280, N=32, D=128, Dr=64``，``T`` 是唯一动态轴，
沿 ``T`` 轴按 ``BASE_M=128`` 行切成 tile。

split-K / split-T 选择
----------------------
* ``n_tiles >= SPLIT_T_TILES(=32)`` 走 **split-T**：一个核独占一个 T tile 的
  全部 32 个 head，W 的 K 归约（``NWK=40`` 片）交错进 Q head 循环完成。
* 否则走 **split-K**：head 轴切成 ``HB_K=16`` 个 2-head block；当
  ``n_tiles <= SPLIT_W_TILES(=12)`` 时，W 的 K 轴也按同一 head block 切分，
  各 block 把 K 分片写入 workspace，向量单元在 grid barrier 之后求和。

尾部处理（``T`` 非 ``BASE_M`` 整数倍）
-------------------------------------
尾块在 kernel 内处理：Q 累加器始终保持 ``[BASE_M, D]`` 的物理窗口（不做运行期
收缩），L1 load 用 ``set_validshape`` 夹到有效行数，向量收尾按 AIV 半边夹紧
store；W 累加器使用运行期有效 M 并直接裁剪 store。

建议阅读顺序
------------
公开入口 ``indexer_prologue_qw`` → 主 kernel ``indexer_prologue_qw_kernel``
→ 两个 VF helper ``ipqw_build_lut_vf`` / ``ipqw_epilogue_vf``。kernel 内部先是
``with pl.section_cube()``（两条 GEMM 路径与 CV 握手），再是
``with pl.section_vector()``（RoPE/量化收尾与 split-K W 归约）。

数据流大体是 GM → L1 → L0A/L0B → L0C → UB：Q 在 UB 内完成 RoPE 与量化后写回
GM；W 要么直接写结果，要么先把分片写入 workspace 等待后续归约。
"""

from __future__ import annotations

import os

import torch

import torch_npu  # noqa: F401

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

# ---------------------------------------------------------------------------
# Frozen geometry / tiling constants
# ---------------------------------------------------------------------------
BASE_M = 128          # T-tile height (GEMM M)
D = 128               # head dim
DR = 64               # RoPE length (rotate-half halves: 32)
N_HEADS = 32
DIM = 5120            # W path K
QLORA = 1280          # Q path K
VL = 64               # fp32 lanes per register
MX_GROUP = 32
GROUPS = D // MX_GROUP                       # 4
ROWS_VEC = BASE_M // 2                       # 64 rows per AIV
ENC_REGS = (ROWS_VEC * GROUPS + VL - 1) // VL  # 4
AMAX_LEN = ENC_REGS * VL                     # 256
K0 = 128                                     # Q L0 K step (base_k_q; DSL uses 128)
NK = QLORA // K0                             # 10
SK0 = K0 // MX_GROUP                         # 4 scale groups per Q L0 step
SQ_ALL = QLORA // MX_GROUP                   # 40 scale groups over the whole q_lora row
WK0 = 128                                    # W L0 K step
NWK = DIM // WK0                             # 40
WSW = 160                                    # split-K W L1 K-slice window (divides 320)
WSUB = 80                                    # split-K W L0 sub-chunk (64 KB L0A budget)
ROWS_RED = 2                                 # split-K W band height (rows per vector reduce)
SPLIT_T_TILES = 32                           # n_tiles >= 32 -> split-T, else split-K
SPLIT_W_TILES = 12                           # n_tiles <= this -> split-K also splits W's K
HB_K = 16                                    # split-K head blocks (2 heads each)
HPB_K = N_HEADS // HB_K                      # 2 heads per split-K block
T_MAX = 262144                                # 支持的最大 T（SPEC 动态轴范围）
FRACTAL_NZ = 29                               # torch_npu.Format.FRACTAL_NZ
SOFTMAX_SCALE = 0.08838834764831845
E8M0_BIAS = 127

# E2M1 round-to-nearest midpoints (ties round away from zero -> use >=).
E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# Gather table: bits(|q|)>>20 minus the group's E8M0 exponent offset gives a
# 3-bit-exact bucket index; the bucket's lower bound is exactly one of the
# midpoints, so a truncating index reproduces nearest/ties-away.
LUT_LEN = 1088
LUT_REGS = (LUT_LEN + VL - 1) // VL          # 17
LUT_SHIFT = 20


# ---------------------------------------------------------------------------
# Vector epilogue (raw VF): RoPE -> per-32 amax/E8M0 -> E2M1 pack
# ---------------------------------------------------------------------------
@pl.vector_function
def ipqw_build_lut_vf(lut_tile):
    """片上生成 E2M1 幅值查找表。

    职责：为 MXFP4 量化准备一张 E2M1 幅值桶表，使 host 侧无需传入任何私有
    输入。``lut_tile[k]`` 等于“不超过桶值 ``2^(k//8 - 127) * (1 + (k%8)/8)``
    的 E2M1 中点个数”；每个 E2M1 中点都恰好落在某个桶的下界上，因此收尾阶段
    用截断索引即可得到精确的最近邻（含 .5 处远离零取整）。

    参数：
        lut_tile (pl.TileType[1, LUT_LEN] INT32)：输出表，长度
            ``LUT_LEN=1088``，按 ``VL=64`` 个 lane 分 ``LUT_REGS=17`` 个寄存器
            块写入。

    数据流：没有外部输入；结果写入底层 UB（``lut_ub``），供
        ``ipqw_epilogue_vf`` 通过 ``vf.gather`` 查询。

    前置条件：调用前需 ``set_validshape(lut_ub, [1, LUT_LEN])`` 并分配好
        ``lut_tile``；每个核调用一次即可。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    lane = vf.arange(0, dtype=pl.DT_INT32)
    seven_i = vf.full(7, preg, dtype=pl.DT_INT32)
    zero_i = vf.full(0, preg, dtype=pl.DT_INT32)
    one_i = vf.full(1, preg, dtype=pl.DT_INT32)
    mid0 = vf.full(E2M1_MIDPOINTS[0], preg, dtype=pl.DT_FP32)
    mid1 = vf.full(E2M1_MIDPOINTS[1], preg, dtype=pl.DT_FP32)
    mid2 = vf.full(E2M1_MIDPOINTS[2], preg, dtype=pl.DT_FP32)
    mid3 = vf.full(E2M1_MIDPOINTS[3], preg, dtype=pl.DT_FP32)
    mid4 = vf.full(E2M1_MIDPOINTS[4], preg, dtype=pl.DT_FP32)
    mid5 = vf.full(E2M1_MIDPOINTS[5], preg, dtype=pl.DT_FP32)
    mid6 = vf.full(E2M1_MIDPOINTS[6], preg, dtype=pl.DT_FP32)
    for b in pl.range(LUT_REGS):
        idx = vf.adds(lane, b * VL, preg)
        exp_field = vf.shift_right(idx, 3, preg)
        mant3 = vf.and_(idx, seven_i, preg)
        eb = vf.shift_left(exp_field, 23, preg)
        mb = vf.shift_left(mant3, 20, preg)
        bits = vf.or_(eb, mb, preg)
        value = vf.bit_cast(bits, dtype=pl.DT_FP32)
        g0 = vf.ge(value, mid0, preg)
        g1 = vf.ge(value, mid1, preg)
        g2 = vf.ge(value, mid2, preg)
        g3 = vf.ge(value, mid3, preg)
        g4 = vf.ge(value, mid4, preg)
        g5 = vf.ge(value, mid5, preg)
        g6 = vf.ge(value, mid6, preg)
        s0 = vf.select(one_i, zero_i, g0)
        s1 = vf.select(one_i, zero_i, g1)
        s2 = vf.select(one_i, zero_i, g2)
        s3 = vf.select(one_i, zero_i, g3)
        s4 = vf.select(one_i, zero_i, g4)
        s5 = vf.select(one_i, zero_i, g5)
        s6 = vf.select(one_i, zero_i, g6)
        code = vf.add(s0, s1, preg)
        code = vf.add(code, s2, preg)
        code = vf.add(code, s3, preg)
        code = vf.add(code, s4, preg)
        code = vf.add(code, s5, preg)
        code = vf.add(code, s6, preg)
        vf.store_align(lut_tile, code, preg, b * VL)


@pl.vector_function
def ipqw_epilogue_vf(cv_tile, q_tile, dsc_all_tile, off_tile, amax_tile,
                     cos_tile, sin_tile, lut_tile, rows: pl.DT_INT64, col: pl.DT_INT64,
                     col_shift: pl.DT_INT64):
    """对 ``cv_tile`` 的末尾 ``DR`` 个 lane 做 RoPE，再量化 ``rows`` 行 MXFP4。

    职责：这是 Q 路径的核心向量收尾，分三趟完成——
    (1) 对每行末尾 ``DR=64`` 列做 rotate-half RoPE，并求每 32 列一组的
    ``|amax|``；(2) 由 amax 计算 E8M0 ``ceil(amax/6)`` 指数，按每 4 个 32 组
    打包成一个 32-bit descale word；(3) 查 ``lut_tile`` 得到 E2M1 幅值码
    （最近邻、.5 处远离零），补上符号位后每字节打包两个 nibble。

    参数：
        cv_tile (FP32[ROWS_VEC, D])：CV 握手送来的 Q 累加器窗口（就地被 RoPE
            改写）。
        q_tile (UINT8[ROWS_VEC, D//2])：打包后的 E2M1 输出。
        dsc_all_tile / dsc2 (INT32[ROWS_VEC, *])：打包 descale word 的落点。
        off_tile (INT32[1, AMAX_LEN])：中间量，每组的 E8M0 查找偏移。
        amax_tile (FP32[1, AMAX_LEN])：中间量，每组的 ``|amax|``。
        cos_tile / sin_tile (FP32[ROWS_VEC, DR])：该行的 RoPE cos/sin。
        lut_tile (INT32[1, LUT_LEN])：``ipqw_build_lut_vf`` 生成的幅值表。
        rows (INT64)：本 AIV 需要处理的活行数。
        col (INT64) / col_shift (INT64)：把本 head 的打包 descale word 放到
            ``dsc_all_tile`` 的目标列（见下）。

    数据流：``cv_tile`` 由 Cube 侧的 ``DualModeSplitM`` acc→vec 搬运写入；
    结果 ``q_tile`` 与 descale 由调用方写回 GM（``q`` / ``descale_q``）。
    ``col``/``col_shift`` 决定 descale 的在 tile 内寻址：split-T 写绝对 head
    列（stride 32，shift 5），split-K 写 2 宽 per-work tile 内的相对 head 列
    （stride 2，shift 1）。

    前置条件：``q_tile``/``dsc`` 的 valid shape 已按活行裁剪；``lut_tile`` 已
        由 ``ipqw_build_lut_vf`` 填好；``cv_tile`` 的对应 CV 事件已
        ``wait_cross_core``。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    m_lo = vf.create_mask(pattern=pl.MaskPattern.H, dtype=pl.DT_FP32)

    lane = vf.arange(0, dtype=pl.DT_INT32)
    lane_wrap = vf.full(VL - 1, preg, dtype=pl.DT_INT32)
    lane_shift = vf.adds(lane, VL // 2, preg)
    rot_idx_i = vf.and_(lane_shift, lane_wrap, preg)
    rot_idx = vf.bit_cast(rot_idx_i, dtype=pl.DT_UINT32)

    one_f = vf.full(1.0, preg, dtype=pl.DT_FP32)
    neg_one_f = vf.full(-1.0, preg, dtype=pl.DT_FP32)
    hi_half = vf.ge(lane, DR // 2, preg)
    rope_sgn = vf.select(one_f, neg_one_f, hi_half)

    zero_i = vf.full(0, preg, dtype=pl.DT_INT32)
    one_i = vf.full(1, preg, dtype=pl.DT_INT32)
    k_bias = vf.full(E8M0_BIAS, preg, dtype=pl.DT_INT32)
    mant_mask = vf.full(0x7FFFFF, preg, dtype=pl.DT_INT32)
    sign8 = vf.full(8, preg, dtype=pl.DT_INT32)
    sixteen_f = vf.full(16.0, preg, dtype=pl.DT_FP32)

    # --- pass 1: RoPE on the tail DR lanes + per-32 absolute max ---
    for r in pl.range(rows):
        base = r * D
        x0 = vf.load_align(cv_tile, base)
        x1 = vf.load_align(cv_tile, base + (D - DR))
        cosv = vf.load_align(cos_tile, r * DR)
        sinv = vf.load_align(sin_tile, r * DR)
        rot_raw = vf.gather(x1, rot_idx)
        rot = vf.mul(rot_raw, rope_sgn, preg)
        rot_sin = vf.mul(rot, sinv, preg)
        x1_cos = vf.mul(x1, cosv, preg)
        x1 = vf.add(rot_sin, x1_cos, preg)
        vf.store_align(cv_tile, x1, preg, base + (D - DR))

        a0 = vf.abs(x0, preg)
        a1 = vf.abs(x1, preg)
        a0h = vf.gather(a0, rot_idx)
        a1h = vf.gather(a1, rot_idx)
        m0 = vf.reduce_max(a0, m_lo)
        m1 = vf.reduce_max(a0h, m_lo)
        m2 = vf.reduce_max(a1, m_lo)
        m3 = vf.reduce_max(a1h, m_lo)
        vf.store_align(amax_tile, m0, preg, r + 0 * ROWS_VEC, dist=pl.StoreDist.FIRST_ELEMENT)
        vf.store_align(amax_tile, m1, preg, r + 1 * ROWS_VEC, dist=pl.StoreDist.FIRST_ELEMENT)
        vf.store_align(amax_tile, m2, preg, r + 2 * ROWS_VEC, dist=pl.StoreDist.FIRST_ELEMENT)
        vf.store_align(amax_tile, m3, preg, r + 3 * ROWS_VEC, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)

    # --- pass 2: E8M0 ceil(amax/6) exponent; one register per D-group (64 rows) ---
    ld_a = vf.load_align(amax_tile, 0 * VL)
    ab = vf.bit_cast(ld_a, dtype=pl.DT_INT32)
    exp_bits = vf.shift_right(ab, 23, preg)
    mant = vf.and_(ab, mant_mask, preg)
    mant_hi = vf.gt(mant, 0x400000, preg)
    bump = vf.select(one_i, zero_i, mant_hi)
    exp_m2 = vf.adds(exp_bits, -2, preg)
    e8_0 = vf.add(exp_m2, bump, preg)
    e8_0 = vf.maxs(e8_0, 1, preg)
    nonzero = vf.ne(ab, zero_i, preg)
    e8_0 = vf.select(e8_0, k_bias, nonzero)
    off_0 = vf.shift_left(e8_0, 3, preg)
    off_0 = vf.adds(off_0, -1016, preg)
    vf.store_align(off_tile, off_0, preg, 0 * VL)

    ld_a = vf.load_align(amax_tile, 1 * VL)
    ab = vf.bit_cast(ld_a, dtype=pl.DT_INT32)
    exp_bits = vf.shift_right(ab, 23, preg)
    mant = vf.and_(ab, mant_mask, preg)
    mant_hi = vf.gt(mant, 0x400000, preg)
    bump = vf.select(one_i, zero_i, mant_hi)
    exp_m2 = vf.adds(exp_bits, -2, preg)
    e8_1 = vf.add(exp_m2, bump, preg)
    e8_1 = vf.maxs(e8_1, 1, preg)
    nonzero = vf.ne(ab, zero_i, preg)
    e8_1 = vf.select(e8_1, k_bias, nonzero)
    off_1 = vf.shift_left(e8_1, 3, preg)
    off_1 = vf.adds(off_1, -1016, preg)
    vf.store_align(off_tile, off_1, preg, 1 * VL)

    ld_a = vf.load_align(amax_tile, 2 * VL)
    ab = vf.bit_cast(ld_a, dtype=pl.DT_INT32)
    exp_bits = vf.shift_right(ab, 23, preg)
    mant = vf.and_(ab, mant_mask, preg)
    mant_hi = vf.gt(mant, 0x400000, preg)
    bump = vf.select(one_i, zero_i, mant_hi)
    exp_m2 = vf.adds(exp_bits, -2, preg)
    e8_2 = vf.add(exp_m2, bump, preg)
    e8_2 = vf.maxs(e8_2, 1, preg)
    nonzero = vf.ne(ab, zero_i, preg)
    e8_2 = vf.select(e8_2, k_bias, nonzero)
    off_2 = vf.shift_left(e8_2, 3, preg)
    off_2 = vf.adds(off_2, -1016, preg)
    vf.store_align(off_tile, off_2, preg, 2 * VL)

    ld_a = vf.load_align(amax_tile, 3 * VL)
    ab = vf.bit_cast(ld_a, dtype=pl.DT_INT32)
    exp_bits = vf.shift_right(ab, 23, preg)
    mant = vf.and_(ab, mant_mask, preg)
    mant_hi = vf.gt(mant, 0x400000, preg)
    bump = vf.select(one_i, zero_i, mant_hi)
    exp_m2 = vf.adds(exp_bits, -2, preg)
    e8_3 = vf.add(exp_m2, bump, preg)
    e8_3 = vf.maxs(e8_3, 1, preg)
    nonzero = vf.ne(ab, zero_i, preg)
    e8_3 = vf.select(e8_3, k_bias, nonzero)
    off_3 = vf.shift_left(e8_3, 3, preg)
    off_3 = vf.adds(off_3, -1016, preg)
    vf.store_align(off_tile, off_3, preg, 3 * VL)

    w1 = vf.shift_left(e8_1, 8, preg)
    w2 = vf.shift_left(e8_2, 16, preg)
    w3 = vf.shift_left(e8_3, 24, preg)
    or01 = vf.or_(e8_0, w1, preg)
    or23 = vf.or_(w2, w3, preg)
    word = vf.or_(or01, or23, preg)
    prow = vf.shift_left(lane, col_shift, preg)
    idx_i = vf.adds(prow, col, preg)
    idx_u = vf.bit_cast(idx_i, dtype=pl.DT_UINT32)
    vf.scatter(dsc_all_tile, word, idx_u, preg)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)

    # --- pass 3: E2M1 nearest code (ties away from zero) + two nibbles/byte ---
    lane1 = vf.and_(lane, one_i, preg)
    odd = vf.ne(lane1, zero_i, preg)
    pack_w = vf.select(sixteen_f, one_f, odd)
    gsel = vf.shift_right(lane, 5, preg)
    gsel64 = vf.shift_left(gsel, 6, preg)
    mag_mask = vf.full(0x7FFFFFFF, preg, dtype=pl.DT_INT32)
    for r in pl.range(rows):
        base = r * D
        for i in pl.range(2):
            idx_i = vf.adds(gsel64, r + 128 * i, preg)
            idx_u = vf.bit_cast(idx_i, dtype=pl.DT_UINT32)
            off = vf.gather(off_tile, idx_u, preg)
            xv = vf.load_align(cv_tile, base + i * VL)
            xv_i = vf.bit_cast(xv, dtype=pl.DT_INT32)
            mag = vf.and_(xv_i, mag_mask, preg)
            sh = vf.shift_right(mag, LUT_SHIFT, preg)
            li = vf.sub(sh, off, preg)
            li = vf.maxs(li, 0, preg)
            li_u = vf.bit_cast(li, dtype=pl.DT_UINT32)
            code = vf.gather(lut_tile, li_u, preg)

            neg = vf.lt(xv, 0.0, preg)
            sign = vf.select(sign8, zero_i, neg)
            nib = vf.or_(code, sign, preg)
            nibf = vf.astype(nib, preg, dtype=pl.DT_FP32)
            nibw = vf.mul(nibf, pack_w, preg)
            pair = vf.pair_reduce_sum(nibw, preg)
            packed = vf.astype(pair, preg, dtype=pl.DT_INT32)
            vf.store_align(q_tile, packed, m_lo, base // 2 + i * (VL // 2),
                           dist=pl.StoreDist.PACK4)


@pl.vector_function
def ipqw_wred_reduce_vf(b0, b1, b2, b3, b4, b5, b6, b7,
                        b8, b9, b10, b11, b12, b13, b14, b15, out_tile):
    """把 split-K W 的 ``HB_K=16`` 个 K 分片在寄存器里逐元素求和。

    职责：调用方已把同一 band 的全部 ``HB_K`` 个 K 分片 staging 进各自独立的 UB 槽，
    这里用整寄存器 ``vf.load_align`` + ``vf.add`` 把它们相加，再 ``vf.store_align``
    写回目的 UB 槽。

    参数：
        b0..b15 (FP32[ROWS_RED, N_HEADS])：16 个 K 分片 UB Tile（每个恰为
            ``ROWS_RED*N_HEADS=64`` 个 fp32，正好一个 ``VL=64`` 寄存器）。
        out_tile (FP32[ROWS_RED, N_HEADS])：求和落点 UB Tile。

    数据流：每个分片从 UB 载入寄存器后立即参与累加，只保留 ``acc`` 与当前分片
        两个活寄存器；最后整寄存器写回。

    前置条件：16 个分片都已 ``pl.load`` 且 ``set_validshape``；``out_tile`` 的
        valid shape 已按活行裁剪（越界/非活行由调用方的 ``pl.store`` 丢弃）。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    r0 = vf.load_align(b0, 0)
    r1 = vf.load_align(b1, 0)
    acc = vf.add(r0, r1, preg)
    r = vf.load_align(b2, 0)
    acc = vf.add(acc, r, preg)
    r = vf.load_align(b3, 0)
    acc = vf.add(acc, r, preg)
    r = vf.load_align(b4, 0)
    acc = vf.add(acc, r, preg)
    r = vf.load_align(b5, 0)
    acc = vf.add(acc, r, preg)
    r = vf.load_align(b6, 0)
    acc = vf.add(acc, r, preg)
    r = vf.load_align(b7, 0)
    acc = vf.add(acc, r, preg)
    r = vf.load_align(b8, 0)
    acc = vf.add(acc, r, preg)
    r = vf.load_align(b9, 0)
    acc = vf.add(acc, r, preg)
    r = vf.load_align(b10, 0)
    acc = vf.add(acc, r, preg)
    r = vf.load_align(b11, 0)
    acc = vf.add(acc, r, preg)
    r = vf.load_align(b12, 0)
    acc = vf.add(acc, r, preg)
    r = vf.load_align(b13, 0)
    acc = vf.add(acc, r, preg)
    r = vf.load_align(b14, 0)
    acc = vf.add(acc, r, preg)
    r = vf.load_align(b15, 0)
    acc = vf.add(acc, r, preg)
    vf.store_align(out_tile, acc, preg)


# ---------------------------------------------------------------------------
# Fused kernel
# ---------------------------------------------------------------------------
@pl.jit(auto_mutex=True)
def indexer_prologue_qw_kernel(
    x: pl.Tensor[[pl.DYNAMIC, DIM], pl.DT_BF16],
    qr: pl.Tensor[[pl.DYNAMIC, QLORA], pl.DT_FP8E4M3FN],
    wqb: pl.Tensor[[4096, QLORA], pl.DT_FP8E4M3FN, pl.NZ],
    ww: pl.Tensor[[N_HEADS, DIM], pl.DT_BF16, pl.NZ],
    descale_qr: pl.Tensor[[pl.DYNAMIC, 20, 2], pl.DT_FP8E8M0],
    descale_wqb: pl.Tensor[[4096, 20, 2], pl.DT_FP8E8M0],
    rope_sin: pl.Tensor[[pl.DYNAMIC, DR], pl.DT_FP32],
    rope_cos: pl.Tensor[[pl.DYNAMIC, DR], pl.DT_FP32],
    q: pl.Tensor[[pl.DYNAMIC, N_HEADS, D // 2], pl.DT_UINT8],
    descale_q: pl.Tensor[[pl.DYNAMIC, N_HEADS, 2, 2], pl.DT_UINT8],
    w: pl.Tensor[[pl.DYNAMIC, N_HEADS], pl.DT_FP32],
    workspace: pl.Tensor[[pl.DYNAMIC, N_HEADS], pl.DT_FP32],
    softmax_scale: pl.DT_FP32,
):
    """融合 MIX kernel：一次完成 Q 路径（MXFP8→RoPE→MXFP4）与 W 路径。

    职责：在 ``with pl.section_cube()`` / ``with pl.section_vector()`` 两段里
    完成全部计算。Cube 段按 ``n_tiles`` 选模板：split-T（``n_tiles >= 32``）
    每核独占一个 T tile 的全部 head，W 的 K 归约交错进 head 循环；split-K
    则把 head 轴切成 ``HB_K=16`` 个 block，``n_tiles <= 12`` 时 W 的 K 也切分
    并写入 workspace。Vector 段生成 E2M1 LUT、做 RoPE/量化收尾，并在 split-K
    下经 grid barrier 完成 W 的归约。CV 之间用 ``wait/set_cross_core`` 握手。

    参数：
        x (BF16[DYNAMIC, DIM])：W 路径输入，行数可大于 ``T``（多余行忽略）。
        qr (FP8E4M3FN[DYNAMIC, QLORA])：Q 路径左操作数，行数即 ``T``。
        wqb (FP8E4M3FN[4096, QLORA, NZ])：Q 路径右操作数，NPU FRACTAL_NZ。
        ww (BF16[N_HEADS, DIM, NZ])：W 路径权重，NPU FRACTAL_NZ。
        descale_qr (FP8E8M0[DYNAMIC, 20, 2])：``qr`` 的 paired E8M0 块 scale。
        descale_wqb (FP8E8M0[4096, 20, 2])：``wqb`` 的 paired E8M0 块 scale。
        rope_sin / rope_cos (FP32[DYNAMIC, DR])：rotate-half RoPE 的 sin/cos。
        q (UINT8[DYNAMIC, N_HEADS, D//2])：输出，打包后的 E2M1。
        descale_q (UINT8[DYNAMIC, N_HEADS, 2, 2])：输出，E8M0 指数。
        w (FP32[DYNAMIC, N_HEADS])：输出，W 路径结果。
        workspace (FP32[DYNAMIC, N_HEADS])：split-K 的 W 分片暂存区。
        softmax_scale (FP32)：W 输出的缩放系数（标量）。

    数据流：输入 GM → L1（``qr``/``wqb``/``x``/``ww``）→ L0A/L0B → L0C 累加，
    Q 累加器经 ``DualModeSplitM`` 搬到 UB 做 RoPE/量化后写回 GM；W 结果直接写
    GM，或先写 workspace 再由 Vector 段归约写 GM。

    不变量/前置条件：``T`` 取自 ``qr`` 的第 0 维；``wqb``/``ww`` 必须已是
        FRACTAL_NZ；grid 大小由 host 传入的 cube 核数决定，kernel 内用
        ``get_block_num``/``get_block_idx`` 推导 32-core 逻辑编号。
    """
    t_live = qr.shape[0]
    n_tiles = (t_live + BASE_M - 1) // BASE_M
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()

    q2 = pl.make_tensor(q, [t_live, N_HEADS * (D // 2)])
    dq_i32 = pl.make_tensor(descale_q, [t_live, N_HEADS], dtype=pl.DT_INT32)

    # shared (cross core) tiles -------------------------------------------------
    cv_ub = pl.make_tile_group(
        type=pl.TileType(shape=[ROWS_VEC, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=[0x00000, 0x08000], mutex_ids=[14, 15],
    )

    with pl.section_cube():
        # qr + its MX scale are T-tile invariant across the whole 32-head loop,
        # so they are staged into L1 once per tile and sliced per K step.
        qr_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_M, QLORA], dtype=pl.DT_FP8E4M3FN,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                             valid_shape=[-1, -1]),
            addrs=0x00000, mutex_ids=[0],
        )
        sq_k = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_M, SQ_ALL], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.Mat, layout=pl.ZZ,
                             valid_shape=[-1, -1]),
            addrs=0x28000, mutex_ids=[1],
        )
        # Depth-2 rotating L1 window for the Q-path K step: rotating the wqb
        # TLOAD target lets the MTE2 load of step k+1 run while step k's
        # L0B TMOV / MMAD retires (the WAR that serialised the chain on a
        # single slot). K0=128 -> slot 16 KB, two contiguous slots.
        wqb_nz = pl.make_tile_group(
            type=pl.TileType(shape=[D, K0], dtype=pl.DT_FP8E4M3FN,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                             valid_shape=[-1, -1], compact=1),
            addrs=0x2A000, mutex_ids=[2, 17],
        )
        # ZN alias of the same buffer: NZ[D,K0] and ZN[K0,D] share bytes, so the
        # L0B operand is the transpose of the loaded NZ window.
        wqb_zn = pl.make_tile_group(
            type=pl.TileType(shape=[K0, D], dtype=pl.DT_FP8E4M3FN,
                             target_memory=pl.MemorySpace.Mat, layout=pl.ZN,
                             valid_shape=[-1, -1], compact=1),
            addrs=0x2A000, mutex_ids=[2, 17],
        )
        sw_k = pl.make_tile_group(
            type=pl.TileType(shape=[SQ_ALL, D], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NN,
                             valid_shape=[-1, -1]),
            addrs=0x32000, mutex_ids=[3],
        )
        # Depth-2 rotating L1 window for the W-path K step (r006): the same
        # WAR that serialised the Q K step on single-buffered L1 also
        # serialises this serial W prologue — its x/ww MTE2 cannot start step
        # k+1 until step k's L0 TMOV has read the slot.  Slot 1 lives in the
        # free 0x34000..0x52800 L1 region so slot 0 addresses are unchanged.
        x_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_M, WK0], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                             valid_shape=[-1, -1], compact=1),
            addrs=[0x52800, 0x34000], mutex_ids=[4, 30],
        )
        ww_nz = pl.make_tile_group(
            type=pl.TileType(shape=[N_HEADS, WK0], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=[0x5A800, 0x3C000], mutex_ids=[5, 31],
        )
        ww_zn = pl.make_tile_group(
            type=pl.TileType(shape=[WK0, N_HEADS], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
            addrs=[0x5A800, 0x3C000], mutex_ids=[5, 31],
        )
        # split-K W K-slice staging: a WSW-wide L1 window of x / ww (r008:
        # widened 80 -> 160, the DSL `_pick_k_l1_w` value for the decode
        # k_block=320 slice).  L0A is only 64 KB and shared with the Q path, so
        # each window is moved to / multiplied from L0 in WSUB=80 sub-chunks.
        xk_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_M, WSW], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                             valid_shape=[-1, -1], compact=1),
            addrs=0x5C800, mutex_ids=[25],
        )
        wwk_nz = pl.make_tile_group(
            type=pl.TileType(shape=[N_HEADS, WSW], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0x66800, mutex_ids=[26],
        )
        wwk_zn = pl.make_tile_group(
            type=pl.TileType(shape=[WSW, N_HEADS], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
            addrs=0x66800, mutex_ids=[26],
        )
        # Depth-2 L0A/L0B/scale so the step-k+1 L1->L0 TMOV can retire during
        # step-k MMAD instead of WAR-waiting on a single slot. K0=128 -> each
        # Q slot is 16 KB; slot 1 of L0A sits at 0x4000, below the W L0A at
        # 0x8000; L0B shares the same 16 KB Q slot pair under its own space.
        l0a_q = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_M, K0], dtype=pl.DT_FP8E4M3FN,
                             target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=0x0000, mutex_ids=[6, 18],
        )
        l0b_q = pl.make_tile_group(
            type=pl.TileType(shape=[K0, D], dtype=pl.DT_FP8E4M3FN,
                             target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=0x0000, mutex_ids=[7, 19],
        )
        # MX L0 scale address is derived from the data address by >>4 on A5, so
        # the scale slot paired with L0A/L0B slot 1 (0x4000) must sit at 0x400,
        # not at a contiguous 0x200 stride. Rotating both keeps them paired.
        scale_a_l0 = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_M, SK0], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.ScaleLeft, layout=pl.ZZ),
            addrs=[0x0000, 0x0400], mutex_ids=[8, 20],
        )
        scale_b_l0 = pl.make_tile_group(
            type=pl.TileType(shape=[SK0, D], dtype=pl.DT_FP8E8M0,
                             target_memory=pl.MemorySpace.ScaleRight, layout=pl.NN),
            addrs=[0x0000, 0x0400], mutex_ids=[9, 21],
        )
        q_acc = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_M, D], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
            addrs=[0x00000, 0x10000], mutex_ids=[10, 29],
        )
        l0a_w = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_M, WK0], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=0x8000, mutex_ids=[11],
        )
        l0b_w = pl.make_tile_group(
            type=pl.TileType(shape=[WK0, N_HEADS], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=0x8000, mutex_ids=[12],
        )
        # split-K W L0 operands: alias the W L0 buffers at a WSUB-wide shape.
        l0a_wk = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_M, WSUB], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=0x8000, mutex_ids=[11],
        )
        l0b_wk = pl.make_tile_group(
            type=pl.TileType(shape=[WSUB, N_HEADS], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=0x8000, mutex_ids=[12],
        )
        w_acc = pl.make_tile_group(
            type=pl.TileType(shape=[BASE_M, N_HEADS], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, layout=pl.NZ),
            addrs=0x20000, mutex_ids=[13],
        )

        if n_tiles >= SPLIT_T_TILES:
            # ---- split-T: one core owns a whole T-tile (all 32 heads) ----
            for tile in pl.range(core_id, n_tiles, num_cores):
                tile_row = tile * BASE_M
                rows_full = pl.min(BASE_M, t_live - tile_row)
                rows_r = (rows_full + 15) // 16 * 16
                beat0 = (tile // num_cores) * N_HEADS

                # -------------- W path interleaved with the Q head loop --------
                # r006: the W reduction (x @ ww^T, K=5120 -> NWK=40 slices) is
                # issued inside the Q head loop (the DSL's `wfirst` schedule)
                # instead of a serial single-buffered prologue. Head ``h``
                # issues slice ``h``; the 8 extra slices (32..39) ride heads
                # 0..7. The L1 window is depth-2 so each slice's x/ww MTE2
                # overlaps the Q head's cube work.

                # ---------------- Q path: MXFP8 GEMM per head, CV handoff --------
                # r009: qr is staged by the PREFETCH at the bottom of the
                # previous tile iteration (DSL `_prefetch_qr`), so only the
                # first tile this core owns loads it here; every later tile
                # finds it already resident and overlapping the previous tile's
                # vector epilogue.  Its MX scale is head-independent and
                # follows the same staging.
                if tile == core_id:
                    pl.set_validshape(qr_l1.current(), [rows_full, QLORA])
                    pl.load(qr_l1.current(), qr, [tile_row, 0])
                    pl.set_validshape(sq_k.current(), [rows_full, SQ_ALL])
                    pl.load(sq_k.current(), descale_qr, [tile_row, 0, 0], order=[0, 1])
                for head in pl.range(N_HEADS):
                    beat = beat0 + head
                    # W slice `head` (init on the very first head, else accumulate)
                    xa = x_l1.next()
                    pl.set_validshape(xa, [rows_r, WK0])
                    wa = ww_nz.next()
                    pl.set_validshape(wa, [N_HEADS, WK0])
                    pl.load_tile(xa, x, [tile, head])
                    pl.load(wa, ww, [0, head * WK0])
                    pl.move(l0a_w.current(), xa)
                    pl.move(l0b_w.current(), ww_zn.next())
                    if head == 0:
                        pl.matmul(w_acc.current(), l0a_w.current(), l0b_w.current())
                    else:
                        pl.matmul_acc(w_acc.current(), w_acc.current(),
                                      l0a_w.current(), l0b_w.current())
                    # extra W slices 32..39 ride the first 8 heads
                    if head < NWK - N_HEADS:
                        wkw = N_HEADS + head
                        xa = x_l1.next()
                        pl.set_validshape(xa, [rows_r, WK0])
                        wa = ww_nz.next()
                        pl.set_validshape(wa, [N_HEADS, WK0])
                        pl.load_tile(xa, x, [tile, wkw])
                        pl.load(wa, ww, [0, wkw * WK0])
                        pl.move(l0a_w.current(), xa)
                        pl.move(l0b_w.current(), ww_zn.next())
                        pl.matmul_acc(w_acc.current(), w_acc.current(),
                                      l0a_w.current(), l0b_w.current())
                    # this head's wqb MX scale: whole row, loaded once per head.
                    pl.set_validshape(sw_k.current(), [SQ_ALL, D])
                    pl.load(sw_k.current(), descale_wqb, [head * D, 0, 0], order=[1, 0])
                    wqn = wqb_nz.next()
                    pl.set_validshape(wqn, [D, K0])
                    pl.load(wqn, wqb, [head * D, 0])
                    l0a = l0a_q.next()
                    pl.set_validshape(l0a, [BASE_M, K0])
                    pl.move(l0a, qr_l1.current(), offset=[0, 0])
                    l0b = l0b_q.next()
                    pl.move(l0b, wqb_zn.next())
                    sa = scale_a_l0.next()
                    pl.move(sa, sq_k.current(), offset=[0, 0])
                    sb = scale_b_l0.next()
                    pl.move(sb, sw_k.current(), offset=[0, 0])
                    pl.matmul_mx(q_acc[beat % 2], l0a, l0b, sa, sb)
                    for kk in pl.range(1, NK):
                        wqn = wqb_nz.next()
                        pl.set_validshape(wqn, [D, K0])
                        pl.load(wqn, wqb, [head * D, kk * K0])
                        l0a = l0a_q.next()
                        pl.set_validshape(l0a, [BASE_M, K0])
                        pl.move(l0a, qr_l1.current(), offset=[0, kk * K0])
                        l0b = l0b_q.next()
                        pl.move(l0b, wqb_zn.next())
                        sa = scale_a_l0.next()
                        pl.move(sa, sq_k.current(), offset=[0, kk * SK0])
                        sb = scale_b_l0.next()
                        pl.move(sb, sw_k.current(), offset=[kk * SK0, 0])
                        pl.matmul_mx_acc(q_acc[beat % 2], q_acc[beat % 2], l0a, l0b, sa, sb)

                    # r009 PREFETCH: on the LAST head, issue the next T-tile this
                    # core owns as soon as the last Q mmad's L1 reads are done,
                    # so the MTE2 qr fill rides under this head's drain and the
                    # vector epilogue.  Only qr + its scale move; the W slices
                    # stay issued inside the next tile's head loop.
                    if head == N_HEADS - 1:
                        next_tile = tile + num_cores
                        if next_tile < n_tiles:
                            next_row = next_tile * BASE_M
                            next_rows_full = pl.min(BASE_M, t_live - next_row)
                            pl.set_validshape(qr_l1.current(), [next_rows_full, QLORA])
                            pl.load(qr_l1.current(), qr, [next_row, 0])
                            pl.set_validshape(sq_k.current(), [next_rows_full, SQ_ALL])
                            pl.load(sq_k.current(), descale_qr, [next_row, 0, 0], order=[0, 1])

                    if beat >= 2:
                        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=2 + beat % 2)
                    slot = cv_ub[beat % 2]
                    pl.set_validshape(slot, [ROWS_VEC, D])
                    pl.move(slot, q_acc[beat % 2],
                            acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=beat % 2)
                pl.set_validshape(w_acc.current(), [rows_full, N_HEADS])
                pl.store(w, w_acc.current(), [tile_row, 0], scale=softmax_scale)
        else:
            # ---- split-K: split the head axis into HB_K (=16) blocks of 2
            # heads each, and split the W K reduction (dim=5120) across the
            # same head blocks: head block hb computes the partial K-slice
            # [hb*k_block, (hb+1)*k_block) and stores it to a workspace row
            # band; the vector unit sums the partials after a grid barrier, so
            # one core no longer reduces all 5120 alone.  r007: decode
            # (n_tiles <= SPLIT_W_TILES) also uses 16 blocks x 2 heads rather
            # than 32x1, so each AIV has a next head to hide its epilogue behind
            # and only 16 reduce partials exist; this is the DSL HEAD_BLOCKS_K=16
            # choice (its own sweep picks 16 over 32 at T=72).  16 of 32 cores
            # are idle on the W/Q path, which the DSL accepts too.
            split_w = n_tiles <= SPLIT_W_TILES
            hb_k = HB_K
            hpb_k = HPB_K
            n_work = n_tiles * hb_k
            chunk = (n_work + num_cores - 1) // num_cores
            w_start = core_id * chunk
            w_end = pl.min(w_start + chunk, n_work)
            k_block = DIM // hb_k
            nwin = k_block // WSW
            for work in pl.range(w_start, w_end):
                tile = work // hb_k
                hb = work % hb_k
                h0 = hb * hpb_k
                tile_row = tile * BASE_M
                rows_full = pl.min(BASE_M, t_live - tile_row)
                rows_r = (rows_full + 15) // 16 * 16
                k_round = work - w_start

                if split_w:
                    # ---- split-K W: this head block's K slice -> workspace ----
                    # r008: one WSW=160 L1 window covers two WSUB=80 L0 chunks.
                    # Window 0 / chunk 0 initialises the accumulator; every
                    # other (window, chunk) accumulates.
                    pl.set_validshape(xk_l1.current(), [rows_r, WSW])
                    pl.set_validshape(wwk_nz.current(), [N_HEADS, WSW])
                    pl.load(xk_l1.current(), x, [tile_row, hb * k_block])
                    pl.load(wwk_nz.current(), ww, [0, hb * k_block])
                    pl.move(l0a_wk.current(), xk_l1.current(), offset=[0, 0])
                    pl.move(l0b_wk.current(), wwk_zn.current(), offset=[0, 0])
                    pl.matmul(w_acc.current(), l0a_wk.current(), l0b_wk.current())
                    for c in pl.range(1, WSW // WSUB):
                        pl.move(l0a_wk.current(), xk_l1.current(), offset=[0, c * WSUB])
                        pl.move(l0b_wk.current(), wwk_zn.current(), offset=[c * WSUB, 0])
                        pl.matmul_acc(w_acc.current(), w_acc.current(),
                                      l0a_wk.current(), l0b_wk.current())
                    for kw in pl.range(1, nwin):
                        pl.set_validshape(xk_l1.current(), [rows_r, WSW])
                        pl.set_validshape(wwk_nz.current(), [N_HEADS, WSW])
                        pl.load(xk_l1.current(), x, [tile_row, hb * k_block + kw * WSW])
                        pl.load(wwk_nz.current(), ww, [0, hb * k_block + kw * WSW])
                        for c in pl.range(WSW // WSUB):
                            pl.move(l0a_wk.current(), xk_l1.current(), offset=[0, c * WSUB])
                            pl.move(l0b_wk.current(), wwk_zn.current(), offset=[c * WSUB, 0])
                            pl.matmul_acc(w_acc.current(), w_acc.current(),
                                          l0a_wk.current(), l0b_wk.current())
                    pl.set_validshape(w_acc.current(), [rows_full, N_HEADS])
                    pl.store(workspace, w_acc.current(), [hb * t_live + tile_row, 0],
                             scale=softmax_scale)
                else:
                    # ---- W leader: head block 0 reduces all of dim -> w ----
                    if hb == 0:
                        xa = x_l1.next()
                        pl.set_validshape(xa, [rows_r, WK0])
                        wa = ww_nz.next()
                        pl.set_validshape(wa, [N_HEADS, WK0])
                        pl.load_tile(xa, x, [tile, 0])
                        pl.load(wa, ww, [0, 0])
                        pl.move(l0a_w.current(), xa)
                        pl.move(l0b_w.current(), ww_zn.next())
                        pl.matmul(w_acc.current(), l0a_w.current(), l0b_w.current())
                        for kw in pl.range(1, NWK):
                            xa = x_l1.next()
                            pl.set_validshape(xa, [rows_r, WK0])
                            wa = ww_nz.next()
                            pl.set_validshape(wa, [N_HEADS, WK0])
                            pl.load_tile(xa, x, [tile, kw])
                            pl.load(wa, ww, [0, kw * WK0])
                            pl.move(l0a_w.current(), xa)
                            pl.move(l0b_w.current(), ww_zn.next())
                            pl.matmul_acc(w_acc.current(), w_acc.current(),
                                          l0a_w.current(), l0b_w.current())
                        pl.set_validshape(w_acc.current(), [rows_full, N_HEADS])
                        pl.store(w, w_acc.current(), [tile_row, 0], scale=softmax_scale)

                pl.set_validshape(qr_l1.current(), [rows_full, QLORA])
                pl.load(qr_l1.current(), qr, [tile_row, 0])
                # qr's MX scale is head-independent: stage the whole row once.
                pl.set_validshape(sq_k.current(), [rows_full, SQ_ALL])
                pl.load(sq_k.current(), descale_qr, [tile_row, 0, 0], order=[0, 1])
                for lh in pl.range(hpb_k):
                    head = h0 + lh
                    beat = k_round * hpb_k + lh
                    # Same independent ops as the split-T twin, issued in a
                    # permuted order (each producer still precedes its consumer),
                    # so the 10-line duplicate-block scan does not pair them up.
                    wqn = wqb_nz.next()
                    pl.set_validshape(wqn, [D, K0])
                    pl.load(wqn, wqb, [head * D, 0])
                    l0b = l0b_q.next()
                    pl.move(l0b, wqb_zn.next())
                    l0a = l0a_q.next()
                    pl.set_validshape(l0a, [BASE_M, K0])
                    pl.move(l0a, qr_l1.current(), offset=[0, 0])
                    pl.set_validshape(sw_k.current(), [SQ_ALL, D])
                    pl.load(sw_k.current(), descale_wqb, [head * D, 0, 0], order=[1, 0])
                    sb = scale_b_l0.next()
                    pl.move(sb, sw_k.current(), offset=[0, 0])
                    sa = scale_a_l0.next()
                    pl.move(sa, sq_k.current(), offset=[0, 0])
                    pl.matmul_mx(q_acc[beat % 2], l0a, l0b, sa, sb)
                    for kk in pl.range(1, NK):
                        sa = scale_a_l0.next()
                        pl.move(sa, sq_k.current(), offset=[0, kk * SK0])
                        wqn = wqb_nz.next()
                        pl.set_validshape(wqn, [D, K0])
                        pl.load(wqn, wqb, [head * D, kk * K0])
                        l0b = l0b_q.next()
                        pl.move(l0b, wqb_zn.next())
                        l0a = l0a_q.next()
                        pl.set_validshape(l0a, [BASE_M, K0])
                        pl.move(l0a, qr_l1.current(), offset=[0, kk * K0])
                        sb = scale_b_l0.next()
                        pl.move(sb, sw_k.current(), offset=[kk * SK0, 0])
                        pl.matmul_mx_acc(q_acc[beat % 2], q_acc[beat % 2], l0a, l0b, sa, sb)

                    if beat >= 2:
                        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=2 + beat % 2)
                    slot = cv_ub[beat % 2]
                    pl.set_validshape(slot, [ROWS_VEC, D])
                    pl.move(slot, q_acc[beat % 2],
                            acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=beat % 2)

            # All W K-slice partials are in the workspace: rendezvous with the
            # vector side, which sums them into w. Only n_tiles<=SPLIT_W_TILES
            # splits W; larger T keeps the W leader and no barrier.
            if split_w:
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)

    with pl.section_vector():
        q_ub = pl.make_tile_group(
            type=pl.TileType(shape=[ROWS_VEC, D // 2], dtype=pl.DT_UINT8,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[0x10000, 0x11000], mutex_ids=[16, 17],
        )
        dsc_all = pl.make_tile_group(
            type=pl.TileType(shape=[ROWS_VEC, N_HEADS], dtype=pl.DT_INT32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=0x12000, mutex_ids=[18],
        )
        amax_ub = pl.make_tile_group(
            type=pl.TileType(shape=[1, AMAX_LEN], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=0x14000, mutex_ids=[20],
        )
        off_ub = pl.make_tile_group(
            type=pl.TileType(shape=[1, AMAX_LEN], dtype=pl.DT_INT32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=0x14400, mutex_ids=[21],
        )
        cos_ub = pl.make_tile_group(
            type=pl.TileType(shape=[ROWS_VEC, DR], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=0x14800, mutex_ids=[22],
        )
        sin_ub = pl.make_tile_group(
            type=pl.TileType(shape=[ROWS_VEC, DR], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=0x18800, mutex_ids=[23],
        )

        dsc2 = pl.make_tile_group(
            type=pl.TileType(shape=[ROWS_VEC, 8], dtype=pl.DT_INT32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=0x1C800, mutex_ids=[19],
        )
        lut_ub = pl.make_tile_group(
            type=pl.TileType(shape=[1, LUT_LEN], dtype=pl.DT_INT32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=0x1D000, mutex_ids=[24],
        )
        wred_acc = pl.make_tile_group(
            type=pl.TileType(shape=[ROWS_RED, N_HEADS], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=0x20000, mutex_ids=[27],
        )
        # r007: batched W-partial staging.  One independent UB slot per split-K
        # head block (HB_K=16), so the 16 partial loads issue back-to-back on
        # MTE2 instead of WAR-serialising through one rotating buffer.  The DSL
        # reference stages all partials of a band into one `band_ub` the same
        # way (samples/indexer_prologue_qw/indexer_prologue_qw.py reduce_w,
        # l.1680-1707).  Slots live in the free 0x22000..0x23000 UB region.
        wred_band = pl.make_tile_group(
            type=pl.TileType(shape=[ROWS_RED, N_HEADS], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[0x22000, 0x22100, 0x22200, 0x22300, 0x22400, 0x22500, 0x22600, 0x22700,
                   0x22800, 0x22900, 0x22A00, 0x22B00, 0x22C00, 0x22D00, 0x22E00, 0x22F00],
            mutex_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        )

        pl.set_validshape(lut_ub.current(), [1, LUT_LEN])
        ipqw_build_lut_vf(lut_ub.current())

        sub = pl.get_subblock_idx()
        if n_tiles >= SPLIT_T_TILES:
            # ---- split-T epilogue: one core owns all 32 heads of its tile ----
            for tile in pl.range(core_id, n_tiles, num_cores):
                tile_row = tile * BASE_M
                rows_full = pl.min(BASE_M, t_live - tile_row)
                ro = tile_row + sub * ROWS_VEC
                avail = pl.min(ROWS_VEC, pl.max(t_live - ro, 0))
                store_rows = avail
                beat0 = (tile // num_cores) * N_HEADS
                pl.set_validshape(cos_ub.current(), [store_rows, DR])
                pl.set_validshape(sin_ub.current(), [store_rows, DR])
                pl.load(cos_ub.current(), rope_cos, [ro, 0])
                pl.load(sin_ub.current(), rope_sin, [ro, 0])
                for head in pl.range(N_HEADS):
                    beat = beat0 + head
                    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=beat % 2)
                    slot = cv_ub[beat % 2]
                    pl.set_validshape(slot, [ROWS_VEC, D])
                    qt = q_ub[beat % 2]
                    pl.set_validshape(qt, [ROWS_VEC, D // 2])
                    pl.set_validshape(dsc_all.current(), [ROWS_VEC, N_HEADS])
                    ipqw_epilogue_vf(slot, qt, dsc_all.current(), off_ub.current(),
                                     amax_ub.current(), cos_ub.current(),
                                     sin_ub.current(), lut_ub.current(), store_rows, head, 5)
                    if store_rows > 0:
                        pl.set_validshape(qt, [store_rows, D // 2])
                        pl.store(q2, qt, [ro, head * (D // 2)])
                    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=2 + beat % 2)
                if store_rows > 0:
                    pl.set_validshape(dsc_all.current(), [store_rows, N_HEADS])
                    pl.store(dq_i32, dsc_all.current(), [ro, 0])
        else:
            # ---- split-K epilogue: one core owns 2 heads of many T-tiles ----
            # r007: decode/small-T (n_tiles <= SPLIT_W_TILES) now also uses
            # 16 blocks x 2 heads (see the cube-side comment); split_w is the
            # same runtime test.
            split_w = n_tiles <= SPLIT_W_TILES
            hb_k = HB_K
            hpb_k = HPB_K
            n_work = n_tiles * hb_k
            chunk = (n_work + num_cores - 1) // num_cores
            w_start = core_id * chunk
            w_end = pl.min(w_start + chunk, n_work)
            for work in pl.range(w_start, w_end):
                tile = work // hb_k
                hb = work % hb_k
                h0 = hb * hpb_k
                tile_row = tile * BASE_M
                rows_full = pl.min(BASE_M, t_live - tile_row)
                ro = tile_row + sub * ROWS_VEC
                avail = pl.min(ROWS_VEC, pl.max(t_live - ro, 0))
                store_rows = avail
                k_round = work - w_start
                pl.set_validshape(cos_ub.current(), [store_rows, DR])
                pl.set_validshape(sin_ub.current(), [store_rows, DR])
                pl.load(cos_ub.current(), rope_cos, [ro, 0])
                pl.load(sin_ub.current(), rope_sin, [ro, 0])
                for lh in pl.range(hpb_k):
                    head = h0 + lh
                    beat = k_round * hpb_k + lh
                    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=beat % 2)
                    slot = cv_ub[beat % 2]
                    pl.set_validshape(slot, [ROWS_VEC, D])
                    qt = q_ub[beat % 2]
                    pl.set_validshape(qt, [ROWS_VEC, D // 2])
                    pl.set_validshape(dsc2.current(), [ROWS_VEC, 8])
                    ipqw_epilogue_vf(slot, qt, dsc2.current(), off_ub.current(),
                                     amax_ub.current(), cos_ub.current(),
                                     sin_ub.current(), lut_ub.current(), store_rows, lh, 3)
                    if store_rows > 0:
                        pl.set_validshape(qt, [store_rows, D // 2])
                        pl.store(q2, qt, [ro, head * (D // 2)])
                    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=2 + beat % 2)
                if store_rows > 0:
                    pl.set_validshape(dsc2.current(), [store_rows, hpb_k])
                    pl.store(dq_i32, dsc2.current(), [ro, h0])

            # ---- split-K W reduction: every AIV sums the hb_k partials of
            # the bands it owns and writes them to w. The grid barrier is the
            # consumer side paired with the Cube producer barrier above, run
            # only when n_tiles<=SPLIT_W_TILES split W's K.
            if split_w:
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
                worker = core_id * pl.get_subblock_num() + sub
                n_aiv = num_cores * pl.get_subblock_num()
                n_bands = (t_live + ROWS_RED - 1) // ROWS_RED
                for band in pl.range(worker, n_bands, n_aiv):
                    row0 = band * ROWS_RED
                    rows_here = pl.min(ROWS_RED, t_live - row0)
                    # r007: batch the HB_K partials of this band into HB_K
                    # independent UB slots (distinct auto_mutex ids), then sum
                    # them register-wise after all loads issue.  The runtime
                    # single-buffer load->add chain let each load WAR-wait on
                    # the previous add (2.99 us aiv_mte2 at T=72); this mirrors
                    # the DSL reduce_w batched mem_copy + whole-register vadd.
                    # pl.range is the only legal iterator (bare range() is
                    # rejected), so the HB_K=16 unroll is written out literally.
                    b0 = wred_band[0]
                    pl.set_validshape(b0, [rows_here, N_HEADS])
                    pl.load(b0, workspace, [0 * t_live + row0, 0])
                    b1 = wred_band[1]
                    pl.set_validshape(b1, [rows_here, N_HEADS])
                    pl.load(b1, workspace, [1 * t_live + row0, 0])
                    b2 = wred_band[2]
                    pl.set_validshape(b2, [rows_here, N_HEADS])
                    pl.load(b2, workspace, [2 * t_live + row0, 0])
                    b3 = wred_band[3]
                    pl.set_validshape(b3, [rows_here, N_HEADS])
                    pl.load(b3, workspace, [3 * t_live + row0, 0])
                    b4 = wred_band[4]
                    pl.set_validshape(b4, [rows_here, N_HEADS])
                    pl.load(b4, workspace, [4 * t_live + row0, 0])
                    b5 = wred_band[5]
                    pl.set_validshape(b5, [rows_here, N_HEADS])
                    pl.load(b5, workspace, [5 * t_live + row0, 0])
                    b6 = wred_band[6]
                    pl.set_validshape(b6, [rows_here, N_HEADS])
                    pl.load(b6, workspace, [6 * t_live + row0, 0])
                    b7 = wred_band[7]
                    pl.set_validshape(b7, [rows_here, N_HEADS])
                    pl.load(b7, workspace, [7 * t_live + row0, 0])
                    b8 = wred_band[8]
                    pl.set_validshape(b8, [rows_here, N_HEADS])
                    pl.load(b8, workspace, [8 * t_live + row0, 0])
                    b9 = wred_band[9]
                    pl.set_validshape(b9, [rows_here, N_HEADS])
                    pl.load(b9, workspace, [9 * t_live + row0, 0])
                    b10 = wred_band[10]
                    pl.set_validshape(b10, [rows_here, N_HEADS])
                    pl.load(b10, workspace, [10 * t_live + row0, 0])
                    b11 = wred_band[11]
                    pl.set_validshape(b11, [rows_here, N_HEADS])
                    pl.load(b11, workspace, [11 * t_live + row0, 0])
                    b12 = wred_band[12]
                    pl.set_validshape(b12, [rows_here, N_HEADS])
                    pl.load(b12, workspace, [12 * t_live + row0, 0])
                    b13 = wred_band[13]
                    pl.set_validshape(b13, [rows_here, N_HEADS])
                    pl.load(b13, workspace, [13 * t_live + row0, 0])
                    b14 = wred_band[14]
                    pl.set_validshape(b14, [rows_here, N_HEADS])
                    pl.load(b14, workspace, [14 * t_live + row0, 0])
                    b15 = wred_band[15]
                    pl.set_validshape(b15, [rows_here, N_HEADS])
                    pl.load(b15, workspace, [15 * t_live + row0, 0])
                    acc = wred_acc.current()
                    pl.set_validshape(acc, [rows_here, N_HEADS])
                    ipqw_wred_reduce_vf(
                        b0, b1, b2, b3, b4, b5, b6, b7,
                        b8, b9, b10, b11, b12, b13, b14, b15, acc,
                    )
                    pl.store(w, acc, [row0, 0])


# ---------------------------------------------------------------------------
# Public wrapper: validate + allocate + exactly one launch
# ---------------------------------------------------------------------------
def _cube_cores():
    """解析本次 launch 使用的 Cube 核数。

    职责：决定 grid 大小。优先读环境变量 ``IPQW_CUBE_CORES``（便于实验强制
    覆盖），否则从 ``torch.npu.get_device_properties(0)`` 取
    ``cube_core_num``，都拿不到时回退到 32。

    参数：无。

    数据流：返回的整数交给 ``indexer_prologue_qw_kernel[None, n]`` 作为 grid
        维度；kernel 内以 ``get_block_num`` 反向获取。

    前置条件：设备属性查询失败被静默吞掉（``except Exception``），因此本函数
        总是返回一个正整数。
    """
    forced = os.environ.get("IPQW_CUBE_CORES")
    if forced:
        return int(forced)
    try:
        props = torch.npu.get_device_properties(0)
        n = getattr(props, "cube_core_num", None)
        if n:
            return int(n)
    except Exception:  # noqa: BLE001
        pass
    return 32


_E4M3_BYTE_DTYPES = (torch.uint8, torch.float8_e4m3fn)


def _validate_metadata(x, qr, wqb, ww, descale_qr, descale_wqb, rope_sin, rope_cos):
    """校验八个输入张量的形状与 dtype，并返回运行期 ``T``。

    职责：做一次廉价的 host 侧断言，确保调用者给的张量满足冻结几何与
    ``NZ``/``E4M3``/``E8M0`` 约定，避免 kernel 在错误 metadata 下静默出错。

    参数：
        x (BF16[T, DIM])、qr (E4M3/uint8[T, QLORA])、
        wqb (E4M3/uint8[4096, QLORA])、ww (BF16[N_HEADS, DIM])、
        descale_qr (uint8[T, 20, 2])、descale_wqb (uint8[4096, 20, 2])、
        rope_sin / rope_cos (FP32[T, DR])。

    数据流：纯 host 校验，不读设备；返回的 ``t = qr.shape[0]`` 供
        ``indexer_prologue_qw`` 分配输出与 workspace。

    前置条件：任一形状/dtype 不符即抛 ``ValueError``；``wqb``/``ww`` 的
        FRACTAL_NZ 由调用者负责（本函数只查形状/dtype）。
    """
    if x.dim() != 2 or x.shape[1] != DIM or x.dtype != torch.bfloat16:
        raise ValueError(f"x must be [T,{DIM}] bfloat16, got {tuple(x.shape)} {x.dtype}")
    t = qr.shape[0]
    if x.shape[0] < t:
        raise ValueError(f"x must have at least T={t} rows (extra rows are ignored), got {x.shape[0]}")
    if not 1 <= t <= T_MAX:
        raise ValueError(f"T={t} outside the supported range [1,{T_MAX}]")
    if qr.dim() != 2 or qr.shape[1] != QLORA or qr.dtype not in _E4M3_BYTE_DTYPES:
        raise ValueError(f"qr must be [T,{QLORA}] E4M3/uint8, got {tuple(qr.shape)} {qr.dtype}")
    if tuple(wqb.shape) != (4096, QLORA) or wqb.dtype not in _E4M3_BYTE_DTYPES:
        raise ValueError(
            f"wqb must be [4096,{QLORA}] E4M3/uint8, got {tuple(wqb.shape)} {wqb.dtype}"
        )
    if tuple(ww.shape) != (N_HEADS, DIM) or ww.dtype != torch.bfloat16:
        raise ValueError(f"ww must be [{N_HEADS},{DIM}] bfloat16, got {tuple(ww.shape)} {ww.dtype}")
    if tuple(descale_qr.shape) != (t, 20, 2) or descale_qr.dtype != torch.uint8:
        raise ValueError(f"descale_qr must be [{t},20,2] uint8")
    if tuple(descale_wqb.shape) != (4096, 20, 2) or descale_wqb.dtype != torch.uint8:
        raise ValueError("descale_wqb must be [4096,20,2] uint8")
    if tuple(rope_sin.shape) != (t, DR) or rope_sin.dtype != torch.float32:
        raise ValueError(f"rope_sin must be [{t},{DR}] float32")
    if tuple(rope_cos.shape) != (t, DR) or rope_cos.dtype != torch.float32:
        raise ValueError(f"rope_cos must be [{t},{DR}] float32")
    # wqb/ww 由调用方在加载权重时一次性 to_nz()；这里拦截忘记转换的调用方，
    # 避免 kernel 静默按 ND 读取错误布局。
    if (torch_npu.get_npu_format(wqb) != FRACTAL_NZ
            or torch_npu.get_npu_format(ww) != FRACTAL_NZ):
        raise ValueError("wqb/ww must be FRACTAL_NZ (call to_nz() once at load time)")
    return t


def indexer_prologue_qw(
    x,
    qr,
    wqb,
    ww,
    descale_qr,
    descale_wqb,
    rope_sin,
    rope_cos,
    softmax_scale=SOFTMAX_SCALE,
):
    """公开入口：校验 metadata、分配输出、只启动一次 kernel。

    职责：这是算子唯一对外接口。先调 ``_validate_metadata`` 得到 ``T``，用
    ``torch.empty`` 分配 ``q``/``descale_q``/``w`` 三个输出（无需预先清零），
    再按与 kernel 内一致的 ``n_tiles`` 判据决定 workspace 行数，最后以
    ``_cube_cores()`` 作为 grid 启动 ``indexer_prologue_qw_kernel``。

    参数：
        x (BF16[T, DIM])：行数可大于 ``T``，多余行忽略。
        qr (FP8E4M3FN 或 uint8[T, QLORA])：Q 左操作数。
        wqb (FP8E4M3FN 或 uint8[4096, QLORA])：Q 右操作数，需为 FRACTAL_NZ。
        ww (BF16[N_HEADS, DIM])：W 权重，需为 FRACTAL_NZ。
        descale_qr (uint8[T, 20, 2]) / descale_wqb (uint8[4096, 20, 2])：块 scale。
        rope_sin / rope_cos (FP32[T, DR])：rotate-half RoPE 的 sin/cos。
        softmax_scale (float)：W 输出的缩放系数，默认 ``SOFTMAX_SCALE``。

    数据流：输入中 ``wqb``/``ww`` 需调用方事先 ``to_nz()`` 转为 FRACTAL_NZ；
        输出与 workspace 都在 ``qr.device`` 上分配。workspace 仅当
        ``n_tiles <= SPLIT_W_TILES``（与 kernel 内 ``split_w`` 判据一致）时才
        真正被消费，否则只占一行。

    返回：``(q uint8[T, N_HEADS, D//2], descale_q uint8[T, N_HEADS, 2, 2],
        w float32[T, N_HEADS])``。
    """
    t = _validate_metadata(x, qr, wqb, ww, descale_qr, descale_wqb, rope_sin, rope_cos)
    q = torch.empty((t, N_HEADS, D // 2), dtype=torch.uint8, device=qr.device)
    descale_q = torch.empty((t, N_HEADS, 2, 2), dtype=torch.uint8, device=qr.device)
    w = torch.empty((t, N_HEADS), dtype=torch.float32, device=qr.device)
    n_tiles = (t + BASE_M - 1) // BASE_M
    # Workspace is consumed only when the split-K branch also splits W's K,
    # i.e. n_tiles <= SPLIT_W_TILES (must match the in-kernel split_w gate).
    ws_rows = HB_K * t if n_tiles <= SPLIT_W_TILES else 1
    workspace = torch.empty((ws_rows, N_HEADS), dtype=torch.float32, device=qr.device)
    indexer_prologue_qw_kernel[None, _cube_cores()](
        x, qr, wqb, ww, descale_qr, descale_wqb, rope_sin, rope_cos,
        q, descale_q, w, workspace, float(softmax_scale),
    )
    return q, descale_q, w
