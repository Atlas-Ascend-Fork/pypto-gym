#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You should not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""flash_kda_cp_preprocess kernel + wrapper（单 ``@pl.jit`` MixKernel 全链）。

一次 launch 完成 8 路输出，块内按 64-token chunk 计算，块间做 summary 递推。
分段：s1a(L2Norm/门控累加/β)→s1b(操作数因子化+emit)→s2(kk/mqk MMAD)→
s3(T/seed 构造)→s4(Neumann 求逆)→s5(u/w MMAD)→Stage 2(φ 合成与 R/ψ 递推)。
Stage 1 交错执行相邻块任务，两个 AIV 子块分别准备 Q 路和 K 路。
Stage 2 局部路径交错执行 φ(r)、递推(r-1) 和预取(r+1)。
共享路径先按块生成 φ，再由各行片读取同一份 φ 递推 R/ψ。
"""

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch_npu  # noqa: F401

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf
from pypto_pro.runtime.tilingkey import TilingKeyField

_OP_DIR = str(Path(__file__).resolve().parents[1])
for _p in (_OP_DIR, _OP_DIR + "/op_kernel"):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ===== 输出合同（8 路 NamedTuple——兼容 tuple 解包与索引） =====


class FlashKDACPOutputs(NamedTuple):
    """flash_kda_cp_preprocess 的 8 路输出（NamedTuple 兼容 tuple 解包）。

    field 顺序即公开合同顺序：
    transition / state_ext / decay / qg / kg / u / w / mqk
    """
    transition: torch.Tensor
    state_ext: torch.Tensor
    decay: torch.Tensor
    qg: torch.Tensor
    kg: torch.Tensor
    u: torch.Tensor
    w: torch.Tensor
    mqk: torch.Tensor


# ===== 常量 =====
CHUNK = 64
D = 128
LANES = 64
VREG = 2
L2_EPS = 1e-6

LAYOUT_BNSD = 0
LAYOUT_BSND = 1
LAYOUT_TND = 2

# Stage 1 GM 操作数布局（每任务 BF16 元素偏移）。
WS_BV_OFF = 20480
WS_BKT_OFF = 28672
WS_OPS_ELEMS = 36864

# 事件（UNICAST_BLOCK；AIV0 与 AIC 双向配对，顺序复用）
EV_READY_OP = 0
EV_READY_KM = 1
EV_READY_TS = 2
EV_FIRST_R_FREE = 7  # Cube MTE1 consumed the first phi as the left operand.
EV_PHI_RELEASE = 4  # The vector store has finished reading the previous UB pair.
EV_READY_WU = 8  # stage2 vec#0→cube#1 的 w/u 就绪（独立于 TS——避免双 set 竞态）
EV_READY_UF = 9  # stage2 cube#2→vec#0(c+1) 的 u 读完成（跨管道读写互斥）
EV_ZERO_READY = 15  # AIV0 publishes the checked zero status to the Cube scalar pipe.
EV_CHAIN_DONE = 14  # stage2 同核链边界：前链末 rank 读毕 ut2/kg2/wt2 → 后链 vec init 可覆写
# km UB 槽反向释放：FIX 写须等 vec V 读完成（固定坐标事件无法表达，按读毕释放）
EV_KM_KK_REL = 10
EV_KM_MQK_REL = 11
# （TND）：φ(r) 读毕 wt2 → 下一轮 transpose(r+1) 可覆写（wt2 单槽跨程序）
EV_WT_REL = 13
# summary 操作数走 GM（parity 双缓冲）；fixpipe 标志对 id 0-7 避开跨核事件
EV_PANEL_L1 = 3  # LEFT 面板（s1b 双 sub 各 set → s2 一 wait 收双路，INTRA）
EV_FX_M0 = 3
EV_FX_R1 = 4
EV_FX_M2 = 5
EV_FX_R3 = 6
EV_FX_W4 = 7
# 共享 φ 的 2b 使用两组事件。R 写回后放行 φ/R 装载，ψ 写回后放行 ψ/uᵀ/kg 装载。
# 与 Stage 1 的 s4/s5 分阶段执行，可分时复用事件 ID 5/6。
EV_FX_RR = 5
EV_FX_RP = 6


# UB 地址（Vector 侧；全部 32B 对齐）
VA_Q = 0x00000
VA_K = 0x04000
VA_QN = 0x08000
VA_KN = 0x0C000
VA_QNF = 0x10000
VA_KNF = 0x18000
VA_GC = 0x20000
VA_TF = 0x38000        # [64,64] FP32 T staging
VA_SEEDF = 0x18000     # [64,64] FP32 seed staging（复用 s1 后已死的 VA_KNF 区）
VA_MQKBF = 0x1C000     # [64,64] BF16 mqk 落盘暂存（同上区域，共享 mutex）
VA_MQKF = 0x10000      # [64,64] FP32 mqk 掩码暂存（复用 s1 后已死的 VA_QNF 区）
VA_BETA = 0x3C000      # [64,1] FP32 sigmoid(beta)
VA_DECAY = 0x3C400     # [1,128] FP32
VA_BIAS = 0x3C600      # [1,128] FP32
VA_ALOG = 0x3C800      # [1,128] FP32

MIN_AIC_CORE_COUNT, MAX_AIC_CORE_COUNT = 1, 32


def _current_aic_core_num(device_index):
    props = torch.npu.get_device_properties(device_index)
    core_num = min(int(getattr(props, "cube_core_num", 0)), MAX_AIC_CORE_COUNT)
    if not MIN_AIC_CORE_COUNT <= core_num <= MAX_AIC_CORE_COUNT:
        raise RuntimeError(f"需要 [1,32] 个 AIC，实际为 {core_num}")
    return core_num


# ===== Vector functions =====

@pl.vector_function
def _l2norm_rows_vf(src, dst, n_rows: pl.DT_INT64):
    """L2Norm（行归约两遍法）：FP32 计算，结果写 FP32 staging；pad 行写零。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg)
    for m in pl.range(0, n_rows):
        base = m * D
        acc = vf.full(0.0, preg)
        for r in pl.range(0, VREG):
            reg = vf.load_align(src, base + r * LANES, dist=pl.LoadDist.UNPK_B16)
            val = vf.astype(reg, preg, dtype=pl.DT_FP32)
            sq = vf.mul(val, val, preg)
            part = vf.reduce_sum(sq, preg)
            acc = vf.add(acc, part, preg)
        sum_b = vf.full(acc, preg)
        sum_b = vf.adds(sum_b, L2_EPS, preg)
        norm = vf.sqrt(sum_b, preg)
        inv = vf.div(one, norm, preg)
        for r in pl.range(0, VREG):
            reg = vf.load_align(src, base + r * LANES, dist=pl.LoadDist.UNPK_B16)
            val = vf.astype(reg, preg, dtype=pl.DT_FP32)
            out = vf.mul(val, inv, preg)
            vf.store_align(dst, out, preg, base + r * LANES)
    zero = vf.full(0.0, preg)
    for m in pl.range(n_rows, CHUNK):
        base = m * D
        for r in pl.range(0, VREG):
            vf.store_align(dst, zero, preg, base + r * LANES)
    vf.mem_bar()


@pl.vector_function
def _f32_to_bf16_vf(src_f32, dst_bf16, n_rows: pl.DT_INT64):
    """FP32 → BF16(RINT) 连续落盘（bit-exact 配方）。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pu16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    for m in pl.range(0, n_rows):
        base = m * D
        r_even, r_odd = vf.load_align(src_f32, base, dist=pl.LoadDist.DINTLV_B32)
        even_b = vf.astype(r_even, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                        round_mode=pl.VFRoundMode.CAST_RINT)
        odd_b = vf.astype(r_odd, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ONE,
                        round_mode=pl.VFRoundMode.CAST_RINT)
        merged = vf.xor(even_b, odd_b, pu16)
        vf.store_align(dst_bf16, merged, pu16, base, dist=pl.StoreDist.NORM_B16)
    vf.mem_bar()


@pl.vector_function
def _zero_rows_from_bf16(tile, from_row: pl.DT_INT64):
    """把 [CHUNK, D] BF16 tile 的 [from_row, CHUNK) 行清零（TND 打包序列的
    chunk 无效行——跨序列 token 共享缓冲，pad 行是下一序列的数据，stage2
    收缩前必须清零使乘积项归零）。from_row ≥ CHUNK 时不做任何事。"""
    if from_row < CHUNK:
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
        pu16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
        zero = vf.full(0.0, preg)
        z_b = vf.astype(zero, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                        round_mode=pl.VFRoundMode.CAST_RINT)
        z_o = vf.astype(zero, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ONE,
                        round_mode=pl.VFRoundMode.CAST_RINT)
        zrow = vf.xor(z_b, z_o, pu16)
        for m in pl.range(from_row, CHUNK):
            base = m * D
            for r in pl.range(0, VREG):
                vf.store_align(tile, zrow, pu16, base + r * LANES,
                            dist=pl.StoreDist.NORM_B16)
        vf.mem_bar()


@pl.vector_function
def _zero_narrow_bf16_rows(tile, from_row: pl.DT_INT64, cols: pl.DT_INT64):
    """Clear the tail of a packed ND row tile with a width of at most 64."""
    mask = vf.update_mask(cols, dtype=pl.DT_UINT16)
    zero = vf.full(0, mask, dtype=pl.DT_UINT16)
    for row in pl.range(from_row, CHUNK):
        vf.store_align(tile, zero, mask, row * cols, dist=pl.StoreDist.NORM_B16)
    vf.mem_bar()


@pl.vector_function
def _f32_to_f16_vf(src_f32, dst_f16, n_rows: pl.DT_INT64):
    """FP32 → FP16(RINT) 连续落盘（DINTLV 交织配方，[n,64] tile 版）。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pu16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    for m in pl.range(0, n_rows):
        base = m * CHUNK
        r_even, r_odd = vf.load_align(src_f32, base, dist=pl.LoadDist.DINTLV_B32)
        even_h = vf.astype(r_even, preg, dtype=pl.DT_FP16, layout=pl.CastLayout.ZERO,
                        round_mode=pl.VFRoundMode.CAST_RINT)
        odd_h = vf.astype(r_odd, preg, dtype=pl.DT_FP16, layout=pl.CastLayout.ONE,
                        round_mode=pl.VFRoundMode.CAST_RINT)
        merged = vf.xor(even_h, odd_h, pu16)
        vf.store_align(dst_f16, merged, pu16, base, dist=pl.StoreDist.NORM_B16)
    vf.mem_bar()


@pl.vector_function
def _gate_cumsum_vf(g_raw, bias_tile, alog_tile, gc_tile, head: pl.DT_INT64,
                    lower_bound: pl.DT_FP32, n_rows: pl.DT_INT64):
    """g_act = lb*sigmoid(exp(a_log[h])*(g+dt_bias)) 后行序 inclusive cumsum。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    b0 = vf.load_align(bias_tile, 0)
    b1 = vf.load_align(bias_tile, LANES)
    al = vf.load_align(alog_tile, head, dist=pl.LoadDist.BRC_B32)
    al_exp = vf.exp(al, preg)
    minus_one = vf.full(-1.0, preg)
    neg_exp = vf.mul(al_exp, minus_one, preg)
    lb_b = vf.full(lower_bound, preg)
    one = vf.full(1.0, preg)
    p0 = vf.full(0.0, preg)
    p1 = vf.full(0.0, preg)
    for m in pl.range(0, n_rows):
        base = m * D
        g0 = vf.load_align(g_raw, base, dist=pl.LoadDist.UNPK_B16)
        f0 = vf.astype(g0, preg, dtype=pl.DT_FP32)
        g1 = vf.load_align(g_raw, base + LANES, dist=pl.LoadDist.UNPK_B16)
        f1 = vf.astype(g1, preg, dtype=pl.DT_FP32)
        x0 = vf.add(f0, b0, preg)
        x0 = vf.mul(x0, neg_exp, preg)
        e0 = vf.exp(x0, preg)
        d0 = vf.add(one, e0, preg)
        s0 = vf.div(lb_b, d0, preg)
        x1 = vf.add(f1, b1, preg)
        x1 = vf.mul(x1, neg_exp, preg)
        e1 = vf.exp(x1, preg)
        d1 = vf.add(one, e1, preg)
        s1 = vf.div(lb_b, d1, preg)
        p0 = vf.add(p0, s0, preg)
        p1 = vf.add(p1, s1, preg)
        vf.store_align(gc_tile, p0, preg, base)
        vf.store_align(gc_tile, p1, preg, base + LANES)
    for m in pl.range(n_rows, CHUNK):
        base = m * D
        vf.store_align(gc_tile, p0, preg, base)
        vf.store_align(gc_tile, p1, preg, base + LANES)
    vf.mem_bar()


@pl.vector_function
def _sigmoid_beta_vf(beta_raw, beta_out, n_rows: pl.DT_INT64):
    """beta = sigmoid(beta_raw)；pad lane 写零。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg)
    raw = vf.load_align(beta_raw, 0, dist=pl.LoadDist.UNPK_B16)
    val = vf.astype(raw, preg, dtype=pl.DT_FP32)
    minus_one = vf.full(-1.0, preg)
    neg = vf.mul(val, minus_one, preg)
    e = vf.exp(neg, preg)
    d = vf.add(one, e, preg)
    s = vf.div(one, d, preg)
    zero = vf.full(0.0, preg)
    vmask = vf.update_mask(n_rows, dtype=pl.DT_FP32)
    out = vf.select(s, zero, vmask)
    vf.store_align(beta_out, out, preg, 0)
    vf.mem_bar()


@pl.vector_function
def _emit_kg_only_vf(kn, gc, kg_f32, n_rows: pl.DT_INT64):
    """kg = exp(G_end - G) * K（pad 行写零）。decay 在 s3 段单独计算。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    gend0 = vf.load_align(gc, (CHUNK - 1) * D)
    gend1 = vf.load_align(gc, (CHUNK - 1) * D + LANES)
    for m in pl.range(0, n_rows):
        base = m * D
        g0 = vf.load_align(gc, base)
        r0 = vf.sub(gend0, g0, preg)
        e0 = vf.exp(r0, preg)
        k0 = vf.load_align(kn, base, dist=pl.LoadDist.UNPK_B16)
        kf0 = vf.astype(k0, preg, dtype=pl.DT_FP32)
        o0 = vf.mul(kf0, e0, preg)
        vf.store_align(kg_f32, o0, preg, base)
        g1 = vf.load_align(gc, base + LANES)
        r1 = vf.sub(gend1, g1, preg)
        e1 = vf.exp(r1, preg)
        k1 = vf.load_align(kn, base + LANES, dist=pl.LoadDist.UNPK_B16)
        kf1 = vf.astype(k1, preg, dtype=pl.DT_FP32)
        o1 = vf.mul(kf1, e1, preg)
        vf.store_align(kg_f32, o1, preg, base + LANES)
    zero = vf.full(0.0, preg)
    for m in pl.range(n_rows, CHUNK):
        base = m * D
        vf.store_align(kg_f32, zero, preg, base)
        vf.store_align(kg_f32, zero, preg, base + LANES)
    vf.mem_bar()


@pl.vector_function
def _emit_decay_only_vf(gc, decay_tile, n_rows: pl.DT_INT64):
    """decay = exp(G_last_valid)（读有效末行而非 pad 行 gc[63]——
    pad 延伸行的 store 落地时序不稳，有效行由 loop1 最早写入）。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    last = (n_rows - 1) * D
    gend0 = vf.load_align(gc, last)
    gend1 = vf.load_align(gc, last + LANES)
    d0 = vf.exp(gend0, preg)
    d1 = vf.exp(gend1, preg)
    vf.store_align(decay_tile, d0, preg, 0)
    vf.store_align(decay_tile, d1, preg, LANES)
    vf.mem_bar()


@pl.vector_function
def _emit_qg_vf(qn, gc, qg_f32, scale: pl.DT_FP32, n_rows: pl.DT_INT64):
    """qg = scale * exp(G) * Q（pad 行写零）。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    sc = vf.full(scale, preg)
    for m in pl.range(0, n_rows):
        base = m * D
        g0 = vf.load_align(gc, base)
        e0 = vf.exp(g0, preg)
        q0 = vf.load_align(qn, base, dist=pl.LoadDist.UNPK_B16)
        qf0 = vf.astype(q0, preg, dtype=pl.DT_FP32)
        r0 = vf.mul(qf0, e0, preg)
        r0 = vf.mul(r0, sc, preg)
        vf.store_align(qg_f32, r0, preg, base)
        g1 = vf.load_align(gc, base + LANES)
        e1 = vf.exp(g1, preg)
        q1 = vf.load_align(qn, base + LANES, dist=pl.LoadDist.UNPK_B16)
        qf1 = vf.astype(q1, preg, dtype=pl.DT_FP32)
        r1 = vf.mul(qf1, e1, preg)
        r1 = vf.mul(r1, sc, preg)
        vf.store_align(qg_f32, r1, preg, base + LANES)
    zero = vf.full(0.0, preg)
    for m in pl.range(n_rows, CHUNK):
        base = m * D
        vf.store_align(qg_f32, zero, preg, base)
        vf.store_align(qg_f32, zero, preg, base + LANES)
    vf.mem_bar()


@pl.vector_function
def _emit_left_vf(xn, gc, out_f32, ref_row: pl.DT_INT64, n_rows: pl.DT_INT64,
                scale: pl.DT_FP32):
    """left[r] = scale * exp(G_i - G_ref) * X_i（16 行；K 侧传 1.0，Q 侧传 scale）。
    2× 展开：每迭代处理 2 行，减半 loop 控制开销。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    sc = vf.full(scale, preg)
    gr0 = vf.load_align(gc, ref_row * D)
    gr1 = vf.load_align(gc, ref_row * D + LANES)
    half = n_rows // 2
    for m2 in pl.range(0, half):
        m = m2 * 2
        src0 = (ref_row + m) * D
        src1 = (ref_row + m + 1) * D
        base0 = m * D
        base1 = (m + 1) * D
        g0 = vf.load_align(gc, src0)
        r0 = vf.sub(g0, gr0, preg)
        e0 = vf.exp(r0, preg)
        x0 = vf.load_align(xn, src0, dist=pl.LoadDist.UNPK_B16)
        xf0 = vf.astype(x0, preg, dtype=pl.DT_FP32)
        o0 = vf.mul(xf0, e0, preg)
        o0 = vf.mul(o0, sc, preg)
        vf.store_align(out_f32, o0, preg, base0)
        g1 = vf.load_align(gc, src0 + LANES)
        r1 = vf.sub(g1, gr1, preg)
        e1 = vf.exp(r1, preg)
        x1 = vf.load_align(xn, src0 + LANES, dist=pl.LoadDist.UNPK_B16)
        xf1 = vf.astype(x1, preg, dtype=pl.DT_FP32)
        o1 = vf.mul(xf1, e1, preg)
        o1 = vf.mul(o1, sc, preg)
        vf.store_align(out_f32, o1, preg, base0 + LANES)
        g0b = vf.load_align(gc, src1)
        r0b = vf.sub(g0b, gr0, preg)
        e0b = vf.exp(r0b, preg)
        x0b = vf.load_align(xn, src1, dist=pl.LoadDist.UNPK_B16)
        xf0b = vf.astype(x0b, preg, dtype=pl.DT_FP32)
        o0b = vf.mul(xf0b, e0b, preg)
        o0b = vf.mul(o0b, sc, preg)
        vf.store_align(out_f32, o0b, preg, base1)
        g1b = vf.load_align(gc, src1 + LANES)
        r1b = vf.sub(g1b, gr1, preg)
        e1b = vf.exp(r1b, preg)
        x1b = vf.load_align(xn, src1 + LANES, dist=pl.LoadDist.UNPK_B16)
        xf1b = vf.astype(x1b, preg, dtype=pl.DT_FP32)
        o1b = vf.mul(xf1b, e1b, preg)
        o1b = vf.mul(o1b, sc, preg)
        vf.store_align(out_f32, o1b, preg, base1 + LANES)
    vf.mem_bar()


@pl.vector_function
def _emit_right_vf(kn, gc, out_f32, ref_row: pl.DT_INT64, col_row0: pl.DT_INT64,
                n_rows: pl.DT_INT64):
    """right[r,c] = exp(G_ref(r) - G_j) * K_j（16 行，行偏移 = col_row0 + m）。
    2× 展开：每迭代处理 2 行。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    gr0 = vf.load_align(gc, ref_row * D)
    gr1 = vf.load_align(gc, ref_row * D + LANES)
    half = n_rows // 2
    for m2 in pl.range(0, half):
        m = m2 * 2
        src0 = (col_row0 + m) * D
        src1 = (col_row0 + m + 1) * D
        base0 = m * D
        base1 = (m + 1) * D
        g0 = vf.load_align(gc, src0)
        r0 = vf.sub(gr0, g0, preg)
        e0 = vf.exp(r0, preg)
        k0 = vf.load_align(kn, src0, dist=pl.LoadDist.UNPK_B16)
        kf0 = vf.astype(k0, preg, dtype=pl.DT_FP32)
        o0 = vf.mul(kf0, e0, preg)
        vf.store_align(out_f32, o0, preg, base0)
        g1 = vf.load_align(gc, src0 + LANES)
        r1 = vf.sub(gr1, g1, preg)
        e1 = vf.exp(r1, preg)
        k1 = vf.load_align(kn, src0 + LANES, dist=pl.LoadDist.UNPK_B16)
        kf1 = vf.astype(k1, preg, dtype=pl.DT_FP32)
        o1 = vf.mul(kf1, e1, preg)
        vf.store_align(out_f32, o1, preg, base0 + LANES)
        g0b = vf.load_align(gc, src1)
        r0b = vf.sub(gr0, g0b, preg)
        e0b = vf.exp(r0b, preg)
        k0b = vf.load_align(kn, src1, dist=pl.LoadDist.UNPK_B16)
        kf0b = vf.astype(k0b, preg, dtype=pl.DT_FP32)
        o0b = vf.mul(kf0b, e0b, preg)
        vf.store_align(out_f32, o0b, preg, base1)
        g1b = vf.load_align(gc, src1 + LANES)
        r1b = vf.sub(gr1, g1b, preg)
        e1b = vf.exp(r1b, preg)
        k1b = vf.load_align(kn, src1 + LANES, dist=pl.LoadDist.UNPK_B16)
        kf1b = vf.astype(k1b, preg, dtype=pl.DT_FP32)
        o1b = vf.mul(kf1b, e1b, preg)
        vf.store_align(out_f32, o1b, preg, base1 + LANES)
    vf.mem_bar()


@pl.vector_function
def _sigmoid_slab_vf(slab, beta_out, n_rows: pl.DT_INT64,
                    nv: pl.DT_INT64, head: pl.DT_INT64):
    """flat slab（[64,nv] 行主序）→ seq 连续 sigmoid(β)：逐行 BRC_B16 取
    (m*nv+head) 的原始 β，寄存器内经 update_mask/select 对逐 lane 累积后
    单次固定 offset 0 落 FP32 槽（向量 store 的偏移须为表达式大步长或
    编译期常量，逐 lane 变化的小步长偏移不受支持）；行 ≥ n_rows 的 lane
    保持初值 0——T̂ pad 行所需的清零随之获得。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg)
    minus_one = vf.full(-1.0, preg)
    acc = vf.full(0.0, preg)
    for m in pl.range(0, n_rows):
        raw = vf.load_align(slab, m * nv + head, dist=pl.LoadDist.BRC_B16)
        val = vf.astype(raw, preg, dtype=pl.DT_FP32)
        neg = vf.mul(val, minus_one, preg)
        e = vf.exp(neg, preg)
        d = vf.add(one, e, preg)
        s = vf.div(one, d, preg)
        hi = vf.update_mask(m + 1, dtype=pl.DT_FP32)
        lo = vf.update_mask(m, dtype=pl.DT_FP32)
        tmp = vf.select(s, acc, hi)
        acc = vf.select(acc, tmp, lo)
    vf.store_align(beta_out, acc, preg, 0)
    vf.mem_bar()


@pl.vector_function
def _emit_bv_vf(v_raw, beta_tile, out_f32, n_rows: pl.DT_INT64):
    """βV：有效行 = sigmoid(β_i)·V_i；pad 行写零。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for m in pl.range(0, n_rows):
        base = m * D
        b = vf.load_align(beta_tile, m, dist=pl.LoadDist.BRC_B32)
        v0 = vf.load_align(v_raw, base, dist=pl.LoadDist.UNPK_B16)
        vf0 = vf.astype(v0, preg, dtype=pl.DT_FP32)
        o0 = vf.mul(vf0, b, preg)
        vf.store_align(out_f32, o0, preg, base)
        v1 = vf.load_align(v_raw, base + LANES, dist=pl.LoadDist.UNPK_B16)
        vf1 = vf.astype(v1, preg, dtype=pl.DT_FP32)
        o1 = vf.mul(vf1, b, preg)
        vf.store_align(out_f32, o1, preg, base + LANES)
    zero = vf.full(0.0, preg)
    for m in pl.range(n_rows, CHUNK):
        base = m * D
        vf.store_align(out_f32, zero, preg, base)
        vf.store_align(out_f32, zero, preg, base + LANES)
    vf.mem_bar()


@pl.vector_function
def _emit_bkt_vf(kn, gc, beta_tile, out_f32, n_rows: pl.DT_INT64):
    """βK̃：有效行 = sigmoid(β_i)·exp(G_i)·K̂_i；pad 行写零。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for m in pl.range(0, n_rows):
        base = m * D
        b = vf.load_align(beta_tile, m, dist=pl.LoadDist.BRC_B32)
        g0 = vf.load_align(gc, base)
        e0 = vf.exp(g0, preg)
        k0 = vf.load_align(kn, base, dist=pl.LoadDist.UNPK_B16)
        kf0 = vf.astype(k0, preg, dtype=pl.DT_FP32)
        o0 = vf.mul(kf0, e0, preg)
        o0 = vf.mul(o0, b, preg)
        vf.store_align(out_f32, o0, preg, base)
        g1 = vf.load_align(gc, base + LANES)
        e1 = vf.exp(g1, preg)
        k1 = vf.load_align(kn, base + LANES, dist=pl.LoadDist.UNPK_B16)
        kf1 = vf.astype(k1, preg, dtype=pl.DT_FP32)
        o1 = vf.mul(kf1, e1, preg)
        o1 = vf.mul(o1, b, preg)
        vf.store_align(out_f32, o1, preg, base + LANES)
    zero = vf.full(0.0, preg)
    for m in pl.range(n_rows, CHUNK):
        base = m * D
        vf.store_align(out_f32, zero, preg, base)
        vf.store_align(out_f32, zero, preg, base + LANES)
    vf.mem_bar()


@pl.vector_function
def _build_t_seed_vf(kk_src, beta_tile, t_f32, seed_f32):
    """T̂ = -tril_strict(β_i·kk)；seed = I + T̂（对角 +1，双 select 法）。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg)
    zero = vf.full(0.0, preg)
    minus_one = vf.full(-1.0, preg)
    for i in pl.range(0, CHUNK):
        base = i * CHUNK
        kk = vf.load_align(kk_src, base)
        b = vf.load_align(beta_tile, i, dist=pl.LoadDist.BRC_B32)
        tb = vf.mul(kk, b, preg)
        keep = vf.update_mask(i, dtype=pl.DT_FP32)
        t_row = vf.select(tb, zero, keep)
        t_row = vf.mul(t_row, minus_one, preg)
        vf.store_align(t_f32, t_row, preg, base)
        inc = vf.add(t_row, one, preg)
        thr = vf.update_mask(i + 1, dtype=pl.DT_FP32)
        c1 = vf.select(inc, t_row, thr)
        seed_row = vf.select(t_row, c1, keep)
        vf.store_align(seed_f32, seed_row, preg, base)
    vf.mem_bar()


@pl.vector_function
def _emit_mqk_vf(mqk_src, mqk_f32):
    """mqk = tril 含对角掩码（[64,64] FP32，行 i 保留 lane<=i）。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, preg)
    for i in pl.range(0, CHUNK):
        base = i * CHUNK
        row = vf.load_align(mqk_src, base)
        keep = vf.update_mask(i + 1, dtype=pl.DT_FP32)
        out = vf.select(row, zero, keep)
        vf.store_align(mqk_f32, out, preg, base)
    vf.mem_bar()


@pl.vector_function
def _f32_to_bf16_64_vf(src_f32, dst_bf16):
    """[64,64] FP32 → BF16(RINT)：双行并置后走 128 宽配方。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pu16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    for m in pl.range(0, CHUNK, 2):
        base = m * CHUNK
        r_even, r_odd = vf.load_align(src_f32, base, dist=pl.LoadDist.DINTLV_B32)
        even_b = vf.astype(r_even, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO,
                        round_mode=pl.VFRoundMode.CAST_RINT)
        odd_b = vf.astype(r_odd, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ONE,
                        round_mode=pl.VFRoundMode.CAST_RINT)
        merged = vf.xor(even_b, odd_b, pu16)
        vf.store_align(dst_bf16, merged, pu16, base, dist=pl.StoreDist.NORM_B16)
    vf.mem_bar()


# ===== 布局相关的装载/存储小助手（layout 为编译期常量） =====


def _ld_seq(tile, tensor, sb, head, row0, layout):
    if layout == LAYOUT_BNSD:
        pl.load(tile, tensor, [sb, head, row0, 0])
    elif layout == LAYOUT_BSND:
        pl.load(tile, tensor, [sb, row0, head, 0], order=[1, 3])
    else:
        pl.load(tile, tensor, [row0, head, 0], order=[0, 2])


def _ld_beta_flat(slab, beta_ptr_, sb, row0, valid_rows, nv, layout, seq_len):
    """β flat slab 装载（NvFlat 路径）：chunk 的 [valid_rows, nv] β 区在物理
    布局中行主序连续（BSND [b,s,h] / TND [T,h]），经 addptr 定位后 [1,2048]
    一次 ND 直读。N 上取整到 16 保 C0 对齐——越界读落在分配粒度内且不消费。"""
    ne = valid_rows * nv
    ne = (ne + 15) // 16 * 16
    pl.set_validshape(slab, [1, ne])
    if layout == LAYOUT_TND:
        bt = pl.make_tensor(pl.addptr(beta_ptr_, row0 * nv), [1, 2048],
                            dtype=pl.DT_BF16)
    else:
        bt = pl.make_tensor(
            pl.addptr(beta_ptr_, sb * seq_len * nv + row0 * nv), [1, 2048],
            dtype=pl.DT_BF16)
    pl.load(slab, bt, [0, 0])


def _ld_beta_w32(win, beta_ptr_, sb, row0, valid_rows, nv, head, layout):
    """β 32 头跨步窗口装载（NvFlat=2，nv>32）：[64,32] tile——
    dim0=64 token（跨步 nv）、dim1=32 头窗口（连续 64B，满足 MTE2 的
    32B 突发要求），与 q 的 [valid,C] 载入同几何。窗口起点 32 头对齐；
    末窗越界读落在分配粒度内且不被消费。返回 col=head-wstart 与
    窗口 stride=32。"""
    wstart = (head // 32) * 32
    col = head - wstart
    vr = valid_rows
    if vr > 64:
        vr = 64
    pl.set_validshape(win, [vr, 32])
    if layout == LAYOUT_TND:
        bt = pl.make_tensor(beta_ptr_, [CHUNK, nv], stride=[nv, 1],
                            dtype=pl.DT_BF16)
        pl.load(win, bt, [row0, wstart], order=[0, 1])
    else:
        bt = pl.make_tensor(beta_ptr_, [1, CHUNK, nv], stride=[1, nv, 1],
                            dtype=pl.DT_BF16)
        pl.load(win, bt, [sb, row0, wstart], order=[1, 2])
    return col


def _store_sub_fractal(tensor, subgrp, sb, head, row0, valid_rows, m, n,
                        layout, scale):
    """单分形 [16,16] 收窄子存（尾块精确行落盘的最小合规单元）。"""
    vr = valid_rows - 16 * m
    if vr > 16:
        vr = 16
    if vr > 0:
        st = subgrp[m * 8 + n]
        pl.set_validshape(st, [vr, 16])
        if layout == LAYOUT_TND:
            pl.store(tensor, st, [row0 + 16 * m, head, 16 * n],
                    order=[0, 2], scale=scale)
        else:
            pl.store(tensor, st, [sb, head, row0 + 16 * m, 16 * n],
                    order=[2, 3], scale=scale)


def _store_uw_subs(tensor, subgrp, lc, sb, head, row0, valid_rows, layout, scale):
    """u/w 精确行落盘双路径。full chunk：单次全量 [CHUNK,D] store（免
    32 次子 store 的 fixpipe 描述符开销）。尾块：32×[16,16] 单分形
    收窄——多 N 分形直写须覆盖完整 M 轴，单分形为唯一合规形态。"""
    if valid_rows == CHUNK:
        pl.set_validshape(lc, [CHUNK, D])
        if layout == LAYOUT_TND:
            pl.store(tensor, lc, [row0, head, 0], order=[0, 2], scale=scale)
        else:
            pl.store(tensor, lc, [sb, head, row0, 0], scale=scale)
    else:
        for m in pl.range(0, 4):
            for n in pl.range(0, 8):
                _store_sub_fractal(tensor, subgrp, sb, head, row0, valid_rows,
                                    m, n, layout, scale)


def _ld_beta(tile_row, tile_col, tensor, sb, head, row0, layout):
    # β 按 seq 连续加载（BNSD 原生连续；[x,1] tile 的 order 步进加载按
    # flat 读，不受支持——host 侧已保证布局）
    if layout == LAYOUT_TND:
        pl.load(tile_row, tensor, [head, row0], order=[0, 1])
    else:
        pl.load(tile_row, tensor, [sb, head, row0], order=[1, 2])


def _st_seq(tensor, tile, sb, head, row0, valid_rows, layout):
    pl.set_validshape(tile, [valid_rows, D])
    if layout == LAYOUT_TND:
        pl.store(tensor, tile, [row0, head, 0], order=[0, 2])
    else:
        pl.store(tensor, tile, [sb, head, row0, 0])


@pl.vector_function
def _diag_add_vf(phi_ub, d_ub):
    """phi[i,i] += d[i]（select 构行 + 整行对齐读改写——无未对齐标量写）"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    d0 = vf.load_align(d_ub, 0)
    d1 = vf.load_align(d_ub, LANES)
    zero = vf.full(0.0, preg)
    for i in pl.range(0, LANES):
        base = i * D
        hi = vf.update_mask(i + 1, dtype=pl.DT_FP32)
        lo = vf.update_mask(i, dtype=pl.DT_FP32)
        a0 = vf.select(d0, zero, hi)
        r0 = vf.select(zero, a0, lo)
        p0 = vf.load_align(phi_ub, base)
        s0 = vf.add(p0, r0, preg)
        vf.store_align(phi_ub, s0, preg, base)
    for i in pl.range(LANES, D):
        base = i * D
        j = i - LANES
        hi = vf.update_mask(j + 1, dtype=pl.DT_FP32)
        lo = vf.update_mask(j, dtype=pl.DT_FP32)
        a1 = vf.select(d1, zero, hi)
        r1 = vf.select(zero, a1, lo)
        p1 = vf.load_align(phi_ub, base + LANES)
        s1 = vf.add(p1, r1, preg)
        vf.store_align(phi_ub, s1, preg, base + LANES)
    vf.mem_bar()


def _st_decay(tensor, tile, sb, head, chunk):
    pl.store(tensor, tile, [sb, head, chunk, 0])


def _right_pair_index(r, c):
    """(r,c) → 右操作数区序号：r(r+1)/2 + c（含对角块，共 10 对）。"""
    return r * (r + 1) // 2 + c


def _ws_view_bf16(ws16_ptr, task_slot, elem_off, rows):
    """ws 操作数区的逐片 rank-2 视图（BF16 指针——2B 元素步进）。"""
    return pl.make_tensor(
        pl.addptr(ws16_ptr, task_slot * WS_OPS_ELEMS + elem_off),
        [rows, D], dtype=pl.DT_BF16)


def _task_ctx(task_start, prefix_start, group_count, group_start,
                chunks_total, tail_rows, token_start, i):
    """任务序言公共坐标推导（chunk/head/valid_rows/row0/task_slot）。

    跨 section 变量不可见，各段自含坐标统一走本推导。
    """
    local_task = task_start + i - prefix_start
    local_head = local_task // group_count
    group_chunk = local_task - local_head * group_count
    chunk = group_start + group_chunk
    valid_rows = CHUNK
    if chunk + 1 == chunks_total:
        valid_rows = tail_rows
    row0 = token_start + chunk * CHUNK
    task_slot = task_start + i
    return chunk, local_head, valid_rows, row0, task_slot


# ===== stage2 体级 helper（m2/ctx 元组；契约即各 helper 解包行）=====


def _summary_task_ctx(schedule, s2_task, s2_dv, nv):
    """解码局部路径与共享 φ 路径使用的 Stage 2 任务。"""
    (metadata, batch_base, storage_base, token_base, length_base, chunk_start_base, chunk_num_base,
     s2_start, s2_end) = schedule
    s2_chain = s2_task // s2_dv
    s2_split = s2_task - s2_chain * s2_dv
    s2_row_off = s2_split * (D // s2_dv)
    s2_rows = D // s2_dv
    s2_batch = s2_chain // nv
    s2_head = s2_chain - s2_batch * nv
    s2_storage = pl.getval(metadata, storage_base + s2_batch)
    s2_logical = pl.getval(metadata, batch_base + s2_batch)
    s2_token = pl.getval(metadata, token_base + s2_batch)
    s2_len = pl.getval(metadata, length_base + s2_batch)
    s2_cstart = pl.getval(metadata, chunk_start_base + s2_batch)
    s2_cnum = pl.getval(metadata, chunk_num_base + s2_batch)
    s2_last_valid = s2_len - (s2_cstart + s2_cnum - 1) * CHUNK
    if s2_last_valid > CHUNK:
        s2_last_valid = CHUNK
    return pl.make_tuple(
        s2_row_off=s2_row_off, s2_rows=s2_rows, s2_head=s2_head, s2_storage=s2_storage,
        s2_logical=s2_logical, s2_token=s2_token, s2_len=s2_len, s2_cstart=s2_cstart,
        s2_cnum=s2_cnum, s2_task=s2_task, s2_start=s2_start, s2_end=s2_end,
        s2_last_valid=s2_last_valid, s2_zword=s2_task * 128,
    )


def _load_first_kg(kg2_l1g, kg_t, valid_rows, layout, token, chunk, head, storage):
    kg = kg2_l1g[0]
    pl.set_validshape(kg, [valid_rows, D])
    if layout == LAYOUT_TND:
        pl.load(kg, kg_t, [token + chunk * CHUNK, head, 0], order=[0, 2])
    else:
        pl.load(kg, kg_t, [storage, head, token + chunk * CHUNK, 0], order=[2, 3])


def _st2_init_cube(m2, ctx, kg_t, trans_t, state_t, id128_view,
                   row_off, rows, w_t, u_t):
    """stage2 链首：R/ψ 接力装载（本 split 的 D 行片）+ 链首 kg 预取。"""
    st_r2g = m2[0]
    st_psi2g = m2[1]
    kg2_l1g = m2[2]
    s2_cstart = ctx[0]
    s2_logical = ctx[1]
    s2_head = ctx[2]
    s2_len = ctx[3]
    s2_storage = ctx[4]
    s2_token = ctx[5]
    layout = ctx[6]
    stra = st_r2g.current()
    pl.set_validshape(stra, [rows, D])
    if s2_cstart > 0:
        pl.load(stra, trans_t,
                [s2_logical, s2_head, row_off, 0])
    else:
        if ctx[7] == 0:
            pl.load(stra, id128_view, [row_off, 0])
    spsi = st_psi2g.current()
    pl.set_validshape(spsi, [rows, D])
    if s2_cstart > 0:
        pl.load(spsi, state_t,
                [s2_logical, s2_head, row_off, 0])
    s2_valid0 = s2_len - s2_cstart * CHUNK
    if s2_valid0 > CHUNK:
        s2_valid0 = CHUNK
    _load_first_kg(kg2_l1g, kg_t, s2_valid0, layout, s2_token, s2_cstart, s2_head, s2_storage)

    if ctx[8] == 0:
        row0 = s2_token + s2_cstart * CHUNK
        wt = m2[5].current()
        ut = m2[6][0]
        pl.set_validshape(wt, [D, s2_valid0])
        pl.set_validshape(ut, [rows, s2_valid0])
        if layout == LAYOUT_TND:
            pl.load(wt, w_t, [row0, s2_head, 0], order=[2, 0])
            pl.load(ut, u_t, [row0, s2_head, row_off], order=[2, 0])
        else:
            pl.load(wt, w_t, [s2_storage, s2_head, row0, 0], order=[3, 2])
            pl.load(ut, u_t, [s2_storage, s2_head, row0, row_off], order=[3, 2])


def _wu_load_chain(m2, view, row0, storage, head, valid, layout, dst):
    """stage2 单张量 w/u 装载链：GM 加载→pad 清零→UB NZ→L1 转置（四路共用）。"""
    wu_ub2g = m2[3]
    wu_nz2g = m2[4]
    wu = wu_ub2g.current()
    pl.set_validshape(wu, [valid, D])
    if layout == LAYOUT_TND:
        pl.load(wu, view,
                [row0, head, 0],
                order=[0, 2])
    else:
        pl.load(wu, view,
                [storage, head, row0, 0],
                order=[2, 3])
    _zero_rows_from_bf16(wu, valid)
    pl.set_validshape(wu, [CHUNK, D])
    wnz = wu_nz2g.current()
    pl.set_validshape(wnz, [CHUNK, D])
    pl.move(wnz, wu)
    pl.set_validshape(dst, [D, CHUNK])
    pl.move(dst, wnz)


def _u_load_chain(u_pair, view, row0, storage, head, valid, layout, dst,
                  col_off, cols):
    """stage2 单张量 u 装载链（窄列版）：只装本 split 的 D 行片。

    u 的行片 = u 的列块 [col_off, col_off+cols)：按列偏移装载 →
    NZ 中转 → 写入本 split 的 [cols, CHUNK] dst。
    """
    u_ub2g = u_pair[0]
    u_nz2g = u_pair[1]
    wu = u_ub2g.current()
    pl.set_validshape(wu, [valid, cols])
    if layout == LAYOUT_TND:
        pl.load(wu, view,
                [row0, head, col_off],
                order=[0, 2])
    else:
        pl.load(wu, view,
                [storage, head, row0, col_off],
                order=[2, 3])
    pl.set_validshape(wu, [CHUNK, cols])
    if valid < CHUNK:
        if cols == D:
            _zero_rows_from_bf16(wu, valid)
        else:
            _zero_narrow_bf16_rows(wu, valid, cols)
    wnz = u_nz2g.current()
    pl.set_validshape(wnz, [CHUNK, cols])
    pl.move(wnz, wu)
    pl.set_validshape(dst, [cols, CHUNK])
    pl.move(dst, wnz)


def _st2_init_vec(m2, ctx, w_t, u_t, u_pair, row_off, rows):
    """stage2 链首：chunk-0 的 w/u 转置预置（vec sub0 段体）。"""
    wu_ub2g = m2[3]
    wu_nz2g = m2[4]
    wt2_l1g = m2[5]
    ut2_l1g = m2[6]
    s2_task = ctx[0]
    s2_start = ctx[1]
    s2_storage = ctx[2]
    s2_logical = ctx[3]
    s2_head = ctx[4]
    s2_token = ctx[5]
    s2_cstart = ctx[6]
    s2_len = ctx[7]
    layout = ctx[8]
    # 句柄预备在 wait 前（运行期 if 内首位 wait 会触发块提前闭合）
    wu_i = wu_ub2g.current()
    wnz_i = wu_nz2g.current()
    wtl_i = wt2_l1g.current()
    utl_i = ut2_l1g[0]
    # 链边界：前链末 rank 的读取与本 init 覆写跨 subblock 无序，显式等前链完成
    if s2_task > s2_start:
        pl.system.wait_cross_core(
            pipe=pl.PipeType.MTE3,
            event_id=EV_CHAIN_DONE,
            sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    s2_r0init = s2_token + s2_cstart * CHUNK
    if ctx[9] == 1:
        # 加载按 valid 收窄，未载 pad 行统一清零
        s2_valid0 = s2_len - s2_cstart * CHUNK
        if s2_valid0 > CHUNK:
            s2_valid0 = CHUNK
        dst_w = wt2_l1g.current()
        dst_u = ut2_l1g[0]
        _wu_load_chain(m2, w_t, s2_r0init, s2_storage, s2_head,
                       s2_valid0, layout, dst_w)
        _u_load_chain(u_pair, u_t, s2_r0init, s2_storage, s2_head,
                      s2_valid0, layout, dst_u, row_off, rows)
        pl.system.set_cross_core(
            pipe=pl.PipeType.MTE3, event_id=EV_READY_WU,
            sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)


def _st2_vecwork(m2, ctx, decay_t, trans_t):
    """stage2：φ(r-1) 对角加 + NZ/L1 转移（vecwork 段体）。"""
    ub_decayg = m2[17]
    phi_ub2g = m2[7]
    phi_nz2g = m2[8]
    phi2_l1g = m2[9]
    s2_cstart = ctx[0]
    s2_r = ctx[1]
    s2_logical = ctx[2]
    s2_head = ctx[3]
    s2_chunk_v = s2_cstart + s2_r - 1
    dcur = ub_decayg.current()
    pl.set_validshape(dcur, [1, D])
    pl.load(dcur, decay_t,
            [s2_logical, s2_head, s2_chunk_v, 0])
    pl.system.wait_cross_core(
        pipe=pl.PipeType.V, event_id=EV_READY_OP,
        sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    _diag_add_vf(phi_ub2g[(s2_r - 1) % 2], dcur)
    pl.system.wait_cross_core(
        pipe=pl.PipeType.MTE3, event_id=EV_READY_KM,
        sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    if ctx[4] and ctx[5] == 1 and s2_r == 1:
        pl.store(trans_t, phi_ub2g[0], [s2_logical, s2_head, 0, 0])
    else:
        phnz = phi_nz2g.current()
        pl.set_validshape(phnz, [D, D])
        pl.move(phnz, phi_ub2g[(s2_r - 1) % 2])
        if ctx[4] and s2_r == 3:
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE3, event_id=EV_FIRST_R_FREE,
                sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
        phl = phi2_l1g[(s2_r - 1) % 2]
        pl.set_validshape(phl, [D, D])
        pl.move(phl, phnz)
    pl.system.set_cross_core(
        pipe=pl.PipeType.MTE3, event_id=EV_READY_TS,
        sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)


def _st2_do_vec_select(ctx, direct_tnd=0):
    """短尾与未开启 TND 直读的分片走 Vector 转置。"""
    s2_r = ctx[0]
    s2_cnum = ctx[1]
    s2_last_valid = ctx[2]
    layout = ctx[3]
    s2_do_vec = 0
    if s2_r + 1 < s2_cnum:
        s2_r1_is_tail = 0
        if s2_r + 2 == s2_cnum:
            if s2_last_valid < CHUNK and (layout != LAYOUT_TND or direct_tnd == 0):
                s2_r1_is_tail = 1
        if layout == LAYOUT_TND:
            if direct_tnd == 0:
                s2_r1_is_tail = 1
        if s2_r1_is_tail == 1:
            s2_do_vec = 1
    return s2_do_vec


def _st2_transpose_body(m2, ctx, w_t, u_t, u_pair, row_off, rows):
    """stage2：chunk r+1 的 w/u 转置链（vec 段体——尾块/TND 路径）。"""
    wu_ub2g = m2[3]
    wu_nz2g = m2[4]
    wt2_l1g = m2[5]
    ut2_l1g = m2[6]
    s2_r = ctx[0]
    s2_cstart = ctx[1]
    s2_len = ctx[2]
    s2_storage = ctx[3]
    s2_head = ctx[4]
    s2_token = ctx[5]
    layout = ctx[6]
    # 等 rank(r-1) 读完 ut2 slot 后再覆写
    if s2_r >= 1:
        pl.system.wait_cross_core(
            pipe=pl.PipeType.MTE3, event_id=EV_READY_UF,
            sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    # 等 φ(r) 读毕 wt2（单槽）
    pl.system.wait_cross_core(
        pipe=pl.PipeType.MTE3, event_id=EV_WT_REL,
        sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    s2_row0_t = s2_token + (s2_cstart + s2_r + 1) * CHUNK
    # 加载按 valid 收窄，未载 pad 行统一清零
    s2_validt = s2_len - (s2_cstart + s2_r + 1) * CHUNK
    if s2_validt > CHUNK:
        s2_validt = CHUNK
    if s2_validt < 0:
        s2_validt = 0
    dst_w = wt2_l1g.current()
    dst_u = ut2_l1g[(s2_r + 1) % 2]
    _wu_load_chain(m2, w_t, s2_row0_t, s2_storage, s2_head,
                   s2_validt, layout, dst_w)
    _u_load_chain(u_pair, u_t, s2_row0_t, s2_storage, s2_head,
                  s2_validt, layout, dst_u, row_off, rows)
    pl.system.set_cross_core(
        pipe=pl.PipeType.MTE3, event_id=EV_READY_WU,
        sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)


def _st2_transpose(m2, sel_ctx, body_ctx, w_t, u_t, u_pair, row_off, rows):
    """stage2：尾块路径选择 + chunk r+1 的 w/u 转置链。"""
    if _st2_do_vec_select(sel_ctx, sel_ctx[4]) == 1:
        _st2_transpose_body(m2, body_ctx, w_t, u_t, u_pair, row_off, rows)


def _st2_phi(m2, ctx):
    """stage2：φ(r) 的 wᵀ@kg matmul + FIX 下发（WU 判定见 _st2_wu_wait）。"""
    wt2_l1g = m2[5]
    kg2_l1g = m2[2]
    l0a_pb2g = m2[10]
    l0b_pb2g = m2[11]
    l0c_phi2g = m2[12]
    phi_ub2g = m2[7]
    s2_r = ctx[0]
    s2_cnum = ctx[1]
    s2_last_valid = ctx[2]
    _st2_wu_wait(ctx)
    wta = wt2_l1g.current()
    kga = kg2_l1g[s2_r % 2]
    la = l0a_pb2g.current()
    k_valid = CHUNK
    if s2_r + 1 == s2_cnum:
        if s2_last_valid < CHUNK:
            k_valid = s2_last_valid
    pl.set_validshape(la, [D, k_valid])
    pl.move(la, wta)
    lb = l0b_pb2g.current()
    pl.set_validshape(lb, [k_valid, D])
    pl.move(lb, kga)
    pl.system.bar_mte1()
    lcp = l0c_phi2g.current()
    pl.set_validshape(lcp, [D, D])
    pl.matmul(lcp, la, lb)
    pl.system.bar_m()
    pl.system.sync_src(set_pipe=pl.PipeType.M,
                    wait_pipe=pl.PipeType.FIX, event_id=EV_FX_M0)
    pl.system.sync_dst(set_pipe=pl.PipeType.M,
                    wait_pipe=pl.PipeType.FIX, event_id=EV_FX_M0)
    ph = phi_ub2g[s2_r % 2]
    pl.set_validshape(ph, [D, D])
    pl.move(ph, lcp,
            acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
    pl.system.set_cross_core(
        pipe=pl.PipeType.FIX, event_id=EV_READY_OP,
        sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    pl.system.set_cross_core(
        pipe=pl.PipeType.FIX, event_id=EV_READY_KM,
        sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    _st2_release_wt(ctx)


def _st2_release_wt(ctx):
    s2_r, s2_cnum, s2_last_valid, layout, direct_tnd, _ = ctx
    # φ(r) 读毕 wt2 → 同轮 vec 的 transpose(r+1) 消费。
    # 仅当 ch(r+1) 走 vec 转置（transpose 会 wait）才 set——直读轮不 set，
    # 否则计数残留跨任务泄漏，放行后继链的提前覆写
    if s2_r + 1 < s2_cnum:
        s2_r1_vec = 0
        if s2_r + 2 == s2_cnum:
            if s2_last_valid < CHUNK and (layout != LAYOUT_TND or direct_tnd == 0):
                s2_r1_vec = 1
        if layout == LAYOUT_TND:
            if direct_tnd == 0:
                s2_r1_vec = 1
        if s2_r1_vec == 1:
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE1, event_id=EV_WT_REL,
                sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)


def _st2_wu_wait(ctx):
    """φ(r) 的 WU 等待判定：仅 vec 生产的 chunk 需要（与 transpose 的
    set 配平；full chunk 同核 auto_mutex 排序，无 set）。
    """
    s2_r = ctx[0]
    s2_cnum = ctx[1]
    s2_last_valid = ctx[2]
    layout = ctx[3]
    s2_wait_wu = 0
    if s2_r == 0:
        s2_wait_wu = ctx[5]
    else:
        if s2_r + 1 == s2_cnum:
            if s2_last_valid < CHUNK and (layout != LAYOUT_TND or ctx[4] == 0):
                s2_wait_wu = 1
    if layout == LAYOUT_TND:
        if ctx[4] == 0:
            s2_wait_wu = 1
    if s2_wait_wu == 1:
        pl.system.wait_cross_core(
            pipe=pl.PipeType.MTE1, event_id=EV_READY_WU,
            sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)


def _st2_rank(m2, ctx, trans_t, state_t):
    """stage2：以 FP32 φ 递推 R/ψ，并累加 BF16 操作数的 uᵀ@kg。

    la/lb 独立名显式重取（跨兄弟 if 同名赋值经 SSA 合并产生悬空引用）。
    """
    l0b_phi2g = m2[13]
    l0a_r2g = m2[14]
    l0c_r2g = m2[15]
    l0c_psi2g = m2[16]
    st_r2g = m2[0]
    st_psi2g = m2[1]
    l0a_pb2g = m2[10]
    l0b_pb2g = m2[11]
    phi2_l1g = m2[9]
    ut2_l1g = m2[6]
    kg2_l1g = m2[2]
    s2_r = ctx[0]
    s2_cnum = ctx[1]
    s2_logical = ctx[2]
    s2_head = ctx[3]
    s2_task = ctx[4]
    s2_end = ctx[5]
    s2_row_off = ctx[6]
    s2_rows = ctx[7]
    pl.system.wait_cross_core(
        pipe=pl.PipeType.MTE1, event_id=EV_READY_TS,
        sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    lb2 = l0b_phi2g.current()
    pl.set_validshape(lb2, [D, D])
    la2 = l0a_r2g.current()
    lcr = l0c_r2g.current()
    pl.set_validshape(lcr, [s2_rows, D])
    if ctx[11] == 0 or s2_r != 1:
        pl.move(lb2, phi2_l1g[(s2_r - 1) % 2])
        pl.set_validshape(la2, [s2_rows, D])
        if ctx[11] and s2_r == 2:
            pl.move(la2, phi2_l1g[0])
            if s2_cnum >= 3:
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE1, event_id=EV_FIRST_R_FREE,
                    sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
        else:
            pl.move(la2, st_r2g.current(), offset=[0, 0])
        pl.system.bar_mte1()
        pl.matmul(lcr, la2, lb2)
    # lcr 的完成性保证：滚动轮由 commit 的 M→FIX 对覆盖、末轮由
    # commit 末轮的 bar_m 覆盖——此处不再排空 M 管线
    _st2_rank_psi(m2, ctx, trans_t, state_t, la2, lb2, lcr)


def _st2_rank_psi(m2, ctx, trans_t, state_t, la2, lb2, lcr):
    """stage2：FP32 递推 ψ，累加 uᵀ@kg 后提交结果。"""
    l0c_psi2g = m2[16]
    st_psi2g = m2[1]
    l0a_ur2g = m2[18]
    l0b_pb2g = m2[11]
    ut2_l1g = m2[6]
    kg2_l1g = m2[2]
    s2_r = ctx[0]
    s2_rows = ctx[7]
    s2_cnum = ctx[1]
    s2_last_valid = ctx[8]
    first = 0
    if ctx[10] == 0:
        if s2_r == 1:
            first = 1
    lcp2 = l0c_psi2g.current()
    pl.set_validshape(lcp2, [s2_rows, D])
    if first == 0:
        pl.set_validshape(la2, [s2_rows, D])
        pl.move(la2, st_psi2g.current(), offset=[0, 0])
        pl.system.bar_mte1()
        pl.matmul(lcp2, la2, lb2)
        pl.system.bar_m()
    # la/lb 独立名显式重取（兄弟 if 同名赋值经 SSA 合并产生悬空引用）
    la_r = l0a_ur2g.current()
    k_valid = CHUNK
    if s2_r == s2_cnum:
        if s2_last_valid < CHUNK:
            k_valid = s2_last_valid
    pl.set_validshape(la_r, [s2_rows, k_valid])
    pl.move(la_r, ut2_l1g[(s2_r - 1) % 2])
    lb_r = l0b_pb2g.current()
    pl.set_validshape(lb_r, [k_valid, D])
    pl.move(lb_r, kg2_l1g[(s2_r - 1) % 2])
    pl.system.bar_mte1()
    if first == 1:
        pl.matmul(lcp2, la_r, lb_r)
    else:
        pl.matmul_acc(lcp2, lcp2, la_r, lb_r)
    # lcp2 完成性：滚动轮由 M→FIX 对覆盖、末轮由 commit 的 bar_m 覆盖
    _st2_rank_commit(m2, ctx, trans_t, state_t, lcr, lcp2)


def _st2_rank_commit(m2, ctx, trans_t, state_t, lcr, lcp2):
    """stage2：rank 结果的落盘/滚动收尾（UF 释放 + 末轮 store）。"""
    st_r2g = m2[0]
    st_psi2g = m2[1]
    s2_r = ctx[0]
    s2_cnum = ctx[1]
    s2_logical = ctx[2]
    s2_head = ctx[3]
    s2_task = ctx[4]
    s2_end = ctx[5]
    s2_last_valid = ctx[8]
    layout = ctx[9]
    # rank 读毕 ut2 槽——由同轮 vec 的 transpose(r+1)（同槽）消费。
    # 仅当 ch(r+1) 走 vec 转置（transpose 才 wait UF）才 set——直读轮
    # 不 set，防计数残留跨任务泄漏
    if s2_r + 1 < s2_cnum:
        s2_r1_vec = 0
        if s2_r + 2 == s2_cnum:
            if s2_last_valid < CHUNK and (layout != LAYOUT_TND or ctx[12] == 0):
                s2_r1_vec = 1
        if layout == LAYOUT_TND:
            if ctx[12] == 0:
                s2_r1_vec = 1
        if s2_r1_vec == 1:
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE1, event_id=EV_READY_UF,
                sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    s2_row_off = ctx[6]
    if s2_r == s2_cnum:
        pl.system.bar_m()
        if ctx[11] == 0 or s2_r != 1:
            pl.store(trans_t, lcr,
                    [s2_logical, s2_head, s2_row_off, 0])
        pl.store(state_t, lcp2,
                [s2_logical, s2_head, s2_row_off, 0])
        # 链边界 flag：后继链 init 覆写须待本链完成（末链不 set 保持计数平衡）
        if s2_task + 1 < s2_end:
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE1,
                event_id=EV_CHAIN_DONE,
                sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    else:
        pl.system.sync_src(set_pipe=pl.PipeType.M,
                        wait_pipe=pl.PipeType.FIX, event_id=EV_FX_R1)
        pl.system.sync_dst(set_pipe=pl.PipeType.M,
                        wait_pipe=pl.PipeType.FIX, event_id=EV_FX_R1)
        if ctx[11] == 0 or s2_r != 1:
            pl.insert(st_r2g.current(), lcr, [0, 0])
        pl.insert(st_psi2g.current(), lcp2, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.FIX,
                        wait_pipe=pl.PipeType.MTE1, event_id=EV_FX_R1)
        pl.system.sync_dst(set_pipe=pl.PipeType.FIX,
                        wait_pipe=pl.PipeType.MTE1, event_id=EV_FX_R1)


def _st2_prefetch(m2, ctx, w_t, u_t, kg_t, row_off, rows):
    """Cube 预取 kg，满块走 GM 转置直读时同时预取 w/u。"""
    wt2_l1g = m2[5]
    ut2_l1g = m2[6]
    kg2_l1g = m2[2]
    s2_r = ctx[0]
    s2_cnum = ctx[1]
    s2_last_valid = ctx[2]
    s2_storage = ctx[3]
    s2_head = ctx[4]
    s2_token = ctx[5]
    s2_cstart = ctx[6]
    s2_len = ctx[7]
    layout = ctx[8]
    # （跨 section 变量不可见——cube 段重算 is_tail）
    s2_r1_is_tail = 0
    if s2_r + 2 == s2_cnum:
        if s2_last_valid < CHUNK:
            s2_r1_is_tail = 1
    if layout == LAYOUT_TND:
        if ctx[11] == 1:
            valid = CHUNK
            if s2_r1_is_tail == 1:
                valid = s2_last_valid
            pf_row0 = s2_token + (s2_cstart + s2_r + 1) * CHUNK
            wtd = wt2_l1g.current()
            pl.set_validshape(wtd, [D, valid])
            pl.load(wtd, w_t, [pf_row0, s2_head, 0], order=[2, 0])
            utd = ut2_l1g[(s2_r + 1) % 2]
            pl.set_validshape(utd, [rows, valid])
            pl.load(utd, u_t, [pf_row0, s2_head, row_off], order=[2, 0])
    else:
        if s2_r1_is_tail == 0:
            pf_row0 = s2_token + (s2_cstart + s2_r + 1) * CHUNK
            wtp = wt2_l1g.current()
            pl.set_validshape(wtp, [D, CHUNK])
            pl.load(wtp, w_t,
                    [s2_storage, s2_head, pf_row0, 0],
                    order=[3, 2])
            utp = ut2_l1g[(s2_r + 1) % 2]
            pl.set_validshape(utp, [rows, CHUNK])
            pl.load(utp, u_t,
                    [s2_storage, s2_head, pf_row0, row_off],
                    order=[3, 2])
    _st2_prefetch_kg(m2, ctx, kg_t)


def _st2_prefetch_kg(m2, ctx, kg_t):
    """stage2：chunk r+1 的 kg 直读预取（M 收窄防越界）。"""
    kg2_l1g = m2[2]
    s2_r = ctx[0]
    s2_cstart = ctx[6]
    s2_len = ctx[7]
    s2_storage = ctx[3]
    s2_head = ctx[4]
    s2_token = ctx[5]
    layout = ctx[8]
    pf_row0 = s2_token + (s2_cstart + s2_r + 1) * CHUNK
    pf_validt = s2_len - (s2_cstart + s2_r + 1) * CHUNK
    if pf_validt > CHUNK:
        pf_validt = CHUNK
    if pf_validt < 0:
        pf_validt = 0
    kgp = kg2_l1g[(s2_r + 1) % 2]
    pl.set_validshape(kgp, [pf_validt, D])
    if layout == LAYOUT_TND:
        pl.load(kgp, kg_t,
                [pf_row0, s2_head, 0],
                order=[0, 2])
    else:
        pl.load(kgp, kg_t,
                [s2_storage, s2_head, pf_row0, 0],
                order=[2, 3])


# ===== stage1 cube 体级 helper（m1c/ctx 元组；契约即各 helper 解包行）=====


def _s2_km_head(m1c, ctx, ws_ops16_ptr):
    """s2 首块（rblk0）：LEFT_K/RIGHT 装载 + kk/mqk matmul + km 槽释放。

    句柄预备在 wait 前（if 内 wait 首位触发块提前闭合）；INTRA flag
    须等双 sub 落盘；L0C→UB 直通。
    """
    l1_rhsg = m1c[0]
    l0a_kmg = m1c[1]
    l0b_kmg = m1c[2]
    l0c_kmg = m1c[3]
    kk_w0g = m1c[4]
    mqk_w0g = m1c[5]
    task_slot = ctx[0]
    rhs_a = l1_rhsg.current()
    pl.system.wait_cross_core(
    pipe=pl.PipeType.MTE2, event_id=EV_READY_OP,
    sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
    pl.system.wait_cross_core(
    pipe=pl.PipeType.MTE1, event_id=EV_PANEL_L1,
    sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
    pl.load(
        rhs_a,
        pl.make_tensor(
    pl.addptr(ws_ops16_ptr, task_slot * WS_OPS_ELEMS),
    [CHUNK, D], dtype=pl.DT_BF16), [0, 0], order=[1, 0])
    lc_kk = l0c_kmg.current()
    lc_mqk = m1c[27].current()
    # kk: LEFT_K @ RIGHT
    la = l0a_kmg.current()
    lb = l0b_kmg.current()
    pl.move(la, m1c[25][task_slot % 2 * 4])
    pl.move(lb, rhs_a)
    pl.matmul(lc_kk, la, lb)
    # mqk: LEFT_Q @ RIGHT（不同 L0C，连续 M 排队——无中间 bar_m）
    lq_a = m1c[28].current()
    pl.move(lq_a, m1c[26][task_slot % 2 * 4])
    pl.matmul(lc_mqk, lq_a, lb)
    pl.system.bar_m()
    pl.set_validshape(lc_kk, [16, CHUNK])
    pl.set_validshape(lc_mqk, [16, CHUNK])
    # km 槽反向释放——s3(prev) 的 kk/mqk 读完成
    pl.system.wait_cross_core(
    pipe=pl.PipeType.FIX, event_id=EV_KM_KK_REL,
    sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    pl.system.wait_cross_core(
    pipe=pl.PipeType.FIX, event_id=EV_KM_MQK_REL,
    sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    pl.move(kk_w0g.current(), lc_kk,
        acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
    pl.move(mqk_w0g.current(), lc_mqk,
        acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)


def _s2_km_rblk(m1c, ctx, kk_w, mqk_w, ws_ops16_ptr):
    """计算后续评分行块。ctx 为任务槽、RIGHT 偏移和行块索引。"""
    l1_rhsg = m1c[0]
    l0a_kmg = m1c[1]
    l0b_kmg = m1c[2]
    l0c_kmg = m1c[3]
    task_slot = ctx[0]
    rhs_off = ctx[1]
    blk = ctx[2]
    rhs_a = l1_rhsg.current()
    pl.load(
        rhs_a,
        pl.make_tensor(
    pl.addptr(ws_ops16_ptr, task_slot * WS_OPS_ELEMS + rhs_off),
    [CHUNK, D], dtype=pl.DT_BF16), [0, 0], order=[1, 0])
    # kk + mqk 双 L0C/双 L0A：M 管线连续排队，单 bar_m 覆盖双结果
    la = l0a_kmg.current()
    lb = l0b_kmg.current()
    lc_kk = l0c_kmg.current()
    pl.move(la, m1c[25][task_slot % 2 * 4 + blk])
    pl.move(lb, rhs_a)
    pl.matmul(lc_kk, la, lb)
    lq_a = m1c[28].current()
    lc_mqk = m1c[27].current()
    pl.move(lq_a, m1c[26][task_slot % 2 * 4 + blk])
    pl.matmul(lc_mqk, lq_a, lb)
    pl.system.bar_m()
    pl.set_validshape(lc_kk, [16, CHUNK])
    pl.set_validshape(lc_mqk, [16, CHUNK])
    pl.move(kk_w.current(), lc_kk,
        acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
    pl.move(mqk_w.current(), lc_mqk,
        acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)


def _s4_seed(m1c, ctx, id64_f16_view):
    """s4 前段：id/T/seed 装载 + M₀/p₀ fixpipe 闭环。

    L0C→L1 用 pl.insert（超块 TINSERT/fixpipe）+ 显式标志对：M→FIX
    （eM0/em2）保证 insert 等 matmul 完成；FIX→MTE1（eR1/eR3）保证后
    续 L1 读取等 insert 完成。T/seed 已由 s3 直通 L1（MTE3 写入）；
    id 常量装载（MTE2，无依赖）。
    """
    l1_idg = m1c[6]
    l1_tg = m1c[7]
    l1_seedg = m1c[8]
    l1_mg = m1c[9]
    l1_powg = m1c[10]
    l0a_ng = m1c[11]
    l0b_idg = m1c[12]
    l0b_ng = m1c[13]
    l0c_ng = m1c[14]
    l0c_powg = m1c[15]
    i = ctx[0]
    id_a = l1_idg.current()
    pl.set_validshape(id_a, [CHUNK, CHUNK])
    pl.load(id_a, id64_f16_view, [0, 0])
    # L1→L0A 的 MTE1 读取须等 s3 的 UB→L1 落地
    pl.system.wait_cross_core(
    pipe=pl.PipeType.MTE1, event_id=EV_READY_TS,
    sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    t_a = l1_tg[i % 2]
    seed_a = l1_seedg[i % 2]
    la = l0a_ng.current()
    lbp = l0b_idg.current()
    lcn = l0c_ng.current()
    pl.set_validshape(lcn, [CHUNK, CHUNK])
    pl.move(la, seed_a)
    pl.move(lbp, id_a)
    pl.system.bar_mte1()
    pl.matmul(lcn, la, lbp)
    pl.system.bar_m()
    pl.system.sync_src(set_pipe=pl.PipeType.M,
                    wait_pipe=pl.PipeType.FIX, event_id=EV_FX_M0)
    pl.system.sync_dst(set_pipe=pl.PipeType.M,
                    wait_pipe=pl.PipeType.FIX, event_id=EV_FX_M0)
    m_cur = l1_mg.current()
    pl.insert(m_cur, lcn, [0, 0])
    pl.set_validshape(m_cur, [CHUNK, CHUNK])
    # p₀ 写 l0c_pow（不覆写 l0c_n 的 M₀——首轮 matmul_acc 要读）
    _s4_seed_p0(m1c, ctx, la)


def _s4_seed_p0(m1c, ctx, la):
    """s4 前段的 p₀ 半段：p₀ = T̂² fixpipe 回写 + M₀/p₀ 合并同步。"""
    l1_tg = m1c[7]
    l1_powg = m1c[10]
    l0b_ng = m1c[13]
    l0c_powg = m1c[15]
    i = ctx[0]
    t_a = l1_tg[i % 2]
    lb = l0b_ng.current()
    lcn_p0 = l0c_powg.current()
    pl.set_validshape(lcn_p0, [CHUNK, CHUNK])
    pl.move(la, t_a)
    pl.move(lb, t_a)
    pl.system.bar_mte1()
    pl.matmul(lcn_p0, la, lb)
    pl.system.bar_m()
    pl.system.sync_src(set_pipe=pl.PipeType.M,
                    wait_pipe=pl.PipeType.FIX, event_id=EV_FX_M2)
    pl.system.sync_dst(set_pipe=pl.PipeType.M,
                    wait_pipe=pl.PipeType.FIX, event_id=EV_FX_M2)
    pow_cur = l1_powg.current()
    pl.insert(pow_cur, lcn_p0, [0, 0])
    pl.set_validshape(pow_cur, [CHUNK, CHUNK])
    # M₀/p₀ 合用一对 FIX→MTE1（读点均在两 insert 之后）
    pl.system.sync_src(set_pipe=pl.PipeType.FIX,
                    wait_pipe=pl.PipeType.MTE1, event_id=EV_FX_R1)
    pl.system.sync_dst(set_pipe=pl.PipeType.FIX,
                    wait_pipe=pl.PipeType.MTE1, event_id=EV_FX_R1)


def _s4_neumann(m1c, ctx):
    """s4 后段：Neumann neu_iters 步迭代——combine + square。

    迭代不变式：lcn_m(L0C)=M_i，l1_m=M_i，l1_pow=p_i。combine 的
    matmul_acc 直接入 L0C（无需 M@I 拷贝 matmul——M_i 已驻留 lcn_m）。
    T̂ 严格下三角 64×64（T^64=0）；FP16 下 T^16 已为零（条目 ~0.15 时
    0.15^16 = 6.6e-14 < FP16 子正规 6e-8）——4 步即覆盖全部非零项。
    """
    l1_mg = m1c[9]
    l1_powg = m1c[10]
    l0a_ng = m1c[11]
    l0b_ng = m1c[13]
    l0c_ng = m1c[14]
    l0c_powg = m1c[15]
    neu_iters = ctx[0]
    for _step in pl.range(0, neu_iters):
        pow_l1 = l1_powg.current()
        m_l1 = l1_mg.current()
        la = l0a_ng.current()
        lb = l0b_ng.current()
        lcn_m = l0c_ng.current()
        pl.set_validshape(lcn_m, [CHUNK, CHUNK])
        pl.move(la, m_l1, offset=[0, 0])
        pl.move(lb, pow_l1, offset=[0, 0])
        pl.system.bar_mte1()
        pl.matmul_acc(lcn_m, lcn_m, la, lb)
        pl.system.bar_m()
        lcn_p = l0c_powg.current()
        pl.set_validshape(lcn_p, [CHUNK, CHUNK])
        pl.move(la, pow_l1, offset=[0, 0])
        pl.system.bar_mte1()
        pl.matmul(lcn_p, la, lb)
        pl.system.bar_m()
        # 两 matmul 均完成 → 一对 M→FIX 覆盖两个 insert
        pl.system.sync_src(set_pipe=pl.PipeType.M,
                        wait_pipe=pl.PipeType.FIX, event_id=EV_FX_M0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M,
                        wait_pipe=pl.PipeType.FIX, event_id=EV_FX_M0)
        pl.insert(m_l1, lcn_m, [0, 0])
        pl.insert(pow_l1, lcn_p, [0, 0])
        # 两个 insert 均落地 → 一对 FIX→MTE1 覆盖下一迭代读取
        pl.system.sync_src(set_pipe=pl.PipeType.FIX,
                        wait_pipe=pl.PipeType.MTE1, event_id=EV_FX_R1)
        pl.system.sync_dst(set_pipe=pl.PipeType.FIX,
                        wait_pipe=pl.PipeType.MTE1, event_id=EV_FX_R1)


def _s5_u(m1c, ctx, u_t, ws_ops16_ptr):
    """s5 前段：u = Ainv@(βV) 的装载/matmul/精确行落盘。

    s5 在 MTE3 写回后装载 BV/BKT。Ainv 从 L0C 写入 L1 时由 FP32 转为 BF16。
    """
    l1_vg = m1c[16]
    l1_bktg = m1c[17]
    l0a_u16g = m1c[18]
    l0b_vg = m1c[19]
    l0c_uwg = m1c[20]
    l1_a16g = m1c[22]
    uw_subg = m1c[23]
    l0c_ng = m1c[14]
    task_slot = ctx[0]
    core_id = ctx[1]
    storage_batch = ctx[2]
    head = ctx[3]
    row0_out = ctx[4]
    valid_rows = ctx[5]
    layout = ctx[6]
    va = l1_vg.current()
    pl.set_validshape(va, [CHUNK, D])
    pl.load(
        va,
        _ws_view_bf16(ws_ops16_ptr, task_slot, WS_BV_OFF, CHUNK),
        [0, 0])
    vb = l1_bktg.current()
    pl.set_validshape(vb, [CHUNK, D])
    pl.load(
        vb,
        _ws_view_bf16(ws_ops16_ptr, task_slot, WS_BKT_OFF, CHUNK),
        [0, 0])
    la = l0a_u16g.current()
    lbv = l0b_vg.current()
    lcu = l0c_uwg.current()
    pl.set_validshape(lbv, [CHUNK, D])
    pl.set_validshape(lcu, [CHUNK, D])
    pl.system.sync_src(set_pipe=pl.PipeType.M,
                    wait_pipe=pl.PipeType.FIX, event_id=EV_FX_M2)
    pl.system.sync_dst(set_pipe=pl.PipeType.M,
                    wait_pipe=pl.PipeType.FIX, event_id=EV_FX_M2)
    a16 = l1_a16g.current()
    pl.set_validshape(a16, [CHUNK, CHUNK])
    pl.insert(a16, l0c_ng.current(), [0, 0])
    pl.system.sync_src(set_pipe=pl.PipeType.FIX,
                    wait_pipe=pl.PipeType.MTE1, event_id=EV_FX_R3)
    pl.system.sync_dst(set_pipe=pl.PipeType.FIX,
                    wait_pipe=pl.PipeType.MTE1, event_id=EV_FX_R3)
    _s5_u_mm(m1c, ctx, u_t, va, a16)


def _s5_u_mm(m1c, ctx, u_t, va, a16):
    """s5 前段的乘法半段：Ainv@BV matmul + u 精确行落盘。"""
    l0a_u16g = m1c[18]
    l0b_vg = m1c[19]
    l0c_uwg = m1c[20]
    uw_subg = m1c[23]
    storage_batch = ctx[2]
    head = ctx[3]
    row0_out = ctx[4]
    valid_rows = ctx[5]
    layout = ctx[6]
    la = l0a_u16g.current()
    lbv = l0b_vg.current()
    lcu = l0c_uwg.current()
    pl.set_validshape(a16, [CHUNK, CHUNK])
    pl.move(la, a16, offset=[0, 0])
    pl.move(lbv, va)
    pl.system.bar_mte1()
    pl.matmul(lcu, la, lbv)
    pl.system.bar_m()
    _store_uw_subs(u_t, uw_subg, lcu, storage_batch, head,
                row0_out, valid_rows, layout, 1.0)


def _s5_w(m1c, ctx, w_t):
    """s5 后段：w = -(Ainv@(βK̃)) 的 sync 对/matmul/精确行落盘。"""
    l1_bktg = m1c[17]
    l0a_u16g = m1c[18]
    l0b_vg = m1c[19]
    l0c_wg = m1c[21]
    w_subg = m1c[24]
    storage_batch = ctx[2]
    head = ctx[3]
    row0_out = ctx[4]
    valid_rows = ctx[5]
    layout = ctx[6]
    # w-TMOV(MTE1 覆写 lbv) 需等 u-matmul(M 读 lbv) 完成
    pl.system.sync_src(set_pipe=pl.PipeType.M,
                    wait_pipe=pl.PipeType.MTE1, event_id=EV_FX_W4)
    pl.system.sync_dst(set_pipe=pl.PipeType.M,
                    wait_pipe=pl.PipeType.MTE1, event_id=EV_FX_W4)
    # 以及 βK̃ load(MTE2) 落地
    pl.system.sync_src(set_pipe=pl.PipeType.MTE2,
                    wait_pipe=pl.PipeType.MTE1, event_id=EV_FX_M0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2,
                    wait_pipe=pl.PipeType.MTE1, event_id=EV_FX_M0)
    vb = l1_bktg.current()
    la = l0a_u16g.current()
    lbv = l0b_vg.current()
    lcw = l0c_wg.current()
    pl.set_validshape(lcw, [CHUNK, D])
    pl.move(lbv, vb)
    pl.system.bar_mte1()
    pl.matmul(lcw, la, lbv)
    pl.system.bar_m()
    _store_uw_subs(w_t, w_subg, lcw, storage_batch, head,
                row0_out, valid_rows, layout, -1.0)
# ===== stage1 vec 体级 helper（m1v0/m1v1/ctx 元组；契约即各 helper 解包行）=====


def _s1a0_body(m1v0, ctx, q_t, g_t, v_t, b_t, alog_t, bias_t, beta_ptr,
               lower_bound):
    """s1a-sub0：q 路径——L2 归一化 + 门控累积 + β 三态装载 + v 装载。"""
    ub_qg = m1v0[0]
    ub_qng = m1v0[1]
    ub_qnfg = m1v0[2]
    ub_biasg = m1v0[3]
    ub_alogg = m1v0[4]
    ub_gcg = m1v0[5]
    ub_beta_slabg = m1v0[6]
    ub_beta_w32g = m1v0[7]
    ub_beta_rowg = m1v0[8]
    ub_beta_colg = m1v0[9]
    ub_betag = m1v0[10]
    storage_batch = ctx[0]
    qk_head = ctx[1]
    head = ctx[2]
    row0 = ctx[3]
    valid_rows = ctx[4]
    use_flat = ctx[5]
    seq_len = ctx[6]
    nv = ctx[7]
    i = ctx[8]
    layout = ctx[9]
    q_tile = ub_qg.current()
    qn_tile = ub_qng.current()
    qnf_tile = ub_qnfg.current()
    pl.set_validshape(q_tile, [valid_rows, D])
    _ld_seq(q_tile, q_t, storage_batch, qk_head, row0, layout)
    _l2norm_rows_vf(q_tile, qnf_tile, valid_rows)
    _f32_to_bf16_vf(qnf_tile, qn_tile, CHUNK)
    pl.set_validshape(q_tile, [valid_rows, D])
    _ld_seq(q_tile, g_t, storage_batch, head, row0, layout)
    bias_tile = ub_biasg.current()
    pl.set_validshape(bias_tile, [1, D])
    pl.load(bias_tile, bias_t, [head, 0])
    gc_tile = ub_gcg.current()
    al = ub_alogg.current()
    pl.set_validshape(al, [1, nv])
    pl.load(al, alog_t, [0, 0])
    pl.set_validshape(al, [1, D])
    _gate_cumsum_vf(q_tile, bias_tile, al, gc_tile,
                    head, lower_bound, valid_rows)
    _s1a0_beta(m1v0, ctx, q_tile, v_t, b_t, beta_ptr)


def _s1a0_beta(m1v0, ctx, q_tile, v_t, b_t, beta_ptr):
    """s1a-sub0 的 β 三态装载半段（NvFlat 0/1/2）+ v 装载。"""
    ub_beta_slabg = m1v0[6]
    ub_beta_w32g = m1v0[7]
    ub_beta_rowg = m1v0[8]
    ub_beta_colg = m1v0[9]
    ub_betag = m1v0[10]
    storage_batch = ctx[0]
    head = ctx[2]
    row0 = ctx[3]
    valid_rows = ctx[4]
    use_flat = ctx[5]
    seq_len = ctx[6]
    nv = ctx[7]
    i = ctx[8]
    layout = ctx[9]
    if use_flat == 1:
        bslab = ub_beta_slabg.current()
        _ld_beta_flat(bslab, beta_ptr, storage_batch,
                    row0, valid_rows, nv, layout,
                    seq_len)
        _sigmoid_slab_vf(bslab, ub_betag[i % 2],
                        valid_rows, nv, head)
    elif use_flat == 2:
        bwin = ub_beta_w32g.current()
        bcol = _ld_beta_w32(
            bwin, beta_ptr, storage_batch,
            row0, valid_rows, nv, head,
            layout)
        _sigmoid_slab_vf(bwin, ub_betag[i % 2],
                        valid_rows, 32, bcol)
    else:
        beta_row = ub_beta_rowg.current()
        beta_col = ub_beta_colg.current()
        pl.set_validshape(beta_row, [1, valid_rows])
        pl.set_validshape(beta_col, [valid_rows, 1])
        _ld_beta(beta_row, beta_col, b_t, storage_batch,
                head, row0, layout)
        _sigmoid_beta_vf(beta_row, ub_betag[i % 2],
                        valid_rows)
    pl.set_validshape(q_tile, [valid_rows, D])
    _ld_seq(q_tile, v_t, storage_batch, head, row0, layout)


def _emit_left_l1(l1_pg, wnzg, stgg, outfg, kn, gc, task_slot, scale):
    """LEFT_Q/LEFT_K × 4 块发射：VF 块写 → BF16 cast → UB ND→NZ → L1 面板
    （s2 免 GM ws 往返；8 槽 = 奇偶(task_slot%2)×4 rblk）。"""
    outf = outfg.current()
    stg = stgg.current()
    wnz = wnzg.current()
    for rblk in pl.range(0, 4):
        _emit_left_vf(kn, gc, outf,
                    rblk * 16, 16, scale)
        _f32_to_bf16_vf(outf, stg, 16)
        pl.set_validshape(stg, [16, D])
        pl.set_validshape(wnz, [16, D])
        pl.move(wnz, stg)
        dst = l1_pg[task_slot % 2 * 4 + rblk]
        pl.set_validshape(dst, [16, D])
        pl.move(dst, wnz)
    # 面板 L1 交接事件：cbuf 写跨处理器不靠 mutex 排序（stage2 的 WU 同款
    # 配方——MTE3 set，消费侧 wait）；INTRA 双 sub 各 set，s2 head 一 wait 收双路
    pl.system.set_cross_core(
        pipe=pl.PipeType.MTE3, event_id=EV_PANEL_L1,
        sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)


def _s1b0_body(m1v0, ctx, qg_t, decay_t, ws_ops16_ptr, scale):
    """s1b-sub0：BV/LEFT_Q 下发 + qg/decay 纯输出（含 s2 启动 flag）。"""
    ub_qg = m1v0[0]
    ub_qng = m1v0[1]
    ub_gcg = m1v0[5]
    ub_betag = m1v0[10]
    ub_outfg = m1v0[11]
    ub_stgg = m1v0[12]
    ub_decayfg = m1v0[13]
    storage_batch = ctx[0]
    head = ctx[1]
    row0_out = ctx[2]
    task_slot = ctx[4]
    logical_batch = ctx[7]
    chunk = ctx[8]
    i = ctx[9]
    layout = ctx[10]
    qn_b = ub_qng.current()
    gc_b = ub_gcg.current()
    outf = ub_outfg.current()
    stg = ub_stgg.current()
    _emit_left_l1(m1v0[25], m1v0[24], m1v0[12], m1v0[11],
                  qn_b, gc_b, task_slot, scale)
    # s2 启动 flag：INTRA 须等双 sub 落盘（LEFT 面板 cbuf 写；BV 属 s5 次轮
    # 消费——round 序保证其后置安全，flag 边沿不再等其 GM 落盘）
    pl.system.set_cross_core(
        pipe=pl.PipeType.MTE3, event_id=EV_READY_OP,
        sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
    _s1b0_bv(m1v0, ctx, ws_ops16_ptr)
    _s1b0_out(m1v0, ctx, qg_t, decay_t, scale)


def _s1b0_bv(m1v0, ctx, ws_ops16_ptr):
    """s1b-sub0 的 BV 半段：βV 生成 + ws 落盘（s5 依赖）。"""
    ub_qg = m1v0[0]
    ub_qng = m1v0[1]
    ub_gcg = m1v0[5]
    ub_betag = m1v0[10]
    ub_outfg = m1v0[11]
    ub_stgg = m1v0[12]
    task_slot = ctx[4]
    valid_rows = ctx[3]
    i = ctx[9]
    qn_b = ub_qng.current()
    gc_b = ub_gcg.current()
    outf = ub_outfg.current()
    stg = ub_stgg.current()
    _emit_bv_vf(ub_qg.current(), ub_betag[i % 2], outf,
                valid_rows)
    _f32_to_bf16_vf(outf, stg, CHUNK)
    pl.set_validshape(stg, [CHUNK, D])
    pl.store(_ws_view_bf16(ws_ops16_ptr, task_slot, WS_BV_OFF, CHUNK),
            stg, [0, 0])


def _s1b0_out(m1v0, ctx, qg_t, decay_t, scale):
    """s1b-sub0 的输出半段：qg + decay 纯输出（decay 移 sub0 平衡负载）。"""
    ub_qng = m1v0[1]
    ub_gcg = m1v0[5]
    ub_outfg = m1v0[11]
    ub_stgg = m1v0[12]
    ub_decayfg = m1v0[13]
    storage_batch = ctx[0]
    head = ctx[1]
    row0_out = ctx[2]
    valid_rows = ctx[3]
    logical_batch = ctx[7]
    chunk = ctx[8]
    layout = ctx[10]
    qn_b = ub_qng.current()
    gc_b = ub_gcg.current()
    outf = ub_outfg.current()
    stg = ub_stgg.current()
    _emit_qg_vf(qn_b, gc_b, outf, scale, valid_rows)
    _f32_to_bf16_vf(outf, stg, CHUNK)
    _st_seq(qg_t, stg, storage_batch, head, row0_out,
            valid_rows, layout)
    decayf0 = ub_decayfg.current()
    _emit_decay_only_vf(gc_b, decayf0, valid_rows)
    pl.set_validshape(decayf0, [1, D])
    _st_decay(decay_t, decayf0, logical_batch,
            head, chunk)


def _s3_body(m1v0, ctx, mqk_t):
    """s3：T/seed/mqk 的 Vector 中转（km 槽释放 + L1 直通 + mqk 落盘）。"""
    ub_km_kkg = m1v0[14]
    ub_tfg = m1v0[15]
    ub_seedfg = m1v0[16]
    ub_betag = m1v0[10]
    ub_ts16g = m1v0[17]
    ub_ts16_nzg = m1v0[18]
    ub_mqkbfg = m1v0[19]
    ub_km_mqkg = m1v0[20]
    ub_mqkfg = m1v0[21]
    l1_tg = m1v0[22]
    l1_seedg = m1v0[23]
    i = ctx[0]
    task_count = ctx[1]
    logical_batch = ctx[3]
    head = ctx[4]
    chunk = ctx[5]
    pl.system.wait_cross_core(
        pipe=pl.PipeType.V, event_id=EV_READY_KM,
        sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    # beta/l1_t/l1_seed 按任务奇偶取槽（gc 有并发写竞态，不取槽）
    kmsrc = ub_km_kkg.current()
    tf = ub_tfg.current()
    seedf = ub_seedfg.current()
    beta_tile = ub_betag[i % 2]
    _build_t_seed_vf(kmsrc, beta_tile, tf, seedf)
    # kk 读毕——释放 km kk 槽（末轮不 set 保持计数平衡）
    if i + 1 < task_count:
        pl.system.set_cross_core(
            pipe=pl.PipeType.V, event_id=EV_KM_KK_REL,
            sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    # T/seed 直通 L1：FP32→FP16(UB ND) → UB NZ → L1 NZ（两跳必须）
    ts16 = ub_ts16g.current()
    ts16_nz = ub_ts16_nzg.current()
    _f32_to_f16_vf(tf, ts16, CHUNK)
    pl.set_validshape(ts16_nz, [CHUNK, CHUNK])
    pl.move(ts16_nz, ts16)
    pl.set_validshape(l1_tg[i % 2], [CHUNK, CHUNK])
    pl.move(l1_tg[i % 2], ts16_nz)
    _f32_to_f16_vf(seedf, ts16, CHUNK)
    pl.set_validshape(ts16_nz, [CHUNK, CHUNK])
    pl.move(ts16_nz, ts16)
    pl.set_validshape(l1_seedg[i % 2], [CHUNK, CHUNK])
    pl.move(l1_seedg[i % 2], ts16_nz)
    _s3_mqk(m1v0, ctx, mqk_t)


def _s3_mqk(m1v0, ctx, mqk_t):
    """s3 的 mqk 半段：mqk 掩码 + km mqk 槽释放 + 落盘 + TS flag。"""
    ub_mqkbfg = m1v0[19]
    ub_km_mqkg = m1v0[20]
    ub_mqkfg = m1v0[21]
    i = ctx[0]
    task_count = ctx[1]
    logical_batch = ctx[3]
    head = ctx[4]
    chunk = ctx[5]
    mqkbf = ub_mqkbfg.current()
    kmsrc2 = ub_km_mqkg.current()
    mqkf = ub_mqkfg.current()
    _emit_mqk_vf(kmsrc2, mqkf)
    # mqk 读毕——释放 km mqk 槽
    if i + 1 < task_count:
        pl.system.set_cross_core(
            pipe=pl.PipeType.V, event_id=EV_KM_MQK_REL,
            sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    _f32_to_bf16_64_vf(mqkf, mqkbf)
    pl.set_validshape(mqkbf, [CHUNK, CHUNK])
    pl.store(mqk_t, mqkbf,
            [logical_batch, head, chunk, 0, 0],
            order=[3, 4])
    pl.system.set_cross_core(
        pipe=pl.PipeType.MTE3, event_id=EV_READY_TS,
        sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)


def _s1a1_body(m1v1, ctx, k_t, g_t, b_t, alog_t, bias_t, beta_ptr,
               lower_bound):
    """s1a-sub1：k 路径——L2 归一化 + 门控累积 + β 三态装载。"""
    ub_k_s1g = m1v1[0]
    ub_kn1g = m1v1[1]
    ub_outf1g = m1v1[2]
    ub_bias_s1g = m1v1[3]
    ub_gc1g = m1v1[4]
    ub_alog_s1g = m1v1[5]
    ub_beta_slab1g = m1v1[6]
    ub_beta_w321g = m1v1[7]
    ub_beta_row_s1g = m1v1[8]
    ub_beta_col_s1g = m1v1[9]
    ub_beta_s1g = m1v1[10]
    storage_batch = ctx[0]
    s1_qk_head = ctx[1]
    s1_head = ctx[2]
    s1_row0 = ctx[3]
    s1_valid_rows = ctx[4]
    use_flat = ctx[5]
    seq_len = ctx[6]
    nv = ctx[7]
    layout = ctx[8]
    k_tile = ub_k_s1g.current()
    kn_tile = ub_kn1g.current()
    knf1_tile = ub_outf1g.current()
    pl.set_validshape(k_tile, [s1_valid_rows, D])
    _ld_seq(k_tile, k_t, storage_batch, s1_qk_head, s1_row0, layout)
    _l2norm_rows_vf(k_tile, knf1_tile, s1_valid_rows)
    _f32_to_bf16_vf(knf1_tile, kn_tile, CHUNK)
    pl.set_validshape(k_tile, [s1_valid_rows, D])
    _ld_seq(k_tile, g_t, storage_batch, s1_head, s1_row0, layout)
    bias1 = ub_bias_s1g.current()
    pl.set_validshape(bias1, [1, D])
    pl.load(bias1, bias_t, [s1_head, 0])
    gc1_tile = ub_gc1g.current()
    al1 = ub_alog_s1g.current()
    pl.set_validshape(al1, [1, nv])
    pl.load(al1, alog_t, [0, 0])
    pl.set_validshape(al1, [1, D])
    _gate_cumsum_vf(k_tile, bias1, al1, gc1_tile,
                    s1_head, lower_bound, s1_valid_rows)
    _s1a1_beta(m1v1, ctx, b_t, beta_ptr)


def _s1a1_beta(m1v1, ctx, b_t, beta_ptr):
    """s1a-sub1 的 β 三态装载半段（NvFlat 0/1/2）。"""
    ub_beta_slab1g = m1v1[6]
    ub_beta_w321g = m1v1[7]
    ub_beta_row_s1g = m1v1[8]
    ub_beta_col_s1g = m1v1[9]
    ub_beta_s1g = m1v1[10]
    storage_batch = ctx[0]
    s1_head = ctx[2]
    s1_row0 = ctx[3]
    s1_valid_rows = ctx[4]
    use_flat = ctx[5]
    seq_len = ctx[6]
    nv = ctx[7]
    layout = ctx[8]
    if use_flat == 1:
        bslab1 = ub_beta_slab1g.current()
        _ld_beta_flat(bslab1, beta_ptr, storage_batch,
                    s1_row0, s1_valid_rows, nv,
                    layout, seq_len)
        _sigmoid_slab_vf(bslab1, ub_beta_s1g.current(),
                        s1_valid_rows, nv, s1_head)
    elif use_flat == 2:
        bwin1 = ub_beta_w321g.current()
        bcol1 = _ld_beta_w32(
            bwin1, beta_ptr, storage_batch,
            s1_row0, s1_valid_rows, nv,
            s1_head, layout)
        _sigmoid_slab_vf(bwin1, ub_beta_s1g.current(),
                        s1_valid_rows, 32, bcol1)
    else:
        br1 = ub_beta_row_s1g.current()
        bc1 = ub_beta_col_s1g.current()
        pl.set_validshape(br1, [1, s1_valid_rows])
        pl.set_validshape(bc1, [s1_valid_rows, 1])
        _ld_beta(br1, bc1, b_t, storage_batch,
                s1_head, s1_row0, layout)
        _sigmoid_beta_vf(br1, ub_beta_s1g.current(),
                        s1_valid_rows)


def _s1b1_body(m1v1, ctx, kg_t, ws_ops16_ptr):
    """s1b-sub1：BKT/LEFT_K/RIGHT 下发 + kg 纯输出（含 s2 启动 flag）。"""
    ub_kn1g = m1v1[1]
    ub_gc1g = m1v1[4]
    ub_beta_s1g = m1v1[10]
    ub_outf1g = m1v1[2]
    ub_stg1g = m1v1[11]
    storage_batch = ctx[0]
    s1_head = ctx[1]
    s1_row0_out = ctx[2]
    s1_valid_rows = ctx[3]
    s1_task_slot = ctx[4]
    core_id = ctx[5]
    layout = ctx[7]
    kn_b1 = ub_kn1g.current()
    gc_b1 = ub_gc1g.current()
    outf1 = ub_outf1g.current()
    stg1 = ub_stg1g.current()
    _emit_left_l1(m1v1[13], m1v1[12], m1v1[11], m1v1[2],
                  kn_b1, gc_b1, s1_task_slot, 1.0)
    _s1b1_right(kn_b1, gc_b1, outf1, stg1, ctx, kg_t,
                ws_ops16_ptr)
    # BKT 后置：s5 次轮消费（round 序全序保证）；stg1 已被 RIGHT/kg 复用完
    _emit_bkt_vf(kn_b1, gc_b1, ub_beta_s1g.current(), outf1,
                s1_valid_rows)
    _f32_to_bf16_vf(outf1, stg1, CHUNK)
    pl.set_validshape(stg1, [CHUNK, D])
    pl.store(_ws_view_bf16(ws_ops16_ptr, s1_task_slot, WS_BKT_OFF, CHUNK),
            stg1, [0, 0])


def _s1b1_right(kn_b1, gc_b1, outf1, stg1, ctx, kg_t,
                ws_ops16_ptr):
    """s1b-sub1 的右乘数半段：RIGHT × 10 下发 + flag + kg 纯输出。

    tile 句柄由 _s1b1_body 传入（复用其已解析实例，避免重复解包）。
    """
    storage_batch = ctx[0]
    s1_head = ctx[1]
    s1_row0_out = ctx[2]
    s1_valid_rows = ctx[3]
    s1_task_slot = ctx[4]
    layout = ctx[7]
    for rblk in pl.range(0, 4):
        for cblk in pl.range(0, rblk + 1):
            _emit_right_vf(kn_b1, gc_b1, outf1,
                        rblk * 16, cblk * 16, 16)
            pair_off = _right_pair_index(
                rblk, cblk) * 2048
            _f32_to_bf16_vf(outf1, stg1, 16)
            pl.set_validshape(stg1, [16, D])
            pl.store(
                _ws_view_bf16(
                    ws_ops16_ptr, s1_task_slot,
                    pair_off, 16),
                stg1, [0, 0])
    # s2 启动 flag：INTRA 须等双 sub 落盘
    pl.system.set_cross_core(
        pipe=pl.PipeType.MTE3, event_id=EV_READY_OP,
        sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
    _emit_kg_only_vf(kn_b1, gc_b1, outf1, s1_valid_rows)
    _f32_to_bf16_vf(outf1, stg1, CHUNK)
    _st_seq(kg_t, stg1, storage_batch, s1_head, s1_row0_out,
            s1_valid_rows, layout)
# ===== vp 2a/2b 体级 helper（m2/ctx 元组；契约即各 helper 解包行）=====


@pl.vector_function
def _vp_zero_count_vf(src, out, n_words: pl.DT_INT64):
    """R 切片全零判定：Σ_blocks max|x| → lane0（== 0.0 当且仅当全为 ±0）。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    acc = vf.full(0.0, preg)
    for i in pl.range(0, n_words):
        x = vf.load_align(src, i * LANES)
        a = vf.abs(x, preg)
        m = vf.reduce_max(a, preg)
        acc = vf.add(acc, m, preg)
    vf.store_align(out, acc, preg, 0)
    vf.store_align(out, acc, preg, LANES)
    vf.mem_bar()


def _vp2a_cube(m2, ctx, w_t, kg_t, vp_r, vp_lcp):
    """2a 单任务：kg 装载 + wᵀ 装载/等待 + φ matmul（批内不排空）。"""
    kg2_l1g = m2[2]
    wt2_l1g = m2[19]
    l0a_pb2g = m2[10]
    l0b_pb2g = m2[11]
    l0c_phi2g = m2[12]
    vp_storage = ctx[0]
    vp_head = ctx[1]
    vp_row0 = ctx[2]
    vp_valid = ctx[3]
    layout = ctx[4]
    vp_need_wu = ctx[5]
    kga = kg2_l1g[vp_r % 2]
    pl.set_validshape(kga, [vp_valid, D])
    if layout == LAYOUT_TND:
        pl.load(kga, kg_t, [vp_row0, vp_head, 0], order=[0, 2])
    else:
        pl.load(kga, kg_t, [vp_storage, vp_head, vp_row0, 0], order=[2, 3])
    wta = wt2_l1g[vp_r % 2]
    pl.set_validshape(wta, [D, CHUNK])
    if vp_need_wu == 1:
        pl.system.wait_cross_core(
            pipe=pl.PipeType.MTE1, event_id=EV_READY_WU,
            sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    else:
        pl.load(wta, w_t, [vp_storage, vp_head, vp_row0, 0], order=[3, 2])
    la = l0a_pb2g.current()
    pl.set_validshape(la, [D, vp_valid])
    pl.move(la, wta)
    lb = l0b_pb2g.current()
    pl.set_validshape(lb, [vp_valid, D])
    pl.move(lb, kga)
    pl.system.bar_mte1()
    lcp = l0c_phi2g[vp_lcp]
    pl.set_validshape(lcp, [D, D])
    pl.matmul(lcp, la, lb)


def _vp2a_flush(m2, vp_r0, vp_r1, vp_task_count):
    """2a 批尾：一次 M→FIX 同步对 + 双 FIX 发布 + 一次 TS 事件。"""
    phi_ub2g = m2[7]
    l0c_phi2g = m2[12]
    ph0 = phi_ub2g[vp_r0 % 2]
    pl.set_validshape(ph0, [D, D])
    pl.system.sync_src(set_pipe=pl.PipeType.M,
                    wait_pipe=pl.PipeType.FIX, event_id=EV_FX_M0)
    pl.system.sync_dst(set_pipe=pl.PipeType.M,
                    wait_pipe=pl.PipeType.FIX, event_id=EV_FX_M0)
    pl.move(ph0, l0c_phi2g[0],
            acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
    if vp_r1 < vp_task_count:
        ph1 = phi_ub2g[vp_r1 % 2]
        pl.set_validshape(ph1, [D, D])
        pl.move(ph1, l0c_phi2g[1],
                acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
    pl.system.set_cross_core(
        pipe=pl.PipeType.FIX, event_id=EV_READY_TS,
        sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)


def _vp2a_w_chain(m2, ctx, w_t):
    """2a：尾块/TND 的 wᵀ 装载链（vec 三跳）+ WU 发布。"""
    wt2_l1g = m2[19]
    vp_storage = ctx[0]
    vp_head = ctx[1]
    vp_row0 = ctx[2]
    vp_valid = ctx[3]
    layout = ctx[4]
    dst_w = wt2_l1g[ctx[5] % 2]
    _wu_load_chain(m2, w_t, vp_row0, vp_storage, vp_head,
                   vp_valid, layout, dst_w)
    pl.system.set_cross_core(
        pipe=pl.PipeType.MTE3, event_id=EV_READY_WU,
        sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)


def _vp2a_diag_store(m2, ctx, decay_t, ws_phi_ptr):
    """2a：φ(r-1) 对角加（decay 行）+ GM cache 落盘（TS 由批首统一收）。"""
    ub_decayg = m2[17]
    phi_ub2g = m2[7]
    vp_logical = ctx[0]
    vp_head = ctx[1]
    vp_chunk = ctx[2]
    vp_r = ctx[3]
    vp_nv = ctx[4]
    vp_capacity = ctx[5]
    vp_slot = (vp_logical * vp_nv + vp_head) * vp_capacity + vp_chunk
    phv = pl.make_tensor(
        pl.addptr(ws_phi_ptr, vp_slot * D * D), [D, D], dtype=pl.DT_FP32)
    dcur = ub_decayg.current()
    pl.set_validshape(dcur, [1, D])
    pl.load(dcur, decay_t, [vp_logical, vp_head, vp_chunk, 0])
    _diag_add_vf(phi_ub2g[vp_r % 2], dcur)
    pl.store(phv, phi_ub2g[vp_r % 2], [0, 0])


def _vp2b_init_cube(m2, ctx, kg_t, trans_t, state_t, id128_view,
                    zeros128_view, u_t, ws_phi_ptr, ws_zero_t):
    """2b 链首 Cube：准备 R/ψ 和首块操作数，初始化 R 存活标志。"""
    st_r2g = m2[0]
    st_psi2g = m2[1]
    kg2_l1g = m2[2]
    ut2_l1g = m2[6]
    phi2_l1g = m2[9]
    s2_cstart = ctx[0]
    s2_logical = ctx[1]
    s2_head = ctx[2]
    s2_storage = ctx[4]
    s2_token = ctx[5]
    s2_valid0 = ctx[6]
    layout = ctx[7]
    s2_row_off = ctx[8]
    s2_rows = ctx[9]
    s2_c0_vec = ctx[10]
    s2_nv = ctx[11]
    s2_capacity = ctx[12]
    s2_zword = ctx[13]
    # 用 FP32 1.0 的位型标记 R 存活。AIV 写入 0 才表示已确认 R 全零。
    pl.setval(ws_zero_t, s2_zword, 0x3F800000)
    ph_slot = (s2_logical * s2_nv + s2_head) * s2_capacity + s2_cstart
    phv = pl.make_tensor(
        pl.addptr(ws_phi_ptr, ph_slot * D * D), [D, D], dtype=pl.DT_FP32)
    stra = st_r2g.current()
    pl.set_validshape(stra, [s2_rows, D])
    if s2_cstart > 0:
        pl.load(stra, trans_t, [s2_logical, s2_head, s2_row_off, 0])
    else:
        pl.load(stra, id128_view, [s2_row_off, 0])
    spsi = st_psi2g.current()
    pl.set_validshape(spsi, [s2_rows, D])
    if s2_cstart > 0:
        pl.load(spsi, state_t, [s2_logical, s2_head, s2_row_off, 0])
    else:
        pl.load(spsi, zeros128_view, [s2_row_off, 0])
    _load_first_kg(kg2_l1g, kg_t, s2_valid0, layout, s2_token, s2_cstart, s2_head, s2_storage)
    phl = phi2_l1g[0]
    pl.set_validshape(phl, [D, D])
    pl.load(phl, phv, [0, 0])
    if s2_c0_vec == 0:
        utp = ut2_l1g[0]
        pl.set_validshape(utp, [s2_rows, CHUNK])
        pl.load(utp, u_t,
                [s2_storage, s2_head, s2_token + s2_cstart * CHUNK,
                 s2_row_off],
                order=[3, 2])


def _vp2b_init_vec(m2, ctx, u_t, u_pair):
    """2b 链首 vec：链边界等待 + chunk-0 u 链（TND/尾块）。"""
    ut2_l1g = m2[6]
    s2_task = ctx[0]
    s2_start = ctx[1]
    s2_c0_vec = ctx[2]
    s2_storage = ctx[3]
    s2_head = ctx[4]
    s2_token = ctx[5]
    s2_cstart = ctx[6]
    s2_valid0 = ctx[8]
    layout = ctx[9]
    s2_row_off = ctx[10]
    s2_rows = ctx[11]
    dst_u = ut2_l1g[0]
    if s2_task > s2_start:
        pl.system.wait_cross_core(
            pipe=pl.PipeType.MTE3, event_id=EV_CHAIN_DONE,
            sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    if s2_c0_vec == 1:
        row0_u = s2_token + s2_cstart * CHUNK
        _u_load_chain(u_pair, u_t, row0_u, s2_storage, s2_head,
                      s2_valid0, layout, dst_u, s2_row_off, s2_rows)
        pl.system.set_cross_core(
            pipe=pl.PipeType.MTE3, event_id=EV_READY_WU,
            sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)


def _vp2b_phi_load(m2, ctx, ws_phi_ptr):
    """2b：chunk r 的 φ GM cache 直读 → L1 双槽。"""
    phi2_l1g = m2[9]
    s2_r = ctx[0]
    s2_logical = ctx[1]
    s2_head = ctx[2]
    s2_cstart = ctx[3]
    s2_nv = ctx[4]
    s2_capacity = ctx[5]
    ph_slot = (s2_logical * s2_nv + s2_head) * s2_capacity + s2_cstart + s2_r
    phv = pl.make_tensor(
        pl.addptr(ws_phi_ptr, ph_slot * D * D), [D, D], dtype=pl.DT_FP32)
    phl = phi2_l1g[s2_r % 2]
    pl.set_validshape(phl, [D, D])
    pl.load(phl, phv, [0, 0])


def _vp2b_rank(m2, ctx, trans_t, state_t, r_live):
    """2b rank(r-1)：R@φ（R 已验证全零且非检查轮时跳过）+ 零检查产制。"""
    l0b_phi2g = m2[13]
    l0a_r2g = m2[14]
    l0c_r2g = m2[15]
    phi2_l1g = m2[9]
    phi_ub2g = m2[7]
    st_r2g = m2[0]
    s2_r = ctx[0]
    s2_cnum = ctx[1]
    s2_rows = ctx[7]
    s2_chk = ctx[11]
    # 早门：R insert(r-1) 毕 → φ/R L0 装载放行（与 ψ mat(r-1) 重叠，
    # 消除 uTkg(r-1) 末到 R mat(r) 之间的 M 空泡）
    if s2_r >= 2:
        pl.system.sync_dst(set_pipe=pl.PipeType.FIX,
                        wait_pipe=pl.PipeType.MTE1, event_id=EV_FX_RR)
    lb2 = l0b_phi2g.current()
    pl.set_validshape(lb2, [D, D])
    pl.move(lb2, phi2_l1g[(s2_r - 1) % 2])
    la2 = l0a_r2g.current()
    pl.set_validshape(la2, [s2_rows, D])
    lcr = l0c_r2g.current()
    pl.set_validshape(lcr, [s2_rows, D])
    s2_r_live = r_live
    if s2_chk == 1:
        s2_r_live = 1
    if s2_r == s2_cnum:
        s2_r_live = 1
    if s2_r_live != 0:
        pl.move(la2, st_r2g.current(), offset=[0, 0])
        pl.system.bar_mte1()
        pl.matmul(lcr, la2, lb2)
        # R-提前插入：R matmul 完成即插入（与 psi/u^T 重叠——下一轮
        # R 装载只需等 R insert 不需等全轮完成，缩短 R 递推关键链）
        pl.system.sync_src(set_pipe=pl.PipeType.M,
                        wait_pipe=pl.PipeType.FIX, event_id=EV_FX_R1)
        pl.system.sync_dst(set_pipe=pl.PipeType.M,
                        wait_pipe=pl.PipeType.FIX, event_id=EV_FX_R1)
        pl.insert(st_r2g.current(), lcr, [0, 0])
        if s2_chk == 1:
            zc = phi_ub2g[0]
            pl.set_validshape(zc, [s2_rows, D])
            pl.move(zc, lcr, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
            pl.system.set_cross_core(
                pipe=pl.PipeType.FIX, event_id=EV_WT_REL,
                sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    # 早门 src：R insert 毕即放行下一轮 φ/R 装载（skip 轮也发射——
    # 槽内遗留值即已验证的全零 R，语义正确）
    if s2_r < s2_cnum:
        pl.system.sync_src(set_pipe=pl.PipeType.FIX,
                        wait_pipe=pl.PipeType.MTE1, event_id=EV_FX_RR)
    _vp2b_rank_psi(m2, ctx, trans_t, state_t, la2, lb2, lcr)


def _vp2b_rank_psi(m2, ctx, trans_t, state_t, la2, lb2, lcr):
    """2b：ψ ← ψ@φ + uᵀ@kg（ut2 消费前 WU 配平）+ 提交。"""
    l0c_psi2g = m2[16]
    st_psi2g = m2[1]
    l0a_ur2g = m2[18]
    l0b_pb2g = m2[11]
    ut2_l1g = m2[6]
    kg2_l1g = m2[2]
    s2_r = ctx[0]
    s2_cnum = ctx[1]
    s2_rows = ctx[7]
    s2_last_valid = ctx[8]
    layout = ctx[9]
    s2_c0_vec = ctx[12]
    la_r = l0a_ur2g.current()
    k_valid = CHUNK
    if s2_r == s2_cnum:
        k_valid = s2_last_valid
    pl.set_validshape(la_r, [s2_rows, k_valid])
    lb_r = l0b_pb2g.current()
    pl.set_validshape(lb_r, [k_valid, D])
    lcp2 = l0c_psi2g.current()
    pl.set_validshape(lcp2, [s2_rows, D])
    vp_wait_wu = 0
    if s2_r == 1:
        vp_wait_wu = s2_c0_vec
    if s2_r == s2_cnum:
        if s2_last_valid < CHUNK:
            vp_wait_wu = 1
    if layout == LAYOUT_TND:
        vp_wait_wu = 1
    if vp_wait_wu == 1:
        pl.system.wait_cross_core(
            pipe=pl.PipeType.MTE1, event_id=EV_READY_WU,
            sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    # 等待上一轮 ψ 写回，再装载本轮 ψ/uᵀ/kg。
    if s2_r >= 2:
        pl.system.sync_dst(set_pipe=pl.PipeType.FIX,
                        wait_pipe=pl.PipeType.MTE1, event_id=EV_FX_RP)
    pl.set_validshape(la2, [s2_rows, D])
    pl.move(la2, st_psi2g.current(), offset=[0, 0])
    pl.system.bar_mte1()
    # ψ@φ 和 uᵀ@kg 连续累加到同一 L0C，提交前由 M→FIX 事件确认完成。
    pl.matmul(lcp2, la2, lb2)
    pl.move(la_r, ut2_l1g[(s2_r - 1) % 2])
    pl.move(lb_r, kg2_l1g[(s2_r - 1) % 2])
    pl.system.bar_mte1()
    pl.matmul_acc(lcp2, lcp2, la_r, lb_r)
    _vp2b_rank_commit(m2, ctx, trans_t, state_t, lcr, lcp2)


def _vp2b_rank_commit(m2, ctx, trans_t, state_t, lcr, lcp2):
    """2b 提交：UF 释放 + R/ψ 滚动或末轮落盘（R 跳过轮不回写）。"""
    st_psi2g = m2[1]
    s2_r = ctx[0]
    s2_cnum = ctx[1]
    s2_logical = ctx[2]
    s2_head = ctx[3]
    s2_task = ctx[4]
    s2_end = ctx[5]
    s2_row_off = ctx[6]
    s2_last_valid = ctx[8]
    layout = ctx[9]
    if s2_r + 1 < s2_cnum:
        s2_r1_vec = 0
        if s2_r + 2 == s2_cnum:
            if s2_last_valid < CHUNK:
                s2_r1_vec = 1
        if layout == LAYOUT_TND:
            s2_r1_vec = 1
        if s2_r1_vec == 1:
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE1, event_id=EV_READY_UF,
                sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    if s2_r == s2_cnum:
        pl.system.sync_src(set_pipe=pl.PipeType.M,
                           wait_pipe=pl.PipeType.FIX, event_id=EV_FX_R1)
        pl.system.sync_dst(set_pipe=pl.PipeType.M,
                           wait_pipe=pl.PipeType.FIX, event_id=EV_FX_R1)
        pl.store(trans_t, lcr,
                [s2_logical, s2_head, s2_row_off, 0])
        pl.store(state_t, lcp2,
                [s2_logical, s2_head, s2_row_off, 0])
        if s2_task + 1 < s2_end:
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE1, event_id=EV_CHAIN_DONE,
                sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    else:
        # psi 插入；R 已在 rank 前段插入（FIX FIFO 保序——psi 后插必后成）
        pl.system.sync_src(set_pipe=pl.PipeType.M,
                        wait_pipe=pl.PipeType.FIX, event_id=EV_FX_R1)
        pl.system.sync_dst(set_pipe=pl.PipeType.M,
                        wait_pipe=pl.PipeType.FIX, event_id=EV_FX_R1)
        pl.insert(st_psi2g.current(), lcp2, [0, 0])
        # 晚门 src：ψ insert 毕 → 下一轮 ψ/uT/kg 装载（dst 在 rank_psi）
        pl.system.sync_src(set_pipe=pl.PipeType.FIX,
                        wait_pipe=pl.PipeType.MTE1, event_id=EV_FX_RP)



def _vp2b_prefetch(m2, ctx, u_t, kg_t):
    """2b 预取：chunk r+1 的 u 直读（dense 满块）+ kg 直读。"""
    ut2_l1g = m2[6]
    s2_r = ctx[0]
    s2_cnum = ctx[1]
    s2_last_valid = ctx[2]
    s2_storage = ctx[3]
    s2_head = ctx[4]
    s2_token = ctx[5]
    s2_cstart = ctx[6]
    layout = ctx[8]
    s2_row_off = ctx[9]
    s2_rows = ctx[10]
    s2_r1_is_tail = 0
    if s2_r + 2 == s2_cnum:
        if s2_last_valid < CHUNK:
            s2_r1_is_tail = 1
    if layout != LAYOUT_TND:
        if s2_r1_is_tail == 0:
            pf_row0 = s2_token + (s2_cstart + s2_r + 1) * CHUNK
            utp = ut2_l1g[(s2_r + 1) % 2]
            pl.set_validshape(utp, [s2_rows, CHUNK])
            pl.load(utp, u_t,
                    [s2_storage, s2_head, pf_row0, s2_row_off],
                    order=[3, 2])
    _st2_prefetch_kg(m2, ctx, kg_t)


def _vp2b_vec_round(m2, ctx, u_t, u_pair, ws_zero_ptr):
    """2b vec 轮：R 零检查消费 + TND/尾块的 u(r+1) 装载链。"""
    ub_decayg = m2[17]
    phi_ub2g = m2[7]
    ut2_l1g = m2[6]
    s2_r = ctx[0]
    s2_cnum = ctx[1]
    s2_last_valid = ctx[2]
    layout = ctx[3]
    s2_cstart = ctx[4]
    s2_len = ctx[5]
    s2_storage = ctx[6]
    s2_head = ctx[7]
    s2_token = ctx[8]
    s2_row_off = ctx[9]
    s2_rows = ctx[10]
    s2_chk_cons = ctx[11]
    s2_zword = ctx[12]
    zcur = ub_decayg.current()
    zsrc = phi_ub2g[0]
    dst_u = ut2_l1g[(s2_r + 1) % 2]
    if s2_chk_cons == 1:
        wzv = pl.make_tensor(
            pl.addptr(ws_zero_ptr, s2_zword), [1, D], dtype=pl.DT_FP32)
        pl.system.wait_cross_core(
            pipe=pl.PipeType.V, event_id=EV_WT_REL,
            sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
        _vp_zero_count_vf(zsrc, zcur, s2_rows * 2)
        pl.store(wzv, zcur, [0, 0])
        pl.system.set_cross_core(
            pipe=pl.PipeType.MTE3, event_id=EV_ZERO_READY,
            sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    s2_do_vec = _st2_do_vec_select((s2_r, s2_cnum, s2_last_valid, layout))
    if s2_do_vec == 1:
        if s2_r >= 1:
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE3, event_id=EV_READY_UF,
                sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
        s2_row0t = s2_token + (s2_cstart + s2_r + 1) * CHUNK
        s2_validt = s2_len - (s2_cstart + s2_r + 1) * CHUNK
        if s2_validt > CHUNK:
            s2_validt = CHUNK
        if s2_validt < 0:
            s2_validt = 0
        _u_load_chain(u_pair, u_t, s2_row0t, s2_storage, s2_head,
                      s2_validt, layout, dst_u, s2_row_off, s2_rows)
        pl.system.set_cross_core(
            pipe=pl.PipeType.MTE3, event_id=EV_READY_WU,
            sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)


# ===== Kernel（Modules 1-4：MIX，每任务 4 段交替） =====


class CpKey:
    Layout = TilingKeyField(bits=2, values=[LAYOUT_BNSD, LAYOUT_BSND, LAYOUT_TND])
    # β 装载路径（NvFlat）：0 seq 连续（BNSD）；1 flat slab（nv<=32 一次
    # 直读）；2 32 头跨步窗口（nv>32）——按 m*stride+off 逐行取值累积 sigmoid
    NvFlat = TilingKeyField(bits=2, values=[0, 1, 2])
    # 摘要链的行片数按链数和核数选择。单行片由一个核处理整条摘要链。
    S2DV = TilingKeyField(bits=2, values=[1, 2, 4, 8])
    SharePhi = TilingKeyField(bits=1, values=[0, 1])
    PadDecay = TilingKeyField(bits=1, values=[0, 1])
    ShortSummary = TilingKeyField(bits=1, values=[0, 1])


def _make_summary_memory():
    # Stage 2 tile 组：stR（insert 目的）/ stPsi / φ 暂存 / BF16 操作数
    st_r2 = pl.make_tile_group(
        type=pl.TileType(shape=[D // S2DV, D + 16], dtype=pl.DT_FP32,
                        target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x38000, mutex_ids=[15])
    st_psi2 = pl.make_tile_group(
        type=pl.TileType(shape=[D // S2DV, D + 16], dtype=pl.DT_FP32,
                        target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x4A000, mutex_ids=[15])
    # phi2_l1 双槽：slot-B 借 Stage 1 的 L1 区（sync_all 分相后复用安全）
    phi2_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[D, D], dtype=pl.DT_FP32,
                        target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[0x5C000, 0x10000], mutex_ids=[16, 21])
    # wᵀ/uᵀ：Cube 直读稠密整块及单行片 TND；其余路径经 AIV 转置。
    wt2_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[D, CHUNK], dtype=pl.DT_BF16,
                        target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x6C000, mutex_ids=[12])
    # ut2/kg2 双槽：生命周期跨两轮（prefetch/transpose → rank），单槽会被覆写
    ut2_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[D // S2DV, CHUNK], dtype=pl.DT_BF16,
                        target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x70000, 0x78000], mutex_ids=[12, 18])
    kg2_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, D], dtype=pl.DT_BF16,
                        target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[0x74000, 0x7C000], mutex_ids=[26, 24])
    # 尾块/TND 的 w/u UB 中转：w/u 顺序复用同一对 tile；地址避开 phi 区
    # （vec MTE2 装载与 cube FIX 写跨程序并发，同址会 NaN）
    wu_ub2 = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, D], dtype=pl.DT_BF16,
                        target_memory=pl.MemorySpace.Vec, layout=pl.ND),
        addrs=0x20000, mutex_ids=[6])
    wu_nz2 = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, D], dtype=pl.DT_BF16,
                        target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
        addrs=0x24000, mutex_ids=[6])
    # 单行片的 u 与 w 分时共用转置缓冲。多行片只装本行片的 u，
    # 使用 phi_nz2 后的独立缓冲，按 w/u 装载顺序复用 mutex 6。
    if S2DV == 1:
        u_ub2 = wu_ub2
        u_nz2 = wu_nz2
    else:
        u_ub2 = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK, D // S2DV], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Vec, layout=pl.ND),
            addrs=0x38000, mutex_ids=[6])
        u_nz2 = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK, D // S2DV], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addrs=0x3A000, mutex_ids=[6])
    return pl.make_tuple(
        st_r2=st_r2, st_psi2=st_psi2, phi2_l1=phi2_l1, wt2_l1=wt2_l1, ut2_l1=ut2_l1, kg2_l1=kg2_l1,
        wu_ub2=wu_ub2, wu_nz2=wu_nz2, u_ub2=u_ub2, u_nz2=u_nz2
    )


def _make_summary_math():
    # L0A/L0B/L0C：BF16 chunk 乘与 FP32 rank 乘分相同址
    l0a_pb2 = pl.make_tile_group(
        type=pl.TileType(shape=[D, CHUNK], dtype=pl.DT_BF16,
                        target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x00000, mutex_ids=[22])
    l0a_r2 = pl.make_tile_group(
        type=pl.TileType(shape=[D // S2DV, D], dtype=pl.DT_FP32,
                        target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x00000, mutex_ids=[22])
    l0b_pb2 = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, D], dtype=pl.DT_BF16,
                        target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x00000, mutex_ids=[23])
    l0b_phi2 = pl.make_tile_group(
        type=pl.TileType(shape=[D, D], dtype=pl.DT_FP32,
                        target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x00000, mutex_ids=[23])
    # 2a 双任务批处理的双槽（槽 B 借 phase1 的 uw_sub Acc 区——2a 期已死）
    l0c_phi2 = pl.make_tile_group(
        type=pl.TileType(shape=[D, D], dtype=pl.DT_FP32,
                        target_memory=pl.MemorySpace.Acc, layout=pl.NZ),
        addrs=[0x00000, 0x18000], mutex_ids=[31, 31])
    l0c_r2 = pl.make_tile_group(
        type=pl.TileType(shape=[D // S2DV, D], dtype=pl.DT_FP32,
                        target_memory=pl.MemorySpace.Acc, layout=pl.NZ),
        addrs=0x20000, mutex_ids=[31])
    l0c_psi2 = pl.make_tile_group(
        type=pl.TileType(shape=[D // S2DV, D], dtype=pl.DT_FP32,
                        target_memory=pl.MemorySpace.Acc, layout=pl.NZ),
        addrs=0x30000, mutex_ids=[31])
    # ψacc 左操作数（uᵀ[rows] @ kg）：形状须等于 ut2 的 [rows, CHUNK]——
    # Mat 源 move 无 offset 时要求同形；与 pb2/r2 分时共用 Left 0x0 区
    l0a_ur2 = pl.make_tile_group(
        type=pl.TileType(shape=[D // S2DV, CHUNK], dtype=pl.DT_BF16,
                        target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x00000, mutex_ids=[22])
    # φ ND 暂存 + NZ 中转（ND 须先 UB 内转 NZ）；phi 区与 wu 区互不重叠
    phi_ub2 = pl.make_tile_group(
        type=pl.TileType(shape=[D, D], dtype=pl.DT_FP32,
                        target_memory=pl.MemorySpace.Vec, layout=pl.ND),
        addrs=[0x00000, 0x10000], mutex_ids=[0, 4])
    phi_nz2 = pl.make_tile_group(
        type=pl.TileType(shape=[D, D], dtype=pl.DT_FP32,
                        target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
        addrs=0x28000, mutex_ids=[28])
    return pl.make_tuple(
        l0a_pb2=l0a_pb2, l0a_r2=l0a_r2, l0b_pb2=l0b_pb2, l0b_phi2=l0b_phi2, l0c_phi2=l0c_phi2,
        l0c_r2=l0c_r2, l0c_psi2=l0c_psi2, l0a_ur2=l0a_ur2, phi_ub2=phi_ub2, phi_nz2=phi_nz2
    )


def _make_stage1_score_vectors():
    # ws_km 直通 UB：L0C→UB 分块写（SingleModeVec0 无 offset），读侧整块同址 tile
    _t_km_blk = pl.TileType(shape=[16, CHUNK], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec)
    ub_km_kk_w0 = pl.make_tile_group(type=_t_km_blk, addrs=VA_Q, mutex_ids=[0])
    ub_km_kk_w1 = pl.make_tile_group(type=_t_km_blk, addrs=VA_Q + 4096, mutex_ids=[0])
    ub_km_kk_w2 = pl.make_tile_group(type=_t_km_blk, addrs=VA_Q + 8192, mutex_ids=[0])
    ub_km_kk_w3 = pl.make_tile_group(type=_t_km_blk, addrs=VA_Q + 12288, mutex_ids=[0])
    ub_km_mqk_w0 = pl.make_tile_group(type=_t_km_blk, addrs=VA_K, mutex_ids=[1])
    ub_km_mqk_w1 = pl.make_tile_group(type=_t_km_blk, addrs=VA_K + 4096, mutex_ids=[1])
    ub_km_mqk_w2 = pl.make_tile_group(type=_t_km_blk, addrs=VA_K + 8192, mutex_ids=[1])
    ub_km_mqk_w3 = pl.make_tile_group(type=_t_km_blk, addrs=VA_K + 12288, mutex_ids=[1])
    ub_km_kk = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, CHUNK], dtype=pl.DT_FP32,
                        target_memory=pl.MemorySpace.Vec),
        addrs=VA_Q, mutex_ids=[0])
    ub_km_mqk = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, CHUNK], dtype=pl.DT_FP32,
                        target_memory=pl.MemorySpace.Vec),
        addrs=VA_K, mutex_ids=[1])
    return pl.make_tuple(
        ub_km_kk_w0=ub_km_kk_w0, ub_km_kk_w1=ub_km_kk_w1, ub_km_kk_w2=ub_km_kk_w2, ub_km_kk_w3=ub_km_kk_w3,
        ub_km_mqk_w0=ub_km_mqk_w0, ub_km_mqk_w1=ub_km_mqk_w1, ub_km_mqk_w2=ub_km_mqk_w2,
        ub_km_mqk_w3=ub_km_mqk_w3, ub_km_kk=ub_km_kk, ub_km_mqk=ub_km_mqk
    )


def _make_stage1_k_vectors(t_bf16_64, t_fp32_64, t_fp32_row, t_bf16_row, t_bf16_col):
    # sub1 的 emit 暂存（s1b 阶段复用 qnf/knf 死区）
    ub_outf1 = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, D], dtype=pl.DT_FP32,
                        target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=VA_QNF, mutex_ids=[4])
    ub_stg1 = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, D], dtype=pl.DT_BF16,
                        target_memory=pl.MemorySpace.Vec),
        addrs=VA_KNF, mutex_ids=[5])
    # sub1 的门控累加与 sub0 使用各自的 UB。
    ub_kn1 = pl.make_tile_group(type=t_bf16_64, addrs=VA_KN, mutex_ids=[3])
    ub_gc1 = pl.make_tile_group(type=t_fp32_64, addrs=VA_GC, mutex_ids=[6])
    # sub1 的 k 装载与输入镜像（程序内访问，mutex 3 为内部分时）
    ub_k_s1 = pl.make_tile_group(type=t_bf16_64, addrs=0x30000, mutex_ids=[3])
    ub_bias_s1 = pl.make_tile_group(type=t_fp32_row, addrs=0x3CE00, mutex_ids=[3])
    ub_alog_s1 = pl.make_tile_group(type=t_fp32_row, addrs=0x3D000, mutex_ids=[3])
    ub_beta_row_s1 = pl.make_tile_group(type=t_bf16_row, addrs=0x3D400, mutex_ids=[3])
    ub_beta_col_s1 = pl.make_tile_group(type=t_bf16_col, addrs=0x3D600, mutex_ids=[3])
    # σ(β) 槽：flat slab 路径按 m 逐元素写（DN 槽 per-offset store 会越 UB 尾）
    ub_beta_s1 = pl.make_tile_group(
        type=pl.TileType(shape=[1, CHUNK], dtype=pl.DT_FP32,
                        target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=0x3D800, mutex_ids=[3])
    ub_gc = pl.make_tile_group(type=t_fp32_64, addrs=VA_GC, mutex_ids=[6])
    return pl.make_tuple(
        ub_outf1=ub_outf1, ub_stg1=ub_stg1, ub_kn1=ub_kn1, ub_gc1=ub_gc1, ub_k_s1=ub_k_s1,
        ub_bias_s1=ub_bias_s1, ub_alog_s1=ub_alog_s1, ub_beta_row_s1=ub_beta_row_s1,
        ub_beta_col_s1=ub_beta_col_s1, ub_beta_s1=ub_beta_s1, ub_gc=ub_gc
    )


def _make_stage1_q_vectors(t_fp32_64v, t_bf16_64, t_fp32_64sq, t_bf16_row, t_bf16_col):
    # outf/stg 复用 qnf/knf 区；同址组必须共享 mutex，否则框架可能重排
    ub_outf = pl.make_tile_group(type=t_fp32_64v, addrs=0x10000, mutex_ids=[4])
    ub_stg = pl.make_tile_group(type=t_bf16_64, addrs=0x18000, mutex_ids=[5])
    ub_tf = pl.make_tile_group(type=t_fp32_64sq, addrs=VA_TF, mutex_ids=[10])
    # T/seed 直通 L1 的 NZ 中转。
    ub_ts16_nz = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, CHUNK], dtype=pl.DT_FP16,
                        target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
        addrs=0x34000, mutex_ids=[9])
    ub_beta_row = pl.make_tile_group(type=t_bf16_row, addrs=VA_BETA, mutex_ids=[11])
    ub_beta_col = pl.make_tile_group(type=t_bf16_col, addrs=VA_BETA, mutex_ids=[11])
    # ub_beta 双槽：s1ab(r+1) 写与 s3(r) 读并发，两槽共享 mutex 12
    ub_beta = pl.make_tile_group(
        type=pl.TileType(shape=[1, CHUNK], dtype=pl.DT_FP32,
                        target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=[0x3CC00, 0x3CD00], mutex_ids=[12, 12])
    # β flat slab（NvFlat=1）：sub0 双槽（奇偶分槽）+ sub1 单槽，UB 尾部空闲区
    t_beta_slab = pl.TileType(shape=[1, 2048], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Vec, layout=pl.ND,
                            valid_shape=[-1, -1])
    ub_beta_slab = pl.make_tile_group(
        type=t_beta_slab, addrs=0x3DC00, mutex_ids=[12])
    ub_beta_slab1 = pl.make_tile_group(
        type=t_beta_slab, addrs=0x3EC00, mutex_ids=[3])
    # nv>32 的 32 头窗口 tile（NvFlat=2 变体；与 slab 同址互斥）
    t_beta_w32 = pl.TileType(shape=[64, 32], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Vec, layout=pl.ND,
                            valid_shape=[-1, -1])
    ub_beta_w32 = pl.make_tile_group(
        type=t_beta_w32, addrs=0x3DC00, mutex_ids=[12])
    ub_beta_w321 = pl.make_tile_group(
        type=t_beta_w32, addrs=0x3EC00, mutex_ids=[3])
    # s1b decay emit 专用暂存（与 outf 分区，与 beta_row/col 共享 mutex 11）
    return pl.make_tuple(
        ub_outf=ub_outf, ub_stg=ub_stg, ub_tf=ub_tf, ub_ts16_nz=ub_ts16_nz, ub_beta_row=ub_beta_row,
        ub_beta_col=ub_beta_col, ub_beta=ub_beta, ub_beta_slab=ub_beta_slab,
        ub_beta_slab1=ub_beta_slab1, ub_beta_w32=ub_beta_w32, ub_beta_w321=ub_beta_w321
    )


def _make_stage1_result_vectors(t_fp32_row, t_fp32_64sq, t_bf16_64sq):
    ub_decayf = pl.make_tile_group(type=t_fp32_row, addrs=0x3CA00, mutex_ids=[11])
    # [1,D] tile valid 动态(-1)：store 前必须 set_validshape，否则读回垃圾
    ub_decay = pl.make_tile_group(type=t_fp32_row, addrs=VA_DECAY, mutex_ids=[13])
    ub_bias = pl.make_tile_group(type=t_fp32_row, addrs=VA_BIAS, mutex_ids=[14])
    ub_alog = pl.make_tile_group(type=t_fp32_row, addrs=VA_ALOG, mutex_ids=[15])
    ub_seedf = pl.make_tile_group(type=t_fp32_64sq, addrs=VA_SEEDF, mutex_ids=[5])
    ub_mqkbf = pl.make_tile_group(type=t_bf16_64sq, addrs=VA_MQKBF, mutex_ids=[5])
    # Neumann FP16 staging（与 ub_mqkbf 同址分时，共享 mutex）
    ub_ts16 = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, CHUNK], dtype=pl.DT_FP16,
                        target_memory=pl.MemorySpace.Vec),
        addrs=VA_MQKBF, mutex_ids=[5])
    ub_mqkf = pl.make_tile_group(type=t_fp32_64sq, addrs=VA_MQKF, mutex_ids=[4])
    return pl.make_tuple(
        ub_decayf=ub_decayf, ub_decay=ub_decay, ub_bias=ub_bias, ub_alog=ub_alog, ub_seedf=ub_seedf,
        ub_mqkbf=ub_mqkbf, ub_ts16=ub_ts16, ub_mqkf=ub_mqkf
    )


def _make_stage1_score_cube():
    # ---- Cube 侧 L1/L0 TileGroup（s2/s5 操作数 BF16；L0C 恒 FP32）----
    t_l1_rhs = pl.TileType(shape=[D, CHUNK], dtype=pl.DT_BF16,
                        target_memory=pl.MemorySpace.Mat, layout=pl.ZN,
                        valid_shape=[-1, -1])
    l1_rhs = pl.make_tile_group(type=t_l1_rhs, addrs=0x08000, mutex_ids=[21])
    # LEFT_K/LEFT_Q 评分面板（s1b 直写 L1，s2 免 GM ws 往返）：借 phase2 的
    # phase2 的 wt2/ut2/kg2 区（0x6E000-0x7E000），经 sync_all 后分相复用；
    # mutex 26/7 在 phase1 期空闲（26=kg2 槽 A、7=无占用）。lk 归 sub1 写、
    # lq 归 sub0 写（单 sub 原则）；8 槽 = 奇偶(task_slot%2)×4 rblk
    t_l1_panel = pl.TileType(shape=[16, D], dtype=pl.DT_BF16,
                        target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                        valid_shape=[-1, -1])
    l1_plk = pl.make_tile_group(
        type=t_l1_panel,
        addrs=[0x6E000, 0x6F000, 0x70000, 0x71000,
               0x72000, 0x73000, 0x74000, 0x75000],
        mutex_ids=[26, 26, 26, 26, 26, 26, 26, 26])
    l1_plq = pl.make_tile_group(
        type=t_l1_panel,
        addrs=[0x76000, 0x77000, 0x78000, 0x79000,
               0x7A000, 0x7B000, 0x7C000, 0x7D000],
        mutex_ids=[7, 7, 7, 7, 7, 7, 7, 7])
    # 面板的 UB NZ 中转（[16,D] 与面板同形——收窄 move 的源/目的类型
    # 必须对齐；sub0/sub1 各一，0x38000/0x3A000 在 phase1 期空闲）
    t_ub_lnz = pl.TileType(shape=[16, D], dtype=pl.DT_BF16,
                        target_memory=pl.MemorySpace.Vec, layout=pl.NZ)
    ub_lnz = pl.make_tile_group(type=t_ub_lnz, addrs=0x38000,
                        mutex_ids=[10])
    ub_lnz1 = pl.make_tile_group(type=t_ub_lnz, addrs=0x3A000,
                        mutex_ids=[10])
    t_l0a = pl.TileType(shape=[16, D], dtype=pl.DT_BF16,
                        target_memory=pl.MemorySpace.Left, layout=pl.NZ)
    t_l0b = pl.TileType(shape=[D, CHUNK], dtype=pl.DT_BF16,
                        target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                        valid_shape=[-1, -1])
    t_l0c_km = pl.TileType(shape=[16, CHUNK], dtype=pl.DT_FP32,
                        target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                        valid_shape=[-1, -1])
    l0a_km = pl.make_tile_group(type=t_l0a, addrs=0x00000, mutex_ids=[22])
    l0b_km = pl.make_tile_group(type=t_l0b, addrs=0x00000, mutex_ids=[23])
    # mqk 专用 L0A：与 l0a_km 分址（kk/mqk 操作数独立，M 管线可连续排队）
    l0a_mqk = pl.make_tile_group(type=t_l0a, addrs=0x04000, mutex_ids=[22])
    l0c_km = pl.make_tile_group(type=t_l0c_km, addrs=0x00000, mutex_ids=[31])
    # mqk 专用 L0C：双 tile 消除 kk/mqk 间的 bar_m（两 matmul 各写各的，
    # M 管线连续执行，bar_m 仅需一次覆盖双结果）
    l0c_mqk = pl.make_tile_group(type=t_l0c_km, addrs=0x04000, mutex_ids=[31])

    return pl.make_tuple(
        l1_rhs=l1_rhs, l1_plk=l1_plk, l1_plq=l1_plq, ub_lnz=ub_lnz, ub_lnz1=ub_lnz1, l0a_km=l0a_km,
        l0b_km=l0b_km, l0a_mqk=l0a_mqk, l0c_km=l0c_km, l0c_mqk=l0c_mqk
    )


def _make_stage1_inverse_cube():
    # Neumann 区 FP16；M/p 的 L0C→L1 回写走 insert 降精度
    t_l1_n = pl.TileType(shape=[CHUNK, CHUNK], dtype=pl.DT_FP16,
                        target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                        valid_shape=[-1, -1])
    # l1_t/l1_seed 双槽（slot-B 在 L1 空闲区），s3 写与 s4(r-1) 读并发
    l1_t = pl.make_tile_group(type=t_l1_n, addrs=[0x10000, 0x24000], mutex_ids=[25, 18])
    l1_seed = pl.make_tile_group(type=t_l1_n, addrs=[0x14000, 0x2C000], mutex_ids=[24, 19])
    # 均为 s4 cube 程序内分时——id/m 共 27、v 共 27：无跨程序假冲突
    l1_id = pl.make_tile_group(type=t_l1_n, addrs=0x1C000, mutex_ids=[27])
    # fixpipe（L0C→L1 超块 insert）回写 tile：目的 shape 必须大于源 [64,64]
    t_l1_nbig = pl.TileType(shape=[CHUNK, 2 * CHUNK], dtype=pl.DT_FP16,
                            target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
    l1_m = pl.make_tile_group(type=t_l1_nbig, addrs=0x28000, mutex_ids=[27])
    l1_pow = pl.make_tile_group(type=t_l1_nbig, addrs=0x30000, mutex_ids=[17])
    t_l0a_n = pl.TileType(shape=[CHUNK, CHUNK], dtype=pl.DT_FP16,
                        target_memory=pl.MemorySpace.Left, layout=pl.NZ)
    t_l0b_n = pl.TileType(shape=[CHUNK, CHUNK], dtype=pl.DT_FP16,
                        target_memory=pl.MemorySpace.Right, layout=pl.ZN)
    t_l0c_n = pl.TileType(shape=[CHUNK, CHUNK], dtype=pl.DT_FP32,
                        target_memory=pl.MemorySpace.Acc, layout=pl.NZ)
    l0a_n = pl.make_tile_group(type=t_l0a_n, addrs=0x08000, mutex_ids=[22])
    # Right 64KB 预算：l0b_km(32KB)@0 与 l0b_v(32KB)@0 同址分时（s2 完成后 s4 复用）
    l0b_n = pl.make_tile_group(type=t_l0b_n, addrs=0x08000, mutex_ids=[23])
    l0c_n = pl.make_tile_group(type=t_l0c_n, addrs=0x08000, mutex_ids=[31])
    t_l0b_id = pl.TileType(shape=[CHUNK, CHUNK], dtype=pl.DT_FP16,
                        target_memory=pl.MemorySpace.Right, layout=pl.ZN)
    l0b_id = pl.make_tile_group(type=t_l0b_id, addrs=0x0C000, mutex_ids=[23])
    l0c_pow = pl.make_tile_group(type=t_l0c_n, addrs=0x14000, mutex_ids=[31])
    return pl.make_tuple(
        l1_t=l1_t, l1_seed=l1_seed, l1_id=l1_id, l1_m=l1_m, l1_pow=l1_pow, l0a_n=l0a_n, l0b_n=l0b_n,
        l0c_n=l0c_n, l0b_id=l0b_id, l0c_pow=l0c_pow
    )


def _make_stage1_uw_cube():
    t_l0b_v = pl.TileType(shape=[CHUNK, D], dtype=pl.DT_BF16,
                        target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                        valid_shape=[-1, -1])
    l1_v = pl.make_tile_group(type=pl.TileType(shape=[CHUNK, D], dtype=pl.DT_BF16,
                                            target_memory=pl.MemorySpace.Mat,
                                            layout=pl.NZ, valid_shape=[-1, -1]),
                            addrs=0x20000, mutex_ids=[27])
    # βK̃ 独立 tile：与 l1_rhs 同址分时（0x8000，s2 已完成），共享 mutex
    l1_bkt = pl.make_tile_group(type=pl.TileType(shape=[CHUNK, D], dtype=pl.DT_BF16,
                                                target_memory=pl.MemorySpace.Mat,
                                                layout=pl.NZ, valid_shape=[-1, -1]),
                                addrs=0x08000, mutex_ids=[21])
    # Ainv BF16：Neumann 末 M 经 insert 降精度落入（l1_t/seed/id 区空隙）
    l1_a16 = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, 2 * CHUNK], dtype=pl.DT_BF16,
                        target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                        valid_shape=[-1, -1]),
        addrs=0x18000, mutex_ids=[30])
    # 与 l0b_km 同址分时，须共享 mutex——防下任务 s2 覆写未读完的 l0b_v
    l0b_v = pl.make_tile_group(type=t_l0b_v, addrs=0x00000, mutex_ids=[23])
    # s5 的 u/w 左操作数（BF16 Ainv）：与 l0a_n 同址分时（0x8000，s4 已完成）
    l0a_u16 = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK, CHUNK], dtype=pl.DT_BF16,
                        target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x08000, mutex_ids=[22])
    t_l0c_uw = pl.TileType(shape=[CHUNK, D], dtype=pl.DT_FP32,
                        target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                        valid_shape=[-1, -1])
    l0c_uw = pl.make_tile_group(type=t_l0c_uw, addrs=0x18000, mutex_ids=[31])
    l0c_w = pl.make_tile_group(type=t_l0c_uw, addrs=0x0C000, mutex_ids=[31])
    return pl.make_tuple(
        l1_v=l1_v, l1_bkt=l1_bkt, l1_a16=l1_a16, l0b_v=l0b_v, l0a_u16=l0a_u16, l0c_uw=l0c_uw, l0c_w=l0c_w
    )


def _make_stage1_uw_stores():
    # u/w 的 32×[16,16] 单分形子 tile：多 N 分形直写须覆盖完整 M 轴（store
    # 约束），行收窄唯一合规形态=单分形子块（valid m_f==tile m_f==1）。
    # Acc 为 ColMajor——fractal(m,n) 位于 (n*4+m)*1KB，组内索引
    # i=m*8+n 对应 addr=base+(n*4+m)*1024
    t_sub16 = pl.TileType(shape=[16, 16], dtype=pl.DT_FP32,
                        target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                        valid_shape=[-1, -1])
    uw_sub = pl.make_tile_group(
        type=t_sub16,
        addrs=[0x18000, 0x19000, 0x1A000, 0x1B000, 0x1C000, 0x1D000, 0x1E000, 0x1F000,
            0x18400, 0x19400, 0x1A400, 0x1B400, 0x1C400, 0x1D400, 0x1E400, 0x1F400,
            0x18800, 0x19800, 0x1A800, 0x1B800, 0x1C800, 0x1D800, 0x1E800, 0x1F800,
            0x18C00, 0x19C00, 0x1AC00, 0x1BC00, 0x1CC00, 0x1DC00, 0x1EC00, 0x1FC00],
        mutex_ids=[31, 31, 31, 31, 31, 31, 31, 31,
                31, 31, 31, 31, 31, 31, 31, 31,
                31, 31, 31, 31, 31, 31, 31, 31,
                31, 31, 31, 31, 31, 31, 31, 31])
    w_sub = pl.make_tile_group(
        type=t_sub16,
        addrs=[0x0C000, 0x0D000, 0x0E000, 0x0F000, 0x10000, 0x11000, 0x12000, 0x13000,
            0x0C400, 0x0D400, 0x0E400, 0x0F400, 0x10400, 0x11400, 0x12400, 0x13400,
            0x0C800, 0x0D800, 0x0E800, 0x0F800, 0x10800, 0x11800, 0x12800, 0x13800,
            0x0CC00, 0x0DC00, 0x0EC00, 0x0FC00, 0x10C00, 0x11C00, 0x12C00, 0x13C00],
        mutex_ids=[31, 31, 31, 31, 31, 31, 31, 31,
                31, 31, 31, 31, 31, 31, 31, 31,
                31, 31, 31, 31, 31, 31, 31, 31,
                31, 31, 31, 31, 31, 31, 31, 31])
    return pl.make_tuple(
        uw_sub=uw_sub, w_sub=w_sub
    )


def _make_kernel_tiles():
    summary_memory = _make_summary_memory()
    summary_math = _make_summary_math()
    t_bf16_64 = pl.TileType(shape=[CHUNK, D], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    t_fp32_64 = pl.TileType(shape=[CHUNK, D], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec)
    t_fp32_64v = pl.TileType(shape=[CHUNK, D], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    t_bf16_row = pl.TileType(shape=[1, CHUNK], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    t_bf16_col = pl.TileType(shape=[CHUNK, 1], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    t_fp32_row = pl.TileType(shape=[1, D], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec)
    t_fp32_64sq = pl.TileType(shape=[CHUNK, CHUNK], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    t_bf16_64sq = pl.TileType(shape=[CHUNK, CHUNK], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Vec)

    # ub_q/ub_k 在 0x28000/0x2C000（避开 km FIX 写区，跨程序并发）
    ub_q = pl.make_tile_group(type=t_bf16_64, addrs=0x28000, mutex_ids=[28])
    stage1_score_vectors = _make_stage1_score_vectors()
    ub_qn = pl.make_tile_group(type=t_bf16_64, addrs=VA_QN, mutex_ids=[2])
    ub_qnf = pl.make_tile_group(type=t_fp32_64, addrs=VA_QNF, mutex_ids=[4])
    stage1_k_vectors = _make_stage1_k_vectors(t_bf16_64, t_fp32_64, t_fp32_row, t_bf16_row, t_bf16_col)
    stage1_q_vectors = _make_stage1_q_vectors(t_fp32_64v, t_bf16_64, t_fp32_64sq, t_bf16_row, t_bf16_col)
    stage1_result_vectors = _make_stage1_result_vectors(t_fp32_row, t_fp32_64sq, t_bf16_64sq)
    stage1_score_cube = _make_stage1_score_cube()
    stage1_inverse_cube = _make_stage1_inverse_cube()
    stage1_uw_cube = _make_stage1_uw_cube()
    stage1_uw_stores = _make_stage1_uw_stores()
    # 2a has two live wᵀ tiles. Slot B borrows the φ L1 region used in 2b.
    vp_wt2 = pl.make_tile_group(
        type=pl.TileType(shape=[D, CHUNK], dtype=pl.DT_BF16,
                        target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x6C000, 0x60000], mutex_ids=[12, 16])
    return _group_kernel_tiles(
        (summary_memory, summary_math, vp_wt2),
        (stage1_score_vectors, stage1_k_vectors, stage1_q_vectors, stage1_result_vectors),
        (stage1_score_cube, stage1_inverse_cube, stage1_uw_cube, stage1_uw_stores),
        (ub_q, ub_qn, ub_qnf))


def _group_kernel_tiles(summary, vectors, cube, q_tiles):
    summary_memory, summary_math, vp_wt2 = summary
    stage1_score_vectors, stage1_k_vectors, stage1_q_vectors, stage1_result_vectors = vectors
    stage1_score_cube, stage1_inverse_cube, stage1_uw_cube, stage1_uw_stores = cube
    ub_q, ub_qn, ub_qnf = q_tiles
    m2 = (
        summary_memory.st_r2, summary_memory.st_psi2, summary_memory.kg2_l1, summary_memory.wu_ub2,
        summary_memory.wu_nz2, summary_memory.wt2_l1, summary_memory.ut2_l1,
        summary_math.phi_ub2, summary_math.phi_nz2, summary_memory.phi2_l1, summary_math.l0a_pb2,
        summary_math.l0b_pb2, summary_math.l0c_phi2,
        summary_math.l0b_phi2, summary_math.l0a_r2, summary_math.l0c_r2, summary_math.l0c_psi2,
        stage1_result_vectors.ub_decay, summary_math.l0a_ur2, vp_wt2)
    u_pair = (summary_memory.u_ub2, summary_memory.u_nz2)
    m1c = (
        stage1_score_cube.l1_rhs, stage1_score_cube.l0a_km,
        stage1_score_cube.l0b_km, stage1_score_cube.l0c_km,
        stage1_score_vectors.ub_km_kk_w0,
        stage1_score_vectors.ub_km_mqk_w0, stage1_inverse_cube.l1_id, stage1_inverse_cube.l1_t,
        stage1_inverse_cube.l1_seed, stage1_inverse_cube.l1_m, stage1_inverse_cube.l1_pow,
        stage1_inverse_cube.l0a_n,
        stage1_inverse_cube.l0b_id, stage1_inverse_cube.l0b_n, stage1_inverse_cube.l0c_n,
        stage1_inverse_cube.l0c_pow,
        stage1_uw_cube.l1_v, stage1_uw_cube.l1_bkt, stage1_uw_cube.l0a_u16, stage1_uw_cube.l0b_v,
        stage1_uw_cube.l0c_uw, stage1_uw_cube.l0c_w, stage1_uw_cube.l1_a16,
        stage1_uw_stores.uw_sub, stage1_uw_stores.w_sub,
        stage1_score_cube.l1_plk, stage1_score_cube.l1_plq, stage1_score_cube.l0c_mqk,
        stage1_score_cube.l0a_mqk)
    m1v0 = (
        ub_q, ub_qn, ub_qnf, stage1_result_vectors.ub_bias, stage1_result_vectors.ub_alog,
        stage1_k_vectors.ub_gc, stage1_q_vectors.ub_beta_slab,
        stage1_q_vectors.ub_beta_w32, stage1_q_vectors.ub_beta_row, stage1_q_vectors.ub_beta_col,
        stage1_q_vectors.ub_beta, stage1_q_vectors.ub_outf, stage1_q_vectors.ub_stg,
        stage1_result_vectors.ub_decayf, stage1_score_vectors.ub_km_kk, stage1_q_vectors.ub_tf,
        stage1_result_vectors.ub_seedf, stage1_result_vectors.ub_ts16,
        stage1_q_vectors.ub_ts16_nz, stage1_result_vectors.ub_mqkbf, stage1_score_vectors.ub_km_mqk,
        stage1_result_vectors.ub_mqkf, stage1_inverse_cube.l1_t, stage1_inverse_cube.l1_seed,
        stage1_score_cube.ub_lnz, stage1_score_cube.l1_plq)
    m1v1 = (
        stage1_k_vectors.ub_k_s1, stage1_k_vectors.ub_kn1, stage1_k_vectors.ub_outf1,
        stage1_k_vectors.ub_bias_s1, stage1_k_vectors.ub_gc1, stage1_k_vectors.ub_alog_s1,
        stage1_q_vectors.ub_beta_slab1, stage1_q_vectors.ub_beta_w321, stage1_k_vectors.ub_beta_row_s1,
        stage1_k_vectors.ub_beta_col_s1,
        stage1_k_vectors.ub_beta_s1, stage1_k_vectors.ub_stg1, stage1_score_cube.ub_lnz1,
        stage1_score_cube.l1_plk)
    return pl.make_tuple(
        summary=m2, summary_u=u_pair, cube=m1c, vec0=m1v0, vec1=m1v1,
        scores=(stage1_score_vectors.ub_km_kk_w1, stage1_score_vectors.ub_km_kk_w2,
                stage1_score_vectors.ub_km_kk_w3, stage1_score_vectors.ub_km_mqk_w1,
                stage1_score_vectors.ub_km_mqk_w2, stage1_score_vectors.ub_km_mqk_w3))


def _stage1_vec_q(m1v0, batch, inputs, outputs, workspace, params, core_id, i):
    layout, use_flat, seq_len, nv, nqk, capacity, scale, lower_bound, _, pad_decay = params
    logical_batch = batch.logical_batch
    storage_batch = batch.storage_batch
    chunks_total = batch.chunks_total
    chunk, head, valid_rows, row0, task_slot = _task_ctx(
        batch.task_start, batch.prefix_start, batch.group_count, batch.group_start,
                  batch.chunks_total, batch.tail_rows, batch.token_start, i)
    qk_head = head // (nv // nqk)
    row0_out = row0
    _s1a0_body(
        m1v0,
        (storage_batch, qk_head,
         head, row0, valid_rows,
         use_flat, seq_len, nv,
         i, layout),
        inputs.q_t, inputs.g_t, inputs.v_t, inputs.b_t,
        inputs.alog_t, inputs.bias_t, inputs.beta_ptr,
        lower_bound)
    _s1b0_body(
        m1v0,
        (storage_batch, head,
         row0_out, valid_rows,
         task_slot, core_id,
         0,
         logical_batch, chunk, i,
         layout),
        outputs.qg_t, outputs.decay_t,
        workspace.ws_ops16_ptr, scale)
    if pad_decay == 1:
        if chunk + 1 == chunks_total:
            # 每条链的末 chunk 负责其未用的 decay 槽。
            pad_row = m1v0[13].current()
            pl.set_validshape(pad_row, [1, D])
            pl.load(pad_row, workspace.zeros128_view, [0, 0])
            for pad_chunk in pl.range(chunks_total, capacity):
                _st_decay(outputs.decay_t, pad_row, logical_batch,
                          head, pad_chunk)


def _stage1_vec_mqk(m1v0, batch, outputs, i, task_count):
    logical_batch = batch.logical_batch
    chunk, head, s3_valid_rows, s3_row0, s3_task_slot = _task_ctx(
        batch.task_start, batch.prefix_start, batch.group_count, batch.group_start,
                  batch.chunks_total, batch.tail_rows, batch.token_start, i)
    _s3_body(
        m1v0,
        (i, task_count,
         0,
         logical_batch, head,
         chunk),
        outputs.mqk_t)


def _stage1_vec_k(m1v1, batch, inputs, outputs, workspace, params, core_id, i):
    layout, use_flat, seq_len, nv, nqk, _, _, lower_bound, _, _ = params
    storage_batch = batch.storage_batch
    s1_chunk, s1_head, s1_valid_rows, s1_row0, s1_task_slot = _task_ctx(
        batch.task_start, batch.prefix_start, batch.group_count, batch.group_start,
                  batch.chunks_total, batch.tail_rows, batch.token_start, i)
    s1_qk_head = s1_head // (nv // nqk)
    s1_row0_out = s1_row0
    _s1a1_body(
        m1v1,
        (storage_batch,
         s1_qk_head, s1_head,
         s1_row0, s1_valid_rows,
         use_flat, seq_len, nv,
         layout),
        inputs.k_t, inputs.g_t, inputs.b_t, inputs.alog_t,
        inputs.bias_t, inputs.beta_ptr,
        lower_bound)
    _s1b1_body(
        m1v1,
        (storage_batch, s1_head,
         s1_row0_out,
         s1_valid_rows,
         s1_task_slot, core_id,
         0, layout),
        outputs.kg_t,
        workspace.ws_ops16_ptr)


def _stage1_cube_score(m1c, batch, score_tiles, workspace, i):
    (ub_km_kk_w1, ub_km_kk_w2, ub_km_kk_w3, ub_km_mqk_w1, ub_km_mqk_w2, ub_km_mqk_w3) = score_tiles
    chunk, head, valid_rows, row0, task_slot = _task_ctx(
        batch.task_start, batch.prefix_start, batch.group_count, batch.group_start,
                  batch.chunks_total, batch.tail_rows, batch.token_start, i)
    # ================= s2：Cube kkt/mqk（rblk 0-3）=================
    _s2_km_head(m1c, (task_slot,), workspace.ws_ops16_ptr)
    _s2_km_rblk(m1c, (task_slot, 2048, 1),
                ub_km_kk_w1, ub_km_mqk_w1,
                workspace.ws_ops16_ptr)
    _s2_km_rblk(m1c, (task_slot, 6144, 2),
                ub_km_kk_w2, ub_km_mqk_w2,
                workspace.ws_ops16_ptr)
    _s2_km_rblk(m1c, (task_slot, 12288, 3),
                ub_km_kk_w3, ub_km_mqk_w3,
                workspace.ws_ops16_ptr)
    pl.system.set_cross_core(
        pipe=pl.PipeType.FIX, event_id=EV_READY_KM,
        sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)


def _stage1_cube_uw(m1c, batch, outputs, workspace, core_id, i, layout, neu_iters):
    storage_batch = batch.storage_batch
    chunk, head, valid_rows, row0, task_slot = _task_ctx(
        batch.task_start, batch.prefix_start, batch.group_count, batch.group_start,
                  batch.chunks_total, batch.tail_rows, batch.token_start, i)
    row0_out = row0
    # ================= s4/s5：Neumann + u/w 落盘 =====
    _s4_seed(m1c, (i,), workspace.id64_f16_view)
    _s4_neumann(m1c, (neu_iters,))
    _s5_u(m1c,
           (task_slot, core_id, storage_batch,
            head, row0_out, valid_rows, layout),
           outputs.u_t, workspace.ws_ops16_ptr)
    _s5_w(m1c,
          (task_slot, core_id, storage_batch,
           head, row0_out, valid_rows, layout),
          outputs.w_t)


def _stage1_vector_round(batch, inputs, outputs, stage1_tiles, workspace, params, core_id, r, task_count):
    _, m1v0, m1v1 = stage1_tiles
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        if sub_id == 0:
            if r < task_count:
                _stage1_vec_q(m1v0, batch, inputs, outputs, workspace, params, core_id, r)
            if r >= 1:
                _stage1_vec_mqk(m1v0, batch, outputs, r - 1, task_count)
        if sub_id == 1:
            if r < task_count:
                _stage1_vec_k(m1v1, batch, inputs, outputs, workspace, params, core_id, r)


def _stage1_batch(batch, inputs, outputs, stage1_tiles, score_tiles, workspace, params, core_id, task_count):
    m1c, _, _ = stage1_tiles
    layout, _, _, _, _, _, _, _, neu_iters, _ = params
    # bootstrap：km 槽初始 set；task_count==0 整段不进（无未消费 set）
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        if sub_id == 0:
            pl.system.set_cross_core(
                pipe=pl.PipeType.V, event_id=EV_KM_KK_REL,
                sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
            pl.system.set_cross_core(
                pipe=pl.PipeType.V, event_id=EV_KM_MQK_REL,
                sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    # 错位流水：vec=[s1ab(r)][s3(r-1)]，cube=[s2(r)][s4(r-1)]，末轮仅收尾；事件逐边配平
    for r in pl.range(0, task_count + 1):
        _stage1_vector_round(batch, inputs, outputs, stage1_tiles, workspace, params, core_id, r, task_count)
        with pl.section_cube():
            if r < task_count:
                _stage1_cube_score(m1c, batch, score_tiles, workspace, r)
            if r >= 1:
                _stage1_cube_uw(m1c, batch, outputs, workspace, core_id, r - 1, layout, neu_iters)


def _run_stage1_round(schedule, inputs, outputs, stage1_tiles, score_tiles, workspace, params):
    (metadata, batch_base, storage_base, token_base, length_base, chunk_start_base, chunk_num_base,
     active_batch_num, prefix_base, core_id, core_start, core_end) = schedule
    for active_idx in pl.range(0, active_batch_num):
        prefix_start = pl.getval(metadata, prefix_base + active_idx)
        prefix_end = pl.getval(metadata, prefix_base + active_idx + 1)
        task_start = core_start
        if task_start < prefix_start:
            task_start = prefix_start
        task_end = core_end
        if task_end > prefix_end:
            task_end = prefix_end
        task_count = task_end - task_start
        if task_count > 0:
            logical_batch = pl.getval(metadata, batch_base + active_idx)
            storage_batch = pl.getval(metadata, storage_base + active_idx)
            token_start = pl.getval(metadata, token_base + active_idx)
            valid_len = pl.getval(metadata, length_base + active_idx)
            group_start = pl.getval(metadata, chunk_start_base + active_idx)
            group_count = pl.getval(metadata, chunk_num_base + active_idx)
            chunks_total = (valid_len + CHUNK - 1) // CHUNK
            tail_rows = valid_len - (chunks_total - 1) * CHUNK
            batch = pl.make_tuple(
                task_start=task_start, prefix_start=prefix_start, group_count=group_count, group_start=group_start,
                chunks_total=chunks_total, tail_rows=tail_rows, token_start=token_start,
                logical_batch=logical_batch, storage_batch=storage_batch
            )
            _stage1_batch(batch, inputs, outputs, stage1_tiles, score_tiles, workspace,
                          params, core_id, task_count)


def _vp2a_task(batch, idx):
    return _task_ctx(batch.task_start, batch.prefix_start, batch.group_count,
                     batch.group_start, batch.chunks_total, batch.tail_rows, batch.token_start, idx)


def _vp2a_store_one(m2, batch, tensors, idx, nv, capacity):
    chunk, head, valid_rows, row0, task_slot = _vp2a_task(batch, idx)
    _vp2a_diag_store(m2, (batch.logical_batch, head, chunk, idx, nv, capacity),
                    tensors.decay_t, tensors.ws_phi_ptr)


def _vp2a_transpose_one(m2, batch, tensors, idx, layout):
    chunk, head, valid_rows, row0, task_slot = _vp2a_task(batch, idx)
    need_wu = 0
    if layout == LAYOUT_TND:
        need_wu = 1
    if valid_rows < CHUNK:
        need_wu = 1
    if need_wu == 1:
        _vp2a_w_chain(m2, (batch.storage_batch, head, row0, valid_rows, layout, idx), tensors.w_t)


def _vp2a_product_one(m2, batch, tensors, idx, layout, slot):
    chunk, head, valid_rows, row0, task_slot = _vp2a_task(batch, idx)
    need_wu = 0
    if layout == LAYOUT_TND:
        need_wu = 1
    if valid_rows < CHUNK:
        need_wu = 1
    _vp2a_cube(m2, (batch.storage_batch, head, row0, valid_rows, layout, need_wu),
               tensors.w_t, tensors.kg_t, idx, slot)


def _vp2a_vector_pair(m2, batch, tensors, params, vp_r0, vp_r1, vp_task_count):
    nv, layout, capacity = params
    # 消费上一对的 φ；这一对的产物在下一轮写回。
    vp_d0 = vp_r0 - 2
    vp_d1 = vp_r0 - 1
    if vp_d0 >= 0:
        pl.system.wait_cross_core(
            pipe=pl.PipeType.V, event_id=EV_READY_TS,
            sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    if vp_d0 >= 0:
        _vp2a_store_one(m2, batch, tensors, vp_d0, nv, capacity)
    if vp_d1 < vp_task_count:
        if vp_d1 >= 0:
            _vp2a_store_one(m2, batch, tensors, vp_d1, nv, capacity)
    if vp_d0 >= 0:
        if vp_r0 < vp_task_count:
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE3, event_id=EV_PHI_RELEASE,
                sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    if vp_r0 < vp_task_count:
        _vp2a_transpose_one(m2, batch, tensors, vp_r0, layout)
    if vp_r1 < vp_task_count:
        _vp2a_transpose_one(m2, batch, tensors, vp_r1, layout)


def _vp2a_cube_pair(m2, batch, tensors, params, vp_r0, vp_r1, vp_task_count):
    _, layout, _ = params
    if vp_r0 < vp_task_count:
        _vp2a_product_one(m2, batch, tensors, vp_r0, layout, 0)
    if vp_r1 < vp_task_count:
        _vp2a_product_one(m2, batch, tensors, vp_r1, layout, 1)
    if vp_r0 < vp_task_count:
        if vp_r0 > 0:
            pl.system.wait_cross_core(
                pipe=pl.PipeType.FIX, event_id=EV_PHI_RELEASE,
                sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
        _vp2a_flush(m2, vp_r0, vp_r1, vp_task_count)


def _vp2a_batch(m2, batch, tensors, params, vp_task_count):
    # 每对共享一次 M→FIX 同步和一次 TS 发布；末轮只回写前一对。
    for vp_r2 in pl.range(0, (vp_task_count + 3) // 2):
        vp_r0 = vp_r2 * 2
        vp_r1 = vp_r0 + 1
        with pl.section_vector():
            sub_id = pl.get_subblock_idx()
            if sub_id == 0:
                _vp2a_vector_pair(m2, batch, tensors, params, vp_r0, vp_r1, vp_task_count)
        with pl.section_cube():
            _vp2a_cube_pair(m2, batch, tensors, params, vp_r0, vp_r1, vp_task_count)


def _run_shared_phi_products(schedule, tensors, params, m2):
    (metadata, batch_base, storage_base, token_base, length_base, chunk_start_base, chunk_num_base,
     active_batch_num, prefix_base, core_start, core_end) = schedule
    for vp_a in pl.range(0, active_batch_num):
        vp_prefix_start = pl.getval(metadata, prefix_base + vp_a)
        vp_prefix_end = pl.getval(metadata, prefix_base + vp_a + 1)
        vp_task_start = core_start
        if vp_task_start < vp_prefix_start:
            vp_task_start = vp_prefix_start
        vp_task_end = core_end
        if vp_task_end > vp_prefix_end:
            vp_task_end = vp_prefix_end
        vp_task_count = vp_task_end - vp_task_start
        if vp_task_count > 0:
            vp_logical = pl.getval(metadata, batch_base + vp_a)
            vp_storage = pl.getval(metadata, storage_base + vp_a)
            vp_token = pl.getval(metadata, token_base + vp_a)
            vp_len = pl.getval(metadata, length_base + vp_a)
            vp_gstart = pl.getval(metadata, chunk_start_base + vp_a)
            vp_gcount = pl.getval(metadata, chunk_num_base + vp_a)
            vp_chunks_total = (vp_len + CHUNK - 1) // CHUNK
            vp_tail = vp_len - (vp_chunks_total - 1) * CHUNK
            batch = pl.make_tuple(
                task_start=vp_task_start, prefix_start=vp_prefix_start, group_count=vp_gcount,
                group_start=vp_gstart, chunks_total=vp_chunks_total, tail_rows=vp_tail,
                token_start=vp_token, logical_batch=vp_logical, storage_batch=vp_storage
            )
            _vp2a_batch(m2, batch, tensors, params, vp_task_count)


def _shared_summary_begin(task, tensors, m2, u_pair, nv, layout, capacity):
    s2_valid0 = task.s2_len - task.s2_cstart * CHUNK
    if s2_valid0 > CHUNK:
        s2_valid0 = CHUNK
    s2_c0_vec = 0
    if layout == LAYOUT_TND:
        s2_c0_vec = 1
    if s2_valid0 < CHUNK:
        s2_c0_vec = 1
    # 链首：R/ψ 接力 + φ(0)/kg(0)/u(0)
    with pl.section_cube():
        _vp2b_init_cube(
            m2,
            (task.s2_cstart, task.s2_logical, task.s2_head,
             task.s2_len, task.s2_storage, task.s2_token,
             s2_valid0, layout, task.s2_row_off, task.s2_rows,
             s2_c0_vec, nv, capacity, task.s2_zword),
            tensors.kg_t, tensors.trans_t, tensors.state_t, tensors.id128_view,
            tensors.zeros128_view, tensors.u_t, tensors.ws_phi_ptr,
            tensors.ws_zero_t)
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        if sub_id == 0:
            _vp2b_init_vec(
                m2,
                (task.s2_task, task.s2_start, s2_c0_vec,
                 task.s2_storage, task.s2_head, task.s2_token,
                 task.s2_cstart, task.s2_len, s2_valid0,
                 layout, task.s2_row_off, task.s2_rows),
                tensors.u_t, u_pair)
    return s2_c0_vec


def _shared_summary_cube(task, tensors, m2, params, s2_r, s2_chk, s2_chk_cons, s2_c0_vec, r_live):
    s2_cnum = task.s2_cnum
    s2_logical = task.s2_logical
    s2_head = task.s2_head
    s2_cstart = task.s2_cstart
    s2_zword = task.s2_zword
    s2_task = task.s2_task
    s2_end = task.s2_end
    s2_row_off = task.s2_row_off
    s2_rows = task.s2_rows
    s2_last_valid = task.s2_last_valid
    s2_storage = task.s2_storage
    s2_token = task.s2_token
    s2_len = task.s2_len
    nv, layout, capacity = params
    if s2_r < s2_cnum:
        if s2_r >= 1:
            _vp2b_phi_load(
                m2, (s2_r, s2_logical, s2_head, s2_cstart, nv, capacity),
                tensors.ws_phi_ptr)
    if s2_chk_cons == 1:
        pl.system.wait_cross_core(
            pipe=pl.PipeType.S, event_id=EV_ZERO_READY,
            sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
        # 等到写入后，失效 AIC 标量缓存中的旧标志。
        pl.system.dcci(tensors.ws_zero_t, s2_zword,
                       cache_line=pl.CacheLine.SINGLE_CACHE_LINE)
        r_live = pl.getval(tensors.ws_zero_t, s2_zword)
    if s2_r >= 1:
        s2_chk_do = s2_chk
        if s2_r >= s2_cnum:
            s2_chk_do = 0
        _vp2b_rank(
            m2,
            (s2_r, s2_cnum, s2_logical, s2_head, s2_task, s2_end,
             s2_row_off, s2_rows, s2_last_valid, layout, s2_zword, s2_chk_do, s2_c0_vec),
            tensors.trans_t, tensors.state_t, r_live)
    if s2_r + 1 < s2_cnum:
        _vp2b_prefetch(
            m2,
            (s2_r, s2_cnum, s2_last_valid, s2_storage, s2_head,
             s2_token, s2_cstart, s2_len, layout, s2_row_off, s2_rows),
            tensors.u_t, tensors.kg_t)
    return r_live


def _shared_summary_rounds(task, tensors, m2, u_pair, params, s2_c0_vec):
    _, layout, _ = params
    s2_cnum = task.s2_cnum
    s2_last_valid = task.s2_last_valid
    s2_cstart = task.s2_cstart
    s2_len = task.s2_len
    s2_storage = task.s2_storage
    s2_head = task.s2_head
    s2_token = task.s2_token
    s2_row_off = task.s2_row_off
    s2_rows = task.s2_rows
    s2_zword = task.s2_zword
    r_live = 1
    for s2_r in pl.range(0, s2_cnum + 1):
        # rank 检查 chunk r-1，其结果在下一轮由 vector 消费。
        s2_chk = 0
        if s2_r >= 1 and (s2_r - 1) % 16 == 15:
            s2_chk = 1
        s2_chk_cons = 0
        if s2_r >= 2 and (s2_r - 2) % 16 == 15 and s2_r - 1 < s2_cnum:
            s2_chk_cons = 1
        with pl.section_vector():
            sub_id = pl.get_subblock_idx()
            if sub_id == 0:
                _vp2b_vec_round(
                    m2,
                    (s2_r, s2_cnum, s2_last_valid,
                    layout, s2_cstart, s2_len,
                    s2_storage, s2_head,
                    s2_token, s2_row_off,
                    s2_rows, s2_chk_cons,
                    s2_zword),
                    tensors.u_t, u_pair, tensors.ws_zero_ptr)
        with pl.section_cube():
            r_live = _shared_summary_cube(
                task, tensors, m2, params, s2_r, s2_chk, s2_chk_cons, s2_c0_vec, r_live)


def _run_shared_summary(schedule, tensors, params, m2, u_pair):
    s2_start, s2_end = schedule[7], schedule[8]
    s2_dv, nv, layout, capacity = params
    for s2_task in pl.range(s2_start, s2_end):
        task = _summary_task_ctx(schedule, s2_task, s2_dv, nv)
        s2_c0_vec = _shared_summary_begin(task, tensors, m2, u_pair, nv, layout, capacity)
        _shared_summary_rounds(task, tensors, m2, u_pair, (nv, layout, capacity), s2_c0_vec)


def _local_summary_begin(task, tensors, m2, u_pair, s2_dv, layout):
    s2_cstart = task.s2_cstart
    s2_logical = task.s2_logical
    s2_head = task.s2_head
    s2_len = task.s2_len
    s2_storage = task.s2_storage
    s2_token = task.s2_token
    s2_row_off = task.s2_row_off
    s2_rows = task.s2_rows
    s2_task = task.s2_task
    s2_start = task.s2_start
    direct_tnd = 0
    if s2_dv == 1:
        direct_tnd = 1
    first_rank = 0
    if s2_dv == 1:
        if s2_cstart == 0:
            first_rank = 1
    s2_first_vec = 1
    if s2_dv == 1:
        if layout == LAYOUT_TND or s2_len - s2_cstart * CHUNK >= CHUNK:
            s2_first_vec = 0
    # 初始化 R/ψ 并预取首块 kg。w/u 由所选读入路径准备。
    with pl.section_cube():
        _st2_init_cube(
            m2,
            (s2_cstart, s2_logical, s2_head,
            s2_len, s2_storage, s2_token, layout, first_rank, s2_first_vec),
            tensors.kg_t, tensors.trans_t, tensors.state_t, tensors.id128_view,
            s2_row_off, s2_rows, tensors.w_t, tensors.u_t)
    # 仅在需要 AIV 转置时准备首块 w/u 并发布 WU 事件。
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        if sub_id == 0:
            _st2_init_vec(
                m2,
                (s2_task, s2_start, s2_storage,
                s2_logical, s2_head, s2_token,
                s2_cstart, s2_len, layout, s2_first_vec),
                tensors.w_t, tensors.u_t, u_pair,
                s2_row_off, s2_rows)
    return direct_tnd, first_rank, s2_first_vec


def _local_summary_cube(task, tensors, m2, layout, direct_tnd, first_rank, s2_first_vec, s2_r):
    s2_cnum, s2_last_valid = task.s2_cnum, task.s2_last_valid
    s2_cstart, s2_len, s2_token = task.s2_cstart, task.s2_len, task.s2_token
    s2_storage, s2_head, s2_logical = task.s2_storage, task.s2_head, task.s2_logical
    s2_row_off, s2_rows = task.s2_row_off, task.s2_rows
    s2_task, s2_end = task.s2_task, task.s2_end
    if s2_r < s2_cnum:
        _st2_phi(
            m2,
            (s2_r, s2_cnum, s2_last_valid,
            layout, direct_tnd, s2_first_vec))
    if s2_r >= 1:
        _st2_rank(
            m2,
            (s2_r, s2_cnum, s2_logical,
            s2_head, s2_task, s2_end,
            s2_row_off, s2_rows,
            s2_last_valid, layout, s2_cstart, first_rank, direct_tnd),
            tensors.trans_t, tensors.state_t)
    if s2_r + 1 < s2_cnum:
        _st2_prefetch(
            m2,
            (s2_r, s2_cnum, s2_last_valid,
            s2_storage, s2_head, s2_token,
            s2_cstart, s2_len, layout,
            s2_row_off, s2_rows, direct_tnd),
            tensors.w_t, tensors.u_t, tensors.kg_t,
            s2_row_off, s2_rows)


def _local_summary_vector(task, tensors, m2, u_pair, layout, direct_tnd, first_rank, s2_r):
    s2_cnum = task.s2_cnum
    s2_row_off = task.s2_row_off
    s2_rows = task.s2_rows
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        if sub_id == 0:
            if s2_r >= 1:
                _st2_vecwork(
                    m2, (task.s2_cstart, s2_r, task.s2_logical, task.s2_head, first_rank, s2_cnum),
                    tensors.decay_t, tensors.trans_t)
            _st2_transpose(
                m2, (s2_r, s2_cnum, task.s2_last_valid, layout, direct_tnd),
                (s2_r, task.s2_cstart, task.s2_len, task.s2_storage, task.s2_head, task.s2_token, layout),
                tensors.w_t, tensors.u_t, u_pair, s2_row_off, s2_rows)


def _local_summary_rounds(task, tensors, m2, u_pair, layout, direct_tnd, first_rank, s2_first_vec):
    s2_cnum = task.s2_cnum
    # vec 轮 r = [vecwork(r-1), (TND)transpose(r+1)]；末轮仅收尾 rank
    for s2_r in pl.range(0, s2_cnum + 1):
        _local_summary_vector(task, tensors, m2, u_pair, layout, direct_tnd, first_rank, s2_r)
        # cube 轮 r = [φ(r), rank(r-1), prefetch(r+1)]；φ 前置
        # 消除 M 空转，phi 双槽隔离跨轮并发
        with pl.section_cube():
            _local_summary_cube(task, tensors, m2, layout, direct_tnd, first_rank, s2_first_vec, s2_r)


def _run_local_summary(schedule, tensors, params, m2, u_pair):
    s2_start, s2_end = schedule[7], schedule[8]
    s2_dv, nv, layout = params
    for s2_task in pl.range(s2_start, s2_end):
        task = _summary_task_ctx(schedule, s2_task, s2_dv, nv)
        direct_tnd, first_rank, s2_first_vec = _local_summary_begin(task, tensors, m2, u_pair, s2_dv, layout)
        _local_summary_rounds(task, tensors, m2, u_pair, layout, direct_tnd, first_rank, s2_first_vec)


def _make_input_views(ptrs, dims, layout):
    batch, nv, nqk, seq_len = dims
    q_ptr = ptrs.q_ptr
    k_ptr = ptrs.k_ptr
    v_ptr = ptrs.v_ptr
    g_ptr = ptrs.g_ptr
    beta_ptr = ptrs.beta_ptr
    a_log_ptr = ptrs.a_log_ptr
    dt_bias_ptr = ptrs.dt_bias_ptr
    if layout == LAYOUT_BNSD:
        q_t = pl.make_tensor(q_ptr, [batch, nqk, seq_len, D], dtype=pl.DT_BF16)
        k_t = pl.make_tensor(k_ptr, [batch, nqk, seq_len, D], dtype=pl.DT_BF16)
        v_t = pl.make_tensor(v_ptr, [batch, nv, seq_len, D], dtype=pl.DT_BF16)
        g_t = pl.make_tensor(g_ptr, [batch, nv, seq_len, D], dtype=pl.DT_BF16)
        b_t = pl.make_tensor(beta_ptr, [batch, nv, seq_len], dtype=pl.DT_BF16)
    elif layout == LAYOUT_BSND:
        # BSND/TND 的 β 由 flat 路径直接从 beta_ptr 读取。
        q_t = pl.make_tensor(q_ptr, [batch, seq_len, nqk, D], dtype=pl.DT_BF16)
        k_t = pl.make_tensor(k_ptr, [batch, seq_len, nqk, D], dtype=pl.DT_BF16)
        v_t = pl.make_tensor(v_ptr, [batch, seq_len, nv, D], dtype=pl.DT_BF16)
        g_t = pl.make_tensor(g_ptr, [batch, seq_len, nv, D], dtype=pl.DT_BF16)
        b_t = pl.make_tensor(beta_ptr, [batch, nv, seq_len], dtype=pl.DT_BF16)
    else:
        q_t = pl.make_tensor(q_ptr, [seq_len, nqk, D], dtype=pl.DT_BF16)
        k_t = pl.make_tensor(k_ptr, [seq_len, nqk, D], dtype=pl.DT_BF16)
        v_t = pl.make_tensor(v_ptr, [seq_len, nv, D], dtype=pl.DT_BF16)
        g_t = pl.make_tensor(g_ptr, [seq_len, nv, D], dtype=pl.DT_BF16)
        b_t = pl.make_tensor(beta_ptr, [nv, seq_len], dtype=pl.DT_BF16)
    alog_t = pl.make_tensor(a_log_ptr, [1, nv], dtype=pl.DT_FP32)
    bias_t = pl.make_tensor(dt_bias_ptr, [nv, D], dtype=pl.DT_FP32)
    return pl.make_tuple(q_t=q_t, k_t=k_t, v_t=v_t, g_t=g_t, b_t=b_t, beta_ptr=beta_ptr,
                         alog_t=alog_t, bias_t=bias_t)


def _make_output_views(ptrs, dims, layout):
    batch, nv, out_rows, capacity = dims
    qg_ptr = ptrs.qg_ptr
    kg_ptr = ptrs.kg_ptr
    u_ptr = ptrs.u_ptr
    w_ptr = ptrs.w_ptr
    decay_ptr = ptrs.decay_ptr
    mqk_ptr = ptrs.mqk_ptr
    trans_ptr = ptrs.trans_ptr
    state_ptr = ptrs.state_ptr
    # 输出缓冲按容量分配：视图行数须与缓冲一致（out_rows），否则跨 head stride 踩踏
    if layout == LAYOUT_TND:
        qg_t = pl.make_tensor(qg_ptr, [out_rows, nv, D], dtype=pl.DT_BF16)
        kg_t = pl.make_tensor(kg_ptr, [out_rows, nv, D], dtype=pl.DT_BF16)
        u_t = pl.make_tensor(u_ptr, [out_rows, nv, D], dtype=pl.DT_BF16)
        w_t = pl.make_tensor(w_ptr, [out_rows, nv, D], dtype=pl.DT_BF16)
    else:
        qg_t = pl.make_tensor(qg_ptr, [batch, nv, out_rows, D], dtype=pl.DT_BF16)
        kg_t = pl.make_tensor(kg_ptr, [batch, nv, out_rows, D], dtype=pl.DT_BF16)
        u_t = pl.make_tensor(u_ptr, [batch, nv, out_rows, D], dtype=pl.DT_BF16)
        w_t = pl.make_tensor(w_ptr, [batch, nv, out_rows, D], dtype=pl.DT_BF16)
    decay_t = pl.make_tensor(decay_ptr, [batch, nv, capacity, D], dtype=pl.DT_FP32)
    mqk_t = pl.make_tensor(mqk_ptr, [batch, nv, capacity, CHUNK, CHUNK],
                        dtype=pl.DT_BF16)
    trans_t = pl.make_tensor(trans_ptr, [batch, nv, D, D], dtype=pl.DT_FP32)
    state_t = pl.make_tensor(state_ptr, [batch, nv, D, D], dtype=pl.DT_FP32)
    return pl.make_tuple(qg_t=qg_t, kg_t=kg_t, u_t=u_t, w_t=w_t, decay_t=decay_t, mqk_t=mqk_t,
                         trans_t=trans_t, state_t=state_t)


def _make_workspace_views(ptrs, batch, nv):
    ws_ops16_ptr = ptrs.ws_ops16_ptr
    constants_ptr = ptrs.constants_ptr
    ws_phi_ptr = ptrs.ws_phi_ptr
    ws_zero_ptr = ptrs.ws_zero_ptr
    id64_f16_ptr = pl.make_ptr(pl.addptr(constants_ptr, 2 * D * D), pl.DT_FP16)
    id64_f16_view = pl.make_tensor(id64_f16_ptr, [CHUNK, CHUNK], dtype=pl.DT_FP16)
    id128_view = pl.make_tensor(constants_ptr, [D, D], dtype=pl.DT_FP32)
    zeros128_view = pl.make_tensor(pl.addptr(constants_ptr, D * D), [D, D],
                                  dtype=pl.DT_FP32)
    ws_zero_t = pl.make_tensor(ws_zero_ptr, [batch * nv * S2DV, 128],
                               dtype=pl.DT_INT32)
    return pl.make_tuple(ws_ops16_ptr=ws_ops16_ptr, id64_f16_view=id64_f16_view, id128_view=id128_view,
                         zeros128_view=zeros128_view, ws_phi_ptr=ws_phi_ptr,
                         ws_zero_t=ws_zero_t, ws_zero_ptr=ws_zero_ptr)


def _dispatch_shared_summary(schedule, outputs, workspace, dims, m2, u_pair):
    (metadata, batch_base, storage_base, token_base, length_base, chunk_start_base, chunk_num_base,
     active_batch_num, prefix_base, core_start, core_end, s2_start, s2_end) = schedule
    nv, layout, capacity, s2_dv = dims
    # ---- 2a products：φ 逐 (head, chunk) 预产制（复用
    # phase1 任务划分——w/kg 同核产出，φ 落 GM cache）----
    _run_shared_phi_products(
        (metadata, batch_base, storage_base, token_base, length_base,
         chunk_start_base, chunk_num_base, active_batch_num, prefix_base,
         core_start, core_end),
        pl.make_tuple(
            w_t=outputs.w_t, kg_t=outputs.kg_t, decay_t=outputs.decay_t, ws_phi_ptr=workspace.ws_phi_ptr
        ),
        (nv, layout, capacity),
        m2,
    )
    # φ cache 产出全局可见后方可进入 roller 轮
    pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
    if s2_dv == S2DV:
        _run_shared_summary(
            (metadata, batch_base, storage_base, token_base, length_base,
             chunk_start_base, chunk_num_base, s2_start, s2_end),
            pl.make_tuple(
                kg_t=outputs.kg_t, u_t=outputs.u_t, trans_t=outputs.trans_t, state_t=outputs.state_t,
                id128_view=workspace.id128_view, zeros128_view=workspace.zeros128_view,
                ws_phi_ptr=workspace.ws_phi_ptr, ws_zero_t=workspace.ws_zero_t, ws_zero_ptr=workspace.ws_zero_ptr
            ),
            (S2DV, nv, layout, capacity),
            m2,
            u_pair,
        )


def _dispatch_summary(schedule, outputs, workspace, params, ranges_base, round_num, tiles):
    (metadata, batch_base, storage_base, token_base, length_base, chunk_start_base, chunk_num_base,
     active_batch_num, prefix_base, core_id, core_start, core_end) = schedule
    layout, _, _, nv, _, capacity, _, _, _, _ = params
    m2, u_pair = tiles.summary, tiles.summary_u
    dv_splits_base = ranges_base + 32 * 2
    stage2_prefix_base = dv_splits_base + active_batch_num
    stage2_ranges_base = stage2_prefix_base + active_batch_num + 1
    s2_dv = pl.getval(metadata, dv_splits_base)
    # 共享 φ 路径先产出按 chunk 缓存，再并行递推各行片。
    s2_start = pl.getval(metadata, stage2_ranges_base + core_id * 2)
    s2_end = pl.getval(metadata, stage2_ranges_base + core_id * 2 + 1)
    s2_legacy_dv = S2DV
    if SharePhi == 1:
        s2_use_shared = 0
        if S2DV >= 4:
            s2_use_shared = 1
        else:
            if round_num == 1:
                s2_use_shared = 1
        if s2_use_shared == 1:
            s2_legacy_dv = -1
            _dispatch_shared_summary(
                (metadata, batch_base, storage_base, token_base, length_base,
                 chunk_start_base, chunk_num_base, active_batch_num, prefix_base,
                 core_start, core_end, s2_start, s2_end),
                outputs, workspace, (nv, layout, capacity, s2_dv), m2, u_pair)
    if ShortSummary == 1:
        # The first row-slice owner computes the entire single-chunk summary.
        s2_start = (s2_start + s2_dv - 1) // s2_dv
        s2_end = (s2_end + s2_dv - 1) // s2_dv
    if ShortSummary == 1 or s2_dv == s2_legacy_dv:
        _run_local_summary(
            (metadata, batch_base, storage_base, token_base, length_base,
             chunk_start_base, chunk_num_base, s2_start, s2_end),
            pl.make_tuple(
                w_t=outputs.w_t, kg_t=outputs.kg_t, decay_t=outputs.decay_t, u_t=outputs.u_t, trans_t=outputs.trans_t,
                state_t=outputs.state_t, id128_view=workspace.id128_view
            ),
            (S2DV, nv, layout),
            m2,
            u_pair,
        )


def _dispatch_rounds(metadata, inputs, outputs, workspace, params, tiles):
    round_num = pl.getval(metadata, HEADER_STAGE12_ROUND_NUM)
    for round_idx in pl.range(0, round_num):
        record = pl.getval(metadata, HEADER_WORDS + round_idx)
        active_batch_num = pl.getval(metadata, record + 1)
        batch_base = record + 2
        storage_base = batch_base + active_batch_num
        token_base = storage_base + active_batch_num
        length_base = token_base + active_batch_num
        chunk_start_base = length_base + active_batch_num + 1
        chunk_num_base = chunk_start_base + active_batch_num
        prefix_base = chunk_num_base + active_batch_num
        ranges_base = prefix_base + active_batch_num + 1
        # AIC 侧 subblock_num=1、AIV 侧=2，同式两侧均得 mix block 号
        core_id = pl.get_block_idx() // pl.get_subblock_num()
        core_start = pl.getval(metadata, ranges_base + core_id * 2)
        core_end = pl.getval(metadata, ranges_base + core_id * 2 + 1)
        schedule = (metadata, batch_base, storage_base, token_base, length_base, chunk_start_base,
                    chunk_num_base, active_batch_num, prefix_base, core_id, core_start, core_end)
        _run_stage1_round(schedule, inputs, outputs, (tiles.cube, tiles.vec0, tiles.vec1),
                          tiles.scores, workspace, params)
        # The stages use different mutexes for shared memory.
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        _dispatch_summary(schedule, outputs, workspace, params, ranges_base, round_num, tiles)
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)


def _run_if_valid_metadata(metadata, inputs, outputs, workspace, params, tiles):
    md_ok = pl.getval(metadata, HEADER_MAGIC)
    md_abi = pl.getval(metadata, HEADER_ABI_VERSION)
    md_status = pl.getval(metadata, HEADER_STATUS)
    md_core = pl.getval(metadata, HEADER_CORE_NUM)

    if md_ok == MAGIC and md_abi == ABI_VERSION and md_status == STATUS_OK:
        if md_core == pl.get_block_num():
            _dispatch_rounds(metadata, inputs, outputs, workspace, params, tiles)


@pl.jit(auto_mutex=True, tiling_key=CpKey, name="flash_kda_cp_preprocess_closeout")
def flash_kda_cp_preprocess_kernel(
    q_ptr: pl.Ptr[pl.DT_BF16],
    k_ptr: pl.Ptr[pl.DT_BF16],
    v_ptr: pl.Ptr[pl.DT_BF16],
    g_ptr: pl.Ptr[pl.DT_BF16],
    beta_ptr: pl.Ptr[pl.DT_BF16],
    a_log_ptr: pl.Ptr[pl.DT_FP32],
    dt_bias_ptr: pl.Ptr[pl.DT_FP32],
    metadata: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    qg_ptr: pl.Ptr[pl.DT_BF16],
    kg_ptr: pl.Ptr[pl.DT_BF16],
    decay_ptr: pl.Ptr[pl.DT_FP32],
    u_ptr: pl.Ptr[pl.DT_BF16],
    w_ptr: pl.Ptr[pl.DT_BF16],
    mqk_ptr: pl.Ptr[pl.DT_BF16],
    ws_ops16_ptr: pl.Ptr[pl.DT_BF16],
    constants_ptr: pl.Ptr[pl.DT_FP32],
    trans_ptr: pl.Ptr[pl.DT_FP32],
    state_ptr: pl.Ptr[pl.DT_FP32],
    batch: pl.DT_INT64,
    nv: pl.DT_INT64,
    nqk: pl.DT_INT64,
    seq_len: pl.DT_INT64,
    out_rows: pl.DT_INT64,
    capacity: pl.DT_INT64,
    slots: pl.DT_INT64,
    scale: pl.DT_FP32,
    lower_bound: pl.DT_FP32,
    neu_iters: pl.DT_INT64,
    ws_phi_ptr: pl.Ptr[pl.DT_FP32],
    ws_zero_ptr: pl.Ptr[pl.DT_INT32],
):
    """执行预处理和摘要递推。每轮的 MIX 屏障保护两个阶段复用的片上缓冲。"""
    layout = Layout
    inputs = _make_input_views(
        pl.make_tuple(q_ptr=q_ptr, k_ptr=k_ptr, v_ptr=v_ptr, g_ptr=g_ptr, beta_ptr=beta_ptr,
                      a_log_ptr=a_log_ptr, dt_bias_ptr=dt_bias_ptr),
        (batch, nv, nqk, seq_len), layout)
    outputs = _make_output_views(
        pl.make_tuple(qg_ptr=qg_ptr, kg_ptr=kg_ptr, u_ptr=u_ptr, w_ptr=w_ptr, decay_ptr=decay_ptr,
                      mqk_ptr=mqk_ptr, trans_ptr=trans_ptr, state_ptr=state_ptr),
        (batch, nv, out_rows, capacity), layout)
    workspace = _make_workspace_views(
        pl.make_tuple(ws_ops16_ptr=ws_ops16_ptr, constants_ptr=constants_ptr, ws_phi_ptr=ws_phi_ptr,
                      ws_zero_ptr=ws_zero_ptr), batch, nv)
    tiles = _make_kernel_tiles()
    params = (layout, NvFlat, seq_len, nv, nqk, capacity, scale, lower_bound, neu_iters, PadDecay)
    _run_if_valid_metadata(metadata, inputs, outputs, workspace, params, tiles)



def _stage1_slot_bound(batch, nv, storage_length, layout_qkv):
    """stage1 任务数上限的形状推导（免 metadata 设备同步）。

    任务 = (batch, 头, chunk) 三元组：dense 精确为 batch·nv·ceil(S/64)；
    TND 取 Σ ceil(len_i/64) ≤ floor(T/64) + B（每条序列至多一个尾块）。
    上界仅用于 ws 分配尺寸，不进入 kernel 逻辑。
    """
    if layout_qkv == "TND":
        chunk_bound = storage_length // CHUNK_SIZE + batch
        return nv * chunk_bound
    return batch * nv * ((storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE)


_CONSTANTS_CACHE = {}


def _constants_for(device):
    """Upload the immutable FP32 I/0 and FP16 I once per device."""
    constants = _CONSTANTS_CACHE.get(device)
    if constants is None:
        words = np.zeros(2 * D * D + CHUNK * CHUNK // 2, dtype=np.float32)
        np.fill_diagonal(words[:D * D].reshape(D, D), 1)
        np.fill_diagonal(words[2 * D * D:].view(np.float16).reshape(CHUNK, CHUNK), 1)
        constants = torch.from_numpy(words).to(device)
        _CONSTANTS_CACHE[device] = constants
    return constants


# ===== metadata 调度器（stage12w ABI v2；按形状、布局和核数排程）=====

MAGIC = 0x4B444132
ABI_VERSION = 2
MAX_AIC_CORES = 32
WORKSPACE_SLOTS = 2048
LONG_DENSE_WORKSPACE_SLOTS = 4096
CHUNK_SIZE = 64
ROUND_FIXED_WORDS = 133
ROUND_WORDS_PER_BATCH = 9

HEADER_MAGIC = 0
HEADER_ABI_VERSION = 1
HEADER_STATUS = 2
HEADER_CORE_NUM = 3
HEADER_STAGE12_ROUND_NUM = 4
HEADER_BATCH = 5
HEADER_VALUE_HEADS = 6
HEADER_HAS_CU = 7
HEADER_WORKSPACE_SLOTS = 8
HEADER_WORDS = 9

STATUS_OK = 0
STATUS_BAD_CORE_NUM = 1
STATUS_BAD_WORKSPACE_SLOTS = 2
STATUS_BAD_FLAGS = 3
STATUS_BAD_SHAPE = 4
STATUS_BAD_CU_START = 5
STATUS_BAD_LENGTH = 6
STATUS_BAD_PADDED_LENGTH = 7
STATUS_BAD_PACKED_END = 8
STATUS_BAD_METADATA_CAPACITY = 9

MAX_ACTIVE_BATCHES = LONG_DENSE_WORKSPACE_SLOTS
MAX_STAGE1_TASKS = LONG_DENSE_WORKSPACE_SLOTS
MAX_STAGE2_TASKS = LONG_DENSE_WORKSPACE_SLOTS * 8
FULL_CHUNK_STAGE1_COST = 4930

SUPPORTED_HEAD_DIM = 128


def _interpolate_stage1_task_cost(valid_row_num, left_row_num, left_cost,
                                right_row_num, right_cost):
    row_span = right_row_num - left_row_num
    numerator = left_cost * row_span + (valid_row_num - left_row_num) * (right_cost - left_cost)
    denominator = row_span * 1000
    return (numerator + denominator // 2) // denominator


def _stage1_task_cost(valid_row_num):
    if valid_row_num <= 0:
        return 0
    if valid_row_num <= 16:
        return _interpolate_stage1_task_cost(valid_row_num, 1, 4892380, 16, 4920438)
    if valid_row_num <= 32:
        return _interpolate_stage1_task_cost(valid_row_num, 16, 4920438, 32, 4970718)
    if valid_row_num <= 48:
        return _interpolate_stage1_task_cost(valid_row_num, 32, 4970718, 48, 4992347)
    if valid_row_num <= 63:
        return _interpolate_stage1_task_cost(valid_row_num, 48, 4992347, 63, 4952972)
    return FULL_CHUNK_STAGE1_COST


def _stage2_task_cost(candidate_idx, chunk_num):
    # S>1 为行拆变体：R/ψ/ψacc 按 1/S 缩减，φ 与装载全宽重复
    # （c(S) ∝ (1.05 + 5.24/S)/6.29，以 c(1)=1839 标定）
    item_init = 352
    chunk_cost = 1839
    if candidate_idx == 1:
        item_init = 176
        chunk_cost = 1073
    elif candidate_idx == 2:
        item_init = 88
        chunk_cost = 690
    elif candidate_idx == 3:
        item_init = 44
        chunk_cost = 497
    return item_init + chunk_num * chunk_cost


def _s2_split_count(chain_count, core_num):
    """stage2 链 D 维分裂数（编译期 tiling key 的 host 侧镜像）。

    链数 ≥ 核数时不拆；否则取满足 链数×S ≤ 核数 的最大 2 的幂
    （上限 8）。行拆使每任务 chunk 成本从 c(1) 降至 c(S)∝1/S+常数，
    波数 ceil(链×S/核) 在该选择下恒为 1。
    """
    splits = 1
    while splits < 8 and chain_count * splits * 2 <= core_num:
        splits *= 2
    return splits


def round_record_words(active_batch_num: int) -> int:
    active_batch_num = int(active_batch_num)
    if active_batch_num <= 0:
        raise ValueError("active_batch_num must be positive")
    return ROUND_FIXED_WORDS + ROUND_WORDS_PER_BATCH * active_batch_num


def metadata_capacity_upper_bound(batch: int, storage_length: int, *, is_packed: bool) -> int:
    batch = int(batch)
    storage_length = int(storage_length)
    if batch <= 0:
        raise ValueError("batch must be positive")
    if storage_length <= 0:
        raise ValueError("storage_length must be positive")
    storage_chunk_num = (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE
    if is_packed:
        chunk_entries = storage_chunk_num + batch - 1
    else:
        chunk_entries = batch * storage_chunk_num
    return HEADER_WORDS + 1 + (1 + ROUND_FIXED_WORDS + ROUND_WORDS_PER_BATCH) * chunk_entries


def chunk_capacity_upper_bound(batch: int, storage_length: int) -> int:
    batch = int(batch)
    storage_length = int(storage_length)
    if batch <= 0:
        raise ValueError("batch must be positive")
    if storage_length <= 0:
        raise ValueError("storage_length must be positive")
    return (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE + batch - 1


class _BatchSet:
    start = 0
    count = 0
    lengths = None
    token_starts = None
    chunks = None
    tail_rows = None


class _ChunkGroup:
    chunk_start = 0
    count = 0
    size = 0
    local_batches = None
    remaining_chunks = None
    chunks_per_batch = None


class _Schedule:
    status = STATUS_OK
    round_count = 0
    active_entries = 0
    round_index = 0
    record_cursor = 0
    selected_splits = 1
    selected_peak_cost = 0
    selected_total_cost = 0
    selected_task_num = 0
    stage1_costs = None
    stage1_ranges = None
    stage1_core_costs = None
    stage2_costs = None
    candidate_ranges = None
    candidate_core_costs = None
    selected_ranges = None


def _allocate_metadata_scratch(batches, group, schedule):
    batches.lengths = [0] * MAX_ACTIVE_BATCHES
    batches.token_starts = [0] * MAX_ACTIVE_BATCHES
    batches.chunks = [0] * MAX_ACTIVE_BATCHES
    batches.tail_rows = [0] * MAX_ACTIVE_BATCHES
    group.local_batches = [0] * MAX_ACTIVE_BATCHES
    group.remaining_chunks = [0] * MAX_ACTIVE_BATCHES
    group.chunks_per_batch = [0] * MAX_ACTIVE_BATCHES
    schedule.stage1_costs = [0] * MAX_STAGE1_TASKS
    schedule.stage1_ranges = [0] * (MAX_AIC_CORES * 2)
    schedule.stage1_core_costs = [0] * MAX_AIC_CORES
    schedule.stage2_costs = [0] * MAX_STAGE2_TASKS
    schedule.candidate_ranges = [0] * (MAX_AIC_CORES * 2)
    schedule.candidate_core_costs = [0] * MAX_AIC_CORES
    schedule.selected_ranges = [0] * (MAX_AIC_CORES * 2)


class _Args:
    """AICPU kernel 参数视图（host 版：cu_seqlens 为 Python list）。"""

    def __init__(self, cu_seqlens, metadata, metadata_word_capacity, batch,
                physical_batches, storage_length, value_heads, has_cu, is_packed,
                core_num, workspace_slots, max_chunk_group):
        self.cu_seqlens = cu_seqlens
        self.metadata = metadata
        self.metadata_word_capacity = metadata_word_capacity
        self.batch = batch
        self.physical_batches = physical_batches
        self.storage_length = storage_length
        self.value_heads = value_heads
        self.has_cu = has_cu
        self.is_packed = is_packed
        self.core_num = core_num
        self.workspace_slots = workspace_slots
        self.max_chunk_group = max_chunk_group


def _select_group_num(remaining_chunk_num, active_batch_num, value_heads,
                    workspace_slots, max_chunk_group):
    max_remaining_chunk_num = 0
    for active_idx in range(0, active_batch_num):
        if remaining_chunk_num[active_idx] > max_remaining_chunk_num:
            max_remaining_chunk_num = remaining_chunk_num[active_idx]

    selected_group_num = 1
    lower_group_num = 1
    upper_group_num = max_remaining_chunk_num
    if upper_group_num > max_chunk_group:
        upper_group_num = max_chunk_group
    workspace_group_limit = workspace_slots // value_heads
    if upper_group_num > workspace_group_limit:
        upper_group_num = workspace_group_limit
    while lower_group_num <= upper_group_num:
        candidate_group_num = (lower_group_num + upper_group_num) // 2
        slot_num = 0
        for active_idx in range(0, active_batch_num):
            current_chunk_num = remaining_chunk_num[active_idx]
            if current_chunk_num > candidate_group_num:
                current_chunk_num = candidate_group_num
            slot_num += value_heads * current_chunk_num
        if slot_num <= workspace_slots:
            selected_group_num = candidate_group_num
            lower_group_num = candidate_group_num + 1
        else:
            upper_group_num = candidate_group_num - 1
    return selected_group_num


def _partition_greedy_end(task_costs, start, remaining_core_num, remaining_cost, max_end):
    """贪心最小 |cost*cores-total| 的段尾（余 1 核时由调用方直取 task_num）。"""
    end = start + 1
    segment_cost = task_costs[start]
    while end < max_end:
        next_cost = segment_cost + task_costs[end]
        next_error = abs(next_cost * remaining_core_num - remaining_cost)
        current_error = abs(segment_cost * remaining_core_num - remaining_cost)
        if next_error <= current_error:
            segment_cost = next_cost
            end += 1
        else:
            break
    return end


def _partition_costs(task_costs, task_num, core_num, core_ranges, core_costs):
    for core_idx in range(0, MAX_AIC_CORES):
        core_ranges[core_idx * 2] = 0
        core_ranges[core_idx * 2 + 1] = 0
        core_costs[core_idx] = 0

    if task_num <= 0:
        return

    active_core_num = task_num
    if active_core_num > core_num:
        active_core_num = core_num
    remaining_cost = 0
    for task_idx in range(0, task_num):
        remaining_cost += task_costs[task_idx]

    start = 0
    for core_idx in range(0, active_core_num):
        remaining_core_num = active_core_num - core_idx
        if remaining_core_num == 1:
            end = task_num
        else:
            end = _partition_greedy_end(
                task_costs, start, remaining_core_num, remaining_cost,
                task_num - (remaining_core_num - 1))

        segment_cost = 0
        for task_idx in range(start, end):
            segment_cost += task_costs[task_idx]
        core_ranges[core_idx * 2] = start
        core_ranges[core_idx * 2 + 1] = end
        core_costs[core_idx] = segment_cost
        remaining_cost -= segment_cost
        start = end


def _initialize_metadata_header(a):
    a.metadata[HEADER_MAGIC] = MAGIC
    a.metadata[HEADER_ABI_VERSION] = ABI_VERSION
    a.metadata[HEADER_STATUS] = STATUS_OK
    a.metadata[HEADER_CORE_NUM] = a.core_num
    a.metadata[HEADER_STAGE12_ROUND_NUM] = 0
    a.metadata[HEADER_BATCH] = a.batch
    a.metadata[HEADER_VALUE_HEADS] = a.value_heads
    a.metadata[HEADER_HAS_CU] = a.has_cu
    a.metadata[HEADER_WORKSPACE_SLOTS] = a.workspace_slots
    if a.metadata_word_capacity > HEADER_WORDS:
        a.metadata[HEADER_WORDS] = HEADER_WORDS + 1


def _metadata_flags_status(a):
    if a.max_chunk_group < 1 or a.max_chunk_group > 512:
        return STATUS_BAD_FLAGS
    if not (a.has_cu == 0 or a.has_cu == 1):
        return STATUS_BAD_FLAGS
    if not (a.is_packed == 0 or a.is_packed == 1):
        return STATUS_BAD_FLAGS
    if a.is_packed == 1 and a.has_cu == 0:
        return STATUS_BAD_FLAGS
    return STATUS_OK


def _metadata_argument_status(a):
    if a.core_num < 1 or a.core_num > MAX_AIC_CORES:
        return STATUS_BAD_CORE_NUM
    if a.workspace_slots not in (WORKSPACE_SLOTS, LONG_DENSE_WORKSPACE_SLOTS):
        return STATUS_BAD_WORKSPACE_SLOTS
    status = _metadata_flags_status(a)
    if status != STATUS_OK:
        return status
    if a.batch < 1 or a.storage_length < 1 or a.value_heads < 1:
        return STATUS_BAD_SHAPE
    if a.value_heads > a.workspace_slots:
        return STATUS_BAD_SHAPE
    if a.is_packed == 1 and a.physical_batches != 1:
        return STATUS_BAD_SHAPE
    if a.is_packed == 0 and a.physical_batches != a.batch:
        return STATUS_BAD_SHAPE
    if a.has_cu == 1:
        if a.cu_seqlens[0] != 0:
            return STATUS_BAD_CU_START
    return STATUS_OK


def _load_batch_set(a, batches, schedule):
    for local_batch_idx in range(0, batches.count):
        batch_idx = batches.start + local_batch_idx
        token_start = 0
        valid_seq_len = a.storage_length
        if a.has_cu == 1:
            token_start = a.cu_seqlens[batch_idx]
            token_end = a.cu_seqlens[batch_idx + 1]
            valid_seq_len = token_end - token_start
        batches.lengths[local_batch_idx] = valid_seq_len
        batches.token_starts[local_batch_idx] = token_start
        if valid_seq_len <= 0:
            schedule.status = STATUS_BAD_LENGTH
            batches.chunks[local_batch_idx] = 0
            batches.tail_rows[local_batch_idx] = 0
        elif a.is_packed == 0 and valid_seq_len > a.storage_length:
            schedule.status = STATUS_BAD_PADDED_LENGTH
            batches.chunks[local_batch_idx] = 0
            batches.tail_rows[local_batch_idx] = 0
        else:
            batches.chunks[local_batch_idx] = (valid_seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
            batches.tail_rows[local_batch_idx] = ((valid_seq_len - 1) % CHUNK_SIZE) + 1


def _collect_active_batches(batches, group):
    group.count = 0
    for local_batch_idx in range(0, batches.count):
        if batches.chunks[local_batch_idx] > group.chunk_start:
            group.local_batches[group.count] = local_batch_idx
            group.remaining_chunks[group.count] = batches.chunks[local_batch_idx] - group.chunk_start
            group.count += 1
    return group.count


def _choose_chunk_group(a, group):
    group.size = _select_group_num(
        group.remaining_chunks,
        group.count,
        a.value_heads,
        a.workspace_slots,
        a.max_chunk_group,
    )
    for active_idx in range(0, group.count):
        group.chunks_per_batch[active_idx] = min(group.remaining_chunks[active_idx], group.size)


def _count_metadata_rounds(a, batches, group, schedule):
    batch_num_per_set = a.workspace_slots // a.value_heads
    batches.start = 0
    while batches.start < a.batch:
        batches.count = min(a.batch - batches.start, batch_num_per_set)
        _load_batch_set(a, batches, schedule)
        group.chunk_start = 0
        while _collect_active_batches(batches, group) > 0:
            _choose_chunk_group(a, group)
            schedule.round_count += 1
            schedule.active_entries += group.count
            group.chunk_start += group.size
        batches.start += batches.count
    if schedule.status == STATUS_OK and a.is_packed == 1:
        if a.cu_seqlens[a.batch] != a.storage_length:
            schedule.status = STATUS_BAD_PACKED_END


def _write_batch_fields(a, batches, group, round_idx, cursor):
    a.metadata[cursor] = round_idx
    a.metadata[cursor + 1] = group.count
    cursor += 2
    for active_idx in range(0, group.count):
        a.metadata[cursor + active_idx] = batches.start + group.local_batches[active_idx]
    cursor += group.count
    for active_idx in range(0, group.count):
        logical_batch_idx = batches.start + group.local_batches[active_idx]
        if a.is_packed == 1:
            a.metadata[cursor + active_idx] = 0
        else:
            a.metadata[cursor + active_idx] = logical_batch_idx
    cursor += group.count
    for active_idx in range(0, group.count):
        local_batch_idx = group.local_batches[active_idx]
        if a.is_packed == 1:
            a.metadata[cursor + active_idx] = batches.token_starts[local_batch_idx]
        else:
            a.metadata[cursor + active_idx] = 0
    cursor += group.count
    for active_idx in range(0, group.count):
        local_batch_idx = group.local_batches[active_idx]
        a.metadata[cursor + active_idx] = batches.lengths[local_batch_idx]
    cursor += group.count
    a.metadata[cursor] = group.size
    cursor += 1
    for active_idx in range(0, group.count):
        a.metadata[cursor + active_idx] = group.chunk_start
    cursor += group.count
    for active_idx in range(0, group.count):
        a.metadata[cursor + active_idx] = group.chunks_per_batch[active_idx]
    cursor += group.count
    return cursor


def _stage1_chunk_cost(batches, local_batch_idx, global_chunk_idx):
    """整 chunk 记满额成本；序列末尾的短尾块按行数折算。"""
    if (global_chunk_idx + 1 == batches.chunks[local_batch_idx]
            and batches.tail_rows[local_batch_idx] < CHUNK_SIZE):
        return _stage1_task_cost(batches.tail_rows[local_batch_idx])
    return FULL_CHUNK_STAGE1_COST


def _fill_stage1_costs(a, batches, group, schedule):
    stage1_task_idx = 0
    for active_idx in range(0, group.count):
        local_batch_idx = group.local_batches[active_idx]
        chunk_total = group.chunks_per_batch[active_idx]
        for _ in range(0, a.value_heads):
            for local_chunk_idx in range(0, chunk_total):
                global_chunk_idx = group.chunk_start + local_chunk_idx
                schedule.stage1_costs[stage1_task_idx] = _stage1_chunk_cost(
                    batches, local_batch_idx, global_chunk_idx)
                stage1_task_idx += 1


def _write_stage1_partition(a, batches, group, schedule, cursor):
    stage1_task_num = 0
    a.metadata[cursor] = 0
    for active_idx in range(0, group.count):
        stage1_task_num += a.value_heads * group.chunks_per_batch[active_idx]
        a.metadata[cursor + active_idx + 1] = stage1_task_num
    cursor += group.count + 1

    _fill_stage1_costs(a, batches, group, schedule)
    _partition_costs(
        schedule.stage1_costs,
        stage1_task_num,
        a.core_num,
        schedule.stage1_ranges,
        schedule.stage1_core_costs,
    )
    for range_idx in range(0, MAX_AIC_CORES * 2):
        a.metadata[cursor + range_idx] = schedule.stage1_ranges[range_idx]
    cursor += MAX_AIC_CORES * 2
    return cursor


def _stage2_candidate_is_better(schedule, max_core_cost, total_cost, task_num, dv_splits_num):
    if schedule.selected_task_num == 0:
        return 1
    if max_core_cost != schedule.selected_peak_cost:
        return int(max_core_cost < schedule.selected_peak_cost)
    if total_cost != schedule.selected_total_cost:
        return int(total_cost < schedule.selected_total_cost)
    if task_num != schedule.selected_task_num:
        return int(task_num < schedule.selected_task_num)
    return int(dv_splits_num < schedule.selected_splits)


def _select_stage2_partition(a, group, schedule):
    schedule.selected_splits = 1
    schedule.selected_peak_cost = 0
    schedule.selected_total_cost = 0
    schedule.selected_task_num = 0
    # 分裂数全局统一（tiling key 为 launch 级）：按总链数选
    forced_splits = _s2_split_count(a.batch * a.value_heads, a.core_num)
    for candidate_idx in (forced_splits.bit_length() - 1,):
        candidate_dv_splits_num = 1 << candidate_idx
        candidate_task_num = 0
        candidate_total_cost = 0
        for active_idx in range(0, group.count):
            task_cost = _stage2_task_cost(candidate_idx, group.chunks_per_batch[active_idx])
            batch_task_num = a.value_heads * candidate_dv_splits_num
            for _ in range(0, batch_task_num):
                schedule.stage2_costs[candidate_task_num] = task_cost
                candidate_task_num += 1
                candidate_total_cost += task_cost
        _partition_costs(
            schedule.stage2_costs,
            candidate_task_num,
            a.core_num,
            schedule.candidate_ranges,
            schedule.candidate_core_costs,
        )
        candidate_max_core_cost = 0
        for core_idx in range(0, MAX_AIC_CORES):
            if schedule.candidate_core_costs[core_idx] > candidate_max_core_cost:
                candidate_max_core_cost = schedule.candidate_core_costs[core_idx]

        candidate_is_better = _stage2_candidate_is_better(
            schedule,
            candidate_max_core_cost,
            candidate_total_cost,
            candidate_task_num,
            candidate_dv_splits_num,
        )
        if candidate_is_better == 1:
            schedule.selected_splits = candidate_dv_splits_num
            schedule.selected_peak_cost = candidate_max_core_cost
            schedule.selected_total_cost = candidate_total_cost
            schedule.selected_task_num = candidate_task_num
            for range_idx in range(0, MAX_AIC_CORES * 2):
                schedule.selected_ranges[range_idx] = schedule.candidate_ranges[range_idx]


def _write_stage2_partition(a, group, schedule, cursor):
    _select_stage2_partition(a, group, schedule)
    for active_idx in range(0, group.count):
        a.metadata[cursor + active_idx] = schedule.selected_splits
    cursor += group.count
    stage2_task_num = 0
    a.metadata[cursor] = 0
    for active_idx in range(0, group.count):
        stage2_task_num += a.value_heads * schedule.selected_splits
        a.metadata[cursor + active_idx + 1] = stage2_task_num
    cursor += group.count + 1
    for range_idx in range(0, MAX_AIC_CORES * 2):
        a.metadata[cursor + range_idx] = schedule.selected_ranges[range_idx]
    cursor += MAX_AIC_CORES * 2
    return cursor


def _write_metadata_rounds(a, batches, group, schedule):
    batch_num_per_set = a.workspace_slots // a.value_heads
    batches.start = 0
    while batches.start < a.batch:
        batches.count = min(a.batch - batches.start, batch_num_per_set)
        _load_batch_set(a, batches, schedule)
        group.chunk_start = 0
        while _collect_active_batches(batches, group) > 0:
            _choose_chunk_group(a, group)
            a.metadata[HEADER_WORDS + schedule.round_index] = schedule.record_cursor
            cursor = _write_batch_fields(a, batches, group, schedule.round_index, schedule.record_cursor)
            cursor = _write_stage1_partition(a, batches, group, schedule, cursor)
            schedule.record_cursor = _write_stage2_partition(a, group, schedule, cursor)
            schedule.round_index += 1
            group.chunk_start += group.size
        batches.start += batches.count
    a.metadata[HEADER_WORDS + schedule.round_count] = schedule.record_cursor


def flash_kda_cp_preprocess_metadata_kernel_run(a):
    """在 Host 生成调度表，原位写入 metadata 并返回状态码。"""
    batches = _BatchSet()
    group = _ChunkGroup()
    schedule = _Schedule()
    _allocate_metadata_scratch(batches, group, schedule)
    if a.metadata_word_capacity < HEADER_WORDS:
        return STATUS_BAD_METADATA_CAPACITY
    _initialize_metadata_header(a)
    schedule.status = _metadata_argument_status(a)
    if schedule.status != STATUS_OK:
        a.metadata[HEADER_STATUS] = schedule.status
        return schedule.status
    _count_metadata_rounds(a, batches, group, schedule)
    if schedule.status != STATUS_OK:
        a.metadata[HEADER_STATUS] = schedule.status
        return schedule.status
    schedule.record_cursor = HEADER_WORDS + schedule.round_count + 1
    used_words = (
        schedule.record_cursor
        + ROUND_FIXED_WORDS * schedule.round_count
        + ROUND_WORDS_PER_BATCH * schedule.active_entries
    )
    if used_words > a.metadata_word_capacity:
        a.metadata[HEADER_STATUS] = STATUS_BAD_METADATA_CAPACITY
        return STATUS_BAD_METADATA_CAPACITY
    a.metadata[HEADER_STAGE12_ROUND_NUM] = schedule.round_count
    _write_metadata_rounds(a, batches, group, schedule)
    return STATUS_OK


def _device_block_num(ref: torch.Tensor) -> int:
    if ref is not None and ref.device.type not in {"npu", "privateuseone"}:
        return MAX_AIC_CORES
    npu = getattr(torch, "npu", None)
    if npu is None:
        return MAX_AIC_CORES
    device_index = ref.device.index if ref is not None else None
    if device_index is None:
        try:
            device_index = npu.current_device()
        except RuntimeError:
            return MAX_AIC_CORES
    props = npu.get_device_properties(device_index)
    cube_core_num = int(getattr(props, "cube_core_num", 0))
    if cube_core_num <= 0:
        raise RuntimeError(f"NPU {device_index} does not expose a valid cube_core_num")
    return min(cube_core_num, MAX_AIC_CORES)


class _ShapeFields(NamedTuple):
    """布局形状提取结果（batch/storage/头数/头宽）。"""
    batch: int
    physical_batches: int
    storage_length: int
    nqk: int
    nv: int
    head_dim: int


def _shape_of(q, v, layout_qkv, cu_seqlens):
    if layout_qkv not in ("TND", "BNSD", "BSND"):
        raise ValueError(f"layout_qkv must be TND, BNSD, or BSND, got {layout_qkv!r}")
    if layout_qkv == "TND":
        fields = _shape_of_tnd(q, v, cu_seqlens)
    else:
        fields = _shape_of_dense(q, v, layout_qkv, cu_seqlens)
    _shape_of_common(q, v, fields.storage_length, fields.head_dim,
                     fields.batch, fields.nv, fields.nqk)
    return fields.batch, fields.physical_batches, fields.storage_length, \
        fields.nqk, fields.nv


def _shape_of_tnd(q, v, cu_seqlens):
    """TND 布局的形状提取。"""
    if cu_seqlens is None:
        raise ValueError("cu_seqlens is required for TND")
    if cu_seqlens.dtype != torch.int32 or cu_seqlens.dim() != 1:
        raise ValueError("cu_seqlens must be rank-1 int32")
    if not cu_seqlens.is_contiguous() or cu_seqlens.device != q.device:
        raise ValueError("cu_seqlens must be contiguous and on the same device")
    if cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must have shape [B+1] with positive batch")
    batch = cu_seqlens.numel() - 1
    if q.dim() != 3 or v.dim() != 3:
        raise ValueError("TND q/v must be rank-3")
    storage_length, nqk, head_dim = q.shape
    nv = v.shape[1]
    if v.shape[0] != storage_length or v.shape[-1] != head_dim:
        raise ValueError("TND v must match the packed token length")
    physical_batches = 1
    return _ShapeFields(batch, physical_batches, storage_length, nqk, nv, head_dim)


def _shape_of_dense(q, v, layout_qkv, cu_seqlens):
    """BNSD/BSND 布局的形状提取。"""
    if cu_seqlens is not None:
        raise ValueError("cu_seqlens is only valid for TND")
    if q.dim() != 4 or v.dim() != 4:
        raise ValueError("BNSD/BSND q/v must be rank-4")
    if layout_qkv == "BNSD":
        physical_batches, nqk, storage_length, head_dim = q.shape
        nv = v.shape[1]
    else:
        physical_batches, storage_length, nqk, head_dim = q.shape
        nv = v.shape[2]
    if v.shape[0] != physical_batches or v.shape[-1] != head_dim:
        raise ValueError("v batch must match q")
    if layout_qkv == "BNSD" and v.shape[2] != storage_length:
        raise ValueError("v length must match q")
    if layout_qkv == "BSND" and v.shape[1] != storage_length:
        raise ValueError("v length must match q")
    batch = physical_batches
    return _ShapeFields(batch, physical_batches, storage_length, nqk, nv, head_dim)


def _shape_of_common(q, v, storage_length, head_dim, batch, nv, nqk):
    """布局无关的通用形状校验。"""
    if head_dim != SUPPORTED_HEAD_DIM:
        raise ValueError(f"FlashKDA only supports D={SUPPORTED_HEAD_DIM}")
    if nqk < 1 or nv % nqk != 0:
        raise ValueError(f"GQA requires Nv % Nqk == 0, got Nv={nv}, Nqk={nqk}")
    if q.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("q/v must be bf16")
    if not q.is_contiguous() or not v.is_contiguous():
        raise ValueError("q/v must be contiguous when generating metadata")
    if storage_length <= 0 or batch < 1 or nv < 1:
        raise ValueError("storage length, batch and heads must be positive")


def _workspace_slots_for(batch, nv, storage_length, layout_qkv, cu_seqlens):
    workspace_slots = WORKSPACE_SLOTS
    if batch == 1 and layout_qkv == "BSND" and cu_seqlens is None:
        slots = _stage1_slot_bound(batch, nv, storage_length, layout_qkv)
        if WORKSPACE_SLOTS < slots <= LONG_DENSE_WORKSPACE_SLOTS:
            workspace_slots = LONG_DENSE_WORKSPACE_SLOTS
    return workspace_slots


def flash_kda_cp_preprocess_metadata(q, v, layout_qkv, cu_seqlens=None):
    """Build the preprocess-local stage12w schedule (public companion entry)."""
    batch, physical_batches, storage_length, _nqk, nv = _shape_of(
        q, v, layout_qkv, cu_seqlens)
    core_num = _device_block_num(q)
    metadata_capacity = metadata_capacity_upper_bound(
        batch, storage_length, is_packed=(layout_qkv == "TND"))
    metadata = np.zeros(metadata_capacity, dtype=np.uint32)
    cu_list = None
    if cu_seqlens is not None:
        cu_list = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
        if cu_list[0] != 0:
            raise ValueError("cu_seqlens must start at 0")
        if cu_list[-1] != storage_length:
            raise ValueError("cu_seqlens must end at T")
        if any(b <= a for a, b in zip(cu_list, cu_list[1:])):
            if any(b < a for a, b in zip(cu_list, cu_list[1:])):
                raise ValueError("cu_seqlens must be nondecreasing")
    workspace_slots = _workspace_slots_for(batch, nv, storage_length, layout_qkv, cu_seqlens)
    a = _Args(cu_list if cu_list is not None else [0],
              metadata,
              metadata_capacity,
              batch,
              physical_batches,
              storage_length,
              nv,
              1 if cu_seqlens is not None else 0,
              1 if layout_qkv == "TND" else 0,
              core_num,
              workspace_slots,
              512)
    status = flash_kda_cp_preprocess_metadata_kernel_run(a)
    if status != STATUS_OK:
        raise ValueError(f"metadata generation failed with status {status}")
    _constants_for(q.device)
    return torch.from_numpy(metadata.astype(np.int32)).to(q.device)


# ===== host 侧解码（测试与诊断用；kernel 内解码为逐字等价的 getval 序列） =====


def _read_round_fields(words, start, active_batch_num, take):
    """按 ABI 顺序读取单条 round 记录的字段。"""
    fields = {
        "stage12_round_idx": words[start],
        "active_batch_num": active_batch_num,
    }
    cursor = start + 2
    for name, count in (
        ("batch_idx", active_batch_num),
        ("storage_batch_idx", active_batch_num),
        ("token_start", active_batch_num),
        ("valid_seq_len", active_batch_num),
    ):
        fields[name], cursor = take(words, cursor, count)
    fields["group_num"] = words[cursor]
    cursor += 1
    for name, count in (
        ("chunk_start_per_group", active_batch_num),
        ("chunk_num_per_group", active_batch_num),
        ("stage1_task_prefix", active_batch_num + 1),
        ("stage1_ranges_flat", MAX_AIC_CORES * 2),
        ("dv_splits_num", active_batch_num),
        ("stage2_task_prefix", active_batch_num + 1),
        ("stage2_ranges_flat", MAX_AIC_CORES * 2),
    ):
        fields[name], cursor = take(words, cursor, count)
    return fields, cursor


def _check_round_fields(fields, batch, value_heads, workspace_slots):
    """单条 round 记录的结构校验。"""
    if fields["group_num"] <= 0:
        raise ValueError("group_num must be positive")
    if any(i >= batch for i in fields["batch_idx"]):
        raise ValueError("batch_idx is outside logical batch")
    if any(l <= 0 for l in fields["valid_seq_len"]):
        raise ValueError("valid_seq_len must be positive")
    if any(c <= 0 for c in fields["chunk_num_per_group"]):
        raise ValueError("chunk_num_per_group must be positive")
    if any(v not in (1, 2, 4, 8) for v in fields["dv_splits_num"]):
        raise ValueError("dv_splits_num must be 1, 2, 4, or 8")
    _decode_round_prefix_checks(
        fields["stage1_task_prefix"], fields["stage2_task_prefix"],
        fields["chunk_num_per_group"], fields["dv_splits_num"], value_heads)
    if fields["stage1_task_prefix"][-1] > workspace_slots:
        raise ValueError("stage1 task total exceeds workspace slots")


def _decode_round_record(words, start, end, batch, value_heads, workspace_slots, take):
    """解码单条 round 记录并做结构校验（与主函数共享 take/decode_ranges）。"""
    if start + 2 > end:
        raise ValueError("metadata round record is truncated")
    active_batch_num = words[start + 1]
    if start + round_record_words(active_batch_num) != end:
        raise ValueError("metadata round record length does not match active batches")
    fields, cursor = _read_round_fields(words, start, active_batch_num, take)
    if cursor != end:
        raise ValueError("metadata round decoder did not consume the full record")
    _check_round_fields(fields, batch, value_heads, workspace_slots)
    return {
        "stage12_round_idx": fields["stage12_round_idx"],
        "active_batch_num": fields["active_batch_num"],
        "batch_idx": fields["batch_idx"],
        "storage_batch_idx": fields["storage_batch_idx"],
        "token_start": fields["token_start"],
        "valid_seq_len": fields["valid_seq_len"],
        "group_num": fields["group_num"],
        "chunk_start_per_group": fields["chunk_start_per_group"],
        "chunk_num_per_group": fields["chunk_num_per_group"],
        "stage1_task_prefix": fields["stage1_task_prefix"],
        "stage1_core_ranges": _md_decode_ranges(fields["stage1_ranges_flat"]),
        "dv_splits_num": fields["dv_splits_num"],
        "stage2_task_prefix": fields["stage2_task_prefix"],
        "stage2_core_ranges": _md_decode_ranges(fields["stage2_ranges_flat"]),
    }


def _md_decode_ranges(flat):
    """核心范围块解码。"""
    if len(flat) != MAX_AIC_CORES * 2:
        raise ValueError("core range block must contain 64 words")
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


def _md_validate_prefix(prefix, name):
    """task prefix 单调性校验。"""
    if not prefix or prefix[0] != 0:
        raise ValueError(f"{name} must start at zero")
    if any(e < s for s, e in zip(prefix, prefix[1:])):
        raise ValueError(f"{name} must be nondecreasing")


def _decode_round_prefix_checks(stage1_prefix, stage2_prefix,
                                chunk_num_per_group, dv_splits_num, value_heads):
    """prefix 结构与任务数一致性校验。"""
    _md_validate_prefix(stage1_prefix, "stage1_task_prefix")
    _md_validate_prefix(stage2_prefix, "stage2_task_prefix")
    for i, chunk_num in enumerate(chunk_num_per_group):
        if stage1_prefix[i + 1] - stage1_prefix[i] != value_heads * chunk_num:
            raise ValueError("stage1_task_prefix does not match chunk tasks")
        if stage2_prefix[i + 1] - stage2_prefix[i] != value_heads * dv_splits_num[i]:
            raise ValueError("stage2_task_prefix does not match DV tasks")


def _decode_header(words):
    """解析并校验 metadata 头部与偏移表，返回 (header, offsets)。"""
    if len(words) < HEADER_WORDS:
        raise ValueError("metadata is shorter than its header")
    if words[HEADER_MAGIC] != MAGIC:
        raise ValueError("metadata magic does not match")
    if words[HEADER_ABI_VERSION] != ABI_VERSION:
        raise ValueError("metadata ABI version does not match")
    status = words[HEADER_STATUS]
    header = {
        "magic": words[HEADER_MAGIC],
        "abi_version": words[HEADER_ABI_VERSION],
        "status": status,
        "core_num": words[HEADER_CORE_NUM],
        "stage12_round_num": words[HEADER_STAGE12_ROUND_NUM],
        "batch": words[HEADER_BATCH],
        "value_heads": words[HEADER_VALUE_HEADS],
        "has_cu": words[HEADER_HAS_CU],
        "workspace_slots": words[HEADER_WORKSPACE_SLOTS],
    }
    if status != STATUS_OK:
        header["used_words"] = HEADER_WORDS
        header["stage12_rounds"] = []
        return header, None
    if not 1 <= header["core_num"] <= MAX_AIC_CORES:
        raise ValueError(f"metadata core_num must be in [1, {MAX_AIC_CORES}]")
    if header["workspace_slots"] not in (WORKSPACE_SLOTS, LONG_DENSE_WORKSPACE_SLOTS):
        raise ValueError("unsupported metadata workspace_slots")
    if header["batch"] <= 0 or header["value_heads"] <= 0:
        raise ValueError("successful metadata requires positive shapes")
    round_num = int(header["stage12_round_num"])
    offset_begin = HEADER_WORDS
    offset_end = offset_begin + round_num + 1
    if offset_end > len(words):
        raise ValueError("metadata offset table is truncated")
    offsets = words[offset_begin:offset_end]
    if offsets[0] < offset_end:
        raise ValueError("metadata first record offset overlaps the offset table")
    if offsets[-1] > len(words):
        raise ValueError("metadata final offset exceeds capacity")
    if any(e <= s for s, e in zip(offsets, offsets[1:])):
        raise ValueError("metadata offsets must be strictly increasing")
    header["used_words"] = offsets[-1]
    header["stage12_rounds"] = []
    return header, offsets


def decode_metadata(raw):
    """Decode and validate metadata (reference diagnostic decoder)."""

    def take(words, cursor, count):
        end = cursor + count
        if end > len(words):
            raise ValueError("metadata record extends past available words")
        return words[cursor:end], end

    words = [int(x) for x in raw]
    header, offsets = _decode_header(words)
    if offsets is None:
        return header
    batch = int(header["batch"])
    value_heads = int(header["value_heads"])
    for index in range(int(header["stage12_round_num"])):
        start, end = offsets[index], offsets[index + 1]
        header["stage12_rounds"].append(
            _decode_round_record(words, start, end, batch, value_heads,
                                 header["workspace_slots"], take))
    return header


def _validate_layout(q, k, v, g, beta, layout_qkv, cu_seqlens):
    """按布局校验形状与 cu_seqlens，返回 (batch, seq_len, n_kv, n_v, dim)。"""
    if layout_qkv not in ("BNSD", "BSND", "TND"):
        raise ValueError("layout_qkv must be BNSD, BSND, or TND")
    if layout_qkv == "TND":
        return _validate_shape_tnd(q, k, v, g, beta, cu_seqlens)
    return _validate_shape_dense(q, k, v, g, beta, layout_qkv, cu_seqlens)


def _validate_shape_tnd(q, k, v, g, beta, cu_seqlens):
    """TND 布局校验。"""
    if cu_seqlens is None:
        raise ValueError("cu_seqlens is required for TND")
    if cu_seqlens.dtype != torch.int32 or cu_seqlens.dim() != 1:
        raise ValueError("cu_seqlens must be rank-1 int32")
    if not cu_seqlens.is_contiguous() or cu_seqlens.device != q.device:
        raise ValueError("cu_seqlens must be contiguous and on the same device")
    if cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must have shape [B+1] with positive batch")
    batch = cu_seqlens.numel() - 1
    if q.dim() != 3 or k.dim() != 3:
        raise ValueError("TND q and k must be rank-3")
    if v.dim() != 3 or g.dim() != 3:
        raise ValueError("TND v and g must be rank-3")
    seq_len, n_kv, dim = q.shape
    n_v = v.shape[1]
    if k.shape != q.shape or g.shape != v.shape or v.shape != (seq_len, n_v, dim):
        raise ValueError("TND k must match q and v/g must match the packed token length")
    if beta.dim() != 2 or beta.shape != (seq_len, n_v):
        raise ValueError("TND beta shape must be [T,Nv]")
    return batch, seq_len, n_kv, n_v, dim


def _validate_shape_dense(q, k, v, g, beta, layout_qkv, cu_seqlens):
    """BNSD/BSND 布局校验。"""
    if cu_seqlens is not None:
        raise ValueError("cu_seqlens is only valid for TND")
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError("BNSD/BSND q and k must be rank-4")
    if v.dim() != 4 or g.dim() != 4:
        raise ValueError("BNSD/BSND v and g must be rank-4")
    batch, dim = q.shape[0], q.shape[3]
    seq_len = q.shape[1] if layout_qkv == "BSND" else q.shape[2]
    n_v = v.shape[2] if layout_qkv == "BSND" else v.shape[1]
    n_kv = q.shape[2] if layout_qkv == "BSND" else q.shape[1]
    beta_shape = (batch, seq_len, n_v) if layout_qkv == "BSND" else (batch, n_v, seq_len)
    if k.shape != q.shape or g.shape != v.shape or v.shape != (*beta_shape, dim):
        raise ValueError("dense k must match q and v/g must match the layout and sequence length")
    if tuple(beta.shape) != beta_shape:
        raise ValueError("dense beta shape must match v without its final dimension")
    return batch, seq_len, n_kv, n_v, dim


def _validate_common_shapes(batch, seq_len, n_kv, n_v, dim, aux):
    """与布局无关的标量/参数形状校验（aux=(a_log, dt_bias, metadata)）。"""
    a_log, dt_bias, _metadata = aux
    if not (dim == D and batch > 0 and seq_len > 0):
        raise ValueError("head dimension requires D=128 and sequence length must be positive")
    if not (n_kv > 0 and n_v > 0 and n_v % n_kv == 0):
        raise ValueError("GQA requires Nv % Nk == 0")
    if tuple(a_log.shape) != (n_v,) or tuple(dt_bias.shape) != (n_v, D):
        raise ValueError("A_log/dt_bias shape must match Nv and D=128")


def _validate_common_devices(qkvgb, aux):
    """与布局无关的 dtype/设备/连续性校验（qkvgb=(q,k,v,g,beta)）。"""
    q, k, v, g, beta = qkvgb
    a_log, dt_bias, metadata = aux
    if not (q.dtype == k.dtype == v.dtype == g.dtype == torch.bfloat16):
        raise ValueError("q, k, v, and g must be bf16")
    if beta.dtype != torch.bfloat16 or beta.dim() not in (2, 3):
        raise ValueError("beta must be bf16 with rank 2 or 3")
    if a_log.dtype != torch.float32 or dt_bias.dtype != torch.float32:
        raise ValueError("A_log and dt_bias must be fp32")
    tensors = (q, k, v, g, beta, a_log, dt_bias)
    if not all(t.device == q.device for t in tensors):
        raise ValueError("all input tensors must be on the same device")
    if q.device.type not in {"npu", "privateuseone"}:
        raise ValueError("FlashKDA CP requires an NPU tensor")
    if not (a_log.is_contiguous() and dt_bias.is_contiguous()):
        raise ValueError("A_log and dt_bias must be contiguous")
    if metadata.dtype != torch.int32 or metadata.dim() != 1 or not metadata.is_contiguous():
        raise ValueError("metadata must be contiguous rank-1 int32")
    if metadata.device != q.device:
        raise ValueError("metadata must be on the same device")
    if not all(t.is_contiguous() for t in (q, k, v, g, beta)):
        raise ValueError("q, k, v, g, and beta must be contiguous")


def _validate_common(qkvgb, aux, batch, seq_len, n_kv, n_v, dim):
    """与布局无关的通用校验（qkvgb=(q,k,v,g,beta)；aux=(a_log,dt_bias,metadata)）。"""
    _validate_common_shapes(batch, seq_len, n_kv, n_v, dim, aux)
    _validate_common_devices(qkvgb, aux)


def _validate_inputs(q, k, v, g, beta, a_log, dt_bias, metadata, layout_qkv, cu_seqlens):
    batch, seq_len, n_kv, n_v, dim = _validate_layout(
        q, k, v, g, beta, layout_qkv, cu_seqlens)
    _validate_common((q, k, v, g, beta), (a_log, dt_bias, metadata),
                     batch, seq_len, n_kv, n_v, dim)
    return batch, seq_len, n_kv, n_v


class _OpBuffers(NamedTuple):
    """主算子 launch 所需的输出与 workspace 缓冲。"""
    qg: torch.Tensor
    kg: torch.Tensor
    decay: torch.Tensor
    u: torch.Tensor
    w: torch.Tensor
    mqk: torch.Tensor
    transition: torch.Tensor
    state_ext: torch.Tensor
    ws_ops16: torch.Tensor


def _allocate_op_buffers(batch, n_v, seq_len, capacity, layout_qkv, device, slots):
    """分配八个输出和工作区。主 kernel 将未使用的 decay 槽写零。"""
    if layout_qkv == "TND":
        buf_shape = (seq_len, n_v, D)
    else:
        buf_shape = (batch, n_v, seq_len, D)
    qg = torch.empty(buf_shape, dtype=torch.bfloat16, device=device)
    kg = torch.empty(buf_shape, dtype=torch.bfloat16, device=device)
    u = torch.empty(buf_shape, dtype=torch.bfloat16, device=device)
    w = torch.empty(buf_shape, dtype=torch.bfloat16, device=device)
    decay = torch.empty((batch, n_v, capacity, D), dtype=torch.float32, device=device)
    mqk = torch.empty((batch, n_v, capacity, CHUNK, CHUNK), dtype=torch.bfloat16,
                    device=device)
    transition = torch.empty((batch, n_v, D, D), dtype=torch.float32, device=device)
    state_ext = torch.empty((batch, n_v, D, D), dtype=torch.float32, device=device)
    # Stage 1 的 BF16 RIGHT 面板、βV 和 βK̃ 暂存区。
    ws_ops16 = torch.empty((slots, WS_OPS_ELEMS), dtype=torch.bfloat16,
                        device=device)
    return _OpBuffers(qg, kg, decay, u, w, mqk, transition, state_ext,
                      ws_ops16)


def _beta_launch_args(n_v, layout_qkv):
    """BNSD 连续读；BSND/TND 按头数选择整行或跨步读取。"""
    layout_code = {"BNSD": LAYOUT_BNSD, "BSND": LAYOUT_BSND,
                   "TND": LAYOUT_TND}[layout_qkv]
    nv_flat = 0
    if layout_qkv in ("BSND", "TND"):
        nv_flat = 1 if n_v <= 32 else 2
    return layout_code, nv_flat


_VP_DUMMY_CACHE = {}


def _vp_workspace(batch, n_v, capacity, s2_split, device, share_phi):
    """为当前调用分配 φ 和零标志。

    不启用共享 φ 时不访问这两个指针，传占位张量。
    ws_phi 槽位 = (logical_batch*nv + head)*capacity + chunk，FP32 [D,D]；
    ws_zero 每个摘要行片分配 128 个 INT32。首词初始化为非零存活标志。
    """
    if not share_phi:
        key = (str(device.type), device.index, "dummy")
        cached = _VP_DUMMY_CACHE.get(key)
        if cached is None:
            cached = (torch.empty(4, dtype=torch.float32, device=device),
                      torch.empty(4, dtype=torch.int32, device=device))
            _VP_DUMMY_CACHE[key] = cached
        return cached
    phi_slots = batch * n_v * capacity
    ws_phi = torch.empty(phi_slots * D * D, dtype=torch.float32, device=device)
    ws_zero = torch.empty(batch * n_v * s2_split * 128, dtype=torch.int32, device=device)
    return ws_phi, ws_zero


def flash_kda_cp_preprocess(
    q, k, v, g, beta, scale, a_log, dt_bias, lower_bound, layout_qkv, metadata,
    cu_seqlens=None,
):
    """返回八项预处理结果，由一次 MIX kernel 完成设备计算。

    TND 的 cu_seqlens 与 metadata 必须来自同一次布局规划。
    序列边界由 flash_kda_cp_preprocess_metadata 校验并写入调度表。
    """
    batch, seq_len, n_kv, n_v = _validate_inputs(
        q, k, v, g, beta, a_log, dt_bias, metadata, layout_qkv, cu_seqlens)
    if not (-5.0 <= float(lower_bound) <= 0.0):
        raise ValueError("lower_bound must be in [-5, 0]")
    device = q.device
    device_index = device.index if device.index is not None else torch.npu.current_device()
    core_num = _current_aic_core_num(device_index)

    capacity = chunk_capacity_upper_bound(batch, seq_len)
    slots = _stage1_slot_bound(batch, n_v, seq_len, layout_qkv)
    bufs = _allocate_op_buffers(batch, n_v, seq_len, capacity, layout_qkv,
                                device, slots)
    layout_code, nv_flat = _beta_launch_args(n_v, layout_qkv)
    constants = _constants_for(device)
    s2_split = _s2_split_count(batch * n_v, core_num)
    short_summary = (layout_qkv == "BSND" and batch == 1 and seq_len <= CHUNK
                     and s2_split == 4)
    if short_summary:
        s2_split = 1
    workspace_slots = _workspace_slots_for(batch, n_v, seq_len, layout_qkv, cu_seqlens)
    # Enough row slices reuse each shared product to cover the GM handoff.
    share_phi = s2_split >= 4 or (
        s2_split == 2 and workspace_slots == LONG_DENSE_WORKSPACE_SLOTS)
    ws_phi, ws_zero = _vp_workspace(batch, n_v, capacity, s2_split, device, share_phi)
    flash_kda_cp_preprocess_kernel[None, core_num, {
        "Layout": layout_code,
        "NvFlat": nv_flat,
        "S2DV": s2_split,
        "SharePhi": int(share_phi),
        "PadDecay": int(batch > 1),
        "ShortSummary": int(short_summary),
    }](
        q, k, v, g, beta, a_log, dt_bias, metadata,
        bufs.qg, bufs.kg, bufs.decay, bufs.u, bufs.w, bufs.mqk, bufs.ws_ops16,
        constants, bufs.transition, bufs.state_ext,
        batch, n_v, n_kv, seq_len, seq_len, capacity, slots,
        float(scale), float(lower_bound), 4,
        ws_phi, ws_zero,
    )
    return FlashKDACPOutputs(bufs.transition, bufs.state_ext, bufs.decay,
                             bufs.qg, bufs.kg, bufs.u, bufs.w, bufs.mqk)
