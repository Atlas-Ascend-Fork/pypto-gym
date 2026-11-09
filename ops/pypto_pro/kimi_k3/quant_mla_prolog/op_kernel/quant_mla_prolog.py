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
# ruff: noqa: F821  # MlaKey 的字段由 parser 注入体内，对 linter 是未定义名

"""quant_mla_prolog —— PyPTO Pro 实现（kimi_k3 MLA prolog）。

支持域：T ≥ 1 任意整数 × quant_mode_c ∈ {0,1}。单 @pl.jit + 单 TilingKey，
6 个 section、3 道 MIX 屏障；结构见 DESIGN.md。数据通路：

    x ⊗ descale_x @ wqa ⊗ descale_wqa → qa_raw[T,1536] FP32    （M1，Cube）
                   @ wkva ⊗ descale_wkva → kva_raw[T,576] FP32 （M2，Cube）
    qa_raw → RMSNorm → BF16 降位 → per-32 MX → qa_mx + qa_scale（M3，Vector）
    qa_mx ⊗ qa_scale @ wqb ⊗ descale_wqb → BF16 降位 → q_head  （M4，Cube）
    kva_raw → RMSNorm(512,γ_kva) ∥ RoPE64 绕 BF16 → per-tensor 量化
            → PA_NZ 散写（slot<0 跳过）→ kv_cache 原位         （M5，Vector）
    q_head[:,h,0:128] @ wkb[h] → q_latent[96,T,512] BF16        （M6，Cube）
    concat(q_latent, q_head RoPE64) → per-(token,head) absmax 量化
            → q[T,96,576] E4M3 + descale_q[T,96] FP32          （M7，Vector）
"""

import functools
import math
from typing import NamedTuple

import pypto_pro.language as pl
import torch
from pypto_pro.language import Vf as vf
from pypto_pro.runtime.jit import get_current_arch
from pypto_pro.runtime.platform import get_memory_limit
from pypto_pro.runtime.tilingkey import TilingKeyField

# 算子关键常量
DIM = 7168                 # x 的特征维，QA 的 K
X_SCALE_BYTES = DIM // 32  # 224，x 一行的 E8M0 字节数
Q_LORA = 1536              # qa 的宽度

BASE_N_32 = 32             # 权重 fractal 宽度，= WEIGHT_FRACTAL（KVA 的 N 分块）
BASE_K = 256               # K 分段 → 28 段
BASE_N_QA = 64             # QA 的 N 分块 → 24 段
N_SEG_QA = Q_LORA // BASE_N_QA      # 24
K_SEG_QA = DIM // BASE_K            # 28

K_L1_QA = 1024             # L1 的 K 窗口宽，内层切 BASE_K 段喂 L0B
K1_WIN_QA = DIM // K_L1_QA          # 7 个 K1024 窗口
K1_SEG_QA = K_L1_QA // BASE_K       # 4 段/窗口（m=128 的取值）


def _k_split(m):
    """按 m 选 front 的 L0 段宽：m<=64 用 512，m=128 只能 256（L0A 65536B）。"""
    base_k = 512 if m <= 64 else BASE_K
    return base_k, K_L1_QA // base_k, base_k // MX_BLOCK

FEATURES = 576             # kva 的宽度（NoPE 512 + RoPE 64）
N_SEG_KVA = FEATURES // BASE_N_32   # 18
K_ACCUM_WAYS = 1
L0C_DEPTH = 2

MX_BLOCK = 32              # 每个 E8M0 系数覆盖 K 向连续 32 元素
SCALE_K_L1 = BASE_K // MX_BLOCK     # 8，L0 侧 scale 的 K 向长度
SCALE_K1_L1 = K_L1_QA // MX_BLOCK   # 32，L1 窗口侧 scale 的 K 向长度
SCALE_GROUP_PAIR = MX_BLOCK * 2     # 64，一个 scale 对覆盖的元素数

WEIGHT_FRACTAL = 32        # 权重 NZ 物理布局的末维

LANES_FP32 = 64            # 单寄存器 FP32 lane 数
VREG_Q_LORA = Q_LORA // LANES_FP32  # 24，一行 1536 需 24 个寄存器
VREG_FEATURES = FEATURES // LANES_FP32  # 9，一行 576 需 9 个寄存器

# M3 常量
# vf 体内不支持标量除法，倒数在模块级先算好
INV_Q_LORA = 1.0 / Q_LORA
NORM_EPS = 1e-06            # wrapper 的默认值；norm_eps 是运行期标量、非特化轴
FP8_MAX = 448.0
E8M0_BIAS = 8              # raw = max(exp - 8, 0)
E8M0_INV_BASE = 254        # 反 scale 的指数位 = 254 - raw
E8M0_UNITY = 127           # E8M0 的 1.0（值 = 2^(raw-127)），pad 行的 scale
FP32_EXP_SHIFT = 23
FP32_EXP_MASK = 255
SCALE_PAIRS = Q_LORA // LANES_FP32   # 24，每个 64-lane 寄存器出一对 (raw0, raw1)
SCALE_BYTES = SCALE_PAIRS * 2        # 48，一行 E8M0 字节数
SCALE_TILE_COLS = 64                 # tile 列数按 32B 对齐补到 64，valid_shape 限回 48

# M5 常量（KV epilogue）
KV_LORA = 512              # NoPE 段宽度，参与 RMSNorm
ROPE = 64                  # RoPE 段宽度，不归一但绕 BF16 一圈
INV_KV_LORA = 1.0 / KV_LORA          # vf 体内禁标量除法，倒数在模块级算好
VREG_KV_LORA = KV_LORA // LANES_FP32  # 8，归一段占 8 个寄存器
FRACTALS = FEATURES // 32  # 18，PA_NZ 的 fractal 块数（末维 C0=32）
C0 = 32                    # PA_NZ 物理末维
PAGE = 128                 # PA_NZ 每页行数
KVQ_BASE = 0x0CE00         # kvq_ub 起址，块视图地址表从它按 32B 步进

# M6 常量（WKB latent 投影）
QK_NOPE = 128              # q_head 每 head 的 NoPE 段宽度，WKB 的 K
BASE_N_WKB = 128           # N 分段：512/128 = 4 段。L0B 的 [128,128] BF16 为
                           # 32768B 单槽即整容量
N_SEG_WKB = KV_LORA // BASE_N_WKB   # 4

GM_DEPTH_WKB = 2            # 跨核 GM buffer depth（写它的 stage 是奇数编号，契约 ≥ 2）
MW_WKB = 32                 # M6/M7 的 m 粒度，与全局 m 解耦

# L1 地址图按 M 计算（结构见 DESIGN.md）；WSC_BYTES 是 2×4096、别按元素数重算
WSC_BYTES = 0x2000


def _l1_rows(m, x_depth=2):
    """L1 地址表（唯一真相源）：`(name, phase, slot_bases, slot_bytes)`。

    相位 A=front/B=M4/C=M6·M7 之间有 sync_all 硬屏障，故同址可复用。
    x_depth=1 时 x_l1 退单槽（M=128 装进 L1 的代价：丢 x 双缓冲）。
    """
    x_slot = m * K_L1_QA
    sx_slot = m * (DIM // MX_BLOCK)
    wseg_slot = K_L1_QA * BASE_N_QA
    wsc_slot = SCALE_K1_L1 * BASE_N_QA
    # 槽间距用 wqb 自己的尺寸（与 wseg_slot 相等只是巧合）
    wqb_slot = K_L1_QB * BASE_N_QB
    wqbsc_slot = SCALE_K1_QB * BASE_N_QB
    qa_slot = m * Q_LORA
    qas_slot = m * (Q_LORA // MX_BLOCK)
    wkb_slot = QK_NOPE * KV_LORA * 2

    sx = x_depth * x_slot
    wseg = sx + sx_slot
    wsc = wseg + 2 * wseg_slot
    qa = wsc + WSC_BYTES
    qas = qa + qa_slot
    x2 = qa if x_depth == 1 else x_slot
    wkb = [qa, 0] if x_depth == 1 else [qa]
    # qh 不能叠 wkb 也不能落进相位 B，取两者较大
    qh_base = max(qa + wkb_slot, qas + qas_slot)
    qh_slot = min(MW_WKB, m) * QK_NOPE * 2

    return [
        ("x", "A", [0], x_slot),
        ("x2", "A", [x2], x_slot),
        ("sx", "A", [sx], sx_slot),
        ("wseg", "A", [wseg, wseg + wseg_slot], wseg_slot),
        ("wsc", "A", [wsc, wsc + wsc_slot], wsc_slot),
        ("wqb", "B", [wseg, wseg + wqb_slot], wqb_slot),
        ("wqbsc", "B", [wsc, wsc + wqbsc_slot], wqbsc_slot),
        ("qa", "B", [qa], qa_slot),
        ("qas", "B", [qas], qas_slot),
        ("wkb", "C", wkb, wkb_slot),
        ("qh", "C", [qh_base, qh_base + qh_slot], qh_slot),
    ], qh_base + 2 * qh_slot


_L1_SCALAR = ("x", "x2", "sx", "wseg", "wsc", "qa", "qas")


def _check_layout(cond, msg):
    """编译期布局不变量闸门：用 raise 而非 assert（``-O`` 下 assert 会被剥掉）。"""
    if not cond:
        raise RuntimeError(msg)


def _l1_layout(m, x_depth=2):
    """按 m 返回 L1 各 buffer 的起址与末端字节数。"""
    rows, end = _l1_rows(m, x_depth)
    a = {name: (bases[0] if name in _L1_SCALAR else list(bases))
         for name, _phase, bases, _slot in rows}
    _check_layout(
        K_L1_QB * BASE_N_QB == K_L1_QA * BASE_N_QA,
        f"QB 与 QA 的 L1 窗口字节不等：{K_L1_QB * BASE_N_QB} != "
        f"{K_L1_QA * BASE_N_QA}")
    return a, end


def _l1_review(rows, cap):
    """地址断言：相位内不重叠、跨相位叠址须为包含关系、不越容量。"""
    flat = [(n, p, b, b + size) for n, p, bases, size in rows for b in bases]
    for name, _phase, begin, stop in flat:
        _check_layout(0 <= begin < stop <= cap,
                      f"{name} 的范围 [{begin},{stop}) 越出 [0,{cap})")
    alias = []
    for i, (ni, pi, bi, ei) in enumerate(flat):
        for nj, pj, bj, ej in flat[i + 1:]:
            if bi >= ej or bj >= ei:
                continue
            _check_layout(
                pi != pj,
                f"相位 {pi} 内 {ni}[{bi},{ei}) 与 {nj}[{bj},{ej}) 重叠")
            _check_layout(
                (bi <= bj and ej <= ei) or (bj <= bi and ei <= ej),
                f"跨相位叠址必须是包含关系，{ni}[{bi},{ei}) 与 "
                f"{nj}[{bj},{ej}) 只是部分重叠")
            alias.append((ni, pi, nj, pj))
    return alias

WKB_SLOT_BYTES = MW_WKB * KV_LORA * 2       # 32768
# slot 恒为 [MW_WKB, 512]；总量 = block_dim * WS_PER_CORE
WS_PER_CORE = GM_DEPTH_WKB * WKB_SLOT_BYTES    # 65536

# M7 常量（Q epilogue）
VREG_NOPE = KV_LORA // LANES_FP32        # 8，NoPE 段占 8 个寄存器，RoPE 占 1 个
INV_FP8_MAX = 1.0 / FP8_MAX              # 1/448，vf 体内禁标量除法故模块级算好
ABSMAX_FLOOR = 1e-8                      # absmax 下限，**先取下限再参与除法**
SCALE_ONE_COLS = 8           # descale_q 暂存（4B 不满 32B 对齐，声明 8 列再 valid 限 1）
QEPI_ROWS = 8                # M7 一槽行数（一次 memcpy 搬多行，GM 事务数降到 1/4）

# M4 常量（QB head projection）
N_HEADS = 96
HEAD_DIM = 192             # QK_NOPE 128 + ROPE 64
QB_WIDTH = N_HEADS * HEAD_DIM        # 18432
BASE_K_QB = 256              # M4 的 K/N 分段受 L0B 65536B 约束
K_L1_QB = 512                # M4 的 K 两级分块：L1 窗口宽
K1_WIN_QB = Q_LORA // K_L1_QB        # 3 个窗口
K1_SEG_QB = K_L1_QB // BASE_K_QB     # 2 段/窗口
SCALE_K1_QB = K_L1_QB // MX_BLOCK    # 16，L1 窗口侧 scale 的 K 向长度
BASE_N_QB = 128
N_SEG_QB = QB_WIDTH // BASE_N_QB     # 144


def _lr_layout(m):
    """按 m 返回 L0A/L0C 地址；a_l0a/a_qb/a_wkb 三相位同址分时复用。"""
    left = m * _k_split(m)[0]
    acc_qa = m * BASE_N_QA * 4
    acc_qb = m * BASE_N_QB * 4
    acc_wkb = min(MW_WKB, m) * BASE_N_WKB * 4
    base = L0C_DEPTH * acc_qa
    return {
        "left": [0, left],
        "l0c": [i * acc_qa for i in range(L0C_DEPTH)],
        "l0c_qb": base,
        "l0c_wkb": [base + acc_qb + i * acc_wkb for i in range(2)],
        "left_end": 2 * left,
        "acc_end": base + acc_qb + 2 * acc_wkb,
    }


K_SEG_QB = Q_LORA // BASE_K_QB       # 6
SCALE_K_QB = BASE_K_QB // MX_BLOCK   # 8，L1 侧 scale 的 K 向长度


@pl.vector_function
def gamma_roundtrip_1536_vf(src_tile, dst_tile):
    """γ 绕 BF16 一圈：FP32 → BF16(RNE) → FP32，逐 bit 同 .to(BF16).float()。"""
    pf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, VREG_Q_LORA, 1):
        off = r * LANES_FP32
        g = vf.load_align(src_tile, off)
        b = vf.astype(g, pf, dtype=pl.DT_BF16,
                      round_mode=pl.VFRoundMode.CAST_RINT)
        w = vf.astype(b, pf, dtype=pl.DT_FP32)
        vf.store_align(dst_tile, w, pf, off)
    vf.mem_bar()


@pl.vector_function
def gamma_roundtrip_512_vf(src_tile, dst_tile):
    """512 宽版（γ_kva 只覆盖 NoPE 段）。与 1536 版同形。"""
    pf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, VREG_KV_LORA, 1):
        off = r * LANES_FP32
        g = vf.load_align(src_tile, off)
        b = vf.astype(g, pf, dtype=pl.DT_BF16,
                      round_mode=pl.VFRoundMode.CAST_RINT)
        w = vf.astype(b, pf, dtype=pl.DT_FP32)
        vf.store_align(dst_tile, w, pf, off)
    vf.mem_bar()


@pl.vector_function
def qa_requant_vf(qa_tile, gamma_tile, out_tile, raw_tile, byte_tile, eps):
    """M3：RMSNorm(1536, γ_qa) → BF16 → per-32 块 MX 量化。"""
    pf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pi = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
    pu = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    m32i = vf.update_mask(32, dtype=pl.DT_INT32)

    # per-32 块：高 32 lane 搬低位算第二个 absmax
    rids = vf.arange(0, dtype=pl.DT_UINT32)
    c31 = vf.full(31, dtype=pl.DT_UINT32)
    rlo = vf.and_(rids, c31, pu)
    rhi = vf.adds(rlo, 32, pu)
    c32u = vf.full(32, dtype=pl.DT_UINT32)
    lt32 = vf.lt(rids, c32u, pu)

    acc = vf.full(0.0, dtype=pl.DT_FP32)
    for i in pl.range(0, VREG_Q_LORA, 1):
        off0 = i * LANES_FP32
        r = vf.load_align(qa_tile, off0)
        sq = vf.mul(r, r, pf)
        ssum = vf.reduce_sum(sq, pf)
        acc = vf.add(acc, ssum, pf)

    inv_n = vf.full(INV_Q_LORA, dtype=pl.DT_FP32)
    mean = vf.mul(acc, inv_n, pf)
    mean_e = vf.adds(mean, eps, pf)
    root = vf.sqrt(mean_e, pf)
    one = vf.full(1.0, dtype=pl.DT_FP32)
    invr = vf.div(one, root, pf)
    invb = vf.full(invr, pf, pos=pl.DuplicatePos.LOWEST)
    vf.mem_bar()

    c255 = vf.full(FP32_EXP_MASK, dtype=pl.DT_INT32)
    c254 = vf.full(E8M0_INV_BASE, dtype=pl.DT_INT32)
    zi = vf.full(0, dtype=pl.DT_INT32)
    zf = vf.full(0.0, dtype=pl.DT_FP32)
    hi_c = vf.full(FP8_MAX, dtype=pl.DT_FP32)
    lo_c = vf.full(-FP8_MAX, dtype=pl.DT_FP32)

    for i in pl.range(0, VREG_Q_LORA, 1):
        off = i * LANES_FP32
        r = vf.load_align(qa_tile, off)
        gm = vf.load_align(gamma_tile, off)
        nrm = vf.mul(r, invb, pf)
        wg = vf.mul(nrm, gm, pf)
        # FP32→BF16 用 CAST_ROUND（ties-away 语义）
        qb = vf.astype(wg, pf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        q32 = vf.astype(qb, pf, dtype=pl.DT_FP32)

        qbits = vf.bit_cast(q32, dtype=pl.DT_INT32)
        qsh = vf.shift_right(qbits, FP32_EXP_SHIFT, pi)
        qexp = vf.and_(qsh, c255, pi)
        e0 = vf.reduce_max(qexp, m32i)
        qexp_hi = vf.gather(qexp, rhi)
        e1 = vf.reduce_max(qexp_hi, m32i)

        s0 = vf.adds(e0, -E8M0_BIAS, pi)
        raw0 = vf.maxs(s0, 0, pi)
        s1 = vf.adds(e1, -E8M0_BIAS, pi)
        raw1 = vf.maxs(s1, 0, pi)

        # 反 scale = (254-raw)<<23；exp==0 时置 0（raw==0 判不出，exp=8 也给 0）
        d0 = vf.sub(c254, raw0, pi)
        b0 = vf.shift_left(d0, FP32_EXP_SHIFT, pi)
        i0 = vf.bit_cast(b0, dtype=pl.DT_FP32)
        nz0 = vf.ne(e0, zi, pi)
        i0s = vf.select(i0, zf, nz0)

        d1 = vf.sub(c254, raw1, pi)
        b1 = vf.shift_left(d1, FP32_EXP_SHIFT, pi)
        i1 = vf.bit_cast(b1, dtype=pl.DT_FP32)
        nz1 = vf.ne(e1, zi, pi)
        i1s = vf.select(i1, zf, nz1)

        # lane<32 取组 0，否则组 1
        i0b = vf.full(i0s, pf, pos=pl.DuplicatePos.LOWEST)
        i1b = vf.full(i1s, pf, pos=pl.DuplicatePos.LOWEST)
        inv = vf.select(i0b, i1b, lt32)

        # 乘反 scale → 再降一次 BF16（此步无需舍入：q32 低 16 位恒 0、inv 是纯
        # 2 的幂）→ clamp → E4M3
        mq = vf.mul(q32, inv, pf)
        mqb = vf.astype(mq, pf, dtype=pl.DT_BF16,
                        round_mode=pl.VFRoundMode.CAST_RINT)
        mq32 = vf.astype(mqb, pf, dtype=pl.DT_FP32)
        cl = vf.min(mq32, hi_c, pf)
        cl2 = vf.max(cl, lo_c, pf)
        f8 = vf.astype(cl2, pf, dtype=pl.DT_FP8E4M3FN,
                       layout=pl.CastLayout.ZERO,
                       round_mode=pl.VFRoundMode.CAST_RINT,
                       saturate=pl.SaturateMode.ON)
        # 连续 FP8 落盘必须 PACK4：四路 CastLayout 是同一组 lane 的四个字节位
        vf.store_align(out_tile, f8, pf, off, dist=pl.StoreDist.PACK4)

        o2 = i * 2
        vf.store_align(raw_tile, raw0, pi, o2,
                       dist=pl.StoreDist.FIRST_ELEMENT_B32)
        o2b = o2 + 1
        vf.store_align(raw_tile, raw1, pi, o2b,
                       dist=pl.StoreDist.FIRST_ELEMENT_B32)
    vf.mem_bar()

    # 第三遍：48 个 INT32 raw 窄化成连续 E8M0 字节（M4 的 A 侧 scale 在片上读）
    m48 = vf.update_mask(SCALE_BYTES, dtype=pl.DT_INT32)
    rr = vf.load_align(raw_tile, 0)
    ru8 = vf.astype(rr, m48, dtype=pl.DT_UINT8,
                    layout=pl.CastLayout.ZERO,
                    saturate=pl.SaturateMode.ON)
    vf.store_align(byte_tile, ru8, m48, 0, dist=pl.StoreDist.PACK4)
    vf.mem_bar()


@pl.vector_function
def kv_epilogue_vf(kv_tile, gamma_tile, out_tile, qscale, eps):
    """M5：RMSNorm(前 512, γ_kva) ∥ RoPE64 绕 BF16 → per-tensor 量化（C1 版）。

    出口量化必须 vf.div 真除法：乘倒数会差 1 个 FP8 ulp。
    """
    pf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)

    acc = vf.full(0.0, dtype=pl.DT_FP32)
    for i in pl.range(0, VREG_KV_LORA, 1):
        off0 = i * LANES_FP32
        r = vf.load_align(kv_tile, off0)
        sq = vf.mul(r, r, pf)
        ssum = vf.reduce_sum(sq, pf)
        acc = vf.add(acc, ssum, pf)

    inv_n = vf.full(INV_KV_LORA, dtype=pl.DT_FP32)
    mean = vf.mul(acc, inv_n, pf)
    mean_e = vf.adds(mean, eps, pf)
    root = vf.sqrt(mean_e, pf)
    one = vf.full(1.0, dtype=pl.DT_FP32)
    invr = vf.div(one, root, pf)
    invb = vf.full(invr, pf, pos=pl.DuplicatePos.LOWEST)
    vf.mem_bar()

    # Scalar 模式广播必须给 dtype 不能给 pos（混写在 codegen 报 vdup 不匹配）
    qs = vf.full(qscale, dtype=pl.DT_FP32)
    hi_c = vf.full(FP8_MAX, dtype=pl.DT_FP32)
    lo_c = vf.full(-FP8_MAX, dtype=pl.DT_FP32)

    for i in pl.range(0, VREG_KV_LORA, 1):
        off = i * LANES_FP32
        r = vf.load_align(kv_tile, off)
        gm = vf.load_align(gamma_tile, off)
        nrm = vf.mul(r, invb, pf)
        wg = vf.mul(nrm, gm, pf)
        sc = vf.div(wg, qs, pf)
        cl = vf.min(sc, hi_c, pf)
        cl2 = vf.max(cl, lo_c, pf)
        f8 = vf.astype(cl2, pf, dtype=pl.DT_FP8E4M3FN,
                       layout=pl.CastLayout.ZERO,
                       round_mode=pl.VFRoundMode.CAST_RINT,
                       saturate=pl.SaturateMode.ON)
        vf.store_align(out_tile, f8, pf, off, dist=pl.StoreDist.PACK4)

    roff = KV_LORA
    rr = vf.load_align(kv_tile, roff)
    # 此段 BF16 用 CAST_RINT（ties-to-even）；QA 路相反用 CAST_ROUND（ties-away）
    rb = vf.astype(rr, pf, dtype=pl.DT_BF16,
                   round_mode=pl.VFRoundMode.CAST_RINT)
    r32 = vf.astype(rb, pf, dtype=pl.DT_FP32)
    rsc = vf.div(r32, qs, pf)
    rcl = vf.min(rsc, hi_c, pf)
    rcl2 = vf.max(rcl, lo_c, pf)
    rf8 = vf.astype(rcl2, pf, dtype=pl.DT_FP8E4M3FN,
                    layout=pl.CastLayout.ZERO,
                    round_mode=pl.VFRoundMode.CAST_RINT,
                    saturate=pl.SaturateMode.ON)
    vf.store_align(out_tile, rf8, pf, roff, dist=pl.StoreDist.PACK4)
    vf.mem_bar()


@pl.vector_function
def kv_epilogue_bf16_vf(kv_tile, gamma_tile, out_tile, eps):
    """M5 的 C0 版：出口不量化、各段直接降 BF16。BF16 落盘必须
    CastLayout.ZERO + PACK_B32（通用 PACK 会选 PK_B16，出来是垃圾）。
    """
    pf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)

    acc = vf.full(0.0, dtype=pl.DT_FP32)
    for i in pl.range(0, VREG_KV_LORA, 1):
        r = vf.load_align(kv_tile, i * LANES_FP32)
        sq = vf.mul(r, r, pf)
        ssum = vf.reduce_sum(sq, pf)
        acc = vf.add(acc, ssum, pf)

    inv_n = vf.full(INV_KV_LORA, dtype=pl.DT_FP32)
    mean = vf.mul(acc, inv_n, pf)
    mean_e = vf.adds(mean, eps, pf)
    root = vf.sqrt(mean_e, pf)
    one = vf.full(1.0, dtype=pl.DT_FP32)
    invr = vf.div(one, root, pf)
    invb = vf.full(invr, pf, pos=pl.DuplicatePos.LOWEST)
    vf.mem_bar()

    for i in pl.range(0, VREG_KV_LORA, 1):
        off = i * LANES_FP32
        r = vf.load_align(kv_tile, off)
        gm = vf.load_align(gamma_tile, off)
        nrm = vf.mul(r, invb, pf)
        wg = vf.mul(nrm, gm, pf)
        b16 = vf.astype(wg, pf, dtype=pl.DT_BF16,
                        layout=pl.CastLayout.ZERO,
                        round_mode=pl.VFRoundMode.CAST_RINT)
        vf.store_align(out_tile, b16, pf, off, dist=pl.StoreDist.PACK_B32)

    # RoPE 64 段只降一次 BF16（参考先绕一圈再降，第二次落在已可表示值上是恒等）
    rr = vf.load_align(kv_tile, KV_LORA)
    rb = vf.astype(rr, pf, dtype=pl.DT_BF16,
                   layout=pl.CastLayout.ZERO,
                   round_mode=pl.VFRoundMode.CAST_RINT)
    vf.store_align(out_tile, rb, pf, KV_LORA, dist=pl.StoreDist.PACK_B32)
    vf.mem_bar()


@pl.vector_function
def q_epilogue_vf(nope_tile, rope_tile, out_tile, scale_tile):
    """concat(NoPE512, RoPE64) → per-token-head absmax 量化，一次 QEPI_ROWS 行。

    行循环必须 pl.range（vf 内 python 原生 for 被拒）。BF16 源 tile 读固定
    dist=UNPK_B16（默认按 32 位 lane 读，一个 lane 吃掉两个 BF16）。
    """
    pf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    c448 = vf.full(INV_FP8_MAX, dtype=pl.DT_FP32)
    hi = vf.full(FP8_MAX, dtype=pl.DT_FP32)
    lo = vf.full(-FP8_MAX, dtype=pl.DT_FP32)

    for row in pl.range(0, QEPI_ROWS, 1):
        n_base = row * KV_LORA
        r_base = row * ROPE
        o_base = row * FEATURES

        amax = vf.full(0.0, dtype=pl.DT_FP32)
        for i in pl.range(0, VREG_NOPE, 1):
            off = n_base + i * LANES_FP32
            rb = vf.load_align(nope_tile, off, dist=pl.LoadDist.UNPK_B16)
            r32 = vf.astype(rb, pf, dtype=pl.DT_FP32)
            ra = vf.abs(r32, pf)
            rm = vf.reduce_max(ra, pf)
            amax = vf.max(amax, rm, pf)
        rrb = vf.load_align(rope_tile, r_base, dist=pl.LoadDist.UNPK_B16)
        rr32 = vf.astype(rrb, pf, dtype=pl.DT_FP32)
        rra = vf.abs(rr32, pf)
        rrm = vf.reduce_max(rra, pf)
        amax = vf.max(amax, rrm, pf)

        # 下限先取再参与除法（顺序固定：clamp_min(1e-8) / 448）
        amax_c = vf.maxs(amax, ABSMAX_FLOOR, pf)
        scale = vf.mul(amax_c, c448, pf)
        # scale 自身就是 descale_q，是终输出，不得只留片上
        vf.store_align(scale_tile, scale, pf, row * SCALE_ONE_COLS,
                       dist=pl.StoreDist.FIRST_ELEMENT_B32)
        sb = vf.full(scale, pf, pos=pl.DuplicatePos.LOWEST)
        vf.mem_bar()

        # 用 vf.div 真除法（NPU 除法比 CPU 少一个 ULP 朝零，乘倒数会偏）
        for i in pl.range(0, VREG_NOPE, 1):
            off = i * LANES_FP32
            rb = vf.load_align(nope_tile, n_base + off,
                               dist=pl.LoadDist.UNPK_B16)
            r32 = vf.astype(rb, pf, dtype=pl.DT_FP32)
            rq = vf.div(r32, sb, pf)
            c1 = vf.min(rq, hi, pf)
            c2 = vf.max(c1, lo, pf)
            f8 = vf.astype(c2, pf, dtype=pl.DT_FP8E4M3FN,
                           layout=pl.CastLayout.ZERO,
                           round_mode=pl.VFRoundMode.CAST_RINT,
                           saturate=pl.SaturateMode.ON)
            vf.store_align(out_tile, f8, pf, o_base + off,
                           dist=pl.StoreDist.PACK4)
        rb2 = vf.load_align(rope_tile, r_base, dist=pl.LoadDist.UNPK_B16)
        r322 = vf.astype(rb2, pf, dtype=pl.DT_FP32)
        rq2 = vf.div(r322, sb, pf)
        d1 = vf.min(rq2, hi, pf)
        d2 = vf.max(d1, lo, pf)
        f82 = vf.astype(d2, pf, dtype=pl.DT_FP8E4M3FN,
                        layout=pl.CastLayout.ZERO,
                        round_mode=pl.VFRoundMode.CAST_RINT,
                        saturate=pl.SaturateMode.ON)
        vf.store_align(out_tile, f82, pf, o_base + KV_LORA,
                       dist=pl.StoreDist.PACK4)
    vf.mem_bar()


# M6/M7 的自动 CV 并行流水 stage：一拍 = 一个 (head, m 块)，q_latent 走跨核
# GM buffer。末拍部分核无 head 由体内 `h < N_HEADS` 兜（if 控制整个 stage 须为
# 编译期常量）。


def _mm_k0_seg(k0, k1, n_width, sc_base, base_k, scale_k, k1_seg,
               xl, sxl, cur_w, cur_sw,
               a_l0a, sa_l0a_mx, b_l0b, sb_l0b_mx, acc):
    """内层 K 段。k0 是 python 常量（边界判断编译期塌掉）；k1 动态、判断留运行期
    且嵌套写（python 的 and 触发 Expr.__bool__）。
    """
    al = a_l0a.next()
    sal = sa_l0a_mx.next()
    br = b_l0b.next()
    sbl = sb_l0b_mx.next()

    pl.set_validshape(br, [base_k, n_width])
    pl.set_validshape(sbl, [scale_k, n_width])

    pl.move(al, xl, [0, k0 * base_k])
    pl.move(sal, sxl, [0, sc_base + k0 * scale_k])
    pl.move(br, cur_w, [k0 * base_k, 0])
    pl.move(sbl, cur_sw, [k0 * scale_k, 0])

    if k0 == 0:
        if k1 == 0:
            pl.matmul_mx(acc, al, br, sal, sbl, phase=pl.AccPhase.Partial)
        else:
            pl.matmul_mx_acc(acc, acc, al, br, sal, sbl,
                             phase=pl.AccPhase.Partial)
    elif k0 == k1_seg - 1:
        if k1 == K1_WIN_QA - 1:
            pl.matmul_mx_acc(acc, acc, al, br, sal, sbl,
                             phase=pl.AccPhase.Final)
        else:
            pl.matmul_mx_acc(acc, acc, al, br, sal, sbl,
                             phase=pl.AccPhase.Partial)
    else:
        pl.matmul_mx_acc(acc, acc, al, br, sal, sbl,
                         phase=pl.AccPhase.Partial)


def _mm_side(k1, n, n_width, base_k, m_rows, m_idx, xl, sxl, w_gm, w_fracs,
             w_k_total, sw_gm, w_m, w_move, wsc,
             a_l0a, sa_l0a_mx, b_l0b, sb_l0b_mx, l0c, slot, out_gm):
    """交织拍内的一侧（QA 或 KVA）：搬本窗口权重 + K 段累加，末窗落盘。两侧共用
    x 窗口；权重走 merged-row 视图、每窗一次 TLOAD。
    """
    # valid_shape 整条 K 链恒定，段首设一次
    cur_w = w_move.current()
    cur_sw = wsc.current()
    n_frac = n_width // WEIGHT_FRACTAL
    w_win = pl.make_tensor(
        pl.addptr(pl.make_ptr(w_gm), k1 * K_L1_QA * WEIGHT_FRACTAL),
        [w_fracs, K_L1_QA, WEIGHT_FRACTAL],
        stride=[w_k_total * WEIGHT_FRACTAL, WEIGHT_FRACTAL, 1],
        dtype=pl.DT_FP8E4M3FN)
    pl.load(w_m.current(), w_win, [n * n_frac, 0, 0], order=[0, 1, 2])
    pl.load(cur_sw, sw_gm,
            [n * n_width, k1 * K_L1_QA // SCALE_GROUP_PAIR, 0], order=[1, 0])

    # 两侧各占一个 L0C 槽（L0C_DEPTH=2 正好够）
    acc = l0c[slot]
    k1_seg = K_L1_QA // base_k
    scale_k = base_k // MX_BLOCK
    sc_base = k1 * SCALE_K1_L1
    _mm_k0_seg(0, k1, n_width, sc_base, base_k, scale_k, k1_seg,
               xl, sxl, cur_w, cur_sw,
               a_l0a, sa_l0a_mx, b_l0b, sb_l0b_mx, acc)
    _mm_k0_seg(1, k1, n_width, sc_base, base_k, scale_k, k1_seg,
               xl, sxl, cur_w, cur_sw,
               a_l0a, sa_l0a_mx, b_l0b, sb_l0b_mx, acc)
    # base_k=512 时每窗只有 2 段，下面两次按 python 常量塌掉（m<=64 走这条）
    if k1_seg == 4:
        _mm_k0_seg(2, k1, n_width, sc_base, base_k, scale_k, k1_seg,
                   xl, sxl, cur_w, cur_sw,
                   a_l0a, sa_l0a_mx, b_l0b, sb_l0b_mx, acc)
        _mm_k0_seg(3, k1, n_width, sc_base, base_k, scale_k, k1_seg,
                   xl, sxl, cur_w, cur_sw,
                   a_l0a, sa_l0a_mx, b_l0b, sb_l0b_mx, acc)
    if k1 == K1_WIN_QA - 1:
        # slot 取法必须是独立简单赋值（框架不接受内联在调用里的 l0c[0]）
        out = l0c[slot]
        pl.store(out_gm, out, [0, m_idx * m_rows, n * n_width],
                 phase=pl.STPhase.Final)


@pl.pipeline.stage
def stage_w_mm(tick, core_id, num_cores, m_rows, n_blocks,
               base_k, tokens,
               x, descale_x, x_a, x_b, sx_l1,
               wqa, wkva, descale_wqa, descale_wkva,
               wva_m, wva_nz, wvb_m, wvb_nz, wsc_a, wsc_b,
               a_l0a, sa_l0a_mx, b_l0b, sb_l0b_mx, l0c,
               qa_partial, kva_partial):
    """Cube：QA/KVA 交织的一拍，共用一个 x 窗口（拍序 = n/m_idx/k1 压一维）。
    x 直接读 [T, DIM]（valid-M 语义），pad 行垃圾被行独立性圈住。
    """
    chain = tick // K1_WIN_QA
    k1 = tick - chain * K1_WIN_QA
    n_local = chain // n_blocks
    m_idx = chain - n_local * n_blocks
    n = core_id + n_local * num_cores
    # 超出两侧段数的核连 x 都不用搬
    n_seg_max = max(N_SEG_QA, N_SEG_KVA)
    if n < n_seg_max:
        m_valid = pl.min(m_rows, tokens - m_idx * m_rows)
        # k1 偶用 x_a、奇用 x_b；本拍窗口由上一拍预取，本拍只发 k1+1 的预取
        half = k1 // 2
        if k1 - half * 2 == 0:
            xl = x_a.current()
            xn = x_b.current()
        else:
            xl = x_b.current()
            xn = x_a.current()
        sxl = sx_l1.current()
        if k1 == 0:
            # 链首自填（整条链只在首窗搬一次全 K 的 scale）
            pl.set_validshape(xl, [m_valid, K_L1_QA])
            pl.load(xl, x, [m_idx * m_rows, 0])
            pl.set_validshape(xl, [m_rows, K_L1_QA])
            pl.set_validshape(sxl, [m_valid, DIM // MX_BLOCK])
            pl.load(sxl, descale_x, [m_idx * m_rows, 0, 0])
            pl.set_validshape(sxl, [m_rows, DIM // MX_BLOCK])
        if k1 + 1 < K1_WIN_QA:
            pl.set_validshape(xn, [m_valid, K_L1_QA])
            pl.load(xn, x, [m_idx * m_rows, (k1 + 1) * K_L1_QA])
            pl.set_validshape(xn, [m_rows, K_L1_QA])
        if n < N_SEG_QA:
            _mm_side(k1, n, BASE_N_QA, base_k, m_rows, m_idx, xl, sxl,
                     wqa, Q_LORA // WEIGHT_FRACTAL, DIM,
                     descale_wqa, wva_m, wva_nz, wsc_a,
                     a_l0a, sa_l0a_mx, b_l0b, sb_l0b_mx, l0c, 0,
                     qa_partial)
        if n < N_SEG_KVA:
            _mm_side(k1, n, BASE_N_32, base_k, m_rows, m_idx, xl, sxl,
                     wkva, FEATURES // WEIGHT_FRACTAL, DIM,
                     descale_wkva, wvb_m, wvb_nz, wsc_b,
                     a_l0a, sa_l0a_mx, b_l0b, sb_l0b_mx, l0c, 1,
                     kva_partial)


def _head_tile_index(tick, core_id, num_cores, n_blocks):
    """本拍的 (head, m 块号, 本核总拍数)：全局拍号按 head-major 拆开。

    全局线性 tile 空间：96 个 head 对多数核数不整除，线性分最慢核拍数最少。
    M6 的 WKB 段与 C1 / C0 两个 Q epilogue 段共用这段索引算术，纯标量、不碰
    tile。`per_core` 一并返回，供 WKB 段判本核末拍。
    """
    per_core = (N_HEADS * n_blocks + num_cores - 1) // num_cores
    gt = core_id * per_core + tick
    h = gt // n_blocks
    return h, gt - h * n_blocks, per_core


@pl.pipeline.stage
def stage_wkb(tick, core_id, num_cores, mw, n_blocks, q_head, wkb,
              wkb_l1, qh_l1, a_wkb, b_wkb, l0c_wkb, ql_gm):
    """Cube：一个 (head, 32 行块) 的 WKB latent 投影，写进跨核 GM slot。"""
    h, m_idx, per_core = _head_tile_index(tick, core_id, num_cores, n_blocks)
    if h < N_HEADS:
        qh = qh_l1.next()
        wl = wkb_l1.current()
        # 第 h 个 head 的 NoPE 段：列 h*192 起、宽 128（后 64 列 RoPE 不用）
        pl.load(qh, q_head, [m_idx * mw, h * HEAD_DIM])
        # W 只在换 head 时需要：由上一 head 末拍预取，首拍 tick==0 补
        if tick == 0:
            pl.load(wl, wkb, [h, 0, 0], order=[1, 2])

        al = a_wkb.next()
        pl.move(al, qh)
        for n in pl.range(0, N_SEG_WKB, 1):
            br = b_wkb.next()
            acc_w = l0c_wkb.next()
            pl.move(br, wl, [0, n * BASE_N_WKB])
            pl.matmul(acc_w, al, br, phase=pl.AccPhase.Final)
            # L0C→GM 随路 FP32→BF16
            pl.store(ql_gm, acc_w, [0, n * BASE_N_WKB],
                     phase=pl.STPhase.Final)
        # 末拍预取下一 head 的 W；本核末拍不预取（第二槽叠在 x_l1 上）
        if m_idx == n_blocks - 1 and tick + 1 < per_core:
            h_next = h + 1
            if h_next < N_HEADS:
                wl_nxt = wkb_l1.next()
                pl.load(wl_nxt, wkb, [h_next, 0, 0],
                        order=[1, 2])


@pl.pipeline.stage
def stage_q_epi(tick, core_id, num_cores, sub_id, sub_num, m_rows, n_blocks,
                q_head, ql_gm, nope_ub, rope_ub, q_ub, dsq_ub,
                q_out, descale_q):
    """Vector：本拍 (head, m 块) 的 concat + absmax 量化。上界按 m_rows（pad 行
    照样算落 q_out，host 返回前 [:tokens] 切回）。
    """
    h, m_idx, _ = _head_tile_index(tick, core_id, num_cores, n_blocks)
    row0 = m_idx * m_rows
    if h < N_HEADS:
        for b in pl.range(sub_id, m_rows // QEPI_ROWS, sub_num):
            t = b * QEPI_ROWS
            nb = nope_ub.next()
            rb = rope_ub.next()
            qb = q_ub.next()
            db = dsq_ub.next()
            pl.set_validshape(db, [QEPI_ROWS, 1])
            pl.load(nb, ql_gm, [t, 0])
            pl.load(rb, q_head, [row0 + t, h * HEAD_DIM + QK_NOPE])
            q_epilogue_vf(nb, rb, qb, db)
            pl.store(q_out, qb, [row0 + t, h * FEATURES])
            pl.store(descale_q, db, [row0 + t, h])


@pl.pipeline.stage
def stage_q_epi_c0(tick, core_id, num_cores, sub_id, sub_num, m_rows,
                   n_blocks, q_head, ql_gm, nope_ub, rope_ub, q_out):
    """Vector：C0 的 Q epilogue —— 纯 concat 搬运、无量化。"""
    h, m_idx, _ = _head_tile_index(tick, core_id, num_cores, n_blocks)
    row0 = m_idx * m_rows
    if h < N_HEADS:
        for b in pl.range(sub_id, m_rows // QEPI_ROWS, sub_num):
            t = b * QEPI_ROWS
            nb = nope_ub.next()
            rb = rope_ub.next()
            pl.load(nb, ql_gm, [t, 0])
            pl.load(rb, q_head, [row0 + t, h * HEAD_DIM + QK_NOPE])
            pl.store(q_out, nb, [row0 + t, h * FEATURES])
            pl.store(q_out, rb, [row0 + t, h * FEATURES + KV_LORA])


# M3 / M4 / M5 的段体（kernel 体在对应 section 内调用，屏障由 section 持有）


def _module3_qa_requant(core_id, nc, tokens_v, gamma_qa, norm_eps,
                         qa_partial, qa_mx, qa_scale_mx,
                         gamma_ub, gamr_ub, qa_out_ub, qamx_ub, qasc_ub,
                         qab_ub, qasc_e8_ub):
    """M3 段体：RMSNorm(1536, γ_qa) → BF16 → per-32 MX 量化，按 token 分核。
    上界按 capacity（M4 要整块 qa_mx 进 L1）。
    """
    # 行归属只用 block 维（sub_id 不进 start/stride，否则要另加守卫）
    sub_id = pl.get_subblock_idx()
    if sub_id == 0:
        # γ 全核共用，T 循环外搬一次，随即片上绕一圈 BF16 驻留 gamr
        gam = gamma_ub.current()
        pl.load(gam, gamma_qa, [0, 0])
        gamr = gamr_ub.current()
        gamma_roundtrip_1536_vf(gam, gamr)
        for t in pl.range(core_id, tokens_v, nc):
            outb = qa_out_ub.next()
            pl.load(outb, qa_partial, [0, t, 0])
            qmx = qamx_ub.next()
            qsc = qasc_ub.next()
            qb8 = qab_ub.next()
            qa_requant_vf(outb, gamr, qmx, qsc, qb8, norm_eps)
            pl.store(qa_mx, qmx, [t, 0])
            qsc8 = qasc_e8_ub.next()
            pl.set_validshape(qsc8, [1, SCALE_BYTES])
            pl.store(qa_scale_mx, qsc8, [t, 0, 0])


def _qb_k_seg(k1, acc, ql, qsl, cw, csw, a_qb, sa_qb, b_qb, sb_qb):
    """M4 的一个 L1 K 窗口内的 L0 段循环：搬 A/B 与各自 scale，再累加 matmul。

    只含 ``pl.*``：调用发生在被 trace 的 kernel 体内，其中的 ``pl.*`` 照常被
    记录，发射序列与内联时相同。抽出来是为了让调用方的嵌套层数落回门禁阈值内。
    """
    for k0 in pl.range(0, K1_SEG_QB, 1):
        al = a_qb.next()
        sal = sa_qb.next()
        br = b_qb.next()
        sbl = sb_qb.next()

        k = k1 * K1_SEG_QB + k0
        pl.move(al, ql, [0, k * BASE_K_QB])
        pl.move(sal, qsl, [0, k * BASE_K_QB // MX_BLOCK])
        pl.move(br, cw, [k0 * BASE_K_QB, 0])
        pl.move(sbl, csw, [k0 * SCALE_K_QB, 0])

        if k == 0:
            pl.matmul_mx(acc, al, br, sal, sbl, phase=pl.AccPhase.Partial)
        elif k == K_SEG_QB - 1:
            pl.matmul_mx_acc(acc, acc, al, br, sal, sbl,
                             phase=pl.AccPhase.Final)
        else:
            pl.matmul_mx_acc(acc, acc, al, br, sal, sbl,
                             phase=pl.AccPhase.Partial)


def _module4_qb_heads(core_id, nc, n_blocks, m_rows,
                       qa_mx, qa_scale_mx, wqb, descale_wqb, q_head,
                       qa_l1, qas_l1, wqb_m, wqb_nz, wqbsc_l1,
                       a_qb, sa_qb, b_qb, sb_qb, l0c_qb):
    """M4 段体：QB head projection，K 两级分块（L1 512 窗口 × 2 段 256）。"""
    n_frac = BASE_N_QB // WEIGHT_FRACTAL
    w_fracs = QB_WIDTH // WEIGHT_FRACTAL
    w_k_total = Q_LORA
    for m_idx in pl.range(0, n_blocks, 1):
        # 行数恒为 m_rows（pad 行是定义过的值）；A 操作数全核共用
        ql = qa_l1.current()
        qsl = qas_l1.current()
        pl.load_tile(ql, qa_mx, [m_idx, 0])
        pl.load(qsl, qa_scale_mx, [m_idx * m_rows, 0, 0])

        for n in pl.range(core_id, N_SEG_QB, nc):
            acc = l0c_qb.current()
            for k1 in pl.range(0, K1_WIN_QB, 1):
                cwm = wqb_m.next()
                w_win = pl.make_tensor(
                    pl.addptr(pl.make_ptr(wqb), k1 * K_L1_QB * WEIGHT_FRACTAL),
                    [w_fracs, K_L1_QB, WEIGHT_FRACTAL],
                    stride=[w_k_total * WEIGHT_FRACTAL, WEIGHT_FRACTAL, 1],
                    dtype=pl.DT_FP8E4M3FN)
                pl.load(cwm, w_win, [n * n_frac, 0, 0], order=[0, 1, 2])
                cw = wqb_nz.current()
                csw = wqbsc_l1.next()
                pl.load(csw, descale_wqb,
                        [n * BASE_N_QB, k1 * K_L1_QB // SCALE_GROUP_PAIR, 0],
                        order=[1, 0])

                _qb_k_seg(k1, acc, ql, qsl, cw, csw,
                          a_qb, sa_qb, b_qb, sb_qb)

            # FP32→BF16 靠 L0C→GM 随路转换（不配 scale 即支持该路径）
            pl.store(q_head, acc, [m_idx * m_rows, n * BASE_N_QB],
                     phase=pl.STPhase.Final)


def _kv_scatter_row(t, cache_index, kv_cache, kvblk_ub, n_cache_fractals):
    """把第 t 行 KV 按 cache_index 散写进 PA_NZ cache。

    行归属由 owner 模型保证唯一（无需原子）；``slot < 0`` 整行不写。只含
    ``pl.*``，抽出来只为降低调用方的嵌套层数，发射序列与内联时相同。
    """
    slot = pl.getval(cache_index, t)
    if slot >= 0:
        page = slot // PAGE
        row = slot % PAGE
        for fb in pl.range(0, n_cache_fractals, 1):
            pl.store(kv_cache, kvblk_ub[fb], [page, fb, row, 0])


def _module5_kv_scatter(core_id, nc, tokens_kv, gamma_kva, kva_partial,
                        qscale_kv, cache_index, kv_cache, quant_mode_c,
                        norm_eps, n_cache_fractals,
                        gkva_ub, gkvar_ub, kva_out_ub, kvq_ub, kvblk_ub):
    """M5 段体（C0/C1 共用骨架）：kva 归约 → KV epilogue → PA_NZ 散写。
    quant_mode_c 是装饰期常量，选枝在编译期折叠；C0 不读 qscale_kv。
    """
    gkva = gkva_ub.current()
    pl.load(gkva, gamma_kva, [0, 0])
    gkvar = gkvar_ub.current()
    gamma_roundtrip_512_vf(gkva, gkvar)
    if quant_mode_c:
        qs_v = pl.getval(qscale_kv, 0)
    if pl.get_subblock_idx() == 0:
        for t in pl.range(core_id, tokens_kv, nc):
            kout = kva_out_ub.next()
            pl.load(kout, kva_partial, [0, t, 0])

            kvq = kvq_ub.current()
            if quant_mode_c:
                kv_epilogue_vf(kout, gkvar, kvq, qs_v, norm_eps)
            else:
                kv_epilogue_bf16_vf(kout, gkvar, kvq, norm_eps)

            _kv_scatter_row(t, cache_index, kv_cache, kvblk_ub,
                            n_cache_fractals)


MIN_AIC_CORE_COUNT, MAX_AIC_CORE_COUNT = 1, 32


@functools.cache
def _device_aic_core_num(device_index):
    """硬件 AIC 上限；stream 配额是运行期量、不进缓存。"""
    props = torch.npu.get_device_properties(device_index)
    core_num = min(int(getattr(props, "cube_core_num", 0)), MAX_AIC_CORE_COUNT)
    if not MIN_AIC_CORE_COUNT <= core_num <= MAX_AIC_CORE_COUNT:
        raise RuntimeError(
            "quant_mla_prolog requires an available AIC count in "
            f"[{MIN_AIC_CORE_COUNT}, {MAX_AIC_CORE_COUNT}], got {core_num}")
    return core_num


def _current_aic_core_num(device_index):
    """本 stream 实际的 Mix launch 宽度。1 Mix block = 1 AIC + 2 AIV。"""
    core_num = _device_aic_core_num(device_index)
    get_limit = getattr(torch.npu, "get_stream_limit", None)
    if get_limit is not None:
        limits = get_limit(torch.npu.current_stream(device_index))
        core_num = min(core_num, int(limits["cube_core_num"]),
                       int(limits["vector_core_num"]) // 2)
    if not MIN_AIC_CORE_COUNT <= core_num <= MAX_AIC_CORE_COUNT:
        raise RuntimeError(
            "quant_mla_prolog needs at least one AIC and two AIVs on "
            f"the current stream; resolved Mix block count is {core_num}")
    return core_num


# 支持域、分核映射与数值边界见 DESIGN.md；容量断言在 _l1_tables / _lr_tables。


# TilingKey：编译期特化轴
class MlaKey:
    """特化轴：MRows（tile 行数，定地址表/tile 形状/front 段宽）与 QuantModeC
    （cache 与 q 的 dtype、fractal 块数）。n_blocks 只改拍数、核数运行期读、
    norm_eps 是运行期标量 —— 均不进 key。
    """

    MRows = TilingKeyField(bits=2, values=[16, 32, 64, 128])
    QuantModeC = TilingKeyField(bits=1, values=[0, 1])


# 按 key 推的表只在 kernel 体内编译期实参位置调（体内 dict 赋值不受支持）。


# 片上容量从当前安装解析。arch 必须取 get_current_arch()（编译器当轮用的）而非
# get_platform_info().arch —— 后者在无卡 shell 里把 A5 误解成 A2，越界查不出来。
_SPACE_KEY = {"Mat": "l1_size", "Left": "l0_a_size",
              "Right": "l0_b_size", "Acc": "l0_c_size", "Vec": "ub_size"}


@functools.cache
def _space_limit(space):
    """space 取 MemorySpace 名：Mat=L1、Left=L0A、Acc=L0C。"""
    arch = get_current_arch()
    limit = get_memory_limit(arch, space)
    if limit <= 0:
        raise RuntimeError(
            f"解析不出 {space} 的容量（arch={arch}，平台配置里的 key 是 "
            f"{_SPACE_KEY[space]}）。不回落到记住的 A5 值。")
    return limit


@functools.cache
def _l1_tables(m):
    """`(L1_ADDR, L1_END, x_depth)`。M>=128 时 x_l1 退单槽（双槽超 L1 容量）。"""
    x_depth = 1 if m >= 128 else 2
    addr, end = _l1_layout(m, x_depth=x_depth)
    cap = _space_limit("Mat")
    _check_layout(end <= cap, f"L1 超限：{end} > {cap}（M={m}）")
    _l1_review(_l1_rows(m, x_depth)[0], cap)
    return addr, end, x_depth


def _l1(m, name):
    return _l1_tables(m)[0][name]


def _x_depth(m):
    return _l1_tables(m)[2]


@functools.cache
def _lr_tables(m):
    """L0A/L0C 地址表（含容量断言）。"""
    lr = _lr_layout(m)
    left_cap = _space_limit("Left")
    acc_cap = _space_limit("Acc")
    _check_layout(lr["left_end"] <= left_cap,
                  f"L0A 超限：{lr['left_end']} > {left_cap}（M={m}）")
    _check_layout(lr["acc_end"] <= acc_cap,
                  f"L0C 超限：{lr['acc_end']} > {acc_cap}（M={m}）")
    return lr


def _lr(m, name):
    return _lr_tables(m)[name]


def _ks(m, i):
    """`_k_split` 的第 i 项。"""
    return _k_split(m)[i]


@functools.cache
def _cache_tables(quant_mode_c):
    """`(cache_c0, cache_fractals, KVBLK_ADDRS, KVBLK_MUTEX)`。块视图地址表在
    kvq_ub 同一行上按 32B 步进切块（C1 18 块 × 32 列 FP8、C0 36 块 × 16 列
    BF16，每块恰好 32B、步进不随 mode 变）。
    """
    cache_c0 = C0 if quant_mode_c else 16
    cache_fractals = FEATURES // cache_c0
    kvq_base = KVQ_BASE
    addrs = [kvq_base + i * 32 for i in range(cache_fractals)]
    _check_layout(addrs[-1] + 32 <= 0x10000,
                  "kvq/kvblk 段越过 M7 槽区起点")
    return cache_c0, cache_fractals, addrs, [15] * cache_fractals


def _cache(quant_mode_c, i):
    """`_cache_tables` 的第 i 项。"""
    return _cache_tables(quant_mode_c)[i]


@pl.jit(auto_mutex=True, tiling_key=MlaKey, name="quant_mla_prolog",
        pipeline=pl.pipeline.PipelineConfig(preload=4))
def quant_mla_prolog_kernel(
    x: pl.Tensor[[pl.DYNAMIC, DIM], pl.DT_FP8E4M3FN],
    descale_x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
    wqa: pl.Tensor[[Q_LORA // WEIGHT_FRACTAL, DIM, WEIGHT_FRACTAL],
                   pl.DT_FP8E4M3FN],
    descale_wqa: pl.Tensor[[Q_LORA, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
    wkva: pl.Tensor[[FEATURES // WEIGHT_FRACTAL, DIM, WEIGHT_FRACTAL],
                    pl.DT_FP8E4M3FN],
    descale_wkva: pl.Tensor[[FEATURES, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
    qa_partial: pl.Tensor[[K_ACCUM_WAYS, pl.DYNAMIC, Q_LORA], pl.DT_FP32],
    kva_partial: pl.Tensor[[K_ACCUM_WAYS, pl.DYNAMIC, FEATURES], pl.DT_FP32],
    wqb: pl.Tensor[[QB_WIDTH // WEIGHT_FRACTAL, Q_LORA, WEIGHT_FRACTAL],
                   pl.DT_FP8E4M3FN],
    descale_wqb: pl.Tensor[[QB_WIDTH, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
    gamma_qa_ptr: pl.Ptr[pl.DT_FP32],
    qa_mx: pl.Tensor[[pl.DYNAMIC, Q_LORA], pl.DT_FP8E4M3FN],
    qa_scale_mx: pl.Tensor[[pl.DYNAMIC, SCALE_PAIRS, 2], pl.DT_FP8E8M0],
    q_head: pl.Tensor[[pl.STATIC, QB_WIDTH], pl.DT_BF16],
    gamma_kva_ptr: pl.Ptr[pl.DT_FP32],
    qscale_kv: pl.Tensor[[1], pl.DT_FP32],
    cache_index: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64],
    kv_cache_ptr: pl.Ptr[pl.DT_UINT8],
    wkb: pl.Tensor[[N_HEADS, QK_NOPE, KV_LORA], pl.DT_BF16],
    ws: pl.Ptr[pl.DT_UINT8],
    q_out_ptr: pl.Ptr[pl.DT_UINT8],
    descale_q: pl.Tensor[[pl.DYNAMIC, N_HEADS], pl.DT_FP32],
    norm_eps: pl.DT_FP32,
    kv_pages: pl.DT_INT64,
):
    """Module 1..7 主体。跨 section 中转量走 GM；地址与 mutex_id 见 DESIGN.md。"""
    # kv_cache / q_out 的 dtype 随 QuantModeC 变而 key 值绑定前不可用：入参收成
    # pl.Ptr[DT_UINT8]，体内 make_tensor 补类型与形状
    if QuantModeC:
        out_dtype = pl.DT_FP8E4M3FN
    else:
        out_dtype = pl.DT_BF16
    # capacity 取 q_head 的 STATIC 行维
    capacity = q_head.shape[0]
    n_blocks = capacity // MRows
    kv_cache = pl.make_tensor(
        kv_cache_ptr,
        [kv_pages, _cache(QuantModeC, 1), PAGE, _cache(QuantModeC, 0)],
        dtype=out_dtype)
    q_out = pl.make_tensor(
        q_out_ptr, [capacity, N_HEADS * FEATURES], dtype=out_dtype)
    gamma_qa = pl.make_tensor(gamma_qa_ptr, [1, Q_LORA], dtype=pl.DT_FP32)
    gamma_kva = pl.make_tensor(gamma_kva_ptr, [1, KV_LORA], dtype=pl.DT_FP32)

    # L1(Mat)。x 只驻当前 K 窗口，两槽轮转、各持一个 mutex；MRows=128 时 b 槽
    # 叠 qa 借 mutex 14（同址必须共享 mutex）。
    x_mutex_b = 14 if _x_depth(MRows) == 1 else 7
    x_a = pl.make_tile_group(
        type=pl.TileType(shape=[MRows, K_L1_QA], dtype=pl.DT_FP8E4M3FN,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                         valid_shape=[-1, -1]),
        addrs=_l1(MRows, "x"), mutex_ids=[6])
    x_b = pl.make_tile_group(
        type=pl.TileType(shape=[MRows, K_L1_QA], dtype=pl.DT_FP8E4M3FN,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                         valid_shape=[-1, -1]),
        addrs=_l1(MRows, "x2"), mutex_ids=[x_mutex_b])
    sx_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[MRows, DIM // MX_BLOCK],
                         dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZZ,
                         valid_shape=[-1, -1]),
        addrs=_l1(MRows, "sx"), mutex_ids=[9])
    wseg_slot = K_L1_QA * BASE_N_QA
    wsc_slot = SCALE_K1_L1 * BASE_N_QA
    # merged-row 合并装载：order=[0,1,2] 把 rank-3 的 (fractal,k) 并进 tile 行，
    # 字节与 [K_win, N_win] NZ tile 相同，一次 TLOAD 覆盖整窗；a 归 QA、b 归 KVA。
    wva_m = pl.make_tile_group(
        type=pl.TileType(shape=[K_L1_QA * (BASE_N_QA // WEIGHT_FRACTAL),
                                WEIGHT_FRACTAL],
                         dtype=pl.DT_FP8E4M3FN,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                         valid_shape=[-1, -1]),
        addrs=_l1(MRows, "wseg"), mutex_ids=[0])
    wva_nz = pl.reinterpret(wva_m, shape=[K_L1_QA, BASE_N_QA], layout=pl.NZ)
    # 声明按整槽给足（同尺寸才能 reinterpret）；KVA 实际只装前 32 列，valid 收窄
    wvb_m = pl.make_tile_group(
        type=pl.TileType(shape=[K_L1_QA * (BASE_N_QA // WEIGHT_FRACTAL),
                                WEIGHT_FRACTAL],
                         dtype=pl.DT_FP8E4M3FN,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                         valid_shape=[-1, -1]),
        addrs=_l1(MRows, "wseg") + wseg_slot, mutex_ids=[1])
    wvb_nz = pl.reinterpret(wvb_m, shape=[K_L1_QA, BASE_N_QA], layout=pl.NZ)
    wsc_a = pl.make_tile_group(
        type=pl.TileType(shape=[SCALE_K1_L1, BASE_N_QA], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NN,
                         valid_shape=[-1, -1]),
        addrs=_l1(MRows, "wsc"), mutex_ids=[2])
    wsc_b = pl.make_tile_group(
        type=pl.TileType(shape=[SCALE_K1_L1, BASE_N_QA], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NN,
                         valid_shape=[-1, -1]),
        addrs=_l1(MRows, "wsc") + wsc_slot, mutex_ids=[3])

    # M4 的 A 操作数：qa_mx 常驻 L1。同址复用两条硬约束（编译期拦）：槽数必须
    # 相同、必须共用 mutex id。
    qa_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[MRows, Q_LORA], dtype=pl.DT_FP8E4M3FN,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=_l1(MRows, "qa"), mutex_ids=[14])
    qas_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[MRows, Q_LORA // MX_BLOCK],
                         dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZZ),
        addrs=_l1(MRows, "qas"), mutex_ids=[14])

    # M4 权重段：K=512 窗口双缓冲，与相位 A 权重槽同址复用（sync_all 隔开）；
    # mutex 必须留在 [0,1]、不与 x_l1 共锁（共锁会把 M4 与 Front 串行化）。
    wqb_m = pl.make_tile_group(
        type=pl.TileType(shape=[K_L1_QB * (BASE_N_QB // WEIGHT_FRACTAL),
                                WEIGHT_FRACTAL],
                         dtype=pl.DT_FP8E4M3FN,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                         valid_shape=[-1, -1]),
        addrs=_l1(MRows, "wqb"), mutex_ids=[0, 1])
    wqb_nz = pl.reinterpret(wqb_m, shape=[K_L1_QB, BASE_N_QB], layout=pl.NZ)
    wqbsc_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[SCALE_K1_QB, BASE_N_QB], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NN,
                         valid_shape=[-1, -1]),
        addrs=_l1(MRows, "wqbsc"), mutex_ids=[2, 3])

    # L0A / L0A_MX（scale 地址 = 主 tile 地址 >> 4）
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[MRows, _ks(MRows, 0)], dtype=pl.DT_FP8E4M3FN,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ, compact=1),
        addrs=_lr(MRows, "left"), mutex_ids=[6, 7])
    sa_l0a_mx = pl.make_tile_group(
        type=pl.TileType(shape=[MRows, _ks(MRows, 2)], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.ScaleLeft, layout=pl.ZZ,
                         compact=1),
        addrs=[a >> 4 for a in _lr(MRows, "left")], mutex_ids=[6, 7])

    # L0B / L0B_MX
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[_ks(MRows, 0), BASE_N_QA], dtype=pl.DT_FP8E4M3FN,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                         valid_shape=[-1, -1], compact=1),
        # 槽距取 0x8000 而非本组槽长：否则 base_k=256 档的槽 1 与 b_qb 槽 0 同址
        # 不同 mutex（编译期 MutexMismatch）
        addrs=[0, 0x08000], mutex_ids=[8, 9])
    sb_l0b_mx = pl.make_tile_group(
        type=pl.TileType(shape=[_ks(MRows, 2), BASE_N_QA], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.ScaleRight, layout=pl.NN,
                         valid_shape=[-1, -1], compact=1),
        addrs=[0, 0x08000 >> 4], mutex_ids=[8, 9])

    # M4 的 L0A/L0B：与 M1/M2 同形、分时复用同址
    a_qb = pl.make_tile_group(
        type=pl.TileType(shape=[MRows, BASE_K_QB], dtype=pl.DT_FP8E4M3FN,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ,
                         compact=1),
        addrs=_lr(MRows, "left"), mutex_ids=[6, 7])
    sa_qb = pl.make_tile_group(
        type=pl.TileType(shape=[MRows, SCALE_K_QB], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.ScaleLeft, layout=pl.ZZ,
                         compact=1),
        addrs=[a >> 4 for a in _lr(MRows, "left")], mutex_ids=[6, 7])
    b_qb = pl.make_tile_group(
        type=pl.TileType(shape=[BASE_K_QB, BASE_N_QB], dtype=pl.DT_FP8E4M3FN,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                         valid_shape=[-1, -1], compact=1),
        addrs=[0x00000, 0x08000], mutex_ids=[8, 9])
    sb_qb = pl.make_tile_group(
        type=pl.TileType(shape=[SCALE_K_QB, BASE_N_QB], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.ScaleRight, layout=pl.NN,
                         valid_shape=[-1, -1], compact=1),
        addrs=[0x00000 >> 4, 0x08000 >> 4], mutex_ids=[8, 9])

    # L0C：Front 段累加器（K_ACCUM_WAYS 路）
    l0c = pl.make_tile_group(
        type=pl.TileType(shape=[MRows, BASE_N_QA], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         valid_shape=[-1, -1], compact=1),
        addrs=_lr(MRows, "l0c"),
        mutex_ids=[10, 11, 12, 13][:L0C_DEPTH])
    l0c_qb = pl.make_tile_group(
        type=pl.TileType(shape=[MRows, BASE_N_QB], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         compact=1),
        # 与 l0c / l0c_wkb 不重叠故不共锁（多余共锁把 M4 与 Front 串行化）
        addrs=_lr(MRows, "l0c_qb"), mutex_ids=[12])

    core_id = pl.get_block_idx() // pl.get_subblock_num()
    # 任务范围按设备侧实际核数算，host 只把它当 block_dim 传
    nc = pl.get_block_num()

    # UB(Vec)：QA 链的输入/输出行缓冲各 depth=2，单 L0C 累加后低址区空出给第二槽
    qa_out_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, Q_LORA], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[0x06000, 0x00000], mutex_ids=[4, 0])

    # KVA 侧行缓冲 depth=2；不与 QA 同址（同一 token 迭代里两路都要在手）
    kva_out_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, FEATURES], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[0x09C00, 0x07800], mutex_ids=[9, 5])

    # M3：γ 驻留 + FP8 输出 + scale 暂存。BF16 绕行结果驻留 gamr_ub
    gamma_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, Q_LORA], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=0x0A500, mutex_ids=[10])
    gamr_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, Q_LORA], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=0x30000, mutex_ids=[22])
    # FP8 输出 depth=2；第二槽用已死的 part_ub 槽 1（mutex 1）
    qamx_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, Q_LORA], dtype=pl.DT_FP8E4M3FN,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[0x0BD00, 0x01800], mutex_ids=[11, 1])
    # scale 两级：先 INT32 暂存（FIRST_ELEMENT_B32 按 32 位粒度写）再窄化成连续
    # E8M0 字节（M4 的 A 侧 scale 在片上读 E8M0，窄化不能出 kernel）。FIRST_ELEMENT_B8
    # 偏移仍按 32 位元素计（有效值只落字节 0/4/8/…），不能按字节下标直写。
    qasc_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, SCALE_BYTES], dtype=pl.DT_INT32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[0x0C400, 0x01E00], mutex_ids=[12, 2])
    # 字节区：8 位类型 tile 列数须满足 32B 对齐（声明 64 列、valid_shape 限回 48）
    qab_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, SCALE_TILE_COLS], dtype=pl.DT_UINT8,
                         target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1]),
        addrs=[0x0C500, 0x01F00], mutex_ids=[13, 3])
    # 同址 E8M0 视图：两个 group 各有独立游标，depth=2 下循环里必须对两者各调
    # 一次 next()，否则第二拍起读错槽（地址与 shape 都对、错得无声）。
    qasc_e8_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, SCALE_TILE_COLS], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1]),
        addrs=[0x0C500, 0x01F00], mutex_ids=[13, 3])

    # M5：γ_kva 驻留 + KV 输出行 + fractal 块视图（γ_kva 只覆盖 NoPE 512 段）
    gkva_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, KV_LORA], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=0x0C600, mutex_ids=[14])
    gkvar_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, KV_LORA], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=0x31800, mutex_ids=[23])
    kvq_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, FEATURES], dtype=out_dtype,
                         target_memory=pl.MemorySpace.Vec),
        addrs=KVQ_BASE, mutex_ids=[15])
    kvblk_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, _cache(QuantModeC, 0)], dtype=out_dtype,
                         target_memory=pl.MemorySpace.Vec),
        addrs=_cache(QuantModeC, 2), mutex_ids=_cache(QuantModeC, 3))

    # M7 片上资源（Q epilogue 相位），各 depth=2
    nope_ub = pl.make_tile_group(
        type=pl.TileType(shape=[QEPI_ROWS, KV_LORA], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[0x10000, 0x12000], mutex_ids=[16, 24])
    rope_ub = pl.make_tile_group(
        type=pl.TileType(shape=[QEPI_ROWS, ROPE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[0x14000, 0x14400], mutex_ids=[17, 25])
    q_ub = pl.make_tile_group(
        type=pl.TileType(shape=[QEPI_ROWS, FEATURES], dtype=pl.DT_FP8E4M3FN,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[0x15000, 0x16200], mutex_ids=[18, 26])
    # 每行 scale 仅 4B 不满 32B 对齐 → 声明 8 列、valid_shape 限回 1 列
    dsq_ub = pl.make_tile_group(
        type=pl.TileType(shape=[QEPI_ROWS, SCALE_ONE_COLS],
                         dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1]),
        addrs=[0x18000, 0x18100], mutex_ids=[19, 27])

    # M6 片上资源（WKB 相位）；N 的 4 段切分交给 move 的源偏移
    wkb_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[QK_NOPE, KV_LORA], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                         valid_shape=[-1, -1]),
        # 第二槽与 x_l1 同址，必须共享它的 mutex 6（同址复用约束）
        addrs=_l1(MRows, "wkb"), mutex_ids=[14, 6][:len(_l1(MRows, "wkb"))])
    qh_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[min(MW_WKB, MRows), QK_NOPE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=_l1(MRows, "qh"), mutex_ids=[15, 28])

    # L0A 双槽：两槽各落在 a_l0a 的 left 区间内（mutex 6/7 对齐），分时复用同址
    a_wkb = pl.make_tile_group(
        type=pl.TileType(shape=[min(MW_WKB, MRows), QK_NOPE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ,
                         compact=1),
        addrs=_lr(MRows, "left"), mutex_ids=[6, 7])
    # L0B 双槽（各占半容量）：与 M4 的 b_qb 同形同址同 mutex（32768B，2↔2）
    b_wkb = pl.make_tile_group(
        type=pl.TileType(shape=[QK_NOPE, BASE_N_WKB], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                         valid_shape=[-1, -1], compact=1),
        addrs=[0x00000, 0x08000], mutex_ids=[8, 9])
    # L0C 双槽（按 MW_WKB 切 MRows，双缓冲比单个整块省 L0C）
    l0c_wkb = pl.make_tile_group(
        type=pl.TileType(shape=[min(MW_WKB, MRows), BASE_N_WKB], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         compact=1),
        addrs=_lr(MRows, "l0c_wkb"), mutex_ids=[13, 14])


    # M6→M7 跨核共享 GM buffer：每核切自己的区域（Cube 与 Vector 同核号）；
    # fwd/bwd 由框架自动插。事件 id 0~3 不与屏障冲突（MIX 屏障占 11~13）。
    ws_core = pl.addptr(ws, core_id * WS_PER_CORE)
    ql_gm = pl.make_tensor(
        ws_core, [MW_WKB, KV_LORA], dtype=pl.DT_BF16,
        depth=GM_DEPTH_WKB,
        fwd_ids=[0, 1],
        bwd_ids=[2, 3],
    )

    tokens = x.shape[0]

    # front 直读 [T, DIM]（valid-M 语义）；pad 垃圾被行独立性圈住

    with pl.section_cube():
        # valid_shape 整条 K 链恒定、段首设一次（不依赖「未设定时默认取静态形状」）
        vs_a = wsc_a.current()
        vs_b = wsc_b.current()
        vc_a = l0c[0]
        vc_b = l0c[1]
        pl.set_validshape(vs_a, [SCALE_K1_L1, BASE_N_QA])
        pl.set_validshape(vs_b, [SCALE_K1_L1, BASE_N_32])
        pl.set_validshape(vc_a, [MRows, BASE_N_QA])
        pl.set_validshape(vc_b, [MRows, BASE_N_32])
        # 权重 merged tile 与其 reinterpret 视图同样整链恒定，KVA 侧按 32 列收窄
        pl.set_validshape(wva_m.current(),
                          [K_L1_QA * (BASE_N_QA // WEIGHT_FRACTAL),
                           WEIGHT_FRACTAL])
        pl.set_validshape(wva_nz.current(), [K_L1_QA, BASE_N_QA])
        pl.set_validshape(wvb_m.current(), [K_L1_QA, WEIGHT_FRACTAL])
        pl.set_validshape(wvb_nz.current(), [K_L1_QA, BASE_N_32])

    # (n, m_idx, k1) 压成一维拍序；QA 与 KVA 交织、取两侧轮数的较大者
    rounds_qa = (N_SEG_QA + nc - 1) // nc
    rounds_kva = (N_SEG_KVA + nc - 1) // nc
    n_ticks_front = max(rounds_qa, rounds_kva) * n_blocks * K1_WIN_QA

    # Module 1 + 2：QA / KVA 下投影（同一条流水；共享 wseg_l1，框架要求每个
    # 循环自带完整的生产/消费对）
    for tick in pl.range(0, n_ticks_front, 1):
        with pl.section_cube():
            stage_w_mm(tick, core_id, nc,
                       MRows, n_blocks, _ks(MRows, 0), tokens,
                       x, descale_x, x_a, x_b, sx_l1,
                       wqa, wkva, descale_wqa, descale_wkva,
                       wva_m, wva_nz, wvb_m, wvb_nz, wsc_a, wsc_b,
                       a_l0a, sa_l0a_mx, b_l0b, sb_l0b_mx, l0c,
                       qa_partial, kva_partial)

    with pl.section_cube():
        # Cube→Vector 的 GM 写后读：auto_mutex 只管 Tile 地址复用，不覆盖跨
        # section 的 GM 可见性
        pl.system.sync_all()

    with pl.section_vector():
        # 与 cube 段尾的 sync_all 配对（MIX 模式必须两侧各一次）
        pl.system.sync_all()

        # QA 链按 token 分核、单子块干活（KVA 链在 B2 之后与 M4+M6 并发）
        _module3_qa_requant(core_id, nc, capacity, gamma_qa, norm_eps,
                            qa_partial, qa_mx, qa_scale_mx,
                            gamma_ub, gamr_ub, qa_out_ub, qamx_ub, qasc_ub,
                            qab_ub, qasc_e8_ub)

        # 普通 section 的 store 要先 bar_all 排空 MTE3 再进屏障（否则可见性不成立）
        pl.system.bar_all()
        pl.system.sync_all()

    with pl.section_cube():
        pl.system.sync_all()

        # qa_l1 / qas_l1 是单槽常驻的 A 操作数，每个 MRows 块重新装一次。
        _module4_qb_heads(core_id, nc, n_blocks, MRows,
                          qa_mx, qa_scale_mx, wqb, descale_wqb, q_head,
                          qa_l1, qas_l1, wqb_m, wqb_nz, wqbsc_l1,
                          a_qb, sa_qb, b_qb, sb_qb, l0c_qb)

        # M4 → M6 是跨核 GM 写后读（head 的列跨 n 段分核）；屏障取 MIX：M6 在
        # 流水循环里，循环要求 Cube/Vector 严格交替。
        pl.system.bar_all()
        pl.system.sync_all()

    with pl.section_vector():
        # CV 重叠：KVA 链与 M4 并发（零读写交集，真依赖 kva_partial 由 B1 保护），
        # 两条 AIV 都跑、按 core_id 分 token。必须排在本段屏障之前 —— 那是
        # M4→M6 的对齐点，排在它后面就串回去了。
        _module5_kv_scatter(core_id, nc, x.shape[0], gamma_kva, kva_partial,
                            qscale_kv, cache_index, kv_cache, QuantModeC,
                            norm_eps, _cache(QuantModeC, 1),
                            gkva_ub, gkvar_ub, kva_out_ub, kvq_ub, kvblk_ub)

        pl.system.sync_all()

    # M6 + M7：自动 CV 并行流水，一拍 = 一个 (head, m 块)；M 粒度与全局解耦
    mw_wkb = min(MW_WKB, MRows)
    nb_wkb = capacity // mw_wkb
    # 拍数不能是 0：零拍让收尾同步配不上，A5 上挂成 aicore timeout 而 rc 仍 0
    n_ticks_wkb = (N_HEADS * nb_wkb + nc - 1) // nc
    sub_id_q = pl.get_subblock_idx()
    sub_num_q = pl.get_subblock_num()

    for tick in pl.range(0, n_ticks_wkb, 1):
        with pl.section_cube():
            stage_wkb(tick, core_id, nc, mw_wkb, nb_wkb, q_head,
                      wkb, wkb_l1, qh_l1, a_wkb, b_wkb, l0c_wkb, ql_gm)
        with pl.section_vector():
            if QuantModeC:
                stage_q_epi(tick, core_id, nc, sub_id_q,
                            sub_num_q, mw_wkb, nb_wkb, q_head, ql_gm,
                            nope_ub, rope_ub, q_ub, dsq_ub,
                            q_out, descale_q)
            else:
                stage_q_epi_c0(tick, core_id, nc, sub_id_q,
                               sub_num_q, mw_wkb, nb_wkb, q_head,
                               ql_gm, nope_ub, rope_ub, q_out)



def _pick_m(tokens):
    """按 T 选物理行数；T>128 用 128 变体、kernel 内按 m_idx 分块。"""
    if tokens <= 8:
        return 16
    if tokens <= 32:
        return 32
    if tokens <= 64:
        return 64
    return 128


def _validate_scale(name, value, outer, inner):
    """校验逻辑 rank-3 的 MX scale（形状逐维取等号；rank-4 的还原属 kernel 外
    数据搬运，违反 wrapper 边界）。
    """
    groups = (inner + 63) // 64
    if value.dtype != torch.float8_e8m0fnu:
        raise ValueError(f"{name} 必须是 E8M0，实际 {value.dtype}")
    if tuple(value.shape) != (outer, groups, 2):
        raise ValueError(
            f"{name} 必须是逻辑 rank-3 [{outer}, {groups}, 2]，"
            f"实际 {value.ndim} 维 {tuple(value.shape)}")


def _require_device(name, value, device):
    """同设备 + 连续（非连续的权重按物理 NZ 逐 fractal 搬运会读错字节）。"""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} 必须是 torch.Tensor，实际 {type(value)}")
    if value.device.type not in ("npu", "privateuseone"):
        raise RuntimeError(f"{name} 必须在 NPU 上，实际 {value.device}")
    if value.device != device:
        raise ValueError(f"{name} 与 x 不在同一设备：{value.device} vs {device}")
    if not value.is_contiguous():
        raise ValueError(f"{name} 必须连续")


def _check_all_on_device(args):
    """全部入参必须是张量且与 x 同设备。顺序即报错优先级。"""
    (x, wqa, wqb, wkva, wkb, descale_x, descale_wqa, descale_wqb, descale_wkva,
     norm_weight_qa, norm_weight_kva, kv_cache, cache_index, qscale_kv) = args
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x 必须是 torch.Tensor，实际 {type(x)}")
    device = x.device
    named = [
        ("x", x), ("wqa", wqa), ("wqb", wqb), ("wkva", wkva), ("wkb", wkb),
        ("descale_x", descale_x), ("descale_wqa", descale_wqa),
        ("descale_wqb", descale_wqb), ("descale_wkva", descale_wkva),
        ("norm_weight_qa", norm_weight_qa), ("norm_weight_kva", norm_weight_kva),
        ("kv_cache", kv_cache), ("cache_index", cache_index),
    ]
    if qscale_kv is not None:
        named.append(("qscale_kv", qscale_kv))
    for name, value in named:
        _require_device(name, value, device)


def _check_x_weights(x, wqa, wqb, wkva, wkb, descale_x, descale_wqa,
                     descale_wqb, descale_wkva):
    """x、三路 NZ 权重与其 MX scale、wkb；返回 T。"""
    if x.dtype != torch.float8_e4m3fn or x.ndim != 2 or x.shape[1] != DIM:
        raise ValueError(
            f"x 必须是 E4M3 [T, {DIM}]，实际 {tuple(x.shape)} {x.dtype}")
    tokens = x.shape[0]
    if tokens <= 0:
        raise ValueError(f"T 必须为正，实际 {tokens}")

    for name, w, outer, inner in (("wqa", wqa, Q_LORA, DIM),
                                   ("wqb", wqb, QB_WIDTH, Q_LORA),
                                   ("wkva", wkva, FEATURES, DIM)):
        want = (outer // WEIGHT_FRACTAL, inner, WEIGHT_FRACTAL)
        if w.dtype != torch.float8_e4m3fn or tuple(w.shape) != want:
            raise ValueError(
                f"{name} 必须是 E4M3 的 NZ {want}，实际 "
                f"{tuple(w.shape)} {w.dtype}")

    _validate_scale("descale_x", descale_x, tokens, DIM)
    _validate_scale("descale_wqa", descale_wqa, Q_LORA, DIM)
    _validate_scale("descale_wqb", descale_wqb, QB_WIDTH, Q_LORA)
    _validate_scale("descale_wkva", descale_wkva, FEATURES, DIM)

    if (wkb.dtype != torch.bfloat16
            or tuple(wkb.shape) != (N_HEADS, QK_NOPE, KV_LORA)):
        raise ValueError(
            f"wkb 必须是 BF16 [{N_HEADS}, {QK_NOPE}, {KV_LORA}]，实际 "
            f"{tuple(wkb.shape)} {wkb.dtype}")
    return tokens


def _check_norm_index(norm_weight_qa, norm_weight_kva, cache_index,
                      qscale_kv, tokens, quant_mode_c):
    """两路 RMSNorm 权重、cache_index、C1 的 qscale_kv。"""
    for name, g, width in (("norm_weight_qa", norm_weight_qa, Q_LORA),
                           ("norm_weight_kva", norm_weight_kva, KV_LORA)):
        if g.dtype != torch.float32 or tuple(g.shape) != (width,):
            raise ValueError(
                f"{name} 必须是 FP32 [{width}]，实际 {tuple(g.shape)} {g.dtype}")
    if (cache_index.dtype != torch.int64
            or tuple(cache_index.shape) != (tokens,)):
        raise ValueError(
            f"cache_index 必须是 INT64 [{tokens}]，实际 "
            f"{tuple(cache_index.shape)} {cache_index.dtype}")
    if quant_mode_c == 1 and (qscale_kv.dtype != torch.float32
                              or tuple(qscale_kv.shape) != (1,)):
        raise ValueError(
            f"qscale_kv 必须是 FP32 [1]，实际 "
            f"{tuple(qscale_kv.shape)} {qscale_kv.dtype}")


def _check_kv_cache(kv_cache, tokens, quant_mode_c):
    """kv_cache 只收逻辑 rank-4；只判容量不读回 cache_index 的取值。"""
    # cache 两种声明都收（字节相同，物理 PA_NZ 与逻辑 rank-4 零拷 view）
    cache_dtype = torch.float8_e4m3fn if quant_mode_c else torch.bfloat16
    if kv_cache.ndim != 4 or kv_cache.dtype != cache_dtype:
        raise ValueError(
            f"quant_mode_c={quant_mode_c} 要求 kv_cache 为 4 维 "
            f"{cache_dtype}，实际 {kv_cache.ndim} 维 {kv_cache.dtype}")
    if (kv_cache.shape[1] != PAGE or kv_cache.shape[2] != 1
            or kv_cache.shape[3] != FEATURES):
        raise ValueError(
            f"kv_cache 必须是逻辑 [blocks, {PAGE}, 1, {FEATURES}]，"
            f"实际 {tuple(kv_cache.shape)}")
    if kv_cache.shape[0] * PAGE < tokens:
        raise ValueError(
            f"kv_cache 容量 {kv_cache.shape[0]}*{PAGE} 装不下 {tokens} 个 token")


def _validate_inputs(args, quant_mode_c):
    """形状/dtype/设备校验。cache_index 只判容量不读回值（D2H 同步代价）。

    四组校验按原顺序串起来：多个入参同时违例时，先触发的那条决定报错文案，
    用例依赖这个优先级。
    """
    (x, wqa, wqb, wkva, wkb, descale_x, descale_wqa, descale_wqb, descale_wkva,
     norm_weight_qa, norm_weight_kva, kv_cache, cache_index, qscale_kv) = args
    _check_all_on_device(args)
    tokens = _check_x_weights(x, wqa, wqb, wkva, wkb, descale_x,
                              descale_wqa, descale_wqb, descale_wkva)
    _check_norm_index(norm_weight_qa, norm_weight_kva, cache_index,
                      qscale_kv, tokens, quant_mode_c)
    _check_kv_cache(kv_cache, tokens, quant_mode_c)


class _GmBuffers(NamedTuple):
    """kernel 除入参外的 GM 侧产物：跨相位中转量、workspace、输出与发射宽度。"""

    qa_partial: torch.Tensor
    kva_partial: torch.Tensor
    qa_mx: torch.Tensor
    qa_scale_mx: torch.Tensor
    q_head: torch.Tensor
    ws: torch.Tensor
    q_out: torch.Tensor
    descale_q: torch.Tensor
    qscale_slot: torch.Tensor
    block_dim: int


def _alloc_gm(x, capacity, quant_mode_c, qscale_kv):
    """中转量、workspace 与输出的 GM 分配；block_dim 同时定任务划分。"""
    qa_partial = torch.empty(
        (K_ACCUM_WAYS, capacity, Q_LORA), dtype=torch.float32, device=x.device
    )
    kva_partial = torch.empty(
        (K_ACCUM_WAYS, capacity, FEATURES), dtype=torch.float32, device=x.device
    )
    qa_mx = torch.empty((capacity, Q_LORA), dtype=torch.float8_e4m3fn,
                        device=x.device)
    # qa_scale_mx 是 M3 产物、M4 的 A 侧 scale，在 GM 上中转
    qa_scale_mx = torch.empty((capacity, SCALE_PAIRS, 2),
                              dtype=torch.float8_e8m0fnu, device=x.device)
    q_head = torch.empty((capacity, QB_WIDTH), dtype=torch.bfloat16,
                         device=x.device)
    # 核数 host 解析后只当 block_dim 传，任务划分由体内 get_block_num() 读
    # （两处必须同数）；带 tiling_key 的 kernel 裸调会被框架拒、必须显式给
    block_dim = _current_aic_core_num(
        x.device.index if x.device.index is not None
        else torch.npu.current_device())
    # kv_cache 原位：未命中的槽位逐 bit 留原值，要保留原值调用方自己 clone。
    # empty 而非 zeros：读未写会在精度上暴露。
    ws = torch.empty((block_dim * WS_PER_CORE,), dtype=torch.uint8,
                     device=x.device)
    # 输出按 capacity 行开、返回前切回 tokens，省掉 Q epilogue 内循环的运行时守卫
    q_out = torch.empty((capacity, N_HEADS, FEATURES),
                        dtype=torch.float8_e4m3fn if quant_mode_c
                        else torch.bfloat16, device=x.device)
    descale_q = torch.empty((capacity, N_HEADS), dtype=torch.float32,
                            device=x.device)

    # C0 时 kernel 签名保留的 qscale_kv 形参用占位张量（编译期常量跳过读取）
    qscale_slot = (qscale_kv if quant_mode_c
                   else torch.empty((1,), dtype=torch.float32, device=x.device))
    return _GmBuffers(
        qa_partial, kva_partial, qa_mx, qa_scale_mx, q_head,
        ws, q_out, descale_q, qscale_slot, block_dim)


def quant_mla_prolog(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkva,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    qscale_kv=None,
    # norm_eps 是 keyword-only 且无默认值：漏传直接 TypeError，不静默按默认算
    *,
    norm_eps,
    quant_mode_aw=1,
    quant_mode_c=0,
):
    """算子入口：返回有序 dict（q / kv_cache_out，（C1）descale_q）。C1：q 与
    kv_cache 是 E4M3；C0：BF16、qscale_kv 必须传 None。kv_cache 原位改写。
    wrapper 零 torch 数据操作：host 只做校验/元数据/torch.empty/单次发射。
    """
    if quant_mode_aw != 1:
        raise ValueError(f"quant_mode_aw 只支持 1，实际 {quant_mode_aw}")
    if quant_mode_c not in (0, 1):
        raise ValueError("quant_mode_c must be 0 or 1")
    if quant_mode_c == 1 and qscale_kv is None:
        raise ValueError("quant_mode_c=1 requires qscale_kv")
    if quant_mode_c == 0 and qscale_kv is not None:
        raise ValueError("quant_mode_c=0 must not pass qscale_kv")
    _validate_inputs(
        (x, wqa, wqb, wkva, wkb, descale_x, descale_wqa, descale_wqb,
         descale_wkva, norm_weight_qa, norm_weight_kva, kv_cache, cache_index,
         qscale_kv), quant_mode_c)
    tokens = x.shape[0]
    # norm_eps 是运行期标量入参，不进 key、不新增编译产物
    if not (norm_eps > 0.0 and math.isfinite(norm_eps)):
        raise ValueError(f"norm_eps 必须为正且有限，实际 {norm_eps}")
    m_phys = _pick_m(tokens)
    # capacity = kernel 内的物理行数；T>128 时由 kernel 内 m_idx 循环分块
    n_blocks = (tokens + m_phys - 1) // m_phys
    capacity = n_blocks * m_phys
    gm = _alloc_gm(x, capacity, quant_mode_c, qscale_kv)
    quant_mla_prolog_kernel[
        None, gm.block_dim, {"MRows": m_phys, "QuantModeC": quant_mode_c}](
        x, descale_x,
        wqa, descale_wqa, wkva, descale_wkva,
        gm.qa_partial, gm.kva_partial, wqb, descale_wqb,
        norm_weight_qa, gm.qa_mx, gm.qa_scale_mx, gm.q_head,
        norm_weight_kva, gm.qscale_slot, cache_index,
        kv_cache, wkb, gm.ws,
        gm.q_out, gm.descale_q,
        float(norm_eps), kv_cache.shape[0])

    # [:tokens] 是输出的连续前缀：kernel 写满 capacity 行、host 只交有效前缀
    outputs = {"q": gm.q_out[:tokens], "kv_cache_out": kv_cache}
    if quant_mode_c:
        outputs["descale_q"] = gm.descale_q[:tokens]
    return outputs
