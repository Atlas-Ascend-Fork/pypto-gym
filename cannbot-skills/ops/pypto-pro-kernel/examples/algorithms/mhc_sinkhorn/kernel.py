# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。将 64 个 4×4 矩阵的相同位置排列在一个 FP32 寄存器中。

每条 lane 负责一个矩阵，行/列归一化跨寄存器计算，迭代不需要跨 lane 归约。
resident=True 只读写 GM 一次。逐轮对照使用同一 VF 和相同算术，每轮写回再读入。
尾批只传有效矩阵，独立 lane 隔离无效位置。双槽与 auto_mutex 排序 DMA 和 VF。
参见[输出坐标与 lane 分工](../../../references/patterns/vf-outer-reduce.md#output-ownership-before-blocking)。

example-query: {\
  "term_groups":[\
    ["4x4","4×4","小矩阵"],\
    ["row and column","行列"],\
    ["normalization","归一化"]],\
  "exclude":[{"term_groups":[\
    ["8x8 matrix","8x8 matrices","8×8 matrix","8x8 小矩阵","8×8矩阵"]],\
    "reason":"The lane shuffle is specialized for 4x4 matrices."}]}
"""

import math

import torch
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
HC2 = 16
MATS = 64
NAT_COLS = 16
NAT_BYTES = MATS * NAT_COLS * 4
VA_IN0 = 0
VA_IN1 = VA_IN0 + NAT_BYTES
VA_OUT0 = VA_IN1 + NAT_BYTES
VA_OUT1 = VA_OUT0 + NAT_BYTES


@pl.vector_function
def _pack_positions(tin, positions):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    # 四层偶奇拆分。第一层由双寄存器 load 完成，位置寄存器按相邻两列成对存储。
    t0, u0 = vf.load_align(tin, 0, dist=pl.LoadDist.DINTLV_B32)
    t1, u1 = vf.load_align(tin, 128, dist=pl.LoadDist.DINTLV_B32)
    t2, u2 = vf.load_align(tin, 256, dist=pl.LoadDist.DINTLV_B32)
    t3, u3 = vf.load_align(tin, 384, dist=pl.LoadDist.DINTLV_B32)
    t4, u4 = vf.load_align(tin, 512, dist=pl.LoadDist.DINTLV_B32)
    t5, u5 = vf.load_align(tin, 640, dist=pl.LoadDist.DINTLV_B32)
    t6, u6 = vf.load_align(tin, 768, dist=pl.LoadDist.DINTLV_B32)
    t7, u7 = vf.load_align(tin, 896, dist=pl.LoadDist.DINTLV_B32)
    t8, u8 = vf.de_interleave(t0, t1)
    t9, u9 = vf.de_interleave(u0, u1)
    t10, u10 = vf.de_interleave(t2, t3)
    t11, u11 = vf.de_interleave(u2, u3)
    t12, u12 = vf.de_interleave(t4, t5)
    t13, u13 = vf.de_interleave(u4, u5)
    t14, u14 = vf.de_interleave(t6, t7)
    t15, u15 = vf.de_interleave(u6, u7)
    t16, u16 = vf.de_interleave(t8, t10)
    t17, u17 = vf.de_interleave(t9, t11)
    t18, u18 = vf.de_interleave(u8, u10)
    t19, u19 = vf.de_interleave(u9, u11)
    t20, u20 = vf.de_interleave(t12, t14)
    t21, u21 = vf.de_interleave(t13, t15)
    t22, u22 = vf.de_interleave(u12, u14)
    t23, u23 = vf.de_interleave(u13, u15)
    p0, p8 = vf.de_interleave(t16, t20)
    p1, p9 = vf.de_interleave(t17, t21)
    p2, p10 = vf.de_interleave(t18, t22)
    p3, p11 = vf.de_interleave(t19, t23)
    p4, p12 = vf.de_interleave(u16, u20)
    p5, p13 = vf.de_interleave(u17, u21)
    p6, p14 = vf.de_interleave(u18, u22)
    p7, p15 = vf.de_interleave(u19, u23)
    vf.store_align(positions + 0, p0, p1, mask, dist=pl.StoreDist.INTLV_B32)
    vf.store_align(positions + 128, p2, p3, mask, dist=pl.StoreDist.INTLV_B32)
    vf.store_align(positions + 256, p4, p5, mask, dist=pl.StoreDist.INTLV_B32)
    vf.store_align(positions + 384, p6, p7, mask, dist=pl.StoreDist.INTLV_B32)
    vf.store_align(positions + 512, p8, p9, mask, dist=pl.StoreDist.INTLV_B32)
    vf.store_align(positions + 640, p10, p11, mask, dist=pl.StoreDist.INTLV_B32)
    vf.store_align(positions + 768, p12, p13, mask, dist=pl.StoreDist.INTLV_B32)
    vf.store_align(positions + 896, p14, p15, mask, dist=pl.StoreDist.INTLV_B32)


@pl.vector_function
def _unpack_positions(positions, tout):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    p0, p1 = vf.load_align(positions, 0, dist=pl.LoadDist.DINTLV_B32)
    p2, p3 = vf.load_align(positions, 128, dist=pl.LoadDist.DINTLV_B32)
    p4, p5 = vf.load_align(positions, 256, dist=pl.LoadDist.DINTLV_B32)
    p6, p7 = vf.load_align(positions, 384, dist=pl.LoadDist.DINTLV_B32)
    p8, p9 = vf.load_align(positions, 512, dist=pl.LoadDist.DINTLV_B32)
    p10, p11 = vf.load_align(positions, 640, dist=pl.LoadDist.DINTLV_B32)
    p12, p13 = vf.load_align(positions, 768, dist=pl.LoadDist.DINTLV_B32)
    p14, p15 = vf.load_align(positions, 896, dist=pl.LoadDist.DINTLV_B32)
    v0, w0 = vf.interleave(p0, p8)
    v1, w1 = vf.interleave(p1, p9)
    v2, w2 = vf.interleave(p2, p10)
    v3, w3 = vf.interleave(p3, p11)
    v4, w4 = vf.interleave(p4, p12)
    v5, w5 = vf.interleave(p5, p13)
    v6, w6 = vf.interleave(p6, p14)
    v7, w7 = vf.interleave(p7, p15)
    v8, w8 = vf.interleave(v0, v4)
    v9, w9 = vf.interleave(v1, v5)
    v10, w10 = vf.interleave(v2, v6)
    v11, w11 = vf.interleave(v3, v7)
    v12, w12 = vf.interleave(w0, w4)
    v13, w13 = vf.interleave(w1, w5)
    v14, w14 = vf.interleave(w2, w6)
    v15, w15 = vf.interleave(w3, w7)
    v16, w16 = vf.interleave(v8, v10)
    v17, w17 = vf.interleave(v9, v11)
    v18, w18 = vf.interleave(w8, w10)
    v19, w19 = vf.interleave(w9, w11)
    v20, w20 = vf.interleave(v12, v14)
    v21, w21 = vf.interleave(v13, v15)
    v22, w22 = vf.interleave(w12, w14)
    v23, w23 = vf.interleave(w13, w15)
    vf.store_align(tout + 0, v16, v17, mask, dist=pl.StoreDist.INTLV_B32)
    vf.store_align(tout + 128, w16, w17, mask, dist=pl.StoreDist.INTLV_B32)
    vf.store_align(tout + 256, v18, v19, mask, dist=pl.StoreDist.INTLV_B32)
    vf.store_align(tout + 384, w18, w19, mask, dist=pl.StoreDist.INTLV_B32)
    vf.store_align(tout + 512, v20, v21, mask, dist=pl.StoreDist.INTLV_B32)
    vf.store_align(tout + 640, w20, w21, mask, dist=pl.StoreDist.INTLV_B32)
    vf.store_align(tout + 768, v22, v23, mask, dist=pl.StoreDist.INTLV_B32)
    vf.store_align(tout + 896, w22, w23, mask, dist=pl.StoreDist.INTLV_B32)


@pl.vector_function
def _softmax_rows(positions, eps):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for row in pl.range(4):
        base = row * 4 * MATS
        a, b = vf.load_align(positions, base, dist=pl.LoadDist.DINTLV_B32)
        c, d = vf.load_align(positions, base + 2 * MATS, dist=pl.LoadDist.DINTLV_B32)
        maximum = vf.max(a, b, mask)
        maximum = vf.max(maximum, c, mask)
        maximum = vf.max(maximum, d, mask)
        a = vf.exp_sub(a, maximum, mask)
        b = vf.exp_sub(b, maximum, mask)
        c = vf.exp_sub(c, maximum, mask)
        d = vf.exp_sub(d, maximum, mask)
        total = vf.add(a, b, mask)
        total = vf.add(total, c, mask)
        total = vf.add(total, d, mask)
        shift = vf.muls(total, eps, mask)
        a = vf.add(a, shift, mask)
        b = vf.add(b, shift, mask)
        c = vf.add(c, shift, mask)
        d = vf.add(d, shift, mask)
        a = vf.div(a, total, mask)
        b = vf.div(b, total, mask)
        c = vf.div(c, total, mask)
        d = vf.div(d, total, mask)
        vf.store_align(positions + base, a, b, mask, dist=pl.StoreDist.INTLV_B32)
        vf.store_align(positions + (base + 2 * MATS), c, d, mask, dist=pl.StoreDist.INTLV_B32)


@pl.vector_function
def _first_columns(positions, eps):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    # 首轮 softmax 后只归一化列，每次读相邻两列的四对位置寄存器。
    for pair in pl.range(2):
        base = pair * 2 * MATS
        a, b = vf.load_align(positions, base, dist=pl.LoadDist.DINTLV_B32)
        c, d = vf.load_align(positions, base + 4 * MATS, dist=pl.LoadDist.DINTLV_B32)
        e, f = vf.load_align(positions, base + 8 * MATS, dist=pl.LoadDist.DINTLV_B32)
        g, h = vf.load_align(positions, base + 12 * MATS, dist=pl.LoadDist.DINTLV_B32)
        total = vf.add(a, c, mask)
        total = vf.add(total, e, mask)
        total = vf.add(total, g, mask)
        denom = vf.adds(total, eps, mask)
        a = vf.div(a, denom, mask)
        c = vf.div(c, denom, mask)
        e = vf.div(e, denom, mask)
        g = vf.div(g, denom, mask)
        total = vf.add(b, d, mask)
        total = vf.add(total, f, mask)
        total = vf.add(total, h, mask)
        denom = vf.adds(total, eps, mask)
        b = vf.div(b, denom, mask)
        d = vf.div(d, denom, mask)
        f = vf.div(f, denom, mask)
        h = vf.div(h, denom, mask)
        vf.store_align(positions + base, a, b, mask, dist=pl.StoreDist.INTLV_B32)
        vf.store_align(positions + (base + 4 * MATS), c, d, mask, dist=pl.StoreDist.INTLV_B32)
        vf.store_align(positions + (base + 8 * MATS), e, f, mask, dist=pl.StoreDist.INTLV_B32)
        vf.store_align(positions + (base + 12 * MATS), g, h, mask, dist=pl.StoreDist.INTLV_B32)


@pl.vector_function
def _store_four(positions, base, a, b, c, d, mask):
    vf.store_align(positions + base, a, b, mask, dist=pl.StoreDist.INTLV_B32)
    vf.store_align(positions + (base + 2 * MATS), c, d, mask, dist=pl.StoreDist.INTLV_B32)


@pl.vector_function
def _normalize(positions, repeats, eps):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    p00, p01 = vf.load_align(positions, 0, dist=pl.LoadDist.DINTLV_B32)
    p02, p03 = vf.load_align(positions, 128, dist=pl.LoadDist.DINTLV_B32)
    p10, p11 = vf.load_align(positions, 256, dist=pl.LoadDist.DINTLV_B32)
    p12, p13 = vf.load_align(positions, 384, dist=pl.LoadDist.DINTLV_B32)
    p20, p21 = vf.load_align(positions, 512, dist=pl.LoadDist.DINTLV_B32)
    p22, p23 = vf.load_align(positions, 640, dist=pl.LoadDist.DINTLV_B32)
    p30, p31 = vf.load_align(positions, 768, dist=pl.LoadDist.DINTLV_B32)
    p32, p33 = vf.load_align(positions, 896, dist=pl.LoadDist.DINTLV_B32)
    for _ in pl.range(2 * repeats):
        total = vf.add(p00, p01, mask)
        total = vf.add(total, p02, mask)
        total = vf.add(total, p03, mask)
        denom = vf.adds(total, eps, mask)
        p00 = vf.div(p00, denom, mask)
        p01 = vf.div(p01, denom, mask)
        p02 = vf.div(p02, denom, mask)
        p03 = vf.div(p03, denom, mask)
        total = vf.add(p10, p11, mask)
        total = vf.add(total, p12, mask)
        total = vf.add(total, p13, mask)
        denom = vf.adds(total, eps, mask)
        p10 = vf.div(p10, denom, mask)
        p11 = vf.div(p11, denom, mask)
        p12 = vf.div(p12, denom, mask)
        p13 = vf.div(p13, denom, mask)
        total = vf.add(p20, p21, mask)
        total = vf.add(total, p22, mask)
        total = vf.add(total, p23, mask)
        denom = vf.adds(total, eps, mask)
        p20 = vf.div(p20, denom, mask)
        p21 = vf.div(p21, denom, mask)
        p22 = vf.div(p22, denom, mask)
        p23 = vf.div(p23, denom, mask)
        total = vf.add(p30, p31, mask)
        total = vf.add(total, p32, mask)
        total = vf.add(total, p33, mask)
        denom = vf.adds(total, eps, mask)
        p30 = vf.div(p30, denom, mask)
        p31 = vf.div(p31, denom, mask)
        p32 = vf.div(p32, denom, mask)
        p33 = vf.div(p33, denom, mask)
        # 寄存器转置，把列归一化映射到同一组行运算；每两次恢复原始排列。
        p01, p10, p23, p32 = p10, p01, p32, p23
        p02, p20, p13, p31 = p20, p02, p31, p13
        p03, p30, p12, p21 = p30, p03, p21, p12
    _store_four(positions, 0, p00, p01, p02, p03, mask)
    _store_four(positions, 256, p10, p11, p12, p13, mask)
    _store_four(positions, 512, p20, p21, p22, p23, mask)
    _store_four(positions, 768, p30, p31, p32, p33, mask)


@pl.vector_function
def _body(data, positions, initial, repeats, eps):
    _pack_positions(data, positions)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    if initial != 0:
        _softmax_rows(positions, eps)
        vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
        _first_columns(positions, eps)
        vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    if repeats != 0:
        _normalize(positions, repeats, eps)
        vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    _unpack_positions(positions, data)


@pl.jit(auto_mutex=True, name='mhc_sinkhorn_posres_hc4')
def kernel(comb2: pl.Tensor[[pl.DYNAMIC, HC2], FP32],
           initial: pl.DT_INT64,
           repeats: pl.DT_INT64,
           eps: pl.DT_FP32,
           out2: pl.Tensor[[pl.DYNAMIC, HC2], FP32]):
    g_in = pl.make_tile_group(
        type=pl.TileType(shape=[MATS, NAT_COLS], dtype=FP32,
                         target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=[VA_IN0, VA_IN1], mutex_ids=[0, 1])
    g_positions = pl.make_tile_group(
        type=pl.TileType(shape=[HC2, MATS], dtype=FP32, target_memory=pl.MemorySpace.Vec),
        addrs=[VA_OUT0, VA_OUT1], mutex_ids=[2, 3])
    with pl.section_vector():
        b = comb2.shape[0]
        nb = (b + MATS - 1) // MATS
        cid = pl.get_block_idx()
        ncore = pl.get_block_num()
        for blk in pl.range(cid, nb, ncore):
            row0 = blk * MATS
            valid = pl.min(MATS, b - row0)
            data = g_in.next()
            positions = g_positions.next()
            pl.set_validshape(data, [valid, HC2])
            pl.load(data, comb2, [row0, 0])
            _body(data, positions, initial, repeats, eps)
            pl.store(out2, data, [row0, 0])


def run(x, steps=20, eps=1e-6, *, resident=True, workers=8):
    """连续有限 FP32 [B,4,4]，返回同形状新张量。steps 包含首轮。"""
    if x.device.type != "npu" or x.dtype != torch.float32 or not x.is_contiguous():
        raise ValueError("expected contiguous NPU FP32 input")
    if x.ndim != 3 or tuple(x.shape[1:]) != (4, 4) or x.shape[0] < 1:
        raise ValueError("expected [B,4,4] with B >= 1")
    if type(steps) is not int or type(workers) is not int:
        raise TypeError("steps and workers must be Host integers")
    valid_schedule = 1 <= steps <= 40 and 1 <= workers <= 32
    if not valid_schedule or not math.isfinite(eps) or not 1e-8 <= eps <= 1:
        raise ValueError("require 1..40 steps, 1..32 workers and eps in [1e-8,1]")
    source = x.view(-1, HC2)
    result = torch.empty_like(source)
    grid = min(workers, (x.shape[0] + MATS - 1) // MATS)
    if resident or steps == 1:
        kernel[None, grid](source, 1, steps - 1, eps, result)
    else:
        other = torch.empty_like(result)
        kernel[None, grid](source, 1, 0, eps, result)
        for _ in range(1, steps):
            kernel[None, grid](result, 0, 1, eps, other)
            result, other = other, result
    return result.view_as(x)
