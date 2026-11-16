# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PyPTO-Pro affine-summary CP KDA backward.

Stage1 produces E_grad/R and clears this head bucket's parameter gradients.
Stage2 merges the suffix, runs state/chunk backward, and fuses activation gradients.
H cache and the shared local dH operand are BF16; rolling dH and CP summaries stay FP32.
Each public wrapper launches one kernel. Production legacy ABI is not used.
"""

__all__ = ['flash_kda_bwd_stage1_wrapper', 'flash_kda_bwd_stage2_wrapper']
import os
import torch
import torch_npu
import pypto
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf
from pypto_pro.runtime.tilingkey import TilingKeyField
P = pl.PipeType
CHUNK_C = 64
HEAD_D = 128
S_MAX = 131072
B_MAX = 64
N_MAX = 96
_STAGE2_WORKSPACE_BYTES_PER_CORE = 6 * HEAD_D * HEAD_D * 4


@pl.vector_function
def s1_zero_vslice(out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p = vf.update_mask(32, dtype=pl.DT_FP32)
    z = vf.full(0.0, p, dtype=pl.DT_FP32)
    for r in pl.range(0, 64, 1):
        vf.store_align(out, z, p, r * 128)


@pl.vector_function
def s1_transposed(src, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pu_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 64, 1):
        for _ in pl.range(0, 1, 1):
            p_1 = vf.update_mask(32, dtype=pl.DT_FP32)
            ix = vf.arange(0, dtype=pl.DT_UINT32)
            ix = vf.shift_left(ix, 7, pu_1)
            ix = vf.adds(ix, r, pu_1)
            x = vf.gather(src, ix, p_1)
            vf.store_align(dst, x, p_1, r * 128)


@pl.vector_function
def s1_transposed_1(src, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pu_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 32, 1):
        for _ in pl.range(0, 1, 1):
            p_1 = vf.update_mask(64, dtype=pl.DT_FP32)
            ix = vf.arange(0, dtype=pl.DT_UINT32)
            ix = vf.shift_left(ix, 7, pu_1)
            ix = vf.adds(ix, r, pu_1)
            x = vf.gather(src, ix, p_1)
            vf.store_align(dst, x, p_1, r * 128)


@pl.vector_function
def s1_zero_vslice_1(out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p = vf.update_mask(32, dtype=pl.DT_FP32)
    z = vf.full(0.0, p, dtype=pl.DT_FP32)
    for r in pl.range(0, 32, 1):
        vf.store_align(out, z, p, r * 128)


@pl.vector_function
def s1_widen(src, dst, rows):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p16_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    for r in pl.range(0, pl.max(rows, 0), 1):
        for _ in pl.range(0, 1, 1):
            p_1 = vf.update_mask(32, dtype=pl.DT_FP32)
            mul_1 = r * 128
            x = vf.load_align(src, mul_1, dist=pl.LoadDist.US_B16)
            y = vf.astype(x, p16_1, dtype=pl.DT_FP32, round_mode=pl.VFRoundMode.CAST_RINT, saturate=pl.SaturateMode.OFF)
            vf.store_align(dst, y, p_1, mul_1)


@pl.vector_function
def s1_narrow(src, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    p32_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p64_1 = vf.create_mask(pattern=pl.MaskPattern.H, dtype=pl.DT_FP16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, p64_1)
    for r in pl.range(0, 32, 1):
        for _ in pl.range(0, 1, 1):
            p_1 = vf.update_mask(32, dtype=pl.DT_FP16)
            mul_1 = r * 128
            x = vf.load_align(src, mul_1)
            y = vf.astype(x, p32_1, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT, saturate=pl.SaturateMode.OFF)
            z = vf.gather(y, ix)
            vf.store_align(dst, z, p_1, mul_1)


@pl.vector_function
def s1_narrow_2(src, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    p32_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p64_1 = vf.create_mask(pattern=pl.MaskPattern.H, dtype=pl.DT_FP16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, p64_1)
    for r in pl.range(0, 64, 1):
        for _ in pl.range(0, 1, 1):
            p_1 = vf.update_mask(32, dtype=pl.DT_FP16)
            mul_1 = r * 128
            x = vf.load_align(src, mul_1)
            y = vf.astype(x, p32_1, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT, saturate=pl.SaturateMode.OFF)
            z = vf.gather(y, ix)
            vf.store_align(dst, z, p_1, mul_1)


@pl.vector_function
def s1_copy_keep32(src, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p = vf.update_mask(32, dtype=pl.DT_FP32)
    for r in pl.range(0, 32, 1):
        mul = r * 64
        x = vf.load_align(src, mul)
        vf.store_align(dst, x, p, mul)


@pl.vector_function
def s1_recurrent_add_product64(dh, g_last, product, out, scale, first_col):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p = vf.update_mask(32, dtype=pl.DT_FP32)
    for r in pl.range(0, 64, 1):
        mul = r * 128
        x = vf.load_align(dh, mul)
        g = vf.load_align(g_last, first_col + r, dist=pl.LoadDist.BRC_B32)
        eg = vf.exp(g, p)
        m = vf.load_align(product, r * 64)
        v = vf.mul(x, eg, p)
        w = vf.muls(m, scale, p)
        y = vf.add(v, w, p)
        vf.store_align(out, y, p, mul)


@pl.vector_function
def s1_narrow_keep32(src, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    p32_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p64_1 = vf.create_mask(pattern=pl.MaskPattern.H, dtype=pl.DT_FP16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, p64_1)
    for r in pl.range(0, 32, 1):
        for _ in pl.range(0, 1, 1):
            p_1 = vf.update_mask(32, dtype=pl.DT_FP16)
            x = vf.load_align(src, r * 64)
            y = vf.astype(x, p32_1, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT, saturate=pl.SaturateMode.OFF)
            z = vf.gather(y, ix)
            vf.store_align(dst, z, p_1, r * 128)


@pl.vector_function
def s1_binary_rhs64(a, b, out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 64, 1):
        for _ in pl.range(0, 1, 1):
            p_1 = vf.update_mask(32, dtype=pl.DT_FP32)
            mul_1 = r * 128
            x = vf.load_align(a, mul_1)
            y = vf.load_align(b, r * 64)
            z = vf.add(x, y, p_1)
            vf.store_align(out, z, p_1, mul_1)


@pl.vector_function
def s1_qg_bf16(raw, g, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    ph = vf.create_mask(pattern=pl.MaskPattern.H, dtype=pl.DT_UINT16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, ph)
    pm = vf.update_mask(64, dtype=pl.DT_BF16)
    for r in pl.range(0, 32):
        for j in pl.range(0, 2):
            off = r * 128 + j * 64
            xb = vf.load_align(raw, off, dist=pl.LoadDist.US_B16)
            x = vf.astype(xb, p16, dtype=pl.DT_FP32)
            gv = vf.load_align(g, off)
            e = vf.exp(gv, p)
            y = vf.mul(x, e, p)
            yb = vf.astype(y, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            packed = vf.gather(yb, ix)
            vf.store_align(dst, packed, pm, off)


@pl.vector_function
def s1_kg_bf16(raw, g, last, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    ph = vf.create_mask(pattern=pl.MaskPattern.H, dtype=pl.DT_UINT16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, ph)
    pm = vf.update_mask(64, dtype=pl.DT_BF16)
    for r in pl.range(0, 32):
        for j in pl.range(0, 2):
            off = r * 128 + j * 64
            xb = vf.load_align(raw, off, dist=pl.LoadDist.US_B16)
            x = vf.astype(xb, p16, dtype=pl.DT_FP32)
            gv = vf.load_align(g, off)
            gl = vf.load_align(last, j * 64)
            gv = vf.sub(gl, gv, p)
            e = vf.exp(gv, p)
            y = vf.mul(x, e, p)
            yb = vf.astype(y, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            packed = vf.gather(yb, ix)
            vf.store_align(dst, packed, pm, off)


@pl.vector_function
def s2_beta_mul32(a, beta, out, first_row):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 32):
        b = vf.load_align(beta, first_row + r, dist=pl.LoadDist.BRC_B32)
        for j in pl.range(0, 2):
            off = r * 128 + j * 64
            x = vf.load_align(a, off)
            y = vf.mul(x, b, p)
            vf.store_align(out, y, p, off)


@pl.vector_function
def s2_beta_mul16(a, beta, out, first_row, local_rows):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, local_rows):
        b = vf.load_align(beta, first_row + r, dist=pl.LoadDist.BRC_B32)
        for j in pl.range(0, 2):
            off = r * 128 + j * 64
            x = vf.load_align(a, off)
            y = vf.mul(x, b, p)
            vf.store_align(out, y, p, off)


@pl.vector_function
def s2_ternary_beta(a, b, beta, out, first_row):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 32):
        z = vf.load_align(beta, first_row + r, dist=pl.LoadDist.BRC_B32)
        for j in pl.range(0, 2):
            off = r * 128 + j * 64
            x = vf.load_align(a, off)
            y = vf.load_align(b, off)
            v = vf.mul(x, y, p)
            result = vf.mul(v, z, p)
            vf.store_align(out, result, p, off)


@pl.vector_function
def s2_unary(a, out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 32, 1):
        for j in pl.range(0, 2, 1):
            mul = j * 64
            p_1 = vf.update_mask(64, dtype=pl.DT_FP32)
            add = r * 128 + mul
            x = vf.load_align(a, add)
            y = vf.muls(x, 1.0, p_1)
            z = vf.exp(y, p_1)
            vf.store_align(out, z, p_1, add)


@pl.vector_function
def s2_exponential(a, out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for _ in pl.range(0, 1, 1):
        for _ in pl.range(0, 1, 1):
            p_1 = vf.update_mask(64, dtype=pl.DT_FP32)
            x = vf.load_align(a, 0)
            z = vf.exp(x, p_1)
            vf.store_align(out, z, p_1)


@pl.vector_function
def s2_widen(src, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p16_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    for r in pl.range(0, 32, 1):
        for j in pl.range(0, 2, 1):
            mul = j * 64
            p_1 = vf.update_mask(64, dtype=pl.DT_FP32)
            add = r * 128 + mul
            x = vf.load_align(src, add, dist=pl.LoadDist.US_B16)
            y = vf.astype(x, p16_1, dtype=pl.DT_FP32, round_mode=pl.VFRoundMode.CAST_RINT, saturate=pl.SaturateMode.OFF)
            vf.store_align(dst, y, p_1, add)


@pl.vector_function
def s2_binary(a, b, out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 32, 1):
        for j in pl.range(0, 2, 1):
            mul = j * 64
            p_1 = vf.update_mask(64, dtype=pl.DT_FP32)
            add = r * 128 + mul
            x = vf.load_align(a, add)
            y = vf.load_align(b, add)
            z = vf.mul(x, y, p_1)
            vf.store_align(out, z, p_1, add)


@pl.vector_function
def s2_zero32(out):
    p_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    z = vf.full(0.0, p_1, dtype=pl.DT_FP32)
    for i in pl.range(0, 256, 1):
        vf.store_align(out, z, p_1, i * 64)


@pl.vector_function
def s2_transposed(src, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pu_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 64, 1):
        for j in pl.range(0, 2, 1):
            mul = j * 64
            p_1 = vf.update_mask(64, dtype=pl.DT_FP32)
            ix = vf.arange(mul, dtype=pl.DT_UINT32)
            ix = vf.shift_left(ix, 7, pu_1)
            ix = vf.adds(ix, r, pu_1)
            x = vf.gather(src, ix, p_1)
            vf.store_align(dst, x, p_1, r * 128 + mul)


@pl.vector_function
def s2_widen_1(src, dst, rows):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p16_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    for r in pl.range(0, pl.max(rows, 0), 1):
        for j in pl.range(0, 2, 1):
            mul = j * 64
            p_1 = vf.update_mask(64, dtype=pl.DT_FP32)
            add = r * 128 + mul
            x = vf.load_align(src, add, dist=pl.LoadDist.US_B16)
            y = vf.astype(x, p16_1, dtype=pl.DT_FP32, round_mode=pl.VFRoundMode.CAST_RINT, saturate=pl.SaturateMode.OFF)
            vf.store_align(dst, y, p_1, add)


@pl.vector_function
def s2_transpose_h_bf16(src, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pb = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    for r in pl.range(0, 64):
        for j in pl.range(0, 2):
            ix = vf.arange(j * 64, dtype=pl.DT_UINT32)
            ix = vf.shift_left(ix, 7, p)
            ix = vf.adds(ix, r, p)
            # UINT32 indices return BF16 in the low half of each 32-bit lane.
            x = vf.gather(src, ix, pb)
            y = vf.astype(x, pb, dtype=pl.DT_FP32)
            vf.store_align(dst, y, p, r * 128 + j * 64)


@pl.vector_function
def s2_transpose_dh_bf16(src, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pb = vf.update_mask(64, dtype=pl.DT_BF16)
    ix16 = vf.arange(0, dtype=pl.DT_UINT16)
    ix16 = vf.shift_left(ix16, 1, pb)
    for r in pl.range(0, 64):
        for j in pl.range(0, 2):
            ix = vf.arange(j * 64, dtype=pl.DT_UINT32)
            ix = vf.shift_left(ix, 7, p)
            ix = vf.adds(ix, r, p)
            x = vf.gather(src, ix, p)
            h = vf.astype(x, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            packed = vf.gather(h, ix16)
            vf.store_align(dst, packed, pb, r * 128 + j * 64)


@pl.vector_function
def s2_widen_state_half(src, dst):
    p = vf.update_mask(64, dtype=pl.DT_FP32)
    pb = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    for r in pl.range(0, 128):
        x = vf.load_align(src, r * 128, dist=pl.LoadDist.US_B16)
        y = vf.astype(x, pb, dtype=pl.DT_FP32)
        vf.store_align(dst, y, p, r * 128)


@pl.vector_function
def s2_state_dot_local_bf16(states, dh_bf16, out):
    p1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p1 = vf.update_mask(1, dtype=pl.DT_FP32)
    for r in pl.range(0, 64, 1):
        mul = r * 128
        a0 = vf.load_align(states, 8192 + mul)
        add = mul + 64
        a1 = vf.load_align(states, 8192 + add)
        pb = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
        b0h = vf.load_align(dh_bf16, mul, dist=pl.LoadDist.US_B16)
        b1h = vf.load_align(dh_bf16, add, dist=pl.LoadDist.US_B16)
        b0 = vf.astype(b0h, pb, dtype=pl.DT_FP32)
        b1 = vf.astype(b1h, pb, dtype=pl.DT_FP32)
        p0 = vf.mul(a0, b0, p_1)
        p2 = vf.mul(a1, b1, p_1)
        sumv = vf.add(p0, p2, p_1)
        y = vf.reduce_sum(sumv, p_1)
        vf.store_align(out, y, p1, r * 8)


@pl.vector_function
def s2_state_scale(a, b, out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for _ in pl.range(0, 1, 1):
        for _ in pl.range(0, 1, 1):
            p_1 = vf.update_mask(64, dtype=pl.DT_FP32)
            ix = vf.arange(0, dtype=pl.DT_UINT32)
            ix = vf.shift_left(ix, 3, p_1)
            x = vf.gather(a, ix, p_1)
            y = vf.load_align(b, 0)
            z = vf.mul(x, y, p_1)
            vf.store_align(out, z, p_1)


@pl.vector_function
def s2_narrow_1(src, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    p32_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p64_1 = vf.create_mask(pattern=pl.MaskPattern.H, dtype=pl.DT_FP16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, p64_1)
    for r in pl.range(0, 64, 1):
        for j in pl.range(0, 2, 1):
            mul = j * 64
            p_1 = vf.update_mask(64, dtype=pl.DT_FP16)
            add = r * 128 + mul
            x = vf.load_align(src, add)
            y = vf.astype(x, p32_1, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT, saturate=pl.SaturateMode.OFF)
            z = vf.gather(y, ix)
            vf.store_align(dst, z, p_1, add)


@pl.vector_function
def s2_kg_func(k, g, g_last, out):
    p_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 32, 1):
        for j in pl.range(0, 2, 1):
            mul_1 = j * 64
            add = r * 128 + mul_1
            kv = vf.load_align(k, add)
            gv = vf.load_align(g, add)
            gc = vf.load_align(g_last, mul_1)
            diff = vf.sub(gc, gv, p_1)
            ev = vf.exp(diff, p_1)
            y = vf.mul(kv, ev, p_1)
            vf.store_align(out, y, p_1, add)


@pl.vector_function
def s2_unary_2(a, out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 32, 1):
        for j in pl.range(0, 2, 1):
            mul = j * 64
            p_1 = vf.update_mask(64, dtype=pl.DT_FP32)
            add = r * 128 + mul
            x = vf.load_align(a, add)
            y = vf.muls(x, -1.0, p_1)
            vf.store_align(out, y, p_1, add)


@pl.vector_function
def s2_column_neg_sum(a, out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    acc = vf.full(0.0, p, dtype=pl.DT_FP32)
    for r in pl.range(0, 64):
        x = vf.load_align(a, r * 128)
        acc = vf.sub(acc, x, p)
    vf.store_align(out, acc, p)


@pl.vector_function
def s2_binary_2(a, b, out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for _ in pl.range(0, 1, 1):
        for _ in pl.range(0, 1, 1):
            p_1 = vf.update_mask(64, dtype=pl.DT_FP32)
            x = vf.load_align(a, 0)
            y = vf.load_align(b, 0)
            z = vf.add(x, y, p_1)
            vf.store_align(out, z, p_1)


@pl.vector_function
def s2_narrow_2(src, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    p32_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p64_1 = vf.create_mask(pattern=pl.MaskPattern.H, dtype=pl.DT_FP16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, p64_1)
    for r in pl.range(0, 32, 1):
        for j in pl.range(0, 2, 1):
            mul = j * 64
            p_1 = vf.update_mask(64, dtype=pl.DT_FP16)
            add = r * 128 + mul
            x = vf.load_align(src, add)
            y = vf.astype(x, p32_1, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT, saturate=pl.SaturateMode.OFF)
            z = vf.gather(y, ix)
            vf.store_align(dst, z, p_1, add)


@pl.vector_function
def s2_triangular(a, out, start, scale):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 32, 1):
        p_count = start + r + 1
        p_1 = vf.update_mask(p_count, dtype=pl.DT_FP32)
        mul = r * 128
        x = vf.load_align(a, mul)
        y = vf.muls(x, scale, p_1)
        vf.store_align(out, y, p_1, mul)


@pl.vector_function
def s2_narrow_3(src, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    p32_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p64_1 = vf.create_mask(pattern=pl.MaskPattern.H, dtype=pl.DT_FP16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, p64_1)
    for r in pl.range(0, 32, 1):
        for _ in pl.range(0, 1, 1):
            p_1 = vf.update_mask(64, dtype=pl.DT_FP16)
            mul_1 = r * 128
            x = vf.load_align(src, mul_1)
            y = vf.astype(x, p32_1, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT, saturate=pl.SaturateMode.OFF)
            z = vf.gather(y, ix)
            vf.store_align(dst, z, p_1, mul_1)


@pl.vector_function
def s2_ternary_1(a, b, c, out, scale):
    p_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 32, 1):
        for j in pl.range(0, 2, 1):
            add = r * 128 + j * 64
            x = vf.load_align(a, add)
            y = vf.load_align(b, add)
            z = vf.load_align(c, add)
            v = vf.muls(y, scale, p_1)
            v2 = vf.add(x, v, p_1)
            outv = vf.mul(v2, z, p_1)
            vf.store_align(out, outv, p_1, add)


@pl.vector_function
def s2_narrow_4(src, dst, rows):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    p32_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p64_1 = vf.create_mask(pattern=pl.MaskPattern.H, dtype=pl.DT_FP16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, p64_1)
    for r in pl.range(0, pl.max(rows, 0), 1):
        for j in pl.range(0, 2, 1):
            mul = j * 64
            p_1 = vf.update_mask(64, dtype=pl.DT_FP16)
            add = r * 128 + mul
            x = vf.load_align(src, add)
            y = vf.astype(x, p32_1, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT, saturate=pl.SaturateMode.OFF)
            z = vf.gather(y, ix)
            vf.store_align(dst, z, p_1, add)


@pl.vector_function
def s2_binary_3(a, b, out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 32, 1):
        for j in pl.range(0, 2, 1):
            mul = j * 64
            p_1 = vf.update_mask(64, dtype=pl.DT_FP32)
            add = r * 128 + mul
            x = vf.load_align(a, add)
            y = vf.load_align(b, add)
            z = vf.sub(x, y, p_1)
            vf.store_align(out, z, p_1, add)


@pl.vector_function
def s2_binary_4(a, b, out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 32, 1):
        for j in pl.range(0, 2, 1):
            mul = j * 64
            p_1 = vf.update_mask(64, dtype=pl.DT_FP32)
            add = r * 128 + mul
            x = vf.load_align(a, add)
            y = vf.load_align(b, add)
            z = vf.add(x, y, p_1)
            vf.store_align(out, z, p_1, add)


@pl.vector_function
def s2_row_product_sum_1(a, b, out):
    p1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p1 = vf.update_mask(1, dtype=pl.DT_FP32)
    for r in pl.range(0, 32, 1):
        mul = r * 128
        a0 = vf.load_align(a, mul)
        add = mul + 64
        a1 = vf.load_align(a, add)
        b0 = vf.load_align(b, mul)
        b1 = vf.load_align(b, add)
        p0 = vf.mul(a0, b0, p_1)
        p2 = vf.mul(a1, b1, p_1)
        sumv = vf.add(p0, p2, p_1)
        y = vf.reduce_sum(sumv, p_1)
        vf.store_align(out, y, p1, mul)


@pl.vector_function
def s2_unary_3(a, out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 32, 1):
        for j in pl.range(0, 2, 1):
            mul = j * 64
            p_1 = vf.update_mask(64, dtype=pl.DT_FP32)
            add = r * 128 + mul
            x = vf.load_align(a, add)
            y = vf.muls(x, -1.0, p_1)
            vf.store_align(out, y, p_1, add)


@pl.vector_function
def s2_row_triple_sum(a, b, c, out):
    p1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p1 = vf.update_mask(1, dtype=pl.DT_FP32)
    for r in pl.range(0, 32, 1):
        mul = r * 128
        a0 = vf.load_align(a, mul)
        add = mul + 64
        a1 = vf.load_align(a, add)
        b0 = vf.load_align(b, mul)
        b1 = vf.load_align(b, add)
        c0 = vf.load_align(c, mul)
        c1 = vf.load_align(c, add)
        p0 = vf.mul(a0, b0, p_1)
        p2 = vf.mul(a1, b1, p_1)
        p3 = vf.mul(p0, c0, p_1)
        p4 = vf.mul(p2, c1, p_1)
        sumv = vf.add(p3, p4, p_1)
        y = vf.reduce_sum(sumv, p_1)
        vf.store_align(out, y, p1, mul)


@pl.vector_function
def s2_binary_6(a, b, out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 32, 1):
        for j in pl.range(0, 2, 1):
            mul = j * 64
            p_1 = vf.update_mask(64, dtype=pl.DT_FP32)
            add = r * 128 + mul
            x = vf.load_align(a, add)
            y = vf.load_align(b, add)
            z = vf.add(x, y, p_1)
            vf.store_align(out, z, p_1, add)


@pl.vector_function
def s2_triangular_1(a, out, start):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 32, 1):
        p_count = start + r + 1 - 1
        p_1 = vf.update_mask(p_count, dtype=pl.DT_FP32)
        mul = r * 128
        x = vf.load_align(a, mul)
        y = vf.muls(x, 1.0, p_1)
        vf.store_align(out, y, p_1, mul)


@pl.vector_function
def s2_triangular_2(a, out, start):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 32, 1):
        p_count = start + r + 1 - 1
        p_1 = vf.update_mask(p_count, dtype=pl.DT_FP32)
        mul = r * 128
        x = vf.load_align(a, mul)
        y = vf.muls(x, -1.0, p_1)
        vf.store_align(out, y, p_1, mul)


@pl.vector_function
def s2_narrow_6(src, dst, rows):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    p32_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p64_1 = vf.create_mask(pattern=pl.MaskPattern.H, dtype=pl.DT_FP16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, p64_1)
    for r in pl.range(0, pl.max(rows, 0), 1):
        for _ in pl.range(0, 1, 1):
            p_1 = vf.update_mask(1, dtype=pl.DT_FP16)
            mul_1 = r * 128
            x = vf.load_align(src, mul_1)
            y = vf.astype(x, p32_1, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT, saturate=pl.SaturateMode.OFF)
            z = vf.gather(y, ix)
            vf.store_align(dst, z, p_1, mul_1)


@pl.vector_function
def s2_reverse_scan(a, out, live):
    p_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    acc = vf.full(0.0, p_1, dtype=pl.DT_FP32)
    for rr in pl.range(0, pl.max(live, 0), 1):
        r = live - 1 - rr
        mul = r * 128
        x = vf.load_align(a, mul)
        acc = vf.add(acc, x, p_1)
        vf.store_align(out, acc, p_1, mul)


@pl.vector_function
def s2_narrow_7(src, dst, rows):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    p32_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p64_1 = vf.create_mask(pattern=pl.MaskPattern.H, dtype=pl.DT_FP16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, p64_1)
    for r in pl.range(0, pl.max(rows, 0), 1):
        for _ in pl.range(0, 1, 1):
            p_1 = vf.update_mask(64, dtype=pl.DT_FP16)
            mul_1 = r * 128
            x = vf.load_align(src, mul_1)
            y = vf.astype(x, p32_1, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT, saturate=pl.SaturateMode.OFF)
            z = vf.gather(y, ix)
            vf.store_align(dst, z, p_1, mul_1)


@pl.vector_function
def s2_single_token_gate(h0, dh0, out):
    """Stable one-token dg[k] = sum_v h0[v,k] * dh0[v,k]."""
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    acc = vf.full(0.0, p, dtype=pl.DT_FP32)
    for row in pl.range(0, 128):
        x = vf.load_align(h0, row * 128)
        y = vf.load_align(dh0, row * 128)
        pb = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
        yh = vf.astype(y, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
        y = vf.astype(yh, pb, dtype=pl.DT_FP32)
        product = vf.mul(x, y, p)
        acc = vf.add(acc, product, p)
    vf.store_align(out, acc, p, 0)


@pl.vector_function
def s2_safe_strict(a, start, local_rows):
    p = vf.update_mask(local_rows * 2, dtype=pl.DT_FP32)
    zero = vf.full(0.0, p, dtype=pl.DT_FP32)
    for r in pl.range(0, local_rows):
        lower = vf.update_mask(start + r, dtype=pl.DT_FP32)
        x = vf.load_align(a, r * 128)
        x = vf.select(x, zero, lower)
        vf.store_align(a, x, p, r * 128)


@pl.vector_function
def s2_safe_diagonal_add(out, da, raw, start, local_rows):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    for r in pl.range(0, local_rows):
        diag = vf.load_align(da, r * 128 + start + r, dist=pl.LoadDist.BRC_B32)
        for j in pl.range(0, 2):
            off = r * 128 + j * 64
            x16 = vf.load_align(raw, off, dist=pl.LoadDist.US_B16)
            x = vf.astype(x16, p16, dtype=pl.DT_FP32)
            delta = vf.mul(x, diag, p)
            previous = vf.load_align(out, off)
            updated = vf.add(previous, delta, p)
            vf.store_align(out, updated, p, off)


@pl.vector_function
def s2_safe_pack_hi(src, dst, columns, local_rows):
    """Emit the high packet; reuse dead source scratch for the low residual."""
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    ph = vf.create_mask(pattern=pl.MaskPattern.H, dtype=pl.DT_FP16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, ph)
    col = vf.arange(0, dtype=pl.DT_UINT16)
    col_block = vf.shift_right(col, 4, p16)
    col_base = vf.shift_left(col_block, 4, p16)
    within = vf.sub(col, col_base, p16)
    nz_blocks = vf.shift_left(col_block, 8, p16)
    nz_index = vf.add(nz_blocks, within, p16)
    for r in pl.range(0, local_rows):
        for j in pl.range(0, (columns + 63) // 64):
            off = r * 128 + j * 64
            pm = vf.update_mask(pl.min(64, columns - j * 64), dtype=pl.DT_FP16)
            x = vf.load_align(src, off)
            hi = vf.astype(x, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            hi32 = vf.astype(hi, p16, dtype=pl.DT_FP32)
            residual = vf.sub(x, hi32, p)
            vf.store_align(src, residual, p, off)
            packed = vf.gather(hi, ix)
            destination = vf.adds(nz_index, r * 16 + j * 1024, p16)
            vf.scatter(dst, packed, destination, pm)


@pl.vector_function
def s2_safe_pack_lo(src, dst, columns, local_rows):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    ph = vf.create_mask(pattern=pl.MaskPattern.H, dtype=pl.DT_FP16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, ph)
    col = vf.arange(0, dtype=pl.DT_UINT16)
    col_block = vf.shift_right(col, 4, p16)
    col_base = vf.shift_left(col_block, 4, p16)
    within = vf.sub(col, col_base, p16)
    nz_blocks = vf.shift_left(col_block, 8, p16)
    nz_index = vf.add(nz_blocks, within, p16)
    for r in pl.range(0, local_rows):
        for j in pl.range(0, (columns + 63) // 64):
            off = r * 128 + j * 64
            pm = vf.update_mask(pl.min(64, columns - j * 64), dtype=pl.DT_FP16)
            x = vf.load_align(src, off)
            part16 = vf.astype(x, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            packed = vf.gather(part16, ix)
            destination = vf.adds(nz_index, r * 16 + j * 1024, p16)
            vf.scatter(dst, packed, destination, pm)


@pl.vector_function
def s2_safe_exp(gate, ref, dst, sign, local_rows):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, local_rows):
        for j in pl.range(0, 2):
            off = r * 128 + j * 64
            g = vf.load_align(gate, off)
            pivot = vf.load_align(ref, j * 64)
            delta = vf.sub(g, pivot, p)
            delta = vf.muls(delta, sign, p)
            e = vf.exp(delta, p)
            vf.store_align(dst, e, p, off)


@pl.vector_function
def s2_safe_scale_bf16(raw, exponent, dst, local_rows):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    for r in pl.range(0, local_rows):
        for j in pl.range(0, 2):
            off = r * 128 + j * 64
            x16 = vf.load_align(raw, off, dist=pl.LoadDist.US_B16)
            x = vf.astype(x16, p16, dtype=pl.DT_FP32)
            e = vf.load_align(exponent, off)
            y = vf.mul(x, e, p)
            vf.store_align(dst, y, p, off)


@pl.vector_function
def s2_safe_mul(a, b, dst, local_rows):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, local_rows):
        for j in pl.range(0, 2):
            off = r * 128 + j * 64
            x = vf.load_align(a, off)
            y = vf.load_align(b, off)
            z = vf.mul(x, y, p)
            vf.store_align(dst, z, p, off)


@pl.vector_function
def s2_safe_add(a, b, dst, local_rows):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, local_rows):
        for j in pl.range(0, 2):
            off = r * 128 + j * 64
            x = vf.load_align(a, off)
            y = vf.load_align(b, off)
            z = vf.add(x, y, p)
            vf.store_align(dst, z, p, off)


@pl.vector_function
def s2_safe_gate_add(dg, correction, raw, sign, local_rows):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    for r in pl.range(0, local_rows):
        for j in pl.range(0, 2):
            off = r * 128 + j * 64
            a = vf.load_align(dg, off)
            b = vf.load_align(correction, off)
            c16 = vf.load_align(raw, off, dist=pl.LoadDist.US_B16)
            c = vf.astype(c16, p16, dtype=pl.DT_FP32)
            product = vf.mul(b, c, p)
            product = vf.muls(product, sign, p)
            z = vf.add(a, product, p)
            vf.store_align(dg, z, p, off)


@pl.vector_function
def s2_safe_beta_add(db, correction, raw, local_rows):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    p1 = vf.update_mask(1, dtype=pl.DT_FP32)
    for r in pl.range(0, local_rows):
        acc = vf.full(0.0, p, dtype=pl.DT_FP32)
        for j in pl.range(0, 2):
            off = r * 128 + j * 64
            a = vf.load_align(correction, off)
            b16 = vf.load_align(raw, off, dist=pl.LoadDist.US_B16)
            b = vf.astype(b16, p16, dtype=pl.DT_FP32)
            product = vf.mul(a, b, p)
            acc = vf.add(acc, product, p)
        total = vf.reduce_sum(acc, p)
        previous = vf.load_align(db, r * 128)
        total = vf.add(previous, total, p1)
        vf.store_align(db, total, p1, r * 128)


@pl.vector_function
def s2_copy32(src, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for i in pl.range(0, 64):
        x = vf.load_align(src, i * 64)
        vf.store_align(dst, x, p, i * 64)


@pl.vector_function
def s2_zero_rows32(out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    z = vf.full(0.0, p, dtype=pl.DT_FP32)
    for i in pl.range(0, 64):
        vf.store_align(out, z, p, i * 64)


@pl.vector_function
def s2_accumulate_dg(owner, b, c, source_row, dest_row):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 32):
        for j in pl.range(0, 2):
            offset = r * 128 + j * 64
            x = vf.load_align(owner, source_row * 128 + offset)
            y = vf.load_align(b, offset)
            z = vf.load_align(c, offset)
            product = vf.mul(y, z, p)
            value = vf.add(x, product, p)
            vf.store_align(owner, value, p, dest_row * 128 + offset)


@pl.vector_function
def s2_add_base_beta(owner, out):
    p = vf.update_mask(1, dtype=pl.DT_FP32)
    for r in pl.range(0, 32):
        x = vf.load_align(owner, (32 + r) * 128)
        y = vf.load_align(owner, r * 128)
        value = vf.add(x, y, p)
        vf.store_align(out, value, p, r * 128)


@pl.vector_function
def s2_zero_bf16_rows(out, begin, end, cols):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pb = vf.update_mask(cols, dtype=pl.DT_BF16)
    z = vf.full(0.0, p, dtype=pl.DT_FP32)
    zb = vf.astype(z, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
    packed, unused = vf.de_interleave(zb, zb)
    for r in pl.range(begin, end):
        vf.store_align(out, packed, pb, r * 128)


@pl.vector_function
def s2_beta_backward(grad, beta, out, rows, token_start):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pb = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    p1 = vf.update_mask(1, dtype=pl.DT_BF16)
    ph = vf.create_mask(pattern=pl.MaskPattern.H, dtype=pl.DT_UINT16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, ph)
    one = vf.full(1.0, p, dtype=pl.DT_FP32)
    for r in pl.range(0, rows):
        d = vf.load_align(grad, r * 128)
        db = vf.astype(d, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
        d = vf.astype(db, pb, dtype=pl.DT_FP32)
        sig = vf.load_align(beta, token_start + r, dist=pl.LoadDist.BRC_B32)
        inv = vf.sub(one, sig, p)
        d = vf.mul(d, sig, p)
        d = vf.mul(d, inv, p)
        db = vf.astype(d, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
        packed = vf.gather(db, ix)
        vf.store_align(out, packed, p1, r * 128)


@pl.vector_function
def s2_zero_params(out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    z = vf.full(0.0, p, dtype=pl.DT_FP32)
    vf.store_align(out, z, p, 0)
    vf.store_align(out, z, p, 64)


@pl.vector_function
def s2_gate_backward(raw, grad, bias, a_log, out, sums, row_start, rows, lower_bound):
    """AIV owns 64 channels; sums hold FP32 bias[64] and scalar A_log."""
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pb = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    p1 = vf.update_mask(1, dtype=pl.DT_FP32)
    ph = vf.create_mask(pattern=pl.MaskPattern.H, dtype=pl.DT_UINT16)
    pout = vf.update_mask(64, dtype=pl.DT_BF16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, ph)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    bs = vf.load_align(sums, 0)
    old_as = vf.load_align(sums, 64)
    biasv = vf.load_align(bias, 0)
    av = vf.load_align(a_log, 0, dist=pl.LoadDist.BRC_B32)
    av = vf.exp(av, p)
    one = vf.full(1.0, p, dtype=pl.DT_FP32)
    zero = vf.full(0.0, p, dtype=pl.DT_FP32)
    for gate_base in pl.range(0, rows, 8):
        asum = vf.full(0.0, p, dtype=pl.DT_FP32)
        for gate_inner in pl.range(0, pl.min(8, rows - gate_base)):
            r = gate_base + gate_inner
            gb = vf.load_align(raw, r * 128, dist=pl.LoadDist.US_B16)
            g = vf.astype(gb, pb, dtype=pl.DT_FP32)
            x = vf.add(g, biasv, p)
            z = vf.mul(av, x, p)
            e = vf.exp_sub(zero, z, p)
            denom = vf.adds(e, 1.0, p)
            sig = vf.div(one, denom, p)
            inv = vf.sub(one, sig, p)
            dg = vf.load_align(grad, (row_start + r) * 128)
            dgb = vf.astype(dg, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            dg = vf.astype(dgb, pb, dtype=pl.DT_FP32)
            dg = vf.muls(dg, lower_bound, p)
            dg = vf.mul(dg, sig, p)
            dg = vf.mul(dg, inv, p)
            dg = vf.mul(dg, av, p)
            bs = vf.add(bs, dg, p)
            ax = vf.mul(dg, x, p)
            asum = vf.add(asum, ax, p)
            dgb = vf.astype(dg, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            packed = vf.gather(dgb, ix)
            vf.store_align(out, packed, pout, r * 128)
        total = vf.reduce_sum(asum, p)
        old_as = vf.add(old_as, total, p1)
    vf.store_align(sums, bs, p, 0)
    vf.store_align(sums, old_as, p1, 64)


def _resolve_num_cores(device, num_cores):
    """Use one logical Cube block and its paired Vector subblocks per core."""
    available = int(torch.npu.get_device_properties(device).cube_core_num)
    if available < 1:
        raise RuntimeError('device reports no Cube cores')
    if num_cores is None:
        configured = os.environ.get('KDA_BWD_NUM_CORES')
        if configured is not None:
            try:
                num_cores = int(configured)
            except ValueError as exc:
                raise ValueError('KDA_BWD_NUM_CORES must be a positive integer') from exc
        else:
            num_cores = available
    if isinstance(num_cores, bool) or not isinstance(num_cores, int):
        raise TypeError('num_cores must be an integer or None')
    if not 1 <= num_cores <= available:
        raise ValueError(f'num_cores must be in [1, {available}], got {num_cores}')
    return num_cores


def _layout_code(layout_qkv):
    if layout_qkv == 'BSND':
        return 0
    if layout_qkv == 'BNSD':
        return 1
    raise ValueError(f'layout_qkv must be BSND or BNSD, got {layout_qkv!r}')


def _check_tensor(name, value, dtype):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f'{name} must be a torch.Tensor')
    if value.dtype != dtype:
        raise TypeError(f'{name} must be {dtype}, got {value.dtype}')
    if value.device.type != 'npu':
        raise ValueError(f'{name} must be on NPU, got {value.device}')
    if not value.is_contiguous():
        raise ValueError(f'{name} must be contiguous in its layout-native storage')


def _check_layout_shape(t, layout_qkv, batch_size, seq_len, num_heads, head_dim=HEAD_D):
    expected = (
        (batch_size, seq_len, num_heads, head_dim)
        if layout_qkv == 'BSND'
        else (batch_size, num_heads, seq_len, head_dim)
    )
    if tuple(t.shape) != expected:
        raise ValueError(f'expected {expected}, got {tuple(t.shape)}')


@pl.vector_function
def cp_identity_state(dst, v0, k0):
    mask = vf.update_mask(32, dtype=pl.DT_FP32)
    ix = vf.arange(0, dtype=pl.DT_INT32)
    ix = vf.adds(ix, v0, mask)
    one = vf.full(1.0, mask, dtype=pl.DT_FP32)
    zero = vf.full(0.0, mask, dtype=pl.DT_FP32)
    for r in pl.range(0, 64):
        same = vf.eq(ix, k0 + r, mask)
        val = vf.select(one, zero, same)
        vf.store_align(dst, val, mask, r * 128)


@pl.vector_function
def cp_add_state(a, b, out):
    mask = vf.update_mask(32, dtype=pl.DT_FP32)
    for r in pl.range(0, 64):
        x = vf.load_align(a, r * 128)
        y = vf.load_align(b, r * 128)
        z = vf.add(x, y, mask)
        vf.store_align(out, z, mask, r * 128)


@pl.vector_function
def s1_recurrent_add_gamma64(dh, g_last, product, out, scale, first_col):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p = vf.update_mask(32, dtype=pl.DT_FP32)
    for r in pl.range(0, 64, 1):
        mul = r * 128
        x = vf.load_align(dh, mul)
        g = vf.load_align(g_last, first_col + r, dist=pl.LoadDist.BRC_B32)
        eg = g
        m = vf.load_align(product, r * 64)
        v = vf.mul(x, eg, p)
        w = vf.muls(m, scale, p)
        y = vf.add(v, w, p)
        vf.store_align(out, y, p, mul)


def _recurrent_cache_args(cache, fallback_state):
    """Optional forward Kr/gamma cache; older training caches remain supported."""
    kr, gam = cache.get('kr'), cache.get('gam')
    layout = _layout_code(cache['meta'].get('prepared_layout', 'BSND'))
    if kr is None and gam is None:
        return cache['w'], fallback_state, 0, layout
    if kr is None or gam is None:
        raise ValueError('recurrent cache requires kr and gam together')
    for name, value, dtype in (('kr', kr, torch.bfloat16), ('gam', gam, torch.float32)):
        _check_tensor(name, value, dtype)
        if value.device != cache['w'].device:
            raise ValueError('recurrent cache device mismatch')
    if kr.shape != cache['w'].shape or tuple(gam.shape) != tuple(cache['mqk'].shape[:3]) + (128,):
        raise ValueError('recurrent cache shape mismatch')
    return kr, gam, 1, layout


@pl.vector_function
def s1v64_s1_zero_vslice(out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p = vf.update_mask(64, dtype=pl.DT_FP32)
    z = vf.full(0.0, p, dtype=pl.DT_FP32)
    for r in pl.range(0, 64, 1):
        vf.store_align(out, z, p, r * 128)


@pl.vector_function
def s1v64_s1_transposed(src, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pu_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 64, 1):
        for _ in pl.range(0, 1, 1):
            p_1 = vf.update_mask(64, dtype=pl.DT_FP32)
            ix = vf.arange(0, dtype=pl.DT_UINT32)
            ix = vf.shift_left(ix, 7, pu_1)
            ix = vf.adds(ix, r, pu_1)
            x = vf.gather(src, ix, p_1)
            vf.store_align(dst, x, p_1, r * 128)


@pl.vector_function
def s1v64_s1_transposed_1(src, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pu_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 64, 1):
        for _ in pl.range(0, 1, 1):
            p_1 = vf.update_mask(64, dtype=pl.DT_FP32)
            ix = vf.arange(0, dtype=pl.DT_UINT32)
            ix = vf.shift_left(ix, 7, pu_1)
            ix = vf.adds(ix, r, pu_1)
            x = vf.gather(src, ix, p_1)
            vf.store_align(dst, x, p_1, r * 128)


@pl.vector_function
def s1v64_s1_zero_vslice_1(out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p = vf.update_mask(64, dtype=pl.DT_FP32)
    z = vf.full(0.0, p, dtype=pl.DT_FP32)
    for r in pl.range(0, 32, 1):
        vf.store_align(out, z, p, r * 128)


@pl.vector_function
def s1v64_s1_widen(src, dst, rows):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p16_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    for r in pl.range(0, pl.max(rows, 0), 1):
        for _ in pl.range(0, 1, 1):
            p_1 = vf.update_mask(64, dtype=pl.DT_FP32)
            mul_1 = r * 128
            x = vf.load_align(src, mul_1, dist=pl.LoadDist.US_B16)
            y = vf.astype(x, p16_1, dtype=pl.DT_FP32, round_mode=pl.VFRoundMode.CAST_RINT, saturate=pl.SaturateMode.OFF)
            vf.store_align(dst, y, p_1, mul_1)


@pl.vector_function
def s1v64_s1_narrow(src, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    p32_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p64_1 = vf.create_mask(pattern=pl.MaskPattern.H, dtype=pl.DT_FP16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, p64_1)
    for r in pl.range(0, 32, 1):
        for _ in pl.range(0, 1, 1):
            p_1 = vf.update_mask(64, dtype=pl.DT_FP16)
            mul_1 = r * 128
            x = vf.load_align(src, mul_1)
            y = vf.astype(x, p32_1, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT, saturate=pl.SaturateMode.OFF)
            z = vf.gather(y, ix)
            vf.store_align(dst, z, p_1, mul_1)


@pl.vector_function
def s1v64_s1_narrow_2(src, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    p32_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p64_1 = vf.create_mask(pattern=pl.MaskPattern.H, dtype=pl.DT_FP16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, p64_1)
    for r in pl.range(0, 64, 1):
        for _ in pl.range(0, 1, 1):
            p_1 = vf.update_mask(64, dtype=pl.DT_FP16)
            mul_1 = r * 128
            x = vf.load_align(src, mul_1)
            y = vf.astype(x, p32_1, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT, saturate=pl.SaturateMode.OFF)
            z = vf.gather(y, ix)
            vf.store_align(dst, z, p_1, mul_1)


@pl.vector_function
def s1v64_s1_copy_keep32(src, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p = vf.update_mask(64, dtype=pl.DT_FP32)
    for r in pl.range(0, 32, 1):
        mul = r * 64
        x = vf.load_align(src, mul)
        vf.store_align(dst, x, p, mul)


@pl.vector_function
def s1v64_s1_recurrent_add_product64(dh, g_last, product, out, scale, first_col):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p = vf.update_mask(64, dtype=pl.DT_FP32)
    for r in pl.range(0, 64, 1):
        mul = r * 128
        x = vf.load_align(dh, mul)
        g = vf.load_align(g_last, first_col + r, dist=pl.LoadDist.BRC_B32)
        eg = vf.exp(g, p)
        m = vf.load_align(product, r * 64)
        v = vf.mul(x, eg, p)
        w = vf.muls(m, scale, p)
        y = vf.add(v, w, p)
        vf.store_align(out, y, p, mul)


@pl.vector_function
def s1v64_s1_narrow_keep32(src, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    p32_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p64_1 = vf.create_mask(pattern=pl.MaskPattern.H, dtype=pl.DT_FP16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, p64_1)
    for r in pl.range(0, 32, 1):
        for _ in pl.range(0, 1, 1):
            p_1 = vf.update_mask(64, dtype=pl.DT_FP16)
            x = vf.load_align(src, r * 64)
            y = vf.astype(x, p32_1, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT, saturate=pl.SaturateMode.OFF)
            z = vf.gather(y, ix)
            vf.store_align(dst, z, p_1, r * 128)


@pl.vector_function
def s1v64_s1_binary_rhs64(a, b, out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 64, 1):
        for _ in pl.range(0, 1, 1):
            p_1 = vf.update_mask(64, dtype=pl.DT_FP32)
            mul_1 = r * 128
            x = vf.load_align(a, mul_1)
            y = vf.load_align(b, r * 64)
            z = vf.add(x, y, p_1)
            vf.store_align(out, z, p_1, mul_1)


@pl.vector_function
def s1v64_cp_identity_state(dst, v0, k0):
    mask = vf.update_mask(64, dtype=pl.DT_FP32)
    ix = vf.arange(0, dtype=pl.DT_INT32)
    ix = vf.adds(ix, v0, mask)
    one = vf.full(1.0, mask, dtype=pl.DT_FP32)
    zero = vf.full(0.0, mask, dtype=pl.DT_FP32)
    for r in pl.range(0, 64):
        same = vf.eq(ix, k0 + r, mask)
        val = vf.select(one, zero, same)
        vf.store_align(dst, val, mask, r * 128)


@pl.vector_function
def s1v64_s1_recurrent_add_gamma64(dh, g_last, product, out, scale, first_col):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p = vf.update_mask(64, dtype=pl.DT_FP32)
    for r in pl.range(0, 64, 1):
        mul = r * 128
        x = vf.load_align(dh, mul)
        g = vf.load_align(g_last, first_col + r, dist=pl.LoadDist.BRC_B32)
        eg = g
        m = vf.load_align(product, r * 64)
        v = vf.mul(x, eg, p)
        w = vf.muls(m, scale, p)
        y = vf.add(v, w, p)
        vf.store_align(out, y, p, mul)


@pl.vector_function
def s1_direct_state_nz(src, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    ph = vf.update_mask(64, dtype=pl.DT_BF16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, ph)
    col = vf.arange(0, dtype=pl.DT_UINT16)
    block = vf.shift_right(col, 4, ph)
    base = vf.shift_left(block, 4, ph)
    within = vf.sub(col, base, ph)
    nzbase = vf.shift_left(block, 10, ph)
    nzindex = vf.add(nzbase, within, ph)
    for r in pl.range(0, 64):
        x = vf.load_align(src, r * 128)
        xb = vf.astype(x, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
        packed = vf.gather(xb, ix)
        destination = vf.adds(nzindex, r * 16, ph)
        vf.scatter(dst, packed, destination, ph)


@pl.vector_function
def s1_direct_du_nz(src, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    ph = vf.update_mask(64, dtype=pl.DT_BF16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, ph)
    col = vf.arange(0, dtype=pl.DT_UINT16)
    block = vf.shift_right(col, 4, ph)
    base = vf.shift_left(block, 4, ph)
    within = vf.sub(col, base, ph)
    nzbase = vf.shift_left(block, 9, ph)
    nzindex = vf.add(nzbase, within, ph)
    for r in pl.range(0, 32):
        x = vf.load_align(src, r * 64)
        xb = vf.astype(x, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
        packed = vf.gather(xb, ix)
        destination = vf.adds(nzindex, r * 16, ph)
        vf.scatter(dst, packed, destination, ph)


@pl.vector_function
def s1_direct_du_nz_cache(src, dst, nd):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    ph = vf.update_mask(64, dtype=pl.DT_BF16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, ph)
    col = vf.arange(0, dtype=pl.DT_UINT16)
    block = vf.shift_right(col, 4, ph)
    base = vf.shift_left(block, 4, ph)
    within = vf.sub(col, base, ph)
    nzbase = vf.shift_left(block, 9, ph)
    nzindex = vf.add(nzbase, within, ph)
    for r in pl.range(0, 32):
        x = vf.load_align(src, r * 64)
        xb = vf.astype(x, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
        packed = vf.gather(xb, ix)
        destination = vf.adds(nzindex, r * 16, ph)
        vf.scatter(dst, packed, destination, ph)
        vf.store_align(nd, packed, ph, r * 128)


@pl.vector_function
def s1_direct_qg_nz(raw, g, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    ph = vf.create_mask(pattern=pl.MaskPattern.H, dtype=pl.DT_UINT16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, ph)
    pm = vf.update_mask(64, dtype=pl.DT_BF16)
    col = vf.arange(0, dtype=pl.DT_UINT16)
    block = vf.shift_right(col, 4, pm)
    base = vf.shift_left(block, 4, pm)
    within = vf.sub(col, base, pm)
    nzbase = vf.shift_left(block, 9, pm)
    nzindex = vf.add(nzbase, within, pm)
    for r in pl.range(0, 32):
        for j in pl.range(0, 2):
            off = r * 128 + j * 64
            xb = vf.load_align(raw, off, dist=pl.LoadDist.US_B16)
            x = vf.astype(xb, p16, dtype=pl.DT_FP32)
            gv = vf.load_align(g, off)
            e = vf.exp(gv, p)
            y = vf.mul(x, e, p)
            yb = vf.astype(y, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            packed = vf.gather(yb, ix)
            destination = vf.adds(nzindex, r * 16 + j * 2048, pm)
            vf.scatter(dst, packed, destination, pm)


@pl.jit(auto_mutex=True, compile_timeout=280)
def chunk_kda_bwd_stage1_kernel(
    q_c: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    k_c: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    gk_c: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    mqk: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    w: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    do_: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    dht: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    batch_size: pl.DT_INT64,
    total_seq_len: pl.DT_INT64,
    num_heads: pl.DT_INT64,
    total_num_chunks: pl.DT_INT64,
    scale: pl.DT_FP32,
    layout_code: pl.DT_INT64,
    has_dht: pl.DT_INT64,
    da_out: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32],
    dbias_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    init_params: pl.DT_INT64,
    summary: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    has_do: pl.DT_INT64,
    last_continues: pl.DT_INT64,
    packed: pl.DT_INT64,
    segments: pl.DT_INT64,
    cu: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64],
    coffs: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64],
    hoffs: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64],
    num_key_heads: pl.DT_INT64,
    st_do__bs: pl.DT_INT64,
    st_do__ts: pl.DT_INT64,
    st_do__hs: pl.DT_INT64,
    st_dht_ms: pl.DT_INT64,
    st_dht_hs: pl.DT_INT64,
    kr_cache: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    gam_cache: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    has_recurrent_cache: pl.DT_INT64,
    cache_layout: pl.DT_INT64,
    local_state: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    local_du: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    has_local_cache: pl.DT_INT64,
    local_du_bf16: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    du_cache_bf16: pl.DT_INT64,
):
    cache_nc = gam_cache.shape[2]
    cache_p = cache_nc * 64
    cache_ts = 128 * (cache_layout + (1 - cache_layout) * num_heads)
    cache_hs = 128 * (cache_layout * cache_p + 1 - cache_layout)
    ratio = num_heads // num_key_heads
    seq_len = total_seq_len
    num_chunks = total_num_chunks
    num_chunk_states = total_num_chunks + 1
    num_cores = pl.get_block_num()
    kg_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[65536],
        mutex_ids=[16],
        depth=1,
    )
    dh_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[128, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[81920],
        mutex_ids=[17],
        depth=1,
    )
    qg_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[128, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[98304],
        mutex_ids=[18],
        depth=1,
    )
    du_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[114688],
        mutex_ids=[19],
        depth=1,
    )
    tail_do_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[122880],
        mutex_ids=[20],
        depth=1,
    )
    with pl.section_cube():
        result_ub_native = pl.make_tile_group(type=pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[229376], mutex_ids=[0], depth=1)
        result_ub = result_ub_native[0]
        al1_64_64_native = pl.make_tile_group(type=pl.TileType(
            shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat), addrs=[0], mutex_ids=[1], depth=1)
        al1_64_64 = al1_64_64_native[0]
        al0_64_64_native = pl.make_tile_group(type=pl.TileType(
            shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left), addrs=[0], mutex_ids=[2], depth=1)
        al0_64_64 = al0_64_64_native[0]
        al0_64_128_native = pl.make_tile_group(type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left), addrs=[8192], mutex_ids=[4], depth=1)
        al0_64_128 = al0_64_128_native[0]
        al1_128_64_native = pl.make_tile_group(type=pl.TileType(
            shape=[128, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat), addrs=[24576], mutex_ids=[5], depth=1)
        al1_128_64 = al1_128_64_native[0]
        al0_128_64_native = pl.make_tile_group(
            type=pl.TileType(shape=[128, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left),
            addrs=[24576],
            mutex_ids=[6],
            depth=1,
        )
        al0_128_64 = al0_128_64_native[0]
        bl1_64_32_native = pl.make_tile_group(type=pl.TileType(
            shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat), addrs=[40960], mutex_ids=[7], depth=1)
        bl1_64_32 = bl1_64_32_native[0]
        bl0_32_64_native = pl.make_tile_group(type=pl.TileType(
            shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right), addrs=[0], mutex_ids=[8], depth=1)
        bl0_32_64 = bl0_32_64_native[0]
        bl0_32_128_native = pl.make_tile_group(
            type=pl.TileType(shape=[128, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right),
            addrs=[8192],
            mutex_ids=[10],
            depth=1,
        )
        bl0_32_128 = bl0_32_128_native[0]
        acc_64_32_native = pl.make_tile_group(type=pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc), addrs=[0], mutex_ids=[11], depth=1)
        acc_64_32 = acc_64_32_native[0]
        acc_128_32_native = pl.make_tile_group(
            type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
            addrs=[16384],
            mutex_ids=[12],
            depth=1,
        )
        acc_128_32 = acc_128_32_native[0]
        sub = 1 - layout_code
        seq_stride = num_heads * 128 * sub + 128 * layout_code
        head_stride = 128 * sub + seq_len * 128 * layout_code
        core = pl.get_block_idx()
        mul_9 = batch_size * num_heads
        for job in pl.range(core, batch_size * num_heads, num_cores):
            b = job // num_heads
            n = job % num_heads
            cp_seg = b
            cp_head = n
            cp_key = cp_head // ratio
            seq_len = total_seq_len
            num_chunks = total_num_chunks
            cp_cbase = cp_seg * total_num_chunks
            cp_hbase = cp_seg * (total_num_chunks + 1)
            cp_tok = 0
            cp_batch = cp_seg
            if packed:
                seq_len = cu[cp_seg + 1] - cu[cp_seg]
                num_chunks = coffs[cp_seg + 1] - coffs[cp_seg]
                cp_cbase = coffs[cp_seg]
                cp_hbase = hoffs[cp_seg]
                cp_tok = cu[cp_seg]
                cp_batch = 0
            valid_s = seq_len
            single_token = seq_len == 1
            for mode in pl.range(0, 2):
                mode_nc = num_chunks
                if mode == 1 and last_continues == 0:
                    mode_nc = 0
                for vs in pl.range(0, 2):
                    v0 = vs * 64
                    for it in pl.range(0, mode_nc):
                        t = num_chunks - 1 - it
                        cp_t = t
                        cp_chunk = cp_cbase + cp_t
                        pl.system.wait_cross_core(pipe=P.MTE1, event_id=4, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        if has_recurrent_cache != 0:
                            saved_kr = pl.make_tensor(
                                pl.addptr(
                                    pl.make_ptr(kr_cache),
                                    cp_batch * num_heads * cache_p * 128
                                    + cp_head * cache_hs
                                    + (cp_chunk - cp_batch * total_num_chunks) * 64 * cache_ts,
                                ),
                                [64, 128],
                                [cache_ts, 1],
                            )
                            pl.load(kg_l1[0], saved_kr, [0, 0])
                        gm_window_27 = pl.make_tensor(
                            pl.addptr(
                                pl.make_ptr(mqk),
                                ((cp_batch * num_heads + cp_head) * total_num_chunks + cp_t + packed * cp_cbase) * 4096,
                            ),
                            [64, 64],
                            [64, 1],
                        )
                        pl.load(al1_64_64, gm_window_27, [0, 0], order=[1, 0])
                        mul_80 = t * 64
                        cmp_6 = seq_len - mul_80 >= 64 and has_do != 0 and (mode == 0)
                        if cmp_6:
                            gm_direct_do_1 = pl.make_tensor(
                                pl.addptr(
                                    pl.make_ptr(do_),
                                    cp_batch * st_do__bs + cp_tok * st_do__ts + cp_head * st_do__hs,
                                ),
                                [seq_len, 128],
                                [st_do__ts, 1],
                            )
                            pl.load(bl1_64_32, gm_direct_do_1, [mul_80, v0])
                        pl.set_validshape(al0_64_64, [64, 64])
                        pl.move(al0_64_64, al1_64_64)
                        pl.set_validshape(al0_64_64, [64, 64])
                        pl.set_validshape(bl0_32_64, [64, 64])
                        if cmp_6:
                            pl.move(bl0_32_64, bl1_64_32)
                        else:
                            pl.move(bl0_32_64, tail_do_l1[0])
                        pl.set_validshape(bl0_32_64, [64, 64])
                        pl.matmul(acc_64_32, al0_64_64, bl0_32_64)
                        pl.system.wait_cross_core(pipe=P.MTE1, event_id=0, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.set_validshape(al0_64_128, [64, 128])
                        pl.move(al0_64_128, kg_l1[0])
                        pl.set_validshape(al0_64_128, [64, 128])
                        pl.set_validshape(bl0_32_128, [128, 64])
                        pl.move(bl0_32_128, dh_l1[0])
                        pl.set_validshape(bl0_32_128, [128, 64])
                        pl.matmul_acc(acc_64_32, acc_64_32, al0_64_128, bl0_32_128)
                        view_23_fix_view = result_ub[0:0 + 32, 0:0 + 64]
                        pl.move(view_23_fix_view, acc_64_32, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                        pl.system.set_cross_core(pipe=P.FIX, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.FIX, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.MTE1, event_id=0, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.set_validshape(al0_128_64, [128, 64])
                        pl.move(al0_128_64, qg_l1[0])
                        pl.set_validshape(al0_128_64, [128, 64])
                        pl.matmul(acc_128_32, al0_128_64, bl0_32_64)
                        view_33_fix_view = result_ub[0:0 + 64, 0:0 + 64]
                        pl.move(view_33_fix_view, acc_128_32, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                        pl.system.set_cross_core(pipe=P.FIX, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.FIX, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        gm_window_69 = pl.make_tensor(
                            pl.addptr(
                                pl.make_ptr(w),
                                cp_batch * (total_num_chunks * 64) * num_heads * 128
                                + packed * cp_cbase * 64 * num_heads * 128
                                + cp_head * (128 * (1 - layout_code) + total_num_chunks * 64 * 128 * layout_code),
                            ),
                            [total_num_chunks * 64, 128],
                            [seq_stride, 1],
                        )
                        pl.load(al1_128_64, gm_window_69, [mul_80, 0], order=[1, 0])
                        pl.set_validshape(al0_128_64, [128, 64])
                        pl.move(al0_128_64, al1_128_64)
                        pl.set_validshape(al0_128_64, [128, 64])
                        pl.set_validshape(bl0_32_64, [64, 64])
                        pl.move(bl0_32_64, du_l1[0])
                        pl.set_validshape(bl0_32_64, [64, 64])
                        pl.matmul(acc_128_32, al0_128_64, bl0_32_64)
                        view_42_fix_view = result_ub[0:0 + 64, 0:0 + 64]
                        pl.move(view_42_fix_view, acc_128_32, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                        pl.system.set_cross_core(pipe=P.FIX, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.FIX, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
    with pl.section_vector():
        u0_native = pl.make_tile_group(type=pl.TileType(
            shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[0], mutex_ids=[0], depth=1)
        u0 = u0_native[0]
        u1_native = pl.make_tile_group(
            type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=[65536],
            mutex_ids=[1],
            depth=1,
        )
        u1 = u1_native[0]
        u2_native = pl.make_tile_group(
            type=pl.TileType(shape=[32, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=[131072],
            mutex_ids=[2],
            depth=1,
        )
        u2 = u2_native[0]
        dh_nz_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addrs=[131072],
            mutex_ids=[2],
        )
        qg_nz_group = pl.make_tile_group(
            type=pl.TileType(shape=[32, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addrs=[131072],
            mutex_ids=[2],
        )
        du_nz_group = pl.make_tile_group(
            type=pl.TileType(shape=[32, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addrs=[131072],
            mutex_ids=[2],
        )
        dh_nz = dh_nz_group[0]
        qg_nz = qg_nz_group[0]
        du_nz = du_nz_group[0]
        state_native = pl.make_tile_group(
            type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=[163840],
            mutex_ids=[6],
            depth=1,
        )
        state = state_native[0]
        gate_last_native = pl.make_tile_group(type=pl.TileType(
            shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[253952], mutex_ids=[7], depth=1)
        gate_last = gate_last_native[0]
        pack_nd_native = pl.make_tile_group(
            type=pl.TileType(shape=[16, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
            addrs=[147456],
            mutex_ids=[8],
            depth=1,
        )
        pack_nz_native = pl.make_tile_group(
            type=pl.TileType(shape=[16, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addrs=[151552],
            mutex_ids=[9],
            depth=1,
        )
        pack_nd = pack_nd_native[0]
        pack_nz = pack_nz_native[0]
        ub16_native = pl.make_tile_group(
            type=pl.TileType(shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
            addrs=[196608],
            mutex_ids=[3],
            depth=1,
        )
        ub16 = ub16_native[0]
        result_ub_1_native = pl.make_tile_group(type=pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[229376], mutex_ids=[4], depth=1)
        result_ub_1 = result_ub_1_native[0]
        keep_ub_native = pl.make_tile_group(type=pl.TileType(
            shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[245760], mutex_ids=[5], depth=1)
        keep_ub = keep_ub_native[0]
        sub_1 = 1 - layout_code
        seq_stride_1 = num_heads * 128 * sub_1 + 128 * layout_code
        head_stride_1 = 128 * sub_1 + seq_len * 128 * layout_code
        core_1 = pl.get_block_idx() // pl.get_subblock_num()
        sub_4 = pl.get_subblock_idx()
        if init_params:
            s2_zero_params(u0)
            da_view = pl.make_tensor(da_out, [num_heads, 1], [1, 1])
            for head in pl.range(core_1, num_heads, num_cores):
                pl.set_validshape(u0, [1, 64])
                pl.store(dbias_out, u0, [head, sub_4 * 64])
                pl.set_validshape(u0, [128, 128])
                if sub_4 == 0:
                    pl.set_validshape(u0, [1, 1])
                    pl.store(da_view, u0, [head, 0])
                    pl.set_validshape(u0, [128, 128])
        mul_9_1 = batch_size * num_heads
        for job_1 in pl.range(core_1, batch_size * num_heads, num_cores):
            b_1 = job_1 // num_heads
            n_1 = job_1 % num_heads
            cp_seg = b_1
            cp_head = n_1
            cp_key = cp_head // ratio
            seq_len = total_seq_len
            num_chunks = total_num_chunks
            cp_cbase = cp_seg * total_num_chunks
            cp_hbase = cp_seg * (total_num_chunks + 1)
            cp_tok = 0
            cp_batch = cp_seg
            if packed:
                seq_len = cu[cp_seg + 1] - cu[cp_seg]
                num_chunks = coffs[cp_seg + 1] - coffs[cp_seg]
                cp_cbase = coffs[cp_seg]
                cp_hbase = hoffs[cp_seg]
                cp_tok = cu[cp_seg]
                cp_batch = 0
            valid_s = seq_len
            single_token = seq_len == 1
            for mode in pl.range(0, 2):
                mode_nc = num_chunks
                if mode == 1 and last_continues == 0:
                    mode_nc = 0
                for vs in pl.range(0, 2):
                    v0_1 = vs * 64
                    s1v64_s1_zero_vslice(state)
                    if mode == 1:
                        s1v64_cp_identity_state(state, v0_1, sub_4 * 64)
                    elif has_dht != 0:
                        seed = pl.make_tensor(pl.addptr(pl.make_ptr(dht), cp_seg * st_dht_ms +
                                              cp_head * st_dht_hs), [128, 128], [128, 1])
                        pl.set_validshape(u1, [64, 64])
                        pl.load(u1, seed, [v0_1, sub_4 * 64])
                        pl.set_validshape(u1, [128, 128])
                        s1v64_s1_transposed(u1, state)
                    if has_local_cache != 0 and mode == 0:
                        s1v64_s1_narrow_2(state, ub16)
                        cached_terminal_dh_view = pl.make_tensor(
                            pl.addptr(
                                pl.make_ptr(local_state),
                                ((cp_hbase + num_chunks) * num_heads + cp_head) * 16384,
                            ),
                            [128, 128],
                            [128, 1],
                        )
                        pl.set_validshape(ub16, [64, 64])
                        pl.store(cached_terminal_dh_view, ub16, [sub_4 * 64, v0_1])
                        pl.set_validshape(ub16, [128, 128])
                    for it_1 in pl.range(0, mode_nc):
                        t_1 = num_chunks - 1 - it_1
                        cp_t = t_1
                        cp_chunk = cp_cbase + cp_t
                        live = 64
                        mul_33 = t_1 * 64
                        sub_12 = seq_len - mul_33
                        cmp_1 = sub_12 < 64
                        if cmp_1:
                            live = sub_12
                        add_12 = b_1 * (num_chunks * num_heads * 64 * 128) + t_1 * (num_heads * 64 * 128) + n_1 * 8192
                        gm_window_6 = pl.make_tensor(pl.addptr(pl.make_ptr(
                            gk_c), (cp_chunk * num_heads + cp_head) * 8192), [64, 128], [128, 1])
                        mul_43 = sub_4 * 64
                        if has_recurrent_cache != 0:
                            saved_gamma = pl.make_tensor(
                                pl.addptr(
                                    pl.make_ptr(gam_cache),
                                    (
                                        (cp_batch * num_heads + cp_head) * cache_nc
                                        + cp_chunk
                                        - cp_batch * total_num_chunks
                                    )
                                    * 128,
                                ),
                                [1, 128],
                                [128, 1],
                            )
                            pl.load(gate_last, saved_gamma, [0, 0])
                        else:
                            pl.load(gate_last, gm_window_6, [live - 1, 0])
                        if cmp_1 or has_do == 0 or mode == 1:
                            s1v64_s1_zero_vslice_1(u0)
                            mul_49 = sub_4 * 32
                            min_ = pl.min(32, pl.max(0, live - mul_49))
                            if min_ > 0 and has_do != 0 and (mode == 0):
                                gm_window_15 = pl.make_tensor(
                                    pl.addptr(
                                        pl.make_ptr(do_),
                                        cp_batch * st_do__bs + cp_tok * st_do__ts + cp_head * st_do__hs,
                                    ),
                                    [seq_len, 128],
                                    [st_do__ts, 1],
                                )
                                pl.set_validshape(ub16, [pl.min(min_, 128), 64])
                                pl.load(ub16, gm_window_15, [mul_33 + mul_49, v0_1])
                                pl.set_validshape(ub16, [128, 128])
                                s1v64_s1_widen(ub16, u0, min_)
                            s1v64_s1_narrow(u0, ub16)
                            pl.set_validshape(ub16, [32, 64])
                            pl.set_validshape(pack_nd, [16, 64])
                            pl.set_validshape(pack_nz, [16, 64])
                            for pack_row in pl.range(0, 32, 16):
                                pl.move(pack_nd, ub16, offset=[pack_row, 0])
                                pl.move(pack_nz, pack_nd)
                                pl.insert(tail_do_l1[0], pack_nz, [sub_4 * 32 + pack_row, 0])
                            pl.set_validshape(ub16, [128, 128])
                        mul_67 = sub_4 * 32
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=3, sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                        pl.system.wait_cross_core(pipe=P.MTE3, event_id=3,
                                                  sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=4, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        if has_recurrent_cache == 0:
                            gm_window_11 = pl.make_tensor(pl.addptr(pl.make_ptr(
                                k_c), (cp_chunk * num_key_heads + cp_key) * 8192), [64, 128], [128, 1])
                            pl.set_validshape(ub16, [32, 128])
                            pl.load(ub16, gm_window_11, [mul_67, 0])
                            pl.set_validshape(ub16, [128, 128])
                            gm_window_12 = pl.make_tensor(pl.addptr(pl.make_ptr(
                                gk_c), (cp_chunk * num_heads + cp_head) * 8192), [64, 128], [128, 1])
                            pl.set_validshape(u1, [32, 128])
                            pl.load(u1, gm_window_12, [mul_67, 0])
                            pl.set_validshape(u1, [128, 128])
                            s1_kg_bf16(ub16, u1, gate_last, ub16)
                            pl.set_validshape(ub16, [32, 128])
                            pl.set_validshape(pack_nd, [16, 128])
                            pl.set_validshape(pack_nz, [16, 128])
                            for pack_row in pl.range(0, 32, 16):
                                pl.move(pack_nd, ub16, offset=[pack_row, 0])
                                pl.move(pack_nz, pack_nd)
                                pl.insert(kg_l1[0], pack_nz, [mul_67 + pack_row, 0])
                            pl.set_validshape(ub16, [128, 128])
                        s1_direct_state_nz(state, dh_nz)
                        pl.insert(dh_l1[0], dh_nz, [mul_43, 0])
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=3, sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                        pl.system.wait_cross_core(pipe=P.MTE3, event_id=3,
                                                  sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=0, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        if has_recurrent_cache == 1:
                            saved_g = pl.make_tensor(pl.addptr(pl.make_ptr(
                                gk_c), (cp_chunk * num_heads + cp_head) * 8192), [64, 128], [128, 1])
                            pl.set_validshape(u1, [32, 128])
                            pl.load(u1, saved_g, [mul_67, 0])
                            pl.set_validshape(u1, [128, 128])
                        gm_window_8 = pl.make_tensor(pl.addptr(pl.make_ptr(
                            q_c), (cp_chunk * num_key_heads + cp_key) * 8192), [64, 128], [128, 1])
                        pl.set_validshape(ub16, [32, 128])
                        pl.load(ub16, gm_window_8, [mul_67, 0])
                        pl.set_validshape(ub16, [128, 128])
                        s1_direct_qg_nz(ub16, u1, qg_nz)
                        pl.insert(qg_l1[0], qg_nz, [mul_67, 0])
                        pl.system.wait_cross_core(pipe=P.V, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        s1v64_s1_copy_keep32(result_ub_1, keep_ub)
                        if has_local_cache != 0 and mode == 0 and du_cache_bf16 == 0:
                            cached_du_view = pl.make_tensor(pl.addptr(pl.make_ptr(
                                local_du), (cp_chunk * num_heads + cp_head) * 8192), [64, 128], [128, 1])
                            pl.set_validshape(keep_ub, [32, 64])
                            pl.store(cached_du_view, keep_ub, [sub_4 * 32, v0_1])
                            pl.set_validshape(keep_ub, [32, 64])
                        pl.system.set_cross_core(pipe=P.V, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        if du_cache_bf16 != 0 and mode == 0:
                            s1_direct_du_nz_cache(keep_ub, du_nz, ub16)
                        else:
                            s1_direct_du_nz(keep_ub, du_nz)
                        pl.insert(du_l1[0], du_nz, [mul_67, 0])
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=3, sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                        pl.system.wait_cross_core(pipe=P.MTE3, event_id=3,
                                                  sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=0, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        if du_cache_bf16 != 0 and mode == 0:
                            cached_du16_view = pl.make_tensor(pl.addptr(pl.make_ptr(
                                local_du_bf16), (cp_chunk * num_heads + cp_head) * 8192), [64, 128], [128, 1])
                            pl.set_validshape(ub16, [32, 64])
                            pl.store(cached_du16_view, ub16, [sub_4 * 32, v0_1])
                            pl.set_validshape(ub16, [128, 128])
                        pl.system.wait_cross_core(pipe=P.V, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        if has_recurrent_cache != 0:
                            s1v64_s1_recurrent_add_gamma64(state, gate_last, result_ub_1, state, scale, mul_43)
                        else:
                            s1v64_s1_recurrent_add_product64(state, gate_last, result_ub_1, state, scale, mul_43)
                        pl.system.set_cross_core(pipe=P.V, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.V, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        s1v64_s1_binary_rhs64(state, result_ub_1, state)
                        if has_local_cache != 0 and mode == 0:
                            s1v64_s1_narrow_2(state, ub16)
                            cached_dh_view = pl.make_tensor(pl.addptr(pl.make_ptr(local_state), ((
                                cp_hbase + cp_t) * num_heads + cp_head) * 16384), [128, 128], [128, 1])
                            pl.set_validshape(ub16, [64, 64])
                            pl.store(cached_dh_view, ub16, [sub_4 * 64, v0_1])
                            pl.set_validshape(ub16, [128, 128])
                        pl.system.set_cross_core(pipe=P.V, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                    if mode == 1 and last_continues == 0:
                        s1v64_s1_zero_vslice(state)
                    s1v64_s1_transposed_1(state, u1)
                    packet = pl.make_tensor(pl.addptr(pl.make_ptr(summary), ((
                        cp_batch * num_heads + cp_head) * 2 + mode) * 16384), [128, 128], [128, 1])
                    pl.set_validshape(u1, [64, 64])
                    pl.store(packet, u1, [v0_1, sub_4 * 64])
                    pl.set_validshape(u1, [128, 128])


@pl.vector_function
def s2_zero_matrix32x64(out):
    """Only the 32x64 triangle frame is consumed; physical row stride is 128."""
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, p, dtype=pl.DT_FP32)
    for row in pl.range(0, 32):
        vf.store_align(out, zero, p, row * 128)


@pl.vector_function
def s2_local_result(result, exponent, raw, grad, dg, sign, local_rows):
    """Restore one Cube result and update both gradients in a single pass."""
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pb = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    for r in pl.range(0, local_rows):
        for j in pl.range(0, 2):
            off = r * 128 + j * 64
            x = vf.load_align(result, off)
            e = vf.load_align(exponent, off)
            correction = vf.mul(x, e, p)
            previous = vf.load_align(grad, off)
            updated = vf.add(previous, correction, p)
            vf.store_align(grad, updated, p, off)
            xb = vf.load_align(raw, off, dist=pl.LoadDist.US_B16)
            raw32 = vf.astype(xb, pb, dtype=pl.DT_FP32)
            product = vf.mul(correction, raw32, p)
            product = vf.muls(product, sign, p)
            g = vf.load_align(dg, off)
            g = vf.add(g, product, p)
            vf.store_align(dg, g, p, off)


@pl.vector_function
def s2_local_beta_result(result, exponent, raw, beta, first_row, grad, dg, db, local_rows):
    """Preserve beta-dot order, then apply beta and update dk/dg."""
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pb = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    p1 = vf.update_mask(1, dtype=pl.DT_FP32)
    for r in pl.range(0, local_rows):
        beta32 = vf.load_align(beta, first_row + r, dist=pl.LoadDist.BRC_B32)
        acc = vf.full(0.0, p, dtype=pl.DT_FP32)
        for j in pl.range(0, 2):
            off = r * 128 + j * 64
            x = vf.load_align(result, off)
            e = vf.load_align(exponent, off)
            correction = vf.mul(x, e, p)
            xb = vf.load_align(raw, off, dist=pl.LoadDist.US_B16)
            raw32 = vf.astype(xb, pb, dtype=pl.DT_FP32)
            product = vf.mul(correction, raw32, p)
            acc = vf.add(acc, product, p)
            correction = vf.mul(correction, beta32, p)
            previous = vf.load_align(grad, off)
            updated = vf.add(previous, correction, p)
            vf.store_align(grad, updated, p, off)
            product = vf.mul(correction, raw32, p)
            product = vf.muls(product, 1.0, p)
            g = vf.load_align(dg, off)
            g = vf.add(g, product, p)
            vf.store_align(dg, g, p, off)
        total = vf.reduce_sum(acc, p)
        previous = vf.load_align(db, r * 128)
        total = vf.add(previous, total, p1)
        vf.store_align(db, total, p1, r * 128)


@pl.vector_function
def o_prepare(g, k, v, du, beta, e, uh, vb, kb):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pb = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    ph = vf.update_mask(64, dtype=pl.DT_BF16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, ph)
    nzcol = vf.arange(0, dtype=pl.DT_UINT16)
    nzblock = vf.shift_right(nzcol, 4, ph)
    nzbase = vf.shift_left(nzblock, 4, ph)
    nzwithin = vf.sub(nzcol, nzbase, ph)
    nzoffset = vf.shift_left(nzblock, 10, ph)
    nzindex = vf.add(nzoffset, nzwithin, ph)
    for r in pl.range(0, 64):
        b = vf.load_align(beta, r, dist=pl.LoadDist.BRC_B32)
        for j in pl.range(0, 2):
            off = r * 128 + j * 64
            gx = vf.load_align(g, off)
            ex = vf.exp(gx, p)
            vf.store_align(e, ex, p, off)
            kx = vf.load_align(k, off, dist=pl.LoadDist.US_B16)
            kx_cast1 = vf.astype(kx, pb, dtype=pl.DT_FP32)
            vx = vf.load_align(v, off, dist=pl.LoadDist.US_B16)
            vx_cast2 = vf.astype(vx, pb, dtype=pl.DT_FP32)
            ux = vf.load_align(du, off)
            hi_cast3 = vf.astype(ux, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            kh = vf.mul(kx_cast1, ex, p)
            kh = vf.mul(kh, b, p)
            vh = vf.mul(vx_cast2, b, p)
            kh_cast6 = vf.astype(kh, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            vh_cast7 = vf.astype(vh, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            hp = vf.gather(hi_cast3, ix)
            kp = vf.gather(kh_cast6, ix)
            vp = vf.gather(vh_cast7, ix)
            destination = vf.adds(nzindex, r * 16 + j * 4096, ph)
            vf.scatter(uh, hp, destination, ph)
            vf.scatter(kb, kp, destination, ph)
            vf.scatter(vb, vp, destination, ph)


@pl.vector_function
def o_prepare_bf16(g, k, v, du, beta, e, uh, vb, kb):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pb = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    ph = vf.update_mask(64, dtype=pl.DT_BF16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, ph)
    nzcol = vf.arange(0, dtype=pl.DT_UINT16)
    nzblock = vf.shift_right(nzcol, 4, ph)
    nzbase = vf.shift_left(nzblock, 4, ph)
    nzwithin = vf.sub(nzcol, nzbase, ph)
    nzoffset = vf.shift_left(nzblock, 10, ph)
    nzindex = vf.add(nzoffset, nzwithin, ph)
    for r in pl.range(0, 64):
        b = vf.load_align(beta, r, dist=pl.LoadDist.BRC_B32)
        for j in pl.range(0, 2):
            off = r * 128 + j * 64
            gx = vf.load_align(g, off)
            ex = vf.exp(gx, p)
            vf.store_align(e, ex, p, off)
            kx = vf.load_align(k, off, dist=pl.LoadDist.US_B16)
            kx_cast1 = vf.astype(kx, pb, dtype=pl.DT_FP32)
            vx = vf.load_align(v, off, dist=pl.LoadDist.US_B16)
            vx_cast2 = vf.astype(vx, pb, dtype=pl.DT_FP32)
            ux = vf.load_align(du, off, dist=pl.LoadDist.US_B16)
            kh = vf.mul(kx_cast1, ex, p)
            kh = vf.mul(kh, b, p)
            vh = vf.mul(vx_cast2, b, p)
            kh_cast6 = vf.astype(kh, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            vh_cast7 = vf.astype(vh, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            hp = vf.gather(ux, ix)
            kp = vf.gather(kh_cast6, ix)
            vp = vf.gather(vh_cast7, ix)
            destination = vf.adds(nzindex, r * 16 + j * 4096, ph)
            vf.scatter(uh, hp, destination, ph)
            vf.scatter(kb, kp, destination, ph)
            vf.scatter(vb, vp, destination, ph)


@pl.vector_function
def o_state_dot(h, dh, last, out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pb = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    p1 = vf.update_mask(1, dtype=pl.DT_FP32)
    for k in pl.range(0, 128):
        acc = vf.full(0.0, p, dtype=pl.DT_FP32)
        for j in pl.range(0, 2):
            ix = vf.arange(j * 64, dtype=pl.DT_UINT32)
            ix = vf.shift_left(ix, 7, p)
            ix = vf.adds(ix, k, p)
            hx = vf.gather(h, ix, pb)
            hx_cast1 = vf.astype(hx, pb, dtype=pl.DT_FP32)
            dx = vf.load_align(dh, k * 128 + j * 64, dist=pl.LoadDist.US_B16)
            dx_cast2 = vf.astype(dx, pb, dtype=pl.DT_FP32)
            prod = vf.mul(hx_cast1, dx_cast2, p)
            acc = vf.add(acc, prod, p)
        total = vf.reduce_sum(acc, p)
        g = vf.load_align(last, k, dist=pl.LoadDist.BRC_B32)
        g = vf.exp(g, p1)
        total = vf.mul(total, g, p1)
        vf.store_align(out, total, p1, k * 8)


@pl.vector_function
def o_dk0(result, g, last, k, hsum, dk, dg):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pb = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    for j in pl.range(0, 2):
        gl = vf.load_align(last, j * 64)
        acc = vf.full(0.0, p, dtype=pl.DT_FP32)
        for r in pl.range(0, 64):
            off = r * 128 + j * 64
            x = vf.load_align(result, off)
            gr = vf.load_align(g, off)
            eg = vf.sub(gl, gr, p)
            eg = vf.exp(eg, p)
            x = vf.mul(x, eg, p)
            vf.store_align(dk, x, p, off)
            kr = vf.load_align(k, off, dist=pl.LoadDist.US_B16)
            kr_cast1 = vf.astype(kr, pb, dtype=pl.DT_FP32)
            prod = vf.mul(x, kr_cast1, p)
            acc = vf.add(acc, prod, p)
            prod = vf.muls(prod, -1.0, p)
            vf.store_align(dg, prod, p, off)
        ix = vf.arange(j * 64, dtype=pl.DT_UINT32)
        ix = vf.shift_left(ix, 3, p)
        hterm = vf.gather(hsum, ix, p)
        acc = vf.add(acc, hterm, p)
        vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
        old = vf.load_align(dg, 63 * 128 + j * 64)
        acc = vf.add(old, acc, p)
        vf.store_align(dg, acc, p, 63 * 128 + j * 64)


@pl.vector_function
def o_dq0(result, e, q, dq, dg, scale):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pb = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    for r in pl.range(0, 64):
        for j in pl.range(0, 2):
            off = r * 128 + j * 64
            x = vf.load_align(result, off)
            ex = vf.load_align(e, off)
            x = vf.muls(x, scale, p)
            x = vf.mul(x, ex, p)
            vf.store_align(dq, x, p, off)
            qr = vf.load_align(q, off, dist=pl.LoadDist.US_B16)
            qr_cast1 = vf.astype(qr, pb, dtype=pl.DT_FP32)
            prod = vf.mul(x, qr_cast1, p)
            old = vf.load_align(dg, off)
            old = vf.add(old, prod, p)
            vf.store_align(dg, old, p, off)


@pl.vector_function
def o_matrix(result, out, scale, strict):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    z = vf.full(0.0, p, dtype=pl.DT_FP32)
    for r in pl.range(0, 64):
        x = vf.load_align(result, r * 128)
        x = vf.muls(x, scale, p)
        valid = vf.update_mask(pl.min(64, r + 1 - strict), dtype=pl.DT_FP32)
        x = vf.select(x, z, valid)
        vf.store_align(out, x, p, r * 64)


@pl.vector_function
def o_copy64x128(src, dst):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for i in pl.range(0, 128):
        x = vf.load_align(src, i * 64)
        vf.store_align(dst, x, p, i * 64)


@pl.vector_function
def o_dv_high(high, v, beta, db):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pb = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    ph = vf.update_mask(64, dtype=pl.DT_BF16)
    p1 = vf.update_mask(1, dtype=pl.DT_FP32)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, ph)
    for r in pl.range(0, 64):
        b = vf.load_align(beta, r, dist=pl.LoadDist.BRC_B32)
        acc = vf.full(0.0, p, dtype=pl.DT_FP32)
        for j in pl.range(0, 2):
            off = r * 128 + j * 64
            x = vf.load_align(high, off)
            vr = vf.load_align(v, off, dist=pl.LoadDist.US_B16)
            vr_cast1 = vf.astype(vr, pb, dtype=pl.DT_FP32)
            prod = vf.mul(x, vr_cast1, p)
            acc = vf.add(acc, prod, p)
            x = vf.mul(x, b, p)
            x_cast2 = vf.astype(x, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            packed = vf.gather(x_cast2, ix)
            vf.store_align(v, packed, ph, off)
        total = vf.reduce_sum(acc, p)
        vf.store_align(db, total, p1, r * 8)


@pl.vector_function
def o_narrow64(src, dst, scale):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    ph = vf.update_mask(64, dtype=pl.DT_BF16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, ph)
    nzcol = vf.arange(0, dtype=pl.DT_UINT16)
    nzblock = vf.shift_right(nzcol, 4, ph)
    nzbase = vf.shift_left(nzblock, 4, ph)
    nzwithin = vf.sub(nzcol, nzbase, ph)
    nzoffset = vf.shift_left(nzblock, 10, ph)
    nzindex = vf.add(nzoffset, nzwithin, ph)
    for r in pl.range(0, 64):
        for j in pl.range(0, 2):
            off = r * 128 + j * 64
            x = vf.load_align(src, off)
            x = vf.muls(x, scale, p)
            x_cast1 = vf.astype(x, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            x_cast1 = vf.gather(x_cast1, ix)
            destination = vf.adds(nzindex, r * 16 + j * 4096, ph)
            vf.scatter(dst, x_cast1, destination, ph)


@pl.vector_function
def o_strict_narrow(result, out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    ph = vf.update_mask(64, dtype=pl.DT_BF16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, ph)
    nzcol = vf.arange(0, dtype=pl.DT_UINT16)
    nzblock = vf.shift_right(nzcol, 4, ph)
    nzbase = vf.shift_left(nzblock, 4, ph)
    nzwithin = vf.sub(nzcol, nzbase, ph)
    nzoffset = vf.shift_left(nzblock, 10, ph)
    nzindex = vf.add(nzoffset, nzwithin, ph)
    z = vf.full(0.0, p, dtype=pl.DT_FP32)
    for r in pl.range(0, 64):
        x = vf.load_align(result, r * 128)
        valid = vf.update_mask(r, dtype=pl.DT_FP32)
        x = vf.select(x, z, valid)
        x_cast1 = vf.astype(x, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
        x_cast1 = vf.gather(x_cast1, ix)
        destination = vf.adds(nzindex, r * 16, ph)
        vf.scatter(out, x_cast1, destination, ph)


@pl.vector_function
def o_dwk(result, e, k, beta, dk, dg, db):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pb = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    p1 = vf.update_mask(1, dtype=pl.DT_FP32)
    for r in pl.range(0, 64):
        b = vf.load_align(beta, r, dist=pl.LoadDist.BRC_B32)
        acc = vf.full(0.0, p, dtype=pl.DT_FP32)
        for j in pl.range(0, 2):
            off = r * 128 + j * 64
            x = vf.load_align(result, off)
            ex = vf.load_align(e, off)
            kr = vf.load_align(k, off, dist=pl.LoadDist.US_B16)
            kr_cast1 = vf.astype(kr, pb, dtype=pl.DT_FP32)
            dbterm = vf.mul(x, kr_cast1, p)
            dbterm = vf.mul(dbterm, ex, p)
            acc = vf.add(acc, dbterm, p)
            x = vf.mul(x, ex, p)
            x = vf.mul(x, b, p)
            old = vf.load_align(dk, off)
            old = vf.add(old, x, p)
            vf.store_align(dk, old, p, off)
            term = vf.mul(x, kr_cast1, p)
            old = vf.load_align(dg, off)
            old = vf.add(old, term, p)
            vf.store_align(dg, old, p, off)
        total = vf.reduce_sum(acc, p)
        old = vf.load_align(db, r * 8)
        total = vf.add(old, total, p1)
        vf.store_align(db, total, p1, r * 8)


@pl.vector_function
def o_pivot(g0, g1, out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for j in pl.range(0, 2):
        a = vf.load_align(g0, j * 64)
        b = vf.load_align(g1, j * 64)
        a = vf.add(a, b, p)
        a = vf.muls(a, 0.5, p)
        vf.store_align(out, a, p, j * 64)


@pl.vector_function
def o_exp(g, pivot, out, rows, first, band_end, sign):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    z = vf.full(0.0, p, dtype=pl.DT_FP32)
    for r in pl.range(0, rows):
        for j in pl.range(0, 2):
            x = vf.load_align(g, r * 128 + j * 64)
            pv = vf.load_align(pivot, j * 64)
            x = vf.sub(x, pv, p)
            x = vf.muls(x, sign, p)
            if first + r <= band_end:
                x = vf.exp(x, p)
            else:
                x = vf.full(0.0, p, dtype=pl.DT_FP32)
            vf.store_align(out, x, p, r * 128 + j * 64)


@pl.vector_function
def o_local_a(src, tmp, first):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    z = vf.full(0.0, p, dtype=pl.DT_FP32)
    for r in pl.range(0, 16):
        x = vf.load_align(src, (first + r) * 64)
        valid = vf.update_mask(first + r, dtype=pl.DT_FP32)
        x = vf.select(x, z, valid)
        vf.store_align(tmp, x, p, r * 128)


@pl.vector_function
def o_diagonal(da, q, k, dq, dk, first):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pb = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    for r in pl.range(0, 32):
        a = vf.load_align(da, (first + r) * 64 + first + r, dist=pl.LoadDist.BRC_B32)
        for j in pl.range(0, 2):
            off = r * 128 + j * 64
            qr = vf.load_align(q, off, dist=pl.LoadDist.US_B16)
            qr_cast1 = vf.astype(qr, pb, dtype=pl.DT_FP32)
            kr = vf.load_align(k, (first + r) * 128 + j * 64, dist=pl.LoadDist.US_B16)
            kr_cast2 = vf.astype(kr, pb, dtype=pl.DT_FP32)
            qd = vf.load_align(dq, off)
            kd = vf.load_align(dk, off)
            qt = vf.mul(a, kr_cast2, p)
            kt = vf.mul(a, qr_cast1, p)
            qd = vf.add(qd, qt, p)
            kd = vf.add(kd, kt, p)
            vf.store_align(dq, qd, p, off)
            vf.store_align(dk, kd, p, off)


@pl.vector_function
def o_reverse_scan(grad):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for j in pl.range(0, 2):
        acc = vf.full(0.0, p, dtype=pl.DT_FP32)
        for i in pl.range(0, 64):
            r = 63 - i
            x = vf.load_align(grad, r * 128 + j * 64)
            acc = vf.add(acc, x, p)
            vf.store_align(grad, acc, p, r * 128 + j * 64)


@pl.vector_function
def o_beta_chain(db, beta, out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pb = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    one = vf.full(1.0, p, dtype=pl.DT_FP32)
    p1 = vf.update_mask(1, dtype=pl.DT_BF16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, pb)
    for r in pl.range(0, 64):
        x = vf.load_align(db, r * 8, dist=pl.LoadDist.BRC_B32)
        b = vf.load_align(beta, r, dist=pl.LoadDist.BRC_B32)
        x_cast1 = vf.astype(x, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
        x_cast2 = vf.astype(x_cast1, pb, dtype=pl.DT_FP32)
        ib = vf.sub(one, b, p)
        x_cast2 = vf.mul(x_cast2, b, p)
        x_cast2 = vf.mul(x_cast2, ib, p)
        x_cast3 = vf.astype(x_cast2, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
        x_cast3 = vf.gather(x_cast3, ix)
        vf.store_align(out, x_cast3, p1, r * 16)


@pl.vector_function
def o_local_beta_result(result, exponent, raw, beta, first_row, grad, dg, db, local_rows):
    """Preserve beta-dot order, then apply beta and update dk/dg."""
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pb = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    p1 = vf.update_mask(1, dtype=pl.DT_FP32)
    for r in pl.range(0, local_rows):
        beta32 = vf.load_align(beta, first_row + r, dist=pl.LoadDist.BRC_B32)
        acc = vf.full(0.0, p, dtype=pl.DT_FP32)
        for j in pl.range(0, 2):
            off = r * 128 + j * 64
            x = vf.load_align(result, off)
            e = vf.load_align(exponent, off)
            correction = vf.mul(x, e, p)
            xb = vf.load_align(raw, off, dist=pl.LoadDist.US_B16)
            raw32 = vf.astype(xb, pb, dtype=pl.DT_FP32)
            product = vf.mul(correction, raw32, p)
            acc = vf.add(acc, product, p)
            correction = vf.mul(correction, beta32, p)
            previous = vf.load_align(grad, off)
            updated = vf.add(previous, correction, p)
            vf.store_align(grad, updated, p, off)
            product = vf.mul(correction, raw32, p)
            product = vf.muls(product, 1.0, p)
            g = vf.load_align(dg, off)
            g = vf.add(g, product, p)
            vf.store_align(dg, g, p, off)
        total = vf.reduce_sum(acc, p)
        previous = vf.load_align(db, r * 8)
        total = vf.add(previous, total, p1)
        vf.store_align(db, total, p1, r * 8)


@pl.vector_function
def o_state_diag(product, last, out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for j in pl.range(0, 2):
        lane = vf.arange(j * 64, dtype=pl.DT_UINT32)
        row = vf.shift_left(lane, 7, p)
        diagonal = vf.add(row, lane, p)
        value = vf.gather(product, diagonal, p)
        gate = vf.load_align(last, j * 64)
        scale = vf.exp(gate, p)
        result = vf.mul(value, scale, p)
        destination = vf.shift_left(lane, 3, p)
        vf.scatter(out, result, destination, p)


@pl.vector_function
def o_pack_pair(src, high, low, columns):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pb = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    ph = vf.update_mask(64, dtype=pl.DT_BF16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, ph)
    col = vf.arange(0, dtype=pl.DT_UINT16)
    block = vf.shift_right(col, 4, pb)
    base = vf.shift_left(block, 4, pb)
    within = vf.sub(col, base, pb)
    nzbase = vf.shift_left(block, 8, pb)
    nzindex = vf.add(nzbase, within, pb)
    for r in pl.range(0, 16):
        for j in pl.range(0, (columns + 63) // 64):
            x = vf.load_align(src, r * 128 + j * 64)
            hi = vf.astype(x, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            restored = vf.astype(hi, pb, dtype=pl.DT_FP32)
            residual = vf.sub(x, restored, p)
            lo = vf.astype(residual, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            hp = vf.gather(hi, ix)
            lp = vf.gather(lo, ix)
            destination = vf.adds(nzindex, r * 16 + j * 1024, pb)
            vf.scatter(high, hp, destination, ph)
            vf.scatter(low, lp, destination, ph)


@pl.vector_function
def o_a_pack(src, high, low, first):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pb = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    ph = vf.update_mask(64, dtype=pl.DT_BF16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, ph)
    col = vf.arange(0, dtype=pl.DT_UINT16)
    block = vf.shift_right(col, 4, pb)
    base = vf.shift_left(block, 4, pb)
    within = vf.sub(col, base, pb)
    nzbase = vf.shift_left(block, 8, pb)
    nzindex = vf.add(nzbase, within, pb)
    zero = vf.full(0.0, p, dtype=pl.DT_FP32)
    for r in pl.range(0, 16):
        for j in pl.range(0, 1):
            x = vf.load_align(src, (first + r) * 64)
            valid = vf.update_mask(first + r, dtype=pl.DT_FP32)
            x = vf.select(x, zero, valid)
            hi = vf.astype(x, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            restored = vf.astype(hi, pb, dtype=pl.DT_FP32)
            residual = vf.sub(x, restored, p)
            lo = vf.astype(residual, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            hp = vf.gather(hi, ix)
            lp = vf.gather(lo, ix)
            destination = vf.adds(nzindex, r * 16 + j * 1024, pb)
            vf.scatter(high, hp, destination, ph)
            vf.scatter(low, lp, destination, ph)


@pl.vector_function
def o_b_pack(raw, exponent, high, low):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pb = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    ph = vf.update_mask(64, dtype=pl.DT_BF16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, ph)
    col = vf.arange(0, dtype=pl.DT_UINT16)
    block = vf.shift_right(col, 4, pb)
    base = vf.shift_left(block, 4, pb)
    within = vf.sub(col, base, pb)
    nzbase = vf.shift_left(block, 8, pb)
    nzindex = vf.add(nzbase, within, pb)
    for r in pl.range(0, 16):
        for j in pl.range(0, 2):
            raw16 = vf.load_align(raw, r * 128 + j * 64, dist=pl.LoadDist.US_B16)
            raw32 = vf.astype(raw16, pb, dtype=pl.DT_FP32)
            e = vf.load_align(exponent, r * 128 + j * 64)
            x = vf.mul(raw32, e, p)
            hi = vf.astype(x, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            restored = vf.astype(hi, pb, dtype=pl.DT_FP32)
            residual = vf.sub(x, restored, p)
            lo = vf.astype(residual, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            hp = vf.gather(hi, ix)
            lp = vf.gather(lo, ix)
            destination = vf.adds(nzindex, r * 16 + j * 1024, pb)
            vf.scatter(high, hp, destination, ph)
            vf.scatter(low, lp, destination, ph)


@pl.vector_function
def o_beta_pack(raw, exponent, beta, first, high, low):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pb = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    ph = vf.update_mask(64, dtype=pl.DT_BF16)
    ix = vf.arange(0, dtype=pl.DT_UINT16)
    ix = vf.shift_left(ix, 1, ph)
    col = vf.arange(0, dtype=pl.DT_UINT16)
    block = vf.shift_right(col, 4, pb)
    base = vf.shift_left(block, 4, pb)
    within = vf.sub(col, base, pb)
    nzbase = vf.shift_left(block, 8, pb)
    nzindex = vf.add(nzbase, within, pb)
    for r in pl.range(0, 16):
        beta32 = vf.load_align(beta, first + r, dist=pl.LoadDist.BRC_B32)
        for j in pl.range(0, 2):
            raw16 = vf.load_align(raw, r * 128 + j * 64, dist=pl.LoadDist.US_B16)
            raw32 = vf.astype(raw16, pb, dtype=pl.DT_FP32)
            e = vf.load_align(exponent, r * 128 + j * 64)
            x = vf.mul(raw32, e, p)
            x = vf.mul(x, beta32, p)
            hi = vf.astype(x, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            restored = vf.astype(hi, pb, dtype=pl.DT_FP32)
            residual = vf.sub(x, restored, p)
            lo = vf.astype(residual, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            hp = vf.gather(hi, ix)
            lp = vf.gather(lo, ix)
            destination = vf.adds(nzindex, r * 16 + j * 1024, pb)
            vf.scatter(high, hp, destination, ph)
            vf.scatter(low, lp, destination, ph)


@pl.vector_function
def o_l2_cached_inplace(norm_out, grad, rstd, rows, offset):
    """Community-style chain using BF16 normalized input and FP32 rstd."""
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pb = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    for r in pl.range(0, rows):
        xh0 = vf.load_align(norm_out, r * 128, dist=pl.LoadDist.US_B16)
        xh1 = vf.load_align(norm_out, r * 128 + 64, dist=pl.LoadDist.US_B16)
        x0 = vf.astype(xh0, pb, dtype=pl.DT_FP32)
        x1 = vf.astype(xh1, pb, dtype=pl.DT_FP32)
        d0 = vf.load_align(grad, r * 128)
        d1 = vf.load_align(grad, r * 128 + 64)
        db0 = vf.astype(d0, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
        db1 = vf.astype(d1, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
        d0 = vf.astype(db0, pb, dtype=pl.DT_FP32)
        d1 = vf.astype(db1, pb, dtype=pl.DT_FP32)
        t0 = vf.mul(d0, x0, p)
        t1 = vf.mul(d1, x1, p)
        a0 = vf.reduce_sum(t0, p)
        a1 = vf.reduce_sum(t1, p)
        dot = vf.add(a0, a1, p)
        dot = vf.full(dot, p)
        c0 = vf.mul(x0, dot, p)
        c1 = vf.mul(x1, dot, p)
        z0 = vf.sub(d0, c0, p)
        z1 = vf.sub(d1, c1, p)
        inv = vf.load_align(rstd, offset + r, dist=pl.LoadDist.BRC_B32)
        y0 = vf.mul(z0, inv, p)
        y1 = vf.mul(z1, inv, p)
        b0 = vf.astype(y0, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
        b1 = vf.astype(y1, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
        packed, unused = vf.de_interleave(b0, b1)
        vf.store_align(norm_out, packed, pb, r * 128)


@pl.vector_function
def o_exp_pair(g, pivot, positive, negative):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 32):
        for j in pl.range(0, 2):
            x = vf.load_align(g, r * 128 + j * 64)
            pv = vf.load_align(pivot, j * 64)
            shifted = vf.sub(x, pv, p)
            neg = vf.muls(shifted, -1.0, p)
            ep = vf.exp(shifted, p)
            en = vf.exp(neg, p)
            vf.store_align(positive, ep, p, r * 128 + j * 64)
            vf.store_align(negative, en, p, r * 128 + j * 64)


@pl.vector_function
def o_zero_exp_half(out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, p, dtype=pl.DT_FP32)
    for r in pl.range(0, 32):
        vf.store_align(out, zero, p, r * 128)
        vf.store_align(out, zero, p, r * 128 + 64)


@pl.inline
def o_owned_stage2(
    q_c: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    k_c: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    v_c: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    gk_c: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    beta_c: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    akk: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    h: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    v_new: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    do_: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    dq: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    dk: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    dv: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    db: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    dgi: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    workspace: pl.Tensor[[pl.DYNAMIC], pl.DT_UINT8],
    batch_size: pl.DT_INT64,
    total_seq_len: pl.DT_INT64,
    num_heads: pl.DT_INT64,
    total_num_chunks: pl.DT_INT64,
    total_num_chunk_states: pl.DT_INT64,
    scale: pl.DT_FP32,
    layout_code: pl.DT_INT64,
    single_token: pl.DT_INT64,
    q_raw: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    k_raw: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    fuse_preprocess: pl.DT_INT64,
    valid_s: pl.DT_INT64,
    g_raw: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    a_log: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32],
    dt_bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    da_out: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32],
    dbias_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    lower_bound: pl.DT_FP32,
    reduce_mode: pl.DT_INT64,
    mqk: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    w: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    dht: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    all_summary: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    recur_state: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    recur_du: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    initial_grad: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    has_do: pl.DT_INT64,
    has_dht: pl.DT_INT64,
    cp_rank: pl.DT_INT64,
    cp_size: pl.DT_INT64,
    first_continues: pl.DT_INT64,
    last_continues: pl.DT_INT64,
    packed: pl.DT_INT64,
    segments: pl.DT_INT64,
    cu: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64],
    coffs: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64],
    hoffs: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64],
    num_key_heads: pl.DT_INT64,
    qk_sum: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    st_do__bs: pl.DT_INT64,
    st_do__ts: pl.DT_INT64,
    st_do__hs: pl.DT_INT64,
    st_dht_ms: pl.DT_INT64,
    st_dht_hs: pl.DT_INT64,
    st_q_raw_bs: pl.DT_INT64,
    st_q_raw_ts: pl.DT_INT64,
    st_q_raw_hs: pl.DT_INT64,
    st_k_raw_bs: pl.DT_INT64,
    st_k_raw_ts: pl.DT_INT64,
    st_k_raw_hs: pl.DT_INT64,
    st_g_raw_bs: pl.DT_INT64,
    st_g_raw_ts: pl.DT_INT64,
    st_g_raw_hs: pl.DT_INT64,
    st_dq_bs: pl.DT_INT64,
    st_dq_ts: pl.DT_INT64,
    st_dq_hs: pl.DT_INT64,
    st_dk_bs: pl.DT_INT64,
    st_dk_ts: pl.DT_INT64,
    st_dk_hs: pl.DT_INT64,
    st_dv_bs: pl.DT_INT64,
    st_dv_ts: pl.DT_INT64,
    st_dv_hs: pl.DT_INT64,
    st_db_bs: pl.DT_INT64,
    st_db_ts: pl.DT_INT64,
    st_db_hs: pl.DT_INT64,
    st_dgi_bs: pl.DT_INT64,
    st_dgi_ts: pl.DT_INT64,
    st_dgi_hs: pl.DT_INT64,
    st_initial_grad_ms: pl.DT_INT64,
    st_initial_grad_hs: pl.DT_INT64,
    has_local_cache: pl.DT_INT64,
    cached_state: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    cached_du_bf16: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    du_cache_bf16: pl.DT_INT64,
    qk_rstd: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    has_qk_rstd: pl.DT_INT64,
):
    uh = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[0, 131072],
        mutex_ids=[0, 1],
    )
    ul = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[16384, 147456],
        mutex_ids=[2, 3],
    )
    vbt = pl.make_tile_group(
        type=pl.TileType(shape=[128, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[32768, 163840],
        mutex_ids=[4, 5],
    )
    kbt = pl.make_tile_group(
        type=pl.TileType(shape=[128, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[49152, 180224],
        mutex_ids=[6, 7],
    )
    dwm = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[65536, 196608],
        mutex_ids=[8, 9],
    )
    tmpm = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[90112, 221184],
        mutex_ids=[10, 11],
    )
    lah = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[0, 131072],
        mutex_ids=[0, 1],
    )
    lal = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[16384, 147456],
        mutex_ids=[2, 3],
    )
    lath = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[8192, 139264],
        mutex_ids=[0, 1],
    )
    latl = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[24576, 155648],
        mutex_ids=[2, 3],
    )
    lbh = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[32768, 163840],
        mutex_ids=[4, 5],
    )
    lbl = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[49152, 180224],
        mutex_ids=[6, 7],
    )
    rbh = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[98304, 229376],
        mutex_ids=[12, 13],
    )
    rbl = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[114688, 245760],
        mutex_ids=[10, 11],
    )
    cores = pl.get_block_num()
    with pl.section_cube():
        hleft_group = pl.make_tile_group(type=pl.TileType(
            shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left), addrs=[0], mutex_ids=[20])
        hleft = hleft_group[0]
        hacc_group = pl.make_tile_group(type=pl.TileType(
            shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc), addrs=[0], mutex_ids=[22])
        hacc = hacc_group[0]
        hresult_c_group = pl.make_tile_group(type=pl.TileType(
            shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[0], mutex_ids=[29])
        hresult_c = hresult_c_group[0]
        at = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=[81920, 212992],
            mutex_ids=[12, 13],
        )
        hmat = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ
            ),
            addrs=[98304, 229376],
            mutex_ids=[14, 15],
        )
        tprod = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=[0, 131072],
            mutex_ids=[0, 1],
        )
        am = pl.make_tile_group(
            type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=[262144, 311296],
            mutex_ids=[16, 17],
        )
        bm = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ
            ),
            addrs=[278528, 327680],
            mutex_ids=[18, 19],
        )
        al_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left), addrs=[0], mutex_ids=[20])
        al = al_group[0]
        al64_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left), addrs=[0], mutex_ids=[20])
        al64 = al64_group[0]
        alo_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left), addrs=[16384], mutex_ids=[26])
        alo = alo_group[0]
        bl_group = pl.make_tile_group(type=pl.TileType(
            shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right), addrs=[0], mutex_ids=[21])
        bl = bl_group[0]
        bln64_group = pl.make_tile_group(type=pl.TileType(
            shape=[128, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right), addrs=[0], mutex_ids=[21])
        bln64 = bln64_group[0]
        blk64_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right), addrs=[0], mutex_ids=[21])
        blk64 = blk64_group[0]
        bl64_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right), addrs=[0], mutex_ids=[21])
        bl64 = bl64_group[0]
        blo_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right), addrs=[32768], mutex_ids=[27])
        blo = blo_group[0]
        acc_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc), addrs=[0], mutex_ids=[22])
        acc = acc_group[0]
        acc64_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc), addrs=[0], mutex_ids=[22])
        acc64 = acc64_group[0]
        sq = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
            addrs=[32768, 49152],
            mutex_ids=[23, 24],
        )
        cresult_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[196608], mutex_ids=[25])
        cresult = cresult_group[0]
        clocal_result_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[204800], mutex_ids=[25])
        clocal_result = clocal_result_group[0]
        cresult64 = cresult[0:64, 0:64]
        core = pl.get_block_idx()
        for pair in pl.range(core, (batch_size * total_num_chunks * num_heads + 1) // 2, cores):
            for phase in pl.range(0, 11):
                for sub in pl.range(0, 2):
                    task = pair * 2 + sub
                    if task < batch_size * total_num_chunks * num_heads and phase != 5:
                        b = task // (total_num_chunks * num_heads)
                        c = task // num_heads % total_num_chunks
                        n = task % num_heads
                        tok = ((b * total_seq_len + c * 64) * num_heads + n) * 128
                        hv = pl.make_tensor(pl.addptr(pl.make_ptr(h), ((b * (total_num_chunks + 1) + c)
                                            * num_heads + n) * 16384), [128, 128], [128, 1])
                        dhv = pl.make_tensor(pl.addptr(pl.make_ptr(cached_state), ((
                            b * (total_num_chunks + 1) + c + 1) * num_heads + n) * 16384), [128, 128], [128, 1])
                        vnv = pl.make_tensor(pl.addptr(pl.make_ptr(v_new), task * 8192), [128, 64], [64, 1])
                        dov = pl.make_tensor(pl.addptr(pl.make_ptr(do_), tok), [64, 128], [num_heads * 128, 1])
                        av = pl.make_tensor(pl.addptr(pl.make_ptr(akk), task * 4096), [64, 64], [64, 1])
                        if phase == 0:
                            pl.load(hmat[sub], hv, [0, 0])
                            pl.load(bm[sub], dhv, [0, 0])
                            pl.system.wait_cross_core(pipe=P.MTE1, event_id=sub * 16,
                                                      sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
                            pl.move(hleft, bm[sub])
                            pl.move(bl, hmat[sub])
                            pl.matmul(hacc, hleft, bl)
                            if sub == 0:
                                pl.move(hresult_c, hacc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                            else:
                                pl.move(hresult_c, hacc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec1)
                            pl.system.set_cross_core(pipe=P.FIX, event_id=4 + sub * 16,
                                                     sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
                        elif phase == 1:
                            pl.load(at[sub], av, [0, 0], order=[1, 0])
                            pl.load(am[sub], vnv, [0, 0], order=[1, 0])
                            pl.load(bm[sub], dhv, [0, 0], order=[1, 0])
                            pl.set_validshape(al, [64, 128])
                            pl.set_validshape(bl, [128, 128])
                            pl.move(al, am[sub])
                            pl.move(bl, bm[sub])
                            pl.set_validshape(acc, [64, 128])
                            pl.matmul(acc, al, bl)
                        elif phase == 2:
                            pl.load(am[sub], dov, [0, 0])
                            pl.set_validshape(bm[sub], [128, 64])
                            pl.load(bm[sub], vnv, [0, 0])
                            pl.set_validshape(al, [64, 128])
                            pl.set_validshape(bln64, [128, 64])
                            pl.move(al, am[sub])
                            pl.move(bln64, bm[sub], offset=[0, 0])
                            pl.set_validshape(acc64, [64, 64])
                            pl.matmul(acc64, al, bln64)
                            pl.set_validshape(bm[sub], [128, 128])
                        elif phase == 3:
                            pl.load(am[sub], dov, [0, 0])
                            pl.set_validshape(al, [64, 128])
                            pl.set_validshape(bl, [128, 128])
                            pl.move(al, am[sub])
                            pl.move(bl, hmat[sub])
                            pl.set_validshape(acc, [64, 128])
                            pl.matmul(acc, al, bl)
                        elif phase == 4:
                            pl.set_validshape(al64, [64, 64])
                            pl.set_validshape(blk64, [64, 128])
                            pl.move(al64, at[sub])
                            pl.move(blk64, uh[sub])
                            pl.set_validshape(acc, [64, 128])
                            pl.matmul(acc, al64, blk64)
                        elif phase == 5:
                            pl.set_validshape(al64, [64, 64])
                            pl.set_validshape(blk64, [64, 128])
                            pl.move(al64, at[sub])
                            pl.move(blk64, ul[sub])
                            pl.set_validshape(acc, [64, 128])
                            pl.matmul(acc, al64, blk64)
                        elif phase == 6:
                            pl.set_validshape(al, [64, 128])
                            pl.set_validshape(bln64, [128, 64])
                            pl.move(al, uh[sub])
                            pl.move(bln64, vbt[sub])
                            pl.set_validshape(sq[sub], [64, 64])
                            pl.matmul(sq[sub], al, bln64)
                        elif phase == 7:
                            pl.set_validshape(al, [64, 128])
                            pl.set_validshape(bl, [128, 128])
                            pl.move(al, uh[sub])
                            pl.move(bl, hmat[sub])
                            pl.set_validshape(acc, [64, 128])
                            pl.matmul(acc, al, bl)
                        elif phase == 8:
                            pl.system.wait_cross_core(pipe=P.MTE1, event_id=3 + sub * 16,
                                                      sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
                            pl.set_validshape(al, [64, 128])
                            pl.set_validshape(bln64, [128, 64])
                            pl.move(al, dwm[sub])
                            pl.move(bln64, kbt[sub])
                            pl.set_validshape(sq[sub], [64, 64])
                            pl.matmul_acc(sq[sub], sq[sub], al, bln64)
                        elif phase == 9:
                            pl.set_validshape(al64, [64, 64])
                            pl.set_validshape(blk64, [64, 128])
                            pl.move(al64, at[sub])
                            pl.move(blk64, dwm[sub])
                            pl.set_validshape(acc, [64, 128])
                            pl.matmul(acc, al64, blk64)
                        else:
                            pl.system.wait_cross_core(pipe=P.MTE1, event_id=3 + sub * 16,
                                                      sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
                            pl.set_validshape(al64, [64, 64])
                            pl.set_validshape(bl64, [64, 64])
                            pl.move(al64, tmpm[sub])
                            pl.move(bl64, at[sub])
                            pl.set_validshape(sq[sub], [64, 64])
                            pl.matmul(sq[sub], al64, bl64)
                            pl.move(tprod[sub], sq[sub])
                            pl.set_validshape(al64, [64, 64])
                            pl.set_validshape(bl64, [64, 64])
                            pl.move(al64, at[sub])
                            pl.move(bl64, tprod[sub])
                            pl.set_validshape(sq[sub], [64, 64])
                            pl.matmul(sq[sub], al64, bl64)
                        if phase >= 1 and phase != 6:
                            if phase != 1:
                                pl.system.wait_cross_core(pipe=P.FIX, event_id=2 + sub * 16,
                                                          sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
                            if phase == 2 or phase == 8 or phase == 10:
                                pl.set_validshape(cresult, [64, 64])
                            else:
                                pl.set_validshape(cresult, [64, 128])
                            if phase == 8 or phase == 10:
                                if sub == 0:
                                    pl.move(cresult64, sq[sub], acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                                else:
                                    pl.move(cresult64, sq[sub], acc_to_vec_mode=pl.AccToVecMode.SingleModeVec1)
                            elif phase == 2:
                                if sub == 0:
                                    pl.move(cresult64, acc64, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                                else:
                                    pl.move(cresult64, acc64, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec1)
                            else:
                                if sub == 0:
                                    pl.move(cresult, acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                                else:
                                    pl.move(cresult, acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec1)
                            pl.system.set_cross_core(pipe=P.FIX, event_id=1 + sub * 16,
                                                     sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
            pl.system.bar_all()
            for _ in pl.range(0, 2):
                for op in pl.range(0, 2):
                    for sub in pl.range(0, 2):
                        task = pair * 2 + sub
                        if task < batch_size * total_num_chunks * num_heads:
                            pl.system.wait_cross_core(pipe=P.MTE1, event_id=3 + op * 3 + sub *
                                                      16, sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
                            pl.set_validshape(al64, [64, 64])
                            pl.set_validshape(alo, [64, 64])
                            pl.set_validshape(blk64, [64, 128])
                            pl.set_validshape(blo, [64, 128])
                            if op == 0:
                                pl.move(al64, lah[sub])
                                pl.move(alo, lal[sub])
                            else:
                                pl.move(al64, lath[sub])
                                pl.move(alo, latl[sub])
                            if op == 0:
                                pl.move(blk64, lbh[sub])
                                pl.move(blo, lbl[sub])
                            else:
                                pl.move(blk64, rbh[sub])
                                pl.move(blo, rbl[sub])
                            pl.set_validshape(acc, [64, 128])
                            pl.matmul(acc, al64, blk64)
                            pl.matmul_acc(acc, acc, alo, blk64)
                            pl.matmul_acc(acc, acc, al64, blo)
                            pl.system.wait_cross_core(pipe=P.FIX, event_id=2 + sub * 16,
                                                      sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
                            if sub == 0:
                                pl.move(clocal_result, acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                            else:
                                pl.move(clocal_result, acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec1)
                            pl.system.set_cross_core(pipe=P.FIX, event_id=1 + sub * 16,
                                                     sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
            for sub in pl.range(0, 2):
                if pair * 2 + sub < batch_size * total_num_chunks * num_heads:
                    pl.system.wait_cross_core(pipe=P.FIX, event_id=2 + sub * 16,
                                              sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
            pl.system.bar_all()
    with pl.section_vector():
        local_lo_group = pl.make_tile_group(
            type=pl.TileType(shape=[16, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addrs=[254976, 246784],
            mutex_ids=[29, 31],
        )
        local_lo = local_lo_group[0]
        hresult_v_group = pl.make_tile_group(type=pl.TileType(
            shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[0], mutex_ids=[14])
        hresult_v = hresult_v_group[0]
        gbuf_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[0], mutex_ids=[14])
        gbuf = gbuf_group[0]
        kinit_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec), addrs=[32768], mutex_ids=[15])
        kinit = kinit_group[0]
        vinit_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec), addrs=[49152], mutex_ids=[19])
        vinit = vinit_group[0]
        ubuf_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[65536], mutex_ids=[16])
        ubuf = ubuf_group[0]
        du16_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec), addrs=[65536], mutex_ids=[16])
        du16 = du16_group[0]
        ebuf_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[98304], mutex_ids=[17])
        ebuf = ebuf_group[0]
        puh_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addrs=[131072],
            mutex_ids=[18],
        )
        puh = puh_group[0]
        pul_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addrs=[147456],
            mutex_ids=[20],
        )
        pul = pul_group[0]
        pvb_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addrs=[163840],
            mutex_ids=[22],
        )
        pvb = pvb_group[0]
        pkb_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addrs=[180224],
            mutex_ids=[23],
        )
        pkb = pkb_group[0]
        result_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[196608], mutex_ids=[19])
        result = result_group[0]
        beta_group = pl.make_tile_group(type=pl.TileType(
            shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[229376], mutex_ids=[24])
        beta = beta_group[0]
        db_base_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[230400], mutex_ids=[25])
        db_base = db_base_group[0]
        hsum_group = pl.make_tile_group(type=pl.TileType(
            shape=[128, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[232448], mutex_ids=[26])
        hsum = hsum_group[0]
        last_group = pl.make_tile_group(type=pl.TileType(
            shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[236544], mutex_ids=[27])
        last = last_group[0]
        pack_nz_group = pl.make_tile_group(
            type=pl.TileType(shape=[16, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addrs=[237568],
            mutex_ids=[28],
        )
        pack_nz = pack_nz_group[0]
        pack_nz64_group = pl.make_tile_group(
            type=pl.TileType(shape=[16, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addrs=[237568],
            mutex_ids=[28],
        )
        pack_nz64 = pack_nz64_group[0]
        hraw_group = pl.make_tile_group(type=pl.TileType(
            shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec), addrs=[0], mutex_ids=[14])
        hraw = hraw_group[0]
        dhraw_group = pl.make_tile_group(type=pl.TileType(
            shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec), addrs=[32768], mutex_ids=[15])
        dhraw = dhraw_group[0]
        gdk_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[32768], mutex_ids=[15])
        gdk = gdk_group[0]
        kbase_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec), addrs=[163840], mutex_ids=[23])
        kbase = kbase_group[0]
        da_base_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[131072], mutex_ids=[18])
        da_base = da_base_group[0]
        dvb_hi_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[147456], mutex_ids=[20])
        dvb_hi = dvb_hi_group[0]
        vwork_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec), addrs=[180224], mutex_ids=[22])
        vwork = vwork_group[0]
        trilpack_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addrs=[180224],
            mutex_ids=[22],
        )
        trilpack = trilpack_group[0]
        dwpack_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addrs=[163840],
            mutex_ids=[23],
        )
        dwpack = dwpack_group[0]
        dl_base_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[147456], mutex_ids=[20])
        dl_base = dl_base_group[0]
        da_local_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[98304], mutex_ids=[17])
        da_local = da_local_group[0]
        dl_local_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[114688], mutex_ids=[17])
        dl_local = dl_local_group[0]
        k_local_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec), addrs=[131072], mutex_ids=[18])
        k_local = k_local_group[0]
        q_local_group = pl.make_tile_group(type=pl.TileType(
            shape=[32, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec), addrs=[147456], mutex_ids=[20])
        q_local = q_local_group[0]
        ei_group = pl.make_tile_group(type=pl.TileType(
            shape=[32, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[155648], mutex_ids=[22])
        ei = ei_group[0]
        ej_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[172032], mutex_ids=[23])
        ej = ej_group[0]
        local_result_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[204800], mutex_ids=[19])
        local_result = local_result_group[0]
        beta_local_group = pl.make_tile_group(type=pl.TileType(
            shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[237568], mutex_ids=[24])
        beta_local = beta_local_group[0]
        norm_rs_group = pl.make_tile_group(type=pl.TileType(
            shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[238080], mutex_ids=[21])
        norm_rs = norm_rs_group[0]
        db_local_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[239616], mutex_ids=[25])
        db_local = db_local_group[0]
        pivot_group = pl.make_tile_group(type=pl.TileType(
            shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[241664], mutex_ids=[26])
        pivot = pivot_group[0]
        gscratch_group = pl.make_tile_group(type=pl.TileType(
            shape=[32, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[242688], mutex_ids=[27])
        gscratch = gscratch_group[0]
        gscratch0_group = pl.make_tile_group(type=pl.TileType(
            shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[242688], mutex_ids=[27])
        gscratch0 = gscratch0_group[0]
        gscratch1_group = pl.make_tile_group(type=pl.TileType(
            shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[243200], mutex_ids=[28])
        gscratch1 = gscratch1_group[0]
        local_nz_group = pl.make_tile_group(
            type=pl.TileType(shape=[16, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addrs=[250880, 242688],
            mutex_ids=[28, 30],
        )
        local_nz = local_nz_group[0]
        raw_chain_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec), addrs=[131072], mutex_ids=[18])
        raw_chain = raw_chain_group[0]
        chain_out_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec), addrs=[147456], mutex_ids=[20])
        chain_out = chain_out_group[0]
        beta_out_group = pl.make_tile_group(type=pl.TileType(
            shape=[64, 16], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec), addrs=[163840], mutex_ids=[22])
        beta_out = beta_out_group[0]
        chain_bias_group = pl.make_tile_group(type=pl.TileType(
            shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[167936], mutex_ids=[26])
        chain_bias = chain_bias_group[0]
        chain_alog_group = pl.make_tile_group(type=pl.TileType(
            shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[168448], mutex_ids=[27])
        chain_alog = chain_alog_group[0]
        chain_sums_group = pl.make_tile_group(type=pl.TileType(
            shape=[2, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=[168960], mutex_ids=[28])
        chain_sums = chain_sums_group[0]
        gdq = gbuf
        gdg = ubuf
        qbase = vwork
        weighted = gscratch[0:16, 0:128]
        pl.set_validshape(chain_alog, [1, 1])
        sub = pl.get_subblock_idx()
        core = pl.get_block_idx() // pl.get_subblock_num()
        for pair in pl.range(core, (batch_size * total_num_chunks * num_heads + 1) // 2, cores):
            task = pair * 2 + sub
            if task < batch_size * total_num_chunks * num_heads:
                b = task // (total_num_chunks * num_heads)
                c = task // num_heads % total_num_chunks
                n = task % num_heads
                tok = ((b * total_seq_len + c * 64) * num_heads + n) * 128
                gv = pl.make_tensor(pl.addptr(pl.make_ptr(gk_c), task * 8192), [64, 128], [128, 1])
                kv = pl.make_tensor(pl.addptr(pl.make_ptr(k_c), task * 8192), [64, 128], [128, 1])
                qv = pl.make_tensor(pl.addptr(pl.make_ptr(q_c), task * 8192), [64, 128], [128, 1])
                vv = pl.make_tensor(pl.addptr(pl.make_ptr(v_c), task * 8192), [64, 128], [128, 1])
                betav = pl.make_tensor(pl.addptr(pl.make_ptr(beta_c), task * 64), [1, 64], [64, 1])
                hout = pl.make_tensor(pl.addptr(pl.make_ptr(h), ((b * (total_num_chunks + 1) + c) * num_heads + n)
                                      * 16384), [128, 128], [128, 1])
                dhout = pl.make_tensor(pl.addptr(pl.make_ptr(cached_state), ((
                    b * (total_num_chunks + 1) + c + 1) * num_heads + n) * 16384), [128, 128], [128, 1])
                dqv = pl.make_tensor(pl.addptr(pl.make_ptr(dq), tok), [64, 128], [num_heads * 128, 1])
                dkv = pl.make_tensor(pl.addptr(pl.make_ptr(dk), tok), [64, 128], [num_heads * 128, 1])
                dvv = pl.make_tensor(pl.addptr(pl.make_ptr(dv), tok), [64, 128], [num_heads * 128, 1])
                dgv = pl.make_tensor(pl.addptr(pl.make_ptr(dgi), tok), [64, 128], [num_heads * 128, 1])
                dbv = pl.make_tensor(
                    pl.addptr(pl.make_ptr(db), (b * total_seq_len + c * 64) * num_heads + n),
                    [64, 1],
                    [num_heads, 1],
                )
                pl.load(gbuf, gv, [0, 0])
                pl.load(kinit, kv, [0, 0])
                pl.load(vinit, vv, [0, 0])
                if du_cache_bf16 != 0:
                    uv16 = pl.make_tensor(pl.addptr(pl.make_ptr(cached_du_bf16), task * 8192), [64, 128], [128, 1])
                    pl.load(du16, uv16, [0, 0])
                else:
                    uv = pl.make_tensor(pl.addptr(pl.make_ptr(recur_du), task * 8192), [64, 128], [128, 1])
                    pl.load(ubuf, uv, [0, 0])
                pl.load(beta, betav, [0, 0])
                pl.load(last, gv, [63, 0])
                if du_cache_bf16 != 0:
                    o_prepare_bf16(gbuf, kinit, vinit, du16, beta, ebuf, puh, pvb, pkb)
                else:
                    o_prepare(gbuf, kinit, vinit, ubuf, beta, ebuf, puh, pvb, pkb)
                pl.insert(uh[sub], puh, [0, 0])
                pl.insert(vbt[sub], pvb, [0, 0])
                pl.insert(kbt[sub], pkb, [0, 0])
                pl.system.set_cross_core(pipe=P.MTE3, event_id=0, sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
                pl.system.bar_all()
                pl.system.wait_cross_core(pipe=P.V, event_id=4, sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
                o_state_diag(hresult_v, last, hsum)
                pl.system.bar_all()
                pl.load(gbuf, gv, [0, 0])
                pl.load(kbase, kv, [0, 0])
                for phase in pl.range(0, 10):
                    if phase != 5 and phase != 4:
                        pl.system.wait_cross_core(pipe=P.V, event_id=1, sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
                        if phase == 0:
                            o_dk0(result, gbuf, last, kbase, hsum, gdk, gdg)
                        elif phase == 1:
                            o_matrix(result, da_base, scale, 0)
                            pl.load(qbase, qv, [0, 0])
                        elif phase == 2:
                            o_dq0(result, ebuf, qbase, gdq, gdg, scale)
                        elif phase == 3:
                            pl.load(vwork, vv, [0, 0])
                            o_dv_high(result, vwork, beta, db_base)
                            pl.store(dvv, vwork, [0, 0])
                        elif phase == 6:
                            o_narrow64(result, dwpack, -1.0)
                            pl.insert(dwm[sub], dwpack, [0, 0])
                            pl.system.set_cross_core(pipe=P.MTE3, event_id=3,
                                                     sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
                        elif phase == 7:
                            o_strict_narrow(result, trilpack)
                            pl.insert(tmpm[sub], trilpack, [0, 0])
                            pl.system.set_cross_core(pipe=P.MTE3, event_id=3,
                                                     sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
                            pl.load(kbase, kv, [0, 0])
                        elif phase == 8:
                            o_dwk(result, ebuf, kbase, beta, gdk, gdg, db_base)
                        else:
                            o_matrix(result, dl_base, -1.0, 1)
                        pl.system.set_cross_core(pipe=P.V, event_id=2, sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
                pl.system.bar_all()
                pl.move(da_local, da_base)
                pl.move(dl_local, dl_base)
                pl.move(beta_local, beta)
                pl.move(db_local, db_base)
                pl.system.bar_all()
                pl.load(k_local, kv, [0, 0])
                if has_qk_rstd != 0:
                    norm_view = pl.make_tensor(pl.addptr(pl.make_ptr(qk_rstd), task * 128), [1, 128], [128, 1])
                    pl.load(norm_rs, norm_view, [0, 0])
                for band in pl.range(0, 2):
                    pl.system.bar_all()
                    first = band * 32
                    pl.load(gscratch0, gv, [first, 0])
                    pl.load(gscratch1, gv, [first + 31, 0])
                    o_pivot(gscratch0, gscratch1, pivot)
                    pl.load(gscratch, gv, [first, 0])
                    pl.load(q_local, qv, [first, 0])
                    ej_band = ej[first:first + 32, 0:128]
                    o_exp_pair(gscratch, pivot, ei, ej_band)
                    if band == 0:
                        ej_tail = ej[32:64, 0:128]
                        o_zero_exp_half(ej_tail)
                    else:
                        pl.load(gscratch, gv, [0, 0])
                        ej_prefix = ej[0:32, 0:128]
                        o_exp(gscratch, pivot, ej_prefix, 32, 0, 63, -1.0)
                    pl.system.bar_all()
                    # A is packed once into both normal and transposed L1 views.
                    for row in pl.range(0, 64, 16):
                        local_nz = local_nz_group.next()
                        local_lo = local_lo_group.next()
                        pl.set_validshape(local_nz, [16, 128])
                        pl.set_validshape(local_lo, [16, 128])
                        pl.set_validshape(local_nz, [16, 64])
                        pl.set_validshape(local_lo, [16, 64])
                        if row < 32:
                            o_a_pack(da_local, local_nz, local_lo, first + row)
                        else:
                            o_a_pack(dl_local, local_nz, local_lo, first + row - 32)
                        pl.insert(lah[sub], local_nz, [row, 0])
                        pl.insert(lath[sub], local_nz, [row, 0])
                        pl.insert(lal[sub], local_lo, [row, 0])
                        pl.insert(latl[sub], local_lo, [row, 0])
                    for row in pl.range(0, 64, 16):
                        local_nz = local_nz_group.next()
                        local_lo = local_lo_group.next()
                        pl.set_validshape(local_nz, [16, 128])
                        pl.set_validshape(local_lo, [16, 128])
                        kr = k_local[row:row + 16, 0:128]
                        er = ej[row:row + 16, 0:128]
                        o_b_pack(kr, er, local_nz, local_lo)
                        pl.insert(lbh[sub], local_nz, [row, 0])
                        pl.insert(lbl[sub], local_lo, [row, 0])
                    pl.system.set_cross_core(pipe=P.MTE3, event_id=3, sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
                    for row in pl.range(0, 64, 16):
                        local_nz = local_nz_group.next()
                        local_lo = local_lo_group.next()
                        pl.set_validshape(local_nz, [16, 128])
                        pl.set_validshape(local_lo, [16, 128])
                        if row < 32:
                            qr = q_local[row:row + 16, 0:128]
                            er2 = ei[row:row + 16, 0:128]
                            o_b_pack(qr, er2, local_nz, local_lo)
                        else:
                            kr2 = k_local[first + row - 32:first + row - 16, 0:128]
                            er3 = ei[row - 32:row - 16, 0:128]
                            o_beta_pack(kr2, er3, beta_local, first + row - 32, local_nz, local_lo)
                        pl.insert(rbh[sub], local_nz, [row, 0])
                        pl.insert(rbl[sub], local_lo, [row, 0])
                    pl.system.set_cross_core(pipe=P.MTE3, event_id=6, sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
                    pl.system.wait_cross_core(pipe=P.V, event_id=1, sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
                    qgrad = gdq[first:first + 32, 0:128]
                    kgrad = gdk[first:first + 32, 0:128]
                    ggrad = gdg[first:first + 32, 0:128]
                    bgrad = db_local[first:first + 32, 0:8]
                    kband = k_local[first:first + 32, 0:128]
                    qresult = local_result[0:32, 0:128]
                    kresult = local_result[32:64, 0:128]
                    s2_local_result(qresult, ei, q_local, qgrad, ggrad, 1.0, 32)
                    o_local_beta_result(kresult, ei, kband, beta_local, first, kgrad, ggrad, bgrad, 32)
                    o_diagonal(da_local, q_local, k_local, qgrad, kgrad, first)
                    pl.system.set_cross_core(pipe=P.V, event_id=2, sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
                    if has_qk_rstd != 0:
                        o_l2_cached_inplace(q_local, qgrad, norm_rs, 32, first)
                        pl.store(dqv, q_local, [first, 0])
                    pl.system.wait_cross_core(pipe=P.V, event_id=1, sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
                    s2_local_result(local_result, ej, k_local, gdk, gdg, -1.0, 64)
                    pl.system.set_cross_core(pipe=P.V, event_id=2, sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
                pl.system.bar_all()
                qrv = pl.make_tensor(pl.addptr(pl.make_ptr(q_raw), tok), [64, 128], [num_heads * 128, 1])
                krv = pl.make_tensor(pl.addptr(pl.make_ptr(k_raw), tok), [64, 128], [num_heads * 128, 1])
                grv = pl.make_tensor(pl.addptr(pl.make_ptr(g_raw), tok), [64, 128], [num_heads * 128, 1])
                if has_qk_rstd != 0:
                    o_l2_cached_inplace(k_local, gdk, norm_rs, 64, 64)
                    pl.store(dkv, k_local, [0, 0])
                else:
                    pl.load(raw_chain, qrv, [0, 0])
                    cp_l2_backward(raw_chain, gdq, chain_out, 64, 1)
                    pl.store(dqv, chain_out, [0, 0])
                    pl.load(raw_chain, krv, [0, 0])
                    cp_l2_backward(raw_chain, gdk, chain_out, 64, 1)
                    pl.store(dkv, chain_out, [0, 0])
                o_beta_chain(db_local, beta_local, beta_out)
                pl.set_validshape(beta_out, [64, 1])
                pl.store(dbv, beta_out, [0, 0])
                pl.set_validshape(beta_out, [64, 16])
                o_reverse_scan(gdg)
                pl.load(raw_chain, grv, [0, 0])
                pl.load(chain_bias, dt_bias, [n, 0])
                alogv = pl.make_tensor(a_log, [num_heads, 1], [1, 1])
                pl.load(chain_alog, alogv, [n, 0])
                for half in pl.range(0, 2):
                    rv = raw_chain[0:64, half * 64:half * 64 + 64]
                    gv2 = gdg[0:64, half * 64:half * 64 + 64]
                    bv2 = chain_bias[0:1, half * 64:half * 64 + 64]
                    ov2 = chain_out[0:64, half * 64:half * 64 + 64]
                    s2_zero_params(chain_sums)
                    s2_gate_backward(rv, gv2, bv2, chain_alog, ov2, chain_sums, 0, 64, lower_bound)
                    biaspart = chain_sums[0:1, 0:64]
                    apart = chain_sums[1:2, 0:1]
                    pl.store(dbias_out, biaspart, [n, half * 64], atomic=pl.AtomicType.AtomicAdd)
                    dav = pl.make_tensor(da_out, [num_heads, 1], [1, 1])
                    pl.store(dav, apart, [n, 0], atomic=pl.AtomicType.AtomicAdd)
                pl.store(dgv, chain_out, [0, 0])
                if c == 0:
                    ev = pl.make_tensor(pl.addptr(pl.make_ptr(all_summary),
                                        (b * num_heads + n) * 2 * 16384), [128, 128], [128, 1])
                    iv = pl.make_tensor(
                        pl.addptr(pl.make_ptr(initial_grad), (b * num_heads + n) * 16384), [128, 128], [128, 1]
                    )
                    for row in pl.range(0, 128, 64):
                        pl.load(gdq, ev, [row, 0])
                        pl.store(iv, gdq, [row, 0])
                pl.system.bar_all()


class _Stage2Tiling:
    LocalTile = TilingKeyField(bits=1, values=[16, 32])
    OwnedHeads = TilingKeyField(bits=1, values=[0, 1])


@pl.jit(tiling_key=_Stage2Tiling, auto_mutex=True, compile_timeout=280)
def chunk_kda_bwd_stage2_kernel(
    q_c: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    k_c: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    v_c: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    gk_c: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    beta_c: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    akk: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    h: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    v_new: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    do_: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    dq: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    dk: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    dv: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    db: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    dgi: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    workspace: pl.Tensor[[pl.DYNAMIC], pl.DT_UINT8],
    batch_size: pl.DT_INT64,
    total_seq_len: pl.DT_INT64,
    num_heads: pl.DT_INT64,
    total_num_chunks: pl.DT_INT64,
    total_num_chunk_states: pl.DT_INT64,
    scale: pl.DT_FP32,
    layout_code: pl.DT_INT64,
    single_token: pl.DT_INT64,
    q_raw: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    k_raw: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    fuse_preprocess: pl.DT_INT64,
    valid_s: pl.DT_INT64,
    g_raw: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    a_log: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32],
    dt_bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    da_out: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32],
    dbias_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    lower_bound: pl.DT_FP32,
    reduce_mode: pl.DT_INT64,
    mqk: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    w: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    dht: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    all_summary: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    recur_state: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    recur_du: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    initial_grad: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    has_do: pl.DT_INT64,
    has_dht: pl.DT_INT64,
    cp_rank: pl.DT_INT64,
    cp_size: pl.DT_INT64,
    first_continues: pl.DT_INT64,
    last_continues: pl.DT_INT64,
    packed: pl.DT_INT64,
    segments: pl.DT_INT64,
    cu: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64],
    coffs: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64],
    hoffs: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64],
    num_key_heads: pl.DT_INT64,
    qk_sum: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    st_do__bs: pl.DT_INT64,
    st_do__ts: pl.DT_INT64,
    st_do__hs: pl.DT_INT64,
    st_dht_ms: pl.DT_INT64,
    st_dht_hs: pl.DT_INT64,
    st_q_raw_bs: pl.DT_INT64,
    st_q_raw_ts: pl.DT_INT64,
    st_q_raw_hs: pl.DT_INT64,
    st_k_raw_bs: pl.DT_INT64,
    st_k_raw_ts: pl.DT_INT64,
    st_k_raw_hs: pl.DT_INT64,
    st_g_raw_bs: pl.DT_INT64,
    st_g_raw_ts: pl.DT_INT64,
    st_g_raw_hs: pl.DT_INT64,
    st_dq_bs: pl.DT_INT64,
    st_dq_ts: pl.DT_INT64,
    st_dq_hs: pl.DT_INT64,
    st_dk_bs: pl.DT_INT64,
    st_dk_ts: pl.DT_INT64,
    st_dk_hs: pl.DT_INT64,
    st_dv_bs: pl.DT_INT64,
    st_dv_ts: pl.DT_INT64,
    st_dv_hs: pl.DT_INT64,
    st_db_bs: pl.DT_INT64,
    st_db_ts: pl.DT_INT64,
    st_db_hs: pl.DT_INT64,
    st_dgi_bs: pl.DT_INT64,
    st_dgi_ts: pl.DT_INT64,
    st_dgi_hs: pl.DT_INT64,
    st_initial_grad_ms: pl.DT_INT64,
    st_initial_grad_hs: pl.DT_INT64,
    has_local_cache: pl.DT_INT64,
    cached_state: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    cached_du_bf16: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    du_cache_bf16: pl.DT_INT64,
    qk_rstd: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    has_qk_rstd: pl.DT_INT64,
):
    if OwnedHeads != 0:
        o_owned_stage2(
            q_c,
            k_c,
            v_c,
            gk_c,
            beta_c,
            akk,
            h,
            v_new,
            do_,
            dq,
            dk,
            dv,
            db,
            dgi,
            workspace,
            batch_size,
            total_seq_len,
            num_heads,
            total_num_chunks,
            total_num_chunk_states,
            scale,
            layout_code,
            single_token,
            q_raw,
            k_raw,
            fuse_preprocess,
            valid_s,
            g_raw,
            a_log,
            dt_bias,
            da_out,
            dbias_out,
            lower_bound,
            reduce_mode,
            mqk,
            w,
            dht,
            all_summary,
            recur_state,
            recur_du,
            initial_grad,
            has_do,
            has_dht,
            cp_rank,
            cp_size,
            first_continues,
            last_continues,
            packed,
            segments,
            cu,
            coffs,
            hoffs,
            num_key_heads,
            qk_sum,
            st_do__bs,
            st_do__ts,
            st_do__hs,
            st_dht_ms,
            st_dht_hs,
            st_q_raw_bs,
            st_q_raw_ts,
            st_q_raw_hs,
            st_k_raw_bs,
            st_k_raw_ts,
            st_k_raw_hs,
            st_g_raw_bs,
            st_g_raw_ts,
            st_g_raw_hs,
            st_dq_bs,
            st_dq_ts,
            st_dq_hs,
            st_dk_bs,
            st_dk_ts,
            st_dk_hs,
            st_dv_bs,
            st_dv_ts,
            st_dv_hs,
            st_db_bs,
            st_db_ts,
            st_db_hs,
            st_dgi_bs,
            st_dgi_ts,
            st_dgi_hs,
            st_initial_grad_ms,
            st_initial_grad_hs,
            has_local_cache,
            cached_state,
            cached_du_bf16,
            du_cache_bf16,
            qk_rstd,
            has_qk_rstd,
        )
    else:
        local_tile = LocalTile  # noqa: F821 - PyPTO compile-time tiling field
        local_rows = local_tile // 2
        local_blocks = 64 // local_tile
        ratio = num_heads // num_key_heads
        seq_len = total_seq_len
        num_chunks = total_num_chunks
        num_chunk_states = total_num_chunks + 1
        rec_num_cores = pl.get_block_num()
        rec_kg_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=[65536],
            mutex_ids=[16],
            depth=1,
        )
        rec_dh_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[128, 32], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=[81920],
            mutex_ids=[17],
            depth=1,
        )
        rec_qg_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[128, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
            addrs=[90112],
            mutex_ids=[18],
            depth=1,
        )
        rec_du_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[64, 32], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=[106496],
            mutex_ids=[19],
            depth=1,
        )
        rec_tail_do_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[64, 32], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=[110592],
            mutex_ids=[20],
            depth=1,
        )
        num_cores = pl.get_block_num()
        safe_a1_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[local_tile, local_tile], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=[65536, 67584],
            mutex_ids=[9, 15],
            depth=2,
        )
        safe_b1_group = pl.make_tile_group(
            type=pl.TileType(shape=[local_tile, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat),
            addrs=[69632, 77824],
            mutex_ids=[10, 16],
            depth=2,
        )
        safe_at1_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[local_tile, local_tile], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN
            ),
            addrs=[86016, 88064],
            mutex_ids=[19, 20],
            depth=2,
        )
        base_a_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=[83968],
            mutex_ids=[23],
            depth=1,
        )
        base_bt_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN
            ),
            addrs=[100352],
            mutex_ids=[24],
            depth=1,
        )
        base_btn64_group = pl.make_tile_group(
            type=pl.TileType(shape=[128, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
            addrs=[133120],
            mutex_ids=[25],
            depth=1,
        )
        base_bn_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ
            ),
            addrs=[149504],
            mutex_ids=[26],
            depth=1,
        )
        cp_r_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ
            ),
            addrs=[196608],
            mutex_ids=[28],
            depth=1,
        )
        with pl.section_cube():
            rec_result_ub_native = pl.make_tile_group(
                type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[229376],
                mutex_ids=[0],
                depth=1,
            )
            rec_result_ub = rec_result_ub_native[0]
            rec_al1_64_64_native = pl.make_tile_group(type=pl.TileType(
                shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat), addrs=[0], mutex_ids=[1], depth=1)
            rec_al1_64_64 = rec_al1_64_64_native[0]
            rec_al0_64_64_native = pl.make_tile_group(type=pl.TileType(
                shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left), addrs=[0], mutex_ids=[2], depth=1)
            rec_al0_64_64 = rec_al0_64_64_native[0]
            rec_al0_64_128_native = pl.make_tile_group(
                type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left),
                addrs=[8192],
                mutex_ids=[4],
                depth=1,
            )
            rec_al0_64_128 = rec_al0_64_128_native[0]
            rec_al1_128_64_native = pl.make_tile_group(
                type=pl.TileType(shape=[128, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat),
                addrs=[24576],
                mutex_ids=[5],
                depth=1,
            )
            rec_al1_128_64 = rec_al1_128_64_native[0]
            rec_al0_128_64_native = pl.make_tile_group(
                type=pl.TileType(shape=[128, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left),
                addrs=[24576],
                mutex_ids=[6],
                depth=1,
            )
            rec_al0_128_64 = rec_al0_128_64_native[0]
            rec_bl1_64_32_native = pl.make_tile_group(
                type=pl.TileType(shape=[64, 32], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat),
                addrs=[40960],
                mutex_ids=[7],
                depth=1,
            )
            rec_bl1_64_32 = rec_bl1_64_32_native[0]
            rec_bl0_32_64_native = pl.make_tile_group(
                type=pl.TileType(shape=[64, 32], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right),
                addrs=[0],
                mutex_ids=[8],
                depth=1,
            )
            rec_bl0_32_64 = rec_bl0_32_64_native[0]
            rec_bl0_32_128_native = pl.make_tile_group(
                type=pl.TileType(shape=[128, 32], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right),
                addrs=[4096],
                mutex_ids=[10],
                depth=1,
            )
            rec_bl0_32_128 = rec_bl0_32_128_native[0]
            rec_acc_64_32_native = pl.make_tile_group(type=pl.TileType(
                shape=[64, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc), addrs=[0], mutex_ids=[11], depth=1)
            rec_acc_64_32 = rec_acc_64_32_native[0]
            rec_acc_128_32_native = pl.make_tile_group(
                type=pl.TileType(shape=[128, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
                addrs=[8192],
                mutex_ids=[12],
                depth=1,
            )
            rec_acc_128_32 = rec_acc_128_32_native[0]
            rec_sub = 1 - layout_code
            rec_seq_stride = num_heads * 128 * rec_sub + 128 * layout_code
            rec_head_stride = 128 * rec_sub + seq_len * 128 * layout_code
            rec_core = pl.get_block_idx()
            rec_mul_9 = batch_size * num_heads * 4
            result_ub_native = pl.make_tile_group(
                type=pl.TileType(shape=[32, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[229376],
                mutex_ids=[0],
                depth=1,
            )
            result_ub = result_ub_native[0]
            al1_native = pl.make_tile_group(type=pl.TileType(
                shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat), addrs=[0], mutex_ids=[1], depth=1)
            al1 = al1_native[0]
            al0_native = pl.make_tile_group(
                type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left),
                addrs=[0],
                mutex_ids=[3],
                depth=1,
            )
            al0 = al0_native[0]
            bl0_native = pl.make_tile_group(
                type=pl.TileType(shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right),
                addrs=[0],
                mutex_ids=[4],
                depth=1,
            )
            bl0 = bl0_native[0]
            acc_native = pl.make_tile_group(type=pl.TileType(
                shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc), addrs=[0], mutex_ids=[5], depth=1)
            acc = acc_native[0]
            bl1_n64_native = pl.make_tile_group(
                type=pl.TileType(shape=[128, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat),
                addrs=[49152],
                mutex_ids=[6],
                depth=1,
            )
            bl1_n64 = bl1_n64_native[0]
            bl0_n64_native = pl.make_tile_group(
                type=pl.TileType(shape=[128, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right),
                addrs=[32768],
                mutex_ids=[7],
                depth=1,
            )
            bl0_n64 = bl0_n64_native[0]
            acc_n64_native = pl.make_tile_group(
                type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
                addrs=[32768],
                mutex_ids=[8],
                depth=1,
            )
            acc_n64 = acc_n64_native[0]
            safe_a0_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[local_tile, local_tile], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left
                ),
                addrs=[16384, 18432],
                mutex_ids=[11, 17],
                depth=2,
            )
            safe_b0_group = pl.make_tile_group(
                type=pl.TileType(shape=[local_tile, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right),
                addrs=[49152, 57344],
                mutex_ids=[12, 18],
                depth=2,
            )
            safe_acc_group = pl.make_tile_group(
                type=pl.TileType(shape=[local_tile, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
                addrs=[49152],
                mutex_ids=[13],
                depth=1,
            )
            safe_acc = safe_acc_group[0]
            safe_result_group = pl.make_tile_group(
                type=pl.TileType(shape=[local_rows, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[131072],
                mutex_ids=[14],
                depth=1,
            )
            safe_result = safe_result_group[0]

            t_l1_group = pl.make_tile_group(
                type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat),
                addrs=[75776],
                mutex_ids=[21],
                depth=1,
            )
            t_l1 = t_l1_group[0]
            t_l0_group = pl.make_tile_group(
                type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right),
                addrs=[57344],
                mutex_ids=[22],
                depth=1,
            )
            t_l0 = t_l0_group[0]
            sub = 1 - layout_code
            seq_stride = num_heads * 128 * sub + 128 * layout_code
            head_stride = 128 * sub + seq_len * 128 * layout_code
            core = pl.get_block_idx()
            mul_9 = batch_size * num_chunks * num_heads
            cp_a0 = pl.make_tile_group(
                type=pl.TileType(shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left),
                addrs=[32768],
                mutex_ids=[29],
                depth=1,
            )
            for job in pl.range(core, segments * num_key_heads, num_cores):
                b = job // num_key_heads
                n = job % num_key_heads * ratio
                cp_seg = b
                cp_head = n
                cp_key = cp_head // ratio
                seq_len = total_seq_len
                num_chunks = total_num_chunks
                cp_cbase = cp_seg * total_num_chunks
                cp_hbase = cp_seg * (total_num_chunks + 1)
                cp_tok = 0
                cp_batch = cp_seg
                if packed:
                    seq_len = cu[cp_seg + 1] - cu[cp_seg]
                    num_chunks = coffs[cp_seg + 1] - coffs[cp_seg]
                    cp_cbase = coffs[cp_seg]
                    cp_hbase = hoffs[cp_seg]
                    cp_tok = cu[cp_seg]
                    cp_batch = 0
                valid_s = seq_len
                single_token = seq_len == 1
                rec_b = b
                rec_n = n
                if has_local_cache == 0:
                    for cp_member in pl.range(0, ratio):
                        cp_head = cp_key * ratio + cp_member
                        n = cp_head
                        rec_n = cp_head
                        for _ in pl.range(0, 4):
                            if last_continues != 0 and (packed == 0 or cp_seg == segments - 1):
                                for _ in pl.range(0, cp_size - cp_rank - 1):
                                    pl.system.wait_cross_core(pipe=P.MTE1, event_id=0,
                                                              sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                    pl.move(cp_a0[0], cp_r_l1[0])
                                    pl.move(rec_bl0_32_128, rec_dh_l1[0])
                                    pl.matmul(rec_acc_128_32, cp_a0[0], rec_bl0_32_128)
                                    view_cp = rec_result_ub[0:64, 0:32]
                                    pl.move(view_cp, rec_acc_128_32, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                                    pl.system.set_cross_core(pipe=P.FIX, event_id=1,
                                                             sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                    pl.system.wait_cross_core(pipe=P.FIX, event_id=2,
                                                              sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.bar_all()
                        pl.system.set_cross_core(pipe=P.FIX, event_id=7, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.MTE1, event_id=6, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                for it in pl.range(0, num_chunks):
                    t = num_chunks - 1 - it
                    cp_t = t
                    cp_chunk = cp_cbase + cp_t
                    rec_t = t
                    cp_t = rec_t
                    cp_chunk = cp_cbase + cp_t
                    for cp_member in pl.range(0, ratio):
                        cp_head = cp_key * ratio + cp_member
                        n = cp_head
                        rec_n = cp_head
                        if has_local_cache == 0:
                            for vs in pl.range(0, 4):
                                rec_v0 = vs * 32
                                pl.system.wait_cross_core(pipe=P.MTE1, event_id=4,
                                                          sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                rec_gm_window_27 = pl.make_tensor(
                                    pl.addptr(
                                        pl.make_ptr(mqk),
                                        (
                                            (cp_batch * num_heads + cp_head) * total_num_chunks
                                            + cp_t
                                            + packed * cp_cbase
                                        )
                                        * 4096,
                                    ),
                                    [64, 64],
                                    [64, 1],
                                )
                                pl.load(rec_al1_64_64, rec_gm_window_27, [0, 0], order=[1, 0])
                                rec_mul_80 = rec_t * 64
                                rec_cmp_6 = seq_len - rec_mul_80 >= 64 and has_do != 0
                                if rec_cmp_6:
                                    rec_gm_direct_do_1 = pl.make_tensor(
                                        pl.addptr(
                                            pl.make_ptr(do_),
                                            cp_batch * st_do__bs + cp_tok * st_do__ts + cp_head * st_do__hs,
                                        ),
                                        [seq_len, 128],
                                        [st_do__ts, 1],
                                    )
                                    pl.load(rec_bl1_64_32, rec_gm_direct_do_1, [rec_mul_80, rec_v0])
                                pl.set_validshape(rec_al0_64_64, [64, 64])
                                pl.move(rec_al0_64_64, rec_al1_64_64)
                                pl.set_validshape(rec_al0_64_64, [64, 64])
                                pl.set_validshape(rec_bl0_32_64, [64, 32])
                                if rec_cmp_6:
                                    pl.move(rec_bl0_32_64, rec_bl1_64_32)
                                else:
                                    pl.move(rec_bl0_32_64, rec_tail_do_l1[0])
                                pl.set_validshape(rec_bl0_32_64, [64, 32])
                                pl.matmul(rec_acc_64_32, rec_al0_64_64, rec_bl0_32_64)
                                pl.system.wait_cross_core(pipe=P.MTE1, event_id=0,
                                                          sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                pl.set_validshape(rec_al0_64_128, [64, 128])
                                pl.move(rec_al0_64_128, rec_kg_l1[0])
                                pl.set_validshape(rec_al0_64_128, [64, 128])
                                pl.set_validshape(rec_bl0_32_128, [128, 32])
                                pl.move(rec_bl0_32_128, rec_dh_l1[0])
                                pl.set_validshape(rec_bl0_32_128, [128, 32])
                                pl.matmul_acc(rec_acc_64_32, rec_acc_64_32, rec_al0_64_128, rec_bl0_32_128)
                                rec_view_23_fix_view = rec_result_ub[0:0 + 32, 0:0 + 32]
                                pl.move(rec_view_23_fix_view, rec_acc_64_32,
                                        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                                pl.system.set_cross_core(pipe=P.FIX, event_id=1,
                                                         sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                pl.system.wait_cross_core(pipe=P.FIX, event_id=2,
                                                          sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                pl.system.wait_cross_core(pipe=P.MTE1, event_id=0,
                                                          sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                pl.set_validshape(rec_al0_128_64, [128, 64])
                                pl.move(rec_al0_128_64, rec_qg_l1[0])
                                pl.set_validshape(rec_al0_128_64, [128, 64])
                                pl.matmul(rec_acc_128_32, rec_al0_128_64, rec_bl0_32_64)
                                rec_view_33_fix_view = rec_result_ub[0:0 + 64, 0:0 + 32]
                                pl.move(rec_view_33_fix_view, rec_acc_128_32,
                                        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                                pl.system.set_cross_core(pipe=P.FIX, event_id=1,
                                                         sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                pl.system.wait_cross_core(pipe=P.FIX, event_id=2,
                                                          sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                pl.system.wait_cross_core(pipe=P.MTE1, event_id=0,
                                                          sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                rec_gm_window_69 = pl.make_tensor(
                                    pl.addptr(
                                        pl.make_ptr(w),
                                        cp_batch * (total_num_chunks * 64) * num_heads * 128
                                        + packed * cp_cbase * 64 * num_heads * 128
                                        + cp_head
                                        * (128 * (1 - layout_code) + total_num_chunks * 64 * 128 * layout_code),
                                    ),
                                    [total_num_chunks * 64, 128],
                                    [rec_seq_stride, 1],
                                )
                                pl.load(rec_al1_128_64, rec_gm_window_69, [rec_mul_80, 0], order=[1, 0])
                                pl.set_validshape(rec_al0_128_64, [128, 64])
                                pl.move(rec_al0_128_64, rec_al1_128_64)
                                pl.set_validshape(rec_al0_128_64, [128, 64])
                                pl.set_validshape(rec_bl0_32_64, [64, 32])
                                pl.move(rec_bl0_32_64, rec_du_l1[0])
                                pl.set_validshape(rec_bl0_32_64, [64, 32])
                                pl.matmul(rec_acc_128_32, rec_al0_128_64, rec_bl0_32_64)
                                rec_view_42_fix_view = rec_result_ub[0:0 + 64, 0:0 + 32]
                                pl.move(rec_view_42_fix_view, rec_acc_128_32,
                                        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                                pl.system.set_cross_core(pipe=P.FIX, event_id=1,
                                                         sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                pl.system.wait_cross_core(pipe=P.FIX, event_id=2,
                                                          sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                            pl.system.bar_all()
                            pl.system.set_cross_core(pipe=P.FIX, event_id=7, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                            pl.system.wait_cross_core(pipe=P.MTE1, event_id=6,
                                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        live = 64
                        mul_11 = t * 64
                        chunk_index = (b * num_chunks + t) * num_heads + n
                        direct_akk = pl.make_tensor(pl.addptr(pl.make_ptr(
                            akk), (cp_chunk * num_heads + cp_head) * 4096), [64, 64], [64, 1])
                        direct_vnew = pl.make_tensor(pl.addptr(pl.make_ptr(
                            v_new), (cp_chunk * num_heads + cp_head) * 8192), [128, 64], [64, 1])
                        sub_5 = seq_len - mul_11
                        if sub_5 < 64:
                            live = sub_5
                        pl.system.wait_cross_core(pipe=P.MTE1, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.load(al1, direct_vnew, [0, 0], order=[1, 0])
                        pl.set_validshape(al0, [64, 128])
                        pl.move(al0, al1)
                        pl.set_validshape(al0, [64, 128])
                        pl.set_validshape(bl0, [128, 128])
                        pl.move(bl0, base_bt_group[0])
                        pl.set_validshape(bl0, [128, 128])
                        pl.matmul(acc, al0, bl0)
                        pl.move(result_ub, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                        pl.system.set_cross_core(pipe=P.FIX, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.FIX, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.MTE1, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        cmp_5 = live == 64 and has_do != 0
                        if cmp_5:
                            do_direct_mm2 = pl.make_tensor(
                                pl.addptr(
                                    pl.make_ptr(do_),
                                    cp_batch * st_do__bs + cp_tok * st_do__ts + cp_head * st_do__hs,
                                ),
                                [seq_len, 128],
                                [st_do__ts, 1],
                            )
                            pl.load(al1, do_direct_mm2, [mul_11, 0])
                        pl.load(bl1_n64, direct_vnew, [0, 0])
                        pl.set_validshape(al0, [64, 128])
                        if cmp_5:
                            pl.move(al0, al1)
                        else:
                            pl.move(al0, base_a_group[0])
                        pl.set_validshape(al0, [64, 128])
                        pl.set_validshape(bl0_n64, [128, 64])
                        pl.move(bl0_n64, bl1_n64)
                        pl.set_validshape(bl0_n64, [128, 64])
                        pl.matmul(acc_n64, al0, bl0_n64)
                        view_54_fix_view = result_ub[0:0 + 32, 0:0 + 64]
                        pl.move(view_54_fix_view, acc_n64, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                        pl.system.set_cross_core(pipe=P.FIX, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.FIX, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.MTE1, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.set_validshape(al0, [64, 128])
                        if cmp_5:
                            pl.move(al0, al1)
                        else:
                            pl.move(al0, base_a_group[0])
                        pl.set_validshape(al0, [64, 128])
                        pl.set_validshape(bl0, [128, 128])
                        pl.move(bl0, base_bt_group[0])
                        pl.set_validshape(bl0, [128, 128])
                        pl.matmul(acc, al0, bl0)
                        pl.move(result_ub, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                        pl.system.set_cross_core(pipe=P.FIX, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.FIX, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.MTE1, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.set_validshape(al1, [64, 64])
                        pl.load(al1, direct_akk, [0, 0], order=[1, 0])
                        pl.set_validshape(al1, [64, 128])
                        pl.set_validshape(al0, [64, 128])
                        pl.move(al0, al1)
                        pl.set_validshape(al0, [64, 128])
                        pl.set_validshape(bl0, [128, 128])
                        pl.move(bl0, base_bn_group[0])
                        pl.set_validshape(bl0, [128, 128])
                        pl.set_validshape(al0, [64, 64])
                        pl.set_validshape(bl0, [64, 128])
                        pl.matmul(acc, al0, bl0)
                        pl.set_validshape(al0, [64, 128])
                        pl.set_validshape(bl0, [128, 128])
                        pl.move(result_ub, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                        pl.system.set_cross_core(pipe=P.FIX, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.FIX, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.MTE1, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.set_validshape(al1, [64, 64])
                        pl.load(al1, direct_akk, [0, 0], order=[1, 0])
                        pl.set_validshape(al1, [64, 128])
                        pl.set_validshape(al0, [64, 128])
                        pl.move(al0, al1)
                        pl.set_validshape(al0, [64, 128])
                        pl.set_validshape(bl0, [128, 128])
                        pl.move(bl0, base_bn_group[0])
                        pl.set_validshape(bl0, [128, 128])
                        pl.set_validshape(al0, [64, 64])
                        pl.set_validshape(bl0, [64, 128])
                        pl.matmul(acc, al0, bl0)
                        pl.set_validshape(al0, [64, 128])
                        pl.set_validshape(bl0, [128, 128])
                        pl.move(result_ub, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                        pl.system.set_cross_core(pipe=P.FIX, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.FIX, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.MTE1, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.set_validshape(al0, [64, 128])
                        pl.move(al0, base_a_group[0])
                        pl.set_validshape(al0, [64, 128])
                        pl.set_validshape(bl0_n64, [128, 64])
                        pl.move(bl0_n64, base_btn64_group[0])
                        pl.set_validshape(bl0_n64, [128, 64])
                        pl.matmul(acc_n64, al0, bl0_n64)
                        pl.system.set_cross_core(pipe=P.M, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.MTE1, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.set_validshape(al0, [64, 128])
                        pl.move(al0, base_a_group[0])
                        pl.set_validshape(al0, [64, 128])
                        pl.set_validshape(bl0, [128, 128])
                        pl.move(bl0, base_bt_group[0])
                        pl.set_validshape(bl0, [128, 128])
                        pl.matmul(acc, al0, bl0)
                        pl.move(result_ub, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                        pl.system.set_cross_core(pipe=P.FIX, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.FIX, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.MTE1, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.set_validshape(al0, [64, 128])
                        pl.move(al0, base_a_group[0])
                        pl.set_validshape(al0, [64, 128])
                        pl.set_validshape(bl0_n64, [128, 64])
                        pl.move(bl0_n64, base_btn64_group[0])
                        pl.set_validshape(bl0_n64, [128, 64])
                        pl.matmul_acc(acc_n64, acc_n64, al0, bl0_n64)
                        view_127_fix_view = result_ub[0:0 + 32, 0:0 + 64]
                        pl.move(view_127_fix_view, acc_n64, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                        pl.system.set_cross_core(pipe=P.FIX, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.FIX, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.MTE1, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.set_validshape(al1, [64, 64])
                        pl.load(al1, direct_akk, [0, 0], order=[1, 0])
                        pl.set_validshape(al1, [64, 128])
                        pl.set_validshape(al0, [64, 128])
                        pl.move(al0, al1)
                        pl.set_validshape(al0, [64, 128])
                        pl.set_validshape(bl0, [128, 128])
                        pl.move(bl0, base_bn_group[0])
                        pl.set_validshape(bl0, [128, 128])
                        pl.set_validshape(al0, [64, 64])
                        pl.set_validshape(bl0, [64, 128])
                        pl.matmul(acc, al0, bl0)
                        pl.set_validshape(al0, [64, 128])
                        pl.set_validshape(bl0, [128, 128])
                        pl.move(result_ub, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                        pl.system.set_cross_core(pipe=P.FIX, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.FIX, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.MTE1, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.set_validshape(al1, [64, 64])
                        pl.set_validshape(al1, [64, 128])
                        pl.set_validshape(bl1_n64, [64, 64])
                        pl.load(bl1_n64, direct_akk, [0, 0], order=[1, 0])
                        pl.set_validshape(bl1_n64, [128, 64])
                        pl.set_validshape(al0, [64, 128])
                        pl.move(al0, base_a_group[0])
                        pl.set_validshape(al0, [64, 128])
                        pl.set_validshape(bl0_n64, [128, 64])
                        pl.move(bl0_n64, bl1_n64)
                        pl.set_validshape(bl0_n64, [128, 64])
                        pl.set_validshape(al0, [64, 64])
                        pl.set_validshape(bl0_n64, [64, 64])
                        pl.matmul(acc_n64, al0, bl0_n64)
                        pl.set_validshape(al0, [64, 128])
                        pl.set_validshape(bl0_n64, [128, 64])
                        pl.move(t_l1, acc_n64)
                        pl.set_validshape(al1, [64, 64])
                        pl.load(al1, direct_akk, [0, 0], order=[1, 0])
                        pl.set_validshape(al1, [64, 128])
                        pl.set_validshape(al0, [64, 128])
                        pl.move(al0, al1)
                        pl.set_validshape(al0, [64, 128])
                        pl.set_validshape(bl0_n64, [128, 64])
                        pl.set_validshape(bl0_n64, [64, 64])
                        pl.move(t_l0, t_l1)
                        pl.set_validshape(bl0_n64, [128, 64])
                        pl.set_validshape(al0, [64, 64])
                        pl.set_validshape(bl0_n64, [64, 64])
                        pl.matmul(acc_n64, al0, t_l0)
                        pl.set_validshape(al0, [64, 128])
                        pl.set_validshape(bl0_n64, [128, 64])
                        view_163_fix_view = result_ub[0:0 + 32, 0:0 + 64]
                        pl.move(view_163_fix_view, acc_n64, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                        pl.system.set_cross_core(pipe=P.FIX, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.FIX, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        for si in pl.range(0, local_blocks):
                            for _ in pl.range(0, si + 1):
                                for op in pl.range(0, 4):
                                    pl.system.wait_cross_core(pipe=P.MTE1, event_id=4,
                                                              sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                    safe_a0_hi = safe_a0_group[0]
                                    safe_a0_lo = safe_a0_group[1]
                                    safe_b0_hi = safe_b0_group[0]
                                    safe_b0_lo = safe_b0_group[1]
                                    if op < 2:
                                        pl.move(safe_a0_hi, safe_a1_group[0])
                                        pl.move(safe_a0_lo, safe_a1_group[1])
                                    else:
                                        pl.move(safe_a0_hi, safe_at1_group[0])
                                        pl.move(safe_a0_lo, safe_at1_group[1])
                                    pl.move(safe_b0_hi, safe_b1_group[0])
                                    pl.move(safe_b0_lo, safe_b1_group[1])
                                    pl.matmul(safe_acc, safe_a0_hi, safe_b0_hi)
                                    pl.matmul_acc(safe_acc, safe_acc, safe_a0_lo, safe_b0_hi)
                                    pl.matmul_acc(safe_acc, safe_acc, safe_a0_hi, safe_b0_lo)
                                    pl.move(safe_result, safe_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                                    pl.system.set_cross_core(pipe=P.FIX, event_id=1,
                                                             sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                    pl.system.wait_cross_core(pipe=P.FIX, event_id=6,
                                                              sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.bar_all()
                        pl.system.set_cross_core(pipe=P.FIX, event_id=7, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.MTE1, event_id=6, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        with pl.section_vector():
            rec_u0_native = pl.make_tile_group(
                type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[0],
                mutex_ids=[0],
                depth=1,
            )
            rec_u0 = rec_u0_native[0]
            rec_u1_native = pl.make_tile_group(
                type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[65536],
                mutex_ids=[1],
                depth=1,
            )
            rec_u1 = rec_u1_native[0]
            rec_u2_native = pl.make_tile_group(
                type=pl.TileType(shape=[32, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[131072],
                mutex_ids=[2],
                depth=1,
            )
            rec_u2 = rec_u2_native[0]
            rec_state_native = pl.make_tile_group(
                type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[163840],
                mutex_ids=[6],
                depth=1,
            )
            rec_state = rec_state_native[0]
            rec_gate_last_native = pl.make_tile_group(
                type=pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[253952],
                mutex_ids=[7],
                depth=1,
            )
            rec_gate_last = rec_gate_last_native[0]
            rec_pack_nd_native = pl.make_tile_group(
                type=pl.TileType(shape=[16, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
                addrs=[147456],
                mutex_ids=[8],
                depth=1,
            )
            rec_pack_nz_native = pl.make_tile_group(
                type=pl.TileType(
                    shape=[16, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ
                ),
                addrs=[151552],
                mutex_ids=[9],
                depth=1,
            )
            rec_pack_nd = rec_pack_nd_native[0]
            rec_pack_nz = rec_pack_nz_native[0]
            rec_ub16_native = pl.make_tile_group(
                type=pl.TileType(shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
                addrs=[196608],
                mutex_ids=[3],
                depth=1,
            )
            rec_ub16 = rec_ub16_native[0]
            rec_result_ub_1_native = pl.make_tile_group(
                type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[229376],
                mutex_ids=[4],
                depth=1,
            )
            rec_result_ub_1 = rec_result_ub_1_native[0]
            rec_keep_ub_native = pl.make_tile_group(
                type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[245760],
                mutex_ids=[5],
                depth=1,
            )
            rec_keep_ub = rec_keep_ub_native[0]
            rec_sub_1 = 1 - layout_code
            rec_seq_stride_1 = num_heads * 128 * rec_sub_1 + 128 * layout_code
            rec_head_stride_1 = 128 * rec_sub_1 + seq_len * 128 * layout_code
            rec_core_1 = pl.get_block_idx() // pl.get_subblock_num()
            rec_sub_4 = pl.get_subblock_idx()
            u0_native = pl.make_tile_group(
                type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[0],
                mutex_ids=[0],
                depth=1,
            )
            u0 = u0_native[0]
            u1_native = pl.make_tile_group(
                type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[65536],
                mutex_ids=[1],
                depth=1,
            )
            u1 = u1_native[0]
            u2_native = pl.make_tile_group(
                type=pl.TileType(shape=[32, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[131072],
                mutex_ids=[2],
                depth=1,
            )
            u2 = u2_native[0]
            expg_group = pl.make_tile_group(
                type=pl.TileType(shape=[32, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[147456],
                mutex_ids=[28],
                depth=1,
            )
            kbeta_group = pl.make_tile_group(
                type=pl.TileType(shape=[32, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[163840],
                mutex_ids=[29],
                depth=1,
            )
            vbeta_group = pl.make_tile_group(
                type=pl.TileType(shape=[32, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[180224],
                mutex_ids=[30],
                depth=1,
            )
            expg_ub = expg_group[0]
            kbeta_ub = kbeta_group[0]
            vbeta_ub = vbeta_group[0]
            state_sum_group = pl.make_tile_group(
                type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[260608],
                mutex_ids=[31],
                depth=1,
            )
            state_sum_ub = state_sum_group[0]
            ub16_native = pl.make_tile_group(
                type=pl.TileType(shape=[128, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
                addrs=[196608],
                mutex_ids=[3],
                depth=1,
            )
            ub16 = ub16_native[0]
            result_ub_1_native = pl.make_tile_group(
                type=pl.TileType(shape=[32, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[229376],
                mutex_ids=[4],
                depth=1,
            )
            result_ub_1 = result_ub_1_native[0]
            safe_result_v = u2[0:local_rows, 0:128]
            safe_diag_raw_group = pl.make_tile_group(
                type=pl.TileType(shape=[16, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
                addrs=[245760],
                mutex_ids=[6],
                depth=1,
            )
            safe_diag_raw = safe_diag_raw_group[0]
            safe_pack_nd_group = pl.make_tile_group(
                type=pl.TileType(shape=[16, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
                addrs=[251904],
                mutex_ids=[7],
                depth=1,
            )
            safe_pack_nz_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[16, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ
                ),
                addrs=[256000],
                mutex_ids=[8],
                depth=1,
            )
            safe_pack_nd = safe_pack_nd_group[0]
            safe_pack_nz = safe_pack_nz_group[0]
            beta_ub_group = pl.make_tile_group(
                type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[260096],
                mutex_ids=[27],
                depth=1,
            )
            beta_ub = beta_ub_group[0]
            param_group = pl.make_tile_group(
                type=pl.TileType(shape=[2, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[260864],
                mutex_ids=[9],
                depth=1,
            )
            param_sum = param_group[0]
            work32 = pl.make_tensor(workspace, [num_cores, 6, 128, 128], [
                                    6 * (128 * 128), 128 * 128, 128, 1], dtype=pl.DT_FP32)
            sub_1 = 1 - layout_code
            seq_stride_1 = num_heads * 128 * sub_1 + 128 * layout_code
            head_stride_1 = 128 * sub_1 + seq_len * 128 * layout_code
            beta_seq_stride = num_heads * sub_1 + layout_code
            beta_head_stride = sub_1 + seq_len * layout_code
            core_1 = pl.get_block_idx() // pl.get_subblock_num()
            sub_4 = pl.get_subblock_idx()
            if fuse_preprocess:
                s2_zero_params(param_sum)
            da_view = pl.make_tensor(da_out, [num_heads, 1], [1, 1])
            mul_9_1 = batch_size * num_chunks * num_heads
            for job_1 in pl.range(core_1, segments * num_key_heads, num_cores):
                b_1 = job_1 // num_key_heads
                n_1 = job_1 % num_key_heads * ratio
                cp_seg = b_1
                cp_head = n_1
                cp_key = cp_head // ratio
                seq_len = total_seq_len
                num_chunks = total_num_chunks
                cp_cbase = cp_seg * total_num_chunks
                cp_hbase = cp_seg * (total_num_chunks + 1)
                cp_tok = 0
                cp_batch = cp_seg
                if packed:
                    seq_len = cu[cp_seg + 1] - cu[cp_seg]
                    num_chunks = coffs[cp_seg + 1] - coffs[cp_seg]
                    cp_cbase = coffs[cp_seg]
                    cp_hbase = hoffs[cp_seg]
                    cp_tok = cu[cp_seg]
                    cp_batch = 0
                valid_s = seq_len
                single_token = seq_len == 1
                rec_b_1 = b_1
                rec_n_1 = n_1
                mul_10 = num_chunks * num_heads
                if has_local_cache == 0:
                    for cp_member in pl.range(0, ratio):
                        cp_head = cp_key * ratio + cp_member
                        n_1 = cp_head
                        rec_n_1 = cp_head
                        for vs in pl.range(0, 4):
                            rec_v0_1 = vs * 32
                            s1_zero_vslice(rec_state)
                            if last_continues != 0 and (packed == 0 or cp_seg == segments - 1):
                                for step in pl.range(0, cp_size - cp_rank - 1):
                                    other = cp_size - 1 - step
                                    cp_r = pl.make_tensor(
                                        pl.addptr(
                                            pl.make_ptr(all_summary),
                                            (((other * batch_size + cp_batch) * num_heads + cp_head) * 2 + 1)
                                            * 16384,
                                        ),
                                        [128, 128],
                                        [128, 1],
                                    )
                                    pl.set_validshape(rec_u1, [128, 64])
                                    pl.load(rec_u1, cp_r, [0, rec_sub_4 * 64])
                                    pl.set_validshape(rec_u1, [128, 128])
                                    s2_transposed(rec_u1, rec_u0)
                                    s2_narrow_1(rec_u0, rec_ub16)
                                    pl.set_validshape(rec_ub16, [64, 128])
                                    pl.set_validshape(rec_pack_nd, [16, 128])
                                    pl.set_validshape(rec_pack_nz, [16, 128])
                                    for row in pl.range(0, 64, 16):
                                        pl.move(rec_pack_nd, rec_ub16, offset=[row, 0])
                                        pl.move(rec_pack_nz, rec_pack_nd)
                                        pl.insert(cp_r_l1[0], rec_pack_nz, [rec_sub_4 * 64 + row, 0])
                                    pl.set_validshape(rec_ub16, [128, 128])
                                    s1_narrow_2(rec_state, rec_ub16)
                                    pl.set_validshape(rec_ub16, [64, 32])
                                    pl.set_validshape(rec_pack_nd, [16, 32])
                                    pl.set_validshape(rec_pack_nz, [16, 32])
                                    for row in pl.range(0, 64, 16):
                                        pl.move(rec_pack_nd, rec_ub16, offset=[row, 0])
                                        pl.move(rec_pack_nz, rec_pack_nd)
                                        pl.insert(rec_dh_l1[0], rec_pack_nz, [rec_sub_4 * 64 + row, 0])
                                    pl.set_validshape(rec_ub16, [128, 128])
                                    pl.system.set_cross_core(pipe=P.MTE3, event_id=3,
                                                             sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                                    pl.system.wait_cross_core(pipe=P.MTE3, event_id=3,
                                                              sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                                    pl.system.set_cross_core(pipe=P.MTE3, event_id=0,
                                                             sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                    cp_e = pl.make_tensor(
                                        pl.addptr(
                                            pl.make_ptr(all_summary),
                                            ((other * batch_size + cp_batch) * num_heads + cp_head) * 2 * 16384,
                                        ),
                                        [128, 128],
                                        [128, 1],
                                    )
                                    pl.set_validshape(rec_u1, [32, 64])
                                    pl.load(rec_u1, cp_e, [rec_v0_1, rec_sub_4 * 64])
                                    pl.set_validshape(rec_u1, [128, 128])
                                    s1_transposed(rec_u1, rec_state)
                                    pl.system.wait_cross_core(pipe=P.V, event_id=1,
                                                              sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                    s1_binary_rhs64(rec_state, rec_result_ub_1, rec_state)
                                    pl.system.set_cross_core(pipe=P.V, event_id=2,
                                                             sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                            if has_dht != 0:
                                cp_seed = pl.make_tensor(pl.addptr(pl.make_ptr(dht), cp_seg *
                                                         st_dht_ms + cp_head * st_dht_hs), [128, 128], [128, 1])
                                pl.set_validshape(rec_u1, [32, 64])
                                pl.load(rec_u1, cp_seed, [rec_v0_1, rec_sub_4 * 64])
                                pl.set_validshape(rec_u1, [128, 128])
                                s1_transposed(rec_u1, rec_u0)
                                cp_add_state(rec_state, rec_u0, rec_state)
                            s1_transposed_1(rec_state, rec_u1)
                            cp_init = pl.make_tensor(
                                pl.addptr(
                                    pl.make_ptr(recur_state), (rec_core_1 * ratio + cp_member) * 2 * 16384
                                ),
                                [128, 128],
                                [128, 1],
                            )
                            pl.set_validshape(rec_u1, [32, 64])
                            pl.store(cp_init, rec_u1, [rec_v0_1, rec_sub_4 * 64])
                            pl.set_validshape(rec_u1, [128, 128])
                        pl.system.bar_all()
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=6, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.V, event_id=7, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                for it in pl.range(0, num_chunks):
                    t_1 = num_chunks - 1 - it
                    cp_t = t_1
                    cp_chunk = cp_cbase + cp_t
                    rec_t_1 = t_1
                    cp_t = rec_t_1
                    cp_chunk = cp_cbase + cp_t
                    read_slot = it % 2
                    next_slot = 1 - read_slot
                    for cp_member in pl.range(0, ratio):
                        cp_head = cp_key * ratio + cp_member
                        n_1 = cp_head
                        rec_n_1 = cp_head
                        if has_local_cache == 0:
                            for vs in pl.range(0, 4):
                                rec_v0_1 = vs * 32
                                cp_previous = pl.make_tensor(pl.addptr(pl.make_ptr(recur_state), ((
                                    rec_core_1 * ratio + cp_member) * 2 + read_slot) * 16384), [128, 128], [128, 1])
                                pl.set_validshape(rec_u1, [32, 64])
                                pl.load(rec_u1, cp_previous, [rec_v0_1, rec_sub_4 * 64])
                                pl.set_validshape(rec_u1, [128, 128])
                                s1_transposed(rec_u1, rec_state)
                                rec_live = 64
                                rec_mul_33 = rec_t_1 * 64
                                rec_sub_12 = seq_len - rec_mul_33
                                rec_cmp_1 = rec_sub_12 < 64
                                if rec_cmp_1:
                                    rec_live = rec_sub_12
                                rec_add_12 = (
                                    rec_b_1 * (num_chunks * num_heads * 64 * 128)
                                    + rec_t_1 * (num_heads * 64 * 128)
                                    + rec_n_1 * 8192
                                )
                                rec_gm_window_6 = pl.make_tensor(pl.addptr(pl.make_ptr(
                                    gk_c), (cp_chunk * num_heads + cp_head) * 8192), [64, 128], [128, 1])
                                rec_mul_43 = rec_sub_4 * 64
                                pl.load(rec_gate_last, rec_gm_window_6, [rec_live - 1, 0])
                                if rec_cmp_1 or has_do == 0:
                                    s1_zero_vslice_1(rec_u0)
                                    rec_mul_49 = rec_sub_4 * 32
                                    rec_min_ = pl.min(32, pl.max(0, rec_live - rec_mul_49))
                                    if rec_min_ > 0 and has_do != 0:
                                        rec_gm_window_15 = pl.make_tensor(
                                            pl.addptr(
                                                pl.make_ptr(do_),
                                                cp_batch * st_do__bs + cp_tok * st_do__ts + cp_head * st_do__hs,
                                            ),
                                            [seq_len, 128],
                                            [st_do__ts, 1],
                                        )
                                        pl.set_validshape(rec_ub16, [pl.min(rec_min_, 128), 32])
                                        pl.load(rec_ub16, rec_gm_window_15, [rec_mul_33 + rec_mul_49, rec_v0_1])
                                        pl.set_validshape(rec_ub16, [128, 128])
                                        s1_widen(rec_ub16, rec_u0, rec_min_)
                                    s1_narrow(rec_u0, rec_ub16)
                                    pl.set_validshape(rec_ub16, [32, 32])
                                    pl.set_validshape(rec_pack_nd, [16, 32])
                                    pl.set_validshape(rec_pack_nz, [16, 32])
                                    for rec_pack_row in pl.range(0, 32, 16):
                                        pl.move(rec_pack_nd, rec_ub16, offset=[rec_pack_row, 0])
                                        pl.move(rec_pack_nz, rec_pack_nd)
                                        pl.insert(rec_tail_do_l1[0], rec_pack_nz, [rec_sub_4 * 32 + rec_pack_row, 0])
                                    pl.set_validshape(rec_ub16, [128, 128])
                                rec_mul_67 = rec_sub_4 * 32
                                pl.system.set_cross_core(pipe=P.MTE3, event_id=3,
                                                         sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                                pl.system.wait_cross_core(pipe=P.MTE3, event_id=3,
                                                          sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                                pl.system.set_cross_core(pipe=P.MTE3, event_id=4,
                                                         sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                if vs == 0:
                                    rec_gm_window_11 = pl.make_tensor(pl.addptr(pl.make_ptr(
                                        k_c), (cp_chunk * num_key_heads + cp_key) * 8192), [64, 128], [128, 1])
                                    pl.set_validshape(rec_ub16, [32, 128])
                                    pl.load(rec_ub16, rec_gm_window_11, [rec_mul_67, 0])
                                    pl.set_validshape(rec_ub16, [128, 128])
                                    rec_gm_window_12 = pl.make_tensor(pl.addptr(pl.make_ptr(
                                        gk_c), (cp_chunk * num_heads + cp_head) * 8192), [64, 128], [128, 1])
                                    pl.set_validshape(rec_u1, [32, 128])
                                    pl.load(rec_u1, rec_gm_window_12, [rec_mul_67, 0])
                                    pl.set_validshape(rec_u1, [128, 128])
                                    s1_kg_bf16(rec_ub16, rec_u1, rec_gate_last, rec_ub16)
                                    pl.set_validshape(rec_ub16, [32, 128])
                                    pl.set_validshape(rec_pack_nd, [16, 128])
                                    pl.set_validshape(rec_pack_nz, [16, 128])
                                    for rec_pack_row in pl.range(0, 32, 16):
                                        pl.move(rec_pack_nd, rec_ub16, offset=[rec_pack_row, 0])
                                        pl.move(rec_pack_nz, rec_pack_nd)
                                        pl.insert(rec_kg_l1[0], rec_pack_nz, [rec_mul_67 + rec_pack_row, 0])
                                    pl.set_validshape(rec_ub16, [128, 128])
                                s1_narrow_2(rec_state, rec_ub16)
                                pl.set_validshape(rec_ub16, [64, 32])
                                pl.set_validshape(rec_pack_nd, [16, 32])
                                pl.set_validshape(rec_pack_nz, [16, 32])
                                for rec_pack_row in pl.range(0, 64, 16):
                                    pl.move(rec_pack_nd, rec_ub16, offset=[rec_pack_row, 0])
                                    pl.move(rec_pack_nz, rec_pack_nd)
                                    pl.insert(rec_dh_l1[0], rec_pack_nz, [rec_mul_43 + rec_pack_row, 0])
                                pl.set_validshape(rec_ub16, [128, 128])
                                pl.system.set_cross_core(pipe=P.MTE3, event_id=3,
                                                         sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                                pl.system.wait_cross_core(pipe=P.MTE3, event_id=3,
                                                          sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                                pl.system.set_cross_core(pipe=P.MTE3, event_id=0,
                                                         sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                pl.system.wait_cross_core(pipe=P.V, event_id=1,
                                                          sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                s1_copy_keep32(rec_result_ub_1, rec_keep_ub)
                                rec_gm_window_47 = pl.make_tensor(pl.addptr(pl.make_ptr(
                                    recur_du), rec_core_1 * 8192), [64, 128], [128, 1])
                                pl.set_validshape(rec_keep_ub, [32, 32])
                                pl.store(rec_gm_window_47, rec_keep_ub, [rec_mul_67, rec_v0_1])
                                pl.set_validshape(rec_keep_ub, [32, 64])
                                pl.system.set_cross_core(pipe=P.V, event_id=2,
                                                         sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                if vs == 0:
                                    rec_gm_window_8 = pl.make_tensor(pl.addptr(pl.make_ptr(
                                        q_c), (cp_chunk * num_key_heads + cp_key) * 8192), [64, 128], [128, 1])
                                    pl.set_validshape(rec_ub16, [32, 128])
                                    pl.load(rec_ub16, rec_gm_window_8, [rec_mul_67, 0])
                                    pl.set_validshape(rec_ub16, [128, 128])
                                    s1_qg_bf16(rec_ub16, rec_u1, rec_ub16)
                                    pl.set_validshape(rec_ub16, [32, 128])
                                    pl.set_validshape(rec_pack_nd, [16, 128])
                                    pl.set_validshape(rec_pack_nz, [16, 128])
                                    for rec_pack_row in pl.range(0, 32, 16):
                                        pl.move(rec_pack_nd, rec_ub16, offset=[rec_pack_row, 0])
                                        pl.move(rec_pack_nz, rec_pack_nd)
                                        pl.insert(rec_qg_l1[0], rec_pack_nz, [rec_mul_67 + rec_pack_row, 0])
                                    pl.set_validshape(rec_ub16, [128, 128])
                                pl.system.set_cross_core(pipe=P.MTE3, event_id=3,
                                                         sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                                pl.system.wait_cross_core(pipe=P.MTE3, event_id=3,
                                                          sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                                pl.system.set_cross_core(pipe=P.MTE3, event_id=0,
                                                         sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                pl.system.wait_cross_core(pipe=P.V, event_id=1,
                                                          sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                s1_recurrent_add_product64(rec_state, rec_gate_last,
                                                           rec_result_ub_1, rec_state, scale, rec_mul_43)
                                pl.system.set_cross_core(pipe=P.V, event_id=2,
                                                         sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                s1_narrow_keep32(rec_keep_ub, rec_ub16)
                                pl.set_validshape(rec_ub16, [32, 32])
                                pl.set_validshape(rec_pack_nd, [16, 32])
                                pl.set_validshape(rec_pack_nz, [16, 32])
                                for rec_pack_row in pl.range(0, 32, 16):
                                    pl.move(rec_pack_nd, rec_ub16, offset=[rec_pack_row, 0])
                                    pl.move(rec_pack_nz, rec_pack_nd)
                                    pl.insert(rec_du_l1[0], rec_pack_nz, [rec_mul_67 + rec_pack_row, 0])
                                pl.set_validshape(rec_ub16, [128, 128])
                                pl.system.set_cross_core(pipe=P.MTE3, event_id=3,
                                                         sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                                pl.system.wait_cross_core(pipe=P.MTE3, event_id=3,
                                                          sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                                pl.system.set_cross_core(pipe=P.MTE3, event_id=0,
                                                         sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                pl.system.wait_cross_core(pipe=P.V, event_id=1,
                                                          sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                s1_binary_rhs64(rec_state, rec_result_ub_1, rec_state)
                                s1_transposed_1(rec_state, rec_u1)
                                cp_current = pl.make_tensor(pl.addptr(pl.make_ptr(recur_state), ((
                                    rec_core_1 * ratio + cp_member) * 2 + next_slot) * 16384), [128, 128], [128, 1])
                                pl.set_validshape(rec_u1, [32, 64])
                                pl.store(cp_current, rec_u1, [rec_v0_1, rec_mul_43])
                                pl.set_validshape(rec_u1, [128, 128])
                                pl.system.set_cross_core(pipe=P.V, event_id=2,
                                                         sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                            pl.system.bar_all()
                            pl.system.set_cross_core(pipe=P.MTE3, event_id=6,
                                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                            pl.system.wait_cross_core(pipe=P.V, event_id=7, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        live_1 = 64
                        mul_11_1 = t_1 * 64
                        sub_5_1 = seq_len - mul_11_1
                        if sub_5_1 < 64:
                            live_1 = sub_5_1
                        mul_14 = mul_10 * 64
                        mul_16 = num_heads * 64
                        gm_window_8 = pl.make_tensor(pl.addptr(pl.make_ptr(
                            beta_c), cp_chunk * num_heads * 64), [num_heads, 64], [64, 1])
                        mul_18 = sub_4 * 32
                        pl.load(beta_ub, gm_window_8, [n_1, 0])
                        mul_20 = core_1 * (6 * 16384)
                        mul_26 = (b_1 * num_chunks * num_heads + t_1 * num_heads + n_1) * 64
                        mul_27 = mul_26 * 128
                        gm_window_10 = pl.make_tensor(pl.addptr(pl.make_ptr(
                            gk_c), (cp_chunk * num_heads + cp_head) * 8192), [64, 128], [128, 1])
                        pl.set_validshape(u0, [32, 128])
                        pl.load(u0, gm_window_10, [mul_18, 0])
                        pl.set_validshape(u0, [128, 128])
                        s2_unary(u0, expg_ub)
                        gm_window_22 = pl.make_tensor(pl.addptr(pl.make_ptr(
                            k_c), (cp_chunk * num_key_heads + cp_key) * 8192), [64, 128], [128, 1])
                        pl.set_validshape(ub16, [32, 128])
                        pl.load(ub16, gm_window_22, [mul_18, 0])
                        pl.set_validshape(ub16, [128, 128])
                        s2_widen(ub16, u0)
                        s2_ternary_beta(u0, expg_ub, beta_ub, kbeta_ub, mul_18)
                        gm_window_26 = pl.make_tensor(pl.addptr(pl.make_ptr(
                            v_c), (cp_chunk * num_heads + cp_head) * 8192), [64, 128], [128, 1])
                        pl.set_validshape(ub16, [32, 128])
                        pl.load(ub16, gm_window_26, [mul_18, 0])
                        pl.set_validshape(ub16, [128, 128])
                        s2_widen(ub16, u0)
                        s2_beta_mul32(u0, beta_ub, vbeta_ub, mul_18)
                        add_11 = b_1 * (mul_14 * 128) + t_1 * (mul_16 * 128) + n_1 * 8192
                        gm_window_12 = pl.make_tensor(pl.addptr(pl.make_ptr(
                            gk_c), (cp_chunk * num_heads + cp_head) * 8192), [64, 128], [128, 1])
                        mul_41 = sub_4 * 64
                        pl.set_validshape(u0, [1, 64])
                        pl.load(u0, gm_window_12, [live_1 - 1, mul_41])
                        pl.set_validshape(u0, [128, 128])
                        s2_exponential(u0, state_sum_ub)
                        mul_114 = b_1 * (num_chunk_states * num_heads * 128 * 128)
                        mul_116 = num_heads * 128 * 128
                        mul_118 = n_1 * 16384
                        gm_window_31 = pl.make_tensor(pl.addptr(pl.make_ptr(
                            h), ((cp_hbase + cp_t) * num_heads + cp_head) * 16384), [128, 128], [128, 1])
                        pl.set_validshape(ub16, [128, 64])
                        pl.load(ub16, gm_window_31, [0, mul_41])
                        pl.set_validshape(ub16, [128, 128])
                        h_transposed = u1[64:128, 0:128]
                        s2_transpose_h_bf16(ub16, h_transposed)
                        if has_local_cache != 0:
                            dh_kv_view = pl.make_tensor(pl.addptr(pl.make_ptr(cached_state), ((
                                cp_hbase + cp_t + 1) * num_heads + cp_head) * 16384), [128, 128], [128, 1])
                            pl.set_validshape(ub16, [64, 128])
                            pl.load(ub16, dh_kv_view, [mul_41, 0])
                            pl.set_validshape(ub16, [128, 128])
                        else:
                            gm_window_35 = pl.make_tensor(pl.addptr(pl.make_ptr(recur_state), ((
                                core_1 * ratio + cp_member) * 2 + read_slot) * 16384), [128, 128], [128, 1])
                            pl.set_validshape(u0, [128, 64])
                            pl.load(u0, gm_window_35, [0, mul_41])
                            pl.set_validshape(u0, [128, 128])
                            s2_transpose_dh_bf16(u0, ub16)
                        pl.system.sync_src(set_pipe=P.MTE2, wait_pipe=P.S, event_id=7)
                        pl.system.sync_dst(set_pipe=P.MTE2, wait_pipe=P.S, event_id=7)
                        # One actual BF16 local dH operand serves gate and Cube.
                        # The rolling state in recur_state remains FP32.
                        s2_state_dot_local_bf16(u1, ub16, u2)
                        s2_state_scale(u2, state_sum_ub, state_sum_ub)
                        pl.set_validshape(ub16, [64, 128])
                        pl.set_validshape(safe_pack_nd, [16, 128])
                        pl.set_validshape(safe_pack_nz, [16, 128])
                        for base_pack_row in pl.range(0, 64, 16):
                            pl.move(safe_pack_nd, ub16, offset=[base_pack_row, 0])
                            pl.move(safe_pack_nz, safe_pack_nd)
                            pl.insert(base_bt_group[0], safe_pack_nz, [mul_41 + base_pack_row, 0])
                        pl.set_validshape(ub16, [128, 128])
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        gm_window_61 = pl.make_tensor(pl.addptr(pl.make_ptr(
                            gk_c), (cp_chunk * num_heads + cp_head) * 8192), [64, 128], [128, 1])
                        pl.set_validshape(u1, [32, 128])
                        pl.load(u1, gm_window_61, [mul_18, 0])
                        pl.set_validshape(u1, [128, 128])
                        gm_window_62 = pl.make_tensor(pl.addptr(pl.make_ptr(
                            gk_c), (cp_chunk * num_heads + cp_head) * 8192), [64, 128], [128, 1])
                        pl.set_validshape(u2, [1, 128])
                        pl.load(u2, gm_window_62, [live_1 - 1, 0])
                        pl.set_validshape(u2, [32, 128])
                        pl.system.wait_cross_core(pipe=P.V, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        base_dk = u0[96:128, 0:128]
                        s2_kg_func(result_ub_1, u1, u2, base_dk)
                        add_91 = mul_20 + 0 * 16384
                        pl.system.set_cross_core(pipe=P.V, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        gm_window_65 = pl.make_tensor(pl.addptr(pl.make_ptr(
                            k_c), (cp_chunk * num_key_heads + cp_key) * 8192), [64, 128], [128, 1])
                        pl.set_validshape(ub16, [32, 128])
                        pl.load(ub16, gm_window_65, [mul_18, 0])
                        pl.set_validshape(ub16, [128, 128])
                        s2_widen(ub16, u1)
                        s2_binary(base_dk, u1, u2)
                        s2_unary_2(u2, u1)
                        add_98 = mul_20 + 1 * 16384
                        gm_window_68 = pl.make_tensor(pl.addptr(pl.make_ptr(work32), add_98), [128, 128], [128, 1])
                        pl.set_validshape(u1, [32, 128])
                        pl.store(gm_window_68, u1, [mul_18, 0])
                        pl.set_validshape(u1, [128, 128])
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=3, sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                        pl.system.wait_cross_core(pipe=P.MTE2, event_id=3,
                                                  sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                        column_dg = pl.make_tensor(pl.addptr(pl.make_ptr(work32), add_98), [128, 128], [128, 1])
                        pl.set_validshape(u0, [64, 64])
                        pl.load(u0, column_dg, [0, mul_41])
                        pl.set_validshape(u0, [128, 128])
                        s2_column_neg_sum(u0, u2)
                        s2_binary_2(state_sum_ub, u2, u1)
                        gm_window_72 = pl.make_tensor(pl.addptr(pl.make_ptr(work32), add_98), [128, 128], [128, 1])
                        pl.set_validshape(u0, [1, 64])
                        pl.load(u0, gm_window_72, [live_1 - 1, mul_41])
                        pl.set_validshape(u0, [128, 128])
                        s2_binary_2(u0, u1, u2)
                        gm_window_73 = pl.make_tensor(pl.addptr(pl.make_ptr(work32), add_98), [128, 128], [128, 1])
                        pl.set_validshape(u2, [1, 64])
                        pl.store(gm_window_73, u2, [live_1 - 1, mul_41])
                        pl.set_validshape(u2, [32, 128])
                        if live_1 < 64 or has_do == 0:
                            s2_zero_rows32(u0)
                            min_ = pl.min(32, pl.max(0, live_1 - mul_18))
                            if min_ > 0 and has_do != 0:
                                gm_window_43 = pl.make_tensor(
                                    pl.addptr(
                                        pl.make_ptr(do_),
                                        cp_batch * st_do__bs + cp_tok * st_do__ts + cp_head * st_do__hs,
                                    ),
                                    [seq_len, 128],
                                    [st_do__ts, 1],
                                )
                                pl.set_validshape(ub16, [pl.min(min_, 128), 128])
                                pl.load(ub16, gm_window_43, [mul_11_1 + mul_18, 0])
                                pl.set_validshape(ub16, [128, 128])
                                s2_widen_1(ub16, u0, min_)
                            s2_narrow_2(u0, ub16)
                            pl.set_validshape(ub16, [32, 128])
                            pl.set_validshape(safe_pack_nd, [16, 128])
                            pl.set_validshape(safe_pack_nz, [16, 128])
                            for base_pack_row in pl.range(0, 32, 16):
                                pl.move(safe_pack_nd, ub16, offset=[base_pack_row, 0])
                                pl.move(safe_pack_nz, safe_pack_nd)
                                pl.insert(base_a_group[0], safe_pack_nz, [mul_18 + base_pack_row, 0])
                            pl.set_validshape(ub16, [128, 128])
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.V, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        s2_zero_rows32(u1)
                        s2_triangular(result_ub_1, u1, mul_18, scale)
                        add_124 = mul_20 + 4 * 16384
                        gm_window_86 = pl.make_tensor(pl.addptr(pl.make_ptr(work32), add_124), [128, 128], [128, 1])
                        pl.set_validshape(u1, [32, 64])
                        pl.store(gm_window_86, u1, [mul_18, 0])
                        pl.set_validshape(u1, [128, 128])
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        add_134 = mul_20 + 2 * 16384
                        s2_narrow_1(h_transposed, ub16)
                        pl.set_validshape(ub16, [64, 128])
                        pl.set_validshape(safe_pack_nd, [16, 128])
                        pl.set_validshape(safe_pack_nz, [16, 128])
                        for base_pack_row in pl.range(0, 64, 16):
                            pl.move(safe_pack_nd, ub16, offset=[base_pack_row, 0])
                            pl.move(safe_pack_nz, safe_pack_nd)
                            pl.insert(base_bt_group[0], safe_pack_nz, [mul_41 + base_pack_row, 0])
                        pl.set_validshape(ub16, [128, 128])
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.V, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        s2_zero_rows32(u0)
                        s2_ternary_1(u0, result_ub_1, expg_ub, u0, scale)
                        gm_window_112 = pl.make_tensor(pl.addptr(pl.make_ptr(work32), add_134), [128, 128], [128, 1])
                        pl.set_validshape(u0, [32, 128])
                        pl.store(gm_window_112, u0, [mul_18, 0])
                        pl.set_validshape(u0, [128, 128])
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=3, sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                        pl.system.wait_cross_core(pipe=P.MTE2, event_id=3,
                                                  sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                        gm_window_113 = pl.make_tensor(pl.addptr(pl.make_ptr(work32), add_98), [128, 128], [128, 1])
                        pl.set_validshape(u1, [32, 128])
                        pl.load(u1, gm_window_113, [mul_18, 0])
                        pl.set_validshape(u1, [128, 128])
                        gm_window_115 = pl.make_tensor(pl.addptr(pl.make_ptr(
                            q_c), (cp_chunk * num_key_heads + cp_key) * 8192), [64, 128], [128, 1])
                        pl.set_validshape(ub16, [32, 128])
                        pl.load(ub16, gm_window_115, [mul_18, 0])
                        pl.set_validshape(ub16, [128, 128])
                        s2_widen(ub16, u2)
                        s2_accumulate_dg(u1, u0, u2, 0, 32)
                        pl.system.sync_src(set_pipe=P.MTE2, wait_pipe=P.S, event_id=7)
                        pl.system.sync_dst(set_pipe=P.MTE2, wait_pipe=P.S, event_id=7)
                        cache_offset = core_1 * 8192
                        if has_local_cache != 0:
                            cache_offset = (cp_chunk * num_heads + cp_head) * 8192
                        e05_gm_window_134 = pl.make_tensor(
                            pl.addptr(pl.make_ptr(recur_du), cache_offset), [64, 128], [128, 1])
                        pl.set_validshape(u0, [32, 128])
                        pl.load(u0, e05_gm_window_134, [mul_18, 0])
                        pl.set_validshape(u0, [128, 128])
                        s2_narrow_2(u0, ub16)
                        pl.set_validshape(ub16, [32, 128])
                        pl.set_validshape(safe_pack_nd, [16, 128])
                        pl.set_validshape(safe_pack_nz, [16, 128])
                        for base_pack_row in pl.range(0, 32, 16):
                            pl.move(safe_pack_nd, ub16, offset=[base_pack_row, 0])
                            pl.move(safe_pack_nz, safe_pack_nd)
                            pl.insert(base_bn_group[0], safe_pack_nz, [mul_18 + base_pack_row, 0])
                        pl.set_validshape(ub16, [128, 128])
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.V, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        dvb_high = u0[64:96, 0:128]
                        s2_copy32(result_ub_1, dvb_high)
                        pl.system.set_cross_core(pipe=P.V, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        s2_widen(ub16, u1)
                        s2_binary_3(u0, u1, u2)
                        s2_narrow_2(u2, ub16)
                        pl.set_validshape(ub16, [32, 128])
                        pl.set_validshape(safe_pack_nd, [16, 128])
                        pl.set_validshape(safe_pack_nz, [16, 128])
                        for base_pack_row in pl.range(0, 32, 16):
                            pl.move(safe_pack_nd, ub16, offset=[base_pack_row, 0])
                            pl.move(safe_pack_nz, safe_pack_nd)
                            pl.insert(base_bn_group[0], safe_pack_nz, [mul_18 + base_pack_row, 0])
                        pl.set_validshape(ub16, [128, 128])
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.V, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        s2_binary_4(dvb_high, result_ub_1, u2)
                        pl.system.set_cross_core(pipe=P.V, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        s2_beta_mul32(u2, beta_ub, u0, mul_18)
                        min_2 = pl.min(32, pl.max(0, live_1 - mul_18))
                        if min_2 > 0:
                            s2_narrow_4(u0, ub16, min_2)
                            gm_window_145 = pl.make_tensor(
                                pl.addptr(
                                    pl.make_ptr(dv),
                                    cp_batch * st_dv_bs + cp_tok * st_dv_ts + cp_head * st_dv_hs,
                                ),
                                [seq_len, 128],
                                [st_dv_ts, 1],
                            )
                            pl.set_validshape(ub16, [pl.min(min_2, 128), 128])
                            pl.store(gm_window_145, ub16, [mul_11_1 + mul_18, 0])
                            pl.set_validshape(ub16, [128, 128])
                        pl.system.sync_src(set_pipe=P.MTE2, wait_pipe=P.S, event_id=7)
                        pl.system.sync_dst(set_pipe=P.MTE2, wait_pipe=P.S, event_id=7)
                        gm_window_147 = pl.make_tensor(pl.addptr(pl.make_ptr(
                            v_c), (cp_chunk * num_heads + cp_head) * 8192), [64, 128], [128, 1])
                        pl.set_validshape(ub16, [32, 128])
                        pl.load(ub16, gm_window_147, [mul_18, 0])
                        pl.set_validshape(ub16, [128, 128])
                        s2_widen(ub16, u1)
                        base_db = u0[32:64, 0:128]
                        s2_row_product_sum_1(u2, u1, base_db)
                        add_228 = mul_20 + 3 * 16384
                        cache_offset = core_1 * 8192
                        if has_local_cache != 0:
                            cache_offset = (cp_chunk * num_heads + cp_head) * 8192
                        e05_gm_window_150 = pl.make_tensor(
                            pl.addptr(pl.make_ptr(recur_du), cache_offset), [64, 128], [128, 1])
                        pl.set_validshape(u0, [32, 128])
                        pl.load(u0, e05_gm_window_150, [mul_18, 0])
                        pl.set_validshape(u0, [128, 128])
                        s2_narrow_2(u0, ub16)
                        pl.set_validshape(ub16, [32, 128])
                        pl.set_validshape(safe_pack_nd, [16, 128])
                        pl.set_validshape(safe_pack_nz, [16, 128])
                        for base_pack_row in pl.range(0, 32, 16):
                            pl.move(safe_pack_nd, ub16, offset=[base_pack_row, 0])
                            pl.move(safe_pack_nz, safe_pack_nd)
                            pl.insert(base_a_group[0], safe_pack_nz, [mul_18 + base_pack_row, 0])
                        pl.set_validshape(ub16, [128, 128])
                        s2_narrow_2(vbeta_ub, ub16)
                        pl.set_validshape(ub16, [32, 128])
                        pl.set_validshape(safe_pack_nd, [16, 128])
                        pl.set_validshape(safe_pack_nz, [16, 128])
                        for base_pack_row in pl.range(0, 32, 16):
                            pl.move(safe_pack_nd, ub16, offset=[base_pack_row, 0])
                            pl.move(safe_pack_nz, safe_pack_nd)
                            pl.insert(base_btn64_group[0], safe_pack_nz, [mul_18 + base_pack_row, 0])
                        pl.set_validshape(ub16, [128, 128])
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.MTE2, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.sync_src(set_pipe=P.MTE2, wait_pipe=P.S, event_id=7)
                        pl.system.sync_dst(set_pipe=P.MTE2, wait_pipe=P.S, event_id=7)
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.V, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        s2_unary_3(result_ub_1, vbeta_ub)
                        s2_narrow_2(vbeta_ub, ub16)
                        pl.set_validshape(ub16, [32, 128])
                        pl.set_validshape(safe_pack_nd, [16, 128])
                        pl.set_validshape(safe_pack_nz, [16, 128])
                        for base_pack_row in pl.range(0, 32, 16):
                            pl.move(safe_pack_nd, ub16, offset=[base_pack_row, 0])
                            pl.move(safe_pack_nz, safe_pack_nd)
                            pl.insert(base_a_group[0], safe_pack_nz, [mul_18 + base_pack_row, 0])
                        pl.set_validshape(ub16, [128, 128])
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        s2_narrow_2(kbeta_ub, ub16)
                        pl.set_validshape(ub16, [32, 128])
                        pl.set_validshape(safe_pack_nd, [16, 128])
                        pl.set_validshape(safe_pack_nz, [16, 128])
                        for base_pack_row in pl.range(0, 32, 16):
                            pl.move(safe_pack_nd, ub16, offset=[base_pack_row, 0])
                            pl.move(safe_pack_nz, safe_pack_nd)
                            pl.insert(base_btn64_group[0], safe_pack_nz, [mul_18 + base_pack_row, 0])
                        pl.set_validshape(ub16, [128, 128])
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.V, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.set_validshape(result_ub_1, [32, 64])
                        pl.set_validshape(kbeta_ub, [32, 64])
                        pl.move(kbeta_ub, result_ub_1)
                        pl.set_validshape(result_ub_1, [32, 128])
                        pl.set_validshape(kbeta_ub, [32, 128])
                        pl.system.set_cross_core(pipe=P.V, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        s2_narrow_2(vbeta_ub, ub16)
                        pl.set_validshape(ub16, [32, 128])
                        pl.set_validshape(safe_pack_nd, [16, 128])
                        pl.set_validshape(safe_pack_nz, [16, 128])
                        for base_pack_row in pl.range(0, 32, 16):
                            pl.move(safe_pack_nd, ub16, offset=[base_pack_row, 0])
                            pl.move(safe_pack_nz, safe_pack_nd)
                            pl.insert(base_bn_group[0], safe_pack_nz, [mul_18 + base_pack_row, 0])
                        pl.set_validshape(ub16, [128, 128])
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.V, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        s2_ternary_beta(result_ub_1, expg_ub, beta_ub, vbeta_ub, mul_18)
                        gm_window_204 = pl.make_tensor(pl.addptr(pl.make_ptr(
                            k_c), (cp_chunk * num_key_heads + cp_key) * 8192), [64, 128], [128, 1])
                        pl.set_validshape(ub16, [32, 128])
                        pl.load(ub16, gm_window_204, [mul_18, 0])
                        pl.set_validshape(ub16, [128, 128])
                        s2_widen(ub16, u2)
                        s2_accumulate_dg(u1, vbeta_ub, u2, 32, 0)
                        gm_window_205 = pl.make_tensor(pl.addptr(pl.make_ptr(work32), add_98), [128, 128], [128, 1])
                        pl.set_validshape(u1, [32, 128])
                        pl.store(gm_window_205, u1, [mul_18, 0])
                        pl.set_validshape(u1, [128, 128])
                        s2_row_triple_sum(result_ub_1, u2, expg_ub, u0)
                        pl.system.set_cross_core(pipe=P.V, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        s2_add_base_beta(u0, u1)
                        gm_window_212 = pl.make_tensor(pl.addptr(pl.make_ptr(work32), add_228), [128, 128], [128, 1])
                        pl.set_validshape(u1, [32, 1])
                        pl.store(gm_window_212, u1, [mul_18, 0])
                        pl.set_validshape(u1, [128, 128])
                        s2_binary_6(base_dk, vbeta_ub, u2)
                        gm_window_215 = pl.make_tensor(pl.addptr(pl.make_ptr(work32), add_91), [128, 128], [128, 1])
                        pl.set_validshape(u2, [32, 128])
                        pl.store(gm_window_215, u2, [mul_18, 0])
                        pl.set_validshape(u2, [32, 128])
                        s2_zero_matrix32x64(u1)
                        s2_triangular_1(kbeta_ub, u1, mul_18)
                        s2_narrow_3(u1, ub16)
                        pl.set_validshape(ub16, [32, 64])
                        pl.set_validshape(safe_pack_nd, [16, 64])
                        pl.set_validshape(safe_pack_nz, [16, 64])
                        for base_pack_row in pl.range(0, 32, 16):
                            pl.move(safe_pack_nd, ub16, offset=[base_pack_row, 0])
                            pl.move(safe_pack_nz, safe_pack_nd)
                            pl.insert(base_a_group[0], safe_pack_nz, [mul_18 + base_pack_row, 0])
                        pl.set_validshape(ub16, [128, 128])
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.V, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        s2_zero_matrix32x64(u1)
                        s2_triangular_2(result_ub_1, u1, mul_18)
                        add_319 = mul_20 + 5 * 16384
                        gm_window_241 = pl.make_tensor(pl.addptr(pl.make_ptr(work32), add_319), [128, 128], [128, 1])
                        pl.set_validshape(u1, [32, 64])
                        pl.store(gm_window_241, u1, [mul_18, 0])
                        pl.set_validshape(u1, [128, 128])
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=3, sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                        pl.system.wait_cross_core(pipe=P.MTE2, event_id=3,
                                                  sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                        safe_gate = pl.make_tensor(pl.addptr(pl.make_ptr(
                            gk_c), (cp_chunk * num_heads + cp_head) * 8192), [128, 128], [128, 1])
                        safe_q = pl.make_tensor(pl.addptr(pl.make_ptr(
                            q_c), (cp_chunk * num_key_heads + cp_key) * 8192), [128, 128], [128, 1])
                        safe_k = pl.make_tensor(pl.addptr(pl.make_ptr(
                            k_c), (cp_chunk * num_key_heads + cp_key) * 8192), [128, 128], [128, 1])
                        safe_dq = pl.make_tensor(pl.addptr(pl.make_ptr(
                            work32), mul_20 + 2 * 16384), [128, 128], [128, 1])
                        safe_dk = pl.make_tensor(pl.addptr(pl.make_ptr(
                            work32), mul_20 + 0 * 16384), [128, 128], [128, 1])
                        safe_dg = pl.make_tensor(pl.addptr(pl.make_ptr(
                            work32), mul_20 + 1 * 16384), [128, 128], [128, 1])
                        safe_db = pl.make_tensor(pl.addptr(pl.make_ptr(
                            work32), mul_20 + 3 * 16384), [128, 128], [128, 1])
                        safe_da = pl.make_tensor(pl.addptr(pl.make_ptr(
                            work32), mul_20 + 4 * 16384), [128, 128], [128, 1])
                        safe_dl = pl.make_tensor(pl.addptr(pl.make_ptr(
                            work32), mul_20 + 5 * 16384), [128, 128], [128, 1])
                        for grad_block in pl.range(0, local_blocks):
                            grad_row = grad_block * local_tile + sub_4 * local_rows
                            ub_row = 32 + grad_block * local_rows
                            init_dq = u0[ub_row + 0:ub_row + 0 + local_rows, 0:128]
                            pl.set_validshape(init_dq, [local_rows, 128])
                            pl.load(init_dq, safe_dq, [grad_row, 0])
                            init_dk = u0[ub_row + 32:ub_row + 32 + local_rows, 0:128]
                            pl.set_validshape(init_dk, [local_rows, 128])
                            pl.load(init_dk, safe_dk, [grad_row, 0])
                            init_dg = u1[ub_row + 0:ub_row + 0 + local_rows, 0:128]
                            pl.set_validshape(init_dg, [local_rows, 128])
                            pl.load(init_dg, safe_dg, [grad_row, 0])
                            init_db = expg_ub[grad_block * local_rows:grad_block * local_rows + local_rows, 0:128]
                            pl.set_validshape(init_db, [16, 1])
                            pl.load(init_db, safe_db, [grad_row, 0])
                        # Only the first rows of ub16 are scratch in this phase.
                        # Keep this subcore's interleaved q/k rows in the upper arena.
                        local_q = ub16[32:64, 0:128]
                        local_k = ub16[64:96, 0:128]
                        for resident_block in pl.range(0, local_blocks):
                            q_rows = local_q[resident_block *
                                local_rows:resident_block * local_rows + local_rows, 0:128]
                            k_rows = local_k[resident_block *
                                local_rows:resident_block * local_rows + local_rows, 0:128]
                            pl.load(q_rows, safe_q, [resident_block * local_tile + sub_4 * local_rows, 0])
                            pl.load(k_rows, safe_k, [resident_block * local_tile + sub_4 * local_rows, 0])
                        cached_ei = kbeta_ub[0:16, 0:128]
                        cached_ej = vbeta_ub[0:16, 0:128]
                        for si in pl.range(0, local_blocks):
                            for sj in pl.range(0, si + 1):
                                pivot = si * local_tile
                                if si == sj:
                                    pivot = si * local_tile + local_rows
                                pl.set_validshape(u1, [1, 128])
                                pl.load(u1, safe_gate, [pivot, 0])
                                pl.set_validshape(u1, [128, 128])
                                pl.set_validshape(u0, [local_rows, 128])
                                pl.load(u0, safe_gate, [sj * local_tile + sub_4 * local_rows, 0])
                                pl.set_validshape(u0, [128, 128])
                                s2_safe_exp(u0, u1, cached_ej, -1.0, local_rows)
                                for op in pl.range(0, 4):
                                    source_row = sj * local_tile + sub_4 * local_rows
                                    target_row = si * local_tile + sub_4 * local_rows
                                    source_sign = -1.0
                                    target_sign = 1.0
                                    if op >= 2:
                                        source_row = si * local_tile + sub_4 * local_rows
                                        target_row = sj * local_tile + sub_4 * local_rows
                                        source_sign = 1.0
                                        target_sign = -1.0
                                    if op != 1:
                                        if op == 2:
                                            s2_safe_scale_bf16(
                                                local_q[si * local_rows: si * local_rows + local_rows, 0:128],
                                                cached_ei,
                                                u2,
                                                local_rows,
                                            )
                                        else:
                                            if op < 2:
                                                s2_safe_scale_bf16(
                                                    local_k[
                                                        sj * local_rows: sj * local_rows + local_rows, 0:128
                                                    ],
                                                    cached_ej,
                                                    u2,
                                                    local_rows,
                                                )
                                            else:
                                                s2_safe_scale_bf16(
                                                    local_k[
                                                        si * local_rows: si * local_rows + local_rows, 0:128
                                                    ],
                                                    cached_ei,
                                                    u2,
                                                    local_rows,
                                                )
                                        if op == 3:
                                            s2_beta_mul16(u2, beta_ub, u2, source_row, local_rows)
                                    if op == 0 or op == 2:
                                        pl.set_validshape(u0, [local_rows, local_tile])
                                        pl.load(u0, safe_da, [si * local_tile + sub_4 * local_rows, sj * local_tile])
                                        pl.set_validshape(u0, [128, 128])
                                    else:
                                        pl.set_validshape(u0, [local_rows, local_tile])
                                        pl.load(u0, safe_dl, [si * local_tile + sub_4 * local_rows, sj * local_tile])
                                        pl.set_validshape(u0, [128, 128])
                                    if op != 1:
                                        pl.set_validshape(safe_pack_nd, [local_rows, 128])
                                        pl.set_validshape(safe_pack_nz, [local_rows, 128])
                                        s2_safe_pack_hi(u2, safe_pack_nz, 128, local_rows)
                                        pl.insert(safe_b1_group[0], safe_pack_nz, [sub_4 * local_rows, 0])
                                        s2_safe_pack_lo(u2, safe_pack_nz, 128, local_rows)
                                        pl.insert(safe_b1_group[1], safe_pack_nz, [sub_4 * local_rows, 0])
                                    if si == sj:
                                        s2_safe_strict(u0, sub_4 * local_rows, local_rows)
                                    pl.set_validshape(safe_pack_nd, [local_rows, local_tile])
                                    pl.set_validshape(safe_pack_nz, [local_rows, local_tile])
                                    s2_safe_pack_hi(u0, safe_pack_nz, local_tile, local_rows)
                                    if op < 2:
                                        pl.insert(safe_a1_group[0], safe_pack_nz, [sub_4 * local_rows, 0])
                                    else:
                                        pl.insert(safe_at1_group[0], safe_pack_nz, [sub_4 * local_rows, 0])
                                    s2_safe_pack_lo(u0, safe_pack_nz, local_tile, local_rows)
                                    if op < 2:
                                        pl.insert(safe_a1_group[1], safe_pack_nz, [sub_4 * local_rows, 0])
                                    else:
                                        pl.insert(safe_at1_group[1], safe_pack_nz, [sub_4 * local_rows, 0])
                                    pl.system.set_cross_core(pipe=P.MTE3, event_id=4,
                                                             sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                    # Ei is first consumed by the op=0 epilogue. Generate it
                                    # after releasing operands to Cube, so its load/exp
                                    # can overlap the first three-product matmul.
                                    if op == 0 and (sj == 0 or si == sj):
                                        pl.set_validshape(u0, [local_rows, 128])
                                        pl.load(u0, safe_gate, [si * local_tile + sub_4 * local_rows, 0])
                                        pl.set_validshape(u0, [128, 128])
                                        s2_safe_exp(u0, u1, cached_ei, 1.0, local_rows)
                                    pl.system.wait_cross_core(pipe=P.V, event_id=1,
                                                              sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                                    grad_ub_row = 32 + target_row // local_tile * local_rows
                                    dq_acc = u0[grad_ub_row:grad_ub_row + local_rows, 0:128]
                                    dk_acc = u0[grad_ub_row + 32:grad_ub_row + 32 + local_rows, 0:128]
                                    dg_acc = u1[grad_ub_row:grad_ub_row + local_rows, 0:128]
                                    if op == 0:
                                        s2_local_result(
                                            safe_result_v,
                                            cached_ei,
                                            local_q[si * local_rows: si * local_rows + local_rows, 0:128],
                                            dq_acc,
                                            dg_acc,
                                            target_sign,
                                            local_rows,
                                        )
                                        if si == sj:
                                            pl.set_validshape(u1, [local_rows, local_tile])
                                            pl.load(u1, safe_da, [target_row, sj * local_tile])
                                            pl.set_validshape(u1, [128, 128])
                                            s2_safe_diagonal_add(
                                                dq_acc,
                                                u1,
                                                local_k[si * local_rows: si * local_rows + local_rows, 0:128],
                                                sub_4 * local_rows,
                                                local_rows,
                                            )
                                    else:
                                        if op == 1:
                                            db_acc = expg_ub[target_row // local_tile * local_rows:target_row //
                                                local_tile * local_rows + local_rows, 0:128]
                                            s2_local_beta_result(
                                                safe_result_v,
                                                cached_ei,
                                                local_k[si * local_rows: si * local_rows + local_rows, 0:128],
                                                beta_ub,
                                                target_row,
                                                dk_acc,
                                                dg_acc,
                                                db_acc,
                                                local_rows,
                                            )
                                        else:
                                            s2_local_result(
                                                safe_result_v,
                                                cached_ej,
                                                local_k[sj * local_rows: sj * local_rows + local_rows, 0:128],
                                                dk_acc,
                                                dg_acc,
                                                target_sign,
                                                local_rows,
                                            )
                                        if si == sj and op == 2:
                                            pl.set_validshape(u1, [local_rows, local_tile])
                                            pl.load(u1, safe_da, [target_row, sj * local_tile])
                                            pl.set_validshape(u1, [128, 128])
                                            s2_safe_diagonal_add(
                                                dk_acc,
                                                u1,
                                                local_q[sj * local_rows: sj * local_rows + local_rows, 0:128],
                                                sub_4 * local_rows,
                                                local_rows,
                                            )
                                    pl.system.set_cross_core(pipe=P.V, event_id=6,
                                                             sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        dq_output = pl.make_tensor(
                            pl.addptr(
                                pl.make_ptr(dq), cp_batch * st_dq_bs + cp_tok * st_dq_ts + cp_key * st_dq_hs
                            ),
                            [seq_len, 128],
                            [st_dq_ts, 1],
                        )
                        dk_output = pl.make_tensor(
                            pl.addptr(
                                pl.make_ptr(dk), cp_batch * st_dk_bs + cp_tok * st_dk_ts + cp_key * st_dk_hs
                            ),
                            [seq_len, 128],
                            [st_dk_ts, 1],
                        )
                        db_output = pl.make_tensor(
                            pl.addptr(
                                pl.make_ptr(db), cp_batch * st_db_bs + cp_tok * st_db_ts + cp_head * st_db_hs
                            ),
                            [seq_len, 1],
                            [st_db_ts, 1],
                        )
                        raw_seq_stride = num_heads * 128 * (1 - layout_code) + 128 * layout_code
                        raw_head_stride = 128 * (1 - layout_code) + valid_s * 128 * layout_code
                        raw_offset = b_1 * valid_s * num_heads * 128 + n_1 * raw_head_stride
                        raw_q_view = pl.make_tensor(
                            pl.addptr(
                                pl.make_ptr(q_raw),
                                cp_batch * st_q_raw_bs + cp_tok * st_q_raw_ts + cp_key * st_q_raw_hs,
                            ),
                            [valid_s, 128],
                            [st_q_raw_ts, 1],
                        )
                        raw_k_view = pl.make_tensor(
                            pl.addptr(
                                pl.make_ptr(k_raw),
                                cp_batch * st_k_raw_bs + cp_tok * st_k_raw_ts + cp_key * st_k_raw_hs,
                            ),
                            [valid_s, 128],
                            [st_k_raw_ts, 1],
                        )
                        raw_g_view = pl.make_tensor(
                            pl.addptr(
                                pl.make_ptr(g_raw),
                                cp_batch * st_g_raw_bs + cp_tok * st_g_raw_ts + cp_head * st_g_raw_hs,
                            ),
                            [valid_s, 128],
                            [st_g_raw_ts, 1],
                        )
                        for grad_block in pl.range(0, local_blocks):
                            grad_row = grad_block * local_tile + sub_4 * local_rows
                            ub_row = 32 + grad_block * local_rows
                            final_dq = u0[ub_row:ub_row + local_rows, 0:128]
                            final_dk = u0[ub_row + 32:ub_row + 32 + local_rows, 0:128]
                            final_db = expg_ub[grad_block * local_rows:grad_block * local_rows + local_rows, 0:128]
                            output_rows = pl.min(local_rows, pl.max(0, live_1 - grad_row))
                            valid_rows = output_rows
                            if fuse_preprocess:
                                valid_rows = pl.min(valid_rows, pl.max(0, valid_s - mul_11_1 - grad_row))
                            if output_rows > 0:
                                gva_q = pl.make_tensor(pl.addptr(pl.make_ptr(
                                    qk_sum), core_1 * 16384), [64, 128], [128, 1])
                                gva_k = pl.make_tensor(pl.addptr(pl.make_ptr(qk_sum), core_1 *
                                                       16384 + 8192), [64, 128], [128, 1])
                                if cp_member != 0:
                                    pl.set_validshape(u2, [local_rows, 128])
                                    pl.load(u2, gva_q, [grad_row, 0])
                                    s2_safe_add(final_dq, u2, final_dq, local_rows)
                                    pl.load(u2, gva_k, [grad_row, 0])
                                    s2_safe_add(final_dk, u2, final_dk, local_rows)
                                    pl.set_validshape(u2, [128, 128])
                                if cp_member + 1 != ratio:
                                    pl.store(gva_q, final_dq, [grad_row, 0])
                                    pl.store(gva_k, final_dk, [grad_row, 0])
                                if cp_member + 1 == ratio:
                                    if fuse_preprocess:
                                        if valid_rows > 0:
                                            pl.set_validshape(safe_diag_raw, [valid_rows, 128])
                                            pl.load(safe_diag_raw, raw_q_view, [mul_11_1 + grad_row, 0])
                                            pl.set_validshape(safe_diag_raw, [local_rows, 128])
                                        cp_l2_backward(safe_diag_raw, final_dq, ub16, valid_rows, ratio)
                                        if valid_rows < output_rows:
                                            s2_zero_bf16_rows(ub16, valid_rows, output_rows, 128)
                                    else:
                                        s2_narrow_4(final_dq, ub16, valid_rows)
                                    pl.set_validshape(ub16, [output_rows, 128])
                                    pl.store(dq_output, ub16, [mul_11_1 + grad_row, 0])
                                    pl.set_validshape(ub16, [128, 128])
                                    if fuse_preprocess:
                                        if valid_rows > 0:
                                            pl.set_validshape(safe_diag_raw, [valid_rows, 128])
                                            pl.load(safe_diag_raw, raw_k_view, [mul_11_1 + grad_row, 0])
                                            pl.set_validshape(safe_diag_raw, [local_rows, 128])
                                        cp_l2_backward(safe_diag_raw, final_dk, ub16, valid_rows, ratio)
                                        if valid_rows < output_rows:
                                            s2_zero_bf16_rows(ub16, valid_rows, output_rows, 128)
                                    else:
                                        s2_narrow_4(final_dk, ub16, valid_rows)
                                    pl.set_validshape(ub16, [output_rows, 128])
                                    pl.store(dk_output, ub16, [mul_11_1 + grad_row, 0])
                                    pl.set_validshape(ub16, [128, 128])
                                if fuse_preprocess:
                                    s2_beta_backward(final_db, beta_ub, ub16, valid_rows, grad_row)
                                    if valid_rows < output_rows:
                                        s2_zero_bf16_rows(ub16, valid_rows, output_rows, 1)
                                else:
                                    s2_narrow_6(final_db, ub16, valid_rows)
                                pl.set_validshape(ub16, [output_rows, 1])
                                pl.store(db_output, ub16, [mul_11_1 + grad_row, 0])
                                pl.set_validshape(ub16, [128, 128])
                            final_dg = u1[ub_row:ub_row + local_rows, 0:128]
                            pl.set_validshape(final_dg, [local_rows, 128])
                            pl.store(safe_dg, final_dg, [grad_row, 0])
                        pl.system.sync_src(set_pipe=P.MTE2, wait_pipe=P.S, event_id=7)
                        pl.system.sync_dst(set_pipe=P.MTE2, wait_pipe=P.S, event_id=7)
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=3, sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                        pl.system.wait_cross_core(pipe=P.MTE2, event_id=3,
                                                  sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                        gm_window_301 = pl.make_tensor(pl.addptr(pl.make_ptr(work32), add_98), [128, 128], [128, 1])
                        pl.set_validshape(u0, [64, 64])
                        pl.load(u0, gm_window_301, [0, mul_41])
                        pl.set_validshape(u0, [128, 128])
                        s2_reverse_scan(u0, u1, live_1)
                        dg_output = pl.make_tensor(
                            pl.addptr(
                                pl.make_ptr(dgi),
                                cp_batch * st_dgi_bs + cp_tok * st_dgi_ts + cp_head * st_dgi_hs,
                            ),
                            [seq_len, 128],
                            [st_dgi_ts, 1],
                        )
                        if not fuse_preprocess:
                            s2_narrow_7(u1, ub16, live_1)
                            pl.set_validshape(ub16, [live_1, 64])
                            pl.store(dg_output, ub16, [mul_11_1, mul_41])
                            pl.set_validshape(ub16, [128, 128])
                        if single_token:
                            state_offset = (b_1 * num_chunk_states * num_heads + t_1 * num_heads + n_1) * 16384
                            h_entry = pl.make_tensor(pl.addptr(pl.make_ptr(
                                h), ((cp_hbase + cp_t) * num_heads + cp_head) * 16384), [128, 128], [128, 1])
                            cache_offset = ((core_1 * ratio + cp_member) * 2 + next_slot) * 16384
                            if has_local_cache != 0:
                                cache_offset = ((cp_hbase + cp_t) * num_heads + cp_head) * 16384
                            dh_entry = pl.make_tensor(pl.addptr(pl.make_ptr(recur_state),
                                                      cache_offset), [128, 128], [128, 1])
                            pl.set_validshape(ub16, [128, 64])
                            pl.load(ub16, h_entry, [0, mul_41])
                            pl.set_validshape(ub16, [128, 128])
                            s2_widen_state_half(ub16, u0)
                            pl.set_validshape(u1, [128, 64])
                            pl.load(u1, dh_entry, [0, mul_41])
                            pl.set_validshape(u1, [128, 128])
                            s2_single_token_gate(u0, u1, u2)
                        if fuse_preprocess:
                            if reduce_mode == 1:
                                s2_zero_params(param_sum)
                            pl.set_validshape(kbeta_ub, [1, 64])
                            pl.load(kbeta_ub, dt_bias, [n_1, mul_41])
                            pl.set_validshape(kbeta_ub, [32, 128])
                            av = pl.make_tensor(a_log, [num_heads, 1], [1, 1])
                            pl.set_validshape(vbeta_ub, [1, 1])
                            pl.load(vbeta_ub, av, [n_1, 0])
                            pl.set_validshape(vbeta_ub, [32, 128])
                            raw_live = pl.min(live_1, pl.max(0, valid_s - mul_11_1))
                            if lower_bound == 0:
                                s2_zero_bf16_rows(ub16, 0, live_1, 64)
                                pl.set_validshape(ub16, [live_1, 64])
                                pl.store(dg_output, ub16, [mul_11_1, mul_41])
                                pl.set_validshape(ub16, [128, 128])
                            else:
                                for gate_row in pl.range(0, raw_live, 64):
                                    gate_rows = pl.min(64, raw_live - gate_row)
                                    pl.set_validshape(ub16, [gate_rows, 64])
                                    pl.load(ub16, raw_g_view, [mul_11_1 + gate_row, mul_41])
                                    pl.set_validshape(ub16, [128, 128])
                                    if single_token:
                                        s2_gate_backward(ub16, u2, kbeta_ub, vbeta_ub, ub16,
                                                         param_sum, gate_row, gate_rows, lower_bound)
                                    else:
                                        s2_gate_backward(ub16, u1, kbeta_ub, vbeta_ub, ub16,
                                                         param_sum, gate_row, gate_rows, lower_bound)
                                    pl.set_validshape(ub16, [gate_rows, 64])
                                    pl.store(dg_output, ub16, [mul_11_1 + gate_row, mul_41])
                                    pl.set_validshape(ub16, [128, 128])
                                if raw_live < live_1:
                                    pad_rows = live_1 - raw_live
                                    s2_zero_bf16_rows(ub16, 0, pad_rows, 64)
                                    pl.set_validshape(ub16, [pad_rows, 64])
                                    pl.store(dg_output, ub16, [mul_11_1 + raw_live, mul_41])
                                    pl.set_validshape(ub16, [128, 128])
                            if reduce_mode == 1:
                                bias_partial = param_sum[0:1, 0:64]
                                a_partial = param_sum[1:2, 0:64]
                                pl.store(dbias_out, bias_partial, [n_1, mul_41], atomic=pl.AtomicType.AtomicAdd)
                                pl.set_validshape(a_partial, [1, 1])
                                pl.store(da_view, a_partial, [n_1, 0], atomic=pl.AtomicType.AtomicAdd)
                                pl.set_validshape(a_partial, [1, 64])
                        elif single_token:
                            s2_narrow_7(u2, ub16, 1)
                            pl.set_validshape(ub16, [1, 64])
                            pl.store(dg_output, ub16, [mul_11_1, mul_41])
                            pl.set_validshape(ub16, [128, 128])
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=3, sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                        pl.system.wait_cross_core(pipe=P.MTE2, event_id=3,
                                                  sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK)
                        pl.system.bar_all()
                        pl.system.set_cross_core(pipe=P.MTE3, event_id=6, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                        pl.system.wait_cross_core(pipe=P.V, event_id=7, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                for cp_member in pl.range(0, ratio):
                    cp_head = cp_key * ratio + cp_member
                    n_1 = cp_head
                    rec_n_1 = cp_head
                    for vs in pl.range(0, 4):
                        cache_offset = ((rec_core_1 * ratio + cp_member) * 2 + num_chunks % 2) * 16384
                        if has_local_cache != 0:
                            cache_offset = (cp_hbase * num_heads + cp_head) * 16384
                        cp_final = pl.make_tensor(pl.addptr(pl.make_ptr(recur_state),
                                                  cache_offset), [128, 128], [128, 1])
                        cp_initial_grad = pl.make_tensor(
                            pl.addptr(
                                pl.make_ptr(initial_grad),
                                cp_seg * st_initial_grad_ms + cp_head * st_initial_grad_hs,
                            ),
                            [128, 128],
                            [128, 1],
                        )
                        s1_zero_vslice_1(rec_u1)
                        if first_continues == 0 or (packed != 0 and cp_seg != 0):
                            pl.set_validshape(rec_u1, [32, 64])
                            if has_local_cache != 0:
                                dh0_summary = pl.make_tensor(pl.addptr(pl.make_ptr(
                                    all_summary), (cp_batch * num_heads + cp_head) * 2 * 16384), [128, 128], [128, 1])
                                pl.load(rec_u1, dh0_summary, [vs * 32, rec_sub_4 * 64])
                            else:
                                pl.load(rec_u1, cp_final, [vs * 32, rec_sub_4 * 64])
                            pl.set_validshape(rec_u1, [128, 128])
                        else:
                            s2_zero32(rec_u1)
                        pl.set_validshape(rec_u1, [32, 64])
                        pl.store(cp_initial_grad, rec_u1, [vs * 32, rec_sub_4 * 64])
                        pl.set_validshape(rec_u1, [128, 128])


def _check_strided(name, value, dtype, device=None):
    if not isinstance(value, torch.Tensor) or value.dtype != dtype or value.device.type != 'npu':
        raise ValueError(f'{name} must be an NPU tensor of dtype {dtype}')
    if device is not None and value.device != device:
        raise ValueError(f'{name} device mismatch')
    if value.stride(-1) != 1:
        raise ValueError(f'{name} last dimension must be contiguous')
    extent = 1
    for stride, length in sorted(((s, n) for s, n in zip(value.stride(), value.shape) if n > 1)):
        if stride < extent:
            raise ValueError(f'{name} overlapping/broadcast views are unsupported')
        extent += (length - 1) * stride


def _stride_args(x, layout, state=False):
    if state:
        return (x.stride(0), x.stride(1))
    return (x.stride(0), x.stride(2), x.stride(1)) if layout == 'BNSD' else x.stride()[:3]


def _bind_output(bindings, name, shape, dtype, device, live, state=False):
    value = bindings.get(name)
    if value is None:
        return torch.empty(shape, dtype=dtype, device=device)
    _check_strided(name, value, dtype, device)
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f'{name} output shape mismatch')
    if state and value.stride(-2) != 128:
        raise ValueError('state matrix must be contiguous')
    if any((value.untyped_storage().data_ptr() == x.untyped_storage().data_ptr() for x in live)):
        raise ValueError(f'{name} output aliases a live tensor')
    return value


def _param_views(buffer, meta):
    n = int(meta['Nv'])
    global_n = int(meta.get('global_Nv', n))
    if buffer.shape == (n * 129,):
        base = 0
        total = n
    elif buffer.shape == (global_n * 129,):
        base = int(meta['hv_start'])
        total = global_n
    else:
        raise ValueError('parameter buffer must cover local or global heads')
    return (buffer[base:base + n], buffer[total + base * 128:total + (base + n) * 128].view(n, 128))


def _cp_dense_meta(cache, cp_meta):
    meta = cache['meta']
    if meta['cache_version'] != 'flash_kda_cp_train_v3_bf16_state':
        raise ValueError('CP backward requires flash_kda_cp_train_v3_bf16_state')
    if not 1 <= meta['Nk'] <= meta['Nv'] or meta['Nv'] % meta['Nk']:
        raise ValueError('Nv must be divisible by Nk')
    batch_size, seq_len, num_heads, num_chunks = map(int, (meta['B'], meta['S'], meta['Nv'], meta['NC']))
    if meta['layout_qkv'] == 'TND':
        seq_len = sum(cp_meta['sequence_lengths'])
        num_chunks = int(meta['J'])
    if not (
        1 <= batch_size <= B_MAX
        and 1 <= num_heads <= N_MAX
        and (1 <= seq_len <= S_MAX)
        and (num_chunks * 64 <= S_MAX)
    ):
        raise ValueError('invalid dense CP dimensions')
    packed = meta['layout_qkv'] == 'TND'
    if meta['layout_qkv'] not in ('BSND', 'BNSD', 'TND'):
        raise ValueError('unsupported layout')
    expected_nc = sum(((n + 63) // 64 for n in cp_meta['sequence_lengths'])) if packed else (seq_len + 63) // 64
    if num_chunks != expected_nc or (packed and batch_size != 1):
        raise ValueError('cache chunk metadata mismatch')
    if 'cp_meta' in cache:
        for key in ('cp_rank', 'cp_size', 'first_continues', 'last_continues'):
            if key in cache['cp_meta'] and cp_meta.get(key) != cache['cp_meta'][key]:
                raise ValueError('CP metadata changed after forward')
    device = cache['q_c'].device
    num_sequences = len(cp_meta['sequence_lengths']) if packed else batch_size
    shapes = {
        'q_c': ((batch_size, num_chunks, meta['Nk'], 64, 128), torch.bfloat16),
        'k_c': ((batch_size, num_chunks, meta['Nk'], 64, 128), torch.bfloat16),
        'v_c': ((batch_size, num_chunks, num_heads, 64, 128), torch.bfloat16),
        'gk_c': ((batch_size, num_chunks, num_heads, 64, 128), torch.float32),
        'beta_c': ((batch_size, num_chunks, num_heads, 64), torch.float32),
        'akk': ((batch_size, num_chunks, num_heads, 64, 64), torch.bfloat16),
        'h': (
            (batch_size, num_chunks + num_sequences if packed else num_chunks + 1, num_heads, 128, 128),
            torch.bfloat16,
        ),
        'v_new': ((batch_size, num_chunks, num_heads, 128, 64), torch.bfloat16),
        'mqk': ((batch_size, num_heads, num_chunks, 64, 64), torch.bfloat16),
        'w': (
            (
                (batch_size, num_heads, num_chunks * 64, 128)
                if meta['layout_qkv'] == 'BNSD'
                else (batch_size, num_chunks * 64, num_heads, 128)
            ),
            torch.bfloat16,
        ),
    }
    for key, (shape, dtype) in shapes.items():
        value = cache[key]
        _check_tensor(key, value, dtype)
        if tuple(value.shape) != shape or value.device != device:
            raise ValueError(f'{key} cache shape/device mismatch')
    return (meta, batch_size, seq_len, num_heads, num_chunks)


def _fragment_args(meta, cp_meta, device):
    packed = meta['layout_qkv'] == 'TND'
    if packed:
        num_sequences = len(cp_meta['sequence_lengths'])
        for key in ('cu_seqlens', 'chunk_offsets', 'h_offsets'):
            x = cp_meta[key]
            if (
                x.shape != (num_sequences + 1,)
                or x.dtype != torch.int64
                or x.device != device
                or (not x.is_contiguous())
            ):
                raise ValueError('packed metadata must be contiguous device int64')
        return (
            1,
            len(cp_meta['sequence_lengths']),
            cp_meta['cu_seqlens'],
            cp_meta['chunk_offsets'],
            cp_meta['h_offsets'],
        )
    empty = torch.empty((1,), dtype=torch.int64, device=device)
    return (0, int(meta['B']), empty, empty, empty)


def _view4(x, packed):
    return x.unsqueeze(0) if packed and x is not None else x


def flash_kda_bwd_stage1_wrapper(
    train_cache, do, scale, d_final_state=None, *, cp_meta, num_cores=None, param_grad_buffer=None
):
    meta, batch_size, seq_len, num_heads, num_chunks = _cp_dense_meta(train_cache, cp_meta)
    q, k, g, m, w = (train_cache[x] for x in ('q_c', 'k_c', 'gk_c', 'mqk', 'w'))
    if float(scale) != meta['scale']:
        raise ValueError('scale differs from forward')
    device = q.device
    fragments = _fragment_args(meta, cp_meta, device)
    packed = bool(fragments[0])
    do = _view4(do, packed)
    if do is not None:
        _check_strided('do', do, torch.bfloat16, device)
        _check_layout_shape(do, 'BSND' if packed else meta['layout_qkv'], batch_size, seq_len, num_heads)
    has_do = do is not None
    if do is None:
        do = torch.empty((1, 1, 1, 1), dtype=torch.bfloat16, device=device)
    has_dht = d_final_state is not None
    if has_dht:
        _check_strided('d_final_state', d_final_state, torch.float32, device)
        if d_final_state.stride(-2) != 128:
            raise ValueError('d_final_state matrix must be contiguous')
        if tuple(d_final_state.shape) != (fragments[1], num_heads, 128, 128):
            raise ValueError('d_final_state shape mismatch')
    else:
        d_final_state = torch.empty((1, 1, 1, 1), dtype=torch.float32, device=device)
    packet = torch.empty((batch_size, num_heads, 2, 128, 128), dtype=torch.float32, device=device)
    params = param_grad_buffer
    if params is None:
        params = torch.empty((num_heads + num_heads * 128,), dtype=torch.float32, device=device)
    _check_tensor('param_grad_buffer', params, torch.float32)
    if params.device != device:
        raise ValueError('parameter buffer device mismatch')
    da, dbias = _param_views(params, meta)
    last = bool(cp_meta.get('last_continues', cp_meta['cp_rank'] + 1 <
                cp_meta['cp_size'])) and (not packed or fragments[1] == 1)
    cores = _resolve_num_cores(device, num_cores)
    # Optional private intermediate; the public packet still contains exactly E/R.
    local_eligible = (cp_meta['cp_size'] == 1 and not packed and seq_len > 1
        and int(meta['Nk']) == num_heads and not last
        and not bool(cp_meta.get('first_continues', False)))
    bf16_eligible = (local_eligible and has_do and meta['layout_qkv'] == 'BSND' and seq_len % 64 == 0
        and -5.0 <= float(meta['lower_bound']) < 0.0 and do.is_contiguous())
    # Budget the actual DU representation before selecting the cached path.
    du_bytes = 2 if bf16_eligible else 4
    cache_bytes = batch_size * num_heads * ((num_chunks + 1) * 16384 * 2 + num_chunks * 8192 * du_bytes)
    cache_limit = 4096 * 1024 * 1024 if bf16_eligible else 512 * 1024 * 1024
    use_local = local_eligible and cache_bytes <= cache_limit
    local_state = torch.empty((batch_size * (num_chunks + 1) * num_heads, 1, 128, 128) if use_local else (1,
                              1, 1, 1), dtype=torch.bfloat16, device=device)
    use_du_bf16 = use_local and bf16_eligible
    local_du = torch.empty(
        (batch_size * num_chunks * num_heads, 64, 128) if use_local and not use_du_bf16 else (1, 1, 1),
        dtype=torch.float32,
        device=device,
    )
    local_du_bf16 = torch.empty((batch_size * num_chunks * num_heads, 64, 128) if use_du_bf16 else (1, 1, 1),
                                dtype=torch.bfloat16, device=device)
    with pypto.options(pass_options={'enable_slice': False}):
        chunk_kda_bwd_stage1_kernel[None, cores](
            q,
            k,
            g,
            m,
            w,
            do,
            d_final_state,
            batch_size,
            seq_len,
            num_heads,
            num_chunks,
            float(scale),
            _layout_code('BSND' if packed else meta['layout_qkv']),
            int(has_dht),
            da,
            dbias,
            1,
            packet,
            int(has_do),
            int(last),
            *fragments,
            int(meta['Nk']),
            *_stride_args(do, meta['layout_qkv']),
            *_stride_args(d_final_state, meta['layout_qkv'], True),
            *_recurrent_cache_args(train_cache, d_final_state),
            local_state,
            local_du,
            int(use_local),
            local_du_bf16,
            int(use_du_bf16),
        )
    if use_du_bf16:
        packet._kda_local_recurrence_all_bf16 = (local_state, local_du_bf16)
    elif use_local:
        packet._kda_local_recurrence_kv_bf16 = (local_state, local_du)
    return (packet, params)


def flash_kda_bwd_stage2_wrapper(
    q_raw,
    k_raw,
    g_raw,
    a_log,
    dt_bias,
    train_cache,
    do,
    all_bwd_summary,
    param_grad_buffer,
    scale,
    lower_bound,
    d_final_state=None,
    *,
    cp_meta,
    num_cores=None,
    output_buffers=None,
):
    meta, batch_size, seq_len, num_heads, num_chunks = _cp_dense_meta(train_cache, cp_meta)
    device = train_cache['q_c'].device
    fragments = _fragment_args(meta, cp_meta, device)
    packed = bool(fragments[0])
    q_raw, k_raw, g_raw, do = (_view4(x, packed) for x in (q_raw, k_raw, g_raw, do))
    if float(scale) != meta['scale'] or float(lower_bound) != meta['lower_bound']:
        raise ValueError('backward configuration differs from forward')
    rank, size = (int(cp_meta['cp_rank']), int(cp_meta['cp_size']))
    if not 0 <= rank < size:
        raise ValueError('invalid CP rank/size')
    if tuple(all_bwd_summary.shape) != (size * batch_size, num_heads, 2, 128, 128):
        raise ValueError('gathered summary shape mismatch')
    values = [
        ('q_raw', q_raw, torch.bfloat16),
        ('k_raw', k_raw, torch.bfloat16),
        ('g_raw', g_raw, torch.bfloat16),
        ('a_log', a_log, torch.float32),
        ('dt_bias', dt_bias, torch.float32),
        ('all_bwd_summary', all_bwd_summary, torch.float32),
        ('param_grad_buffer', param_grad_buffer, torch.float32),
    ]
    if do is not None:
        values.append(('do', do, torch.bfloat16))
    if d_final_state is not None:
        values.append(('d_final_state', d_final_state, torch.float32))
    for name, value, dtype in values:
        if name in ('q_raw', 'k_raw', 'g_raw', 'do', 'd_final_state'):
            _check_strided(name, value, dtype, device)
        else:
            _check_tensor(name, value, dtype)
        if value.device != device:
            raise ValueError(f'{name} device mismatch')
    layout = 'BSND' if packed else meta['layout_qkv']
    _check_layout_shape(q_raw, layout, batch_size, seq_len, meta['Nk'])
    _check_layout_shape(k_raw, layout, batch_size, seq_len, meta['Nk'])
    _check_layout_shape(g_raw, layout, batch_size, seq_len, num_heads)
    if do is not None:
        _check_layout_shape(do, layout, batch_size, seq_len, num_heads)
    if tuple(a_log.shape) != (num_heads,) or tuple(dt_bias.shape) != (num_heads, 128):
        raise ValueError('parameter shape mismatch')
    if d_final_state is not None and d_final_state.shape != (fragments[1], num_heads, 128, 128):
        raise ValueError('d_final_state shape mismatch')
    if d_final_state is not None and d_final_state.stride(-2) != 128:
        raise ValueError('d_final_state matrix must be contiguous')
    has_do = do is not None
    if do is None:
        do = torch.empty((1, 1, 1, 1), dtype=torch.bfloat16, device=device)
    has_dht = d_final_state is not None
    if d_final_state is None:
        d_final_state = torch.empty((1, 1, 1, 1), dtype=torch.float32, device=device)
    shape = (
        (batch_size, seq_len, num_heads, 128)
        if meta['layout_qkv'] in ('BSND', 'TND')
        else (batch_size, num_heads, seq_len, 128)
    )
    beta_shape = shape[:-1]
    qshape = (
        (batch_size, seq_len, int(meta['Nk']), 128)
        if meta['layout_qkv'] in ('BSND', 'TND')
        else (batch_size, int(meta['Nk']), seq_len, 128)
    )
    bindings = {} if output_buffers is None else dict(output_buffers)
    if set(bindings) - {'dq', 'dk', 'dv', 'dbeta', 'dg', 'dh_initial_state'}:
        raise ValueError('unknown output binding')
    originals = dict(bindings)
    if packed:
        bindings = {key: _view4(value, True) if key != 'dh_initial_state' else value for key, value in bindings.items()}
    live = [x for x in train_cache.values() if isinstance(x, torch.Tensor)] + [x for _, x, _ in values]

    def allocate(name, sh, dtype=torch.bfloat16, state=False):
        x = _bind_output(bindings, name, sh, dtype, device, live, state)
        live.append(x)
        return x
    dq, dk = (allocate(name, qshape) for name in ('dq', 'dk'))
    dv, dg = (allocate(name, shape) for name in ('dv', 'dg'))
    db = allocate('dbeta', beta_shape)
    dh0 = allocate('dh_initial_state', (fragments[1], num_heads, 128, 128), torch.float32, True)
    cores = _resolve_num_cores(device, num_cores)
    work = torch.empty(cores * _STAGE2_WORKSPACE_BYTES_PER_CORE, dtype=torch.uint8, device=device)
    owned_eligible = (not packed and has_do and layout == 'BSND' and seq_len % 64 == 0
        and -5.0 <= lower_bound < 0.0
        and all(x.is_contiguous() for x in (q_raw, k_raw, g_raw, do, dq, dk, dv, db, dg, dh0)))
    local_recurrence = getattr(all_bwd_summary, '_kda_local_recurrence_kv_bf16', None) if size == 1 else None
    local_bf16 = getattr(all_bwd_summary, '_kda_local_recurrence_all_bf16',
                         None) if size == 1 and owned_eligible else None
    du_cache_bf16 = local_bf16 is not None
    has_local_cache = du_cache_bf16 or local_recurrence is not None
    cached_state = torch.empty((1, 1, 1, 1), dtype=torch.bfloat16, device=device)
    cached_du_bf16 = torch.empty((1, 1, 1), dtype=torch.bfloat16, device=device)
    if du_cache_bf16:
        cached_state, cached_du_bf16 = local_bf16
        states = torch.empty((1, 1, 1, 1), dtype=torch.float32, device=device)
        du = torch.empty((1, 1, 1), dtype=torch.float32, device=device)
    elif has_local_cache:
        cached_state, du = local_recurrence
        states = torch.empty((1, 1, 1, 1), dtype=torch.float32, device=device)
    else:
        # A strided output, cloned packet or other fallback must retain the FP32
        # recurrence path; BF16 DU is private to the owned-head fast route.
        states = torch.empty((cores * (num_heads // meta['Nk']), 2, 128, 128), dtype=torch.float32, device=device)
        du = torch.empty((cores, 64, 128), dtype=torch.float32, device=device)
    qk_sum = torch.empty((cores, 2, 64, 128), dtype=torch.float32, device=device)
    da, dbias = _param_views(param_grad_buffer, meta)
    first = bool(cp_meta.get('first_continues', rank > 0))
    last = bool(cp_meta.get('last_continues', rank + 1 < size))
    owned_heads = has_local_cache and owned_eligible
    qk_rstd = train_cache.get('qk_rstd')
    has_qk_rstd = owned_heads and qk_rstd is not None
    if has_qk_rstd:
        if (
            qk_rstd.shape != (batch_size, num_chunks, num_heads, 2, 64)
            or qk_rstd.dtype != torch.float32
            or qk_rstd.device != device
            or not qk_rstd.is_contiguous()
        ):
            raise ValueError('qk_rstd must be contiguous FP32 [B,NC,N,2,64] on input device')
    else:
        qk_rstd = torch.empty((1, 1, 1, 1, 1), dtype=torch.float32, device=device)
    with pypto.options(pass_options={'enable_slice': False}):
        chunk_kda_bwd_stage2_kernel[
            None, cores, {"LocalTile": 32 if -5.0 <= lower_bound < 0.0 else 16, "OwnedHeads": int(owned_heads)}
        ](
            *(train_cache[x] for x in ('q_c', 'k_c', 'v_c', 'gk_c', 'beta_c', 'akk', 'h', 'v_new')),
            do,
            dq,
            dk,
            dv,
            db,
            dg,
            work,
            batch_size,
            seq_len,
            num_heads,
            num_chunks,
            num_chunks + 1,
            float(scale),
            _layout_code('BSND' if packed else meta['layout_qkv']),
            int(seq_len == 1),
            q_raw,
            k_raw,
            1,
            seq_len,
            g_raw,
            a_log,
            dt_bias,
            da,
            dbias,
            float(lower_bound),
            1,
            train_cache['mqk'],
            train_cache['w'],
            d_final_state,
            all_bwd_summary,
            states,
            du,
            dh0,
            int(has_do),
            int(has_dht),
            rank,
            size,
            int(first),
            int(last),
            *fragments,
            int(meta['Nk']),
            qk_sum,
            *_stride_args(do, layout),
            *_stride_args(d_final_state, layout, True),
            *_stride_args(q_raw, layout),
            *_stride_args(k_raw, layout),
            *_stride_args(g_raw, layout),
            *_stride_args(dq, layout),
            *_stride_args(dk, layout),
            *_stride_args(dv, layout),
            *_stride_args(db, layout),
            *_stride_args(dg, layout),
            *_stride_args(dh0, layout, True),
            int(has_local_cache),
            cached_state,
            cached_du_bf16,
            int(du_cache_bf16),
            qk_rstd,
            int(has_qk_rstd),
        )
    result = (dq, dk, dv, db, dg)
    result = tuple((x[0] for x in result)) + (dh0,) if packed else result + (dh0,)
    return tuple(
        (
            originals.get(name, x)
            for name, x in zip(('dq', 'dk', 'dv', 'dbeta', 'dg', 'dh_initial_state'), result)
        )
    )


@pl.vector_function
def cp_l2_backward(raw, grad, out, rows, ratio):
    """Full 128-feature rows; preserve the core BF16 gradient boundary."""
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pb = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    for r in pl.range(0, rows):
        xb0 = vf.load_align(raw, r * 128, dist=pl.LoadDist.US_B16)
        xb1 = vf.load_align(raw, r * 128 + 64, dist=pl.LoadDist.US_B16)
        x0 = vf.astype(xb0, pb, dtype=pl.DT_FP32)
        x1 = vf.astype(xb1, pb, dtype=pl.DT_FP32)
        d0 = vf.load_align(grad, r * 128)
        d1 = vf.load_align(grad, r * 128 + 64)
        if ratio == 1:
            db0 = vf.astype(d0, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            db1 = vf.astype(d1, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            d0 = vf.astype(db0, pb, dtype=pl.DT_FP32)
            d1 = vf.astype(db1, pb, dtype=pl.DT_FP32)
        xx0 = vf.mul(x0, x0, p)
        xx1 = vf.mul(x1, x1, p)
        norm0 = vf.reduce_sum(xx0, p)
        norm1 = vf.reduce_sum(xx1, p)
        norm = vf.add(norm0, norm1, p)
        norm = vf.full(norm, p)
        norm = vf.adds(norm, 1e-06, p)
        denom = vf.sqrt(norm, p)
        dx0 = vf.mul(d0, x0, p)
        dx1 = vf.mul(d1, x1, p)
        dot0 = vf.reduce_sum(dx0, p)
        dot1 = vf.reduce_sum(dx1, p)
        dot = vf.add(dot0, dot1, p)
        dot = vf.full(dot, p)
        cube = vf.mul(norm, denom, p)
        corr0 = vf.mul(x0, dot, p)
        corr1 = vf.mul(x1, dot, p)
        corr0 = vf.div(corr0, cube, p)
        corr1 = vf.div(corr1, cube, p)
        y0 = vf.div(d0, denom, p)
        y1 = vf.div(d1, denom, p)
        y0 = vf.sub(y0, corr0, p)
        y1 = vf.sub(y1, corr1, p)
        b0 = vf.astype(y0, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
        b1 = vf.astype(y1, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
        packed, unused = vf.de_interleave(b0, b1)
        vf.store_align(out, packed, pb, r * 128)
