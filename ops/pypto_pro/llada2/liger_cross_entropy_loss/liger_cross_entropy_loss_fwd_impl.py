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
"""PyPTO-Pro liger_cross_entropy forward kernel implementation.

Forward of the Liger-Kernel fused cross-entropy loss, float32 logits.

The contract is `LigerCrossEntropyFunction.forward` from Liger-Kernel: one
streaming pass computes the per-row online-softmax statistics, a cross-core
rendezvous makes the global normalizers available to every lane, and a second
pass writes the gradient intermediate back into the logits in place -- the
memory-saving trick the upstream triton kernel uses, which is what lets the
backward be a single scaling.

Formula, all intermediates in float32, for each row r of x[BT, V] with class
weights w[V]:

    x_cap = softcap * tanh(x / softcap)          (identity when disabled)
    m     = max(x_cap)   d = sum(exp(x_cap - m))   lse = m + ln(d)
    loss  = ((lse - x_cap[y]) * w[y]) * (1 - ls)
            + (-eps * sum(x_cap * w) + eps * lse * sum(w))
    z     = lse_square_scale * lse^2
    loss  = loss / D_loss + z / D_z               (mean only; else D = 1)
    dx    = A * exp(x_cap - lse) - B * w,  dx[y] -= C
    dx   *= 1 - (x_cap / softcap)^2               (chain rule, 1 when disabled)

with eps = ls / V, D_loss = sum_non_ignore_weight and D_z = n_non_ignore. Rows
whose target equals `ignore_index` contribute zero everywhere and have their dx
row zeroed, matching the triton kernel's early return.

The unweighted path needs no separate branch: substituting w = 1 turns the
weighted formulas into the unweighted ones exactly, so the kernel keeps one
weighted code path and pre-fills its weight tile with ones.

Two pieces of the gradient pass run under a zero-or-one trip count rather than
unconditionally, because both are per-element and both dominate a pass that is
vector-bound rather than bandwidth-bound. The softcap chain rule is a provable
no-op when the cap is disabled (`inv_softcap` is zero, so the factor is one),
and the first-occurrence argmax feeds nothing but `token_accuracy` and
`predicted_tokens`. A caller that asks for those pays exactly what it did
before; one that wants only the loss and the gradient does not.

Both streaming passes carry their GM tiles in rotating buffers, so the load of
the next column tile overlaps the vector work on the current one. Everything
downstream of the widen is UB-to-register and stays single-slot -- the row
accumulator serializes the vector work anyway, so only the transfers are worth
hiding.

The kernel body is machine-generated. Its source of truth is an EasyASC kernel
translated by `easyasc.targets.pypto`; regenerating it is described in the
README beside this file.
"""

import torch

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import get_platform_info

# The runner's older PyPTO build leaves gaps in its DT_* pybind
# coverage: some constants (DT_BF16 among them) arrive as bare ints
# that VF-layer kwarg checks reject ("expected ir::DataType, but got
# int"), while others on the same build are real DataType objects
# (add_rms_norm_dynamic_quant exercised FP16/FP32/INT8/INT16/UINT32
# there). Rebuild the broken ones from the enum value AT MODULE
# SCOPE -- the VF parser inlines helper calls and rejects
# isinstance, so the fixups must be plain names by the time a
# decorated body mentions them. A healthy build passes through.
_DTT = type(pl.DT_FP32)


def _dt_fix(v):
    return v if isinstance(v, _DTT) else _DTT(v)


_PL_BF16 = _dt_fix(pl.DT_BF16)
_PL_FP16 = _dt_fix(pl.DT_FP16)
_PL_FP32 = _dt_fix(pl.DT_FP32)
_PL_INT8 = _dt_fix(pl.DT_INT8)
_PL_INT16 = _dt_fix(pl.DT_INT16)
_PL_INT32 = _dt_fix(pl.DT_INT32)
_PL_INT64 = _dt_fix(pl.DT_INT64)
_PL_UINT8 = _dt_fix(pl.DT_UINT8)
_PL_UINT16 = _dt_fix(pl.DT_UINT16)
_PL_UINT32 = _dt_fix(pl.DT_UINT32)
_PL_UINT64 = _dt_fix(pl.DT_UINT64)


@pl.vector_function
def argmax_reset_vf(acc):
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lce_ar_big = vf.full(3e+38, _tmp_maskreg_0, dtype=_PL_FP32)
    vf.store_align(acc + 0, lce_ar_big, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def fill_ones_vf(dst, groups):
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lce_fo_one = vf.full(1.0, _tmp_maskreg_0, dtype=_PL_FP32)
    for lce_fo_group in pl.range(0, groups, 1):
        vf.store_align(dst + (lce_fo_group * 64), lce_fo_one, _tmp_maskreg_0)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def glob_finish_vf(glob, cols, has_weight_f, is_mean):
    _zero_int32 = vf.full(0, dtype=_PL_INT32)
    _cm0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    _div_sign = vf.full(-2147483648, _cm0, dtype=_PL_INT32)
    _div_expmask = vf.full(2139095040, _cm0, dtype=_PL_INT32)
    _div_bias = vf.full(127, _cm0, dtype=_PL_INT32)
    _div_thr = vf.full(-64, _cm0, dtype=_PL_INT32)
    _zero_fp32 = vf.full(0, dtype=_PL_FP32)
    _div_one = vf.full(1.0, _cm0, dtype=_PL_FP32)
    _tmp_maskreg_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lce_gf_all_f = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lce_gf_on = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=_PL_FP32)
    lce_gf_one = vf.full(1.0, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_gf_count = vf.load_align(glob, 0, dist=pl.LoadDist.BRC_B32)
    lce_gf_sum_w = vf.load_align(glob, 1, dist=pl.LoadDist.BRC_B32)
    lce_gf_weight_sum = vf.load_align(glob, 2, dist=pl.LoadDist.BRC_B32)
    lce_gf_cols_i = vf.full(cols, _tmp_maskreg_1, dtype=_PL_INT32)
    lce_gf_cols_f = vf.astype(lce_gf_cols_i, _tmp_maskreg_1, dtype=_PL_FP32, layout=pl.CastLayout.ZERO)
    lce_gf_cols_f = vf.full(lce_gf_cols_f, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_gf_flag = vf.full(has_weight_f, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_gf_on = vf.gt(lce_gf_flag, 0.5, lce_gf_all_f)
    lce_gf_weight_sum = vf.select(lce_gf_weight_sum, lce_gf_cols_f, lce_gf_on)
    lce_gf_sum_w = vf.select(lce_gf_sum_w, lce_gf_count, lce_gf_on)
    _t1 = vf.div(lce_gf_one, lce_gf_count, _tmp_maskreg_0)
    _t2 = vf.bit_cast(_t1, dtype=_PL_INT32)
    _t3 = vf.or_(_t2, _div_sign, _tmp_maskreg_0)
    _t4 = vf.eq(_t1, 0.0, _tmp_maskreg_0)
    _t5 = vf.ge(_t3, -8388608, _tmp_maskreg_0)
    _t5 = vf.or_(_t5, _t4, _tmp_maskreg_0)
    _t6 = vf.bit_cast(lce_gf_one, dtype=_PL_INT32)
    _t7 = vf.and_(_t6, _div_expmask, _tmp_maskreg_0)
    _t7 = vf.shift_right(_t7, 23, _tmp_maskreg_0)
    _t8 = vf.sub(_t7, _div_bias, _tmp_maskreg_0)
    _t9 = vf.lt(_t8, -64, _tmp_maskreg_0)
    _t10 = vf.sub(_div_thr, _t8, _tmp_maskreg_0)
    _t10 = vf.max(_t10, _zero_int32, _tmp_maskreg_0)
    _t10 = vf.adds(_t10, 127, _t9)
    _t10 = vf.shift_left(_t10, 23, _t9)
    _t12 = vf.bit_cast(_t10, dtype=_PL_FP32)
    _t11 = vf.select(_t12, _div_one, _t9)
    _t13 = vf.mul(lce_gf_one, _t11, _tmp_maskreg_0)
    _t14 = vf.mul(lce_gf_count, _t11, _tmp_maskreg_0)
    _t15 = vf.muls(_t14, -1.0, _tmp_maskreg_0)
    _t16 = vf.adds(_t2, -1, _tmp_maskreg_0)
    _t17 = vf.adds(_t2, 1, _tmp_maskreg_0)
    _t18 = vf.bit_cast(_t16, dtype=_PL_FP32)
    _t19 = vf.bit_cast(_t17, dtype=_PL_FP32)
    _t20 = vf.move(_t13, _tmp_maskreg_0)
    _t20 = vf.mul_add_dst(_t1, _t15, _tmp_maskreg_0)
    _t20 = vf.abs(_t20, _tmp_maskreg_0)
    _t21 = vf.move(_t13, _tmp_maskreg_0)
    _t21 = vf.mul_add_dst(_t18, _t15, _tmp_maskreg_0)
    _t21 = vf.abs(_t21, _tmp_maskreg_0)
    _t22 = vf.move(_t13, _tmp_maskreg_0)
    _t22 = vf.mul_add_dst(_t19, _t15, _tmp_maskreg_0)
    _t22 = vf.abs(_t22, _tmp_maskreg_0)
    _t24 = vf.lt(_t20, _t21, _tmp_maskreg_0)
    _t20 = vf.select(_t20, _t21, _t24)
    _t23 = vf.select(_t1, _t18, _t24)
    _t25 = vf.lt(_t22, _t20, _tmp_maskreg_0)
    _t23 = vf.select(_t19, _t23, _t25)
    lce_gf_inv_n = vf.select(_t1, _t23, _t5)
    lce_gf_flag = vf.full(is_mean, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_gf_on = vf.gt(lce_gf_flag, 0.5, lce_gf_all_f)
    lce_gf_denom = vf.select(lce_gf_sum_w, lce_gf_one, lce_gf_on)
    _t26 = vf.div(lce_gf_one, lce_gf_denom, _tmp_maskreg_0)
    _t27 = vf.bit_cast(_t26, dtype=_PL_INT32)
    _t28 = vf.or_(_t27, _div_sign, _tmp_maskreg_0)
    _t29 = vf.eq(_t26, 0.0, _tmp_maskreg_0)
    _t30 = vf.ge(_t28, -8388608, _tmp_maskreg_0)
    _t30 = vf.or_(_t30, _t29, _tmp_maskreg_0)
    _t31 = vf.bit_cast(lce_gf_one, dtype=_PL_INT32)
    _t32 = vf.and_(_t31, _div_expmask, _tmp_maskreg_0)
    _t32 = vf.shift_right(_t32, 23, _tmp_maskreg_0)
    _t33 = vf.sub(_t32, _div_bias, _tmp_maskreg_0)
    _t34 = vf.lt(_t33, -64, _tmp_maskreg_0)
    _t35 = vf.sub(_div_thr, _t33, _tmp_maskreg_0)
    _t35 = vf.max(_t35, _zero_int32, _tmp_maskreg_0)
    _t35 = vf.adds(_t35, 127, _t34)
    _t35 = vf.shift_left(_t35, 23, _t34)
    _t37 = vf.bit_cast(_t35, dtype=_PL_FP32)
    _t36 = vf.select(_t37, _div_one, _t34)
    _t38 = vf.mul(lce_gf_one, _t36, _tmp_maskreg_0)
    _t39 = vf.mul(lce_gf_denom, _t36, _tmp_maskreg_0)
    _t40 = vf.muls(_t39, -1.0, _tmp_maskreg_0)
    _t41 = vf.adds(_t27, -1, _tmp_maskreg_0)
    _t42 = vf.adds(_t27, 1, _tmp_maskreg_0)
    _t43 = vf.bit_cast(_t41, dtype=_PL_FP32)
    _t44 = vf.bit_cast(_t42, dtype=_PL_FP32)
    _t45 = vf.move(_t38, _tmp_maskreg_0)
    _t45 = vf.mul_add_dst(_t26, _t40, _tmp_maskreg_0)
    _t45 = vf.abs(_t45, _tmp_maskreg_0)
    _t46 = vf.move(_t38, _tmp_maskreg_0)
    _t46 = vf.mul_add_dst(_t43, _t40, _tmp_maskreg_0)
    _t46 = vf.abs(_t46, _tmp_maskreg_0)
    _t47 = vf.move(_t38, _tmp_maskreg_0)
    _t47 = vf.mul_add_dst(_t44, _t40, _tmp_maskreg_0)
    _t47 = vf.abs(_t47, _tmp_maskreg_0)
    _t49 = vf.lt(_t45, _t46, _tmp_maskreg_0)
    _t45 = vf.select(_t45, _t46, _t49)
    _t48 = vf.select(_t26, _t43, _t49)
    _t50 = vf.lt(_t47, _t45, _tmp_maskreg_0)
    _t48 = vf.select(_t44, _t48, _t50)
    lce_gf_inv = vf.select(_t26, _t48, _t30)
    vf.store_align(glob + 3, lce_gf_inv, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    lce_gf_denom = vf.select(lce_gf_count, lce_gf_one, lce_gf_on)
    _t51 = vf.div(lce_gf_one, lce_gf_denom, _tmp_maskreg_0)
    _t52 = vf.bit_cast(_t51, dtype=_PL_INT32)
    _t53 = vf.or_(_t52, _div_sign, _tmp_maskreg_0)
    _t54 = vf.eq(_t51, 0.0, _tmp_maskreg_0)
    _t55 = vf.ge(_t53, -8388608, _tmp_maskreg_0)
    _t55 = vf.or_(_t55, _t54, _tmp_maskreg_0)
    _t56 = vf.bit_cast(lce_gf_one, dtype=_PL_INT32)
    _t57 = vf.and_(_t56, _div_expmask, _tmp_maskreg_0)
    _t57 = vf.shift_right(_t57, 23, _tmp_maskreg_0)
    _t58 = vf.sub(_t57, _div_bias, _tmp_maskreg_0)
    _t59 = vf.lt(_t58, -64, _tmp_maskreg_0)
    _t60 = vf.sub(_div_thr, _t58, _tmp_maskreg_0)
    _t60 = vf.max(_t60, _zero_int32, _tmp_maskreg_0)
    _t60 = vf.adds(_t60, 127, _t59)
    _t60 = vf.shift_left(_t60, 23, _t59)
    _t62 = vf.bit_cast(_t60, dtype=_PL_FP32)
    _t61 = vf.select(_t62, _div_one, _t59)
    _t63 = vf.mul(lce_gf_one, _t61, _tmp_maskreg_0)
    _t64 = vf.mul(lce_gf_denom, _t61, _tmp_maskreg_0)
    _t65 = vf.muls(_t64, -1.0, _tmp_maskreg_0)
    _t66 = vf.adds(_t52, -1, _tmp_maskreg_0)
    _t67 = vf.adds(_t52, 1, _tmp_maskreg_0)
    _t68 = vf.bit_cast(_t66, dtype=_PL_FP32)
    _t69 = vf.bit_cast(_t67, dtype=_PL_FP32)
    _t70 = vf.move(_t63, _tmp_maskreg_0)
    _t70 = vf.mul_add_dst(_t51, _t65, _tmp_maskreg_0)
    _t70 = vf.abs(_t70, _tmp_maskreg_0)
    _t71 = vf.move(_t63, _tmp_maskreg_0)
    _t71 = vf.mul_add_dst(_t68, _t65, _tmp_maskreg_0)
    _t71 = vf.abs(_t71, _tmp_maskreg_0)
    _t72 = vf.move(_t63, _tmp_maskreg_0)
    _t72 = vf.mul_add_dst(_t69, _t65, _tmp_maskreg_0)
    _t72 = vf.abs(_t72, _tmp_maskreg_0)
    _t74 = vf.lt(_t70, _t71, _tmp_maskreg_0)
    _t70 = vf.select(_t70, _t71, _t74)
    _t73 = vf.select(_t51, _t68, _t74)
    _t75 = vf.lt(_t72, _t70, _tmp_maskreg_0)
    _t73 = vf.select(_t69, _t73, _t75)
    lce_gf_inv = vf.select(_t51, _t73, _t55)
    vf.store_align(glob + 4, lce_gf_inv, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.store_align(glob + 2, lce_gf_weight_sum, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.store_align(glob + 1, lce_gf_sum_w, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.store_align(glob + 5, lce_gf_inv_n, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def glob_reset_vf(glob):
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lce_gr_zero = vf.full(0.0, _tmp_maskreg_0, dtype=_PL_FP32)
    vf.store_align(glob + 0, lce_gr_zero, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.store_align(glob + 1, lce_gr_zero, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.store_align(glob + 2, lce_gr_zero, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def narrow_float_vf(src, dst, back, valid, groups):
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    for lce_nf_group in pl.range(0, groups, 1):
        lce_nf_active = vf.update_mask(pl.max(0, valid - lce_nf_group * 64), dtype=_PL_FP32)
        _m0 = (lce_nf_group * 64)
        lce_nf_value = vf.load_align(src, _m0)
        vf.store_align(dst + _m0, lce_nf_value, _tmp_maskreg_0)
        vf.store_align(back + _m0, lce_nf_value, _tmp_maskreg_0)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def red_reset_vf(red):
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lce_rr0_zero = vf.full(0.0, _tmp_maskreg_0, dtype=_PL_FP32)
    vf.store_align(red + 0, lce_rr0_zero, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.store_align(red + 1, lce_rr0_zero, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.store_align(red + 2, lce_rr0_zero, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.store_align(red + 3, lce_rr0_zero, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def reduce_scale_vf(glob, red):
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lce_rs_all_f = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lce_rs_inv_n = vf.load_align(glob, 5, dist=pl.LoadDist.BRC_B32)
    lce_rs_count = vf.load_align(glob, 0, dist=pl.LoadDist.BRC_B32)
    lce_rs_total = vf.load_align(red, 2, dist=pl.LoadDist.BRC_B32)
    lce_rs_total = vf.mul(lce_rs_total, lce_rs_inv_n, lce_rs_all_f)
    vf.store_align(red + 2, lce_rs_total, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.store_align(red + 3, lce_rs_count, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def row_prep_vf(
    s_lse,
    s_xy,
    s_wy,
    s_sxw,
    s_valid,
    glob,
    o_a,
    o_b,
    o_c,
    loss_f,
    zloss_f,
    rows,
    chunks,
    eps_value,
    one_minus_ls,
    z_scale,
):
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lce_rp_all_f = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lce_rp_keep = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=_PL_FP32)
    lce_rp_zero = vf.full(0.0, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_rp_inv_dl = vf.load_align(glob, 3, dist=pl.LoadDist.BRC_B32)
    lce_rp_inv_dz = vf.load_align(glob, 4, dist=pl.LoadDist.BRC_B32)
    lce_rp_weight_sum = vf.load_align(glob, 2, dist=pl.LoadDist.BRC_B32)
    for lce_rp_chunk in pl.range(0, chunks, 1):
        lce_rp_active = vf.update_mask((rows - lce_rp_chunk * 64), dtype=_PL_FP32)
        _m0 = (lce_rp_chunk * 64)
        lce_rp_lse = vf.load_align(s_lse, _m0)
        lce_rp_xy = vf.load_align(s_xy, _m0)
        lce_rp_wy = vf.load_align(s_wy, _m0)
        lce_rp_sxw = vf.load_align(s_sxw, _m0)
        lce_rp_valid = vf.load_align(s_valid, _m0)
        lce_rp_loss = vf.sub(lce_rp_lse, lce_rp_xy, lce_rp_active)
        lce_rp_loss = vf.mul(lce_rp_loss, lce_rp_wy, lce_rp_active)
        lce_rp_loss = vf.muls(lce_rp_loss, one_minus_ls, lce_rp_active)
        lce_rp_smooth = vf.mul(lce_rp_lse, lce_rp_weight_sum, lce_rp_active)
        lce_rp_smooth = vf.sub(lce_rp_smooth, lce_rp_sxw, lce_rp_active)
        lce_rp_smooth = vf.muls(lce_rp_smooth, eps_value, lce_rp_active)
        lce_rp_loss = vf.add(lce_rp_loss, lce_rp_smooth, lce_rp_active)
        lce_rp_z = vf.mul(lce_rp_lse, lce_rp_lse, lce_rp_active)
        lce_rp_z = vf.muls(lce_rp_z, z_scale, lce_rp_active)
        lce_rp_z = vf.mul(lce_rp_z, lce_rp_inv_dz, lce_rp_active)
        lce_rp_loss = vf.mul(lce_rp_loss, lce_rp_inv_dl, lce_rp_active)
        lce_rp_loss = vf.add(lce_rp_loss, lce_rp_z, lce_rp_active)
        lce_rp_keep = vf.gt(lce_rp_valid, 0.5, lce_rp_all_f)
        lce_rp_loss = vf.select(lce_rp_loss, lce_rp_zero, lce_rp_keep)
        lce_rp_z = vf.select(lce_rp_z, lce_rp_zero, lce_rp_keep)
        vf.store_align(loss_f + _m0, lce_rp_loss, _tmp_maskreg_0)
        vf.store_align(zloss_f + _m0, lce_rp_z, _tmp_maskreg_0)
        lce_rp_coef_a = vf.muls(lce_rp_wy, one_minus_ls, lce_rp_active)
        lce_rp_scratch = vf.muls(lce_rp_weight_sum, eps_value, lce_rp_active)
        lce_rp_coef_a = vf.add(lce_rp_coef_a, lce_rp_scratch, lce_rp_active)
        lce_rp_coef_a = vf.mul(lce_rp_coef_a, lce_rp_inv_dl, lce_rp_active)
        lce_rp_scratch = vf.muls(lce_rp_lse, z_scale, lce_rp_active)
        lce_rp_scratch = vf.muls(lce_rp_scratch, 2.0, lce_rp_active)
        lce_rp_scratch = vf.mul(lce_rp_scratch, lce_rp_inv_dz, lce_rp_active)
        lce_rp_coef_a = vf.add(lce_rp_coef_a, lce_rp_scratch, lce_rp_active)
        lce_rp_coef_b = vf.muls(lce_rp_inv_dl, eps_value, lce_rp_active)
        lce_rp_coef_b = vf.muls(lce_rp_coef_b, -1.0, lce_rp_active)
        lce_rp_coef_c = vf.muls(lce_rp_wy, one_minus_ls, lce_rp_active)
        lce_rp_coef_c = vf.mul(lce_rp_coef_c, lce_rp_inv_dl, lce_rp_active)
        vf.store_align(o_a + _m0, lce_rp_coef_a, _tmp_maskreg_0)
        vf.store_align(o_b + _m0, lce_rp_coef_b, _tmp_maskreg_0)
        vf.store_align(o_c + _m0, lce_rp_coef_c, _tmp_maskreg_0)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def row_reset_vf(acc):
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lce_rr_zero = vf.full(0.0, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_rr_ninf = vf.full(-3e+38, _tmp_maskreg_0, dtype=_PL_FP32)
    vf.store_align(acc + 0, lce_rr_ninf, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.store_align(acc + 1, lce_rr_zero, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.store_align(acc + 2, lce_rr_zero, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.store_align(acc + 3, lce_rr_zero, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.store_align(acc + 4, lce_rr_zero, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def scan_add_vf(src, acc, slot, valid, groups):
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lce_sa_all_f = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lce_sa_zero = vf.full(0.0, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_sa_running = vf.full(0.0, _tmp_maskreg_0, dtype=_PL_FP32)
    for lce_sa_group in pl.range(0, groups, 1):
        lce_sa_active = vf.update_mask((valid - lce_sa_group * 64), dtype=_PL_FP32)
        lce_sa_value = vf.load_align(src, (lce_sa_group * 64))
        lce_sa_value = vf.select(lce_sa_value, lce_sa_zero, lce_sa_active)
        lce_sa_running = vf.add(lce_sa_running, lce_sa_value, lce_sa_all_f)
    lce_sa_total = vf.reduce_sum(lce_sa_running, lce_sa_all_f)
    lce_sa_carried = vf.load_align(acc, slot, dist=pl.LoadDist.BRC_B32)
    lce_sa_total = vf.add(lce_sa_total, lce_sa_carried, lce_sa_all_f)
    vf.store_align(acc + slot, lce_sa_total, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def softcap_tile_vf(cap, valid, groups, softcap, two_over_cap):
    _zero_int32 = vf.full(0, dtype=_PL_INT32)
    _cm0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    _div_sign = vf.full(-2147483648, _cm0, dtype=_PL_INT32)
    _div_expmask = vf.full(2139095040, _cm0, dtype=_PL_INT32)
    _div_bias = vf.full(127, _cm0, dtype=_PL_INT32)
    _div_thr = vf.full(-64, _cm0, dtype=_PL_INT32)
    _zero_fp32 = vf.full(0, dtype=_PL_FP32)
    _div_one = vf.full(1.0, _cm0, dtype=_PL_FP32)
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lce_sc_one = vf.full(1.0, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_sc_two = vf.full(2.0, _tmp_maskreg_0, dtype=_PL_FP32)
    for lce_sc_group in pl.range(0, groups, 1):
        lce_sc_active = vf.update_mask((valid - lce_sc_group * 64), dtype=_PL_FP32)
        _m0 = (lce_sc_group * 64)
        lce_sc_value = vf.load_align(cap, _m0)
        lce_sc_scaled = vf.muls(lce_sc_value, two_over_cap, lce_sc_active)
        lce_sc_expo = vf.exp(lce_sc_scaled, lce_sc_active)
        lce_sc_expo = vf.adds(lce_sc_expo, 1.0, lce_sc_active)
        _t1 = vf.div(lce_sc_two, lce_sc_expo, lce_sc_active)
        _t2 = vf.bit_cast(_t1, dtype=_PL_INT32)
        _t3 = vf.or_(_t2, _div_sign, lce_sc_active)
        _t4 = vf.eq(_t1, 0.0, lce_sc_active)
        _t5 = vf.ge(_t3, -8388608, lce_sc_active)
        _t5 = vf.or_(_t5, _t4, lce_sc_active)
        _t6 = vf.bit_cast(lce_sc_two, dtype=_PL_INT32)
        _t7 = vf.and_(_t6, _div_expmask, lce_sc_active)
        _t7 = vf.shift_right(_t7, 23, lce_sc_active)
        _t8 = vf.sub(_t7, _div_bias, lce_sc_active)
        _t9 = vf.lt(_t8, -64, lce_sc_active)
        _t10 = vf.sub(_div_thr, _t8, lce_sc_active)
        _t10 = vf.max(_t10, _zero_int32, lce_sc_active)
        _t10 = vf.adds(_t10, 127, _t9)
        _t10 = vf.shift_left(_t10, 23, _t9)
        _t12 = vf.bit_cast(_t10, dtype=_PL_FP32)
        _t11 = vf.select(_t12, _div_one, _t9)
        _t13 = vf.mul(lce_sc_two, _t11, lce_sc_active)
        _t14 = vf.mul(lce_sc_expo, _t11, lce_sc_active)
        _t15 = vf.muls(_t14, -1.0, lce_sc_active)
        _t16 = vf.adds(_t2, -1, lce_sc_active)
        _t17 = vf.adds(_t2, 1, lce_sc_active)
        _t18 = vf.bit_cast(_t16, dtype=_PL_FP32)
        _t19 = vf.bit_cast(_t17, dtype=_PL_FP32)
        _t20 = vf.move(_t13, lce_sc_active)
        _t20 = vf.mul_add_dst(_t1, _t15, lce_sc_active)
        _t20 = vf.abs(_t20, lce_sc_active)
        _t21 = vf.move(_t13, lce_sc_active)
        _t21 = vf.mul_add_dst(_t18, _t15, lce_sc_active)
        _t21 = vf.abs(_t21, lce_sc_active)
        _t22 = vf.move(_t13, lce_sc_active)
        _t22 = vf.mul_add_dst(_t19, _t15, lce_sc_active)
        _t22 = vf.abs(_t22, lce_sc_active)
        _t24 = vf.lt(_t20, _t21, lce_sc_active)
        _t20 = vf.select(_t20, _t21, _t24)
        _t23 = vf.select(_t1, _t18, _t24)
        _t25 = vf.lt(_t22, _t20, lce_sc_active)
        _t23 = vf.select(_t19, _t23, _t25)
        lce_sc_tanh = vf.select(_t1, _t23, _t5)
        lce_sc_tanh = vf.sub(lce_sc_one, lce_sc_tanh, lce_sc_active)
        lce_sc_value = vf.muls(lce_sc_tanh, softcap, lce_sc_active)
        vf.store_align(cap + _m0, lce_sc_value, _tmp_maskreg_0)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def stage1_row_finish_vf(acc, tgt, s_max, s_lse, s_xy, s_wy, s_sxw, s_valid, slot, ignore_value):
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lce_rf_all_f = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lce_rf_keep = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=_PL_FP32)
    lce_rf_one = vf.full(1.0, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_rf_zero = vf.full(0.0, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_rf_target_i = vf.load_align(tgt, slot, dist=pl.LoadDist.BRC_B32)
    lce_rf_target_f = vf.astype(lce_rf_target_i, _tmp_maskreg_0, dtype=_PL_FP32, layout=pl.CastLayout.ZERO)
    lce_rf_target_f = vf.full(lce_rf_target_f, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_rf_keep = vf.ne(lce_rf_target_f, ignore_value, lce_rf_all_f)
    lce_rf_valid = vf.select(lce_rf_one, lce_rf_zero, lce_rf_keep)
    lce_rf_row_max = vf.load_align(acc, 0, dist=pl.LoadDist.BRC_B32)
    lce_rf_total = vf.load_align(acc, 1, dist=pl.LoadDist.BRC_B32)
    lce_rf_xy = vf.load_align(acc, 2, dist=pl.LoadDist.BRC_B32)
    lce_rf_wy = vf.load_align(acc, 3, dist=pl.LoadDist.BRC_B32)
    lce_rf_sxw = vf.load_align(acc, 4, dist=pl.LoadDist.BRC_B32)
    lce_rf_lse = vf.ln(lce_rf_total, _tmp_maskreg_0)
    lce_rf_lse = vf.add(lce_rf_lse, lce_rf_row_max, lce_rf_all_f)
    lce_rf_wy = vf.mul(lce_rf_wy, lce_rf_valid, lce_rf_all_f)
    vf.store_align(s_max + slot, lce_rf_row_max, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.store_align(s_lse + slot, lce_rf_lse, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.store_align(s_xy + slot, lce_rf_xy, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.store_align(s_wy + slot, lce_rf_wy, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.store_align(s_sxw + slot, lce_rf_sxw, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.store_align(s_valid + slot, lce_rf_valid, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def stage1_tile_vf(cap, wtile, tgt, acc, slot, col0, valid, groups):
    _cm0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    _cm1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    _tmp_maskreg_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lce_s1_all_f = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lce_s1_is_target = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=_PL_FP32)
    lce_s1_ninf = vf.full(-3e+38, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_s1_zero = vf.full(0.0, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_s1_tile_max_v = vf.full(-3e+38, _tmp_maskreg_0, dtype=_PL_FP32)
    for lce_s1a_group in pl.range(0, groups, 1):
        lce_s1_active = vf.update_mask((valid - lce_s1a_group * 64), dtype=_PL_FP32)
        lce_s1_value = vf.load_align(cap, (lce_s1a_group * 64))
        lce_s1_guarded = vf.select(lce_s1_value, lce_s1_ninf, lce_s1_active)
        lce_s1_tile_max_v = vf.max(lce_s1_tile_max_v, lce_s1_guarded, lce_s1_all_f)
    lce_s1_tile_max = vf.reduce_max(lce_s1_tile_max_v, lce_s1_all_f)
    lce_s1_tile_max = vf.full(lce_s1_tile_max, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_s1_prev_max = vf.load_align(acc, 0, dist=pl.LoadDist.BRC_B32)
    lce_s1_carried = vf.load_align(acc, 1, dist=pl.LoadDist.BRC_B32)
    lce_s1_new_max = vf.max(lce_s1_prev_max, lce_s1_tile_max, lce_s1_all_f)
    lce_s1_correction = vf.exp_sub(lce_s1_prev_max, lce_s1_new_max, lce_s1_all_f)
    lce_s1_carried = vf.mul(lce_s1_carried, lce_s1_correction, lce_s1_all_f)
    lce_s1_target_i = vf.load_align(tgt, slot, dist=pl.LoadDist.BRC_B32)
    lce_s1_target_f = vf.astype(lce_s1_target_i, _tmp_maskreg_1, dtype=_PL_FP32, layout=pl.CastLayout.ZERO)
    lce_s1_target_f = vf.full(lce_s1_target_f, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_s1_base_i = vf.full(col0, _tmp_maskreg_1, dtype=_PL_INT32)
    lce_s1_base_f = vf.astype(lce_s1_base_i, _tmp_maskreg_1, dtype=_PL_FP32, layout=pl.CastLayout.ZERO)
    lce_s1_base_f = vf.full(lce_s1_base_f, _tmp_maskreg_0, dtype=_PL_FP32)
    _t1 = vf.arange(0, dtype=_PL_INT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    lce_s1_index_f = vf.astype(_t1, _cm1, dtype=_PL_FP32, layout=pl.CastLayout.ZERO)
    lce_s1_index_f = vf.add(lce_s1_index_f, lce_s1_base_f, lce_s1_all_f)
    lce_s1_dacc = vf.full(0.0, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_s1_sxacc = vf.full(0.0, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_s1_xyacc = vf.full(0.0, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_s1_wyacc = vf.full(0.0, _tmp_maskreg_0, dtype=_PL_FP32)
    for lce_s1b_group in pl.range(0, groups, 1):
        lce_s1_active = vf.update_mask((valid - lce_s1b_group * 64), dtype=_PL_FP32)
        _m0 = (lce_s1b_group * 64)
        lce_s1_value = vf.load_align(cap, _m0)
        lce_s1_weight = vf.load_align(wtile, _m0)
        lce_s1_expo = vf.exp_sub(lce_s1_value, lce_s1_new_max, lce_s1_active)
        lce_s1_dacc = vf.add(lce_s1_dacc, lce_s1_expo, lce_s1_all_f)
        lce_s1_product = vf.mul(lce_s1_value, lce_s1_weight, lce_s1_active)
        lce_s1_sxacc = vf.add(lce_s1_sxacc, lce_s1_product, lce_s1_all_f)
        lce_s1_is_target = vf.eq(lce_s1_index_f, lce_s1_target_f, lce_s1_all_f)
        lce_s1_picked = vf.select(lce_s1_value, lce_s1_zero, lce_s1_is_target)
        lce_s1_xyacc = vf.add(lce_s1_xyacc, lce_s1_picked, lce_s1_all_f)
        lce_s1_picked = vf.select(lce_s1_weight, lce_s1_zero, lce_s1_is_target)
        lce_s1_wyacc = vf.add(lce_s1_wyacc, lce_s1_picked, lce_s1_all_f)
        lce_s1_index_f = vf.adds(lce_s1_index_f, 64.0, lce_s1_all_f)
    lce_s1_dsum = vf.reduce_sum(lce_s1_dacc, lce_s1_all_f)
    lce_s1_sxsum = vf.reduce_sum(lce_s1_sxacc, lce_s1_all_f)
    lce_s1_xysum = vf.reduce_sum(lce_s1_xyacc, lce_s1_all_f)
    lce_s1_wysum = vf.reduce_sum(lce_s1_wyacc, lce_s1_all_f)
    lce_s1_carried = vf.add(lce_s1_carried, lce_s1_dsum, lce_s1_all_f)
    vf.store_align(acc + 0, lce_s1_new_max, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.store_align(acc + 1, lce_s1_carried, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    lce_s1_carried = vf.load_align(acc, 4, dist=pl.LoadDist.BRC_B32)
    lce_s1_sxsum = vf.add(lce_s1_sxsum, lce_s1_carried, lce_s1_all_f)
    vf.store_align(acc + 4, lce_s1_sxsum, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    lce_s1_carried = vf.load_align(acc, 2, dist=pl.LoadDist.BRC_B32)
    lce_s1_xysum = vf.add(lce_s1_xysum, lce_s1_carried, lce_s1_all_f)
    vf.store_align(acc + 2, lce_s1_xysum, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    lce_s1_carried = vf.load_align(acc, 3, dist=pl.LoadDist.BRC_B32)
    lce_s1_wysum = vf.add(lce_s1_wysum, lce_s1_carried, lce_s1_all_f)
    vf.store_align(acc + 3, lce_s1_wysum, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def stage2_row_finish_vf(acc, tgt, s_valid, acc_out, pred_out, slot):
    _tmp_maskreg_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lce_pf_all_f = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lce_pf_keep = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=_PL_FP32)
    lce_pf_same = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=_PL_FP32)
    lce_pf_one = vf.full(1.0, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_pf_zero = vf.full(0.0, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_pf_minus_one = vf.full(-1.0, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_pf_best = vf.load_align(acc, 0, dist=pl.LoadDist.BRC_B32)
    lce_pf_valid = vf.load_align(s_valid, slot, dist=pl.LoadDist.BRC_B32)
    lce_pf_target_i = vf.load_align(tgt, slot, dist=pl.LoadDist.BRC_B32)
    lce_pf_target_f = vf.astype(lce_pf_target_i, _tmp_maskreg_1, dtype=_PL_FP32, layout=pl.CastLayout.ZERO)
    lce_pf_target_f = vf.full(lce_pf_target_f, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_pf_keep = vf.gt(lce_pf_valid, 0.5, lce_pf_all_f)
    lce_pf_same = vf.eq(lce_pf_best, lce_pf_target_f, lce_pf_all_f)
    lce_pf_hit = vf.select(lce_pf_one, lce_pf_zero, lce_pf_same)
    lce_pf_hit = vf.select(lce_pf_hit, lce_pf_zero, lce_pf_keep)
    lce_pf_pred_f = vf.select(lce_pf_best, lce_pf_minus_one, lce_pf_keep)
    lce_pf_pred_i = vf.astype(
        lce_pf_pred_f,
        _tmp_maskreg_1,
        dtype=_PL_INT32,
        layout=pl.CastLayout.ZERO,
        round_mode=pl.VFRoundMode.CAST_RINT,
    )
    vf.store_align(acc_out + slot, lce_pf_hit, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.store_align(pred_out + slot, lce_pf_pred_i, _tmp_maskreg_1, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def stage2_tile_vf(
    cap,
    raw,
    wtile,
    out,
    tgt,
    s_a,
    s_b,
    s_c,
    s_lse,
    s_max,
    s_valid,
    acc,
    slot,
    col0,
    valid,
    groups,
    has_grad_f,
    inv_softcap,
    has_softcap,
    has_metrics,
):
    _cm0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    _cm1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    _tmp_maskreg_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lce_s2_all_f = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    lce_s2_is_target = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=_PL_FP32)
    lce_s2_is_max = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=_PL_FP32)
    lce_s2_keep = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=_PL_FP32)
    lce_s2_with_grad = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=_PL_FP32)
    lce_s2_one = vf.full(1.0, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_s2_zero = vf.full(0.0, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_s2_big = vf.full(3e+38, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_s2_best_v = vf.full(3e+38, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_s2_coef_a = vf.load_align(s_a, slot, dist=pl.LoadDist.BRC_B32)
    lce_s2_coef_b = vf.load_align(s_b, slot, dist=pl.LoadDist.BRC_B32)
    lce_s2_coef_c = vf.load_align(s_c, slot, dist=pl.LoadDist.BRC_B32)
    lce_s2_lse = vf.load_align(s_lse, slot, dist=pl.LoadDist.BRC_B32)
    lce_s2_row_max = vf.load_align(s_max, slot, dist=pl.LoadDist.BRC_B32)
    lce_s2_valid = vf.load_align(s_valid, slot, dist=pl.LoadDist.BRC_B32)
    lce_s2_keep = vf.gt(lce_s2_valid, 0.5, lce_s2_all_f)
    lce_s2_grad_flag = vf.full(has_grad_f, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_s2_with_grad = vf.gt(lce_s2_grad_flag, 0.5, lce_s2_all_f)
    lce_s2_target_i = vf.load_align(tgt, slot, dist=pl.LoadDist.BRC_B32)
    lce_s2_target_f = vf.astype(lce_s2_target_i, _tmp_maskreg_1, dtype=_PL_FP32, layout=pl.CastLayout.ZERO)
    lce_s2_target_f = vf.full(lce_s2_target_f, _tmp_maskreg_0, dtype=_PL_FP32)
    lce_s2_base_i = vf.full(col0, _tmp_maskreg_1, dtype=_PL_INT32)
    lce_s2_base_f = vf.astype(lce_s2_base_i, _tmp_maskreg_1, dtype=_PL_FP32, layout=pl.CastLayout.ZERO)
    lce_s2_base_f = vf.full(lce_s2_base_f, _tmp_maskreg_0, dtype=_PL_FP32)
    _t1 = vf.arange(0, dtype=_PL_INT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    lce_s2_index_f = vf.astype(_t1, _cm1, dtype=_PL_FP32, layout=pl.CastLayout.ZERO)
    lce_s2_index_f = vf.add(lce_s2_index_f, lce_s2_base_f, lce_s2_all_f)
    for lce_s2_group in pl.range(0, groups, 1):
        lce_s2_active = vf.update_mask((valid - lce_s2_group * 64), dtype=_PL_FP32)
        _m0 = (lce_s2_group * 64)
        lce_s2_value = vf.load_align(cap, _m0)
        lce_s2_raw = vf.load_align(raw, _m0)
        lce_s2_weight = vf.load_align(wtile, _m0)
        lce_s2_prob = vf.exp_sub(lce_s2_value, lce_s2_lse, lce_s2_active)
        lce_s2_grad = vf.mul(lce_s2_prob, lce_s2_coef_a, lce_s2_active)
        lce_s2_scratch = vf.mul(lce_s2_weight, lce_s2_coef_b, lce_s2_active)
        lce_s2_grad = vf.add(lce_s2_grad, lce_s2_scratch, lce_s2_active)
        lce_s2_is_target = vf.eq(lce_s2_index_f, lce_s2_target_f, lce_s2_all_f)
        lce_s2_picked = vf.select(lce_s2_coef_c, lce_s2_zero, lce_s2_is_target)
        lce_s2_grad = vf.sub(lce_s2_grad, lce_s2_picked, lce_s2_active)
        for _ in pl.range(0, has_softcap, 1):
            lce_s2_chain = vf.muls(lce_s2_value, inv_softcap, lce_s2_active)
            lce_s2_chain = vf.mul(lce_s2_chain, lce_s2_chain, lce_s2_active)
            lce_s2_chain = vf.sub(lce_s2_one, lce_s2_chain, lce_s2_all_f)
            lce_s2_grad = vf.mul(lce_s2_grad, lce_s2_chain, lce_s2_active)
        lce_s2_grad = vf.select(lce_s2_grad, lce_s2_raw, lce_s2_with_grad)
        lce_s2_grad = vf.select(lce_s2_grad, lce_s2_zero, lce_s2_keep)
        vf.store_align(out + _m0, lce_s2_grad, _tmp_maskreg_0)
        for _ in pl.range(0, has_metrics, 1):
            lce_s2_is_max = vf.eq(lce_s2_value, lce_s2_row_max, lce_s2_active)
            lce_s2_picked = vf.select(lce_s2_index_f, lce_s2_big, lce_s2_is_max)
            lce_s2_best_v = vf.min(lce_s2_best_v, lce_s2_picked, lce_s2_all_f)
        lce_s2_index_f = vf.adds(lce_s2_index_f, 64.0, lce_s2_all_f)
    lce_s2_best = vf.reduce_min(lce_s2_best_v, lce_s2_all_f)
    lce_s2_carried = vf.load_align(acc, 0, dist=pl.LoadDist.BRC_B32)
    lce_s2_best = vf.min(lce_s2_best, lce_s2_carried, lce_s2_all_f)
    vf.store_align(acc + 0, lce_s2_best, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def widen1_float_vf(src, cap, valid, groups):
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    for lce_w1f_group in pl.range(0, groups, 1):
        lce_w1f_active = vf.update_mask((valid - lce_w1f_group * 64), dtype=_PL_FP32)
        _m0 = (lce_w1f_group * 64)
        lce_w1f_value = vf.load_align(src, _m0)
        vf.store_align(cap + _m0, lce_w1f_value, _tmp_maskreg_0)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def widen2_float_vf(src, raw, cap, valid, groups):
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    for lce_wf_group in pl.range(0, groups, 1):
        lce_wf_active = vf.update_mask((valid - lce_wf_group * 64), dtype=_PL_FP32)
        _m0 = (lce_wf_group * 64)
        lce_wf_value = vf.load_align(src, _m0)
        vf.store_align(raw + _m0, lce_wf_value, _tmp_maskreg_0)
        vf.store_align(cap + _m0, lce_wf_value, _tmp_maskreg_0)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.jit(auto_mutex=True)
def liger_cross_entropy_loss_fwd_kernel(
    target: pl.Tensor[[1, pl.DYNAMIC], pl.DT_INT32],
    weight: pl.Tensor[[1, pl.DYNAMIC], pl.DT_FP32],
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    loss1d: pl.Tensor[[1, pl.DYNAMIC], pl.DT_FP32],
    zloss1d: pl.Tensor[[1, pl.DYNAMIC], pl.DT_FP32],
    acc1d: pl.Tensor[[1, pl.DYNAMIC], pl.DT_FP32],
    pred1d: pl.Tensor[[1, pl.DYNAMIC], pl.DT_INT32],
    redf: pl.Tensor[[1, pl.DYNAMIC], pl.DT_FP32],
    redn: pl.Tensor[[1, pl.DYNAMIC], pl.DT_FP32],
    lce_ws_max: pl.Tensor[[1, pl.DYNAMIC], pl.DT_FP32],
    lce_ws_lse: pl.Tensor[[1, pl.DYNAMIC], pl.DT_FP32],
    lce_ws_xy: pl.Tensor[[1, pl.DYNAMIC], pl.DT_FP32],
    lce_ws_wy: pl.Tensor[[1, pl.DYNAMIC], pl.DT_FP32],
    lce_ws_sxw: pl.Tensor[[1, pl.DYNAMIC], pl.DT_FP32],
    lce_ws_valid: pl.Tensor[[1, pl.DYNAMIC], pl.DT_FP32],
    lce_ws_lossf: pl.Tensor[[1, pl.DYNAMIC], pl.DT_FP32],
    lce_ws_zlossf: pl.Tensor[[1, pl.DYNAMIC], pl.DT_FP32],
    bt: pl.DT_INT32,
    vv: pl.DT_INT32,
    nred: pl.DT_INT32,
    ntwo: pl.DT_INT32,
    nunit: pl.DT_INT32,
    hasw: pl.DT_INT32,
    hassc: pl.DT_INT32,
    epsv: pl.DT_FP32,
    omlv: pl.DT_FP32,
    zsv: pl.DT_FP32,
    ignv: pl.DT_FP32,
    meanv: pl.DT_FP32,
    capv: pl.DT_FP32,
    twocv: pl.DT_FP32,
    invcv: pl.DT_FP32,
    haswf: pl.DT_FP32,
    hasgf: pl.DT_FP32,
    hasmet: pl.DT_INT32,
):
    g_lce_xn_buf = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 4096], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
        ),
        addrs=0x0,
        mutex_ids=[0, 1],
    )
    g_lce_xo_buf = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 4096], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
        ),
        addrs=0x8000,
        mutex_ids=[2, 3],
    )
    g_lce_xr_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, 4096], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x10000,
        mutex_ids=[4],
    )
    g_lce_xc_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, 4096], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x14000,
        mutex_ids=[4],
    )
    g_lce_dx_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, 4096], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x18000,
        mutex_ids=[4],
    )
    g_lce_w_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, 4096], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1c000,
        mutex_ids=[5],
    )
    g_lce_tgt_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, 512], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x20000,
        mutex_ids=[6],
    )
    g_lce_s_max = pl.make_tile_group(
        type=pl.TileType(shape=[1, 512], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x20800,
        mutex_ids=[7],
    )
    g_lce_s_lse = pl.make_tile_group(
        type=pl.TileType(shape=[1, 512], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x21000,
        mutex_ids=[8],
    )
    g_lce_s_xy = pl.make_tile_group(
        type=pl.TileType(shape=[1, 512], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x21800,
        mutex_ids=[9],
    )
    g_lce_s_wy = pl.make_tile_group(
        type=pl.TileType(shape=[1, 512], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x22000,
        mutex_ids=[10],
    )
    g_lce_s_sxw = pl.make_tile_group(
        type=pl.TileType(shape=[1, 512], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x22800,
        mutex_ids=[11],
    )
    g_lce_s_valid = pl.make_tile_group(
        type=pl.TileType(shape=[1, 512], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x23000,
        mutex_ids=[12],
    )
    g_lce_s_a = pl.make_tile_group(
        type=pl.TileType(shape=[1, 512], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x23800,
        mutex_ids=[4],
    )
    g_lce_s_b = pl.make_tile_group(
        type=pl.TileType(shape=[1, 512], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x24000,
        mutex_ids=[4],
    )
    g_lce_s_c = pl.make_tile_group(
        type=pl.TileType(shape=[1, 512], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x24800,
        mutex_ids=[4],
    )
    g_lce_lossf_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, 512], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x25000,
        mutex_ids=[4],
    )
    g_lce_zlossf_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, 512], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x25800,
        mutex_ids=[4],
    )
    g_lce_lossr_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, 512], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x26000,
        mutex_ids=[13],
    )
    g_lce_zlossr_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, 512], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x26800,
        mutex_ids=[14],
    )
    g_lce_lossn_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, 512], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x27000,
        mutex_ids=[15],
    )
    g_lce_zlossn_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, 512], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x27800,
        mutex_ids=[16],
    )
    g_lce_accn_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, 512], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x28000,
        mutex_ids=[17],
    )
    g_lce_predn_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, 512], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x28800,
        mutex_ids=[18],
    )
    g_lce_scan_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, 4096], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x29000,
        mutex_ids=[19],
    )
    g_lce_acc_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x2d000,
        mutex_ids=[4],
    )
    g_lce_glob_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x2d100,
        mutex_ids=[4],
    )
    g_lce_red_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x2d200,
        mutex_ids=[20],
    )
    g_lce_redn_ub = pl.make_tile_group(
        type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x2d300,
        mutex_ids=[21],
    )
    with pl.section_vector():
        t1_lce_xr_ub = g_lce_xr_ub.next()
        t2_lce_xc_ub = g_lce_xc_ub.next()
        t3_lce_dx_ub = g_lce_dx_ub.next()
        t4_lce_w_ub = g_lce_w_ub.next()
        t5_lce_tgt_ub = g_lce_tgt_ub.next()
        t6_lce_s_max = g_lce_s_max.next()
        t7_lce_s_lse = g_lce_s_lse.next()
        t8_lce_s_xy = g_lce_s_xy.next()
        t9_lce_s_wy = g_lce_s_wy.next()
        t10_lce_s_sxw = g_lce_s_sxw.next()
        t11_lce_s_valid = g_lce_s_valid.next()
        t12_lce_s_a = g_lce_s_a.next()
        t13_lce_s_b = g_lce_s_b.next()
        t14_lce_s_c = g_lce_s_c.next()
        t15_lce_lossf_ub = g_lce_lossf_ub.next()
        t16_lce_zlossf_ub = g_lce_zlossf_ub.next()
        t17_lce_lossr_ub = g_lce_lossr_ub.next()
        t18_lce_zlossr_ub = g_lce_zlossr_ub.next()
        t19_lce_lossn_ub = g_lce_lossn_ub.next()
        t20_lce_zlossn_ub = g_lce_zlossn_ub.next()
        t21_lce_accn_ub = g_lce_accn_ub.next()
        t22_lce_predn_ub = g_lce_predn_ub.next()
        t23_lce_scan_ub = g_lce_scan_ub.next()
        t24_lce_acc_ub = g_lce_acc_ub.next()
        t25_lce_glob_ub = g_lce_glob_ub.next()
        t26_lce_red_ub = g_lce_red_ub.next()
        t27_lce_redn_ub = g_lce_redn_ub.next()
        fill_ones_vf(t4_lce_w_ub, ((4096 + 64 - 1) // 64))
        glob_reset_vf(t25_lce_glob_ub)
        pl.system.bar_all()
        _k14 = (bt + pl.get_block_num() - 1)
        _k24 = (_k14 // pl.get_block_num())
        _k29 = (_k24 * pl.get_block_idx())
        _k30 = (bt - _k29)
        _k34 = (pl.min(_k24, _k30))
        _k35 = (pl.max(0, _k34))
        _k36 = (_k35 + 512 - 1)
        _k37 = (_k36 // 512)
        for lce_b1 in pl.range(0, _k37, 1):
            _k0 = (lce_b1 * 512)
            _k31 = (_k29 + _k0)
            _k38 = ((_k29 + _k35) - _k31)
            _k40 = (pl.min(512, _k38))
            pl.set_validshape(t5_lce_tgt_ub, [1, _k40])
            pl.load(t5_lce_tgt_ub, target, [0, _k31])
            pl.system.bar_all()
            for lce_r1 in pl.range(0, _k40, 1):
                row_reset_vf(t24_lce_acc_ub)
                pl.system.bar_all()
                for lce_t1 in pl.range(0, ((vv + 4096 - 1) // 4096), 1):
                    t28_lce_xn_buf = g_lce_xn_buf.next()
                    _k5 = (lce_t1 * 4096)
                    _k10 = (vv - _k5)
                    _k17 = (pl.min(4096, _k10))
                    pl.set_validshape(t28_lce_xn_buf, [1, _k17])
                    pl.load(t28_lce_xn_buf, x, [(_k31 + lce_r1), _k5])
                    for _ in pl.range(0, hasw, 1):
                        pl.set_validshape(t4_lce_w_ub, [1, _k17])
                        pl.load(t4_lce_w_ub, weight, [0, _k5])
                    _k22 = (_k17 + 64 - 1)
                    _k27 = (_k22 // 64)
                    widen1_float_vf(t28_lce_xn_buf, t2_lce_xc_ub, _k17, _k27)
                    for _ in pl.range(0, hassc, 1):
                        softcap_tile_vf(t2_lce_xc_ub, _k17, _k27, capv, twocv)
                    stage1_tile_vf(t2_lce_xc_ub, t4_lce_w_ub, t5_lce_tgt_ub, t24_lce_acc_ub, lce_r1, _k5, _k17, _k27)
                pl.system.bar_all()
                stage1_row_finish_vf(
                    t24_lce_acc_ub,
                    t5_lce_tgt_ub,
                    t6_lce_s_max,
                    t7_lce_s_lse,
                    t8_lce_s_xy,
                    t9_lce_s_wy,
                    t10_lce_s_sxw,
                    t11_lce_s_valid,
                    lce_r1,
                    ignv,
                )
                pl.system.bar_all()
            pl.set_validshape(t6_lce_s_max, [1, _k40])
            pl.store(lce_ws_max, t6_lce_s_max, [0, _k31])
            pl.set_validshape(t7_lce_s_lse, [1, _k40])
            pl.store(lce_ws_lse, t7_lce_s_lse, [0, _k31])
            pl.set_validshape(t8_lce_s_xy, [1, _k40])
            pl.store(lce_ws_xy, t8_lce_s_xy, [0, _k31])
            pl.set_validshape(t9_lce_s_wy, [1, _k40])
            pl.store(lce_ws_wy, t9_lce_s_wy, [0, _k31])
            pl.set_validshape(t10_lce_s_sxw, [1, _k40])
            pl.store(lce_ws_sxw, t10_lce_s_sxw, [0, _k31])
            pl.set_validshape(t11_lce_s_valid, [1, _k40])
            pl.store(lce_ws_valid, t11_lce_s_valid, [0, _k31])
            pl.system.bar_all()
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=0, sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK)
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=0, sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK)
        _k2 = (bt + 4096 - 1)
        _k12 = (_k2 // 4096)
        for lce_sc in pl.range(0, _k12, 1):
            _k4 = (lce_sc * 4096)
            _k9 = (bt - _k4)
            _k16 = (pl.min(4096, _k9))
            pl.set_validshape(t23_lce_scan_ub, [1, _k16])
            pl.load(t23_lce_scan_ub, lce_ws_valid, [0, _k4])
            pl.system.bar_all()
            _k21 = (_k16 + 64 - 1)
            _k26 = (_k21 // 64)
            scan_add_vf(t23_lce_scan_ub, t25_lce_glob_ub, 0, _k16, _k26)
            pl.system.bar_all()
            pl.set_validshape(t23_lce_scan_ub, [1, _k16])
            pl.load(t23_lce_scan_ub, lce_ws_wy, [0, _k4])
            pl.system.bar_all()
            scan_add_vf(t23_lce_scan_ub, t25_lce_glob_ub, 1, _k16, _k26)
            pl.system.bar_all()
        for lce_wt_loop in pl.range(0, (((vv + 4096 - 1) // 4096) * hasw), 1):
            _k7 = (lce_wt_loop * 4096)
            _k13 = (vv - _k7)
            _k19 = (pl.min(4096, _k13))
            pl.set_validshape(t23_lce_scan_ub, [1, _k19])
            pl.load(t23_lce_scan_ub, weight, [0, _k7])
            pl.system.bar_all()
            scan_add_vf(t23_lce_scan_ub, t25_lce_glob_ub, 2, _k19, ((_k19 + 64 - 1) // 64))
            pl.system.bar_all()
        glob_finish_vf(t25_lce_glob_ub, vv, haswf, meanv)
        pl.system.bar_all()
        for lce_b2 in pl.range(0, _k37, 1):
            _k1 = (lce_b2 * 512)
            _k32 = (_k29 + _k1)
            _k39 = ((_k29 + _k35) - _k32)
            _k41 = (pl.min(512, _k39))
            pl.set_validshape(t5_lce_tgt_ub, [1, _k41])
            pl.load(t5_lce_tgt_ub, target, [0, _k32])
            pl.set_validshape(t6_lce_s_max, [1, _k41])
            pl.load(t6_lce_s_max, lce_ws_max, [0, _k32])
            pl.set_validshape(t7_lce_s_lse, [1, _k41])
            pl.load(t7_lce_s_lse, lce_ws_lse, [0, _k32])
            pl.set_validshape(t8_lce_s_xy, [1, _k41])
            pl.load(t8_lce_s_xy, lce_ws_xy, [0, _k32])
            pl.set_validshape(t9_lce_s_wy, [1, _k41])
            pl.load(t9_lce_s_wy, lce_ws_wy, [0, _k32])
            pl.set_validshape(t10_lce_s_sxw, [1, _k41])
            pl.load(t10_lce_s_sxw, lce_ws_sxw, [0, _k32])
            pl.set_validshape(t11_lce_s_valid, [1, _k41])
            pl.load(t11_lce_s_valid, lce_ws_valid, [0, _k32])
            pl.system.bar_all()
            _k42 = (_k41 + 64 - 1)
            _k43 = (_k42 // 64)
            row_prep_vf(
                t7_lce_s_lse,
                t8_lce_s_xy,
                t9_lce_s_wy,
                t10_lce_s_sxw,
                t11_lce_s_valid,
                t25_lce_glob_ub,
                t12_lce_s_a,
                t13_lce_s_b,
                t14_lce_s_c,
                t15_lce_lossf_ub,
                t16_lce_zlossf_ub,
                _k41,
                _k43,
                epsv,
                omlv,
                zsv,
            )
            narrow_float_vf(t15_lce_lossf_ub, t19_lce_lossn_ub, t17_lce_lossr_ub, _k41, _k43)
            narrow_float_vf(t16_lce_zlossf_ub, t20_lce_zlossn_ub, t18_lce_zlossr_ub, _k41, _k43)
            pl.system.bar_all()
            pl.set_validshape(t19_lce_lossn_ub, [1, _k41])
            pl.store(loss1d, t19_lce_lossn_ub, [0, _k32])
            pl.set_validshape(t20_lce_zlossn_ub, [1, _k41])
            pl.store(zloss1d, t20_lce_zlossn_ub, [0, _k32])
            pl.set_validshape(t17_lce_lossr_ub, [1, _k41])
            pl.store(lce_ws_lossf, t17_lce_lossr_ub, [0, _k32])
            pl.set_validshape(t18_lce_zlossr_ub, [1, _k41])
            pl.store(lce_ws_zlossf, t18_lce_zlossr_ub, [0, _k32])
            pl.system.bar_all()
            for lce_r2 in pl.range(0, _k41, 1):
                argmax_reset_vf(t24_lce_acc_ub)
                pl.system.bar_all()
                for lce_t2 in pl.range(0, ((vv + 4096 - 1) // 4096), 1):
                    t29_lce_xn_buf = g_lce_xn_buf.next()
                    _k6 = (lce_t2 * 4096)
                    _k11 = (vv - _k6)
                    _k18 = (pl.min(4096, _k11))
                    pl.set_validshape(t29_lce_xn_buf, [1, _k18])
                    _k33 = (_k32 + lce_r2)
                    pl.load(t29_lce_xn_buf, x, [_k33, _k6])
                    for _ in pl.range(0, hasw, 1):
                        pl.set_validshape(t4_lce_w_ub, [1, _k18])
                        pl.load(t4_lce_w_ub, weight, [0, _k6])
                    _k23 = (_k18 + 64 - 1)
                    _k28 = (_k23 // 64)
                    widen2_float_vf(t29_lce_xn_buf, t1_lce_xr_ub, t2_lce_xc_ub, _k18, _k28)
                    for _ in pl.range(0, hassc, 1):
                        softcap_tile_vf(t2_lce_xc_ub, _k18, _k28, capv, twocv)
                    stage2_tile_vf(
                        t2_lce_xc_ub,
                        t1_lce_xr_ub,
                        t4_lce_w_ub,
                        t3_lce_dx_ub,
                        t5_lce_tgt_ub,
                        t12_lce_s_a,
                        t13_lce_s_b,
                        t14_lce_s_c,
                        t7_lce_s_lse,
                        t6_lce_s_max,
                        t11_lce_s_valid,
                        t24_lce_acc_ub,
                        lce_r2,
                        _k6,
                        _k18,
                        _k28,
                        hasgf,
                        invcv,
                        hassc,
                        hasmet,
                    )
                    t30_lce_xo_buf = g_lce_xo_buf.next()
                    narrow_float_vf(t3_lce_dx_ub, t30_lce_xo_buf, t1_lce_xr_ub, _k18, _k28)
                    pl.set_validshape(t30_lce_xo_buf, [1, _k18])
                    pl.store(x, t30_lce_xo_buf, [_k33, _k6])
                pl.system.bar_all()
                stage2_row_finish_vf(
                    t24_lce_acc_ub,
                    t5_lce_tgt_ub,
                    t11_lce_s_valid,
                    t21_lce_accn_ub,
                    t22_lce_predn_ub,
                    lce_r2,
                )
                pl.system.bar_all()
            pl.set_validshape(t21_lce_accn_ub, [1, _k41])
            pl.store(acc1d, t21_lce_accn_ub, [0, _k32])
            pl.set_validshape(t22_lce_predn_ub, [1, _k41])
            pl.store(pred1d, t22_lce_predn_ub, [0, _k32])
            pl.system.bar_all()
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK)
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK)
        for _ in pl.range(0, (1 - (pl.min(1, pl.get_block_idx()))), 1):
            red_reset_vf(t26_lce_red_ub)
            pl.system.bar_all()
            for lce_rd in pl.range(0, _k12, 1):
                _k3 = (lce_rd * 4096)
                _k8 = (bt - _k3)
                _k15 = (pl.min(4096, _k8))
                pl.set_validshape(t23_lce_scan_ub, [1, _k15])
                pl.load(t23_lce_scan_ub, lce_ws_lossf, [0, _k3])
                pl.system.bar_all()
                _k20 = (_k15 + 64 - 1)
                _k25 = (_k20 // 64)
                scan_add_vf(t23_lce_scan_ub, t26_lce_red_ub, 0, _k15, _k25)
                pl.system.bar_all()
                pl.set_validshape(t23_lce_scan_ub, [1, _k15])
                pl.load(t23_lce_scan_ub, lce_ws_zlossf, [0, _k3])
                pl.system.bar_all()
                scan_add_vf(t23_lce_scan_ub, t26_lce_red_ub, 1, _k15, _k25)
                pl.system.bar_all()
                pl.set_validshape(t23_lce_scan_ub, [1, _k15])
                pl.load(t23_lce_scan_ub, acc1d, [0, _k3])
                pl.system.bar_all()
                scan_add_vf(t23_lce_scan_ub, t26_lce_red_ub, 2, _k15, _k25)
                pl.system.bar_all()
            reduce_scale_vf(t25_lce_glob_ub, t26_lce_red_ub)
            narrow_float_vf(t26_lce_red_ub, t27_lce_redn_ub, t23_lce_scan_ub, 2, 1)
            pl.system.bar_all()
            pl.set_validshape(t26_lce_red_ub, [1, 4])
            pl.store(redf, t26_lce_red_ub, [0, 0])
            pl.set_validshape(t27_lce_redn_ub, [1, 2])
            pl.store(redn, t27_lce_redn_ub, [0, 0])
            pl.system.bar_all()


# ---------------------------------------------------------------------------
# host wrapper
# ---------------------------------------------------------------------------

# The kernel's private scratch. `split_workspace` is kernel-internal on the
# AscendC path; PyPTO Pro has no equivalent, so the emitted signature appends
# one tensor per workspace after the public tensors. All eight are float32
# [1, BT] and carry per-row statistics across the rendezvous.
_FWD_WORKSPACES = 8


def liger_cross_entropy_loss_fwd_wrapper(
    logits,                        # [BT, V] float32, read and written in place
    target,                        # [BT]    int64
    weight=None,                   # [V]     float, optional class weights
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    reduction="mean",
    softcap=None,
    return_z_loss=False,
    return_token_accuracy=False,
    return_predicted_tokens=False,
):
    """Returns (loss, z_loss, token_accuracy, predicted_tokens, saved_input).

    Like the upstream triton kernel, the gradient intermediate is left in
    `logits` when `logits.requires_grad` is set. `saved_input` is the buffer it
    lives in -- here that IS `logits`, viewed as [BT, V], because this kernel
    writes in place rather than into a second allocation. Hand it to
    `liger_cross_entropy_loss_bwd_wrapper` to finish the backward; that is what
    `LigerCrossEntropyFunction` saves in its context.

    It is returned unconditionally, including when `requires_grad` is clear --
    the caller decides whether to keep it, exactly as the autograd Function
    does.
    """
    if logits.ndim != 2:
        raise ValueError("logits must be [BT, V], got {}".format(tuple(logits.shape)))
    if target.ndim != 1 or target.shape[0] != logits.shape[0]:
        raise ValueError("target must be [BT] matching the logit rows")
    if reduction not in ("none", "sum", "mean"):
        raise ValueError("reduction must be 'mean', 'sum' or 'none', got {}".format(reduction))
    if not (0.0 <= label_smoothing <= 1.0):
        raise ValueError("label_smoothing must be in [0, 1], got {}".format(label_smoothing))
    if softcap is not None and softcap <= 0:
        raise ValueError("softcap must be > 0 or None, got {}".format(softcap))
    if logits.dtype != torch.float32:
        raise ValueError("this kernel is the float32 variant, got {}".format(logits.dtype))
    if not logits.is_contiguous():
        # The gradient intermediate is written back into this exact buffer. A
        # non-contiguous input would be copied on the way to the kernel, the
        # kernel would write the copy, and the caller's tensor would come back
        # holding its original activations -- a plausible loss with a silently
        # missing gradient. Refuse instead.
        raise ValueError("logits must be contiguous; call .contiguous() first")

    rows, cols = int(logits.shape[0]), int(logits.shape[1])
    device = logits.device

    if weight is not None:
        if weight.ndim != 1 or weight.shape[0] != cols:
            raise ValueError("weight must be a [V] tensor, got {}".format(tuple(weight.shape)))
        if not torch.is_floating_point(weight):
            raise ValueError("weight must be floating point, got {}".format(weight.dtype))

    keep = target != ignore_index
    if int((target * keep).max()) >= cols or int((target * keep).min()) < 0:
        raise AssertionError("target out of bounds for V = {}".format(cols))

    # Dtype adaptation at the kernel boundary only: class indices are int32
    # (every legal target round-trips exactly) and class weights are float32,
    # which is what the reference widens them to before any arithmetic.
    target_i32 = target.to(torch.int32).contiguous().reshape(1, rows)
    x2d = logits.detach().contiguous().reshape(rows, cols)
    if weight is None:
        # Never read: the kernel pre-fills its weight tile with ones and the GM
        # load runs under a zero trip count.
        weight2d = torch.empty((1, cols), dtype=torch.float32, device=device)
        has_weight = 0
    else:
        weight2d = weight.detach().to(torch.float32).contiguous().reshape(1, cols)
        has_weight = 1

    loss1d = torch.zeros((1, rows), dtype=torch.float32, device=device)
    zloss1d = torch.zeros((1, rows), dtype=torch.float32, device=device)
    acc1d = torch.zeros((1, rows), dtype=torch.float32, device=device)
    pred1d = torch.zeros((1, rows), dtype=torch.int32, device=device)
    redf = torch.zeros((1, 4), dtype=torch.float32, device=device)
    redn = torch.zeros((1, 2), dtype=torch.float32, device=device)
    workspaces = [torch.zeros((1, rows), dtype=torch.float32, device=device)
                  for _ in range(_FWD_WORKSPACES)]

    cap = float(softcap) if softcap is not None else 0.0
    # This is a Vector-only kernel. Cap the launch by both the row-level task
    # split and the current device's AIV capacity. The INTER_BLOCK rendezvous
    # requires every launched block to be resident, so exceeding the platform
    # limit can deadlock rather than merely leave excess blocks idle.
    block_dim = min(rows, get_platform_info().vector_core_num)
    liger_cross_entropy_loss_fwd_kernel[None, block_dim](
        target_i32, weight2d, x2d, loss1d, zloss1d, acc1d, pred1d, redf, redn,
        *workspaces,
        rows, cols, 4, 2, 1, has_weight, 1 if softcap is not None else 0,
        float(label_smoothing) / float(cols), 1.0 - float(label_smoothing),
        float(lse_square_scale), float(ignore_index),
        1.0 if reduction == "mean" else 0.0,
        cap, (2.0 / cap) if cap else 0.0, (1.0 / cap) if cap else 0.0,
        float(has_weight), 1.0 if logits.requires_grad else 0.0,
        # The first-occurrence argmax scan is the only per-element work in the
        # gradient pass that nothing but these two outputs consumes, so it runs
        # under a zero-or-one trip count rather than unconditionally.
        1 if (return_token_accuracy or return_predicted_tokens) else 0,
    )

    if reduction == "none":
        loss = loss1d.reshape(rows)
        z_loss = zloss1d.reshape(rows) if return_z_loss else None
        accuracy = acc1d.reshape(rows) if return_token_accuracy else None
    else:
        loss = redn[0, 0]
        z_loss = redn[0, 1] if return_z_loss else None
        accuracy = redf[0, 2] if return_token_accuracy else None
    predicted = (pred1d.reshape(rows).to(torch.int64)
                 if return_predicted_tokens else None)
    return loss, z_loss, accuracy, predicted, x2d
