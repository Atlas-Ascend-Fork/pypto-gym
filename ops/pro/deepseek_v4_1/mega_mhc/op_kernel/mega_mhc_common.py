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
"""Shared Vector/pipeline helpers for the MegaMhc kernel.

Kept in a separate module so the batch16 and batch8 specializations share one
implementation. Only plain compile-time constants live here; Width/BlockJobs are
injected into the kernel module and stay in mega_mhc.py.
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

F32 = pl.DT_FP32
B16 = pl.DT_BF16
CHUNK = 256
SLOW_BATCH = 1
SLOW_CHUNK = 512
SLOW_PLANE = 8


@pl.vector_function
def _slow_dot_first(r0, r1, r2, r3, w, partial, scores, route):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
    for output in pl.range(0, 24, SLOW_PLANE):
        if SLOW_BATCH > 0:
            if SLOW_PLANE > 0:
                a0_0 = vf.full(0.0, p, dtype=F32)
                if False:
                    a0_0 = vf.load_align(partial, [0 + output + 0, 0])
            if SLOW_PLANE > 1:
                a0_1 = vf.full(0.0, p, dtype=F32)
                if False:
                    a0_1 = vf.load_align(partial, [0 + output + 1, 0])
            if SLOW_PLANE > 2:
                a0_2 = vf.full(0.0, p, dtype=F32)
                if False:
                    a0_2 = vf.load_align(partial, [0 + output + 2, 0])
            if SLOW_PLANE > 3:
                a0_3 = vf.full(0.0, p, dtype=F32)
                if False:
                    a0_3 = vf.load_align(partial, [0 + output + 3, 0])
            if SLOW_PLANE > 4:
                a0_4 = vf.full(0.0, p, dtype=F32)
                if False:
                    a0_4 = vf.load_align(partial, [0 + output + 4, 0])
            if SLOW_PLANE > 5:
                a0_5 = vf.full(0.0, p, dtype=F32)
                if False:
                    a0_5 = vf.load_align(partial, [0 + output + 5, 0])
            if SLOW_PLANE > 6:
                a0_6 = vf.full(0.0, p, dtype=F32)
                if False:
                    a0_6 = vf.load_align(partial, [0 + output + 6, 0])
            if SLOW_PLANE > 7:
                a0_7 = vf.full(0.0, p, dtype=F32)
                if False:
                    a0_7 = vf.load_align(partial, [0 + output + 7, 0])
        if SLOW_BATCH > 1:
            if SLOW_PLANE > 0:
                a1_0 = vf.full(0.0, p, dtype=F32)
                if False:
                    a1_0 = vf.load_align(partial, [24 + output + 0, 0])
            if SLOW_PLANE > 1:
                a1_1 = vf.full(0.0, p, dtype=F32)
                if False:
                    a1_1 = vf.load_align(partial, [24 + output + 1, 0])
            if SLOW_PLANE > 2:
                a1_2 = vf.full(0.0, p, dtype=F32)
                if False:
                    a1_2 = vf.load_align(partial, [24 + output + 2, 0])
            if SLOW_PLANE > 3:
                a1_3 = vf.full(0.0, p, dtype=F32)
                if False:
                    a1_3 = vf.load_align(partial, [24 + output + 3, 0])
            if SLOW_PLANE > 4:
                a1_4 = vf.full(0.0, p, dtype=F32)
                if False:
                    a1_4 = vf.load_align(partial, [24 + output + 4, 0])
            if SLOW_PLANE > 5:
                a1_5 = vf.full(0.0, p, dtype=F32)
                if False:
                    a1_5 = vf.load_align(partial, [24 + output + 5, 0])
            if SLOW_PLANE > 6:
                a1_6 = vf.full(0.0, p, dtype=F32)
                if False:
                    a1_6 = vf.load_align(partial, [24 + output + 6, 0])
            if SLOW_PLANE > 7:
                a1_7 = vf.full(0.0, p, dtype=F32)
                if False:
                    a1_7 = vf.load_align(partial, [24 + output + 7, 0])
        if SLOW_BATCH > 2:
            if SLOW_PLANE > 0:
                a2_0 = vf.full(0.0, p, dtype=F32)
                if False:
                    a2_0 = vf.load_align(partial, [48 + output + 0, 0])
            if SLOW_PLANE > 1:
                a2_1 = vf.full(0.0, p, dtype=F32)
                if False:
                    a2_1 = vf.load_align(partial, [48 + output + 1, 0])
            if SLOW_PLANE > 2:
                a2_2 = vf.full(0.0, p, dtype=F32)
                if False:
                    a2_2 = vf.load_align(partial, [48 + output + 2, 0])
            if SLOW_PLANE > 3:
                a2_3 = vf.full(0.0, p, dtype=F32)
                if False:
                    a2_3 = vf.load_align(partial, [48 + output + 3, 0])
            if SLOW_PLANE > 4:
                a2_4 = vf.full(0.0, p, dtype=F32)
                if False:
                    a2_4 = vf.load_align(partial, [48 + output + 4, 0])
            if SLOW_PLANE > 5:
                a2_5 = vf.full(0.0, p, dtype=F32)
                if False:
                    a2_5 = vf.load_align(partial, [48 + output + 5, 0])
            if SLOW_PLANE > 6:
                a2_6 = vf.full(0.0, p, dtype=F32)
                if False:
                    a2_6 = vf.load_align(partial, [48 + output + 6, 0])
            if SLOW_PLANE > 7:
                a2_7 = vf.full(0.0, p, dtype=F32)
                if False:
                    a2_7 = vf.load_align(partial, [48 + output + 7, 0])
        if SLOW_BATCH > 3:
            if SLOW_PLANE > 0:
                a3_0 = vf.full(0.0, p, dtype=F32)
                if False:
                    a3_0 = vf.load_align(partial, [72 + output + 0, 0])
            if SLOW_PLANE > 1:
                a3_1 = vf.full(0.0, p, dtype=F32)
                if False:
                    a3_1 = vf.load_align(partial, [72 + output + 1, 0])
            if SLOW_PLANE > 2:
                a3_2 = vf.full(0.0, p, dtype=F32)
                if False:
                    a3_2 = vf.load_align(partial, [72 + output + 2, 0])
            if SLOW_PLANE > 3:
                a3_3 = vf.full(0.0, p, dtype=F32)
                if False:
                    a3_3 = vf.load_align(partial, [72 + output + 3, 0])
            if SLOW_PLANE > 4:
                a3_4 = vf.full(0.0, p, dtype=F32)
                if False:
                    a3_4 = vf.load_align(partial, [72 + output + 4, 0])
            if SLOW_PLANE > 5:
                a3_5 = vf.full(0.0, p, dtype=F32)
                if False:
                    a3_5 = vf.load_align(partial, [72 + output + 5, 0])
            if SLOW_PLANE > 6:
                a3_6 = vf.full(0.0, p, dtype=F32)
                if False:
                    a3_6 = vf.load_align(partial, [72 + output + 6, 0])
            if SLOW_PLANE > 7:
                a3_7 = vf.full(0.0, p, dtype=F32)
                if False:
                    a3_7 = vf.load_align(partial, [72 + output + 7, 0])
        for col in pl.range(0, SLOW_CHUNK, 64):
            if SLOW_BATCH > 0:
                rv0 = vf.load_align(r0, [route, col])
            if SLOW_BATCH > 1:
                rv1 = vf.load_align(r1, [route, col])
            if SLOW_BATCH > 2:
                rv2 = vf.load_align(r2, [route, col])
            if SLOW_BATCH > 3:
                rv3 = vf.load_align(r3, [route, col])
            if SLOW_PLANE > 0:
                wv0 = vf.load_align(w, [output + 0, col])
            if SLOW_PLANE > 1:
                wv1 = vf.load_align(w, [output + 1, col])
            if SLOW_PLANE > 2:
                wv2 = vf.load_align(w, [output + 2, col])
            if SLOW_PLANE > 3:
                wv3 = vf.load_align(w, [output + 3, col])
            if SLOW_PLANE > 0:
                if SLOW_BATCH > 0:
                    a0_0 = vf.mul_add_dst(rv0, wv0, p)
                if SLOW_BATCH > 1:
                    a1_0 = vf.mul_add_dst(rv1, wv0, p)
                if SLOW_BATCH > 2:
                    a2_0 = vf.mul_add_dst(rv2, wv0, p)
                if SLOW_BATCH > 3:
                    a3_0 = vf.mul_add_dst(rv3, wv0, p)
            if SLOW_PLANE > 1:
                if SLOW_BATCH > 0:
                    a0_1 = vf.mul_add_dst(rv0, wv1, p)
                if SLOW_BATCH > 1:
                    a1_1 = vf.mul_add_dst(rv1, wv1, p)
                if SLOW_BATCH > 2:
                    a2_1 = vf.mul_add_dst(rv2, wv1, p)
                if SLOW_BATCH > 3:
                    a3_1 = vf.mul_add_dst(rv3, wv1, p)
            if SLOW_PLANE > 2:
                if SLOW_BATCH > 0:
                    a0_2 = vf.mul_add_dst(rv0, wv2, p)
                if SLOW_BATCH > 1:
                    a1_2 = vf.mul_add_dst(rv1, wv2, p)
                if SLOW_BATCH > 2:
                    a2_2 = vf.mul_add_dst(rv2, wv2, p)
                if SLOW_BATCH > 3:
                    a3_2 = vf.mul_add_dst(rv3, wv2, p)
            if SLOW_PLANE > 3:
                if SLOW_BATCH > 0:
                    a0_3 = vf.mul_add_dst(rv0, wv3, p)
                if SLOW_BATCH > 1:
                    a1_3 = vf.mul_add_dst(rv1, wv3, p)
                if SLOW_BATCH > 2:
                    a2_3 = vf.mul_add_dst(rv2, wv3, p)
                if SLOW_BATCH > 3:
                    a3_3 = vf.mul_add_dst(rv3, wv3, p)
            if SLOW_PLANE > 4:
                wv0 = vf.load_align(w, [output + 4, col])
            if SLOW_PLANE > 5:
                wv1 = vf.load_align(w, [output + 5, col])
            if SLOW_PLANE > 6:
                wv2 = vf.load_align(w, [output + 6, col])
            if SLOW_PLANE > 7:
                wv3 = vf.load_align(w, [output + 7, col])
            if SLOW_PLANE > 4:
                if SLOW_BATCH > 0:
                    a0_4 = vf.mul_add_dst(rv0, wv0, p)
                if SLOW_BATCH > 1:
                    a1_4 = vf.mul_add_dst(rv1, wv0, p)
                if SLOW_BATCH > 2:
                    a2_4 = vf.mul_add_dst(rv2, wv0, p)
                if SLOW_BATCH > 3:
                    a3_4 = vf.mul_add_dst(rv3, wv0, p)
            if SLOW_PLANE > 5:
                if SLOW_BATCH > 0:
                    a0_5 = vf.mul_add_dst(rv0, wv1, p)
                if SLOW_BATCH > 1:
                    a1_5 = vf.mul_add_dst(rv1, wv1, p)
                if SLOW_BATCH > 2:
                    a2_5 = vf.mul_add_dst(rv2, wv1, p)
                if SLOW_BATCH > 3:
                    a3_5 = vf.mul_add_dst(rv3, wv1, p)
            if SLOW_PLANE > 6:
                if SLOW_BATCH > 0:
                    a0_6 = vf.mul_add_dst(rv0, wv2, p)
                if SLOW_BATCH > 1:
                    a1_6 = vf.mul_add_dst(rv1, wv2, p)
                if SLOW_BATCH > 2:
                    a2_6 = vf.mul_add_dst(rv2, wv2, p)
                if SLOW_BATCH > 3:
                    a3_6 = vf.mul_add_dst(rv3, wv2, p)
            if SLOW_PLANE > 7:
                if SLOW_BATCH > 0:
                    a0_7 = vf.mul_add_dst(rv0, wv3, p)
                if SLOW_BATCH > 1:
                    a1_7 = vf.mul_add_dst(rv1, wv3, p)
                if SLOW_BATCH > 2:
                    a2_7 = vf.mul_add_dst(rv2, wv3, p)
                if SLOW_BATCH > 3:
                    a3_7 = vf.mul_add_dst(rv3, wv3, p)
        if SLOW_BATCH > 0:
            if SLOW_PLANE > 0:
                if False:
                    value = vf.reduce_sum(a0_0, p)
                    vf.store_align(scores, value, p, 0 + output + 0, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_0, p, [0 + output + 0, 0])
            if SLOW_PLANE > 1:
                if False:
                    value = vf.reduce_sum(a0_1, p)
                    vf.store_align(scores, value, p, 0 + output + 1, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_1, p, [0 + output + 1, 0])
            if SLOW_PLANE > 2:
                if False:
                    value = vf.reduce_sum(a0_2, p)
                    vf.store_align(scores, value, p, 0 + output + 2, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_2, p, [0 + output + 2, 0])
            if SLOW_PLANE > 3:
                if False:
                    value = vf.reduce_sum(a0_3, p)
                    vf.store_align(scores, value, p, 0 + output + 3, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_3, p, [0 + output + 3, 0])
            if SLOW_PLANE > 4:
                if False:
                    value = vf.reduce_sum(a0_4, p)
                    vf.store_align(scores, value, p, 0 + output + 4, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_4, p, [0 + output + 4, 0])
            if SLOW_PLANE > 5:
                if False:
                    value = vf.reduce_sum(a0_5, p)
                    vf.store_align(scores, value, p, 0 + output + 5, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_5, p, [0 + output + 5, 0])
            if SLOW_PLANE > 6:
                if False:
                    value = vf.reduce_sum(a0_6, p)
                    vf.store_align(scores, value, p, 0 + output + 6, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_6, p, [0 + output + 6, 0])
            if SLOW_PLANE > 7:
                if False:
                    value = vf.reduce_sum(a0_7, p)
                    vf.store_align(scores, value, p, 0 + output + 7, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_7, p, [0 + output + 7, 0])
        if SLOW_BATCH > 1:
            if SLOW_PLANE > 0:
                if False:
                    value = vf.reduce_sum(a1_0, p)
                    vf.store_align(scores, value, p, 64 + output + 0, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_0, p, [24 + output + 0, 0])
            if SLOW_PLANE > 1:
                if False:
                    value = vf.reduce_sum(a1_1, p)
                    vf.store_align(scores, value, p, 64 + output + 1, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_1, p, [24 + output + 1, 0])
            if SLOW_PLANE > 2:
                if False:
                    value = vf.reduce_sum(a1_2, p)
                    vf.store_align(scores, value, p, 64 + output + 2, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_2, p, [24 + output + 2, 0])
            if SLOW_PLANE > 3:
                if False:
                    value = vf.reduce_sum(a1_3, p)
                    vf.store_align(scores, value, p, 64 + output + 3, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_3, p, [24 + output + 3, 0])
            if SLOW_PLANE > 4:
                if False:
                    value = vf.reduce_sum(a1_4, p)
                    vf.store_align(scores, value, p, 64 + output + 4, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_4, p, [24 + output + 4, 0])
            if SLOW_PLANE > 5:
                if False:
                    value = vf.reduce_sum(a1_5, p)
                    vf.store_align(scores, value, p, 64 + output + 5, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_5, p, [24 + output + 5, 0])
            if SLOW_PLANE > 6:
                if False:
                    value = vf.reduce_sum(a1_6, p)
                    vf.store_align(scores, value, p, 64 + output + 6, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_6, p, [24 + output + 6, 0])
            if SLOW_PLANE > 7:
                if False:
                    value = vf.reduce_sum(a1_7, p)
                    vf.store_align(scores, value, p, 64 + output + 7, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_7, p, [24 + output + 7, 0])
        if SLOW_BATCH > 2:
            if SLOW_PLANE > 0:
                if False:
                    value = vf.reduce_sum(a2_0, p)
                    vf.store_align(scores, value, p, 128 + output + 0, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_0, p, [48 + output + 0, 0])
            if SLOW_PLANE > 1:
                if False:
                    value = vf.reduce_sum(a2_1, p)
                    vf.store_align(scores, value, p, 128 + output + 1, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_1, p, [48 + output + 1, 0])
            if SLOW_PLANE > 2:
                if False:
                    value = vf.reduce_sum(a2_2, p)
                    vf.store_align(scores, value, p, 128 + output + 2, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_2, p, [48 + output + 2, 0])
            if SLOW_PLANE > 3:
                if False:
                    value = vf.reduce_sum(a2_3, p)
                    vf.store_align(scores, value, p, 128 + output + 3, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_3, p, [48 + output + 3, 0])
            if SLOW_PLANE > 4:
                if False:
                    value = vf.reduce_sum(a2_4, p)
                    vf.store_align(scores, value, p, 128 + output + 4, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_4, p, [48 + output + 4, 0])
            if SLOW_PLANE > 5:
                if False:
                    value = vf.reduce_sum(a2_5, p)
                    vf.store_align(scores, value, p, 128 + output + 5, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_5, p, [48 + output + 5, 0])
            if SLOW_PLANE > 6:
                if False:
                    value = vf.reduce_sum(a2_6, p)
                    vf.store_align(scores, value, p, 128 + output + 6, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_6, p, [48 + output + 6, 0])
            if SLOW_PLANE > 7:
                if False:
                    value = vf.reduce_sum(a2_7, p)
                    vf.store_align(scores, value, p, 128 + output + 7, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_7, p, [48 + output + 7, 0])
        if SLOW_BATCH > 3:
            if SLOW_PLANE > 0:
                if False:
                    value = vf.reduce_sum(a3_0, p)
                    vf.store_align(scores, value, p, 192 + output + 0, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_0, p, [72 + output + 0, 0])
            if SLOW_PLANE > 1:
                if False:
                    value = vf.reduce_sum(a3_1, p)
                    vf.store_align(scores, value, p, 192 + output + 1, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_1, p, [72 + output + 1, 0])
            if SLOW_PLANE > 2:
                if False:
                    value = vf.reduce_sum(a3_2, p)
                    vf.store_align(scores, value, p, 192 + output + 2, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_2, p, [72 + output + 2, 0])
            if SLOW_PLANE > 3:
                if False:
                    value = vf.reduce_sum(a3_3, p)
                    vf.store_align(scores, value, p, 192 + output + 3, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_3, p, [72 + output + 3, 0])
            if SLOW_PLANE > 4:
                if False:
                    value = vf.reduce_sum(a3_4, p)
                    vf.store_align(scores, value, p, 192 + output + 4, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_4, p, [72 + output + 4, 0])
            if SLOW_PLANE > 5:
                if False:
                    value = vf.reduce_sum(a3_5, p)
                    vf.store_align(scores, value, p, 192 + output + 5, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_5, p, [72 + output + 5, 0])
            if SLOW_PLANE > 6:
                if False:
                    value = vf.reduce_sum(a3_6, p)
                    vf.store_align(scores, value, p, 192 + output + 6, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_6, p, [72 + output + 6, 0])
            if SLOW_PLANE > 7:
                if False:
                    value = vf.reduce_sum(a3_7, p)
                    vf.store_align(scores, value, p, 192 + output + 7, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_7, p, [72 + output + 7, 0])


@pl.vector_function
def _slow_dot_accum(r0, r1, r2, r3, w, partial, scores, route):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
    for output in pl.range(0, 24, SLOW_PLANE):
        if SLOW_BATCH > 0:
            if SLOW_PLANE > 0:
                a0_0 = vf.full(0.0, p, dtype=F32)
                if True:
                    a0_0 = vf.load_align(partial, [0 + output + 0, 0])
            if SLOW_PLANE > 1:
                a0_1 = vf.full(0.0, p, dtype=F32)
                if True:
                    a0_1 = vf.load_align(partial, [0 + output + 1, 0])
            if SLOW_PLANE > 2:
                a0_2 = vf.full(0.0, p, dtype=F32)
                if True:
                    a0_2 = vf.load_align(partial, [0 + output + 2, 0])
            if SLOW_PLANE > 3:
                a0_3 = vf.full(0.0, p, dtype=F32)
                if True:
                    a0_3 = vf.load_align(partial, [0 + output + 3, 0])
            if SLOW_PLANE > 4:
                a0_4 = vf.full(0.0, p, dtype=F32)
                if True:
                    a0_4 = vf.load_align(partial, [0 + output + 4, 0])
            if SLOW_PLANE > 5:
                a0_5 = vf.full(0.0, p, dtype=F32)
                if True:
                    a0_5 = vf.load_align(partial, [0 + output + 5, 0])
            if SLOW_PLANE > 6:
                a0_6 = vf.full(0.0, p, dtype=F32)
                if True:
                    a0_6 = vf.load_align(partial, [0 + output + 6, 0])
            if SLOW_PLANE > 7:
                a0_7 = vf.full(0.0, p, dtype=F32)
                if True:
                    a0_7 = vf.load_align(partial, [0 + output + 7, 0])
        if SLOW_BATCH > 1:
            if SLOW_PLANE > 0:
                a1_0 = vf.full(0.0, p, dtype=F32)
                if True:
                    a1_0 = vf.load_align(partial, [24 + output + 0, 0])
            if SLOW_PLANE > 1:
                a1_1 = vf.full(0.0, p, dtype=F32)
                if True:
                    a1_1 = vf.load_align(partial, [24 + output + 1, 0])
            if SLOW_PLANE > 2:
                a1_2 = vf.full(0.0, p, dtype=F32)
                if True:
                    a1_2 = vf.load_align(partial, [24 + output + 2, 0])
            if SLOW_PLANE > 3:
                a1_3 = vf.full(0.0, p, dtype=F32)
                if True:
                    a1_3 = vf.load_align(partial, [24 + output + 3, 0])
            if SLOW_PLANE > 4:
                a1_4 = vf.full(0.0, p, dtype=F32)
                if True:
                    a1_4 = vf.load_align(partial, [24 + output + 4, 0])
            if SLOW_PLANE > 5:
                a1_5 = vf.full(0.0, p, dtype=F32)
                if True:
                    a1_5 = vf.load_align(partial, [24 + output + 5, 0])
            if SLOW_PLANE > 6:
                a1_6 = vf.full(0.0, p, dtype=F32)
                if True:
                    a1_6 = vf.load_align(partial, [24 + output + 6, 0])
            if SLOW_PLANE > 7:
                a1_7 = vf.full(0.0, p, dtype=F32)
                if True:
                    a1_7 = vf.load_align(partial, [24 + output + 7, 0])
        if SLOW_BATCH > 2:
            if SLOW_PLANE > 0:
                a2_0 = vf.full(0.0, p, dtype=F32)
                if True:
                    a2_0 = vf.load_align(partial, [48 + output + 0, 0])
            if SLOW_PLANE > 1:
                a2_1 = vf.full(0.0, p, dtype=F32)
                if True:
                    a2_1 = vf.load_align(partial, [48 + output + 1, 0])
            if SLOW_PLANE > 2:
                a2_2 = vf.full(0.0, p, dtype=F32)
                if True:
                    a2_2 = vf.load_align(partial, [48 + output + 2, 0])
            if SLOW_PLANE > 3:
                a2_3 = vf.full(0.0, p, dtype=F32)
                if True:
                    a2_3 = vf.load_align(partial, [48 + output + 3, 0])
            if SLOW_PLANE > 4:
                a2_4 = vf.full(0.0, p, dtype=F32)
                if True:
                    a2_4 = vf.load_align(partial, [48 + output + 4, 0])
            if SLOW_PLANE > 5:
                a2_5 = vf.full(0.0, p, dtype=F32)
                if True:
                    a2_5 = vf.load_align(partial, [48 + output + 5, 0])
            if SLOW_PLANE > 6:
                a2_6 = vf.full(0.0, p, dtype=F32)
                if True:
                    a2_6 = vf.load_align(partial, [48 + output + 6, 0])
            if SLOW_PLANE > 7:
                a2_7 = vf.full(0.0, p, dtype=F32)
                if True:
                    a2_7 = vf.load_align(partial, [48 + output + 7, 0])
        if SLOW_BATCH > 3:
            if SLOW_PLANE > 0:
                a3_0 = vf.full(0.0, p, dtype=F32)
                if True:
                    a3_0 = vf.load_align(partial, [72 + output + 0, 0])
            if SLOW_PLANE > 1:
                a3_1 = vf.full(0.0, p, dtype=F32)
                if True:
                    a3_1 = vf.load_align(partial, [72 + output + 1, 0])
            if SLOW_PLANE > 2:
                a3_2 = vf.full(0.0, p, dtype=F32)
                if True:
                    a3_2 = vf.load_align(partial, [72 + output + 2, 0])
            if SLOW_PLANE > 3:
                a3_3 = vf.full(0.0, p, dtype=F32)
                if True:
                    a3_3 = vf.load_align(partial, [72 + output + 3, 0])
            if SLOW_PLANE > 4:
                a3_4 = vf.full(0.0, p, dtype=F32)
                if True:
                    a3_4 = vf.load_align(partial, [72 + output + 4, 0])
            if SLOW_PLANE > 5:
                a3_5 = vf.full(0.0, p, dtype=F32)
                if True:
                    a3_5 = vf.load_align(partial, [72 + output + 5, 0])
            if SLOW_PLANE > 6:
                a3_6 = vf.full(0.0, p, dtype=F32)
                if True:
                    a3_6 = vf.load_align(partial, [72 + output + 6, 0])
            if SLOW_PLANE > 7:
                a3_7 = vf.full(0.0, p, dtype=F32)
                if True:
                    a3_7 = vf.load_align(partial, [72 + output + 7, 0])
        for col in pl.range(0, SLOW_CHUNK, 64):
            if SLOW_BATCH > 0:
                rv0 = vf.load_align(r0, [route, col])
            if SLOW_BATCH > 1:
                rv1 = vf.load_align(r1, [route, col])
            if SLOW_BATCH > 2:
                rv2 = vf.load_align(r2, [route, col])
            if SLOW_BATCH > 3:
                rv3 = vf.load_align(r3, [route, col])
            if SLOW_PLANE > 0:
                wv0 = vf.load_align(w, [output + 0, col])
            if SLOW_PLANE > 1:
                wv1 = vf.load_align(w, [output + 1, col])
            if SLOW_PLANE > 2:
                wv2 = vf.load_align(w, [output + 2, col])
            if SLOW_PLANE > 3:
                wv3 = vf.load_align(w, [output + 3, col])
            if SLOW_PLANE > 0:
                if SLOW_BATCH > 0:
                    a0_0 = vf.mul_add_dst(rv0, wv0, p)
                if SLOW_BATCH > 1:
                    a1_0 = vf.mul_add_dst(rv1, wv0, p)
                if SLOW_BATCH > 2:
                    a2_0 = vf.mul_add_dst(rv2, wv0, p)
                if SLOW_BATCH > 3:
                    a3_0 = vf.mul_add_dst(rv3, wv0, p)
            if SLOW_PLANE > 1:
                if SLOW_BATCH > 0:
                    a0_1 = vf.mul_add_dst(rv0, wv1, p)
                if SLOW_BATCH > 1:
                    a1_1 = vf.mul_add_dst(rv1, wv1, p)
                if SLOW_BATCH > 2:
                    a2_1 = vf.mul_add_dst(rv2, wv1, p)
                if SLOW_BATCH > 3:
                    a3_1 = vf.mul_add_dst(rv3, wv1, p)
            if SLOW_PLANE > 2:
                if SLOW_BATCH > 0:
                    a0_2 = vf.mul_add_dst(rv0, wv2, p)
                if SLOW_BATCH > 1:
                    a1_2 = vf.mul_add_dst(rv1, wv2, p)
                if SLOW_BATCH > 2:
                    a2_2 = vf.mul_add_dst(rv2, wv2, p)
                if SLOW_BATCH > 3:
                    a3_2 = vf.mul_add_dst(rv3, wv2, p)
            if SLOW_PLANE > 3:
                if SLOW_BATCH > 0:
                    a0_3 = vf.mul_add_dst(rv0, wv3, p)
                if SLOW_BATCH > 1:
                    a1_3 = vf.mul_add_dst(rv1, wv3, p)
                if SLOW_BATCH > 2:
                    a2_3 = vf.mul_add_dst(rv2, wv3, p)
                if SLOW_BATCH > 3:
                    a3_3 = vf.mul_add_dst(rv3, wv3, p)
            if SLOW_PLANE > 4:
                wv0 = vf.load_align(w, [output + 4, col])
            if SLOW_PLANE > 5:
                wv1 = vf.load_align(w, [output + 5, col])
            if SLOW_PLANE > 6:
                wv2 = vf.load_align(w, [output + 6, col])
            if SLOW_PLANE > 7:
                wv3 = vf.load_align(w, [output + 7, col])
            if SLOW_PLANE > 4:
                if SLOW_BATCH > 0:
                    a0_4 = vf.mul_add_dst(rv0, wv0, p)
                if SLOW_BATCH > 1:
                    a1_4 = vf.mul_add_dst(rv1, wv0, p)
                if SLOW_BATCH > 2:
                    a2_4 = vf.mul_add_dst(rv2, wv0, p)
                if SLOW_BATCH > 3:
                    a3_4 = vf.mul_add_dst(rv3, wv0, p)
            if SLOW_PLANE > 5:
                if SLOW_BATCH > 0:
                    a0_5 = vf.mul_add_dst(rv0, wv1, p)
                if SLOW_BATCH > 1:
                    a1_5 = vf.mul_add_dst(rv1, wv1, p)
                if SLOW_BATCH > 2:
                    a2_5 = vf.mul_add_dst(rv2, wv1, p)
                if SLOW_BATCH > 3:
                    a3_5 = vf.mul_add_dst(rv3, wv1, p)
            if SLOW_PLANE > 6:
                if SLOW_BATCH > 0:
                    a0_6 = vf.mul_add_dst(rv0, wv2, p)
                if SLOW_BATCH > 1:
                    a1_6 = vf.mul_add_dst(rv1, wv2, p)
                if SLOW_BATCH > 2:
                    a2_6 = vf.mul_add_dst(rv2, wv2, p)
                if SLOW_BATCH > 3:
                    a3_6 = vf.mul_add_dst(rv3, wv2, p)
            if SLOW_PLANE > 7:
                if SLOW_BATCH > 0:
                    a0_7 = vf.mul_add_dst(rv0, wv3, p)
                if SLOW_BATCH > 1:
                    a1_7 = vf.mul_add_dst(rv1, wv3, p)
                if SLOW_BATCH > 2:
                    a2_7 = vf.mul_add_dst(rv2, wv3, p)
                if SLOW_BATCH > 3:
                    a3_7 = vf.mul_add_dst(rv3, wv3, p)
        if SLOW_BATCH > 0:
            if SLOW_PLANE > 0:
                if False:
                    value = vf.reduce_sum(a0_0, p)
                    vf.store_align(scores, value, p, 0 + output + 0, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_0, p, [0 + output + 0, 0])
            if SLOW_PLANE > 1:
                if False:
                    value = vf.reduce_sum(a0_1, p)
                    vf.store_align(scores, value, p, 0 + output + 1, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_1, p, [0 + output + 1, 0])
            if SLOW_PLANE > 2:
                if False:
                    value = vf.reduce_sum(a0_2, p)
                    vf.store_align(scores, value, p, 0 + output + 2, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_2, p, [0 + output + 2, 0])
            if SLOW_PLANE > 3:
                if False:
                    value = vf.reduce_sum(a0_3, p)
                    vf.store_align(scores, value, p, 0 + output + 3, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_3, p, [0 + output + 3, 0])
            if SLOW_PLANE > 4:
                if False:
                    value = vf.reduce_sum(a0_4, p)
                    vf.store_align(scores, value, p, 0 + output + 4, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_4, p, [0 + output + 4, 0])
            if SLOW_PLANE > 5:
                if False:
                    value = vf.reduce_sum(a0_5, p)
                    vf.store_align(scores, value, p, 0 + output + 5, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_5, p, [0 + output + 5, 0])
            if SLOW_PLANE > 6:
                if False:
                    value = vf.reduce_sum(a0_6, p)
                    vf.store_align(scores, value, p, 0 + output + 6, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_6, p, [0 + output + 6, 0])
            if SLOW_PLANE > 7:
                if False:
                    value = vf.reduce_sum(a0_7, p)
                    vf.store_align(scores, value, p, 0 + output + 7, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_7, p, [0 + output + 7, 0])
        if SLOW_BATCH > 1:
            if SLOW_PLANE > 0:
                if False:
                    value = vf.reduce_sum(a1_0, p)
                    vf.store_align(scores, value, p, 64 + output + 0, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_0, p, [24 + output + 0, 0])
            if SLOW_PLANE > 1:
                if False:
                    value = vf.reduce_sum(a1_1, p)
                    vf.store_align(scores, value, p, 64 + output + 1, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_1, p, [24 + output + 1, 0])
            if SLOW_PLANE > 2:
                if False:
                    value = vf.reduce_sum(a1_2, p)
                    vf.store_align(scores, value, p, 64 + output + 2, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_2, p, [24 + output + 2, 0])
            if SLOW_PLANE > 3:
                if False:
                    value = vf.reduce_sum(a1_3, p)
                    vf.store_align(scores, value, p, 64 + output + 3, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_3, p, [24 + output + 3, 0])
            if SLOW_PLANE > 4:
                if False:
                    value = vf.reduce_sum(a1_4, p)
                    vf.store_align(scores, value, p, 64 + output + 4, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_4, p, [24 + output + 4, 0])
            if SLOW_PLANE > 5:
                if False:
                    value = vf.reduce_sum(a1_5, p)
                    vf.store_align(scores, value, p, 64 + output + 5, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_5, p, [24 + output + 5, 0])
            if SLOW_PLANE > 6:
                if False:
                    value = vf.reduce_sum(a1_6, p)
                    vf.store_align(scores, value, p, 64 + output + 6, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_6, p, [24 + output + 6, 0])
            if SLOW_PLANE > 7:
                if False:
                    value = vf.reduce_sum(a1_7, p)
                    vf.store_align(scores, value, p, 64 + output + 7, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_7, p, [24 + output + 7, 0])
        if SLOW_BATCH > 2:
            if SLOW_PLANE > 0:
                if False:
                    value = vf.reduce_sum(a2_0, p)
                    vf.store_align(scores, value, p, 128 + output + 0, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_0, p, [48 + output + 0, 0])
            if SLOW_PLANE > 1:
                if False:
                    value = vf.reduce_sum(a2_1, p)
                    vf.store_align(scores, value, p, 128 + output + 1, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_1, p, [48 + output + 1, 0])
            if SLOW_PLANE > 2:
                if False:
                    value = vf.reduce_sum(a2_2, p)
                    vf.store_align(scores, value, p, 128 + output + 2, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_2, p, [48 + output + 2, 0])
            if SLOW_PLANE > 3:
                if False:
                    value = vf.reduce_sum(a2_3, p)
                    vf.store_align(scores, value, p, 128 + output + 3, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_3, p, [48 + output + 3, 0])
            if SLOW_PLANE > 4:
                if False:
                    value = vf.reduce_sum(a2_4, p)
                    vf.store_align(scores, value, p, 128 + output + 4, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_4, p, [48 + output + 4, 0])
            if SLOW_PLANE > 5:
                if False:
                    value = vf.reduce_sum(a2_5, p)
                    vf.store_align(scores, value, p, 128 + output + 5, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_5, p, [48 + output + 5, 0])
            if SLOW_PLANE > 6:
                if False:
                    value = vf.reduce_sum(a2_6, p)
                    vf.store_align(scores, value, p, 128 + output + 6, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_6, p, [48 + output + 6, 0])
            if SLOW_PLANE > 7:
                if False:
                    value = vf.reduce_sum(a2_7, p)
                    vf.store_align(scores, value, p, 128 + output + 7, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_7, p, [48 + output + 7, 0])
        if SLOW_BATCH > 3:
            if SLOW_PLANE > 0:
                if False:
                    value = vf.reduce_sum(a3_0, p)
                    vf.store_align(scores, value, p, 192 + output + 0, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_0, p, [72 + output + 0, 0])
            if SLOW_PLANE > 1:
                if False:
                    value = vf.reduce_sum(a3_1, p)
                    vf.store_align(scores, value, p, 192 + output + 1, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_1, p, [72 + output + 1, 0])
            if SLOW_PLANE > 2:
                if False:
                    value = vf.reduce_sum(a3_2, p)
                    vf.store_align(scores, value, p, 192 + output + 2, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_2, p, [72 + output + 2, 0])
            if SLOW_PLANE > 3:
                if False:
                    value = vf.reduce_sum(a3_3, p)
                    vf.store_align(scores, value, p, 192 + output + 3, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_3, p, [72 + output + 3, 0])
            if SLOW_PLANE > 4:
                if False:
                    value = vf.reduce_sum(a3_4, p)
                    vf.store_align(scores, value, p, 192 + output + 4, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_4, p, [72 + output + 4, 0])
            if SLOW_PLANE > 5:
                if False:
                    value = vf.reduce_sum(a3_5, p)
                    vf.store_align(scores, value, p, 192 + output + 5, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_5, p, [72 + output + 5, 0])
            if SLOW_PLANE > 6:
                if False:
                    value = vf.reduce_sum(a3_6, p)
                    vf.store_align(scores, value, p, 192 + output + 6, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_6, p, [72 + output + 6, 0])
            if SLOW_PLANE > 7:
                if False:
                    value = vf.reduce_sum(a3_7, p)
                    vf.store_align(scores, value, p, 192 + output + 7, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_7, p, [72 + output + 7, 0])


@pl.vector_function
def _slow_dot_last(r0, r1, r2, r3, w, partial, scores, route):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
    for output in pl.range(0, 24, SLOW_PLANE):
        if SLOW_BATCH > 0:
            if SLOW_PLANE > 0:
                a0_0 = vf.full(0.0, p, dtype=F32)
                if True:
                    a0_0 = vf.load_align(partial, [0 + output + 0, 0])
            if SLOW_PLANE > 1:
                a0_1 = vf.full(0.0, p, dtype=F32)
                if True:
                    a0_1 = vf.load_align(partial, [0 + output + 1, 0])
            if SLOW_PLANE > 2:
                a0_2 = vf.full(0.0, p, dtype=F32)
                if True:
                    a0_2 = vf.load_align(partial, [0 + output + 2, 0])
            if SLOW_PLANE > 3:
                a0_3 = vf.full(0.0, p, dtype=F32)
                if True:
                    a0_3 = vf.load_align(partial, [0 + output + 3, 0])
            if SLOW_PLANE > 4:
                a0_4 = vf.full(0.0, p, dtype=F32)
                if True:
                    a0_4 = vf.load_align(partial, [0 + output + 4, 0])
            if SLOW_PLANE > 5:
                a0_5 = vf.full(0.0, p, dtype=F32)
                if True:
                    a0_5 = vf.load_align(partial, [0 + output + 5, 0])
            if SLOW_PLANE > 6:
                a0_6 = vf.full(0.0, p, dtype=F32)
                if True:
                    a0_6 = vf.load_align(partial, [0 + output + 6, 0])
            if SLOW_PLANE > 7:
                a0_7 = vf.full(0.0, p, dtype=F32)
                if True:
                    a0_7 = vf.load_align(partial, [0 + output + 7, 0])
        if SLOW_BATCH > 1:
            if SLOW_PLANE > 0:
                a1_0 = vf.full(0.0, p, dtype=F32)
                if True:
                    a1_0 = vf.load_align(partial, [24 + output + 0, 0])
            if SLOW_PLANE > 1:
                a1_1 = vf.full(0.0, p, dtype=F32)
                if True:
                    a1_1 = vf.load_align(partial, [24 + output + 1, 0])
            if SLOW_PLANE > 2:
                a1_2 = vf.full(0.0, p, dtype=F32)
                if True:
                    a1_2 = vf.load_align(partial, [24 + output + 2, 0])
            if SLOW_PLANE > 3:
                a1_3 = vf.full(0.0, p, dtype=F32)
                if True:
                    a1_3 = vf.load_align(partial, [24 + output + 3, 0])
            if SLOW_PLANE > 4:
                a1_4 = vf.full(0.0, p, dtype=F32)
                if True:
                    a1_4 = vf.load_align(partial, [24 + output + 4, 0])
            if SLOW_PLANE > 5:
                a1_5 = vf.full(0.0, p, dtype=F32)
                if True:
                    a1_5 = vf.load_align(partial, [24 + output + 5, 0])
            if SLOW_PLANE > 6:
                a1_6 = vf.full(0.0, p, dtype=F32)
                if True:
                    a1_6 = vf.load_align(partial, [24 + output + 6, 0])
            if SLOW_PLANE > 7:
                a1_7 = vf.full(0.0, p, dtype=F32)
                if True:
                    a1_7 = vf.load_align(partial, [24 + output + 7, 0])
        if SLOW_BATCH > 2:
            if SLOW_PLANE > 0:
                a2_0 = vf.full(0.0, p, dtype=F32)
                if True:
                    a2_0 = vf.load_align(partial, [48 + output + 0, 0])
            if SLOW_PLANE > 1:
                a2_1 = vf.full(0.0, p, dtype=F32)
                if True:
                    a2_1 = vf.load_align(partial, [48 + output + 1, 0])
            if SLOW_PLANE > 2:
                a2_2 = vf.full(0.0, p, dtype=F32)
                if True:
                    a2_2 = vf.load_align(partial, [48 + output + 2, 0])
            if SLOW_PLANE > 3:
                a2_3 = vf.full(0.0, p, dtype=F32)
                if True:
                    a2_3 = vf.load_align(partial, [48 + output + 3, 0])
            if SLOW_PLANE > 4:
                a2_4 = vf.full(0.0, p, dtype=F32)
                if True:
                    a2_4 = vf.load_align(partial, [48 + output + 4, 0])
            if SLOW_PLANE > 5:
                a2_5 = vf.full(0.0, p, dtype=F32)
                if True:
                    a2_5 = vf.load_align(partial, [48 + output + 5, 0])
            if SLOW_PLANE > 6:
                a2_6 = vf.full(0.0, p, dtype=F32)
                if True:
                    a2_6 = vf.load_align(partial, [48 + output + 6, 0])
            if SLOW_PLANE > 7:
                a2_7 = vf.full(0.0, p, dtype=F32)
                if True:
                    a2_7 = vf.load_align(partial, [48 + output + 7, 0])
        if SLOW_BATCH > 3:
            if SLOW_PLANE > 0:
                a3_0 = vf.full(0.0, p, dtype=F32)
                if True:
                    a3_0 = vf.load_align(partial, [72 + output + 0, 0])
            if SLOW_PLANE > 1:
                a3_1 = vf.full(0.0, p, dtype=F32)
                if True:
                    a3_1 = vf.load_align(partial, [72 + output + 1, 0])
            if SLOW_PLANE > 2:
                a3_2 = vf.full(0.0, p, dtype=F32)
                if True:
                    a3_2 = vf.load_align(partial, [72 + output + 2, 0])
            if SLOW_PLANE > 3:
                a3_3 = vf.full(0.0, p, dtype=F32)
                if True:
                    a3_3 = vf.load_align(partial, [72 + output + 3, 0])
            if SLOW_PLANE > 4:
                a3_4 = vf.full(0.0, p, dtype=F32)
                if True:
                    a3_4 = vf.load_align(partial, [72 + output + 4, 0])
            if SLOW_PLANE > 5:
                a3_5 = vf.full(0.0, p, dtype=F32)
                if True:
                    a3_5 = vf.load_align(partial, [72 + output + 5, 0])
            if SLOW_PLANE > 6:
                a3_6 = vf.full(0.0, p, dtype=F32)
                if True:
                    a3_6 = vf.load_align(partial, [72 + output + 6, 0])
            if SLOW_PLANE > 7:
                a3_7 = vf.full(0.0, p, dtype=F32)
                if True:
                    a3_7 = vf.load_align(partial, [72 + output + 7, 0])
        for col in pl.range(0, SLOW_CHUNK, 64):
            if SLOW_BATCH > 0:
                rv0 = vf.load_align(r0, [route, col])
            if SLOW_BATCH > 1:
                rv1 = vf.load_align(r1, [route, col])
            if SLOW_BATCH > 2:
                rv2 = vf.load_align(r2, [route, col])
            if SLOW_BATCH > 3:
                rv3 = vf.load_align(r3, [route, col])
            if SLOW_PLANE > 0:
                wv0 = vf.load_align(w, [output + 0, col])
            if SLOW_PLANE > 1:
                wv1 = vf.load_align(w, [output + 1, col])
            if SLOW_PLANE > 2:
                wv2 = vf.load_align(w, [output + 2, col])
            if SLOW_PLANE > 3:
                wv3 = vf.load_align(w, [output + 3, col])
            if SLOW_PLANE > 0:
                if SLOW_BATCH > 0:
                    a0_0 = vf.mul_add_dst(rv0, wv0, p)
                if SLOW_BATCH > 1:
                    a1_0 = vf.mul_add_dst(rv1, wv0, p)
                if SLOW_BATCH > 2:
                    a2_0 = vf.mul_add_dst(rv2, wv0, p)
                if SLOW_BATCH > 3:
                    a3_0 = vf.mul_add_dst(rv3, wv0, p)
            if SLOW_PLANE > 1:
                if SLOW_BATCH > 0:
                    a0_1 = vf.mul_add_dst(rv0, wv1, p)
                if SLOW_BATCH > 1:
                    a1_1 = vf.mul_add_dst(rv1, wv1, p)
                if SLOW_BATCH > 2:
                    a2_1 = vf.mul_add_dst(rv2, wv1, p)
                if SLOW_BATCH > 3:
                    a3_1 = vf.mul_add_dst(rv3, wv1, p)
            if SLOW_PLANE > 2:
                if SLOW_BATCH > 0:
                    a0_2 = vf.mul_add_dst(rv0, wv2, p)
                if SLOW_BATCH > 1:
                    a1_2 = vf.mul_add_dst(rv1, wv2, p)
                if SLOW_BATCH > 2:
                    a2_2 = vf.mul_add_dst(rv2, wv2, p)
                if SLOW_BATCH > 3:
                    a3_2 = vf.mul_add_dst(rv3, wv2, p)
            if SLOW_PLANE > 3:
                if SLOW_BATCH > 0:
                    a0_3 = vf.mul_add_dst(rv0, wv3, p)
                if SLOW_BATCH > 1:
                    a1_3 = vf.mul_add_dst(rv1, wv3, p)
                if SLOW_BATCH > 2:
                    a2_3 = vf.mul_add_dst(rv2, wv3, p)
                if SLOW_BATCH > 3:
                    a3_3 = vf.mul_add_dst(rv3, wv3, p)
            if SLOW_PLANE > 4:
                wv0 = vf.load_align(w, [output + 4, col])
            if SLOW_PLANE > 5:
                wv1 = vf.load_align(w, [output + 5, col])
            if SLOW_PLANE > 6:
                wv2 = vf.load_align(w, [output + 6, col])
            if SLOW_PLANE > 7:
                wv3 = vf.load_align(w, [output + 7, col])
            if SLOW_PLANE > 4:
                if SLOW_BATCH > 0:
                    a0_4 = vf.mul_add_dst(rv0, wv0, p)
                if SLOW_BATCH > 1:
                    a1_4 = vf.mul_add_dst(rv1, wv0, p)
                if SLOW_BATCH > 2:
                    a2_4 = vf.mul_add_dst(rv2, wv0, p)
                if SLOW_BATCH > 3:
                    a3_4 = vf.mul_add_dst(rv3, wv0, p)
            if SLOW_PLANE > 5:
                if SLOW_BATCH > 0:
                    a0_5 = vf.mul_add_dst(rv0, wv1, p)
                if SLOW_BATCH > 1:
                    a1_5 = vf.mul_add_dst(rv1, wv1, p)
                if SLOW_BATCH > 2:
                    a2_5 = vf.mul_add_dst(rv2, wv1, p)
                if SLOW_BATCH > 3:
                    a3_5 = vf.mul_add_dst(rv3, wv1, p)
            if SLOW_PLANE > 6:
                if SLOW_BATCH > 0:
                    a0_6 = vf.mul_add_dst(rv0, wv2, p)
                if SLOW_BATCH > 1:
                    a1_6 = vf.mul_add_dst(rv1, wv2, p)
                if SLOW_BATCH > 2:
                    a2_6 = vf.mul_add_dst(rv2, wv2, p)
                if SLOW_BATCH > 3:
                    a3_6 = vf.mul_add_dst(rv3, wv2, p)
            if SLOW_PLANE > 7:
                if SLOW_BATCH > 0:
                    a0_7 = vf.mul_add_dst(rv0, wv3, p)
                if SLOW_BATCH > 1:
                    a1_7 = vf.mul_add_dst(rv1, wv3, p)
                if SLOW_BATCH > 2:
                    a2_7 = vf.mul_add_dst(rv2, wv3, p)
                if SLOW_BATCH > 3:
                    a3_7 = vf.mul_add_dst(rv3, wv3, p)
        if SLOW_BATCH > 0:
            if SLOW_PLANE > 0:
                if True:
                    value = vf.reduce_sum(a0_0, p)
                    vf.store_align(scores, value, p, 0 + output + 0, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_0, p, [0 + output + 0, 0])
            if SLOW_PLANE > 1:
                if True:
                    value = vf.reduce_sum(a0_1, p)
                    vf.store_align(scores, value, p, 0 + output + 1, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_1, p, [0 + output + 1, 0])
            if SLOW_PLANE > 2:
                if True:
                    value = vf.reduce_sum(a0_2, p)
                    vf.store_align(scores, value, p, 0 + output + 2, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_2, p, [0 + output + 2, 0])
            if SLOW_PLANE > 3:
                if True:
                    value = vf.reduce_sum(a0_3, p)
                    vf.store_align(scores, value, p, 0 + output + 3, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_3, p, [0 + output + 3, 0])
            if SLOW_PLANE > 4:
                if True:
                    value = vf.reduce_sum(a0_4, p)
                    vf.store_align(scores, value, p, 0 + output + 4, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_4, p, [0 + output + 4, 0])
            if SLOW_PLANE > 5:
                if True:
                    value = vf.reduce_sum(a0_5, p)
                    vf.store_align(scores, value, p, 0 + output + 5, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_5, p, [0 + output + 5, 0])
            if SLOW_PLANE > 6:
                if True:
                    value = vf.reduce_sum(a0_6, p)
                    vf.store_align(scores, value, p, 0 + output + 6, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_6, p, [0 + output + 6, 0])
            if SLOW_PLANE > 7:
                if True:
                    value = vf.reduce_sum(a0_7, p)
                    vf.store_align(scores, value, p, 0 + output + 7, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_7, p, [0 + output + 7, 0])
        if SLOW_BATCH > 1:
            if SLOW_PLANE > 0:
                if True:
                    value = vf.reduce_sum(a1_0, p)
                    vf.store_align(scores, value, p, 64 + output + 0, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_0, p, [24 + output + 0, 0])
            if SLOW_PLANE > 1:
                if True:
                    value = vf.reduce_sum(a1_1, p)
                    vf.store_align(scores, value, p, 64 + output + 1, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_1, p, [24 + output + 1, 0])
            if SLOW_PLANE > 2:
                if True:
                    value = vf.reduce_sum(a1_2, p)
                    vf.store_align(scores, value, p, 64 + output + 2, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_2, p, [24 + output + 2, 0])
            if SLOW_PLANE > 3:
                if True:
                    value = vf.reduce_sum(a1_3, p)
                    vf.store_align(scores, value, p, 64 + output + 3, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_3, p, [24 + output + 3, 0])
            if SLOW_PLANE > 4:
                if True:
                    value = vf.reduce_sum(a1_4, p)
                    vf.store_align(scores, value, p, 64 + output + 4, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_4, p, [24 + output + 4, 0])
            if SLOW_PLANE > 5:
                if True:
                    value = vf.reduce_sum(a1_5, p)
                    vf.store_align(scores, value, p, 64 + output + 5, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_5, p, [24 + output + 5, 0])
            if SLOW_PLANE > 6:
                if True:
                    value = vf.reduce_sum(a1_6, p)
                    vf.store_align(scores, value, p, 64 + output + 6, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_6, p, [24 + output + 6, 0])
            if SLOW_PLANE > 7:
                if True:
                    value = vf.reduce_sum(a1_7, p)
                    vf.store_align(scores, value, p, 64 + output + 7, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_7, p, [24 + output + 7, 0])
        if SLOW_BATCH > 2:
            if SLOW_PLANE > 0:
                if True:
                    value = vf.reduce_sum(a2_0, p)
                    vf.store_align(scores, value, p, 128 + output + 0, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_0, p, [48 + output + 0, 0])
            if SLOW_PLANE > 1:
                if True:
                    value = vf.reduce_sum(a2_1, p)
                    vf.store_align(scores, value, p, 128 + output + 1, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_1, p, [48 + output + 1, 0])
            if SLOW_PLANE > 2:
                if True:
                    value = vf.reduce_sum(a2_2, p)
                    vf.store_align(scores, value, p, 128 + output + 2, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_2, p, [48 + output + 2, 0])
            if SLOW_PLANE > 3:
                if True:
                    value = vf.reduce_sum(a2_3, p)
                    vf.store_align(scores, value, p, 128 + output + 3, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_3, p, [48 + output + 3, 0])
            if SLOW_PLANE > 4:
                if True:
                    value = vf.reduce_sum(a2_4, p)
                    vf.store_align(scores, value, p, 128 + output + 4, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_4, p, [48 + output + 4, 0])
            if SLOW_PLANE > 5:
                if True:
                    value = vf.reduce_sum(a2_5, p)
                    vf.store_align(scores, value, p, 128 + output + 5, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_5, p, [48 + output + 5, 0])
            if SLOW_PLANE > 6:
                if True:
                    value = vf.reduce_sum(a2_6, p)
                    vf.store_align(scores, value, p, 128 + output + 6, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_6, p, [48 + output + 6, 0])
            if SLOW_PLANE > 7:
                if True:
                    value = vf.reduce_sum(a2_7, p)
                    vf.store_align(scores, value, p, 128 + output + 7, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_7, p, [48 + output + 7, 0])
        if SLOW_BATCH > 3:
            if SLOW_PLANE > 0:
                if True:
                    value = vf.reduce_sum(a3_0, p)
                    vf.store_align(scores, value, p, 192 + output + 0, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_0, p, [72 + output + 0, 0])
            if SLOW_PLANE > 1:
                if True:
                    value = vf.reduce_sum(a3_1, p)
                    vf.store_align(scores, value, p, 192 + output + 1, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_1, p, [72 + output + 1, 0])
            if SLOW_PLANE > 2:
                if True:
                    value = vf.reduce_sum(a3_2, p)
                    vf.store_align(scores, value, p, 192 + output + 2, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_2, p, [72 + output + 2, 0])
            if SLOW_PLANE > 3:
                if True:
                    value = vf.reduce_sum(a3_3, p)
                    vf.store_align(scores, value, p, 192 + output + 3, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_3, p, [72 + output + 3, 0])
            if SLOW_PLANE > 4:
                if True:
                    value = vf.reduce_sum(a3_4, p)
                    vf.store_align(scores, value, p, 192 + output + 4, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_4, p, [72 + output + 4, 0])
            if SLOW_PLANE > 5:
                if True:
                    value = vf.reduce_sum(a3_5, p)
                    vf.store_align(scores, value, p, 192 + output + 5, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_5, p, [72 + output + 5, 0])
            if SLOW_PLANE > 6:
                if True:
                    value = vf.reduce_sum(a3_6, p)
                    vf.store_align(scores, value, p, 192 + output + 6, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_6, p, [72 + output + 6, 0])
            if SLOW_PLANE > 7:
                if True:
                    value = vf.reduce_sum(a3_7, p)
                    vf.store_align(scores, value, p, 192 + output + 7, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_7, p, [72 + output + 7, 0])


@pl.vector_function
def _slow_finish_pair(scores, stats, scales, bases, prev, post, comb, pre_eps, post_scale, repeat, eps, count):
    pall = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
    pu = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    ids = vf.arange(0, dtype=pl.DT_UINT32)
    c4 = vf.full(4, pu, dtype=pl.DT_UINT32)
    c8 = vf.full(8, pu, dtype=pl.DT_UINT32)
    c24 = vf.full(24, pu, dtype=pl.DT_UINT32)
    c31 = vf.full(31, pu, dtype=pl.DT_UINT32)
    c32 = vf.full(32, pu, dtype=pl.DT_UINT32)
    c7 = vf.full(7, pu, dtype=pl.DT_UINT32)
    uthree = vf.full(3, pu, dtype=pl.DT_UINT32)
    uone = vf.full(1, pu, dtype=pl.DT_UINT32)
    localid = vf.and_(ids, c31, pu)
    tokenbase = vf.and_(ids, c32, pu)
    first = vf.lt(ids, c32, pu)
    pcount = vf.update_mask(count * 32, dtype=F32)
    p24 = vf.lt(localid, c24, pcount)
    p8 = vf.lt(localid, c8, pcount)
    lt4 = vf.lt(localid, c4, pu)
    lt8 = vf.lt(localid, c8, pu)
    s0 = vf.load_align(scales, 0, dist=pl.LoadDist.BRC_B32)
    s1 = vf.load_align(scales, 1, dist=pl.LoadDist.BRC_B32)
    s2 = vf.load_align(scales, 2, dist=pl.LoadDist.BRC_B32)
    scale01 = vf.select(s0, s1, lt4)
    scale = vf.select(scale01, s2, lt8)
    rawbase = vf.load_align(bases, 0)
    base = vf.gather(rawbase, localid)
    q0 = vf.load_align(scores, 0)
    q1 = vf.load_align(scores, 64)
    q0 = vf.gather(q0, localid)
    q1 = vf.gather(q1, localid)
    q = vf.select(q0, q1, first)
    inv0 = vf.load_align(stats, 0, dist=pl.LoadDist.BRC_B32)
    inv1 = vf.load_align(stats, 8, dist=pl.LoadDist.BRC_B32)
    inv = vf.select(inv0, inv1, first)
    value = vf.mul(q, inv, p24)
    value = vf.mul(value, scale, p24)
    value = vf.add(value, base, p24)
    one = vf.full(1.0, pall, dtype=F32)
    neg = vf.muls(value, -1.0, p8)
    expv = vf.exp(neg, p8)
    den = vf.adds(expv, 1.0, p8)
    sig = vf.div(one, den, p8)
    cols = vf.and_(ids, c7, pu)
    row = vf.shift_right(ids, 3, pu)
    row01 = vf.and_(row, uone, pu)
    idx = vf.shift_left(row01, 5, pu)
    idx = vf.add(idx, cols, pu)
    pv = vf.gather(sig, idx)
    pv = vf.adds(pv, pre_eps, pall)
    pstore = vf.update_mask(count * 8, dtype=F32)
    pstore = vf.lt(cols, c4, pstore)
    vf.store_align(prev, pv, pstore, 0)
    postidx = vf.adds(idx, 4, pu)
    pp = vf.gather(sig, postidx)
    pp = vf.muls(pp, post_scale, pall)
    vf.store_align(post, pp, pstore, 0)
    active = vf.lt(cols, c4, pcount)
    matrixrow = vf.and_(row, uthree, pu)
    qidx = vf.shift_left(matrixrow, 2, pu)
    qidx = vf.add(qidx, cols, pu)
    qidx = vf.add(qidx, tokenbase, pu)
    qidx = vf.adds(qidx, 8, pu)
    v = vf.gather(value, qidx)
    mx = vf.reduce_max(v, active, datablock=True)
    mb = vf.gather(mx, row)
    v = vf.sub(v, mb, active)
    v = vf.exp(v, active)
    sums = vf.reduce_sum(v, active, datablock=True)
    sb = vf.gather(sums, row)
    v = vf.div(v, sb, active)
    v = vf.adds(v, eps, active)
    colidx = vf.add(tokenbase, cols, pu)
    i1 = vf.adds(colidx, 8, pu)
    i2 = vf.adds(colidx, 16, pu)
    i3 = vf.adds(colidx, 24, pu)
    c0 = vf.gather(v, colidx)
    c1 = vf.gather(v, i1)
    c2 = vf.gather(v, i2)
    c3 = vf.gather(v, i3)
    cs = vf.add(c0, c1, active)
    cs = vf.add(cs, c2, active)
    cs = vf.add(cs, c3, active)
    cs = vf.adds(cs, eps, active)
    v = vf.div(v, cs, active)
    for _ in pl.range(repeat - 1):
        sums = vf.reduce_sum(v, active, datablock=True)
        sb = vf.gather(sums, row)
        sb = vf.adds(sb, eps, active)
        v = vf.div(v, sb, active)
        c0 = vf.gather(v, colidx)
        c1 = vf.gather(v, i1)
        c2 = vf.gather(v, i2)
        c3 = vf.gather(v, i3)
        cs = vf.add(c0, c1, active)
        cs = vf.add(cs, c2, active)
        cs = vf.add(cs, c3, active)
        cs = vf.adds(cs, eps, active)
        v = vf.div(v, cs, active)
    vf.store_align(comb, v, active, 0)


@pl.vector_function
def _slow_dot_last_finish_single(
    r0,
    r1,
    r2,
    r3,
    w,
    partial,
    scores,
    route,
    stats,
    scales,
    bases,
    prev,
    post,
    comb,
    pre_eps,
    post_scale,
    repeat,
    eps,
    count,
):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
    for output in pl.range(0, 24, SLOW_PLANE):
        if SLOW_BATCH > 0:
            if SLOW_PLANE > 0:
                a0_0 = vf.full(0.0, p, dtype=F32)
                if True:
                    a0_0 = vf.load_align(partial, [0 + output + 0, 0])
            if SLOW_PLANE > 1:
                a0_1 = vf.full(0.0, p, dtype=F32)
                if True:
                    a0_1 = vf.load_align(partial, [0 + output + 1, 0])
            if SLOW_PLANE > 2:
                a0_2 = vf.full(0.0, p, dtype=F32)
                if True:
                    a0_2 = vf.load_align(partial, [0 + output + 2, 0])
            if SLOW_PLANE > 3:
                a0_3 = vf.full(0.0, p, dtype=F32)
                if True:
                    a0_3 = vf.load_align(partial, [0 + output + 3, 0])
            if SLOW_PLANE > 4:
                a0_4 = vf.full(0.0, p, dtype=F32)
                if True:
                    a0_4 = vf.load_align(partial, [0 + output + 4, 0])
            if SLOW_PLANE > 5:
                a0_5 = vf.full(0.0, p, dtype=F32)
                if True:
                    a0_5 = vf.load_align(partial, [0 + output + 5, 0])
            if SLOW_PLANE > 6:
                a0_6 = vf.full(0.0, p, dtype=F32)
                if True:
                    a0_6 = vf.load_align(partial, [0 + output + 6, 0])
            if SLOW_PLANE > 7:
                a0_7 = vf.full(0.0, p, dtype=F32)
                if True:
                    a0_7 = vf.load_align(partial, [0 + output + 7, 0])
        if SLOW_BATCH > 1:
            if SLOW_PLANE > 0:
                a1_0 = vf.full(0.0, p, dtype=F32)
                if True:
                    a1_0 = vf.load_align(partial, [24 + output + 0, 0])
            if SLOW_PLANE > 1:
                a1_1 = vf.full(0.0, p, dtype=F32)
                if True:
                    a1_1 = vf.load_align(partial, [24 + output + 1, 0])
            if SLOW_PLANE > 2:
                a1_2 = vf.full(0.0, p, dtype=F32)
                if True:
                    a1_2 = vf.load_align(partial, [24 + output + 2, 0])
            if SLOW_PLANE > 3:
                a1_3 = vf.full(0.0, p, dtype=F32)
                if True:
                    a1_3 = vf.load_align(partial, [24 + output + 3, 0])
            if SLOW_PLANE > 4:
                a1_4 = vf.full(0.0, p, dtype=F32)
                if True:
                    a1_4 = vf.load_align(partial, [24 + output + 4, 0])
            if SLOW_PLANE > 5:
                a1_5 = vf.full(0.0, p, dtype=F32)
                if True:
                    a1_5 = vf.load_align(partial, [24 + output + 5, 0])
            if SLOW_PLANE > 6:
                a1_6 = vf.full(0.0, p, dtype=F32)
                if True:
                    a1_6 = vf.load_align(partial, [24 + output + 6, 0])
            if SLOW_PLANE > 7:
                a1_7 = vf.full(0.0, p, dtype=F32)
                if True:
                    a1_7 = vf.load_align(partial, [24 + output + 7, 0])
        if SLOW_BATCH > 2:
            if SLOW_PLANE > 0:
                a2_0 = vf.full(0.0, p, dtype=F32)
                if True:
                    a2_0 = vf.load_align(partial, [48 + output + 0, 0])
            if SLOW_PLANE > 1:
                a2_1 = vf.full(0.0, p, dtype=F32)
                if True:
                    a2_1 = vf.load_align(partial, [48 + output + 1, 0])
            if SLOW_PLANE > 2:
                a2_2 = vf.full(0.0, p, dtype=F32)
                if True:
                    a2_2 = vf.load_align(partial, [48 + output + 2, 0])
            if SLOW_PLANE > 3:
                a2_3 = vf.full(0.0, p, dtype=F32)
                if True:
                    a2_3 = vf.load_align(partial, [48 + output + 3, 0])
            if SLOW_PLANE > 4:
                a2_4 = vf.full(0.0, p, dtype=F32)
                if True:
                    a2_4 = vf.load_align(partial, [48 + output + 4, 0])
            if SLOW_PLANE > 5:
                a2_5 = vf.full(0.0, p, dtype=F32)
                if True:
                    a2_5 = vf.load_align(partial, [48 + output + 5, 0])
            if SLOW_PLANE > 6:
                a2_6 = vf.full(0.0, p, dtype=F32)
                if True:
                    a2_6 = vf.load_align(partial, [48 + output + 6, 0])
            if SLOW_PLANE > 7:
                a2_7 = vf.full(0.0, p, dtype=F32)
                if True:
                    a2_7 = vf.load_align(partial, [48 + output + 7, 0])
        if SLOW_BATCH > 3:
            if SLOW_PLANE > 0:
                a3_0 = vf.full(0.0, p, dtype=F32)
                if True:
                    a3_0 = vf.load_align(partial, [72 + output + 0, 0])
            if SLOW_PLANE > 1:
                a3_1 = vf.full(0.0, p, dtype=F32)
                if True:
                    a3_1 = vf.load_align(partial, [72 + output + 1, 0])
            if SLOW_PLANE > 2:
                a3_2 = vf.full(0.0, p, dtype=F32)
                if True:
                    a3_2 = vf.load_align(partial, [72 + output + 2, 0])
            if SLOW_PLANE > 3:
                a3_3 = vf.full(0.0, p, dtype=F32)
                if True:
                    a3_3 = vf.load_align(partial, [72 + output + 3, 0])
            if SLOW_PLANE > 4:
                a3_4 = vf.full(0.0, p, dtype=F32)
                if True:
                    a3_4 = vf.load_align(partial, [72 + output + 4, 0])
            if SLOW_PLANE > 5:
                a3_5 = vf.full(0.0, p, dtype=F32)
                if True:
                    a3_5 = vf.load_align(partial, [72 + output + 5, 0])
            if SLOW_PLANE > 6:
                a3_6 = vf.full(0.0, p, dtype=F32)
                if True:
                    a3_6 = vf.load_align(partial, [72 + output + 6, 0])
            if SLOW_PLANE > 7:
                a3_7 = vf.full(0.0, p, dtype=F32)
                if True:
                    a3_7 = vf.load_align(partial, [72 + output + 7, 0])
        for col in pl.range(0, SLOW_CHUNK, 64):
            if SLOW_BATCH > 0:
                rv0 = vf.load_align(r0, [route, col])
            if SLOW_BATCH > 1:
                rv1 = vf.load_align(r1, [route, col])
            if SLOW_BATCH > 2:
                rv2 = vf.load_align(r2, [route, col])
            if SLOW_BATCH > 3:
                rv3 = vf.load_align(r3, [route, col])
            if SLOW_PLANE > 0:
                wv0 = vf.load_align(w, [output + 0, col])
            if SLOW_PLANE > 1:
                wv1 = vf.load_align(w, [output + 1, col])
            if SLOW_PLANE > 2:
                wv2 = vf.load_align(w, [output + 2, col])
            if SLOW_PLANE > 3:
                wv3 = vf.load_align(w, [output + 3, col])
            if SLOW_PLANE > 0:
                if SLOW_BATCH > 0:
                    a0_0 = vf.mul_add_dst(rv0, wv0, p)
                if SLOW_BATCH > 1:
                    a1_0 = vf.mul_add_dst(rv1, wv0, p)
                if SLOW_BATCH > 2:
                    a2_0 = vf.mul_add_dst(rv2, wv0, p)
                if SLOW_BATCH > 3:
                    a3_0 = vf.mul_add_dst(rv3, wv0, p)
            if SLOW_PLANE > 1:
                if SLOW_BATCH > 0:
                    a0_1 = vf.mul_add_dst(rv0, wv1, p)
                if SLOW_BATCH > 1:
                    a1_1 = vf.mul_add_dst(rv1, wv1, p)
                if SLOW_BATCH > 2:
                    a2_1 = vf.mul_add_dst(rv2, wv1, p)
                if SLOW_BATCH > 3:
                    a3_1 = vf.mul_add_dst(rv3, wv1, p)
            if SLOW_PLANE > 2:
                if SLOW_BATCH > 0:
                    a0_2 = vf.mul_add_dst(rv0, wv2, p)
                if SLOW_BATCH > 1:
                    a1_2 = vf.mul_add_dst(rv1, wv2, p)
                if SLOW_BATCH > 2:
                    a2_2 = vf.mul_add_dst(rv2, wv2, p)
                if SLOW_BATCH > 3:
                    a3_2 = vf.mul_add_dst(rv3, wv2, p)
            if SLOW_PLANE > 3:
                if SLOW_BATCH > 0:
                    a0_3 = vf.mul_add_dst(rv0, wv3, p)
                if SLOW_BATCH > 1:
                    a1_3 = vf.mul_add_dst(rv1, wv3, p)
                if SLOW_BATCH > 2:
                    a2_3 = vf.mul_add_dst(rv2, wv3, p)
                if SLOW_BATCH > 3:
                    a3_3 = vf.mul_add_dst(rv3, wv3, p)
            if SLOW_PLANE > 4:
                wv0 = vf.load_align(w, [output + 4, col])
            if SLOW_PLANE > 5:
                wv1 = vf.load_align(w, [output + 5, col])
            if SLOW_PLANE > 6:
                wv2 = vf.load_align(w, [output + 6, col])
            if SLOW_PLANE > 7:
                wv3 = vf.load_align(w, [output + 7, col])
            if SLOW_PLANE > 4:
                if SLOW_BATCH > 0:
                    a0_4 = vf.mul_add_dst(rv0, wv0, p)
                if SLOW_BATCH > 1:
                    a1_4 = vf.mul_add_dst(rv1, wv0, p)
                if SLOW_BATCH > 2:
                    a2_4 = vf.mul_add_dst(rv2, wv0, p)
                if SLOW_BATCH > 3:
                    a3_4 = vf.mul_add_dst(rv3, wv0, p)
            if SLOW_PLANE > 5:
                if SLOW_BATCH > 0:
                    a0_5 = vf.mul_add_dst(rv0, wv1, p)
                if SLOW_BATCH > 1:
                    a1_5 = vf.mul_add_dst(rv1, wv1, p)
                if SLOW_BATCH > 2:
                    a2_5 = vf.mul_add_dst(rv2, wv1, p)
                if SLOW_BATCH > 3:
                    a3_5 = vf.mul_add_dst(rv3, wv1, p)
            if SLOW_PLANE > 6:
                if SLOW_BATCH > 0:
                    a0_6 = vf.mul_add_dst(rv0, wv2, p)
                if SLOW_BATCH > 1:
                    a1_6 = vf.mul_add_dst(rv1, wv2, p)
                if SLOW_BATCH > 2:
                    a2_6 = vf.mul_add_dst(rv2, wv2, p)
                if SLOW_BATCH > 3:
                    a3_6 = vf.mul_add_dst(rv3, wv2, p)
            if SLOW_PLANE > 7:
                if SLOW_BATCH > 0:
                    a0_7 = vf.mul_add_dst(rv0, wv3, p)
                if SLOW_BATCH > 1:
                    a1_7 = vf.mul_add_dst(rv1, wv3, p)
                if SLOW_BATCH > 2:
                    a2_7 = vf.mul_add_dst(rv2, wv3, p)
                if SLOW_BATCH > 3:
                    a3_7 = vf.mul_add_dst(rv3, wv3, p)
        if SLOW_BATCH > 0:
            if SLOW_PLANE > 0:
                if True:
                    value = vf.reduce_sum(a0_0, p)
                    vf.store_align(scores, value, p, 0 + output + 0, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_0, p, [0 + output + 0, 0])
            if SLOW_PLANE > 1:
                if True:
                    value = vf.reduce_sum(a0_1, p)
                    vf.store_align(scores, value, p, 0 + output + 1, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_1, p, [0 + output + 1, 0])
            if SLOW_PLANE > 2:
                if True:
                    value = vf.reduce_sum(a0_2, p)
                    vf.store_align(scores, value, p, 0 + output + 2, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_2, p, [0 + output + 2, 0])
            if SLOW_PLANE > 3:
                if True:
                    value = vf.reduce_sum(a0_3, p)
                    vf.store_align(scores, value, p, 0 + output + 3, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_3, p, [0 + output + 3, 0])
            if SLOW_PLANE > 4:
                if True:
                    value = vf.reduce_sum(a0_4, p)
                    vf.store_align(scores, value, p, 0 + output + 4, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_4, p, [0 + output + 4, 0])
            if SLOW_PLANE > 5:
                if True:
                    value = vf.reduce_sum(a0_5, p)
                    vf.store_align(scores, value, p, 0 + output + 5, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_5, p, [0 + output + 5, 0])
            if SLOW_PLANE > 6:
                if True:
                    value = vf.reduce_sum(a0_6, p)
                    vf.store_align(scores, value, p, 0 + output + 6, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_6, p, [0 + output + 6, 0])
            if SLOW_PLANE > 7:
                if True:
                    value = vf.reduce_sum(a0_7, p)
                    vf.store_align(scores, value, p, 0 + output + 7, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a0_7, p, [0 + output + 7, 0])
        if SLOW_BATCH > 1:
            if SLOW_PLANE > 0:
                if True:
                    value = vf.reduce_sum(a1_0, p)
                    vf.store_align(scores, value, p, 64 + output + 0, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_0, p, [24 + output + 0, 0])
            if SLOW_PLANE > 1:
                if True:
                    value = vf.reduce_sum(a1_1, p)
                    vf.store_align(scores, value, p, 64 + output + 1, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_1, p, [24 + output + 1, 0])
            if SLOW_PLANE > 2:
                if True:
                    value = vf.reduce_sum(a1_2, p)
                    vf.store_align(scores, value, p, 64 + output + 2, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_2, p, [24 + output + 2, 0])
            if SLOW_PLANE > 3:
                if True:
                    value = vf.reduce_sum(a1_3, p)
                    vf.store_align(scores, value, p, 64 + output + 3, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_3, p, [24 + output + 3, 0])
            if SLOW_PLANE > 4:
                if True:
                    value = vf.reduce_sum(a1_4, p)
                    vf.store_align(scores, value, p, 64 + output + 4, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_4, p, [24 + output + 4, 0])
            if SLOW_PLANE > 5:
                if True:
                    value = vf.reduce_sum(a1_5, p)
                    vf.store_align(scores, value, p, 64 + output + 5, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_5, p, [24 + output + 5, 0])
            if SLOW_PLANE > 6:
                if True:
                    value = vf.reduce_sum(a1_6, p)
                    vf.store_align(scores, value, p, 64 + output + 6, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_6, p, [24 + output + 6, 0])
            if SLOW_PLANE > 7:
                if True:
                    value = vf.reduce_sum(a1_7, p)
                    vf.store_align(scores, value, p, 64 + output + 7, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a1_7, p, [24 + output + 7, 0])
        if SLOW_BATCH > 2:
            if SLOW_PLANE > 0:
                if True:
                    value = vf.reduce_sum(a2_0, p)
                    vf.store_align(scores, value, p, 128 + output + 0, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_0, p, [48 + output + 0, 0])
            if SLOW_PLANE > 1:
                if True:
                    value = vf.reduce_sum(a2_1, p)
                    vf.store_align(scores, value, p, 128 + output + 1, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_1, p, [48 + output + 1, 0])
            if SLOW_PLANE > 2:
                if True:
                    value = vf.reduce_sum(a2_2, p)
                    vf.store_align(scores, value, p, 128 + output + 2, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_2, p, [48 + output + 2, 0])
            if SLOW_PLANE > 3:
                if True:
                    value = vf.reduce_sum(a2_3, p)
                    vf.store_align(scores, value, p, 128 + output + 3, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_3, p, [48 + output + 3, 0])
            if SLOW_PLANE > 4:
                if True:
                    value = vf.reduce_sum(a2_4, p)
                    vf.store_align(scores, value, p, 128 + output + 4, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_4, p, [48 + output + 4, 0])
            if SLOW_PLANE > 5:
                if True:
                    value = vf.reduce_sum(a2_5, p)
                    vf.store_align(scores, value, p, 128 + output + 5, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_5, p, [48 + output + 5, 0])
            if SLOW_PLANE > 6:
                if True:
                    value = vf.reduce_sum(a2_6, p)
                    vf.store_align(scores, value, p, 128 + output + 6, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_6, p, [48 + output + 6, 0])
            if SLOW_PLANE > 7:
                if True:
                    value = vf.reduce_sum(a2_7, p)
                    vf.store_align(scores, value, p, 128 + output + 7, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a2_7, p, [48 + output + 7, 0])
        if SLOW_BATCH > 3:
            if SLOW_PLANE > 0:
                if True:
                    value = vf.reduce_sum(a3_0, p)
                    vf.store_align(scores, value, p, 192 + output + 0, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_0, p, [72 + output + 0, 0])
            if SLOW_PLANE > 1:
                if True:
                    value = vf.reduce_sum(a3_1, p)
                    vf.store_align(scores, value, p, 192 + output + 1, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_1, p, [72 + output + 1, 0])
            if SLOW_PLANE > 2:
                if True:
                    value = vf.reduce_sum(a3_2, p)
                    vf.store_align(scores, value, p, 192 + output + 2, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_2, p, [72 + output + 2, 0])
            if SLOW_PLANE > 3:
                if True:
                    value = vf.reduce_sum(a3_3, p)
                    vf.store_align(scores, value, p, 192 + output + 3, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_3, p, [72 + output + 3, 0])
            if SLOW_PLANE > 4:
                if True:
                    value = vf.reduce_sum(a3_4, p)
                    vf.store_align(scores, value, p, 192 + output + 4, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_4, p, [72 + output + 4, 0])
            if SLOW_PLANE > 5:
                if True:
                    value = vf.reduce_sum(a3_5, p)
                    vf.store_align(scores, value, p, 192 + output + 5, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_5, p, [72 + output + 5, 0])
            if SLOW_PLANE > 6:
                if True:
                    value = vf.reduce_sum(a3_6, p)
                    vf.store_align(scores, value, p, 192 + output + 6, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_6, p, [72 + output + 6, 0])
            if SLOW_PLANE > 7:
                if True:
                    value = vf.reduce_sum(a3_7, p)
                    vf.store_align(scores, value, p, 192 + output + 7, dist=pl.StoreDist.FIRST_ELEMENT_B32)
                else:
                    vf.store_align(partial, a3_7, p, [72 + output + 7, 0])
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    pall = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
    pu = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    p24 = vf.update_mask(24, dtype=F32)
    p8 = vf.update_mask(8, dtype=F32)
    p4 = vf.update_mask(4, dtype=F32)
    p32 = vf.update_mask(32, dtype=pl.DT_UINT32)
    ids = vf.arange(0, dtype=pl.DT_UINT32)
    four = vf.full(4, pu, dtype=pl.DT_UINT32)
    eight = vf.full(8, pu, dtype=pl.DT_UINT32)
    lt4 = vf.lt(ids, four, pu)
    lt8 = vf.lt(ids, eight, pu)
    s0 = vf.load_align(scales, 0, dist=pl.LoadDist.BRC_B32)
    s1 = vf.load_align(scales, 1, dist=pl.LoadDist.BRC_B32)
    s2 = vf.load_align(scales, 2, dist=pl.LoadDist.BRC_B32)
    scale01 = vf.select(s0, s1, lt4)
    scale = vf.select(scale01, s2, lt8)
    one = vf.full(1.0, pall, dtype=F32)
    base = vf.load_align(bases, 0)
    idx4 = vf.adds(ids, 4, pu)
    seven = vf.full(7, pu, dtype=pl.DT_UINT32)
    cols = vf.and_(ids, seven, pu)
    rows = vf.shift_right(ids, 3, pu)
    active = vf.lt(cols, four, p32)
    r4 = vf.shift_left(rows, 2, pu)
    qidx = vf.add(r4, cols, pu)
    qidx = vf.adds(qidx, 8, pu)
    i1 = vf.adds(cols, 8, pu)
    i2 = vf.adds(cols, 16, pu)
    i3 = vf.adds(cols, 24, pu)
    inv = vf.load_align(stats, 0, dist=pl.LoadDist.BRC_B32)
    q = vf.load_align(scores, 0)
    value = vf.mul(q, inv, p24)
    value = vf.mul(value, scale, p24)
    value = vf.add(value, base, p24)
    neg = vf.muls(value, -1.0, p8)
    expv = vf.exp(neg, p8)
    den = vf.adds(expv, 1.0, p8)
    sig = vf.div(one, den, p8)
    sigpost = vf.gather(sig, idx4)
    pp = vf.muls(sigpost, post_scale, p4)
    vf.store_align(post, pp, p4, 0)
    pv = vf.adds(sig, pre_eps, p4)
    vf.store_align(prev, pv, p4, 0)
    v = vf.gather(value, qidx)
    mx = vf.reduce_max(v, active, datablock=True)
    mb = vf.gather(mx, rows)
    v = vf.sub(v, mb, active)
    v = vf.exp(v, active)
    sums = vf.reduce_sum(v, active, datablock=True)
    sb = vf.gather(sums, rows)
    v = vf.div(v, sb, active)
    v = vf.adds(v, eps, active)
    c0 = vf.gather(v, cols)
    c1 = vf.gather(v, i1)
    c2 = vf.gather(v, i2)
    c3 = vf.gather(v, i3)
    cs = vf.add(c0, c1, active)
    cs = vf.add(cs, c2, active)
    cs = vf.add(cs, c3, active)
    cs = vf.adds(cs, eps, active)
    v = vf.div(v, cs, active)
    for _ in pl.range(repeat - 1):
        sums = vf.reduce_sum(v, active, datablock=True)
        sb = vf.gather(sums, rows)
        sb = vf.adds(sb, eps, active)
        v = vf.div(v, sb, active)
        c0 = vf.gather(v, cols)
        c1 = vf.gather(v, i1)
        c2 = vf.gather(v, i2)
        c3 = vf.gather(v, i3)
        cs = vf.add(c0, c1, active)
        cs = vf.add(cs, c2, active)
        cs = vf.add(cs, c3, active)
        cs = vf.adds(cs, eps, active)
        v = vf.div(v, cs, active)
    vf.store_align(comb, v, active, 0)


@pl.vector_function
def range_init(scales, stats):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
    pu = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    p3 = vf.update_mask(3, dtype=F32)
    sv = vf.load_align(scales, [0, 0])
    bits = vf.bit_cast(sv, dtype=pl.DT_UINT32)
    signless = vf.full(2147483647, pu, dtype=pl.DT_UINT32)
    absolute = vf.and_(bits, signless, pu)
    exponent = vf.shift_right(absolute, 23, pu)
    ei = vf.bit_cast(exponent, dtype=pl.DT_INT32)
    ef = vf.astype(ei, p, dtype=F32)
    peak = vf.reduce_max(ef, p3)
    peak = vf.full(peak, p)
    threshold = vf.full(131.0, p, dtype=F32)
    unsafe = vf.ge(peak, threshold, p)
    one = vf.full(1.0, p, dtype=F32)
    zero = vf.full(0.0, p, dtype=F32)
    bad = vf.select(one, zero, unsafe)
    minimum = vf.full(255.0, p, dtype=F32)
    maximum = vf.full(0.0, p, dtype=F32)
    vf.store_align(stats, minimum, p, [0, 0])
    vf.store_align(stats, maximum, p, [1, 0])
    vf.store_align(stats, bad, p, [2, 0])


@pl.vector_function
def range_panel(w, stats, rows):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
    pu = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    signless = vf.full(2147483647, pu, dtype=pl.DT_UINT32)
    zero = vf.full(0, pu, dtype=pl.DT_UINT32)
    empty = vf.full(255.0, p, dtype=F32)
    minimum = vf.load_align(stats, [0, 0])
    maximum = vf.load_align(stats, [1, 0])
    for row in pl.range(rows):
        for col in pl.range(0, 256, 64):
            value = vf.load_align(w, [row, col])
            bits = vf.bit_cast(value, dtype=pl.DT_UINT32)
            absolute = vf.and_(bits, signless, pu)
            exponent = vf.shift_right(absolute, 23, pu)
            ei = vf.bit_cast(exponent, dtype=pl.DT_INT32)
            ef = vf.astype(ei, p, dtype=F32)
            nonzero = vf.ne(absolute, zero, pu)
            nonzero_exp = vf.select(ef, empty, nonzero)
            minimum = vf.min(minimum, nonzero_exp, p)
            maximum = vf.max(maximum, ef, p)
    vf.store_align(stats, minimum, p, [0, 0])
    vf.store_align(stats, maximum, p, [1, 0])


@pl.vector_function
def range_record(stats, out):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
    pu = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    pi = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
    p8 = vf.update_mask(8, dtype=pl.DT_INT32)
    minimum = vf.load_align(stats, [0, 0])
    maximum = vf.load_align(stats, [1, 0])
    bad = vf.load_align(stats, [2, 0])
    minimum = vf.reduce_min(minimum, p)
    maximum = vf.reduce_max(maximum, p)
    minimum = vf.full(minimum, p)
    maximum = vf.full(maximum, p)
    zero = vf.full(0.0, p, dtype=F32)
    ids = vf.arange(0, dtype=pl.DT_UINT32)
    c0 = vf.full(0, pu, dtype=pl.DT_UINT32)
    c1 = vf.full(1, pu, dtype=pl.DT_UINT32)
    c2 = vf.full(2, pu, dtype=pl.DT_UINT32)
    m0 = vf.eq(ids, c0, pu)
    m1 = vf.eq(ids, c1, pu)
    m2 = vf.eq(ids, c2, pu)
    record = vf.select(bad, zero, m0)
    record = vf.select(minimum, record, m1)
    record = vf.select(maximum, record, m2)
    integer = vf.astype(record, pi, dtype=pl.DT_INT32)
    vf.store_align(out, integer, p8, [0, 0])


@pl.vector_function
def range_merge(records, out, shards):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
    pu = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    pi = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
    p16 = vf.update_mask(16, dtype=pl.DT_INT32)
    minimum = vf.full(255.0, p, dtype=F32)
    maximum = vf.full(0.0, p, dtype=F32)
    bad = vf.full(0.0, p, dtype=F32)
    zero = vf.full(0.0, p, dtype=F32)
    one = vf.full(1.0, p, dtype=F32)
    low_bound = vf.full(32.0, p, dtype=F32)
    high_bound = vf.full(222.0, p, dtype=F32)
    span_bound = vf.full(64.0, p, dtype=F32)
    for shard in pl.range(shards):
        raw = vf.load_align(records, [shard // 4 * 4, 0])
        values = vf.astype(raw, p, dtype=F32)
        for sub in pl.range(2):
            i0 = vf.full(shard % 4 * 16 + sub * 8, pu, dtype=pl.DT_UINT32)
            i1 = vf.full(shard % 4 * 16 + sub * 8 + 1, pu, dtype=pl.DT_UINT32)
            i2 = vf.full(shard % 4 * 16 + sub * 8 + 2, pu, dtype=pl.DT_UINT32)
            b = vf.gather(values, i0)
            lo = vf.gather(values, i1)
            hi = vf.gather(values, i2)
            bad = vf.max(bad, b, p)
            minimum = vf.min(minimum, lo, p)
            maximum = vf.max(maximum, hi, p)
    low = vf.lt(minimum, low_bound, p)
    high = vf.gt(maximum, high_bound, p)
    span = vf.sub(maximum, minimum, p)
    wide = vf.gt(span, span_bound, p)
    bad = vf.select(one, bad, low)
    bad = vf.select(one, bad, high)
    bad = vf.select(one, bad, wide)
    integer = vf.astype(bad, pi, dtype=pl.DT_INT32)
    vf.store_align(out, integer, p16, [0, 0])


@pl.vector_function
def _post128_pack_first(r, x, u, post, comb, prev, partial, z0, z1, z2, z3, basecol, count, batch):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
    ph = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=B16)
    pu = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    for row in pl.range(count):
        ss = vf.full(0.0, p, dtype=F32)
        us = vf.full(0.0, p, dtype=F32)
        xb_lo = vf.load_align(x, [row, basecol + 0], dist=pl.LoadDist.UNPK_B16)
        xv_lo = vf.astype(xb_lo, p, dtype=F32)
        rb = vf.load_align(r, [row * 4 + 0, basecol + 0], dist=pl.LoadDist.UNPK_B16)
        r0_lo = vf.astype(rb, p, dtype=F32)
        rb = vf.load_align(r, [row * 4 + 1, basecol + 0], dist=pl.LoadDist.UNPK_B16)
        r1_lo = vf.astype(rb, p, dtype=F32)
        rb = vf.load_align(r, [row * 4 + 2, basecol + 0], dist=pl.LoadDist.UNPK_B16)
        r2_lo = vf.astype(rb, p, dtype=F32)
        rb = vf.load_align(r, [row * 4 + 3, basecol + 0], dist=pl.LoadDist.UNPK_B16)
        r3_lo = vf.astype(rb, p, dtype=F32)
        xb_hi = vf.load_align(x, [row, basecol + 64], dist=pl.LoadDist.UNPK_B16)
        xv_hi = vf.astype(xb_hi, p, dtype=F32)
        rb = vf.load_align(r, [row * 4 + 0, basecol + 64], dist=pl.LoadDist.UNPK_B16)
        r0_hi = vf.astype(rb, p, dtype=F32)
        rb = vf.load_align(r, [row * 4 + 1, basecol + 64], dist=pl.LoadDist.UNPK_B16)
        r1_hi = vf.astype(rb, p, dtype=F32)
        rb = vf.load_align(r, [row * 4 + 2, basecol + 64], dist=pl.LoadDist.UNPK_B16)
        r2_hi = vf.astype(rb, p, dtype=F32)
        rb = vf.load_align(r, [row * 4 + 3, basecol + 64], dist=pl.LoadDist.UNPK_B16)
        r3_hi = vf.astype(rb, p, dtype=F32)
        c = vf.load_align(comb, [row, 0], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul(r0_lo, c, p)
        v0_hi = vf.mul(r0_hi, c, p)
        c = vf.load_align(comb, [row, 4], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul_add_dst(r1_lo, c, p)
        v0_hi = vf.mul_add_dst(r1_hi, c, p)
        c = vf.load_align(comb, [row, 8], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul_add_dst(r2_lo, c, p)
        v0_hi = vf.mul_add_dst(r2_hi, c, p)
        c = vf.load_align(comb, [row, 12], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul_add_dst(r3_lo, c, p)
        v0_hi = vf.mul_add_dst(r3_hi, c, p)
        pv = vf.load_align(post, [row, 0], dist=pl.LoadDist.BRC_B32)
        xp = vf.mul(xv_lo, pv, p)
        v_lo = vf.add(v_lo, xp, p)
        xp = vf.mul(xv_hi, pv, p)
        v0_hi = vf.add(v0_hi, xp, p)
        b_lo = vf.astype(v_lo, ph, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        b_hi = vf.astype(v0_hi, ph, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        v_lo = vf.astype(b_lo, p, dtype=F32)
        v0_hi = vf.astype(b_hi, p, dtype=F32)
        packed = vf.bit_cast(b_lo, dtype=pl.DT_UINT32)
        vf.store_align(r, packed, pu, [row * 4 + 0, basecol + 0], dist=pl.StoreDist.PACK)
        packed = vf.bit_cast(b_hi, dtype=pl.DT_UINT32)
        vf.store_align(r, packed, pu, [row * 4 + 0, basecol + 64], dist=pl.StoreDist.PACK)
        dense, unused = vf.de_interleave(b_lo, b_hi)
        vf.store_align(
            z0,
            dense,
            ph,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=batch + 1,
            repeat_stride=1,
            post_update=True,
        )
        ss = vf.mul_add_dst(v_lo, v_lo, p)
        av = vf.load_align(prev, [row, 0], dist=pl.LoadDist.BRC_B32)
        uv_lo = vf.mul(v_lo, av, p)
        uv_hi = vf.mul(v0_hi, av, p)
        c = vf.load_align(comb, [row, 1], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul(r0_lo, c, p)
        v1_hi = vf.mul(r0_hi, c, p)
        c = vf.load_align(comb, [row, 5], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul_add_dst(r1_lo, c, p)
        v1_hi = vf.mul_add_dst(r1_hi, c, p)
        c = vf.load_align(comb, [row, 9], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul_add_dst(r2_lo, c, p)
        v1_hi = vf.mul_add_dst(r2_hi, c, p)
        c = vf.load_align(comb, [row, 13], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul_add_dst(r3_lo, c, p)
        v1_hi = vf.mul_add_dst(r3_hi, c, p)
        pv = vf.load_align(post, [row, 1], dist=pl.LoadDist.BRC_B32)
        xp = vf.mul(xv_lo, pv, p)
        v_lo = vf.add(v_lo, xp, p)
        xp = vf.mul(xv_hi, pv, p)
        v1_hi = vf.add(v1_hi, xp, p)
        b_lo = vf.astype(v_lo, ph, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        b_hi = vf.astype(v1_hi, ph, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        v_lo = vf.astype(b_lo, p, dtype=F32)
        v1_hi = vf.astype(b_hi, p, dtype=F32)
        packed = vf.bit_cast(b_lo, dtype=pl.DT_UINT32)
        vf.store_align(r, packed, pu, [row * 4 + 1, basecol + 0], dist=pl.StoreDist.PACK)
        packed = vf.bit_cast(b_hi, dtype=pl.DT_UINT32)
        vf.store_align(r, packed, pu, [row * 4 + 1, basecol + 64], dist=pl.StoreDist.PACK)
        dense, unused = vf.de_interleave(b_lo, b_hi)
        vf.store_align(
            z1,
            dense,
            ph,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=batch + 1,
            repeat_stride=1,
            post_update=True,
        )
        ss = vf.mul_add_dst(v_lo, v_lo, p)
        av = vf.load_align(prev, [row, 1], dist=pl.LoadDist.BRC_B32)
        q = vf.mul(v_lo, av, p)
        uv_lo = vf.add(uv_lo, q, p)
        q = vf.mul(v1_hi, av, p)
        uv_hi = vf.add(uv_hi, q, p)
        c = vf.load_align(comb, [row, 2], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul(r0_lo, c, p)
        v2_hi = vf.mul(r0_hi, c, p)
        c = vf.load_align(comb, [row, 6], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul_add_dst(r1_lo, c, p)
        v2_hi = vf.mul_add_dst(r1_hi, c, p)
        c = vf.load_align(comb, [row, 10], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul_add_dst(r2_lo, c, p)
        v2_hi = vf.mul_add_dst(r2_hi, c, p)
        c = vf.load_align(comb, [row, 14], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul_add_dst(r3_lo, c, p)
        v2_hi = vf.mul_add_dst(r3_hi, c, p)
        pv = vf.load_align(post, [row, 2], dist=pl.LoadDist.BRC_B32)
        xp = vf.mul(xv_lo, pv, p)
        v_lo = vf.add(v_lo, xp, p)
        xp = vf.mul(xv_hi, pv, p)
        v2_hi = vf.add(v2_hi, xp, p)
        b_lo = vf.astype(v_lo, ph, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        b_hi = vf.astype(v2_hi, ph, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        v_lo = vf.astype(b_lo, p, dtype=F32)
        v2_hi = vf.astype(b_hi, p, dtype=F32)
        packed = vf.bit_cast(b_lo, dtype=pl.DT_UINT32)
        vf.store_align(r, packed, pu, [row * 4 + 2, basecol + 0], dist=pl.StoreDist.PACK)
        packed = vf.bit_cast(b_hi, dtype=pl.DT_UINT32)
        vf.store_align(r, packed, pu, [row * 4 + 2, basecol + 64], dist=pl.StoreDist.PACK)
        dense, unused = vf.de_interleave(b_lo, b_hi)
        vf.store_align(
            z2,
            dense,
            ph,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=batch + 1,
            repeat_stride=1,
            post_update=True,
        )
        ss = vf.mul_add_dst(v_lo, v_lo, p)
        av = vf.load_align(prev, [row, 2], dist=pl.LoadDist.BRC_B32)
        q = vf.mul(v_lo, av, p)
        uv_lo = vf.add(uv_lo, q, p)
        q = vf.mul(v2_hi, av, p)
        uv_hi = vf.add(uv_hi, q, p)
        c = vf.load_align(comb, [row, 3], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul(r0_lo, c, p)
        v3_hi = vf.mul(r0_hi, c, p)
        c = vf.load_align(comb, [row, 7], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul_add_dst(r1_lo, c, p)
        v3_hi = vf.mul_add_dst(r1_hi, c, p)
        c = vf.load_align(comb, [row, 11], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul_add_dst(r2_lo, c, p)
        v3_hi = vf.mul_add_dst(r2_hi, c, p)
        c = vf.load_align(comb, [row, 15], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul_add_dst(r3_lo, c, p)
        v3_hi = vf.mul_add_dst(r3_hi, c, p)
        pv = vf.load_align(post, [row, 3], dist=pl.LoadDist.BRC_B32)
        xp = vf.mul(xv_lo, pv, p)
        v_lo = vf.add(v_lo, xp, p)
        xp = vf.mul(xv_hi, pv, p)
        v3_hi = vf.add(v3_hi, xp, p)
        b_lo = vf.astype(v_lo, ph, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        b_hi = vf.astype(v3_hi, ph, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        v_lo = vf.astype(b_lo, p, dtype=F32)
        v3_hi = vf.astype(b_hi, p, dtype=F32)
        packed = vf.bit_cast(b_lo, dtype=pl.DT_UINT32)
        vf.store_align(r, packed, pu, [row * 4 + 3, basecol + 0], dist=pl.StoreDist.PACK)
        packed = vf.bit_cast(b_hi, dtype=pl.DT_UINT32)
        vf.store_align(r, packed, pu, [row * 4 + 3, basecol + 64], dist=pl.StoreDist.PACK)
        dense, unused = vf.de_interleave(b_lo, b_hi)
        vf.store_align(
            z3,
            dense,
            ph,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=batch + 1,
            repeat_stride=1,
            post_update=True,
        )
        ss = vf.mul_add_dst(v_lo, v_lo, p)
        av = vf.load_align(prev, [row, 3], dist=pl.LoadDist.BRC_B32)
        q = vf.mul(v_lo, av, p)
        uv_lo = vf.add(uv_lo, q, p)
        q = vf.mul(v3_hi, av, p)
        uv_hi = vf.add(uv_hi, q, p)
        ss = vf.mul_add_dst(v0_hi, v0_hi, p)
        ss = vf.mul_add_dst(v1_hi, v1_hi, p)
        ss = vf.mul_add_dst(v2_hi, v2_hi, p)
        ss = vf.mul_add_dst(v3_hi, v3_hi, p)
        ub = vf.astype(uv_lo, ph, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        ur = vf.astype(ub, p, dtype=F32)
        us = vf.mul_add_dst(ur, ur, p)
        packed = vf.bit_cast(ub, dtype=pl.DT_UINT32)
        vf.store_align(u, packed, pu, [row, 0 + (basecol + 0)], dist=pl.StoreDist.PACK)
        ub = vf.astype(uv_hi, ph, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        ur = vf.astype(ub, p, dtype=F32)
        us = vf.mul_add_dst(ur, ur, p)
        packed = vf.bit_cast(ub, dtype=pl.DT_UINT32)
        vf.store_align(u, packed, pu, [row, 0 + (basecol + 64)], dist=pl.StoreDist.PACK)
        vf.store_align(partial, ss, p, [row * 2, 0])
        vf.store_align(partial, us, p, [row * 2 + 1, 0])
    zero = vf.full(0.0, ph, dtype=B16)
    for _ in pl.range(count, batch):
        vf.store_align(
            z0,
            zero,
            ph,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=batch + 1,
            repeat_stride=1,
            post_update=True,
        )
        vf.store_align(
            z1,
            zero,
            ph,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=batch + 1,
            repeat_stride=1,
            post_update=True,
        )
        vf.store_align(
            z2,
            zero,
            ph,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=batch + 1,
            repeat_stride=1,
            post_update=True,
        )
        vf.store_align(
            z3,
            zero,
            ph,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=batch + 1,
            repeat_stride=1,
            post_update=True,
        )


@pl.vector_function
def _post128_pack_next(r, x, u, post, comb, prev, partial, z0, z1, z2, z3, basecol, count, batch):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
    ph = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=B16)
    pu = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    for row in pl.range(count):
        ss = vf.load_align(partial, [row * 2, 0])
        us = vf.load_align(partial, [row * 2 + 1, 0])
        xb_lo = vf.load_align(x, [row, basecol + 0], dist=pl.LoadDist.UNPK_B16)
        xv_lo = vf.astype(xb_lo, p, dtype=F32)
        rb = vf.load_align(r, [row * 4 + 0, basecol + 0], dist=pl.LoadDist.UNPK_B16)
        r0_lo = vf.astype(rb, p, dtype=F32)
        rb = vf.load_align(r, [row * 4 + 1, basecol + 0], dist=pl.LoadDist.UNPK_B16)
        r1_lo = vf.astype(rb, p, dtype=F32)
        rb = vf.load_align(r, [row * 4 + 2, basecol + 0], dist=pl.LoadDist.UNPK_B16)
        r2_lo = vf.astype(rb, p, dtype=F32)
        rb = vf.load_align(r, [row * 4 + 3, basecol + 0], dist=pl.LoadDist.UNPK_B16)
        r3_lo = vf.astype(rb, p, dtype=F32)
        xb_hi = vf.load_align(x, [row, basecol + 64], dist=pl.LoadDist.UNPK_B16)
        xv_hi = vf.astype(xb_hi, p, dtype=F32)
        rb = vf.load_align(r, [row * 4 + 0, basecol + 64], dist=pl.LoadDist.UNPK_B16)
        r0_hi = vf.astype(rb, p, dtype=F32)
        rb = vf.load_align(r, [row * 4 + 1, basecol + 64], dist=pl.LoadDist.UNPK_B16)
        r1_hi = vf.astype(rb, p, dtype=F32)
        rb = vf.load_align(r, [row * 4 + 2, basecol + 64], dist=pl.LoadDist.UNPK_B16)
        r2_hi = vf.astype(rb, p, dtype=F32)
        rb = vf.load_align(r, [row * 4 + 3, basecol + 64], dist=pl.LoadDist.UNPK_B16)
        r3_hi = vf.astype(rb, p, dtype=F32)
        c = vf.load_align(comb, [row, 0], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul(r0_lo, c, p)
        v0_hi = vf.mul(r0_hi, c, p)
        c = vf.load_align(comb, [row, 4], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul_add_dst(r1_lo, c, p)
        v0_hi = vf.mul_add_dst(r1_hi, c, p)
        c = vf.load_align(comb, [row, 8], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul_add_dst(r2_lo, c, p)
        v0_hi = vf.mul_add_dst(r2_hi, c, p)
        c = vf.load_align(comb, [row, 12], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul_add_dst(r3_lo, c, p)
        v0_hi = vf.mul_add_dst(r3_hi, c, p)
        pv = vf.load_align(post, [row, 0], dist=pl.LoadDist.BRC_B32)
        xp = vf.mul(xv_lo, pv, p)
        v_lo = vf.add(v_lo, xp, p)
        xp = vf.mul(xv_hi, pv, p)
        v0_hi = vf.add(v0_hi, xp, p)
        b_lo = vf.astype(v_lo, ph, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        b_hi = vf.astype(v0_hi, ph, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        v_lo = vf.astype(b_lo, p, dtype=F32)
        v0_hi = vf.astype(b_hi, p, dtype=F32)
        packed = vf.bit_cast(b_lo, dtype=pl.DT_UINT32)
        vf.store_align(r, packed, pu, [row * 4 + 0, basecol + 0], dist=pl.StoreDist.PACK)
        packed = vf.bit_cast(b_hi, dtype=pl.DT_UINT32)
        vf.store_align(r, packed, pu, [row * 4 + 0, basecol + 64], dist=pl.StoreDist.PACK)
        dense, unused = vf.de_interleave(b_lo, b_hi)
        vf.store_align(
            z0,
            dense,
            ph,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=batch + 1,
            repeat_stride=1,
            post_update=True,
        )
        ss = vf.mul_add_dst(v_lo, v_lo, p)
        av = vf.load_align(prev, [row, 0], dist=pl.LoadDist.BRC_B32)
        uv_lo = vf.mul(v_lo, av, p)
        uv_hi = vf.mul(v0_hi, av, p)
        c = vf.load_align(comb, [row, 1], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul(r0_lo, c, p)
        v1_hi = vf.mul(r0_hi, c, p)
        c = vf.load_align(comb, [row, 5], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul_add_dst(r1_lo, c, p)
        v1_hi = vf.mul_add_dst(r1_hi, c, p)
        c = vf.load_align(comb, [row, 9], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul_add_dst(r2_lo, c, p)
        v1_hi = vf.mul_add_dst(r2_hi, c, p)
        c = vf.load_align(comb, [row, 13], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul_add_dst(r3_lo, c, p)
        v1_hi = vf.mul_add_dst(r3_hi, c, p)
        pv = vf.load_align(post, [row, 1], dist=pl.LoadDist.BRC_B32)
        xp = vf.mul(xv_lo, pv, p)
        v_lo = vf.add(v_lo, xp, p)
        xp = vf.mul(xv_hi, pv, p)
        v1_hi = vf.add(v1_hi, xp, p)
        b_lo = vf.astype(v_lo, ph, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        b_hi = vf.astype(v1_hi, ph, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        v_lo = vf.astype(b_lo, p, dtype=F32)
        v1_hi = vf.astype(b_hi, p, dtype=F32)
        packed = vf.bit_cast(b_lo, dtype=pl.DT_UINT32)
        vf.store_align(r, packed, pu, [row * 4 + 1, basecol + 0], dist=pl.StoreDist.PACK)
        packed = vf.bit_cast(b_hi, dtype=pl.DT_UINT32)
        vf.store_align(r, packed, pu, [row * 4 + 1, basecol + 64], dist=pl.StoreDist.PACK)
        dense, unused = vf.de_interleave(b_lo, b_hi)
        vf.store_align(
            z1,
            dense,
            ph,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=batch + 1,
            repeat_stride=1,
            post_update=True,
        )
        ss = vf.mul_add_dst(v_lo, v_lo, p)
        av = vf.load_align(prev, [row, 1], dist=pl.LoadDist.BRC_B32)
        q = vf.mul(v_lo, av, p)
        uv_lo = vf.add(uv_lo, q, p)
        q = vf.mul(v1_hi, av, p)
        uv_hi = vf.add(uv_hi, q, p)
        c = vf.load_align(comb, [row, 2], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul(r0_lo, c, p)
        v2_hi = vf.mul(r0_hi, c, p)
        c = vf.load_align(comb, [row, 6], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul_add_dst(r1_lo, c, p)
        v2_hi = vf.mul_add_dst(r1_hi, c, p)
        c = vf.load_align(comb, [row, 10], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul_add_dst(r2_lo, c, p)
        v2_hi = vf.mul_add_dst(r2_hi, c, p)
        c = vf.load_align(comb, [row, 14], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul_add_dst(r3_lo, c, p)
        v2_hi = vf.mul_add_dst(r3_hi, c, p)
        pv = vf.load_align(post, [row, 2], dist=pl.LoadDist.BRC_B32)
        xp = vf.mul(xv_lo, pv, p)
        v_lo = vf.add(v_lo, xp, p)
        xp = vf.mul(xv_hi, pv, p)
        v2_hi = vf.add(v2_hi, xp, p)
        b_lo = vf.astype(v_lo, ph, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        b_hi = vf.astype(v2_hi, ph, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        v_lo = vf.astype(b_lo, p, dtype=F32)
        v2_hi = vf.astype(b_hi, p, dtype=F32)
        packed = vf.bit_cast(b_lo, dtype=pl.DT_UINT32)
        vf.store_align(r, packed, pu, [row * 4 + 2, basecol + 0], dist=pl.StoreDist.PACK)
        packed = vf.bit_cast(b_hi, dtype=pl.DT_UINT32)
        vf.store_align(r, packed, pu, [row * 4 + 2, basecol + 64], dist=pl.StoreDist.PACK)
        dense, unused = vf.de_interleave(b_lo, b_hi)
        vf.store_align(
            z2,
            dense,
            ph,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=batch + 1,
            repeat_stride=1,
            post_update=True,
        )
        ss = vf.mul_add_dst(v_lo, v_lo, p)
        av = vf.load_align(prev, [row, 2], dist=pl.LoadDist.BRC_B32)
        q = vf.mul(v_lo, av, p)
        uv_lo = vf.add(uv_lo, q, p)
        q = vf.mul(v2_hi, av, p)
        uv_hi = vf.add(uv_hi, q, p)
        c = vf.load_align(comb, [row, 3], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul(r0_lo, c, p)
        v3_hi = vf.mul(r0_hi, c, p)
        c = vf.load_align(comb, [row, 7], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul_add_dst(r1_lo, c, p)
        v3_hi = vf.mul_add_dst(r1_hi, c, p)
        c = vf.load_align(comb, [row, 11], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul_add_dst(r2_lo, c, p)
        v3_hi = vf.mul_add_dst(r2_hi, c, p)
        c = vf.load_align(comb, [row, 15], dist=pl.LoadDist.BRC_B32)
        v_lo = vf.mul_add_dst(r3_lo, c, p)
        v3_hi = vf.mul_add_dst(r3_hi, c, p)
        pv = vf.load_align(post, [row, 3], dist=pl.LoadDist.BRC_B32)
        xp = vf.mul(xv_lo, pv, p)
        v_lo = vf.add(v_lo, xp, p)
        xp = vf.mul(xv_hi, pv, p)
        v3_hi = vf.add(v3_hi, xp, p)
        b_lo = vf.astype(v_lo, ph, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        b_hi = vf.astype(v3_hi, ph, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        v_lo = vf.astype(b_lo, p, dtype=F32)
        v3_hi = vf.astype(b_hi, p, dtype=F32)
        packed = vf.bit_cast(b_lo, dtype=pl.DT_UINT32)
        vf.store_align(r, packed, pu, [row * 4 + 3, basecol + 0], dist=pl.StoreDist.PACK)
        packed = vf.bit_cast(b_hi, dtype=pl.DT_UINT32)
        vf.store_align(r, packed, pu, [row * 4 + 3, basecol + 64], dist=pl.StoreDist.PACK)
        dense, unused = vf.de_interleave(b_lo, b_hi)
        vf.store_align(
            z3,
            dense,
            ph,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=batch + 1,
            repeat_stride=1,
            post_update=True,
        )
        ss = vf.mul_add_dst(v_lo, v_lo, p)
        av = vf.load_align(prev, [row, 3], dist=pl.LoadDist.BRC_B32)
        q = vf.mul(v_lo, av, p)
        uv_lo = vf.add(uv_lo, q, p)
        q = vf.mul(v3_hi, av, p)
        uv_hi = vf.add(uv_hi, q, p)
        ss = vf.mul_add_dst(v0_hi, v0_hi, p)
        ss = vf.mul_add_dst(v1_hi, v1_hi, p)
        ss = vf.mul_add_dst(v2_hi, v2_hi, p)
        ss = vf.mul_add_dst(v3_hi, v3_hi, p)
        ub = vf.astype(uv_lo, ph, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        ur = vf.astype(ub, p, dtype=F32)
        us = vf.mul_add_dst(ur, ur, p)
        packed = vf.bit_cast(ub, dtype=pl.DT_UINT32)
        vf.store_align(u, packed, pu, [row, 0 + (basecol + 0)], dist=pl.StoreDist.PACK)
        ub = vf.astype(uv_hi, ph, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
        ur = vf.astype(ub, p, dtype=F32)
        us = vf.mul_add_dst(ur, ur, p)
        packed = vf.bit_cast(ub, dtype=pl.DT_UINT32)
        vf.store_align(u, packed, pu, [row, 0 + (basecol + 64)], dist=pl.StoreDist.PACK)
        vf.store_align(partial, ss, p, [row * 2, 0])
        vf.store_align(partial, us, p, [row * 2 + 1, 0])
    zero = vf.full(0.0, ph, dtype=B16)
    for _ in pl.range(count, batch):
        vf.store_align(
            z0,
            zero,
            ph,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=batch + 1,
            repeat_stride=1,
            post_update=True,
        )
        vf.store_align(
            z1,
            zero,
            ph,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=batch + 1,
            repeat_stride=1,
            post_update=True,
        )
        vf.store_align(
            z2,
            zero,
            ph,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=batch + 1,
            repeat_stride=1,
            post_update=True,
        )
        vf.store_align(
            z3,
            zero,
            ph,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=batch + 1,
            repeat_stride=1,
            post_update=True,
        )


@pl.vector_function
def pack_fn_bf2_128(w, hi, lo, rows, offset):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    ph = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    for row in pl.range(rows):
        a = vf.load_align(w, [row, offset])
        b = vf.load_align(w, [row, offset + 64])
        ah = vf.astype(a, ph, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
        bh = vf.astype(b, ph, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
        av = vf.astype(ah, p, dtype=pl.DT_FP32)
        bv = vf.astype(bh, p, dtype=pl.DT_FP32)
        da = vf.sub(a, av, p)
        db = vf.sub(b, bv, p)
        al = vf.astype(da, ph, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
        bl = vf.astype(db, ph, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
        hdense, hodd = vf.de_interleave(ah, bh)
        ldense, lodd = vf.de_interleave(al, bl)
        vf.store_align(
            hi,
            hdense,
            ph,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=17,
            repeat_stride=1,
            post_update=True,
        )
        vf.store_align(
            lo,
            ldense,
            ph,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=17,
            repeat_stride=1,
            post_update=True,
        )
    zero = vf.full(0.0, ph, dtype=pl.DT_BF16)
    for _ in pl.range(rows, 16):
        vf.store_align(
            hi,
            zero,
            ph,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=17,
            repeat_stride=1,
            post_update=True,
        )
        vf.store_align(
            lo,
            zero,
            ph,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=17,
            repeat_stride=1,
            post_update=True,
        )


@pl.vector_function
def _partial_record(scores, partial, record, count):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
    pu = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    p24 = vf.update_mask(24, dtype=F32)
    p32 = vf.update_mask(32, dtype=F32)
    ids = vf.arange(0, dtype=pl.DT_UINT32)
    i24 = vf.full(24, pu, dtype=pl.DT_UINT32)
    i25 = vf.full(25, pu, dtype=pl.DT_UINT32)
    m24 = vf.eq(ids, i24, pu)
    m25 = vf.eq(ids, i25, pu)
    zero = vf.full(0.0, p, dtype=F32)
    for row in pl.range(count):
        pair = vf.load_align(scores, [row // 2 * 2, 0])
        row_mask = vf.full(31, pu, dtype=pl.DT_UINT32)
        row_index = vf.and_(ids, row_mask, pu)
        row_offset = vf.full(row % 2 * 32, pu, dtype=pl.DT_UINT32)
        row_index = vf.add(row_index, row_offset, pu)
        q = vf.gather(pair, row_index)
        ss = vf.load_align(partial, [row * 2, 0])
        su = vf.load_align(partial, [row * 2 + 1, 0])
        ss = vf.reduce_sum(ss, p)
        su = vf.reduce_sum(su, p)
        ss = vf.full(ss, p)
        su = vf.full(su, p)
        value = vf.select(q, zero, p24)
        value = vf.select(ss, value, m24)
        value = vf.select(su, value, m25)
        vf.store_align(record, value, p32, [row, 0])


@pl.pipeline.stage
def _prefetch_produce(
    job,
    shard,
    tokens,
    x,
    residual,
    prev,
    post,
    comb,
    rout,
    rg,
    xg,
    pg,
    cg,
    vg,
    ug,
    partialg,
    a0g,
    a1g,
    a2g,
    a3g,
    gz00,
    gz01,
    gz10,
    gz11,
    gz20,
    gz21,
    gz30,
    gz31,
    batch,
    u_slots,
):
    sub = pl.get_subblock_idx()
    start = job * (2 * batch) + sub * batch
    count = pl.max(0, pl.min(batch, tokens - start))
    future_start = (job + 1) * (2 * batch) + sub * batch
    future_count = pl.max(0, pl.min(batch, tokens - future_start))
    future_rb = rg[(job + 1) % 2]
    future_xb = xg[(job + 1) % 2]
    future_pp = pg[(job + 1) % 2]
    future_cc = cg[(job + 1) % 2]
    future_vv = vg[(job + 1) % 2]
    pl.set_validshape(future_rb, [future_count * 4, CHUNK])
    pl.set_validshape(future_xb, [future_count, CHUNK])
    pl.set_validshape(future_pp, [future_count, 4])
    pl.set_validshape(future_cc, [future_count, 16])
    pl.set_validshape(future_vv, [future_count, 4])
    if future_count > 0:
        pl.load(future_rb, residual, [future_start * 4, shard * CHUNK])
        pl.load(future_xb, x, [future_start, shard * CHUNK])
        pl.load(future_pp, post, [future_start, 0])
        pl.load(future_cc, comb, [future_start, 0])
        pl.load(future_vv, prev, [future_start, 0])
    rb = rg[job % 2]
    xb = xg[job % 2]
    pp = pg[job % 2]
    cc = cg[job % 2]
    vv = vg[job % 2]
    uu = ug[job % u_slots]
    partial = partialg[job % 2]
    pl.set_validshape(rb, [count * 4, CHUNK])
    pl.set_validshape(xb, [count, CHUNK])
    pl.set_validshape(pp, [count, 4])
    pl.set_validshape(cc, [count, 16])
    pl.set_validshape(vv, [count, 4])
    z00 = gz00.next()
    z01 = gz01.next()
    z10 = gz10.next()
    z11 = gz11.next()
    z20 = gz20.next()
    z21 = gz21.next()
    z30 = gz30.next()
    z31 = gz31.next()
    _post128_pack_first(rb, xb, uu, pp, cc, vv, partial, z00, z10, z20, z30, 0, count, batch)
    _post128_pack_next(rb, xb, uu, pp, cc, vv, partial, z01, z11, z21, z31, 128, count, batch)
    if count > 0:
        pl.store(rout, rb, [start * 4, shard * CHUNK])
        pl.set_validshape(uu, [count, CHUNK])
    aa0 = a0g.next()
    pl.set_validshape(z00, [batch, 128])
    pl.set_validshape(z01, [batch, 128])
    pl.insert(aa0, z00, [sub * batch, 0])
    pl.insert(aa0, z01, [sub * batch, 128])
    aa1 = a1g.next()
    pl.set_validshape(z10, [batch, 128])
    pl.set_validshape(z11, [batch, 128])
    pl.insert(aa1, z10, [sub * batch, 0])
    pl.insert(aa1, z11, [sub * batch, 128])
    aa2 = a2g.next()
    pl.set_validshape(z20, [batch, 128])
    pl.set_validshape(z21, [batch, 128])
    pl.insert(aa2, z20, [sub * batch, 0])
    pl.insert(aa2, z21, [sub * batch, 128])
    aa3 = a3g.next()
    pl.set_validshape(z30, [batch, 128])
    pl.set_validshape(z31, [batch, 128])
    pl.insert(aa3, z30, [sub * batch, 0])
    pl.insert(aa3, z31, [sub * batch, 128])


@pl.pipeline.stage
def _overlap_project(
    job, a0g, a1g, a2g, a3g, b0hg, b0lg, b1hg, b1lg, b2hg, b2lg, b3hg, b3lg, la, lb, accg, scoreg, batch
):
    aa0 = a0g.next()
    aa1 = a1g.next()
    aa2 = a2g.next()
    aa3 = a3g.next()
    acc = accg.current()
    weight0h = b0hg.current()
    pl.gemm(acc, aa0, weight0h, l0a=la, l0b=lb, trans_b=True, m=2 * batch, n=24, k=CHUNK, init=True)
    weight0l = b0lg.current()
    pl.gemm(acc, aa0, weight0l, l0a=la, l0b=lb, trans_b=True, m=2 * batch, n=24, k=CHUNK, init=False)
    weight1h = b1hg.current()
    pl.gemm(acc, aa1, weight1h, l0a=la, l0b=lb, trans_b=True, m=2 * batch, n=24, k=CHUNK, init=False)
    weight1l = b1lg.current()
    pl.gemm(acc, aa1, weight1l, l0a=la, l0b=lb, trans_b=True, m=2 * batch, n=24, k=CHUNK, init=False)
    weight2h = b2hg.current()
    pl.gemm(acc, aa2, weight2h, l0a=la, l0b=lb, trans_b=True, m=2 * batch, n=24, k=CHUNK, init=False)
    weight2l = b2lg.current()
    pl.gemm(acc, aa2, weight2l, l0a=la, l0b=lb, trans_b=True, m=2 * batch, n=24, k=CHUNK, init=False)
    weight3h = b3hg.current()
    pl.gemm(acc, aa3, weight3h, l0a=la, l0b=lb, trans_b=True, m=2 * batch, n=24, k=CHUNK, init=False)
    weight3l = b3lg.current()
    pl.gemm(acc, aa3, weight3l, l0a=la, l0b=lb, trans_b=True, m=2 * batch, n=24, k=CHUNK, init=False)
    scores = scoreg.next()
    pl.move(scores, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


@pl.vector_function
def _norm_chunk_rows(u, gamma, inverse, count):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
    ph = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=B16)
    pu = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    for token in pl.range(count):
        ur = vf.load_align(inverse, [token, 0], dist=pl.LoadDist.BRC_B32)
        for col in pl.range(0, CHUNK, 64):
            ub = vf.load_align(u, [token, col], dist=pl.LoadDist.UNPK_B16)
            gb = vf.load_align(gamma, col, dist=pl.LoadDist.UNPK_B16)
            uv = vf.astype(ub, p, dtype=F32)
            gv = vf.astype(gb, p, dtype=F32)
            y = vf.mul(uv, ur, p)
            y = vf.mul(y, gv, p)
            yb = vf.astype(y, ph, dtype=B16, round_mode=pl.VFRoundMode.CAST_RINT)
            packed = vf.bit_cast(yb, dtype=pl.DT_UINT32)
            vf.store_align(u, packed, pu, [token, col], dist=pl.StoreDist.PACK)
