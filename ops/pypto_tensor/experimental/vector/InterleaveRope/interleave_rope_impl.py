#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
interleave_rope PyPTO implementation (ASC ops-transformer 兼容计算流)

数学计算 (与 ASC ops-transformer posembedding/interleave_rope 一致, half-split cos/sin 配对):

  x 为 interleave 排布: x_even[k] = x[..., 2k], x_odd[k] = x[..., 2k+1]  (k = 0..31)
  cos/sin 为半区配对排布 (等价 cat(freqs, freqs), 与 ASC 相同, 不做奇偶假设):
      c_lo = cos[..., 0:32],  c_hi = cos[..., 32:64]
      s_lo = sin[..., 0:32],  s_hi = sin[..., 32:64]

  输出 layout (split-half, 与 ASC 相同):
      out[..., 0:32 ] = y_even = x_even · c_lo − x_odd · s_lo
      out[..., 32:64] = y_odd  = x_even · s_hi + x_odd · c_hi

计算流 (对齐 ASC kernel):
  1. 仅对 x 做奇偶抽取 (gathermask PM=1/2, 等价 ASC GatherMask stride 1/2;
     950 用 deinterleave 单指令取双半)
  2. cos/sin 零 gather —— 直接 view 切前后半区 ([..., 0:32] / [..., 32:64]),
     等价 ASC 的地址偏移配对
  3. 全 dtype 强制 FP32 计算 (等价 ASC 的 Cast fp32 → mul/sub/add → Cast back)
  4. 出口 cast: bf16 用 CAST_RINT, fp16 用 CAST_NONE (与 ASC RoundMode 一致)

Wrapper 做：输入校验 / 输出张量分配 / 按 (arch, N, dtype, S, S_cs) 派发。
Kernel 全程 4D，无 5D reshape/concat。

kernel 实例：{N=1, N=128} × {bf16, fp16} + 变体 (broadcast / short_s / unroll / 950)。
"""

from dataclasses import dataclass

import pypto
import torch

N_TILE_128 = 32
B_TILE_128_SHORT = 4
S_TILE_128 = 16
S_TILE_128_SHORT = 2
S_UNROLL_128 = [4, 2, 1]
S_TILE_1 = 64
D = 64
HALF = 32
ASCEND_950_NPUARCH = "DAV_3510"


@dataclass
class RopeTileConfig:
    def __init__(self):
        self.n_length = 32
        self.gather_tile = [1, 16, 8, 64]
        self.elem_tile = [1, 16, 8, 32]
        self.unroll_list = {64, 32, 1}


@dataclass
class RopeTileConfig950:
    def __init__(self):
        self.n_length = 32
        self.gather_tile = [1, 16, 16, 64]
        self.elem_tile = [1, 16, 16, 32]
        self.unroll_list = {64, 32, 1}


@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "stitch_function_max_num": 512,
        "device_sched_mode": 3,
    },
    pass_options={"vec_nbuffer_setting": {-2: 1, 1: 8}},
)
def interleave_rope_kernel_n128_bf16(
    x: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    out: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_BF16),
    tile_config = RopeTileConfig()
):
    pypto.experimental.set_operation_options(combine_axis=True)
    batch = x.shape[0]
    seq_len = x.shape[2]
    dim = x.shape[3]
    n_length = tile_config.n_length
    gather_tile = tile_config.gather_tile
    elem_tile = tile_config.elem_tile

    for b_idx in pypto.loop(batch, name="b_loop"):
        for n_blk in pypto.loop(128 // n_length, name="n_loop"):
            n_off = n_blk * n_length
            for s_blk, unroll_length in pypto.loop_unroll(
                0, seq_len, 1, name="s_loop", idx_name="bs_blk_offset", unroll_list=tile_config.unroll_list
            ):

                x_t = pypto.view(x, [1, n_length, unroll_length, dim], [b_idx, n_off, s_blk, 0],
                                 valid_shape=[1, n_length, unroll_length, dim])
                cos_t = pypto.view(cos, [1, 1, unroll_length, dim], [b_idx, 0, s_blk, 0],
                                   valid_shape=[1, 1, unroll_length, dim])
                sin_t = pypto.view(sin, [1, 1, unroll_length, dim], [b_idx, 0, s_blk, 0],
                                   valid_shape=[1, 1, unroll_length, dim])

                pypto.set_pass_options(sg_set_scope=1)
                pypto.set_vec_tile_shapes(gather_tile[0], gather_tile[1], gather_tile[2], gather_tile[3])

                # 计算流开头：x/cos/sin 先 cast 到 FP32（与加载融合）
                x_f = pypto.cast(x_t, pypto.DT_FP32)
                cos_f = pypto.cast(cos_t, pypto.DT_FP32)
                sin_f = pypto.cast(sin_t, pypto.DT_FP32)

                # cos/sin 半区 view（FP32 域，零 gather）
                c_lo = pypto.view(cos_f, [1, 1, unroll_length, HALF], [0, 0, 0, 0])
                c_hi = pypto.view(cos_f, [1, 1, unroll_length, HALF], [0, 0, 0, HALF])
                s_lo = pypto.view(sin_f, [1, 1, unroll_length, HALF], [0, 0, 0, 0])
                s_hi = pypto.view(sin_f, [1, 1, unroll_length, HALF], [0, 0, 0, HALF])

                # 仅 x 做奇偶抽取（FP32 域，等价 ASC GatherMask real/imag）
                x_e = pypto.gathermask(x_f, pattern_mode=1)
                x_o = pypto.gathermask(x_f, pattern_mode=2)

                pypto.set_vec_tile_shapes(elem_tile[0], elem_tile[1], elem_tile[2], elem_tile[3])
                # FP32 计算（ASC 全 dtype 强制 fp32）
                ye_f = pypto.sub(pypto.mul(x_e, c_lo), pypto.mul(x_o, s_lo))
                yo_f = pypto.add(pypto.mul(x_e, s_hi), pypto.mul(x_o, c_hi))
                # bf16 出口 CAST_RINT（与 ASC 一致）
                ye = pypto.cast(ye_f, pypto.DT_BF16, mode=pypto.CastMode.CAST_RINT)
                yo = pypto.cast(yo_f, pypto.DT_BF16, mode=pypto.CastMode.CAST_RINT)

                pypto.assemble(ye, [b_idx, n_off, s_blk, 0], out)
                pypto.assemble(yo, [b_idx, n_off, s_blk, HALF], out)

                pypto.set_pass_options(sg_set_scope=-1)


@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "stitch_function_max_num": 512,
        "device_sched_mode": 3,
    },
)
def interleave_rope_kernel_n128_bf16_unroll(
    x: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    out: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_BF16),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128, D)
    B = x.shape[0]
    S = x.shape[2]
    s_loops = (S + S_TILE_128 - 1) // S_TILE_128
    for b in pypto.loop(B, name="b_loop"):
        for n_blk in pypto.loop(128 // N_TILE_128, name="n_loop"):
            n_off = n_blk * N_TILE_128
            for s_blk in pypto.loop(s_loops, name="s_loop", unroll_list=S_UNROLL_128):
                s_off = s_blk * S_TILE_128
                valid_s = (S - s_off).min(S_TILE_128)
                vshape_x = [1, N_TILE_128, valid_s, D]
                vshape_cs = [1, 1, valid_s, HALF]
                x_t = pypto.view(x, [1, N_TILE_128, S_TILE_128, D], [b, n_off, s_off, 0],
                                 valid_shape=vshape_x)
                c_lo = pypto.view(cos, [1, 1, S_TILE_128, HALF], [b, 0, s_off, 0],
                                  valid_shape=vshape_cs)
                c_hi = pypto.view(cos, [1, 1, S_TILE_128, HALF], [b, 0, s_off, HALF],
                                  valid_shape=vshape_cs)
                s_lo = pypto.view(sin, [1, 1, S_TILE_128, HALF], [b, 0, s_off, 0],
                                  valid_shape=vshape_cs)
                s_hi = pypto.view(sin, [1, 1, S_TILE_128, HALF], [b, 0, s_off, HALF],
                                  valid_shape=vshape_cs)
                x_e = pypto.gathermask(x_t, pattern_mode=1)
                x_o = pypto.gathermask(x_t, pattern_mode=2)
                pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128, HALF)
                xe_f = pypto.cast(x_e, pypto.DT_FP32)
                xo_f = pypto.cast(x_o, pypto.DT_FP32)
                cl_f = pypto.cast(c_lo, pypto.DT_FP32)
                ch_f = pypto.cast(c_hi, pypto.DT_FP32)
                sl_f = pypto.cast(s_lo, pypto.DT_FP32)
                sh_f = pypto.cast(s_hi, pypto.DT_FP32)
                ye_f = pypto.sub(pypto.mul(xe_f, cl_f), pypto.mul(xo_f, sl_f))
                yo_f = pypto.add(pypto.mul(xe_f, sh_f), pypto.mul(xo_f, ch_f))
                ye = pypto.cast(ye_f, pypto.DT_BF16, mode=pypto.CastMode.CAST_RINT)
                yo = pypto.cast(yo_f, pypto.DT_BF16, mode=pypto.CastMode.CAST_RINT)
                pypto.assemble(ye, [b, n_off, s_off, 0], out)
                pypto.assemble(yo, [b, n_off, s_off, HALF], out)
                pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128, D)


@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "stitch_function_max_num": 512,
        "device_sched_mode": 3,
    },
    pass_options={"vec_nbuffer_setting": {-2: 1, -1: 4}},
)
def interleave_rope_kernel_n128_bf16_950(
    x:   pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    out: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_BF16),
    tile_config=RopeTileConfig950()
):
    pypto.experimental.set_operation_options(combine_axis=True)
    batch = x.shape[0]
    seq_len = x.shape[2]
    dim = x.shape[3]
    n_length = tile_config.n_length
    gather_tile = tile_config.gather_tile
    elem_tile = tile_config.elem_tile

    for b_idx in pypto.loop(batch, name="b_loop"):
        for n_blk in pypto.loop(128 // n_length, name="n_loop"):
            n_off = n_blk * n_length
            for s_blk, unroll_length in pypto.loop_unroll(
                0, seq_len, 1, name="s_loop", idx_name="bs_blk_offset", unroll_list=tile_config.unroll_list
            ):

                x_t = pypto.view(x, [1, n_length, unroll_length, dim], [b_idx, n_off, s_blk, 0],
                                 valid_shape=[1, n_length, unroll_length, dim])
                # cos/sin 半区 view：零 gather，等价 ASC 的地址偏移配对
                c_lo = pypto.view(cos, [1, 1, unroll_length, HALF], [b_idx, 0, s_blk, 0],
                                  valid_shape=[1, 1, unroll_length, HALF])
                c_hi = pypto.view(cos, [1, 1, unroll_length, HALF], [b_idx, 0, s_blk, HALF],
                                  valid_shape=[1, 1, unroll_length, HALF])
                s_lo = pypto.view(sin, [1, 1, unroll_length, HALF], [b_idx, 0, s_blk, 0],
                                  valid_shape=[1, 1, unroll_length, HALF])
                s_hi = pypto.view(sin, [1, 1, unroll_length, HALF], [b_idx, 0, s_blk, HALF],
                                  valid_shape=[1, 1, unroll_length, HALF])

                pypto.set_pass_options(sg_set_scope=1)
                pypto.set_vec_tile_shapes(gather_tile[0], gather_tile[1], gather_tile[2], gather_tile[3])

                # 计算流开头：全部输入先 cast 到 FP32，再进行后续计算流
                x_f = pypto.cast(x_t, pypto.DT_FP32)
                cl_f = pypto.cast(c_lo, pypto.DT_FP32)
                ch_f = pypto.cast(c_hi, pypto.DT_FP32)
                sl_f = pypto.cast(s_lo, pypto.DT_FP32)
                sh_f = pypto.cast(s_hi, pypto.DT_FP32)

                # 950: x 在 FP32 域用 deinterleave 单指令取奇偶双半；cos/sin 仍为零 gather 半区 view
                x_e, x_o = pypto.deinterleave(x_f)

                pypto.set_vec_tile_shapes(elem_tile[0], elem_tile[1], elem_tile[2], elem_tile[3])
                ye_f = pypto.sub(pypto.mul(x_e, cl_f), pypto.mul(x_o, sl_f))
                yo_f = pypto.add(pypto.mul(x_e, sh_f), pypto.mul(x_o, ch_f))
                ye = pypto.cast(ye_f, pypto.DT_BF16, mode=pypto.CastMode.CAST_RINT)
                yo = pypto.cast(yo_f, pypto.DT_BF16, mode=pypto.CastMode.CAST_RINT)

                pypto.assemble(ye, [b_idx, n_off, s_blk, 0], out)
                pypto.assemble(yo, [b_idx, n_off, s_blk, HALF], out)
                pypto.set_pass_options(sg_set_scope=-1)


@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "stitch_function_max_num": 512,
        "device_sched_mode": 3,
    },
)
def interleave_rope_kernel_n128_bf16_broadcast(
    x: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, 1, 1, 64], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, 1, 1, 64], pypto.DT_BF16),
    out: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_BF16),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128, D)
    B = x.shape[0]
    S = x.shape[2]
    s_loops = (S + S_TILE_128 - 1) // S_TILE_128
    for b in pypto.loop(B, name="b_loop"):
        # cos/sin 半区 view 提升到 n/s 循环之外（每 batch 取一次，跨 N/S tile 复用）
        c_lo = pypto.view(cos, [1, 1, 1, HALF], [b, 0, 0, 0])
        c_hi = pypto.view(cos, [1, 1, 1, HALF], [b, 0, 0, HALF])
        s_lo = pypto.view(sin, [1, 1, 1, HALF], [b, 0, 0, 0])
        s_hi = pypto.view(sin, [1, 1, 1, HALF], [b, 0, 0, HALF])
        for n_blk in pypto.loop(128 // N_TILE_128, name="n_loop"):
            n_off = n_blk * N_TILE_128
            for s_blk in pypto.loop(s_loops, name="s_loop"):
                s_off = s_blk * S_TILE_128
                valid_s = (S - s_off).min(S_TILE_128)
                x_t = pypto.view(
                    x, [1, N_TILE_128, S_TILE_128, D], [b, n_off, s_off, 0],
                    valid_shape=[1, N_TILE_128, valid_s, D])
                pypto.set_pass_options(sg_set_scope=1)
                x_e = pypto.gathermask(x_t, pattern_mode=1)
                x_o = pypto.gathermask(x_t, pattern_mode=2)
                pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128, HALF)
                xe_f = pypto.cast(x_e, pypto.DT_FP32)
                xo_f = pypto.cast(x_o, pypto.DT_FP32)
                cl_f = pypto.cast(c_lo, pypto.DT_FP32)
                ch_f = pypto.cast(c_hi, pypto.DT_FP32)
                sl_f = pypto.cast(s_lo, pypto.DT_FP32)
                sh_f = pypto.cast(s_hi, pypto.DT_FP32)
                ye_f = pypto.sub(pypto.mul(xe_f, cl_f), pypto.mul(xo_f, sl_f))
                yo_f = pypto.add(pypto.mul(xe_f, sh_f), pypto.mul(xo_f, ch_f))
                ye = pypto.cast(ye_f, pypto.DT_BF16, mode=pypto.CastMode.CAST_RINT)
                yo = pypto.cast(yo_f, pypto.DT_BF16, mode=pypto.CastMode.CAST_RINT)
                pypto.assemble(ye, [b, n_off, s_off, 0], out)
                pypto.assemble(yo, [b, n_off, s_off, HALF], out)
                pypto.set_pass_options(sg_set_scope=-1)
                pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128, D)


@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "stitch_function_max_num": 512,
        "device_sched_mode": 3,
    },
)
def interleave_rope_kernel_n128_bf16_short_s(
    x: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    out: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_BF16),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128_SHORT, D)
    B = x.shape[0]
    S = x.shape[2]
    s_loops = (S + S_TILE_128_SHORT - 1) // S_TILE_128_SHORT
    for b in pypto.loop(B, name="b_loop"):
        for n_blk in pypto.loop(128 // N_TILE_128, name="n_loop"):
            n_off = n_blk * N_TILE_128
            for s_blk in pypto.loop(s_loops, name="s_loop"):
                s_off = s_blk * S_TILE_128_SHORT
                valid_s = (S - s_off).min(S_TILE_128_SHORT)
                vshape_x = [1, N_TILE_128, valid_s, D]
                vshape_cs = [1, 1, valid_s, HALF]
                x_t = pypto.view(x, [1, N_TILE_128, S_TILE_128_SHORT, D], [b, n_off, s_off, 0],
                                 valid_shape=vshape_x)
                c_lo = pypto.view(cos, [1, 1, S_TILE_128_SHORT, HALF], [b, 0, s_off, 0],
                                  valid_shape=vshape_cs)
                c_hi = pypto.view(cos, [1, 1, S_TILE_128_SHORT, HALF], [b, 0, s_off, HALF],
                                  valid_shape=vshape_cs)
                s_lo = pypto.view(sin, [1, 1, S_TILE_128_SHORT, HALF], [b, 0, s_off, 0],
                                  valid_shape=vshape_cs)
                s_hi = pypto.view(sin, [1, 1, S_TILE_128_SHORT, HALF], [b, 0, s_off, HALF],
                                  valid_shape=vshape_cs)
                x_e = pypto.gathermask(x_t, pattern_mode=1)
                x_o = pypto.gathermask(x_t, pattern_mode=2)
                pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128_SHORT, HALF)
                xe_f = pypto.cast(x_e, pypto.DT_FP32)
                xo_f = pypto.cast(x_o, pypto.DT_FP32)
                cl_f = pypto.cast(c_lo, pypto.DT_FP32)
                ch_f = pypto.cast(c_hi, pypto.DT_FP32)
                sl_f = pypto.cast(s_lo, pypto.DT_FP32)
                sh_f = pypto.cast(s_hi, pypto.DT_FP32)
                ye_f = pypto.sub(pypto.mul(xe_f, cl_f), pypto.mul(xo_f, sl_f))
                yo_f = pypto.add(pypto.mul(xe_f, sh_f), pypto.mul(xo_f, ch_f))
                ye = pypto.cast(ye_f, pypto.DT_BF16, mode=pypto.CastMode.CAST_RINT)
                yo = pypto.cast(yo_f, pypto.DT_BF16, mode=pypto.CastMode.CAST_RINT)
                pypto.assemble(ye, [b, n_off, s_off, 0], out)
                pypto.assemble(yo, [b, n_off, s_off, HALF], out)
                pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128_SHORT, D)


@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "stitch_function_max_num": 512,
        "device_sched_mode": 3,
    },
)
def interleave_rope_kernel_n128_bf16_short_s_btile(
    x: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    out: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_BF16),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(B_TILE_128_SHORT, N_TILE_128, S_TILE_128_SHORT, D)
    B = x.shape[0]
    S = x.shape[2]
    b_loops = (B + B_TILE_128_SHORT - 1) // B_TILE_128_SHORT
    s_loops = (S + S_TILE_128_SHORT - 1) // S_TILE_128_SHORT
    for b_blk in pypto.loop(b_loops, name="b_loop"):
        b_off = b_blk * B_TILE_128_SHORT
        valid_b = (B - b_off).min(B_TILE_128_SHORT)
        for n_blk in pypto.loop(128 // N_TILE_128, name="n_loop"):
            n_off = n_blk * N_TILE_128
            for s_blk in pypto.loop(s_loops, name="s_loop"):
                s_off = s_blk * S_TILE_128_SHORT
                valid_s = (S - s_off).min(S_TILE_128_SHORT)
                vshape_x = [valid_b, N_TILE_128, valid_s, D]
                vshape_cs = [valid_b, 1, valid_s, HALF]
                x_t = pypto.view(x, [B_TILE_128_SHORT, N_TILE_128, S_TILE_128_SHORT, D],
                                 [b_off, n_off, s_off, 0],
                                 valid_shape=vshape_x)
                c_lo = pypto.view(cos, [B_TILE_128_SHORT, 1, S_TILE_128_SHORT, HALF],
                                  [b_off, 0, s_off, 0],
                                  valid_shape=vshape_cs)
                c_hi = pypto.view(cos, [B_TILE_128_SHORT, 1, S_TILE_128_SHORT, HALF],
                                  [b_off, 0, s_off, HALF],
                                  valid_shape=vshape_cs)
                s_lo = pypto.view(sin, [B_TILE_128_SHORT, 1, S_TILE_128_SHORT, HALF],
                                  [b_off, 0, s_off, 0],
                                  valid_shape=vshape_cs)
                s_hi = pypto.view(sin, [B_TILE_128_SHORT, 1, S_TILE_128_SHORT, HALF],
                                  [b_off, 0, s_off, HALF],
                                  valid_shape=vshape_cs)
                x_e = pypto.gathermask(x_t, pattern_mode=1)
                x_o = pypto.gathermask(x_t, pattern_mode=2)
                pypto.set_vec_tile_shapes(B_TILE_128_SHORT, N_TILE_128, S_TILE_128_SHORT, HALF)
                xe_f = pypto.cast(x_e, pypto.DT_FP32)
                xo_f = pypto.cast(x_o, pypto.DT_FP32)
                cl_f = pypto.cast(c_lo, pypto.DT_FP32)
                ch_f = pypto.cast(c_hi, pypto.DT_FP32)
                sl_f = pypto.cast(s_lo, pypto.DT_FP32)
                sh_f = pypto.cast(s_hi, pypto.DT_FP32)
                ye_f = pypto.sub(pypto.mul(xe_f, cl_f), pypto.mul(xo_f, sl_f))
                yo_f = pypto.add(pypto.mul(xe_f, sh_f), pypto.mul(xo_f, ch_f))
                ye = pypto.cast(ye_f, pypto.DT_BF16, mode=pypto.CastMode.CAST_RINT)
                yo = pypto.cast(yo_f, pypto.DT_BF16, mode=pypto.CastMode.CAST_RINT)
                pypto.assemble(ye, [b_off, n_off, s_off, 0], out)
                pypto.assemble(yo, [b_off, n_off, s_off, HALF], out)
                pypto.set_vec_tile_shapes(B_TILE_128_SHORT, N_TILE_128, S_TILE_128_SHORT, D)


@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
)
def interleave_rope_kernel_n128_fp16(
    x: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_FP16),
    cos: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_FP16),
    sin: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_FP16),
    out: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_FP16),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128, D)
    B = x.shape[0]
    S = x.shape[2]
    s_loops = (S + S_TILE_128 - 1) // S_TILE_128
    for b in pypto.loop(B, name="b_loop"):
        for n_blk in pypto.loop(128 // N_TILE_128, name="n_loop"):
            n_off = n_blk * N_TILE_128
            for s_blk in pypto.loop(s_loops, name="s_loop"):
                s_off = s_blk * S_TILE_128
                valid_s = (S - s_off).min(S_TILE_128)
                vshape_x = [1, N_TILE_128, valid_s, D]
                vshape_cs = [1, 1, valid_s, HALF]
                x_t = pypto.view(x, [1, N_TILE_128, S_TILE_128, D], [b, n_off, s_off, 0],
                                 valid_shape=vshape_x)
                c_lo = pypto.view(cos, [1, 1, S_TILE_128, HALF], [b, 0, s_off, 0],
                                  valid_shape=vshape_cs)
                c_hi = pypto.view(cos, [1, 1, S_TILE_128, HALF], [b, 0, s_off, HALF],
                                  valid_shape=vshape_cs)
                s_lo = pypto.view(sin, [1, 1, S_TILE_128, HALF], [b, 0, s_off, 0],
                                  valid_shape=vshape_cs)
                s_hi = pypto.view(sin, [1, 1, S_TILE_128, HALF], [b, 0, s_off, HALF],
                                  valid_shape=vshape_cs)
                x_e = pypto.gathermask(x_t, pattern_mode=1)
                x_o = pypto.gathermask(x_t, pattern_mode=2)
                pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128, HALF)
                xe_f = pypto.cast(x_e, pypto.DT_FP32)
                xo_f = pypto.cast(x_o, pypto.DT_FP32)
                cl_f = pypto.cast(c_lo, pypto.DT_FP32)
                ch_f = pypto.cast(c_hi, pypto.DT_FP32)
                sl_f = pypto.cast(s_lo, pypto.DT_FP32)
                sh_f = pypto.cast(s_hi, pypto.DT_FP32)
                ye_f = pypto.sub(pypto.mul(xe_f, cl_f), pypto.mul(xo_f, sl_f))
                yo_f = pypto.add(pypto.mul(xe_f, sh_f), pypto.mul(xo_f, ch_f))
                # fp16 出口 CAST_NONE（与 ASC 一致）
                ye = pypto.cast(ye_f, pypto.DT_FP16, mode=pypto.CastMode.CAST_NONE)
                yo = pypto.cast(yo_f, pypto.DT_FP16, mode=pypto.CastMode.CAST_NONE)
                pypto.assemble(ye, [b, n_off, s_off, 0], out)
                pypto.assemble(yo, [b, n_off, s_off, HALF], out)
                pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128, D)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
)
def interleave_rope_kernel_n1_bf16(
    x: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    out: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(1, 1, S_TILE_1, D)
    B = x.shape[0]
    S = x.shape[2]
    s_loops = (S + S_TILE_1 - 1) // S_TILE_1
    for b in pypto.loop(B, name="b_loop"):
        for s_blk in pypto.loop(s_loops, name="s_loop"):
            s_off = s_blk * S_TILE_1
            valid_s = (S - s_off).min(S_TILE_1)
            vshape_x = [1, 1, valid_s, D]
            vshape_cs = [1, 1, valid_s, HALF]
            x_t = pypto.view(x, [1, 1, S_TILE_1, D], [b, 0, s_off, 0], valid_shape=vshape_x)
            c_lo = pypto.view(cos, [1, 1, S_TILE_1, HALF], [b, 0, s_off, 0], valid_shape=vshape_cs)
            c_hi = pypto.view(cos, [1, 1, S_TILE_1, HALF], [b, 0, s_off, HALF], valid_shape=vshape_cs)
            s_lo = pypto.view(sin, [1, 1, S_TILE_1, HALF], [b, 0, s_off, 0], valid_shape=vshape_cs)
            s_hi = pypto.view(sin, [1, 1, S_TILE_1, HALF], [b, 0, s_off, HALF], valid_shape=vshape_cs)
            x_e = pypto.gathermask(x_t, pattern_mode=1)
            x_o = pypto.gathermask(x_t, pattern_mode=2)
            pypto.set_vec_tile_shapes(1, 1, S_TILE_1, HALF)
            xe_f = pypto.cast(x_e, pypto.DT_FP32)
            xo_f = pypto.cast(x_o, pypto.DT_FP32)
            cl_f = pypto.cast(c_lo, pypto.DT_FP32)
            ch_f = pypto.cast(c_hi, pypto.DT_FP32)
            sl_f = pypto.cast(s_lo, pypto.DT_FP32)
            sh_f = pypto.cast(s_hi, pypto.DT_FP32)
            ye_f = pypto.sub(pypto.mul(xe_f, cl_f), pypto.mul(xo_f, sl_f))
            yo_f = pypto.add(pypto.mul(xe_f, sh_f), pypto.mul(xo_f, ch_f))
            ye = pypto.cast(ye_f, pypto.DT_BF16, mode=pypto.CastMode.CAST_RINT)
            yo = pypto.cast(yo_f, pypto.DT_BF16, mode=pypto.CastMode.CAST_RINT)
            pypto.assemble(ye, [b, 0, s_off, 0], out)
            pypto.assemble(yo, [b, 0, s_off, HALF], out)
            pypto.set_vec_tile_shapes(1, 1, S_TILE_1, D)


@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
)
def interleave_rope_kernel_n1_bf16_broadcast(
    x: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, 1, 1, 64], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, 1, 1, 64], pypto.DT_BF16),
    out: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(1, 1, S_TILE_1, D)
    B = x.shape[0]
    S = x.shape[2]
    s_loops = (S + S_TILE_1 - 1) // S_TILE_1
    for b in pypto.loop(B, name="b_loop"):
        c_lo = pypto.view(cos, [1, 1, 1, HALF], [b, 0, 0, 0])
        c_hi = pypto.view(cos, [1, 1, 1, HALF], [b, 0, 0, HALF])
        s_lo = pypto.view(sin, [1, 1, 1, HALF], [b, 0, 0, 0])
        s_hi = pypto.view(sin, [1, 1, 1, HALF], [b, 0, 0, HALF])
        for s_blk in pypto.loop(s_loops, name="s_loop"):
            s_off = s_blk * S_TILE_1
            valid_s = (S - s_off).min(S_TILE_1)
            x_t = pypto.view(x, [1, 1, S_TILE_1, D], [b, 0, s_off, 0],
                             valid_shape=[1, 1, valid_s, D])
            x_e = pypto.gathermask(x_t, pattern_mode=1)
            x_o = pypto.gathermask(x_t, pattern_mode=2)
            pypto.set_vec_tile_shapes(1, 1, S_TILE_1, HALF)
            xe_f = pypto.cast(x_e, pypto.DT_FP32)
            xo_f = pypto.cast(x_o, pypto.DT_FP32)
            cl_f = pypto.cast(c_lo, pypto.DT_FP32)
            ch_f = pypto.cast(c_hi, pypto.DT_FP32)
            sl_f = pypto.cast(s_lo, pypto.DT_FP32)
            sh_f = pypto.cast(s_hi, pypto.DT_FP32)
            ye_f = pypto.sub(pypto.mul(xe_f, cl_f), pypto.mul(xo_f, sl_f))
            yo_f = pypto.add(pypto.mul(xe_f, sh_f), pypto.mul(xo_f, ch_f))
            ye = pypto.cast(ye_f, pypto.DT_BF16, mode=pypto.CastMode.CAST_RINT)
            yo = pypto.cast(yo_f, pypto.DT_BF16, mode=pypto.CastMode.CAST_RINT)
            pypto.assemble(ye, [b, 0, s_off, 0], out)
            pypto.assemble(yo, [b, 0, s_off, HALF], out)
            pypto.set_vec_tile_shapes(1, 1, S_TILE_1, D)


@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
)
def interleave_rope_kernel_n1_fp16(
    x: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_FP16),
    cos: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_FP16),
    sin: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_FP16),
    out: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_FP16),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(1, 1, S_TILE_1, D)
    B = x.shape[0]
    S = x.shape[2]
    s_loops = (S + S_TILE_1 - 1) // S_TILE_1
    for b in pypto.loop(B, name="b_loop"):
        for s_blk in pypto.loop(s_loops, name="s_loop"):
            s_off = s_blk * S_TILE_1
            valid_s = (S - s_off).min(S_TILE_1)
            vshape_x = [1, 1, valid_s, D]
            vshape_cs = [1, 1, valid_s, HALF]
            x_t = pypto.view(x, [1, 1, S_TILE_1, D], [b, 0, s_off, 0], valid_shape=vshape_x)
            c_lo = pypto.view(cos, [1, 1, S_TILE_1, HALF], [b, 0, s_off, 0], valid_shape=vshape_cs)
            c_hi = pypto.view(cos, [1, 1, S_TILE_1, HALF], [b, 0, s_off, HALF], valid_shape=vshape_cs)
            s_lo = pypto.view(sin, [1, 1, S_TILE_1, HALF], [b, 0, s_off, 0], valid_shape=vshape_cs)
            s_hi = pypto.view(sin, [1, 1, S_TILE_1, HALF], [b, 0, s_off, HALF], valid_shape=vshape_cs)
            x_e = pypto.gathermask(x_t, pattern_mode=1)
            x_o = pypto.gathermask(x_t, pattern_mode=2)
            pypto.set_vec_tile_shapes(1, 1, S_TILE_1, HALF)
            xe_f = pypto.cast(x_e, pypto.DT_FP32)
            xo_f = pypto.cast(x_o, pypto.DT_FP32)
            cl_f = pypto.cast(c_lo, pypto.DT_FP32)
            ch_f = pypto.cast(c_hi, pypto.DT_FP32)
            sl_f = pypto.cast(s_lo, pypto.DT_FP32)
            sh_f = pypto.cast(s_hi, pypto.DT_FP32)
            ye_f = pypto.sub(pypto.mul(xe_f, cl_f), pypto.mul(xo_f, sl_f))
            yo_f = pypto.add(pypto.mul(xe_f, sh_f), pypto.mul(xo_f, ch_f))
            ye = pypto.cast(ye_f, pypto.DT_FP16, mode=pypto.CastMode.CAST_NONE)
            yo = pypto.cast(yo_f, pypto.DT_FP16, mode=pypto.CastMode.CAST_NONE)
            pypto.assemble(ye, [b, 0, s_off, 0], out)
            pypto.assemble(yo, [b, 0, s_off, HALF], out)
            pypto.set_vec_tile_shapes(1, 1, S_TILE_1, D)


_KERNELS = {
    (128, torch.bfloat16): interleave_rope_kernel_n128_bf16,
    (128, torch.float16): interleave_rope_kernel_n128_fp16,
    (1, torch.bfloat16): interleave_rope_kernel_n1_bf16,
    (1, torch.float16): interleave_rope_kernel_n1_fp16,
}


def interleave_rope_wrapper(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """ASC 兼容 interleave_rope：cos/sin 半区配对，FP32 内部计算，split-half 输出。"""

    assert x.is_contiguous(), "x must be contiguous"
    assert cos.is_contiguous() and sin.is_contiguous(), "cos/sin must be contiguous"
    assert x.dim() == 4 and cos.dim() == 4 and sin.dim() == 4
    B, N, S, Dd = x.shape
    assert Dd == 64, f"D must be 64, got {Dd}"
    assert N in (1, 128), f"N must be 1 or 128, got {N}"
    assert cos.shape[0] == B and cos.shape[1] == 1 and cos.shape[3] == 64
    assert sin.shape[0] == B and sin.shape[1] == 1 and sin.shape[3] == 64
    S_cs = cos.shape[2]
    assert S_cs in (1, S), f"S_cs must be 1 or {S}, got {S_cs}"
    assert cos.dtype == x.dtype and sin.dtype == x.dtype, \
        f"cos/sin dtype must match x; got x={x.dtype} cos={cos.dtype} sin={sin.dtype}"
    assert x.dtype in (torch.bfloat16, torch.float16), \
        f"unsupported dtype {x.dtype}"

    out = torch.empty_like(x)
    if (pypto.platform.npuarch == ASCEND_950_NPUARCH and N == 128 and x.dtype == torch.bfloat16
            and S == 1024 and S_cs == S and B in (2, 8)):
        kernel = interleave_rope_kernel_n128_bf16_950
    elif N == 128 and x.dtype == torch.bfloat16 and S == 1:
        kernel = interleave_rope_kernel_n128_bf16_short_s
    elif N == 128 and x.dtype == torch.bfloat16 and S == 2:
        kernel = interleave_rope_kernel_n128_bf16_short_s_btile
    elif S_cs == 1 and x.dtype == torch.bfloat16 and N == 128:
        kernel = interleave_rope_kernel_n128_bf16_broadcast
    elif S_cs == 1 and x.dtype == torch.bfloat16 and N == 1:
        kernel = interleave_rope_kernel_n1_bf16_broadcast
    elif S_cs == 1:
        raise AssertionError("S_cs=1 broadcast is only supported in BF16 kernels")
    elif N == 128 and x.dtype == torch.bfloat16 and S > 2048:
        kernel = interleave_rope_kernel_n128_bf16_unroll
    else:
        kernel = _KERNELS[(N, x.dtype)]
    kernel(x, cos, sin, out)
    return out
