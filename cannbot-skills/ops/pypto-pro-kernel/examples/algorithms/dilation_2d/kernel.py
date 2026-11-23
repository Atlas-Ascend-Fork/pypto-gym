# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。NHWC 灰度形态学膨胀，3×3 逐通道滤波器。

有限连续 FP16 x[N,H,W,C] 和 filter[3,3,C]，1≤W≤128、1≤C≤32。
stride、rate 各取 1 或 2，两个空间方向使用同一值。SAME 多出的填充放在右/下。
UB 中每像素占 32 个 FP16，stride=1 时一次算四个像素；stride=2 时只取一个。
band_rows=1/2 比较逐行与双行共享输入。stride=rate=1 时两个输出行读 4 个输入行，
逐行对照读 6 个。每个原始行槽与其 DMA 视图共用 mutex，左右保护区保持负无穷。
参见[窗口读写坐标](../../../references/patterns/vf-neighborhood.md#state-the-three-coordinate-systems)。

example-query: {\
  "aliases":["灰度膨胀","grayscale dilation"],\
  "term_groups":[\
    ["NHWC","逐通道","灰度"],\
    ["max-plus","morphological dilation","灰度膨胀","形态学膨胀"]],\
  "exclude":[{"term_groups":[\
    ["5x5 filter","5×5 filter","5x5 footprint","5×5滤波器","5×5结构元素"]],\
    "reason":"The row window and filter taps implement a 3x3 footprint."}]}
"""

from functools import lru_cache

import torch
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf


FP16 = pl.DT_FP16
CHANNELS = 32
PIXELS = 4


@pl.vector_function
def _clear(row, registers):
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP16)
    neg = vf.full(float("-inf"), all_lanes, dtype=FP16)
    for i in pl.range(registers):
        vf.store_align(row, neg, all_lanes, [i * PIXELS, 0])


@pl.vector_function
def _repeat_channels(weights, channels):
    live = vf.update_mask(channels, dtype=FP16)
    for tap in pl.range(9):
        data = vf.load_align(weights, [tap, 0])
        for p in pl.range(1, PIXELS):
            vf.store_align(weights, data, live, [tap, p * CHANNELS])


@pl.vector_function
def _max_row(out, top, mid, bottom, weights, groups, offset, stride, rate, lanes):
    pixels = PIXELS if stride == 1 else 1
    mask = vf.update_mask(lanes, dtype=FP16)
    w0 = vf.load_align(weights, [0, 0])
    w1 = vf.load_align(weights, [1, 0])
    w2 = vf.load_align(weights, [2, 0])
    w3 = vf.load_align(weights, [3, 0])
    w4 = vf.load_align(weights, [4, 0])
    w5 = vf.load_align(weights, [5, 0])
    w6 = vf.load_align(weights, [6, 0])
    w7 = vf.load_align(weights, [7, 0])
    w8 = vf.load_align(weights, [8, 0])
    for group in pl.range(groups):
        x = offset + group * (pixels * stride)
        value = vf.load_align(top, [x, 0])
        best = vf.add(value, w0, mask)
        value = vf.load_align(top, [x + rate, 0])
        candidate = vf.add(value, w1, mask)
        best = vf.max(best, candidate, mask)
        value = vf.load_align(top, [x + 2 * rate, 0])
        candidate = vf.add(value, w2, mask)
        best = vf.max(best, candidate, mask)
        value = vf.load_align(mid, [x, 0])
        candidate = vf.add(value, w3, mask)
        best = vf.max(best, candidate, mask)
        value = vf.load_align(mid, [x + rate, 0])
        candidate = vf.add(value, w4, mask)
        best = vf.max(best, candidate, mask)
        value = vf.load_align(mid, [x + 2 * rate, 0])
        candidate = vf.add(value, w5, mask)
        best = vf.max(best, candidate, mask)
        value = vf.load_align(bottom, [x, 0])
        candidate = vf.add(value, w6, mask)
        best = vf.max(best, candidate, mask)
        value = vf.load_align(bottom, [x + rate, 0])
        candidate = vf.add(value, w7, mask)
        best = vf.max(best, candidate, mask)
        value = vf.load_align(bottom, [x + 2 * rate, 0])
        candidate = vf.add(value, w8, mask)
        best = vf.max(best, candidate, mask)
        vf.store_align(out, best, mask, [group * pixels, 0])


@lru_cache(maxsize=32)
def _make_kernel(width, stride, rate, band_rows):
    span = (band_rows - 1) * stride + 2 * rate + 1
    # 左侧四像素保护区，两端均可读取一个完整寄存器；整槽按寄存器对齐。
    physical = ((width + 2 * PIXELS + 2 * rate + PIXELS - 1) // PIXELS) * PIXELS
    slot_bytes = physical * CHANNELS * 2
    offsets = [i * slot_bytes for i in range(span)]
    interior_offsets = [start + PIXELS * CHANNELS * 2 for start in offsets]
    ids = list(range(span))
    pixels = PIXELS if stride == 1 else 1

    @pl.jit(auto_mutex=True)
    def kernel(x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], FP16],
               filt: pl.Tensor[[9, pl.DYNAMIC], FP16],
               y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], FP16],
               pad_y: pl.DT_INT64, pad_x: pl.DT_INT64):
        with pl.section_vector():
            row_type = pl.TileType(shape=[physical, CHANNELS], dtype=FP16,
                                   target_memory=pl.MemorySpace.Vec)
            raw = pl.make_tile_group(type=row_type, addrs=offsets, mutex_ids=ids)
            interior = pl.make_tile_group(
                type=pl.TileType(shape=[width, CHANNELS], dtype=FP16,
                                 target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
                addrs=interior_offsets, mutex_ids=ids)
            weights = pl.make_tile_group(
                type=pl.TileType(shape=[9, 128], dtype=FP16, target_memory=pl.MemorySpace.Vec,
                                 valid_shape=[-1, -1]),
                addrs=[span * slot_bytes], mutex_ids=[span]).current()
            out = pl.make_tile_group(
                type=row_type, addrs=[span * slot_bytes + 9 * 256], mutex_ids=[span + 1]).current()
            n, height, channels = x.shape[0], x.shape[1], x.shape[3]
            oh, ow = y.shape[1], y.shape[2]
            bands = (oh + band_rows - 1) // band_rows
            pl.set_validshape(weights, [9, channels])
            pl.load(weights, filt, [0, 0])
            pl.set_validshape(weights, [9, 128])
            _repeat_channels(weights, channels)
            lanes = channels if pixels == 1 else 128
            for task in pl.range(pl.get_block_idx(), n * bands, pl.get_block_num()):
                b = task // bands
                oy = (task % bands) * band_rows
                first_row = oy * stride - pad_y
                for k in pl.range(span):
                    _clear(raw[k], physical // PIXELS)
                for k in pl.range(pl.max(0, -first_row), pl.min(span, height - first_row)):
                    pl.set_validshape(interior[k], [width, channels])
                    pl.load(interior[k], x, [b, first_row + k, 0, 0])
                for q in pl.range(pl.min(band_rows, oh - oy)):
                    s = q * stride
                    _max_row(out, raw[s], raw[s + rate], raw[s + 2 * rate], weights,
                             (ow + pixels - 1) // pixels, PIXELS - pad_x, stride, rate, lanes)
                    pl.set_validshape(out, [ow, channels])
                    pl.store(y, out, [b, oy + q, 0, 0])

    return kernel


def run(x, filt, stride=1, rate=1, padding="SAME", *, band_rows=2, workers=8):
    """逐通道计算 max(x + filt)，输出连续 FP16 [N,OH,OW,C]。"""
    if x.ndim != 4 or tuple(filt.shape) != (3, 3, x.shape[-1]):
        raise ValueError("expected NHWC input and [3,3,C] filter")
    if x.device.type != "npu" or filt.device != x.device or any(
        t.dtype != torch.float16 or not t.is_contiguous() for t in (x, filt)
    ):
        raise ValueError("expected contiguous FP16 inputs on one NPU")
    n, h, w, c = x.shape
    if h < 1 or not 1 <= w <= 128 or not 1 <= c <= CHANNELS:
        raise ValueError("require H >= 1, W in [1,128] and C in [1,32]")
    if stride not in (1, 2) or rate not in (1, 2) or band_rows not in (1, 2):
        raise ValueError("stride, rate and band_rows must be 1 or 2")
    if padding not in ("VALID", "SAME") or not 1 <= workers <= 32:
        raise ValueError("expected VALID or SAME and workers in [1,32]")
    extent = 2 * rate + 1
    if padding == "SAME":
        oh, ow = (h + stride - 1) // stride, (w + stride - 1) // stride
        py = max(0, (oh - 1) * stride + extent - h) // 2
        px = max(0, (ow - 1) * stride + extent - w) // 2
    else:
        oh, ow = max(0, (h - extent) // stride + 1), max(0, (w - extent) // stride + 1)
        py = px = 0
    y = x.new_empty((n, oh, ow, c))
    if y.numel():
        tasks = n * ((oh + band_rows - 1) // band_rows)
        kernel = _make_kernel(w, stride, rate, band_rows)
        kernel[None, min(workers, tasks)](x, filt.view(9, c), y, py, px)
    return y
