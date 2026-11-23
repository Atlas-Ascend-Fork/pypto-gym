# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：混合。用受限工作区计算 BF16 线性投影的交叉熵。

连续 BF16 hidden[T,H] 和 weight[V,H]，1≤T≤64，H 为 64 或 128；同设备 INT32
labels[T] 的值由调用者保证为有效词号或 ignore_index。输入须产生有限 FP32 logits。
输出 FP32 [1]，sum 或有效 token 的 mean；全部忽略时 sum=0，mean=NaN。

Cube 直接算 W_s @ hidden.T，使一个词的 token 分数在 GM 连续；VF 的一条 lane 保持
一个 token 的最大值、指数和及其舍入补偿、目标分数。词表每 64 行一块。streamed=True 复用
一个 64×64 FP32 分数块，对照先保存所有块。两路调用相同的 Cube/VF，单 stream 的
启动顺序保证写后读和重用，核内使用 auto_mutex。
参见[在线归一化状态](../../../references/patterns/online-softmax.md#the-recurrence-and-where-state-lives)。

example-query: {\
  "term_groups":[\
    ["vocabulary","词表"],\
    ["projection","投影"],\
    ["cross entropy","cross-entropy","交叉熵"]],\
  "exclude":[{"term_groups":[\
    ["with label smoothing","smoothed target distribution","带标签平滑","启用标签平滑"]],\
    "reason":"The loss uses one unsmoothed target score per valid token."}]}
"""

from functools import lru_cache

import torch
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf


BF16, FP32, INT32 = pl.DT_BF16, pl.DT_FP32, pl.DT_INT32
BLOCK = 64
# 每个有效词的 max-target 至多是两个有限 FP32 数的跨度，最多归约 BLOCK 个 token。
LOSS_SCALE = 2.0 * BLOCK
INV_LOSS_SCALE = 1.0 / LOSS_SCALE


@lru_cache(maxsize=2)
def _make_projection(hidden_dim):
    @pl.jit(auto_mutex=True)
    def project(hidden: pl.Tensor[[pl.DYNAMIC, hidden_dim], BF16],
                weight: pl.Tensor[[pl.DYNAMIC, hidden_dim], BF16],
                scores: pl.Tensor[[BLOCK, BLOCK], FP32],
                start: pl.DT_INT64, live: pl.DT_INT64):
        w_type = pl.TileType(shape=[BLOCK, hidden_dim], dtype=BF16, target_memory=pl.MemorySpace.Mat)
        w = pl.make_tile_group(type=w_type, addrs=[0], mutex_ids=[0]).current()
        x = pl.make_tile_group(type=w_type, addrs=[2 * BLOCK * hidden_dim], mutex_ids=[1]).current()
        left = pl.make_tile_group(
            type=pl.TileType(shape=[BLOCK, hidden_dim], dtype=BF16,
                             target_memory=pl.MemorySpace.Left, compact=1),
            addrs=[0], mutex_ids=[2])
        right = pl.make_tile_group(
            type=pl.TileType(shape=[hidden_dim, BLOCK], dtype=BF16, target_memory=pl.MemorySpace.Right),
            addrs=[0], mutex_ids=[3])
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[BLOCK, BLOCK], dtype=FP32,
                             target_memory=pl.MemorySpace.Acc, compact=1),
            addrs=[0], mutex_ids=[4]).current()
        with pl.section_cube():
            tokens = hidden.shape[0]
            pl.set_validshape(w, [live, hidden_dim])
            pl.load(w, weight, [start, 0])
            pl.set_validshape(x, [tokens, hidden_dim])
            pl.load(x, hidden, [0, 0])
            # 满物理块做乘法。填充行只产生无效输出行/列，VF 只读取有效区域。
            pl.set_validshape(w, [BLOCK, hidden_dim])
            pl.set_validshape(x, [BLOCK, hidden_dim])
            pl.gemm(acc, w, x, l0a=left, l0b=right, trans_b=True)
            pl.store(scores, acc, [0, 0])

    return project


@pl.vector_function
def _merge(tile, labels, state, start, live, tokens):
    fp = vf.update_mask(tokens, dtype=FP32)
    ip = vf.update_mask(tokens, dtype=INT32)
    label = vf.load_align(labels, [0, 0])
    old_max = vf.full(float("-inf"), fp, dtype=FP32)
    old_sum = vf.full(0.0, fp, dtype=FP32)
    correction = vf.full(0.0, fp, dtype=FP32)
    target = vf.full(0.0, fp, dtype=FP32)
    if start != 0:
        old_max = vf.load_align(state, [0, 0])
        old_sum = vf.load_align(state, [1, 0])
        target = vf.load_align(state, [2, 0])
        correction = vf.load_align(state, [3, 0])

    new_max = vf.move(old_max, fp)
    for v in pl.range(live):
        value = vf.load_align(tile, [v, 0])
        new_max = vf.max(new_max, value, fp)
        word = vf.full(start + v, ip, dtype=INT32)
        match = vf.eq(label, word, ip)
        target = vf.select(value, target, match)

    scale = vf.exp_sub(old_max, new_max, fp)
    total = vf.mul(old_sum, scale, fp)
    correction = vf.mul(correction, scale, fp)
    for v in pl.range(live):
        value = vf.load_align(tile, [v, 0])
        exp_value = vf.exp_sub(value, new_max, fp)
        # Kahan 补偿求和。小于 total 半个 ULP 的项仍可累积，跨词表块沿用同一补偿。
        adjusted = vf.sub(exp_value, correction, fp)
        next_total = vf.add(total, adjusted, fp)
        added = vf.sub(next_total, total, fp)
        correction = vf.sub(added, adjusted, fp)
        total = vf.move(next_total, fp)
    vf.store_align(state, new_max, fp, [0, 0])
    vf.store_align(state, total, fp, [1, 0])
    vf.store_align(state, target, fp, [2, 0])
    vf.store_align(state, correction, fp, [3, 0])


@pl.vector_function
def _finish(state, labels, result, tokens, ignore_index, mean):
    fp = vf.update_mask(tokens, dtype=FP32)
    ip = vf.update_mask(tokens, dtype=INT32)
    scalar = vf.update_mask(1, dtype=FP32)
    label = vf.load_align(labels, [0, 0])
    ignored = vf.full(ignore_index, ip, dtype=INT32)
    keep = vf.ne(label, ignored, ip)
    maximum = vf.load_align(state, [0, 0])
    total = vf.load_align(state, [1, 0])
    target = vf.load_align(state, [2, 0])
    # 在减法之前按 2 的幂缩小量级，给单个差值和跨 token 的和同时留出范围。
    # mean 先除以有效 token 数再恢复尺度，避免可表示的均值因中间溢出变为 inf。
    scaled_max = vf.muls(maximum, INV_LOSS_SCALE, fp)
    scaled_target = vf.muls(target, INV_LOSS_SCALE, fp)
    delta = vf.sub(scaled_max, scaled_target, fp)
    log_sum = vf.log(total, fp)
    scaled_log_sum = vf.muls(log_sum, INV_LOSS_SCALE, fp)
    loss = vf.add(delta, scaled_log_sum, fp)
    zero = vf.full(0.0, fp, dtype=FP32)
    masked = vf.select(loss, zero, keep)
    numerator = vf.reduce_sum(masked, fp)
    if mean != 0:
        one = vf.full(1.0, fp, dtype=FP32)
        counts = vf.select(one, zero, keep)
        count = vf.reduce_sum(counts, fp)
        numerator = vf.div(numerator, count, scalar)
    reduced = vf.muls(numerator, LOSS_SCALE, scalar)
    vf.store_align(result, reduced, scalar, [0, 0])


@pl.jit(auto_mutex=True)
def _consume(scores: pl.Tensor[[BLOCK, BLOCK], FP32], labels: pl.Tensor[[pl.DYNAMIC], INT32],
             state: pl.Tensor[[4, BLOCK], FP32], result: pl.Tensor[[1], FP32],
             start: pl.DT_INT64, live: pl.DT_INT64, last: pl.DT_INT64,
             ignore_index: pl.DT_INT32, mean: pl.DT_INT64):
    with pl.section_vector():
        tile = pl.make_tile_group(
            type=pl.TileType(shape=[BLOCK, BLOCK], dtype=FP32, target_memory=pl.MemorySpace.Vec),
            addrs=[0], mutex_ids=[0]).current()
        indexes = pl.make_tile_group(
            type=pl.TileType(shape=[1, BLOCK], dtype=INT32, target_memory=pl.MemorySpace.Vec),
            addrs=[0x4000], mutex_ids=[1]).current()
        stats = pl.make_tile_group(
            type=pl.TileType(shape=[4, BLOCK], dtype=FP32, target_memory=pl.MemorySpace.Vec),
            addrs=[0x4100], mutex_ids=[2]).current()
        out = pl.make_tile_group(
            type=pl.TileType(shape=[1, BLOCK], dtype=FP32, target_memory=pl.MemorySpace.Vec),
            addrs=[0x4500], mutex_ids=[3]).current()
        tokens = labels.shape[0]
        pl.set_validshape(tile, [live, tokens])
        pl.load(tile, scores, [0, 0])
        pl.set_validshape(indexes, [1, tokens])
        pl.load(indexes, labels, [0])
        pl.set_validshape(stats, [4, tokens])
        if start != 0:
            pl.load(stats, state, [0, 0])
        _merge(tile, indexes, stats, start, live, tokens)
        if last != 0:
            _finish(stats, indexes, out, tokens, ignore_index, mean)
            pl.set_validshape(out, [1, 1])
            pl.store(result, out, [0])
        else:
            pl.store(state, stats, [0, 0])


def _check_shapes(hidden, weight, labels):
    if hidden.ndim != 2 or weight.ndim != 2 or labels.shape != hidden.shape[:1]:
        raise ValueError("expected hidden[T,H], weight[V,H], labels[T]")
    t, h = hidden.shape
    v = weight.shape[0]
    valid_hidden_dim = h in (64, 128) and weight.shape[1] == h
    if not 1 <= t <= BLOCK or not valid_hidden_dim or not 1 <= v < 2**31:
        raise ValueError("require 1..64 tokens, H in {64,128} and a positive INT32 vocabulary")


def _check_inputs(hidden, weight, labels, reduction, ignore_index):
    if hidden.dtype != torch.bfloat16 or weight.dtype != hidden.dtype or labels.dtype != torch.int32:
        raise ValueError("expected BF16 hidden/weight and INT32 labels")
    if hidden.device.type != "npu" or any(
        x.device != hidden.device or not x.is_contiguous() for x in (hidden, weight, labels)
    ):
        raise ValueError("expected contiguous inputs on one NPU")
    if reduction not in ("mean", "sum") or not -(2**31) <= ignore_index < 2**31:
        raise ValueError("expected sum or mean and INT32 ignore_index")


def run(hidden, weight, labels, reduction="mean", ignore_index=-100, *, streamed=True):
    """计算 FP32 标量 loss（形状 [1]），不读取设备标签到 Host。"""
    _check_shapes(hidden, weight, labels)
    _check_inputs(hidden, weight, labels, reduction, ignore_index)
    h, v = hidden.shape[1], weight.shape[0]
    slabs = (v + BLOCK - 1) // BLOCK
    state = hidden.new_empty((4, BLOCK), dtype=torch.float32)
    result = state.new_empty((1,))
    workspace = state.new_empty((1 if streamed else slabs, BLOCK, BLOCK))
    project = _make_projection(h)

    def consume(scores, start, live):
        _consume[None, 1](scores, labels, state, result, start, live,
                          int(start + live == v), ignore_index, int(reduction == "mean"))

    for slab in range(slabs):
        start = slab * BLOCK
        live = min(BLOCK, v - start)
        scores = workspace[0 if streamed else slab]
        project[None, 1](hidden, weight, scores, start, live)
        if streamed:
            consume(scores, start, live)
    if not streamed:
        for slab in range(slabs):
            start = slab * BLOCK
            consume(workspace[slab], start, min(BLOCK, v - start))
    return result
