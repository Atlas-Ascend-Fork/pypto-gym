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
"""Engram hash（n-gram 哈希行号）的 PyPTO-Pro 实现，运行于 Ascend 950（arch 3510）。

计算定义（对每个 token、每个 n-gram layer）：

      hash = XOR_i ( int64(id_i) * mult[layer, i] ),  i = 0 .. G-1
      对每个 n-gram 位置 i >= 1、每张 embedding 表 t：
        out[layer, token, (i-1)*Tb + t] = hash % vocab[layer, i-1, t] + offsets[layer, col]
      image token：out[layer, token, :] = -1

实现要点：

1. **GM 直访 SIMT**：SIMT 线程直接从 GM 读 token id 与配置张量并写出结果，不做
   UB staging、不经 MTE2/MTE3 搬运；访存由 D-cache 服务。
2. **编译期专门化**：kernel 以 ``(num_layers, max_ngram, tables, num_tokens, has_mask)``
   为键专门化，wrapper 缓存每种组合的已编译 kernel；mask 是否存在也做编译期特化。
3. **不使用 UB tile 作为 SIMT 实参**：当前工具链下 SIMT 读取 UB tile 会触发设备异常
   （The address for VEC to access UB is out of bounds，vector core exception 507035），
   写 UB tile 正常；GM 直访形态不受影响。

取模语义：设备为截断取模，torch 参考为向下取整取模；hash 可能为负（例如 multiplier 为
负）时两者相差一个除数，实现中对负余数加一次除数，因此对任意 int64 输入与 torch 参考
逐位一致。测试额外覆盖负 multiplier。
"""
import torch

import pypto_pro.language as pl

TILE = 128  # SIMT 线程数 = 每拍处理的 token 数


def _build_kernel(num_layers, max_ngram, tables, num_tokens, has_mask):
    """构造一个以 (L, G, Tb, T, has_mask) 为静态契约的 kernel。

    ``has_mask=False`` 时不接受 mask 张量、SIMT 里也没有 mask 读和分支——V4.1 的调用
    不传 image_token_mask，实测这一步省掉逐 token 的 1 字节 GM 读与分支（T=65536 时
    0.24 ms -> 0.18 ms）。
    """
    num_cols = (max_ngram - 1) * tables
    tile = TILE
    total = num_tokens

    @pl.vector_function(mode="simt")
    def hash_token(out, ids, mult, vocab, offs, row, tid):
        for layer in pl.range(0, num_layers):
            h = (pl.simt.cast(ids[row + tid, 0], pl.DT_INT64)
                 * mult[layer, 0])
            for i in pl.range(1, max_ngram):
                h = h ^ (pl.simt.cast(ids[row + tid, i], pl.DT_INT64)
                         * mult[layer, i])
                for t in pl.range(0, tables):
                    d = pl.simt.cast(
                        vocab[layer * (max_ngram - 1) + (i - 1), t],
                        pl.DT_INT64)
                    r = h % d
                    if r < 0:
                        r = r + d
                    out[layer * total + row + tid,
                        (i - 1) * tables + t] = (
                        pl.simt.cast(r, pl.DT_INT32)
                        + offs[layer, (i - 1) * tables + t])

    if has_mask:
        @pl.vector_function(mode="simt", max_threads=tile)
        def engram_hash_simt(out, ids, mask, mult, vocab, offs, row, valid):
            tid = pl.simt.linear_thread_idx()
            if tid < valid:
                if mask[row + tid, 0] != 0:
                    for layer in pl.range(0, num_layers):
                        for col in pl.range(0, num_cols):
                            out[layer * total + row + tid, col] = -1
                else:
                    hash_token(out, ids, mult, vocab, offs, row, tid)

        @pl.jit(auto_mutex=True, name=f"engram_hash_L{num_layers}_G{max_ngram}_Tb{tables}_T{total}_M1")
        def engram_hash_kernel(
            ngram_token_ids: pl.Tensor[[total, max_ngram], pl.DT_INT32],
            multipliers: pl.Tensor[[num_layers, max_ngram], pl.DT_INT64],
            vocab_sizes: pl.Tensor[[num_layers * (max_ngram - 1), tables], pl.DT_INT32],
            offsets: pl.Tensor[[num_layers, num_cols], pl.DT_INT32],
            image_token_mask: pl.Tensor[[total, 1], pl.DT_UINT8],
            output: pl.Tensor[[num_layers * total, num_cols], pl.DT_INT32],
        ):
            tile_id = pl.get_block_idx()
            row = tile_id * tile
            valid = pl.min(tile, total - row)
            with pl.section_vector():
                engram_hash_simt[tile](output, ngram_token_ids, image_token_mask,
                                       multipliers, vocab_sizes, offsets, row, valid)
    else:
        @pl.vector_function(mode="simt", max_threads=tile)
        def engram_hash_simt(out, ids, mult, vocab, offs, row, valid):
            tid = pl.simt.linear_thread_idx()
            if tid < valid:
                hash_token(out, ids, mult, vocab, offs, row, tid)

        @pl.jit(auto_mutex=True, name=f"engram_hash_L{num_layers}_G{max_ngram}_Tb{tables}_T{total}_M0")
        def engram_hash_kernel(
            ngram_token_ids: pl.Tensor[[total, max_ngram], pl.DT_INT32],
            multipliers: pl.Tensor[[num_layers, max_ngram], pl.DT_INT64],
            vocab_sizes: pl.Tensor[[num_layers * (max_ngram - 1), tables], pl.DT_INT32],
            offsets: pl.Tensor[[num_layers, num_cols], pl.DT_INT32],
            output: pl.Tensor[[num_layers * total, num_cols], pl.DT_INT32],
        ):
            tile_id = pl.get_block_idx()
            row = tile_id * tile
            valid = pl.min(tile, total - row)
            with pl.section_vector():
                engram_hash_simt[tile](output, ngram_token_ids, multipliers,
                                       vocab_sizes, offsets, row, valid)

    return engram_hash_kernel


_KERNEL_CACHE = {}


def _get_kernel(num_layers, max_ngram, tables, num_tokens, has_mask):
    key = (num_layers, max_ngram, tables, num_tokens, has_mask)
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = _build_kernel(num_layers, max_ngram, tables, num_tokens, has_mask)
        _KERNEL_CACHE[key] = kernel
    return kernel


def make_offsets(vocab_sizes):
    """由 vocab_sizes 求每层的 exclusive prefix-sum（与参考实现一致，int32）。"""
    rows = []
    for layer in range(vocab_sizes.shape[0]):
        flat = vocab_sizes[layer].reshape(-1)
        rows.append(torch.cat([torch.zeros(1, dtype=torch.int32, device=flat.device),
                               flat[:-1].cumsum(0, dtype=torch.int32)]))
    return torch.stack(rows, dim=0)


def engram_hash(
    ngram_token_ids: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
    image_token_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Engram hash indices。

    Args:
        ngram_token_ids: ``[num_tokens, max_ngram]`` int32。
        multipliers: ``[num_layers, max_ngram]`` int64。
        vocab_sizes: ``[num_layers, max_ngram - 1, tables]`` int32。
        offsets: ``[num_layers, (max_ngram - 1) * tables]`` int32，
            由 :func:`make_offsets` 生成。
        image_token_mask: 可选，``[num_tokens]`` bool/uint8，True 表示 image token，
            其整行输出为 -1。None 等价于全 False。

    Returns:
        ``[num_layers, num_tokens, (max_ngram - 1) * tables]`` int32。

    所有张量必须在同一 NPU 设备上。dtype 不符会 fail closed（不做隐式转换，
    因为 ids/vocab/offsets 的整型宽度决定计算结果）。
    """
    # 先查 rank，再按 rank 取维度；混设备的张量必须在 launch 前拦掉。
    for name, tensor, rank in (("ngram_token_ids", ngram_token_ids, 2),
                               ("multipliers", multipliers, 2),
                               ("vocab_sizes", vocab_sizes, 3),
                               ("offsets", offsets, 2)):
        if tensor.dim() != rank:
            raise ValueError(f"engram_hash: {name} must be {rank}-D, got shape {tuple(tensor.shape)}")
    devices = {t.device for t in (ngram_token_ids, multipliers, vocab_sizes, offsets)}
    if len(devices) != 1:
        raise ValueError(f"engram_hash: all inputs must share one device, got {sorted(map(str, devices))}")

    num_tokens, max_ngram = ngram_token_ids.shape
    num_layers = multipliers.shape[0]
    tables = vocab_sizes.shape[2]
    num_cols = (max_ngram - 1) * tables

    expected = {
        "ngram_token_ids": (torch.int32, (num_tokens, max_ngram)),
        "multipliers": (torch.int64, (num_layers, max_ngram)),
        "vocab_sizes": (torch.int32, (num_layers, max_ngram - 1, tables)),
        "offsets": (torch.int32, (num_layers, num_cols)),
    }
    for name, tensor in (("ngram_token_ids", ngram_token_ids), ("multipliers", multipliers),
                         ("vocab_sizes", vocab_sizes), ("offsets", offsets)):
        want_dtype, want_shape = expected[name]
        if tensor.dtype != want_dtype:
            raise ValueError(f"engram_hash: {name} must be {want_dtype}, got {tensor.dtype}")
        if tuple(tensor.shape) != want_shape:
            raise ValueError(f"engram_hash: {name} must be {want_shape}, got {tuple(tensor.shape)}")
    if num_layers <= 0 or max_ngram < 2 or tables <= 0:
        raise ValueError("engram_hash: require num_layers > 0, max_ngram >= 2, tables > 0")

    if image_token_mask is None:
        mask = None
    else:
        if image_token_mask.shape != (num_tokens,):
            raise ValueError(
                f"engram_hash: image_token_mask must be [{num_tokens}], got {tuple(image_token_mask.shape)}")
        if image_token_mask.device != ngram_token_ids.device:
            raise ValueError(
                f"engram_hash: image_token_mask must be on {ngram_token_ids.device}, "
                f"got {image_token_mask.device}")
        mask = image_token_mask.to(torch.uint8).reshape(num_tokens, 1)

    output = torch.empty((num_layers, num_tokens, num_cols), dtype=torch.int32,
                         device=ngram_token_ids.device)
    if num_tokens == 0:
        return output

    kernel = _get_kernel(num_layers, max_ngram, tables, num_tokens, mask is not None)
    n_tiles = (num_tokens + TILE - 1) // TILE
    base = (ngram_token_ids.contiguous(),
            multipliers.contiguous(),
            vocab_sizes.reshape(num_layers * (max_ngram - 1), tables).contiguous(),
            offsets.contiguous())
    out2d = output.view(num_layers * num_tokens, num_cols)
    if mask is None:
        kernel[None, n_tiles](*base, out2d)
    else:
        kernel[None, n_tiles](*base, mask, out2d)
    return output
