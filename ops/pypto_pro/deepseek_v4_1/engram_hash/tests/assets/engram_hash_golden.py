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
"""Engram hash 的 torch 参考（CPU / 任意设备，纯 FP32 无关的整型运算，逐位一致）。

本文件为 engram_hash 的 torch 参考实现（整型运算，与设备实现逐位一致），只保留
hash 部分以便本目录自包含：

    prod = ids[None].to(int64) * multipliers[:, None]
    hash = XOR_i prod[:, :, i]
    out[layer, :, (i-1)*Tb + t] = hash % vocab[layer, i-1, t] + offsets[layer, col]
    out[mask, :] = -1
"""
import torch


def engram_hash_ref(ngram_token_ids, multipliers, vocab_sizes, offsets, image_token_mask=None):
    """参考实现：输入输出布局与 ``engram_hash`` 的 wrapper 一致。"""
    num_layers = multipliers.shape[0]
    max_ngram_size = multipliers.shape[1]

    prod = ngram_token_ids.to(torch.int64).unsqueeze(0) * multipliers.unsqueeze(1)

    pieces = [[] for _ in range(num_layers)]
    hashes = prod[:, :, 0].clone()
    for i in range(1, max_ngram_size):
        hashes.bitwise_xor_(prod[:, :, i])
        for layer in range(num_layers):
            pieces[layer].append(
                (hashes[layer].unsqueeze(-1)
                 % vocab_sizes[layer, i - 1].to(torch.int64).unsqueeze(0)).to(torch.int32))

    stacked = torch.stack([torch.cat(p, dim=-1) for p in pieces], dim=0)
    output = stacked + offsets.unsqueeze(1)
    if image_token_mask is not None:
        output = output.masked_fill(image_token_mask.view(1, -1, 1).bool(), -1)
    return output
