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


"""PyPTO fused attention softmax kernel for Gemma-4.

3-pass tiled softmax: max -> sum -> normalize+write.
S_TILE = 64, FP32 internal, bf16 I/O.

Pass 1: row-wise global max  (numerically stable exp)
Pass 2: exp(x - max) sum     (denominator)
Pass 3: exp(x - max) / sum   (normalize + write)
"""

import pypto
import torch
from torch._dynamo import allow_in_graph

S_TILE = 1024


@pypto.frontend.jit
def attn_softmax_kernel(
    scores: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),
    scale: float,
    n_s_tiles: int,
):
    M = scores.shape[0]

    for m_idx in pypto.loop(M, name="M_LOOP"):
        # Pass 1: find global max
        gmax = pypto.tensor([1], pypto.DT_FP32, "gmax")
        for t in range(n_s_tiles):
            s_off = t * S_TILE
            pypto.set_vec_tile_shapes(1, S_TILE)
            row = pypto.view(
                scores, [1, S_TILE], [m_idx, s_off], valid_shape=[1, S_TILE]
            )
            r = pypto.cast(row, pypto.DT_FP32)
            r = pypto.mul(r, scale)
            tm = pypto.amax(r, dim=-1, keepdim=True)
            if t == 0:
                gmax[:] = tm
            else:
                gmax[:] = pypto.maximum(gmax, tm)

        # Pass 2: compute total exp sum
        tsum = pypto.tensor([1], pypto.DT_FP32, "tsum")
        for t in range(n_s_tiles):
            s_off = t * S_TILE
            pypto.set_vec_tile_shapes(1, S_TILE)
            row = pypto.view(
                scores, [1, S_TILE], [m_idx, s_off], valid_shape=[1, S_TILE]
            )
            r = pypto.cast(row, pypto.DT_FP32)
            r = pypto.mul(r, scale)
            r = pypto.sub(r, gmax)
            r = pypto.exp(r)
            ts = pypto.sum(r, dim=-1, keepdim=True)
            if t == 0:
                tsum[:] = ts
            else:
                tsum[:] = pypto.add(tsum, ts)

        # Pass 3: normalize and write
        for t in range(n_s_tiles):
            s_off = t * S_TILE
            pypto.set_vec_tile_shapes(1, S_TILE)
            row = pypto.view(
                scores, [1, S_TILE], [m_idx, s_off], valid_shape=[1, S_TILE]
            )
            r = pypto.cast(row, pypto.DT_FP32)
            r = pypto.mul(r, scale)
            r = pypto.sub(r, gmax)
            r = pypto.exp(r)
            r = pypto.div(r, tsum, precision_type=pypto.PrecisionType.INTRINSIC)
            r = pypto.cast(r, pypto.DT_BF16)
            pypto.assemble(r, [m_idx, s_off], out)


@allow_in_graph
def attn_softmax_wrapper(scores: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """User-facing wrapper for attention softmax.

    Accepts scores [M, S] dtype bfloat16. Pads S to a multiple of S_TILE
    with -1e9 (effectively -inf for softmax), runs the kernel, slices back.
    """
    assert scores.dtype == torch.bfloat16
    M, S = scores.shape
    n_tiles = (S + S_TILE - 1) // S_TILE
    padded_S = n_tiles * S_TILE
    FILL_VAL = -1e9
    if padded_S > S:
        pad = torch.full(
            (M, padded_S - S), FILL_VAL, dtype=torch.bfloat16, device=scores.device
        )
        scores_padded = torch.cat([scores, pad], dim=1)
    else:
        scores_padded = scores

    out_padded = torch.empty(M, padded_S, dtype=torch.bfloat16, device=scores.device)
    attn_softmax_kernel(scores_padded, out_padded, scale, n_tiles)
    return out_padded[:, :S]
