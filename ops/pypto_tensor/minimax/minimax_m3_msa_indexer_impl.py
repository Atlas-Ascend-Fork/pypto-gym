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
"""MiniMax-M3 MSA lightning-INDEXER (block selection) — PyPTO tile kernel.

Decode step (1 query): scores the ``sparse_num_index_heads`` (=4) index-query heads against the single
(MQA) index key over all keys, then max-pools each ``sparse_block_size`` (=128) token block. The PyPTO
kernel does the O(ctx) heavy part — the BF16 score matmul into a per-block ``[NPAD, nb*BY]`` score
buffer; the cheap tail (block max-pool over BY, head-max over the 4 index heads, top-k over the ~nb
block scores, and forcing the local block in) runs in eager torch on the tiny ``[nb]`` vector.

Cube constraint: the matmul M-axis must be 16-aligned, so the 4 index heads are zero-padded to 16
(``NPAD``); the pad rows produce score 0 and are dropped on the host (slice ``[:NIDX]``).

Block top-k selection matches the in-repo modeling indexer (``MiniMaxM3Attention._msa_block_indices`` /
``_msa_decode_block_table``): top-(topk-local) over the non-local blocks + the ``sparse_local_block``
most-recent blocks forced in. Validated to pick the IDENTICAL block set as that torch reference.
"""
import torch
import pypto
from torch._dynamo import allow_in_graph

NIDX = 4                   # config.sparse_attention_config.sparse_num_index_heads
NPAD = 16                  # cube M must be 16-aligned; pad the 4 index heads to 16
D = 128                    # sparse_index_dim
BY = 128                   # sparse_block_size
TOPK, LOCAL = 16, 1        # sparse_topk_blocks, sparse_local_block

_kernel_cache = {}


def _get_indexer_kernel(nb):
    """JIT-compile (and cache) the index-score kernel for ``nb`` key blocks.

    Emits the BF16 score matmul of the 16-padded index heads against each ``BY``-wide key block into a
    ``[NPAD, nb*BY]`` FP32 score buffer (the O(ctx) part); the block max-pool / top-k tail runs in eager
    torch on the host. Cached per ``nb`` because the tensor shapes are static.
    """
    if nb in _kernel_cache:
        return _kernel_cache[nb]

    @pypto.frontend.jit(
        runtime_options={"device_sched_mode": 1, "stitch_function_max_num": 64},
        pass_options={"cube_l1_reuse_setting": {-1: 4}, "vec_nbuffer_setting": {-1: 2}},
    )
    def kernel(
        idx_q: pypto.Tensor([NPAD, D], pypto.DT_BF16),       # 4 real heads + 12 zero-pad rows
        idx_k: pypto.Tensor([nb * BY, D], pypto.DT_BF16),    # single (MQA) index key over all keys
        scores_out: pypto.Tensor([NPAD, nb * BY], pypto.DT_FP32),
    ):
        for blk in range(nb):                                # static unroll -> static assemble offsets
            pypto.set_vec_tile_shapes(NPAD, BY)
            q_v = pypto.view(idx_q, [NPAD, D], [0, 0])
            k_blk = pypto.view(idx_k, [BY, D], [blk * BY, 0])
            pypto.set_cube_tile_shapes([NPAD, D], [D, BY], [NPAD, BY])   # [M,K],[K,N],[M,N]
            s = pypto.matmul(q_v, k_blk, pypto.DT_FP32, a_trans=False, b_trans=True)  # [NPAD, BY]
            s_v = pypto.mul(s, 1.0)                           # vector stage (matmul -> vec -> GM)
            pypto.assemble(s_v, [0, blk * BY], scores_out)

    _kernel_cache[nb] = kernel
    return kernel


@allow_in_graph
@torch.no_grad()
def minimax_m3_msa_indexer(idx_q, idx_k, nb):
    """Lightning-indexer block selection for a single decode query.

    Args:
        idx_q: [NIDX, D] bf16 — index-query heads (post index_q_norm + RoPE).
        idx_k: [nb*BY, D] bf16 — single index key over all keys (post index_k_norm + RoPE).
        nb:    int — number of 128-token key blocks (ceil(seq_len / 128)).
    Returns:
        [1, min(TOPK, nb)] int32 — selected block ids: top-(TOPK-LOCAL) by max-pooled score + the LOCAL
        most-recent blocks. When ``nb <= TOPK`` there is nothing to prune, so all ``nb`` blocks are
        returned (``arange(nb)``) — mirrors the in-repo ``_msa_decode_block_table`` guard and avoids the
        ``topk(k)`` out-of-range crash when fewer than ``TOPK`` blocks exist (short context).
    """
    dev = idx_q.device
    q_pad = torch.zeros(NPAD, D, dtype=idx_q.dtype, device=dev)
    q_pad[:NIDX] = idx_q
    scores = torch.zeros(NPAD, nb * BY, dtype=torch.float32, device=dev)
    _get_indexer_kernel(nb)(q_pad.contiguous(), idx_k.contiguous(), scores)
    blk = scores[:NIDX].view(NIDX, nb, BY).amax(-1).amax(0)          # block max-pool, then head max
    ksel = min(TOPK, nb)
    if LOCAL > 0 and nb > ksel:
        ids = blk[:nb - LOCAL].topk(ksel - LOCAL, sorted=False).indices.to(torch.int32)
        loc = torch.arange(nb - LOCAL, nb, device=dev, dtype=torch.int32)
        return torch.cat([ids, loc]).view(1, ksel)
    return torch.arange(nb, device=dev, dtype=torch.int32).view(1, nb)
