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
"""MiniMax-M3 MSA (MiniMax Sparse Attention) block-sparse DECODE attention — PyPTO tile kernel.

Decode step (Q seq-len = 1) for the M3 GQA attention (Hq=64, Hkv=4, head_dim=128), attending ONLY the
lightning-indexer-selected key blocks (``sparse_block_size`` = 128, ``sparse_topk_blocks`` = 16). The
GQA group (``Hq // Hkv`` = 16 query heads sharing one KV head) is batched into the cube M-axis, and the
kernel runs an online-softmax flash over the selected keys in ``NTILE``-wide chunks (bigger matmuls than
per-128-block). The current (partial) block is masked column-wise via ``valid_mask``.

The selected KV blocks are gathered into a compact tensor on the host (per KV head); the block selection
is shared across heads because the M3 indexer max-pools the scores over the index heads. Pair with
``minimax_m3_msa_indexer_impl`` for the selection.

NOTE (honest): for the M3 decode shapes the native ``npu_fused_infer_attention_score`` (paged
block-sparse via ``block_table``) is currently ~5x faster than this PyPTO kernel — same structural
finding as the MoE grouped-GEMM (a hand-tuned native decode kernel beats the tile-DSL for this
memory-bound, M=16 GQA-decode shape). This kernel is the PyPTO-native MSA path; the native op is the
speed oracle/target. Follow-ups to close the gap: single-shot softmax (drop online recombine), fuse the
indexer block-max + top-k, and an on-device gather (``gather_in_ub`` + ``block_table``).
"""
import math
import os

import torch
import pypto
from torch._dynamo import allow_in_graph

HQ, HKV, D = 64, 4, 128
GROUP = HQ // HKV          # 16 q-heads per kv-head -> cube M (16-aligned)
BY = 128                   # KV block == config.sparse_attention_config.sparse_block_size
SCALE = 1.0 / math.sqrt(D)
LARGE_NEG = -3.0e38
NTILE = int(os.environ.get("MSA_NTILE", "512"))    # online-softmax chunk width (512 fastest in sweep)

_kernel_cache = {}


def _get_decode_kernel(bsz, topk):
    """JIT-compile (and cache) the decode kernel for a (batch, topk) shape."""
    key = (bsz, topk)
    if key in _kernel_cache:
        return _kernel_cache[key]
    total_outer = bsz * HKV
    sel = topk * BY
    nchunk = sel // NTILE

    @pypto.frontend.jit(
        runtime_options={"device_sched_mode": 1, "stitch_function_max_num": 64},
        pass_options={"cube_l1_reuse_setting": {-1: 3}, "cube_nbuffer_setting": {-1: 4},
                      "vec_nbuffer_setting": {-1: 1}},   # cube pipeline depth (the GEMM-winning knob)
    )
    def kernel(
        q_2d: pypto.Tensor([bsz * HQ, D], pypto.DT_BF16),
        k_cmp: pypto.Tensor([bsz * HKV * topk * BY, D], pypto.DT_BF16),
        v_cmp: pypto.Tensor([bsz * HKV * topk * BY, D], pypto.DT_BF16),
        valid_mask: pypto.Tensor([bsz, sel], pypto.DT_FP32),
        output_2d: pypto.Tensor([bsz * HQ, D], pypto.DT_BF16),
    ):
        dtype = q_2d.dtype
        pypto.experimental.set_operation_options(combine_axis=True)
        for outer in range(total_outer):           # static unroll (B*HKV is static) -> better scheduling
            b_idx = outer // HKV
            hkv = outer % HKV
            q_ofs = b_idx * HQ + hkv * GROUP

            mi = pypto.tensor([GROUP, 1], pypto.DT_FP32, "mi")
            li = pypto.tensor([GROUP, 1], pypto.DT_FP32, "li")
            oi = pypto.tensor([GROUP, D], pypto.DT_FP32, "oi")

            for c in range(nchunk):           # static unroll over key chunks (N=NTILE matmuls)
                kv_ofs = outer * sel + c * NTILE
                pypto.set_vec_tile_shapes(GROUP, NTILE)
                q_block = pypto.view(q_2d, [GROUP, D], [q_ofs, 0])
                k_ch = pypto.view(k_cmp, [NTILE, D], [kv_ofs, 0])
                v_ch = pypto.view(v_cmp, [NTILE, D], [kv_ofs, 0])

                pypto.set_cube_tile_shapes([GROUP, D], [D, NTILE], [GROUP, NTILE])  # [M,K],[K,N],[M,N]
                s = pypto.matmul(q_block, k_ch, pypto.DT_FP32, a_trans=False, b_trans=True)
                s = pypto.mul(s, SCALE)
                mask_row = pypto.view(valid_mask, [1, NTILE], [b_idx, c * NTILE])   # row-broadcast
                inv = pypto.add(pypto.mul(mask_row, -1.0), 1.0)
                s = pypto.add(pypto.mul(s, mask_row), pypto.mul(inv, LARGE_NEG))

                m_c = pypto.amax(s, dim=-1, keepdim=True)
                p = pypto.exp(pypto.sub(s, m_c))
                l_c = pypto.sum(p, dim=-1, keepdim=True)
                p_bf16 = pypto.cast(p, dtype)
                pypto.set_cube_tile_shapes([GROUP, 128], [128, D], [GROUP, D])      # K tiled by 128
                o_c = pypto.matmul(p_bf16, v_ch, pypto.DT_FP32)

                if c == 0:
                    mi[:] = m_c
                    li[:] = l_c
                    oi[:] = o_c
                else:
                    mi_new = pypto.maximum(mi, m_c)
                    alpha = pypto.exp(pypto.sub(mi, mi_new))
                    beta = pypto.exp(pypto.sub(m_c, mi_new))
                    li[:] = pypto.add(pypto.mul(alpha, li), pypto.mul(beta, l_c))
                    oi[:] = pypto.add(pypto.mul(oi, alpha), pypto.mul(o_c, beta))
                    mi[:] = mi_new
            out = pypto.div(oi, li)
            pypto.set_vec_tile_shapes(GROUP, D)
            pypto.assemble(pypto.cast(out, dtype), [q_ofs, 0], output_2d)

    _kernel_cache[key] = kernel
    return kernel


@allow_in_graph
@torch.no_grad()
def minimax_m3_msa_sparse_decode(q, k_blocks, v_blocks, block_ids, seq_len):
    """MSA block-sparse decode attention (B sequences, 1 query token each).

    Args:
        q:         [B, HQ, D] bf16 — the decode query (post q_proj + q_norm + RoPE).
        k_blocks:  [B, HKV, nb, BY, D] bf16 — paged KV cache keys (post k_norm + RoPE).
        v_blocks:  [B, HKV, nb, BY, D] bf16 — paged KV cache values.
        block_ids: [B, topk] int — indexer-selected block ids (shared across heads).
        seq_len:   int — total KV length (the current/last selected block may be partial).
    Returns:
        [B, HQ, D] bf16 attention output.
    """
    bsz, topk = block_ids.shape
    dev = q.device
    cur_block = (seq_len - 1) // BY
    cur_valid = seq_len - cur_block * BY
    k_cmp = torch.empty(bsz * HKV * topk * BY, D, dtype=torch.bfloat16, device=dev)
    v_cmp = torch.empty(bsz * HKV * topk * BY, D, dtype=torch.bfloat16, device=dev)
    valid_mask = torch.ones(bsz, topk * BY, dtype=torch.float32, device=dev)
    for b in range(bsz):
        sel = block_ids[b].long()
        for j, blk in enumerate(sel.tolist()):
            if blk == cur_block:
                valid_mask[b, j * BY + cur_valid:(j + 1) * BY] = 0.0
            elif blk > cur_block:
                valid_mask[b, j * BY:(j + 1) * BY] = 0.0
        for hkv in range(HKV):
            base = (b * HKV + hkv) * topk * BY
            k_cmp[base:base + topk * BY] = k_blocks[b, hkv, sel].reshape(topk * BY, D)
            v_cmp[base:base + topk * BY] = v_blocks[b, hkv, sel].reshape(topk * BY, D)
    out = torch.zeros(bsz * HQ, D, dtype=torch.bfloat16, device=dev)
    _get_decode_kernel(bsz, topk)(q.reshape(bsz * HQ, D).contiguous(), k_cmp, v_cmp, valid_mask, out)
    return out.reshape(bsz, HQ, D)
