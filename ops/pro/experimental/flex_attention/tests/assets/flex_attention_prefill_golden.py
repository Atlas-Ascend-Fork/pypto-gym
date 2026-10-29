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
"""FlexAttention prefill(bf16 hybrid-mask) golden/reference implementations.

包含:
  - flash_attention_ref_tnd / flash_attention_ref_fp32: TND 参考实现
  - cosine_similarity: 精度度量
  - hybrid_bounds / dense_mask_from_bounds: hybrid mask 期望构造
  - normalize_spans / hybrid_mask: mm_prefix_range 规范化与 mask spec
"""

import math

import torch

from flex_attention_kernel_consts import NEG_INF, TKV

# 950 PR hybrid mask 的 span 预算(与 prefill 用例中的 HYBRID_MAX_SPANS 保持一致).
HYBRID_MAX_SPANS = 16


def flash_attention_ref_tnd(q_tnd, k_tnd, v_tnd, seq_q_list, seq_kv_list, d, masks):
    """Reference attention operating on TND tensors, per-batch.

    ``masks[bi]`` is a bool tensor of shape [sq, skv] where True == masked
    (the cell is invisible and gets NEG_INF before softmax). It is the single
    source of truth shared with the kernel: both come from the same spans via
    the bound rule (host: hybrid_bounds, device: compute_row_bounds_vf).

    This mirrors the *on-device flash-attention compute flow* rather than a
    single fused fp32 softmax, so the reference stays faithful to what the
    kernel actually computes:
      * KV is consumed in TKV-wide blocks with an online (running max/sum)
        softmax, exactly like the Cube/Vector pipeline;
      * the P (probability) tile is cast to bf16 *before* the P@V matmul, as the
        kernel does (``vf.astype(..., DT_BF16)`` in ``process_vec1_*``), so P@V
        runs in bf16 with fp32 accumulation;
      * the running denominator uses the fp32 exp sum (the kernel reduces the
        sum from the fp32 exp, before the bf16 cast).
    Modelling the bf16-P quantization is what lets the reference agree with the
    kernel to ~bf16 ulp even for sliding-window masks, where the attention peak
    lands in a later block and earlier blocks' bf16 P is downscaled by exp_corr.

    GQA: q head ``ni`` reads kv head ``ni // (n_head_q // n_head_kv)``, matching
    the kernel's ``ctx.kv_n_idx``.
    """
    q_tnd = q_tnd.cpu()
    k_tnd = k_tnd.cpu()
    v_tnd = v_tnd.cpu()
    seq_q_list = seq_q_list.cpu() if torch.is_tensor(seq_q_list) else seq_q_list
    seq_kv_list = seq_kv_list.cpu() if torch.is_tensor(seq_kv_list) else seq_kv_list
    scale_val = 1.0 / math.sqrt(d)
    n = q_tnd.shape[1]
    group = n // k_tnd.shape[1]
    o_tnd = torch.zeros_like(q_tnd)
    q_off = 0
    kv_off = 0
    for bi, sq_val in enumerate(seq_q_list):
        sq = int(sq_val)
        skv = int(seq_kv_list[bi])
        mask = masks[bi].cpu()
        for ni in range(n):
            kvi = ni // group
            qi = q_tnd[q_off:q_off + sq, ni, :].float()      # [sq, d]
            ki = k_tnd[kv_off:kv_off + skv, kvi, :].float()  # [skv, d]
            vi = v_tnd[kv_off:kv_off + skv, kvi, :].float()  # [skv, d]
            running_m = torch.full((sq, 1), float(NEG_INF))
            running_l = torch.zeros((sq, 1))
            acc = torch.zeros((sq, d))
            for c0 in range(0, skv, TKV):
                c1 = min(c0 + TKV, skv)
                qk = torch.matmul(qi, ki[c0:c1].T) * scale_val          # [sq, blk] fp32
                qk = qk.masked_fill(mask[:, c0:c1], NEG_INF)
                blk_max = qk.max(dim=1, keepdim=True).values            # [sq, 1]
                new_m = torch.maximum(running_m, blk_max)
                corr = torch.exp(running_m - new_m)                     # [sq, 1]
                p = torch.exp(qk - new_m)                               # [sq, blk] fp32
                p16 = p.bfloat16().float()                              # bf16 P (FA flow)
                acc = acc * corr + torch.matmul(p16, vi[c0:c1])
                running_l = running_l * corr + p.sum(dim=1, keepdim=True)
                running_m = new_m
            o_tnd[q_off:q_off + sq, ni, :] = (acc / running_l).bfloat16()
        q_off += sq
        kv_off += skv
    return o_tnd


def flash_attention_ref_fp32(q_tnd, k_tnd, v_tnd, seq_q_list, seq_kv_list, d, masks):
    """Naive fused fp32 attention -- the reference the operator grades against.

    Deliberately *not* shaped like the kernel: one fp32 softmax over the whole
    row, no KV blocking, no bf16 P quantization. flash_attention_ref_tnd mirrors
    the kernel's flow (which is what lets it agree to ~1 ulp); this one is
    independent, so it also cross-checks that reference rather than sharing its
    assumptions. The doc's criterion against it is cosine similarity >= 0.99.
    """
    q_tnd, k_tnd, v_tnd = q_tnd.cpu(), k_tnd.cpu(), v_tnd.cpu()
    scale_val = 1.0 / math.sqrt(d)
    n = q_tnd.shape[1]
    group = n // k_tnd.shape[1]
    o_tnd = torch.zeros_like(q_tnd)
    q_off = kv_off = 0
    for bi, sq_val in enumerate(seq_q_list):
        sq, skv = int(sq_val), int(seq_kv_list[bi])
        mask = masks[bi].cpu()
        for ni in range(n):
            kvi = ni // group
            qi = q_tnd[q_off:q_off + sq, ni, :].float()
            ki = k_tnd[kv_off:kv_off + skv, kvi, :].float()
            vi = v_tnd[kv_off:kv_off + skv, kvi, :].float()
            sc = torch.matmul(qi, ki.T) * scale_val
            sc = sc.masked_fill(mask, float("-inf"))
            o_tnd[q_off:q_off + sq, ni, :] = torch.matmul(torch.softmax(sc, dim=-1), vi).bfloat16()
        q_off += sq
        kv_off += skv
    return o_tnd


def cosine_similarity(a, b):
    """Whole-tensor cosine similarity, the doc's accuracy metric.

    Accumulated in float64 deliberately. At these sizes (millions of elements)
    an fp32 dot product and fp32 norms drift by more than the deviation being
    measured: with fp32 reductions cos(x, x) came back as 0.9969 instead of 1.0,
    which would have swamped the real ~1e-6 signal and reported every case as
    far worse than it is.
    """
    x, y = a.double().flatten(), b.double().flatten()
    return float(torch.dot(x, y) / (x.norm() * y.norm()))


def hybrid_bounds(sq, skv, spans):
    """Per-row visibility bound -- the host mirror of the on-device rule.

    This is the *only* mask quantity the host computes, and it exists purely to
    drive the reference and the core-balancing cost model; nothing derived here
    is shipped to the kernel, which reads mm_prefix_range and rebuilds the mask
    itself. Query row r sits at absolute position ``q_abs = skv - sq + r`` and
    sees KV columns ``[0, bound[r])``.
    """
    q_abs = torch.arange(sq, dtype=torch.int64) + (skv - sq)
    bound = q_abs + 1
    for s, e in spans:
        hit = (q_abs >= s) & (q_abs <= e)
        bound = torch.where(hit, torch.clamp(bound, min=e + 1), bound)
    return torch.clamp(bound, max=skv)


def normalize_spans(mm_prefix_range, b):
    """Validate mm_prefix_range and return (device tensor, per-batch span list).

    Accepts what the operator accepts: ``None`` (pure causal), ``[max_spans, 2]``
    broadcast to every request, or ``[b, max_spans, 2]``; ``-1`` marks padding.
    The device tensor keeps the operator's own ``[b, max_spans, 2]`` layout --
    only padded out to the full span budget, so the kernel's span loop is a
    fixed statically unrolled length and padding slots stay inert (read
    unsigned, ``s = -1`` never matches a real position). The [2, max_spans]
    transpose the UB staging tile needs is done by the load, not here.
    """
    dev = torch.full((b, HYBRID_MAX_SPANS, 2), -1, dtype=torch.int32)
    if mm_prefix_range is None:
        return dev, [[] for _ in range(b)]

    t = torch.as_tensor(mm_prefix_range, dtype=torch.int32)
    if t.dim() == 2:                                    # [max_spans, 2] -> broadcast
        t = t.unsqueeze(0).expand(b, -1, -1)
    assert t.shape[0] == b and t.shape[2] == 2, "mm_prefix_range must be [b, max_spans, 2]"
    assert t.shape[1] <= HYBRID_MAX_SPANS, \
        "at most %d spans per request" % HYBRID_MAX_SPANS

    dev[:, :t.shape[1], :] = t
    per_batch = [[(int(s), int(e)) for s, e in t[bi].tolist() if int(s) >= 0] for bi in range(b)]
    return dev, per_batch


def dense_mask_from_bounds(bound, skv):
    """Dense [sq, skv] reference mask (True == masked) from per-row bounds."""
    cols = torch.arange(skv, dtype=torch.int64)
    return ~(cols[None, :] < bound[:, None])




def hybrid_mask(mm_prefix_range):
    """Mask spec: the operator's mm_prefix_range, verbatim."""
    return mm_prefix_range

