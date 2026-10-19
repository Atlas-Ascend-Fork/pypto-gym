# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Test for mtgr_ragged_segment_attention (Phase 1c)

Phase 1c additions vs Phase 1b:
- New inputs: mask0 / mask1 [1024,1024] FP32 templates, rules[seq_num]
  INT32 (0=causal, 1=full-visibility, 2=diagonal-mask),
  segment_starts[Batch, SeqNum+1] INT32 (host-side cumsum).

Three-state markers on stdout:
    [PRECISION_PASS]  -- all cases ran AND precision passes
    [PRECISION_FAIL]  -- one or more cases failed precision
    (no marker + non-zero exit) -- runtime failure
"""

from __future__ import annotations
import logging
import time
import math
from typing import Optional, Sequence, Tuple
import sys
import os

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

import numpy as np
import torch
from numpy.testing import assert_allclose

import pypto

from mtgr_segment_attention.mtgr_ragged_segment_attention_impl import (
    mtgr_ragged_segment_attention,
    Q_TILE_SIZE,
    K_TILE_SIZE,
    MASK_TEMPLATE_SIZE,
)


# Local test constants (no longer module-level in impl)
NUM_HEADS = 8
HEAD_DIM = 64
HIDDEN_DIM = NUM_HEADS * HEAD_DIM
SCALE = 1.0 / (HEAD_DIM ** 0.5)
Q_TILE = 128
K_TILE = 128
ATOL = 1e-3
RTOL = 0.0078125
MACRO_BLOCK_SIZE = 1024


# --------------------------------------------------------------------------- #
# golden
# --------------------------------------------------------------------------- #
def _build_combined_mask_for_batch(
    segment_lengths_b: Sequence[int],
    rules_list: Sequence[int],
    mask0_np: np.ndarray,
    mask1_np: np.ndarray,
) -> np.ndarray:
    """构建单个 batch 的完整 [seq_total, seq_total] attention mask。

    返回值语义：0 = visible, 1 = blocked。
    """
    seg_lens = np.asarray(segment_lengths_b, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(seg_lens)])
    s_total = int(starts[-1])
    seq_num = len(seg_lens)
    full = np.ones((s_total, s_total), dtype=np.float32)
    for sq in range(seq_num):
        rule_val = rules_list[sq]
        qs, qe = int(starts[sq]), int(starts[sq + 1])
        for sk in range(seq_num):
            ks, ke = int(starts[sk]), int(starts[sk + 1])
            if sk > sq:
                continue
            if sk < sq:
                full[qs:qe, ks:ke] = 0.0
            else:
                qlen, klen = qe - qs, ke - ks
                q_macro_count = (qlen + MACRO_BLOCK_SIZE - 1) // MACRO_BLOCK_SIZE
                k_macro_count = (klen + MACRO_BLOCK_SIZE - 1) // MACRO_BLOCK_SIZE
                for mq in range(q_macro_count):
                    mq_s = qs + mq * MACRO_BLOCK_SIZE
                    mq_e = min(qs + (mq + 1) * MACRO_BLOCK_SIZE, qe)
                    mql = mq_e - mq_s
                    for mk in range(k_macro_count):
                        mk_s = ks + mk * MACRO_BLOCK_SIZE
                        mk_e = min(ks + (mk + 1) * MACRO_BLOCK_SIZE, ke)
                        mkl = mk_e - mk_s
                        if rule_val == 0:
                            if mk < mq:
                                full[mq_s:mq_e, mk_s:mk_e] = 0.0
                            elif mk == mq:
                                full[mq_s:mq_e, mk_s:mk_e] = mask0_np[:mql, :mkl]
                        elif rule_val == 1:
                            full[mq_s:mq_e, mk_s:mk_e] = 0.0
                        elif rule_val == 2:
                            if mk == mq:
                                full[mq_s:mq_e, mk_s:mk_e] = mask1_np[:mql, :mkl]
    return full


def mtgr_ragged_segment_attention_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask0: torch.Tensor,
    mask1: torch.Tensor,
    rules: torch.Tensor,
    segment_lengths: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    num_heads: int,
    head_dim: int,
    batch_size: int,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """简化版 golden：整段 mask + 标准 attention。

    与 kernel 精度对齐的关键：P 矩阵做一次 bf16 round。
    """
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    q_cpu = q.detach().cpu().float()
    k_cpu = k.detach().cpu().float()
    v_cpu = v.detach().cpu().float()
    mask0_np = mask0.detach().cpu().float().numpy()
    mask1_np = mask1.detach().cpu().float().numpy()
    rules_list = rules.detach().cpu().to(torch.int64).tolist()

    cu_q = cu_seqlens_q.detach().cpu().to(torch.int64).tolist()
    cu_k = cu_seqlens_k.detach().cpu().to(torch.int64).tolist()
    seg_lens = segment_lengths.detach().cpu().to(torch.int64).numpy()

    hidden_dim = num_heads * head_dim
    total_seq_q = q_cpu.shape[0]
    output = torch.zeros(total_seq_q, hidden_dim, dtype=torch.float32)
    l_output = torch.zeros(total_seq_q, 1, dtype=torch.float32)
    m_output = torch.zeros(total_seq_q, 1, dtype=torch.float32)

    for b in range(batch_size):
        q_start, q_end = cu_q[b], cu_q[b + 1]
        k_start, k_end = cu_k[b], cu_k[b + 1]
        seq_len = q_end - q_start
        if seq_len == 0:
            continue
        seg_lens_b = seg_lens[b]

        # ── 1. 构建本 batch 的完整 mask [seq_len, seq_len] ──
        mask_np = _build_combined_mask_for_batch(
            seg_lens_b.tolist(), rules_list, mask0_np, mask1_np,
        )
        mask_t = torch.from_numpy(mask_np)  # 0=visible, 1=blocked

        # ── 2. 逐 head 做标准 attention ──
        for h in range(num_heads):
            h_off = h * head_dim
            q_h = q_cpu[q_start:q_end, h_off:h_off + head_dim]  # [S, D]
            k_h = k_cpu[k_start:k_end, h_off:h_off + head_dim]
            v_h = v_cpu[k_start:k_end, h_off:h_off + head_dim]

            scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * scale
            # 应用 mask：blocked 位置减去大数 → softmax 后趋近 0
            scores = scores - mask_t * 40000.0

            # online softmax 数值路径：先 max-shift，exp 后做 bf16 round，再归一化
            m = scores.max(dim=-1, keepdim=True)[0]
            exp_s = torch.exp(scores - m)
            # bf16 round 在未归一化的 exp 上（匹配 kernel 逐 tile 的 pij_bf16）
            exp_bf16 = exp_s.to(torch.bfloat16).to(torch.float32)
            l = exp_bf16.sum(dim=-1, keepdim=True)
            p_bf16 = exp_bf16 / l

            out_h = torch.matmul(p_bf16, v_h)
            output[q_start:q_end, h_off:h_off + head_dim] = \
                                out_h.to(torch.bfloat16).to(torch.float32)

    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, l_output, m_output


# --------------------------------------------------------------------------- #
# Mask templates and segment helpers
# --------------------------------------------------------------------------- #
def build_mask0(size=MASK_TEMPLATE_SIZE):
    """Lower-triangular template: mask0[i, j] = 1 if j > i else 0."""
    return torch.triu(torch.ones(size, size, dtype=torch.float32), diagonal=1)


def build_mask1(size=MASK_TEMPLATE_SIZE):
    """Diagonal-only template: mask1[i, j] = 0 if i == j else 1."""
    m = torch.ones(size, size, dtype=torch.float32)
    m.fill_diagonal_(0.0)
    return m


def build_segment_starts(segment_lengths_per_batch):
    """Compute [Batch, SeqNum+1] int32 cumulative starts on host (B-scheme).

    segment_lengths_per_batch: List[List[int]]  (Batch x SeqNum)
    """
    arr = np.asarray(segment_lengths_per_batch, dtype=np.int64)
    batch_size, seq_num = arr.shape
    starts = np.zeros((batch_size, seq_num + 1), dtype=np.int32)
    for b in range(batch_size):
        cum = 0
        for s in range(seq_num):
            starts[b, s] = cum
            cum += int(arr[b, s])
        starts[b, seq_num] = cum
    return starts


def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        logger.warning("Please set TILE_FWK_DEVICE_ID before running:")
        logger.warning("  export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        return int(os.environ['TILE_FWK_DEVICE_ID'])
    except ValueError:
        return None


def create_inputs(segment_lengths_per_batch, rules_list, device, seed=42):
    """Build q/k/v/mask0/mask1/rules/segment_starts/cu_seqlens for a case."""
    seq_lens = [int(sum(s)) for s in segment_lengths_per_batch]
    total_q = sum(seq_lens)
    total_k = sum(seq_lens)  # Q/K segments aligned

    torch.manual_seed(seed)
    q = torch.randn(total_q, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_k, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_k, HIDDEN_DIM, dtype=torch.bfloat16, device=device)

    cu_seqlens_q = torch.tensor(
        [0] + list(np.cumsum(seq_lens)), dtype=torch.int32, device=device,
    )
    cu_seqlens_k = torch.tensor(
        [0] + list(np.cumsum(seq_lens)), dtype=torch.int32, device=device,
    )

    mask0 = build_mask0().to(device=device)
    mask1 = build_mask1().to(device=device)

    rules = torch.tensor(rules_list, dtype=torch.int32, device=device)
    segment_starts_np = build_segment_starts(segment_lengths_per_batch)
    segment_starts = torch.from_numpy(segment_starts_np).to(device=device)

    # Golden side wants segment_lengths (Batch x SeqNum int32).
    segment_lengths = torch.tensor(
        segment_lengths_per_batch, dtype=torch.int32, device=device,
    )

    return (
        q, k, v, mask0, mask1, rules, segment_starts,
        segment_lengths, cu_seqlens_q, cu_seqlens_k,
    )


def compare_with_golden(out_kernel, out_golden, atol, rtol):
    a = out_kernel.float().cpu()
    b = out_golden.float().cpu()
    diff = (a - b).abs()
    max_abs = diff.max().item()
    denom = b.abs().clamp(min=1e-12)
    rel = diff / denom
    max_rel = rel.max().item()
    ok = diff <= (atol + rtol * b.abs())
    fail_mask = ~ok
    fail_count = int(fail_mask.sum().item())
    total = int(fail_mask.numel())

    passed = True
    try:
        assert_allclose(a.numpy(), b.numpy(), atol=atol, rtol=rtol)
    except AssertionError as e:
        passed = False
        logger.error("  [assert_allclose] failed: %s", str(e).splitlines()[0])

    worst = []
    if fail_count > 0:
        flat_diff = diff.flatten()
        topk = min(5, fail_count)
        idxs = torch.topk(flat_diff, topk).indices.tolist()
        for idx in idxs:
            worst.append({
                "flat_idx": idx,
                "kernel": float(a.flatten()[idx]),
                "golden": float(b.flatten()[idx]),
                "abs_err": float(flat_diff[idx]),
            })
    return passed, max_abs, max_rel, fail_count, total, worst


def run_case(label, device, segment_lengths_per_batch, rules_list):
    batch_size = len(segment_lengths_per_batch)
    seq_num = len(rules_list)
    seq_lens = [sum(s) for s in segment_lengths_per_batch]
    total_q = sum(seq_lens)

    logger.info("=" * 60)
    logger.info("Case: %s", label)
    logger.info("  batch_size=%d, seq_num=%d", batch_size, seq_num)
    logger.info("  segment_lengths=%s", segment_lengths_per_batch)
    logger.info("  rules=%s  (0=causal, 1=full, 2=diagonal)", rules_list)
    logger.info("  total_q=%d", total_q)
    logger.info("  NUM_HEADS=%d, HEAD_DIM=%d, Q_tile=%d, K_tile=%d",
                NUM_HEADS, HEAD_DIM, Q_TILE_SIZE, K_TILE_SIZE)
    logger.info("=" * 60)

    (q, k, v, mask0, mask1, rules, segment_starts,
     segment_lengths, cu_seqlens_q, cu_seqlens_k) = create_inputs(
        segment_lengths_per_batch, rules_list, device,
    )

    out = torch.empty(total_q, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    l_out = torch.empty(total_q, 1, dtype=torch.float32, device=device)
    m_out = torch.empty(total_q, 1, dtype=torch.float32, device=device)

    logger.info("Running kernel...")
    start = time.time()
    mtgr_ragged_segment_attention(
        q, k, v, mask0, mask1, rules, segment_starts,
        out, l_out, m_out, cu_seqlens_q, cu_seqlens_k,
        batch_size, seq_num, NUM_HEADS, HEAD_DIM,
    )
    elapsed = time.time() - start
    logger.info("  Kernel time: %.3fs", elapsed)

    logger.info("Computing golden reference (CPU)...")
    golden_out, _, _ = mtgr_ragged_segment_attention_golden(
        q, k, v, mask0, mask1, rules, segment_lengths,
        cu_seqlens_q, cu_seqlens_k,
        num_heads=NUM_HEADS, head_dim=HEAD_DIM, batch_size=batch_size,
        scale=SCALE,
    )

    passed, max_abs, max_rel, fc, total, worst = compare_with_golden(
        out, golden_out, ATOL, RTOL,
    )
    logger.info("  max_abs_err = %.6e", max_abs)
    logger.info("  max_rel_err = %.6e", max_rel)
    logger.info("  fail_count  = %d / %d", fc, total)
    if not passed:
        logger.warning("  worst offenders (top %d):", len(worst))
        for w in worst:
            logger.warning("    flat_idx=%6d  kernel=%+.6e  golden=%+.6e  abs_err=%.6e",
                           w['flat_idx'], w['kernel'], w['golden'], w['abs_err'])
    return passed


# --------------------------------------------------------------------------- #
# OL21 test entries
# --------------------------------------------------------------------------- #
def test_level0_phase1c_single_seg_type0_512(device):
    """C1: single triangular segment (=Phase 1a equivalent)."""
    return run_case("C1 single-seg causal", device, [[1600, 8, 200, 1200]], [0, 1, 2, 2])


def test_level0_phase1c_seg_type0_then_full(device):
    """C2: causal(16) + fully-enabled(512)."""
    return run_case("C2 causal+full", device, [[1600, 8, 200, 1200]], [0, 1, 0, 2])


def test_level0_phase1c_three_segments(device):
    """C3: causal(16) + full(8) + diagonal(488)."""
    return run_case("C3 three-seg", device, [[1600, 8, 200, 1200], [1700, 8, 300, 1024]], [0, 1, 2, 2])


def test_level0_phase1c_4seg_large_len(device):
    """C8: 4-seg with large first segment (2000+8+100+1000), rules [0,1,0,2]."""
    return run_case(
        "C8 4-seg large [2000,8,100,1000]",
        device,
        [[2200, 8, 200, 1024], [1700, 8, 300, 1100], [2440, 8, 200, 2048], [1600, 8, 300, 1800],\
        [3300, 8, 200, 1300], [1700, 8, 300, 2048], [1780, 8, 300, 1200], [2048, 8, 500, 1800]],
        [0, 1, 0, 2],
    )


def test_level1_phase1c_3seg_large_len(device):
    """C9: 3-seg with large first segment (2000+8+100+1000), rules [0,1,0,2]."""
    return run_case(
        "C9 3-seg large [1600,8,1024]",
        device,
        [[1600, 200, 1024]],
        [1, 0, 2],
    )


def main():
    logger.info("=" * 60)
    logger.info("mtgr_ragged_segment_attention -- Phase 1c Test")
    logger.info("Input shape: [total_seq, %d]", HIDDEN_DIM)
    logger.info("=" * 60)

    device_id = get_device_id()
    if device_id is None:
        return 2

    try:
        import torch_npu
        torch.npu.set_device(device_id)
        device = f'npu:{device_id}'
    except Exception as e:
        logger.error("[error] failed to init NPU: %s", e, exc_info=True)
        return 3

    cases = [
        ("C1", test_level0_phase1c_single_seg_type0_512),
        ("C2", test_level0_phase1c_seg_type0_then_full),
        ("C3", test_level0_phase1c_three_segments),
        ("C8", test_level0_phase1c_4seg_large_len),
        ("C9", test_level1_phase1c_3seg_large_len),
    ]

    try:
        all_passed = True
        results = []
        for name, fn in cases:
            logger.info(">>>> Running %s <<<<", name)
            try:
                ok = fn(device)
            except Exception as e:
                logger.error("[case-runtime-error] %s: %s", name, e, exc_info=True)
                return 1
            results.append((name, ok))
            if not ok:
                all_passed = False

        logger.info("=" * 60)
        for name, ok in results:
            logger.info("  %s: %s", name, 'PASS' if ok else 'FAIL')
        logger.info("=" * 60)
        if all_passed:
            logger.info("[PRECISION_PASS] All Phase 1c cases passed (atol=%s, rtol=%s).", ATOL, RTOL)
            return 0
        else:
            logger.warning("[PRECISION_FAIL] One or more Phase 1c cases failed precision check.")
            return 0  # precision-fail is not a runtime fail
    except Exception as e:
        logger.error("[runtime-error] %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
