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

"""FlexAttention mask_info host-side test (kernel: flex_attention/op_kernel).

The host interface flex_attention() takes a compile_mask.md mask_info dict and
covers the bitmask interval schedule (primary path): FP16 (10 cfgs) + BF16 (5 cfgs),
band masks (causal + sliding window) via pre_tokens/next_tokens.

Golden masks are derived independently from the same band formulas.

Usage:
    python3 python/tests/st/pypto_pro/frontend/fa/test_flex_attention.py
"""

import logging
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "op_kernel"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flex_attention import (  # noqa: E402
    flex_attention,
    CHUNK, MASK_NUM_LIMIT, TKV, TD, QK_PRELOAD, FIFO_SIZE, NEG_INF,
)
from mask_specs import (  # noqa: E402
    FULL_SPEC, CAUSAL_SPEC, PREFIX_LM_SPEC, SLIDING_WINDOW_SPEC,
    COMPOSED_SPEC, BIDIRECTIONAL_SLIDING_WINDOW_SPEC, SPU_FWD_SPEC,
    FOUR_STAGE_SPEC, HSTU_FWD_SPEC, BATCHIFIED_FULL_SPEC,
    BATCHIFIED_CAUSAL_SPEC, BLOCK_DIFFUSION_SPEC,
    JETSPEC_TRAINING_SPEC, SVG_SPATIAL_SPEC,
)

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

TS = 128  # Q tile (matches kernel)


def flash_attention_ref_tnd(q_tnd, k_tnd, v_tnd, seq_q_list, seq_kv_list, d, masks,
                            torch_dtype=torch.float16, return_stats=False):
    """Reference attention operating on TND tensors, per-batch.

    ``masks[bi]`` is a bool tensor of shape [sq, skv] where True == masked
    (the cell is invisible and gets NEG_INF before softmax). It mirrors the
    on-device flash-attention compute flow: block-wide online softmax, P cast
    to the IO dtype before the P@V matmul, fp32 running denominator.

    When return_stats=True, additionally returns (softmax_max, softmax_sum):
      softmax_max [tq, n] fp32: per-Q global max(S), S = QK*scale + mask
      softmax_sum [tq, n] fp32: per-Q sum(exp(S - max_final))
    """
    q_tnd = q_tnd.cpu()
    k_tnd = k_tnd.cpu()
    v_tnd = v_tnd.cpu()
    seq_q_list = seq_q_list.cpu() if torch.is_tensor(seq_q_list) else seq_q_list
    seq_kv_list = seq_kv_list.cpu() if torch.is_tensor(seq_kv_list) else seq_kv_list
    scale_val = 1.0 / math.sqrt(d)
    n = q_tnd.shape[1]
    o_tnd = torch.zeros_like(q_tnd)
    max_tnd = torch.zeros((q_tnd.shape[0], n), dtype=torch.float32)
    sum_tnd = torch.zeros((q_tnd.shape[0], n), dtype=torch.float32)
    q_off = 0
    kv_off = 0
    for bi, sq_val in enumerate(seq_q_list):
        sq = int(sq_val)
        skv = int(seq_kv_list[bi])
        mask = masks[bi].cpu()
        for ni in range(n):
            qi = q_tnd[q_off:q_off + sq, ni, :].float()      # [sq, d]
            ki = k_tnd[kv_off:kv_off + skv, ni, :].float()   # [skv, d]
            vi = v_tnd[kv_off:kv_off + skv, ni, :].float()   # [skv, d]
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
                p16 = p.to(torch_dtype).float()                          # IO-dtype P (FA flow)
                acc = acc * corr + torch.matmul(p16, vi[c0:c1])
                running_l = running_l * corr + p.sum(dim=1, keepdim=True)
                running_m = new_m
            o_tnd[q_off:q_off + sq, ni, :] = (acc / running_l).to(torch_dtype)
            max_tnd[q_off:q_off + sq, ni] = running_m.squeeze(1)
            sum_tnd[q_off:q_off + sq, ni] = running_l.squeeze(1)
        q_off += sq
        kv_off += skv
    if return_stats:
        return o_tnd, max_tnd, sum_tnd
    return o_tnd


def _band_vis(seq_q_list, seq_kv_list, pre_tokens, next_tokens):
    """Per-batch [sq, skv] bool visibility (True == visible) for a band mask:
    visible(r, c) <=> L(r) <= c < R(r) with L = max(r-pre, 0), R = min(r+next+1, skv).
    """
    vis = []
    for bi, skv in enumerate(seq_kv_list):
        sq = seq_q_list[bi]
        m = torch.zeros((sq, skv), dtype=torch.bool)
        for r in range(sq):
            lo = max(r - pre_tokens, 0)
            hi = min(r + next_tokens + 1, skv)
            if hi > lo:
                m[r, lo:hi] = True
        vis.append(m)
    return vis


def _build_band_mask_info(seq_q_list, seq_kv_list, pre_tokens, next_tokens=0, mask_num=1):
    """Test-side mask_info generator (band formula descriptor + bitmask schedule).

    Returns (mask_info dict, dense_masks list).
    """
    b = len(seq_q_list)
    block_m = max((sq + TS - 1) // TS for sq in seq_q_list)
    col_block = max((skv + TKV - 1) // TKV for skv in seq_kv_list)
    word = (col_block + 31) // 32
    pad = block_m * TS

    vis = _band_vis(seq_q_list, seq_kv_list, pre_tokens, next_tokens)

    maskr = torch.zeros((b, pad), dtype=torch.int32)
    if mask_num > 0:
        mask_start = torch.zeros((b, 1, pad), dtype=torch.int32)
        mask_len = torch.zeros((b, 1, pad), dtype=torch.int32)
    else:
        mask_start = mask_len = None
    for bi, sq in enumerate(seq_q_list):
        skv = seq_kv_list[bi]
        for r in range(sq):
            lo = max(r - pre_tokens, 0)
            hi = min(r + next_tokens + 1, skv)
            # maskgen-aligned encoding: maskr = right bound; gap [0, lo) is masked.
            maskr[bi, r] = hi
            if mask_num > 0:
                mask_start[bi, 0, r] = 0
                mask_len[bi, 0, r] = lo

    block_compute = torch.zeros((b, block_m, word), dtype=torch.int64)
    block_mask = torch.zeros((b, block_m, word), dtype=torch.int64)
    block_range = torch.zeros((b, block_m), dtype=torch.int32)
    for bi, sq in enumerate(seq_q_list):
        skv = seq_kv_list[bi]
        for m in range((sq + TS - 1) // TS):
            r0, r1 = m * TS, min((m + 1) * TS, sq)
            max_blk = 0
            for kj in range(col_block):
                c0, c1 = kj * TKV, min((kj + 1) * TKV, skv)
                sub = vis[bi][r0:r1, c0:c1]
                if bool(sub.any()):
                    max_blk = kj
                    block_compute[bi, m, kj // 32] |= 1 << (kj % 32)
                    if not bool(sub.all()):
                        block_mask[bi, m, kj // 32] |= 1 << (kj % 32)
            block_range[bi, m] = max_blk + 1
    mask_info = {"maskr": maskr, "mask_num": mask_num,
                 "block_range": block_range.to(torch.int32),
                 "block_compute": block_compute,
                 "block_mask": block_mask.to(torch.uint32)}
    if mask_num > 0:
        mask_info.update({"mask_start": mask_start, "mask_len": mask_len})

    dense_masks = [~m for m in vis]  # True == masked (golden)
    return mask_info, dense_masks


# ================================================================
#  maskgen generation-logic simulations (torch ports)
#  Reference: maskgen/codegen/simd.py simulate_generated_program
#  + maskgen/reference/block_range.py compute_block_range
# ================================================================

def _maskgen_simulate(seq_q_list, seq_kv_list, row_intervals_fn, mask_num,
                      tile_m=TS, tile_n=TKV):
    """Port of maskgen simulate_generated_program (forward SIMD path).

    row_intervals_fn(bi, r) -> tuple[(start, end), ...]: visible half-open
    intervals of row r in batch bi (equivalent to maskgen evaluate_row_program
    output). Endpoint masks supply exact intervals here.

    Returns maskgen-native tensors (unpadded):
        maskr        [B, max(seq_q)] int32
        mask_start   [B, mask_num, max(seq_q)] int32 or None (mask_num == 0)
        mask_len     [B, mask_num, max(seq_q)] int32 or None
        block_range  [B, q_tiles] int32
        block_compute [B, q_tiles, k_words] uint32
        block_mask    [B, q_tiles, k_words] uint32
    """
    b = len(seq_q_list)
    max_seq_q = max(seq_q_list)
    max_seq_k = max(seq_kv_list)
    q_tiles = (max_seq_q + tile_m - 1) // tile_m
    k_tiles = (max_seq_k + tile_n - 1) // tile_n
    k_words = (k_tiles + 31) // 32

    maskr = torch.zeros((b, max_seq_q), dtype=torch.int32)
    mask_start = mask_len = None
    if mask_num > 0:
        mask_start = torch.zeros((b, mask_num, max_seq_q), dtype=torch.int32)
        mask_len = torch.zeros((b, mask_num, max_seq_q), dtype=torch.int32)

    # 1) per-row maskr + masked-gap packing (row step of simulate_generated_program)
    for bi in range(b):
        for r in range(seq_q_list[bi]):
            intervals = row_intervals_fn(bi, r)
            if not intervals:
                continue
            maskr[bi, r] = intervals[-1][1]
            previous_end = 0
            gap_idx = 0
            for start, end in intervals:
                if previous_end < start:
                    mask_start[bi, gap_idx, r] = previous_end
                    mask_len[bi, gap_idx, r] = start - previous_end
                    gap_idx += 1
                previous_end = end

    # 2) compressed-interval reconstruction (what decode_mask decodes) + block words
    def compressed_intervals(bi, r):
        visible = []
        visible_start = 0
        if mask_num > 0:
            for plane in range(mask_num):
                gap_len = int(mask_len[bi, plane, r])
                if gap_len <= 0:
                    continue
                gap_start = int(mask_start[bi, plane, r])
                if visible_start < gap_start:
                    visible.append((visible_start, gap_start))
                visible_start = gap_start + gap_len
        row_maskr = int(maskr[bi, r])
        if visible_start < row_maskr:
            visible.append((visible_start, row_maskr))
        return tuple(visible)

    block_compute = torch.zeros((b, q_tiles, k_words), dtype=torch.uint32)
    block_mask = torch.zeros((b, q_tiles, k_words), dtype=torch.uint32)
    for bi in range(b):
        sq = seq_q_list[bi]
        skv = seq_kv_list[bi]
        bi_q_tiles = (sq + tile_m - 1) // tile_m
        bi_k_tiles = (skv + tile_n - 1) // tile_n
        bi_k_words = (bi_k_tiles + 31) // 32
        for q_tile in range(bi_q_tiles):
            q_start = q_tile * tile_m
            q_end = min(q_start + tile_m, sq)
            for k_word in range(bi_k_words):
                compute_word = 0
                mask_word = 0
                for bit in range(32):
                    k_tile = k_word * 32 + bit
                    if k_tile >= bi_k_tiles:
                        continue
                    k_start = k_tile * tile_n
                    k_end = min(k_start + tile_n, skv)
                    any_visible = False
                    all_visible = True
                    for r in range(q_start, q_end):
                        intervals = compressed_intervals(bi, r)
                        row_overlaps = any(
                            start < k_end and k_start < end for start, end in intervals
                        )
                        row_covers = any(
                            start <= k_start and k_end <= end for start, end in intervals
                        )
                        any_visible = any_visible or row_overlaps
                        all_visible = all_visible and row_covers
                    if any_visible:
                        compute_word |= 1 << bit
                        if not all_visible:
                            mask_word |= 1 << bit
                block_compute[bi, q_tile, k_word] = compute_word & 0xFFFFFFFF
                block_mask[bi, q_tile, k_word] = mask_word & 0xFFFFFFFF

    # 3) block_range = per-Q-tile ceil(max(maskr) / tile_n)
    block_range = torch.zeros((b, q_tiles), dtype=torch.int32)
    for bi in range(b):
        sq = seq_q_list[bi]
        bi_q_tiles = (sq + tile_m - 1) // tile_m
        for q_tile in range(bi_q_tiles):
            q_start = q_tile * tile_m
            q_end = min(q_start + tile_m, sq)
            maximum = int(maskr[bi, q_start:q_end].max())
            block_range[bi, q_tile] = (maximum + tile_n - 1) // tile_n

    return maskr, mask_start, mask_len, block_range, block_compute, block_mask


def _maskgen_simulate_scan(seq_q_list, seq_kv_list, visible_predicate, mask_num,
                           tile_m=TS, tile_n=TKV):
    """maskgen predicate-scan path simulation: derive intervals by scanning columns.

    visible_predicate(bi, r, c) -> bool: whether row r may attend to column c.
    Mirrors the on-device scan (columns 0..seq_kv-1); output format matches
    _maskgen_simulate. Scan and endpoint paths share the same packing core.
    """
    def row_intervals_fn(bi, r):
        skv = seq_kv_list[bi]
        intervals = []
        start = None
        for c in range(skv):
            if visible_predicate(bi, r, c):
                if start is None:
                    start = c
            elif start is not None:
                intervals.append((start, c))
                start = None
        if start is not None:
            intervals.append((start, skv))
        return tuple(intervals)

    return _maskgen_simulate(seq_q_list, seq_kv_list, row_intervals_fn, mask_num,
                             tile_m=tile_m, tile_n=tile_n)


def _band_intervals(r, skv, pre_tokens, next_tokens):
    lo = max(r - pre_tokens, 0)
    hi = min(r + next_tokens + 1, skv)
    return () if lo >= hi else ((lo, hi),)


def _pad_maskgen_to_info(maskr, mask_start, mask_len, block_range, block_compute,
                         block_mask, mask_num):
    """Pad maskgen-native six outputs to the kernel's mask_info dict format.

    maskgen native: maskr [B, max_seq_q], mask_start/len [B, mask_num, max_seq_q],
                    block_range [B, q_tiles], block_compute/block_mask [B, q_tiles, k_words] uint32
    kernel format:  maskr [B, Pf], mask_start/len [B, mask_num, Pf],
                    block_range [B, Rf], block_compute [B, Rf, Wf] int64,
                    block_mask [B, Rf, Wf] uint32
    """
    b = maskr.shape[0]
    block_m = block_range.shape[1]
    word = block_compute.shape[2]
    pad = block_m * TS

    maskr_p = torch.zeros((b, pad), dtype=torch.int32)
    maskr_p[:, :maskr.shape[1]] = maskr
    if mask_num > 0:
        mask_start_p = torch.zeros((b, mask_num, pad), dtype=torch.int32)
        mask_len_p = torch.zeros((b, mask_num, pad), dtype=torch.int32)
        mask_start_p[:, :, :mask_start.shape[2]] = mask_start
        mask_len_p[:, :, :mask_len.shape[2]] = mask_len
    else:
        mask_start_p = mask_len_p = None

    block_range_p = torch.zeros((b, block_m), dtype=torch.int32)
    block_range_p[:, :block_range.shape[1]] = block_range
    block_compute_p = torch.zeros((b, block_m, word), dtype=torch.int64)
    block_compute_p[:, :block_compute.shape[1], :block_compute.shape[2]] = block_compute.long()
    block_mask_p = torch.zeros((b, block_m, word), dtype=torch.uint32)
    block_mask_p[:, :block_mask.shape[1], :block_mask.shape[2]] = block_mask

    mask_info = {"maskr": maskr_p, "mask_num": mask_num,
                 "block_range": block_range_p,
                 "block_compute": block_compute_p,
                 "block_mask": block_mask_p}
    if mask_num > 0:
        mask_info.update({"mask_start": mask_start_p, "mask_len": mask_len_p})
    return mask_info


def _build_band_mask_info_maskgen(seq_q_list, seq_kv_list, pre_tokens, next_tokens=0,
                                  mask_num=1):
    """_build_band_mask_info 的 maskgen 生成逻辑版（端点掩码路径）。

    Band 可见区间经 _maskgen_simulate 打包，输出 mask_info dict 格式与
    原 _build_band_mask_info 一致（含 dense_masks，供 golden 参考）。
    """
    maskr, mask_start, mask_len, block_range, block_compute, block_mask = _maskgen_simulate(
        seq_q_list, seq_kv_list,
        lambda bi, r: _band_intervals(r, seq_kv_list[bi], pre_tokens, next_tokens),
        mask_num)

    mask_info = _pad_maskgen_to_info(maskr, mask_start, mask_len, block_range, block_compute,
                                     block_mask, mask_num)

    dense_masks = [~m for m in _band_vis(seq_q_list, seq_kv_list, pre_tokens, next_tokens)]
    return mask_info, dense_masks


def _fixed_intervals_mask_info(seq_q_list, seq_kv_list, intervals, mask_num):
    """Build mask_info + dense_masks for a fixed multi-interval visibility pattern.

    intervals: tuple[(start, end), ...] visible half-open intervals, identical for
    every row (clamped to each batch's kv length). Internal gaps between consecutive
    intervals plus a leading gap [0, first_start) (when the first interval does not
    start at 0) fill the mask_num gap slots.
    """
    def row_intervals_fn(bi, r):
        skv = seq_kv_list[bi]
        out = []
        for s, e in intervals:
            e = min(e, skv)
            if s < e:
                out.append((s, e))
        return tuple(out)

    maskr, mask_start, mask_len, block_range, block_compute, block_mask = _maskgen_simulate(
        seq_q_list, seq_kv_list, row_intervals_fn, mask_num)
    mask_info = _pad_maskgen_to_info(maskr, mask_start, mask_len, block_range, block_compute,
                                     block_mask, mask_num)

    dense_masks = []
    for bi, skv in enumerate(seq_kv_list):
        sq = seq_q_list[bi]
        m = torch.zeros((sq, skv), dtype=torch.bool)
        for s, e in intervals:
            m[:, s:min(e, skv)] = True
        dense_masks.append(~m)
    return mask_info, dense_masks


def _build_spec_mask_info(seq_q_list, seq_kv_list, spec, mask_num, **tensors):
    """MaskSpec → mask_info + dense_masks, 通过扫描 visible 谓词模拟。

    tensors: CPU tensor，key 对应 spec.tensor_args 的 name（如 seq_lens=...）。
    mask_num 必须 ≥ 该 pattern 的实际最大间隙数。
    """
    b = len(seq_q_list)
    scalars_by_batch = []
    for bi in range(b):
        targs = {arg.name: tensors[arg.name] for arg in spec.tensor_args}
        scalars = spec.scalar_setup(bi, **targs)
        scalars_by_batch.append(scalars)

    def visible_predicate(bi, r, c):
        for region in spec.regions:
            if region.when(r, **scalars_by_batch[bi]):
                return region.visible(r, c, **scalars_by_batch[bi])
        return False

    maskr, mask_start, mask_len, block_range, bc, bm = _maskgen_simulate_scan(
        seq_q_list, seq_kv_list, visible_predicate, mask_num)
    mask_info = _pad_maskgen_to_info(maskr, mask_start, mask_len, block_range, bc, bm, mask_num)

    dense_masks = []
    for bi, skv in enumerate(seq_kv_list):
        sq = seq_q_list[bi]
        m = torch.zeros((sq, skv), dtype=torch.bool)
        for r in range(sq):
            for c in range(skv):
                m[r, c] = visible_predicate(bi, r, c)
        dense_masks.append(~m)
    return mask_info, dense_masks


def _reconstruct_intervals_from_mask_info(maskr, mask_start, mask_len, mask_num, bi, r):
    """Reconstruct a row's visible intervals from maskr + gaps (maskgen
    compressed_row_intervals logic, i.e. what decode_mask decodes)."""
    visible = []
    visible_start = 0
    if mask_num > 0:
        for plane in range(mask_num):
            gap_len = int(mask_len[bi, plane, r])
            if gap_len <= 0:
                continue
            gap_start = int(mask_start[bi, plane, r])
            if visible_start < gap_start:
                visible.append((visible_start, gap_start))
            visible_start = gap_start + gap_len
    row_maskr = int(maskr[bi, r])
    if visible_start < row_maskr:
        visible.append((visible_start, row_maskr))
    return tuple(visible)


def _assert_mask_info_equal(mi_a, mi_b):
    assert set(mi_a) == set(mi_b), f"mask_info keys differ: {set(mi_a)} vs {set(mi_b)}"
    for key in mi_a:
        if isinstance(mi_a[key], int):
            assert mi_a[key] == mi_b[key], f"mask_info['{key}'] differs"
        else:
            torch.testing.assert_close(mi_a[key], mi_b[key], rtol=0, atol=0)


def _flex_attention_from_info(q, k, v, seq_q_list, seq_kv_list, attn_out, softmax_max, softmax_sum, mask_info):
    """Unpack a mask_info dict and call the expanded flex_attention() interface."""
    device = q.device
    def _to_dev(x):
        return x.to(device) if x is not None else None
    actual_seq_q = torch.tensor(seq_q_list, dtype=torch.int64, device=device).cumsum(0)
    actual_seq_kv = torch.tensor(seq_kv_list, dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
    flex_attention(q, k, v, actual_seq_q, actual_seq_kv,
                   _to_dev(mask_info["maskr"]),
                   _to_dev(mask_info.get("mask_start")),
                   _to_dev(mask_info.get("mask_len")),
                   _to_dev(mask_info["block_range"]),
                   _to_dev(mask_info["block_compute"]),
                   _to_dev(mask_info["block_mask"]),
                   attn_out, softmax_max, softmax_sum)


def test_maskgen_sim_parity():
    """CPU-only: verify that original, endpoint-sim, and scan-sim produce identical mask_info."""
    configs = [
        ([512], [512], 100000, 0, 1),
        ([512], [512], 127, 0, 1),
        ([200], [200], 100000, 0, 1),
        ([160], [160], 100000, 0, 1),
        ([128, 256, 200], [128, 256, 200], 100000, 0, 1),
        ([384, 100], [384, 100], 255, 0, 1),
        ([128], [128], 100000, 0, 0),  # mask_num=0
    ]
    for seq_q_list, seq_kv_list, pre, nxt, mask_num in configs:
        # original
        mi_orig, _ = _build_band_mask_info(seq_q_list, seq_kv_list, pre, nxt, mask_num=mask_num)
        # endpoint-sim
        mi_sim, _ = _build_band_mask_info_maskgen(seq_q_list, seq_kv_list, pre, nxt, mask_num=mask_num)
        _assert_mask_info_equal(mi_orig, mi_sim)

        # scan-sim (via _band_vis)
        vis = _band_vis(seq_q_list, seq_kv_list, pre, nxt)
        def pred(bi, r, c):
            return bool(vis[bi][r, c])
        maskr, mask_start, mask_len, block_range, bc, bm = _maskgen_simulate_scan(
            seq_q_list, seq_kv_list, pred, mask_num)
        # pad and compare
        b = len(seq_q_list)
        block_m = max((sq + TS - 1) // TS for sq in seq_q_list)
        col_block = max((skv + TKV - 1) // TKV for skv in seq_kv_list)
        word = (col_block + 31) // 32
        pad = block_m * TS
        maskr_p = torch.zeros((b, pad), dtype=torch.int32)
        maskr_p[:, :maskr.shape[1]] = maskr
        torch.testing.assert_close(maskr_p, mi_orig["maskr"], rtol=0, atol=0)
        if mask_num > 0:
            mask_start_p = torch.zeros((b, mask_num, pad), dtype=torch.int32)
            mask_len_p = torch.zeros((b, mask_num, pad), dtype=torch.int32)
            mask_start_p[:, :, :mask_start.shape[2]] = mask_start
            mask_len_p[:, :, :mask_len.shape[2]] = mask_len
            torch.testing.assert_close(mask_start_p, mi_orig["mask_start"], rtol=0, atol=0)
            torch.testing.assert_close(mask_len_p, mi_orig["mask_len"], rtol=0, atol=0)
        block_range_p = torch.zeros((b, block_m), dtype=torch.int32)
        block_range_p[:, :block_range.shape[1]] = block_range
        torch.testing.assert_close(block_range_p, mi_orig["block_range"], rtol=0, atol=0)
        block_compute_p = torch.zeros((b, block_m, word), dtype=torch.int64)
        block_compute_p[:, :bc.shape[1], :bc.shape[2]] = bc.long()
        torch.testing.assert_close(block_compute_p, mi_orig["block_compute"], rtol=0, atol=0)
        block_mask_p = torch.zeros((b, block_m, word), dtype=torch.uint32)
        block_mask_p[:, :bm.shape[1], :bm.shape[2]] = bm
        torch.testing.assert_close(block_mask_p, mi_orig["block_mask"], rtol=0, atol=0)

    logging.info("  [PASS] maskgen sim parity (endpoint ↔ scan ↔ original)")


# Each cfg: (seq_q_list, seq_kv_list, n, d, pre_tokens, atol); next_tokens = 0 (causal band).
# The three softmax branch widths (s2 == 128 / <= 64 / 64 < s2 < 128) are selected by the
# KV-tile width of the last block, covered by the tail cases below.
# Head-dim rows: d=64 (d_loops=1, tail=0), d=96 (d_loops=1 + tail=32),
# d=32 (d_loops=0, tail-only path), d=100 (non-16-aligned probe: GM stride +
# matmul K-dim partial fragment; if it fails the assert tightens to d % 16 == 0).
F16_CFGS = [
    ([512], [512], 1, TD, 100000, 1e-3),  # pure causal, aligned s2=128 path
    ([512], [512], 1, TD, 127, 1e-3),     # sliding window -> leading-block skip (hole > 0)
    ([512], [512], 1, TD, 255, 1e-3),     # multi-block window
    ([200], [200], 1, TD, 100000, 1e-3),  # q-tail(72) + kv tail s2=72 -> unalign path
    ([160], [160], 1, TD, 100000, 1e-3),  # kv tail s2=32 -> unalign64 path
    ([256], [256], 2, TD, 100000, 1e-3),  # multi-head n=2
    ([384, 100], [384, 100], 2, TD, 255, 1e-3),  # varlen + multi-head + window (sq > skv)
    ([512], [512], 1, 64, 100000, 1e-3),  # head-dim 64: d_loops=1, tail=0
    ([512], [512], 1, 96, 100000, 1e-3),  # head-dim 96: d_loops=1 + tail=32
    ([200], [200], 1, 32, 127, 1e-3),     # head-dim 32: d_loops=0, tail-only + varlen tail
]
BF16_CFGS = [
    ([512], [512], 1, TD, 100000, 1e-2),
    ([512], [512], 1, TD, 127, 1e-2),
    ([200], [200], 1, TD, 100000, 1e-2),
    ([512], [512], 2, 64, 100000, 1e-2),  # multi-head head-dim 64
    ([512], [512], 1, 100, 100000, 1e-2),  # non-16-aligned head-dim probe
]


def run_fa_cfgs(test_cfgs, torch_dtype, tag):
    failures = []
    for test_cfg in test_cfgs:
        seq_q_list, seq_kv_list, n, d, pre, atol = test_cfg
        b = len(seq_q_list)
        tq = sum(seq_q_list)
        tkv = sum(seq_kv_list)
        logging.info("\nFA-MASKINFO %s (b=%s, seq_q=%s, seq_kv=%s, n=%s, d=%s) pre=%s cores=platform",
                     tag, b, seq_q_list, seq_kv_list, n, d, pre)

        q = torch.rand((tq, n, d), dtype=torch_dtype).to(ST_DEVICE)
        k = torch.rand((tkv, n, d), dtype=torch_dtype).to(ST_DEVICE)
        v = torch.rand((tkv, n, d), dtype=torch_dtype).to(ST_DEVICE)
        o = torch.zeros((tq, n, d), dtype=torch_dtype, device=ST_DEVICE)
        smax = torch.zeros((n, tq), dtype=torch.float32, device=ST_DEVICE)
        ssum = torch.zeros((n, tq), dtype=torch.float32, device=ST_DEVICE)

        mask_info, dense_masks = _build_band_mask_info_maskgen(
            seq_q_list, seq_kv_list, pre, 0)
        _flex_attention_from_info(q, k, v, seq_q_list, seq_kv_list, o, smax, ssum, mask_info)

        o_ref = flash_attention_ref_tnd(
            q, k, v, seq_q_list, seq_kv_list, d, dense_masks, torch_dtype)
        o = o.cpu()
        diff = (o - o_ref).abs().max().item()
        out_is_zero = bool(o.abs().max().item() == 0.0)
        status = "PASS"
        try:
            assert not out_is_zero, "kernel output is all-zero (o not written back to host)"
            torch.testing.assert_close(o, o_ref, rtol=atol, atol=atol)
        except AssertionError as e:
            status = "FAIL"
            failures.append((test_cfg, diff, str(e).splitlines()[0]))
        logging.info("  [%s] %s b=%d seq_q=%s seq_kv=%s n=%d atol=%.0e max|diff|=%.5f",
                     status, tag, b, seq_q_list, seq_kv_list, n, atol, diff)

    if failures:
        msg = "\n".join("  cfg=%s max|diff|=%.5f : %s" % (c, df, m) for c, df, m in failures)
        raise AssertionError("%d/%d precision cases failed:\n%s" % (len(failures), len(test_cfgs), msg))
    logging.info("  PASS (%s)", tag)


@pytest.mark.soc("950")
def test_fa_mask_info():
    torch.npu.set_device(ST_DEVICE)
    torch.manual_seed(42)
    run_fa_cfgs(F16_CFGS, torch.float16, "f16")
    run_fa_cfgs(BF16_CFGS, torch.bfloat16, "bf16")


@pytest.mark.soc("950")
def test_fa_mask_info_none_desc():
    """mask_num == 0: mask_start/mask_len keys omitted -> kernel receives None."""
    torch.npu.set_device(ST_DEVICE)
    torch.manual_seed(7)
    seq_q_list, seq_kv_list, n, d, pre, atol = [128], [128], 1, TD, 100000, 1e-3
    q = torch.rand((128, n, d), dtype=torch.float16).to(ST_DEVICE)
    k = torch.rand((128, n, d), dtype=torch.float16).to(ST_DEVICE)
    v = torch.rand((128, n, d), dtype=torch.float16).to(ST_DEVICE)
    o = torch.zeros((128, n, d), dtype=torch.float16, device=ST_DEVICE)
    smax = torch.zeros((n, 128), dtype=torch.float32, device=ST_DEVICE)
    ssum = torch.zeros((n, 128), dtype=torch.float32, device=ST_DEVICE)

    mi, masks = _build_band_mask_info_maskgen(seq_q_list, seq_kv_list, pre, 0, mask_num=0)
    assert "mask_start" not in mi and "mask_len" not in mi and mi["mask_num"] == 0
    _flex_attention_from_info(q, k, v, seq_q_list, seq_kv_list, o, smax, ssum, mi)
    o_ref = flash_attention_ref_tnd(q, k, v, seq_q_list, seq_kv_list, d, masks, torch.float16)
    torch.testing.assert_close(o.cpu(), o_ref, rtol=atol, atol=atol)
    logging.info("  [PASS] none-descriptor (mask_num=0, mask_start/mask_len=None)")


@pytest.mark.soc("950")
def test_fa_mask_info_required_validation():
    """Contract Required params must be present; invalid values must raise."""
    torch.npu.set_device(ST_DEVICE)
    q = torch.rand((128, 1, TD), dtype=torch.float16).to(ST_DEVICE)
    k = torch.rand((128, 1, TD), dtype=torch.float16).to(ST_DEVICE)
    v = torch.rand((128, 1, TD), dtype=torch.float16).to(ST_DEVICE)

    mi, _ = _build_band_mask_info_maskgen([128], [128], 100000, 0)
    oval = torch.zeros((128, 1, TD), dtype=torch.float16, device=ST_DEVICE)
    smax_val = torch.zeros((1, 128), dtype=torch.float32, device=ST_DEVICE)
    ssum_val = torch.zeros((1, 128), dtype=torch.float32, device=ST_DEVICE)

    def _expect_raise(**kwargs):
        bad = dict(mi)
        bad.update(kwargs)
        with pytest.raises((AssertionError, ValueError)):
            _flex_attention_from_info(q, k, v, [128], [128], oval, smax_val, ssum_val, bad)

    # mask_num is derived from mask_start.shape[1] (0 when mask_start is
    # absent); no explicit mask_num argument exists any more.
    _expect_raise(maskr=None)                                              # maskr is Required
    _expect_raise(mask_start=mi["mask_start"].reshape(-1))                 # mask_start must be rank-3
    _expect_raise(mask_len=torch.zeros((1, 1, 64), dtype=torch.int32))     # mask_len shape must match mask_start
    _expect_raise(mask_len=None)                                           # mask_len required when mask_start present
    oversized = torch.zeros((1, MASK_NUM_LIMIT + 1, mi["mask_start"].shape[2]),
                            dtype=torch.int32)                             # derived mask_num > sanity limit
    _expect_raise(mask_start=oversized, mask_len=oversized)
    _expect_raise(block_range=None)                                        # block_range is Required
    _expect_raise(block_compute=None)                                       # block_compute is Required
    _expect_raise(block_mask=None)                                          # block_mask is Required
    logging.info("  [PASS] required-key validation")


@pytest.mark.soc("950")
def test_fa_mask_info_multi_hole():
    """mask_num=2/3/18 多间隙 mask：maskgen 模拟生成 → flex_attention → golden 对比。

    mask_num=18 > CHUNK=16 走 HoleChunk=1 分块解码（base 16 槽 + refine 2 槽尾块）。"""
    torch.npu.set_device(ST_DEVICE)
    torch.manual_seed(7)

    seq_q_list = [128]
    seq_kv_list = [128]
    n, d = 1, TD
    tq = sum(seq_q_list)
    tkv = sum(seq_kv_list)
    q = torch.rand((tq, n, d), dtype=torch.float16).to(ST_DEVICE)
    k = torch.rand((tkv, n, d), dtype=torch.float16).to(ST_DEVICE)
    v = torch.rand((tkv, n, d), dtype=torch.float16).to(ST_DEVICE)
    o = torch.zeros((tq, n, d), dtype=torch.float16, device=ST_DEVICE)
    smax = torch.zeros((n, tq), dtype=torch.float32, device=ST_DEVICE)
    ssum = torch.zeros((n, tq), dtype=torch.float32, device=ST_DEVICE)

    cases = [
        (2, ((3, 5), (8, 10))),                     # leading [0,3) + internal [5,8) = 2 gaps
        (2, ((0, 2), (5, 8), (11, 14))),            # 2 internal gaps [2,5),[8,11)
        (3, ((0, 2), (5, 8), (11, 14), (17, 20))),  # 3 internal gaps
        (3, ((3, 5), (8, 10), (13, 15))),           # leading + 2 internal = 3 gaps
        # 18 gaps (leading [0,2) + 17 unit gaps): HoleChunk=1, 2 chunks (16 + 2)
        (18, tuple((2 + 2 * i, 3 + 2 * i) for i in range(18))),
    ]

    for mask_num, intervals in cases:
        # CPU: 验证二包重建一致（maskr、间隙打包、压缩重建 == 原区间）
        expected_gaps = []
        prev_end = 0
        for s, e in intervals:
            if prev_end < s:
                expected_gaps.append((prev_end, s))
            prev_end = e
        assert len(expected_gaps) <= mask_num, \
            f"gap count {len(expected_gaps)} exceeds mask_num={mask_num}"

        maskr, mask_start, mask_len, block_range, bc, bm = _maskgen_simulate(
            seq_q_list, seq_kv_list, lambda bi, r: tuple(
                (s, min(e, seq_kv_list[bi])) for s, e in intervals), mask_num)
        assert int(maskr[0, 0]) == intervals[-1][1], "maskr != last interval end"
        for plane, (gs, ge) in enumerate(expected_gaps):
            assert int(mask_start[0, plane, 0]) == gs, \
                f"mask_start[{plane}] wrong: {int(mask_start[0, plane, 0])} != {gs}"
            assert int(mask_len[0, plane, 0]) == ge - gs, \
                f"mask_len[{plane}] wrong: {int(mask_len[0, plane, 0])} != {ge - gs}"
        # 重建校验
        recon = _reconstruct_intervals_from_mask_info(maskr, mask_start, mask_len, mask_num, 0, 0)
        clamped = tuple((s, min(e, seq_kv_list[0])) for s, e in intervals)
        assert recon == clamped, f"reconstruction mismatch: {recon} != {clamped}"

        # NPU: flex_attention
        mi, dense_masks = _fixed_intervals_mask_info(seq_q_list, seq_kv_list, intervals, mask_num)
        _flex_attention_from_info(q, k, v, seq_q_list, seq_kv_list, o, smax, ssum, mi)
        o_ref = flash_attention_ref_tnd(q, k, v, seq_q_list, seq_kv_list, d, dense_masks,
                                        torch.float16)
        torch.testing.assert_close(o.cpu(), o_ref, rtol=1e-3, atol=1e-3)
        logging.info("  [PASS] multi-hole mask_num=%d intervals=%s gaps=%s",
                     mask_num, intervals, expected_gaps)


@pytest.mark.soc("950")
def test_fa_mask_patterns():
    """maskgen 各 pattern 的 visible 谓词 → 模拟生成 mask_info → flex_attention 验证。"""
    torch.npu.set_device(ST_DEVICE)
    torch.manual_seed(7)

    PATTERN_CFGS = [
        ("causal", CAUSAL_SPEC, 1, [128], [128], {"seq_lens": [128]}, 1e-3),
        ("full", FULL_SPEC, 0, [128], [128], {"seq_lens": [128]}, 1e-3),
        ("prefix_lm", PREFIX_LM_SPEC, 1, [128], [128],
         {"seq_lens": [128], "prefix_lengths": [32]}, 1e-3),
        ("sliding_window", SLIDING_WINDOW_SPEC, 1, [128], [128],
         {"seq_lens": [128], "window_sizes": [64]}, 1e-3),
        ("bidirectional_sw", BIDIRECTIONAL_SLIDING_WINDOW_SPEC, 1, [128], [128],
         {"seq_lens": [128], "window_sizes": [32]}, 1e-3),
        ("composed", COMPOSED_SPEC, 2, [128], [128], {"seq_lens": [128]}, 1e-3),
        ("spu", SPU_FWD_SPEC, 2, [256], [256],
         {"context_length": [64], "labelgroup_size": [4],
          "cu_seqlens": [0, 256]}, 1e-3),
        ("four_stage", FOUR_STAGE_SPEC, 2, [128], [128],
         {"context_sizes": [32], "history_sizes": [32], "realtime_sizes": [32],
          "label_nums": [8], "label_group_sizes": [4]}, 1e-3),
        ("hstu", HSTU_FWD_SPEC, 2, [256], [256],
         {"context_length": [32], "history_length": [32], "label_group_size": [4],
          "cu_seqlen": [0, 256]}, 1e-3),
        ("batchified_causal", BATCHIFIED_CAUSAL_SPEC, 1, [128], [128],
         {"seq_lens": [128], "group_sizes": [16]}, 1e-3),
        ("batchified_full", BATCHIFIED_FULL_SPEC, 1, [128], [128],
         {"seq_lens": [128], "group_sizes": [16]}, 1e-3),
        ("block_diffusion", BLOCK_DIFFUSION_SPEC, 2, [256], [256],
         {"base_seq_lens": [128], "block_sizes": [32]}, 1e-3),
        ("jetspec", JETSPEC_TRAINING_SPEC, 1, [128], [128],
         {"prefix_lengths": [64], "sampled_lengths": [64], "block_sizes": [8]}, 1e-3),
        ("svg", SVG_SPATIAL_SPEC, 2, [256], [256],
         {"seq_lens": [256], "prompt_lengths": [32], "token_per_frames": [4],
          "frame_widths": [16], "round_widths": [8], "attn_sink_flags": [1]}, 1e-3),
    ]

    n, d = 1, TD
    for name, spec, mask_num, sq_list, skv_list, tensors, atol in PATTERN_CFGS:
        tq, tkv = sum(sq_list), sum(skv_list)
        q = torch.rand((tq, n, d), dtype=torch.float16).to(ST_DEVICE)
        k = torch.rand((tkv, n, d), dtype=torch.float16).to(ST_DEVICE)
        v = torch.rand((tkv, n, d), dtype=torch.float16).to(ST_DEVICE)
        o = torch.zeros((tq, n, d), dtype=torch.float16, device=ST_DEVICE)
        smax = torch.zeros((n, tq), dtype=torch.float32, device=ST_DEVICE)
        ssum = torch.zeros((n, tq), dtype=torch.float32, device=ST_DEVICE)
        tensor_inputs = {k: torch.tensor(v, dtype=torch.int32, device=ST_DEVICE)
                        for k, v in tensors.items()}
        mi, masks = _build_spec_mask_info(sq_list, skv_list, spec, mask_num, **tensor_inputs)
        _flex_attention_from_info(q, k, v, sq_list, skv_list, o, smax, ssum, mi)
        o_ref = flash_attention_ref_tnd(
            q, k, v, sq_list, skv_list, d, masks, torch.float16)
        torch.testing.assert_close(o.cpu(), o_ref, rtol=atol, atol=atol)
        logging.info("  [PASS] pattern=%-20s mask_num=%d",
                     name, mask_num)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.npu.set_device(ST_DEVICE)
    logging.info("FlexAttention mask_info (QK_PRELOAD=%s, FIFO=%s, decode CHUNK=%s, mask_num sanity=%s)",
                 QK_PRELOAD, FIFO_SIZE, CHUNK, MASK_NUM_LIMIT)
    logging.info("%s", '=' * 60)
    test_maskgen_sim_parity()
    test_fa_mask_info()
    test_fa_mask_info_none_desc()
    test_fa_mask_info_multi_hole()
    test_fa_mask_patterns()
    test_fa_mask_info_required_validation()
    logging.info("\nAll tests passed!")
