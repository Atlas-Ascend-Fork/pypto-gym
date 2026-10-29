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
"""FlexAttention golden/reference implementations (测试专用, 不随 whl/run 交付).

包含:
  - flash_attention_ref_tnd: TND 参考实现(含可选 softmax 统计量)
  - band/spec 掩码的 mask_info 构造与 maskgen 逻辑模拟
  - mask_info 一致性/区间重建等 golden 校验 helper
"""

import math

import torch

from flex_attention_kernel_consts import NEG_INF, TKV, TS


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


