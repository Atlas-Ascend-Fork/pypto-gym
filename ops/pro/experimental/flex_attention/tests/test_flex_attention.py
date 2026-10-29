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
    pytest ops/pro/experimental/flex_attention/tests/test_flex_attention.py --mode jit
    pytest ops/pro/experimental/flex_attention/tests/test_flex_attention.py --mode aclnn
"""

import logging
import os

import pytest
import torch

from op_caller import get_mode, get_op_caller
from flex_attention_kernel_consts import MASK_NUM_LIMIT, TKV, TD, TS
from flex_attention_mask_specs import (
    FULL_SPEC, CAUSAL_SPEC, PREFIX_LM_SPEC, SLIDING_WINDOW_SPEC,
    COMPOSED_SPEC, BIDIRECTIONAL_SLIDING_WINDOW_SPEC, SPU_FWD_SPEC,
    FOUR_STAGE_SPEC, HSTU_FWD_SPEC, BATCHIFIED_FULL_SPEC,
    BATCHIFIED_CAUSAL_SPEC, BLOCK_DIFFUSION_SPEC,
    JETSPEC_TRAINING_SPEC, SVG_SPATIAL_SPEC,
)
from flex_attention_golden import (
    flash_attention_ref_tnd, _band_vis, _build_band_mask_info,
    _maskgen_simulate, _maskgen_simulate_scan, _build_band_mask_info_maskgen,
    _fixed_intervals_mask_info, _build_spec_mask_info,
    _reconstruct_intervals_from_mask_info, _assert_mask_info_equal,
)
from flex_attention_call import flex_attention_from_info

flex_attention = get_op_caller("flex_attention")

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _flex_attention_from_info(q, k, v, seq_q_list, seq_kv_list, attn_out,
                              softmax_max, softmax_sum, mask_info):
    """Unpack a mask_info dict and call the current-mode flex_attention()."""
    flex_attention_from_info(flex_attention, q, k, v, seq_q_list, seq_kv_list,
                             attn_out, softmax_max, softmax_sum, mask_info)


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
def test_fa_mask_info_skewed_varlen_work_ranges():
    """Skewed varlen [1, 255]: every work_id must be launched.

    Regression for the tiling fallback that derived total_work from an even
    tq/batch split: it under-estimated [1, 255] (1+1 vs 1+2 tiles) and silently
    dropped the last Q tile, leaving rows 129..255 unwritten.
    """
    torch.npu.set_device(ST_DEVICE)
    torch.manual_seed(1)
    seq_q_list, seq_kv_list = [1, 255], [1, 255]
    n, d, pre = 1, TD, 100000
    tq, tkv = sum(seq_q_list), sum(seq_kv_list)
    q = torch.rand((tq, n, d), dtype=torch.float16).to(ST_DEVICE)
    k = torch.rand((tkv, n, d), dtype=torch.float16).to(ST_DEVICE)
    v = torch.rand((tkv, n, d), dtype=torch.float16).to(ST_DEVICE)
    o = torch.zeros((tq, n, d), dtype=torch.float16, device=ST_DEVICE)
    smax = torch.zeros((n, tq), dtype=torch.float32, device=ST_DEVICE)
    ssum = torch.zeros_like(smax)

    mi, _ = _build_band_mask_info_maskgen(seq_q_list, seq_kv_list, pre, 0)
    _flex_attention_from_info(q, k, v, seq_q_list, seq_kv_list, o, smax, ssum, mi)
    zero_rows = (o.cpu().abs().amax(dim=(1, 2)) == 0).nonzero().flatten().tolist()
    assert not zero_rows, f"work ids dropped: {len(zero_rows)} rows unwritten, first={zero_rows[:8]}"


@pytest.mark.soc("950")
def test_fa_mask_info_bshd_layout():
    """bshd layout [B, S, H, D] -> delivery-form output [B*S, H, D]; parity with varlen."""
    torch.npu.set_device(ST_DEVICE)
    torch.manual_seed(9)
    b, s1, s2, n, d = 2, 128, 128, 1, TD
    q4 = torch.rand((b, s1, n, d), dtype=torch.float16).to(ST_DEVICE)
    k4 = torch.rand((b, s2, n, d), dtype=torch.float16).to(ST_DEVICE)
    v4 = torch.rand((b, s2, n, d), dtype=torch.float16).to(ST_DEVICE)
    tq, tkv = b * s1, b * s2
    o = torch.zeros((tq, n, d), dtype=torch.float16, device=ST_DEVICE)
    smax = torch.zeros((n, tq), dtype=torch.float32, device=ST_DEVICE)
    ssum = torch.zeros_like(smax)

    mi, masks = _build_band_mask_info_maskgen([s1] * b, [s2] * b, 100000, 0)
    _to_dev = lambda x: x.to(ST_DEVICE) if x is not None else None  # noqa: E731
    flex_attention(q4, k4, v4, None, None,
                   _to_dev(mi["maskr"]),
                   _to_dev(mi.get("mask_start")),
                   _to_dev(mi.get("mask_len")),
                   _to_dev(mi["block_range"]),
                   _to_dev(mi["block_compute"]),
                   _to_dev(mi["block_mask"]),
                   o, smax, ssum, layout="bshd")
    o_ref = flash_attention_ref_tnd(q4.reshape(tq, n, d), k4.reshape(tkv, n, d),
                                    v4.reshape(tkv, n, d), [s1] * b, [s2] * b, d, masks,
                                    torch.float16)
    torch.testing.assert_close(o.cpu(), o_ref, rtol=1e-3, atol=1e-3)
    logging.info("  [PASS] bshd layout (B=%d, S=%d)", b, s1)


@pytest.mark.soc("950")
@pytest.mark.skipif(get_mode() == "aclnn",
                    reason="Python host contract validation; built-in path validates via tiling/framework")
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
