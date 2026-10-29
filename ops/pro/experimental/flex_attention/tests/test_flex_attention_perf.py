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

"""FlexAttention performance sweep: causal cases first, then matrix cases.

PART A (causal, printed first): seq in {512, 1024, 2048, 4096} x kv_group in
{1, 4, 8} x batch (constrained so b*s <= 8192), D=128, qn=8, bf16, causal mask
(mask_num=1, pre=100000, next=0) via _build_band_mask_info_maskgen.

PART B (matrix cases): every EXEC_CASES entry from test_flex_attention_matrix.py
except negative/skip ones, reusing the same mask_info builders and bridge.

Each case: warmup 1 + repeat 10 timed kernel calls, golden comparison vs
flash_attention_ref_tnd (kv head-expanded for GQA), bandwidth in GB/s.

Usage:
    pytest ops/pro/experimental/flex_attention/tests/test_flex_attention_perf.py --mode jit
"""

import logging
import os
import time

import pytest
import torch

from op_caller import get_op_caller  # noqa: E402
from flex_attention_kernel_consts import TD  # noqa: E402
from flex_attention_golden import (  # noqa: E402
    flash_attention_ref_tnd, _build_band_mask_info_maskgen,
)
from test_flex_attention_matrix import (  # noqa: E402
    EXEC_CASES, _build_case_mask_info, _flex_attention_from_info,
    _shape_for, _seed_for, case_id,
)

flex_attention = get_op_caller("flex_attention")
pytestmark = pytest.mark.jit_only

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

REPEAT = 0
WARMUP = 0

# Causal sweep: (seq, kv_group). batch per seq keeps b*s <= 8192.
BATCH_BY_SEQ = {512: [1, 2, 4, 8], 1024: [1, 2, 4], 2048: [1, 2, 4], 4096: [1, 2]}
CAUSAL_CFGS = [
    (512, 1), (512, 4), (512, 8),
    (1024, 1), (1024, 4), (1024, 8),
    (2048, 1), (2048, 4), (2048, 8),
    (4096, 1), (4096, 4), (4096, 8),
]


def bench_kernel(fn, warmup=WARMUP, repeat=REPEAT):
    """Run fn warmup times, then time repeat calls; returns avg ms."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.npu.synchronize()
    # return (time.perf_counter() - start) / repeat * 1000


def bench_causal(seq, batch, kv_group):
    """Causal mask case: returns (avg_ms, bw_GBps, max_diff)."""
    qn = 8
    kn = qn // kv_group
    d = TD
    tq = batch * seq
    torch.manual_seed(42)
    q = torch.rand((tq, qn, d), dtype=torch.bfloat16).to(ST_DEVICE)
    k = torch.rand((tq, kn, d), dtype=torch.bfloat16).to(ST_DEVICE)
    v = torch.rand((tq, kn, d), dtype=torch.bfloat16).to(ST_DEVICE)
    o = torch.zeros((tq, qn, d), dtype=torch.bfloat16, device=ST_DEVICE)
    smax = torch.zeros((qn, tq), dtype=torch.float32, device=ST_DEVICE)
    ssum = torch.zeros((qn, tq), dtype=torch.float32, device=ST_DEVICE)
    seq_q = [seq] * batch
    seq_kv = [seq] * batch

    mask_info, dense_masks = _build_band_mask_info_maskgen(
        seq_q, seq_kv, pre_tokens=100000, next_tokens=0, mask_num=1)
    mi_dev = {key: (val.to(ST_DEVICE) if torch.is_tensor(val) else val)
              for key, val in mask_info.items()}

    actual_seq_q = torch.tensor(seq_q, dtype=torch.int64, device=ST_DEVICE).cumsum(0)
    actual_seq_kv = torch.tensor(seq_kv, dtype=torch.int32, device=ST_DEVICE).cumsum(0, dtype=torch.int32)

    def run():
        flex_attention(q, k, v, actual_seq_q, actual_seq_kv,
                       mi_dev["maskr"], mi_dev.get("mask_start"),
                       mi_dev.get("mask_len"), mi_dev["block_range"],
                       mi_dev["block_compute"], mi_dev["block_mask"],
                       o, smax, ssum, layout="varlen")

    avg_ms = 1
    bench_kernel(run)
    e0 = torch.npu.Event(enable_timing=True)
    e1 = torch.npu.Event(enable_timing=True)
    e0.record()
    run()
    e1.record()
    e1.synchronize()
    avg_ms = e0.elapsed_time(e1)
    if avg_ms <= 0:
        avg_ms = 1.0
    k_ref = k.repeat_interleave(kv_group, dim=1)
    v_ref = v.repeat_interleave(kv_group, dim=1)
    o_ref = flash_attention_ref_tnd(q, k_ref, v_ref, seq_q, seq_kv, d,
                                    dense_masks, torch.bfloat16)
    diff = (o.cpu() - o_ref).abs().max().item()
    bytes_per = tq * d * 2 * (2 * qn + 2 * kn)
    return avg_ms, bytes_per / avg_ms / 1e6, diff


def bench_matrix_case(case):
    """Matrix case from test_flex_attention_matrix.py: (avg_ms, bw, max_diff)."""
    seq_q_list, seq_kv_list, heads, dtype, atol = _shape_for(case)
    b = len(seq_q_list)
    tq, tkv = sum(seq_q_list), sum(seq_kv_list)
    d = TD
    torch.manual_seed(_seed_for(case))
    hkv = heads // case.kv_group
    layout = case.layout
    if layout == "bshd":
        s, sk = seq_q_list[0], seq_kv_list[0]
        q = torch.rand((b, s, heads, d), dtype=dtype).to(ST_DEVICE)
        k = torch.rand((b, sk, hkv, d), dtype=dtype).to(ST_DEVICE)
        v = torch.rand((b, sk, hkv, d), dtype=dtype).to(ST_DEVICE)
        q_flat = q.reshape(tq, heads, d)
        k_flat = k.reshape(tkv, hkv, d)
        v_flat = v.reshape(tkv, hkv, d)
    else:
        q = torch.rand((tq, heads, d), dtype=dtype).to(ST_DEVICE)
        k = torch.rand((tkv, hkv, d), dtype=dtype).to(ST_DEVICE)
        v = torch.rand((tkv, hkv, d), dtype=dtype).to(ST_DEVICE)
        q_flat, k_flat, v_flat = q, k, v
    o = torch.zeros((tq, heads, d), dtype=dtype, device=ST_DEVICE)
    smax = torch.zeros((heads, tq), dtype=torch.float32, device=ST_DEVICE)
    ssum = torch.zeros((heads, tq), dtype=torch.float32, device=ST_DEVICE)

    mask_info, dense_masks = _build_case_mask_info(case, seq_q_list, seq_kv_list)

    def run():
        _flex_attention_from_info(q, k, v, seq_q_list, seq_kv_list, o, smax, ssum,
                                  mask_info, layout=layout)

    avg_ms = 1
    bench_kernel(run)
    e0 = torch.npu.Event(enable_timing=True)
    e1 = torch.npu.Event(enable_timing=True)
    e0.record()
    run()
    e1.record()
    e1.synchronize()
    avg_ms = e0.elapsed_time(e1)
    if case.kv_group > 1:
        k_ref = k_flat.repeat_interleave(case.kv_group, dim=1)
        v_ref = v_flat.repeat_interleave(case.kv_group, dim=1)
    else:
        k_ref, v_ref = k_flat, v_flat
    o_ref = flash_attention_ref_tnd(q_flat, k_ref, v_ref, seq_q_list, seq_kv_list,
                                    d, dense_masks, dtype)
    diff = (o.cpu() - o_ref).abs().max().item()
    bytes_per = d * 2 * (2 * tq * heads + 2 * tkv * hkv)
    return avg_ms, bytes_per / avg_ms / 1e6, diff


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["a", "b", "all"], default="all",
                        help="a=causal sweep, b=matrix cases, all=both")
    args = parser.parse_args()

    torch.npu.set_device(ST_DEVICE)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    failures = []

    if args.part in ("a", "all"):
        print("=" * 80)
        print("PART A: causal sweep (mask_num=1, pre=100000, next=0, D=128, qn=8, bf16)")
        print(f"{'seq':>6} {'batch':>5} {'qn':>3} {'kn':>3} {'kv_group':>8} "
              f"{'time(ms)':>10} {'bw(GB/s)':>10} {'max|diff|':>12}  "
              f"{'seq_q_list':<24} {'seq_kv_list':<24} {'qn':>2} {'hkv':>3} {'d':>3}")
        for seq, kv_group in CAUSAL_CFGS:
            for batch in BATCH_BY_SEQ[seq]:
                avg_ms, bw, diff = bench_causal(seq, batch, kv_group)
                kn = 8 // kv_group
                sq = str([seq] * batch).replace(" ", "")
                skv = str([seq] * batch).replace(" ", "")
                print(f"{seq:>6} {batch:>5} {8:>3} {kn:>3} {kv_group:>8} "
                      f"{avg_ms:>10.3f} {bw:>10.1f} {diff:>12.6f}  "
                      f"{sq:<24} {skv:<24} {8:>2} {kn:>3} {TD:>3}")
                if diff > 1e-2:
                    failures.append(f"causal seq={seq} batch={batch} kv_group={kv_group} "
                                    f"diff={diff}")

    if args.part in ("b", "all"):
        print("=" * 80)
        print("PART B: matrix cases (test_flex_attention_matrix.py EXEC_CASES)")
        print(f"{'case':<40} {'time(ms)':>10} {'bw(GB/s)':>10} {'max|diff|':>12}  "
              f"{'seq_q_list':<24} {'seq_kv_list':<24} {'qn':>2} {'hkv':>3} {'d':>3}")
        for case in EXEC_CASES:
            if case.expected_error or case.skip:
                continue
            seq_q_list, seq_kv_list, heads, _, atol = _shape_for(case)
            avg_ms, bw, diff = bench_matrix_case(case)
            hkv = heads // case.kv_group
            sq = str(seq_q_list).replace(" ", "")
            skv = str(seq_kv_list).replace(" ", "")
            print(f"{case_id(case):<40} {avg_ms:>10.3f} {bw:>10.1f} {diff:>12.6f}  "
                  f"{sq:<24} {skv:<24} {heads:>2} {hkv:>3} {TD:>3}")
            if diff > atol:
                failures.append(f"{case_id(case)} diff={diff} atol={atol}")

    if failures:
        raise AssertionError(f"{len(failures)} precision failures:\n" +
                             "\n".join(f"  {f}" for f in failures))
    logging.info("\nAll perf cases passed!")
