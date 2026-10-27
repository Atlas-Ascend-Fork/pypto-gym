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

"""FlexAttention matrix test cases (mirrors test_sdpa_matrix.py report structure).

Matrix dimensions: mask family (7), layout (varlen/bshd), qk (eq/neq),
fwd_source (regionmask/compilemask), kv_group (GQA). Adapted to the PyPTO
flex_attention kernel: forward-only, bitmask interval schedule (dense/bwd/
callable axes from the GPU report do not apply).

Stripe patterns (callable_modulo mask_num=6, piecewise mask_num=21) execute on
the kernel: mask_num <= CHUNK runs the single-pass decoder, mask_num > CHUNK
runs the HoleChunk=1 chunked decoder (base + refine passes).

Usage:
    python3 python/tests/st/pypto_pro/frontend/fa/test_flex_attention_matrix.py
"""

import dataclasses
import itertools
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
    flex_attention, CHUNK, MASK_NUM_LIMIT, TD, TKV,
)
from mask_specs import (  # noqa: E402
    FOUR_STAGE_SPEC, HSTU_FWD_SPEC, SPU_FWD_SPEC,
    REALTIME_VAR_SPEC, CALLABLE_MODULO_SPEC, PIECEWISE_SPEC,
)
from test_flex_attention import (  # noqa: E402
    flash_attention_ref_tnd, _build_spec_mask_info, _fixed_intervals_mask_info,
    _maskgen_simulate, _pad_maskgen_to_info, _reconstruct_intervals_from_mask_info,
    _assert_mask_info_equal,
)

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

TS = 128


# ================================================================
#  AttentionCase
# ================================================================

@dataclasses.dataclass
class AttentionCase:
    mask: str
    layout: str
    qk: str
    fwd_source: str
    kv_group: int = 1
    skip: bool = False
    expected_error: str | None = None


def case_id(c: AttentionCase) -> str:
    cid = f"{c.mask}-{c.layout}-{c.qk}-{c.fwd_source}"
    if c.kv_group != 1:
        cid += f"-g{c.kv_group}"
    return cid


# ================================================================
#  Shape / seed / dtype per mask family
# ================================================================

def _shape_for(case: AttentionCase):
    if case.mask == "banded":
        return [64, 64], [96, 96], 2, torch.float16, 1e-3
    if case.mask == "piecewise":
        return [193], [193], 4, torch.bfloat16, 1e-2
    if case.mask == "callable_modulo":
        return [64, 64], [64, 64], 8, torch.bfloat16, 1e-2
    # varlen/bshd share equal lengths (report: varlen uses fix_max_seqlen,
    # L = B*S); the layout differs in how tensors reach the operator.
    return [64, 64], [64, 64], 8, torch.bfloat16, 1e-2


def _seed_for(case: AttentionCase):
    return {"banded": 20260804, "4stage": 114,
            "callable_modulo": 116, "piecewise": 7}.get(case.mask, 115)


# ================================================================
#  MaskSpec + tensor inputs per mask family
# ================================================================

def _spec_for(case: AttentionCase, b: int, seq_q_list):
    if case.mask == "4stage":
        return FOUR_STAGE_SPEC, {
            "context_sizes": torch.full((b,), 16, dtype=torch.int32),
            "history_sizes": torch.full((b,), 16, dtype=torch.int32),
            "realtime_sizes": torch.full((b,), 16, dtype=torch.int32),
            "label_nums": torch.full((b,), 4, dtype=torch.int32),
            "label_group_sizes": torch.full((b,), 4, dtype=torch.int32)}
    if case.mask == "3stage":
        return HSTU_FWD_SPEC, {
            "context_length": torch.full((b,), 20, dtype=torch.int32),
            "history_length": torch.full((b,), 20, dtype=torch.int32),
            "label_group_size": torch.full((b,), 12, dtype=torch.int32),
            "cu_seqlen": torch.tensor([0] + [64] * b, dtype=torch.int32)}
    if case.mask == "realtime_var":
        return REALTIME_VAR_SPEC, {
            "context_sizes": torch.full((b,), 16, dtype=torch.int32),
            "history_sizes": torch.full((b,), 16, dtype=torch.int32),
            "realtime_sizes": torch.full((b,), 16, dtype=torch.int32),
            "label_nums": torch.full((b,), 4, dtype=torch.int32),
            "label_group_sizes": torch.full((b,), 4, dtype=torch.int32),
            "realtime_offsets": torch.full((b,), 22, dtype=torch.int32)}
    if case.mask == "labelsec":
        cu = [0] + [64] * b
        acc = list(itertools.accumulate(cu))
        return SPU_FWD_SPEC, {
            "context_length": torch.full((b,), 32, dtype=torch.int32),
            "labelgroup_size": torch.full((b,), 8, dtype=torch.int32),
            "cu_seqlens": torch.tensor(acc, dtype=torch.int32)}
    if case.mask == "callable_modulo":
        return CALLABLE_MODULO_SPEC, {
            "context_sizes": torch.full((b,), 16, dtype=torch.int32),
            "history_sizes": torch.full((b,), 16, dtype=torch.int32),
            "realtime_sizes": torch.full((b,), 16, dtype=torch.int32),
            "label_nums": torch.full((b,), 4, dtype=torch.int32),
            "label_group_sizes": torch.full((b,), 4, dtype=torch.int32),
            "mods": torch.full((b,), 6, dtype=torch.int32),
            "rems": torch.full((b,), 3, dtype=torch.int32)}
    if case.mask == "piecewise":
        return PIECEWISE_SPEC, {
            "seq_lens": torch.tensor(seq_q_list, dtype=torch.int32),
            "row_l2s": torch.full((b,), 128, dtype=torch.int32),
            "col_splits": torch.full((b,), 128, dtype=torch.int32),
            "mods": torch.full((b,), 6, dtype=torch.int32),
            "rems": torch.full((b,), 3, dtype=torch.int32)}
    raise ValueError(f"unknown mask {case.mask}")


# ================================================================
#  MaskInfo builders (regionmask = scan, compilemask = endpoint)
# ================================================================

def _build_spec_mask_info_endpoint(seq_q_list, seq_kv_list, spec, mask_num, **tensors):
    """Derive intervals from the spec predicate via scanning, then pack via
    _maskgen_simulate (endpoint path).  Produces identical mask_info to
    _build_spec_mask_info (scan path) — cf. test_maskgen_sim_parity."""
    b = len(seq_q_list)
    scalars_by_batch = []
    for bi in range(b):
        targs = {arg.name: tensors[arg.name] for arg in spec.tensor_args}
        scalars_by_batch.append(spec.scalar_setup(bi, **targs))

    def row_intervals_fn(bi, r):
        scalars = scalars_by_batch[bi]
        intervals, start = [], None
        skv = seq_kv_list[bi]
        for c in range(skv):
            visible = False
            for region in spec.regions:
                if region.when(r, **scalars):
                    visible = region.visible(r, c, **scalars)
                    break
            if visible:
                if start is None:
                    start = c
            elif start is not None:
                intervals.append((start, c))
                start = None
        if start is not None:
            intervals.append((start, skv))
        return tuple(intervals)

    maskr, mask_start, mask_len, block_range, bc, bm = _maskgen_simulate(
        seq_q_list, seq_kv_list, row_intervals_fn, mask_num)
    mask_info = _pad_maskgen_to_info(maskr, mask_start, mask_len, block_range, bc, bm, mask_num)

    dense_masks = []
    for bi, skv in enumerate(seq_kv_list):
        sq = seq_q_list[bi]
        m = torch.zeros((sq, skv), dtype=torch.bool)
        for r in range(sq):
            for c in range(skv):
                visible = False
                for region in spec.regions:
                    if region.when(r, **scalars_by_batch[bi]):
                        visible = region.visible(r, c, **scalars_by_batch[bi])
                        break
                m[r, c] = visible
        dense_masks.append(~m)
    return mask_info, dense_masks


def _build_case_mask_info(case: AttentionCase, seq_q_list, seq_kv_list):
    b = len(seq_q_list)
    if case.mask == "banded":
        skv = seq_kv_list[0]
        return _fixed_intervals_mask_info(
            seq_q_list, seq_kv_list, ((3, skv),), 1)
    spec, inputs = _spec_for(case, b, seq_q_list)
    req = _required_mask_num(case)
    if case.fwd_source == "regionmask":
        return _build_spec_mask_info(seq_q_list, seq_kv_list, spec, req, **inputs)
    return _build_spec_mask_info_endpoint(seq_q_list, seq_kv_list, spec, req, **inputs)


# ================================================================
#  flex_attention bridge
# ================================================================

def _flex_attention_from_info(q, k, v, seq_q_list, seq_kv_list, attn_out, softmax_max, softmax_sum,
                              mask_info, layout="varlen"):
    device = q.device
    def _to_dev(x):
        return x.to(device) if x is not None else None
    if layout == "bshd":
        flex_attention(q, k, v, None, None,
                       _to_dev(mask_info["maskr"]),
                       _to_dev(mask_info.get("mask_start")),
                       _to_dev(mask_info.get("mask_len")),
                       _to_dev(mask_info["block_range"]),
                       _to_dev(mask_info["block_compute"]),
                       _to_dev(mask_info["block_mask"]),
                       attn_out, softmax_max, softmax_sum, layout="bshd")
        return
    actual_seq_q = torch.tensor(seq_q_list, dtype=torch.int64, device=device).cumsum(0)
    actual_seq_kv = torch.tensor(seq_kv_list, dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
    flex_attention(q, k, v, actual_seq_q, actual_seq_kv,
                   _to_dev(mask_info["maskr"]),
                   _to_dev(mask_info.get("mask_start")),
                   _to_dev(mask_info.get("mask_len")),
                   _to_dev(mask_info["block_range"]),
                   _to_dev(mask_info["block_compute"]),
                   _to_dev(mask_info["block_mask"]),
                   attn_out, softmax_max, softmax_sum, layout="varlen")


# ================================================================
#  Negative case runner
# ================================================================

def _run_negative(case: AttentionCase):
    cid = case_id(case)
    seq_q_list, seq_kv_list, heads, dtype, _ = _shape_for(case)
    b = len(seq_q_list)
    tq, tkv = sum(seq_q_list), sum(seq_kv_list)
    torch.manual_seed(114)
    q = torch.rand((tq, heads, TD), dtype=dtype).to(ST_DEVICE)
    k = torch.rand((tkv, heads, TD), dtype=dtype).to(ST_DEVICE)
    v = torch.rand((tkv, heads, TD), dtype=dtype).to(ST_DEVICE)
    o = torch.zeros((tq, heads, TD), dtype=dtype, device=ST_DEVICE)
    smax = torch.zeros((heads, tq), dtype=torch.float32, device=ST_DEVICE)
    ssum = torch.zeros((heads, tq), dtype=torch.float32, device=ST_DEVICE)
    mask_info, _ = _build_case_mask_info(case, seq_q_list, seq_kv_list)
    bad = dict(mask_info)
    if case.expected_error == "mask_num_overflow":
        # Kernel supports arbitrary mask_num via the chunked decoder; only the
        # host sanity limit rejects. mask_num is derived from
        # mask_start.shape[1], so overflow is injected via an oversized
        # mask_start/mask_len pair.
        ms = mask_info["mask_start"]
        big = torch.zeros((ms.shape[0], MASK_NUM_LIMIT + 1, ms.shape[2]),
                          dtype=ms.dtype)
        bad["mask_start"] = big
        bad["mask_len"] = big.clone()
    elif case.expected_error == "missing_maskr":
        bad["maskr"] = None
    elif case.expected_error == "missing_block_compute":
        bad["block_compute"] = None
    elif case.expected_error == "missing_mask_len":
        bad["mask_len"] = None
    with pytest.raises((AssertionError, ValueError)):
        _flex_attention_from_info(q, k, v, seq_q_list, seq_kv_list, o, smax, ssum, bad,
                                  layout=case.layout)
    logging.info("  [PASS] %s (expected_error=%s)", cid, case.expected_error)


# ================================================================
#  Numerical case runner
# ================================================================

def _run_case(case: AttentionCase):
    cid = case_id(case)
    if case.expected_error:
        _run_negative(case)
        return
    seq_q_list, seq_kv_list, heads, dtype, atol = _shape_for(case)
    b = len(seq_q_list)
    tq, tkv = sum(seq_q_list), sum(seq_kv_list)
    d = TD
    torch.manual_seed(_seed_for(case))
    hkv = heads // case.kv_group
    layout = case.layout
    if layout == "bshd":
        s = seq_q_list[0]
        sk = seq_kv_list[0]
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
        q_flat = q
        k_flat = k
        v_flat = v
    o = torch.zeros((tq, heads, d), dtype=dtype, device=ST_DEVICE)
    smax = torch.zeros((heads, tq), dtype=torch.float32, device=ST_DEVICE)
    ssum = torch.zeros((heads, tq), dtype=torch.float32, device=ST_DEVICE)
    mask_info, dense_masks = _build_case_mask_info(case, seq_q_list, seq_kv_list)
    # CPU: verify regionmask == compilemask parity for non-banded specs
    if case.mask != "banded" and case.fwd_source == "regionmask":
        spec, inputs = _spec_for(case, b, seq_q_list)
        req = _required_mask_num(case)
        mi_end, _ = _build_spec_mask_info_endpoint(seq_q_list, seq_kv_list, spec, req, **inputs)
        _assert_mask_info_equal(mask_info, mi_end)
    _flex_attention_from_info(q, k, v, seq_q_list, seq_kv_list, o, smax, ssum, mask_info,
                              layout=layout)
    # Reference expects MHA (q_heads == kv_heads). Expand GQA k/v to match.
    if case.kv_group > 1:
        k_ref = k_flat.repeat_interleave(case.kv_group, dim=1)
        v_ref = v_flat.repeat_interleave(case.kv_group, dim=1)
    else:
        k_ref = k_flat
        v_ref = v_flat
    o_ref = flash_attention_ref_tnd(q_flat, k_ref, v_ref, seq_q_list, seq_kv_list,
                                    d, dense_masks, dtype)
    torch.testing.assert_close(o.cpu(), o_ref, rtol=atol, atol=atol)
    logging.info("  [PASS] %s (max|diff|=%s)", cid,
                 (o.cpu() - o_ref).abs().max().item())


# ================================================================
#  Stripe-pattern mask_info verification (CPU only, skip execution)
# ================================================================

def _required_mask_num(case: AttentionCase):
    if case.mask == "callable_modulo":
        return 6
    if case.mask == "piecewise":
        return 21
    return 1


def _verify_stripe_mask_info(case: AttentionCase):
    seq_q_list, seq_kv_list, _, _, _ = _shape_for(case)
    b = len(seq_q_list)
    spec, inputs = _spec_for(case, b, seq_q_list)
    req = _required_mask_num(case)
    mask_info, dense_masks = _build_spec_mask_info(seq_q_list, seq_kv_list, spec, req, **inputs)
    assert mask_info["mask_num"] == req
    for bi in range(b):
        sq, skv = seq_q_list[bi], seq_kv_list[bi]
        dm = dense_masks[bi].cpu().bool()
        for r in (0, min(40, sq - 1), sq - 1):
            recon = _reconstruct_intervals_from_mask_info(
                mask_info["maskr"], mask_info["mask_start"],
                mask_info["mask_len"], req, bi, r)
            vis = torch.zeros(skv, dtype=torch.bool)
            for s, e in recon:
                vis[s:e] = True
            torch.testing.assert_close(vis, ~dm[r], rtol=0, atol=0)
    logging.info("  [VERIFY] %s mask_info reconstruction matches predicate (mask_num=%d)",
                 case_id(case), req)


# ================================================================
#  Case lists
# ================================================================

CORE_CASES = [
    # 4stage: 2 layouts × 2 fwd_source
    AttentionCase("4stage", "varlen", "eq", "regionmask"),
    AttentionCase("4stage", "bshd", "eq", "regionmask"),
    AttentionCase("4stage", "varlen", "eq", "compilemask"),
    AttentionCase("4stage", "bshd", "eq", "compilemask"),
    # banded: bshd × neq × 2 fwd_source
    AttentionCase("banded", "bshd", "neq", "regionmask"),
    AttentionCase("banded", "bshd", "neq", "compilemask"),
    # 3stage / realtime_var / labelsec
    AttentionCase("3stage", "varlen", "eq", "regionmask"),
    AttentionCase("3stage", "bshd", "eq", "regionmask"),
    AttentionCase("realtime_var", "varlen", "eq", "regionmask"),
    AttentionCase("realtime_var", "bshd", "eq", "regionmask"),
    AttentionCase("labelsec", "varlen", "eq", "regionmask"),
]

GQA_CASES = [
    # 4stage GQA cross: 2 fwd_source × 2 kv_group
    AttentionCase("4stage", "varlen", "eq", "regionmask", kv_group=4),
    AttentionCase("4stage", "varlen", "eq", "compilemask", kv_group=4),
    AttentionCase("4stage", "varlen", "eq", "regionmask", kv_group=8),
    AttentionCase("4stage", "varlen", "eq", "compilemask", kv_group=8),
    # banded GQA: 2 fwd_source × kv_group=2
    AttentionCase("banded", "bshd", "neq", "regionmask", kv_group=2),
    AttentionCase("banded", "bshd", "neq", "compilemask", kv_group=2),
    # semantic GQA: representative kv_group=4
    AttentionCase("3stage", "varlen", "eq", "regionmask", kv_group=4),
    AttentionCase("realtime_var", "varlen", "eq", "regionmask", kv_group=4),
]

NEGATIVE_CASES = [
    AttentionCase("4stage", "varlen", "eq", "regionmask", expected_error="mask_num_overflow"),
    AttentionCase("4stage", "varlen", "eq", "regionmask", expected_error="missing_maskr"),
    AttentionCase("4stage", "varlen", "eq", "regionmask", expected_error="missing_block_compute"),
    AttentionCase("4stage", "varlen", "eq", "regionmask", expected_error="missing_mask_len"),
]

STRIPE_CASES = [
    # callable_modulo (mask_num=6 <= CHUNK): single-pass decoder.
    AttentionCase("callable_modulo", "varlen", "eq", "compilemask"),
    # piecewise (mask_num=21 > CHUNK): HoleChunk=1 chunked decoder (16 + 5).
    AttentionCase("piecewise", "varlen", "eq", "compilemask"),
]

EXEC_CASES = CORE_CASES + GQA_CASES + NEGATIVE_CASES + STRIPE_CASES


# ================================================================
#  Tests
# ================================================================

@pytest.mark.soc("950")
@pytest.mark.parametrize("case", EXEC_CASES, ids=lambda c: case_id(c))
def test_fa_matrix_case(case):
    torch.npu.set_device(ST_DEVICE)
    _run_case(case)


@pytest.mark.soc("950")
@pytest.mark.parametrize("case", STRIPE_CASES, ids=lambda c: case_id(c))
def test_fa_matrix_stripe_generation(case):
    torch.npu.set_device(ST_DEVICE)
    # CPU-only mask_info generation + reconstruction verification (the cases
    # also execute on the kernel via test_fa_matrix_case).
    _verify_stripe_mask_info(case)
    logging.info("  [VERIFY] %s also executes on kernel "
                 "(mask_num=%d, chunked=%s)", case_id(case),
                 _required_mask_num(case), _required_mask_num(case) > CHUNK)


# ================================================================
#  CPU-only: spec parity between regionmask and compilemask paths
# ================================================================

def test_new_spec_sim_parity():
    """Verify that for all new specs, regionmask (scan) and compilemask (endpoint)
    produce identical mask_info."""
    b = 2
    sq_list = [64, 64]
    skv_list = [64, 64]
    for spec, name in [(REALTIME_VAR_SPEC, "realtime_var"),
                       (CALLABLE_MODULO_SPEC, "callable_modulo"),
                       (PIECEWISE_SPEC, "piecewise")]:
        req = {"realtime_var": 1, "callable_modulo": 6, "piecewise": 21}[name]
        if name == "piecewise":
            sq_list_p = [64, 64]
            skv_list_p = [64, 64]
        else:
            sq_list_p = sq_list
            skv_list_p = skv_list
        tensors = {arg.name: torch.full((b,), 16, dtype=torch.int32)
                   for arg in spec.tensor_args}
        if name == "realtime_var":
            tensors["realtime_offsets"] = torch.full((b,), 22, dtype=torch.int32)
        if name == "callable_modulo":
            tensors["mods"] = torch.full((b,), 6, dtype=torch.int32)
            tensors["rems"] = torch.full((b,), 3, dtype=torch.int32)
        if name == "piecewise":
            tensors["seq_lens"] = torch.tensor(sq_list_p, dtype=torch.int32)
            tensors["row_l2s"] = torch.full((b,), 128, dtype=torch.int32)
            tensors["col_splits"] = torch.full((b,), 128, dtype=torch.int32)
            tensors["mods"] = torch.full((b,), 6, dtype=torch.int32)
            tensors["rems"] = torch.full((b,), 3, dtype=torch.int32)
        mi_scan, _ = _build_spec_mask_info(sq_list_p, skv_list_p, spec, req, **tensors)
        mi_end, _ = _build_spec_mask_info_endpoint(sq_list_p, skv_list_p, spec, req, **tensors)
        _assert_mask_info_equal(mi_scan, mi_end)
    logging.info("  [PASS] new spec parity (regionmask == compilemask)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.npu.set_device(ST_DEVICE)
    logging.info("FlexAttention matrix test suite")
    logging.info("  %d exec cases (incl. %d stripe: chunked=%s)",
                 len(EXEC_CASES), len(STRIPE_CASES),
                 [(_required_mask_num(c), _required_mask_num(c) > CHUNK) for c in STRIPE_CASES])
    logging.info("%s", "=" * 60)
    test_new_spec_sim_parity()
    failures = []
    for c in EXEC_CASES:
        try:
            _run_case(c)
        except Exception as e:
            failures.append((case_id(c), str(e)))
    if failures:
        msg = "\n".join(f"  {cid}: {err}" for cid, err in failures)
        raise AssertionError(f"{len(failures)} failures:\n{msg}")
    logging.info("\nAll tests passed!")