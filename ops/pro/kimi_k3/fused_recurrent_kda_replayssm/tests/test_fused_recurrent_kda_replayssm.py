#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""fused_recurrent_kda_replayssm PyPTO-Pro tests: one test function per case.

The kernel and the host wrapper live under ``op_kernel/``; the CPU golden and
the numeric comparator live under ``tests/assets/``.  Each ``test_xxx`` prepares
one case's inputs, runs the public wrapper on the NPU and compares the BF16
output and the FP16 U/K + FP32 decay records against the independent FP64 CPU
golden (``atol = rtol = 5e-3``); the read-only checkpoint and the unused record
suffix are checked with zero tolerance.

Run: ``TILE_FWK_DEVICE_ID=0 pytest ops/pro/kimi_k3/fused_recurrent_kda_replayssm/tests --mode jit -q``
"""

import importlib.util
import logging
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from fused_recurrent_kda_replayssm import fused_recurrent_kda

_GEN_GOLDEN = Path(__file__).resolve().parent / "assets" / "gen_golden.py"
_spec = importlib.util.spec_from_file_location(
    "fused_recurrent_kda_replayssm_gen_golden", _GEN_GOLDEN)
golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(golden)
check_precision = golden.check_precision

logging.basicConfig(level=logging.INFO, format="%(message)s")
_LOG = logging.getLogger(__name__)

DEVICE_ID = os.environ.get("TILE_FWK_DEVICE_ID", "0")

pytestmark = pytest.mark.soc("950")


def require_a5():
    try:
        torch.npu.set_device(int(DEVICE_ID))
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Device {name} is not A5 (Ascend950). Skip.")


def _npu(x, dtype=None):
    t = torch.from_numpy(np.ascontiguousarray(x))
    if dtype is not None:
        t = t.to(dtype)
    return t.to(f"npu:{DEVICE_ID}")


def _run_case(layout, batch, seq, heads, *, cu_seqlens=None, capacity=None,
              rows=None, workers=None, slot=2, seed=1234):
    inp = golden.make_inputs(batch=batch, seq=seq, heads=heads, seed=seed,
                             layout=layout, cu_seqlens=cu_seqlens,
                             capacity=capacity)
    cap = inp["capacity"]
    rec_shape = (batch, cap, heads, 128)
    recs = [torch.full(rec_shape, golden.SENTINEL, dtype=dt, device=f"npu:{DEVICE_ID}")
            for dt in (torch.float16, torch.float16, torch.float32)]
    state = _npu(inp["state"].astype(np.float32))
    state_before = state.cpu().clone()
    cu = None
    if layout == "TND":
        cu_np = np.cumsum([0] + list(inp["seqlens"])).astype(np.int32)
        cu = torch.from_numpy(cu_np).to(torch.int32).to(f"npu:{DEVICE_ID}")
    out = fused_recurrent_kda(
        _npu(inp["mixedqkv"], torch.bfloat16), state,
        _npu(inp["beta"], torch.bfloat16), _npu(inp["g"], torch.bfloat16),
        None, _npu(inp["A_log"]), _npu(inp["dt_bias"]), -5.0, layout,
        replay_u=recs[0], replay_k=recs[1], replay_decay=recs[2],
        cu_seqlens=cu, rows=rows, workers=workers, slots=slot)
    torch.npu.synchronize()
    assert torch.equal(state.cpu(), state_before), "read-only checkpoint changed"
    return inp, out.cpu(), [r.cpu() for r in recs]


def _compare(inp, out, recs):
    layout = inp["layout"]
    cap = inp["capacity"]
    out_ref, u_ref, k_ref, d_ref = golden.reference(inp)
    res = check_precision(out.float().numpy(), out_ref)
    assert res["ok"], f"out mismatch: {res}"
    for name, rec, ref in (("replay_u", recs[0], u_ref),
                           ("replay_k", recs[1], k_ref),
                           ("replay_decay", recs[2], d_ref)):
        expected = np.full((inp["batch"], cap, inp["heads"], 128),
                           golden.SENTINEL, dtype=np.float64)
        if layout == "TND":
            offset = 0
            for b, length in enumerate(inp["seqlens"]):
                expected[b, :length] = ref[offset:offset + length]
                offset += length
            valid = expected[:, :max(inp["seqlens"])]
        else:
            expected[:, :inp["seq"]] = ref
            valid = expected[:, :inp["seq"]]
        got = rec.float().numpy().astype(np.float64)
        res = check_precision(got[:, :valid.shape[1]], valid)
        assert res["ok"], f"{name} mismatch: {res}"
        assert np.array_equal(got[:, valid.shape[1]:],
                              np.full_like(got[:, valid.shape[1]:], golden.SENTINEL)), \
            f"{name} unused suffix changed"
    _LOG.info("%s %s: out max_abs=%.3e", inp["layout"],
              (inp["batch"], inp["seq"], inp["heads"]), res["max_abs"])


@pytest.mark.parametrize("layout,batch,seq,heads", [
    ("BSND", 2, 8, 6),
    ("BNSD", 2, 5, 3),
    ("TND", 3, 0, 2),
])
def test_layouts(layout, batch, seq, heads):
    require_a5()
    cu = [5, 1, 3] if layout == "TND" else None
    inp, out, recs = _run_case(layout, batch, seq, heads, cu_seqlens=cu)
    _compare(inp, out, recs)


def test_minimal_single_token():
    require_a5()
    inp, out, recs = _run_case("BSND", 1, 1, 1)
    _compare(inp, out, recs)


def test_long_sequence():
    require_a5()
    inp, out, recs = _run_case("BSND", 2, 17, 4)
    _compare(inp, out, recs)


def test_max_heads():
    require_a5()
    inp, out, recs = _run_case("BSND", 1, 4, 96)
    _compare(inp, out, recs)


def test_record_capacity_suffix():
    require_a5()
    inp, out, recs = _run_case("BSND", 2, 4, 3, capacity=9)
    _compare(inp, out, recs)


@pytest.mark.parametrize("rows", [16, 32, 64])
def test_rows(rows):
    require_a5()
    inp, out, recs = _run_case("BSND", 4, 6, 4, rows=rows)
    _compare(inp, out, recs)


@pytest.mark.parametrize("workers", [1, 8, 32, 64])
def test_worker_sweep(workers):
    require_a5()
    inp, out, recs = _run_case("BSND", 4, 6, 4, workers=workers)
    _compare(inp, out, recs)


def test_repeat_determinism():
    require_a5()
    inp = golden.make_inputs(batch=2, seq=8, heads=6, seed=99, layout="BSND")
    cap = inp["capacity"]
    rec_shape = (2, cap, 6, 128)
    mixed = _npu(inp["mixedqkv"], torch.bfloat16)
    state = _npu(inp["state"].astype(np.float32))
    beta = _npu(inp["beta"], torch.bfloat16)
    gate = _npu(inp["g"], torch.bfloat16)
    a_log, dt_bias = _npu(inp["A_log"]), _npu(inp["dt_bias"])
    first = None
    for _ in range(10):
        recs = [torch.full(rec_shape, golden.SENTINEL, dtype=dt, device=f"npu:{DEVICE_ID}")
                for dt in (torch.float16, torch.float16, torch.float32)]
        out = fused_recurrent_kda(mixed, state, beta, gate, None, a_log, dt_bias,
                                  -5.0, "BSND", replay_u=recs[0],
                                  replay_k=recs[1], replay_decay=recs[2])
        torch.npu.synchronize()
        cur = (out.cpu(), recs[0].cpu(), recs[1].cpu())
        if first is None:
            first = cur
        else:
            assert all(torch.equal(a, b) for a, b in zip(first, cur)), \
                "non-deterministic output"
