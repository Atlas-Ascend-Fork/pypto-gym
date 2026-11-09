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

"""fused_recurrent_kda_snapshot PyPTO-Pro test: one test function per case.

The kernel and the host wrapper live under
``op_kernel/fused_recurrent_kda_snapshot.py``; the CPU golden and the numeric
comparator live under ``tests/assets/``.  Each ``test_xxx`` prepares one case's
inputs, runs the public wrapper on the NPU and compares the updated state pool
and the output against the independent FP64 CPU golden (``atol = rtol = 5e-3``
for an FP32 pool, ``1e-2`` for a BF16 pool).

Run: ``pytest ops/pro/kimi_k3/fused_recurrent_kda_snapshot/tests --mode jit -q``
"""

import importlib.util
import logging
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from fused_recurrent_kda_snapshot import fused_recurrent_kda

_GEN_GOLDEN = Path(__file__).resolve().parent / "assets" / "gen_golden.py"
_spec = importlib.util.spec_from_file_location(
    "fused_recurrent_kda_snapshot_gen_golden", _GEN_GOLDEN)
golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(golden)
check_precision = golden.check_precision

logging.basicConfig(level=logging.INFO, format="%(message)s")
_LOG = logging.getLogger(__name__)

DEVICE_ID = os.environ.get("TILE_FWK_DEVICE_ID", "0")


def require_a5():
    try:
        torch.npu.set_device(int(DEVICE_ID))
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Device {name} is not A5 (Ascend950). Skip.")


def _device():
    import torch_npu  # noqa: F401

    torch.npu.set_device(int(DEVICE_ID))
    return "npu"


_BF16_FIELDS = ("mixedqkv", "g", "beta")


def _to_npu(inp):
    out = {}
    for key, value in inp.items():
        if not isinstance(value, np.ndarray):
            out[key] = value
            continue
        tensor = torch.from_numpy(value)
        if key in _BF16_FIELDS:
            out[key] = tensor.to(torch.bfloat16).npu()
        elif key == "state":
            out[key] = tensor.to(torch.bfloat16 if tensor.dtype == torch.bfloat16
                                 else torch.float32).npu()
        elif value.dtype == np.int32:
            out[key] = tensor.int().npu()
        elif value.dtype == np.float32:
            out[key] = tensor.float().npu()
        else:
            out[key] = tensor.npu()
    return out


def run_case(layout="BSND", batch=2, seq=8, heads=6, dim=128, seed=0, *,
             state_dtype=np.float32, pool=None, use_indices=False, accepted=None,
             seq_lens=None, rows=32, workers=None, scale=None, lower_bound=-5.0,
             check=True):
    require_a5()
    device = _device()
    cu = None
    seqlens = None
    if layout == "TND":
        seqlens = seq_lens if seq_lens is not None else [seq] * batch
        cu = np.zeros(batch + 1, dtype=np.int32)
        for i, length in enumerate(seqlens):
            cu[i + 1] = cu[i] + length
    indices = None
    if use_indices:
        token_count = int(cu[-1]) if layout == "TND" else batch * seq
        rng = np.random.default_rng(seed + 991)
        indices = rng.permutation(pool if pool is not None else token_count).astype(np.int32)
        indices = indices[:token_count]
    inp = golden.make_inputs(
        batch=batch, seq=seq, heads=heads, dim=dim, seed=seed, layout=layout,
        state_dtype=state_dtype, pool=pool, state_indices=indices, accepted=accepted,
        cu_seqlens=seqlens, scale=scale, lower_bound=lower_bound)
    state_ref, out_ref = golden.reference(inp)
    dev = _to_npu(inp)
    if state_dtype == "bf16":
        dev["state"] = dev["state"].to(torch.bfloat16)
    state_t = dev["state"].clone()
    kwargs = {}
    if layout == "TND":
        kwargs["cu_seqlens"] = torch.from_numpy(cu).int().npu()
    if indices is not None:
        kwargs["ssm_state_indices"] = torch.from_numpy(indices).int().npu()
    if accepted is not None:
        kwargs["num_accepted_tokens"] = torch.from_numpy(
            np.asarray(accepted, dtype=np.int32)).int().npu()
    out = None
    _, out = fused_recurrent_kda(
        dev["mixedqkv"], state_t, dev["beta"], dev["g"], scale, dev["A_log"],
        dev["dt_bias"], lower_bound, layout, rows=rows, workers=workers, **kwargs)
    torch.npu.synchronize()
    result = {
        "state": state_t.cpu().float().numpy(),
        "out": out.cpu().float().numpy(),
        "state_ref": state_ref,
        "out_ref": out_ref,
        "inp": inp,
        "device": device,
    }
    if check:
        _check(result, state_dtype, pool)
    return result


def _check(result, state_dtype, pool):
    got = result["state"]
    want = result["state_ref"]
    if pool is not None:
        active = result["inp"]["ssm_state_indices"]
        got_cmp = got[active]
        want_cmp = want[active]
    else:
        got_cmp, want_cmp = got, want
    state_tol = 5e-3 if state_dtype == np.float32 else 1e-2
    passed, summary = check_precision(got_cmp, want_cmp, rtol=state_tol, atol=state_tol)
    assert passed, f"state mismatch: {summary}"
    passed, summary = check_precision(result["out"], result["out_ref"], rtol=5e-3, atol=5e-3)
    assert passed, f"out mismatch: {summary}"


@pytest.mark.soc("950")
def test_smoke_bsnd():
    run_case(layout="BSND", batch=2, seq=8, heads=6, seed=3)


@pytest.mark.soc("950")
@pytest.mark.parametrize("layout", ["BSND", "BNSD", "TND"])
@pytest.mark.parametrize("shape", [(1, 8, 6), (2, 8, 6), (2, 1, 3), (3, 5, 2)])
def test_layouts(layout, shape):
    batch, seq, heads = shape
    run_case(layout=layout, batch=batch, seq=seq, heads=heads, seed=17 + batch + seq,
             seq_lens=[seq - 1, seq, 1] if layout == "TND" and batch == 3 else None)


@pytest.mark.soc("950")
@pytest.mark.parametrize("layout", ["BSND", "BNSD", "TND"])
def test_state_indices_and_accepted(layout):
    run_case(layout=layout, batch=2, seq=8, heads=4, seed=41, pool=40,
             use_indices=True, accepted=[0, 5])


@pytest.mark.soc("950")
def test_bf16_state():
    run_case(layout="BSND", batch=2, seq=8, heads=4, seed=53, state_dtype="bf16")


@pytest.mark.soc("950")
@pytest.mark.parametrize("rows", [32, 64])
def test_row_items(rows):
    run_case(layout="BSND", batch=3, seq=12, heads=5, seed=67, rows=rows)


@pytest.mark.soc("950")
def test_single_token_sequences():
    run_case(layout="BSND", batch=3, seq=1, heads=7, seed=101, accepted=[0, 1, 1])
    run_case(layout="TND", batch=2, seq=1, heads=3, seed=103, seq_lens=[1, 1])


@pytest.mark.soc("950")
def test_max_heads_and_accepted_edges():
    run_case(layout="BSND", batch=1, seq=4, heads=96, seed=107)
    run_case(layout="BNSD", batch=2, seq=6, heads=5, seed=109, accepted=[6, 0],
             pool=20, use_indices=True)


@pytest.mark.soc("950")
def test_long_sequence():
    run_case(layout="BSND", batch=1, seq=33, heads=2, seed=113)


@pytest.mark.soc("950")
def test_worker_sweep():
    for workers in (1, 3, 13, 56, 64):
        run_case(layout="TND", batch=2, seq=6, heads=3, seed=71, workers=workers,
                 seq_lens=[6, 5])


@pytest.mark.soc("950")
def test_head_dim_blocks():
    run_case(layout="BNSD", batch=1, seq=4, heads=17, seed=83)
    run_case(layout="BSND", batch=4, seq=3, heads=33, seed=89)
