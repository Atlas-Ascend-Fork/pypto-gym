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

"""commit_recurrent_kda_replayssm PyPTO-Pro test: one test function per case.

The kernel and the host wrapper live under
``op_kernel/commit_recurrent_kda_replayssm.py``; the CPU golden and the numeric
comparator live under ``tests/assets/``.  Each ``test_xxx`` prepares one case's
inputs, runs the public wrapper on the NPU and compares the in-place result
against the independent FP32 CPU golden (``atol = rtol = 1e-6``).

Run: ``pytest ops/pro/kimi_k3/commit_recurrent_kda_replayssm/tests --mode jit -q``
"""

import importlib.util
import logging
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from commit_recurrent_kda_replayssm import commit_recurrent_kda_replayssm

_GEN_GOLDEN = Path(__file__).resolve().parent / "assets" / "gen_golden.py"
_spec = importlib.util.spec_from_file_location(
    "commit_recurrent_kda_replayssm_gen_golden", _GEN_GOLDEN)
golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(golden)
assert_close = golden.assert_close

logging.basicConfig(level=logging.INFO, format="%(message)s")
_LOG = logging.getLogger(__name__)

DEVICE_ID = os.environ.get("TILE_FWK_DEVICE_ID", "0")
DEVICE = f"npu:{DEVICE_ID}"

# name -> (batch, seq, heads, accepted)
CASES = {
    "bench_shape_accepted4": (16, 8, 6, 4),
    "accepted1": (16, 8, 6, 1),
    "accepted_all": (16, 8, 6, 8),
    "single_batch_head": (1, 8, 1, 4),
    "mixed_accepted": (7, 4, 6, [0, 4, 1, 0, 2, 3, 0]),
    "wide_heads": (2, 8, 32, 4),
    "accepted_zero": (4, 16, 6, [0, 16, 3, 0]),
    "large_batch": (32, 8, 6, 4),
    "seq_one": (16, 1, 6, 1),
    "tiny": (3, 3, 2, 2),
    # S larger than the fused record block -> multi-chunk path
    "multichunk_mixed": (4, 32, 6, [0, 32, 20, 5]),
    "multichunk_tail": (2, 40, 3, [40, 39]),
    "single_head_full": (8, 8, 1, 8),
}

ROW_BLOCKS = (16, 32, 64, 128)


def require_a5(device=None):
    if device is None:
        device = DEVICE
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Device {name} is not A5 (Ascend950). Skip.")


def _to_npu(arr, dtype):
    return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype).to(DEVICE)


def _device_inputs(state, ru, rk, rd, acc):
    return (
        torch.from_numpy(state).to(DEVICE),
        _to_npu(ru, torch.float16),
        _to_npu(rk, torch.float16),
        _to_npu(rd, torch.float32),
        _to_npu(acc, torch.int32),
    )


def _run_case(name, rows=64):
    batch, seq, heads, accepted = CASES[name]
    require_a5()
    state, ru, rk, rd, acc = golden.make_inputs(batch, seq, heads, accepted=accepted)
    expect = golden.reference(state, ru, rk, rd, acc)
    state_d, ru_d, rk_d, rd_d, acc_d = _device_inputs(state, ru, rk, rd, acc)

    ru_before = ru_d.clone()
    acc_before = acc_d.clone()
    commit_recurrent_kda_replayssm(state_d, ru_d, rk_d, rd_d, acc_d, rows=rows)
    torch.npu.synchronize()

    rel = assert_close(state_d.cpu().numpy(), expect)
    _LOG.info("  [%s rows=%d] floored rel err %.3e", name, rows, rel)
    assert rel < 1e-5, f"floored rel err {rel:.3e}"
    assert torch.equal(ru_d, ru_before), "replay_u was modified"
    assert torch.equal(acc_d, acc_before), "num_accepted_tokens was modified"


@pytest.mark.soc("950")
@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_golden(name):
    _run_case(name, rows=64)


@pytest.mark.soc("950")
@pytest.mark.parametrize("rows", ROW_BLOCKS)
def test_row_blocks(rows):
    _run_case("mixed_accepted", rows=rows)


@pytest.mark.soc("950")
def test_repeatable():
    """Repeated identical launches must be deterministic (no loop-carried race)."""
    require_a5()
    state, ru, rk, rd, acc = golden.make_inputs(
        7, 8, 6, accepted=[0, 8, 1, 0, 4, 2, 0])
    expect = golden.reference(state, ru, rk, rd, acc)
    state_d, ru_d, rk_d, rd_d, acc_d = _device_inputs(state, ru, rk, rd, acc)
    for _ in range(15):
        state_d.copy_(torch.from_numpy(state).to(DEVICE))
        commit_recurrent_kda_replayssm(state_d, ru_d, rk_d, rd_d, acc_d)
        torch.npu.synchronize()
        np.testing.assert_array_equal(state_d.cpu().numpy(), expect)


@pytest.mark.soc("950")
def test_empty_recurrence_is_identity():
    """A batch whose accepted count is zero keeps its checkpoint bit for bit."""
    require_a5()
    state, ru, rk, rd, acc = golden.make_inputs(4, 5, 2, accepted=0)
    state = np.random.default_rng(7).standard_normal(state.shape).astype(np.float32)
    state_d, ru_d, rk_d, rd_d, acc_d = _device_inputs(state, ru, rk, rd, acc)
    commit_recurrent_kda_replayssm(state_d, ru_d, rk_d, rd_d, acc_d)
    torch.npu.synchronize()
    np.testing.assert_array_equal(state_d.cpu().numpy(), state)


@pytest.mark.soc("950")
def test_rejects_unsupported_row_block():
    require_a5()
    state, ru, rk, rd, acc = golden.make_inputs(2, 4, 2, accepted=2)
    state_d, ru_d, rk_d, rd_d, acc_d = _device_inputs(state, ru, rk, rd, acc)
    with pytest.raises(AssertionError):
        commit_recurrent_kda_replayssm(state_d, ru_d, rk_d, rd_d, acc_d, rows=96)
