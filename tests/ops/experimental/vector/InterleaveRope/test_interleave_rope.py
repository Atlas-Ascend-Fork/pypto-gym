#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Precision test for interleave_rope (PyPTO, ASC ops-transformer 兼容计算流).
#
# Contract (ASC half-split cos/sin pairing):
#   x_even[k]=x[...,2k], x_odd[k]=x[...,2k+1]
#   y[..., 0:32 ] = x_even·cos[..., 0:32] − x_odd·sin[..., 0:32]
#   y[..., 32:64] = x_even·sin[..., 32:64] + x_odd·cos[..., 32:64]
#   cos/sin 为任意逐位值（ASC example 风格，不假设前后半区相等）；
#   kernel 内部 FP32 计算，仅 x 做 gathermask，cos/sin 零 gather（半区 view）。
#
# Test levels (loaded from test_cases.json):
#   - level0: bf16, [1,1,1024,64],   S_cs=S       (smallest functional, n1 kernel)
#   - level1: bf16, [1,128,2048,64], S_cs=S       (typical multi-head, main kernel)
#   - level2: bf16, [2,128,4096,64], S_cs=1       (broadcast path)
#   - level3: fp16, [1,1,1024,64],   S_cs=S       (fp16 dispatch)
#   - level4: bf16, [4,128,8192,64], S_cs=S       (max dynamic axes, unroll kernel)
#   - level5: bf16, [4,128,2,64],    S_cs=S       (short sequence, short_s_btile)
#   - level6: bf16, [4,1,2,64],      S_cs=S       (short sequence N=1)
#   - level7: bf16, [2,1,8192,64],   S_cs=S       (large sequence N=1)
#   - level8: bf16, [2,128,8192,64], S_cs=S       (large sequence multi-head)
#   - level9: bf16, [8,128,1024,64], S_cs=S       (950 path, deinterleave)
#   - level10: bf16, [2,128,1024,64], S_cs=S      (950 path, deinterleave)
#
# Compares the PyPTO kernel output against the pure-PyTorch golden using
# arbitrary positional cos/sin data (strictest pairing test, ASC example style).
# -----------------------------------------------------------------------------
from __future__ import annotations

import os
import sys

_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))


import json
import os
import sys
import traceback

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401  # required to enable npu backend
from numpy.testing import assert_allclose

_HERE = os.path.dirname(os.path.abspath(__file__))

from experimental.vector.InterleaveRope.interleave_rope_impl import interleave_rope_wrapper  # noqa: E402
from interleave_rope_golden import interleave_rope_golden

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Required test levels for this targeted 950 branch coverage.
REQUIRED_LEVELS = ("level9", "level10")


def _device() -> str:
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    torch.npu.set_device(device_id)
    return f"npu:{device_id}"


def _load_cases(path: str = None) -> list:
    if path is None:
        path = os.path.join(_HERE, "test_cases.json")
    with open(path, "r") as f:
        cfg = json.load(f)
    cases = cfg["test_cases"]
    ids = {c["id"] for c in cases}
    for lvl in REQUIRED_LEVELS:
        if lvl not in ids:
            raise RuntimeError(f"missing required test level '{lvl}' in {path}")
    return cases


def _make_inputs(case: dict, device: str):
    seed = case.get("seed", 42)
    torch.manual_seed(seed)
    inp = case["input"]
    dtype = DTYPE_MAP[inp["dtype"]]
    x_shape = tuple(inp["shape"])
    cos_shape = tuple(inp["cos_shape"])
    sin_shape = tuple(inp["sin_shape"])

    x = torch.randn(x_shape, dtype=torch.float32).to(dtype).to(device)
    # cos/sin: 任意逐位值 clamp 到 [-1,1]（ASC example 风格——前后半区独立，
    # 不假设重复；对半区配对契约是最严格的测试数据）
    cos = torch.randn(cos_shape, dtype=torch.float32).clamp_(-1.0, 1.0).to(dtype).to(device)
    sin = torch.randn(sin_shape, dtype=torch.float32).clamp_(-1.0, 1.0).to(dtype).to(device)
    return x, cos, sin


def _run_case(case: dict, device: str) -> bool:
    case_id = case["id"]
    rtol = case.get("rtol", 0.0078125)
    atol = case.get("atol", 1e-4)

    print("=" * 60)
    print(f"Test: {case_id} - {case.get('description', '')}")
    print("=" * 60)

    x, cos, sin = _make_inputs(case, device)

    # Golden runs on CPU.
    y_g = interleave_rope_golden(
        x.detach().cpu(), cos.detach().cpu(), sin.detach().cpu()
    )

    # PyPTO kernel runs on NPU.
    y_out = interleave_rope_wrapper(x, cos, sin)
    torch.npu.synchronize()

    # Cast to fp32 numpy arrays for comparison.
    actual = y_out.detach().cpu().float().numpy()
    expected = y_g.float().numpy()

    diff = np.abs(actual - expected)
    max_abs = float(diff.max())
    denom = np.maximum(np.abs(expected), 1e-12)
    max_rel = float((diff / denom).max())
    print(f"  [y] shape={tuple(y_out.shape)} dtype={y_out.dtype}")
    print(f"  [y] max_abs_err={max_abs:.6e} max_rel_err={max_rel:.6e}")

    try:
        assert_allclose(actual, expected, atol=atol, rtol=rtol)
        return True
    except AssertionError as exc:
        print(f"  [y] FAIL: {exc}")
        return False


def test_level0(device: str) -> bool:
    cases = [c for c in _load_cases() if c["id"] == "level0"]
    return _run_case(cases[0], device)


def test_level1(device: str) -> bool:
    cases = [c for c in _load_cases() if c["id"] == "level1"]
    return _run_case(cases[0], device)


def test_level2(device: str) -> bool:
    cases = [c for c in _load_cases() if c["id"] == "level2"]
    return _run_case(cases[0], device)


def test_level3(device: str) -> bool:
    cases = [c for c in _load_cases() if c["id"] == "level3"]
    return _run_case(cases[0], device)


def test_level4(device: str) -> bool:
    cases = [c for c in _load_cases() if c["id"] == "level4"]
    return _run_case(cases[0], device)


def test_level5(device: str) -> bool:
    cases = [c for c in _load_cases() if c["id"] == "level5"]
    return _run_case(cases[0], device)


def test_level6(device: str) -> bool:
    cases = [c for c in _load_cases() if c["id"] == "level6"]
    return _run_case(cases[0], device)


def test_level7(device: str) -> bool:
    cases = [c for c in _load_cases() if c["id"] == "level7"]
    return _run_case(cases[0], device)


def test_level8(device: str) -> bool:
    """level8: bf16 path, [2,128,8192,64], S_cs=S (large sequence multi-head)."""
    cases = [c for c in _load_cases() if c["id"] == "level8"]
    return _run_case(cases[0], device)


@pytest.mark.soc("950")
def test_level9(device: str) -> bool:
    cases = [c for c in _load_cases() if c["id"] == "level9"]
    return _run_case(cases[0], device)


@pytest.mark.soc("950")
def test_level10(device: str) -> bool:
    cases = [c for c in _load_cases() if c["id"] == "level10"]
    return _run_case(cases[0], device)


def main() -> int:
    try:
        device = _device()
        print(f"Using device: {device}")

        all_ok = True
        runners = (test_level9, test_level10)
        for runner in runners:
            try:
                ok = runner(device)
            except Exception:
                traceback.print_exc()
                ok = False
            all_ok = all_ok and ok

        if all_ok:
            print("[PRECISION_PASS]")
            return 0
        print("[PRECISION_FAIL]")
        return 1
    except Exception:
        traceback.print_exc()
        print("[PRECISION_FAIL]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
