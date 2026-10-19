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
# Precision test for inplace_add_rms_norm (PyPTO).
#
# Test levels (loaded from test_cases.json):
#   - level0: bf16, [1,16,7168]    smallest functional
#   - level1: bf16, [16,128,7168]  B*S=2048 P0
#   - level2: bf16, [8,128,7168]   B*S=1024 min
#   - level3: bf16, [64,128,7168]  B*S=8192 max
#   - level4: bf16, [144,1,7168]   S=1
#   - level5: bf16, [1,1024,7168]  B=1 boundary
#
# Compares the PyPTO kernel output against the pure-PyTorch golden using
# numpy.testing.assert_allclose with atol/rtol from test_cases.json.
# Also verifies inplace semantics:
#   1) x1/x2.data_ptr() unchanged across the call
#   2) returned final_y / x_add are aliases of x1 / x2
#   3) rstd is a fresh tensor (independent data_ptr)
#   4) x1 / x2 contents overwritten
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

from experimental.vector.InplaceAddRmsNorm.inplace_add_rms_norm_impl import npu_inplace_add_rms_norm  # noqa: E402
from inplace_add_rms_norm_golden import inplace_add_rms_norm_golden

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Required test levels - default set covers all P0 shapes.
REQUIRED_LEVELS = ("level0", "level1", "level2", "level3")


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
    x1_shape = tuple(inp["x1_shape"])
    x2_shape = tuple(inp["x2_shape"])
    gamma_shape = tuple(inp["gamma_shape"])

    x1 = (torch.randn(x1_shape, dtype=torch.float32) * 0.1).to(dtype).to(device)
    x2 = (torch.randn(x2_shape, dtype=torch.float32) * 0.1).to(dtype).to(device)
    gamma = torch.randn(gamma_shape, dtype=torch.float32).to(dtype).to(device)
    return x1, x2, gamma


def _verify_inplace_semantics(x1, x2, fy_i, xa_i, rstd_i, p1_before, p2_before, case,
                              x1_init_cpu, x2_init_cpu):
    assert x1.data_ptr() == p1_before, \
        f"[INPLACE FAIL] x1 data_ptr changed ({p1_before} -> {x1.data_ptr()})"
    assert x2.data_ptr() == p2_before, \
        f"[INPLACE FAIL] x2 data_ptr changed ({p2_before} -> {x2.data_ptr()})"
    assert fy_i.data_ptr() == x1.data_ptr(), \
        "[INPLACE FAIL] returned final_y is not alias of x1"
    assert xa_i.data_ptr() == x2.data_ptr(), \
        "[INPLACE FAIL] returned x_add is not alias of x2"
    assert rstd_i.data_ptr() not in (p1_before, p2_before), \
        "[INPLACE FAIL] rstd should be a fresh tensor"
    assert tuple(rstd_i.shape) == (case["input"]["x1_shape"][0],
                                   case["input"]["x1_shape"][1], 1), \
        f"rstd shape mismatch: {tuple(rstd_i.shape)}"
    assert x1.dtype == torch.bfloat16
    assert x2.dtype == torch.bfloat16
    assert rstd_i.dtype == torch.bfloat16
    actual_y = x1.detach().cpu().float().numpy()
    actual_xadd = x2.detach().cpu().float().numpy()
    diff_x1 = float(np.abs(actual_y - x1_init_cpu.float().numpy()).max())
    diff_x2 = float(np.abs(actual_xadd - x2_init_cpu.float().numpy()).max())
    assert diff_x1 > 0, "[INPLACE FAIL] x1 content not overwritten"
    assert diff_x2 > 0, "[INPLACE FAIL] x2 content not overwritten"
    print(f"  [inplace] data_ptr unchanged + alias OK; overwrite diff "
          f"x1={diff_x1:.4f} x2={diff_x2:.4f}")
    return actual_y, actual_xadd


def _compare_and_report(actual_y, actual_xadd, actual_rstd, expected_y, expected_xadd,
                         expected_rstd, atol, rtol):
    def _err(a, e):
        d = np.abs(a - e)
        denom = np.maximum(np.abs(e), 1e-12)
        return float(d.max()), float((d / denom).max())
    y_abs, y_rel = _err(actual_y, expected_y)
    xa_abs, xa_rel = _err(actual_xadd, expected_xadd)
    rstd_abs, rstd_rel = _err(actual_rstd, expected_rstd)
    print(f"  [y]      max_abs={y_abs:.6e} max_rel={y_rel:.6e}")
    print(f"  [x_add]  max_abs={xa_abs:.6e} max_rel={xa_rel:.6e}")
    print(f"  [rstd]   max_abs={rstd_abs:.6e} max_rel={rstd_rel:.6e}")
    try:
        assert_allclose(actual_y, expected_y, atol=atol, rtol=rtol)
        assert_allclose(actual_xadd, expected_xadd, atol=atol, rtol=rtol)
        assert_allclose(actual_rstd, expected_rstd, atol=atol, rtol=rtol)
        return True
    except AssertionError as exc:
        print(f"  FAIL: {exc}")
        return False


def _run_case(case: dict, device: str) -> bool:
    case_id = case["id"]
    rtol = case.get("rtol", 1e-2)
    atol = case.get("atol", 1e-2)
    eps = case.get("eps", 1.0e-6)

    print("=" * 60)
    print(f"Test: {case_id} - {case.get('description', '')}")
    print("=" * 60)

    x1, x2, gamma = _make_inputs(case, device)
    x1_g = x1.detach().cpu().clone()
    x2_g = x2.detach().cpu().clone()
    g_g = gamma.detach().cpu().clone()
    fy_g, xa_g, rstd_g = inplace_add_rms_norm_golden(x1_g, x2_g, g_g, eps)

    p1_before = x1.data_ptr()
    p2_before = x2.data_ptr()
    x1_init_cpu = x1.detach().cpu().clone()
    x2_init_cpu = x2.detach().cpu().clone()

    fy_i, xa_i, rstd_i = npu_inplace_add_rms_norm(x1, x2, gamma, eps)
    torch.npu.synchronize()

    actual_y, actual_xadd = _verify_inplace_semantics(
        x1, x2, fy_i, xa_i, rstd_i, p1_before, p2_before, case, x1_init_cpu, x2_init_cpu)

    actual_rstd = rstd_i.detach().cpu().float().numpy()
    expected_y = fy_g.float().numpy()
    expected_xadd = xa_g.float().numpy()
    expected_rstd = rstd_g.float().numpy()

    return _compare_and_report(actual_y, actual_xadd, actual_rstd,
                               expected_y, expected_xadd, expected_rstd, atol, rtol)


def test_level0(device: str) -> bool:
    """level0: bf16, [1,16,7168] smallest functional."""
    cases = [c for c in _load_cases() if c["id"] == "level0"]
    return _run_case(cases[0], device)


@pytest.mark.soc("950")
def test_level1(device: str) -> bool:
    cases = [c for c in _load_cases() if c["id"] == "level1"]
    return _run_case(cases[0], device)


def test_level2(device: str) -> bool:
    cases = [c for c in _load_cases() if c["id"] == "level2"]
    return _run_case(cases[0], device)


@pytest.mark.soc("950")
def test_level3(device: str) -> bool:
    cases = [c for c in _load_cases() if c["id"] == "level3"]
    return _run_case(cases[0], device)


def main() -> int:
    try:
        device = _device()
        print(f"Using device: {device}")

        all_ok = True
        # 取消注释你要跑的 level；默认跑 level0 + level1 + level4
        for runner in (test_level0, test_level1, test_level2, test_level3):
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
