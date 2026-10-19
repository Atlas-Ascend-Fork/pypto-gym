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
"""Pytest harness for engram kernel.

Precision Standard 2.1 three-way comparison for all 5 outputs:
  * NPU PyPTO-Pro kernel output;
  * NPU benchmark from the small torch implementation at kernel dtype;
  * FP32 golden reference on CPU.

Run on NPU:
    pytest test_engram_forward_pypto_pro.py -v
or direct:
    python test_engram_forward_pypto_pro.py
"""

import logging
import os
import sys

import torch
import pytest

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Ensure we can import the golden reference that ships next to this test.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Locate the repo root (the directory containing 'src') and add it to sys.path
# so the kernel impl under src/pypto_gym/.../engram/ is importable as a package.
_REPO_ROOT = _THIS_DIR
while not os.path.isdir(os.path.join(_REPO_ROOT, "src")):
    _REPO_ROOT = os.path.dirname(_REPO_ROOT)
    if _REPO_ROOT == os.path.dirname(_REPO_ROOT):
        break
_SRC_DIR = _REPO_ROOT
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
_GOLDEN_DIR = os.path.join(_REPO_ROOT, "tests", "ops", "experimental",
"ops_transformer", "engram")
if _GOLDEN_DIR not in sys.path:
    sys.path.insert(0, _GOLDEN_DIR)

from engram_golden import (
    _get_device,
    engram_forward_with_cache,
)
from compare import _compare
from ops.pypto_pro.experimental.ops_transformer.engram.engram_forward_impl import (
    engram_forward_wrapper,
)

FORWARD_NAMES = ["value_out", "score_back", "key_back", "value_back", "gate_back"]


# ═══════════════════════════════════════════════════════════════════
# Golden references shared with the backward test
# ═══════════════════════════════════════════════════════════════════

def _forward_outputs(hs, emb, kpw, vpw, kg, qg):
    """Return forward outputs in the kernel's output order."""
    value_out, cache = engram_forward_with_cache(hs, emb, kpw, vpw, kg, qg)
    if hs.dtype == torch.bfloat16:
        value_out = value_out.to(torch.bfloat16)
    return (
        value_out,
        cache["scores"],
        cache["keys"],
        cache["value"],
        cache["gates"],
    )


# ═══════════════════════════════════════════════════════════════════
# Input construction
# ═══════════════════════════════════════════════════════════════════

def _make_case(device, b, s, m_h=4, h=1280, de=512, dtype=torch.bfloat16, seed=42):
    """Construct forward inputs at the kernel input dtype."""
    torch.manual_seed(seed)
    hidden_states = torch.randn(b, s, m_h, h, dtype=dtype, device=device)
    embeddings = torch.randn(b, s, de, dtype=dtype, device=device)
    key_proj_weights = torch.randn(m_h, de, h, dtype=dtype, device=device)
    value_proj_weights = torch.randn(de, h, dtype=dtype, device=device)
    key_gamma = torch.randn(m_h, h, dtype=dtype, device=device)
    query_gamma = torch.randn(m_h, h, dtype=dtype, device=device)
    return (
        hidden_states,
        embeddings,
        key_proj_weights,
        value_proj_weights,
        key_gamma,
        query_gamma,
    )


# ═══════════════════════════════════════════════════════════════════
# Core case runner: kernel vs benchmark vs FP64 golden
# ═══════════════════════════════════════════════════════════════════

def run_engram_forward_case(b, s, m_h=4, h=1280, de=512, seed=42):
    """Run one case. Returns dict of per-output PASS/FAIL booleans."""
    device = _get_device()

    inputs = _make_case(device, b, s, m_h, h, de, seed=seed)

    # Kernel
    npu_outputs = engram_forward_wrapper(*inputs)

    # Benchmark: same inputs at low-precision dtype
    with torch.no_grad():
        benchmark_outputs = _forward_outputs(*inputs)

    # CPU FP32 golden: all tensor inputs widened
    golden_args = tuple(
        arg.detach().cpu().to(torch.float64)
        for arg in inputs
    )
    with torch.no_grad():
        golden_outputs = _forward_outputs(*golden_args)

    results = {}
    for name, nt, bt, gt in zip(
        FORWARD_NAMES, npu_outputs, benchmark_outputs, golden_outputs
    ):
        result, mare, mere, rmse, small_value = _compare(nt, bt, gt)
        log.info(
            f"  {name:30s} MARE={mare:.4f} MERE={mere:.4f} "
            f"RMSE={rmse:.4f} SmallVal={small_value:.4f} [{result}]"
        )
        results[name] = result == "PASS"

    return results


# ═══════════════════════════════════════════════════════════════════
# Test case matrix
# ═══════════════════════════════════════════════════════════════════

CASES = [
    pytest.param(1, 512, 4, 1280, 512, id="b1_s512_mhc4_hiddensize1280_dime512"),
    pytest.param(16, 512, 4, 2560, 512, id="b16_s512_mhc4_hiddensize2560_dime512"),
    pytest.param(1, 127, 4, 1280, 1024, id="b1_s127_mhc4_hiddensize1280_dime1024"),
    pytest.param(1, 1023, 4, 2560, 1024, id="b1_s1023_mhc4_hiddensize2560_dime1024"),
    pytest.param(5, 513, 16, 2560, 512, id="b5_s513_mhc16_hiddensize2560_dime512"),
]


@pytest.mark.soc("950")
@pytest.mark.parametrize("b,s,m_h,h,de", CASES)
def test_engram_forward_pypto_pro(b, s, m_h, h, de):
    """Precision Standard 2.1 three-way comparison for all 5 outputs."""
    results = run_engram_forward_case(b, s, m_h, h, de)
    failed = [n for n, ok in results.items() if not ok]
    assert not failed, f"Precision check failed for: {failed}"
    log.info("[PRECISION_PASS]")


# ═══════════════════════════════════════════════════════════════════
# Direct execution entry point (no pytest needed)
# ═══════════════════════════════════════════════════════════════════

def main():
    all_pass = True
    for case in CASES:
        b, s, m_h, h, de = case.values
        name = case.id
        try:
            results = run_engram_forward_case(b, s, m_h, h, de)
            ok = all(results.values())
        except Exception as exc:
            log.info("  EXCEPTION in %s: %s", name, exc)
            ok = False
        log.info("  %-30s %s", name, "PASS" if ok else "FAIL")
        all_pass = all_pass and ok

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
