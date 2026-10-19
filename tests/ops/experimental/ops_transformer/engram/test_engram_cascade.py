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
"""Pytest harness for engram cascade kernel (forward → backward).

Precision Standard 2.1 three-way comparison for all 6 gradients over the full
forward → backward cascade. 三条路径全部走 torch.autograd 自动反向:
  * NPU PyPTO kernel: EngramFunc (wrapper fwd + bwd) → autograd backward (被测);
  * NPU benchmark / CPU golden: forward_with_cache + autograd backward.

Three paths share the same BF16 inputs (no FP64 master), so the npu kernel and
the golden see identical input values; 三路径都只写 forward, 反向由 autograd 触发.

Run on NPU:
    pytest test_engram_cascade.py -v
or direct:
    python test_engram_cascade.py
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

from engram_golden import (
    engram_forward_with_cache,
    _get_device,
)
from compare import _compare
from engram_autograd import engram_autograd

GRAD_NAMES = [
    "grad_hidden_states",
    "grad_embeddings",
    "grad_key_proj_weights",
    "grad_value_proj_weights",
    "grad_key_gamma",
    "grad_query_gamma",
]


# ═══════════════════════════════════════════════════════════════════
# Input construction
# ═══════════════════════════════════════════════════════════════════

def _make_case(device, b, s, m_h=4, h=1280, de=512, dtype=torch.bfloat16, seed=42):
    """Construct cascade inputs: 6 forward inputs + grad_output, all BF16.

    Three paths (npu / benchmark / golden) share this single BF16 draw — no
    FP64 master — so the npu kernel and the golden see identical values.
    gamma = 1; weights ×0.5 to prevent BF16 matmul overflow.
    """
    torch.manual_seed(seed)
    hidden_states = torch.randn(b, s, m_h, h, dtype=dtype, device=device)
    embeddings = torch.randn(b, s, de, dtype=dtype, device=device)
    key_proj_weights = torch.randn(m_h, de, h, dtype=dtype, device=device) * 0.5
    value_proj_weights = torch.randn(de, h, dtype=dtype, device=device) * 0.5
    key_gamma = torch.ones(m_h, h, dtype=dtype, device=device)
    query_gamma = torch.ones(m_h, h, dtype=dtype, device=device)
    grad_output = torch.randn(b, s, m_h, h, dtype=dtype, device=device)
    return (hidden_states, embeddings, key_proj_weights, value_proj_weights,
            key_gamma, query_gamma, grad_output)


# ═══════════════════════════════════════════════════════════════════
# Core case runner: kernel cascade vs autograd benchmark vs FP32 golden
# ═══════════════════════════════════════════════════════════════════

def run_engram_cascade_case(b, s, m_h=4, h=1280, de=512):
    """Run one cascade case. Returns dict of per-gradient PASS/FAIL booleans."""
    device = _get_device()
    clamp_value, eps = 1e-6, 1e-6

    hidden, emb, kpw, vpw, kg, qg, grad_output = _make_case(device, b, s, m_h, h, de)
    leaves = (hidden, emb, kpw, vpw, kg, qg)  # 顺序与 GRAD_NAMES 一一对应

    # npu: EngramFunc forward → autograd backward (被测级联, 自动微分)
    npu_leaves = [t.clone().requires_grad_(True) for t in leaves]
    npu_value_out = engram_autograd(*npu_leaves)[0]
    npu_value_out.backward(grad_output.to(npu_value_out.dtype))
    npu_grads = [t.grad for t in npu_leaves]

    # benchmark: forward_with_cache + autograd backward (NPU BF16)
    bm_leaves = [t.clone().requires_grad_(True) for t in leaves]
    bm_value_out, _ = engram_forward_with_cache(*bm_leaves, clamp_value, eps)
    bm_value_out.backward(grad_output.to(bm_value_out.dtype))
    bm_grads = [t.grad for t in bm_leaves]

    # golden: forward_with_cache + autograd backward (CPU FP32; 输入数值=同一份 BF16 升 FP32 表示)
    g_leaves = [t.detach().cpu().to(torch.float32).requires_grad_(True) for t in leaves]
    g_value_out, _ = engram_forward_with_cache(*g_leaves, clamp_value, eps)
    g_value_out.backward(grad_output.detach().cpu().to(torch.float32))
    g_grads = [t.grad for t in g_leaves]

    results = {}
    for name, nt, bt, gt in zip(GRAD_NAMES, npu_grads, bm_grads, g_grads):
        result, mare, mere, rmse, small_value = _compare(nt, bt, gt)
        log.info(
            f"  {name:30s} MARE={mare:.4f} MERE={mere:.4f} "
            f"RMSE={rmse:.4f} SmallVal={small_value:.4f} [{result}]"
        )
        results[name] = result == "PASS"
    return results


# ═══════════════════════════════════════════════════════════════════
# Test case matrix: (b, s, m_h, h, de)
# ═══════════════════════════════════════════════════════════════════

CASES = [
    pytest.param(1, 4096, 4, 1536, 640, id="b1_s4096_mhc4_h1536_de640"),
    pytest.param(1, 4096, 4, 2048, 1280, id="b1_s4096_mhc4_h2048_de1280"),
]


@pytest.mark.soc("910")
@pytest.mark.parametrize("b,s,m_h,h,de", CASES)
def test_engram_cascade_pypto(b, s, m_h, h, de):
    """Precision Standard 2.1 three-way comparison for all 6 gradients over the cascade."""
    results = run_engram_cascade_case(b, s, m_h, h, de)
    failed = [n for n, ok in results.items() if not ok]
    assert not failed, f"Precision check failed for: {failed}"


# ═══════════════════════════════════════════════════════════════════
# Direct execution entry point (no pytest needed)
# ═══════════════════════════════════════════════════════════════════

def main():
    all_pass = True
    for case in CASES:
        b, s, m_h, h, de = case.values
        name = case.id
        try:
            results = run_engram_cascade_case(b, s, m_h, h, de)
            ok = all(results.values())
        except Exception as exc:
            log.info("  EXCEPTION in %s: %s", name, exc)
            ok = False
        log.info("  %-30s %s", name, "PASS" if ok else "FAIL")
        all_pass = all_pass and ok

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
