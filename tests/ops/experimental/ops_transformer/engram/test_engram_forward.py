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
  * NPU PyPTO kernel output;
  * NPU benchmark from the small torch implementation at kernel dtype;
  * FP32 golden reference on CPU.

Run on NPU (直接调用):
    pytest test_engram_forward.py -v
    python test_engram_forward.py
    python test_engram_forward.py -k pypto       # 仅直接调用 pytest

推理模式 (return_cache=False, cache 返回 None, 仅比较 value_out):
    python test_engram_forward.py --no-cache
    pytest test_engram_forward.py -k pypto       # parametrize 已覆盖 True/False

ACLGraph 入图 (PyPTO + npugraph_ex):
    python test_engram_forward.py --acl
    pytest test_engram_forward.py -k acl
"""

import logging
import os
import sys

import torch
import pytest
import torch_npu
from torch._subclasses.fake_tensor import FakeTensor

try:
    from torch._dynamo import allow_in_graph
except Exception:

    def allow_in_graph(fn):
        return fn

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
    _get_device,
    engram_forward_with_cache,
)
from compare import _compare
from ops.pypto_tensor.experimental.ops_transformer.engram.engram_forward_impl import (
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

def run_engram_forward_case(b, s, m_h=4, h=1280, de=512, seed=42, return_cache=True):
    """Run one case. Returns dict of per-output PASS/FAIL booleans.

    return_cache=True (训练): 5 元组输出, 全量三方对比;
    return_cache=False (推理): wrapper 返回 (value_out, None, ...), 精度比较也只比较 value_out.
    """
    device = _get_device()

    inputs = _make_case(device, b, s, m_h, h, de, seed=seed)

    # Kernel: 恒返回 5 元组 (return_cache=False 时后 4 项为 None, zip 按 compare_names 截断)
    npu_outputs = engram_forward_wrapper(*inputs, return_cache=return_cache)

    # Benchmark: same inputs at low-precision dtype
    with torch.no_grad():
        benchmark_outputs = _forward_outputs(*inputs)

    # CPU FP64 golden: all tensor inputs widened
    golden_args = tuple(
        arg.detach().cpu().to(torch.float64)
        for arg in inputs
    )
    with torch.no_grad():
        golden_outputs = _forward_outputs(*golden_args)

    compare_names = FORWARD_NAMES if return_cache else FORWARD_NAMES[:1]
    results = {}
    for name, nt, bt, gt in zip(
        compare_names, npu_outputs, benchmark_outputs, golden_outputs
    ):
        result, mare, mere, rmse, small_value = _compare(nt, bt, gt)
        log.info(
            f"  {name:30s} MARE={mare:.4f} MERE={mere:.4f} "
            f"RMSE={rmse:.4f} SmallVal={small_value:.4f} [{result}]"
        )
        results[name] = result == "PASS"

    return results


# ═══════════════════════════════════════════════════════════════════
# Test case matrix: (b, s, m_h, h, de)
# return_cache 不进矩阵: pytest 由第二层 parametrize 覆盖, 直调 main() 由
# 命令行 --no-cache 控制 (与 --acl 同款风格)
# ═══════════════════════════════════════════════════════════════════

CASES = [
    pytest.param(1, 4096, 4, 1536, 640, id="b1_s4096_mhc4_h1536_de640"),
    pytest.param(1, 4099, 4, 2048, 1280, id="b1_s4099_mhc4_h2048_de1280"),
]


@pytest.mark.soc("910")
@pytest.mark.parametrize("return_cache", [True, False])
@pytest.mark.parametrize("b,s,m_h,h,de", CASES)
def test_engram_forward_pypto(b, s, m_h, h, de, return_cache):
    """Precision Standard 2.1 three-way comparison.

    return_cache=True: all 5 outputs; return_cache=False (推理): value_out only (cache 为 None).
    """
    results = run_engram_forward_case(b, s, m_h, h, de, return_cache=return_cache)
    expected = FORWARD_NAMES if return_cache else FORWARD_NAMES[:1]
    assert list(results.keys()) == expected, \
        f"return_cache={return_cache} should compare {expected}, got {list(results.keys())}"
    failed = [n for n, ok in results.items() if not ok]
    assert not failed, f"Precision check failed for: {failed}"
    log.info("[PRECISION_PASS]")


# ═══════════════════════════════════════════════════════════════════
# Direct execution entry point (no pytest needed)
# ═══════════════════════════════════════════════════════════════════

def main():
    is_acl = "--acl" in sys.argv
    no_cache = "--no-cache" in sys.argv
    mode = "ACLGRAPH" if is_acl else ("DIRECT (no-cache)" if no_cache else "DIRECT")
    log.info(f"=== Running {mode} mode ===")
    if is_acl and no_cache:
        log.info("  Note: ACLGraph op 签名固定训练模式, --no-cache 被忽略")
    all_pass = True
    for case in CASES:
        b, s, m_h, h, de = case.values
        name = case.id
        try:
            if is_acl:
                results = run_engram_forward_acl_case(b, s, m_h, h, de)
            else:
                results = run_engram_forward_case(b, s, m_h, h, de, return_cache=not no_cache)
            ok = all(results.values())
        except Exception as exc:
            log.info("  EXCEPTION in %s: %s", name, exc)
            ok = False
        log.info("  %-30s %s", name, "PASS" if ok else "FAIL")
        all_pass = all_pass and ok

    return 0 if all_pass else 1


# ═══════════════════════════════════════════════════════════════════
# ACLGraph 入图测试 (PyPTO + npugraph_ex)
# 参考 pypto/python/tests/st/operator/pg/test_pg_lightning_indexer_prolog_quant_hif8.py
# ═══════════════════════════════════════════════════════════════════
pyptolib = torch.library.Library("pypto", "FRAGMENT")
pyptolib.define(
    "engram_forward_graph("
    "Tensor hidden_states, Tensor embeddings, Tensor weight_key, Tensor weight_value, "
    "Tensor gamma_key, Tensor gamma_query, "
    "float clamp_value, float eps"
    ") -> (Tensor output, Tensor score_cache, Tensor key_cache, "
    "Tensor value_cache, Tensor gate_cache)"
)


class EngramForwardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        (
            hidden_states, embeddings, weight_key, weight_value,
            gamma_key, gamma_query, clamp_value, eps,
        ) = args
        return torch.ops.pypto.engram_forward_graph(
            hidden_states, embeddings, weight_key, weight_value,
            gamma_key, gamma_query, clamp_value, eps,
        )


@torch.library.impl(pyptolib, "engram_forward_graph", "Meta")
def _engram_forward_graph_meta(
    hidden_states, embeddings, weight_key, weight_value,
    gamma_key, gamma_query, clamp_value, eps,
):
    b, s, m, h = hidden_states.shape
    output = torch.empty((b, s, m, h), dtype=hidden_states.dtype, device="meta")
    score_cache = torch.empty((b, s, m), dtype=torch.float32, device="meta")
    key_cache = torch.empty((b, s, m, h), dtype=hidden_states.dtype, device="meta")
    value_cache = torch.empty((b, s, h), dtype=hidden_states.dtype, device="meta")
    gate_cache = torch.empty((b, s, m), dtype=torch.float32, device="meta")
    return output, score_cache, key_cache, value_cache, gate_cache


def _engram_forward_graph_pypto(
    hidden_states, embeddings, weight_key, weight_value,
    gamma_key, gamma_query, clamp_value, eps,
):
    """直接复用 engram_forward_wrapper (内部分配输出 + 调 kernel + reshape)."""
    if isinstance(hidden_states, FakeTensor):
        return _engram_forward_graph_meta(
            hidden_states, embeddings, weight_key, weight_value,
            gamma_key, gamma_query, clamp_value, eps,
        )
    return engram_forward_wrapper(
        hidden_states, embeddings, weight_key, weight_value,
        gamma_key, gamma_query, clamp_value, eps,
    )


try:
    _engram_forward_graph_pypto = allow_in_graph(_engram_forward_graph_pypto)
    torch.library.impl(pyptolib, "engram_forward_graph", "NPU")(_engram_forward_graph_pypto)
except Exception as _e:
    if "could not parse dispatch key: NPU" in str(_e):
        log.warning("Skip NPU registration: torchair not installed")
    else:
        log.warning(f"Skip: Unexpected error: {_e}")


def run_engram_forward_acl_case(b, s, m_h=4, h=1280, de=512, seed=42):
    """ACLGraph 入图: torch.compile(npugraph_ex), 同样三方精度对比."""
    device = _get_device()
    torch_npu.npu.config.allow_internal_format = True

    inputs = _make_case(device, b, s, m_h, h, de, seed=seed)

    # Benchmark + FP64 golden (与 run_engram_forward_case 同源)
    with torch.no_grad():
        benchmark_outputs = _forward_outputs(*inputs)
    golden_args = tuple(arg.detach().cpu().to(torch.float64) for arg in inputs)
    with torch.no_grad():
        golden_outputs = _forward_outputs(*golden_args)

    model = EngramForwardModel()
    compile_forward = torch.compile(model, fullgraph=True, backend="npugraph_ex", dynamic=False)
    npu_outputs = compile_forward(*inputs, CLAMP_VALUE_DEFAULT, EPS_DEFAULT)

    results = {}
    for name, nt, bt, gt in zip(FORWARD_NAMES, npu_outputs, benchmark_outputs, golden_outputs):
        result, mare, mere, rmse, small_value = _compare(nt, bt, gt)
        log.info(
            f"  [ACL] {name:26s} MARE={mare:.4f} MERE={mere:.4f} "
            f"RMSE={rmse:.4f} SmallVal={small_value:.4f} [{result}]"
        )
        results[name] = result == "PASS"
    return results


CLAMP_VALUE_DEFAULT = 1e-6
EPS_DEFAULT = 1e-6


@pytest.mark.soc("910")
@pytest.mark.parametrize("b,s,m_h,h,de", CASES)
def test_engram_forward_acl(b, s, m_h, h, de):
    """ACLGraph 入图精度测试 (三方对比)."""
    results = run_engram_forward_acl_case(b, s, m_h, h, de)
    failed = [n for n, ok in results.items() if not ok]
    assert not failed, f"[ACL] Precision check failed for: {failed}"
    log.info("[ACL_PRECISION_PASS]")


if __name__ == "__main__":
    sys.exit(main())
