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
"""Pytest harness for engram_backward kernel.

Precision Standard 2.1 three-way comparison for all 6 gradient outputs:
  * NPU PyPTO kernel output;
  * NPU benchmark from the small torch implementation at kernel dtype;
  * FP64 golden output from the same small implementation on CPU.

Run on NPU (直接调用):
    pytest test_engram_backward.py -v
    python test_engram_backward.py
    python test_engram_backward.py -k pypto      # 仅直接调用 pytest

ACLGraph 入图 (PyPTO + npugraph_ex):
    python test_engram_backward.py --acl
    pytest test_engram_backward.py -k acl
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
    engram_backward_golden,
    engram_forward_with_cache,
    _get_device,
)
from compare import _compare
from ops.pypto_tensor.experimental.ops_transformer.engram.engram_backward_impl import (
    engram_backward_wrapper,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


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
    """Construct backward inputs: forward inputs → forward_with_cache → grad_output."""
    torch.manual_seed(seed)
    hidden_states = torch.randn(b, s, m_h, h, dtype=dtype, device=device)
    embeddings = torch.randn(b, s, de, dtype=dtype, device=device)
    key_proj_weights = torch.randn(m_h, de, h, dtype=dtype, device=device) * 0.5
    value_proj_weights = torch.randn(de, h, dtype=dtype, device=device) * 0.5
    key_gamma = torch.ones(m_h, h, dtype=dtype, device=device)
    query_gamma = torch.ones(m_h, h, dtype=dtype, device=device)

    clamp_value, eps = 1e-6, 1e-6
    with torch.no_grad():
        _, cache = engram_forward_with_cache(
            hidden_states, embeddings, key_proj_weights, value_proj_weights,
            key_gamma, query_gamma, clamp_value, eps,
        )

    grad_output = torch.randn(b, s, m_h, h, dtype=dtype, device=device)

    # Args in engram_backward_wrapper signature order
    kernel_args = (
        grad_output,
        hidden_states, embeddings, key_proj_weights, value_proj_weights,
        key_gamma, query_gamma,
        cache["scores"].float(), cache["gates"].float(),
        cache["keys"], cache["value"],
    )
    kwargs = {"clamp_value": clamp_value, "eps": eps}
    return kernel_args, kwargs


# ═══════════════════════════════════════════════════════════════════
# Core case runner: kernel vs benchmark vs FP64 golden
# ═══════════════════════════════════════════════════════════════════

def run_engram_backward_case(b, s, m_h=4, h=1280, de=512):
    """Run one case. Returns dict of per-gradient PASS/FAIL booleans."""
    device = _get_device()

    kernel_args, kwargs = _make_case(device, b, s, m_h, h, de)

    # Run kernel wrapper (scores / gates widened to FP32)
    npu_grads = engram_backward_wrapper(*kernel_args, **kwargs)

    # Benchmark: same inputs at low-precision dtype
    with torch.no_grad():
        benchmark_grads = engram_backward_golden(*kernel_args, **kwargs)

    # CPU FP64 golden: all tensor inputs widened
    golden_args = tuple(arg.detach().cpu().to(torch.float64) for arg in kernel_args)
    with torch.no_grad():
        golden_grads = engram_backward_golden(*golden_args, **kwargs)

    results = {}
    for name, nt, bt, gt in zip(GRAD_NAMES, npu_grads, benchmark_grads, golden_grads):
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
    pytest.param(1, 4099, 4, 2048, 1280, id="b1_s4099_mhc4_h2048_de1280"),
]


@pytest.mark.soc("910")
@pytest.mark.parametrize("b,s,m_h,h,de", CASES)
def test_engram_backward_pypto(b, s, m_h, h, de):
    """Precision Standard 2.1 three-way comparison for all 6 gradients."""
    results = run_engram_backward_case(b, s, m_h, h, de)
    failed = [n for n, ok in results.items() if not ok]
    assert not failed, f"Precision check failed for: {failed}"


# ═══════════════════════════════════════════════════════════════════
# Direct execution entry point (no pytest needed)
# ═══════════════════════════════════════════════════════════════════

def main():
    is_acl = "--acl" in sys.argv
    log.info(f"=== Running {'ACLGRAPH' if is_acl else 'DIRECT'} mode ===")
    all_pass = True
    for case in CASES:
        b, s, m_h, h, de = case.values
        name = case.id
        try:
            if is_acl:
                results = run_engram_backward_acl_case(b, s, m_h, h, de)
            else:
                results = run_engram_backward_case(b, s, m_h, h, de)
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
# op 签名顺序对齐 engram_backward_wrapper:
#   grad_out, hidden_states, embeddings, weight_key, weight_value,
#   gamma_key, gamma_query, score_cache, gate_cache, key_cache, value_cache,
#   clamp_value, eps
# ═══════════════════════════════════════════════════════════════════
pyptolib = torch.library.Library("pypto", "FRAGMENT")
pyptolib.define(
    "engram_backward_graph("
    "Tensor grad_out, Tensor hidden_states, Tensor embeddings, "
    "Tensor weight_key, Tensor weight_value, Tensor gamma_key, Tensor gamma_query, "
    "Tensor score_cache, Tensor gate_cache, Tensor key_cache, Tensor value_cache, "
    "float clamp_value, float eps"
    ") -> (Tensor grad_hidden, Tensor grad_embeddings, Tensor grad_weight_key, "
    "Tensor grad_weight_value, Tensor grad_gamma_key, Tensor grad_gamma_query)"
)


class EngramBackwardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        (
            grad_out, hidden_states, embeddings, weight_key, weight_value,
            gamma_key, gamma_query, score_cache, gate_cache, key_cache, value_cache,
            clamp_value, eps,
        ) = args
        return torch.ops.pypto.engram_backward_graph(
            grad_out, hidden_states, embeddings, weight_key, weight_value,
            gamma_key, gamma_query, score_cache, gate_cache, key_cache, value_cache,
            clamp_value, eps,
        )


@torch.library.impl(pyptolib, "engram_backward_graph", "Meta")
def _engram_backward_graph_meta(
    grad_out, hidden_states, embeddings, weight_key, weight_value,
    gamma_key, gamma_query, score_cache, gate_cache, key_cache, value_cache,
    clamp_value, eps,
):
    b, s, m, h = hidden_states.shape
    de = embeddings.shape[-1]
    grad_hidden = torch.empty((b, s, m, h), dtype=hidden_states.dtype, device="meta")
    grad_embeddings = torch.empty((b, s, de), dtype=hidden_states.dtype, device="meta")
    grad_weight_key = torch.empty_like(weight_key, memory_format=torch.preserve_format)
    grad_weight_value = torch.empty_like(weight_value, memory_format=torch.preserve_format)
    grad_gamma_key = torch.empty_like(gamma_key, memory_format=torch.preserve_format)
    grad_gamma_query = torch.empty_like(gamma_query, memory_format=torch.preserve_format)
    return (grad_hidden, grad_embeddings, grad_weight_key,
            grad_weight_value, grad_gamma_key, grad_gamma_query)


def _engram_backward_graph_pypto(
    grad_out, hidden_states, embeddings, weight_key, weight_value,
    gamma_key, gamma_query, score_cache, gate_cache, key_cache, value_cache,
    clamp_value, eps,
):
    """直接复用 engram_backward_wrapper (内部分配 output+workspace + 调 kernel + reshape)."""
    if isinstance(grad_out, FakeTensor):
        return _engram_backward_graph_meta(
            grad_out, hidden_states, embeddings, weight_key, weight_value,
            gamma_key, gamma_query, score_cache, gate_cache, key_cache, value_cache,
            clamp_value, eps,
        )
    return engram_backward_wrapper(
        grad_out, hidden_states, embeddings, weight_key, weight_value,
        gamma_key, gamma_query, score_cache, gate_cache, key_cache, value_cache,
        clamp_value, eps,
    )


try:
    _engram_backward_graph_pypto = allow_in_graph(_engram_backward_graph_pypto)
    torch.library.impl(pyptolib, "engram_backward_graph", "NPU")(_engram_backward_graph_pypto)
except Exception as _e:
    if "could not parse dispatch key: NPU" in str(_e):
        log.warning("Skip NPU registration: torchair not installed")
    else:
        log.warning(f"Skip: Unexpected error: {_e}")


def run_engram_backward_acl_case(b, s, m_h=4, h=1280, de=512):
    """ACLGraph 入图: torch.compile(npugraph_ex), 同样三方精度对比 6 个梯度."""
    device = _get_device()
    torch_npu.npu.config.allow_internal_format = True

    kernel_args, kwargs = _make_case(device, b, s, m_h, h, de)

    # Benchmark + FP64 golden (与 run_engram_backward_case 同源)
    with torch.no_grad():
        benchmark_grads = engram_backward_golden(*kernel_args, **kwargs)
    golden_args = tuple(arg.detach().cpu().to(torch.float64) for arg in kernel_args)
    with torch.no_grad():
        golden_grads = engram_backward_golden(*golden_args, **kwargs)

    model = EngramBackwardModel()
    compile_forward = torch.compile(model, fullgraph=True, backend="npugraph_ex", dynamic=False)
    npu_grads = compile_forward(*kernel_args, kwargs["clamp_value"], kwargs["eps"])

    results = {}
    for name, nt, bt, gt in zip(GRAD_NAMES, npu_grads, benchmark_grads, golden_grads):
        result, mare, mere, rmse, small_value = _compare(nt, bt, gt)
        log.info(
            f"  [ACL] {name:26s} MARE={mare:.4f} MERE={mere:.4f} "
            f"RMSE={rmse:.4f} SmallVal={small_value:.4f} [{result}]"
        )
        results[name] = result == "PASS"
    return results


@pytest.mark.soc("910")
@pytest.mark.parametrize("b,s,m_h,h,de", CASES)
def test_engram_backward_acl(b, s, m_h, h, de):
    """ACLGraph 入图精度测试 (三方对比)."""
    results = run_engram_backward_acl_case(b, s, m_h, h, de)
    failed = [n for n, ok in results.items() if not ok]
    assert not failed, f"[ACL] Precision check failed for: {failed}"
    log.info("[ACL_PRECISION_PASS]")


if __name__ == "__main__":
    sys.exit(main())
