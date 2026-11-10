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

"""engram_gate PyPTO-Pro test: one test function per case, plus a ``main`` entry.

The kernel and the host wrapper live under ``op_kernel/engram_gate.py``; the CPU
golden and the Scheme A comparator live under ``tests/assets/``. Each test
prepares one case's inputs, runs the public wrapper on the NPU and compares the
output against the independent CPU golden with the official Scheme A checker.

Besides the five SPEC cases, the suite covers hc/dim boundaries, all-true and
single-true image masks, non-contiguous inputs, and the eps/clamp contract.
"""

import logging
import os
import sys

import pytest
import torch

# tests/ , tests/assets/ and op_kernel/ all sit next to each other.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OP_DIR = os.path.dirname(_THIS_DIR)
# 强制把本算子的目录放到 sys.path 最前: 根 conftest 已注入原 jit-only 算子的
# 同名模块, 仅做 membership 判断会解析到旧模块.
for _path in (_THIS_DIR, os.path.join(_THIS_DIR, "assets"),
              os.path.join(_OP_DIR, "op_kernel")):
    while _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

import engram_gate_golden_cpu  # noqa: E402
from engram_gate_precision import check_precision  # noqa: E402

try:
    from op_caller import get_op_caller  # noqa: E402

    engram_gate_wrapper = get_op_caller(
        "engram_gate", jit_attr="engram_gate_wrapper",
        aclnn_attr="engram_gate_wrapper")
except ImportError:  # standalone run outside pytest (jit only)
    from engram_gate import engram_gate_wrapper  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
_LOG = logging.getLogger(__name__)

pytestmark = pytest.mark.soc("950")

# name, t, hc, dim, seed, mask_kind
_CASES = {
    "partial_image_mask": (5, 4, 5120, 20260925, "partial"),
    "model_token_count_nomask": (72, 4, 5120, 20260992, "all_false"),
    "chunked_one_chunk_tail_one": (3, 4, 8449, 20269372, "all_false"),
    "model_dim_plus_one_partial_mask": (5, 4, 5121, 20266041, "partial"),
    # Mirrors the default compile-only shape (2, 4, 5120) with no masked token.
    "compile_default_shape": (2, 4, 5120, 20260922, "all_false"),
}

# Delivery-form boundary coverage: hc/dim edges and mask branches.
_EXTRA_CASES = {
    "t1_hc1_d64": (1, 1, 64, 20261001, "all_false"),
    "hc3_d65_partial": (2, 3, 65, 20261002, "partial"),
    "hc5_d63_alltrue": (2, 5, 63, 20261003, "all_true"),
    "hc8_d8512_single_true": (1, 8, 8512, 20261004, "single_true"),
    "hc16_d5121_partial": (2, 16, 5121, 20261005, "partial"),
    "hc32_d129": (2, 32, 129, 20261006, "all_false"),
    "hc56_d64": (1, 56, 64, 20261007, "all_false"),
}


def _run_case_spec(t, hc, dim, seed, mask_kind, name=""):
    """Run one shape/mask spec through the public wrapper and compare."""
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", 0)))
    cpu_inputs = engram_gate_golden_cpu.make_inputs(
        {"t": t, "hc": hc, "dim": dim, "seed": seed, "mask_kind": mask_kind})
    expected = engram_gate_golden_cpu.reference(cpu_inputs)
    actual = engram_gate_wrapper(*[item.to("npu") for item in cpu_inputs])
    torch.npu.synchronize()
    passed, summary = check_precision(actual.cpu(), expected)
    _LOG.info("[%s] status=%s :: %s",
              name or f"t{t}_hc{hc}_d{dim}", "PASS" if passed else "FAIL", summary)
    if not passed:
        raise AssertionError(f"{name or (t, hc, dim)} precision failed: {summary}")


def _run_case(name):
    t, hc, dim, seed, mask_kind = _CASES[name]
    _run_case_spec(t, hc, dim, seed, mask_kind, name)


def test_partial_image_mask():
    _run_case("partial_image_mask")


def test_model_token_count_nomask():
    _run_case("model_token_count_nomask")


def test_chunked_one_chunk_tail_one():
    _run_case("chunked_one_chunk_tail_one")


def test_model_dim_plus_one_partial_mask():
    _run_case("model_dim_plus_one_partial_mask")


def test_compile_default_shape():
    _run_case("compile_default_shape")


@pytest.mark.parametrize("name", list(_EXTRA_CASES), ids=list(_EXTRA_CASES))
def test_boundary_cases(name):
    t, hc, dim, seed, mask_kind = _EXTRA_CASES[name]
    _run_case_spec(t, hc, dim, seed, mask_kind, name)


def test_non_contiguous_inputs():
    """Strided token/dim views must produce the same values as contiguous inputs."""
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", 0)))
    t, hc, dim = 5, 4, 5120
    cpu_inputs = engram_gate_golden_cpu.make_inputs(
        {"t": t, "hc": hc, "dim": dim, "seed": 20261008, "mask_kind": "partial"})
    expected = engram_gate_golden_cpu.reference(cpu_inputs)
    x, key, value, weight, image_mask = cpu_inputs

    def _strided_last(src):
        base = torch.empty(*src.shape[:-1], src.shape[-1] * 2, dtype=src.dtype)
        base[..., ::2] = src
        base[..., 1::2] = 0
        return base[..., ::2]

    strided = [_strided_last(x), _strided_last(key), _strided_last(value),
               _strided_last(weight), image_mask]
    actual = engram_gate_wrapper(*[item.to("npu") for item in strided])
    torch.npu.synchronize()
    passed, summary = check_precision(actual.cpu(), expected)
    _LOG.info("[non_contiguous] status=%s :: %s", "PASS" if passed else "FAIL", summary)
    if not passed:
        raise AssertionError(f"non_contiguous precision failed: {summary}")


def test_eps_contract():
    """eps/clamp_value other than 1e-6 are rejected by both modes."""
    t, hc, dim = 1, 1, 64
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", 0)))
    cpu_inputs = engram_gate_golden_cpu.make_inputs(
        {"t": t, "hc": hc, "dim": dim, "seed": 20261009, "mask_kind": "all_false"})
    device_inputs = [item.to("npu") for item in cpu_inputs]
    with pytest.raises(ValueError):
        engram_gate_wrapper(*device_inputs, eps=1.0e-5)
    with pytest.raises(ValueError):
        engram_gate_wrapper(*device_inputs, clamp_value=1.0e-5)


def main():
    """Run every SPEC case by calling its test function, in the same order as the suite."""
    test_partial_image_mask()
    test_model_token_count_nomask()
    test_chunked_one_chunk_tail_one()
    test_model_dim_plus_one_partial_mask()
    test_compile_default_shape()
    for name in _EXTRA_CASES:
        test_boundary_cases(name)
    test_non_contiguous_inputs()
    test_eps_contract()


if __name__ == "__main__":
    raise SystemExit(main())
