# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
"""engram_gate PyPTO-Pro test: one test function per case, plus a ``main`` entry.

The kernel and the host wrapper live under
``ops/pypto_pro/deepseek_v4_1/engram_gate/engram_gate.py``. Each
``test_xxx`` prepares one case's inputs, runs the public wrapper on the NPU and
compares against the independent CPU golden with the official Scheme A checker.

Run: ``python test_engram_gate.py``
"""

import logging
import os
import sys

import pytest
import torch

# Make the sibling golden/precision modules importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Locate the repo root (the directory containing 'src') so the kernel module is importable.
_REPO_ROOT = _HERE
while not os.path.isdir(os.path.join(_REPO_ROOT, "src")):
    _PARENT = os.path.dirname(_REPO_ROOT)
    if _PARENT == _REPO_ROOT:
        break
    _REPO_ROOT = _PARENT
_SRC_DIR = _REPO_ROOT
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import engram_gate_golden_cpu  # noqa: E402
from precision_compare import check_precision  # noqa: E402
from ops.pypto_pro.deepseek_v4_1.engram_gate.engram_gate import (  # noqa: E402
    engram_gate_wrapper,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
_LOG = logging.getLogger(__name__)

# name, t, hc, dim, seed, mask_kind
_CASES = {
    "partial_image_mask": (5, 4, 5120, 20260925, "partial"),
    "model_token_count_nomask": (72, 4, 5120, 20260992, "all_false"),
    "chunked_one_chunk_tail_one": (3, 4, 8449, 20269372, "all_false"),
    "model_dim_plus_one_partial_mask": (5, 4, 5121, 20266041, "partial"),
    # Mirrors the default compile-only shape (2, 4, 5120) with no masked token.
    "compile_default_shape": (2, 4, 5120, 20260922, "all_false"),
}


def _run_case(name):
    """Run one case on the NPU through the public wrapper and check its precision."""
    t, hc, dim, seed, mask_kind = _CASES[name]
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", 0)))
    cpu_inputs = engram_gate_golden_cpu.make_inputs(
        {"t": t, "hc": hc, "dim": dim, "seed": seed, "mask_kind": mask_kind})
    expected = engram_gate_golden_cpu.reference(cpu_inputs)
    actual = engram_gate_wrapper(*[item.to("npu") for item in cpu_inputs])
    torch.npu.synchronize()
    passed, summary = check_precision(actual.cpu(), expected)
    _LOG.info("[%s] status=%s :: %s", name, "PASS" if passed else "FAIL", summary)
    if not passed:
        raise AssertionError(f"{name} precision failed: {summary}")


@pytest.mark.soc("950")
def test_partial_image_mask():
    _run_case("partial_image_mask")


@pytest.mark.soc("950")
def test_model_token_count_nomask():
    _run_case("model_token_count_nomask")


@pytest.mark.soc("950")
def test_chunked_one_chunk_tail_one():
    _run_case("chunked_one_chunk_tail_one")


@pytest.mark.soc("950")
def test_model_dim_plus_one_partial_mask():
    _run_case("model_dim_plus_one_partial_mask")


@pytest.mark.soc("950")
def test_compile_default_shape():
    _run_case("compile_default_shape")


def main():
    test_partial_image_mask()
    test_model_token_count_nomask()
    test_chunked_one_chunk_tail_one()
    test_model_dim_plus_one_partial_mask()
    test_compile_default_shape()


if __name__ == "__main__":
    raise SystemExit(main())
