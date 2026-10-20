# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
"""mhc_pre_fuse PyPTO-Pro test: one test function per case, plus a ``main`` entry.

The kernel and the host wrapper live under
``op_kernel/mhc_pre_fuse.py``; the CPU golden and the Scheme A comparator live
under ``tests/assets/``.  Each ``test_xxx`` prepares one case's inputs, runs the
public wrapper on the NPU and compares the three outputs against the independent
CPU golden with the official Scheme A checker.

Run: ``python test_mhc_pre_fuse.py``
"""

import logging
import os
import sys

import pytest
import torch

# tests/ , tests/assets/ and op_kernel/ all sit next to each other.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OP_DIR = os.path.dirname(_THIS_DIR)
for _path in (_THIS_DIR, os.path.join(_THIS_DIR, "assets"),
              os.path.join(_OP_DIR, "op_kernel")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import mhc_pre_fuse_golden_cpu  # noqa: E402
from mhc_pre_fuse import mhc_pre_fuse_wrapper  # noqa: E402
from mhc_pre_fuse_precision import check_precision  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
_LOG = logging.getLogger(__name__)

# The five cases this suite runs; they are also the cases timed in RESULTS.md.
_CASES = {
    "t512_h4096": (512, 4096),
    "t512_h7168": (512, 7168),
    "t8192_h4096": (8192, 4096),
    "t8192_h5120": (8192, 5120),
    "t8192_h7168": (8192, 7168),
}


def _run_case(name):
    """Run one case on the NPU through the public wrapper and check its precision."""
    t, hidden = _CASES[name]
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", 0)))
    cpu_inputs = mhc_pre_fuse_golden_cpu.make_inputs(
        {"t": t, "hidden": hidden, "seed": 20260000 + t + hidden})
    expected = mhc_pre_fuse_golden_cpu.reference(cpu_inputs)
    actual = mhc_pre_fuse_wrapper(*[item.to("npu") for item in cpu_inputs])
    torch.npu.synchronize()
    passed, summary = check_precision([a.cpu() for a in actual], expected)
    _LOG.info("[%s] status=%s :: %s", name, "PASS" if passed else "FAIL", summary)
    if not passed:
        raise AssertionError(f"{name} precision failed: {summary}")


@pytest.mark.soc("950")
def test_t512_h4096():
    _run_case("t512_h4096")


@pytest.mark.soc("950")
def test_t512_h7168():
    _run_case("t512_h7168")


@pytest.mark.soc("950")
def test_t8192_h4096():
    _run_case("t8192_h4096")


@pytest.mark.soc("950")
def test_t8192_h5120():
    _run_case("t8192_h5120")


@pytest.mark.soc("950")
def test_t8192_h7168():
    _run_case("t8192_h7168")


def main():
    """Run every case by calling its test function, in the same order as the suite."""
    test_t512_h4096()
    test_t512_h7168()
    test_t8192_h4096()
    test_t8192_h5120()
    test_t8192_h7168()


if __name__ == "__main__":
    raise SystemExit(main())
