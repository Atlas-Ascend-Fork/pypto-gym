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

"""mhc_pre_fuse PyPTO-Pro test: one test function per case, plus a ``main`` entry.

The kernel and the host wrapper live under ``op_kernel/mhc_pre_fuse.py``; the
CPU golden and the Scheme A comparator live under ``tests/assets/``.  Each test
prepares one case's inputs, runs the public wrapper on the NPU and compares the
three outputs against the independent CPU golden.

Besides the original five performance shapes, the suite covers the delivery
form's tiling-key branches (Tb / HBlk / RRem), non-default knobs and the
leading split axis of gemm_out_mul / gemm_out_sqrsum.
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

import mhc_pre_fuse_golden_cpu  # noqa: E402
from mhc_pre_fuse_golden_cpu import Knobs  # noqa: E402
from mhc_pre_fuse_precision import check_precision  # noqa: E402

try:
    from op_caller import get_op_caller  # noqa: E402

    mhc_pre_fuse_wrapper = get_op_caller(
        "mhc_pre_fuse",
        jit_attr="mhc_pre_fuse_wrapper",
        aclnn_attr="mhc_pre_fuse_wrapper")
except ImportError:  # standalone run outside pytest (jit only)
    from mhc_pre_fuse import mhc_pre_fuse_wrapper  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
_LOG = logging.getLogger(__name__)

pytestmark = pytest.mark.soc("950")

# The five cases this suite runs; they are also the cases timed in RESULTS.md.
_CASES = {
    "t512_h4096": (512, 4096),
    "t512_h7168": (512, 7168),
    "t8192_h4096": (8192, 4096),
    "t8192_h5120": (8192, 5120),
    "t8192_h7168": (8192, 7168),
}

# Delivery-form branches: Tb (T%8), HBlk (H divisibility) and RRem (tail).
_EXTRA_CASES = {
    "t12_h512_tb4_r3": (12, 512),
    "t6_h512_tb2_r3": (6, 512),
    "t3_h640_tb1_hblk128": (3, 640),
    "t7_h320_tb1_hblk64": (7, 320),
    "t64_h320_hblk64_r0": (64, 320),
    "t64_h640_hblk128_r0": (64, 640),
    "t64_h960_hblk192_r0": (64, 960),
    "t64_h1280_hblk256_r0": (64, 1280),
    "t64_h1536_hblk256_r1": (64, 1536),
    "t8_h3072_hblk512_r1": (8, 3072),
    "t8_h3584_hblk512_r2": (8, 3584),
}


def _run_case_shape(t, hidden, knobs=None, split=False, name=""):
    """Run one shape through the public wrapper and compare all three outputs."""
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", 0)))
    case = {"t": t, "hidden": hidden, "seed": 20260000 + t + hidden}
    cpu_inputs = mhc_pre_fuse_golden_cpu.make_inputs(case)
    if knobs is None:
        expected = mhc_pre_fuse_golden_cpu.reference(cpu_inputs)
    else:
        expected = mhc_pre_fuse_golden_cpu.reference(cpu_inputs, knobs=knobs)
    call_inputs = list(cpu_inputs)
    if split:
        call_inputs[0] = call_inputs[0].unsqueeze(0)
        call_inputs[1] = call_inputs[1].unsqueeze(0)
    device_inputs = [item.to("npu") for item in call_inputs]
    if knobs is None:
        actual = mhc_pre_fuse_wrapper(*device_inputs)
    else:
        actual = mhc_pre_fuse_wrapper(*device_inputs, knobs=knobs)
    torch.npu.synchronize()
    passed, summary = check_precision([a.cpu() for a in actual], expected)
    _LOG.info("[%s] status=%s :: %s",
              name or f"t{t}_h{hidden}", "PASS" if passed else "FAIL", summary)
    if not passed:
        raise AssertionError(f"{name or (t, hidden)} precision failed: {summary}")


def _run_case(name):
    t, hidden = _CASES[name]
    _run_case_shape(t, hidden, name=name)


def test_t512_h4096():
    _run_case("t512_h4096")


def test_t512_h7168():
    _run_case("t512_h7168")


def test_t8192_h4096():
    _run_case("t8192_h4096")


def test_t8192_h5120():
    _run_case("t8192_h5120")


def test_t8192_h7168():
    _run_case("t8192_h7168")


@pytest.mark.parametrize("name", list(_EXTRA_CASES), ids=list(_EXTRA_CASES))
def test_delivery_branches(name):
    t, hidden = _EXTRA_CASES[name]
    _run_case_shape(t, hidden, name=name)


def test_nondefault_knobs():
    knobs = Knobs(rms_eps=1.0e-5, mhc_pre_eps=0.0, mhc_sinkhorn_eps=1.0e-4,
                  mhc_post_mult_value=2.0, sinkhorn_repeat=3)
    _run_case_shape(64, 1280, knobs=knobs, name="nondefault_knobs")


def test_split_axis_inputs():
    _run_case_shape(64, 1280, split=True, name="split_axis")


def main():
    """Run every case by calling its test function, in the same order as the suite."""
    test_t512_h4096()
    test_t512_h7168()
    test_t8192_h4096()
    test_t8192_h5120()
    test_t8192_h7168()
    for name in _EXTRA_CASES:
        test_delivery_branches(name)
    test_nondefault_knobs()
    test_split_axis_inputs()


if __name__ == "__main__":
    raise SystemExit(main())
