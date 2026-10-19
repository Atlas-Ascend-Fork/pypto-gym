# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""EngramGateBwd 测试套件：每个用例一个 ``test_xxx`` 函数，外加 ``main`` 入口。

kernel 位于同算子目录的 ``op_kernel/engram_gate_bwd.py``，golden 与精度对比脚本位于 ``tests/assets/``。
每个用例按规格构造输入，走公开 wrapper 上板运行，再用官方 Scheme A 精度标准比对独立 CPU golden。
运行：``python3 -m pytest <本文件> -q`` 或直接 ``python3 <本文件>``。

用例是正向模型实际会跑的 12 个形状：两档 token 数 × 六个 hidden size，``image_mask`` 全 False。
"""

import logging
import os
import sys

import pytest
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OP_DIR = os.path.dirname(_THIS_DIR)
# golden 与精度对比脚本在 tests/assets/，kernel 在同级的 op_kernel/ 目录。
for _path in (_THIS_DIR, os.path.join(_THIS_DIR, "assets"),
              os.path.join(_OP_DIR, "op_kernel")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import engram_gate_bwd_golden_cpu as golden  # noqa: E402
from engram_gate_bwd_precision import check_precision  # noqa: E402
from engram_gate_bwd import engram_gate_bwd_wrapper  # noqa: E402


def _npu_available():
    try:
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except Exception:  # noqa: BLE001
        return False


pytestmark = [
    pytest.mark.soc("950"),
    pytest.mark.skipif(not _npu_available(), reason="torch_npu / NPU not available"),
]

logging.basicConfig(level=logging.INFO, format="%(message)s")
_LOG = logging.getLogger(__name__)

# name, t, hc, dim, seed, mask_kind
_CASES = {
    "t512_h2048": (512, 4, 2048, 20261014, "all_false"),
    "t512_h2560": (512, 4, 2560, 20261015, "all_false"),
    "t512_h3072": (512, 4, 3072, 20261016, "all_false"),
    "t512_h4096": (512, 4, 4096, 20261017, "all_false"),
    "t512_h6144": (512, 4, 6144, 20261018, "all_false"),
    "t512_h7168": (512, 4, 7168, 20261019, "all_false"),
    "t8001_h2048": (8001, 4, 2048, 20261020, "all_false"),
    "t8001_h2560": (8001, 4, 2560, 20261021, "all_false"),
    "t8001_h3072": (8001, 4, 3072, 20261022, "all_false"),
    "t8001_h4096": (8001, 4, 4096, 20261023, "all_false"),
    "t8001_h6144": (8001, 4, 6144, 20261024, "all_false"),
    "t8001_h7168": (8001, 4, 7168, 20261025, "all_false"),
}


def _run_case(name):
    """按规格构造输入，上板跑公开 wrapper，并用官方精度标准比对 CPU golden。"""
    t, hc, dim, seed, mask_kind = _CASES[name]
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", 0)))
    cpu_inputs = golden.make_inputs(
        {"t": t, "hc": hc, "dim": dim, "seed": seed, "mask_kind": mask_kind})
    expected = golden.reference(cpu_inputs)
    actual = engram_gate_bwd_wrapper(*[item.to("npu") for item in cpu_inputs])
    torch.npu.synchronize()
    passed, summary = check_precision(tuple(item.cpu() for item in actual), expected)
    _LOG.info("[%s] status=%s :: %s", name, "PASS" if passed else "FAIL", summary)
    if not passed:
        raise AssertionError(f"{name} precision failed: {summary}")


def test_t512_h2048():
    _run_case("t512_h2048")


def test_t512_h2560():
    _run_case("t512_h2560")


def test_t512_h3072():
    _run_case("t512_h3072")


def test_t512_h4096():
    _run_case("t512_h4096")


def test_t512_h6144():
    _run_case("t512_h6144")


def test_t512_h7168():
    _run_case("t512_h7168")


def test_t8001_h2048():
    _run_case("t8001_h2048")


def test_t8001_h2560():
    _run_case("t8001_h2560")


def test_t8001_h3072():
    _run_case("t8001_h3072")


def test_t8001_h4096():
    _run_case("t8001_h4096")


def test_t8001_h6144():
    _run_case("t8001_h6144")


def test_t8001_h7168():
    _run_case("t8001_h7168")


def main():
    test_t512_h2048()
    test_t512_h2560()
    test_t512_h3072()
    test_t512_h4096()
    test_t512_h6144()
    test_t512_h7168()
    test_t8001_h2048()
    test_t8001_h2560()
    test_t8001_h3072()
    test_t8001_h4096()
    test_t8001_h6144()
    test_t8001_h7168()


if __name__ == "__main__":
    raise SystemExit(main())
