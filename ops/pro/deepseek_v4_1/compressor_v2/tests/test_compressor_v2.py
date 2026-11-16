# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Official CompressorV2 single-test defaults with reproducible input seeds."""

import logging
import os
import sys
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

_REPO_ROOT = Path(__file__).resolve().parents[5]
for _path in (_REPO_ROOT, _REPO_ROOT / "tests" / "common"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from op_caller import get_op_caller
from ops.pro.deepseek_v4_1.compressor_v2.tests.assets.compressor_v2_golden_cpu import make_inputs, reference
from ops.pro.deepseek_v4_1.compressor_v2.tests.assets.compressor_v2_kernel_consts import CASES as _CASES
from ops.pro.deepseek_v4_1.compressor_v2.tests.assets.compressor_v2_precision import check, check_guards

_LOGGER = logging.getLogger(__name__)

compressor = get_op_caller(
    "compressor_v2",
    jit_module="ops.pro.deepseek_v4_1.compressor_v2.op_kernel.compressor_v2",
    jit_attr="compressor",
    aclnn_attr="compressor",
)
_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
_DEVICE = f"npu:{_DEVICE_ID}"


def to_device(values, case):
    """Build the specified cache view, retaining a trailing allocation guard."""
    device_values = [None if value is None else value.to(_DEVICE) for value in values]
    old = values[3]
    block_elements = old.shape[1] * old.shape[2]
    block_stride = block_elements + case["cache_stride_extra"]
    backing = torch.full(
        (old.shape[0] * block_stride + 64,), 7717.0, device=_DEVICE, dtype=torch.float32
    )
    cache = backing.as_strided(
        old.shape, (block_stride, old.shape[2], 1), storage_offset=case["cache_offset"]
    )
    cache.copy_(device_values[3])
    device_values[3] = cache
    return tuple(device_values), backing


def call(values, ratio):
    x, kv, gate, cache, table, cu, used, starts = values
    return compressor(
        x,
        kv,
        gate,
        cache,
        ratio,
        state_block_table=table,
        cu_seqlens=cu,
        seqused=used,
        start_pos=starts,
    )


def run_case(case, repeats=1):
    """Run public wrapper calls and validate every specified result."""
    torch.set_num_threads(8)
    torch.npu.set_device(_DEVICE_ID)
    cpu_values = make_inputs(case)
    expected = reference(cpu_values, case["ratio"])
    values, backing = to_device(cpu_values, case)
    for repeat in range(repeats):
        if repeat:
            values[3].copy_(cpu_values[3])
            torch.npu.synchronize()
        out = call(values, case["ratio"])
        torch.npu.synchronize()
        check(out, values[3], expected, values[0].dtype)
        check_guards(backing, values[3])
    if expected[1].any():
        corrupted = out.cpu().clone()
        corrupted[expected[1]] = float("nan")
        try:
            check(corrupted, values[3], expected, values[0].dtype)
        except AssertionError:
            pass
        else:
            raise AssertionError(
                "the correctness criterion accepted a corrupted output"
            )
    _LOGGER.info("PASS %s", case["name"])
    return out


@pytest.mark.soc("950")
def test_prefill0():
    run_case(_CASES["Prefill0"])


@pytest.mark.soc("950")
def test_prefill1():
    run_case(_CASES["Prefill1"], repeats=3)


@pytest.mark.soc("950")
def test_prefill2():
    run_case(_CASES["Prefill2"])


@pytest.mark.soc("950")
def test_decode0_start8192():
    run_case(_CASES["decode0_start8192"])


@pytest.mark.soc("950")
def test_decode0_start8195():
    run_case(_CASES["decode0_start8195"])


@pytest.mark.soc("950")
def test_decode1_start8192():
    run_case(_CASES["decode1_start8192"])


@pytest.mark.soc("950")
def test_decode1_start8195():
    run_case(_CASES["decode1_start8195"])


@pytest.mark.soc("950")
def test_decode2_start8192():
    run_case(_CASES["decode2_start8192"])


@pytest.mark.soc("950")
def test_decode2_start8195():
    run_case(_CASES["decode2_start8195"])


@pytest.mark.soc("950")
def test_decode3_start8193():
    run_case(_CASES["decode3_start8193"])


@pytest.mark.soc("950")
def test_decode4_start8192():
    run_case(_CASES["decode4_start8192"])


@pytest.mark.soc("950")
def test_decode4_start8193():
    run_case(_CASES["decode4_start8193"])


@pytest.mark.soc("950")
def test_decode4_start8191():
    run_case(_CASES["decode4_start8191"])


@pytest.mark.soc("950")
def test_decode5_start8193():
    run_case(_CASES["decode5_start8193"])


@pytest.mark.soc("950")
def test_fullload0():
    run_case(_CASES["FullLoad0"])


@pytest.mark.soc("950")
def test_fullload1():
    run_case(_CASES["FullLoad1"])


@pytest.mark.soc("950")
def test_fullload2():
    run_case(_CASES["FullLoad2"])


@pytest.mark.soc("950")
def test_fullload3():
    run_case(_CASES["FullLoad3"])


@pytest.mark.soc("950")
def test_fullload4():
    run_case(_CASES["FullLoad4"])


@pytest.mark.soc("950")
def test_fullload5():
    run_case(_CASES["FullLoad5"])


@pytest.mark.soc("950")
def test_fullload6():
    run_case(_CASES["FullLoad6"])


@pytest.mark.soc("950")
def test_fullload7():
    run_case(_CASES["FullLoad7"])


@pytest.mark.soc("950")
def test_fullload8():
    run_case(_CASES["FullLoad8"])


@pytest.mark.soc("950")
def test_contiguous0():
    run_case(_CASES["Contiguous0"])


@pytest.mark.soc("950")
def test_noncontiguous0():
    run_case(_CASES["NonContiguous0"])


def main():
    test_prefill0()
    test_prefill1()
    test_prefill2()
    test_decode0_start8192()
    test_decode0_start8195()
    test_decode1_start8192()
    test_decode1_start8195()
    test_decode2_start8192()
    test_decode2_start8195()
    test_decode3_start8193()
    test_decode4_start8192()
    test_decode4_start8193()
    test_decode4_start8191()
    test_decode5_start8193()
    test_fullload0()
    test_fullload1()
    test_fullload2()
    test_fullload3()
    test_fullload4()
    test_fullload5()
    test_fullload6()
    test_fullload7()
    test_fullload8()
    test_contiguous0()
    test_noncontiguous0()
    _LOGGER.info("COMPRESSOR_V2_OK %s", len(_CASES))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
