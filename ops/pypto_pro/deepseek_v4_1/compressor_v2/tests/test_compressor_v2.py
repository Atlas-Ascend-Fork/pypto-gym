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
import sys
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

_LOGGER = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
for _path in (_THIS_DIR / "assets", _THIS_DIR.parent / "op_kernel"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from compressor_v2 import compressor
from compressor_v2_golden_cpu import make_inputs, reference
from compressor_v2_precision import check, check_guards

# fmt: off
_FULLLOAD3_BLOCKS = [
    17, 12, 7, 38, 44, 43, 52, 16, 56, 18, 14, 11, 35, 20, 26, 42,
    57, 13, 1, 62, 41, 47, 0, 3, 37, 48, 50, 49, 46, 24, 39, 36,
    55, 51, 29, 61, 34, 53, 8, 19, 32, 31, 30, 15, 23, 21, 25, 45,
    63, 54, 58, 28, 6, 5, 4, 27, 9, 2, 59, 40, 22, 60, 33, 10,
]

_FULLLOAD4_BLOCKS = [
    37, 5, 33, 54, 36, 41, 44, 28, 16, 7, 0, 25, 56, 18, 63, 21,
    52, 42, 6, 61, 58, 39, 53, 31, 55, 47, 2, 32, 34, 15, 3, 35,
    4, 17, 10, 20, 38, 40, 24, 46, 13, 29, 48, 19, 23, 59, 9, 22,
    12, 30, 27, 26, 14, 43, 60, 49, 11, 50, 1, 57, 45, 62, 8, 51,
]

# name, layout, batch, sequence, dim, ratio, block_size, start_pos, seed, pad, offset, blocks
_CASE_ROWS = [
    ("Prefill0", "TH", 1, 8192, 512, 4, 128, 0, 2026102801, 14, 6, [0]),
    ("Prefill1", "TH", 1, 8192, 128, 4, 128, 0, 2026102802, 16, 6, [0]),
    ("Prefill2", "TH", 1, 8192, 512, 128, 128, 0, 2026102803, 5, 1, [0]),
    ("decode0_start8192", "TH", 1, 1, 512, 4, 128, 8192, 2026102804, 38, 20, [0]),
    ("decode0_start8195", "TH", 1, 1, 512, 4, 128, 8195, 2026102805, 16, 6, [0]),
    ("decode1_start8192", "TH", 1, 1, 128, 4, 128, 8192, 2026102806, 23, 0, [0]),
    ("decode1_start8195", "TH", 1, 1, 128, 4, 128, 8195, 2026102807, 49, 24, [0]),
    ("decode2_start8192", "TH", 1, 1, 512, 128, 128, 8192, 2026102808, 24, 17, [0]),
    ("decode2_start8195", "TH", 1, 1, 512, 128, 128, 8195, 2026102809, 26, 18, [0]),
    ("decode3_start8193", "TH", 8, 3, 512, 4, 128, 8193, 2026102810, 10, 0, [3, 2, 0, 6, 5, 1, 4, 7]),
    ("decode4_start8192", "TH", 8, 3, 128, 4, 128, 8192, 2026102811, 35, 25, [4, 5, 2, 1, 7, 3, 0, 6]),
    ("decode4_start8193", "TH", 8, 3, 128, 4, 128, 8193, 2026102812, 35, 24, [4, 0, 1, 6, 7, 2, 5, 3]),
    ("decode4_start8191", "TH", 8, 3, 128, 4, 128, 8191, 2026102813, 11, 6, [4, 1, 7, 0, 6, 3, 2, 5]),
    ("decode5_start8193", "TH", 8, 3, 512, 128, 128, 8193, 2026102814, 5, 1, [6, 1, 5, 4, 0, 2, 7, 3]),
    ("FullLoad0", "BSH", 1, 4, 512, 4, 128, 0, 2026102815, 31, 8, [0]),
    ("FullLoad1", "BSH", 1, 4, 128, 4, 128, 0, 2026102816, 29, 0, [0]),
    ("FullLoad2", "BSH", 1, 4, 512, 128, 131, 0, 2026102817, 3, 0, [0]),
    ("FullLoad3", "BSH", 64, 4, 512, 4, 128, 0, 2026102818, 18, 11, _FULLLOAD3_BLOCKS),
    ("FullLoad4", "BSH", 64, 4, 128, 4, 128, 0, 2026102819, 38, 31, _FULLLOAD4_BLOCKS),
    ("FullLoad5", "BSH", 8, 2, 512, 4, 128, 0, 2026102820, 47, 11, [6, 1, 3, 7, 0, 2, 4, 5]),
    ("FullLoad6", "BSH", 1, 1, 512, 4, 128, 100, 2026102821, 23, 2, [0]),
    ("FullLoad7", "BSH", 1, 1, 128, 4, 128, 100, 2026102822, 21, 12, [0]),
    ("FullLoad8", "BSH", 1, 1, 512, 128, 128, 100, 2026102823, 22, 14, [0]),
    ("Contiguous0", "TH", 2, 3, 512, 4, 128, 8193, 2026102824, 0, 0, [0, 1]),
    ("NonContiguous0", "TH", 2, 3, 512, 4, 128, 8193, 2026102825, 36, 4, [1, 0]),
]
# fmt: on

_CASES = {
    name: {
        "name": name,
        "layout": layout,
        "dtype": "bfloat16",
        "batch": batch,
        "sequence": sequence,
        "hidden": 4096,
        "dim": dim,
        "ratio": ratio,
        "block_size": block_size,
        "starts": [start] * batch,
        "seed": seed,
        "blocks": blocks,
        "cache_stride_extra": pad,
        "cache_offset": offset,
    }
    for name, layout, batch, sequence, dim, ratio, block_size, start, seed, pad, offset, blocks in _CASE_ROWS
}


def to_device(values, case):
    """Build the specified cache view, retaining a trailing allocation guard."""
    device_values = [None if value is None else value.to("npu:0") for value in values]
    old = values[3]
    block_elements = old.shape[1] * old.shape[2]
    block_stride = block_elements + case["cache_stride_extra"]
    backing = torch.full(
        (old.shape[0] * block_stride + 64,), 7717.0, device="npu:0", dtype=torch.float32
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


def run_case(case):
    """Run one public wrapper call and validate every specified result."""
    torch.set_num_threads(8)
    torch.npu.set_device(0)
    cpu_values = make_inputs(case)
    expected = reference(cpu_values, case["ratio"])
    values, backing = to_device(cpu_values, case)
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
    return out


@pytest.mark.soc("950")
def test_prefill0():
    run_case(_CASES["Prefill0"])


@pytest.mark.soc("950")
def test_prefill1():
    run_case(_CASES["Prefill1"])


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
    for case in _CASES.values():
        run_case(case)
        _LOGGER.info("PASS %s", case["name"])
    _LOGGER.info("COMPRESSOR_V2_OK %s", len(_CASES))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
