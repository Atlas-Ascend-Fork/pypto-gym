# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Precision tests for the PyPTO QkvRmsNormRopeCache custom op."""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict

import pytest
import torch
import torch_npu  # noqa: F401
from numpy.testing import assert_allclose

LOGGER = logging.getLogger(__name__)
CASE_FILE = Path(__file__).with_name("test_cases.json")
P0_SHAPE_MARKERS = [[8, 2304], [2, 9216]]
QkvNormRopeInputs = collections.namedtuple(
    "QkvNormRopeInputs",
    ["qkv", "q_gamma", "k_gamma", "cos", "sin", "index",
     "q_out", "k_cache", "v_cache", "k_scale", "v_scale"],
)
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while REPO_ROOT != REPO_ROOT.parent and not (REPO_ROOT / "src").is_dir():
    REPO_ROOT = REPO_ROOT.parent
if (REPO_ROOT / "src").is_dir():
    sys_path_items = [
        REPO_ROOT / "src",
        REPO_ROOT / "ops" / "pypto_tensor",
    ]
    for path_item in reversed(sys_path_items):
        path_text = str(path_item)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

qkv_rms_norm_rope_cache_golden = importlib.import_module(
    "qkv_rms_norm_rope_cache_golden"
).qkv_rms_norm_rope_cache_golden
qkv_rms_norm_rope_cache_wrapper = importlib.import_module(
    "experimental.vector.QkvRmsNormRopeCache.qkv_rms_norm_rope_cache_impl"
).qkv_rms_norm_rope_cache_wrapper


def load_cases(path: Path = CASE_FILE):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)["test_cases"]


def npu_device():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    torch.npu.set_device(device_id)
    return torch.device(f"npu:{device_id}")


def make_inputs(case: Dict, device: torch.device):
    torch.manual_seed(int(case["seed"]))
    batch, seq, num_qkv, dim = case["qkv_size"]
    num_q, num_k, num_v = case["head_nums"]
    tokens = batch * seq
    qkv = (torch.randn(tokens, num_qkv * dim, dtype=torch.float32, device=device) * 0.2).to(torch.bfloat16)
    q_gamma = (torch.rand(dim, dtype=torch.float32, device=device) + 0.5).to(torch.bfloat16)
    k_gamma = (torch.rand(dim, dtype=torch.float32, device=device) + 0.5).to(torch.bfloat16)
    cos = torch.randn(tokens, dim, dtype=torch.float32, device=device).cos().to(torch.bfloat16)
    sin = torch.randn(tokens, dim, dtype=torch.float32, device=device).sin().to(torch.bfloat16)
    index_values = case.get("index_values")
    if index_values is None:
        index = torch.arange(tokens, dtype=torch.int64, device=device)
    else:
        if len(index_values) != tokens:
            raise ValueError(f"index_values length {len(index_values)} must equal tokens {tokens}")
        index = torch.tensor(index_values, dtype=torch.int64, device=device)
    q_out = torch.zeros(tokens, num_q * dim, dtype=torch.bfloat16, device=device)
    c0 = int(case.get("c0", 32))
    cache_dtype_name = case.get("cache_dtype", "int8")
    if cache_dtype_name != "int8":
        raise ValueError("current network cases require int8 cache_dtype")
    cache_dtype = torch.int8
    k_cache = torch.zeros(
        case["block_num"], num_k * dim // c0, case["block_size"], c0, dtype=cache_dtype, device=device
    )
    v_cache = torch.zeros(
        case["block_num"], num_v * dim // c0, case["block_size"], c0, dtype=cache_dtype, device=device
    )
    if cache_dtype == torch.int8:
        k_scale = (torch.rand(num_k, dim, dtype=torch.float32, device=device) * 0.1) + 0.1
        v_scale = (torch.rand(num_v, dim, dtype=torch.float32, device=device) * 0.1) + 0.1
    else:
        k_scale = None
        v_scale = None
    return QkvNormRopeInputs(qkv, q_gamma, k_gamma, cos, sin, index, q_out, k_cache, v_cache, k_scale, v_scale)


def run_single_case(case: Dict):
    device = npu_device()
    inputs = make_inputs(case, device)
    golden = qkv_rms_norm_rope_cache_golden(
        *inputs,
        qkv_size=case["qkv_size"],
        head_nums=case["head_nums"],
        epsilon=case["epsilon"],
        cache_mode=case["cache_mode"],
        is_output_qkv=case["is_output_qkv"],
    )
    result = qkv_rms_norm_rope_cache_wrapper(
        *inputs,
        qkv_size=case["qkv_size"],
        head_nums=case["head_nums"],
        epsilon=case["epsilon"],
        cache_mode=case["cache_mode"],
        is_output_qkv=case["is_output_qkv"],
    )
    result = tuple(x.detach().cpu() for x in result)
    golden = tuple(x.detach().cpu() for x in golden[:len(result)])

    for idx, (actual, expected) in enumerate(zip(result, golden)):
        assert actual.shape == expected.shape, f"output {idx} shape mismatch: {actual.shape} != {expected.shape}"
        assert actual.dtype == expected.dtype, f"output {idx} dtype mismatch: {actual.dtype} != {expected.dtype}"
        atol = float(case.get("atol", 4e-3))
        rtol = float(case.get("rtol", 4e-3))
        if expected.dtype == torch.int8:
            atol = float(case.get("cache_atol", atol))
            rtol = float(case.get("cache_rtol", rtol))
        try:
            assert_allclose(
                actual.to(torch.float32).numpy(),
                expected.to(torch.float32).numpy(),
                rtol=rtol,
                atol=atol,
            )
        except AssertionError:
            diff = (actual.to(torch.float32) - expected.to(torch.float32)).abs()
            LOGGER.error(
                "output %s mismatch: max_diff=%s, actual_nonzero=%s, expected_nonzero=%s",
                idx, float(diff.max()), int(actual.count_nonzero()), int(expected.count_nonzero())
            )
            raise
    return result


@pytest.mark.soc("950")
def test_level0_npu_precision():
    run_single_case(load_cases()[0])


@pytest.mark.soc("950")
def test_level1_npu_precision():
    run_single_case(load_cases()[1])


def run_benchmark_case(case: Dict, warmup: int, repeat: int):
    device = npu_device()
    inputs = make_inputs(case, device)
    for _ in range(warmup):
        qkv_rms_norm_rope_cache_wrapper(
            *inputs,
            qkv_size=case["qkv_size"],
            head_nums=case["head_nums"],
            epsilon=case["epsilon"],
            cache_mode=case["cache_mode"],
            is_output_qkv=case["is_output_qkv"],
        )
    torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        qkv_rms_norm_rope_cache_wrapper(
            *inputs,
            qkv_size=case["qkv_size"],
            head_nums=case["head_nums"],
            epsilon=case["epsilon"],
            cache_mode=case["cache_mode"],
            is_output_qkv=case["is_output_qkv"],
        )
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    avg_ms = elapsed_ms / repeat
    tokens = int(case["qkv_size"][0]) * int(case["qkv_size"][1])
    LOGGER.info(
        "[PERF] %s avg_ms=%.6f repeat=%s warmup=%s tokens=%s ms_per_token=%.6f",
        case["id"], avg_ms, repeat, warmup, tokens, avg_ms / tokens,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id", nargs="?")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--run-mode", choices=["npu"], default="npu")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    cases = load_cases()
    if args.list:
        for case in cases:
            LOGGER.info("%s: %s", case["id"], case["description"])
        return
    selected = [case for case in cases if args.case_id in (None, case["id"])]
    if not selected:
        raise SystemExit(f"unknown case_id: {args.case_id}")
    if args.benchmark:
        for case in selected:
            run_benchmark_case(case, args.warmup, args.repeat)
        return
    try:
        for case in selected:
            LOGGER.info("Running %s (npu)", case["id"])
            run_single_case(case)
            LOGGER.info("PASS %s", case["id"])
    except AssertionError:
        LOGGER.error("[PRECISION_FAIL]")
        raise
    LOGGER.info("[PRECISION_PASS]")


if __name__ == "__main__":
    main()
