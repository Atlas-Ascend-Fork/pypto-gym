# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback

import pytest
import torch
import torch_npu  # noqa: F401

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

_repo = THIS_DIR
while _repo != os.path.dirname(_repo) and not os.path.isdir(os.path.join(_repo, "src")):
    _repo = os.path.dirname(_repo)
if os.path.isdir(os.path.join(_repo, "src")):
    sys.path.insert(0, _repo)
    sys.path.insert(0, os.path.join(_repo, "ops", "pypto_tensor"))

from experimental.vector.GatherPaKvCache.gather_pa_kv_cache_impl import (  # noqa: E402
    gather_pa_kv_cache_wrapper,
)
from gather_pa_kv_cache_golden import gather_pa_kv_cache_golden, make_case  # noqa: E402

LOGGER = logging.getLogger(__name__)
REQUIRED_LEVELS = tuple(f"level{idx}" for idx in range(10))


def _device() -> str:
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    torch.npu.set_device(device_id)
    return f"npu:{device_id}"


def _load_cases(path: str = None) -> list[dict]:
    if path is None:
        path = os.path.join(THIS_DIR, "test_cases.json")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cases = cfg["test_cases"]
    ids = {case["id"] for case in cases}
    for level in REQUIRED_LEVELS:
        if level not in ids:
            raise RuntimeError(f"missing required test level '{level}' in {path}")
    return cases


def _choose_gather_cases(cases: list[dict], selected: list[str]) -> list[dict]:
    wanted = set(selected)
    if not wanted:
        return cases
    case_by_id = {case["id"]: case for case in cases}
    missing = sorted(wanted.difference(case_by_id))
    if missing:
        raise RuntimeError(f"unknown gather case(s): {missing}")
    return [case_by_id[case_id] for case_id in selected]


def _list_gather_cases(cases: list[dict]) -> None:
    for case in cases:
        LOGGER.info("%s: %s", case["id"], case.get("description", ""))


def _run_gather_cases(cases: list[dict], device: str) -> bool:
    passed = True
    for case in cases:
        try:
            passed = bool(_run_case(case, device)) and passed
        except Exception:
            traceback.print_exc()
            passed = False
    return passed


def _make_inputs(case: dict, device: str) -> tuple[dict, dict]:
    inp = case["input"]
    if inp.get("is_seq_lens_cumsum") is not True:
        raise ValueError(f"{case['id']} must use cumsum seq_lens")
    case_cpu = make_case(
        total_tokens=int(inp["total_tokens"]),
        q_count=int(inp["q_count"]),
        num_blocks=int(inp["key_cache_shape"][0]),
        block_table_cols=int(inp["block_tables_shape"][1]),
        block_size=int(inp["key_cache_shape"][1]),
        key_num_heads=int(inp["key_cache_shape"][2]),
        key_dim=int(inp["key_cache_shape"][3]),
        value_num_heads=int(inp["value_cache_shape"][2]),
        value_dim=int(inp["value_cache_shape"][3]),
        is_seq_lens_cumsum=True,
        compute_golden=False,
        seed=int(case.get("seed", 42)),
    )
    case_dev = dict(case_cpu)
    for name in (
        "key_cache",
        "value_cache",
        "block_tables",
        "seq_lens",
        "key_ref",
        "value_ref",
        "seq_offset",
    ):
        case_dev[name] = case_cpu[name].to(device).contiguous()
    return case_cpu, case_dev


def _assert_shapes(case: dict, key_out: torch.Tensor, value_out: torch.Tensor) -> None:
    expected = case["output"]
    key_shape = tuple(expected["key_shape"])
    value_shape = tuple(expected["value_shape"])
    if tuple(key_out.shape) != key_shape:
        raise AssertionError(f"key output shape mismatch: got {tuple(key_out.shape)}, expected {key_shape}")
    if tuple(value_out.shape) != value_shape:
        raise AssertionError(f"value output shape mismatch: got {tuple(value_out.shape)}, expected {value_shape}")
    if key_out.dtype != torch.bfloat16 or value_out.dtype != torch.bfloat16:
        raise AssertionError(f"output dtype mismatch: key={key_out.dtype}, value={value_out.dtype}")


def _run_case(case: dict, device: str) -> bool:
    case_id = case["id"]
    inp = case["input"]
    LOGGER.info("=" * 60)
    LOGGER.info("Test: %s - %s", case_id, case.get("description", ""))
    LOGGER.info("=" * 60)

    case_cpu, case_dev = _make_inputs(case, device)
    key_out, value_out = gather_pa_kv_cache_wrapper(
        case_dev["key_cache"],
        case_dev["value_cache"],
        case_dev["block_tables"],
        case_dev["seq_lens"],
        case_dev["key_ref"],
        case_dev["value_ref"],
        case_dev["seq_offset"],
        cache_mode=inp.get("cache_mode", "Norm"),
        is_seq_lens_cumsum=True,
        run_mode="npu",
    )
    torch.npu.synchronize()
    _assert_shapes(case, key_out, value_out)

    key_ref, value_ref = gather_pa_kv_cache_golden(
        case_cpu["key_cache"],
        case_cpu["value_cache"],
        case_cpu["block_tables"],
        case_cpu["seq_lens"],
        case_cpu["key_ref"],
        case_cpu["value_ref"],
        case_cpu["seq_offset"],
        cache_mode=inp.get("cache_mode", "Norm"),
        is_seq_lens_cumsum=True,
    )
    key_equal = torch.equal(key_out.detach().cpu(), key_ref)
    value_equal = torch.equal(value_out.detach().cpu(), value_ref)
    LOGGER.info("  [key] shape=%s equal=%s", tuple(key_out.shape), key_equal)
    LOGGER.info("  [value] shape=%s equal=%s", tuple(value_out.shape), value_equal)
    return bool(key_equal and value_equal)


@pytest.mark.soc("950")
def test_level0(device_id=None, run_mode="npu"):
    del device_id, run_mode
    device = _device()
    cases = [case for case in _load_cases() if case["id"] == "level0"]
    assert _run_case(cases[0], device)


@pytest.mark.soc("950")
def test_level9(device_id=None, run_mode="npu"):
    del device_id, run_mode
    device = _device()
    cases = [case for case in _load_cases() if case["id"] == "level9"]
    assert _run_case(cases[0], device)


def main() -> int:
    parser = argparse.ArgumentParser(description="Precision test for gather_pa_kv_cache ND network sweep")
    parser.add_argument("cases", nargs="*", help="case ids from test_cases.json, e.g. level0 level9")
    parser.add_argument("--list", action="store_true", help="list available cases and exit")
    parser.add_argument("--run-mode", default="npu", choices=["npu"], help="execution backend")
    args = parser.parse_args()

    try:
        cases = _load_cases()
        if args.list:
            _list_gather_cases(cases)
            return 0
        device = _device()
        selected_cases = _choose_gather_cases(cases, args.cases)
        LOGGER.info("Using gather device: %s", device)
        if _run_gather_cases(selected_cases, device):
            LOGGER.info("[PRECISION_PASS]")
            return 0
    except Exception:
        traceback.print_exc()
    LOGGER.error("[PRECISION_FAIL]")
    return 1


if __name__ == "__main__":
    sys.exit(main())
