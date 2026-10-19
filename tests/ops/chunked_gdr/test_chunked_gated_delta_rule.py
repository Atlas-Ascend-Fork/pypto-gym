# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------



"""PyPTO chunked_gated_delta_rule operator test — self-contained.

This is a fully self-contained test file with NO external dependencies
beyond chunked_gated_delta_rule_impl and chunked_gated_delta_rule_golden.
All performance utilities and test case configurations are inlined.

Precision comparison test for the Chunked Gated Delta Rule Linear Attention
kernel implementation. Tests both aligned and unaligned sequence scenarios,
with mixed Nv=4/8 configurations.

Three-state output:
  - [PRECISION_PASS]: all precision checks passed
  - [PRECISION_FAIL]: precision check failed (numerical mismatch)
  - No marker + exit ≠ 0: runtime failure (code crash, logic error, etc.)

Performance tracking (enabled with --perf flag):
  When --perf is specified, each test case records task_time, AICore time
  and AICore utilization from merged_swimlane.json, printing a summary
  table at the end.

Test cases (8 total, mixed Nv=4/8):
  Nv=8 (large cases, GQA emphasis):
    - aligned_gqa (B=1, Nqk=2, Nv=8, T=128)
    - aligned_multi_batch_gqa (B=2, Nqk=2, Nv=8, T=512)
    - unaligned_gqa (B=1, Nqk=2, Nv=8, T=130)
  Nv=4 (small/medium cases):
    - aligned_large_gqa (B=2, Nqk=4, Nv=4, T=512)
    - unaligned_single (B=1, Nqk=2, Nv=4, T=130)
    - aligned_single_chunk_L64 (B=1, Nqk=2, Nv=4, T=64)
    - aligned_multi_chunk_L64 (B=1, Nqk=2, Nv=4, T=128)
    - aligned_single_chunk_L32 (B=1, Nqk=2, Nv=4, T=32)
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch_npu
from numpy.testing import assert_allclose

_CUR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CUR))
_IMPL = Path(__file__).resolve().parents[3] / "ops/pypto_tensor/experimental/attention/chunked_gdr/"
sys.path.insert(0, str(_IMPL))

import pytest

from chunked_gated_delta_rule_golden import chunked_gated_delta_rule_golden
from chunked_gated_delta_rule_impl import chunked_gated_delta_rule_wrapper

# ═══════════════════════════════════════════════════════════════════
# Inlined perf_test_utils: Environment configuration
# ═══════════════════════════════════════════════════════════════════

_DEFAULT_ASCEND_HOME = "/home/developer/Ascend/cann-9.0.0"
_DEFAULT_PYPTO_PATH = "/mnt/workspace/gitCode/cann/pypto/python"
_FORK_PYPTO_PATH = "/mnt/workspace/gitCode/cann/mce/pypto_fork/pypto_6304/python"


def get_device():
    if "TILE_FWK_DEVICE_ID" in os.environ:
        device_id = int(os.environ["TILE_FWK_DEVICE_ID"])
        return f"npu:{device_id}"
    return "cpu"


def _validate_ascend_path():
    ascend = os.environ.get("ASCEND_HOME_PATH", _DEFAULT_ASCEND_HOME)
    if not os.path.isdir(ascend):
        return f"ASCEND_HOME_PATH not found: {ascend}"
    os.environ["ASCEND_HOME_PATH"] = ascend
    print(f"[ENV] ASCEND_HOME_PATH = {ascend}")
    return None


def _validate_pto_isa_path():
    pto_isa_path = os.environ.get(
        "PTO_TILE_LIB_CODE_PATH",
        "/mnt/workspace/gitCode/cann/mce/pypto_fork/pto-isa"
    )
    if os.path.isdir(pto_isa_path):
        os.environ["PTO_TILE_LIB_CODE_PATH"] = pto_isa_path
        print(f"[ENV] PTO_TILE_LIB_CODE_PATH = {pto_isa_path}")
        return None
    return f"PTO_TILE_LIB_CODE_PATH not found: {pto_isa_path}"


def _validate_pypto_path():
    pypto_path = os.environ.get("PYPTO_PATH", _DEFAULT_PYPTO_PATH)
    if not os.path.isdir(pypto_path):
        if os.path.isdir(_FORK_PYPTO_PATH):
            pypto_path = _FORK_PYPTO_PATH
        else:
            return f"PyPTO path not found: {pypto_path}"
    existing = os.environ.get("PYTHONPATH", "")
    if pypto_path not in existing:
        os.environ["PYTHONPATH"] = pypto_path + ((":" + existing) if existing else "")
    if pypto_path not in sys.path:
        sys.path.insert(0, pypto_path)
    print(f"[ENV] PyPTO path = {pypto_path}")
    return None


def _run_npu_smi(warnings):
    try:
        import subprocess
        result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            warnings.append("npu-smi info returned non-zero")
        else:
            for line in result.stdout.strip().split("\n")[:4]:
                print(f"[NPU] {line}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        warnings.append("npu-smi not available")


def _check_env():
    errors = []
    warnings = []

    err = _validate_ascend_path()
    if err:
        errors.append(err)

    devid = os.environ.get("TILE_FWK_DEVICE_ID", "0")
    os.environ["TILE_FWK_DEVICE_ID"] = devid
    print(f"[ENV] TILE_FWK_DEVICE_ID = {devid}")

    warn = _validate_pto_isa_path()
    if warn:
        warnings.append(warn)

    err = _validate_pypto_path()
    if err:
        errors.append(err)

    _run_npu_smi(warnings)

    try:
        print(f"[ENV] torch_npu OK")
    except ImportError:
        errors.append("torch_npu import failed")

    try:
        import pypto  # noqa: F401
        print(f"[ENV] pypto OK  (path: {pypto.__file__})")
        has_dynamic = hasattr(pypto, 'DYNAMIC')
        print(f"[ENV] pypto DYNAMIC support: {has_dynamic}")
    except ImportError:
        errors.append("pypto import failed")

    print()
    if warnings:
        for w in warnings:
            print(f"[WARN] {w}")
        print()

    if errors:
        print("=" * 60)
        print("Environment check FAILED:")
        for i, e in enumerate(errors, 1):
            print(f"\n  [{i}] {e}")
        print("\n" + "=" * 60)
        raise RuntimeError("Environment check failed: see errors above")

    print("[ENV] All checks passed. Starting tests...\n")


def get_device():
    """Get NPU device string."""
    did = os.environ.get("TILE_FWK_DEVICE_ID", "0")
    return f"npu:{did}"


# ═══════════════════════════════════════════════════════════════════
# Inlined perf_test_utils: Logging
# ═══════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
_fmt = logging.Formatter(fmt='%(asctime)s [%(levelname)s] %(message)s', datefmt='[%H:%M:%S]')
_handler = logging.StreamHandler()
_handler.setFormatter(_fmt)
logger.handlers.clear()
logger.addHandler(_handler)


# ═══════════════════════════════════════════════════════════════════
# Inlined perf_test_utils: Performance tracking
# ═══════════════════════════════════════════════════════════════════

_PERF_OUTPUT_BASE = os.path.abspath(os.path.join(os.getcwd(), "output"))
_perf_records = []
_perf_timestamp = None


def _take_perf_timestamp():
    """Record a timestamp BEFORE the kernel call.

    After the kernel completes, _collect_updated_dirs() will look for
    merged_swimlane.json files modified after this timestamp (with a
    0.5s buffer for filesystem timing jitter).
    """
    global _perf_timestamp
    _perf_timestamp = time.time()


def _collect_updated_dirs():
    """Collect output dirs whose merged_swimlane.json was modified after
    _take_perf_timestamp() (minus a 0.5s buffer).

    Returns:
        List of absolute paths to output dirs with updated swimlane data.
    """
    if _perf_timestamp is None or not os.path.isdir(_PERF_OUTPUT_BASE):
        return []
    updated = []
    for name in os.listdir(_PERF_OUTPUT_BASE):
        d = os.path.join(_PERF_OUTPUT_BASE, name)
        if not os.path.isdir(d):
            continue
        swim = os.path.join(d, "merged_swimlane.json")
        if os.path.isfile(swim):
            mtime = os.path.getmtime(swim)
            if mtime >= _perf_timestamp - 0.5:
                updated.append(d)
    updated.sort()
    return updated


def _identify_kernel(output_dir):
    """Identify the kernel type from the output directory structure.

    Returns:
        "Aligned" or "Unaligned" or "Unknown"
    """
    kernel_aicore = os.path.join(output_dir, "kernel_aicore")
    if not os.path.isdir(kernel_aicore):
        return "Unknown"

    dir_name = os.path.basename(output_dir)
    if "aligned" in dir_name.lower():
        return "Aligned"
    elif "unaligned" in dir_name.lower():
        return "Unaligned"

    return "CGDR"


def _parse_swimlane(output_dir):
    """Parse merged_swimlane.json to extract performance metrics.

    Returns:
        (task_time, aicore_time, utilization) tuple, or None if parsing fails
    """
    swim_path = os.path.join(output_dir, "merged_swimlane.json")
    if not os.path.exists(swim_path):
        return None

    with open(swim_path) as f:
        data = json.load(f)

    evts = data.get("traceEvents", [])
    if not evts:
        return None

    tid2core = {}
    for ev in evts:
        if ev.get("ph") == "M" and ev.get("name") == "thread_name":
            tid2core[ev["tid"]] = ev["args"]["name"]

    all_tasks = [ev for ev in evts if ev.get("cat") == "event" and ev.get("ph") == "X"]
    real_tasks = [t for t in all_tasks if "(fake)" not in t.get("name", "")]

    if not real_tasks:
        return None

    global_first = min(t["ts"] for t in real_tasks)
    global_last = max(t["ts"] + t["dur"] for t in real_tasks)
    task_time = global_last - global_first

    aicore_time = sum(t["dur"] for t in real_tasks)

    active_core_ids = set()
    for t in real_tasks:
        core_name = tid2core.get(t["tid"], "")
        m = re.search(r"(\d+)$", core_name)
        if m:
            active_core_ids.add(int(m.group(1)))

    num_active_cores = len(active_core_ids) if active_core_ids else 1

    util = (aicore_time / (task_time * num_active_cores) * 100) if task_time > 0 else 0

    return task_time, aicore_time, util


def _record_perf_from_dirs(test_name, updated_dirs, kernel_filter=None):
    """Record performance data from updated output directories."""
    best_per_type = {}
    for d in updated_dirs:
        ktype = _identify_kernel(d)
        if kernel_filter and ktype not in kernel_filter:
            continue
        if ktype == "Unknown":
            continue
        swim = os.path.join(d, "merged_swimlane.json")
        if not os.path.isfile(swim):
            continue
        mtime = os.path.getmtime(swim)
        if ktype not in best_per_type or mtime > best_per_type[ktype][1]:
            best_per_type[ktype] = (d, mtime)

    for ktype, (d, _) in best_per_type.items():
        result = _parse_swimlane(d)
        if result is not None:
            task_time, aicore_time, util = result
            _perf_records.append({
                "test_name": test_name,
                "kernel": ktype,
                "task_time_us": task_time,
                "aicore_time_us": aicore_time,
                "utilization": util,
            })
        else:
            _perf_records.append({
                "test_name": test_name,
                "kernel": ktype,
                "task_time_us": None,
                "aicore_time_us": None,
                "utilization": None,
            })


def _print_perf_summary():
    """Print formatted performance summary table."""
    if not _perf_records:
        logger.info("\n[PERF] No performance data collected.")
        return

    max_test_len = max(len(r["test_name"]) for r in _perf_records)
    W_TEST = max(max_test_len + 2, 20)
    W_KERNEL = 10
    W_TASK = 14
    W_AICORE = 14
    W_UTIL = 10

    def _fmt_us(val):
        if val is None:
            return "N/A".rjust(W_TASK)
        if val >= 1000:
            return f"{val / 1000:.2f} ms".rjust(W_TASK)
        return f"{val:.1f} us".rjust(W_TASK)

    def _fmt_pct(val):
        if val is None:
            return "N/A".rjust(W_UTIL)
        return f"{val:.1f}%".rjust(W_UTIL)

    sep = "+" + "-" * (W_TEST + 2) + "+" + "-" * (W_KERNEL + 2) + \
          "+" + "-" * (W_TASK + 2) + "+" + "-" * (W_AICORE + 2) + \
          "+" + "-" * (W_UTIL + 2) + "+"

    header = "|" + " Test Case".ljust(W_TEST + 1) + \
             "|" + " Kernel".ljust(W_KERNEL + 1) + \
             "|" + " Task Time".ljust(W_TASK + 1) + \
             "|" + " AICore Time".ljust(W_AICORE + 1) + \
             "|" + " AICore Util".ljust(W_UTIL + 1) + "|"

    logger.info("")
    logger.info("=" * len(sep))
    logger.info("Performance Summary (from merged_swimlane.json)")
    logger.info("=" * len(sep))
    logger.info(sep)
    logger.info(header)
    logger.info(sep)

    current_test = None
    for rec in _perf_records:
        if rec["test_name"] != current_test:
            test_str = " " + rec["test_name"]
            current_test = rec["test_name"]
        else:
            test_str = ""
        row = "|" + test_str.ljust(W_TEST + 1) + \
              "|" + (" " + rec["kernel"]).ljust(W_KERNEL + 1) + \
              "|" + _fmt_us(rec["task_time_us"]) + " " + \
              "|" + _fmt_us(rec["aicore_time_us"]) + " " + \
              "|" + _fmt_pct(rec["utilization"]) + " |"
        logger.info(row)

    logger.info(sep)
    logger.info("")
    logger.info("指标说明 (与 PyPTO toolkit 定义一致):")
    logger.info("  Task Time   = 首 task 到末 task 的 wall-clock 耗时")
    logger.info("  AICore Time = 所有 task 的 dur 之和（总计算量）")
    logger.info("  AICore Util = AICoreTime / (TaskTime × 活跃物理核数) × 100%")
    logger.info("=" * len(sep))


def clear_perf_records():
    """Clear all collected performance records (for fresh runs)."""
    global _perf_records, _perf_timestamp
    _perf_records = []
    _perf_timestamp = None


# ═══════════════════════════════════════════════════════════════════
# Test configuration (inlined from test_cases.json)
# ═══════════════════════════════════════════════════════════════════

D = 128

_perf_flags = {"enabled": False}


def gen_dims(params):
    """Generate dimension parameters from input params dict."""
    chunk_size = params.get("chunk_size", 128)
    dims = {
        "T": params["T"],
        "B": params["B"],
        "Nqk": params["Nqk"],
        "Nv": params["Nv"],
        "D": D,
        "L": chunk_size,
    }
    if "seq_lens" in params:
        dims["seq_lens"] = params["seq_lens"]
    return dims


def gen_inputs(dims, dtype=torch.float32):
    """Generate input tensors for testing (matching qwen3_next data ranges).

    Now includes zeros tensors and eye tensor sized based on chunk_size L.
    """
    t = dims["T"]
    b = dims["B"]
    nqk = dims["Nqk"]
    nv = dims["Nv"]
    d = dims["D"]
    l = dims["L"]
    inverse_shape = l // 8

    query = torch.rand([t, nqk, d], dtype=dtype) * (1.3655 + 0.2785) - (1.3655 + 0.2785)
    key = torch.rand([t, nqk, d], dtype=dtype) * (1.4664 + 0.2785) - (1.4664 + 0.2785)
    value = torch.rand([t, nv, d], dtype=dtype) * (1.6488 + 0.2785) - (1.6488 + 0.2785)
    beta = torch.rand([t, nv], dtype=dtype) * (0.8927 - 0.0889) - (0.8927 - 0.0889)
    gate = torch.rand([t, nv], dtype=dtype) * (-0.1343 + 37.5452) - (-0.1343 + 37.5452)
    states = torch.zeros([b, nv, d, d], dtype=dtype)

    seq_lens = dims.get("seq_lens")
    if seq_lens is not None:
        if len(seq_lens) != b:
            raise ValueError(f"seq_lens length {len(seq_lens)} != B={b}")
        t = sum(seq_lens)
        offsets = [0]
        for sl in seq_lens:
            offsets.append(offsets[-1] + sl)
        act_seq_len = torch.tensor(offsets, dtype=torch.int32)
    else:
        t = dims["T"]
        seq_len_per_batch = t // b
        act_seq_len = torch.tensor(
            [i * seq_len_per_batch for i in range(b + 1)],
            dtype=torch.int32
        )

    mask = torch.tril(-torch.ones([l, l], dtype=dtype), diagonal=-1)
    tril_mask = torch.ones([l, l], dtype=dtype).tril()
    eye_data = torch.eye(l, dtype=dtype)

    is_aligned = all(
        (int(act_seq_len[i + 1]) - int(act_seq_len[i])) % l == 0
        for i in range(b)
    )

    return {
        "query": query,
        "key": key,
        "value": value,
        "beta": beta,
        "gate": gate,
        "states": states,
        "act_seq_len": act_seq_len,
        "mask": mask,
        "tril_mask": tril_mask,
        "eye": eye_data,
    }


def compare(actual, expected, name, rtol=1e-3, atol_abs=0, atol_rel=1e-3):
    """Compare two tensors with tolerance: tolerance = atol_abs + atol_rel * |expected|."""
    diff = torch.abs(actual.float() - expected.float())
    tolerance = atol_abs + atol_rel * torch.abs(expected.float())
    out_of_tolerance = (diff > tolerance).sum().item()
    total = actual.numel()
    max_diff = torch.max(diff).item()

    print(f"  {name}: max_diff={max_diff:.6e}, out_of_tol={out_of_tolerance}/{total}")

    if out_of_tolerance > 0:
        raise AssertionError(
            f"{name} precision FAIL: {out_of_tolerance}/{total} elements out of tolerance "
            f"(max_diff={max_diff:.6e}, tolerance={atol_abs}+{atol_rel}*|expected|)"
        )

    assert_allclose(
        actual.cpu().numpy(),
        expected.cpu().numpy(),
        rtol=rtol,
        atol=atol_abs,
    )


TEST_CONFIGS = [
    {
        "id": "aligned_gqa",
        "description": "GQA mode, aligned (B=1, Nqk=2, Nv=8, T=128)",
        "params": {"B": 1, "Nqk": 2, "Nv": 8, "T": 128, "chunk_size": 128},
    },
    {
        "id": "aligned_multi_batch_gqa",
        "description": "Multi-batch + GQA, aligned (B=2, Nqk=2, Nv=8, T=512)",
        "params": {"B": 2, "Nqk": 2, "Nv": 8, "T": 512, "chunk_size": 128},
    },
    {
        "id": "unaligned_gqa",
        "description": "GQA + unaligned (B=1, Nqk=2, Nv=8, T=130)",
        "params": {"B": 1, "Nqk": 2, "Nv": 8, "T": 130, "chunk_size": 128},
    },
    {
        "id": "aligned_large_gqa",
        "description": "Large-scale aligned (B=2, Nqk=4, Nv=4, T=512)",
        "params": {"B": 2, "Nqk": 4, "Nv": 4, "T": 512, "chunk_size": 128},
    },
    {
        "id": "unaligned_single",
        "description": "Unaligned: 1 full + 1 partial chunk (B=1, Nqk=2, Nv=4, T=130)",
        "params": {"B": 1, "Nqk": 2, "Nv": 4, "T": 130, "chunk_size": 128},
    },
    {
        "id": "aligned_single_chunk_L64",
        "description": "Single chunk L=64 aligned (B=1, Nqk=2, Nv=4, T=64)",
        "params": {"B": 1, "Nqk": 2, "Nv": 4, "T": 64, "chunk_size": 64},
    },
    {
        "id": "aligned_multi_chunk_L64",
        "description": "Two chunks L=64 aligned (B=1, Nqk=2, Nv=4, T=128, 2 chunks of 64)",
        "params": {"B": 1, "Nqk": 2, "Nv": 4, "T": 128, "chunk_size": 64},
    },
    {
        "id": "aligned_single_chunk_L32",
        "description": "Single chunk L=32 aligned (B=1, Nqk=2, Nv=4, T=32, INVERSE_SHAPE=4)",
        "params": {"B": 1, "Nqk": 2, "Nv": 4, "T": 32, "chunk_size": 32},
    },
]


@pytest.mark.skip(reason="large test case")
def test_level0_smoke():
    device = get_device()
    if device.startswith("npu"):
        device_id = int(device.split(":")[1])
    """Level 0: Basic smoke test — aligned single chunk only."""

    config = TEST_CONFIGS[0]
    run_single_test(config, device_id)
    logger.info(f"\n[LEVEL0] Smoke test passed: {config['id']}")


@pytest.mark.skip(reason="large test case")
def test_level1_comprehensive():
    device = get_device()
    if device.startswith("npu"):
        device_id = int(device.split(":")[1])
    run_all_tests(device_id=device_id)
    logger.info(f"\n[LEVEL1] All comprehensive tests passed")


def run_single_test(config, device_id=0):
    """Execute a single test case: golden vs kernel comparison."""
    case_id = config["id"]
    description = config.get("description", "")
    params = config["params"]

    perf_label = f"{case_id} [B={params['B']},Nqk={params['Nqk']},Nv={params['Nv']},T={params['T']}]"

    logger.info(f"{'=' * 60}")
    logger.info(f"Test: {case_id} — {description}")
    logger.info(f"{'=' * 60}")

    torch.manual_seed(42)

    dims = gen_dims(params)
    inputs = gen_inputs(dims, torch.float32)
    chunk_size = dims["L"]

    golden_attn, golden_state = chunked_gated_delta_rule_golden(
        inputs["query"], inputs["key"], inputs["value"],
        inputs["beta"], inputs["gate"], inputs["states"],
        inputs["mask"], inputs["tril_mask"], inputs["eye"],
        inputs["act_seq_len"],
        chunk_size=chunk_size,
    )

    torch.npu.set_device(device_id)

    if _perf_flags["enabled"]:
        _take_perf_timestamp()

    pto_attn, pto_state = chunked_gated_delta_rule_wrapper(
        inputs["query"], inputs["key"], inputs["value"],
        inputs["beta"], inputs["gate"], inputs["states"],
        act_seq_len=inputs["act_seq_len"],
        chunk_size=chunk_size,
        mask=inputs["mask"],
        tril_mask=inputs["tril_mask"],
        eye=inputs["eye"],
        enable_perf_debug=_perf_flags["enabled"],
    )

    if _perf_flags["enabled"]:
        updated_dirs = _collect_updated_dirs()
        _record_perf_from_dirs(perf_label, updated_dirs)

    logger.info(f"  Input shapes: query={inputs['query'].shape}, value={inputs['value'].shape}")
    logger.info(f"  Output shapes: attn={pto_attn.shape}, state={pto_state.shape}")

    if torch.isnan(pto_attn).any() or torch.isinf(pto_attn).any():
        raise AssertionError(f"core_attn_out contains NaN/Inf")
    if torch.isnan(pto_state).any() or torch.isinf(pto_state).any():
        raise AssertionError(f"last_state_data contains NaN/Inf")

    compare(pto_attn, golden_attn, "core_attn_out",
            rtol=1e-3, atol_abs=1e-3, atol_rel=1e-3)

    compare(pto_state, golden_state, "last_state_data",
            rtol=1e-3, atol_abs=1e-3, atol_rel=1e-3)

    logger.info(f"  ✓ {case_id} passed")
    return True


def run_all_tests(device_id=0, case_id=None):
    """Run all or selected test cases."""
    configs = TEST_CONFIGS

    if case_id:
        config = None
        for c in configs:
            if c["id"] == case_id:
                config = c
                break
        if config is None:
            logger.error(f"unknown case '{case_id}'")
            logger.error(f"Valid cases: {', '.join([c['id'] for c in configs])}")
        to_run = [config]
    else:
        to_run = configs

    passed = 0
    try:
        for config in to_run:
            run_single_test(config, device_id)
            passed += 1

        logger.info(f"\n{'=' * 60}")
        logger.info(f"All tests passed! ({passed}/{len(to_run)} cases)")
        logger.info(f"[PRECISION_PASS]")
        logger.info(f"{'=' * 60}")

        if _perf_flags["enabled"]:
            _print_perf_summary()

        return True

    except AssertionError as e:
        logger.error(f"\n{'=' * 60}")
        logger.error(f"[PRECISION_FAIL] {e}")
        raise RuntimeError("Test failed") from e

    except Exception as e:
        logger.error(f"\nRuntime error: {e}")
        raise RuntimeError("Test execution failed with critical error") from e


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="chunked_gated_delta_rule precision test with optional perf tracking"
    )
    parser.add_argument("case_id", type=str, nargs="?", help="Specific case ID to run")
    parser.add_argument("--list", action="store_true", help="List all test cases")
    parser.add_argument("--perf", action="store_true",
                        help="Enable performance tracking")
    parser.add_argument("--device_id", type=int, default=0,
                        help="NPU device ID (default: 0 or TILE_FWK_DEVICE_ID env)")
    args = parser.parse_args()

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", args.device_id))

    if args.perf:
        _perf_flags["enabled"] = True
        clear_perf_records()
        logger.info("[PERF] Performance tracking enabled")

    if args.list:
        print("\nAvailable test cases:\n")
        for config in TEST_CONFIGS:
            params = config.get("params", {})
            t_val = params.get("T", 0)
            cs = params.get("chunk_size", 128)
            is_aligned = t_val % cs == 0
            version = "aligned" if is_aligned else "unaligned"
            cs_str = f"L={cs}" if cs != 128 else ""
            print(f"  {config['id']}  — {config.get('description', '')}  [{version}] {cs_str}")
        sys.exit(0)

    run_all_tests(device_id=device_id, case_id=args.case_id)