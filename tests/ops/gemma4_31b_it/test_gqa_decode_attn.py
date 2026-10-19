# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""
GQA decode attention kernel precision test.

Validates: PyPTO kernel vs Golden (pure PyTorch with KV head averaging).
Test cases cover global/local attention with various Skv lengths.

Each test case runs in a subprocess to avoid pypto JIT state contamination
across different DYNAMIC Skv shapes.
"""

import math
import os
import sys
import json
import argparse
import subprocess

# Architecture
Nq = 32
Nkv_orig = 16
Nkv = 4
GROUPS = Nq // Nkv
D = 256
SCALE = 1.0
W = 1024
B = 1
Sq = 1


def load_test_cases(json_path):
    if not os.path.exists(json_path):
        raise RuntimeError(f"Test cases file not found: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)


def run_single_case(case_data, device):
    """Run a single test case (called in subprocess via --_run_one)."""
    import torch
    import numpy as np
    from numpy.testing import assert_allclose

    from ops.pypto_tensor.gemma4_31b_it.gqa_decode_attn.gqa_decode_attn_impl import (
        gqa_decode_attn_wrapper)
    from gqa_decode_attn_golden import gqa_decode_attn_golden

    case_id = case_data["id"]
    description = case_data.get("description", "")
    Skv = case_data["Skv"]
    layer_kind = case_data["layer_kind"]
    effective_skv = min(Skv, W) if layer_kind == "local" else Skv

    print("=" * 60)
    print(f"Test: {case_id} -- {description}")
    print(f"  Skv={Skv}, kind={layer_kind}, effective_skv={effective_skv}")
    print("=" * 60)

    torch.manual_seed(case_data.get("seed", 42) + Skv)

    q = torch.randn(B, Nq, Sq, D, dtype=torch.bfloat16, device=device) * 0.3
    k = torch.randn(B, Nkv_orig, Skv, D, dtype=torch.bfloat16, device=device) * 0.3
    v = torch.randn(B, Nkv_orig, Skv, D, dtype=torch.bfloat16, device=device) * 0.3
    mask = torch.zeros(B, 1, Sq, Skv, dtype=torch.float32, device=device)

    ref = gqa_decode_attn_golden(q, k, v, mask, scaling=SCALE, layer_kind=layer_kind)
    got = gqa_decode_attn_wrapper(q, k, v, mask, SCALE, layer_kind)

    ref_np = ref.float().cpu().numpy()
    got_np = got.float().cpu().numpy()
    max_diff = float(np.abs(ref_np - got_np).max())
    mean_diff = float(np.abs(ref_np - got_np).mean())

    rtol = case_data.get("rtol", 5e-3)
    atol = case_data.get("atol", 5e-3)

    print(f"  Output shape: {tuple(got.shape)}")
    print(f"  Max diff : {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")

    try:
        assert_allclose(ref_np, got_np, rtol=rtol, atol=atol)
        print("  [PRECISION_PASS]")
        return True
    except AssertionError as e:
        print(f"  [PRECISION_FAIL] {e}", file=sys.stderr)
        return False


def run_case_subprocess(case_data, device_str, python_exe, script_path, json_path):
    """Run a single case in an isolated subprocess to avoid JIT state contamination."""
    case_id = case_data["id"]
    env = os.environ.copy()
    # CWD must be PR root so 'src.*' imports work; also add test dir to PYTHONPATH for golden
    pr_root = os.path.abspath(os.path.join(os.path.dirname(script_path), "..", "..", ".."))
    test_dir = os.path.dirname(script_path)
    env["PYTHONPATH"] = test_dir + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [python_exe, script_path, case_id, "--device", device_str, "--json", json_path, "--_run_one"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=pr_root)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode == 0


def _setup_npu_device(args):
    """Set up NPU device from --device arg for single-case mode."""
    device = args.device
    if device.startswith("npu"):
        device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
        import torch
        import torch_npu  # noqa: F401
        torch.npu.set_device(device_id)
        device = f"npu:{device_id}"
    return device


def _run_single_inprocess(args, gqa_cases):
    """Run a single test case in the current process."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)
    import torch
    device = _setup_npu_device(args)
    match = [c for c in gqa_cases if c["id"] == args.case_id]
    if not match:
        raise RuntimeError("Test execution failed")
    ok = run_single_case(match[0], device)
    sys.exit(0 if ok else 1)


def _run_all_cases_subprocess(args, gqa_cases):
    """Run all (or filtered) test cases in subprocesses for JIT isolation."""
    if args.case_id:
        match = [c for c in gqa_cases if c["id"] == args.case_id]
        if not match:
            print(f"ERROR: unknown case '{args.case_id}'")
            raise RuntimeError("Test execution failed")
        to_run = match
    else:
        to_run = gqa_cases

    python_exe = sys.executable
    script_path = os.path.abspath(__file__)
    passed = 0
    for case_data in to_run:
        print(f"\n>> Running {case_data['id']}")
        if run_case_subprocess(case_data, args.device, python_exe, script_path, args.json):
            passed += 1
        else:
            print(f"\nFailed at {case_data['id']}")
            raise RuntimeError("Test failed")

    print("\n" + "=" * 60)
    print(f"All GQA decode attention tests passed! ({passed}/{len(to_run)} cases)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="GQA decode attention kernel precision test")
    parser.add_argument("case_id", type=str, nargs="?", help="Case ID to run (omit for all)")
    parser.add_argument("--list", action="store_true", help="List available cases")
    parser.add_argument("--device", type=str,
                        default="npu",
                        help="Device: cpu or npu:<id>")
    parser.add_argument("--json", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_cases.json"))
    parser.add_argument("--_run_one", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    test_cases = load_test_cases(args.json)
    gqa_cases = [c for c in test_cases.get("test_cases", []) if c.get("op_name", "") == "gqa_decode_attn"]
    if not gqa_cases:
        print("ERROR: No gqa_decode_attn test cases found")
        raise RuntimeError("Test execution failed")

    if args.list:
        print(f"\nGQA decode attention test cases:\n")
        for c in gqa_cases:
            print(f"  {c['id']}  -- {c.get('description', '')}  [Skv={c['Skv']}, kind={c['layer_kind']}]")
        return

    if args._run_one:
        _run_single_inprocess(args, gqa_cases)
    else:
        _run_all_cases_subprocess(args, gqa_cases)


if __name__ == "__main__":
    main()
