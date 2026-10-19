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
Attention SoftMax kernel precision test.

Validates: PyPTO kernel vs Golden (pure PyTorch) implementation.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from numpy.testing import assert_allclose

_IMPL = Path(__file__).resolve().parents[3] / "ops" / "pypto_tensor" / "gemma4_31b_it"
sys.path.insert(0, str(_IMPL))

from attn_softmax.attn_softmax_impl import attn_softmax_wrapper
from attn_softmax_golden import attn_softmax_golden


_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def load_test_cases(json_path):
    if not os.path.exists(json_path):
        raise RuntimeError(f"Test cases file not found: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)


def _bf16_to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).cpu().numpy()


def run_single_case(case_data, device):
    case_id = case_data["id"]
    description = case_data.get("description", "")
    print("=" * 60)
    print(f"Test: {case_id} -- {description}")
    print("=" * 60)

    torch.manual_seed(case_data.get("seed", 42))

    inputs = case_data["input"]
    shape = inputs[0]["shape"]
    dtype = _DTYPE_MAP[inputs[0]["dtype"]]
    scale = case_data.get("scale", 1.0)

    scores = torch.randn(shape, dtype=dtype, device=device)

    result = attn_softmax_wrapper(scores, scale=scale)
    golden = attn_softmax_golden(scores, scale=scale)

    r_np = _bf16_to_np(result)
    g_np = _bf16_to_np(golden)

    max_diff = np.abs(r_np - g_np).max()
    mean_diff = np.abs(r_np - g_np).mean()

    print(f"  Shape    : {tuple(scores.shape)}, scale={scale}")
    print(f"  Max diff : {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")

    rtol = case_data.get("rtol", 0.0078125)
    atol = case_data.get("atol", 0.0001)
    try:
        assert_allclose(r_np, g_np, rtol=rtol, atol=atol)
        print("  [PRECISION_PASS]")
    except AssertionError as e:
        print(f"  [PRECISION_FAIL] {e}", file=sys.stderr)
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Attention SoftMax kernel precision test")
    parser.add_argument("case_id", type=str, nargs="?", help="Case ID to run (omit for all)")
    parser.add_argument("--list", action="store_true", help="List available cases")
    parser.add_argument("--device", type=str,
                        default="npu",
                        help="Device: cpu or npu:<id>")
    parser.add_argument("--json", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_cases.json"))
    args = parser.parse_args()

    test_cases = load_test_cases(args.json)
    softmax_cases = [c for c in test_cases.get("test_cases", []) if c.get("op_name", "") == "attn_softmax"]
    if not softmax_cases:
        print("ERROR: No attn_softmax test cases found")
        raise RuntimeError("Test execution failed")

    if args.list:
        print(f"\nAttention SoftMax test cases:\n")
        for c in softmax_cases:
            print(f"  {c['id']}  -- {c.get('description', '')}  [shape={c['input'][0]['shape']}]")
        return

    device = args.device
    if device.startswith("npu"):
        device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
        import torch_npu  # noqa: F401
        torch.npu.set_device(device_id)
        device = f"npu:{device_id}"

    if args.case_id:
        match = [c for c in softmax_cases if c["id"] == args.case_id]
        if not match:
            print(f"ERROR: unknown case '{args.case_id}'")
            raise RuntimeError("Test execution failed")
        to_run = match
    else:
        to_run = softmax_cases

    passed = 0
    for case_data in to_run:
        print(f"\n>> Running {case_data['id']}")
        if run_single_case(case_data, device):
            passed += 1
        else:
            print(f"\nFailed at {case_data['id']}")
            raise RuntimeError("Test failed")

    print("\n" + "=" * 60)
    print(f"All Attention SoftMax tests passed! ({passed}/{len(to_run)} cases)")
    print("=" * 60)


if __name__ == "__main__":
    main()
