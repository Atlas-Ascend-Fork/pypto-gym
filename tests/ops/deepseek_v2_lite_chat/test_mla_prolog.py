#!/usr/bin/env python3
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
MLA KV Prolog 精度测试脚本
遍历 test_cases.json 执行精度对比
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

_CUR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CUR))
_IMPL = Path(__file__).resolve().parents[3] / "ops/pypto_tensor/deepseek_v2_lite_chat/mla_prolog"
sys.path.insert(0, str(_IMPL))

import numpy as np
from mla_prolog import mla_prolog_hybrid_optimized
from mla_prolog_golden import mla_prolog_golden
from numpy.testing import assert_allclose


def prep_env():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)


def get_device():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    return f"npu:{device_id}"


def load_test_cases():
    json_path = os.path.join(os.path.dirname(__file__), "test_cases.json")
    if not os.path.exists(json_path):
        raise RuntimeError(f"Test cases file not found: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)


def _prepare_case_inputs(case_data):
    """Parse test case JSON and create input tensors for a single case."""
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    int_dtype_map = {"int64": torch.long, "int32": torch.int32}

    inputs = case_data["input"]

    hidden_dtype = dtype_map.get(inputs["hidden_states"]["dtype"], torch.float16)
    hidden_states = torch.randn(inputs["hidden_states"]["shape"], dtype=hidden_dtype, device="cpu")

    kv_a_weight = torch.randn(inputs["kv_a_weight"]["shape"],
                               dtype=dtype_map.get(inputs["kv_a_weight"]["dtype"], torch.float16), device="cpu")
    kv_b_weight = torch.randn(inputs["kv_b_weight"]["shape"],
                               dtype=dtype_map.get(inputs["kv_b_weight"]["dtype"], torch.float16), device="cpu")
    ln_weight = torch.randn(inputs["ln_weight"]["shape"],
                             dtype=dtype_map.get(inputs["ln_weight"]["dtype"], torch.float16), device="cpu")
    eps = inputs["eps"]["value"]
    cos = torch.randn(inputs["cos"]["shape"], dtype=dtype_map.get(inputs["cos"]["dtype"], torch.float16), device="cpu")
    sin = torch.randn(inputs["sin"]["shape"], dtype=dtype_map.get(inputs["sin"]["dtype"], torch.float16), device="cpu")

    pos_ids_dtype = int_dtype_map.get(inputs["pos_ids"]["dtype"], torch.long)
    bsz = inputs["hidden_states"]["shape"][0]
    seq_len = inputs["hidden_states"]["shape"][1]
    pos_ids = torch.arange(seq_len, dtype=pos_ids_dtype, device="cpu").unsqueeze(0).expand(bsz, -1)

    return hidden_states, kv_a_weight, kv_b_weight, ln_weight, eps, cos, sin, pos_ids


def _verify_precision(k_nope_impl, k_nope_golden, value_impl, value_golden,
                      k_pe_impl, k_pe_golden, case_data):
    """Verify precision of all three outputs and raise on failure."""
    rtol = case_data.get("rtol", 1e-2)
    atol = case_data.get("atol", 1e-2)

    print("\n[精度验证 - k_nope]")
    print(f"  Max diff: {torch.abs(k_nope_golden - k_nope_impl).max().item():.6e}")
    try:
        assert_allclose(k_nope_impl.cpu().numpy(), k_nope_golden.cpu().numpy(), rtol=rtol, atol=atol)
        print(f"[PRECISION_PASS] k_nope diff < {rtol}")
    except AssertionError as e:
        print(f"[PRECISION_FAIL] k_nope: {e}", file=sys.stderr)
        raise

    print("\n[精度验证 - value]")
    print(f"  Max diff: {torch.abs(value_golden - value_impl).max().item():.6e}")
    try:
        assert_allclose(value_impl.cpu().numpy(), value_golden.cpu().numpy(), rtol=rtol, atol=atol)
        print(f"[PRECISION_PASS] value diff < {rtol}")
    except AssertionError as e:
        print(f"[PRECISION_FAIL] value: {e}", file=sys.stderr)
        raise

    print("\n[精度验证 - k_pe]")
    print(f"  Max diff: {torch.abs(k_pe_golden - k_pe_impl).max().item():.6e}")
    try:
        assert_allclose(k_pe_impl.cpu().numpy(), k_pe_golden.cpu().numpy(), rtol=rtol, atol=atol)
        print(f"[PRECISION_PASS] k_pe diff < {rtol}")
    except AssertionError as e:
        print(f"[PRECISION_FAIL] k_pe: {e}", file=sys.stderr)
        raise


def _verify_output_shapes(k_nope_impl, value_impl, k_pe_impl, case_data):
    """Verify output tensor shapes and dtypes match spec."""
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    outputs = case_data["output"]

    expected_k_nope_shape = torch.Size(outputs["k_nope"]["shape"])
    expected_k_nope_dtype = dtype_map.get(outputs["k_nope"]["dtype"], torch.float16)
    assert k_nope_impl.shape == expected_k_nope_shape, \
        f"k_nope shape mismatch: {k_nope_impl.shape} vs {expected_k_nope_shape}"
    assert k_nope_impl.dtype == expected_k_nope_dtype, \
        f"k_nope dtype mismatch: {k_nope_impl.dtype} vs {expected_k_nope_dtype}"

    expected_value_shape = torch.Size(outputs["value"]["shape"])
    expected_value_dtype = dtype_map.get(outputs["value"]["dtype"], torch.float16)
    assert value_impl.shape == expected_value_shape, \
        f"value shape mismatch: {value_impl.shape} vs {expected_value_shape}"
    assert value_impl.dtype == expected_value_dtype, \
        f"value dtype mismatch: {value_impl.dtype} vs {expected_value_dtype}"

    expected_k_pe_shape = torch.Size(outputs["k_pe"]["shape"])
    expected_k_pe_dtype = dtype_map.get(outputs["k_pe"]["dtype"], torch.float16)
    assert k_pe_impl.shape == expected_k_pe_shape, \
        f"k_pe shape mismatch: {k_pe_impl.shape} vs {expected_k_pe_shape}"
    assert k_pe_impl.dtype == expected_k_pe_dtype, \
        f"k_pe dtype mismatch: {k_pe_impl.dtype} vs {expected_k_pe_dtype}"


def run_single_case(case_data, device):
    case_id = case_data["id"]
    description = case_data.get("description", "")

    print("=" * 60)
    print(f"Test: {case_id} — {description}")
    print("=" * 60)

    torch.manual_seed(case_data.get("seed", 42))

    hidden_states, kv_a_weight, kv_b_weight, ln_weight, eps, cos, sin, pos_ids = \
        _prepare_case_inputs(case_data)
    
    hidden_states = hidden_states.npu()
    kv_a_weight = kv_a_weight.npu()
    kv_b_weight = kv_b_weight.npu()
    ln_weight = ln_weight.npu()
    cos = cos.npu()
    sin = sin.npu()
    pos_ids = pos_ids.npu()

    k_nope_golden, value_golden, k_pe_golden = mla_prolog_golden(
        hidden_states, kv_a_weight, kv_b_weight, ln_weight, eps, cos, sin, pos_ids)

    k_nope_impl, value_impl, k_pe_impl = mla_prolog_hybrid_optimized(
        hidden_states, kv_a_weight, kv_b_weight, ln_weight, eps, cos, sin, pos_ids)

    _verify_precision(k_nope_impl, k_nope_golden, value_impl, value_golden,
                      k_pe_impl, k_pe_golden, case_data)
    _verify_output_shapes(k_nope_impl, value_impl, k_pe_impl, case_data)


def test_deepseek_v2_lite_chat_mla_prolog():
    parser = argparse.ArgumentParser(description="MLA KV Prolog 精度测试")
    parser.add_argument("case_id", nargs="?", help="运行单个用例")
    parser.add_argument("--list", action="store_true", help="列出所有用例")
    args, _ = parser.parse_known_args()

    test_cases = load_test_cases()
    cases = test_cases.get("test_cases", [])

    if args.list:
        print(f"\nTest cases from test_cases.json:\n")
        for case in cases:
            print(f"  {case['id']} — {case.get('description', '')}")
        return

    prep_env()
    device = get_device()
    print("=================")
    print(device)

    to_run = cases if not args.case_id else [c for c in cases if c["id"] == args.case_id]

    try:
        for case_data in to_run:
            run_single_case(case_data, device)
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    test_deepseek_v2_lite_chat_mla_prolog()
