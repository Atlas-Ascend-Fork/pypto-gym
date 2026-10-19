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
RoPE PyPTO Implementation 精度测试脚本

验证 PyPTO implementation (rope_impl.py)
与 Golden 参考实现的精度一致性

关键：导入 impl 函数（PyPTO kernel实现），测试核心算子功能

"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch_npu

_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

import numpy as np
from numpy.testing import assert_allclose

try:
    from rope_golden import apply_multimodal_rotary_pos_emb_golden, apply_rotary_pos_emb_vision_golden
except ImportError:
    _p = os.path.dirname(__file__)
    sys.path.insert(0, _p)
    from rope_golden import (
        apply_rotary_pos_emb_vision_golden,
        apply_multimodal_rotary_pos_emb_golden
    )

from spatial_ssrl_3b.rope.rope_impl import (
    apply_multimodal_rotary_pos_emb_pto_impl,
    apply_rotary_pos_emb_vision_pto_impl,
)


def prep_env():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)


def load_test_cases():
    json_path = os.path.join(os.path.dirname(__file__), "test_cases_rope.json")
    if not os.path.exists(json_path):
        raise RuntimeError(f"Test cases file not found: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)


def run_single_case(case_data):
    case_id = case_data["id"]
    description = case_data.get("description", "")
    
    print("=" * 60)
    print(f"Test: {case_id} — {description}")
    print("=" * 60)
    
    torch.manual_seed(case_data.get("seed", 42))
    
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    
    inputs = case_data["input"]
    rope_type = inputs.get("rope_type", "multimodal")
    
    dtype = dtype_map.get(inputs["q"]["dtype"], torch.float16)
    
    q = torch.randn(inputs["q"]["shape"], dtype=dtype).npu()
    k = torch.randn(inputs["k"]["shape"], dtype=dtype).npu()
    cos = torch.randn(inputs["cos"]["shape"], dtype=dtype).npu()
    sin = torch.randn(inputs["sin"]["shape"], dtype=dtype).npu()
    
    if rope_type == "multimodal":
        mrope_section = inputs.get("mrope_section", [16, 24, 24])
        unsqueeze_dim = inputs.get("unsqueeze_dim", 1)
        
        q_golden_cpu, k_golden_cpu = apply_multimodal_rotary_pos_emb_golden(
            q.cpu(), k.cpu(), cos.cpu(), sin.cpu(), mrope_section, unsqueeze_dim
        )
        q_golden = q_golden_cpu.npu()
        k_golden = k_golden_cpu.npu()
        
        q_impl, k_impl = apply_multimodal_rotary_pos_emb_pto_impl(
            q, k, cos, sin, mrope_section, unsqueeze_dim
        )
    else:  # vision
        q_golden_cpu, k_golden_cpu = apply_rotary_pos_emb_vision_golden(
            q.cpu(), k.cpu(), cos.cpu(), sin.cpu()
        )
        q_golden = q_golden_cpu.npu()
        k_golden = k_golden_cpu.npu()
        
        q_impl, k_impl = apply_rotary_pos_emb_vision_pto_impl(q, k, cos, sin)
    
    q_diff = torch.abs(q_golden - q_impl).max().item()
    k_diff = torch.abs(k_golden - k_impl).max().item()
    
    print(f"  q max diff: {q_diff:.6e}")
    print(f"  k max diff: {k_diff:.6e}")
    
    rtol = case_data.get("rtol", 5e-3)
    atol = case_data.get("atol", 5e-3)
    
    try:
        assert_allclose(q_impl.cpu().numpy(), q_golden.cpu().numpy(), rtol=rtol, atol=atol)
        assert_allclose(k_impl.cpu().numpy(), k_golden.cpu().numpy(), rtol=rtol, atol=atol)
        print(f"[PRECISION_PASS] diff < {rtol}")
    except AssertionError as e:
        print(f"[PRECISION_FAIL] {e}", file=sys.stderr)
        raise
    
    expected_q_shape = case_data["output"]["q_shape"]
    expected_k_shape = case_data["output"]["k_shape"]
    expected_dtype = dtype_map.get(case_data["output"]["dtype"], torch.float16)
    
    assert q_impl.shape == torch.Size(expected_q_shape), \
        f"q shape mismatch: {q_impl.shape} vs {expected_q_shape}"
    assert k_impl.shape == torch.Size(expected_k_shape), \
        f"k shape mismatch: {k_impl.shape} vs {expected_k_shape}"
    assert q_impl.dtype == expected_dtype, f"q dtype mismatch: {q_impl.dtype} vs {expected_dtype}"
    assert k_impl.dtype == expected_dtype, f"k dtype mismatch: {k_impl.dtype} vs {expected_dtype}"
    
    if not torch.isnan(q_impl).any() and not torch.isnan(k_impl).any():
        print("[NaN_CHECK_PASS] No NaN in output")
    else:
        print("[NaN_CHECK_FAIL] NaN detected in output")
        raise RuntimeError("NaN detected in PyPTO kernel output")
    
    if not torch.isinf(q_impl).any() and not torch.isinf(k_impl).any():
        print("[INF_CHECK_PASS] No Inf in output")
    else:
        print("[INF_CHECK_FAIL] Inf detected in output")
        raise RuntimeError("Inf detected in PyPTO kernel output")


def test_rope_spatial_ssrl_3b():
    parser = argparse.ArgumentParser(description="RoPE PyPTO Kernel 精度测试")
    parser.add_argument("case_id", nargs="?", help="运行单个用例")
    parser.add_argument("--list", action="store_true", help="列出所有用例")
    args, _ = parser.parse_known_args()
    
    test_cases = load_test_cases()
    cases = test_cases.get("test_cases", [])
    
    if args.list:
        print(f"\nTest cases from test_cases_rope.json:\n")
        for case in cases:
            print(f"  {case['id']} — {case.get('description', '')}")
        return
    
    prep_env()
    
    print("\n验证 PyPTO implementation (rope_impl.py):")
    print("  - @pypto.frontend.jit 装饰器")
    print("  - pypto tensor operations (neg, concat, add, mul)")
    print("  - 导入 impl 函数（PyPTO kernel实现）")
    print("  - Vision RoPE impl: apply_rotary_pos_emb_vision_pto_impl")
    print("  - Multimodal RoPE impl: apply_multimodal_rotary_pos_emb_pto_impl")
    print("")
    
    to_run = cases if not args.case_id else [c for c in cases if c["id"] == args.case_id]
    
    try:
        for case_data in to_run:
            run_single_case(case_data)
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("PyPTO implementation验证成功")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    test_rope_spatial_ssrl_3b()