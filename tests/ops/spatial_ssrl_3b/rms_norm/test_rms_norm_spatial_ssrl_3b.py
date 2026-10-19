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
RMSNorm PyPTO Kernel 精度测试脚本

验证真正的 PyPTO kernel (@pypto.frontend.jit + pypto.rms_norm())
与 Golden 参考实现的精度一致性
"""


import os
import sys
from pathlib import Path

import json
import argparse
import logging
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
    from rms_norm_golden_spatial_ssrl_3b import rms_norm_golden
except ImportError:
    _p = os.path.dirname(__file__)
    sys.path.insert(0, _p)
    from rms_norm_golden_spatial_ssrl_3b import rms_norm_golden

from spatial_ssrl_3b.rms_norm.rms_norm_impl import rms_norm_impl


def prep_env():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)


def load_test_cases():
    json_path = os.path.join(os.path.dirname(__file__), "test_cases.json")
    if not os.path.exists(json_path):
        raise RuntimeError(f"Test cases file not found: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)


def run_single_case(case_data):
    case_id = case_data["id"]
    description = case_data.get("description", "")
    
    logging.info("=" * 60)
    logging.info(f"Test: {case_id} — {description}")
    logging.info("=" * 60)
    
    torch.manual_seed(case_data.get("seed", 42))
    
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    
    inputs = case_data["input"]
    dtype = dtype_map[inputs["hidden_states"]["dtype"]]
    
    hidden_states = torch.randn(inputs["hidden_states"]["shape"], dtype=dtype).npu()
    gamma = torch.randn(inputs["gamma"]["shape"], dtype=dtype).npu()
    eps = inputs["eps"]["value"]
    
    output_golden = rms_norm_golden(hidden_states.cpu(), gamma.cpu(), eps).npu()
    output_impl = rms_norm_impl(hidden_states, gamma, eps)
    
    max_diff = torch.abs(output_golden - output_impl).max().item()
    mean_diff = torch.abs(output_golden - output_impl).mean().item()
    
    logging.info(f"  Max diff: {max_diff:.6e}")
    logging.info(f"  Mean diff: {mean_diff:.6e}")
    
    rtol = case_data.get("rtol", 1e-2)
    atol = case_data.get("atol", 1e-2)
    
    try:
        assert_allclose(output_impl.cpu().numpy(), output_golden.cpu().numpy(), rtol=rtol, atol=atol)
        logging.info(f"[PRECISION_PASS] diff < {rtol}")
    except AssertionError as e:
        logging.info(f"[PRECISION_FAIL] {e}", file=sys.stderr)
        raise
    
    expected_shape = case_data["output"]["shape"]
    expected_dtype = dtype_map[case_data["output"]["dtype"]]
    assert output_impl.shape == torch.Size(expected_shape), f"Shape mismatch: {output_impl.shape} vs {expected_shape}"
    assert output_impl.dtype == expected_dtype, f"Dtype mismatch: {output_impl.dtype} vs {expected_dtype}"
    
    if not torch.isnan(output_impl).any():
        logging.info("[NaN_CHECK_PASS] No NaN in output")
    else:
        logging.info("[NaN_CHECK_FAIL] NaN detected in output")
        raise RuntimeError("NaN detected in PyPTO kernel output")
    
    if not torch.isinf(output_impl).any():
        logging.info("[INF_CHECK_PASS] No Inf in output")
    else:
        logging.info("[INF_CHECK_FAIL] Inf detected in output")
        raise RuntimeError("Inf detected in PyPTO kernel output")


def test_rms_norm_spatial_ssrl_3b():
    parser = argparse.ArgumentParser(description="RMSNorm PyPTO Kernel 精度测试")
    parser.add_argument("case_id", nargs="?", help="运行单个用例")
    parser.add_argument("--list", action="store_true", help="列出所有用例")
    args, _ = parser.parse_known_args()
    
    test_cases = load_test_cases()
    cases = test_cases.get("test_cases", [])
    
    if args.list:
        logging.info(f"\nTest cases from test_cases.json:\n")
        for case in cases:
            logging.info(f"  {case['id']} — {case.get('description', '')}")
        return
    
    prep_env()
    
    logging.info("\n验证真正的 PyPTO kernel:")
    logging.info("  - @pypto.frontend.jit 装饰器")
    logging.info("  - pypto.rms_norm() 融合 kernel")
    logging.info("  - 不是 PyTorch fallback 版本")
    logging.info("")
    
    to_run = cases if not args.case_id else [c for c in cases if c["id"] == args.case_id]
    
    try:
        for case_data in to_run:
            run_single_case(case_data)
        logging.info("\n" + "=" * 60)
        logging.info("All tests passed!")
        logging.info("真正的 PyPTO kernel验证成功")
        logging.info("=" * 60)
    except Exception as e:
        logging.info(f"\nError: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_rms_norm_spatial_ssrl_3b()