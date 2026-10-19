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
MRoPE 精度测试脚本
遍历 mrope_test_cases.json 执行精度对比
"""

import os
import sys
from pathlib import Path

import json
import logging
import pytest
import torch
import torch_npu

_CUR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CUR))
_IMPL = Path(__file__).resolve().parents[4] / "ops/pypto_tensor/gutenocr_3b/mrope"
sys.path.insert(0, str(_IMPL))

import numpy as np
from numpy.testing import assert_allclose

from mrope_golden import mrope_golden
from mrope_impl import mrope_pto_correct


def load_test_cases():
    json_path = os.path.join(os.path.dirname(__file__), "mrope_test_cases.json")
    if not os.path.exists(json_path):
        raise RuntimeError(f"Test cases file not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
        return data.get("test_cases", [])


@pytest.fixture(scope="module")
def device():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    return f"npu:{device_id}"


@pytest.fixture(scope="module")
def test_cases():
    return load_test_cases()


def run_single_case(case_data):
    case_id = case_data["id"]
    description = case_data.get("description", "")

    logging.info("\n" + "=" * 60)
    logging.info(f"Test: {case_id} — {description}")
    logging.info("=" * 60)

    if torch.npu.is_available():
        dev_id = os.environ.get('TILE_FWK_DEVICE_ID', '0')
        torch.npu.set_device(f'npu:{dev_id}')

    torch.manual_seed(case_data.get("seed", 42))

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}

    inputs = case_data["input"]
    q_dtype = dtype_map[inputs["q"]["dtype"]]
    k_dtype = dtype_map[inputs["k"]["dtype"]]
    cos_dtype = dtype_map[inputs["cos"]["dtype"]]

    q = torch.randn(inputs["q"]["shape"], dtype=q_dtype, device="cpu")
    k = torch.randn(inputs["k"]["shape"], dtype=k_dtype, device="cpu")
    cos = torch.randn(inputs["cos"]["shape"], dtype=cos_dtype, device="cpu")
    sin = torch.randn(inputs["sin"]["shape"], dtype=cos_dtype, device="cpu")

    mrope_section = inputs["mrope_section"]
    unsqueeze_dim = inputs["unsqueeze_dim"]

    q_npu = q.npu()
    k_npu = k.npu()
    cos_npu = cos.npu()
    sin_npu = sin.npu()

    q_golden_cpu, k_golden_cpu = mrope_golden(q, k, cos, sin, mrope_section, unsqueeze_dim)
    q_golden = q_golden_cpu.npu()
    k_golden = k_golden_cpu.npu()

    q_impl, k_impl = mrope_pto_correct(q_npu, k_npu, cos_npu, sin_npu, mrope_section, unsqueeze_dim)

    logging.info("\n[precision validation - q]")
    q_diff = torch.abs(q_golden - q_impl).max().item()
    logging.info(f"  Max diff: {q_diff:.6e}")

    logging.info("[precision validation - k]")
    k_diff = torch.abs(k_golden - k_impl).max().item()
    logging.info(f"  Max diff: {k_diff:.6e}")

    rtol = case_data.get("rtol", 1e-5)
    atol = case_data.get("atol", 1e-5)

    if q_impl.dtype == torch.bfloat16:
        assert_allclose(q_impl.cpu().float().numpy(), q_golden.cpu().float().numpy(), rtol=rtol, atol=atol)
        assert_allclose(k_impl.cpu().float().numpy(), k_golden.cpu().float().numpy(), rtol=rtol, atol=atol)
    else:
        assert_allclose(q_impl.cpu().numpy(), q_golden.cpu().numpy(), rtol=rtol, atol=atol)
        assert_allclose(k_impl.cpu().numpy(), k_golden.cpu().numpy(), rtol=rtol, atol=atol)
    logging.info(f"[PRECISION_PASS] diff < {rtol}")

    outputs = case_data["output"]
    expected_q_shape = torch.Size(outputs["q_shape"])
    expected_k_shape = torch.Size(outputs["k_shape"])
    assert q_impl.shape == expected_q_shape, f"q shape mismatch: {q_impl.shape} vs {expected_q_shape}"
    assert k_impl.shape == expected_k_shape, f"k shape mismatch: {k_impl.shape} vs {expected_k_shape}"


@pytest.mark.parametrize("case_data", load_test_cases(), ids=lambda c: c["id"])
def test_mrope(case_data):
    run_single_case(case_data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v", "-s"])
