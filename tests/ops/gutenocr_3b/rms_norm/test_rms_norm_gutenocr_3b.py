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
RMSNorm 精度测试脚本
遍历 test_cases.json 执行精度对比
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
_IMPL = Path(__file__).resolve().parents[4] / "ops/pypto_tensor/gutenocr_3b/rms_norm"
sys.path.insert(0, str(_IMPL))

import numpy as np
from numpy.testing import assert_allclose

from rms_norm_golden_gutenocr_3b import rms_norm_golden
from rms_norm_impl import rms_norm_pto_native


def prep_env():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)


def get_device():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    return f"npu:{device_id}"


def load_test_cases():
    json_path = os.path.join(os.path.dirname(__file__), "rms_norm_test_cases.json")
    if not os.path.exists(json_path):
        raise RuntimeError(f"Test cases file not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
        return data.get("test_cases", [])


@pytest.fixture(scope="module")
def device():
    prep_env()
    return get_device()


@pytest.fixture(scope="module")
def test_cases():
    return load_test_cases()


def run_single_case(case_data):
    case_id = case_data["id"]
    description = case_data.get("description", "")

    logging.info("\n" + "=" * 60)
    logging.info(f"Test: {case_id} — {description}")
    logging.info("=" * 60)

    torch.manual_seed(case_data.get("seed", 42))

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}

    inputs = case_data["input"]

    hidden_dtype = dtype_map[inputs["hidden_states"]["dtype"]]
    hidden_states = torch.randn(inputs["hidden_states"]["shape"], dtype=hidden_dtype, device="cpu")

    weight_dtype = dtype_map[inputs["weight"]["dtype"]]
    weight = torch.randn(inputs["weight"]["shape"], dtype=weight_dtype, device="cpu")

    eps = inputs["eps"]["value"]

    hidden_states_npu = hidden_states.npu()
    weight_npu = weight.npu()

    output_golden = rms_norm_golden(hidden_states_npu, weight_npu, eps)

    output_impl = rms_norm_pto_native(hidden_states_npu, weight_npu, eps)

    logging.info("\n[精度验证 - output]")
    output_diff = torch.abs(output_golden - output_impl).max().item()
    logging.info(f"  Max diff: {output_diff:.6e}")

    rtol = case_data.get("rtol", 1e-2)
    atol = case_data.get("atol", 1e-2)

    if output_impl.dtype == torch.bfloat16:
        output_impl_fp32 = output_impl.cpu().float()
        output_golden_fp32 = output_golden.cpu().float()
        assert_allclose(output_impl_fp32.numpy(), output_golden_fp32.numpy(), rtol=rtol, atol=atol)
    else:
        assert_allclose(output_impl.cpu().numpy(), output_golden.cpu().numpy(), rtol=rtol, atol=atol)
    logging.info(f"[PRECISION_PASS] output diff < {rtol}")

    outputs = case_data["output"]

    expected_output_shape = torch.Size(outputs["output"]["shape"])
    expected_output_dtype = dtype_map[outputs["output"]["dtype"]]
    assert output_impl.shape == expected_output_shape, \
        f"output shape mismatch: {output_impl.shape} vs {expected_output_shape}"
    assert output_impl.dtype == expected_output_dtype, \
        f"output dtype mismatch: {output_impl.dtype} vs {expected_output_dtype}"


@pytest.mark.parametrize("case_data", load_test_cases(), ids=lambda c: c["id"])
def test_rms_norm(device, case_data):
    run_single_case(case_data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v", "-s"])