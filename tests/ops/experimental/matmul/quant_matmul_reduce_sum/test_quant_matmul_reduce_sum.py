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

"""PyPTO quant_matmul_reduce_sum operator test.

测试说明：
  - golden 实现来自 quant_matmul_reduce_sum_golden.py
  - kernel 与 host 封装来自 quant_matmul_reduce_sum_impl.py
  - 精度对比使用 numpy.testing.assert_allclose
"""

import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

import argparse

import pytest
import pypto
import torch
import torch_npu  # noqa: F401
from numpy.testing import assert_allclose
from torch.nn import functional as F

from quant_matmul_reduce_sum_golden import quant_matmul_reduce_sum_golden
from experimental.matmul.quant_matmul_reduce_sum.quant_matmul_reduce_sum_impl import (
    QuantMatmulReduceSumConfig,
    quant_matmul_reduce_sum_wrapper,
)


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

def get_device_id():
    return int(os.environ.get('TILE_FWK_DEVICE_ID', 0))


# ---------------------------------------------------------------------------
# Utility: ND -> Fractal NZ conversion (for test data preparation)
# ---------------------------------------------------------------------------

def trans_nd_to_fractal_nz(data: torch.Tensor, keep_m_dim=False):
    """Transform ND tensor to fractal NZ format.

    Args:
        data: Input tensor of shape [..., M, N]
        keep_m_dim: Whether to keep M dimension unchanged (default: False)

    Returns:
        torch.Tensor: Transformed tensor in fractal NZ format
    """
    def _gen_axes_for_transpose(offset, base):
        return [x for x in range(offset)] + [x + offset for x in base]

    def _ceil_div(a, b):
        return (a + b - 1) // b

    ori_shape = data.shape
    m_ori, n_ori = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_num = len(batch_ori)
    m0 = 16
    n0 = 32 // data.dtype.itemsize
    if data.dtype == torch.int32:
        n0 = 16
    m1, n1 = _ceil_div(m_ori, m0), _ceil_div(n_ori, n0)
    padding_m = m1 * m0 - m_ori
    padding_n = n1 * n0 - n_ori
    if not keep_m_dim:
        pad_list = [0, padding_n, 0, padding_m] + [0, 0] * batch_num
        data = F.pad(data, pad_list, "constant")
        array_trans = _gen_axes_for_transpose(len(data.shape) - 2, [2, 0, 1, 3])
        data = data.reshape(batch_ori + (m1, m0, n1, n0)).permute(*array_trans).contiguous()
    else:
        pad_list = [0, padding_n, 0, 0] + [0, 0] * batch_num
        data = F.pad(data, pad_list, "constant")
        array_trans = _gen_axes_for_transpose(len(data.shape) - 2, [1, 0, 2])
        data = data.reshape(batch_ori + (m_ori, n1, n0)).permute(*array_trans).contiguous()
    return data


# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------

TEST_CONFIGS = [
    QuantMatmulReduceSumConfig(
        ori=[2, 128, 128, 128],
        m_tile_shape=[128, 128],
        k_tile_shape=[128, 128],
        n_tile_shape=[128, 128],
        in_dtype=pypto.DT_INT8,
        out_dtype=pypto.DT_BF16,
        x2_format_nz=True,
        vec_tile_shapes=[64, 64, 64],
        description="P0 core: batch=2, m=128, k=128, n=128, NZ format",
    ),
]


# ---------------------------------------------------------------------------
# Test function (pytest-compatible)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("config", TEST_CONFIGS, ids=[c.description for c in TEST_CONFIGS])
def test_quant_matmul_reduce_sum(config: QuantMatmulReduceSumConfig):
    """Test quantized matmul reduce sum: generate data, run golden & PyPTO, compare."""
    b, m, k, n = config.ori

    device_id = get_device_id()
    torch.npu.set_device(device_id)
    torch_npu.npu.config.allow_internal_format = True

    torch.manual_seed(42)
    x1 = torch.randint(-10, 10, (b, m, k), dtype=torch.int8).npu()
    x2_nd = torch.randint(-10, 10, (b, k, n), dtype=torch.int8).npu()
    x2_nz = torch_npu.npu_format_cast(x2_nd, torch_npu.Format.FRACTAL_NZ)
    x1_scale = torch.randn((b, m), dtype=torch.float32).uniform_(0.5, 1.5).npu()
    x2_scale = torch.randn((n,), dtype=torch.bfloat16).uniform_(0.5, 1.5).npu()

    # Select x2 input based on format configuration
    if config.x2_format_nz:
        x2_input = x2_nz
    else:
        x2_input = x2_nd

    pypto_out = quant_matmul_reduce_sum_wrapper(x1, x2_input, x1_scale, x2_scale, config)
    golden_out = quant_matmul_reduce_sum_golden(x1.cpu(), x2_nd.cpu(), x1_scale.cpu(), x2_scale.cpu())

    pypto_out_cpu = pypto_out.cpu().float()
    golden_out_cpu = golden_out.cpu().float()

    assert_allclose(pypto_out_cpu, golden_out_cpu, rtol=0.001, atol=0.001)
    print(f"[PASS] {config.description}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PyPTO quant_matmul_reduce_sum operator test")
    parser.add_argument(
        "case",
        type=int,
        nargs="?",
        metavar="N",
        help="Run single 1-based case index (default: run all)",
    )
    parser.add_argument("--list", action="store_true", help="List all test cases and exit")
    args = parser.parse_args()

    if args.list:
        for i, cfg in enumerate(TEST_CONFIGS, 1):
            print(f"  {i}. {cfg.description}")
        return

    if args.case is not None:
        if not 1 <= args.case <= len(TEST_CONFIGS):
            print(f"ERROR: case must be 1..{len(TEST_CONFIGS)}", file=sys.stderr)
            raise RuntimeError("Test execution failed")
        test_quant_matmul_reduce_sum(TEST_CONFIGS[args.case - 1])
        return

    for test_config in TEST_CONFIGS:
        test_quant_matmul_reduce_sum(test_config)


if __name__ == "__main__":
    main()
