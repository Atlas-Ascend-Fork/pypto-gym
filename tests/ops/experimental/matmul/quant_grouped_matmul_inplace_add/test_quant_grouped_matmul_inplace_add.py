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

import os
import sys

import pypto
import torch_npu  # noqa: F401  # must come before pypto kernel imports

_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))


from dataclasses import dataclass

import torch
from experimental.matmul.quant_grouped_matmul_inplace_add.quant_grouped_matmul_inplace_add import (
    ShapeConfig,
    scaled_matmul_kernel,
)
from numpy.testing import assert_allclose
from quant_grouped_matmul_inplace_add_golden import GmmGoldenInputs, gen_golden


@dataclass
class QuantGroupedMatmulInplaceAddInputs:
    """
    Input parameters for generating quantized grouped matmul inplace add output.

    Attributes:
        a: Input tensor of shape [K, M] or [M, K] depending on a_trans flag
        b: Weight tensor of shape [K, N] or [N, K] depending on b_trans flag
        scaled_a: Scale factors for input tensor in MXFP8 format
        scaled_b: Scale factors for weight tensor in MXFP8 format
        y: Output tensor of shape [num_groups, M, N] (also serves as initial value for inplace add)
        tile_config: Tile configuration for computation including group splitting
    """
    a: torch.Tensor
    b: torch.Tensor
    scaled_a: torch.Tensor
    scaled_b: torch.Tensor
    y: torch.Tensor
    tile_config: 'ShapeConfig'


def quant_grouped_matmul_inplace_add(inputs: QuantGroupedMatmulInplaceAddInputs) -> torch.Tensor:
    """
    Generate quantized grouped matmul inplace add output using PyPTO scaled matrix multiplication.

    Args:
        inputs: Input parameters including tensors, scales, and tile config

    Returns:
        torch.Tensor: Output tensor of shape [num_groups, M, N] in FP32 after inplace accumulation
    """
    a = inputs.a.npu()
    b = inputs.b.npu()
    scaled_a = inputs.scaled_a.npu()
    scaled_b = inputs.scaled_b.npu()
    y = inputs.y.npu()

    # Execute scaled matrix multiplication kernel with inplace add
    scaled_matmul_kernel(a, b, scaled_a, scaled_b, y, inputs.tile_config)

    return y.to(torch.float32)


import pytest

_QUANT_GMM_TEST_CONFIGS = [
    ShapeConfig(
        ori_shape=[768, 6144, 4096],
        num_groups=32,
        m_tile_shape=[128, 128],
        k_tile_shape=[64, 192],
        n_tile_shape=[256, 1024],
        vector_tile_shape=[1, 32, 512, 2],
        in_dtype=pypto.DT_FP8E4M3,
        a_trans=True,
        b_trans=False,
        a_format_nz=False,
        b_format_nz=False,
        c_format_nz=False,
        description="Case1",
    ),
    ShapeConfig(
        ori_shape=[768, 8192, 4096],
        num_groups=32,
        m_tile_shape=[128, 128],
        k_tile_shape=[64, 192],
        n_tile_shape=[256, 1024],
        vector_tile_shape=[1, 32, 512, 2],
        in_dtype=pypto.DT_FP8E4M3,
        a_trans=True,
        b_trans=False,
        a_format_nz=False,
        b_format_nz=False,
        c_format_nz=False,
        description="Case2",
    ),
    ShapeConfig(
        ori_shape=[2048, 6144, 4096],
        num_groups=16,
        m_tile_shape=[128, 128],
        k_tile_shape=[64, 192],
        n_tile_shape=[256, 1024],
        vector_tile_shape=[1, 32, 512, 2],
        in_dtype=pypto.DT_FP8E5M2,
        a_trans=True,
        b_trans=False,
        a_format_nz=False,
        b_format_nz=False,
        c_format_nz=False,
        description="Case3",
    ),
    ShapeConfig(
        ori_shape=[2048, 7168, 4096],
        num_groups=16,
        m_tile_shape=[128, 128],
        k_tile_shape=[64, 192],
        n_tile_shape=[256, 1024],
        vector_tile_shape=[1, 32, 512, 2],
        in_dtype=pypto.DT_FP8E5M2,
        a_trans=True,
        b_trans=False,
        a_format_nz=False,
        b_format_nz=False,
        c_format_nz=False,
        description="Case4",
    ),
]


def _generate_mxfp8_a(m, k, num_groups, torch_dtype, a_trans):
    """Generate MXFP8-format input tensor and its scale factors."""
    if a_trans:
        a = torch.randn((k, m), dtype=torch.float32).uniform_(0, 1).to(torch_dtype)
        scaled_a = torch.randn(
    (k // 64 + num_groups, m, 2),
    dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
    else:
        a = torch.randn((m, k), dtype=torch.float32).uniform_(0, 1).to(torch_dtype)
        scaled_a = torch.randn(
    (m, k // 64 + num_groups, 2),
    dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
    return a, scaled_a


def _generate_mxfp8_b(k, n, num_groups, torch_dtype, b_trans):
    """Generate MXFP8-format weight tensor and its scale factors."""
    if b_trans:
        b = torch.randn((n, k), dtype=torch.float32).uniform_(0, 1).to(torch_dtype)
        scaled_b = torch.randn(
    (n, k // 64 + num_groups, 2),
    dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
    else:
        b = torch.randn((k, n), dtype=torch.float32).uniform_(0, 1).to(torch_dtype)
        scaled_b = torch.randn(
    (k // 64 + num_groups, n, 2),
    dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
    return b, scaled_b


@pytest.mark.soc("950")
@pytest.mark.parametrize("tile_config", _QUANT_GMM_TEST_CONFIGS)
def test_quant_grouped_matmul_inplace_add(tile_config):
    """
    Test the quantized grouped matrix multiplication with inplace add.

    This function runs a complete test for a given configuration:
    1. Generate test data with MXFP8 format
    2. Compute golden (reference) output using PyTorch
    3. Compute output using PyPTO
    4. Compare results

    Args:
        tile_config: Configuration parameters for the test case
    """
    # Extract configuration parameters
    m = tile_config.ori_shape[0]
    k = tile_config.ori_shape[1]
    n = tile_config.ori_shape[2]
    num_groups = tile_config.num_groups
    in_dtype = tile_config.in_dtype
    a_trans = tile_config.a_trans
    b_trans = tile_config.b_trans

    # Map PyPTO data type to torch dtype
    torch_dtype_map = {
        pypto.DT_FP8E4M3: torch.float8_e4m3fn,
        pypto.DT_FP8E5M2: torch.float8_e5m2,
    }
    torch_dtype = torch_dtype_map.get(in_dtype, torch.float8_e4m3fn)

    # Generate input and weight tensors in MXFP8 format
    a, scaled_a = _generate_mxfp8_a(m, k, num_groups, torch_dtype, a_trans)
    b, scaled_b = _generate_mxfp8_b(k, n, num_groups, torch_dtype, b_trans)

    # Initialize output tensor with random values (for inplace add)
    y_init = torch.randn((num_groups, m, n), dtype=torch.float32)
    y_init_npu = y_init.clone().npu()

    # Compute golden (reference) output
    golden = gen_golden(GmmGoldenInputs(
        a=a,
        b=b,
        scaled_a=scaled_a,
        scaled_b=scaled_b,
        y=y_init,
        num_groups=num_groups,
        a_trans=a_trans,
        b_trans=b_trans,
    ))

    # Compute output using PyPTO
    result = quant_grouped_matmul_inplace_add(QuantGroupedMatmulInplaceAddInputs(
        a=a,
        b=b,
        scaled_a=scaled_a,
        scaled_b=scaled_b,
        y=y_init_npu,
        tile_config=tile_config,
    ))

    # Verify results
    assert_allclose(golden.cpu().numpy(), result.cpu().numpy(), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    for cfg in _QUANT_GMM_TEST_CONFIGS:
        test_quant_grouped_matmul_inplace_add(cfg)