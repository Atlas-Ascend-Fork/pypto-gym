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
from typing import List

import torch
from experimental.matmul.transpose_quant_batch_matmul.transpose_quant_batch_matmul_impl import (
    ShapeConfig,
    transpose_quant_batch_mat_mul_kernel,
)
from numpy.testing import assert_allclose
from transpose_quant_batch_matmul_golden import TqbmmGoldenInputs, gen_golden


@dataclass
class TransposeQuantBatchMatMulInputs:
    """
    Input parameters for generating transpose quantized batch matmul output.

    Attributes:
        x1: Input tensor of shape [M, B, K] in FP8 format
        x2: Input tensor of shape [B, K, N] or [B, N, K] in FP8 format
        x1Scale: Scale factors for x1 in MXFP8 format, shape [M, B, K//64, 2] in E8M0
        x2Scale: Scale factors for x2 in MXFP8 format
        permX1: Permutation for x1 input
        permX2: Permutation for x2 input
        permY: Permutation for output
        dtype: Output dtype flag (1=FP16, 27=BF16)
        tile_config: Tile configuration for computation
    """
    x1: torch.Tensor
    x2: torch.Tensor
    x1Scale: torch.Tensor
    x2Scale: torch.Tensor
    permX1: List[int]
    permX2: List[int]
    permY: List[int]
    dtype: int
    tile_config: 'ShapeConfig'


def transpose_quant_batch_matmul(inputs: TransposeQuantBatchMatMulInputs) -> torch.Tensor:
    """
    Generate transpose quantized batch matmul output using PyPTO scaled_mm kernel.

    Wrapper preprocessing flow:
      1. Move tensors to NPU (no permute, keep original layout)
      2. Construct output tensor [M, B, N]
      3. Call kernel

    Args:
        inputs: Input parameters including tensors, scales, permutations and tile config

    Returns:
        torch.Tensor: Output tensor of shape [M, B, N] in target dtype
    """
    x1 = inputs.x1
    x2 = inputs.x2
    x1Scale = inputs.x1Scale
    x2Scale = inputs.x2Scale
    tile_config = inputs.tile_config

    M, B, K = x1.shape
    N = x2.shape[-1] if tile_config.permX2 == [0, 1, 2] else x2.shape[1]

    # Move to NPU without permute
    x1 = x1.npu()
    x1Scale = x1Scale.npu()
    x2 = x2.npu()
    x2Scale = x2Scale.npu()

    torch_out_dtype = torch.float16 if tile_config.out_dtype == pypto.DT_FP16 else torch.bfloat16
    out_batch = torch.zeros(M, B, N, dtype=torch_out_dtype).npu()

    transpose_quant_batch_mat_mul_kernel(
        x1, x2, x1Scale, x2Scale, out_batch,
        tile_config
    )

    return out_batch


import pytest

_TQBMM_TEST_CONFIGS = [
    ShapeConfig(
        ori_shape=[8, 128, 512],
        batch_size=128,
        m_tile_shape=[128, 128],
        k_tile_shape=[128, 128],
        n_tile_shape=[512, 512],
        vector_tile_shape=[1, 512, 1024, 32],
        in_dtype=pypto.DT_FP8E5M2,
        out_dtype=pypto.DT_BF16,
        permX1=[1, 0, 2],
        permX2=[0, 2, 1],
        permY=[1, 0, 2],
        description="test1",
    ),
    ShapeConfig(
        ori_shape=[128, 128, 512],
        batch_size=128,
        m_tile_shape=[128, 128],
        k_tile_shape=[128, 128],
        n_tile_shape=[512, 512],
        vector_tile_shape=[1, 512, 1024, 32],
        in_dtype=pypto.DT_FP8E4M3,
        out_dtype=pypto.DT_FP16,
        permX1=[1, 0, 2],
        permX2=[0, 1, 2],
        permY=[1, 0, 2],
        description="test2",
    ),
    ShapeConfig(
        ori_shape=[8192, 128, 512],
        batch_size=128,
        m_tile_shape=[512, 512],
        k_tile_shape=[128, 128],
        n_tile_shape=[128, 128],
        vector_tile_shape=[1, 512, 1024, 32],
        in_dtype=pypto.DT_FP8E4M3,
        out_dtype=pypto.DT_FP16,
        permX1=[1, 0, 2],
        permX2=[0, 1, 2],
        permY=[1, 0, 2],
        description="test3",
    ),
]


FILTERED_CONFIGS = [cfg for cfg in _TQBMM_TEST_CONFIGS if cfg.description != "test3"]
@pytest.mark.soc("950")
@pytest.mark.parametrize("tile_config", FILTERED_CONFIGS)
def test_transpose_quant_batch_matmul(tile_config):
    """
    Test the transpose quantized batch matrix multiplication.

    This function runs a complete test for a given configuration:
    1. Generate test data with MXFP8 format
    2. Compute golden (reference) output using PyTorch
    3. Compute output using PyPTO
    4. Compare results

    Args:
        tile_config: Configuration parameters for the test case
    """
    M = tile_config.ori_shape[0]
    K = tile_config.ori_shape[1]
    N = tile_config.ori_shape[2]
    B = tile_config.batch_size
    in_dtype = tile_config.in_dtype
    out_dtype_int = 1 if tile_config.out_dtype == pypto.DT_FP16 else 27
    permX1 = tile_config.permX1
    permX2 = tile_config.permX2
    permY = tile_config.permY

    # Map PyPTO data type to torch dtype
    torch_dtype_map = {
        pypto.DT_FP8E4M3: torch.float8_e4m3fn,
        pypto.DT_FP8E5M2: torch.float8_e5m2,
    }
    torch_dtype = torch_dtype_map.get(in_dtype, torch.float8_e4m3fn)

    data_range = 0.05 if tile_config.out_dtype == pypto.DT_BF16 else 1.0

    # Generate input tensor x1 in MXFP8 format
    x1 = torch.randn((M, B, K), dtype=torch.float32).uniform_(0, data_range).to(torch_dtype)
    x1Scale = torch.randn((M, B, K // 64, 2), dtype=torch.float32).uniform_(0.9, 1.1).to(torch.float8_e8m0fnu)

    # Generate input tensor x2 in MXFP8 format — shape depends on permX2
    if permX2 == [0, 2, 1]:
        x2 = torch.randn((B, N, K), dtype=torch.float32).uniform_(0, data_range).to(torch_dtype)
        x2Scale = torch.randn((B, N, K // 64, 2), dtype=torch.float32).uniform_(0, data_range).to(torch.float8_e8m0fnu)
    else:
        x2 = torch.randn((B, K, N), dtype=torch.float32).uniform_(0, data_range).to(torch_dtype)
        x2Scale = torch.randn((B, K // 64, N, 2), dtype=torch.float32).uniform_(0, data_range).to(torch.float8_e8m0fnu)

    # Compute golden (reference) output
    golden = gen_golden(TqbmmGoldenInputs(
        x1=x1, x2=x2, x1Scale=x1Scale, x2Scale=x2Scale,
        permX1=permX1, permX2=permX2, permY=permY, dtype=out_dtype_int
    ))

    # Compute output using PyPTO
    result = transpose_quant_batch_matmul(TransposeQuantBatchMatMulInputs(
        x1=x1, x2=x2, x1Scale=x1Scale, x2Scale=x2Scale,
        permX1=permX1, permX2=permX2, permY=permY,
        dtype=out_dtype_int, tile_config=tile_config
    ))

    # Verify results
    assert_allclose(golden.float().cpu().numpy(), result.float().cpu().numpy(), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    for cfg in _TQBMM_TEST_CONFIGS:
        test_transpose_quant_batch_matmul(cfg)