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
Grouped Matrix Multiplication with Inplace Add using MXFP8 Quantization

This module implements grouped matrix multiplication with inplace addition and MXFP8 quantization using PyPTO.
Supports K-axis grouped GEMM operations with different weight groups and MXFP8 quantization format,
where results are accumulated inplace to the output tensor.
"""

from dataclasses import dataclass
import pypto


@dataclass
class ShapeConfig:
    """
    Configuration parameters for quantized grouped matrix multiplication with inplace add.

    Attributes:
        ori_shape: Original shape [M, K, N]
        num_groups: Number of groups to split K-axis uniformly
        m_tile_shape: Tile shape for M dimension in cube operation
        k_tile_shape: Tile shape for K dimension in cube operation
        n_tile_shape: Tile shape for N dimension in cube operation
        vector_tile_shape: Tile shapes for vector operations
        in_dtype: Input data type (default: DT_FP8E4M3)
        a_trans: Whether input tensor is transposed (default: True)
        b_trans: Whether weight tensor is transposed (default: False)
        a_format_nz: Whether input uses NZ format (default: False)
        b_format_nz: Whether weight uses NZ format (default: False)
        c_format_nz: Whether output uses NZ format (default: False)
        description: Description of the test case
    """
    ori_shape: list
    num_groups: int
    m_tile_shape: list
    k_tile_shape: list
    n_tile_shape: list
    vector_tile_shape: list
    in_dtype: pypto.DataType = pypto.DT_FP8E4M3
    a_trans: bool = True
    b_trans: bool = False
    a_format_nz: bool = False
    b_format_nz: bool = False
    c_format_nz: bool = False
    description: str = ""


@pypto.frontend.jit(
    pass_options={
        "cube_nbuffer_setting": {-1: 4},
        "vec_nbuffer_setting": {-2: 1, -1: 4}
    },
    runtime_options={
        "stitch_function_max_num": 8
    }
)
def scaled_matmul_kernel(
    a: pypto.Tensor(),
    b: pypto.Tensor(),
    scaled_a: pypto.Tensor(),
    scaled_b: pypto.Tensor(),
    y: pypto.Tensor(),
    tile_config: ShapeConfig
):
    """
    Scaled matrix multiplication kernel for grouped GEMM with inplace add and MXFP8 quantization.

    This kernel performs K-axis grouped matrix multiplication with MXFP8 quantization,
    where each group uses a different K-axis block, and results are accumulated inplace
    to the output tensor.

    Args:
        a: Input tensor of shape [K, M] or [M, K]
        b: Weight tensor of shape [K, N] or [N, K]
        scaled_a: Scale factors for input tensor in MXFP8 format
        scaled_b: Scale factors for weight tensor in MXFP8 format
        y: Output tensor of shape [num_groups, M, N] (serves as initial value and result)
        tile_config: Tile configuration including num_groups, shapes and tile parameters

    Note:
        - K-axis must be 64-aligned for MX quantization
        - K must be divisible by num_groups for uniform splitting
        - Inner axis (M or N) must be 32-byte aligned
        - Performance optimized with parallel loop + batch add
    """
    num_groups = tile_config.num_groups
    m = tile_config.ori_shape[0]
    n = tile_config.ori_shape[2]
    k = tile_config.ori_shape[1]
    k_block = k // num_groups

    a_trans = tile_config.a_trans
    b_trans = tile_config.b_trans

    # Set tile shapes before loop execution
    pypto.set_cube_tile_shapes(
        tile_config.m_tile_shape,
        tile_config.k_tile_shape,
        tile_config.n_tile_shape
    )
    pypto.set_vec_tile_shapes(
        tile_config.vector_tile_shape[0],
        tile_config.vector_tile_shape[1],
        tile_config.vector_tile_shape[2],
        tile_config.vector_tile_shape[3]
    )

    # Create intermediate tensor to collect results from all groups
    mm_result_tensor = pypto.tensor([num_groups, m, n], pypto.DT_FP32)

    # Parallel loop: each group computes scaled_mm independently
    for i in pypto.loop(num_groups, parallel=True):
        begin = i * k_block
        end = (i + 1) * k_block
        scale_offset = begin // 64 + i
        scale_length = k_block // 64

        # Extract input tensor for current group
        if a_trans:
            x = a[begin:end, :]
            scaled_x = scaled_a[scale_offset:scale_offset + scale_length, :, :]
        else:
            x = a[:, begin:end]
            scaled_x = scaled_a[:, scale_offset:scale_offset + scale_length, :]

        # Extract weight tensor for current group
        if b_trans:
            weight = b[:, begin:end]
            scaled_weight = scaled_b[scale_offset:scale_offset + scale_length, :]
        else:
            weight = b[begin:end, :]
            scaled_weight = scaled_b[scale_offset:scale_offset + scale_length, :, :]

        # Compute scaled matrix multiplication for current group
        scale_a_trans = a_trans
        mm_result_tensor[i] = pypto.scaled_mm(
            x,
            weight,
            pypto.DT_FP32,
            scaled_x,
            scaled_weight,
            a_trans=a_trans,
            scale_a_trans=scale_a_trans,
            b_trans=b_trans
        )

    # Batch accumulation: perform inplace add after loop (optimal performance)
    y[:, :, :] = pypto.add(y, mm_result_tensor)
