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

"""quant_matmul_reduce_sum PyPTO 算子实现。

Quantized matrix multiplication followed by scale dequantization and batch reduce sum.
Golden 参考实现与单测入口见：
  tests/ops/experimental/matmul/quant_matmul_reduce_sum/

本模块提供 JIT kernel、配置数据结构与 host 侧 wrapper。
"""

from dataclasses import dataclass

import pypto
import torch
import torch_npu  # type: ignore[reportMissingImports]


@dataclass
class QuantMatmulReduceSumConfig:
    """quant_matmul_reduce_sum 配置。

    Attributes:
        ori: Original shape [batch, m, k, n]
        m_tile_shape: Tile shape for M dimension in cube operation
        k_tile_shape: Tile shape for K dimension in cube operation
        n_tile_shape: Tile shape for N dimension in cube operation
        in_dtype: Input data type
        out_dtype: Output data type
        x2_format_nz: Whether x2 uses NZ format (default: True)
        vec_tile_shapes: Tile shapes for vector operations (default: [1, 256, 256])
        description: Description of the test case
    """
    ori: list
    m_tile_shape: list
    k_tile_shape: list
    n_tile_shape: list
    in_dtype: pypto.DataType
    out_dtype: pypto.DataType
    x2_format_nz: bool = True
    vec_tile_shapes: list = None
    description: str = ""

    def __post_init__(self):
        if self.vec_tile_shapes is None:
            self.vec_tile_shapes = [1, 256, 256]


def quant_matmul_reduce_sum_pypto(config: QuantMatmulReduceSumConfig):
    """Create quantized matmul reduce sum kernel using PyPTO.

    Args:
        config: Configuration parameters for the operation

    Returns:
        function: JIT-compiled kernel function
    """
    batch, m, k, n = config.ori
    x1_shape = [batch, m, k]
    x2_shape = [batch, k, n]
    x1_scale_shape = [batch, m]
    x2_scale_shape = [n]
    out_shape = [m, n]
    x2_format = pypto.TileOpFormat.TILEOP_NZ if config.x2_format_nz else pypto.TileOpFormat.TILEOP_ND

    @pypto.frontend.jit()
    def quant_matmul_reduce_sum_kernel(
        x1: pypto.Tensor(x1_shape, pypto.DT_INT8),
        x2: pypto.Tensor(x2_shape, pypto.DT_INT8, format=x2_format),
        x1_scale: pypto.Tensor(x1_scale_shape, pypto.DT_FP32),
        x2_scale: pypto.Tensor(x2_scale_shape, pypto.DT_BF16),
        out: pypto.Tensor(out_shape, pypto.DT_BF16),
    ):
        pypto.set_cube_tile_shapes(config.m_tile_shape, config.k_tile_shape, config.n_tile_shape)
        pypto.set_vec_tile_shapes(*config.vec_tile_shapes)

        if config.x2_format_nz:
            pypto.set_matrix_size([m, k, n])

        # Matmul computation and convert to FP32
        matmul_result = pypto.matmul(x1, x2, pypto.DT_INT32)
        matmul_result_fp32 = pypto.cast(matmul_result, pypto.DT_FP32)

        # Convert x2_scale to FP32 and broadcast
        x2_scale_fp32 = pypto.cast(x2_scale, pypto.DT_FP32)
        x2_scale_2d = pypto.unsqueeze(x2_scale_fp32, 0)
        x2_scale_broadcast = pypto.expand_clone(x2_scale_2d, [m, n])

        # Broadcast x1_scale
        x1_scale_2d = pypto.unsqueeze(x1_scale, 2)
        x1_scale_broadcast = pypto.expand_clone(x1_scale_2d, [batch, m, n])

        # Compute scale multiplication
        scale_mul = pypto.mul(x1_scale_broadcast, x2_scale_broadcast)

        # Fused multiply and reduce sum
        scaled = pypto.mul(matmul_result_fp32, scale_mul)
        out_fp32 = pypto.sum(scaled, 0)
        out_bf16 = pypto.cast(out_fp32, pypto.DT_BF16)

        out.move(out_bf16)

    return quant_matmul_reduce_sum_kernel


def quant_matmul_reduce_sum_wrapper(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x1_scale: torch.Tensor,
    x2_scale: torch.Tensor,
    config: QuantMatmulReduceSumConfig,
) -> torch.Tensor:
    """Host-side wrapper: allocate output, launch kernel, return result.

    Args:
        x1: INT8 input tensor of shape [batch, m, k]
        x2: INT8 input tensor of shape [batch, k, n] (ND or NZ format)
        x1_scale: FP32 scale tensor of shape [batch, m]
        x2_scale: BF16 scale tensor of shape [n]
        config: Operation configuration

    Returns:
        torch.Tensor: BF16 output tensor of shape [m, n]
    """
    _, m, _, n = config.ori
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x1.device)
    kernel = quant_matmul_reduce_sum_pypto(config)
    kernel(x1, x2, x1_scale, x2_scale, out)
    return out
