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
Transpose Quantized Batch Matrix Multiplication with MXFP8 Quantization

This module implements batch matrix multiplication with transpose and MXFP8 quantization using PyPTO.
Supports multiple perm combinations (permX1, permX2, permY) and output dtype (FP16/BF16).

M-axis splitting approach (non-parallel loops):
  - Outer LOOP_M: splits M into m_chunk_size tiles, iterates without parallel=True
  - Inner LOOP_B: iterates over B batches without parallel=True
  - Reshape x1 to [M, B*K], x1Scale to [M, B*K//64, 2] — same as original
  - Each iteration slices row range [m_begin:m_end] + column range [begin:end] from reshaped 2D
  - Assemble mm_result to local_out at [m_begin, out_pos]
  - Final reshape local_out [M, B*N] → out [M, B, N]

This avoids parallel loop unroll (LoopUnroll + ExpandFunction), keeping the IR graph small
for fast compilation. Loop iterations are handled at runtime, not unrolled into the IR.
"""

from dataclasses import dataclass
from typing import List
import pypto


@dataclass
class ShapeConfig:
    """
    Configuration parameters for transpose quantized batch matrix multiplication.

    Attributes:
        ori_shape: Original shape [M, K, N] — M is dynamic in kernel, only used for golden/wrapper
        batch_size: B-axis size (compile-time fixed)
        m_tile_shape: Tile shape for M dimension in cube operation
        k_tile_shape: Tile shape for K dimension in cube operation
        n_tile_shape: Tile shape for N dimension in cube operation
        vector_tile_shape: Tile shapes for vector operations
        num_k_groups: Number of groups to split K-axis (default: 1)
        num_n_groups: Number of groups to split N-axis (default: 1)
        in_dtype: Input data type (default: DT_FP8E4M3)
        out_dtype: Output data type (default: DT_BF16)
        permX1: Permutation for x1 input (default: [1, 0, 2] = batch first)
        permX2: Permutation for x2 input (default: [0, 1, 2] = K,N layout)
        permY: Permutation for output (default: [1, 0, 2] = M,B,N layout)
        description: Description of the test case
    """
    ori_shape: list
    batch_size: int
    m_tile_shape: list
    k_tile_shape: list
    n_tile_shape: list
    vector_tile_shape: list
    num_k_groups: int = 1
    num_n_groups: int = 1
    in_dtype: pypto.DataType = pypto.DT_FP8E4M3
    out_dtype: pypto.DataType = pypto.DT_BF16
    permX1: List[int] = None
    permX2: List[int] = None
    permY: List[int] = None
    description: str = ""

    def __post_init__(self):
        if self.permX1 is None:
            self.permX1 = [1, 0, 2]
        if self.permX2 is None:
            self.permX2 = [0, 1, 2]
        if self.permY is None:
            self.permY = [1, 0, 2]


@pypto.frontend.jit(
    pass_options={
        "cube_l1_reuse_setting": {-1: 2},
        "cube_nbuffer_setting": {-1: 4},
        "vec_nbuffer_setting": {-2: 1, -1: 2},
    },
    runtime_options={"stitch_function_max_num": 1024, "device_sched_mode": 1},
)
def transpose_quant_batch_mat_mul_kernel(
    x1: pypto.Tensor(),
    x2: pypto.Tensor(),
    x1Scale: pypto.Tensor(),
    x2Scale: pypto.Tensor(),
    out: pypto.Tensor(),
    tile_config: ShapeConfig
):
    """
    Transpose quantized batch matrix multiplication kernel using MXFP8 quantization.

    M-axis splitting approach — non-parallel nested loops avoid IR graph explosion:
      Outer LOOP_M (non-parallel): splits M into m_chunk_size tiles
      Inner LOOP_B (non-parallel): iterates over B batches
      Reshape [M, B*K] and assemble pattern preserved from original — only row slicing changed.

    Args:
        x1: Input tensor of shape [M, B, K] in FP8 format
        x2: Input tensor of shape [B, K, N] or [B, N, K] in FP8 format
        x1Scale: Scale factors for x1 in MXFP8 format, shape [M, B, K//64, 2] in E8M0
        x2Scale: Scale factors for x2 in MXFP8 format, shape varies by permX2
        out: Output tensor of shape [M, B, N]
        tile_config: Tile configuration including batch_size, shapes and tile parameters

    Note:
        - M-axis splitting with non-parallel loop: avoids parallel loop unroll expansion
        - B-axis non-parallel loop: no LoopUnroll, IR stays small
        - permX2 determines whether x2 uses scaled_mm with b_trans=True
        - Reshape x1→[M, B*K] and assemble→local_out pattern unchanged from original
    """
    pypto.experimental.auto_mix_partition(1)

    M = tile_config.ori_shape[0]
    K = tile_config.ori_shape[1]
    N = tile_config.ori_shape[2]
    B = tile_config.batch_size
    out_dtype = tile_config.out_dtype
    permX2 = tile_config.permX2

    pypto.set_cube_tile_shapes(
        tile_config.m_tile_shape, tile_config.k_tile_shape, tile_config.n_tile_shape
    )
    pypto.set_vec_tile_shapes(
        tile_config.vector_tile_shape[0], tile_config.vector_tile_shape[1],
        tile_config.vector_tile_shape[2], tile_config.vector_tile_shape[3]
    )

    x1_reshape = pypto.reshape(x1, [M, B * K], inplace=True)
    x1_scale_reshape = pypto.reshape(x1Scale, [M, B * K // 64, 2], inplace=True)

    local_out = pypto.Tensor(shape=(M, B * N), dtype=out_dtype)
    for b_idx in range(B):
        begin = b_idx * K
        end = (b_idx + 1) * K

        x1_slice = x1_reshape[:, begin:end]
        x1_scale_slice = x1_scale_reshape[:, begin // 64:end // 64, :]

        x2_slice = x2[b_idx, :, :]
        x2_scale_slice = x2Scale[b_idx, :, :, :]

        if permX2 == [0, 1, 2]:
            mm_result = pypto.scaled_mm(
                x1_slice, x2_slice, out_dtype,
                x1_scale_slice, x2_scale_slice
            )
        else:
            mm_result = pypto.scaled_mm(
                x1_slice, x2_slice, out_dtype,
                x1_scale_slice, x2_scale_slice,
                b_trans=True, scale_b_trans=True
            )

        out_pos = b_idx * N
        pypto.assemble(mm_result, [0, out_pos], local_out)

    # Reshape: local_out [M, B*N] → out [M, B, N]
    out[:, :, :] = pypto.reshape(local_out, [M, B, N])
