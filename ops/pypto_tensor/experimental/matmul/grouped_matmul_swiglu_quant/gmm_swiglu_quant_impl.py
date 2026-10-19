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

"""GMM MXFP8 SwiGLU quantization — PyPTO kernel 与 host 封装。

Golden 与单测：`tests/ops/experimental/matmul/grouped_matmul_swiglu_quant/`
"""

from dataclasses import dataclass

import pypto
import torch
import torch_npu  # type: ignore[reportMissingImports]


@dataclass(frozen=True)
class TransposeConfig:
    """Transpose flags for the left-hand-side and right-hand-side matrices."""

    a_trans: bool = False
    b_trans: bool = False


@dataclass(frozen=True)
class GroupedMatmulInputs:
    """Grouped matmul tensors shared by both the reference path and the kernel path."""

    a: torch.Tensor
    b: torch.Tensor
    scaled_a: torch.Tensor
    scaled_b: torch.Tensor
    group_list: list[int]


@dataclass
class ShapeConfig:
    """Configuration for matmul tile shapes and layout flags."""

    ori_shape: list
    tile_size: int
    m_tile_shape: list
    k_tile_shape: list
    n_tile_shape: list
    vector_tile_shape: list
    a_trans: bool = False
    b_trans: bool = False
    a_format_nz: bool = False
    b_format_nz: bool = False
    c_format_nz: bool = False


@pypto.frontend.jit(
    runtime_options={"device_sched_mode": 3},
)
def scaled_matmul_kernel(
    a: pypto.Tensor(),
    b: pypto.Tensor(),
    scaled_a: pypto.Tensor(),
    scaled_b: pypto.Tensor(),
    out: pypto.Tensor(),
    out_quant: pypto.Tensor(),
    group_list,
    tile_config,
):
    """Run grouped scaled matmul, then apply SwiGLU and per-token quantization."""
    num_groups = b.shape[0]
    n_size = b.shape[-1]
    begin = 0
    end = 0

    for i in range(num_groups):
        begin = end
        end = end + group_list[i]

        x = a[begin:end, :]
        weight = b[i]
        scaled_x = scaled_a[begin:end, :, :]
        pypto.set_vec_tile_shapes(
            tile_config.vector_tile_shape[0],
            tile_config.vector_tile_shape[1],
            tile_config.vector_tile_shape[2],
            tile_config.vector_tile_shape[3],
        )
        scaled_weight = scaled_b[i]

        pypto.set_cube_tile_shapes(
            tile_config.m_tile_shape,
            tile_config.k_tile_shape,
            tile_config.n_tile_shape,
            enable_split_k=True,
        )
        current_mm_out = pypto.scaled_mm(x, weight, pypto.DT_FP32, scaled_x, scaled_weight)
        # Split the matmul output into the SwiGLU value and gate tensors.
        value = current_mm_out[:, : n_size // 2]
        gate = current_mm_out[:, n_size // 2:]
        pypto.set_vec_tile_shapes(64, 256)
        silu_value = value * pypto.sigmoid(value)
        swiglu_out = pypto.mul(silu_value, gate)

        # Quantize each token with its own scale.
        x_bf16 = pypto.cast(swiglu_out, pypto.DT_BF16, pypto.CastMode.CAST_RINT)
        x_fp32 = pypto.cast(x_bf16, pypto.DT_FP32)
        x_abs = pypto.abs(x_fp32)
        x_max = pypto.amax(x_abs, -1, True)
        shape_0, shape_1 = x_max.shape[:2]
        x_scale = pypto.div(pypto.full([shape_0, shape_1], 127.0, pypto.DT_FP32), x_max)
        x_mul = pypto.mul(x_fp32, x_scale)

        x_mul_round = pypto.round(x_mul)
        x_fp16 = pypto.cast(x_mul_round, pypto.DT_FP16, pypto.CastMode.CAST_RINT)
        x_int8 = pypto.cast(x_fp16, pypto.DT_INT8)
        x_scale_quant = pypto.div(pypto.full([shape_0, shape_1], 1.0, pypto.DT_FP32), x_scale)

        pypto.assemble(x_int8, [begin, 0], out)
        pypto.assemble(x_scale_quant, [begin, 0], out_quant)


def gen_mxfp8(inputs: GroupedMatmulInputs, tile_config: ShapeConfig):
    """Launch the PyPTO kernel and return the quantized output plus scales."""
    a = inputs.a.npu()
    b = inputs.b.npu()
    scaled_a = inputs.scaled_a.npu()
    scaled_b = inputs.scaled_b.npu()

    out_shape = (a.shape[0], b.shape[-1] // 2)
    out = torch.zeros(out_shape, dtype=torch.int8).npu()

    out_quant_shape = (a.shape[0], 1)
    out_quant = torch.zeros(out_quant_shape, dtype=torch.float32).npu()

    scaled_matmul_kernel(a, b, scaled_a, scaled_b, out, out_quant, inputs.group_list, tile_config)
    out = out.to(torch.float32)
    out_quant = out_quant.squeeze(dim=1)

    return out, out_quant
