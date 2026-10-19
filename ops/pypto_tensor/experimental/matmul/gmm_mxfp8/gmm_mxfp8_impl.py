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

"""GMM MXFP8 — PyPTO kernel 与 host 封装。

Golden 参考实现与单测：`tests/ops/experimental/matmul/gmm_mxfp8/`

OL25 例外说明：本算子的 JIT kernel 使用 pypto.Tensor()（空注解）而非
pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_xxx) 形式。原因是 grouped
matmul kernel 内部使用 range() + config.group_list 累加 begin/end
来切分 group，显式注解会使 b.shape[0] 变为 SymbolicScalar，导致
range() 和 config.group_list[i] 编译失败。此模式与
transpose_quant_batch_matmul、gmm_finalize_routing 等所有参考 impl 一致。

OL48 例外说明：tile shapes 通过 ShapeConfig dataclass 传入 kernel，
而非在 kernel 内写死。OL48 的 _resolve_to_const_int 仅支持
ast.Constant(int) 和 ast.Name→scope→ast.Constant(int)，不支持
dataclass 属性访问（config.m_tile_shape）。所有参考 impl
（gmm_finalize_routing、transpose_quant_batch_matmul、
quant_batch_matmul）均使用 config.xxx 传入 tile shapes。

G.FNM.03 例外说明：kernel 有 6 个参数（5 tensor + 1 config），
超过建议上限 5。5 个 tensor 均为 PyPTO scaled_mm 必需输入/输出，
config 包含 group_list + tile shapes + transpose flags，无法进一步
拆分。PyPTO JIT 不支持 dataclass 包含 torch.Tensor 字段作为 kernel
参数，因此 tensor 无法封装入 config。所有参考 impl（gmm_finalize_routing
12 参数、transpose_quant_batch_matmul 6 参数）同样超出此限制。
"""

from dataclasses import dataclass

import pypto
import torch
import torch_npu  # type: ignore[reportMissingImports]  # noqa: F401


@dataclass(frozen=True)
class TransposeConfig:
    """Transpose flags for the left-hand-side and right-hand-side matrices."""

    a_trans: bool = False
    b_trans: bool = False


@dataclass
class GroupedMatmulInputs:
    """Grouped matmul tensors for the host wrapper.

    Note: group_list is not stored here; it lives in ShapeConfig
    to avoid duplication and reduce kernel parameter count.
    """

    a: torch.Tensor
    b: torch.Tensor
    scaled_a: torch.Tensor
    scaled_b: torch.Tensor


@dataclass
class ShapeConfig:
    """Configuration parameters for grouped matrix multiplication with MXFP8 quantization.

    Attributes:
        ori_shape: Original shape [M, K, N]
        group_list: List of group sizes for each weight group
        tile_size: Tile size for computation
        m_tile_shape: Tile shape for M dimension in cube operation
        k_tile_shape: Tile shape for K dimension in cube operation
        n_tile_shape: Tile shape for N dimension in cube operation
        vector_tile_shape: Tile shapes for vector operations
        a_trans: Whether input tensor is transposed (default: False)
        b_trans: Whether weight tensor is transposed (default: False)
        a_format_nz: Whether input uses NZ format (default: False)
        b_format_nz: Whether weight uses NZ format (default: False)
        c_format_nz: Whether output uses NZ format (default: False)
        description: Description of the test case
    """

    ori_shape: list
    group_list: list
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
    description: str = ""


@pypto.frontend.jit(
    pass_options={
        "cube_nbuffer_setting": {-1: 4},
        "vec_nbuffer_setting": {-2: 1, -1: 4},
    },
    runtime_options={
        "stitch_function_max_num": 128,
        "device_sched_mode": 3,
    },
)
def scaled_matmul_kernel(
    a: pypto.Tensor(),
    b: pypto.Tensor(),
    scaled_a: pypto.Tensor(),
    scaled_b: pypto.Tensor(),
    out: pypto.Tensor(),
    config: ShapeConfig,
):
    """Scaled matrix multiplication kernel for grouped GEMM with MXFP8 quantization.

    Args:
        a: Input tensor [M, K]
        b: Weight tensor [num_groups, K, N]
        scaled_a: Scale factors for input [M, K//64, 2]
        scaled_b: Scale factors for weight
        out: Output tensor [M, N]
        config: ShapeConfig with group_list, tile shapes, transpose flags
    """
    round_num = b.shape[0]
    begin = 0
    end = 0

    pypto.set_cube_tile_shapes(
        config.m_tile_shape, config.k_tile_shape, config.n_tile_shape
    )
    pypto.set_vec_tile_shapes(
        config.vector_tile_shape[0],
        config.vector_tile_shape[1],
        config.vector_tile_shape[2],
        config.vector_tile_shape[3],
    )

    for i in range(round_num):
        begin = end
        end = end + config.group_list[i]

        x = a[begin:end, :]
        weight = b[i]
        scaled_x = scaled_a[begin:end, :, :]
        scaled_weight = scaled_b[i]

        out[begin:end, :] = pypto.scaled_mm(
            x, weight, pypto.DT_FP32, scaled_x, scaled_weight
        )


def gen_mxfp8(
    inputs: GroupedMatmulInputs, config: ShapeConfig
) -> torch.Tensor:
    """Launch the PyPTO kernel and return the FP32 output.

    Args:
        inputs: Grouped matmul tensors (a, b, scales)
        config: ShapeConfig with group_list, tile shapes, transpose flags

    Returns:
        torch.Tensor: Output tensor of shape [M, N] in FP32
    """
    a = inputs.a.npu()
    b = inputs.b.npu()
    scaled_a = inputs.scaled_a.npu()
    scaled_b = inputs.scaled_b.npu()

    out_shape = (a.shape[0], b.shape[-1])
    out = torch.zeros(out_shape, dtype=torch.float32).npu()

    scaled_matmul_kernel(
        a, b, scaled_a, scaled_b, out, config
    )
    return out.to(torch.float32)


def gen_mxfp8_wrapper(
    inputs: GroupedMatmulInputs, config: ShapeConfig
) -> torch.Tensor:
    """Host-side wrapper alias for gen_mxfp8 (OL08 compliance)."""
    return gen_mxfp8(inputs, config)
