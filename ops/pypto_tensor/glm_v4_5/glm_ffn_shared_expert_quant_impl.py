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
GLM-4.5 FFN Shared Expert Quantization Module

This module implements the quantized FFN computation for shared experts in MoE architecture.
Shared experts are used across all tokens and tasks, learning general feature representations
while reducing the total parameter count through weight sharing.

Main Functions:
    - ffn_shared_expert_quant: Main function for shared expert FFN quantization
    - share_expert_moe_main: JIT compiled kernel for shared expert computation
    - expert_infer_base: Base inference function for shared expert computation
"""
import os
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
import pypto
from .glm_ffn_common_interface import symmetric_quantization_per_token, dequant_dynamic, swiglu
from .utils.get_format import get_format


def check_args(
    hidden_states,
    w13,
    w13_scale,
    w2,
    w2_scale
):
    assert hidden_states.dim() == 2
    assert hidden_states.shape[1] == 5120
    assert get_format(hidden_states) == 'ND'
    assert hidden_states.dtype == torch.bfloat16

    assert w13.dim() == 2
    assert w13.shape[0] == 5120
    assert w13.shape[1] == 384
    assert get_format(w13) == 'NZ'
    assert w13.dtype == torch.int8

    assert w13_scale.dim() == 1
    assert w13_scale.shape[0] == 384
    assert get_format(w13_scale) == 'ND'
    assert w13_scale.dtype == torch.bfloat16

    assert w2.dim() == 2
    assert w2.shape[0] == 192
    assert w2.shape[1] == 5120
    assert get_format(w2) == 'NZ'
    assert w2.dtype == torch.int8

    assert w2_scale.dim() == 1
    assert w2_scale.shape[0] == 5120
    assert get_format(w2_scale) == 'ND'
    assert w2_scale.dtype == torch.bfloat16


def expert_infer_base(hidden_states, w13_params, w2_params, ffn_res, tiling_params, offset_params):
    """
    Base inference function for shared expert computation.

    This function performs FFN computation for shared expert:
    1. Per-token quantization: hidden_states_quant = Quantize(hidden_states)
    2. Quantized matrix multiplication: up_proj = MatMul(hidden_states_quant, w13)
    3. Dequantization: up_proj_dequant = Dequantize(up_proj, w13_scale, hidden_states_scale)
    4. SwiGLU activation: swiglu_out = SwiGLU(up_proj_dequant)
    5. Per-token quantization: down_proj_quant = Quantize(swiglu_out)
    6. Quantized matrix multiplication: down_proj = MatMul(down_proj_quant, w2)
    7. Dequantization: output = Dequantize(down_proj, w2_scale, down_proj_scale)

    Args:
        hidden_states: Input hidden states [num_tokens, hidden_size]
        w13_params: Tuple of (w13, w13_scale)
        w2_params: Tuple of (w2, w2_scale)
        ffn_res: Output tensor [num_tokens, hidden_size]
        tiling_params: Tuple of (vec_tile_shape, mm1_cube_tile_shape, mm2_cube_tile_shape)
        offset_params: Tuple of (share_loop_idx, loop_base)

    Note:
        This function processes tokens in tiles of size loop_base (typically 8)
        to support efficient computation on NPU.
    """
    w13, w13_scale = w13_params
    w2, w2_scale = w2_params
    unroll_offset, unroll_level = offset_params
    vec_tile_shape, mm1_cube_tile_shape, mm2_cube_tile_shape = tiling_params

    hidden_size = hidden_states.shape[1]
    intermediate_size = w2.shape[0]
    x_dtype = hidden_states.dtype

    pypto.set_vec_tile_shapes(vec_tile_shape[0], vec_tile_shape[1])
    hidden_states_offset = [unroll_offset, 0]
    hidden_states_actual = pypto.view(hidden_states, [unroll_level, hidden_size], hidden_states_offset)

    hidden_states_quant, hidden_states_scale = symmetric_quantization_per_token(hidden_states_actual)

    pypto.set_cube_tile_shapes([unroll_level, unroll_level],
                               [mm1_cube_tile_shape[1], mm1_cube_tile_shape[1] * 2],
                               [mm1_cube_tile_shape[2], mm1_cube_tile_shape[2]], True)
    up_proj = pypto.matmul(hidden_states_quant, w13, pypto.DT_INT32)

    w13_scale_2d = pypto.unsqueeze(w13_scale, 0)
    pypto.set_vec_tile_shapes(4, intermediate_size * 2)
    up_proj_dequant = dequant_dynamic(up_proj, w13_scale_2d, hidden_states_scale)
    swiglu_out = swiglu(up_proj_dequant)

    down_proj_quant, down_proj_scale = symmetric_quantization_per_token(swiglu_out)

    pypto.set_cube_tile_shapes([unroll_level, unroll_level],
                               [mm2_cube_tile_shape[1], mm2_cube_tile_shape[1] * 2],
                               [mm2_cube_tile_shape[2], mm2_cube_tile_shape[2]], False)
    down_proj = pypto.matmul(down_proj_quant, w2, pypto.DT_INT32)

    w2_scale_2d = pypto.unsqueeze(w2_scale, 0)
    pypto.set_vec_tile_shapes(4, hidden_size)
    down_proj_dequant = dequant_dynamic(down_proj, w2_scale_2d, down_proj_scale)
    out = pypto.cast(down_proj_dequant, x_dtype)
    pypto.assemble(out, hidden_states_offset, ffn_res)


@pypto.frontend.jit(
    runtime_options={"device_sched_mode": 1},
)
def share_expert_moe_main(
    hidden_states: pypto.tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    w13: pypto.tensor(),
    w13_scale: pypto.tensor(),
    w2: pypto.tensor(),
    w2_scale: pypto.tensor(),
    ffn_res: pypto.tensor([pypto.DYNAMIC, ...], pypto.DT_BF16)
):

    vec_tile_shape = (4, 5120)
    mm1_cube_tile_shape = (8, 256, 256)
    mm2_cube_tile_shape = (8, 192, 256)

    token_nums = hidden_states.shape[0]
    for share_loop_idx, loop_base in pypto.loop_unroll(
        token_nums,
        unroll_list=[1, 2, 4, 8, 16, 32, 64, 128],
        name="share_loop_idx"):
        expert_infer_base(
            hidden_states=hidden_states,
            w13_params=[w13, w13_scale],
            w2_params=[w2, w2_scale],
            ffn_res=ffn_res,
            tiling_params=[vec_tile_shape, mm1_cube_tile_shape, mm2_cube_tile_shape],
            offset_params=[share_loop_idx, loop_base]
    )


@allow_in_graph
def ffn_shared_expert_quant(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    ffn_res: torch.Tensor
) -> None:
    """
    Quantized FFN computation for shared experts in MoE architecture.

    This function computes FFN output using quantized operations for shared experts.
    Shared experts are used across all tokens and tasks, learning general feature
    representations while reducing the total parameter count through weight sharing.

    Args:
        hidden_states: Input hidden states [num_tokens, hidden_size]
        w13: Gate and up projection weights (int8) [hidden_size, intermediate_size * 2]
        w13_scale: w13 weight scales [intermediate_size * 2]
        w2: Down projection weights (int8) [intermediate_size, hidden_size]
        w2_scale: w2 weight scales [hidden_size]
        ffn_res: Output tensor [num_tokens, hidden_size]

    Note:
        This function is decorated with @allow_in_graph to enable integration
        with PyTorch's compilation graph. The computation uses per-token quantization
        for better accuracy compared to per-channel quantization.
    """
    if not isinstance(hidden_states, FakeTensor):
        check_args(hidden_states, w13, w13_scale, w2, w2_scale)

    inputs = [hidden_states, w13, w13_scale, w2, w2_scale, ffn_res]
    share_expert_moe_main(*inputs)
