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
GLM-4.5 Gate Module for MoE Expert Routing

This module implements the gate operation that projects hidden states from
the model's main dimension (d_model) to the router-specific dimension (d_router).
This projection is used to compute router logits for expert selection in MoE architectures.

Main Functions:
    - gate: Main gate function for expert routing
    - select_experts_mm_kernel: JIT compiled kernel for matrix multiplication
"""
import os
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
import pypto
from .utils.get_format import get_format


def check_args(
    gate_weight: torch.Tensor,
    hidden_states: torch.Tensor
) -> None:
    """
    Validate input arguments for gate operation.

    Args:
        gate_weight: Gate weight matrix
        hidden_states: Input hidden states
    """
    assert gate_weight.dim() == 2
    assert gate_weight.shape[0] == 160
    assert gate_weight.shape[1] == 5120
    assert get_format(gate_weight) == 'ND'
    assert gate_weight.dtype == torch.float32

    assert hidden_states.dim() == 2
    assert hidden_states.shape[1] == 5120
    assert get_format(hidden_states) == 'ND'
    assert hidden_states.dtype == torch.float32



@pypto.frontend.jit(
    runtime_options={}
)
def select_experts_mm_kernel(
    hidden_states: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    mm_weight: pypto.Tensor([], pypto.DT_FP32),
    router_logits_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32)
):
    """
    JIT compiled kernel for gate matrix multiplication.

    This kernel performs the matrix multiplication: router_logits = hidden_states @ weight^T
    to project hidden states from d_model to d_router dimension for expert routing.

    Args:
        hidden_states: Input hidden states [num_tokens, hidden_size]
        mm_weight: Gate weight matrix [num_router_experts, hidden_size]
        router_logits_out: Output router logits [num_tokens, num_router_experts]

    Note:
        This function processes inputs in tiles of size 32 to support dynamic batch sizes.
        The computation uses cube tiling for efficient matrix multiplication on NPU.
    """
    bs = hidden_states.shape[0]
    _ne = mm_weight.shape[0]
    h_num = hidden_states.shape[1]

    view_shape = (32, h_num)

    bs_loop = (bs + view_shape[0] - 1) // view_shape[0]

    for bs_idx in pypto.loop(bs_loop, name="LOOP_MOE_MM_L0", idx_name="bs_idx"):

        tile_hidden_states = pypto.view(hidden_states, view_shape,
                                        [bs_idx * view_shape[0], 0],
                                        valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]),
                                                        h_num])

        pypto.set_cube_tile_shapes([32, 32], [512, 1024], [16, 16])

        res = pypto.matmul(tile_hidden_states, mm_weight, tile_hidden_states.dtype, b_trans=True)

        router_logits_out[bs_idx * view_shape[0]:, 0:] = res




@allow_in_graph
def gate(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    router_logits_out: torch.Tensor
):
    """
    Gate operation for expert routing in MoE architecture.

    This function projects hidden states from the model's main dimension (d_model)
    to the router-specific dimension (d_router) using a learned weight matrix.
    The output router logits are used by the expert selection mechanism to determine
    which experts should process each token.

    Args:
        gate_weight: Gate weight matrix [num_router_experts, hidden_size]
        hidden_states: Input hidden states [num_tokens, hidden_size]
        router_logits_out: Output router logits [num_tokens, num_router_experts]

    Returns:
        router_logits_out: Router logits tensor [num_tokens, num_router_experts]

    Note:
        This function is decorated with @allow_in_graph to enable integration
        with PyTorch's compilation graph.
    """
    if isinstance(hidden_states, FakeTensor):
        return router_logits_out
    check_args(gate_weight, hidden_states)

    inputs = [hidden_states, gate_weight, router_logits_out]
    select_experts_mm_kernel(*inputs)
