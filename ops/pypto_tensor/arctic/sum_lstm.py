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
arctic lstm Example for PyPTO

This example demonstrates:
- Special lstm provided by arcitc


lstm is the core mechanism in arcitc lstm-based speculators
"""

import os
import logging
from dataclasses import dataclass

import pypto
from torch._dynamo import allow_in_graph

BATCH_SIZE = 32
D_GATE = 4096
D_GATE_4 = 16384


@dataclass
class LstmConfig:
    """Hyperparameters for LSTM."""
    alpha: float = 0.1
    eps_cell: float = 1e-6
    eps_state: float = 1e-6


@dataclass
class LstmTileConfig:
    """Tiling configuration for NPU optimization."""
    def __init__(self):
        self.tile_bs = 1          # Batch dimension tile size
        self.unroll_list = [1, 2, 4]    # Loop unrolling strategy
        self.h_tile = 4096         # Hidden dimension tile size (aligned to 128 bytes)


def rms_norm_pure(x: pypto.Tensor, epsilon: float) -> pypto.Tensor:
    """
    Pure RMSNorm without learnable parameters.
    Formula: x * rsqrt(mean(x^2) + eps)
    """
    input_dtype = x.dtype
    x_fp32 = pypto.cast(x, pypto.DT_FP32)

    y = pypto.mul(x_fp32, x_fp32)
    y = pypto.mul(y, 1.0 / x.shape[-1])
    y = pypto.sum(y, -1, keepdim=True)

    y = pypto.add(y, epsilon)
    y = pypto.sqrt(y)

    output = pypto.div(x_fp32, y)
    return pypto.cast(output, input_dtype)


def gelu_activation_core(x: pypto.Tensor) -> pypto.Tensor:
    """
    GELU activation function: x * 0.5 * (1 + erf(x / sqrt(2)))

    Approximated as: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

    Parameters
    ----------
    x : pypto.Tensor
        Input tensor

    Returns
    -------
    pypto.Tensor
        GELU activated tensor
    """
    # Use the sigmoid approximation: x * sigmoid(1.702 * x)
    x_scaled = pypto.mul(x, 1.702)
    sigmoid = pypto.sigmoid(x_scaled)
    return pypto.mul(x, sigmoid)


def sum_lstm_compute(
    states_4d: pypto.Tensor,
    z4_4d: pypto.Tensor,
    prev_cell: pypto.Tensor,
    w_cell: pypto.Tensor,
    b_cell: pypto.Tensor,
    w_state: pypto.Tensor,
    b_state: pypto.Tensor,
    config: LstmConfig,
    tile_config: LstmTileConfig,
    h_out: pypto.Tensor,
    c_out: pypto.Tensor,
):
    """Core computation logic for Snowflake Arctic LSTM."""
    # Dimensions
    batch_size = states_4d.shape[0]
    hidden_dim_4 = states_4d.shape[1] # 4 * H
    hidden_dim = prev_cell.shape[1]   # H

    # Pre-broadcast 1D weights to [1, H] for correct vector multiplication
    if w_cell is not None:
        w_cell_b_half = pypto.reshape(w_cell, [1, hidden_dim], inplace=True)
        b_cell_b_half = pypto.reshape(b_cell, [1, hidden_dim], inplace=True)

    if w_state is not None:
        w_state_b_half = pypto.reshape(w_state, [1, hidden_dim], inplace=True)
        b_state_b_half = pypto.reshape(b_state, [1, hidden_dim], inplace=True)


    # Main Loop over Batch Dimension
    for bs_offset, unroll_length in pypto.loop_unroll(
        0, batch_size, 1,
        name="LSTM_BATCH_LOOP",
        idx_name="bs_offset",
        unroll_list=tile_config.unroll_list
    ):
        current_tile_bs = unroll_length
        output_offset = [bs_offset, 0]
        pypto.set_vec_tile_shapes(current_tile_bs, tile_config.h_tile)
        if w_cell is not None:
            w_cell_b = pypto.cast(w_cell_b_half, pypto.DT_FP32)
            b_cell_b = pypto.cast(b_cell_b_half, pypto.DT_FP32)
        if w_state is not None:
            b_state_b = pypto.cast(b_state_b_half, pypto.DT_FP32)
            w_state_b = pypto.cast(w_state_b_half, pypto.DT_FP32)
        # Set vector tile shape for current batch
        pypto.set_vec_tile_shapes(1, hidden_dim_4)

        # === Step 1: Input Fusion (states + alpha * z4) ===
        pypto.set_semantic_label("Input_Fusion")
        states_tile_half = pypto.view(states_4d, [current_tile_bs, hidden_dim_4], [bs_offset, 0])
        z4_tile_half = pypto.view(z4_4d, [current_tile_bs, hidden_dim_4], [bs_offset, 0])
        x_dtype = states_4d.dtype
        states_tile = pypto.cast(states_tile_half, pypto.DT_FP32)
        z4_tile = pypto.cast(z4_tile_half, pypto.DT_FP32)
        z4_scaled = pypto.mul(z4_tile, config.alpha)
        fused = pypto.add(states_tile, z4_scaled)

        # === Step 2: Logical Split ===
        # Reshape [BS, 4H] -> [BS, 4, H] logic handled by stride/view
        pre_f = pypto.view(fused, [current_tile_bs, hidden_dim], [0, 0])
        pre_i = pypto.view(fused, [current_tile_bs, hidden_dim], [0, hidden_dim * 1])
        pre_o = pypto.view(fused, [current_tile_bs, hidden_dim], [0, hidden_dim * 2])
        pre_c = pypto.view(fused, [current_tile_bs, hidden_dim], [0, hidden_dim * 3])

        # === Step 3: Gates ===
        pypto.set_semantic_label("Gate_Sigmoid")
        pypto.set_vec_tile_shapes(1, tile_config.h_tile)
        f_gate = pypto.sigmoid(pre_f)
        i_gate = pypto.sigmoid(pre_i)
        o_gate = pypto.sigmoid(pre_o)

        # === Step 4: Pre-Cell Path ===
        pypto.set_semantic_label("rms_norm_pure")
        c_cand_norm = rms_norm_pure(pre_c, config.eps_cell)

        if w_cell is not None:
            c_cand_norm = pypto.mul(c_cand_norm, w_cell_b)
        if b_cell is not None:
            c_cand_norm = pypto.add(c_cand_norm, b_cell_b)

        pypto.set_semantic_label("gelu_activation_core")
        c_act = gelu_activation_core(c_cand_norm)

        # === Step 5: Cell Update (c_new = prev * f + act * i) ===
        pypto.set_semantic_label("Cell_Update")
        prev_cell_tile_half = pypto.view(prev_cell, [current_tile_bs, hidden_dim], [bs_offset, 0])
        prev_cell_tile = pypto.cast(prev_cell_tile_half, pypto.DT_FP32)
        term1 = pypto.mul(prev_cell_tile, f_gate)
        term2 = pypto.mul(c_act, i_gate)
        c_new_tile = pypto.add(term1, term2)
        c_new_tile_out = pypto.cast(c_new_tile, x_dtype)
        pypto.assemble(c_new_tile_out, output_offset, c_out)

        # === Step 6: Post-Cell Path ===
        pypto.set_semantic_label("Post_Cell_Process")
        h_temp = rms_norm_pure(c_new_tile, config.eps_state)

        if w_state is not None:
            h_temp = pypto.mul(h_temp, w_state_b)
        if b_state is not None:
            h_temp = pypto.add(h_temp, b_state_b)

        pypto.set_semantic_label("gelu_activation_core 2")
        h_act = gelu_activation_core(h_temp)

        # === Step 7: Final Output (h_new = h_act * o_gate) ===
        h_new_tile = pypto.mul(h_act, o_gate)
        h_new_tile_out = pypto.cast(h_new_tile, x_dtype)
        pypto.assemble(h_new_tile_out, output_offset, h_out)


@allow_in_graph
def sum_lstm(run_mode: str = "npu"):

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(
        runtime_options={"device_sched_mode": 1},
    )
    def sum_lstm_kernel(
        states_4d: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP16),
        z4_4d: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP16),
        prev_cell: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP16),
        w_cell: pypto.Tensor([...], pypto.DT_FP16),
        b_cell: pypto.Tensor([...], pypto.DT_FP16),
        w_state: pypto.Tensor([...], pypto.DT_FP16),
        b_state: pypto.Tensor([...], pypto.DT_FP16),
        h_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP16),
        c_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP16),
        config: LstmConfig,
    ):

        tile_cfg = LstmTileConfig()

        sum_lstm_compute(
            states_4d, z4_4d, prev_cell,
            w_cell, b_cell, w_state, b_state,
            config, tile_cfg,
            h_out, c_out
        )
    return sum_lstm_kernel
