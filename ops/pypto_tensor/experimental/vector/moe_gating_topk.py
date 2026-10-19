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
MOE Gating TopK Operator Implementation

This module implements the MOE Gating TopK operator for MoE (Mixture of Experts) architecture.
It selects top-k experts for each input sample with optional group-based expert selection.

Formula:
    norm_out = sigmoid(x) or softmax(x)
    if group_count > 1:
        group_scores = max(norm_out) or sum(topk2(norm_out))
        selected_groups = topk(group_scores, k_group)
        mask = create_mask(selected_groups)
        norm_out = norm_out * mask

    topk_experts, topk_indices = topk(norm_out, k)
    weights = gather(original_norm_out, topk_indices)
    weights = weights / (sum(weights) + eps) * routed_scaling_factor

Output:
    y: Selected and normalized weights [batch_size, k]
    indices: Selected expert indices [batch_size, k]
    y2: Original normalized values [batch_size, num_experts]
"""
import logging
import os
from dataclasses import dataclass

import numpy as np
import pypto
import torch
import torch_npu
from numpy.testing import assert_allclose


@dataclass
class MoEGatingTopKConfig:
    k: int
    k_group: int
    group_count: int
    group_select_mode: int
    norm_type: int
    routed_scaling_factor: float
    eps: float


def moe_gating_topk_core(
    x,
    bias,
    config: MoEGatingTopKConfig) -> tuple[pypto.Tensor, pypto.Tensor, pypto.Tensor]:
    """
    Core computation function for MOE Gating TopK

    Args:
        x: Input tensor [batch_size, num_experts]
        bias: Bias tensor [num_experts] or None
        config: Configuration object containing all parameters

    Returns:
        y_out: Selected and normalized weights [batch_size, k]
        expect_idx_out: Selected expert indices [batch_size, k]
        original_norm_out: Original normalized values [batch_size, num_experts]
    """
    bs = x.shape[0]
    num_experts = x.shape[1]
    x = pypto.cast(x, pypto.DT_FP32)

    if config.norm_type == 1:
        norm_out = pypto.sigmoid(x)
    else:
        norm_out = pypto.softmax(x, -1)
    original_norm_out = norm_out

    if bias is not None:
        bias_expanded = pypto.unsqueeze(bias, 0)
        bias_expanded = pypto.expand_clone(bias_expanded, [bs, num_experts])
        norm_out = pypto.add(norm_out, bias_expanded)

    # When group_count > 1, perform group-based expert selection
    if config.group_count > 1:
        group_unit = num_experts // config.group_count
        group = pypto.reshape(norm_out, [bs, config.group_count, group_unit])
        pypto.set_vec_tile_shapes(bs, config.group_count, group_unit)

        if config.group_select_mode == 1:
            # Use topk2 sum for group selection
            # Select top 2 values from each group and sum them
            group_topk = pypto.topk(group, 2, -1, True)[0]
            group_topk = pypto.sum(group_topk, -1)
        else:
            # Use max for group selection
            # Select maximum value from each group
            group_topk = pypto.amax(group, -1, False)

        # Select top-k groups (smallest indices for largest values)
        group_topk, group_topk_id = pypto.topk(group_topk, config.k_group, -1, True)

        # Initialize mask with zeros
        mask = pypto.full([bs, config.group_count], 0.0, group_topk.dtype)
        # Set selected group positions to 1.0
        topk_group_mask_scatter = pypto.scatter_(mask, 1, group_topk_id, 1.0)
        # Expand mask to match expert dimension
        topk_group_unsqueeze = pypto.unsqueeze(topk_group_mask_scatter, -1)
        pypto.set_vec_tile_shapes(bs, config.group_count, num_experts)
        expand = pypto.expand_clone(topk_group_unsqueeze, [bs, config.group_count, group_unit])
        reshape = pypto.reshape(expand, [bs, num_experts])

        # Mask non-selected experts by setting them to 0.0
        pypto.set_vec_tile_shapes(bs, num_experts)
        twm_not = pypto.logical_not(reshape)
        norm_out = pypto.where(twm_not, 0.0, norm_out)

    # Returns both values and indices
    expect_out, expect_idx_out = pypto.topk(norm_out, config.k, -1, True)

    # Use the indices to gather corresponding weights
    y = pypto.gather(original_norm_out, 1, expect_idx_out)

    # Add eps to avoid division by zero
    if config.norm_type == 1:
        y_sum_eps = pypto.sum(y, -1, True) + config.eps
        y_out = pypto.div(y, y_sum_eps) * config.routed_scaling_factor
    else:
        y_out = y * config.routed_scaling_factor

    return y_out, expect_idx_out, original_norm_out


def moe_gating_topk(
    x_shape: tuple[int, int],
    bias_shape: tuple[int],
    config: MoEGatingTopKConfig,
    run_mode: str = "npu"):
    """
    MOE Gating TopK operator with group-based expert selection

    Args:
        x_shape: Input tensor shape [batch_size, num_experts]
        bias_shape: Bias tensor shape [num_experts]
        config: Configuration object containing all parameters
        run_mode: Execution mode ('npu' or 'sim')

    Returns:
        JIT compiled function
    """
    # Input validation
    num_experts = x_shape[1]
    if config.k <= 0 or config.k > num_experts:
        raise ValueError(f"k must be in range [1, {num_experts}], got {config.k}")
    if config.k_group <= 0 or config.k_group > config.group_count:
        raise ValueError(f"k_group must be in range [1, {config.group_count}], got {config.k_group}")
    if config.group_count > 1 and num_experts % config.group_count != 0:
        raise ValueError(f"num_experts ({num_experts}) must be divisible by group_count ({config.group_count})")
    if config.group_select_mode not in [0, 1]:
        raise ValueError(f"group_select_mode must be 0 or 1, got {config.group_select_mode}")
    if config.norm_type not in [0, 1]:
        raise ValueError(f"norm_type must be 0 (softmax) or 1 (sigmoid), got {config.norm_type}")

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    out_shape = [x_shape[0], config.k]

    @pypto.frontend.jit(
        runtime_options={"run_mode": mode,
        "stitch_function_max_num": 128,
        "device_sched_mode": 1}
    )
    def moe_gating_topk_kernel(
        x: pypto.Tensor(x_shape, pypto.DT_FP32),
        bias: pypto.Tensor(bias_shape, pypto.DT_FP32),
        y_out: pypto.Tensor(out_shape, pypto.DT_FP32),
        expect_idx_out: pypto.tensor(out_shape, pypto.DT_INT32),
        norm_out: pypto.tensor(x_shape, pypto.DT_FP32)

    ):

        # Loop-based processing with batch size 32
        # This improves performance by processing multiple samples in parallel
        pypto.set_vec_tile_shapes(64, 512)
        bs_size = 32

        bs_loop = (x_shape[0] + bs_size - 1) // bs_size
        for bs_index in pypto.loop(0, bs_loop, 1, name="LOOP_MOEGATE_L0", idx_name="bs_idx"):
            b_offset = bs_index * bs_size
            # Process each batch block
            # Note: The last block may have fewer than bs_size samples
            input_view = pypto.view(x, [bs_size, x_shape[1]], [b_offset, 0])
            pypto.set_vec_tile_shapes(bs_size, x_shape[1])

            # Call core computation function
            y_out_loop, expect_idx_out_loop, norm_out_loop = moe_gating_topk_core(
                input_view,
                bias,
                config
            )
            # Write results to output tensors
            y_out[b_offset:b_offset + bs_size, 0:] = y_out_loop
            expect_idx_out[b_offset:b_offset + bs_size, 0:] = expect_idx_out_loop
            norm_out[b_offset:b_offset + bs_size, 0:] = norm_out_loop
    return moe_gating_topk_kernel


def moe_gating_topk_cpu(
    x,
    bias,
    config: MoEGatingTopKConfig):
    """
    CPU reference implementation for MOE Gating TopK

    Args:
        x: Input tensor [batch_size, num_experts]
        bias: Bias tensor [num_experts] or None
        config: Configuration object containing all parameters

    Returns:
        y: Selected weights [batch_size, k]
        indices: Selected expert indices [batch_size, k]
        y2: Original normalized values [batch_size, num_experts]
    """
    # Step 1: Apply normalization (softmax or sigmoid)
    if config.norm_type == 0:
        # Use softmax for normalization
        x_max = torch.max(x, dim=-1, keepdim=True).values
        exp_x = torch.exp(x - x_max)
        x = exp_x / torch.sum(exp_x, dim=-1, keepdim=True)
        x = x.squeeze(-1)
    else:
        x = 1 / (1 + torch.exp(-x))

    original_x = x

    # Step 2: Add bias if provided
    if bias is not None:
        x = x + bias

    # Step 3: Group selection (optional)
    if config.group_count > 1:
        # Reshape to [batch_size, group_count, experts_per_group]
        x = x.reshape(x.shape[0], config.group_count, -1)
        if config.group_select_mode == 0:
            # Use max for group selection
            group_x = torch.amax(x, dim=-1)
        else:
            # Use topk2 sum for group selection
            group_x, _ = torch.topk(x, 2, dim=-1)
            group_x = group_x[..., -2:].sum(dim=-1)

        # Select top-k groups
        indices = torch.argsort(-group_x, dim=-1, stable=True)[:, :config.k_group]
        mask = torch.ones((x.shape[0], config.group_count), dtype=torch.bool, device=x.device)
        mask.scatter_(1, indices, False)
        x = torch.where(mask.unsqueeze(-1), float('-inf'), x)
        x = x.reshape(x.shape[0], -1)

    # Step 4: Select top-k experts
    indices = torch.argsort(-x, dim=-1, stable=True)
    indices = indices[:, :config.k]
    y = torch.gather(original_x, 1, indices)

    # Step 5: Renormalize if sigmoid was used
    if config.norm_type == 1:
        y /= (torch.sum(y, dim=-1, keepdim=True) + config.eps)

    # Step 6: Apply scaling factor
    y = y * config.routed_scaling_factor
    y2 = original_x
    return y, indices, y2


def get_default_moe_gating_topk_config():
    """
    Get default configuration for MOE Gating TopK operator

    Returns:
        MoEGatingTopKConfig: Default configuration object
    """
    return MoEGatingTopKConfig(
        k=4,
        k_group=1,
        group_count=4,
        group_select_mode=0,
        norm_type=1,
        routed_scaling_factor=1.0,
        eps=1e-20
    )


def test_moe_gating_topk_single(config, batch_size, num_experts):
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch_npu.npu.set_device(device_id)

    x = np.random.uniform(0, 2, (batch_size, num_experts)).astype(np.float32)
    bias = np.random.uniform(0, 2, (num_experts,)).astype(np.float32)
    x_tensor = torch.tensor(x).npu()
    bias_tensor = torch.tensor(bias).npu()

    x_tensor_cpu = torch.tensor(x).cpu()
    bias_tensor_cpu = torch.tensor(bias).cpu()

    y_cpu, expert_idx_cpu, out_cpu = moe_gating_topk_cpu(
        x=x_tensor_cpu,
        bias=bias_tensor_cpu,
        config=config)

    y_out = torch.rand((batch_size, config.k), dtype=torch.float32, device=f'npu:{device_id}')
    expect_idx_out = torch.rand((batch_size, config.k), dtype=torch.int32, device=f'npu:{device_id}')
    norm_out = torch.rand((batch_size, num_experts), dtype=torch.float32, device=f'npu:{device_id}')

    moe_gating_topk(
        x_shape=x_tensor.shape,
        bias_shape=bias_tensor.shape,
        config=config)(x_tensor, bias_tensor, y_out, expect_idx_out, norm_out)

    y_cpu_tensor_list = y_cpu.cpu().flatten().tolist()
    expert_idx_cpu_tensor_list = expert_idx_cpu.cpu().flatten().tolist()
    out_cpu_tensor_list = out_cpu.cpu().flatten().tolist()

    y_npu_tensor_list = y_out.cpu().flatten().tolist()
    expert_idx_npu_tensor_list = expect_idx_out.cpu().flatten().tolist()
    out_npu_tensor_list = norm_out.cpu().flatten().tolist()

    assert_allclose(np.array(y_cpu_tensor_list),
                    np.array(y_npu_tensor_list),
                    rtol=5e-3, atol=5e-3)

    assert_allclose(np.array(expert_idx_cpu_tensor_list),
                    np.array(expert_idx_npu_tensor_list),
                    rtol=5e-3, atol=5e-3)

    assert_allclose(np.array(out_cpu_tensor_list),
                    np.array(out_npu_tensor_list),
                    rtol=5e-3, atol=5e-3)


def test_moe_gating_topk():
    test_cases = [
        (
            MoEGatingTopKConfig(
                k=2, k_group=1, group_count=1, group_select_mode=0,
                norm_type=1, routed_scaling_factor=1.0, eps=1e-20
            ),
            32, 16
        ),
        (
            MoEGatingTopKConfig(
                k=4, k_group=1, group_count=4, group_select_mode=0,
                norm_type=1, routed_scaling_factor=1.0, eps=1e-20
            ),
            64, 32
        ),
        (
            MoEGatingTopKConfig(
                k=4, k_group=2, group_count=4, group_select_mode=1,
                norm_type=0, routed_scaling_factor=1.0, eps=1e-20
            ),
            128, 64
        ),
        (
            MoEGatingTopKConfig(
                k=8, k_group=2, group_count=8, group_select_mode=0,
                norm_type=1, routed_scaling_factor=2.0, eps=1e-20
            ),
            256, 128
        ),
    ]

    for i, (config, batch_size, num_experts) in enumerate(test_cases):
        test_moe_gating_topk_single(config, batch_size, num_experts)
        msg = (
            f"Test case {i + 1} passed: k={config.k}, "
            f"group_count={config.group_count}, norm_type={config.norm_type}, "
            f"batch_size={batch_size}, num_experts={num_experts}"
        )
        logging.info(msg)

    logging.info("✓ moe_gating_topk test passed")


def main():
    test_moe_gating_topk()


if __name__ == "__main__":
    main()
