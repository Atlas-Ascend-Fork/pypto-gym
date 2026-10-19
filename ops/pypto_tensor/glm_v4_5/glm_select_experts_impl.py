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
GLM-4.5 Expert Selection Module for MoE Architecture

This module implements the expert selection logic for Mixture of Experts (MoE) architecture.
It intelligently assigns input tokens to different expert networks based on router logits,
supporting group-based top-k selection and weight renormalization.

Main Functions:
    - select_experts: Main function for expert selection
    - select_experts_kernel: JIT compiled kernel implementation
    - process_main_loop_interation: Process a single batch iteration
"""
import os
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
import pypto
from .utils.get_format import get_format


def check_args(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    topk_group: int,
    num_expert_group: int,
    e_score_correction_bias: torch.Tensor
) -> None:
    assert router_logits.dim() == 2
    assert router_logits.shape[1] == 160
    assert get_format(router_logits) == 'ND'
    assert router_logits.dtype == torch.float32

    assert e_score_correction_bias.dim() == 1
    assert e_score_correction_bias.shape[0] == 160
    assert get_format(e_score_correction_bias) == 'ND'
    assert e_score_correction_bias.dtype == torch.bfloat16

    assert isinstance(top_k, int)
    assert isinstance(renormalize, bool)
    assert isinstance(topk_group, int)
    assert isinstance(num_expert_group, int)


def process_main_loop_interation(
    bs_idx,
    logits_input,
    e_score_bias_2d,
    weight_k,
    ids_k,
    bs,
    ne,
    view_shape,
    view_first,
    topk,
    topk_group,
    num_expert_group,
    renormalize_flag
):
    """
    Process a single batch iteration for expert selection.

    This function performs the following operations:
    1. Apply sigmoid to router logits
    2. Add expert score correction bias
    3. Group experts and select top-k groups
    4. Mask non-selected experts
    5. Select top-k experts from masked logits
    6. Optionally renormalize expert weights

    Args:
        bs_idx: Current batch index
        logits_input: Router logits [num_tokens, num_router_experts]
        e_score_bias_2d: Expert score correction bias [1, num_router_experts]
        weight_k: Output tensor for top-k weights [num_tokens, topk]
        ids_k: Output tensor for top-k expert IDs [num_tokens, topk]
        bs: Batch size (number of tokens)
        ne: Number of experts
        view_shape: Shape for view operation
        view_first: First dimension of view shape
        topk: Number of top experts to select
        topk_group: Number of expert groups for top-k selection
        num_expert_group: Number of experts per group
        renormalize_flag: Whether to renormalize expert weights
    """
    tile_logits = pypto.view(logits_input, view_shape,
                                [bs_idx * view_shape[0], 0],
                                valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]), ne])

    pypto.set_vec_tile_shapes(view_first, ne)
    tile_logits_fp32 = pypto.cast(tile_logits, pypto.DT_FP32)
    e_score_bias_2d_cast = pypto.cast(e_score_bias_2d, tile_logits_fp32.dtype)

    topk_weights = pypto.sigmoid(tile_logits_fp32)

    topk_weights_add = pypto.add(topk_weights, e_score_bias_2d_cast)
    group_unit = ne // num_expert_group
    r1 = pypto.reshape(topk_weights_add,
                        [view_shape[0], num_expert_group, group_unit],
                        valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]), num_expert_group,
                                    group_unit])

    pypto.set_vec_tile_shapes(view_first, num_expert_group, group_unit)
    max1 = pypto.amax(r1, -1, False)
    group_weight = max1

    pypto.set_vec_tile_shapes(view_first, num_expert_group)
    _, topk_group_indices = pypto.topk(group_weight, topk_group, -1, True)

    topk_group_mask = pypto.full([view_shape[0], num_expert_group], 0.0, group_weight.dtype,
                                    valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]),
                                                num_expert_group])

    topk_group_mask_scatter_trans = pypto.scatter_(topk_group_mask, 1, topk_group_indices, 1.0)

    twm_unsqueeze = pypto.unsqueeze(topk_group_mask_scatter_trans, -1)

    pypto.set_vec_tile_shapes(view_first, num_expert_group, ne)
    twm_expand = pypto.expand_clone(twm_unsqueeze, [view_shape[0], num_expert_group, group_unit],
                                    valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]),
                                                    num_expert_group, group_unit])

    pypto.set_vec_tile_shapes(view_first, num_expert_group, group_unit)
    twm_reshape = pypto.reshape(twm_expand,
                                [view_shape[0], ne],
                                valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]), ne])

    pypto.set_vec_tile_shapes(view_first, ne)
    twm_not = pypto.logical_not(twm_reshape)

    topk_weights_maskfill = pypto.where(twm_not, 0.0, topk_weights_add)

    _, topk_ids = pypto.topk(topk_weights_maskfill, topk, -1, True)

    tw_gather = pypto.gather(topk_weights, 1, topk_ids)

    pypto.set_vec_tile_shapes(view_first, topk)
    if pypto.symbolic_scalar(renormalize_flag):
        denominator = pypto.sum(tw_gather, -1, True)
        topk_weight_out = pypto.div(tw_gather, denominator)
    else:
        denominator = tw_gather
        topk_weight_out = denominator

    weight_k[bs_idx * view_shape[0]:, 0:] = topk_weight_out
    ids_k[bs_idx * view_shape[0]:, 0:] = topk_ids


@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128}
)
def select_experts_kernel(
    logits: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    e_score_bias_input: pypto.Tensor([], pypto.DT_BF16),
    weight_k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    index_k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_INT32),
    renormalize=True,
    topk_group=1,
    num_expert_group=1
):
    batch_size = logits.shape[0]
    number_experts = logits.shape[1]
    topk = index_k.shape[1]
    view_shape = (1, number_experts)
    view_first = 1
    bs_loop = (batch_size + view_shape[0] - 1) // view_shape[0]

    pypto.set_vec_tile_shapes(number_experts)
    e_score_bias_2d = pypto.reshape(e_score_bias_input, [1, number_experts], inplace=True)

    for bs_index in pypto.loop(bs_loop, name="LOOP_MOEGATE_L0", idx_name="bs_idx"):
        process_main_loop_interation(
            bs_index,
            logits,
            e_score_bias_2d,
            weight_k,
            index_k,
            batch_size,
            number_experts,
            view_shape,
            view_first,
            topk,
            topk_group,
            num_expert_group,
            renormalize
        )


@allow_in_graph
def select_experts(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    topk_group: int,
    num_expert_group: int,
    e_score_correction_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor
):
    """
    Select top-k experts for each token based on router logits.

    This function implements the expert selection mechanism for MoE architecture.
    It uses a two-stage selection process:
    1. First selects top-k expert groups
    2. Then selects top-k experts from the selected groups

    Args:
        router_logits: Router logits [num_tokens, num_router_experts]
        top_k: Number of top experts to select per token
        renormalize: Whether to renormalize the routing weights
        topk_group: Number of expert groups to select from
        num_expert_group: Number of experts in each group
        e_score_correction_bias: Correction bias to apply to expert scores [num_router_experts]
        topk_weights: Output tensor for top-k expert weights [num_tokens, topk]
        topk_ids: Output tensor for top-k expert IDs [num_tokens, topk]

    Note:
        This function is decorated with @allow_in_graph to enable integration
        with PyTorch's compilation graph.
    """
    if isinstance(router_logits, FakeTensor):
        return
    check_args(
        router_logits,
        top_k,
        renormalize,
        topk_group,
        num_expert_group,
        e_score_correction_bias
    )
    select_experts_kernel(
        router_logits,
        e_score_correction_bias,
        topk_weights,
        topk_ids,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group
    )
