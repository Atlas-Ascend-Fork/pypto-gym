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
"""
import os
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
import pypto
from .utils.get_format import get_format
from .glm_ffn_common_interface import symmetric_quantization_per_token, dequant_dynamic, swiglu


def check_cond(cond, msg):
    if not cond:
        raise ValueError(msg)


def powers_of_2(n: int) -> set[int]:
    check_cond(n > 0, "n must be positive")
    result = set()
    power = 0
    while True:
        current = 1 << power
        if current > n:
            break
        result.add(current)
        power += 1
    return result


def check_args(
    gate_weight: torch.Tensor,
    hidden_states: torch.Tensor,
    top_k: int,
    renormalize: bool,
    topk_group: int,
    num_expert_group: int,
    e_score_correction_bias: torch.Tensor,
    w13,
    w13_scale,
    w2,
    w2_scale,
) -> None:
    check_cond(gate_weight.dim() == 2, "invalid gate weight dim.")
    check_cond(gate_weight.shape[0] == 160, "invalid gate weight shape.")
    check_cond(gate_weight.shape[1] == 5120, "invalid gate weight shape.")
    check_cond(get_format(gate_weight) == 'ND', "invalid gate weight format.")
    check_cond((gate_weight.dtype == torch.float32), "invalid gate weight dtype.")
    check_cond(hidden_states.dim() == 2, "invalid hidden states dim.")
    check_cond(hidden_states.shape[1] == 5120, "invalid hidden states shape.")
    check_cond(get_format(hidden_states) == 'ND', "invalid hidden states format.")
    check_cond(hidden_states.dtype == torch.bfloat16, "invalid hidden states dtype.")

    check_cond(e_score_correction_bias.dim() == 1, "invalid bias dim.")
    check_cond(e_score_correction_bias.shape[0] == 160, "invalid bias shape.")
    check_cond(get_format(e_score_correction_bias) == 'ND', "invalid bias format.")
    check_cond(e_score_correction_bias.dtype == torch.bfloat16, "invalid bias dtype.")
    check_cond(isinstance(top_k, int), "invalid topk dtype.")
    check_cond(isinstance(renormalize, bool), "invalid renormalize dtype.")
    check_cond(isinstance(topk_group, int), "invalid topk_group dtype.")
    check_cond(isinstance(num_expert_group, int), "invalid num_expert_group dtype.")

    check_cond(w13.dim() == 2, "invalid w13 dim.")
    check_cond(w13.shape[0] == 5120, "invalid w13 shape.")
    check_cond(w13.shape[1] == 384, "invalid w13 shape.")
    check_cond(get_format(w13) == 'NZ', "invalid w13 format.")
    check_cond(w13.dtype == torch.int8, "invalid w13 dtype.")
    check_cond(w13_scale.dim() == 1, "invalid w13_scale dim.")
    check_cond(w13_scale.shape[0] == 384, "invalid w13_scale shape.")
    check_cond(get_format(w13_scale) == 'ND', "invalid w13_scale format.")
    check_cond(w13_scale.dtype == torch.bfloat16, "invalid w13_scale dtype.")
    check_cond(w2.dim() == 2, "invalid w2 dim.")
    check_cond(w2.shape[0] == 192, "invalid w2 shape.")
    check_cond(w2.shape[1] == 5120, "invalid w2 shape.")
    check_cond(get_format(w2) == 'NZ', "invalid w2 format.")
    check_cond(w2.dtype == torch.int8, "invalid w2 dtype.")
    check_cond(w2_scale.dim() == 1, "invalid w2_scale dim.")
    check_cond(w2_scale.shape[0] == 5120, "invalid w2_scale shape.")
    check_cond(get_format(w2_scale) == 'ND', "invalid w2_scale format.")
    check_cond(w2_scale.dtype == torch.bfloat16, "invalid hidden states dtype.")


ND = pypto.TileOpFormat.TILEOP_ND
NZ = pypto.TileOpFormat.TILEOP_NZ


@pypto.frontend.jit(
    runtime_options={"device_sched_mode": 1,
                    "stitch_function_max_num": 128},
    pass_options={"cube_l1_reuse_setting": {-1: 2}}
)
def moe_fusion_kernel(
    hidden_states: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16, format=ND),
    mm_weight: pypto.Tensor([], pypto.DT_FP32, format=ND),
    e_score_bias_input: pypto.Tensor([], pypto.DT_BF16, format=ND),
    w13: pypto.Tensor([], pypto.DT_INT8, format=NZ),
    w13_scale: pypto.Tensor([], pypto.DT_BF16, format=ND),
    w2: pypto.Tensor([], pypto.DT_INT8, format=NZ),
    w2_scale: pypto.Tensor([], pypto.DT_BF16, format=ND),
    weight_k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32, format=ND),
    ids_k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_INT32, format=ND),
    ffn_res: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16, format=ND),
    topk_group,
    num_expert_group,
):
    bs = hidden_states.shape[0]

    ne = mm_weight.shape[0]
    topk = ids_k.shape[1]

    pypto.experimental.set_operation_options(combine_axis=True)

    vec_tile_shape = (4, 5120)
    mm1_cube_tile_shape = (8, 256, 256)
    mm2_cube_tile_shape = (8, 192, 256)
    hidden_size = hidden_states.shape[1]
    intermediate_size = w2.shape[0]

    pypto.set_vec_tile_shapes(ne)
    e_score_bias_2d = pypto.reshape(e_score_bias_input, [1, ne], inplace=True)

    for bs_idx, tile_batch in pypto.loop_unroll(0, bs, 1, name="LOOP_MOE_FUSION_L0", idx_name="bs_idx",
                                                unroll_list=powers_of_2(32)):
        tile_hidden_states = hidden_states[bs_idx:bs_idx + tile_batch, :]

        pypto.set_vec_tile_shapes(min(tile_batch, 4), 5120)
        tile_hidden_states_fp32 = pypto.cast(tile_hidden_states, pypto.DT_FP32)
        mm_weight_fp32 = pypto.cast(mm_weight, pypto.DT_FP32)
        pypto.set_cube_tile_shapes([min(tile_batch, 32), min(tile_batch, 32)], [512, 1024], [16, 16])
        res = pypto.matmul(tile_hidden_states_fp32, mm_weight_fp32, tile_hidden_states_fp32.dtype, b_trans=True)

        tile_logits = res
        view_first = 1

        pypto.set_vec_tile_shapes(view_first, ne)
        tile_logits_fp32 = pypto.cast(tile_logits, pypto.DT_FP32)
        e_score_bias_2d_tile = pypto.tensor([tile_batch, ne], e_score_bias_2d.dtype, "e_score_bias_2d_tile")
        for tmp_idx in range(tile_batch):
            pypto.assemble(e_score_bias_2d, [tmp_idx, 0], e_score_bias_2d_tile)
        e_score_bias_2d_cast = pypto.cast(e_score_bias_2d_tile, tile_logits_fp32.dtype)

        topk_weights = pypto.sigmoid(tile_logits_fp32)

        topk_weights_add = pypto.add(topk_weights, e_score_bias_2d_cast)
        group_unit = ne // num_expert_group
        r1 = pypto.reshape(topk_weights_add, [tile_batch, num_expert_group, group_unit])

        pypto.set_vec_tile_shapes(view_first, num_expert_group, group_unit)
        max1 = pypto.amax(r1, -1, False)
        group_weight = max1

        pypto.set_vec_tile_shapes(view_first, num_expert_group)
        _, topk_group_indices = pypto.topk(group_weight, topk_group, -1, True)

        topk_group_mask = pypto.full([tile_batch, num_expert_group], 0.0, group_weight.dtype)

        topk_group_mask_scatter_trans = pypto.scatter_(topk_group_mask, 1, topk_group_indices, 1.0)

        twm_unsqueeze = pypto.unsqueeze(topk_group_mask_scatter_trans, -1)

        pypto.set_vec_tile_shapes(view_first, num_expert_group, ne)
        twm_expand = pypto.expand_clone(twm_unsqueeze, [tile_batch, num_expert_group, group_unit])

        pypto.set_vec_tile_shapes(view_first, num_expert_group, group_unit)
        twm_reshape = pypto.reshape(twm_expand, [tile_batch, ne])

        pypto.set_vec_tile_shapes(view_first, ne)
        twm_not = pypto.logical_not(twm_reshape)

        topk_weights_maskfill = pypto.where(twm_not, 0.0, topk_weights_add)

        _, topk_ids = pypto.topk(topk_weights_maskfill, topk, -1, True)

        tw_gather = pypto.gather(topk_weights, 1, topk_ids)

        pypto.set_vec_tile_shapes(view_first, topk)
        denominator = pypto.sum(tw_gather, -1, True)

        topk_weight_out = pypto.div(tw_gather, denominator)

        weight_k[bs_idx:bs_idx + tile_batch, :] = topk_weight_out
        ids_k[bs_idx:bs_idx + tile_batch, :] = topk_ids

        pypto.set_vec_tile_shapes(vec_tile_shape[0], vec_tile_shape[1])
        hidden_states_offset = [bs_idx, 0]

        hidden_states_quant, hidden_states_scale = symmetric_quantization_per_token(tile_hidden_states)

        pypto.set_cube_tile_shapes([tile_batch, tile_batch],
                                [mm1_cube_tile_shape[1], mm1_cube_tile_shape[1] * 2],
                                [mm1_cube_tile_shape[2], mm1_cube_tile_shape[2]], True)
        up_proj = pypto.matmul(hidden_states_quant, w13, pypto.DT_INT32)

        w13_scale_2d = pypto.unsqueeze(w13_scale, 0)
        pypto.set_vec_tile_shapes(8, intermediate_size * 2)
        up_proj_dequant = dequant_dynamic(up_proj, w13_scale_2d, hidden_states_scale)
        swiglu_out = swiglu(up_proj_dequant)

        down_proj_quant, down_proj_scale = symmetric_quantization_per_token(swiglu_out)

        pypto.set_cube_tile_shapes([tile_batch, tile_batch],
                                [mm2_cube_tile_shape[1], mm2_cube_tile_shape[1] * 2],
                                [mm2_cube_tile_shape[2], mm2_cube_tile_shape[2]], False)
        down_proj = pypto.matmul(down_proj_quant, w2, pypto.DT_INT32)

        w2_scale_2d = pypto.unsqueeze(w2_scale, 0)
        pypto.set_vec_tile_shapes(4, hidden_size)
        down_proj_dequant = dequant_dynamic(down_proj, w2_scale_2d, down_proj_scale)
        out = pypto.cast(down_proj_dequant, hidden_states.dtype)
        pypto.assemble(out, hidden_states_offset, ffn_res)


@allow_in_graph
def moe_fusion(
    gate_weight: torch.Tensor,  # gate matmul weights
    hidden_states: torch.Tensor,  # Hidden states of shape (num_tokens, hidden_size).
    top_k: int,  # number of top k experts.
    renormalize: bool,
    topk_group: int,
    num_expert_group: int,
    e_score_bias: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    ffn_res: torch.Tensor,
):
    if isinstance(hidden_states, FakeTensor):
        return
    check_args(
        gate_weight,
        hidden_states,
        top_k,
        renormalize,
        topk_group,
        num_expert_group,
        e_score_bias,
        w13,
        w13_scale,
        w2,
        w2_scale,
    )

    _bs = hidden_states.shape[0]
    _hidden_size = hidden_states.shape[1]
    inputs = [hidden_states, gate_weight, e_score_bias, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids, ffn_res]

    moe_fusion_kernel(*inputs, topk_group, num_expert_group)


def moe_fusion_pto(
    gate_layer,
    hidden_states,
    share_layer,
    top_k,
    renormalize,
    topk_group=None,
    num_expert_group=None,
    e_score_correction_bias=None,
):
    bs = hidden_states.shape[0]
    _ne = gate_layer.weight.shape[0]
    device_info = hidden_states.device
    topk_weights = torch.empty((bs, top_k), dtype=torch.float32, device=device_info)
    topk_ids = torch.empty((bs, top_k), dtype=torch.int32, device=device_info)

    ffn_res = torch.empty_like(hidden_states, device=device_info)
    w13_int8 = share_layer.gate_up_proj.weight
    w13_scale = share_layer.gate_up_proj.weight_scale
    w2_int8 = share_layer.down_proj.weight
    w2_scale = share_layer.down_proj.weight_scale

    moe_fusion(
        gate_layer.weight,
        hidden_states,
        top_k,
        renormalize,
        topk_group,
        num_expert_group,
        e_score_correction_bias,
        w13_int8,
        w13_scale,
        w2_int8,
        w2_scale,
        topk_weights,
        topk_ids,
        ffn_res,
    )
    return topk_weights, topk_ids, ffn_res
