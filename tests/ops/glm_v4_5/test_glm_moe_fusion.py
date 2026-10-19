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
from dataclasses import dataclass
import os
from typing import Any
import torch
import torch_npu

import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

import numpy as np
from numpy.testing import assert_allclose
import pypto
from glm_v4_5.glm_moe_fusion_impl import moe_fusion, check_cond


def gen_quan_per_channel_weight_nz(x):
    x_fp32 = x.to(torch.float32)
    max_value = x_fp32.abs().max(dim=0, keepdim=True)[0]
    scale_quant = 127.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_rint = torch.round(y_fp32).to(torch.int32)
    y_round = torch.round(y_rint).to(torch.float16)
    y_int8 = torch.trunc(y_round).to(torch.int8)
    y_int8_nz = torch_npu.npu_format_cast(y_int8, 29)
    scale_dequant = (1 / scale_quant)
    return y_int8_nz, scale_dequant


def _compute_router_golden(hidden_states, mm_weight, e_score_bias, bs, ne,
                           num_expert_group, top_k, topk_group, renormalize):
    """Compute golden router output: topk_weights and topk_ids."""
    result = torch.matmul(hidden_states.to(torch.float32), mm_weight.to(torch.float32).t())
    router_logits_fp32 = result.to(torch.float)
    original_weights = router_logits_fp32.sigmoid()
    bias_2d = e_score_bias.unsqueeze(0)
    topk_weights_g_add = original_weights + bias_2d
    tw_view = topk_weights_g_add.view(bs, num_expert_group, -1)
    grouped_weights = tw_view.max(dim=-1).values

    topk_group_indices_g = torch.topk(grouped_weights.to(torch.float32),
                                      k=topk_group,
                                      dim=-1,
                                      sorted=False)[1]
    topk_group_mask = torch.zeros_like(grouped_weights)
    topk_group_mask.scatter_(1, topk_group_indices_g, 1)
    tgm_unsquee = topk_group_mask.unsqueeze(-1)
    tgm_expand = tgm_unsquee.expand(bs, num_expert_group, ne // num_expert_group)
    topk_weight_mask = tgm_expand.reshape(bs, -1)
    logical_not_tmp = ~topk_weight_mask.bool()
    topk_weights_fill = topk_weights_g_add.masked_fill(
        logical_not_tmp, 0.0)

    topk_ids_int64 = torch.topk(topk_weights_fill.to(torch.float32), k=top_k, dim=-1, sorted=False)[1]
    topk_ids_int32 = topk_ids_int64.to(torch.int32)

    topk_weights_gather = original_weights.gather(1, topk_ids_int64)
    if renormalize:
        topk_weights_out = topk_weights_gather / topk_weights_gather.sum(dim=-1, keepdim=True)
    else:
        topk_weights_out = topk_weights_gather

    topk_weight_2_tensor_list = topk_weights_out.cpu().flatten().tolist()
    topk_ids_tensor_list = topk_ids_int32.cpu().flatten().tolist()
    return topk_weight_2_tensor_list, topk_ids_tensor_list


def _compute_moe_golden(hidden_states, w13, w13_scale, w2, w2_scale, x_dtype):
    """Compute golden MoE FFN output via dynamic quant + matmul + swiglu chain."""
    quantized_x, dynamic_scale = torch_npu.npu_dynamic_quant(hidden_states)
    output_w13 = torch_npu.npu_quant_matmul(
        quantized_x,
        w13,
        w13_scale,
        pertoken_scale=dynamic_scale,
        bias=None,
        output_dtype=x_dtype
    )
    swiglu_out = torch_npu.npu_swiglu(output_w13)
    quantized_x2, x_scale = torch_npu.npu_dynamic_quant(swiglu_out)
    golden = torch_npu.npu_quant_matmul(
        quantized_x2,
        w2,
        w2_scale,
        pertoken_scale=x_scale,
        bias=None,
        output_dtype=x_dtype
    )
    return golden


@dataclass
class _RunSingleMoeIterInputs:
    bs: Any
    hidden_size: Any
    intermediate_size: Any
    x_dtype: Any
    ne: Any
    h_num: Any
    top_k: Any
    topk_group: Any
    num_expert_group: Any
    renormalize: Any
    enable_graph: Any
    device_id: Any


def _run_single_moe_iter(cfg: _RunSingleMoeIterInputs):


    """Run one batch-size iteration of moe_fusion test."""
    hidden_states = torch.rand(
        (cfg.bs, cfg.hidden_size),
        dtype=cfg.x_dtype, device=f'npu:{cfg.device_id}') * 0.05
    weight_gate_upper_tensor = torch.rand((cfg.hidden_size, cfg.intermediate_size * 2),
                                        dtype=cfg.x_dtype, device=f'npu:{cfg.device_id}') * 0.05
    w13, w13_scale = gen_quan_per_channel_weight_nz(weight_gate_upper_tensor)
    w13_scale = w13_scale.reshape(-1).to(cfg.x_dtype)
    weight_down_proj_tensor = torch.rand((cfg.intermediate_size, cfg.hidden_size),
                                        dtype=cfg.x_dtype, device=f'npu:{cfg.device_id}') * 0.05
    w2, w2_scale = gen_quan_per_channel_weight_nz(weight_down_proj_tensor)
    w2_scale = w2_scale.reshape(-1).to(cfg.x_dtype)
    ffn_res = torch.empty((cfg.bs, cfg.hidden_size), dtype=cfg.x_dtype, device=f'npu:{cfg.device_id}')

    mm_weight = torch.rand((cfg.ne, cfg.h_num), dtype=torch.float32, device=f'npu:{cfg.device_id}')
    e_score_bias = torch.rand((cfg.ne), dtype=torch.bfloat16, device=f'npu:{cfg.device_id}')
    topk_weights = torch.empty((cfg.bs, cfg.top_k), dtype=torch.float32, device=f'npu:{cfg.device_id}')
    topk_ids = torch.empty((cfg.bs, cfg.top_k), dtype=torch.int32, device=f'npu:{cfg.device_id}')

    inputs = [mm_weight, hidden_states, cfg.top_k, cfg.renormalize, cfg.topk_group, cfg.num_expert_group,
              e_score_bias, w13, w13_scale, w2, w2_scale]
    outputs = [topk_weights, topk_ids, ffn_res]

    if cfg.enable_graph:
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            moe_fusion(*inputs, *outputs)
        g.replay()
    else:
        moe_fusion(*inputs, *outputs)

    topk_weight_list, topk_ids_list = _compute_router_golden(
        hidden_states, mm_weight, e_score_bias, cfg.bs, cfg.ne,
        cfg.num_expert_group, cfg.top_k, cfg.topk_group, cfg.renormalize)
    golden = _compute_moe_golden(hidden_states, w13, w13_scale, w2, w2_scale, cfg.x_dtype)

    assert_allclose(np.array(topk_weights.cpu().flatten().tolist()), np.array(topk_weight_list),
                    rtol=5e-3, atol=5e-3)
    assert_allclose(np.array(topk_ids.cpu().flatten().tolist()), np.array(topk_ids_list),
                    rtol=5e-3, atol=5e-3)
    assert_allclose(np.array(ffn_res.cpu().flatten().tolist()), np.array(golden.cpu().flatten().tolist()),
                    rtol=0.0078125, atol=0.0001)

    import pypto.pypto_impl as pypto_impl
    total_elapsed = pypto_impl.GetCompilerMonitorTotalElapsed()
    check_cond(total_elapsed <= 60, f"glm_moe_fusion compile elapsed timeout {total_elapsed}s > 60s.")


def test_moe_fusion():
    enable_graph = False
    ne = 160
    h_num = 5120
    top_k = 8
    topk_group = 1
    num_expert_group = 1
    renormalize = True
    x_dtype = torch.bfloat16
    intermediate_size = 192
    hidden_size = h_num

    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    torch.manual_seed(0)
    for bs in [32, 32, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]:
        _run_single_moe_iter(_RunSingleMoeIterInputs(
            bs, hidden_size, intermediate_size, x_dtype, ne, h_num,
            top_k, topk_group, num_expert_group, renormalize,
            enable_graph, device_id))


def main():
    pypto.set_host_options(compile_monitor_enable=1,
        compile_timeout=10,
        compile_timeout_stage=5,
        compile_monitor_print_interval=2)
    test_moe_fusion()


if __name__ == "__main__":
    main()
