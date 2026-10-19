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
import os
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
from glm_v4_5.glm_select_experts_impl import select_experts


def gen_row_idx_gloden(hidden_states, top_k):
    num_tokens = hidden_states.shape[0]
    row_idx_len = num_tokens * top_k
    row_idx = (torch.arange(0, row_idx_len, dtype=torch.int32,
                            device=hidden_states.device).view(top_k, -1).permute(1, 0).contiguous())
    return row_idx


def _compute_select_experts_golden(router_logits, e_score_bias, bs, ne, top_k,
                                    topk_group, num_expert_group, renormalize):
    router_logits_fp32 = router_logits.to(torch.float)
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
    tgm_expand = tgm_unsquee.expand(
        bs, num_expert_group, ne // num_expert_group)
    topk_weight_mask = tgm_expand.reshape(bs, -1)
    logical_not_tmp = ~topk_weight_mask.bool()
    topk_weights_fill = topk_weights_g_add.masked_fill(
        logical_not_tmp, 0.0)

    topk_ids_int64 = torch.topk(topk_weights_fill.to(torch.float32),
                                k=top_k,
                                dim=-1,
                                sorted=False)[1]
    topk_ids_int32 = topk_ids_int64.to(torch.int32)

    topk_weights_gather = original_weights.gather(1, topk_ids_int64)

    if renormalize:
        topk_weights_out = topk_weights_gather / \
            topk_weights_gather.sum(dim=-1, keepdim=True)
    else:
        topk_weights_out = topk_weights_gather

    golden_weights = np.array(topk_weights_out.cpu().flatten().tolist())
    golden_ids = np.array(topk_ids_int32.cpu().flatten().tolist())
    return golden_weights, golden_ids


def test_select_experts():
    bs = 32
    ne = 160
    top_k = 8
    topk_group = 1
    num_expert_group = 1
    renormalize = True
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    for i in range(0, 2):
        if i == 1:
            bs = 1026
        torch.manual_seed(0)
        np.random.seed(0)
        router_logits = torch.rand((bs, ne), dtype=torch.float32, device=f'npu:{device_id}')
        e_score_bias = torch.rand((ne), dtype=torch.bfloat16, device=f'npu:{device_id}')
        topk_weights = torch.rand((bs, top_k), dtype=torch.float32, device=f'npu:{device_id}')
        topk_ids = torch.rand((bs, top_k), dtype=torch.int32, device=f'npu:{device_id}')

        inputs = [router_logits, top_k, renormalize, topk_group, num_expert_group, e_score_bias, topk_weights, topk_ids]
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            select_experts(*inputs)
        g.replay()

        golden_weights, golden_ids = _compute_select_experts_golden(
            router_logits, e_score_bias, bs, ne, top_k, topk_group, num_expert_group, renormalize)

        assert_allclose(np.array(topk_weights.cpu().flatten().tolist()),
                        golden_weights,
                        rtol=5e-3, atol=5e-3)

        assert_allclose(np.array(topk_ids.cpu().flatten().tolist()),
                        golden_ids,
                        rtol=5e-3, atol=5e-3)


def main():
    test_select_experts()


if __name__ == "__main__":
    main()
