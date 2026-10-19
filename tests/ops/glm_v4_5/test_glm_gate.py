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
from glm_v4_5.glm_gate_impl import gate
import pytest


@pytest.mark.soc("950", "910")
def test_select_experts_mm():
    bs = 64
    ne = 160
    h_num = 5120

    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    for i in range(0, 1):
        if i == 1:
            bs = 1026
        torch.manual_seed(0)
        np.random.seed(0)
        hidden_states = torch.rand((bs, h_num), dtype=torch.float32, device=f'npu:{device_id}')
        mm_weight = torch.rand((ne, h_num), dtype=torch.float32, device=f'npu:{device_id}')
        router_logits_out = torch.rand((bs, ne), dtype=torch.float32, device=f'npu:{device_id}')

        inputs = [hidden_states, mm_weight, router_logits_out]

        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            gate(*inputs)
        g.replay()

        result = torch.matmul(hidden_states, mm_weight.t())
        result_list = result.cpu().flatten().tolist()

        assert_allclose(np.array(router_logits_out.cpu().flatten().tolist()),
                        np.array(result_list),
                        rtol=5e-3, atol=5e-3)


def main():
    test_select_experts_mm()


if __name__ == "__main__":
    main()
