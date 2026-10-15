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
sys.path.insert(0, os.path.join(_p, 'src'))
sys.path.insert(0, os.path.join(_p, 'src', 'pypto_gym', 'ops', 'pypto_tensor'))

import numpy as np
from numpy.testing import assert_allclose
from glm_v4_5.glm_ffn_shared_expert_quant_impl import ffn_shared_expert_quant
import pytest
import collections


FfnGenInputOutput = collections.namedtuple("FfnGenInputOutput",
    ["hidden_states", "w13", "w13_scale", "w2", "w2_scale", "ffn_res"])


def ffn_golden_quan_per_token(x):
    x_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    max_value = x_fp32.abs().max(dim=1, keepdim=True)[0]
    scale_quant = 127.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_rint = torch.round(y_fp32).to(torch.int32)
    y_round = torch.round(y_rint).to(torch.float16)
    y_int8 = torch.trunc(y_round).to(torch.int8)
    scale_dequant = (1 / scale_quant)
    return y_int8, scale_dequant


def ffn_golden_quan_per_channel(x):
    x_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    max_value = x_fp32.abs().max(dim=0, keepdim=True)[0]
    scale_quant = 127.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_rint = torch.round(y_fp32).to(torch.int32)
    y_round = torch.round(y_rint).to(torch.float16)
    y_int8 = torch.trunc(y_round).to(torch.int8)
    scale_dequant = (1 / scale_quant)
    return y_int8, scale_dequant


def moe_torch_npu(hidden_states, w13, w13_scale, w2, w2_scale):
    x_dtype = hidden_states.dtype
    quantized_x, dynamic_scale = torch_npu.npu_dynamic_quant(hidden_states)
    output_w13 = torch_npu.npu_quant_matmul(
            quantized_x,
            w13,
            w13_scale,
            pertoken_scale=dynamic_scale,
            bias=None,
            output_dtype=x_dtype,
        )
    swiglu_out = torch_npu.npu_swiglu(output_w13)
    quantized_x, x_scale = torch_npu.npu_dynamic_quant(swiglu_out)
    output = torch_npu.npu_quant_matmul(
            quantized_x,
            w2,
            w2_scale,
            pertoken_scale=x_scale,
            bias=None,
            output_dtype=x_dtype,
        )
    return output


def gen_input(
    b: int,
    s: int,
    hidden_size: int,
    intermediate_size: int,
    dtypes: torch.dtype,
    device_id: int
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(1)
    hidden_states = torch.randn((b * s, hidden_size), dtype=dtypes, device=f'npu:{device_id}') * 0.01 * 2 - 0.01

    weight_gate_upper_tensor = torch.randn((hidden_size, intermediate_size * 2),
                                           dtype=dtypes, device=f'npu:{device_id}') * 0.01 * 2 - 0.01
    w13, w13_scale = ffn_golden_quan_per_channel(weight_gate_upper_tensor)
    w13_scale = w13_scale.reshape(-1).to(dtypes)

    weight_down_proj_tensor = torch.randn((intermediate_size, hidden_size),
                                          dtype=dtypes, device=f'npu:{device_id}') * 0.01 * 2 - 0.01
    w2, w2_scale = ffn_golden_quan_per_channel(weight_down_proj_tensor)
    w2_scale = w2_scale.reshape(-1).to(dtypes)

    ffn_res = torch.empty((b * s, hidden_size), dtype=dtypes, device=f'npu:{device_id}')
    return FfnGenInputOutput(hidden_states, w13, w13_scale, w2, w2_scale, ffn_res)


@pytest.mark.soc("950", "910")
def test_ffn_share() -> None:
    x_dtype = torch.bfloat16
    s = 1
    intermediate_size = 192
    hidden_size = 5120
    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    for b in [1, 2]:
        hidden_states, w13, w13_scale, w2, w2_scale, ffn_res = \
            gen_input(b, s, hidden_size, intermediate_size, x_dtype, device_id)
        w13 = torch_npu.npu_format_cast(w13, 29)
        w2 = torch_npu.npu_format_cast(w2, 29)
        ffn_shared_expert_quant(hidden_states, w13, w13_scale, w2, w2_scale, ffn_res)

        golden = moe_torch_npu(hidden_states, w13, w13_scale, w2, w2_scale)
        assert_allclose(np.array(ffn_res.cpu().flatten().tolist()), np.array(golden.cpu().flatten().tolist()),
                        rtol=0.0078125, atol=0.0001)


def main():
    test_ffn_share()


if __name__ == "__main__":
    main()
