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
Fused SwiGLU Operator Test
"""
import os
import math
import logging
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
import pytest
from numpy.testing import assert_allclose


logging.basicConfig(level=logging.INFO, format="%(message)s")


def get_device_id():
    return int(os.environ.get('TILE_FWK_DEVICE_ID', 0))


def golden_fused_swiglu_fwd(x, w_g, w_fc, b_g, b_fc):
    gate = x @ w_g + b_g
    fc = x @ w_fc + b_fc
    gate_silu = gate * torch.sigmoid(gate)
    y = (gate_silu * fc).to(x.dtype)
    return y


def test_fwd(device_id):
    m, k, n = 220000, 512, 1024
    torch.npu.set_device(device_id)
    logging.info(f"\n=== Forward Test [{m}, {k}] @ [{k}, {n}] ===")
    device = f'npu:{device_id}'
    np.random.seed(0)
    torch.manual_seed(0)

    x = torch.randn(m, k, dtype=torch.bfloat16, device=device) / math.sqrt(m)
    w_g = torch.randn(k, n, dtype=torch.bfloat16, device=device) / math.sqrt(k)
    w_fc = torch.randn(k, n, dtype=torch.bfloat16, device=device) / math.sqrt(k)
    b_g = torch.randn(1, n, dtype=torch.bfloat16, device=device) / math.sqrt(n)
    b_fc = torch.randn(1, n, dtype=torch.bfloat16, device=device) / math.sqrt(n)
    y_golden = golden_fused_swiglu_fwd(x, w_g, w_fc, b_g, b_fc)
    y_out = torch.empty(m, n, dtype=torch.bfloat16, device=device)

    from experimental.ops_transformer.fused_swiglu.fused_swiglu_impl import fused_swiglu_fwd_kernel
    fused_swiglu_fwd_kernel(x, w_g, w_fc, b_g, b_fc, y_out)

    assert_allclose(y_out.cpu().float().numpy(), y_golden.cpu().float().numpy(), rtol=0.0078125, atol=0.0001)
    logging.info(f"\n=== Forward Pass ===")


def main():
    device_id = get_device_id()
    torch.npu.set_device(device_id)
    test_fwd(device_id)


if __name__ == "__main__":
    main()