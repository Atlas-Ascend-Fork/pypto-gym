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
Fused SwiGLU Grad Operator Test
"""
import logging
import math
import os
import sys

import torch
import torch_npu

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


def golden_fused_swiglu_bwd(dy, g, fc, w_g, w_fc, x):
    dy_fp32 = dy.float()
    g_fp32 = g.float()
    fc_fp32 = fc.float()
    w_g_fp32 = w_g.float()
    w_fc_fp32 = w_fc.float()
    x_fp32 = x.float()
    sigmoid_g = torch.sigmoid(g_fp32)
    silu_g = g_fp32 * sigmoid_g
    dg = dy_fp32 * fc_fp32 * sigmoid_g * (1 + g_fp32 * (1 - sigmoid_g))
    dfc = dy_fp32 * silu_g
    db_g = (dg.sum(dim=0, keepdim=True)).to(dy.dtype)
    db_fc = (dfc.sum(dim=0, keepdim=True)).to(dy.dtype)
  
    dw_g = (x_fp32.to(dy.dtype).float().T @ dg.to(dy.dtype).float()).to(dy.dtype)
    dw_fc = (x_fp32.to(dy.dtype).float().T @ dfc.to(dy.dtype).float()).to(dy.dtype)
  
    dx = (dg.to(dy.dtype).float() @ w_g_fp32.to(dy.dtype).float().T + \
            dfc.to(dy.dtype).float() @ w_fc_fp32.to(dy.dtype).float().T).to(dy.dtype)
    return dx, dw_g, dw_fc, db_g, db_fc


def test_bwd(device_id):
    m, k, n = 220000, 512, 1024
    torch.npu.set_device(device_id)
    logging.info(f"\n=== Backward Test [{m}, {k}] @ [{k}, {n}] ===")
    device = f'npu:{device_id}'
    np.random.seed(0)
    torch.manual_seed(0)

    x = torch.randn(m, k, dtype=torch.bfloat16, device=device) / math.sqrt(m)
    w_g = torch.randn(k, n, dtype=torch.bfloat16, device=device) / math.sqrt(k)
    w_fc = torch.randn(k, n, dtype=torch.bfloat16, device=device) / math.sqrt(k)
    dy = torch.randn(m, n, dtype=torch.bfloat16, device=device) / math.sqrt(m)
    g = torch.randn(m, n, dtype=torch.bfloat16, device=device) / math.sqrt(m)
    fc = torch.randn(m, n, dtype=torch.bfloat16, device=device) / math.sqrt(m)
    dx_golden, dw_g_golden, dw_fc_golden, db_g_golden, db_fc_golden = golden_fused_swiglu_bwd(dy, g, fc, w_g, w_fc, x)

    dx_out = torch.empty(m, k, dtype=torch.bfloat16, device=device)
    dw_g_out = torch.zeros(k, n, dtype=torch.bfloat16, device=device)
    dw_fc_out = torch.zeros(k, n, dtype=torch.bfloat16, device=device)
    db_g_out = torch.zeros(1, n, dtype=torch.bfloat16, device=device)
    db_fc_out = torch.zeros(1, n, dtype=torch.bfloat16, device=device)
    dg_out = torch.empty(m, n, dtype=torch.bfloat16, device=device)
    dfc_out = torch.empty(m, n, dtype=torch.bfloat16, device=device)

    from experimental.ops_transformer.fused_swiglu_grad.fused_swiglu_grad_impl import (
        fused_swiglu_bwd_b_kernel,
        fused_swiglu_bwd_w_kernel,
        fused_swiglu_bwd_x_kernel,
    )
    fused_swiglu_bwd_b_kernel(dy, g, fc, dg_out, dfc_out, db_g_out, db_fc_out)
    fused_swiglu_bwd_w_kernel(x, dg_out, dfc_out, dw_g_out, dw_fc_out)
    fused_swiglu_bwd_x_kernel(dg_out, dfc_out, w_g, w_fc, dx_out)

    assert_allclose(dx_out.cpu().float().numpy(), dx_golden.cpu().float().numpy(), rtol=0.0078125, atol=0.0001)
    assert_allclose(dw_g_out.cpu().float().numpy(), dw_g_golden.cpu().float().numpy(), rtol=0.0078125, atol=0.0001)
    assert_allclose(dw_fc_out.cpu().float().numpy(), dw_fc_golden.cpu().float().numpy(), rtol=0.0078125, atol=0.0001)
    assert_allclose(db_g_out.cpu().float().numpy(), db_g_golden.cpu().float().numpy(), rtol=0.0078125, atol=0.0001)
    assert_allclose(db_fc_out.cpu().float().numpy(), db_fc_golden.cpu().float().numpy(), rtol=0.0078125, atol=0.0001)
    logging.info(f"\n=== Backward Pass ===")


def main():
    device_id = get_device_id()
    torch.npu.set_device(device_id)
    test_bwd(device_id)


if __name__ == "__main__":
    main()
