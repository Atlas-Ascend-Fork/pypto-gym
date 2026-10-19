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

"""GMM MXFP8 test — validates PyPTO kernel against torch golden reference."""

# ── Import path setup (must precede all other imports) ──
import os
import sys

_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

# ── Standard library imports ──
from dataclasses import dataclass

# ── Third-party imports ──
import pytest
import torch
import torch_npu  # noqa: F401  # must come before pypto kernel imports
import pypto
from numpy.testing import assert_allclose

# ── Application / local imports ──
from experimental.matmul.gmm_mxfp8.gmm_mxfp8_impl import (
    GroupedMatmulInputs,
    ShapeConfig,
    TransposeConfig,
    gen_mxfp8,
)
from gmm_mxfp8_golden import GmmGoldenInputs, gen_golden

# ── OL20: handle TILE_FWK_DEVICE_ID (defaults to 0) ──
device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))


_GMM_MXF8_TEST_CONFIGS = [
    ShapeConfig(
        ori_shape=[16, 512, 7168],
        group_list=[7, 9],
        tile_size=256,
        m_tile_shape=[9, 9],
        k_tile_shape=[256, 256],
        n_tile_shape=[256, 256],
        vector_tile_shape=[1, 8, 256, 32],
        a_trans=False,
        b_trans=False,
        description="testcase m16 k512 n7168 g[7,9]",
    ),
]


@pytest.mark.soc("950")
@pytest.mark.parametrize("tile_config", _GMM_MXF8_TEST_CONFIGS)
def test_gmm_mxfp8(tile_config):
    """Validate the PyPTO kernel against the PyTorch reference implementation."""
    torch.npu.set_device(device_id)
    m_size, k_size, n_size = tile_config.ori_shape
    group_list, b_trans = tile_config.group_list, tile_config.b_trans
    num_groups = len(group_list)

    # Generate input tensors in MXFP8 format
    a = torch.randn(
        (m_size, k_size), dtype=torch.float32
    ).uniform_(0, 1).to(torch.float8_e4m3fn)
    scaled_a = torch.randn(
        (m_size, k_size // 64, 2), dtype=torch.float32
    ).uniform_(0, 1).to(torch.float8_e8m0fnu)
    if b_trans:
        b = torch.randn(
            (num_groups, n_size, k_size), dtype=torch.float32
        ).uniform_(0, 1).to(torch.float8_e4m3fn)
        scaled_b = torch.randn(
            (num_groups, n_size, k_size // 64, 2), dtype=torch.float32
        ).uniform_(0, 1).to(torch.float8_e8m0fnu)
    else:
        b = torch.randn(
            (num_groups, k_size, n_size), dtype=torch.float32
        ).uniform_(0, 1).to(torch.float8_e4m3fn)
        scaled_b = torch.randn(
            (num_groups, k_size // 64, n_size, 2), dtype=torch.float32
        ).uniform_(0, 1).to(torch.float8_e8m0fnu)

    golden = gen_golden(GmmGoldenInputs(
        a=a, b=b, scaled_a=scaled_a, scaled_b=scaled_b,
        group_list=group_list,
        transpose=TransposeConfig(a_trans=tile_config.a_trans, b_trans=b_trans),
    ))
    result = gen_mxfp8(
        GroupedMatmulInputs(a=a, b=b, scaled_a=scaled_a, scaled_b=scaled_b),
        tile_config,
    )
    assert_allclose(
        golden.float().cpu().numpy(),
        result.float().cpu().numpy(),
        rtol=1e-3, atol=1e-3,
    )


if __name__ == "__main__":
    for cfg in _GMM_MXF8_TEST_CONFIGS:
        test_gmm_mxfp8(cfg)
