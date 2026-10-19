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

"""PyPTO grouped_matmul_finalize_routing operator test.

测试说明：
  - golden 实现来自 gmm_finalize_routing_golden.py
  - kernel 与 host 封装来自 gmm_finalize_routing_impl.py
  - 精度对比使用 numpy.testing.assert_allclose
"""


from dataclasses import dataclass
import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

import argparse
import sys

from typing import Any
import pytest
import pypto
import torch
import torch_npu  # noqa: F401
from numpy.testing import assert_allclose

from gmm_finalize_routing_golden import gen_golden, make_group_list
from experimental.matmul.grouped_matmul_finalize_routing.gmm_finalize_routing_impl import (
    FinalizeRoutingConfig,
    FinalizeRoutingGoldenInputs,
    FinalizeRoutingInputs,
    gen_pypto,
)

RTOL = 1e-3
ATOL = 1e-3


@dataclass
class _BuildFinalizeRoutingTensorsOutputs:
    x1: Any
    x2: Any
    scale: Any
    pertoken_scale: Any
    group_list: Any
    shared_input: Any
    logit: Any
    row_index: Any
    out: Any


def _build_finalize_routing_tensors(config, torch_dtype, scale_k):


    """构造 grouped_matmul_finalize_routing 的输入张量。"""
    x1 = torch.randn((config.m, config.k), dtype=torch.float32).uniform_(0, 1).to(torch_dtype)
    if config.transpose_x2:
        x2 = torch.randn((config.num_experts, config.n, config.k), dtype=torch.float32).uniform_(0, 1).to(torch_dtype)
        scale = torch.randn((config.n, scale_k, 2), dtype=torch.float32).uniform_(0, 1)
    else:
        x2 = torch.randn((config.num_experts, config.k, config.n), dtype=torch.float32).uniform_(0, 1).to(torch_dtype)
        scale = torch.randn((scale_k, config.n, 2), dtype=torch.float32).uniform_(0, 1)
    scale = scale.to(torch.float8_e8m0fnu)

    pertoken_scale = torch.randn((config.m, scale_k, 2), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
    group_list = make_group_list(config.m, config.num_experts, config.group_list_type)
    shared_input = torch.randn((config.batch, config.n), dtype=torch.float32).uniform_(-0.5, 0.5).to(torch.bfloat16)
    logit = torch.randn((config.m,), dtype=torch.float32).uniform_(0, 1)
    row_index = torch.arange(config.m, dtype=torch.int64) % config.batch
    out = torch.zeros((config.batch, config.n), dtype=torch.float32)
    return _BuildFinalizeRoutingTensorsOutputs(
    x1, x2, scale, pertoken_scale, group_list,
    shared_input, logit, row_index, out)


TEST_CONFIGS = [
    FinalizeRoutingConfig(
        batch=4, topk=8, k=7168, n=4096, num_experts=8,
        vector_tile_shape=[1, 4, 128, 4],
        in_dtype=pypto.DT_FP8E4M3, transpose_x2=True, group_list_type=1,
        description="case1 batch4 topk8 k7168 n4096 e8",
    ),
    FinalizeRoutingConfig(
        batch=128, topk=8, k=7168, n=4096, num_experts=8,
        vector_tile_shape=[1, 4, 128, 4],
        in_dtype=pypto.DT_FP8E4M3, transpose_x2=True, group_list_type=1,
        description="case2 batch128 topk8 k7168 n4096 e8",
    ),
]


@pytest.mark.soc("950")
@pytest.mark.parametrize("config", TEST_CONFIGS)
def test_gmm_finalize_routing(config):
    """单配置测试：构造数据、运行 golden 与 PyPTO，并执行数值对齐校验。"""
    torch_dtype_map = {
        pypto.DT_FP8E4M3: torch.float8_e4m3fn,
        pypto.DT_FP8E5M2: torch.float8_e5m2,
    }
    torch_dtype = torch_dtype_map.get(config.in_dtype, torch.float8_e4m3fn)
    scale_k = (config.k + 63) // 64

    result = _build_finalize_routing_tensors(config, torch_dtype, scale_k)
    x1 = result.x1
    x2 = result.x2
    scale = result.scale
    pertoken_scale = result.pertoken_scale
    group_list = result.group_list
    shared_input = result.shared_input
    logit = result.logit
    row_index = result.row_index
    out = result.out

    golden = gen_golden(
        FinalizeRoutingGoldenInputs(
            x1=x1,
            x2=x2,
            scale=scale,
            pertoken_scale=pertoken_scale,
            group_list=group_list,
            shared_input=shared_input,
            logit=logit,
            row_index=row_index,
            out=out,
            config=config,
        )
    )

    result = gen_pypto(
        FinalizeRoutingInputs(
            x1=x1,
            x2=x2,
            scale=scale,
            pertoken_scale=pertoken_scale,
            group_list=group_list,
            shared_input=shared_input,
            logit=logit,
            row_index=row_index,
            out=out,
            config=config,
        )
    )

    assert_allclose(golden.cpu().numpy(), result.cpu().numpy(), rtol=RTOL, atol=ATOL)
    print(config.description, "PASSED")


def main():
    parser = argparse.ArgumentParser(description="PyPTO grouped_matmul_finalize_routing operator test")
    parser.add_argument(
        "case",
        type=int,
        nargs="?",
        metavar="N",
        help="Run single 1-based case index (default: run all)",
    )
    args = parser.parse_args()

    if args.case is not None:
        if not 1 <= args.case <= len(TEST_CONFIGS):
            print(f"ERROR: case must be 1..{len(TEST_CONFIGS)}", file=sys.stderr)
            raise RuntimeError("Test execution failed")
        test_gmm_finalize_routing(TEST_CONFIGS[args.case - 1])
        return

    for test_config in TEST_CONFIGS:
        test_gmm_finalize_routing(test_config)


if __name__ == "__main__":
    main()
