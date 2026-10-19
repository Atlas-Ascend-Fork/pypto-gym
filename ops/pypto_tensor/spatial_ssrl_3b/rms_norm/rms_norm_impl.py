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
PyPTO RMSNorm 实现
"""

import logging
import torch
from torch._dynamo import allow_in_graph
import pypto


@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128},
    pass_options={"cube_l1_reuse_setting": {-1: 4}},
)
def rms_norm_impl_kernel(
    hidden_states: pypto.tensor(),
    weight: pypto.tensor(),
    output: pypto.tensor(),
    epsilon
):
    rank = hidden_states.dim
    tile_shapes = [128 for _ in range(rank)]
    pypto.set_vec_tile_shapes(*tile_shapes)

    y = pypto.rms_norm(hidden_states, weight, epsilon)
    output[:] = y


@allow_in_graph
def rms_norm_impl(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    output = torch.empty_like(hidden_states)
    rms_norm_impl_kernel(hidden_states, weight, output, eps)
    return output


if __name__ == "__main__":
    torch.manual_seed(42)
    
    hidden = torch.randn(1, 31, 2048, dtype=torch.float16, device="cpu")
    weight = torch.ones(2048, dtype=torch.float16, device="cpu")
    
    output = rms_norm_impl(hidden, weight, 1e-6)
    logging.info(f"Input shape: {hidden.shape}, dtype: {hidden.dtype}")
    logging.info(f"Output shape: {output.shape}, dtype: {output.dtype}")
    logging.info(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
