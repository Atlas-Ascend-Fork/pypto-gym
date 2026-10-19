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

import logging
import torch
import pypto

_logger = logging.getLogger(__name__)

PYPTO_AVAILABLE = True
PYPTO_KERNEL_AVAILABLE = True


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def rms_norm_kernel(hidden_states: pypto.tensor(),
                    weight: pypto.tensor(),
                    output: pypto.tensor(),
                    eps):
    rank = hidden_states.dim
    tile_shapes = [128 for _ in range(rank)]
    tile_shapes[-1] = 2048
    pypto.set_vec_tile_shapes(*tile_shapes)
    y = pypto.rms_norm(hidden_states, weight, eps)
    output[:] = y


def rms_norm_pto_native(hidden_states, weight, eps=1e-6):
    out = torch.empty_like(hidden_states)
    rms_norm_kernel(hidden_states, weight, out, eps)
    return out


__all__ = [
    'rms_norm_pto_native',
    'PYPTO_AVAILABLE',
    'PYPTO_KERNEL_AVAILABLE',
]
