#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from dataclasses import dataclass

import pypto


@dataclass(frozen=True)
class RMSPropConfig:
    lr: float = 0.001
    rho: float = 0.9
    momentum: float = 0.9
    epsilon: float = 1e-7


@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "stitch_function_max_num": 128,
    }
)
def apply_rms_prop_kernel(
    var: pypto.Tensor([], pypto.DT_FP32),
    ms: pypto.Tensor([], pypto.DT_FP32),
    mom: pypto.Tensor([], pypto.DT_FP32),
    grad: pypto.Tensor([], pypto.DT_FP32),
    config: RMSPropConfig,
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(32, 512)

    grad_sq = grad * grad
    ms_new = ms + (grad_sq - ms) * (1.0 - config.rho)
    mom_new = mom * config.momentum + (grad * config.lr) / pypto.sqrt(ms_new + config.epsilon)
    var_new = var - mom_new

    var.move(var_new)
    ms.move(ms_new)
    mom.move(mom_new)
