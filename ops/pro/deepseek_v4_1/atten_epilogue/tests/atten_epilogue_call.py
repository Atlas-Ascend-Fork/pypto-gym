#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""AttenEpilogue 用例共享调用 helper(非 golden; jit/aclnn 两种入口签名一致).

交付形态下输出由调用方预分配(与 op_host 注册的输出一致), 因此两种模式的入口都是
``entry(o, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos, out)``.
"""

import torch

from atten_epilogue_kernel_consts import OUT_FEATURES


def make_output(o, rows=None):
    """Allocate a bf16 [rows, 5120] output (rows defaults to T)."""
    return torch.empty((rows if rows is not None else o.shape[0], OUT_FEATURES),
                       dtype=torch.bfloat16, device=o.device)


def run_epilogue(entry, args, out=None):
    """Run the current-mode entry over `args` and return the output tensor."""
    if out is None:
        out = make_output(args[0])
    entry(*args, out)
    return out
