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

"""为 tests/ops/qwen3_5_9b 下的测试提供 npu_device fixture。

在模块顶层 import torch_npu，使得 collection 阶段（先于 fixture）
import kernel 文件触发 `@pypto.frontend.jit` 装饰器调用 `torch.npu.is_available()`
时 `torch.npu` 已经可用。
"""
import os

import torch  # noqa: F401
import torch_npu  # noqa: F401
import pytest


@pytest.fixture
def npu_device(device):
    """`npu:N` device string with `torch.npu.set_device` already called."""
    torch.npu.set_device(device)
    os.environ["TILE_FWK_DEVICE_ID"] = str(device)
    return f"npu:{device}"

"""
Qwen3-Next PyPTO 融合算子库

实际集成的算子：
- Gated Delta Rule (chunk 融合版): chunk_gated_delta_rule, chunk_gated_delta_rule_unaligned

融合范围：
- l2norm + pre_attn + inverse_pto + inverse_matmul + cal_value_and_key_cumdecay + recurrent_state_attn_all
- 支持对齐和未对齐两种 chunk 模式

应用场景：
- Qwen3-Next 模型的 Gated Delta Rule prefill 推理加速
"""

from .gated_delta_rule_impl import (
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_unaligned,
)

__all__ = [
    'chunk_gated_delta_rule',
    'chunk_gated_delta_rule_unaligned',
]
