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
Arctic PyPTO 融合算子库

实际集成的算子：
- Sum LSTM: 基于 LSTM 的 speculative decoding 融合 kernel

融合范围：
- 输入融合 (states + alpha * z4)
- RMSNorm 归一化 (RMSNorm 替代 LayerNorm)
- GELU 激活 + 门控机制 (forget/input/output gates)
- 细胞状态更新 + 隐藏状态输出

应用场景：
- Arctic-Inference 框架的 LSTM-based speculator，用于加速大语言模型的 speculative decoding
"""

from .sum_lstm import sum_lstm, LstmConfig, LstmTileConfig

__all__ = [
    'sum_lstm',
    'LstmConfig',
    'LstmTileConfig',
]
