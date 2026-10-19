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
QAT (Quantization-Aware Training) PyPTO 融合算子库

实际集成的算子：
- 非对称 per-group 量化 (forward + backward): ai_infra_qat_asymmetric_per_group / _backward
- 对称 per-channel 量化 (forward + backward): ai_infra_qat_symmetric_per_channel / _backward
- 对称 per-tensor 量化 (forward + backward): ai_infra_qat_symmetric_per_tensor / _backward

融合范围：
- 量化 + 反量化 + 梯度反传的端到端融合
- 支持 group_size=128、bit=4/8 等可配置量化参数

应用场景：
- 大语言模型的量化感知训练 (QAT)，在训练过程中模拟量化误差以提升推理精度
"""

from .qat_impl import (
    ai_infra_qat_asymmetric_per_group,
    ai_infra_qat_asymmetric_per_group_backward,
    ai_infra_qat_symmetric_per_channel,
    ai_infra_qat_symmetric_per_channel_backward,
    ai_infra_qat_symmetric_per_tensor,
    ai_infra_qat_symmetric_per_tensor_backward,
)

__all__ = [
    'ai_infra_qat_asymmetric_per_group',
    'ai_infra_qat_asymmetric_per_group_backward',
    'ai_infra_qat_symmetric_per_channel',
    'ai_infra_qat_symmetric_per_channel_backward',
    'ai_infra_qat_symmetric_per_tensor',
    'ai_infra_qat_symmetric_per_tensor_backward',
]
