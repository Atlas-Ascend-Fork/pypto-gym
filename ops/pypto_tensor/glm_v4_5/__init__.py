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
GLM V4.5 PyPTO 融合算子库

实际集成的算子：
- Attention (原生): IFA flash attention + RMSNorm bias + RoPE
- Attention (融合版): RMSNorm bias + RoPE + attention 融合 kernel
- Attention (预量化版): RMSNorm bias + RoPE + 量化预处理的 attention 前处理
- FFN Common Interface: SwiGLU + 对称量化 + 动态反量化
- FFN Shared Expert (量化版): 共享专家 FFN + SwiGLU + 量化
- Gate: MoE 专家选择的门控计算
- MoE Fusion: MoE 融合 kernel (gate + shared expert + routed experts)
- Select Experts: MoE 专家选择与 token 分发 kernel

融合范围：
- RMSNorm bias + RoPE + Attention 的端到端融合
- Gate + Shared Expert + Routed Experts 的 MoE 融合
"""

from .glm_attention_impl import (
    attention as glm_attention,
    attention_for_950 as glm_attention_for_950,
    ifa_func_kernel as glm_ifa_func_kernel,
    IfaTileShapeConfig,
    IfaConfig,
)
from .glm_attention_fusion_impl import (
    attention as glm_attention_fusion,
    ifa_func_kernel as glm_ifa_fusion_kernel,
    AttentionTileConfig,
    AttentionConfig,
)
from .glm_attention_pre_quant_impl import (
    attention_pre_quant,
)
from .glm_ffn_common_interface import (
    symmetric_quantization_per_token,
    dequant_dynamic,
    swiglu,
)
from .glm_ffn_shared_expert_quant_impl import (
    ffn_shared_expert_quant,
)
from .glm_gate_impl import (
    gate,
)
from .glm_moe_fusion_impl import (
    moe_fusion,
    moe_fusion_pto,
)
from .glm_select_experts_impl import (
    select_experts,
)

__all__ = [
    # Attention (原生)
    'glm_attention',
    'glm_attention_for_950',
    'glm_ifa_func_kernel',
    'IfaTileShapeConfig',
    'IfaConfig',
    # Attention (融合)
    'glm_attention_fusion',
    'glm_ifa_fusion_kernel',
    'AttentionTileConfig',
    'AttentionConfig',
    # Attention (预量化)
    'attention_pre_quant',
    # FFN Common Interface
    'symmetric_quantization_per_token',
    'dequant_dynamic',
    'swiglu',
    # FFN Shared Expert
    'ffn_shared_expert_quant',
    # Gate
    'gate',
    # MoE Fusion
    'moe_fusion',
    'moe_fusion_pto',
    # Select Experts
    'select_experts',
]
