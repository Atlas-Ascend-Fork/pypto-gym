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
GutenOCR-3B PyPTO 融合算子库

实际集成的算子：
- SwiGLU MLP (融合版): gate_proj + up_proj + down_proj 三合一融合 kernel
- MRoPE (Multimodal Rotary Position Embedding): temporal + height + width 三维位置编码
- RMSNorm (BF16 优化版): pypto.rms_norm 融合实现

融合范围：
- SwiGLU MLP: gate_proj(x) + SiLU + up_proj(x) + element-wise mul + down_proj(hidden)
- MRoPE: concat(temporal, height, width) + cos/sin + rotation + concat output
- RMSNorm: mean(x^2) + rsqrt(mean+eps) + x * rsqrt * weight

推荐配置：
- SwiGLU MLP: 所有 batch 推荐启用 (唯一稳定有效算子, 端到端 +3%~+16%)
- MRoPE: 仅 Batch <= 4 推荐启用 (固化开销问题)
- RMSNorm: 不推荐启用 (固化开销抵消优化)
"""

from .swiglu_mlp import swiglu_mlp_fused, swiglu_mlp_fused_static
from .mrope import mrope_pto_correct, mrope_torch_fallback
from .rms_norm import rms_norm_pto_native

USE_PTO_SWIGLU_MLP = False
USE_PTO_MROPE = False
USE_PTO_RMS_NORM = False


def rms_norm_wrapper(hidden_states, weight, eps=1e-6):
    """Wrapper for RMSNorm PTO call from modeling code."""
    return rms_norm_pto_native(hidden_states, weight, eps)


def swiglu_mlp_wrapper(mlp_module, hidden_states):
    """
    Bridge: Qwen2MLP → swiglu_mlp_fused / swiglu_mlp_fused_static kernel.

    Args:
        mlp_module: nn.Module with .gate_proj, .up_proj, .down_proj (nn.Linear)
        hidden_states: [batch_size, seq_len, hidden_size] or [batch_size, hidden_size]
    Returns:
        output: same shape as hidden_states
    """
    import torch

    # Flatten to [batch*seq, hidden_size]
    orig_shape = hidden_states.shape
    if hidden_states.dim() == 3:
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
    else:
        x = hidden_states

    batch_size = x.shape[0]

    gate_weight = mlp_module.gate_proj.weight.data.T.contiguous()
    up_weight = mlp_module.up_proj.weight.data.T.contiguous()
    down_weight = mlp_module.down_proj.weight.data.T.contiguous()

    # Handle optional bias
    hidden_size = mlp_module.hidden_size
    inter_size = mlp_module.intermediate_size

    def _zeros(shape):
        return torch.zeros(shape, dtype=x.dtype, device=x.device)
    gate_bias = (mlp_module.gate_proj.bias if mlp_module.gate_proj.bias is not None
                 else _zeros(inter_size))
    up_bias = (mlp_module.up_proj.bias if mlp_module.up_proj.bias is not None
               else _zeros(inter_size))
    down_bias = (mlp_module.down_proj.bias if mlp_module.down_proj.bias is not None
                 else _zeros(hidden_size))

    try:
        result = torch.empty(batch_size, hidden_size, dtype=x.dtype, device=x.device)
        swiglu_mlp_fused(x, gate_weight, gate_bias, up_weight, up_bias,
                         down_weight, down_bias, result)
        return result.reshape(orig_shape)
    except Exception:
        # PyPTO kernel unavailable — fall back to torch path
        gate = torch.nn.functional.silu(torch.nn.functional.linear(x, gate_weight.T, gate_bias))
        up = torch.nn.functional.linear(x, up_weight.T, up_bias)
        down = torch.nn.functional.linear(gate * up, down_weight.T, down_bias)
        return down.reshape(orig_shape)


__all__ = [
    'USE_PTO_SWIGLU_MLP',
    'USE_PTO_MROPE',
    'USE_PTO_RMS_NORM',
    'rms_norm_wrapper',
    'swiglu_mlp_wrapper',
    'swiglu_mlp_fused',
    'swiglu_mlp_fused_static',
    'mrope_pto_correct',
    'mrope_torch_fallback',
    'rms_norm_pto_native',
]
