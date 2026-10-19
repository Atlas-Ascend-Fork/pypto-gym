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
DeepSeek-V2-Lite-Chat PyPTO Kernels Module
全局开关和算子导入管理（sys.modules注入）
"""

import sys
import torch


# ===== 导入算子实现（触发torch.library注册） =====
from .rms_norm.rms_norm_impl import (
    rms_norm_wrapper,
    rms_norm_pypto,
    pyptolib,  # 导出torch.library对象（RoPE共用）
)

from .rope.rope_impl import (
    apply_rotary_pos_emb_wrapper,
    apply_rotary_pos_emb_pto,
)

# MLA Prolog 混合优化版本（移除第一个matmul）
from .mla_prolog import (
    mla_prolog_wrapper,
    mla_prolog_wrapper,
    USE_PTO_MLA_PROLOG as _USE_PTO_MLA_PROLOG,
)

# 兼容别名（供modeling调用）
mla_mla_prolog_v2 = mla_prolog_wrapper  # 兼容现有代码


# ===== 全局开关 =====
USE_PTO = False  # 主开关
USE_PTO_RMS_NORM = False  # RMSNorm单独开关
USE_PTO_ROPE = False  # RoPE单独开关
USE_PTO_MLA_PROLOG = False  # KV融合算子单独开关
USE_ACL_GRAPH = False  # aclgraph模式开关


# ===== 算子接口（供modeling调用） =====

def rms_norm(hidden_states, weight, eps=1e-6):
    """
    RMSNorm接口：根据USE_PTO_RMS_NORM开关选择实现
    
    Args:
        hidden_states: [batch, seq_len, hidden_size]
        weight: [hidden_size]
        eps: float
    
    Returns:
        output: [batch, seq_len, hidden_size]
    """
    if USE_PTO_RMS_NORM:
        # 使用torch.ops.pypto调用（支持aclgraph）
        return torch.ops.pypto.rms_norm(hidden_states, weight, eps)
    else:
        # Fallback: 原始torch实现
        input_dtype = hidden_states.dtype
        hidden_states_fp32 = hidden_states.to(torch.float32)
        variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
        hidden_states_normed = hidden_states_fp32 * torch.rsqrt(variance + eps)
        return (weight * hidden_states_normed).to(input_dtype)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """
    RoPE接口：根据USE_PTO_ROPE开关选择实现
    
    Args:
        q: [bsz, num_heads, seq_len, head_dim]
        k: [bsz, num_heads, seq_len, head_dim] (MLA: k只有1个head)
        cos: [seq_len, head_dim] (cached)
        sin: [seq_len, head_dim] (cached)
        position_ids: [bsz, seq_len]
        unsqueeze_dim: int (default 1)
    
    Returns:
        q_embed: [bsz, num_heads, seq_len, head_dim]
        k_embed: [bsz, num_heads, seq_len, head_dim]
    """
    if USE_PTO_ROPE:
        # 使用torch.ops.pypto调用（支持aclgraph）
        return torch.ops.pypto.apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim)
    else:
        # Fallback: 原始torch实现（DeepSeek特殊实现）
        # 索引cos/sin + unsqueeze
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        
        # q重组
        b, h, s, d = q.shape
        q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        
        # k重组
        b, h, s, d = k.shape
        k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        
        # rotate_half
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)
        
        # 应用旋转
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        
        return q_embed, k_embed


def mla_prolog(hidden_states, kv_a_weight, kv_b_weight, ln_weight, eps, cos, sin):
    """
    KV Prolog接口：根据USE_PTO_MLA_PROLOG开关选择实现
    
    Args:
        hidden_states: [bsz, seq_len, hidden_size]
        kv_a_weight: [kv_lora_rank + rope_dim, hidden_size]
        kv_b_weight: [num_heads * 256, kv_lora_rank]
        ln_weight: [kv_lora_rank]
        eps: float
        cos: [seq_len, rope_dim]
        sin: [seq_len, rope_dim]
    
    Returns:
        k_nope: [bsz, num_heads, seq_len, 128]
        value: [bsz, num_heads, seq_len, 128]
        k_pe: [bsz, 1, seq_len, rope_dim]
    """
    if USE_PTO_MLA_PROLOG:
        # 使用torch.ops.pypto调用（支持aclgraph）
        return torch.ops.pypto.mla_prolog(
            hidden_states, kv_a_weight, kv_b_weight, ln_weight, cos, sin
        )
    else:
        # Fallback: 原始torch实现（暂时使用wrapper）
        return mla_prolog_wrapper(
            hidden_states, kv_a_weight, kv_b_weight, ln_weight, eps, cos, sin
        )


# ===== 模块导出 =====
__all__ = [
    "USE_PTO",
    "USE_PTO_RMS_NORM",
    "USE_PTO_ROPE",
    "USE_PTO_MLA_PROLOG",
    "USE_ACL_GRAPH",
    "rms_norm",
    "apply_rotary_pos_emb",
    "mla_prolog",
    "mla_mla_prolog_v2",
    "rms_norm_wrapper",
    "rms_norm_pypto",
    "apply_rotary_pos_emb_wrapper",
    "apply_rotary_pos_emb_pto",
    "mla_prolog_wrapper",
    "mla_prolog_pypto",
    "pyptolib",
]