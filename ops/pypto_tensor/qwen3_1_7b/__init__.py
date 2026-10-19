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
Qwen3-1.7B PyPTO fused kernel module.

Switch variable: USE_PTO_ROPE (bool, default False)
  - True: Q/K RMSNorm + RoPE fused via PyPTO kernel
  - False: original PyTorch q_norm/k_norm + apply_rotary_pos_emb
"""
import torch

USE_PTO_ROPE = False


def qk_rope_wrapper(q_proj_out, k_proj_out, cos, sin, q_norm_weight, k_norm_weight,
                     q_num_heads, kv_num_heads, head_dim):
    """PTO fused Q/K RMSNorm + RoPE — dispatched from Qwen3Attention.forward.

    Shapes:
      q_proj_out:    [B, S, q_num_heads * head_dim]
      k_proj_out:    [B, S, kv_num_heads * head_dim]
      cos, sin:      [B, S, head_dim]
      q_norm_weight: [head_dim]
      k_norm_weight: [head_dim]

    Returns:
      query_states:  [B, q_num_heads, S, head_dim]    (transposed for attention)
      key_states:    [B, kv_num_heads, S, head_dim]
    """
    from .rope.rrms_norm_rope_impl import qwen3_qk_rope_q, qwen3_qk_rope_k

    batch, seq_len = q_proj_out.shape[0], q_proj_out.shape[1]
    orig_dtype = q_proj_out.dtype  # may be float16; kernel expects bfloat16

    # Flatten batch*seq_len into the dynamic sequence dim expected by the PTO kernel
    q_3d = q_proj_out.view(batch, seq_len, q_num_heads, head_dim).reshape(-1, q_num_heads, head_dim)
    k_3d = k_proj_out.view(batch, seq_len, kv_num_heads, head_dim).reshape(-1, kv_num_heads, head_dim)

    cos_2d = cos.reshape(-1, head_dim).to(torch.bfloat16)
    sin_2d = sin.reshape(-1, head_dim).to(torch.bfloat16)

    out_q_bf16 = torch.empty(q_3d.shape, dtype=torch.bfloat16, device=q_3d.device)
    out_k_bf16 = torch.empty(k_3d.shape, dtype=torch.bfloat16, device=k_3d.device)
    qwen3_qk_rope_q(q_3d.to(torch.bfloat16), cos_2d, sin_2d, q_norm_weight.to(torch.bfloat16), out_q_bf16)
    qwen3_qk_rope_k(k_3d.to(torch.bfloat16), cos_2d, sin_2d, k_norm_weight.to(torch.bfloat16), out_k_bf16)

    query_states = out_q_bf16.to(orig_dtype).reshape(batch, seq_len, q_num_heads, head_dim).transpose(1, 2).contiguous()
    key_states = out_k_bf16.to(orig_dtype).reshape(batch, seq_len, kv_num_heads, head_dim).transpose(1, 2).contiguous()

    return query_states, key_states
