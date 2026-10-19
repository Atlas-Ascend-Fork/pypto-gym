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
PyPTO RoPE Implementation

实现真正的 PyPTO kernel，而不是 fallback
使用 pypto.frontend.jit + pypto tensor operations

策略：
- Vision RoPE: 完全使用 PyPTO kernel
- Multimodal RoPE: 预处理使用 PyTorch，核心 rotate 使用 PyPTO kernel
"""

import logging
from typing import Tuple, List
import torch
from torch._dynamo import allow_in_graph
import pypto


@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128},
    pass_options={"cube_l1_reuse_setting": {-1: 4}},
)
def apply_rotary_pos_emb_vision_kernel(
    q: pypto.tensor(),
    k: pypto.tensor(),
    cos: pypto.tensor(),
    sin: pypto.tensor(),
    q_out: pypto.tensor(),
    k_out: pypto.tensor()
):
    """Vision RoPE kernel implementation"""
    rank = q.dim
    tile_shapes = [32 for _ in range(rank)]
    pypto.set_vec_tile_shapes(*tile_shapes)
    
    head_dim = q.shape[-1]
    
    cos_expanded = cos.unsqueeze(-2)
    sin_expanded = sin.unsqueeze(-2)

    q_half = head_dim // 2
    q1 = q[..., :q_half]
    q2 = q[..., q_half:]
    
    k1 = k[..., :q_half]
    k2 = k[..., q_half:]

    neg_q2 = pypto.neg(q2)
    neg_k2 = pypto.neg(k2)
    
    q_rotated = pypto.concat([neg_q2, q1], dim=-1)
    k_rotated = pypto.concat([neg_k2, k1], dim=-1)

    q_embed = pypto.add(pypto.mul(q, cos_expanded), pypto.mul(q_rotated, sin_expanded))
    k_embed = pypto.add(pypto.mul(k, cos_expanded), pypto.mul(k_rotated, sin_expanded))

    q_out[:] = q_embed
    k_out[:] = k_embed


@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128},
    pass_options={"cube_l1_reuse_setting": {-1: 4}},
)
def apply_rotary_pos_emb_kernel(
    q: pypto.tensor(),
    k: pypto.tensor(),
    cos: pypto.tensor(),
    sin: pypto.tensor(),
    q_out: pypto.tensor(),
    k_out: pypto.tensor()
):
    """通用 RoPE kernel implementation"""
    rank = q.dim
    tile_shapes = [32 for _ in range(rank)]
    pypto.set_vec_tile_shapes(*tile_shapes)

    head_dim = q.shape[-1]
    half_dim = head_dim // 2
    
    q1 = q[..., :half_dim]
    q2 = q[..., half_dim:]
    k1 = k[..., :half_dim]
    k2 = k[..., half_dim:]

    neg_q2 = pypto.neg(q2)
    neg_k2 = pypto.neg(k2)
    
    q_rotated = pypto.concat([neg_q2, q1], dim=-1)
    k_rotated = pypto.concat([neg_k2, k1], dim=-1)

    q_embed = pypto.add(pypto.mul(q, cos), pypto.mul(q_rotated, sin))
    k_embed = pypto.add(pypto.mul(k, cos), pypto.mul(k_rotated, sin))

    q_out[:] = q_embed
    k_out[:] = k_embed


@allow_in_graph
def apply_rotary_pos_emb_vision_pto_impl(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vision RoPE PyPTO implementation wrapper
    
    Args:
        q: query tensor, shape [seq_len, num_heads, head_dim]
        k: key tensor, shape [seq_len, num_heads, head_dim]
        cos: cosine tensor, shape [seq_len, head_dim // 2]
        sin: sine tensor, shape [seq_len, head_dim // 2]
    
    Returns:
        q_embed, k_embed: 旋转后的 query 和 key
    """
    q = q.contiguous()
    k = k.contiguous()
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    apply_rotary_pos_emb_vision_kernel(q, k, cos, sin, q_out, k_out)
    
    return q_out, k_out


@allow_in_graph
def apply_multimodal_rotary_pos_emb_pto_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: List[int],
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multimodal RoPE PyPTO implementation wrapper
    
    预处理 cos/sin 使用 PyTorch，核心 rotate 使用 PyPTO kernel
    
    Args:
        q: query tensor, shape [batch, num_heads, seq_len, head_dim]
        k: key tensor, shape [batch, num_kv_heads, seq_len, head_dim]
        cos: cosine tensor, shape [num_sections, batch, seq_len, head_dim]
        sin: sine tensor, shape [num_sections, batch, seq_len, head_dim]
        mrope_section: 多模态 RoPE section 列表，如 [16, 24, 24]
        unsqueeze_dim: unsqueeze 维度，默认为 1
    
    Returns:
        q_embed, k_embed: 旋转后的 query 和 key
    """
    mrope_section_expanded = mrope_section * 2
    
    cos_list = []
    sin_list = []
    start_idx = 0
    for i, section in enumerate(mrope_section_expanded):
        end_idx = start_idx + section
        cos_slice = cos[..., start_idx:end_idx]
        sin_slice = sin[..., start_idx:end_idx]
        
        cos_list.append(cos_slice[i % 3])
        sin_list.append(sin_slice[i % 3])
        
        start_idx = end_idx
    
    cos_merged = torch.cat(cos_list, dim=-1).unsqueeze(unsqueeze_dim).contiguous()
    sin_merged = torch.cat(sin_list, dim=-1).unsqueeze(unsqueeze_dim).contiguous()

    # GQA: q and k may have different head counts — split kernel calls
    cos_q = cos_merged.expand(-1, q.shape[1], -1, -1).contiguous()
    sin_q = sin_merged.expand(-1, q.shape[1], -1, -1).contiguous()
    cos_k = cos_merged.expand(-1, k.shape[1], -1, -1).contiguous()
    sin_k = sin_merged.expand(-1, k.shape[1], -1, -1).contiguous()

    q = q.contiguous()
    k = k.contiguous()
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    apply_rotary_pos_emb_kernel(q, q, cos_q, sin_q, q_out, q_out)  # q-only: dummy k
    apply_rotary_pos_emb_kernel(k, k, cos_k, sin_k, k_out, k_out)  # k-only: dummy q
    
    return q_out, k_out


if __name__ == "__main__":
    logging.info("=== PyPTO RoPE Implementation Test ===")
    
    import os
    import torch_npu
    
    _dev_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    device = f"npu:{_dev_id}"
    
    seq_len, num_heads, head_dim = 31, 16, 128
    q_vision = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float16, device=device)
    k_vision = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float16, device=device)
    cos_vision = torch.randn(seq_len, head_dim, dtype=torch.float16, device=device)
    sin_vision = torch.randn(seq_len, head_dim, dtype=torch.float16, device=device)
    
    try:
        q_embed_vision, k_embed_vision = \
            apply_rotary_pos_emb_vision_pto_impl(q_vision, k_vision, cos_vision, sin_vision)
        
        logging.info(f"Vision RoPE:")
        logging.info(f"  q_embed shape: {q_embed_vision.shape}")
        logging.info(f"  k_embed shape: {k_embed_vision.shape}")
        logging.info(f"  q_embed device: {q_embed_vision.device}")
        logging.info(f"[TEST_PASS] Vision RoPE PyPTO kernel works")
    except Exception as e:
        logging.info(f"[TEST_ERROR] Vision RoPE: {e}")
        import traceback
        traceback.print_exc()
    
    batch_size, num_heads, seq_len, head_dim = 1, 16, 31, 128
    q_multimodal = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    k_multimodal = torch.randn(batch_size, 2, seq_len, head_dim, dtype=torch.float16, device=device)
    cos_multimodal = torch.randn(3, batch_size, seq_len, head_dim, dtype=torch.float16, device=device)
    sin_multimodal = torch.randn(3, batch_size, seq_len, head_dim, dtype=torch.float16, device=device)
    mrope_section = [16, 24, 24]
    
    try:
        q_embed_multimodal, k_embed_multimodal = apply_multimodal_rotary_pos_emb_pto_impl(
            q_multimodal, k_multimodal, cos_multimodal, sin_multimodal, mrope_section
        )
        
        logging.info(f"\nMultimodal RoPE:")
        logging.info(f"  q_embed shape: {q_embed_multimodal.shape}")
        logging.info(f"  k_embed shape: {k_embed_multimodal.shape}")
        logging.info(f"  q_embed device: {q_embed_multimodal.device}")
        logging.info(f"[TEST_PASS] Multimodal RoPE PyPTO kernel works")
    except Exception as e:
        logging.info(f"[TEST_ERROR] Multimodal RoPE: {e}")
        import traceback
        traceback.print_exc()
