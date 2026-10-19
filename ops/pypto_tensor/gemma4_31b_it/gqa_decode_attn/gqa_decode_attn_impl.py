# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""GQA decode attention with KV head averaging for Gemma-4-31B.

Groups of 4 adjacent KV heads are averaged before the kernel call, reducing
K/V memory bandwidth by 75%:
    k_reduced[i] = mean(k[4i], k[4i+1], k[4i+2], k[4i+3])   for i in 0..3
    v_reduced[i] = mean(v[4i], v[4i+1], v[4i+2], v[4i+3])

Effective dimensions after averaging:
    Nkv_orig=16 -> Nkv=4, GROUPS=2 -> GROUPS=8, D=256 unchanged.

Inputs (to kernel):
    q       [Nq=32, D=256]                    bf16 -> reshaped to [Nkv=4, GROUPS=8, D=256]
    k_full  [Nkv=4, Skv_padded, D=256]        bf16  (already averaged)
    v_full  [Nkv=4, Skv_padded, D=256]        bf16  (already averaged)
    mask    [Nkv=4, GROUPS=8, Skv_padded]      fp32
    out     [Nq=32, D=256]                    bf16 <- reshaped from [Nkv=4, GROUPS=8, D=256]

SRAM budget (Ascend 910B, pypto-9.0.0-beta.1: 192KB UB per core):
    K tile: Nkv=4 * S2_TILE * D=256 * 2B = 4 * 64 * 256 * 2 = 128KB
    V tile: same = 128KB
    Each K or V tile fits within 192KB UB.

Cube tiles:
    QK^T: [4,4], [128,128], [64,64]   -- batch=Nkv(4), M=GROUPS(8), N=S2_TILE, K=D(256 in 2x128)
    PV:   [4,4], [64,64], [128,128]   -- batch=Nkv(4), M=GROUPS(8), N=D(256 in 2x128), K=S2_TILE
"""

import os
import sys
import contextlib
import tempfile

import pypto
import torch
import torch.nn.functional as F
from torch._dynamo import allow_in_graph

# Original Gemma-4-31B architecture
Nq = 32
Nkv_orig = 16

# KV head averaging: groups of 4
KV_GROUP_SIZE = 4   # number of adjacent KV heads to average
Nkv = Nkv_orig // KV_GROUP_SIZE  # 4
GROUPS = Nq // Nkv  # 8
D = 256
SCALE = 1.0  # Model uses QK RMSNorm which absorbs scaling
W = 1024  # sliding window size

# S2_TILE: 256 — Nkv=4 * 256 * 256 * 2B = 512KB per K/V tile
S2_TILE = 256

# Dynamic dimension for KV sequence length
Skv_dyn = pypto.frontend.dynamic("Skv")

# Build directory isolation
_BUILD_DIR = os.environ.get(
    "GEMMA4_PTO_BUILD_DIR",
    os.path.join(tempfile.gettempdir(), "gemma4_pto_build", "gqa_decode"),
)
os.makedirs(_BUILD_DIR, exist_ok=True)


def _local_softmax(sij, mask, s2_start, s2_valid):
    pypto.set_vec_tile_shapes(Nkv, GROUPS, S2_TILE)
    mask_tile = pypto.view(mask, [Nkv, GROUPS, S2_TILE], [0, 0, s2_start],
                           valid_shape=[Nkv, GROUPS, s2_valid])
    sij_scaled = pypto.add(pypto.mul(sij, SCALE), mask_tile)
    m_ij = pypto.amax(sij_scaled, -1, keepdim=True)
    p_ij = pypto.exp(pypto.sub(sij_scaled, m_ij))
    l_ij = pypto.sum(p_ij, -1, keepdim=True)
    return m_ij, p_ij, l_ij


def _compensated_pv(p_ij, v_tile):
    p_hi = pypto.tensor([Nkv, GROUPS, S2_TILE], pypto.DT_BF16, "p_hi")
    p_hi[:] = pypto.cast(p_ij, pypto.DT_BF16)
    p_lo = pypto.tensor([Nkv, GROUPS, S2_TILE], pypto.DT_BF16, "p_lo")
    p_lo[:] = pypto.cast(pypto.sub(p_ij, pypto.cast(p_hi, pypto.DT_FP32)), pypto.DT_BF16)
    pypto.set_cube_tile_shapes([4, 4], [64, 64], [128, 128])
    o_hi = pypto.matmul(p_hi, v_tile, pypto.DT_FP32)
    o_lo = pypto.matmul(p_lo, v_tile, pypto.DT_FP32)
    pypto.set_vec_tile_shapes(Nkv, GROUPS, D)
    return pypto.add(o_hi, o_lo)


@pypto.frontend.jit(
    runtime_options={"device_sched_mode": 1, "run_mode": pypto.RunMode.NPU, "stitch_function_max_num": 512},
    pass_options={"cube_l1_reuse_setting": {0: 8}, "vec_nbuffer_setting": {-1: 8}},
)
def gemma4_decode_attn_gqa(
    q: pypto.Tensor([Nq, D], pypto.DT_BF16),
    k_full: pypto.Tensor([Nkv, Skv_dyn, D], pypto.DT_BF16),
    v_full: pypto.Tensor([Nkv, Skv_dyn, D], pypto.DT_BF16),
    mask: pypto.Tensor([Nkv, GROUPS, Skv_dyn], pypto.DT_FP32),
    out: pypto.Tensor([Nq, D], pypto.DT_BF16),
):
    """GQA decode attention with online softmax (Nkv=4, GROUPS=8)."""
    Skv = k_full.shape[1]
    s2_loop = (Skv + S2_TILE - 1) // S2_TILE

    pypto.set_vec_tile_shapes(Nkv, GROUPS, D)
    q_3d = pypto.tensor([Nkv, GROUPS, D], pypto.DT_BF16, "q_3d")
    q_3d[:] = pypto.reshape(q, [Nkv, GROUPS, D])

    # Online softmax accumulators
    oi = pypto.tensor([Nkv, GROUPS, D], pypto.DT_FP32, "oi")
    li = pypto.tensor([Nkv, GROUPS, 1], pypto.DT_FP32, "li")
    mi = pypto.tensor([Nkv, GROUPS, 1], pypto.DT_FP32, "mi")

    for s2_idx in pypto.loop(s2_loop, name="LOOP_S2", idx_name="s2_idx", unroll_list=[2, 1]):
        s2_start = s2_idx * S2_TILE
        s2_valid = pypto.min(s2_start + S2_TILE, Skv) - s2_start

        pypto.set_vec_tile_shapes(Nkv, S2_TILE, D)
        k_tile = pypto.tensor([Nkv, S2_TILE, D], pypto.DT_BF16, "k_tile")
        k_tile[:] = pypto.view(k_full, [Nkv, S2_TILE, D], [0, s2_start, 0],
                               valid_shape=[Nkv, s2_valid, D])
        v_tile = pypto.tensor([Nkv, S2_TILE, D], pypto.DT_BF16, "v_tile")
        v_tile[:] = pypto.view(v_full, [Nkv, S2_TILE, D], [0, s2_start, 0],
                               valid_shape=[Nkv, s2_valid, D])

        pypto.set_cube_tile_shapes([4, 4], [128, 128], [64, 64])
        sij = pypto.matmul(q_3d, k_tile, pypto.DT_FP32, b_trans=True)

        m_ij, p_ij, l_ij = _local_softmax(sij, mask, s2_start, s2_valid)
        o_ij = _compensated_pv(p_ij, v_tile)

        pypto.set_vec_tile_shapes(Nkv, GROUPS, D)
        if pypto.is_loop_begin(s2_idx):
            if pypto.is_loop_end(s2_idx):
                out[:] = pypto.reshape(
                    pypto.cast(pypto.div(o_ij, l_ij, precision_type=pypto.PrecisionType.INTRINSIC),
                               pypto.DT_BF16),
                    [Nq, D])
            else:
                oi[:] = o_ij
            li[:] = l_ij
            mi[:] = m_ij
        else:
            mi_new = pypto.maximum(mi, m_ij)
            alpha = pypto.exp(pypto.sub(mi, mi_new))
            beta = pypto.exp(pypto.sub(m_ij, mi_new))
            li_new = pypto.add(pypto.mul(alpha, li), pypto.mul(beta, l_ij))
            oi_new = pypto.add(pypto.mul(oi, alpha), pypto.mul(o_ij, beta))
            if pypto.is_loop_end(s2_idx):
                out[:] = pypto.reshape(
                    pypto.cast(pypto.div(oi_new, li_new, precision_type=pypto.PrecisionType.INTRINSIC),
                               pypto.DT_BF16),
                    [Nq, D])
            else:
                oi[:] = oi_new
            li[:] = li_new
            mi[:] = mi_new


def _reduce_kv_heads(key_states, value_states):
    """Average groups of 4 adjacent KV heads: [B, 16, Skv, D] -> [B, 4, Skv, D].

    k_reduced[i] = mean(k[4i], k[4i+1], k[4i+2], k[4i+3])
    v_reduced[i] = mean(v[4i], v[4i+1], v[4i+2], v[4i+3])
    """
    k_reduced = (key_states[:, 0::4] + key_states[:, 1::4] + key_states[:, 2::4] + key_states[:, 3::4]).float() / 4.0
    k_reduced = k_reduced.to(key_states.dtype)
    v_reduced = (value_states[:, 0::4] + value_states[:, 1::4] + \
                 value_states[:, 2::4] + value_states[:, 3::4]).float() / 4.0
    v_reduced = v_reduced.to(value_states.dtype)
    return k_reduced, v_reduced


def _torch_fallback(query_states, key_states, value_states, attention_mask, scaling):
    """Pure PyTorch fallback for prefill (Sq > 1), with KV head averaging."""
    B, Nq_l, Sq, D_l = query_states.shape

    # Average KV heads first
    k_reduced, v_reduced = _reduce_kv_heads(key_states, value_states)
    _, Nkv_l, Skv, _ = k_reduced.shape
    GROUPS_l = Nq_l // Nkv_l

    k = k_reduced[:, :, None, :, :].expand(B, Nkv_l, GROUPS_l, Skv, D_l)
    k = k.reshape(B, Nq_l, Skv, D_l)
    v = v_reduced[:, :, None, :, :].expand(B, Nkv_l, GROUPS_l, Skv, D_l)
    v = v.reshape(B, Nq_l, Skv, D_l)

    scores = torch.matmul(query_states.float(), k.float().transpose(-2, -1)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask.float()
    attn_weights = F.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, v.float())
    return out.transpose(1, 2).to(torch.bfloat16)


@allow_in_graph
def gqa_decode_attn_wrapper(query_states, key_states, value_states, attention_mask, scaling, layer_kind):
    """GQA decode attention forward with KV head averaging.

    Averages groups of 4 adjacent KV heads (16->4) before running the GQA kernel.
    Reduces K/V memory bandwidth by 75% at the cost of some precision.

    Args:
        query_states  : [B=1, Nq=32, Sq, D=256] bf16
        key_states    : [B=1, Nkv=16, Skv, D=256] bf16  (original 16 KV heads)
        value_states  : [B=1, Nkv=16, Skv, D=256] bf16  (original 16 KV heads)
        attention_mask : [B=1, 1, Sq, Skv] fp32 (0 or -inf)
        scaling       : float (1.0)
        layer_kind    : str ("global" or "local")

    Returns:
        [B=1, Sq, Nq=32, D=256] bf16
    """
    B_in, Nq_in, Sq, D_in = query_states.shape

    # Prefill fallback
    if Sq != 1:
        return _torch_fallback(query_states, key_states, value_states, attention_mask, scaling)

    assert B_in == 1, f"GQA kernel expects B=1, got {B_in}"
    assert Nq_in == Nq
    assert D_in == D

    _, Nkv_in, Skv, _ = key_states.shape
    assert Nkv_in == Nkv_orig, f"Expected {Nkv_orig} KV heads, got {Nkv_in}"
    device = query_states.device

    # Sliding window: slice for local layers with Skv > W
    if layer_kind == "local" and Skv > W:
        key_states = key_states[:, :, Skv - W:, :]
        value_states = value_states[:, :, Skv - W:, :]
        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, Skv - W:]
        Skv = W

    # Average groups of 4 KV heads: [1, 16, Skv, D] -> [1, 4, Skv, D]
    k_reduced, v_reduced = _reduce_kv_heads(key_states, value_states)

    # Pad Skv to S2_TILE multiple
    Skv_padded = ((Skv + S2_TILE - 1) // S2_TILE) * S2_TILE
    if Skv_padded > Skv:
        pad_k = torch.zeros(1, Nkv, Skv_padded - Skv, D, dtype=torch.bfloat16, device=device)
        k_padded = torch.cat([k_reduced, pad_k], dim=2)
        pad_v = torch.zeros(1, Nkv, Skv_padded - Skv, D, dtype=torch.bfloat16, device=device)
        v_padded = torch.cat([v_reduced, pad_v], dim=2)
    else:
        k_padded = k_reduced.contiguous()
        v_padded = v_reduced.contiguous()

    if attention_mask is not None:
        mask_base = attention_mask[0, 0, 0, :].float()
    else:
        mask_base = torch.zeros(Skv, dtype=torch.float32, device=device)

    if Skv_padded > Skv:
        mask_pad = torch.full((Skv_padded - Skv,), -1e30, dtype=torch.float32, device=device)
        mask_full = torch.cat([mask_base, mask_pad])
    else:
        mask_full = mask_base

    mask_3d = mask_full.view(1, 1, Skv_padded).expand(Nkv, GROUPS, Skv_padded).contiguous()

    # Squeeze batch dim
    q_kernel = query_states[0, :, 0, :].contiguous()
    k_kernel = k_padded[0].contiguous()
    v_kernel = v_padded[0].contiguous()

    out_kernel = torch.empty(Nq, D, dtype=torch.bfloat16, device=device)

    gemma4_decode_attn_gqa(q_kernel, k_kernel, v_kernel, mask_3d, out_kernel)

    result = out_kernel.view(1, Nq, 1, D).transpose(1, 2)
    return result
