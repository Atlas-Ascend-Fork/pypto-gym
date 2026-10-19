#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Precision test for the Pangu-7B single-layer fused kernel (BSH KV cache).

Validates ``pangu_fused_layer_v2_bsh`` against a pure-PyTorch golden reference,
covering one full decoder layer: RMSNorm -> QKV GEMM (+bias) -> RoPE -> KV-cache
write -> GQA attention (online softmax) -> O GEMM (+bias) -> RMSNorm -> SwiGLU FFN.

Usage::

    # pytest
    pytest tests/ops/openpangu_v5_7b/test_pangu_fused_layer.py -v --forked

    # standalone
    python3 tests/ops/openpangu_v5_7b/test_pangu_fused_layer.py
"""

import math
import os
import sys
from pathlib import Path
from typing import Tuple
import logging
import torch
import pytest

# Add pypto-gym src to sys.path
_CUR = Path(__file__).resolve().parent
_SRC = _CUR.parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Model constants (must match DynamicFusedLayerConfigV2BSH defaults)
# ---------------------------------------------------------------------------
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 12800
QKV_SIZE = (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
KV_SIZE = NUM_KV_HEADS * HEAD_DIM
SCALE = 1.0 / (HEAD_DIM ** 0.5)
RMS_NORM_EPS = 1e-5
ROPE_THETA = 16000000.0


# ---------------------------------------------------------------------------
# Golden reference (pure PyTorch)
# ---------------------------------------------------------------------------
def _rms_norm(x, residual, weight, eps):
    new_residual = x.float() + residual.float()
    variance = new_residual.pow(2).mean(-1, keepdim=True)
    hidden_normed = new_residual * torch.rsqrt(variance + eps)
    return (hidden_normed * weight.float()).to(x.dtype), new_residual.to(x.dtype)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def fused_layer_golden(
    hidden_states, residual, cos, sin,
    key_cache, value_cache,
    input_ln_weight, post_ln_weight,
    qkv_weight, qkv_bias, o_weight, o_bias,
    gate_weight, up_weight, down_weight,
    actual_kv_len,
):
    """PyTorch golden reference for one fused decoder layer (BSH KV cache).

    ``cos``/``sin`` are [max_seq, head_dim]; the kernel indexes ``cos[prev_len]``
    where ``prev_len = actual_kv_len`` (matching the kernel's
    ``actual_kv_len_val = actual_kv_len[0] + 1; prev_len = actual_kv_len_val - 1``).
    """
    prev_len = int(actual_kv_len.item())

    # 1. Input RMSNorm
    hidden_normed, new_residual1 = _rms_norm(
        hidden_states, residual, input_ln_weight, RMS_NORM_EPS
    )

    # 2. QKV projection (+bias)
    qkv = torch.matmul(hidden_normed.float(), qkv_weight.t().float())
    qkv = qkv + qkv_bias.float()

    q = qkv[:, :, : NUM_HEADS * HEAD_DIM]
    k = qkv[:, :, NUM_HEADS * HEAD_DIM: NUM_HEADS * HEAD_DIM + KV_SIZE]
    v = qkv[:, :, NUM_HEADS * HEAD_DIM + KV_SIZE:]

    q = q.view(NUM_HEADS, HEAD_DIM).float()
    k = k.view(NUM_KV_HEADS, HEAD_DIM).float()

    # 3. RoPE (GPT-NeoX rotate_half, no deinterleave — matches NPU native op)
    cos_cur = cos[prev_len].float()
    sin_cur = sin[prev_len].float()
    q_embed = q * cos_cur + _rotate_half(q) * sin_cur
    k_embed = k * cos_cur + _rotate_half(k) * sin_cur

    # 4. Update KV cache (BSH)
    key_cache[prev_len] = k_embed.to(key_cache.dtype).view(KV_SIZE)
    value_cache[prev_len] = v.to(value_cache.dtype).view(KV_SIZE)

    # 5. GQA attention
    q_gqa = q_embed.view(NUM_KV_HEADS, NUM_HEADS // NUM_KV_HEADS, HEAD_DIM)
    k_cache_bnsd = key_cache[: prev_len + 1].view(-1, NUM_KV_HEADS, HEAD_DIM).permute(1, 0, 2)
    v_cache_bnsd = value_cache[: prev_len + 1].view(-1, NUM_KV_HEADS, HEAD_DIM).permute(1, 0, 2)

    scores = torch.matmul(q_gqa, k_cache_bnsd.transpose(-2, -1)) * SCALE
    attn_weights = torch.softmax(scores, dim=-1)
    attn_output = torch.matmul(attn_weights, v_cache_bnsd)

    # 6. O projection (+bias)
    attn_flat = attn_output.view(1, NUM_HEADS * HEAD_DIM)
    attn_proj = torch.matmul(attn_flat.float(), o_weight.t().float()) + o_bias.float()

    # 7. Post-attention RMSNorm
    hidden_normed2, new_residual2 = _rms_norm(
        attn_proj.to(hidden_states.dtype),
        new_residual1.to(hidden_states.dtype),
        post_ln_weight,
        RMS_NORM_EPS,
    )

    # 8. FFN (SwiGLU)
    gate = torch.matmul(hidden_normed2.float(), gate_weight.t().float())
    up = torch.matmul(hidden_normed2.float(), up_weight.t().float())
    activated = torch.nn.functional.silu(gate) * up
    output = torch.matmul(activated, down_weight.t().float())

    return output.to(hidden_states.dtype), new_residual2.to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Test fixture
# ---------------------------------------------------------------------------
def _make_inputs(device, dtype, actual_kv_len, max_kv_len):
    """Create random inputs and weights for a single decode step."""
    std_h = 1.0 / math.sqrt(HIDDEN_SIZE)
    std_w = 1.0 / math.sqrt(HEAD_DIM)
    std_i = 1.0 / math.sqrt(INTERMEDIATE_SIZE)

    hidden_states = torch.randn(1, 1, HIDDEN_SIZE, dtype=dtype, device=device) * std_h
    residual = torch.zeros(1, 1, HIDDEN_SIZE, dtype=dtype, device=device)

    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=device) / HEAD_DIM))
    t = torch.arange(max_kv_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)

    key_cache = torch.zeros(max_kv_len, KV_SIZE, dtype=dtype, device=device)
    value_cache = torch.zeros(max_kv_len, KV_SIZE, dtype=dtype, device=device)
    if actual_kv_len > 0:
        key_cache[:actual_kv_len] = torch.randn(actual_kv_len, KV_SIZE, dtype=dtype, device=device) * std_w
        value_cache[:actual_kv_len] = torch.randn(actual_kv_len, KV_SIZE, dtype=dtype, device=device) * std_w

    input_ln_weight = torch.ones(HIDDEN_SIZE, dtype=dtype, device=device)
    post_ln_weight = torch.ones(HIDDEN_SIZE, dtype=dtype, device=device)
    qkv_weight = torch.randn(QKV_SIZE, HIDDEN_SIZE, dtype=dtype, device=device) * std_h
    qkv_bias = torch.randn(QKV_SIZE, dtype=dtype, device=device) * 0.01
    o_weight = torch.randn(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, dtype=dtype, device=device) * std_h
    o_bias = torch.randn(HIDDEN_SIZE, dtype=dtype, device=device) * 0.01
    gate_weight = torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype=dtype, device=device) * std_i
    up_weight = torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype=dtype, device=device) * std_i
    down_weight = torch.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE, dtype=dtype, device=device) * std_i

    kv_len_tensor = torch.tensor([actual_kv_len], dtype=torch.int64, device=device)

    return {
        "hidden_states": hidden_states,
        "residual": residual,
        "cos": cos,
        "sin": sin,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "input_ln_weight": input_ln_weight,
        "post_ln_weight": post_ln_weight,
        "qkv_weight": qkv_weight,
        "qkv_bias": qkv_bias,
        "o_weight": o_weight,
        "o_bias": o_bias,
        "gate_weight": gate_weight,
        "up_weight": up_weight,
        "down_weight": down_weight,
        "actual_kv_len": kv_len_tensor,
    }


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@pytest.mark.skip("test")
@pytest.mark.soc("910")
def test_pangu_fused_layer_decode_step():
    """Single decode step: kernel output vs golden reference."""
    import torch_npu
    from ops.pypto_tensor.openpangu_v5_7b.pangu_fused_layer_dynamic_v2_bsh import (
        PanguFusedLayerV2BSHModule,
    )

    device = "npu:0"
    dtype = torch.bfloat16
    actual_kv_len = 128
    max_kv_len = 256

    inputs = _make_inputs(device, dtype, actual_kv_len, max_kv_len)

    # Golden
    kc_golden = inputs["key_cache"].clone()
    vc_golden = inputs["value_cache"].clone()
    ref_output, ref_residual = fused_layer_golden(
        inputs["hidden_states"], inputs["residual"],
        inputs["cos"], inputs["sin"],
        kc_golden, vc_golden,
        inputs["input_ln_weight"], inputs["post_ln_weight"],
        inputs["qkv_weight"], inputs["qkv_bias"],
        inputs["o_weight"], inputs["o_bias"],
        inputs["gate_weight"], inputs["up_weight"], inputs["down_weight"],
        inputs["actual_kv_len"],
    )

    # Kernel
    module = PanguFusedLayerV2BSHModule()
    kc_kernel = inputs["key_cache"].clone()
    vc_kernel = inputs["value_cache"].clone()
    output, new_residual = module(
        inputs["hidden_states"], inputs["residual"],
        inputs["cos"], inputs["sin"],
        kc_kernel, vc_kernel,
        inputs["input_ln_weight"], inputs["post_ln_weight"],
        inputs["qkv_weight"], inputs["qkv_bias"],
        inputs["o_weight"], inputs["o_bias"],
        inputs["gate_weight"], inputs["up_weight"], inputs["down_weight"],
        inputs["actual_kv_len"],
    )
    torch.npu.synchronize()

    output_diff = (output - ref_output).abs().max().item()
    residual_diff = (new_residual - ref_residual).abs().max().item()
    assert output_diff < 0.5, f"output diff {output_diff:.4f} exceeds tolerance 0.5"
    assert residual_diff < 0.5, f"residual diff {residual_diff:.4f} exceeds tolerance 0.5"


if __name__ == "__main__":
    test_pangu_fused_layer_decode_step()
