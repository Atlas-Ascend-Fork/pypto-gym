# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent pure-torch CPU golden for ``mega_recurrent_kda_replayssm``.

The golden fixes the public dtype boundaries:

* projections are FP32 GEMMs rounded to the declared output dtype (BF16; the
  first-stage decay projection round-trips through BF16);
* causal K=4 Conv1D + SiLU runs with a BF16 history rounding each step and a
  ``(p0 + p1) + (p2 + p3)`` accumulation order;
* Q/K are L2-normalised with eps ``1e-6``;
* the delta-rule recurrence runs in FP32 over the ``S=8`` proposal tokens;
* RMSNorm + sigmoid gate + output projection close the fragment.

Reference: ``tests/ops/kimi_linear_48b_a3b`` (KDA goldens).
"""

from __future__ import annotations

__all__ = [
    "HEAD_DIM", "NUM_HEADS", "HIDDEN", "SEQ_LEN", "CONV_KERNEL",
    "linear_reference", "run_causal_conv1d_silu", "run_golden",
]

import torch

HEAD_DIM = 128
NUM_HEADS = 6
HIDDEN = 7168
SEQ_LEN = 8
CONV_KERNEL = 4


def linear_reference(value: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """``value @ weight^T`` accumulated in FP32, rounded to ``out_dtype``."""
    return torch.matmul(value.float(), weight.float().transpose(-1, -2)).to(out_dtype)


def run_causal_conv1d_silu(raw_qkv: torch.Tensor, conv_weight: torch.Tensor, conv_state: torch.Tensor,
                           conv_state_indices: torch.Tensor, conv_num_accepted: torch.Tensor) -> torch.Tensor:
    """Pooled K=4 causal Conv1D + SiLU with the model-native speculative cache.

    ``conv_weight`` is the canonical ``[CONV_KERNEL, 3*P]`` layout. ``raw_qkv`` is
    ``[B, S, 3*P]``. ``conv_state`` is updated in place (``conv_state_indices == 0``
    is the immutable null slot and leaves both output and state untouched).
    """
    batch, seq, channels = map(int, raw_qkv.shape)
    if tuple(conv_weight.shape) != (CONV_KERNEL, channels):
        raise ValueError("conv_weight must be [4, 3*P]")
    result = torch.zeros_like(raw_qkv, dtype=torch.bfloat16)
    original_state = conv_state.clone()
    filter_fp32 = conv_weight.float()
    for b in range(batch):
        cache_index = int(conv_state_indices[b])
        if cache_index == 0:
            continue
        accepted = int(conv_num_accepted[b])
        offset = max(accepted - 1, 0)
        stop = offset + CONV_KERNEL - 1
        history = original_state[cache_index, offset:stop].float().clone()
        for token in range(seq):
            current = raw_qkv[b, token].float()
            window = torch.cat((history, current.unsqueeze(0)), dim=0)
            products = window * filter_fp32
            convolved = products[0] + products[1] + (products[2] + products[3])
            result[b, token] = torch.nn.functional.silu(convolved).to(torch.bfloat16)
            history = torch.cat((history[1:], current.to(torch.bfloat16).float().unsqueeze(0)), dim=0)
        conv_state[cache_index, :2].copy_(original_state[cache_index, offset + 1:stop])
        conv_state[cache_index, 2:2 + seq].copy_(raw_qkv[b])
    return result


def canonical_conv_weight(conv_weight: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """Normalise any accepted Conv layout to ``[CONV_KERNEL, 3*P]``."""
    projection = num_heads * head_dim
    if tuple(conv_weight.shape) == (CONV_KERNEL, 3 * projection):
        return conv_weight
    if tuple(conv_weight.shape) == (3, num_heads, CONV_KERNEL, head_dim):
        return conv_weight.permute(2, 0, 1, 3).reshape(CONV_KERNEL, 3 * projection).contiguous()
    if tuple(conv_weight.shape) == (3, CONV_KERNEL, head_dim):
        return (conv_weight[:, None]
                .expand(3, num_heads, CONV_KERNEL, head_dim)
                .permute(2, 0, 1, 3).reshape(CONV_KERNEL, 3 * projection).contiguous())
    raise ValueError("conv_weight must be [4,3*P], [3,N,4,D] or [3,4,D]")


def _golden_front(hidden_states, qkv_weight, conv_weight, conv_state,
                  conv_state_indices, conv_num_accepted, shape):
    """Projections + causal Conv1D + Q/K L2 norm (dtype contract: BF16 boundaries)."""
    b, s, n, d = shape
    raw_qkv = linear_reference(hidden_states, qkv_weight, torch.float32)
    convolved = run_causal_conv1d_silu(
        raw_qkv, conv_weight, conv_state,
        conv_state_indices, conv_num_accepted).view(b, s, 3, n, d)
    query, key, value = (convolved[:, :, i].float() for i in range(3))
    query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
    key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    return raw_qkv, convolved, query, key, value


def _golden_decay(hidden_states, weights, acts, shape):
    """Decay activation plus the beta/gate projections consumed by the recurrence."""
    b, s, n, d = shape
    decay_a_weight, decay_b_weight, beta_weight, gate_weight = weights
    a_log, dt_bias, lower_bound = acts
    decay_a = linear_reference(hidden_states, decay_a_weight, torch.bfloat16)
    raw_decay = linear_reference(decay_a, decay_b_weight, torch.bfloat16).view(b, s, n, d)
    beta = torch.sigmoid(
        linear_reference(hidden_states, beta_weight, torch.bfloat16).float())
    gate = linear_reference(hidden_states, gate_weight, torch.bfloat16).view(b, s, n, d)
    decay = torch.exp(lower_bound * torch.sigmoid(
        torch.exp(a_log).view(1, 1, n, 1)
        * (raw_decay.float() + dt_bias.view(1, 1, n, d))))
    return decay, beta, gate


def _golden_recurrence(recurrent_state, query, key, value, decay, beta, scale, shape):
    """The FP32 delta-rule recurrence over the ``S`` proposal tokens."""
    b, s, n, d = shape
    current = recurrent_state.clone().float()
    replay_u = torch.empty(b, s, n, d, dtype=torch.float16)
    recurrent = torch.empty(b, s, n, d, dtype=torch.float32)
    snapshots = []
    for t in range(s):
        current = current * decay[:, t].unsqueeze(-2)
        delta = (value[:, t] - (current * key[:, t].unsqueeze(-2)).sum(-1)) * beta[:, t].unsqueeze(-1)
        replay_u[:, t] = delta.to(torch.float16)
        current = current + delta.unsqueeze(-1) * key[:, t].unsqueeze(-2)
        snapshots.append(current.clone())
        recurrent[:, t] = (current * query[:, t].unsqueeze(-2)).sum(-1) * float(scale)
    return replay_u, recurrent, snapshots


def run_golden(hidden_states, qkv_weight, decay_a_weight, decay_b_weight, beta_weight,
               gate_weight, output_norm_weight, output_weight, conv_weight,
               conv_state, recurrent_state, conv_state_indices, conv_num_accepted,
               a_log, dt_bias, scale, lower_bound, rms_norm_eps, num_heads=NUM_HEADS):
    """Return a dict with the public outputs plus every intermediate tensor.

    ``num_heads`` is the TP knob (``N = 96 / TP``, one of 6/12/24); it defaults to
    the module-level ``NUM_HEADS`` so every existing caller is unchanged.
    """
    conv_state = conv_state.clone()
    b, s, _ = map(int, hidden_states.shape)
    shape = (b, s, int(num_heads), HEAD_DIM)
    n, d = shape[2], shape[3]
    projection = n * d

    raw_qkv, convolved, query, key, value = _golden_front(
        hidden_states, qkv_weight, conv_weight, conv_state,
        conv_state_indices, conv_num_accepted, shape)
    decay, beta, gate = _golden_decay(
        hidden_states, (decay_a_weight, decay_b_weight, beta_weight, gate_weight),
        (a_log, dt_bias, lower_bound), shape)
    replay_u, recurrent, snapshots = _golden_recurrence(
        recurrent_state, query, key, value, decay, beta, scale, shape)

    inv_rms = torch.rsqrt(recurrent.square().mean(-1, keepdim=True)
                          + float(rms_norm_eps))
    normalized = recurrent * inv_rms * output_norm_weight.float()
    gated = (normalized * torch.sigmoid(gate.float())).to(torch.bfloat16)
    output = linear_reference(gated.reshape(b, s, projection), output_weight, torch.bfloat16)

    return {
        "output": output,
        "conv_state": conv_state,
        "recurrent_state": recurrent_state,
        "replay_u": replay_u,
        "replay_k": key.to(torch.float16),
        "replay_decay": decay,
        "raw_qkv": raw_qkv,
        "convolved": convolved,
        "query": query,
        "key": key,
        "value": value,
        "decay": decay,
        "beta": beta,
        "gate": gate,
        "recurrent": recurrent,
        "gated": gated,
        "snapshots": snapshots,
    }
