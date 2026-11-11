#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent CPU golden for the fused ``mega_recurrent_kda`` operator.

Pure-torch CPU reference implementing the operator math:

    projections -> causal Conv1D(K=4)+SiLU -> Q/K RMS norm -> decay/beta
    activation -> KDA delta-rule recurrence -> RMSNorm + sigmoid gate
    -> output projection

All arithmetic runs in FP32 except where the operator contract fixes a BF16
rounding point (``decay_a`` intermediate, ``convolved`` result, ``gated``
result).  The returned
tuple contains the public output plus the two in-place state buffers and the
intermediate workspace regions used by the precision test.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import torch

HEAD_DIM = 128
CONV_KERNEL_SIZE = 4
QK_EPSILON = 1e-6


@dataclasses.dataclass
class MegaRecurrentKDAWeights:
    qkv_projection_weight: torch.Tensor
    decay_projection_a_weight: torch.Tensor
    decay_projection_b_weight: torch.Tensor
    beta_projection_weight: torch.Tensor
    output_gate_projection_weight: torch.Tensor
    output_norm_weight: torch.Tensor
    output_projection_weight: torch.Tensor
    conv1d_weight: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor


class MegaRecurrentKDAGolden(NamedTuple):
    output: torch.Tensor
    raw_qkv: torch.Tensor
    convolved_qkv: torch.Tensor
    decay_a: torch.Tensor
    raw_decay: torch.Tensor
    raw_beta: torch.Tensor
    raw_output_gate: torch.Tensor
    recurrent_output: torch.Tensor
    gated_output: torch.Tensor
    conv_state: torch.Tensor
    recurrent_state: torch.Tensor


class CpuArguments(NamedTuple):
    hidden_states: torch.Tensor
    qkv_projection_weight: torch.Tensor
    decay_projection_a_weight: torch.Tensor
    decay_projection_b_weight: torch.Tensor
    beta_projection_weight: torch.Tensor
    output_gate_projection_weight: torch.Tensor
    output_norm_weight: torch.Tensor
    output_projection_weight: torch.Tensor
    conv1d_weight: torch.Tensor
    conv_state: torch.Tensor
    recurrent_state: torch.Tensor
    conv_state_indices: torch.Tensor
    ssm_state_indices: torch.Tensor
    conv_num_accepted_tokens: torch.Tensor
    num_accepted_tokens: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor


def _linear(value: torch.Tensor, weight: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    """``value @ weight`` (weight is ``[K, N]``) accumulated in FP32, rounded once."""
    result = torch.matmul(value.float(), weight.float())
    return result.to(output_dtype)


def _conv1d_silu(raw_qkv: torch.Tensor, conv1d_weight: torch.Tensor, conv_state: torch.Tensor,
                 conv_state_indices: torch.Tensor, conv_num_accepted_tokens: torch.Tensor) -> torch.Tensor:
    """Model-native causal Conv1D + SiLU with the FP32/BF16 history boundary.

    ``conv_state_indices == 0`` marks the immutable null slot; those rows produce
    an all-zero convolution output and leave ``conv_state`` untouched.
    """
    batch_size, seq_len, channels = map(int, raw_qkv.shape)
    if tuple(conv1d_weight.shape) != (CONV_KERNEL_SIZE, channels):
        raise ValueError("conv1d_weight must be [4, 3*P]")
    if int(conv_state.shape[1]) < seq_len + CONV_KERNEL_SIZE - 2:
        raise ValueError("conv_state state_length must be at least S+2")
    result = torch.zeros_like(raw_qkv, dtype=torch.bfloat16)
    original_state = conv_state.clone()
    filter_fp32 = conv1d_weight.float()
    for batch_index in range(batch_size):
        cache_index = int(conv_state_indices[batch_index])
        if cache_index == 0:
            continue
        accepted = int(conv_num_accepted_tokens[batch_index])
        offset = max(accepted - 1, 0)
        history_stop = offset + CONV_KERNEL_SIZE - 1
        history = original_state[cache_index, offset:history_stop].float().clone()
        if int(history.shape[0]) != CONV_KERNEL_SIZE - 1:
            raise ValueError("accepted Conv history exceeds state_length")
        for token in range(seq_len):
            current = raw_qkv[batch_index, token].float()
            window = torch.cat((history, current.unsqueeze(0)), dim=0)
            products = window * filter_fp32
            convolved = products[0] + products[1] + (products[2] + products[3])
            result[batch_index, token] = torch.nn.functional.silu(convolved).to(torch.bfloat16)
            history = torch.cat((history[1:], current.to(torch.bfloat16).float().unsqueeze(0)), dim=0)
        selected = original_state[cache_index, offset:history_stop]
        conv_state[cache_index, :2].copy_(selected[1:])
        proposal_stop = 2 + seq_len
        conv_state[cache_index, 2:proposal_stop].copy_(raw_qkv[batch_index])
    return result


def _pooled_recurrent(convolved_qkv, raw_decay, raw_beta, recurrent_state, ssm_state_indices,
                      num_accepted_tokens, a_log, dt_bias, scale, lower_bound) -> torch.Tensor:
    batch_size, seq_len, _, num_heads, head_dim = convolved_qkv.shape
    query = convolved_qkv[:, :, 0].float()
    key = convolved_qkv[:, :, 1].float()
    value = convolved_qkv[:, :, 2].float()
    query = query * torch.rsqrt(query.square().sum(dim=-1, keepdim=True) + QK_EPSILON)
    key = key * torch.rsqrt(key.square().sum(dim=-1, keepdim=True) + QK_EPSILON)
    decay = float(lower_bound) * torch.sigmoid(
        torch.exp(a_log.float()).view(1, 1, num_heads, 1)
        * (raw_decay.float() + dt_bias.float().view(1, 1, num_heads, head_dim)))
    beta = torch.sigmoid(raw_beta.float())
    output = torch.empty(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32,
                         device=convolved_qkv.device)
    state_indices = ssm_state_indices.reshape(batch_size, seq_len)
    for batch_index in range(batch_size):
        accepted = int(num_accepted_tokens[batch_index])
        initial_token = accepted - 1 if accepted > 0 else 0
        current = recurrent_state[int(state_indices[batch_index, initial_token])].float().clone()
        for token in range(seq_len):
            current = current * torch.exp(decay[batch_index, token]).unsqueeze(-2)
            key_value = (current * key[batch_index, token].unsqueeze(-2)).sum(dim=-1)
            delta = (value[batch_index, token] - key_value) * beta[batch_index, token].unsqueeze(-1)
            current = current + delta.unsqueeze(-1) * key[batch_index, token].unsqueeze(-2)
            output[batch_index, token] = (current * query[batch_index, token].unsqueeze(-2)).sum(dim=-1) * float(scale)
            recurrent_state[int(state_indices[batch_index, token])].copy_(current.to(recurrent_state.dtype))
    return output


def run_mega_recurrent_kda_golden(hidden_states, weights, conv_state, recurrent_state, conv_state_indices,
                                  ssm_state_indices, conv_num_accepted_tokens, num_accepted_tokens,
                                  scale, lower_bound, rms_norm_eps) -> MegaRecurrentKDAGolden:
    """Run the full fused fragment on CPU with independent arithmetic."""
    if hidden_states.device.type != "cpu":
        raise ValueError("golden must use independent CPU arithmetic")
    batch_size, seq_len, hidden_size = map(int, hidden_states.shape)
    num_heads = int(weights.output_gate_projection_weight.shape[1]) // HEAD_DIM
    projection_size = num_heads * HEAD_DIM
    raw_qkv = _linear(hidden_states, weights.qkv_projection_weight, torch.float32)
    convolved_flat = _conv1d_silu(raw_qkv, weights.conv1d_weight, conv_state, conv_state_indices,
                                  conv_num_accepted_tokens)
    convolved_qkv = convolved_flat.view(batch_size, seq_len, 3, num_heads, HEAD_DIM)
    decay_a = _linear(hidden_states, weights.decay_projection_a_weight, torch.bfloat16)
    raw_decay = _linear(decay_a, weights.decay_projection_b_weight, torch.bfloat16).view(
        batch_size, seq_len, num_heads, HEAD_DIM)
    raw_beta = _linear(hidden_states, weights.beta_projection_weight, torch.bfloat16)[
        ..., :num_heads]
    raw_output_gate = _linear(hidden_states, weights.output_gate_projection_weight, torch.bfloat16).view(
        batch_size, seq_len, num_heads, HEAD_DIM)
    recurrent_output = _pooled_recurrent(convolved_qkv, raw_decay, raw_beta, recurrent_state, ssm_state_indices,
                                         num_accepted_tokens, weights.a_log, weights.dt_bias, scale, lower_bound)
    inverse_rms = torch.rsqrt(recurrent_output.square().mean(dim=-1, keepdim=True) + float(rms_norm_eps))
    normalized = recurrent_output * inverse_rms * weights.output_norm_weight.float()
    gated_output = (normalized * torch.sigmoid(raw_output_gate.float())).to(torch.bfloat16)
    output = _linear(gated_output.view(batch_size, seq_len, projection_size),
                     weights.output_projection_weight, torch.bfloat16).view(batch_size, seq_len, hidden_size)
    return MegaRecurrentKDAGolden(output=output, raw_qkv=raw_qkv, convolved_qkv=convolved_qkv, decay_a=decay_a,
                                  raw_decay=raw_decay, raw_beta=raw_beta, raw_output_gate=raw_output_gate,
                                  recurrent_output=recurrent_output, gated_output=gated_output,
                                  conv_state=conv_state, recurrent_state=recurrent_state)


def make_cpu_arguments(batch_size, seq_len, seed, num_heads=6):
    """Deterministic legal inputs for one case, all on CPU (BF16 weights except FP32 states)."""
    generator = torch.Generator().manual_seed(seed)
    hidden_size, head_dim = (7168, 128)
    projection_size = num_heads * head_dim

    def randn(shape, dtype=torch.bfloat16, scale=0.02):
        return (torch.randn(shape, generator=generator) * scale).to(dtype)

    conv_indices = torch.randperm(batch_size, generator=generator).int() + 1
    conv_indices[::7] = 0
    state_indices = torch.randperm(batch_size * seq_len, generator=generator).int() + 1
    accepted = torch.arange(batch_size, dtype=torch.int32) % (seq_len + 1)
    # Beta is handed over zero-padded to the 128-wide proj2 N tile; the kernel
    # reads only the first ``num_heads`` columns.  ``pad_beta`` keeps the RNG
    # draw at its original position so seeded cases stay reproducible.

    def pad_beta(beta):
        out = torch.zeros((hidden_size, head_dim), dtype=torch.bfloat16)
        out[:, :num_heads].copy_(beta)
        return out

    return CpuArguments(
        randn((batch_size, seq_len, hidden_size)),
        randn((hidden_size, 3 * projection_size)),
        randn((hidden_size, head_dim)),
        randn((head_dim, projection_size)),
        pad_beta(randn((hidden_size, num_heads))),
        randn((hidden_size, projection_size)),
        randn((head_dim,), torch.float32, scale=0.2),
        randn((projection_size, hidden_size)),
        randn((CONV_KERNEL_SIZE, 3 * projection_size)),
        randn((batch_size + 1, seq_len + 2, 3 * projection_size)),
        randn((batch_size * seq_len + 1, num_heads, head_dim, head_dim), torch.float32, scale=0.002),
        conv_indices,
        state_indices,
        accepted,
        (seq_len - accepted).contiguous(),
        randn((num_heads,), torch.float32, scale=0.1),
        randn((num_heads, head_dim), torch.float32, scale=0.1),
    )


_ARGUMENT_NAMES = (
    "hidden_states",
    "qkv_projection_weight",
    "decay_projection_a_weight",
    "decay_projection_b_weight",
    "beta_projection_weight",
    "output_gate_projection_weight",
    "output_norm_weight",
    "output_projection_weight",
    "conv1d_weight",
    "conv_state",
    "recurrent_state",
    "conv_state_indices",
    "ssm_state_indices",
    "conv_num_accepted_tokens",
    "num_accepted_tokens",
    "a_log",
    "dt_bias",
)


def weights_from_arguments(arguments):
    return MegaRecurrentKDAWeights(*arguments[1:9], arguments[15], arguments[16])
