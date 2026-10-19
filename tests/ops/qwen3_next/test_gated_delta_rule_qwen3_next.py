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
Qwen3-next Gated Delta Rule STest Module

This module provides test cases and wrapper functions for the Chunk Gated Delta Rule
attention mechanism. It includes both PyPTO implementation calls and PyTorch reference
implementations for validation.

Main Functions:
    - gen_dims: Generate dimension parameters
    - gen_inputs: Generate input tensors
    - gen_data: Generate test data based on case name
    - do_test_chunk_gated_delta_rule: Execute test case
    - pypto_chunk_gated_delta_rule_dyn: Dynamic wrapper for PyPTO implementation

Example:
    pytest tests/ops/qwen3_next/test_gated_delta_rule.py
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F
import torch_npu

_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))


import pypto
from qwen3_next.gated_delta_rule_impl import chunk_gated_delta_rule, chunk_gated_delta_rule_unaligned


def gen_dims(params):
    """Generate dimension parameters from input params."""
    dims = {}
    dims["T"] = params["T"]
    dims["B"] = params["B"]
    dims["Nqk"] = params["Nqk"]
    dims["Nv"] = params["Nv"]
    dims["D"] = 128
    dims["L"] = 128
    return dims


def gen_inputs(dims, dtype=torch.float32):
    """Generate input tensors for testing."""
    t = dims["T"]
    b = dims["B"]
    nqk = dims["Nqk"]
    nv = dims["Nv"]
    d = dims["D"]
    l = dims["L"]

    # Generate input data
    input_query = torch.rand([t, nqk, d], dtype=dtype) * (1.3655 + 0.2785) - (1.3655 + 0.2785)
    input_key = torch.rand([t, nqk, d], dtype=dtype) * (1.4664 + 0.2785) - (1.4664 + 0.2785)
    input_value = torch.rand([t, nv, d], dtype=dtype) * (1.6488 + 0.2785) - (1.6488 + 0.2785)
    input_beta = torch.rand([t, nv], dtype=dtype) * (0.8927 - 0.0889) - (0.8927 - 0.0889)
    input_gate = torch.rand([t, nv], dtype=dtype) * (-0.1343 + 37.5452) - (-0.1343 + 37.5452)
    input_states = torch.zeros([b, nv, d, d], dtype=dtype)

    # Generate act_seq_len based on B and T
    seq_len_per_batch = t // b
    act_seq_len = [i * seq_len_per_batch for i in range(b + 1)]
    act_seq_len = torch.tensor(act_seq_len, dtype=torch.int32)

    # Helper tensors
    mask = torch.tril(-torch.ones([l, l], dtype=dtype), diagonal=-1)
    tril_mask = torch.ones([l, l], dtype=dtype).tril()
    eye_data = torch.eye(16, dtype=dtype).repeat(1, 8)
    eye_data_unaligned = torch.eye(16, dtype=dtype)

    return {
        "query": input_query,
        "key": input_key,
        "value": input_value,
        "beta": input_beta,
        "gate": input_gate,
        "states": input_states,
        "act_seq_len": act_seq_len,
        "mask": mask,
        "tril_mask": tril_mask,
        "eye_data": eye_data,
        "eye_data_unaligned": eye_data_unaligned,
    }


def golden_chunk_gated_delta_rule(inputs: dict, dims: dict):
    """Calculate golden output using PyTorch reference implementation."""
    query = inputs["query"]
    key = inputs["key"]
    value = inputs["value"]
    beta = inputs["beta"]
    gate = inputs["gate"]
    states = inputs["states"]
    act_seq_len = inputs["act_seq_len"]

    core_attn_out, final_state = segs_chunk_gated_delta_rule(
        query=query.clone(),
        key=key.clone(),
        value=value.clone(),
        gate=gate.clone(),
        beta=beta.clone(),
        act_seq_len=act_seq_len.clone(),
        chunk_size=128,
        initial_state=states.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    return {
        "core_attn_out": core_attn_out,
        "final_state": final_state,
    }


def gen_data(case_name):
    """Generate test data based on case name."""
    if case_name.startswith("ChunkGatedDeltaRuleSTest.b2_nqk2_nv4_s1k"):
        params = {"T": 1024 * 2, "B": 2, "Nqk": 2, "Nv": 4, }
    elif case_name.startswith("ChunkGatedDeltaRuleSTest.b2_nqk4_nv8_s4k"):
        params = {"T": 1024 * 8, "B": 2, "Nqk": 4, "Nv": 8, }
    elif case_name.startswith("ChunkGatedDeltaRuleSTest.b2_nqk2_nv4_s8k"):
        params = {"T": 1024 * 16, "B": 2, "Nqk": 2, "Nv": 4, }
    elif case_name.startswith("ChunkGatedDeltaRuleSTest.b2_nqk4_nv8_s8k"):
        params = {"T": 1024 * 16, "B": 2, "Nqk": 4, "Nv": 8, }
    elif case_name.startswith("ChunkGatedDeltaRuleSTest.b1_nqk16_nv32_s32k"):
        params = {"T": 1024 * 32, "B": 1, "Nqk": 16, "Nv": 32, }
    elif case_name.startswith("ChunkGatedDeltaRuleSTest.b1_nqk2_nv4_s256k"):
        params = {"T": 1024 * 256, "B": 1, "Nqk": 2, "Nv": 4, }
    elif case_name.startswith("ChunkGatedDeltaRuleSTest.b1_nqk2_nv4_s512k"):
        params = {"T": 1024 * 512, "B": 1, "Nqk": 2, "Nv": 4, }
    elif case_name.startswith("ChunkGatedDeltaRuleSTest.b1_nqk2_nv4_s1m"):
        params = {"T": 1024 * 1024, "B": 1, "Nqk": 2, "Nv": 4, }
    elif case_name.startswith("ChunkGatedDeltaRuleSTest.b1_nqk2_nv4_s1026"):
        params = {"T": 1026, "B": 1, "Nqk": 2, "Nv": 4, }
    elif case_name.startswith("ChunkGatedDeltaRuleSTest.b1_nqk2_nv4_s2059"):
        params = {"T": 2059, "B": 1, "Nqk": 2, "Nv": 4, }
    else:
        raise Exception(f"Can't get func to gen golden, Case({case_name})")

    seed = 0
    torch.manual_seed(seed)
    dims = gen_dims(params)
    inputs = gen_inputs(dims, torch.float32)
    outputs = golden_chunk_gated_delta_rule(inputs, dims)
    return dims, inputs, outputs


def gen_zero_tensor(t):
    """Generate zero tensor with same shape and dtype."""
    return torch.zeros_like(t).npu()


def pypto_chunk_gated_delta_rule_dyn(dims, inputs: dict, outputs: dict):
    """Dynamic wrapper for PyPTO chunk gated delta rule."""
    b = dims["B"]
    nqk = dims["Nqk"]
    nv = dims["Nv"]
    d = dims["D"]
    l = dims["L"]
    act_seq_len = inputs["act_seq_len"]

    if (act_seq_len % l != 0).any():
        input_data = [inputs["query"], inputs["key"], inputs["value"], inputs["beta"], inputs["gate"], inputs["states"],
                inputs["mask"], inputs["tril_mask"], inputs["eye_data"], inputs["act_seq_len"]]
        output_data = [outputs["core_attn_out"], outputs["final_state"]]
        chunk_gated_delta_rule_unaligned(b, nqk, nv, d, l)(*input_data, *output_data)
    else:
        input_data = [inputs["query"], inputs["key"], inputs["value"], inputs["beta"], inputs["gate"], inputs["states"],
                inputs["mask"], inputs["tril_mask"], inputs["eye_data"], inputs["act_seq_len"]]
        output_data = [outputs["core_attn_out"], outputs["final_state"]]
        chunk_gated_delta_rule(b, nqk, nv, d, l)(*input_data, *output_data)

    torch_npu.npu.synchronize()


def do_test_chunk_gated_delta_rule(case_name):
    """Execute test case for chunk gated delta rule."""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    dims, inputs_data, golden_data = gen_data(case_name)

    # Move inputs to NPU
    inputs = {
        "query": inputs_data["query"].npu(),
        "key": inputs_data["key"].npu(),
        "value": inputs_data["value"].npu(),
        "beta": inputs_data["beta"].npu(),
        "gate": inputs_data["gate"].npu(),
        "states": inputs_data["states"].npu(),
        "act_seq_len": inputs_data["act_seq_len"].npu(),
        "mask": inputs_data["mask"].npu(),
        "tril_mask": inputs_data["tril_mask"].npu(),
        "eye_data": inputs_data["eye_data"].npu(),
        "eye_data_unaligned": inputs_data["eye_data_unaligned"].npu(),
    }

    # Golden outputs
    core_attn_out_golden = golden_data["core_attn_out"]
    final_state_golden = golden_data["final_state"]

    # Prepare output tensors
    outputs = {
        "core_attn_out": gen_zero_tensor(core_attn_out_golden),
        "final_state": gen_zero_tensor(final_state_golden),
    }

    # Run PyPTO implementation
    pypto_chunk_gated_delta_rule_dyn(dims, inputs, outputs)

    # Compare results
    compare(actual=outputs["core_attn_out"].cpu(), expected=core_attn_out_golden, name="core_attn_out", rtol=1e-3,
        atol_abs=0.005, atol_rel=1e-3)
    compare(actual=outputs["final_state"].cpu(), expected=final_state_golden, name="final_state", rtol=1e-3,
        atol_abs=0.005, atol_rel=1e-3)


def compare(**kwargs):
    """Compare two tensors with tolerance."""
    actual = kwargs.get("actual")
    expected = kwargs.get("expected")
    name = kwargs.get("name")
    rtol = kwargs.get("rtol")
    atol_abs = kwargs.get("atol_abs")
    atol_rel = kwargs.get("atol_rel")

    diff = torch.abs(actual.float() - expected.float())
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()

    tolerance = atol_abs + atol_rel * torch.abs(expected.float())
    out_of_tolerance = (diff > tolerance).sum().item()
    total = actual.numel()
    diff_thd = total * atol_abs

    if out_of_tolerance > diff_thd:
        raise AssertionError(f"{name} comparison failed: {out_of_tolerance} elements out of tolerance")


def segs_chunk_gated_delta_rule(**kwargs):
    """Segmented chunk gated delta rule for batch processing."""
    query = kwargs.get("query")
    key = kwargs.get("key")
    value = kwargs.get("value")
    g = kwargs.get("gate")
    beta = kwargs.get("beta")
    act_seq_len = kwargs.get("act_seq_len")
    chunk_size = kwargs.get("chunk_size")
    initial_state = kwargs.get("initial_state")
    output_final_state = kwargs.get("output_final_state")
    use_qk_l2norm_in_kernel = kwargs.get("use_qk_l2norm_in_kernel")


    t, n1, d = query.shape
    t, n, d = value.shape
    batch = act_seq_len.shape[0] - 1

    query = query.repeat_interleave(n // n1, dim=1)
    key = key.repeat_interleave(n // n1, dim=1)

    final_state = torch.zeros([batch, n, d, d], dtype=torch.float32, device=query.device)

    query, key, value, beta, g = \
        [x.transpose(0, 1).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]
    final_attn = torch.zeros([t, n, d], dtype=torch.float32, device=query.device)

    for b_idx in range(batch):
        s = act_seq_len[b_idx + 1] - act_seq_len[b_idx]
        b_ofs = act_seq_len[b_idx]
        seg_s = 128
        pad_size = (chunk_size - s % chunk_size) % chunk_size
        pad_seq_length = s + pad_size
        batch_query, batch_key, batch_value = \
            [F.pad(x[:, b_ofs:b_ofs + s], (0, 0, 0, pad_size)) for x in (query, key, value)]
        batch_beta, batch_g = [F.pad(x[:, b_ofs:b_ofs + s], (0, pad_size)) for x in (beta, g)]
        result_list = []
        recurrent_state = initial_state[b_idx:b_idx + 1, ...]
        for s_idx in range(0, pad_seq_length, seg_s):
            chunk_query, chunk_key, chunk_value = \
                [x[:, s_idx:s_idx + seg_s, :].reshape(1, n, seg_s, d) for x in (batch_query, batch_key, batch_value)]
            chunk_gate, chunk_beta = [x[:, s_idx:s_idx + seg_s].reshape(1, n, seg_s) for x in (batch_g, batch_beta)]
            cur_attn, cur_state = segs_chunk_gated_delta_rule_sub(query=chunk_query, key=chunk_key, value=chunk_value,
                g=chunk_gate, beta=chunk_beta, chunk_size=chunk_size, initial_state=recurrent_state,
                output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,)
            result_list.append(cur_attn.squeeze(0))
            recurrent_state = cur_state
        batch_attn = torch.cat(result_list, dim=0)[:s]
        final_attn[b_ofs:b_ofs + s] = batch_attn
        final_state[b_idx:b_idx + 1, ...] = recurrent_state
    return final_attn, final_state


def segs_chunk_gated_delta_rule_sub_inverse(attn, chunk_size):
    for index in range(1, chunk_size):
        line = attn[..., index, :index].clone()
        sub = attn[..., :index, :index].clone()
        attn[..., index, :index] = line + (line.unsqueeze(-1) * sub).sum(-2)
    return attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)


def segs_chunk_gated_delta_rule_sub_cycle(**kwargs):
    query = kwargs.get("query")
    key = kwargs.get("key")
    value = kwargs.get("value")
    decay_mask = kwargs.get("decay_mask")
    k_cumdecay = kwargs.get("k_cumdecay")
    g = kwargs.get("g")
    last_recurrent_state = kwargs.get("last_recurrent_state")
    total_sequence_length = kwargs.get("total_sequence_length")
    chunk_size = kwargs.get("chunk_size")

    attn_out = torch.zeros_like(value).to(query.device)
    attn_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for index in range(0, total_sequence_length // chunk_size):
        q_index, k_index, v_index = query[:, :, index], key[:, :, index], value[:, :, index]
        attn = (q_index @ k_index.transpose(-1, -2) * decay_mask[:, :, index]).masked_fill_(attn_mask, 0)
        v_new = v_index - (k_cumdecay[:, :, index]) @ last_recurrent_state
        attn_out[:, :, index] = (q_index * g[:, :, index, :, None].exp()) @ last_recurrent_state + attn @ v_new
        last_recurrent_state = last_recurrent_state * g[:, :, index, -1, None, None].exp() + \
            (k_index * (g[:, :, index, -1, None] - g[:, :, index]).exp()[..., None]).transpose(-1, -2) @ v_new

    return attn_out, last_recurrent_state


def segs_chunk_gated_delta_rule_sub(**kwargs):
    """PyTorch reference implementation of chunk gated delta rule."""
    query = kwargs.get("query")
    key = kwargs.get("key")
    value = kwargs.get("value")
    g = kwargs.get("g")
    beta = kwargs.get("beta")
    chunk_size = kwargs.get("chunk_size")
    initial_state = kwargs.get("initial_state")
    output_final_state = kwargs.get("output_final_state")
    use_qk_l2norm_in_kernel = kwargs.get("use_qk_l2norm_in_kernel")

    b, n, s, d = value.shape

    initial_state = initial_state.transpose(3, 2)
    if use_qk_l2norm_in_kernel:
        query = query * torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = [F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value)]
    beta, g = [F.pad(x, (0, pad_size)) for x in (beta, g)]

    total_sequence_length = sequence_length + pad_size
    query = query * (1 / (query.shape[-1] ** 0.5))

    v_beta, k_beta = [x * beta.unsqueeze(-1) for x in (value, key)]
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()

    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)

    attn = segs_chunk_gated_delta_rule_sub_inverse(attn, chunk_size)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    if initial_state is None:
        last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=query.device).to(value)
    else:
        last_recurrent_state = initial_state.to(value)

    attn_out, last_recurrent_state = segs_chunk_gated_delta_rule_sub_cycle(query=query, key=key, value=value,
        decay_mask=decay_mask, k_cumdecay=k_cumdecay, g=g, last_recurrent_state=last_recurrent_state,
        total_sequence_length=total_sequence_length, chunk_size=chunk_size)

    if not output_final_state:
        last_recurrent_state = None
    attn_out = attn_out.reshape(attn_out.shape[0], attn_out.shape[1], -1, attn_out.shape[-1])
    attn_out = attn_out[:, :, :sequence_length].transpose(1, 2).contiguous()

    last_recurrent_state = last_recurrent_state.transpose(3, 2)

    return attn_out, last_recurrent_state


# ==================== Test Cases ====================
# Test case: B:2, Nqk:2, Nv:4, S:4K
@pytest.mark.skip(reason="large test case")
def test_b2_nqk2_nv4_s1k():
    do_test_chunk_gated_delta_rule("ChunkGatedDeltaRuleSTest.b2_nqk2_nv4_s1k")


# Test case: B:2, Nqk:4, Nv:8, S:4K
@pytest.mark.skip(reason="large test case")
def test_b2_nqk4_nv8_s4k():
    do_test_chunk_gated_delta_rule("ChunkGatedDeltaRuleSTest.b2_nqk4_nv8_s4k")


# Test case: B:2, Nqk:2, Nv:4, S:8K
@pytest.mark.skip(reason="large test case")
def test_b2_nqk2_nv4_s8k():
    do_test_chunk_gated_delta_rule("ChunkGatedDeltaRuleSTest.b2_nqk2_nv4_s8k")


# Test case: B:2, Nqk:4, Nv:8, S:8K
@pytest.mark.skip(reason="large test case")
def test_b2_nqk4_nv8_s8k():
    do_test_chunk_gated_delta_rule("ChunkGatedDeltaRuleSTest.b2_nqk4_nv8_s8k")


# Test case: B:1, Nqk:16, Nv:32, S:32K
@pytest.mark.skip(reason="large test case")
def test_b1_nqk16_nv32_s32k():
    do_test_chunk_gated_delta_rule("ChunkGatedDeltaRuleSTest.b1_nqk16_nv32_s32k")


# Test case: B:1, Nqk:2, Nv:4, S:256K
@pytest.mark.skip(reason="large test case")
def test_b1_nqk2_nv4_s256k():
    do_test_chunk_gated_delta_rule("ChunkGatedDeltaRuleSTest.b1_nqk2_nv4_s256k")


# Test case: B:1, Nqk:2, Nv:4, S:512K
@pytest.mark.skip(reason="large test case")
def test_b1_nqk2_nv4_s512k():
    do_test_chunk_gated_delta_rule("ChunkGatedDeltaRuleSTest.b1_nqk2_nv4_s512k")


# Test case: B:1, Nqk:2, Nv:4, S:1M
@pytest.mark.skip(reason="large test case")
def test_b1_nqk2_nv4_s1m():
    do_test_chunk_gated_delta_rule("ChunkGatedDeltaRuleSTest.b1_nqk2_nv4_s1m")


# Test case: B:1, Nqk:2, Nv:4, S:1026
@pytest.mark.skip(reason="large test case")
def test_b1_nqk2_nv4_s1026():
    do_test_chunk_gated_delta_rule("ChunkGatedDeltaRuleSTest.b1_nqk2_nv4_s1026")


# Test case: B:1, Nqk:2, Nv:4, S:4108
@pytest.mark.skip(reason="non-divisible test case")
def test_b1_nqk2_nv4_s2059():
    do_test_chunk_gated_delta_rule("ChunkGatedDeltaRuleSTest.b1_nqk2_nv4_s2059")


if __name__ == "__main__":
    test_b2_nqk2_nv4_s1k()