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
"""Test module for deepseekv4_compressor."""

import os
import sys
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch_npu
import pytest

import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

import numpy as np
from numpy.testing import assert_allclose

from deepseek_v4.compressor_impl import compressor_pypto, npu_compressor, CompressorArgs


np.random.seed(0)
torch.manual_seed(0)
np.set_printoptions(formatter={"float": "{:.6f}".format})


def overlap_transform(tensor: torch.Tensor, value: float) -> torch.Tensor:
    # tensor shape: [batch_size, seq_len, ratio, 2*dim]
    b, s, ratio, d = tensor.size()
    d = d//2
    new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
    new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
    new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
    return new_tensor


def rms_norm_golden(x: torch.Tensor, eps: float, weight: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    var = x.square().mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (weight * x).to(dtype)


def apply_rotary_pos_emb_v2(
    x: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    mode: str = "half",
) -> torch.Tensor:
    input_dtype = x.dtype
    if input_dtype != torch.float32:
        x = x.to(torch.float32)
    if cos.dtype != torch.float32:
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)
    if mode == "half":
        b, s, d = x.shape
        x = x.reshape(b, s, d // 2, 2).permute(0, 1, 3, 2).reshape(b, s, d)

        x1, x2 = x.chunk(2, dim=-1)
        p = torch.cat((-x2, x1), dim=-1)
    else:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        p = torch.stack((-x2, x1), dim=-1).flatten(-2)
    x_embed = (x * cos) + (p * sin)
    x_embed = x_embed.to(input_dtype)

    return x_embed


@dataclass
class _CompressOverlapStepInputs:
    kv_total: Any
    score_total: Any
    ape: Any
    pos: Any
    b_idx: Any
    i: Any
    start_pos_dy: Any
    kv_state: Any
    score_state: Any
    kv_block_table: Any
    score_block_table: Any
    block_size: Any
    ratio: Any
    d: Any
    kv: Any
    score: Any


def _compress_overlap_step(inputs: _CompressOverlapStepInputs):
    """Overlap-mode (ratio==4) compression step for one token position."""
    block_offset = (inputs.start_pos_dy[inputs.b_idx] + inputs.i) // inputs.block_size
    kv_block_idx = inputs.kv_block_table[inputs.b_idx, block_offset]
    score_block_idx = inputs.score_block_table[inputs.b_idx, block_offset]
    cur_pos = (inputs.start_pos_dy[inputs.b_idx] + inputs.i) % inputs.block_size
    inputs.kv_state[kv_block_idx, cur_pos, :] = inputs.kv.squeeze(0)
    inputs.score_state[score_block_idx, cur_pos, :] = inputs.score.squeeze(0)
    should_compress = (inputs.start_pos_dy[inputs.b_idx] + inputs.i + 1) % inputs.ratio == 0
    if should_compress:
        pre_block_offset = (inputs.start_pos_dy[inputs.b_idx] + inputs.i
                            - 2 * inputs.ratio + 1) // inputs.block_size
        pre_kv_block_idx = inputs.kv_block_table[inputs.b_idx, pre_block_offset]
        pre_score_block_idx = inputs.score_block_table[inputs.b_idx, pre_block_offset]
        pre_start = (inputs.start_pos_dy[inputs.b_idx] + inputs.i - 2 * inputs.ratio + 1) % inputs.block_size
        pre_end = pre_start + inputs.ratio
        cur_start = (inputs.start_pos_dy[inputs.b_idx] + inputs.i - inputs.ratio + 1) % inputs.block_size
        cur_end = cur_start + inputs.ratio
        if inputs.start_pos_dy[inputs.b_idx] < inputs.ratio:
            kv_state_tmp = torch.cat([
                inputs.kv_state[pre_kv_block_idx, pre_start:pre_end, :inputs.d] * 0,
                inputs.kv_state[kv_block_idx, cur_start:cur_end, inputs.d:]], dim=0)
            score_state_tmp = torch.cat([
                inputs.score_state[pre_score_block_idx, pre_start:pre_end, :inputs.d] - float("inf"),
                inputs.score_state[score_block_idx, cur_start:cur_end, inputs.d:]], dim=0)
        else:
            kv_state_tmp = torch.cat([
                inputs.kv_state[pre_kv_block_idx, pre_start:pre_end, :inputs.d],
                inputs.kv_state[kv_block_idx, cur_start:cur_end, inputs.d:]], dim=0)
            score_state_tmp = torch.cat([
                inputs.score_state[pre_score_block_idx, pre_start:pre_end, :inputs.d],
                inputs.score_state[score_block_idx, cur_start:cur_end, inputs.d:]], dim=0)
        kv_new = (kv_state_tmp * score_state_tmp.softmax(dim=0)).sum(dim=0, keepdim=False)
    else:
        kv_new = None
    return kv_new, should_compress


@dataclass
class _CompressNonOverlapStepInputs:
    kv_total: Any
    score_total: Any
    ape: Any
    pos: Any
    b_idx: Any
    i: Any
    start_pos_dy: Any
    kv_state: Any
    score_state: Any
    kv_block_table: Any
    score_block_table: Any
    block_size: Any
    ratio: Any
    d: Any
    kv: Any
    score: Any


def _compress_non_overlap_step(inputs: _CompressNonOverlapStepInputs):
    """Non-overlap-mode compression step for one token position."""
    block_offset = (inputs.start_pos_dy[inputs.b_idx] + inputs.i) // inputs.block_size
    kv_block_idx = inputs.kv_block_table[inputs.b_idx, block_offset]
    score_block_idx = inputs.score_block_table[inputs.b_idx, block_offset]
    cur_pos = (inputs.start_pos_dy[inputs.b_idx] + inputs.i) % inputs.block_size
    inputs.kv_state[kv_block_idx, cur_pos, :] = inputs.kv.squeeze(0)
    inputs.score_state[score_block_idx, cur_pos, :] = inputs.score.squeeze(0)
    should_compress = (inputs.start_pos_dy[inputs.b_idx] + inputs.i + 1) % inputs.ratio == 0
    if should_compress:
        kv_tmp = torch.cat((inputs.kv_state[kv_block_idx, :-1, :], inputs.kv), dim=0)
        score_tmp = torch.cat((inputs.score_state[score_block_idx, :-1, :], inputs.score), dim=0)
        kv_new = (kv_tmp * score_tmp.softmax(dim=0)).sum(dim=0, keepdim=False)
    else:
        kv_new = None
    return kv_new, should_compress


def _compress_post_process(kv_new, dtype, eps, weight, rope_head_dim, sin, cos, b_idx, hadamard, rotate):
    """Apply RMSNorm, RoPE, and optional Hadamard to a compressed KV token."""
    kv = rms_norm_golden(kv_new.to(dtype), eps, weight)
    kv_rope = kv[..., -rope_head_dim:].clone()
    kv_new = kv.clone()
    kv_new[..., -rope_head_dim:] = apply_rotary_pos_emb_v2(
        kv_rope, sin[b_idx, ...], cos[b_idx, ...], "interleave")
    if rotate:
        return torch.matmul(kv_new, hadamard)
    return kv_new


@dataclass
class GoldenCompressInputs:
    x: Any
    sin: Any
    cos: Any
    wkv: Any
    wgate: Any
    ape: Any
    weight: Any
    kv_state: Any
    score_state: Any
    kv_block_table: Any
    score_block_table: Any
    hadamard: Any
    ratio: Any
    start_pos_dy: Any
    rope_head_dim: Any
    rotate: Any
    eps: Any = 1e-6


def golden_compress(inputs: GoldenCompressInputs):
    x = inputs.x
    sin = inputs.sin
    cos = inputs.cos
    wkv = inputs.wkv
    wgate = inputs.wgate
    ape = inputs.ape
    weight = inputs.weight
    kv_state = inputs.kv_state
    score_state = inputs.score_state
    kv_block_table = inputs.kv_block_table
    score_block_table = inputs.score_block_table
    hadamard = inputs.hadamard
    ratio = inputs.ratio
    start_pos_dy = inputs.start_pos_dy
    rope_head_dim = inputs.rope_head_dim
    rotate = inputs.rotate
    eps = inputs.eps
    bsz, s1, _ = x.size()
    overlap = ratio == 4
    dtype = x.dtype
    x = x.float()

    wkv = wkv.transpose(-2, -1).to(torch.float32)
    wgate = wgate.transpose(-2, -1).to(torch.float32)
    d = wkv.size(1) // (1 + overlap)

    kv_total = torch.matmul(x, wkv)
    score_total = torch.matmul(x, wgate)

    block_size = kv_state.shape[1]
    kv_output = torch.zeros(
        (min(bsz * s1, bsz * s1 // ratio + bsz), d),
        dtype=torch.bfloat16, device=x.device)
    for b_idx in range(bsz):
        for i in range(s1):
            pos = (start_pos_dy[b_idx] + i) % ratio
            kv = kv_total[b_idx, i:i + 1, :].clone()
            score = score_total[b_idx, i:i + 1, :].clone()
            score += ape[pos]
            if overlap:
                kv_new, should_compress = _compress_overlap_step(_CompressOverlapStepInputs(
                    kv_total=kv_total, score_total=score_total, ape=ape, pos=pos,
                    b_idx=b_idx, i=i, start_pos_dy=start_pos_dy,
                    kv_state=kv_state, score_state=score_state, kv_block_table=kv_block_table,
                    score_block_table=score_block_table,
                    block_size=block_size, ratio=ratio, d=d, kv=kv, score=score))
            else:
                kv_new, should_compress = _compress_non_overlap_step(_CompressNonOverlapStepInputs(
                    kv_total=kv_total, score_total=score_total, ape=ape, pos=pos,
                    b_idx=b_idx, i=i, start_pos_dy=start_pos_dy,
                    kv_state=kv_state, score_state=score_state, kv_block_table=kv_block_table,
                    score_block_table=score_block_table,
                    block_size=block_size, ratio=ratio, d=d, kv=kv, score=score))
            if should_compress:
                kv_output[b_idx, :] = _compress_post_process(
                    kv_new, dtype, eps, weight, rope_head_dim, sin, cos, b_idx, hadamard, rotate)
    return kv_output


def gen_inputs(
    bsz: int,
    seq: int,
    h: int,
    d: int,
    rope_head_dim: int,
    ratio: int,
    device: str,
):
    torch.manual_seed(42)
    overlap = ratio == 4
    coff = 1 + overlap
    x = torch.rand((bsz, seq, h), dtype=torch.bfloat16, device=device)
    rope_axis0 = min(bsz * seq, bsz * seq // ratio + bsz)
    sin = torch.rand((rope_axis0, rope_head_dim), dtype=torch.bfloat16, device=device)
    cos = torch.rand((rope_axis0, rope_head_dim), dtype=torch.bfloat16, device=device)
    wkv = torch.rand((coff * d, h), dtype=torch.bfloat16, device=device)
    wgate = torch.rand((coff * d, h), dtype=torch.bfloat16, device=device)
    ape = torch.rand((ratio, coff * d), dtype=torch.float32, device=device)
    weight = torch.ones(d, dtype=torch.float32, device=device)
    if overlap:
        block_table = (
            torch.ones(bsz, 100, dtype=torch.int32, device=device)
            + torch.arange(bsz, dtype=torch.int32, device=device).view(-1, 1) * 2
        )
    else:
        block_table = (
            torch.arange(100, dtype=torch.int32, device=device) % 2
            + 1
            + torch.arange(bsz, dtype=torch.int32, device=device).view(-1, 1) * 2
        )
    kv_state = torch.zeros(
        (block_table.max() + 1, 128, coff * d), dtype=torch.float32, device=device
    )
    score_state = torch.zeros(
        (block_table.max() + 1, 128, coff * d), dtype=torch.float32, device=device
    )
    hadamard = torch.rand((d, d), dtype=torch.bfloat16, device=device) * (d**-0.5)
    return (x, sin, cos, wkv, wgate, ape, weight,
            kv_state, score_state, block_table, hadamard)


@dataclass
class CompressorForwardInputs:
    x: Any
    kv_state: Any
    score_state: Any
    kv_block_table: Any
    score_block_table: Any
    sin: Any
    cos: Any
    wkv: Any
    wgate: Any
    ape: Any
    weight: Any
    hadamard: Any
    st: Any
    ra: Any
    rope_head_dim: Any
    ro: Any


class Compressor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: CompressorForwardInputs):
        args = CompressorArgs(
            x=inputs.x, kv_state=inputs.kv_state, score_state=inputs.score_state,
            kv_block_table=inputs.kv_block_table, score_block_table=inputs.score_block_table,
            sin=inputs.sin, cos=inputs.cos, wkv=inputs.wkv, wgate=inputs.wgate,
            ape=inputs.ape, weight=inputs.weight,
            hadamard=inputs.hadamard, start_pos=inputs.st, ratio=inputs.ra,
            rope_head_dim=inputs.rope_head_dim, rotate=inputs.ro)
        return compressor_pypto(args)


def compile_model(model):
    # aclgraph模式npugraph_ex调用
    compile_options = {
        "frozen_parameter": True,
        "static_kernel_compile": False,
    }
    compiled_model = torch.compile(model, dynamic=False, fullgraph=True, backend="npugraph_ex", options=compile_options)

    return compiled_model


def _run_compressor_test(ra, ro, bsz, seq, h, d, rope_head_dim, device, enable_acl_graph=False):
    """Common test runner for compressor tests."""
    st = torch.tensor([8192] * bsz, dtype=torch.int32, device=device)
    print(f"test_compressor_decode (ratio: {ra}, rotate: {ro}) begin!")

    x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state, block_table, hadamard = \
        gen_inputs(bsz, seq, h, d, rope_head_dim, ra, device)

    if enable_acl_graph:
        compressor_model = Compressor().npu()
        compressor_model = compile_model(compressor_model)
        out, kv_state_out, score_state_out = compressor_model(CompressorForwardInputs(
            x=x, kv_state=kv_state, score_state=score_state,
            kv_block_table=block_table, score_block_table=block_table,
            sin=sin, cos=cos, wkv=wkv, wgate=wgate,
            ape=ape, weight=weight, hadamard=hadamard, st=st, ra=ra,
            rope_head_dim=rope_head_dim, ro=ro))
        torch_npu.npu.synchronize()
    elif ra == 128:
        args = CompressorArgs(
            x=x, kv_state=kv_state, score_state=score_state,
            kv_block_table=block_table, score_block_table=block_table,
            sin=sin, cos=cos, wkv=wkv, wgate=wgate, ape=ape, weight=weight,
            hadamard=hadamard, start_pos=st, ratio=ra,
            rope_head_dim=rope_head_dim, rotate=ro)
        out, kv_state_out, score_state_out = npu_compressor(args)
    else:
        args = CompressorArgs(
            x=x, kv_state=kv_state, score_state=score_state,
            kv_block_table=block_table, score_block_table=block_table,
            sin=sin, cos=cos, wkv=wkv, wgate=wgate, ape=ape, weight=weight,
            hadamard=hadamard, start_pos=st, ratio=ra,
            rope_head_dim=rope_head_dim, rotate=ro)
        out, kv_state_out, score_state_out = compressor_pypto(args)

    kv = golden_compress(GoldenCompressInputs(
        x=x, sin=sin, cos=cos, wkv=wkv, wgate=wgate, ape=ape, weight=weight,
        kv_state=kv_state, score_state=score_state, kv_block_table=block_table,
        score_block_table=block_table,
        hadamard=hadamard, ratio=ra, start_pos_dy=st, rope_head_dim=rope_head_dim, rotate=ro))
    _assert_compressor_outputs(kv_state_out, kv_state, score_state_out, score_state, out, kv)
    print("test_compressor_decode passed!")


def _assert_compressor_outputs(kv_state_out, kv_state, score_state_out, score_state, out, kv):
    """Assert compressor outputs match goldens."""
    assert_allclose(kv_state_out.cpu().float().numpy(), kv_state.cpu().float().numpy(), rtol=1e-3, atol=1e-3)
    assert_allclose(score_state_out.cpu().float().numpy(), score_state.cpu().float().numpy(), rtol=1e-3, atol=1e-3)
    if kv is not None:
        assert_allclose(out.cpu().float().numpy(), kv.cpu().float().numpy(), rtol=0.0078125, atol=1e-4)


def _compressor_prep():
    """Common setup for compressor tests: get device and configure."""
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    torch.npu.set_device(device_id)
    torch_npu.npu.config.allow_internal_format = True
    return device


@pytest.mark.soc("950", "910")
def test_comp_128(enable_acl_graph=False):
    """Test Compressor"""
    print("=" * 60)
    print("Test: Compressor")
    print("=" * 60)
    device = _compressor_prep()
    _run_compressor_test(ra=128, ro=False, bsz=64, seq=2, h=4096, d=512, rope_head_dim=64,
                         device=device, enable_acl_graph=enable_acl_graph)


@pytest.mark.skip(reason="large test case")
def test_comp_4(enable_acl_graph=False):
    """Test Compressor"""
    print("Test: Compressor")
    print("=" * 60)
    device = _compressor_prep()
    _run_compressor_test(ra=4, ro=False, bsz=64, seq=2, h=4096, d=512, rope_head_dim=64,
                         device=device, enable_acl_graph=enable_acl_graph)


@pytest.mark.skip(reason="large test case")
def test_comp_indexer(enable_acl_graph=False):
    """Test Compressor"""
    print("=" * 60)
    print("Test: Compressor")
    print("=" * 60)
    device = _compressor_prep()
    _run_compressor_test(ra=4, ro=True, bsz=64, seq=2, h=4096, d=128, rope_head_dim=64,
                         device=device, enable_acl_graph=enable_acl_graph)


if __name__ == "__main__":
    test_comp_128()
    test_comp_4()
    test_comp_indexer()