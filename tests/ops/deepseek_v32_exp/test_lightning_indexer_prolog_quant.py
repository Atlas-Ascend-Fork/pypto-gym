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
"""
from dataclasses import dataclass
import os
import math
import logging
from typing import Any
import torch
import torch_npu

import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

import pytest
import pypto

from deepseek_v32_exp.lightning_indexer_prolog_quant_impl import (
    IndexerPrologQuantInput, IndexerPrologQuantOutput, IndexerPrologQuantAttr, IndexerPrologQuantConfigs,
    lightning_indexer_prolog_quant)
from common_utils import compare


def gen_dims(params):
    dims = {}
    dims["s2"] = params["s2"]
    dims["b"] = params["b"]
    dims["t"] = params["b"] * params["s1"]
    dims["h"] = 7168
    dims["q_lora_rank"] = 1536
    dims["idx_head_dim"] = 128
    dims["idx_n_heads"] = 64
    dims["rope_head_dim"] = 64
    dims["block_size"] = 128
    dims["block_num"] = dims["b"] * dims["s2"] // dims["block_size"]
    dims["n_kv"] = 1
    return dims


def gen_block_table(act_seq, block_size, s1):
    b = act_seq.shape[0]
    block_num = 0
    block_num_each = []
    max_kv = max(act_seq)
    for cur_s in act_seq:
        cur_block_num = math.ceil(cur_s / block_size)
        block_num_each.append(cur_block_num)
        block_num += cur_block_num
    block_table_shape = [b, math.ceil(max_kv / block_size)]
    block_idx_list = torch.arange(0, block_num, 1)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))].to(torch.int32)

    block_idx = 0
    block_table_bidx = 0
    block_table = -torch.ones(block_table_shape, dtype=torch.int32)

    for cur_block in block_num_each:
        for j in range(cur_block):
            block_table[block_table_bidx, j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_bidx += 1

    cache_index = -torch.ones((b, s1), dtype=torch.int64)
    for i in range(b):
        cur_act = act_seq[i]
        for j in range(s1):
            pos = cur_act - s1 + j
            block_idx_in_seq = pos // block_size
            global_block_id = block_table[i, block_idx_in_seq]

            offset_in_block = pos % block_size
            global_index = global_block_id * block_size + offset_in_block
            cache_index[i, j] = global_index

    return block_num, block_table, cache_index


def gen_cache_tensor(k_cache_bsnd, block_table, block_num, block_size):
    dtype = k_cache_bsnd.dtype
    b, s2, n_kv, d = k_cache_bsnd.shape
    k_cache = torch.zeros((block_num, block_size, n_kv, d), dtype=dtype)
    s2_new = ((s2 + block_size - 1) // block_size) * block_size  # ceil to block_size
    k_cache_raw = torch.zeros((b, s2_new, n_kv, d), dtype=dtype)
    k_cache_raw[:, :s2, :, :] = k_cache_bsnd

    for b_idx in range(b):
        for block_idx, cache_block_idx in enumerate(block_table[b_idx]):
            block_offset = block_idx * block_size
            if cache_block_idx == -1:
                continue
            else:
                k_cache[cache_block_idx, :, :, :] = k_cache_raw[
                    b_idx, block_offset: (block_offset + block_size), :, :
                ]

    return k_cache


def gen_inputs(dims, dtype=torch.bfloat16, qunat_dtype=torch.int8):
    b, t, n, d = dims["b"], dims["t"], dims["idx_n_heads"], dims["idx_head_dim"]
    s = t // b
    h = dims["h"]
    q_lora_rank = dims["q_lora_rank"]
    block_size = dims["block_size"]
    n_kv = dims["n_kv"]
    s2 = dims["s2"]
    rope_head_dim = dims["rope_head_dim"]

    x = torch.empty((b, s, h), dtype=dtype).uniform_(-1, 1)
    q_norm = torch.randint(low=-128, high=128, size=(b, s, q_lora_rank), dtype=qunat_dtype)
    q_norm_scale = torch.empty((b, s, 1), dtype=torch.float32).uniform_(-1, 1)
    w_idx_qb = torch.randint(low=-128, high=128, size=(q_lora_rank, n * d), dtype=qunat_dtype)
    w_idx_qb_scale = torch.empty((n * d, 1), dtype=torch.float32).uniform_(-1, 1)
    w_idx_k = torch.empty((h, d), dtype=dtype).uniform_(-1, 1)
    w_idx_proj = torch.empty((h, n), dtype=dtype).uniform_(-1, 1)
    ln_gamma = torch.ones((d,), dtype=dtype)
    ln_beta = torch.zeros((d,), dtype=dtype)

    random_angles = (torch.rand(b, s, rope_head_dim, dtype=torch.float32) * 2 * torch.pi)
    cos = torch.cos(random_angles).to(dtype)
    sin = torch.sin(random_angles).to(dtype)

    hadamard_q = torch.empty((d, d), dtype=dtype).uniform_(-1, 1)  # (128, 128)
    hadamard_k = torch.empty((d, d), dtype=dtype).uniform_(-1, 1)

    act_seq = torch.tensor([s2] * b)  # (b,)
    k_cache_bsnd = torch.randint(low=-128, high=128, size=(b, s2, n_kv, d), dtype=qunat_dtype)
    k_scale_cache_bsnd = torch.empty((b, s2, n_kv, 1), dtype=torch.float16).uniform_(-1, 1)
    block_num, block_table, k_cache_index = gen_block_table(act_seq, block_size, s)
    k_cache = gen_cache_tensor(k_cache_bsnd, block_table, block_num, block_size)
    k_scale_cache = gen_cache_tensor(k_scale_cache_bsnd, block_table, block_num, block_size)

    return {
        "token_x": x,  # input0, bf16
        "q_norm": q_norm,  # input1, int8
        "q_norm_scale": q_norm_scale,  # input2, fp32
        "w_idx_qb": w_idx_qb,  # input3, int8
        "w_idx_qb_scale": w_idx_qb_scale,  # input4, fp32
        "w_idx_k": w_idx_k,  # input5, bf16
        "w_idx_proj": w_idx_proj,  # input6, bf16
        "layer_norm_gamma": ln_gamma,  # input7, bf16
        "layer_norm_beta": ln_beta,  # input8, bf16
        "cos_idx_rope": cos,  # input9, bf16
        "sin_idx_rope": sin,  # input10, bf16
        "hadamard_q": hadamard_q,  # input11, bf16
        "hadamard_k": hadamard_k,  # input12, bf16
        "idx_k_cache": k_cache,  # input13, int8  # (block_num, block_size, n_kv, d)
        "idx_k_scale_cache": k_scale_cache,  # input14, fp16  # (block_num, block_size, n_kv, 1)
        "idx_k_cache_index": k_cache_index,  # input15, int64  (b, s)/（t,)
        "idx_block_table": block_table,  # input16, int32  (b, ceil(s2, block_size))
        "act_seq": act_seq,  # input17, int32
    }


def scatter_update_pa_bsnd(cache, k_bsnd, cache_index, axis):
    block_number, block_size, n_kv, d = cache.shape
    res = cache.reshape(block_number * block_size * n_kv, d)
    b, s1 = cache_index.shape

    if axis == -2:
        for b_i in range(b):
            for s1_i in range(s1):
                index_value = cache_index[b_i][s1_i]
                res[index_value, :] = k_bsnd[b_i, s1_i, :, :]

    return res.reshape(block_number, block_size, n_kv, d)


def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x_dtype = x.dtype
    if x_dtype != torch.float32:
        x = x.to(torch.float32)
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x = (x - mean) / torch.sqrt(var + eps)
    return (x * gamma.to(torch.float32) + beta.to(torch.float32)).to(x_dtype)


def quant_int8(x: torch.Tensor):
    # pertoken
    x_dtype = x.dtype  # bf16, (b, s, n, d)
    x_fp32 = x.to(torch.float32)
    max_value = torch.amax(torch.abs(x_fp32), dim=-1, keepdim=True)
    scale_quant = 127.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_fp32 = y_fp32.view(x.shape)
    y_int32 = torch.round(y_fp32).to(torch.int32)  # rint mode
    y_int8 = torch.trunc(y_int32.to(x_dtype)).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    # (b, s, n, d) int8, (b, s, n, 1) fp32
    return y_int8, scale_dequant


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def single_rope(x, cos_in, sin_in):
    logging.info("Entering into single_rope")
    # x: (b, s, n, d), cos_in: (b, s, d), sin_in: (b, s, d)
    x_dtype = x.dtype
    b, s, n, d = x.shape
    x_cast = x.to(torch.float32)
    cos_cast = cos_in.to(torch.float32)
    sin_cast = sin_in.to(torch.float32)
    cos_re = cos_cast.unsqueeze(2)  # (b, s, 1, d)
    sin_re = sin_cast.unsqueeze(2)  # (b, s, 1, d)
    res = x_cast * cos_re + rotate_half(x_cast) * sin_re  # (b, s, n, d)
    return res.to(x_dtype)


def _indexer_compute_q(inputs, dims):
    """Compute query (q) path: matmul -> dequant -> rope -> hadamard -> quant."""
    b, t, n, d = dims["b"], dims["t"], dims["idx_n_heads"], dims["idx_head_dim"]
    s = t // b
    rope_head_dim = dims["rope_head_dim"]
    x_dtype = inputs["token_x"].dtype

    q_norm = inputs["q_norm"]
    q_norm_scale = inputs["q_norm_scale"]
    w_idx_qb = inputs["w_idx_qb"]
    w_idx_qb_scale = inputs["w_idx_qb_scale"]
    cos = inputs["cos_idx_rope"]
    sin = inputs["sin_idx_rope"]
    hadamard_q = inputs["hadamard_q"]

    q = torch.matmul(q_norm.to(torch.int32), w_idx_qb.to(torch.int32))  # (b, s, n * d)
    q_fp32 = q.to(torch.float32)
    q_fp32 = q_fp32 * q_norm_scale
    q_fp32 = q_fp32 * w_idx_qb_scale.reshape(1, n * d)
    q_bf16 = q_fp32.reshape(b, s, n, d).to(torch.bfloat16)
    q_rope, q_nope = torch.split(q_bf16, [rope_head_dim, d - rope_head_dim], dim=-1)
    q_rope = single_rope(q_rope, cos, sin)
    q = torch.cat([q_rope, q_nope], dim=-1)
    q = torch.matmul(q.to(torch.float32), hadamard_q.to(torch.float32)).to(x_dtype)
    q_int8, q_scale = quant_int8(q)
    return q_int8, q_scale.to(torch.float16)


def _indexer_compute_k(inputs, dims):
    """Compute key (k) path: matmul -> layernorm -> rope -> hadamard -> quant -> cache update."""
    b, t, n, d = dims["b"], dims["t"], dims["idx_n_heads"], dims["idx_head_dim"]
    s = t // b
    rope_head_dim = dims["rope_head_dim"]
    x_dtype = inputs["token_x"].dtype

    x = inputs["token_x"]
    w_idx_k = inputs["w_idx_k"]
    cos = inputs["cos_idx_rope"]
    sin = inputs["sin_idx_rope"]
    layer_norm_gamma = inputs["layer_norm_gamma"]
    layer_norm_beta = inputs["layer_norm_beta"]
    hadamard_k = inputs["hadamard_k"]
    idx_k_cache = inputs["idx_k_cache"]
    idx_k_scale_cache = inputs["idx_k_scale_cache"]
    cache_index = inputs["idx_k_cache_index"]

    k = torch.matmul(x.to(torch.float32), w_idx_k.to(torch.float32))  # (b, s, d)
    k = layer_norm(k, layer_norm_gamma, layer_norm_beta).to(x_dtype)
    k_rope, k_nope = torch.split(k, [rope_head_dim, d - rope_head_dim], dim=-1)
    k_rope = single_rope(k_rope.unsqueeze(2), cos, sin).squeeze(2)
    k = torch.cat([k_rope, k_nope], dim=-1)
    k = torch.matmul(k.to(torch.float32), hadamard_k.to(torch.float32)).to(x_dtype)
    k_int8, k_scale = quant_int8(k)
    k_scale = k_scale.to(torch.float16)

    k_cache = idx_k_cache.clone()
    k_scale_cache = idx_k_scale_cache.clone()
    scatter_update_pa_bsnd(k_cache, k_int8.reshape(b, s, 1, d), cache_index, -2)
    scatter_update_pa_bsnd(k_scale_cache, k_scale.reshape(b, s, 1, 1), cache_index, -2)
    return k_cache, k_scale_cache


def _indexer_compute_weights(inputs, dims):
    """Compute attention weights path."""
    x = inputs["token_x"]
    w_idx_proj = inputs["w_idx_proj"]
    n = dims["idx_n_heads"]
    d = dims["idx_head_dim"]
    x_dtype = x.dtype

    # matmul use float32 for arm, arm平台matmul在bfloat16数据类型下表现跟x86不一致，通过升精度保证正确性
    weights = torch.matmul(x.to(torch.float32),
                           w_idx_proj.to(torch.float32)).to(x_dtype).to(torch.float32)  # (b, s, n)
    weights = weights * (n ** -0.5) * (d ** -0.5)
    return weights.to(torch.float16)


def indexer_prolog(inputs: dict, dims: dict):
    x_dtype = inputs["token_x"].dtype

    q_int8, q_scale = _indexer_compute_q(inputs, dims)
    k_cache, k_scale_cache = _indexer_compute_k(inputs, dims)
    weights = _indexer_compute_weights(inputs, dims)

    outputs = {"query": q_int8, "query_scale": q_scale,
               "idx_k_cache_out": k_cache, "idx_k_scale_cache_out": k_scale_cache,
               "weights": weights}
    return outputs


def gen_data(case_name):
    if case_name.startswith("QuantLightningIndexerPrologSTest.b4_s1_2_s2_64k"):
        params = {
            "b": 4,
            "s1": 2,
            "s2": 1024 * 64
        }
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b8_s1_2_s2_64k"):
        params = {
            "b": 8,
            "s1": 2,
            "s2": 1024 * 64
        }
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b1_s1_4k_s2_64k"):
        params = {
            "b": 1,
            "s1": 1024 * 4,
            "s2": 1024 * 64
        }
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b2_s1_4k_s2_64k"):
        params = {
            "b": 2,
            "s1": 1024 * 4,
            "s2": 1024 * 64
        }
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b128_s1_4_s2_8k"):
        params = {
            "b": 128,
            "s1": 4,
            "s2": 1024 * 8
        }
    else:
        raise Exception(f"Can't get func to gen golden, Case({case_name})")

    seed = 7
    # PyTorch 随机数生成器
    torch.manual_seed(seed)
    dims = gen_dims(params)
    inputs = gen_inputs(dims, torch.bfloat16)
    outputs = indexer_prolog(inputs, dims)
    return dims, inputs, outputs


def gen_zero_tensor(t):
    return torch.zeros_like(t).npu()


def _build_indexer_pytorch_inputs(inputs_data, dims):
    """Build IndexerPrologQuantInput from raw data dict."""
    t = dims["t"]
    h = dims["h"]
    q_lora_rank = dims["q_lora_rank"]
    rope_head_dim = dims["rope_head_dim"]
    return IndexerPrologQuantInput(
        x=inputs_data["token_x"].npu().reshape(t, h),
        q_norm=inputs_data["q_norm"].npu().reshape(t, q_lora_rank),
        q_norm_scale=inputs_data["q_norm_scale"].npu().reshape(t, 1),
        w_qb=torch_npu.npu_format_cast(inputs_data["w_idx_qb"].npu().contiguous(), torch_npu.Format.FRACTAL_NZ),
        w_qb_scale=inputs_data["w_idx_qb_scale"].npu(),
        wk=torch_npu.npu_format_cast(inputs_data["w_idx_k"].npu().contiguous(), torch_npu.Format.FRACTAL_NZ),
        w_proj=torch_npu.npu_format_cast(
            inputs_data["w_idx_proj"].npu().contiguous(), torch_npu.Format.FRACTAL_NZ),
        ln_gamma_k=inputs_data["layer_norm_gamma"].npu(),
        ln_beta_k=inputs_data["layer_norm_beta"].npu(),
        cos_idx_rope=inputs_data["cos_idx_rope"].npu().reshape(t, rope_head_dim),
        sin_idx_rope=inputs_data["sin_idx_rope"].npu().reshape(t, rope_head_dim),
        hadamard_q=inputs_data["hadamard_q"].npu(),
        hadamard_k=inputs_data["hadamard_k"].npu(),
        k_cache=inputs_data["idx_k_cache"].npu(),
        k_cache_scale=inputs_data["idx_k_scale_cache"].npu(),
        k_cache_index=inputs_data["idx_k_cache_index"].npu().reshape(t)
    )


@dataclass
class _BuildIndexerOutputsOutputs:
    outputs: Any
    q_int8_golden: Any
    q_scale_golden: Any
    k_cache_golden: Any
    k_cache_scale_golden: Any
    weights_golden: Any



def _build_indexer_outputs(golden_data, inputs, dims):


    """Build IndexerPrologQuantOutput from golden data."""
    t = dims["t"]
    head_num = dims["idx_n_heads"]
    idx_head_dim = dims["idx_head_dim"]

    q_int8_golden = golden_data["query"].reshape(t, head_num, idx_head_dim)
    q_scale_golden = golden_data["query_scale"].reshape(t, head_num, 1)
    k_cache_golden = golden_data["idx_k_cache_out"]
    k_cache_scale_golden = golden_data["idx_k_scale_cache_out"]
    weights_golden = golden_data["weights"].reshape(t, head_num)

    outputs = IndexerPrologQuantOutput(
        q_int8=gen_zero_tensor(q_int8_golden),
        q_scale=gen_zero_tensor(q_scale_golden),
        k_int8=inputs.k_cache,
        k_scale=inputs.k_cache_scale,
        weights=gen_zero_tensor(weights_golden)
    )
    return _BuildIndexerOutputsOutputs(
    outputs, q_int8_golden, q_scale_golden,
    k_cache_golden, k_cache_scale_golden, weights_golden)


def _run_and_compare(inputs, outputs, configs, q_int8_golden, q_scale_golden,
                     k_cache_golden, k_cache_scale_golden, weights_golden):
    """Execute kernel and compare with golden outputs."""
    attrs = IndexerPrologQuantAttr(
        eps=1e-6,
        layerout_query="TND",
        layerout_key="PA_BSND",
    )
    tensors = [tensor for _, tensor in vars(inputs).items()] + \
              [tensor for _, tensor in vars(outputs).items()]
    lightning_indexer_prolog_quant(*tensors, configs, attrs)

    compare(outputs.q_int8.cpu(), q_int8_golden, "q_int8", atol=1, rtol=0, max_error_ratio=0)
    compare(outputs.q_scale.cpu(), q_scale_golden, "q_scale", atol=0.000025, rtol=0, max_error_ratio=0.005)
    compare(outputs.k_int8.cpu(), k_cache_golden, "k_int8", atol=1, rtol=0, max_error_ratio=0)
    compare(outputs.k_scale.cpu(), k_cache_scale_golden, "k_scale", atol=0.000025, rtol=0, max_error_ratio=0)
    compare(outputs.weights.cpu(), weights_golden, "weights", atol=0.000025, rtol=0., max_error_ratio=0)


def do_test_lighting_indexer_prolog_quant(case_name, configs):
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    print(f"=== run test case: {case_name} ===")

    dims, inputs_data, golden_data = gen_data(case_name)

    torch_npu.npu.config.allow_internal_format = True

    inputs = _build_indexer_pytorch_inputs(inputs_data, dims)
    out = _build_indexer_outputs(golden_data, inputs, dims)
    outputs = out.outputs
    q_int8_golden = out.q_int8_golden
    q_scale_golden = out.q_scale_golden
    k_cache_golden = out.k_cache_golden
    k_cache_scale_golden = out.k_cache_scale_golden
    weights_golden = out.weights_golden

    _run_and_compare(inputs, outputs, configs, q_int8_golden, q_scale_golden,
                     k_cache_golden, k_cache_scale_golden, weights_golden)

    print(f"=== {case_name}: PASS ===")


@pytest.mark.soc("950", "910")
@pytest.mark.skip(reason="large test case")
def test_b4_s1_2_s2_64k():
    configs = IndexerPrologQuantConfigs(
        q_linear=[16, 16, 512, 512, 128, 128],
        q_hd=[32, 32, 128, 128, 128, 128],
        k_linear=[16, 16, 512, 512, 64, 64],
        w_linear=[16, 16, 1024, 1024, 32, 32],
        unroll_list=[32, 16, 8, 4, 2, 1],
        cube_l1_reuse_setting={1: 4},
        block_size=128,
        t_sub_tile=1,
        chunk_size=2,
        vec_nbuffer_setting={-1: 1},
    )
    do_test_lighting_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b4_s1_2_s2_64k", configs)


@pytest.mark.skip(reason="large test case")
def test_b8_s1_2_s2_64k():
    configs = IndexerPrologQuantConfigs(
        q_linear=[16, 16, 512, 512, 128, 128],
        q_hd=[32, 32, 128, 128, 128, 128],
        k_linear=[16, 16, 512, 512, 64, 64],
        w_linear=[16, 16, 1024, 1024, 32, 32],
        unroll_list=[32, 16, 8, 4, 2, 1],
        cube_l1_reuse_setting={1: 4},
        block_size=128,
        t_sub_tile=1,
        chunk_size=2,
        vec_nbuffer_setting={-1: 1},
    )
    do_test_lighting_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b8_s1_2_s2_64k", configs)


@pytest.mark.skip(reason="large test case")
def test_b1_s1_4k_s2_64k():
    configs = IndexerPrologQuantConfigs(
        q_linear=[16, 16, 512, 512, 128, 128],
        q_hd=[32, 32, 128, 128, 128, 128],
        k_linear=[16, 16, 512, 512, 64, 64],
        w_linear=[16, 16, 1024, 1024, 32, 32],
        unroll_list=[32, 16, 8, 4, 2, 1],
        cube_l1_reuse_setting={1: 4},
        block_size=128,
        t_sub_tile=1,
        chunk_size=2,
        vec_nbuffer_setting={-1: 1},
    )
    do_test_lighting_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b1_s1_4k_s2_64k", configs)


@pytest.mark.skip(reason="large test case")
def test_b2_s1_4k_s2_64k():
    configs = IndexerPrologQuantConfigs(
        q_linear=[16, 16, 512, 512, 128, 128],
        q_hd=[32, 32, 128, 128, 128, 128],
        k_linear=[16, 16, 512, 512, 64, 64],
        w_linear=[16, 16, 1024, 1024, 32, 32],
        unroll_list=[32, 16, 8, 4, 2, 1],
        cube_l1_reuse_setting={1: 4},
        block_size=128,
        t_sub_tile=1,
        chunk_size=2,
        vec_nbuffer_setting={-1: 1},
    )
    do_test_lighting_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b2_s1_4k_s2_64k", configs)


@pytest.mark.skip(reason="large test case")
def test_b128_s1_4_s2_8k():
    configs = IndexerPrologQuantConfigs(
        q_linear=[128, 128, 256, 256, 256, 256],
        q_hd=[128, 128, 64, 64, 128, 128],
        k_linear=[64, 64, 256, 256, 128, 128],
        w_linear=[32, 32, 512, 512, 64, 64],
        unroll_list=[128, 64, 32, 16, 8, 4, 2, 1],
        cube_l1_reuse_setting={1: 4, 3: 4},
        block_size=128,
        t_sub_tile=2,
        chunk_size=1,
        vec_nbuffer_setting={-1: 1},
    )
    do_test_lighting_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b128_s1_4_s2_8k", configs)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO
    )
    test_b4_s1_2_s2_64k()
