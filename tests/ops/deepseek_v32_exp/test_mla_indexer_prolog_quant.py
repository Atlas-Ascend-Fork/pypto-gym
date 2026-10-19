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
import sys
import math
import logging
from pathlib import Path
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
from deepseek_v32_exp.mla_prolog_quant_impl import MlaTileConfig
from common_utils import compare

PRINT_DEBUG = False


def prep_env():

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    torch_npu.npu.config.allow_internal_format = True


def quant(input_t, is_pertoken: bool = True, has_smooth=False, smooth_cq=None):
    input_fp32 = input_t.to(torch.float32)
    if has_smooth:
        input_fp32 = input_fp32 * smooth_cq
    abs_res = torch.abs(input_fp32)
    reduce_idx = -1
    if not is_pertoken:
        reduce_idx = -2
        logging.debug("This PerChannel Quant!!")

    max_value = torch.max(abs_res, dim=reduce_idx, keepdims=True)[0]
    scale_quant = 127 / max_value
    out_fp32 = input_fp32 * scale_quant
    out_int32 = torch.round(out_fp32).to(torch.int32)
    out_fp16 = out_int32.to(torch.float16)
    out_int8 = torch.trunc(out_fp16).to(torch.int8)
    scale_dequant = 1 / scale_quant

    return out_int8, scale_dequant


def rms_norm(x, gamma):
    x_dtype = x.dtype
    mean_coff = 1.0 / x.shape[-1]

    x_f32 = x.to(torch.float32)
    square = x_f32 * x_f32
    mean_res = square * mean_coff

    reduce_sum = torch.sum(mean_res, dim=-1, keepdims=True)
    reduce_sqrt = torch.sqrt(reduce_sum)
    res_div = x_f32 / reduce_sqrt

    res = res_div * gamma

    if x_dtype != torch.float32:
        res = res.to(x_dtype)
    return res


def scatter_update_4d(cache, key_states, indices, axis):
    # inputs: cache, key_states, indices
    # key_states shape: [b*s1*1, d]
    block_number, block_size, n2, d = cache.shape
    res = cache.reshape(block_number * block_size * n2, d)
    b, s1 = indices.shape

    if axis == -2:
        for b_i in range(b):
            for s1_i in range(s1):
                index_value = indices[b_i][s1_i]
                res[index_value][:] = key_states[b_i * s1 + s1_i][:]
    return res.reshape(block_number, block_size, n2, d)


def scatter_update_2d(cache, k_bsnd, cache_index, axis):
    block_number, block_size, n_kv, d = cache.shape
    res = cache.reshape(block_number * block_size * n_kv, d)
    b, s1 = cache_index.shape

    if axis == -2:
        for b_i in range(b):
            for s1_i in range(s1):
                index_value = cache_index[b_i][s1_i]
                res[index_value, :] = k_bsnd[b_i, s1_i, :, :]

    return res.reshape(block_number, block_size, n_kv, d)


def apply_rotary_pos_emb_v2(q, k, cos, sin, unsqueeze_dim=2):
    input_dtype = q.dtype
    if input_dtype != torch.float32:
        q = q.to(torch.float32)
        k = k.to(torch.float32)
    if cos.dtype != torch.float32:
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)

    cos = torch.unsqueeze(cos, dim=unsqueeze_dim)  # [b,s,1,qk_d]
    sin = torch.unsqueeze(sin, dim=unsqueeze_dim)  # [b,s,1,qk_d]

    b, s, h, d = q.shape
    q = q.reshape(b, s, h, d // 2, 2).permute(0, 1, 2, 4, 3).reshape(b, s, h, d)  # [b,s,n,qk_d]

    b, s, h, d = k.shape
    k = k.reshape(b, s, h, d // 2, 2).permute(0, 1, 2, 4, 3).reshape(b, s, h, d)  # [b,s,1,qk_d]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    if input_dtype != torch.float32:
        q_embed, k_embed = q_embed.to(input_dtype), k_embed.to(input_dtype)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def single_rope(x, cos_in, sin_in):
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


def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x_dtype = x.dtype
    if x_dtype != torch.float32:
        x = x.to(torch.float32)
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x = (x - mean) / torch.sqrt(var + eps)
    return (x * gamma.to(torch.float32) + beta.to(torch.float32)).to(x_dtype)


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

    block_table = -torch.ones(block_table_shape, dtype=torch.int32)

    block_idx = 0
    block_table_bidx = 0
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


def _build_kv_cache(k_bsnd, b, block_size, block_table, block_num, kv_lora_rank,
                     qk_rope_head_dim, dtype):
    skv_max = k_bsnd.shape[1]
    per_batch_max_num = math.ceil(skv_max / block_size)
    k_tensor_bsnd = torch.zeros((b, per_batch_max_num * block_size, 1, kv_lora_rank + qk_rope_head_dim)).to(dtype)
    k_tensor_bsnd[:, :k_bsnd.shape[1], :, :] = k_bsnd[:, :, :, :]
    k_cache_tensor = torch.zeros([block_num, block_size, 1, kv_lora_rank + qk_rope_head_dim]).to(dtype)
    for b_idx in range(b):
        for block_i, kv_cache_blk_id in enumerate(block_table[b_idx]):
            block_offset = block_i * block_size
            if kv_cache_blk_id == -1:
                continue
            else:
                k_cache_tensor[kv_cache_blk_id, 0:block_size, :, :] = k_tensor_bsnd[
                    b_idx, block_offset:(block_offset + block_size), :, :]
    kv_cache = k_cache_tensor[:, :, :, : kv_lora_rank]
    kr_cache = k_cache_tensor[:, :, :, kv_lora_rank:]
    return kv_cache, kr_cache


def gen_mla_prolog_quant_v32_inputs(params, dtypes, actual_seq, is_quant=(False, False),
                                    is_nz=False, has_smooth=False, block_size=128, cache_mode='BSND'):
    dtype, w_dtype = dtypes
    is_quant_a, is_quant_b = is_quant
    b, s, s1, h = params.get('b'), params.get('s'), params.get('s1'), params.get('h')
    n = params.get('num_heads')
    q_lora_rank = params.get('q_lora_rank')
    qk_nope_head_dim = params.get('qk_nope_head_dim')
    qk_rope_head_dim = params.get('qk_rope_head_dim')
    kv_lora_rank = params.get('kv_lora_rank')
    block_num, block_table, cache_index = gen_block_table(actual_seq, block_size, s1)

    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    res = [None] * 17
    res[0] = torch.empty([b, s, h]).uniform_(-1, 1).to(dtype)
    res[4] = dict()

    w_dq, w_uqqr, w_dkvkr, _res = _prepare_weights(_PrepareWeightsInputs(
        h, q_lora_rank, n, q_head_dim, kv_lora_rank, qk_rope_head_dim,
        w_dtype, is_quant_a, is_quant_b, is_nz, has_smooth, res))
    if _res is not None:
        res = _res

    res[1], res[2], res[5] = w_dq, w_uqqr, w_dkvkr
    w_uk = torch.empty([n, qk_nope_head_dim, kv_lora_rank]).uniform_(-0.1, 0.1).to(w_dtype)
    res[6] = w_uk
    res[7] = torch.empty([q_lora_rank]).uniform_(-1, 1).to(dtype)
    res[8] = torch.empty([kv_lora_rank]).uniform_(-1, 1).to(dtype)
    res[9] = torch.empty([b, s, qk_rope_head_dim]).uniform_(-0.1, 0.1).to(dtype)
    res[10] = torch.empty([b, s, qk_rope_head_dim]).uniform_(-0.1, 0.1).to(dtype)
    res[11] = cache_index
    k_bsnd = torch.empty([b, actual_seq.max(), 1, kv_lora_rank + qk_rope_head_dim]).uniform_(-1, 1).to(dtype)
    kv_cache, kr_cache, kv_quant_scale_cache = _prepare_kv_cache(
        k_bsnd, b, block_size, block_table, block_num, kv_lora_rank, qk_rope_head_dim, dtype, is_quant_b)
    res[12], res[13], res[14], res[15], res[16] = kv_cache, kr_cache, kv_quant_scale_cache, block_num, block_table

    return res


@dataclass
class _PrepareWeightsInputs:
    h: Any
    q_lora_rank: Any
    n: Any
    q_head_dim: Any
    kv_lora_rank: Any
    qk_rope_head_dim: Any
    w_dtype: Any
    is_quant_a: Any
    is_quant_b: Any
    is_nz: Any
    has_smooth: Any
    res: Any


def _prepare_weights(inputs: _PrepareWeightsInputs):


    """Prepare and quantize weight matrices for MLA prolog."""
    w_dq = torch.empty([inputs.h, inputs.q_lora_rank]).uniform_(-0.1, 0.1).to(inputs.w_dtype)
    w_uqqr = torch.empty([inputs.q_lora_rank, inputs.n * inputs.q_head_dim]).uniform_(-0.1, 0.1).to(inputs.w_dtype)
    w_dkvkr = torch.empty(
            [inputs.h, inputs.kv_lora_rank + inputs.qk_rope_head_dim]
            ).uniform_(-0.1, 0.1).to(inputs.w_dtype)

    if inputs.is_quant_a:
        w_dq, w_qa_scale = quant(w_dq, False)
        w_dkvkr, w_kva_scale = quant(w_dkvkr, False)
        inputs.res[4]['w_dq'] = w_qa_scale
        inputs.res[4]['w_dkvkr'] = w_kva_scale
        if inputs.is_nz:
            w_dq = w_dq.reshape(inputs.h, inputs.q_lora_rank // 32, 32).permute(1, 0, 2)
            w_dkvkr = w_dkvkr.reshape(
            inputs.h, (inputs.kv_lora_rank + inputs.qk_rope_head_dim) // 32, 32
            ).permute(1, 0, 2)
    elif inputs.is_nz:
        w_dq = w_dq.reshape(inputs.h, inputs.q_lora_rank // 16, 16).permute(1, 0, 2)
        w_dkvkr = w_dkvkr.reshape(inputs.h, (inputs.kv_lora_rank + inputs.qk_rope_head_dim) // 16, 16).permute(1, 0, 2)

    if inputs.is_quant_b:
        w_uqqr, w_qb_scale = quant(w_uqqr, False)
        inputs.res[4]['w_uqqr'] = w_qb_scale
        if inputs.has_smooth:
            smooth_cq = torch.empty([1, inputs.q_lora_rank]).uniform_(-1, 1).to(torch.float32)
            inputs.res[3] = smooth_cq
        if inputs.is_nz:
            w_uqqr = w_uqqr.reshape(inputs.q_lora_rank, inputs.n * inputs.q_head_dim // 32, 32).permute(1, 0, 2)
    return w_dq, w_uqqr, w_dkvkr, inputs.res


def _prepare_kv_cache(k_bsnd, b, block_size, block_table, block_num,
                       kv_lora_rank, qk_rope_head_dim, dtype, is_quant_b):
    """Prepare KV cache tensors with optional quantization."""
    kv_cache, kr_cache = _build_kv_cache(k_bsnd, b, block_size, block_table, block_num,
                                          kv_lora_rank, qk_rope_head_dim, dtype)
    kv_quant_scale_cache = None
    if is_quant_b:
        kv_cache_shape = [block_num, block_size, 1, kv_lora_rank]
        kv_cache_split = kv_cache.reshape(-1, 4, kv_lora_rank // 4)
        kv_cache, kv_quant_scale_cache = quant(kv_cache_split, True)
        kv_cache = kv_cache.reshape(kv_cache_shape)
        kv_quant_scale_cache = kv_quant_scale_cache.reshape([block_num, block_size, 1, 4])
    return kv_cache, kr_cache, kv_quant_scale_cache


def gen_indexer_prolog_inputs(params, block_num, block_table, mla_inputs, mla_goldens):
    q_lora_rank = params['q_lora_rank']
    b = params['b']
    s2 = params['s2']
    h = params['h']
    n2 = params['n2']
    idx_head_dim = params['idx_head_dim']
    dtype = params['dtype']
    block_size = params['block_size']
    idx_n_heads = params['idx_n_heads']
    quant_dtype = torch.int8

    x = mla_inputs['x']
    cos = mla_inputs['cos']
    sin = mla_inputs['sin']
    cache_index = mla_inputs['cache_index']
    rms_norm_out = mla_goldens['rms_norm_out']
    rms_norm_scale_out = mla_goldens['rms_norm_scale_out']

    w_idx_qb = torch.randint(low=-128, high=128, size=(q_lora_rank, idx_n_heads * idx_head_dim), dtype=quant_dtype)
    w_idx_qb_scale = torch.empty((1, idx_n_heads * idx_head_dim), dtype=torch.float32).uniform_(-1, 1)
    w_idx_k = torch.empty((h, idx_head_dim), dtype=dtype).uniform_(-1, 1)
    w_idx_proj = torch.empty((h, idx_n_heads), dtype=dtype).uniform_(-1, 1)

    ln_gamma = torch.ones((idx_head_dim,), dtype=dtype)
    ln_beta = torch.zeros((idx_head_dim,), dtype=dtype)

    hadamard_q = torch.empty((idx_head_dim, idx_head_dim), dtype=dtype).uniform_(-1, 1)  # (128, 128)
    hadamard_k = torch.empty((idx_head_dim, idx_head_dim), dtype=dtype).uniform_(-1, 1)

    k_cache_bsnd = torch.rand((b * s2 * n2, idx_head_dim), dtype=torch.float32) * 2 - 1
    k_cache_bsnd, k_scale_cache_bsnd = quant(k_cache_bsnd)
    k_cache_bsnd = k_cache_bsnd.reshape(b, s2, n2, idx_head_dim).to(dtype=quant_dtype)
    k_scale_cache_bsnd = k_scale_cache_bsnd.reshape(b, s2, n2, 1).to(torch.float16)

    k_cache = gen_cache_tensor(k_cache_bsnd, block_table, block_num, block_size)
    k_scale_cache = gen_cache_tensor(k_scale_cache_bsnd, block_table, block_num, block_size)

    return {
        'token_x': x,  # input0, bf16
        'q_norm': rms_norm_out,  # input1, int8
        'q_norm_scale': rms_norm_scale_out,  # input2, fp32
        'w_idx_qb': w_idx_qb,  # input3, int8
        'w_idx_qb_scale': w_idx_qb_scale,  # input4, fp32
        'w_idx_k': w_idx_k,  # input5, bf16
        'w_idx_proj': w_idx_proj,  # input6, bf16
        'layer_norm_gamma': ln_gamma,  # input7, bf16
        'layer_norm_beta': ln_beta,  # input8, bf16
        'cos_idx_rope': cos,  # input9, bf16
        'sin_idx_rope': sin,  # input10, bf16
        'hadamard_q': hadamard_q,  # input11, bf16
        'hadamard_k': hadamard_k,  # input12, bf16
        'idx_k_cache': k_cache,  # input13, int8  # (block_num, block_size, n_kv, d)
        'idx_k_scale_cache': k_scale_cache,  # input14, fp16  # (block_num, block_size, n_kv, 1)
        'idx_k_cache_index': cache_index,  # input15, int64  (b, s)/（t,)
        'idx_block_table': block_table,  # input16, int32  (b, ceil(s2, block_size))
    }


@dataclass
class _MlaComputeQPathInputs:
    x: Any
    w_dq: Any
    w_uqqr: Any
    w_uk: Any
    gamma_cq: Any
    cos: Any
    is_quant_a: Any
    is_quant_b: Any
    has_smooth: Any
    smooth_cq: Any
    w_qa_scale: Any
    w_qb_scale: Any
    dtype: Any


def _mla_compute_q_path(inputs: _MlaComputeQPathInputs):


    b, s, h = inputs.x.shape
    qk_rope_head_dim = inputs.cos.shape[2]
    n, qk_nope_head_dim, kv_lora_rank = inputs.w_uk.shape
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    x_2d = inputs.x.reshape(b * s, h)
    if inputs.is_quant_a:
        x_2d_quant, x_2d_scale_dequant = quant(x_2d, True)
        q_a_proj = torch.matmul(x_2d_quant.to(torch.int32), inputs.w_dq.to(torch.int32))
        q_a_proj_fp32 = q_a_proj.to(torch.float32)
        q_a_proj_fp32_dequant = q_a_proj_fp32 * x_2d_scale_dequant
        q_a_proj = q_a_proj_fp32_dequant * inputs.w_qa_scale
    else:
        q_a_proj = torch.matmul(x_2d.to(torch.float32), inputs.w_dq.to(torch.float32))
    q_a_proj = q_a_proj.to(torch.bfloat16)
    q_a_layernorm = rms_norm(q_a_proj, inputs.gamma_cq)
    q_a_layernorm_scale_dequant = None
    if inputs.is_quant_b:
        if inputs.has_smooth:
            q_a_layernorm, q_a_layernorm_scale_dequant = quant(q_a_layernorm, True, True, inputs.smooth_cq)
        else:
            q_a_layernorm, q_a_layernorm_scale_dequant = quant(q_a_layernorm, True)
        q_b_proj = torch.matmul(q_a_layernorm.to(torch.int32), inputs.w_uqqr.to(torch.int32)).to(q_a_layernorm.device)
        q_b_proj_fp32 = q_b_proj.to(torch.float32)
        q_b_proj_fp32_dequant = q_b_proj_fp32 * q_a_layernorm_scale_dequant
        q_b_proj = q_b_proj_fp32_dequant * inputs.w_qb_scale
    else:
        q_b_proj = torch.matmul(q_a_layernorm.to(torch.float32), inputs.w_uqqr.to(torch.float32))
    q_b_proj = q_b_proj.to(inputs.dtype)
    q_reshape = q_b_proj.reshape(b, s, n, q_head_dim)
    q_nope = q_reshape[:, :, :, 0:qk_nope_head_dim]
    q_nope_r = q_nope.reshape(b * s, n, qk_nope_head_dim)
    q_nope_t = q_nope_r.permute(1, 0, 2)
    q_nope_new = torch.matmul(q_nope_t.to(torch.float32), inputs.w_uk.to(torch.float32))
    q_nope_new = q_nope_new.to(inputs.dtype)
    q_nope_new_t = q_nope_new.permute(1, 0, 2)
    q_nope = q_nope_new_t.reshape(b, s, n, kv_lora_rank)
    return q_nope, q_reshape, q_a_layernorm, q_a_layernorm_scale_dequant, x_2d


@dataclass
class _MlaComputeKvPathInputs:
    x_2d: Any
    w_dkvkr: Any
    gamma_ckv: Any
    is_quant_a: Any
    w_kva_scale: Any
    is_quant_b: Any
    dtype: Any
    b: Any
    s: Any
    kv_lora_rank: Any
    qk_rope_head_dim: Any


def _mla_compute_kv_path(inputs: _MlaComputeKvPathInputs):

    bs, h = inputs.x_2d.shape
    if inputs.is_quant_a:
        x_2d_quant, x_2d_scale_dequant = quant(inputs.x_2d, True)
        kv_a_proj = torch.matmul(x_2d_quant.to(torch.int32), inputs.w_dkvkr.to(torch.int32))
        kv_a_proj_fp32 = kv_a_proj.to(torch.float32)
        kv_a_proj_fp32_dequant = kv_a_proj_fp32 * x_2d_scale_dequant
        kv_a_proj = kv_a_proj_fp32_dequant * inputs.w_kva_scale
    else:
        kv_a_proj = torch.matmul(inputs.x_2d.to(torch.float32), inputs.w_dkvkr.to(torch.float32))
    kv_a_proj = kv_a_proj.to(inputs.dtype)
    kv_reshape = kv_a_proj.reshape(inputs.b, inputs.s, inputs.kv_lora_rank + inputs.qk_rope_head_dim)
    compressed_kv = kv_reshape[:, :, 0:inputs.kv_lora_rank]
    compressed_kv_norm = rms_norm(compressed_kv, inputs.gamma_ckv)
    compressed_kv_quant_scale = None
    if inputs.is_quant_b:
        compressed_kv_norm_split = compressed_kv_norm.reshape(inputs.b * inputs.s, 4, inputs.kv_lora_rank // 4)
        compressed_kv_norm, compressed_kv_quant_scale = quant(compressed_kv_norm_split, True)
        compressed_kv_quant_scale = compressed_kv_quant_scale.reshape(inputs.b, inputs.s, 1, 4)
    compressed_kv_r = compressed_kv_norm.reshape(inputs.b, inputs.s, 1, inputs.kv_lora_rank)
    k_nope = compressed_kv_r.reshape(inputs.b * inputs.s * 1, inputs.kv_lora_rank)
    return kv_a_proj, kv_reshape, k_nope, compressed_kv_quant_scale


@dataclass
class _MlaComputeRopeCacheInputs:
    q_reshape: Any
    kv_reshape: Any
    cos: Any
    sin: Any
    k_nope: Any
    compressed_kv_quant_scale: Any
    kv_cache: Any
    kr_cache: Any
    cache_index: Any
    kv_quant_scale_cache: Any
    is_quant_b: Any


def _mla_compute_rope_cache(inputs: _MlaComputeRopeCacheInputs):


    qk_nope_head_dim = inputs.q_reshape.shape[-1] - inputs.cos.shape[2]
    qk_rope_head_dim = inputs.cos.shape[2]
    kv_lora_rank = inputs.kv_reshape.shape[-1] - qk_rope_head_dim
    b, s = inputs.q_reshape.shape[0], inputs.q_reshape.shape[1]
    q_pe = inputs.q_reshape[:, :, :, qk_nope_head_dim:]
    k_pe = inputs.kv_reshape[:, :, kv_lora_rank:]
    k_pe_r = k_pe.reshape(b, s, 1, qk_rope_head_dim)
    q_embed, k_embed = apply_rotary_pos_emb_v2(q_pe, k_pe_r, inputs.cos, inputs.sin, 2)
    k_embed_r = k_embed.reshape(b * 1 * s, qk_rope_head_dim)
    kv_cache_tmp = inputs.kv_cache.clone()
    kv_cache_out = scatter_update_4d(kv_cache_tmp, inputs.k_nope, inputs.cache_index, -2)
    kr_cache_tmp = inputs.kr_cache.clone()
    kr_cache_out = scatter_update_4d(kr_cache_tmp, k_embed_r, inputs.cache_index, -2)
    if inputs.is_quant_b:
        compressed_kv_quant_scale = inputs.compressed_kv_quant_scale.reshape(-1, 4)
        kv_quant_scale_cache_tmp = inputs.kv_quant_scale_cache.clone()
        kv_quant_scale_cache_out = \
            scatter_update_4d(kv_quant_scale_cache_tmp, compressed_kv_quant_scale, inputs.cache_index, -2)
    else:
        kv_quant_scale_cache_out = None
    return q_embed, kv_cache_out, kr_cache_out, kv_quant_scale_cache_out


def mla_prolog_quant_v32_compute(inputs):
    dtype = inputs.get('dtype')
    is_quant_a = inputs.get('is_quant_a')
    is_quant_b = inputs.get('is_quant_b')
    has_smooth = inputs.get('has_smooth')
    gamma_cq = inputs.get('gamma_cq')
    gamma_ckv = inputs.get('gamma_ckv')
    x = inputs.get('x')
    w_dq = inputs.get('w_dq')
    w_uqqr = inputs.get('w_uqqr')
    w_uk = inputs.get('w_uk')
    w_dkvkr = inputs.get('w_dkvkr')
    cos = inputs.get('cos')
    sin = inputs.get('sin')
    kv_cache = inputs.get('kv_cache')
    kr_cache = inputs.get('kr_cache')
    kv_quant_scale_cache = None
    if is_quant_b:
        kv_quant_scale_cache = inputs.get('kv_quant_scale_cache')
    cache_index = inputs.get('cache_index')
    smooth_cq = None
    w_qa_scale = None
    w_kva_scale = None
    w_qb_scale = None
    if is_quant_a:
        w_qa_scale = inputs.get('w_qa_scale')
        w_kva_scale = inputs.get('w_kva_scale')
    if is_quant_b:
        w_qb_scale = inputs.get('w_qb_scale')
        if has_smooth:
            smooth_cq = inputs.get('smooth_cq')

    q_nope, q_reshape, q_a_layernorm, q_a_layernorm_scale_dequant, x_2d = \
        _mla_compute_q_path(_MlaComputeQPathInputs(
            x, w_dq, w_uqqr, w_uk, gamma_cq, cos, is_quant_a, is_quant_b,
            has_smooth, smooth_cq, w_qa_scale, w_qb_scale, dtype))

    b, s, h = x.shape
    qk_rope_head_dim = cos.shape[2]
    n, qk_nope_head_dim, kv_lora_rank = w_uk.shape

    kv_a_proj, kv_reshape, k_nope, compressed_kv_quant_scale = \
        _mla_compute_kv_path(_MlaComputeKvPathInputs(
            x_2d=x_2d, w_dkvkr=w_dkvkr, gamma_ckv=gamma_ckv, is_quant_a=is_quant_a,
            w_kva_scale=w_kva_scale, is_quant_b=is_quant_b, dtype=dtype, b=b, s=s,
            kv_lora_rank=kv_lora_rank, qk_rope_head_dim=qk_rope_head_dim))

    q_embed, kv_cache_out, kr_cache_out, kv_quant_scale_cache_out = \
        _mla_compute_rope_cache(_MlaComputeRopeCacheInputs(
            q_reshape, kv_reshape, cos, sin, k_nope,
            compressed_kv_quant_scale, kv_cache, kr_cache,
            cache_index, kv_quant_scale_cache, is_quant_b))

    res = [q_nope, q_embed, q_a_layernorm, q_a_layernorm_scale_dequant, kv_cache_out,
            kr_cache_out, kv_quant_scale_cache_out]
    return res


def _indexer_q_path(q_norm, q_norm_scale, w_idx_qb, w_idx_qb_scale, n, d, x_dtype):
    q = torch.matmul(q_norm.to(torch.int32), w_idx_qb.to(torch.int32))
    q_fp32 = q.to(torch.float32)
    q_fp32 = q_fp32 * q_norm_scale
    q_fp32 = q_fp32 * w_idx_qb_scale.reshape(1, n * d)
    return q_fp32.reshape(q_norm.shape[0], q_norm.shape[1], n, d).to(torch.bfloat16)


def _indexer_q_rope_hadamard(q_bf16, cos, sin, rope_head_dim, hadamard_q, x_dtype):
    q_rope, q_nope = torch.split(q_bf16, [rope_head_dim, q_bf16.shape[-1] - rope_head_dim], dim=-1)
    q_rope = single_rope(q_rope, cos, sin)
    q = torch.cat([q_rope, q_nope], dim=-1)
    q = torch.matmul(q.to(torch.float32), hadamard_q.to(torch.float32)).to(x_dtype)
    q_int8, q_scale = quant(q)
    q_scale = q_scale.to(torch.float16)
    return q_int8, q_scale


@dataclass
class IndexerKPathInputs:
    x: Any
    w_idx_k: Any
    layer_norm_gamma: Any
    layer_norm_beta: Any
    cos: Any
    sin: Any
    rope_head_dim: Any
    hadamard_k: Any
    x_dtype: Any
    b: Any
    s: Any
    d: Any


def _indexer_k_path(inputs: IndexerKPathInputs):

    k = torch.matmul(inputs.x.to(torch.float32), inputs.w_idx_k.to(torch.float32))
    k = layer_norm(k, inputs.layer_norm_gamma, inputs.layer_norm_beta).to(inputs.x_dtype)
    k_rope, k_nope = torch.split(k, [inputs.rope_head_dim, inputs.d - inputs.rope_head_dim], dim=-1)
    k_rope = single_rope(k_rope.unsqueeze(2), inputs.cos, inputs.sin).squeeze(2)
    k = torch.cat([k_rope, k_nope], dim=-1)
    k = torch.matmul(k.to(torch.float32), inputs.hadamard_k.to(torch.float32)).to(inputs.x_dtype)
    k_int8, k_scale = quant(k)
    k_scale = k_scale.to(torch.float16)
    return k_int8, k_scale


def indexer_prolog(inputs: dict, dims: dict):
    b, t, n, d = dims['b'], dims['t'], dims['idx_n_heads'], dims['idx_head_dim']
    s = t // b
    rope_head_dim = dims['rope_head_dim']
    x = inputs['token_x']
    q_norm = inputs['q_norm']
    q_norm_scale = inputs['q_norm_scale']
    w_idx_qb = inputs['w_idx_qb']
    w_idx_qb_scale = inputs['w_idx_qb_scale']
    w_idx_k = inputs['w_idx_k']
    w_idx_proj = inputs['w_idx_proj']
    layer_norm_gamma = inputs['layer_norm_gamma']
    layer_norm_beta = inputs['layer_norm_beta']
    cos = inputs['cos_idx_rope']
    sin = inputs['sin_idx_rope']
    hadamard_q = inputs['hadamard_q']
    hadamard_k = inputs['hadamard_k']
    idx_k_cache = inputs['idx_k_cache']
    idx_k_scale_cache = inputs['idx_k_scale_cache']
    cache_index = inputs['idx_k_cache_index']
    x_dtype = x.dtype

    q_bf16 = _indexer_q_path(q_norm.reshape(b, s, -1), q_norm_scale.reshape(b, s, -1),
                             w_idx_qb, w_idx_qb_scale, n, d, x_dtype)
    q_int8, q_scale = _indexer_q_rope_hadamard(q_bf16, cos, sin, rope_head_dim, hadamard_q, x_dtype)

    k_int8, k_scale = _indexer_k_path(IndexerKPathInputs(
        x=x, w_idx_k=w_idx_k, layer_norm_gamma=layer_norm_gamma, layer_norm_beta=layer_norm_beta,
        cos=cos, sin=sin, rope_head_dim=rope_head_dim, hadamard_k=hadamard_k,
        x_dtype=x_dtype, b=b, s=s, d=d))
    k_cache = idx_k_cache.clone()
    k_scale_cache = idx_k_scale_cache.clone()
    scatter_update_2d(k_cache, k_int8.reshape(b, s, 1, d), cache_index, -2)
    scatter_update_2d(k_scale_cache, k_scale.reshape(b, s, 1, 1), cache_index, -2)

    weights = torch.matmul(x.to(torch.float32),
        w_idx_proj.to(torch.float32)).to(x_dtype).to(torch.float32)
    weights = weights * (n ** -0.5) * (d ** -0.5)
    weights = weights.to(torch.float16)

    outputs = {'q_int8': q_int8, 'q_scale': q_scale,
               'idx_k_cache_out': k_cache, 'idx_k_scale_cache_out': k_scale_cache,
               'weights': weights}
    return outputs


def _prepare_mla_data(params, dtypes):
    actual_seq = params['actual_seq']
    is_quant = (False, True)
    is_nz = False
    has_smooth = False
    block_size = params['block_size']
    cache_mode = 'PA_BSND'
    q_lora_rank = params['q_lora_rank']
    qk_rope_head_dim = params['qk_rope_head_dim']
    kv_lora_rank = params['kv_lora_rank']
    t = params['t']
    h = params['h']
    n1 = params['n1']
    dtype = params['dtype']

    (x, w_dq, w_uqqr, smooth_cq, scale_data, w_dkvkr, w_uk, gamma_cq, gamma_ckv, cos, sin, kv_len, kv_cache,
     kr_cache, kv_quant_scale_cache, block_num, block_table) = \
        gen_mla_prolog_quant_v32_inputs(params, dtypes, actual_seq, is_quant, is_nz,
                                        has_smooth, block_size, cache_mode)

    mla_inputs = {'dtype': dtype, 'is_quant_a': False, 'is_quant_b': True, 'has_smooth': False}
    mla_inputs['cache_mode'] = cache_mode
    mla_inputs['gamma_cq'] = gamma_cq
    mla_inputs['gamma_ckv'] = gamma_ckv
    mla_inputs['x'] = x
    mla_inputs['w_dq'] = w_dq
    mla_inputs['w_uqqr'] = w_uqqr
    mla_inputs['w_uk'] = w_uk
    mla_inputs['w_dkvkr'] = w_dkvkr
    mla_inputs['cos'] = cos
    mla_inputs['sin'] = sin
    mla_inputs['kv_cache'] = kv_cache
    mla_inputs['kr_cache'] = kr_cache
    mla_inputs['kv_quant_scale_cache'] = kv_quant_scale_cache
    mla_inputs['cache_index'] = kv_len
    mla_inputs['w_qb_scale'] = scale_data['w_uqqr']

    res = mla_prolog_quant_v32_compute(mla_inputs)
    q_nope, q_rope, rms_norm_out, rms_norm_scale_out, kv_cache_out, kr_cache_out, \
            kv_quant_scale_cache_out = res
    mla_goldens = dict(q_nope=q_nope, q_rope=q_rope, rms_norm_out=rms_norm_out,
                       rms_norm_scale_out=rms_norm_scale_out, kv_cache_out=kv_cache_out,
                       kr_cache_out=kr_cache_out, kv_quant_scale_cache_out=kv_quant_scale_cache_out)

    mla_inputs_npu = {}
    for k, v in mla_inputs.items():
        mla_inputs_npu[k] = v.npu().contiguous() if isinstance(v, torch.Tensor) else v
    mla_inputs_npu['x'] = mla_inputs_npu['x'].reshape(t, h).contiguous()
    mla_inputs_npu['cos'] = mla_inputs_npu['cos'].reshape(t, qk_rope_head_dim).contiguous()
    mla_inputs_npu['sin'] = mla_inputs_npu['sin'].reshape(t, qk_rope_head_dim).contiguous()
    mla_inputs_npu['cache_index'] = mla_inputs_npu['cache_index'].reshape(t).contiguous()
    mla_inputs_npu['w_qb_scale'] = mla_inputs_npu['w_qb_scale'].reshape(-1, 1).contiguous()
    mla_inputs_npu['w_dq'] = torch_npu.npu_format_cast(mla_inputs_npu['w_dq'], torch_npu.Format.FRACTAL_NZ)
    mla_inputs_npu['w_uqqr'] = torch_npu.npu_format_cast(mla_inputs_npu['w_uqqr'], torch_npu.Format.FRACTAL_NZ)
    mla_inputs_npu['w_dkvkr'] = torch_npu.npu_format_cast(mla_inputs_npu['w_dkvkr'], torch_npu.Format.FRACTAL_NZ)
    mla_goldens['q_nope'] = mla_goldens['q_nope'].reshape(t, n1, kv_lora_rank).contiguous().cpu()
    mla_goldens['q_rope'] = mla_goldens['q_rope'].reshape(t, n1, qk_rope_head_dim).contiguous().cpu()
    return mla_inputs_npu, mla_goldens, mla_inputs, block_num, block_table


def _prepare_indexer_data(params, block_num, block_table, mla_inputs, mla_goldens):
    t = params['t']
    h = params['h']
    head_num = params['idx_n_heads']
    idx_head_dim = params['idx_head_dim']
    q_lora_rank = params['q_lora_rank']
    rope_head_dim = params['rope_head_dim']

    ip_inputs = gen_indexer_prolog_inputs(params, block_num, block_table, mla_inputs, mla_goldens)
    ip_goldens = indexer_prolog(ip_inputs, params)
    ip_inputs_npu = {}
    for k, v in ip_inputs.items():
        ip_inputs_npu[k] = v.npu().contiguous() if isinstance(v, torch.Tensor) else v
    ip_inputs_npu['token_x'] = ip_inputs_npu['token_x'].reshape(t, h).contiguous()
    ip_inputs_npu['cos_idx_rope'] = ip_inputs_npu['cos_idx_rope'].reshape(t, rope_head_dim).contiguous()
    ip_inputs_npu['sin_idx_rope'] = ip_inputs_npu['sin_idx_rope'].reshape(t, rope_head_dim).contiguous()
    ip_inputs_npu['idx_k_cache_index'] = ip_inputs_npu['idx_k_cache_index'].reshape(t).contiguous()
    ip_inputs_npu['w_idx_qb_scale'] = ip_inputs_npu['w_idx_qb_scale'].reshape(-1, 1).contiguous()
    ip_inputs_npu['q_norm'] = ip_inputs_npu['q_norm'].reshape(t, q_lora_rank).contiguous()
    ip_inputs_npu['q_norm_scale'] = ip_inputs_npu['q_norm_scale'].reshape(t, 1).contiguous()
    ip_inputs_npu['w_idx_qb_nz'] = torch_npu.npu_format_cast(ip_inputs_npu['w_idx_qb'], torch_npu.Format.FRACTAL_NZ)
    ip_inputs_npu['w_idx_k_nz'] = torch_npu.npu_format_cast(ip_inputs_npu['w_idx_k'], torch_npu.Format.FRACTAL_NZ)
    ip_inputs_npu['w_idx_proj_nz'] = torch_npu.npu_format_cast(ip_inputs_npu['w_idx_proj'],
                                                               torch_npu.Format.FRACTAL_NZ)
    ip_goldens['q_int8'] = ip_goldens['q_int8'].reshape(t, head_num, idx_head_dim).contiguous()
    ip_goldens['q_scale'] = ip_goldens['q_scale'].reshape(t, head_num, 1).contiguous()
    ip_goldens['weights'] = ip_goldens['weights'].reshape(t, head_num).contiguous()
    return ip_inputs_npu, ip_goldens


def _build_inputs_dict(mla_inputs_npu, ip_inputs_npu):
    """Build the combined inputs dictionary for the test."""
    return {
        'x': mla_inputs_npu['x'],
        'w_dq': mla_inputs_npu['w_dq'],
        'w_uqqr': mla_inputs_npu['w_uqqr'],
        'w_qb_scale': mla_inputs_npu['w_qb_scale'],
        'w_uk': mla_inputs_npu['w_uk'],
        'w_dkvkr': mla_inputs_npu['w_dkvkr'],
        'gamma_cq': mla_inputs_npu['gamma_cq'],
        'gamma_ckv': mla_inputs_npu['gamma_ckv'],
        'cos': mla_inputs_npu['cos'],
        'sin': mla_inputs_npu['sin'],
        'cache_index': mla_inputs_npu['cache_index'],
        'kv_cache': mla_inputs_npu['kv_cache'],
        'kr_cache': mla_inputs_npu['kr_cache'],
        'kv_quant_scale_cache': mla_inputs_npu['kv_quant_scale_cache'],
        'w_idx_qb_nz': ip_inputs_npu['w_idx_qb_nz'],
        'w_idx_qb_scale': ip_inputs_npu['w_idx_qb_scale'],
        'w_idx_k_nz': ip_inputs_npu['w_idx_k_nz'],
        'w_idx_proj_nz': ip_inputs_npu['w_idx_proj_nz'],
        'layer_norm_gamma': ip_inputs_npu['layer_norm_gamma'],
        'layer_norm_beta': ip_inputs_npu['layer_norm_beta'],
        'hadamard_q': ip_inputs_npu['hadamard_q'],
        'hadamard_k': ip_inputs_npu['hadamard_k'],
        'idx_k_cache': ip_inputs_npu['idx_k_cache'],
        'idx_k_scale_cache': ip_inputs_npu['idx_k_scale_cache'],
    }


def _build_goldens_dict(mla_goldens, ip_goldens, t, head_num, idx_head_dim):
    """Build the combined goldens dictionary for the test."""
    return {
        'q_nope': mla_goldens['q_nope'],
        'q_rope': mla_goldens['q_rope'],
        'rms_norm_out': mla_goldens['rms_norm_out'],
        'rms_norm_scale_out': mla_goldens['rms_norm_scale_out'],
        'kv_cache_out': mla_goldens['kv_cache_out'],
        'kr_cache_out': mla_goldens['kr_cache_out'],
        'kv_quant_scale_cache_out': mla_goldens['kv_quant_scale_cache_out'],
        'q_int8': ip_goldens['q_int8'].reshape(t, head_num, idx_head_dim),
        'q_scale': ip_goldens['q_scale'].reshape(t, head_num, 1),
        'idx_k_cache_out': ip_goldens['idx_k_cache_out'],
        'idx_k_scale_cache_out': ip_goldens['idx_k_scale_cache_out'],
        'weights': ip_goldens['weights'].reshape(t, head_num),
    }


def _build_outputs_dict(goldens, mla_inputs_npu, ip_inputs_npu):
    """Build the combined outputs dictionary for the test."""
    return {
        'q_nope': gen_zero_tensor(goldens['q_nope']),
        'q_rope': gen_zero_tensor(goldens['q_rope']),
        'kv_cache_out': mla_inputs_npu['kv_cache'],
        'kr_cache_out': mla_inputs_npu['kr_cache'],
        'kv_quant_scale_cache_out': mla_inputs_npu['kv_quant_scale_cache'],
        'q_int8': gen_zero_tensor(goldens['q_int8']),
        'q_scale': gen_zero_tensor(goldens['q_scale']),
        'idx_k_cache_out': ip_inputs_npu['idx_k_cache'],
        'idx_k_scale_cache_out': ip_inputs_npu['idx_k_scale_cache'],
        'weights': gen_zero_tensor(goldens['weights'])
    }


def gen_test_data(params):
    t = params['t']
    head_num = params['idx_n_heads']
    idx_head_dim = params['idx_head_dim']
    dtype = params['dtype']
    dtypes = (dtype, dtype)

    mla_inputs_npu, mla_goldens, mla_inputs, block_num, block_table = _prepare_mla_data(params, dtypes)
    ip_inputs_npu, ip_goldens = _prepare_indexer_data(params, block_num, block_table, mla_inputs, mla_goldens)

    inputs = _build_inputs_dict(mla_inputs_npu, ip_inputs_npu)
    goldens = _build_goldens_dict(mla_goldens, ip_goldens, t, head_num, idx_head_dim)
    outputs = _build_outputs_dict(goldens, mla_inputs_npu, ip_inputs_npu)
    return inputs, outputs, goldens


def gen_zero_tensor(t):
    return torch.zeros_like(t).npu()


def check(case_name, outputs, goldens):
    ########### mla ###########
    compare(outputs['q_nope'].cpu(), goldens['q_nope'], 'qNope', atol=0.005, rtol=0.0078125,
            max_error_ratio=0.005)
    compare(outputs['q_rope'].cpu(), goldens['q_rope'], 'qRope', atol=0.005, rtol=0.0078125, max_error_ratio=0.005)
    compare(outputs['kv_cache_out'].cpu(), goldens['kv_cache_out'], 'kv', atol=1, rtol=0, max_error_ratio=0)
    compare(outputs['kr_cache_out'].cpu(), goldens['kr_cache_out'], 'kr',
            atol=0.0001, rtol=0.0078125, max_error_ratio=0.005)
    compare(outputs['kv_quant_scale_cache_out'].cpu(), goldens['kv_quant_scale_cache_out'],
            'kScaleCache', atol=0.000025, rtol=0.005, max_error_ratio=0.005)

    ########### ip ###########
    compare(outputs['q_int8'].cpu(), goldens['q_int8'], 'q_int8', atol=2, rtol=0, max_error_ratio=0)
    compare(outputs['q_scale'].cpu(), goldens['q_scale'], 'q_scale', atol=0.000025, rtol=0.006)
    compare(outputs['idx_k_cache_out'].cpu(), goldens['idx_k_cache_out'], 'k_int8', atol=1, rtol=0, max_error_ratio=0)
    compare(outputs['idx_k_scale_cache_out'].cpu(), goldens['idx_k_scale_cache_out'],
            'k_scale', atol=0.000025, rtol=0, max_error_ratio=0.005)
    compare(outputs['weights'].cpu(), goldens['weights'], 'weights', atol=0.000025, rtol=0, max_error_ratio=0.005)
    logging.debug(f'=== {case_name}: PASS ===')


def convert_torch_tensor(tensor_dict, dynamic_axis_dict, name_prefix):
    dynamic_count = 0
    pypto_tensors = []
    for name, tensor in tensor_dict.items():
        if name in dynamic_axis_dict.keys():
            dynamic_axis = dynamic_axis_dict[name]
            pypto_tensors.append(pypto.from_torch(tensor, name_prefix + name, dynamic_axis=dynamic_axis))
            dynamic_count += 1
        else:
            pypto_tensors.append(pypto.from_torch(tensor, name_prefix + name))
    assert dynamic_count == len(dynamic_axis_dict)
    return pypto_tensors


def do_test(case_name, params, mla_epsilon_cq, mla_epsilon_ckv, mla_cache_mode, mla_tile_config, ip_attrs,
            ip_configs, rope_tile_shape, is_prefill=False):
    prep_env()

    logging.debug(f'=== run test case: {case_name} ===')
    inputs, outputs, goldens = gen_test_data(params)

    pto_inputs = [
        inputs["x"],
        inputs["w_dq"],
        inputs["w_uqqr"],
        inputs["w_qb_scale"],
        inputs["w_uk"],
        inputs["w_dkvkr"],
        inputs["gamma_cq"],
        inputs["gamma_ckv"],
        inputs["cos"],
        inputs["sin"],
        inputs["cache_index"],
        inputs["kv_cache"],
        inputs["kr_cache"],
        inputs["kv_quant_scale_cache"],
        inputs["w_idx_qb_nz"],
        inputs["w_idx_qb_scale"],
        inputs["w_idx_k_nz"],
        inputs["w_idx_proj_nz"],
        inputs["layer_norm_gamma"],
        inputs["layer_norm_beta"],
        inputs["hadamard_q"],
        inputs["hadamard_k"],
        inputs["idx_k_cache"],
        inputs["idx_k_scale_cache"],
    ]
    pto_outputs = [
        outputs["q_nope"],
        outputs["q_rope"],
        outputs["kv_cache_out"],
        outputs["kr_cache_out"],
        outputs["kv_quant_scale_cache_out"],
        outputs["q_int8"],
        outputs["q_scale"],
        outputs["idx_k_cache_out"],
        outputs["idx_k_scale_cache_out"],
        outputs["weights"]
    ]

    import deepseek_v32_exp.mla_indexer_prolog_quant_impl as mla_lp_quant
    if is_prefill:
        fun = mla_lp_quant.mla_indexer_prolog_quant_p
    else:
        fun = mla_lp_quant.mla_indexer_prolog_quant_d

    fun(*pto_inputs, *pto_outputs, mla_epsilon_cq, mla_epsilon_ckv, mla_cache_mode, mla_tile_config,
        ip_attrs, ip_configs, rope_tile_shape)
    torch_npu.npu.synchronize()
    check(case_name, outputs, goldens)


def _setup_mla_tile_config(tile_bs, m_tile_max, prefill=False):
    from deepseek_v32_exp.mla_prolog_quant_impl import MlaTileConfig, RopeTileShapeConfig
    mla_tile_config = MlaTileConfig()
    mla_tile_config.tile_bs = tile_bs
    c0 = 16
    m_tile_value = (min(m_tile_max, mla_tile_config.tile_bs) + c0 - 1) // c0 * c0
    mv_tile_value = min(8, mla_tile_config.tile_bs)
    mla_tile_config.m_tile = m_tile_value
    mla_tile_config.pre_quant_cube_tile = [m_tile_value, m_tile_value, 256, 256, 128, 128]
    mla_tile_config.mv_tile = mv_tile_value
    mla_tile_config.q_vec_tile0 = 1
    mla_tile_config.q_vec_tile1 = 32
    mla_tile_config.k_vec_tile0 = 2
    mla_tile_config.k_vec_tile1 = 512
    if prefill:
        mla_tile_config.q_vec_tile0 = 32
        mla_tile_config.q_vec_tile1 = 128
        mla_tile_config.k_vec_tile0 = 32
        mla_tile_config.k_vec_tile1 = 512
    return mla_tile_config


params_base = {
    'n1': 128,
    'n2': 1,
    'h': 7168,
    'num_heads': 128,  # mla
    'idx_n_heads': 64,  # ip
    'q_lora_rank': 1536,
    'qk_nope_head_dim': 128,  # mla
    'idx_head_dim': 128,  # ip
    'qk_rope_head_dim': 64,  # mla
    'rope_head_dim': 64,  # ip
    'kv_lora_rank': 512,
    'block_size': 128,
    'dtype': torch.bfloat16
}


@pytest.mark.soc("950", "910")
@pytest.mark.skip(reason="env error case")
def test_b_4_s1_2_tilebs_8_d():
    seed = 6
    torch.manual_seed(seed)
    b = 4
    s1 = 2
    s2 = 1024
    params = params_base
    params_base.update({
        'b': b, 's': s1, 't': b * s1, 's1': s1, 's2': s2,
        'actual_seq': torch.tensor([s2] * b, dtype=torch.int32).unsqueeze(-1),
    })
    mla_tile_config = _setup_mla_tile_config(8, 32)
    mla_tile_config.unroll_list = [8, 4, 2, 1]
    from deepseek_v32_exp.mla_prolog_quant_impl import RopeTileShapeConfig
    rope_tile_shape = RopeTileShapeConfig(two_dim=[128, 128], three_dim=[128, 128, 128], four_dim=[16, 128, 128, 128])
    mla_cache_mode = 'PA_BSND'
    mla_epsilon_cq = 1e-5
    mla_epsilon_ckv = 1e-5
    import deepseek_v32_exp.lightning_indexer_prolog_quant_impl as ip
    ip_attrs = ip.IndexerPrologQuantAttr(eps=1e-6, layerout_query='TND', layerout_key='PA_BSND')
    ip_configs = ip.IndexerPrologQuantConfigs(
        q_linear=[16, 16, 256, 256, 128, 128], q_hd=[64, 64, 128, 128, 128, 128],
        k_linear=[16, 16, 256, 256, 128, 128], w_linear=[16, 16, 256, 256, 128, 128],
        unroll_list=[128, 64, 32, 16, 8, 4, 2, 1], cube_l1_reuse_setting={1: 4},
        block_size=128, t_sub_tile=1, chunk_size=2, vec_nbuffer_setting={-1: 1})
    do_test("mla_prolog_indexer_prolog_quant.test_b_4_s1_2_tilebs_8",
            params, mla_epsilon_cq, mla_epsilon_ckv,
            mla_cache_mode, mla_tile_config, ip_attrs, ip_configs, rope_tile_shape, False)


@pytest.mark.skip(reason="prefill test cast")
def test_t_32_tilebs_16_p():
    seed = 7
    torch.manual_seed(seed)
    b = 16
    s1 = 2
    s2 = 1024
    params = params_base
    params_base.update({
        'b': b, 's': s1, 't': b * s1, 's1': s1, 's2': s2,
        'actual_seq': torch.tensor([s2] * b, dtype=torch.int32).unsqueeze(-1),
    })
    mla_tile_config = _setup_mla_tile_config(16, 128, prefill=True)
    mla_tile_config.cube_l1_reuse_setting = {0: 2, 1: 1, 2: 1, 3: 4, 4: 4, 5: 1}
    mla_tile_config.unroll_list = [32, 16, 8, 4, 2, 1]
    mla_tile_config.dynamic_unaligned_enable = True
    from deepseek_v32_exp.mla_prolog_quant_impl import RopeTileShapeConfig
    rope_tile_shape = RopeTileShapeConfig(two_dim=[32, 64], three_dim=[32, 32, 128], four_dim=[16, 128, 128, 128])
    mla_cache_mode = 'PA_BSND'
    mla_epsilon_cq = 1e-5
    mla_epsilon_ckv = 1e-5
    import deepseek_v32_exp.lightning_indexer_prolog_quant_impl as ip
    ip_attrs = ip.IndexerPrologQuantAttr(eps=1e-6, layerout_query='TND', layerout_key='PA_BSND')
    ip_configs = ip.IndexerPrologQuantConfigs(
        q_linear=[16, 16, 512, 512, 128, 128], q_hd=[32, 32, 128, 128, 128, 128],
        k_linear=[16, 16, 512, 512, 64, 64], w_linear=[16, 16, 1024, 1024, 32, 32],
        unroll_list=[32, 16, 8, 4, 2, 1], cube_l1_reuse_setting={1: 4},
        block_size=128, t_sub_tile=1, chunk_size=2, vec_nbuffer_setting={-1: 1})
    do_test("mla_prolog_indexer_prolog_prefill.test_t_32_tilebs_16",
            params, mla_epsilon_cq, mla_epsilon_ckv,
            mla_cache_mode, mla_tile_config, ip_attrs, ip_configs, rope_tile_shape, True)


@pytest.mark.skip(reason="large shape")
def test_t_512_tilebs_128_p():
    seed = 15
    torch.manual_seed(seed)
    b = 128
    s1 = 4
    s2 = 1024
    params = params_base
    params_base.update({
        'b': b, 's': s1, 't': b * s1, 's1': s1, 's2': s2,
        'actual_seq': torch.tensor([s2] * b, dtype=torch.int32).unsqueeze(-1),
    })
    mla_tile_config = _setup_mla_tile_config(128, 128, prefill=True)
    mla_tile_config.cube_l1_reuse_setting = {0: 2, 1: 1, 2: 1, 3: 4, 4: 4, 5: 1}
    mla_tile_config.unroll_list = [128, 64, 32, 16, 8, 4, 2, 1]
    mla_tile_config.dynamic_unaligned_enable = True
    from deepseek_v32_exp.mla_prolog_quant_impl import RopeTileShapeConfig
    rope_tile_shape = RopeTileShapeConfig(two_dim=[32, 64], three_dim=[32, 32, 128], four_dim=[16, 128, 128, 128])
    mla_cache_mode = 'PA_BSND'
    mla_epsilon_cq = 1e-5
    mla_epsilon_ckv = 1e-5
    import deepseek_v32_exp.lightning_indexer_prolog_quant_impl as ip
    ip_attrs = ip.IndexerPrologQuantAttr(eps=1e-6, layerout_query='TND', layerout_key='PA_BSND')
    ip_configs = ip.IndexerPrologQuantConfigs(
        q_linear=[128, 128, 256, 256, 256, 256], q_hd=[128, 128, 64, 64, 128, 128],
        k_linear=[64, 64, 256, 256, 128, 128], w_linear=[32, 32, 512, 512, 64, 64],
        unroll_list=[128, 64, 32, 16, 8, 4, 2, 1], cube_l1_reuse_setting={1: 4, 3: 4},
        block_size=128, t_sub_tile=2, chunk_size=1, vec_nbuffer_setting={-1: 1})
    do_test("mla_prolog_indexer_prolog_prefill.test_t_512_tilebs_128", params, mla_epsilon_cq, mla_epsilon_ckv,
            mla_cache_mode, mla_tile_config, ip_attrs, ip_configs, rope_tile_shape, True)


if __name__ == '__main__':
    if PRINT_DEBUG:
        logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                            level=logging.DEBUG)
    test_b_4_s1_2_tilebs_8_d()
