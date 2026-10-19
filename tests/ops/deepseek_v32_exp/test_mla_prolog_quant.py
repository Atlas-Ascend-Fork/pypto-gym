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
'''
'''
import os
import math
import time
import logging
from dataclasses import dataclass
from typing import Any
from pathlib import Path
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
from deepseek_v32_exp.mla_prolog_quant_impl import mla_prolog_quant_p, mla_prolog_quant_d, MlaTileConfig
from common_utils import compare
import collections


MlaPrologQuantResult = collections.namedtuple('MlaPrologQuantResult', [
    'q_out', 'q_embed', 'q_a_layernorm', 'q_a_layernorm_scale_dequant',
    'kv_cache_out', 'kr_cache_out', 'kv_quant_scale_cache_out'
])
GenBlockTableOutput = collections.namedtuple("GenBlockTableOutput", ["block_num", "block_table", "cache_index"])


def prep_env():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    torch_npu.npu.config.allow_internal_format = True


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


def scatter_update(inputs, axis):
    # inputs: cache, key_states, indices
    # key_states shape: [b*s1*1, d]
    cache, key_states, indices = inputs
    block_number, block_size, n2, d = cache.shape
    res = cache.reshape(block_number * block_size * n2, d)
    b, s1 = indices.shape

    if axis == -2:
        for b_i in range(b):
            for s1_i in range(s1):
                index_value = indices[b_i][s1_i]
                res[index_value][:] = key_states[b_i * s1 + s1_i][:]
    return res.reshape(block_number, block_size, n2, d)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.concatenate((-x2, x1), dim=-1)


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


def tensor_to_file(t: torch.Tensor, output: Path):
    with open(str(output), "wb") as f:
        dtype = t.dtype
        if dtype == torch.bfloat16:
            dtype = torch.int16
        for each in t:
            f.write(each.view(dtype).cpu().numpy().tobytes())


@dataclass
class _ComputeQPathInputs:
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


def _compute_q_path(inputs: _ComputeQPathInputs):
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
        q_b_proj = torch.matmul(q_a_layernorm.to(torch.int32).cpu(),
                    inputs.w_uqqr.to(torch.int32).cpu()).to(q_a_layernorm.device)
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
    q_out = q_nope_new_t.reshape(b, s, n, kv_lora_rank)
    return q_out, q_reshape, q_a_layernorm, q_a_layernorm_scale_dequant, x_2d


def _compute_kv_path(x_2d, w_dkvkr, gamma_ckv, is_quant_a, w_kva_scale, is_quant_b, dtype):
    b_s, h = x_2d.shape
    if is_quant_a:
        x_2d_quant, x_2d_scale_dequant = quant(x_2d, True)
        kv_a_proj = torch.matmul(x_2d_quant.to(torch.int32), w_dkvkr.to(torch.int32))
        kv_a_proj_fp32 = kv_a_proj.to(torch.float32)
        kv_a_proj_fp32_dequant = kv_a_proj_fp32 * x_2d_scale_dequant
        kv_a_proj = kv_a_proj_fp32_dequant * w_kva_scale
    else:
        kv_a_proj = torch.matmul(x_2d.to(torch.float32), w_dkvkr.to(torch.float32))
    return kv_a_proj


@dataclass
class _ComputeRopeCacheInputs:
    q_reshape: Any
    kv_reshape: Any
    cos: Any
    sin: Any
    k_nope: Any
    kv_cache: Any
    kr_cache: Any
    cache_index: Any
    kv_quant_scale_cache: Any
    is_quant_b: Any
    compressed_kv_quant_scale: Any


def _compute_rope_cache(inputs: _ComputeRopeCacheInputs):
    b, s = inputs.q_reshape.shape[0], inputs.q_reshape.shape[1]
    qk_nope_head_dim = inputs.q_reshape.shape[-1] - inputs.cos.shape[2]
    qk_rope_head_dim = inputs.cos.shape[2]
    kv_lora_rank = inputs.kv_reshape.shape[-1] - qk_rope_head_dim
    q_pe = inputs.q_reshape[:, :, :, qk_nope_head_dim:]
    k_pe = inputs.kv_reshape[:, :, kv_lora_rank:]
    k_pe_r = k_pe.reshape(b, s, 1, qk_rope_head_dim)
    q_embed, k_embed = apply_rotary_pos_emb_v2(q_pe, k_pe_r, inputs.cos, inputs.sin, 2)
    k_embed_r = k_embed.reshape(b * 1 * s, qk_rope_head_dim)
    kv_cache_out = scatter_update([inputs.kv_cache.clone(), inputs.k_nope, inputs.cache_index], -2)
    kr_cache_out = scatter_update([inputs.kr_cache.clone(), k_embed_r, inputs.cache_index], -2)
    kv_quant_scale_cache_out = None
    if inputs.is_quant_b:
        ckvs = inputs.compressed_kv_quant_scale.reshape(-1, 4)
        kv_quant_scale_cache_out = scatter_update([inputs.kv_quant_scale_cache.clone(), ckvs, inputs.cache_index], -2)
    return q_embed, kv_cache_out, kr_cache_out, kv_quant_scale_cache_out


def _parse_mla_quant_inputs(inputs):
    """Extract all fields from the inputs dict for mla_prolog_quant_v32."""
    dtype = inputs.get("dtype")
    is_quant_a = inputs.get("is_quant_a")
    is_quant_b = inputs.get("is_quant_b")
    has_smooth = inputs.get("has_smooth")
    cache_mode = inputs.get("cache_mode")
    gamma_cq = inputs.get("gamma_cq")
    gamma_ckv = inputs.get("gamma_ckv")
    x = inputs.get("x")
    w_dq = inputs.get("w_dq")
    w_uqqr = inputs.get("w_uqqr")
    w_uk = inputs.get("w_uk")
    w_dkvkr = inputs.get("w_dkvkr")
    cos = inputs.get("cos")
    sin = inputs.get("sin")
    kv_cache = inputs.get("kv_cache")
    kr_cache = inputs.get("kr_cache")
    kv_quant_scale_cache = None
    if is_quant_b:
        kv_quant_scale_cache = inputs.get("kv_quant_scale_cache")
    cache_index = inputs.get("cache_index")
    w_qa_scale = None
    w_kva_scale = None
    w_qb_scale = None
    smooth_cq = None
    if is_quant_a:
        w_qa_scale = inputs.get("w_qa_scale")
        w_kva_scale = inputs.get("w_kva_scale")
    if is_quant_b:
        w_qb_scale = inputs.get("w_qb_scale")
        if has_smooth:
            smooth_cq = inputs.get("smooth_cq")
    return (dtype, is_quant_a, is_quant_b, has_smooth, cache_mode, gamma_cq, gamma_ckv,
            x, w_dq, w_uqqr, w_uk, w_dkvkr, cos, sin, kv_cache, kr_cache,
            kv_quant_scale_cache, cache_index, w_qa_scale, w_kva_scale, w_qb_scale, smooth_cq)


def mla_prolog_quant_v32_compute(inputs):
    (dtype, is_quant_a, is_quant_b, has_smooth, cache_mode, gamma_cq, gamma_ckv,
     x, w_dq, w_uqqr, w_uk, w_dkvkr, cos, sin, kv_cache, kr_cache,
     kv_quant_scale_cache, cache_index, w_qa_scale, w_kva_scale, w_qb_scale, smooth_cq) = \
        _parse_mla_quant_inputs(inputs)

    q_out, q_reshape, q_a_layernorm, q_a_layernorm_scale_dequant, x_2d = \
        _compute_q_path(_ComputeQPathInputs(
            x=x, w_dq=w_dq, w_uqqr=w_uqqr, w_uk=w_uk, gamma_cq=gamma_cq, cos=cos,
            is_quant_a=is_quant_a, is_quant_b=is_quant_b,
            has_smooth=has_smooth, smooth_cq=smooth_cq, w_qa_scale=w_qa_scale, w_qb_scale=w_qb_scale, dtype=dtype))

    kv_a_proj = _compute_kv_path(x_2d, w_dkvkr, gamma_ckv, is_quant_a, w_kva_scale, is_quant_b, dtype)
    b, s, h = x.shape
    qk_rope_head_dim = cos.shape[2]
    n, qk_nope_head_dim, kv_lora_rank = w_uk.shape
    kv_a_proj = kv_a_proj.to(dtype)
    kv_reshape = kv_a_proj.reshape(b, s, kv_lora_rank + qk_rope_head_dim)
    compressed_kv = kv_reshape[:, :, 0:kv_lora_rank]
    compressed_kv_norm = rms_norm(compressed_kv, gamma_ckv)
    compressed_kv_quant_scale = None
    if is_quant_b:
        compressed_kv_norm_split = compressed_kv_norm.reshape(b * s, 4, kv_lora_rank // 4)
        compressed_kv_norm, compressed_kv_quant_scale = quant(compressed_kv_norm_split, True)
        compressed_kv_quant_scale = compressed_kv_quant_scale.reshape(b, s, 1, 4)
    compressed_kv_r = compressed_kv_norm.reshape(b, s, 1, kv_lora_rank)
    k_nope = compressed_kv_r.reshape(b * s * 1, kv_lora_rank)

    q_embed, kv_cache_out, kr_cache_out, kv_quant_scale_cache_out = \
        _compute_rope_cache(_ComputeRopeCacheInputs(
            q_reshape=q_reshape, kv_reshape=kv_reshape, cos=cos, sin=sin, k_nope=k_nope,
            kv_cache=kv_cache, kr_cache=kr_cache,
            cache_index=cache_index, kv_quant_scale_cache=kv_quant_scale_cache,
            is_quant_b=is_quant_b, compressed_kv_quant_scale=compressed_kv_quant_scale))

    return MlaPrologQuantResult(q_out, q_embed, q_a_layernorm, q_a_layernorm_scale_dequant, kv_cache_out,
                               kr_cache_out, kv_quant_scale_cache_out)


def gen_block_table(act_seq, block_size, s1, need_indices=False):
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

    if need_indices:
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
    else:
        cache_index = None

    if need_indices:
        return GenBlockTableOutput(block_num, block_table, cache_index)
    else:
        return GenBlockTableOutput(block_num, block_table, cache_index)


@dataclass
class _MakeWeightsAndKvInputs:
    is_quant_a: Any
    is_quant_b: Any
    has_smooth: Any
    is_nz: Any
    b: Any
    h: Any
    s: Any
    n: Any
    q_lora_rank: Any
    qk_nope_head_dim: Any
    qk_rope_head_dim: Any
    kv_lora_rank: Any
    q_head_dim: Any
    w_dtype: Any
    dtype: Any
    res: Any
    block_size: Any
    block_num: Any
    block_table: Any
    skv_max: Any


def _make_weights_and_kv(inputs: _MakeWeightsAndKvInputs):
    w_dq = torch.empty([inputs.h, inputs.q_lora_rank]).uniform_(-0.1, 0.1).to(inputs.w_dtype)
    w_uqqr = torch.empty([inputs.q_lora_rank, inputs.n * inputs.q_head_dim]).uniform_(-0.1, 0.1).to(inputs.w_dtype)
    w_dkvkr = (torch.empty([inputs.h, inputs.kv_lora_rank + inputs.qk_rope_head_dim])
               .uniform_(-0.1, 0.1).to(inputs.w_dtype))
    inputs.res[4] = dict()
    if inputs.is_quant_a:
        w_dq, w_qa_scale = quant(w_dq, False)
        w_dkvkr, w_kva_scale = quant(w_dkvkr, False)
        inputs.res[4]["w_dq"] = w_qa_scale
        inputs.res[4]["w_dkvkr"] = w_kva_scale
    if inputs.is_quant_b:
        w_uqqr, w_qb_scale = quant(w_uqqr, False)
        inputs.res[4]["w_uqqr"] = w_qb_scale
        if inputs.has_smooth:
            inputs.res[3] = torch.empty([1, inputs.q_lora_rank]).uniform_(-1, 1).to(torch.float32)
    inputs.res[1], inputs.res[2], inputs.res[5] = w_dq, w_uqqr, w_dkvkr
    w_uk = torch.empty([inputs.n, inputs.qk_nope_head_dim, inputs.kv_lora_rank]).uniform_(-0.1, 0.1).to(inputs.w_dtype)
    inputs.res[6] = w_uk
    inputs.res[7] = torch.empty([inputs.q_lora_rank]).uniform_(-1, 1).to(inputs.dtype)
    inputs.res[8] = torch.empty([inputs.kv_lora_rank]).uniform_(-1, 1).to(inputs.dtype)
    inputs.res[9] = torch.empty([inputs.b, inputs.s, inputs.qk_rope_head_dim]).uniform_(-0.1, 0.1).to(inputs.dtype)
    inputs.res[10] = torch.empty([inputs.b, inputs.s, inputs.qk_rope_head_dim]).uniform_(-0.1, 0.1).to(inputs.dtype)
    k_bsnd = torch.empty(
        [inputs.b, inputs.skv_max, 1, inputs.kv_lora_rank + inputs.qk_rope_head_dim]
    ).uniform_(-1, 1).to(inputs.dtype)
    per_batch_max_num = math.ceil(inputs.skv_max / inputs.block_size)
    k_tensor_bsnd = torch.zeros(
        (inputs.b, per_batch_max_num * inputs.block_size, 1,
         inputs.kv_lora_rank + inputs.qk_rope_head_dim)
    ).to(inputs.dtype)
    k_tensor_bsnd[:, :k_bsnd.shape[1], :, :] = k_bsnd[:, :, :, :]
    k_cache_tensor = torch.zeros(
        [inputs.block_num, inputs.block_size, 1,
         inputs.kv_lora_rank + inputs.qk_rope_head_dim]
    ).to(inputs.dtype)
    for b_idx in range(inputs.b):
        for block_i, kv_cache_blk_id in enumerate(inputs.block_table[b_idx]):
            block_offset = block_i * inputs.block_size
            if kv_cache_blk_id == -1:
                continue
            else:
                k_cache_tensor[kv_cache_blk_id, 0:inputs.block_size, :, :] = k_tensor_bsnd[
                    b_idx, block_offset:(block_offset + inputs.block_size), :, :]
    kv_cache = k_cache_tensor[:, :, :, : inputs.kv_lora_rank]
    kr_cache = k_cache_tensor[:, :, :, inputs.kv_lora_rank:]
    kv_quant_scale_cache = None
    kv_cache_shape = [inputs.block_num, inputs.block_size, 1, inputs.kv_lora_rank]
    if inputs.is_quant_b:
        kv_cache_split = kv_cache.reshape(-1, 4, inputs.kv_lora_rank // 4)
        kv_cache, kv_quant_scale_cache = quant(kv_cache_split, True)
        kv_cache = kv_cache.reshape(kv_cache_shape)
        kv_quant_scale_cache = kv_quant_scale_cache.reshape([inputs.block_num, inputs.block_size, 1, 4])
    return kv_cache, kr_cache, kv_quant_scale_cache


def gen_mla_prolog_quant_v32_input_data(params, dtypes, actual_seq, is_quant=(False, False),
                                        has_smooth=False, block_size=128, cache_mode="BSND"):
    dtype, w_dtype = dtypes
    is_quant_a, is_quant_b = is_quant
    b, s, s1, h = params.get("b"), params.get("s"), params.get("s1"), params.get("h")
    n = params.get("n1")
    q_lora_rank = params.get("q_lora_rank")
    qk_nope_head_dim = params.get("qk_nope_head_dim")
    qk_rope_head_dim = params.get("qk_rope_head_dim")
    kv_lora_rank = params.get("kv_lora_rank")
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    block_num, block_table, cache_index = gen_block_table(actual_seq, block_size, s1, need_indices=True)
    skv_max = actual_seq.max()

    res = [None] * 17
    res[0] = torch.empty([b, s, h]).uniform_(-1, 1).to(dtype)
    res[11] = cache_index
    kv_cache, kr_cache, kv_quant_scale_cache = _make_weights_and_kv(_MakeWeightsAndKvInputs(
        is_quant_a=is_quant_a, is_quant_b=is_quant_b, has_smooth=has_smooth, is_nz=False,
        b=b, h=h, s=s, n=n, q_lora_rank=q_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim, qk_rope_head_dim=qk_rope_head_dim,
        kv_lora_rank=kv_lora_rank, q_head_dim=q_head_dim, w_dtype=w_dtype, dtype=dtype,
        res=res, block_size=block_size, block_num=block_num, block_table=block_table, skv_max=skv_max))
    res[12], res[13], res[14], res[15], res[16] = kv_cache, kr_cache, kv_quant_scale_cache, block_num, block_table
    return res


def gen_mla_prolog_quant_v32_data(params, dtypes, actual_seq, is_quant=(False, False),
                                  has_smooth=False, block_size=128, cache_mode="BSND"):
    dtype, w_dtype = dtypes
    x, w_dq, w_uqqr, smooth_cq, scale_data, w_dkvkr, w_uk, gamma_cq, gamma_ckv, cos, sin, kv_len, \
        kv_cache, kr_cache, kv_quant_scale_cache, block_num, block_table = \
        gen_mla_prolog_quant_v32_input_data(params, dtypes, actual_seq, is_quant, has_smooth,
                                            block_size, cache_mode)
    is_quant_a, is_quant_b = is_quant

    inputs = {"dtype": dtype, "is_quant_a": is_quant_a, "is_quant_b": is_quant_b, "has_smooth": has_smooth}
    inputs["cache_mode"] = cache_mode
    inputs["gamma_cq"] = gamma_cq
    inputs["gamma_ckv"] = gamma_ckv
    inputs["x"] = x
    inputs["w_dq"] = w_dq
    inputs["w_uqqr"] = w_uqqr
    inputs["w_uk"] = w_uk
    inputs["w_dkvkr"] = w_dkvkr
    inputs["cos"] = cos
    inputs["sin"] = sin
    inputs["kv_cache"] = kv_cache
    inputs["kr_cache"] = kr_cache
    inputs["kv_quant_scale_cache"] = kv_quant_scale_cache
    inputs["cache_index"] = kv_len
    if is_quant_a:
        inputs["w_qa_scale"] = scale_data["w_dq"]
        inputs["w_kva_scale"] = scale_data["w_dkvkr"]
    if is_quant_b:
        inputs["w_qb_scale"] = scale_data["w_uqqr"]
        if has_smooth:
            inputs["smooth_cq"] = smooth_cq

    result = mla_prolog_quant_v32_compute(inputs)
    q_out = result.q_out
    q_embed = result.q_embed
    rms_norm_out = result.q_a_layernorm
    rms_norm_scale = result.q_a_layernorm_scale_dequant
    kv_cache_out = result.kv_cache_out
    kr_cache_out = result.kr_cache_out
    kv_quant_scale_cache_out = result.kv_quant_scale_cache_out
    outputs = {"q_golden": q_out, "q_rope": q_embed, "kr_golden": kr_cache_out, "kv_golden": kv_cache_out}
    outputs["kv_quant_scale_cache_golden"] = kv_quant_scale_cache_out
    outputs["rms_norm_golden"] = rms_norm_out
    outputs["rms_norm_scale_golden"] = rms_norm_scale

    if torch_npu.npu.is_available():
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.npu()

    return inputs, outputs


def convert_pypto_to_torch_type(pypto_type):
    if pypto_type == pypto.DT_INT8:
        return torch.int8
    elif pypto_type == pypto.DT_INT32:
        return torch.int32
    elif pypto_type == pypto.DT_FP32:
        return torch.float32
    elif pypto_type == pypto.DT_FP16:
        return torch.float16
    elif pypto_type == pypto.DT_BF16:
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported pypto.DataType: {pypto_type}")


@dataclass
class _RunAndCompareInputs:
    output_q_norm_data: Any
    output_q_norm_scale_data: Any
    output_q_nope_data: Any
    output_q_rope_data: Any
    output_kv_cache_data: Any
    output_kr_cache_data: Any
    golden1: Any
    golden2: Any
    golden3: Any
    golden4: Any
    golden5: Any
    golden6: Any
    golden7: Any
    is_quant_b: Any
    k_scale: Any


def _run_and_compare(inputs: _RunAndCompareInputs):
    print("qNope =======")
    compare(inputs.output_q_nope_data.cpu(), inputs.golden1.cpu(), "qNope",
            atol=0.005, rtol=0.0078125, max_error_ratio=0.005)
    print("qRope =======")
    compare(inputs.output_q_rope_data.cpu(), inputs.golden2.cpu(), "qRope",
            atol=0.005, rtol=0.0078125, max_error_ratio=0.005)
    if inputs.is_quant_b:
        print("qNorm =======")
        compare(inputs.output_q_norm_data.cpu(), inputs.golden6.cpu(), "qNorm",
                atol=1.0, rtol=0.0, max_error_ratio=0.005)
        print("qNormScale =======")
        compare(inputs.output_q_norm_scale_data.cpu(), inputs.golden7.cpu(), "qNormScale",
                atol=0.000025, rtol=0.005, max_error_ratio=0.005)
    else:
        print("qNorm =======")
        compare(inputs.output_q_norm_data.cpu(), inputs.golden6.cpu(), "qNorm",
                atol=0.0001, rtol=0.0078125, max_error_ratio=0.005)
    print("kv =======")
    if inputs.is_quant_b:
        compare(inputs.output_kv_cache_data.cpu(), inputs.golden3.cpu(), "kv", atol=1.0, rtol=0.0, max_error_ratio=0)
    else:
        compare(inputs.output_kv_cache_data.cpu(), inputs.golden3.cpu(), "kv",
                atol=0.0001, rtol=0.0078125, max_error_ratio=0)
    print("kr =======")
    compare(inputs.output_kr_cache_data.cpu(), inputs.golden4.cpu(), "kr",
            atol=0.0001, rtol=0.0078125, max_error_ratio=0)
    if inputs.is_quant_b:
        print("kScaleCache =======")
        compare(inputs.k_scale.cpu(), inputs.golden5.cpu(), "kScaleCache", atol=0.000025, rtol=0.005, max_error_ratio=0)


def _setup_input_output_data(input_tensors, params, is_quant_b, d_type, dtype_kv_quant):
    b, s, t = params['b'], params['s'], params['b'] * params['s']
    s2, n1 = params['s2'], params['n1']
    n2 = 1
    h, q_lora_rank = params['h'], params['q_lora_rank']
    qk_nope_head_dim = params['qk_nope_head_dim']
    qk_rope_head_dim = params['qk_rope_head_dim']
    kv_lora_rank = params["kv_lora_rank"]
    block_size = params['block_size']
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    block_num = b * ((s2 + block_size - 1) // block_size)

    q_norm_out_shape = [t, q_lora_rank]
    kv_cache_shape = [block_num, block_size, n2, kv_lora_rank]
    kr_cache_shape = [block_num, block_size, n2, qk_rope_head_dim]
    k_scale_cache_out_shape = [block_num, block_size, n2, 4]
    q_nope_out_shape = [t, n1, kv_lora_rank]
    q_rope_out_shape = [t, n1, qk_rope_head_dim]

    output_q_norm_data = torch.empty(q_norm_out_shape, dtype=convert_pypto_to_torch_type(dtype_kv_quant)).npu()
    output_q_norm_scale_data = torch.empty([t, 1], dtype=torch.float32).npu()
    output_q_nope_data = torch.empty(q_nope_out_shape, dtype=convert_pypto_to_torch_type(d_type)).npu()
    output_q_rope_data = torch.empty(q_rope_out_shape, dtype=convert_pypto_to_torch_type(d_type)).npu()
    output_kv_cache_data = input_tensors["kv_cache"].reshape(kv_cache_shape).npu()
    output_kr_cache_data = input_tensors["kr_cache"].reshape(kr_cache_shape).npu()

    w_dq_shape = [h, q_lora_rank]
    w_dkv_kr_shape = [h, kv_lora_rank + qk_rope_head_dim]
    w_uq_qr_shape = [q_lora_rank, n1 * q_head_dim]
    w_dq_nz = torch_npu.npu_format_cast(
        input_tensors["w_dq"].reshape(w_dq_shape).npu().contiguous(),
        torch_npu.Format.FRACTAL_NZ)
    w_dkvkr_nz = torch_npu.npu_format_cast(
        input_tensors["w_dkvkr"].reshape(w_dkv_kr_shape).npu().contiguous(),
        torch_npu.Format.FRACTAL_NZ)
    w_uqqr_nz = torch_npu.npu_format_cast(
        input_tensors["w_uqqr"].reshape(w_uq_qr_shape).npu().contiguous(),
        torch_npu.Format.FRACTAL_NZ)
    input_tensors["w_uqqr"] = w_uqqr_nz
    input_tensors["w_dkvkr"] = w_dkvkr_nz
    input_tensors["w_dq"] = w_dq_nz

    token_x_data = input_tensors["x"].reshape([t, h]).npu()
    w_dq_data = input_tensors["w_dq"].reshape(w_dq_shape).npu()
    w_uq_qr_data = input_tensors["w_uqqr"].reshape(w_uq_qr_shape).npu()
    w_uk_data = input_tensors["w_uk"].reshape([n1, qk_nope_head_dim, kv_lora_rank]).npu()
    w_dkv_kr_data = input_tensors["w_dkvkr"].reshape(w_dkv_kr_shape).npu()
    rmsnorm_gamma_cq_data = input_tensors["gamma_cq"].reshape([q_lora_rank]).npu()
    rmsnorm_gamma_ckv_data = input_tensors["gamma_ckv"].reshape([kv_lora_rank]).npu()
    rope_cos_data = input_tensors["cos"].reshape([t, qk_rope_head_dim]).npu()
    rope_sin_data = input_tensors["sin"].reshape([t, qk_rope_head_dim]).npu()
    cache_index_data = input_tensors["cache_index"].reshape([t]).npu()
    kv_cache_data = input_tensors["kv_cache"].reshape(kv_cache_shape).npu()
    kr_cache_data = input_tensors["kr_cache"].reshape(kr_cache_shape).npu()

    if is_quant_b:
        k_scale = input_tensors["kv_quant_scale_cache"].npu()
        k_scale_cache_data = k_scale
        k_scale_cache_data_out = k_scale
    else:
        k_scale_cache_data = torch.zeros(k_scale_cache_out_shape, dtype=torch.float32).npu()
        k_scale_cache_data_out = torch.zeros(k_scale_cache_out_shape, dtype=torch.float32).npu()
        k_scale = None

    dequant_scale_w_uq_qr_shape = [n1 * q_head_dim, 1]
    if is_quant_b:
        dequant_scale_w_uq_qr_data = input_tensors["w_qb_scale"].reshape(dequant_scale_w_uq_qr_shape).npu()
    else:
        dequant_scale_w_uq_qr_data = torch.Tensor().npu()

    input_data = [token_x_data, w_dq_data, w_uq_qr_data, dequant_scale_w_uq_qr_data,
                w_uk_data, w_dkv_kr_data, rmsnorm_gamma_cq_data, rmsnorm_gamma_ckv_data,
                rope_cos_data, rope_sin_data, cache_index_data,
                kv_cache_data, kr_cache_data, k_scale_cache_data]
    output_data = [output_q_norm_data, output_q_norm_scale_data, output_q_nope_data,
                output_q_rope_data, output_kv_cache_data, output_kr_cache_data, k_scale_cache_data_out]
    return input_data, output_data, (output_q_norm_data, output_q_norm_scale_data, output_q_nope_data,
                output_q_rope_data, output_kv_cache_data, output_kr_cache_data, k_scale)


@dataclass
class MlaPrologQuantV32Inputs:
    params: Any
    input_tensors: Any
    golden_data: Any
    dtype: Any
    w_dtype: Any
    is_quant_a: Any
    is_quant_b: Any
    nz: Any
    tile_config: Any
    cache_mode: Any
    is_p: Any


def mla_prolog_quant_v32(inputs: MlaPrologQuantV32Inputs):
    params = inputs.params
    input_tensors = inputs.input_tensors
    golden_data = inputs.golden_data
    dtype = inputs.dtype
    w_dtype = inputs.w_dtype
    is_quant_a = inputs.is_quant_a
    is_quant_b = inputs.is_quant_b
    nz = inputs.nz
    tile_config = inputs.tile_config
    cache_mode = inputs.cache_mode
    is_p = inputs.is_p

    d_type = pypto.DT_FP16 if dtype == pypto.DT_FP16 else pypto.DT_BF16
    dtype_qa = pypto.DT_INT8 if (is_quant_a and w_dtype == pypto.DT_INT8) else dtype
    dtype_qb = pypto.DT_INT8 if (is_quant_b and w_dtype == pypto.DT_INT8) else dtype
    dtype_kv_quant = dtype_qb

    b, s, t = params['b'], params['s'], params['b'] * params['s']
    s2, n1 = params['s2'], params['n1']
    n2 = 1
    kv_lora_rank = params["kv_lora_rank"]
    qk_rope_head_dim = params['qk_rope_head_dim']

    input_data, output_data, compare_tensors = _setup_input_output_data(
        input_tensors, params, is_quant_b, d_type, dtype_kv_quant)
    output_q_norm_data, output_q_norm_scale_data, output_q_nope_data, \
        output_q_rope_data, output_kv_cache_data, output_kr_cache_data, k_scale = compare_tensors

    from deepseek_v32_exp.mla_prolog_quant_impl import RopeTileShapeConfig
    rope_tile_shape = RopeTileShapeConfig(two_dim=[32, 64], three_dim=[32, 32, 128], four_dim=[16, 128, 128, 128])
    if is_p:
        mla_prolog_quant_p(*input_data, *output_data, 1e-5, 1e-5, cache_mode, tile_config, rope_tile_shape)
    else:
        mla_prolog_quant_d(*input_data, *output_data, 1e-5, 1e-5, cache_mode, tile_config, rope_tile_shape)
    torch_npu.npu.synchronize()

    golden1 = golden_data["q_golden"].reshape([t, n1, kv_lora_rank])
    golden2 = golden_data["q_rope"] .reshape([t, n1, qk_rope_head_dim])
    golden3 = golden_data["kv_golden"].reshape(output_kv_cache_data.shape)
    golden4 = golden_data["kr_golden"].reshape(output_kr_cache_data.shape)
    golden5 = golden6 = golden7 = None
    if is_quant_b:
        block_size = params['block_size']
        block_num = b * ((s2 + block_size - 1) // block_size)
        golden5 = golden_data["kv_quant_scale_cache_golden"].reshape([block_num, block_size, n2, 4])
    golden6 = golden_data["rms_norm_golden"].reshape([t, params['q_lora_rank']])
    if is_quant_b:
        golden7 = golden_data["rms_norm_scale_golden"].reshape([t, 1])

    _run_and_compare(_RunAndCompareInputs(
        output_q_norm_data=output_q_norm_data, output_q_norm_scale_data=output_q_norm_scale_data,
        output_q_nope_data=output_q_nope_data,
        output_q_rope_data=output_q_rope_data, output_kv_cache_data=output_kv_cache_data,
        output_kr_cache_data=output_kr_cache_data,
        golden1=golden1, golden2=golden2, golden3=golden3, golden4=golden4,
        golden5=golden5, golden6=golden6, golden7=golden7,
        is_quant_b=is_quant_b, k_scale=k_scale))


@pytest.mark.skip(reason="large shape")
def test_b128_s4k4_pa_nd_bf16_quantb_p():
    '''
    mla_prolog prefill测试函数
    '''
    torch.manual_seed(5)
    prep_env()
    params = {
        'b': 128,
        't': 128,
        's': 1,
        's1': 1,
        's2': 1024,
        'n1': 128,
        'h': 7168,
        'q_lora_rank': 1536,
        'qk_nope_head_dim': 128,
        'qk_rope_head_dim': 64,
        'kv_lora_rank': 512,
        'block_size': 128
    }
    dtype = pypto.DT_BF16
    w_dtype = pypto.DT_INT8
    is_quant_a, is_quant_b, is_nz = False, True, False
    cache_mode = "PA_BSND"
    tile_config = MlaTileConfig()
    tile_config.tile_bs = 128
    c0 = 16
    m_tile_value = (min(128, tile_config.tile_bs) + c0 - 1) // c0 * c0
    mv_tile_value = min(8, tile_config.tile_bs)
    tile_config.m_tile = m_tile_value

    tile_config.pre_quant_cube_tile[0] = m_tile_value
    tile_config.pre_quant_cube_tile[1] = m_tile_value
    tile_config.cube_qb_tile = [m_tile_value, m_tile_value, 256, 256, 256, 256]
    tile_config.cube_wuk_tile = [m_tile_value, m_tile_value, 128, 128, 128, 128]
    tile_config.mv_tile = mv_tile_value
    tile_config.q_vec_tile0 = 32
    tile_config.q_vec_tile1 = 128
    tile_config.k_vec_tile0 = 32
    tile_config.k_vec_tile1 = 512
    tile_config.unroll_list = [128, 64, 32, 16, 8, 4, 2, 1]

    actual_seq = torch.tensor([params["s2"]] * params["b"], dtype=torch.int32).unsqueeze(-1)
    input_tensors, golden_data = gen_mla_prolog_quant_v32_data(params, (torch.bfloat16, torch.bfloat16), actual_seq, \
                    (is_quant_a, is_quant_b), False, 128, "PA_BSND")
    mla_prolog_quant_v32(MlaPrologQuantV32Inputs(
        params=params, input_tensors=input_tensors, golden_data=golden_data,
        dtype=dtype, w_dtype=w_dtype, is_quant_a=is_quant_a, is_quant_b=is_quant_b,
        nz=is_nz, tile_config=tile_config, cache_mode=cache_mode, is_p=True))


@pytest.mark.soc("950", "910")
@pytest.mark.skip(reason="large shape")
def test_b4_s64k2_pa_nd_bf16_quantb_d():
    '''
    mla_prolog decode测试函数
    '''
    torch.manual_seed(5)
    prep_env()
    params = {
        'b': 4,
        't': 8,
        's': 2,
        's1': 2,
        's2': 1024,
        'n1': 128,
        'h': 7168,
        'q_lora_rank': 1536,
        'qk_nope_head_dim': 128,
        'qk_rope_head_dim': 64,
        'kv_lora_rank': 512,
        'block_size': 128
    }
    dtype = pypto.DT_BF16
    w_dtype = pypto.DT_INT8
    is_quant_a, is_quant_b, is_nz = False, True, False
    cache_mode = "PA_BSND"
    tile_config = MlaTileConfig()
    tile_config.tile_bs = 8

    c0 = 16
    m_tile_value = (min(32, tile_config.tile_bs) + c0 - 1) // c0 * c0
    mv_tile_value = min(8, tile_config.tile_bs)
    tile_config.m_tile = m_tile_value

    tile_config.pre_quant_cube_tile = [m_tile_value, m_tile_value, 256, 256, 128, 128]
    tile_config.cube_qb_tile = [m_tile_value, m_tile_value, 256, 256, 256, 256]
    tile_config.cube_wuk_tile = [tile_config.m_tile, tile_config.m_tile, 128, 128, 128, 128]
    tile_config.mv_tile = mv_tile_value
    tile_config.q_vec_tile0 = 1
    tile_config.q_vec_tile1 = 32
    tile_config.k_vec_tile0 = 2
    tile_config.k_vec_tile1 = 512
    tile_config.unroll_list = [8, 4, 2, 1]

    actual_seq = torch.tensor([params["s2"]] * params["b"], dtype=torch.int32).unsqueeze(-1)
    input_tensors, golden_data = gen_mla_prolog_quant_v32_data(params, (torch.bfloat16, torch.bfloat16), actual_seq, \
                    (is_quant_a, is_quant_b), False, 128, "PA_BSND")
    mla_prolog_quant_v32(MlaPrologQuantV32Inputs(
        params=params, input_tensors=input_tensors, golden_data=golden_data,
        dtype=dtype, w_dtype=w_dtype, is_quant_a=is_quant_a, is_quant_b=is_quant_b,
        nz=is_nz, tile_config=tile_config, cache_mode=cache_mode, is_p=False))


@pytest.mark.skip(reason="large shape")
def test_b64_s64k2_pa_nd_bf16_quantb_d():
    '''
    mla_prolog decode int8量化高吞吐测试用例
    '''
    torch.manual_seed(5)
    prep_env()
    params = {
        'b': 64,
        't': 128,
        's': 2,
        's1': 2,
        's2': 1024,
        'n1': 128,
        'h': 7168,
        'q_lora_rank': 1536,
        'qk_nope_head_dim': 128,
        'qk_rope_head_dim': 64,
        'kv_lora_rank': 512,
        'block_size': 128
    }
    dtype = pypto.DT_BF16
    w_dtype = pypto.DT_INT8
    is_quant_a, is_quant_b, is_nz = False, True, False
    cache_mode = "PA_BSND"
    tile_config = MlaTileConfig()
    tile_config.tile_bs = 32

    tile_config.m_tile = 128 

    tile_config.cube_qb_tile = [128, 128, 128, 256, 256, 256]
    tile_config.cube_wuk_tile = [tile_config.m_tile, tile_config.m_tile, 128, 256, 256, 256]
    tile_config.mv_tile = 8
    tile_config.q_vec_tile0 = 32
    tile_config.q_vec_tile1 = 128
    tile_config.k_vec_tile0 = 32
    tile_config.k_vec_tile1 = 512
    
    if pypto.platform.npuarch == 'DAV_3510':
        tile_config.pre_quant_cube_tile = [128, 128, 256, 256, 128, 128]
        tile_config.unroll_list = [128, 64, 32, 8, 4, 2, 1]
    else:
        tile_config.pre_quant_cube_tile = [32, 32, 256, 256, 128, 128]
        tile_config.unroll_list = [64, 32, 8, 4, 2, 1]

    actual_seq = torch.tensor([params["s2"]] * params["b"], dtype=torch.int32).unsqueeze(-1)
    input_tensors, golden_data = gen_mla_prolog_quant_v32_data(params, (torch.bfloat16, torch.bfloat16), actual_seq, \
                    (is_quant_a, is_quant_b), False, 128, "PA_BSND")
    mla_prolog_quant_v32(MlaPrologQuantV32Inputs(
        params=params, input_tensors=input_tensors, golden_data=golden_data,
        dtype=dtype, w_dtype=w_dtype, is_quant_a=is_quant_a, is_quant_b=is_quant_b,
        nz=is_nz, tile_config=tile_config, cache_mode=cache_mode, is_p=False))


@pytest.mark.soc("950")
def test_b4_s64k2_pa_nd_bf16_d():
    '''
    mla_prolog decode非量化测试函数
    '''
    torch.manual_seed(5)
    prep_env()
    params = {
        'b': 4,
        't': 8,
        's': 2,
        's1': 2,
        's2': 64 * 1024,
        'n1': 128,
        'h': 7168,
        'q_lora_rank': 1536,
        'qk_nope_head_dim': 128,
        'qk_rope_head_dim': 64,
        'kv_lora_rank': 512,
        'block_size': 128
    }
    dtype = pypto.DT_BF16
    w_dtype = pypto.DT_INT8
    is_quant_a, is_quant_b, is_nz = False, False, False
    cache_mode = "PA_BSND"
    tile_config = MlaTileConfig()
    tile_config.tile_bs = 8

    c0 = 16
    m_tile_value = (min(32, tile_config.tile_bs) + c0 - 1) // c0 * c0
    mv_tile_value = min(8, tile_config.tile_bs)
    tile_config.m_tile = m_tile_value

    tile_config.pre_quant_cube_tile = [m_tile_value, m_tile_value, 64, 256, 128, 128]
    tile_config.cube_qb_tile = [m_tile_value, m_tile_value, 64, 256, 256, 256]
    tile_config.cube_wuk_tile = [tile_config.m_tile, tile_config.m_tile, 128, 128, 128, 128]
    tile_config.mv_tile = mv_tile_value
    tile_config.q_vec_tile0 = 1
    tile_config.q_vec_tile1 = 32
    tile_config.k_vec_tile0 = 2
    tile_config.k_vec_tile1 = 512
    tile_config.unroll_list = [8, 4, 2, 1]

    actual_seq = torch.tensor([params["s2"]] * params["b"], dtype=torch.int32).unsqueeze(-1)
    input_tensors, golden_data = gen_mla_prolog_quant_v32_data(params, (torch.bfloat16, torch.bfloat16), actual_seq, \
                    (is_quant_a, is_quant_b), False, 128, "PA_BSND")
    mla_prolog_quant_v32(MlaPrologQuantV32Inputs(
        params=params, input_tensors=input_tensors, golden_data=golden_data,
        dtype=dtype, w_dtype=w_dtype, is_quant_a=is_quant_a, is_quant_b=is_quant_b,
        nz=is_nz, tile_config=tile_config, cache_mode=cache_mode, is_p=False))


@pytest.mark.soc("950")
def test_b64_s64k2_pa_nd_bf16_d():
    '''
    mla_prolog decode非量化测试函数
    '''
    torch.manual_seed(5)
    prep_env()
    params = {
        'b': 64,
        't': 128,
        's': 2,
        's1': 2,
        's2': 64 * 1024,
        'n1': 128,
        'h': 7168,
        'q_lora_rank': 1536,
        'qk_nope_head_dim': 128,
        'qk_rope_head_dim': 64,
        'kv_lora_rank': 512,
        'block_size': 128
    }
    dtype = pypto.DT_BF16
    w_dtype = pypto.DT_INT8
    is_quant_a, is_quant_b, is_nz = False, False, False
    cache_mode = "PA_BSND"
    tile_config = MlaTileConfig()
    tile_config.tile_bs = 128
    c0 = 16
    m_tile_value = (min(128, tile_config.tile_bs) + c0 - 1) // c0 * c0
    mv_tile_value = min(8, tile_config.tile_bs)
    tile_config.m_tile = m_tile_value

    if pypto.platform.npuarch == 'DAV_3510':
        tile_config.pre_quant_cube_tile = [m_tile_value, m_tile_value, 64, 256, 128, 128]
        tile_config.cube_qb_tile = [128, 128, 64, 256, 256, 256]
        tile_config.cube_wuk_tile = [tile_config.m_tile, tile_config.m_tile, 128, 128, 128, 128]
    else:
        tile_config.pre_quant_cube_tile = [32, 32, 64, 256, 128, 128]
        tile_config.cube_qb_tile = [128, 128, 64, 256, 256, 256]
        tile_config.cube_wuk_tile = [tile_config.m_tile, tile_config.m_tile, 128, 256, 256, 256]
    tile_config.mv_tile = mv_tile_value
    tile_config.q_vec_tile0 = 32
    tile_config.q_vec_tile1 = 128
    tile_config.k_vec_tile0 = 32
    tile_config.k_vec_tile1 = 512
    if pypto.platform.npuarch == 'DAV_3510':
        tile_config.unroll_list = [128, 64, 32, 16, 8, 4, 2, 1]
    else:
        tile_config.unroll_list = [64, 32, 16, 8, 4, 2, 1]

    actual_seq = torch.tensor([params["s2"]] * params["b"], dtype=torch.int32).unsqueeze(-1)
    input_tensors, golden_data = gen_mla_prolog_quant_v32_data(params, (torch.bfloat16, torch.bfloat16), actual_seq, \
                    (is_quant_a, is_quant_b), False, 128, "PA_BSND")
    mla_prolog_quant_v32(MlaPrologQuantV32Inputs(
        params=params, input_tensors=input_tensors, golden_data=golden_data,
        dtype=dtype, w_dtype=w_dtype, is_quant_a=is_quant_a, is_quant_b=is_quant_b,
        nz=is_nz, tile_config=tile_config, cache_mode=cache_mode, is_p=False))



if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO
    )
    test_b4_s64k2_pa_nd_bf16_quantb_d()
