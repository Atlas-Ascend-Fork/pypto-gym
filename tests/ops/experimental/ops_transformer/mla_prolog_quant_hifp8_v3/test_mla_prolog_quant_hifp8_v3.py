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
MLA Prolog Quant HiFP8 Operator Test

本测试文件用于验证 MLA (Multi-Head Attention) Prolog V3 量化算子的实现正确性，
支持 HiFP8 数据格式的量化场景。

主要测试内容:
  - MLA Prolog decode 模式下的量化计算
  - Q/KV 的量化精度验证
  - KV Cache 和 KR Cache 的更新操作
  - RoPE (Rotary Position Embedding) 位置编码应用
"""
from dataclasses import dataclass
import os

import math
import time
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

from common_utils import compare
from experimental.ops_transformer.mla_prolog_quant_hifp8_v3.mla_prolog_quant_hifp8_v3_impl import (
    mla_prolog_quant, MlaTileConfig, RopeTileShapeConfig
)
import collections


MlaPrologHifpResult = collections.namedtuple('MlaPrologHifpResult', [
    'q_out', 'q_embed', 'q_a_layernorm', 'q_a_layernorm_scale_dequant',
    'kv_cache_out', 'kr_cache_out'
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


def quant_hif8(x: torch.Tensor, quant_dim: int = -1):
    x_fp32 = x.to(torch.float32)
    max_value = torch.amax(torch.abs(x_fp32), dim=quant_dim, keepdim=True)
    scale_quant = 32768.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_fp32 = y_fp32.view(x.shape)
    y_hif8 = torch_npu.npu_dtype_cast(y_fp32, torch_npu.hifloat8)
    scale_dequant = 1.0 / scale_quant
    return y_hif8, scale_dequant


def tensor_to_file(t: torch.Tensor, output: Path):
    with open(str(output), "wb") as f:
        dtype = t.dtype
        if dtype == torch.bfloat16:
            dtype = torch.int16
        for each in t:
            f.write(each.view(dtype).cpu().numpy().tobytes())


def _extract_compute_inputs(inputs):
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
    w_dq_scale = None
    w_dkvkr_scale = None
    w_uqqr_scale = None
    smooth_cq = None
    if is_quant_a:
        w_dq_scale = inputs.get("w_dq_scale")
        w_dkvkr_scale = inputs.get("w_dkvkr_scale")
    if is_quant_b:
        w_uqqr_scale = inputs.get("w_uqqr_scale")
        if has_smooth:
            smooth_cq = inputs.get("smooth_cq")
    return (dtype, is_quant_a, is_quant_b, has_smooth, cache_mode,
            gamma_cq, gamma_ckv, x, w_dq, w_uqqr, w_uk, w_dkvkr,
            cos, sin, kv_cache, kr_cache, kv_quant_scale_cache, cache_index,
            w_dq_scale, w_dkvkr_scale, w_uqqr_scale, smooth_cq)


@dataclass
class _ComputeQPathInputs:
    x_2d: Any
    x_2d_quant: Any
    x_2d_scale_dequant: Any
    w_dq: Any
    w_dq_scale: Any
    w_uqqr: Any
    w_uqqr_scale: Any
    gamma_cq: Any
    w_uk: Any
    is_quant_a: Any
    is_quant_b: Any
    dtype: Any
    b: Any
    s: Any
    n: Any
    q_lora_rank: Any
    qk_nope_head_dim: Any
    qk_rope_head_dim: Any
    kv_lora_rank: Any


@dataclass
class _ComputeKvAndCacheInputs:
    x_2d: Any
    x_2d_quant: Any
    x_2d_scale_dequant: Any
    w_dkvkr: Any
    w_dkvkr_scale: Any
    gamma_ckv: Any
    cos: Any
    sin: Any
    q_reshape: Any
    kv_cache: Any
    kr_cache: Any
    cache_index: Any
    is_quant_a: Any
    dtype: Any
    b: Any
    s: Any
    kv_lora_rank: Any
    qk_rope_head_dim: Any
    qk_nope_head_dim: Any


def _compute_kv_and_cache(inputs: _ComputeKvAndCacheInputs):

    # shape is: [b * s, h] @ [h, kv_lora_rank + qk_rope_head_dim] -> [b * s, kv_lora_rank + qk_rope_head_dim]
    if inputs.is_quant_a:
        kv_a_proj = torch_npu.npu_quant_matmul(
            inputs.x_2d_quant, inputs.w_dkvkr,
            inputs.w_dkvkr_scale.view(inputs.kv_lora_rank + inputs.qk_rope_head_dim), 
            pertoken_scale=inputs.x_2d_scale_dequant.view(inputs.b * inputs.s),
            x1_dtype=torch_npu.hifloat8,
            x2_dtype=torch_npu.hifloat8, output_dtype=torch_npu.float32)
    else:
        # matmul use float32 for arm, arm平台matmul在bfloat16数据类型下表现与x86平台不一致，通过升精度保证正确性
        kv_a_proj = torch.matmul(inputs.x_2d.to(torch.float32),
                                 inputs.w_dkvkr.to(torch.float32))  # [b * s, kv_lora_rank + qk_rope_head_dim]

    kv_a_proj = kv_a_proj.to(inputs.dtype)
    kv_reshape = kv_a_proj.reshape(inputs.b, inputs.s, inputs.kv_lora_rank + inputs.qk_rope_head_dim)

    compressed_kv = kv_reshape[:, :, 0:inputs.kv_lora_rank]  # [b, s, kv_lora_rank]
    compressed_kv_norm = rms_norm(compressed_kv, inputs.gamma_ckv)
    compressed_kv_quant_scale = None

    compressed_kv_r = compressed_kv_norm.reshape(inputs.b, inputs.s, 1, inputs.kv_lora_rank)
    k_nope = compressed_kv_r.reshape(inputs.b * inputs.s * 1, inputs.kv_lora_rank)

    """ RoPE """
    q_pe = inputs.q_reshape[:, :, :, inputs.qk_nope_head_dim:]  # [b, s, n, qk_rope_head_dim]

    k_pe = kv_reshape[:, :, inputs.kv_lora_rank:]  # [b, s, qk_rope_head_dim]
    k_pe_r = k_pe.reshape(inputs.b, inputs.s, 1, inputs.qk_rope_head_dim)

    # q_embed: [b, s, n, qk_rope_head_dim], k_embed: [b, s, 1, qk_rope_head_dim]
    q_embed, k_embed = apply_rotary_pos_emb_v2(q_pe, k_pe_r, inputs.cos, inputs.sin, 2)
    k_embed_r = k_embed.reshape(inputs.b * 1 * inputs.s, inputs.qk_rope_head_dim)

    """ kv_cache output, [b,1,s2,kv_lora_rank] """
    kv_cache_tmp = inputs.kv_cache.clone()
    kv_cache_out = scatter_update([kv_cache_tmp, k_nope, inputs.cache_index], -2)

    """ kr_cache output, [b,1,s2,qk_rope_head_dim] """
    kr_cache_tmp = inputs.kr_cache.clone()
    kr_cache_out = scatter_update([kr_cache_tmp, k_embed_r, inputs.cache_index], -2)

    return q_embed, kv_cache_out, kr_cache_out


@dataclass
class _ComputeQPathInputs:
    x: Any
    w_dq: Any
    w_uqqr: Any
    w_uk: Any
    gamma_cq: Any
    cos: Any
    dtype: Any
    is_quant_a: Any
    is_quant_b: Any
    w_dq_scale: Any
    w_uqqr_scale: Any


@dataclass


class _ComputeQPathOutputs:
    q_out: Any
    q_reshape: Any
    q_a_layernorm: Any
    q_a_layernorm_scale_dequant: Any
    x_2d: Any
    x_2d_quant: Any
    x_2d_scale_dequant: Any


def _compute_q_path(inputs: _ComputeQPathInputs):


    """Compute Q path: proj -> layernorm -> q_b_proj -> q_nope -> q_out."""
    b, s, h = inputs.x.shape
    qk_rope_head_dim = inputs.cos.shape[2]
    n, qk_nope_head_dim, kv_lora_rank = inputs.w_uk.shape
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    q_lora_rank = inputs.w_dq.shape[1]
    x_2d = inputs.x.reshape(b * s, h)
    x_2d_quant, x_2d_scale_dequant = quant_hif8(x_2d)
    if inputs.is_quant_a:
        q_a_proj = torch_npu.npu_quant_matmul(x_2d_quant, inputs.w_dq, inputs.w_dq_scale.view(q_lora_rank),
            pertoken_scale=x_2d_scale_dequant.view(b * s), x1_dtype=torch_npu.hifloat8,
            x2_dtype=torch_npu.hifloat8, output_dtype=torch_npu.float32)
    else:
        q_a_proj = torch.matmul(x_2d.to(torch.float32), inputs.w_dq.to(torch.float32))
    q_a_proj = q_a_proj.to(torch.bfloat16)
    q_a_layernorm = rms_norm(q_a_proj, inputs.gamma_cq)
    q_a_layernorm_scale_dequant = None
    if inputs.is_quant_b:
        q_a_layernorm, q_a_layernorm_scale_dequant = quant_hif8(q_a_layernorm)
        q_b_proj = torch_npu.npu_quant_matmul(q_a_layernorm, inputs.w_uqqr, inputs.w_uqqr_scale.view(n * q_head_dim),
            pertoken_scale=q_a_layernorm_scale_dequant.view(b * s), x1_dtype=torch_npu.hifloat8,
            x2_dtype=torch_npu.hifloat8, output_dtype=torch_npu.float32)
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
    return _ComputeQPathOutputs(
            q_out, q_reshape, q_a_layernorm, q_a_layernorm_scale_dequant,
            x_2d, x_2d_quant, x_2d_scale_dequant)


@dataclass
class _ComputeKvPathInputs:
    x_2d: Any
    x_2d_quant: Any
    x_2d_scale_dequant: Any
    w_dkvkr: Any
    gamma_ckv: Any
    is_quant_a: Any
    w_dkvkr_scale: Any
    dtype: Any
    b: Any
    s: Any
    kv_lora_rank: Any
    qk_rope_head_dim: Any


def _compute_kv_path(inputs: _ComputeKvPathInputs):


    """Compute KV path: proj -> rms_norm -> k_nope."""
    if inputs.is_quant_a:
        kv_a_proj = torch_npu.npu_quant_matmul(
            inputs.x_2d_quant, inputs.w_dkvkr,
            inputs.w_dkvkr_scale.view(inputs.kv_lora_rank + inputs.qk_rope_head_dim),
            pertoken_scale=inputs.x_2d_scale_dequant.view(inputs.b * inputs.s), x1_dtype=torch_npu.hifloat8,
            x2_dtype=torch_npu.hifloat8, output_dtype=torch_npu.float32)
    else:
        kv_a_proj = torch.matmul(inputs.x_2d.to(torch.float32), inputs.w_dkvkr.to(torch.float32))
    kv_a_proj = kv_a_proj.to(inputs.dtype)
    kv_reshape = kv_a_proj.reshape(inputs.b, inputs.s, inputs.kv_lora_rank + inputs.qk_rope_head_dim)
    compressed_kv = kv_reshape[:, :, 0:inputs.kv_lora_rank]
    compressed_kv_norm = rms_norm(compressed_kv, inputs.gamma_ckv)
    compressed_kv_r = compressed_kv_norm.reshape(inputs.b, inputs.s, 1, inputs.kv_lora_rank)
    k_nope = compressed_kv_r.reshape(inputs.b * inputs.s * 1, inputs.kv_lora_rank)
    return kv_a_proj, kv_reshape, k_nope


def _compute_rope_and_cache(q_reshape, kv_reshape, cos, sin,
                              kv_cache, kr_cache, k_nope, cache_index):
    """Compute RoPE embeddings and update caches."""
    qk_nope_head_dim = q_reshape.shape[-1] - cos.shape[2]
    qk_rope_head_dim = cos.shape[2]
    kv_lora_rank = kv_reshape.shape[-1] - qk_rope_head_dim
    b, s = q_reshape.shape[0], q_reshape.shape[1]
    q_pe = q_reshape[:, :, :, qk_nope_head_dim:]
    k_pe = kv_reshape[:, :, kv_lora_rank:]
    k_pe_r = k_pe.reshape(b, s, 1, qk_rope_head_dim)
    q_embed, k_embed = apply_rotary_pos_emb_v2(q_pe, k_pe_r, cos, sin, 2)
    k_embed_r = k_embed.reshape(b * 1 * s, qk_rope_head_dim)
    kv_cache_tmp = kv_cache.clone()
    kv_cache_out = scatter_update([kv_cache_tmp, k_nope, cache_index], -2)
    kr_cache_tmp = kr_cache.clone()
    kr_cache_out = scatter_update([kr_cache_tmp, k_embed_r, cache_index], -2)
    return q_embed, kv_cache_out, kr_cache_out


def mla_prolog_quant_v32_compute(inputs):
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
    if is_quant_a:
        w_dq_scale = inputs.get("w_dq_scale")
        w_dkvkr_scale = inputs.get("w_dkvkr_scale")
    if is_quant_b:
        w_uqqr_scale = inputs.get("w_uqqr_scale")
        if has_smooth:
            smooth_cq = inputs.get("smooth_cq")

    q_outputs = \
        _compute_q_path(_ComputeQPathInputs(x=x, w_dq=w_dq, w_uqqr=w_uqqr, w_uk=w_uk,
                        gamma_cq=gamma_cq, cos=cos, dtype=dtype,
                        is_quant_a=is_quant_a, is_quant_b=is_quant_b,
                        w_dq_scale=w_dq_scale if is_quant_a else None,
                        w_uqqr_scale=w_uqqr_scale if is_quant_b else None))
    q_out = q_outputs.q_out
    q_reshape = q_outputs.q_reshape
    q_a_layernorm = q_outputs.q_a_layernorm
    q_a_layernorm_scale_dequant = q_outputs.q_a_layernorm_scale_dequant
    x_2d = q_outputs.x_2d
    x_2d_quant = q_outputs.x_2d_quant
    x_2d_scale_dequant = q_outputs.x_2d_scale_dequant

    b, s, h = x.shape
    qk_rope_head_dim = cos.shape[2]
    n, qk_nope_head_dim, kv_lora_rank = w_uk.shape

    kv_a_proj, kv_reshape, k_nope = _compute_kv_path(
        _ComputeKvPathInputs(x_2d=x_2d, x_2d_quant=x_2d_quant,
        x_2d_scale_dequant=x_2d_scale_dequant, w_dkvkr=w_dkvkr, gamma_ckv=gamma_ckv,
        is_quant_a=is_quant_a, w_dkvkr_scale=w_dkvkr_scale if is_quant_a else None,
        dtype=dtype, b=b, s=s, kv_lora_rank=kv_lora_rank, qk_rope_head_dim=qk_rope_head_dim))

    q_embed, kv_cache_out, kr_cache_out = _compute_rope_and_cache(
        q_reshape, kv_reshape, cos, sin, kv_cache, kr_cache, k_nope, cache_index)

    return MlaPrologHifpResult(q_out, q_embed, q_a_layernorm, q_a_layernorm_scale_dequant, kv_cache_out,
                              kr_cache_out)


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
class _AssembleInputDataResInputs:
    x: Any
    w_dq: Any
    w_uqqr: Any
    smooth_cq: Any
    scale_data: Any
    w_dkvkr: Any
    w_uk: Any
    gamma_cq: Any
    gamma_ckv: Any
    cos: Any
    sin: Any
    cache_index: Any
    kv_cache: Any
    kr_cache: Any
    block_num: Any
    block_table: Any


def _assemble_input_data_res(inputs: _AssembleInputDataResInputs):


    res = [None] * 16
    res[0] = inputs.x
    res[1] = inputs.w_dq
    res[2] = inputs.w_uqqr
    res[3] = inputs.smooth_cq
    res[4] = inputs.scale_data
    res[5] = inputs.w_dkvkr
    res[6] = inputs.w_uk
    res[7] = inputs.gamma_cq
    res[8] = inputs.gamma_ckv
    res[9] = inputs.cos
    res[10] = inputs.sin
    res[11] = inputs.cache_index
    res[12] = inputs.kv_cache
    res[13] = inputs.kr_cache
    res[14] = inputs.block_num
    res[15] = inputs.block_table
    return res


def _create_and_quantize_weights(x_shape, w_qa_shape, w_qb_shape, w_kv_a_shape,
                                 dtype, w_dtype, is_quant_a, is_quant_b, has_smooth,
                                 smooth_cq_shape):
    x = torch.empty(x_shape).uniform_(-1, 1).to(dtype)
    w_dq = torch.empty(w_qa_shape).uniform_(-0.1, 0.1).to(w_dtype).npu()
    w_uqqr = torch.empty(w_qb_shape).uniform_(-0.1, 0.1).to(w_dtype).npu()
    w_dkvkr = torch.empty(w_kv_a_shape).uniform_(-0.1, 0.1).to(w_dtype).npu()
    scale_data = dict()
    smooth_cq = None

    if is_quant_a:
        w_dq, w_qa_scale = quant_hif8(w_dq, -2)
        w_dkvkr, w_kva_scale = quant_hif8(w_dkvkr, -2)
        scale_data["w_dq"] = w_qa_scale
        scale_data["w_dkvkr"] = w_kva_scale

    if is_quant_b:
        w_uqqr, w_qb_scale = quant_hif8(w_uqqr, -2)
        scale_data["w_uqqr"] = w_qb_scale
        # smooth_data
        if has_smooth:
            smooth_cq = torch.empty(smooth_cq_shape).uniform_(-1, 1).to(torch.float32)

    return x, w_dq, w_uqqr, w_dkvkr, scale_data, smooth_cq


def _build_kv_kr_cache(b, block_table, block_num, block_size, skv_max,
                       k_bsnd, kv_lora_rank, qk_rope_head_dim, dtype):
    # kv paddIng
    per_batch_max_num = math.ceil(skv_max / block_size)
    k_tensor_bsnd = torch.zeros((b, per_batch_max_num * block_size, 1, kv_lora_rank + qk_rope_head_dim)).to(dtype)
    k_tensor_bsnd[:, :k_bsnd.shape[1], :, :] = k_bsnd[:, :, :, :]
    # kv_cache
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


def gen_mla_prolog_quant_v32_input_data(params, dtypes, actual_seq, is_quant=(False, False),
                                        has_smooth=False, block_size=128, cache_mode="BSND"):
    dtype, w_dtype = dtypes

    is_quant_a, is_quant_b = is_quant
    b = params.get("b")
    s = params.get("s")  # s=1 or 2
    s1 = params.get("s1")  # s2=4k
    h = params.get("h")
    n = params.get("n1")
    q_lora_rank = params.get("q_lora_rank")
    qk_nope_head_dim = params.get("qk_nope_head_dim")
    qk_rope_head_dim = params.get("qk_rope_head_dim")
    kv_lora_rank = params.get("kv_lora_rank")
    block_num, block_table, cache_index = gen_block_table(actual_seq, block_size, s1, need_indices=True)

    skv_max = actual_seq.max()
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    x_shape = [b, s, h]
    w_qa_shape = [h, q_lora_rank]
    w_qb_shape = [q_lora_rank, n * q_head_dim]
    w_kv_a_shape = [h, kv_lora_rank + qk_rope_head_dim]
    w_kv_b_k_shape = [n, qk_nope_head_dim, kv_lora_rank]
    gamma_cq_shape = [q_lora_rank]
    gamma_ckv_shape = [kv_lora_rank]
    cos_shape = [b, s, qk_rope_head_dim]
    kv_bsnd_shape = [b, skv_max, 1, kv_lora_rank + qk_rope_head_dim]
    kv_cache_shape = [block_num, block_size, 1, kv_lora_rank]
    kr_cache_shape = [block_num, block_size, 1, qk_rope_head_dim]
    kv_quant_scale_cache_shape = [block_num, block_size, 1, 4]
    smooth_cq_shape = [1, q_lora_rank]

    x, w_dq, w_uqqr, w_dkvkr, scale_data, smooth_cq = \
        _create_and_quantize_weights(x_shape, w_qa_shape, w_qb_shape, w_kv_a_shape,
                                     dtype, w_dtype, is_quant_a, is_quant_b, has_smooth,
                                     smooth_cq_shape)

    w_uk = torch.empty(w_kv_b_k_shape).uniform_(-0.1, 0.1).to(w_dtype)
    gamma_cq = torch.empty(gamma_cq_shape).uniform_(-1, 1).to(dtype)  # [q_lora_rank]
    gamma_ckv = torch.empty(gamma_ckv_shape).uniform_(-1, 1).to(dtype)  # [kv_lora_rank]
    cos = torch.empty(cos_shape).uniform_(-0.1, 0.1).to(dtype)  # [b, s, qk_rope_head_dim]
    sin = torch.empty(cos_shape).uniform_(-0.1, 0.1).to(dtype)  # [b, s, qk_rope_head_dim]
    k_bsnd = torch.empty(kv_bsnd_shape).uniform_(-1, 1).to(dtype)

    kv_cache, kr_cache = _build_kv_kr_cache(b, block_table, block_num, block_size, skv_max,
                                             k_bsnd, kv_lora_rank, qk_rope_head_dim, dtype)

    return _assemble_input_data_res(_AssembleInputDataResInputs(
        x=x, w_dq=w_dq, w_uqqr=w_uqqr, smooth_cq=smooth_cq, scale_data=scale_data,
        w_dkvkr=w_dkvkr, w_uk=w_uk, gamma_cq=gamma_cq, gamma_ckv=gamma_ckv,
        cos=cos, sin=sin, cache_index=cache_index, kv_cache=kv_cache, kr_cache=kr_cache,
        block_num=block_num, block_table=block_table))


def gen_mla_prolog_quant_v32_data(params, dtypes, actual_seq, is_quant=(False, False),
                                  has_smooth=False, block_size=128, cache_mode="BSND"):
    dtype, w_dtype = dtypes
    x, w_dq, w_uqqr, smooth_cq, scale_data, w_dkvkr, w_uk, gamma_cq, gamma_ckv, cos, sin, kv_len, \
        kv_cache, kr_cache, block_num, block_table = \
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
    inputs["cache_index"] = kv_len
    if is_quant_a:
        inputs["w_dq_scale"] = scale_data["w_dq"]
        inputs["w_dkvkr_scale"] = scale_data["w_dkvkr"]
    if is_quant_b:
        inputs["w_uqqr_scale"] = scale_data["w_uqqr"]
        if has_smooth:
            inputs["smooth_cq"] = smooth_cq

    if torch_npu.npu.is_available():
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.npu()

    q_out, q_embed, rms_norm_out, rms_norm_scale, kv_cache_out, kr_cache_out = mla_prolog_quant_v32_compute(inputs)
    outputs = {"q_golden": q_out, "q_rope": q_embed, "kr_golden": kr_cache_out, "kv_golden": kv_cache_out}

    outputs["rms_norm_golden"] = rms_norm_out
    outputs["rms_norm_scale_golden"] = rms_norm_scale

    return inputs, outputs


def _compute_mla_shapes_and_golden(params, golden_data):
    b = params['b']
    s = params['s']
    t = b * s
    s2 = params['s2']
    n1 = params['n1']
    n2 = 1
    h = params['h']
    q_lora_rank = params['q_lora_rank']
    qk_nope_head_dim = params['qk_nope_head_dim']
    qk_rope_head_dim = params['qk_rope_head_dim']
    kv_lora_rank = params["kv_lora_rank"]
    block_size = params['block_size']
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    token_x_shape = [t, h]
    w_dq_shape = [h, q_lora_rank]
    w_uq_qr_shape = [q_lora_rank, n1 * q_head_dim]
    dequant_scale_w_uq_qr_shape = [n1 * q_head_dim, 1]
    w_dkv_kr_shape = [h, kv_lora_rank + qk_rope_head_dim]
    w_uk_shape = [n1, qk_nope_head_dim, kv_lora_rank]
    rope_cos_shape = [t, qk_rope_head_dim]
    rmsnorm_gamma_cq_shape = [q_lora_rank]
    rmsnorm_gamma_ckv_shape = [kv_lora_rank]
    cache_index_shape = [t]
    block_num = b * ((s2 + block_size - 1) // block_size)
    kv_cache_shape = [block_num, block_size, n2, kv_lora_rank]
    kr_cache_shape = [block_num, block_size, n2, qk_rope_head_dim]
    # output
    kv_cache_out_shape = [block_num, block_size, n2, kv_lora_rank]
    kr_cache_out_shape = [block_num, block_size, n2, qk_rope_head_dim]
    q_nope_out_shape = [t, n1, kv_lora_rank]
    q_rope_out_shape = [t, n1, qk_rope_head_dim]

    # golden data
    golden1 = golden_data["q_golden"].reshape(q_nope_out_shape)
    golden2 = golden_data["q_rope"] .reshape(q_rope_out_shape)

    golden3 = golden_data["kv_golden"].reshape(kv_cache_out_shape)
    golden4 = golden_data["kr_golden"].reshape(kr_cache_out_shape)

    shapes = {
        "token_x_shape": token_x_shape, "w_dq_shape": w_dq_shape,
        "w_uq_qr_shape": w_uq_qr_shape, "w_dkv_kr_shape": w_dkv_kr_shape,
        "w_uk_shape": w_uk_shape, "rope_cos_shape": rope_cos_shape,
        "rmsnorm_gamma_cq_shape": rmsnorm_gamma_cq_shape,
        "rmsnorm_gamma_ckv_shape": rmsnorm_gamma_ckv_shape,
        "cache_index_shape": cache_index_shape,
        "kv_cache_shape": kv_cache_shape, "kr_cache_shape": kr_cache_shape,
        "q_nope_out_shape": q_nope_out_shape, "q_rope_out_shape": q_rope_out_shape,
    }

    return shapes, golden1, golden2, golden3, golden4


def _prepare_mla_input_output_tensors(input_tensors, shapes, dtype, is_quant_a, is_quant_b, nz):
    output_q_nope_data = torch.empty(shapes["q_nope_out_shape"], dtype=dtype).npu()
    output_q_rope_data = torch.empty(shapes["q_rope_out_shape"], dtype=dtype).npu()
    output_kv_cache_data = input_tensors["kv_cache"].reshape(shapes["kv_cache_shape"]).npu()
    output_kr_cache_data = input_tensors["kr_cache"].reshape(shapes["kr_cache_shape"]).npu()

    if nz:
        w_dq_nz = torch_npu.npu_format_cast(input_tensors["w_dq"].reshape(shapes["w_dq_shape"]).npu().contiguous(), \
                                            torch_npu.Format.FRACTAL_NZ)
        w_dkvkr_nz = torch_npu.npu_format_cast(
            input_tensors["w_dkvkr"].reshape(shapes["w_dkv_kr_shape"]).npu().contiguous(),
            torch_npu.Format.FRACTAL_NZ)
        w_uqqr_nz = torch_npu.npu_format_cast(
            input_tensors["w_uqqr"].reshape(shapes["w_uq_qr_shape"]).npu().contiguous(),
            torch_npu.Format.FRACTAL_NZ)
        input_tensors["w_uqqr"] = w_uqqr_nz
        input_tensors["w_dkvkr"] = w_dkvkr_nz
        input_tensors["w_dq"] = w_dq_nz

    # input data
    token_x_data = input_tensors["x"].reshape(shapes["token_x_shape"]).npu()
    w_dq_data = input_tensors["w_dq"].reshape(shapes["w_dq_shape"]).npu()
    w_uq_qr_data = input_tensors["w_uqqr"].reshape(shapes["w_uq_qr_shape"]).npu()
    w_uk_data = input_tensors["w_uk"].reshape(shapes["w_uk_shape"]).npu()
    w_dkv_kr_data = input_tensors["w_dkvkr"].reshape(shapes["w_dkv_kr_shape"]).npu()
    rmsnorm_gamma_cq_data =  \
                    input_tensors["gamma_cq"].reshape(shapes["rmsnorm_gamma_cq_shape"]).npu()
    rmsnorm_gamma_ckv_data = input_tensors["gamma_ckv"].reshape(shapes["rmsnorm_gamma_ckv_shape"]).npu()
    rope_cos_data = input_tensors["cos"].reshape(shapes["rope_cos_shape"]).npu()
    rope_sin_data = input_tensors["sin"].reshape(shapes["rope_cos_shape"]).npu()
    cache_index_data = input_tensors["cache_index"].reshape(shapes["cache_index_shape"]).npu()
    kv_cache_data = input_tensors["kv_cache"].reshape(shapes["kv_cache_shape"]).npu()
    kr_cache_data = input_tensors["kr_cache"].reshape(shapes["kr_cache_shape"]).npu()

    if is_quant_a:
        w_dq_scale_data = input_tensors["w_dq_scale"].npu()
        w_dkvkr_scale_data = input_tensors["w_dkvkr_scale"].npu()
    else:
        w_dq_scale_data = torch.Tensor().npu()
        w_dkvkr_scale_data = torch.Tensor().npu()

    if is_quant_b:
        w_uqqr_scale_data =  \
                input_tensors["w_uqqr_scale"].npu()
    else:
        w_uqqr_scale_data = torch.Tensor().npu()

    input_data = [token_x_data, w_dq_data, w_dq_scale_data, w_uq_qr_data, w_uqqr_scale_data,
                w_uk_data, w_dkv_kr_data, w_dkvkr_scale_data, rmsnorm_gamma_cq_data, rmsnorm_gamma_ckv_data,
                rope_cos_data, rope_sin_data, cache_index_data,
                kv_cache_data, kr_cache_data]
    output_data = [output_q_nope_data, output_q_rope_data, output_kv_cache_data, output_kr_cache_data]

    return input_data, output_data


def _compare_mla_outputs(output_q_nope_data, output_q_rope_data, output_kv_cache_data, output_kr_cache_data,
                         golden1, golden2, golden3, golden4):
    torch_npu.npu.synchronize()

    ########### compare #######
    logging.info("qNope =======")
    compare(output_q_nope_data.cpu(), golden1.cpu(), "qNope", atol=0.005, rtol=0.0078125, max_error_ratio=0.005)
    logging.info("qRope =======")
    compare(output_q_rope_data.cpu(), golden2.cpu(), "qRope", atol=0.005, rtol=0.0078125, max_error_ratio=0.005)
    logging.info("kv =======")
    compare(output_kv_cache_data.cpu(), golden3.cpu(), "kv", atol=0.0001, rtol=0.0078125, max_error_ratio=0)
    logging.info("kr =======")
    compare(output_kr_cache_data.cpu(), golden4.cpu(), "kr", atol=0.0001, rtol=0.0078125, max_error_ratio=0)


def mla_prolog_quant_v32(params, input_tensors, golden_data, dtype, is_quant_a, \
                        is_quant_b, nz, tile_config):

    shapes, golden1, golden2, golden3, golden4 = _compute_mla_shapes_and_golden(params, golden_data)
    input_data, output_data = _prepare_mla_input_output_tensors(
        input_tensors, shapes, dtype, is_quant_a, is_quant_b, nz)

    rope_tile_shape = RopeTileShapeConfig(two_dim=[32, 64], three_dim=[32, 32, 128], four_dim=[16, 128, 128, 128])
    mla_prolog_quant(*input_data, *output_data, 1e-5, 1e-5, tile_config, rope_tile_shape)

    _compare_mla_outputs(output_data[0], output_data[1], output_data[2], output_data[3],
                         golden1, golden2, golden3, golden4)


@pytest.mark.soc("950")
def test_b4_s64k2_pa_nd_bf16_quant():
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
    dtype = torch.bfloat16
    is_quant_a, is_quant_b, is_nz = True, True, False
    cache_mode = "PA_BSND"
    tile_config = MlaTileConfig()
    tile_config.tile_bs = 8

    c0 = 16
    m_tile_value = (min(32, tile_config.tile_bs) + c0 - 1) // c0 * c0
    mv_tile_value = min(8, tile_config.tile_bs)
    tile_config.m_tile = m_tile_value

    tile_config.pre_quant_cube_tile = [m_tile_value, m_tile_value, 256, 256, 128, 128]
    tile_config.mv_tile = mv_tile_value
    tile_config.q_vec_tile0 = 1
    tile_config.q_vec_tile1 = 32
    tile_config.k_vec_tile0 = 2
    tile_config.k_vec_tile1 = 512
    tile_config.unroll_list = [8, 4, 2, 1]

    actual_seq = torch.tensor([params["s2"]] * params["b"], dtype=torch.int32).unsqueeze(-1)
    input_tensors, golden_data = gen_mla_prolog_quant_v32_data(params, (torch.bfloat16, torch.bfloat16), actual_seq, \
                    (is_quant_a, is_quant_b), False, 128, cache_mode)
    mla_prolog_quant_v32(params, input_tensors, golden_data, dtype, \
                        is_quant_a, is_quant_b, is_nz, tile_config)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO
    )
    test_b4_s64k2_pa_nd_bf16_quant()
