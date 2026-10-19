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
import logging
import math
from pathlib import Path
import enum
import torch
import torch_npu
import numpy as np

import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

import pytest
import pypto

from experimental.ops_transformer.mla_prolog_mxfp_quant_v3.mla_prolog_mxfp_quant_impl import (
    mla_prolog_quant, MlaTileConfig, RopeTileShapeConfig
)
from common_utils import compare
import collections


MlaPrologMxfpResult = collections.namedtuple('MlaPrologMxfpResult', [
    'q_nope_new_t', 'q_embed', 'q_a_layernorm', 'q_a_layernorm_scale_dequant',
    'kv_cache_out', 'kr_cache_out'
])
GenBlockTableOutput = collections.namedtuple("GenBlockTableOutput", ["block_num", "block_table", "cache_index"])


FP8_E4M3_TARGET_MAX_POW2 = 8
FP8_E4M3_MAX_POS = 448.0
FP8_E4M3_MIN_NORMAL = 2 ** -6
FP8_E4M3_EXP_BIAS = 7
FP8_E4M3_MBITS = 3
F32_EXP_BIAS = 127
F32_MBITS = 23
E8M0_EXPONENT_BIAS = 127
QUANT_MX_GROUP_COLS = 32
QUANT_MX_SCALE_GROUP_COLS = 64


def prep_env():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))
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


def apply_rotary_pos_emb_v2(q, k, cos, sin, unsqueeze_dim=1):
    input_dtype = q.dtype
    if input_dtype != torch.float32:
        q = q.to(torch.float32)
        k = k.to(torch.float32)
    if cos.dtype != torch.float32:
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)

    cos = torch.unsqueeze(cos, dim=unsqueeze_dim)  # [t,1,qk_d]
    sin = torch.unsqueeze(sin, dim=unsqueeze_dim)  # [t,1,qk_d]

    t, h, d = q.shape
    q = q.reshape(t, h, d // 2, 2).permute(0, 1, 3, 2).reshape(t, h, d)  # [t,n,qk_d]

    t, h, d = k.shape
    k = k.reshape(t, h, d // 2, 2).permute(0, 1, 3, 2).reshape(t, h, d)  # [t,1,qk_d]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    if input_dtype != torch.float32:
        q_embed, k_embed = q_embed.to(input_dtype), k_embed.to(input_dtype)
    return q_embed, k_embed


def compute_shared_exponents(max_abs: np.ndarray) -> np.ndarray:
    nan_mask = np.isnan(max_abs)
    bits = max_abs.astype(np.float32, copy=False).view(np.int32)
    fp_exponent = ((bits >> F32_MBITS) & 0xFF).astype(np.int32)
    biased = np.clip(fp_exponent - FP8_E4M3_TARGET_MAX_POW2, 0, 254).astype(np.uint8)
    biased[nan_mask] = 0xFF
    return biased


def compute_scalings_from_exponents(e8m0: np.ndarray) -> np.ndarray:
    e8m0_i32 = e8m0.astype(np.int32)
    scale_exp = np.int32(254) - e8m0_i32
    result = (scale_exp << F32_MBITS).astype(np.int32).view(np.float32)
    result[scale_exp == 0] = np.float32(math.ldexp(1.0, -E8M0_EXPONENT_BIAS))
    result[e8m0 == 0xFF] = np.float32(np.nan)
    return result


def encode_e4m3_fn_vectorized(values: np.ndarray) -> np.ndarray:
    shift = F32_MBITS - FP8_E4M3_MBITS
    magic_adder = np.int32((1 << (shift - 1)) - 1)
    denorm_exp = (F32_EXP_BIAS - FP8_E4M3_EXP_BIAS) + shift + 1
    denorm_mask_int = np.int32(denorm_exp << F32_MBITS)
    denorm_mask_float = np.array(denorm_mask_int, dtype=np.int32).view(np.float32)
    val_to_add = np.int32(((FP8_E4M3_EXP_BIAS - F32_EXP_BIAS) << F32_MBITS) + int(magic_adder))

    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.int32)
    sign = ((bits >> 24) & np.int32(0x80)).astype(np.uint8)
    abs_bits = bits & np.int32(0x7FFFFFFF)
    abs_val = abs_bits.view(np.float32).copy()

    nan_mask = np.isnan(values)
    saturate_mask = abs_val >= np.float32(FP8_E4M3_MAX_POS)
    denormal_mask = (~saturate_mask) & (abs_val < np.float32(FP8_E4M3_MIN_NORMAL)) & (~nan_mask)
    normal_mask = (~saturate_mask) & (~denormal_mask) & (~nan_mask)

    denorm_result = (abs_val + denorm_mask_float).view(np.int32) - denorm_mask_int
    denorm_result = denorm_result.astype(np.uint8)

    mant_odd = ((abs_bits >> np.int32(shift)) & np.int32(1)).astype(np.int32)
    normal_result = abs_bits + val_to_add + mant_odd
    normal_result = ((normal_result >> np.int32(shift)) & np.int32(0x7F)).astype(np.uint8)

    result = np.where(saturate_mask, np.uint8(0x7E), np.uint8(0))
    result = np.where(denormal_mask, denorm_result, result)
    result = np.where(normal_mask, normal_result, result)
    result = np.where(nan_mask, np.uint8(0x7F), result)
    return (result | sign).astype(np.uint8)


def quant_mx_golden_bytes(input_tensor: torch.Tensor, is_trans=False):
    if is_trans:
        input_tensor = input_tensor.T
    x = input_tensor.cpu().numpy().astype(np.float32, copy=False)
    cols = x.shape[-1]
    rows = x.size // cols
    group_cols = (cols + QUANT_MX_GROUP_COLS - 1) // QUANT_MX_GROUP_COLS
    scale_group_cols = (cols + QUANT_MX_SCALE_GROUP_COLS - 1) // QUANT_MX_SCALE_GROUP_COLS

    x_flat = x.reshape(rows, cols)
    padded_cols = group_cols * QUANT_MX_GROUP_COLS
    x_padded = np.zeros((rows, padded_cols), dtype=np.float32)
    x_padded[:, :cols] = x_flat
    x_grouped = x_padded.reshape(rows, group_cols, QUANT_MX_GROUP_COLS)

    max_abs = np.max(np.abs(x_grouped), axis=2).astype(np.float32)
    e8m0 = compute_shared_exponents(max_abs)
    group_scaling = compute_scalings_from_exponents(e8m0)
    quant_grouped = encode_e4m3_fn_vectorized(x_grouped * group_scaling[:, :, np.newaxis])

    quant = quant_grouped.reshape(rows, padded_cols)[:, :cols].reshape(x.shape)
    scale_shape = list(x.shape[:-1]) + [scale_group_cols, 2]
    scale = np.zeros(scale_shape, dtype=np.uint8)
    scale.reshape(rows, scale_group_cols * 2)[:, :group_cols] = e8m0.reshape(rows, group_cols)
    if is_trans:
        return torch.from_numpy(np.transpose(quant, (1, 0)).copy()).view(torch.float8_e4m3fn), \
                torch.from_numpy(np.transpose(scale, (1, 0, 2)).copy()).view(torch.float8_e8m0fnu)
    else:
        return torch.from_numpy(quant.copy()).view(torch.float8_e4m3fn), \
                torch.from_numpy(scale.copy()).view(torch.float8_e8m0fnu)


def tensor_to_file(t: torch.Tensor, output: Path):
    with open(str(output), "wb") as f:
        dtype = t.dtype
        if dtype == torch.bfloat16:
            dtype = torch.int16
        for each in t:
            f.write(each.view(dtype).cpu().numpy().tobytes())



def _compute_q_path(inputs, t, h, q_lora_rank, q_head_dim, qk_nope_head_dim, kv_lora_rank, n):
    """Compute Q projection path: q_a_proj -> layernorm -> q_b_proj -> q_nope_new_t."""
    dtype = inputs.get("dtype")
    is_quant_a = inputs.get("is_quant_a")
    is_quant_b = inputs.get("is_quant_b")
    x = inputs.get("x")
    w_dq = inputs.get("w_dq")
    gamma_cq = inputs.get("gamma_cq")
    w_uqqr = inputs.get("w_uqqr")
    w_uk = inputs.get("w_uk")

    # shape is: [t, h] @ [h, q_lora_rank] -> [t, q_lora_rank]
    if is_quant_a:
        # no smooth
        x_scale = inputs.get("x_scale")
        w_dq_scale = inputs.get("w_dq_scale")
        q_a_proj = torch_npu.npu_quant_matmul(x.view(t, h).npu(), w_dq.view(h, q_lora_rank).npu(), \
            w_dq_scale.npu(), pertoken_scale=x_scale.view(t, h // 64, 2).npu(), \
            x1_dtype=torch.float8_e4m3fn, x2_dtype=torch.float8_e4m3fn, output_dtype=torch.float32, \
        pertoken_scale_dtype=torch.float8_e8m0fnu, scale_dtype=torch.float8_e8m0fnu, group_sizes=[1, 1, 32])
    else:
        # matmul use float32 for arm, arm平台matmul在bfloat16数据类型下表现与x86平台不一致，通过升精度保证正确性
        q_a_proj = torch.matmul(x.to(torch.float32), w_dq.to(torch.float32))  # [t, q_lora_rank]

    q_a_layernorm = rms_norm(q_a_proj, gamma_cq)

    # shape is: [t, q_lora_rank] @ [q_lora_rank, n * q_head_dim] -> [t, n * q_head_dim]
    q_a_layernorm_scale_dequant = None
    if is_quant_b:
        q_a_layernorm, q_a_layernorm_scale_dequant = quant_mx_golden_bytes(q_a_layernorm)  # scale: [t,1]
        w_uqqr_scale = inputs.get("w_uqqr_scale")
        q_b_proj = torch_npu.npu_quant_matmul(q_a_layernorm.view(t, q_lora_rank).npu(), \
            w_uqqr.view(q_lora_rank, n * q_head_dim).npu(), w_uqqr_scale.npu(), \
            pertoken_scale=q_a_layernorm_scale_dequant.view(t, q_lora_rank // 64, 2).npu(), \
            x1_dtype=torch.float8_e4m3fn, x2_dtype=torch.float8_e4m3fn, output_dtype=torch.float32, \
        pertoken_scale_dtype=torch.float8_e8m0fnu, scale_dtype=torch.float8_e8m0fnu, group_sizes=[1, 1, 32])
    else:
        q_b_proj = torch.matmul(q_a_layernorm.to(torch.float32), w_uqqr.to(torch.float32))  # [b * s, n * q_head_dim]

    q_b_proj = q_b_proj.to(dtype)
    q_reshape = q_b_proj.reshape(t, n, q_head_dim)
    q_nope = q_reshape[:, :, 0:qk_nope_head_dim]  # [t, n, qk_nope_head_dim]
    q_nope_t = q_nope.permute(1, 0, 2)  # [n, t, qk_nope_head_dim]
    # shape is: [n, t, qk_nope_head_dim] @ [n, qk_nope_head_dim, kv_lora_rank] -> [n, t, kv_lora_rank]
    # matmul use float32 for arm, arm平台matmul在bfloat16数据类型下表现与x86平台不一致，通过升精度保证正确性
    q_nope_new = torch.matmul(q_nope_t.to(torch.float32), w_uk.to(torch.float32))
    q_nope_new = q_nope_new.to(dtype)
    q_nope_new_t = q_nope_new.permute(1, 0, 2)  # [t, n, kv_lora_rank]

    return q_nope_new_t, q_reshape, q_a_layernorm, q_a_layernorm_scale_dequant


def _compute_kv_rope_cache_path(inputs, t, h, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim, q_reshape):
    """Compute KV projection, RoPE embedding, and cache scatter update."""
    dtype = inputs.get("dtype")
    is_quant_a = inputs.get("is_quant_a")
    x = inputs.get("x")
    w_dkvkr = inputs.get("w_dkvkr")
    gamma_ckv = inputs.get("gamma_ckv")
    cos = inputs.get("cos")
    sin = inputs.get("sin")
    kv_cache = inputs.get("kv_cache")
    kr_cache = inputs.get("kr_cache")
    cache_index = inputs.get("cache_index")

    """ kv """
    # shape is: [t, h] @ [h, kv_lora_rank + qk_rope_head_dim] -> [t, kv_lora_rank + qk_rope_head_dim]
    if is_quant_a:
        x_scale = inputs.get("x_scale")
        w_dkvkr_scale = inputs.get("w_dkvkr_scale")
        kv_a_proj = torch_npu.npu_quant_matmul(x.view(t, h).npu(), \
            w_dkvkr.view(h, kv_lora_rank + qk_rope_head_dim).npu(), w_dkvkr_scale.npu(), \
            pertoken_scale=x_scale.view(t, h // 64, 2).npu(), x1_dtype=torch.float8_e4m3fn, \
            x2_dtype=torch.float8_e4m3fn, output_dtype=torch.float32, \
        pertoken_scale_dtype=torch.float8_e8m0fnu, scale_dtype=torch.float8_e8m0fnu, group_sizes=[1, 1, 32])
    else:
        # matmul use float32 for arm, arm平台matmul在bfloat16数据类型下表现与x86平台不一致，通过升精度保证正确性
        kv_a_proj = torch.matmul(x.to(torch.float32),
                                 w_dkvkr.to(torch.float32))  # [b * s, kv_lora_rank + qk_rope_head_dim]

    kv_a_proj = kv_a_proj.to(dtype)
    kv_reshape = kv_a_proj.reshape(t, kv_lora_rank + qk_rope_head_dim)

    compressed_kv = kv_reshape[:, 0:kv_lora_rank]  # [t, kv_lora_rank]
    compressed_kv_norm = rms_norm(compressed_kv, gamma_ckv)
    compressed_kv_quant_scale = None

    compressed_kv_r = compressed_kv_norm.reshape(t, 1, kv_lora_rank)
    k_nope = compressed_kv_r.reshape(t * 1, kv_lora_rank)

    """ RoPE """
    q_pe = q_reshape[:, :, qk_nope_head_dim:]  # [t, n, qk_rope_head_dim]

    k_pe = kv_reshape[:, kv_lora_rank:]  # [t, qk_rope_head_dim]
    k_pe_r = k_pe.reshape(t, 1, qk_rope_head_dim)

    # q_embed: [t, n, qk_rope_head_dim], k_embed: [t, 1, qk_rope_head_dim]
    q_embed, k_embed = apply_rotary_pos_emb_v2(q_pe, k_pe_r, cos, sin, 1)
    k_embed_r = k_embed.reshape(t * 1, qk_rope_head_dim)

    """ kv_cache output, [b,1,s2,kv_lora_rank] """
    kv_cache_tmp = kv_cache.clone()
    kv_cache_out = scatter_update([kv_cache_tmp, k_nope, cache_index], -2)

    """ kr_cache output, [b,1,s2,qk_rope_head_dim] """
    kr_cache_tmp = kr_cache.clone()
    kr_cache_out = scatter_update([kr_cache_tmp, k_embed_r, cache_index], -2)

    return q_embed, kv_cache_out, kr_cache_out


def mla_prolog_quant_v32_compute(inputs):
    x = inputs.get("x")
    cos = inputs.get("cos")
    w_uk = inputs.get("w_uk")
    w_dq = inputs.get("w_dq")
    t, h = x.shape
    qk_rope_head_dim = cos.shape[1]
    n, qk_nope_head_dim, kv_lora_rank = w_uk.shape
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    q_lora_rank = w_dq.shape[1]

    q_nope_new_t, q_reshape, q_a_layernorm, q_a_layernorm_scale_dequant = \
        _compute_q_path(inputs, t, h, q_lora_rank, q_head_dim, qk_nope_head_dim, kv_lora_rank, n)
    q_embed, kv_cache_out, kr_cache_out = \
        _compute_kv_rope_cache_path(inputs, t, h, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim, q_reshape)

    return MlaPrologMxfpResult(q_nope_new_t, q_embed, q_a_layernorm, q_a_layernorm_scale_dequant, kv_cache_out,
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



def _init_and_quantize_weights(x_shape, w_qa_shape, w_qb_shape, w_kv_a_shape, smooth_cq_shape,
                                is_quant_a, is_quant_b, has_smooth):
    """Create input tensors and apply MX quantization if enabled."""
    res = [None] * 16
    x = torch.empty(x_shape).uniform_(-1, 1).to(torch.float32)
    res[0] = x
    w_dq = torch.empty(w_qa_shape).uniform_(-0.1, 0.1).to(torch.float32).npu()
    w_uqqr = torch.empty(w_qb_shape).uniform_(-0.1, 0.1).to(torch.float32).npu()
    w_dkvkr = torch.empty(w_kv_a_shape).uniform_(-0.1, 0.1).to(torch.float32).npu()
    res[4] = dict()

    if is_quant_a:
        x_quant, x_scale = quant_mx_golden_bytes(x)
        res[0] = x_quant
        res[4]["x"] = x_scale
        w_dq, w_qa_scale = quant_mx_golden_bytes(w_dq, is_trans=True)
        w_dkvkr, w_kva_scale = quant_mx_golden_bytes(w_dkvkr, is_trans=True)
        res[4]["w_dq"] = w_qa_scale
        res[4]["w_dkvkr"] = w_kva_scale

    if is_quant_b:
        w_uqqr, w_qb_scale = quant_mx_golden_bytes(w_uqqr, is_trans=True)
        res[4]["w_uqqr"] = w_qb_scale
        # smooth_data
        if has_smooth:
            smooth_cq = torch.empty(smooth_cq_shape).uniform_(-1, 1).to(torch.float32)
            res[3] = smooth_cq

    res[1] = w_dq
    res[2] = w_uqqr
    res[5] = w_dkvkr

    return res


def _build_kv_kr_cache(block_table, block_num, block_size, b, kv_lora_rank, qk_rope_head_dim, dtype, skv_max):
    """Build kv_cache and kr_cache from block table and random KV data."""
    kv_bsnd_shape = [b, skv_max, 1, kv_lora_rank + qk_rope_head_dim]
    k_bsnd = torch.empty(kv_bsnd_shape).uniform_(-1, 1).to(dtype)
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
    t = params.get("t")
    s1 = t // b
    h = params.get("h")
    n = params.get("n1")
    q_lora_rank = params.get("q_lora_rank")
    qk_nope_head_dim = params.get("qk_nope_head_dim")
    qk_rope_head_dim = params.get("qk_rope_head_dim")
    kv_lora_rank = params.get("kv_lora_rank")
    block_num, block_table, cache_index = gen_block_table(actual_seq, block_size, s1, need_indices=True)

    skv_max = actual_seq.max()
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    x_shape = [t, h]
    w_qa_shape = [h, q_lora_rank]
    w_qb_shape = [q_lora_rank, n * q_head_dim]
    w_kv_a_shape = [h, kv_lora_rank + qk_rope_head_dim]
    smooth_cq_shape = [1, q_lora_rank]

    res = _init_and_quantize_weights(x_shape, w_qa_shape, w_qb_shape, w_kv_a_shape, smooth_cq_shape,
                                      is_quant_a, is_quant_b, has_smooth)

    w_kv_b_k_shape = [n, qk_nope_head_dim, kv_lora_rank]
    gamma_cq_shape = [q_lora_rank]
    gamma_ckv_shape = [kv_lora_rank]
    cos_shape = [t, qk_rope_head_dim]
    res[6] = torch.empty(w_kv_b_k_shape).uniform_(-0.1, 0.1).to(w_dtype)
    res[7] = torch.empty(gamma_cq_shape).uniform_(-1, 1).to(dtype)
    res[8] = torch.empty(gamma_ckv_shape).uniform_(-1, 1).to(dtype)
    res[9] = torch.empty(cos_shape).uniform_(-0.1, 0.1).to(dtype)
    res[10] = torch.empty(cos_shape).uniform_(-0.1, 0.1).to(dtype)
    res[11] = cache_index.reshape(t, 1)

    kv_cache, kr_cache = _build_kv_kr_cache(block_table, block_num, block_size, b,
                                              kv_lora_rank, qk_rope_head_dim, dtype, skv_max)
    res[12] = kv_cache
    res[13] = kr_cache
    res[14] = block_num
    res[15] = block_table

    return res



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
        inputs["x_scale"] = scale_data["x"]
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



def _define_mla_prolog_shapes(params):
    """Compute all tensor shapes from model parameters."""
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

    return dict(
        b=b, s=s, t=t, s2=s2, n1=n1, n2=n2, h=h,
        q_lora_rank=q_lora_rank, qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim, kv_lora_rank=kv_lora_rank,
        block_size=block_size, q_head_dim=q_head_dim, block_num=block_num,
        token_x_shape=token_x_shape, w_dq_shape=w_dq_shape,
        w_uq_qr_shape=w_uq_qr_shape, dequant_scale_w_uq_qr_shape=dequant_scale_w_uq_qr_shape,
        w_dkv_kr_shape=w_dkv_kr_shape, w_uk_shape=w_uk_shape,
        rope_cos_shape=rope_cos_shape, rmsnorm_gamma_cq_shape=rmsnorm_gamma_cq_shape,
        rmsnorm_gamma_ckv_shape=rmsnorm_gamma_ckv_shape, cache_index_shape=cache_index_shape,
        kv_cache_shape=kv_cache_shape, kr_cache_shape=kr_cache_shape,
        kv_cache_out_shape=kv_cache_out_shape, kr_cache_out_shape=kr_cache_out_shape,
        q_nope_out_shape=q_nope_out_shape, q_rope_out_shape=q_rope_out_shape,
    )


def _apply_nz_format(input_tensors, shapes):
    """Apply FRACTAL_NZ format cast to weight tensors."""
    w_dq_nz = torch_npu.npu_format_cast(
        input_tensors["w_dq"].reshape(shapes["w_dq_shape"]).npu().contiguous(),
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


def _prepare_mla_prolog_tensors(shapes, input_tensors, golden_data, dtype, is_quant_a, is_quant_b, nz):
    """Prepare input and output tensors for mla_prolog_quant kernel invocation."""
    # golden data
    golden1 = golden_data["q_golden"].reshape(shapes["q_nope_out_shape"])
    golden2 = golden_data["q_rope"].reshape(shapes["q_rope_out_shape"])
    golden3 = golden_data["kv_golden"].reshape(shapes["kv_cache_out_shape"])
    golden4 = golden_data["kr_golden"].reshape(shapes["kr_cache_out_shape"])

    output_q_nope_data = torch.empty(shapes["q_nope_out_shape"], dtype=dtype).npu()
    output_q_rope_data = torch.empty(shapes["q_rope_out_shape"], dtype=dtype).npu()
    output_kv_cache_data = input_tensors["kv_cache"].reshape(shapes["kv_cache_shape"]).npu()
    output_kr_cache_data = input_tensors["kr_cache"].reshape(shapes["kr_cache_shape"]).npu()

    if nz:
        _apply_nz_format(input_tensors, shapes)


    # input data
    token_x_data = input_tensors["x"].reshape(shapes["token_x_shape"]).npu()
    w_dq_data = input_tensors["w_dq"].reshape(shapes["w_dq_shape"]).npu()
    w_uq_qr_data = input_tensors["w_uqqr"].reshape(shapes["w_uq_qr_shape"]).npu()
    w_uk_data = input_tensors["w_uk"].reshape(shapes["w_uk_shape"]).npu()
    w_dkv_kr_data = input_tensors["w_dkvkr"].reshape(shapes["w_dkv_kr_shape"]).npu()
    rmsnorm_gamma_cq_data = \
                    input_tensors["gamma_cq"].reshape(shapes["rmsnorm_gamma_cq_shape"]).npu()
    rmsnorm_gamma_ckv_data = input_tensors["gamma_ckv"].reshape(shapes["rmsnorm_gamma_ckv_shape"]).npu()
    rope_cos_data = input_tensors["cos"].reshape(shapes["rope_cos_shape"]).npu()
    rope_sin_data = input_tensors["sin"].reshape(shapes["rope_cos_shape"]).npu()
    cache_index_data = input_tensors["cache_index"].reshape(shapes["cache_index_shape"]).npu()
    kv_cache_data = input_tensors["kv_cache"].reshape(shapes["kv_cache_shape"]).npu()
    kr_cache_data = input_tensors["kr_cache"].reshape(shapes["kr_cache_shape"]).npu()

    if is_quant_a:
        x_scale_data = input_tensors["x_scale"].npu()
        w_dq_scale_data = input_tensors["w_dq_scale"].npu()
        w_dkvkr_scale_data = input_tensors["w_dkvkr_scale"].npu()
    else:
        x_scale_data = torch.Tensor().npu()
        w_dq_scale_data = torch.Tensor().npu()
        w_dkvkr_scale_data = torch.Tensor().npu()

    if is_quant_b:
        w_uqqr_scale_data = \
                input_tensors["w_uqqr_scale"].npu()
    else:
        w_uqqr_scale_data = torch.Tensor().npu()

    input_data = [token_x_data, x_scale_data, w_dq_data, w_dq_scale_data, w_uq_qr_data, w_uqqr_scale_data,
                w_uk_data, w_dkv_kr_data, w_dkvkr_scale_data, rmsnorm_gamma_cq_data, rmsnorm_gamma_ckv_data,
                rope_cos_data, rope_sin_data, cache_index_data,
                kv_cache_data, kr_cache_data]
    output_data = [output_q_nope_data, output_q_rope_data, output_kv_cache_data, output_kr_cache_data]
    goldens = [golden1, golden2, golden3, golden4]

    return input_data, output_data, goldens


def _compare_mla_prolog_results(output_data, goldens):
    """Compare kernel outputs against golden references."""
    output_q_nope_data, output_q_rope_data, output_kv_cache_data, output_kr_cache_data = output_data
    golden1, golden2, golden3, golden4 = goldens
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
    shapes = _define_mla_prolog_shapes(params)
    input_data, output_data, goldens = \
        _prepare_mla_prolog_tensors(shapes, input_tensors, golden_data, dtype, is_quant_a, is_quant_b, nz)
    rope_tile_shape = RopeTileShapeConfig(two_dim=[32, 64], three_dim=[32, 32, 128], four_dim=[16, 128, 128, 128])
    mla_prolog_quant(*input_data, *output_data, 1e-5, 1e-5, tile_config, rope_tile_shape)
    _compare_mla_prolog_results(output_data, goldens)



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
        's2': 64*1024,
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


@pytest.mark.soc("950")
def test_b64_s64k2_pa_nd_bf16_quant():
    '''
    mla_prolog decode测试函数
    '''
    torch.manual_seed(5)
    prep_env()
    params = {
        'b': 64,
        't': 128,
        's': 2,
        's1': 2,
        's2': 64*1024,
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
    tile_config.tile_bs = 128

    c0 = 16
    m_tile_value = (min(128, tile_config.tile_bs) + c0 - 1) // c0 * c0
    mv_tile_value = min(8, tile_config.tile_bs)
    tile_config.m_tile = m_tile_value

    tile_config.pre_quant_cube_tile = [m_tile_value, m_tile_value, 256, 256, 128, 128]
    tile_config.mv_tile = mv_tile_value
    tile_config.q_vec_tile0 = 32
    tile_config.q_vec_tile1 = 128
    tile_config.k_vec_tile0 = 32
    tile_config.k_vec_tile1 = 512
    tile_config.unroll_list = [128, 64, 8, 4, 2, 1]

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
