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
import os
import math
import logging
from dataclasses import dataclass
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
import numpy as np
import pypto

from deepseek_v32_exp.sparse_flash_attention_quant_impl \
    import sparse_flash_attention_quant_d, sparse_flash_attention_quant_p, \
           sparse_flash_attention_d_950, SaTileShapeConfig
from common_utils import compare, gen_uniform_data



def _gather_kv_cache(s2_tile_cur, topk_indices_tmp, kn, kr, kn_scales,
                      block_size, block_table, b_idx, s2_start, dk, dv, input_dtype, kn_dtype):
    """Gather KV cache entries for attention computation."""
    slc_kn = torch.zeros([s2_tile_cur, dk], dtype=kn_dtype)
    slc_kr = torch.zeros([s2_tile_cur, dv], dtype=input_dtype)
    slc_kn_scales = torch.zeros([s2_tile_cur, 4], dtype=torch.float32)
    offset = torch.zeros([s2_tile_cur], dtype=torch.int32)
    for cur_s2_idx in range(s2_tile_cur):
        topk_index = topk_indices_tmp[cur_s2_idx]
        block_idx_in_batch = topk_index // block_size
        slc_block_idx = block_table[b_idx, block_idx_in_batch]
        tail = topk_index % block_size
        offset[cur_s2_idx] = slc_block_idx * block_size + tail
    for cur_s2_idx in range(s2_tile_cur):
        slc_idx = offset[cur_s2_idx]
        slc_kn[cur_s2_idx, :] = kn[slc_idx, :]
        slc_kr[cur_s2_idx, :] = kr[slc_idx, :]
        slc_kn_scales[cur_s2_idx, :] = kn_scales[slc_idx, :]
    return slc_kn, slc_kr, slc_kn_scales


def _compute_s2_tile_attention(qi, slc_kn, slc_kr, slc_kn_scales, scalar,
                                input_dtype, dk, is_kn_quant, dv):
    """Compute single S2 tile softmax and attention for one head."""
    if is_kn_quant:
        kn_bs = slc_kn.reshape(-1, 128).to(torch.float)
        kn_scales_tmp = slc_kn_scales.reshape(-1, 1)
        kn_tmp = kn_bs * kn_scales_tmp
        kn_tmp = kn_tmp.reshape(-1, dk).to(input_dtype)
    else:
        kn_tmp = slc_kn
    kr_tmp = slc_kr
    vj = kn_tmp
    kj_view = torch.cat([kn_tmp, kr_tmp], dim=-1)
    sij = torch.matmul(qi.to(torch.float32), kj_view.transpose(1, 0).to(torch.float32)).to(torch.float32)
    sij_scale = sij * scalar
    tilda_mij = sij_scale.amax(dim=-1, keepdims=True)
    t_sub = sij_scale - tilda_mij
    tilda_pij = torch.exp(t_sub)
    tilda_pij_f16 = tilda_pij.to(input_dtype)
    q1 = torch.matmul(tilda_pij_f16.to(torch.float32), vj.to(torch.float32)).to(torch.float32)
    tilda_lij = tilda_pij.sum(dim=-1, keepdims=True)
    return q1, tilda_lij, tilda_mij


def _flash_update(oi_tmp, li_update, mi_update, q1, tilda_lij, tilda_mij,
                  bn_per_batch, s2_idx, n1, tmp_out, b_idx, s1_idx):
    """Online flash attention update step."""
    if s2_idx == 0:
        oi_tmp = q1
        if bn_per_batch == 1:
            oi_update = oi_tmp / tilda_lij
        else:
            oi_update = oi_tmp
        li_update = tilda_lij
        mi_update = tilda_mij
        tmp_out[b_idx, s1_idx, :] = tilda_lij.reshape(n1)
        return oi_tmp, oi_update, li_update, mi_update
    mi_new = torch.maximum(mi_update, tilda_mij)
    t1 = mi_update - mi_new
    t2 = torch.exp(t1)
    t3 = tilda_mij - mi_new
    t4 = torch.exp(t3)
    t5 = t4 * tilda_lij
    t6 = t2 * li_update
    li_new = t6 + t5
    q3 = oi_tmp * t2
    q2 = q1 * t4
    oi_tmp = q3 + q2
    if s2_idx == bn_per_batch - 1:
        oi_update = oi_tmp / li_new
    else:
        oi_update = oi_tmp
    return oi_tmp, oi_update, li_new, mi_new


def compute_attention(input_data, params, s2_tile):
    """
    计算注意力机制，支持不同批次的序列长度不同
    使用PyTorch实现
    """
    q, kn, kr, kn_scales, topk_indices, block_table, actual_seq = input_data
    block_size, scalar, topk, d_v, is_kn_quant = params
    b, s1, n1, dq = q.shape
    _, dk = kn.shape
    _, dv = kr.shape
    if topk_indices.ndim > 2:
        topk_indices = topk_indices.reshape(b * s1, topk)
    atten_out_shape = [b, s1, n1, d_v]
    input_dtype = q.dtype
    kn_dtype = kn.dtype
    attention_output = torch.zeros(atten_out_shape, dtype=input_dtype)
    tmp_out = torch.zeros([b, s1, n1], dtype=input_dtype)
    for b_idx in range(b):
        cur_k_seq = actual_seq[b_idx]
        for s1_idx in range(s1):
            cur_seq = min(max(cur_k_seq - s1 + 1 + s1_idx, 0), topk)
            bn_per_batch = math.ceil(cur_seq / s2_tile)
            qi = q[b_idx, s1_idx, :, :]
            oi_tmp, li_update, mi_update = None, None, None
            for s2_idx in range(bn_per_batch):
                s2_tile_cur = min(s2_tile, cur_seq - s2_idx * s2_tile)
                s2_start = s2_tile * s2_idx
                topk_indices_tmp = topk_indices[b_idx * s1 + s1_idx, s2_start:s2_start + s2_tile_cur]
                slc_kn, slc_kr, slc_kn_scales = _gather_kv_cache(
                    s2_tile_cur, topk_indices_tmp, kn, kr, kn_scales,
                    block_size, block_table, b_idx, s2_start, dk, dv, input_dtype, kn_dtype)
                q1, tilda_lij, tilda_mij = _compute_s2_tile_attention(
                    qi, slc_kn, slc_kr, slc_kn_scales, scalar, input_dtype, dk, is_kn_quant, dv)
                oi_tmp, oi_update, li_update, mi_update = _flash_update(
                    oi_tmp, li_update, mi_update, q1, tilda_lij, tilda_mij,
                    bn_per_batch, s2_idx, n1, tmp_out, b_idx, s1_idx)
            attention_output[b_idx, s1_idx, :, :] = oi_update.to(input_dtype)
    return attention_output, tmp_out


def compute_attention_no_flash(input_data, params, s2_tile):
    """
    计算注意力机制，支持不同批次的序列长度不同
    使用PyTorch实现
    no flash 版本
    """
    q, kn, kr, kn_scales, topk_indices, block_table, actual_seq = input_data
    block_size, scalar, topk, d_v, is_kn_quant = params
    b, s1, n1, dq = q.shape
    _, dk = kn.shape
    _, dv = kr.shape
    if topk_indices.ndim > 2:
        topk_indices = topk_indices.reshape(b * s1, topk)
    atten_out_shape = [b, s1, n1, d_v]
    input_dtype = q.dtype
    kn_dtype = kn.dtype
    attention_output = torch.zeros(atten_out_shape, dtype=input_dtype)
    tmp_out = torch.zeros([b, s1, n1], dtype=input_dtype)
    for b_idx in range(b):
        cur_k_seq = actual_seq[b_idx]
        for s1_idx in range(s1):
            cur_seq = min(max(cur_k_seq - s1 + 1 + s1_idx, 0), topk)
            bn_per_batch = math.ceil(cur_seq / s2_tile)
            qi = q[b_idx, s1_idx, :, :]
            for s2_idx in range(bn_per_batch):
                s2_tile_cur = min(s2_tile, cur_seq - s2_idx * s2_tile)
                s2_start = s2_tile * s2_idx
                topk_indices_tmp = topk_indices[b_idx * s1 + s1_idx, s2_start:s2_start + s2_tile_cur]
                slc_kn, slc_kr, slc_kn_scales = _gather_kv_cache(
                    s2_tile_cur, topk_indices_tmp, kn, kr, kn_scales,
                    block_size, block_table, b_idx, s2_start, dk, dv, input_dtype, kn_dtype)
                if is_kn_quant:
                    kn_bs = slc_kn.reshape(-1, 128).to(torch.float)
                    kn_scales_tmp = slc_kn_scales.reshape(-1, 1)
                    kn_tmp = kn_bs * kn_scales_tmp
                    kn_tmp = kn_tmp.reshape(-1, 512).to(input_dtype)
                else:
                    kn_tmp = slc_kn
                kj_view = torch.cat([kn_tmp, slc_kr], dim=-1)
                sij = torch.matmul(qi.to(torch.float32), kj_view.transpose(1, 0).to(torch.float32)).to(torch.float32)
                sij_scale = sij * scalar
                tilda_mij = sij_scale.amax(dim=-1, keepdims=True)
                t_sub = sij_scale - tilda_mij
                tilda_pij = torch.exp(t_sub)
                tilda_lij = tilda_pij.sum(dim=-1, keepdims=True)
                tmp_softmax = (tilda_pij / tilda_lij).to(input_dtype)
                atten_out_part = torch.matmul(tmp_softmax.to(torch.float32), kn_tmp.to(torch.float32)).to(torch.float32)
            attention_output[b_idx, s1_idx, :, :] = atten_out_part.to(input_dtype)
    return attention_output, tmp_out


def gen_block_table(act_seq, block_size, s1, need_indices=False):
    block_num = 0
    block_num_each = []
    b = act_seq.shape[0]
    max_kv = max(act_seq)
    for cur_s in act_seq:
        cur_block_num = math.ceil(cur_s / block_size)
        block_num_each.append(cur_block_num)
        block_num += cur_block_num
    block_table_shape = [b, math.ceil(max_kv / block_size)]
    block_idx_list = torch.arange(0, block_num, 1)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))].to(torch.int32)

    block_table = -torch.ones(block_table_shape, dtype=torch.int32)

    block_table_bidx = 0
    block_idx = 0
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

    return block_num, block_table, cache_index


def _gen_topk_indices(b, s_q, n_kv, topk, slc_actual_seq):
    """Generate topk indices for sparse flash attention."""
    topk_indices = torch.zeros(b, s_q, topk).to(torch.int32)
    for b_i in range(b):
        for s_q_i in range(s_q):
            if slc_actual_seq[b_i] < topk:
                topk_indices[b_i, s_q_i, :slc_actual_seq[b_i]] = torch.arange(0, slc_actual_seq[b_i])
            else:
                perm = torch.randperm(slc_actual_seq[b_i])
                topk_indices[b_i, s_q_i, :] = perm[:topk]
    return topk_indices.reshape(b * s_q, n_kv * topk)


def _prepare_kv_data(kn_bsnd_tmp, kr, block_num, block_size,
                     kv_lora_rank, qk_rope_dim, is_kn_quant):
    """Prepare and optionally quantize KN/KR cache data."""
    kn_bsnd_reshape = kn_bsnd_tmp.reshape(block_num * block_size, 4, 128).to(torch.float32)
    kn_scales = kn_bsnd_reshape.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    if is_kn_quant == 1:
        kn_quant = kn_bsnd_tmp.reshape(block_num * block_size, 4, 128) / kn_scales
        kn = torch.round(kn_quant).clamp(-128, 127).to(torch.int8)
    else:
        kn = kn_bsnd_tmp
    kn = kn.reshape(block_num * block_size, kv_lora_rank)
    kn_scales = kn_scales.reshape(block_num * block_size, 4)
    kr = kr.reshape(block_num * block_size, qk_rope_dim)
    return kn, kn_scales, kr


def gen_gather_select_attention_golden(dtype, bn1n2s1, is_kn_quant, actual_seq, is_soc_950):
    block_size = 128
    torch.manual_seed(42)
    b, n_q, n_kv, s_q = bn1n2s1
    kv_lora_rank = 512
    qk_rope_dim = 64
    topk = 2048
    np.random.seed(None)
    # q head dim
    d_q = kv_lora_rank + qk_rope_dim
    # k head dim
    d_k = kv_lora_rank + qk_rope_dim
    # v head dim
    d_v = kv_lora_rank
    scalar = d_q ** -0.5
    if isinstance(actual_seq, int):
        actual_seq = [actual_seq] * b
    elif isinstance(actual_seq, list):
        if len(actual_seq) == b:
            actual_seq = actual_seq
        else:
            raise RuntimeError("unsupported actual_seq list length")
    else:
        raise RuntimeError("unsupported actual_seq data type")
    # 1. 定义shape
    shape_q = [b, s_q, n_q, d_q]

    block_num_per_batch = []
    block_num_min = 0
    block_num = 0
    for actual_seq_tmp in actual_seq:
        block_num_per_batch.append(math.ceil(actual_seq_tmp / block_size))
        block_num_min += math.ceil(actual_seq_tmp / block_size)
    block_num = block_num_min

    shape_kn = [block_num, block_size, kv_lora_rank]
    shape_kr = [block_num, block_size, qk_rope_dim]

    max_kv_seq = max(actual_seq)
    block_num, block_table, _ = gen_block_table(torch.tensor(actual_seq), block_size, s_q, need_indices=False)
    slc_actual_seq = [min(actual_seq[i], topk) for i in range(b)]
    topk_indices = _gen_topk_indices(b, s_q, n_kv, topk, slc_actual_seq)

    q_bsnd = gen_uniform_data(shape_q, -1, 1, dtype)
    kn, kn_scales, kr = _prepare_kv_data(gen_uniform_data(shape_kn, -1, 1, dtype),
                                          gen_uniform_data(shape_kr, -1, 1, dtype),
                                          block_num, block_size, kv_lora_rank, qk_rope_dim,
                                          is_kn_quant)

    # 3. 计算attention
    params = [block_size, scalar, topk, kv_lora_rank, is_kn_quant]
    input_data = [q_bsnd, kn, kr, kn_scales, topk_indices, block_table, actual_seq]

    if is_soc_950:
        s2_tile = 512
        atten_out, _ = compute_attention(input_data, params, s2_tile)
    else:
        s2_tile = 2048
        atten_out, _ = compute_attention_no_flash(input_data, params, s2_tile)

    q_nope, q_rope, input_params, input_data_map = _build_output_params(
        q_bsnd, kn, kr, kn_scales, topk_indices, block_table, actual_seq,
        b, s_q, n_q, n_kv, max_kv_seq, kv_lora_rank, qk_rope_dim, block_num,
        block_size, topk, is_kn_quant, scalar)

    return input_params, input_data_map, atten_out


def _build_output_params(q_bsnd, kn, kr, kn_scales, topk_indices, block_table,
                          actual_seq, b, s_q, n_q, n_kv, max_kv_seq, kv_lora_rank,
                          qk_rope_dim, block_num, block_size, topk, is_kn_quant, scalar):
    """Build output parameter lists from computed data."""
    q_nope = q_bsnd[:, :, :, :kv_lora_rank].reshape(b * s_q * n_q, kv_lora_rank)
    q_rope = q_bsnd[:, :, :, kv_lora_rank:].reshape(b * s_q * n_q, qk_rope_dim)
    input_params = [b, s_q, n_q, n_kv, max_kv_seq, kv_lora_rank, qk_rope_dim, block_num, block_size, topk,
                    is_kn_quant, scalar]
    input_data_map = [q_nope, q_rope, kn, kr, kn_scales, topk_indices, block_table, actual_seq]
    return q_nope, q_rope, input_params, input_data_map


def _select_tile_config(is_p, is_soc_950):
    """Select appropriate tile config for sparse flash attention."""
    if is_soc_950:
        return SaTileShapeConfig(
            g_tile=128, s_kv_tile=512, gather_vec_tile_shape=[32, 576],
            c1_tile_shape=[128, 128, 256, 256, 128, 128], v1_tile_shape=[64, 128],
            c2_tile_shape=[128, 128, 128, 128, 256, 256], v2_tile_shape=[128, 128])
    if is_p:
        return SaTileShapeConfig(
            g_tile=128, s_kv_tile=2048, gather_vec_tile_shape=[32, 512],
            c1_tile_shape=[128, 128, 128, 128, 128, 128], v1_tile_shape=[8, 2048],
            c2_tile_shape=[128, 128, 128, 128, 128, 128], v2_tile_shape=[64, 128])
    return SaTileShapeConfig(
        g_tile=128, s_kv_tile=2048, gather_vec_tile_shape=[32, 512],
        c1_tile_shape=[128, 128, 128, 128, 128, 128], v1_tile_shape=[8, 2048],
        c2_tile_shape=[128, 128, 128, 128, 128, 128], v2_tile_shape=[64, 256])


def do_test_sparse_attention_func(bn1n2s1, actual_seq, input_params, input_data, atten_out, is_p, is_soc_950):
    b, n1, n2, s1 = bn1n2s1
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    tile_config = _select_tile_config(is_p, is_soc_950)

    b, s1, n_q, n_kv, max_kv_seq, kv_lora_rank, qk_rope_dim, block_num, block_size, topk, \
        is_kn_quant, softmax_scale = input_params
    q_nope, q_rope, kn, kr, kn_scales, topk_indices, block_table, kv_actual_seqs = input_data
    kv_act_seqs = torch.tensor(actual_seq, dtype=torch.int32)

    if is_soc_950:
        key_2d = torch.cat([kn, kr], dim=-1)
        pto_inputs = [q_nope.npu(), q_rope.npu(), key_2d.npu(), kn_scales.npu(),
                      topk_indices.npu(), block_table.npu(), kv_act_seqs.npu()]
    else:
        pto_inputs = [q_nope.npu(), q_rope.npu(), kn.npu(), kr.npu(), kn_scales.npu(),
                      topk_indices.npu(), block_table.npu(), kv_act_seqs.npu()]

    calc_attention_out = torch.zeros([b, s1, n_q, kv_lora_rank], dtype=torch.bfloat16)
    calc_attention_out_npu = calc_attention_out.npu()
    pto_outputs = [calc_attention_out_npu]

    max_blocknum_perbatch = math.ceil(max_kv_seq / block_size)

    if is_soc_950:
        for _ in range(1):
            sparse_flash_attention_d_950(*pto_inputs, *pto_outputs, n_q, n_kv, softmax_scale, topk, block_size,
                max_blocknum_perbatch, tile_config)
    else:
        if is_p:
            sparse_flash_attention_quant_p(*pto_inputs, *pto_outputs, n_q, n_kv, softmax_scale, topk, block_size,
                max_blocknum_perbatch, tile_config)
        else:
            sparse_flash_attention_quant_d(*pto_inputs, *pto_outputs, n_q, n_kv, softmax_scale, topk, block_size,
                max_blocknum_perbatch, tile_config)

    torch_npu.npu.synchronize()
    compare(calc_attention_out_npu.cpu(), atten_out, "atten_out", atol=0.0001, rtol=0.005, max_error_count=100)


def get_case_config(case_name: str):
    # case参数配置字典，key为case名称，value为对应的参数元组(bn1n2s1, is_kn_quant, actual_seq)
    test_case_config = {
        "sfa_bf16_b4_s2_seq64K_total_int8_d": (
            (4, 128, 1, 2), 1, [65536, 16381, 666, 15], 0
        ),
        "sfa_bf16_b4_s2_seq64K_per_int8_d": (
            (4, 128, 1, 2), 1, [65536] * 4, 0
        ),
        "sfa_bf16_b4_s2_seq64K_per_bf16_d": (
            (4, 128, 1, 2), 0, [65536] * 4, 0
        ),
        "sfa_bf16_b1_s256_seq64K_int8_p": (
            (1, 128, 1, 256), 1, [65536], 0
        ),
        "sfa_bf16_b4_s2_seq64K_per_bf16_d_950": (
            (4, 128, 1, 2), 0, [65536] * 4, 1
        ),
        "sfa_bf16_b64_s2_seq64K_per_bf16_d_950": (
            (64, 128, 1, 2), 0, [65536] * 64, 1
        ),
        "sfa_bf16_b64_s2_seq64K_uniform_per_bf16_d_950": (
            (64, 128, 1, 2), 0,
            [16384] * 8 + [32768] * 8 + [49152] * 8 + [65536] * 16 + [81920] * 8 + [98304] * 8 + [114688] * 8, 1
        ),
    }
    case_config = test_case_config.get(case_name)
    return case_config


def do_test_sfa_entry(case_name: str, is_p: bool, is_soc_950: bool):
    case_config = get_case_config(case_name)
    if not case_config:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    bn1n2s1, is_kn_quant, actual_seq, is_soc_950 = case_config

    input_params, input_data, atten_out = gen_gather_select_attention_golden(
        torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, is_soc_950
    )
    do_test_sparse_attention_func(
        bn1n2s1, actual_seq, input_params, input_data, atten_out, is_p, is_soc_950
    )
    return True


@pytest.mark.soc("950", "910")
def test_sfa_bf16_b4_s2_seq64k_total_int8_d():
    '''
    sfa decode测试函数
    '''
    do_test_sfa_entry("sfa_bf16_b4_s2_seq64K_total_int8_d", is_p=False, is_soc_950=False)


@pytest.mark.skip(reason="perf")
def test_sfa_bf16_b4_s2_seq64k_per_int8_d():
    '''
    sfa decode测试函数
    '''
    do_test_sfa_entry("sfa_bf16_b4_s2_seq64K_per_int8_d", is_p=False, is_soc_950=False)


@pytest.mark.soc("950")
@pytest.mark.skip(reason="perf")
def test_sfa_bf16_b4_s2_seq64k_per_bf16_d_950():
    '''
    sfa decode非量化950 mix切分测试用例
    '''
    do_test_sfa_entry("sfa_bf16_b4_s2_seq64K_per_bf16_d_950", is_p=False, is_soc_950=True)


@pytest.mark.soc("950")
@pytest.mark.skip(reason="perf")
def test_sfa_bf16_b64_s2_seq64k_per_bf16_d_950():
    '''
    sfa decode非量化950 mix切分测试用例
    '''
    do_test_sfa_entry("sfa_bf16_b64_s2_seq64K_per_bf16_d_950", is_p=False, is_soc_950=True)


@pytest.mark.soc("950")
@pytest.mark.skip(reason="bf16 perf")
def test_sfa_bf16_b4_s2_seq64k_per_bf16_d():
    '''
    sfa decode非量化测试函数
    '''
    do_test_sfa_entry("sfa_bf16_b4_s2_seq64K_per_bf16_d", is_p=False, is_soc_950=False)


@pytest.mark.skip(reason="large test case")
def test_sfa_bf16_b1_s256_seq64k_int8_p():
    '''
    sfa prefill测试函数
    '''
    do_test_sfa_entry("sfa_bf16_b1_s256_seq64K_int8_p", is_p=True, is_soc_950=False)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO
    )
    test_sfa_bf16_b4_s2_seq64k_total_int8_d()
    test_sfa_bf16_b4_s2_seq64k_per_int8_d()
    test_sfa_bf16_b1_s256_seq64k_int8_p()
    if pypto.platform.npuarch == 'DAV_3510':
        test_sfa_bf16_b4_s2_seq64k_per_bf16_d_950()
        test_sfa_bf16_b64_s2_seq64k_per_bf16_d_950()
