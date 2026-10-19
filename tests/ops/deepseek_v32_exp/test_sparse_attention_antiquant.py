#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# ... (license header unchanged)
# -----------------------------------------------------------------------------------------------------------
"""
"""
import math
import os
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

import numpy as np
import pytest
import pypto

from deepseek_v32_exp.sparse_attention_antiquant_impl import (
    SaTileShapeConfig,
    sparse_attention_antiquant_d,
    sparse_attention_antiquant_p,
    sparse_attention_antiquant_d_950,
)
from common_utils import compare


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    """
    PyTorch版本的均匀分布数据生成，与NumPy版本行为完全一致
    严格保持 [min_value, max_value) 左闭右开区间特性
    """
    # 特殊情况：全零张量
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=dtype)
    # 布尔类型处理：等概率生成True/False
    if dtype == torch.bool:
        # 生成[0,2)的整数，转换为bool即等概率True/False
        return torch.randint(0, 2, data_shape, dtype=dtype)
    # 浮点类型：[min_value, max_value)
    if torch.is_floating_point(torch.tensor(0, dtype=dtype)):
        # torch.rand生成[0,1)，缩放后得到[min_value, max_value)
        return min_value + (max_value - min_value) * torch.rand(data_shape, dtype=dtype)
    # 整数类型：[min_value, max_value)
    else:
        # torch.randint的high参数为开区间，直接对应[min_value, max_value)
        return torch.randint(low=min_value, high=max_value, size=data_shape, dtype=dtype)


def compute_attention_aq(input_data, params, s2_tile):
    """
    SA, 存8算16, Page nope cache, 计算流非FA
    使用PyTorch实现
    """
    q_nope, q_rope, nope_cache_2d, topk_indices, block_table, actual_seq = input_data
    nq, block_size, scalar, topk, kv_lora_rank, qk_rope_dim = params
    b_s1_nq, _ = q_nope.shape
    b = len(actual_seq)
    b_s1 = b_s1_nq // nq
    s1 = b_s1 // b

    if topk_indices.ndim > 2:
        topk_indices = topk_indices.reshape(b * s1, topk)

    atten_out_shape = [b, s1, nq, kv_lora_rank]
    input_dtype = q_nope.dtype
    q_nope = q_nope.reshape(b, s1, nq, -1)
    q_rope = q_rope.reshape(b, s1, nq, -1)

    # 初始化输出张量
    attention_output = torch.zeros(atten_out_shape, dtype=input_dtype)
    tmp_out = torch.zeros([b, s1, nq, kv_lora_rank], dtype=input_dtype)

    for b_idx in range(b):
        cur_k_seq = actual_seq[b_idx]
        for s1_idx in range(s1):
            cur_seq = min(max(cur_k_seq - s1 + 1 + s1_idx, 0), topk)
            bn_per_batch = math.ceil(cur_seq / s2_tile)

            qi = torch.zeros([nq, kv_lora_rank + qk_rope_dim], dtype=input_dtype)
            qi[:, :kv_lora_rank] = q_nope[b_idx, s1_idx, :, :]
            qi[:, kv_lora_rank:] = q_rope[b_idx, s1_idx, :, :]

            for s2_idx in range(bn_per_batch):
                s2_tile_cur = min(s2_tile, cur_seq - s2_idx * s2_tile)
                s2_start = s2_tile * s2_idx
                s2_end = s2_start + s2_tile_cur

                topk_indices_tmp = topk_indices[b_idx * s1 + s1_idx, s2_start:s2_end]
                slc_nope = torch.zeros([s2_tile_cur, kv_lora_rank + 2 * qk_rope_dim + 4 * 4], dtype=torch.int8)
                slc_kv_up = torch.zeros([s2_tile_cur, kv_lora_rank + qk_rope_dim], dtype=input_dtype)

                # 当前b&s1&s2 topk_index  --->  kvCache的offset
                offset = torch.zeros([s2_tile_cur], dtype=torch.int32)
                for cur_s2_idx in range(s2_tile_cur):
                    s2_idx_tmp = s2_start + cur_s2_idx
                    topk_index = topk_indices_tmp[s2_idx_tmp]
                    block_idx_in_batch = topk_index // block_size
                    slc_block_idx = block_table[b_idx, block_idx_in_batch]
                    tail = topk_index % block_size
                    offset[cur_s2_idx] = slc_block_idx * block_size + tail

                # 索引 kvCache
                for cur_s2_idx in range(s2_tile_cur):
                    slc_idx = offset[cur_s2_idx]
                    slc_nope[cur_s2_idx, :] = nope_cache_2d[slc_idx, :]

                # 存8算16
                slc_kv_int8 = slc_nope[:, :kv_lora_rank]
                slc_kv_scales_vint8 = slc_nope[:, kv_lora_rank + 2 * qk_rope_dim:]
                slc_kv_scales = slc_kv_scales_vint8.view(torch.float32).reshape(-1, 1)
                slc_kv_fp32 = slc_kv_int8.reshape(-1, 128).to(torch.float)
                slc_kv = slc_kv_fp32 * slc_kv_scales
                slc_kr_vin8 = slc_nope[:, kv_lora_rank:kv_lora_rank + 2 * qk_rope_dim]

                slc_kv_up[:, :kv_lora_rank] = slc_kv.to(input_dtype).reshape(-1, kv_lora_rank)
                slc_kv_up[:, kv_lora_rank:] = slc_kr_vin8.view(input_dtype)
                vj = slc_kv_up[:, :kv_lora_rank]

                # C1
                sij = torch.matmul(qi.to(torch.float32), slc_kv_up.transpose(1, 0).to(torch.float32)).to(torch.float32)

                # V1
                sij_scale = sij * scalar  # (nq, s2_tile)
                tilda_mij = sij_scale.amax(dim=-1, keepdims=True)  # (nq, 1)
                t_sub = sij_scale - tilda_mij  # (nq, s2_tile)
                tilda_pij = torch.exp(t_sub)  # (nq, s2_tile)
                tilda_lij_reduce = tilda_pij.sum(dim=-1, keepdims=True)  # (nq, 1)
                t_softmax = tilda_pij / tilda_lij_reduce
                tilda_pij_f16 = t_softmax.to(input_dtype)

                # C2
                q1 = torch.matmul(tilda_pij_f16.to(torch.float32), vj.to(torch.float32)).to(torch.float32)

            attention_output[b_idx, s1_idx, :, :] = q1.to(input_dtype)

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


def gen_gather_select_attention_golden_aq(dtype, bn1n2s1, is_kn_quant, actual_seq):
    # 默认 量化场景
    block_size = 128
    torch.manual_seed(42)
    b, n_q, n_kv, s_q = bn1n2s1  # 48, 128, 1, 1
    kv_lora_rank = 512
    qk_rope_dim = 64
    topk = 2048
    np.random.seed(None)

    # q head dim
    d_q = kv_lora_rank + qk_rope_dim

    # k head dim
    _d_k = kv_lora_rank + qk_rope_dim

    # v head dim
    _d_v = kv_lora_rank

    scalar = d_q**-0.5
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

    shape_kn = [block_num, block_size, n_kv, kv_lora_rank]
    shape_kr = [block_num, block_size, n_kv, qk_rope_dim]

    # 2、生成数据
    max_kv_seq = max(actual_seq)
    block_num, block_table, _ = gen_block_table(torch.tensor(actual_seq), block_size, s_q, need_indices=False)
    topk_indices = torch.zeros(b, s_q, topk).to(torch.int32)
    slc_actual_seq = []
    for i in range(b):
        slc_actual_seq.append(min(actual_seq[i], topk))

    for b_i in range(b):
        for s_q_i in range(s_q):
            if slc_actual_seq[b_i] < topk:
                topk_indices[b_i, s_q_i, :slc_actual_seq[b_i]] = torch.arange(0, slc_actual_seq[b_i])
            else:
                perm = torch.randperm(slc_actual_seq[b_i])
                topk_indices[b_i, s_q_i, :] = perm[:topk]

    topk_indices = topk_indices.reshape(b * s_q, n_kv * topk)

    q_bsnd = gen_uniform_data(shape_q, -1, 1, dtype)
    kn_bsnd = gen_uniform_data(shape_kn, -1, 1, dtype)

    kn_bsnd_reshape = kn_bsnd.reshape(block_num * block_size, 4, 128).to(torch.float32)
    kn_scales = kn_bsnd_reshape.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    kn_quant_fp32 = kn_bsnd.reshape(block_num * block_size, 4, 128) / kn_scales
    kn_quant = torch.round(kn_quant_fp32).clamp(-128, 127).to(torch.int8)

    kr = gen_uniform_data(shape_kr, -1, 1, dtype)

    # 2D
    kn_quant = kn_quant.reshape(block_num * block_size, kv_lora_rank)
    kn_scales = kn_scales.reshape(block_num * block_size, 4)
    kr = kr.reshape(block_num * block_size, qk_rope_dim)

    # nope_cache: kv尾轴512 int8， kr尾轴64 bf16/fp16，kv scale尾轴4 fp32，共656
    nope_cache_2d = torch.zeros([block_num * block_size, kv_lora_rank + qk_rope_dim * 2 + 4 * 4], dtype=torch.int8)

    # [:, 0:512]
    nope_cache_2d[:, :kv_lora_rank] = kn_quant

    # [:, 512:640]
    nope_cache_2d[:, kv_lora_rank:kv_lora_rank + qk_rope_dim * 2] = kr.view(torch.int8)

    # [:, 640:656]
    nope_cache_2d[:, kv_lora_rank + qk_rope_dim * 2:] = kn_scales.view(torch.int8)

    # q split to [nope + rope]
    q_nope = q_bsnd[:, :, :, :kv_lora_rank]
    q_rope = q_bsnd[:, :, :, kv_lora_rank:]
    q_nope = q_nope.reshape(b * s_q * n_q, kv_lora_rank)
    q_rope = q_rope.reshape(b * s_q * n_q, qk_rope_dim)

    # 3. 计算attention
    params = [n_q, block_size, scalar, topk, kv_lora_rank, qk_rope_dim]
    input_data = [q_nope, q_rope, nope_cache_2d, topk_indices, block_table, actual_seq]

    s2_tile = 2048
    atten_out, tmp_out = compute_attention_aq(input_data, params, s2_tile)

    # input params
    input_params = [b, s_q, n_q, n_kv, max_kv_seq, kv_lora_rank, qk_rope_dim, block_num, block_size, topk, scalar]
    input_data_map = [q_nope, q_rope, nope_cache_2d, topk_indices, block_table, actual_seq]

    return input_params, input_data_map, atten_out


def do_test_sparse_attention_func_aq(bn1n2s1, actual_seq, input_params, input_data, atten_out, is_p):
    b, n1, n2, s1 = bn1n2s1

    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    if is_p:
        tile_config = SaTileShapeConfig(
            g_tile=128,
            s_kv_tile=2048,
            c1_tile_shape=[128, 128, 128, 128, 128, 128],
            v1_tile_shape=[8, 2048],
            c2_tile_shape=[128, 128, 128, 128, 128, 128],  # C1的N轴与C2的K轴一致
            v2_tile_shape=[64, 128],
        )
    else:
        tile_config = SaTileShapeConfig(
            g_tile=128,
            s_kv_tile=2048,
            c1_tile_shape=[128, 128, 128, 128, 128, 128],
            v1_tile_shape=[8, 2048],
            c2_tile_shape=[128, 128, 128, 128, 128, 128],
            v2_tile_shape=[64, 128]
        )

    b, s1, n_q, n_kv, max_kv_seq, kv_lora_rank, qk_rope_dim, block_num, block_size, topk, softmax_scale = input_params
    q_nope, q_rope, nope_cache_2d, topk_indices, block_table, kv_actual_seqs = input_data
    kv_act_seqs = torch.tensor(actual_seq, dtype=torch.int32)

    calc_attention_out = torch.zeros([b * s1 * n_q, kv_lora_rank], dtype=torch.bfloat16)
    calc_attention_out_npu = calc_attention_out.npu()

    q_nope_npu = q_nope.npu()
    q_rope_npu = q_rope.npu()
    nope_cache_npu = nope_cache_2d.npu()
    topk_indices_npu = topk_indices.npu()
    block_table_npu = block_table.npu()
    kv_act_seqs_npu = kv_act_seqs.npu()

    pto_inputs = [q_nope_npu, q_rope_npu, nope_cache_npu, topk_indices_npu, block_table_npu, kv_act_seqs_npu]
    pto_outputs = [calc_attention_out_npu]

    max_blocknum_perbatch = math.ceil(max_kv_seq / block_size)

    if is_p:
        sparse_attention_antiquant_p(
            *pto_inputs, *pto_outputs, n_q, n_kv, softmax_scale, topk, block_size, max_blocknum_perbatch, tile_config)
    else:
        sparse_attention_antiquant_d(
            *pto_inputs,
            *pto_outputs,
            n_q,
            n_kv,
            softmax_scale,
            topk,
            block_size,
            max_blocknum_perbatch,
            tile_config,
        )
    calc_attention_out_npu = calc_attention_out_npu.reshape(b, s1, n_q, kv_lora_rank)
    torch_npu.npu.synchronize()
    compare(calc_attention_out_npu.cpu(), atten_out, "atten_out", atol=0.0001, rtol=0.005, max_error_count=100)


def compute_attention_aq_950(input_data, params, s2_tile):
    """
    SA, 存8算16, Page nope cache, 计算流FA (online softmax)
    使用PyTorch实现
    """
    q_nope, q_rope, nope_cache_2d, topk_indices, block_table, actual_seq = input_data
    nq, block_size, scalar, topk, kv_lora_rank, qk_rope_dim = params
    b_s1_nq, _ = q_nope.shape
    b = len(actual_seq)
    b_s1 = b_s1_nq // nq
    s1 = b_s1 // b

    if topk_indices.ndim > 2:
        topk_indices = topk_indices.reshape(b * s1, topk)

    atten_out_shape = [b, s1, nq, kv_lora_rank]
    input_dtype = q_nope.dtype
    q_nope = q_nope.reshape(b, s1, nq, -1)
    q_rope = q_rope.reshape(b, s1, nq, -1)

    attention_output = torch.zeros(atten_out_shape, dtype=input_dtype)
    tmp_out = torch.zeros([b, s1, nq, kv_lora_rank], dtype=input_dtype)

    for b_idx in range(b):
        cur_k_seq = actual_seq[b_idx]
        for s1_idx in range(s1):
            cur_seq = min(max(cur_k_seq - s1 + 1 + s1_idx, 0), topk)
            bn_per_batch = math.ceil(cur_seq / s2_tile)

            qi = torch.zeros([nq, kv_lora_rank + qk_rope_dim], dtype=input_dtype)
            qi[:, :kv_lora_rank] = q_nope[b_idx, s1_idx, :, :]
            qi[:, kv_lora_rank:] = q_rope[b_idx, s1_idx, :, :]

            mi = None
            li = None
            oi = None

            for s2_idx in range(bn_per_batch):
                s2_tile_cur = min(s2_tile, cur_seq - s2_idx * s2_tile)
                s2_start = s2_tile * s2_idx
                s2_end = s2_start + s2_tile_cur

                topk_indices_tmp = topk_indices[b_idx * s1 + s1_idx, s2_start:s2_end]
                slc_nope = torch.zeros([s2_tile_cur, kv_lora_rank + 2 * qk_rope_dim + 4 * 4], dtype=torch.int8)
                slc_kv_up = torch.zeros([s2_tile_cur, kv_lora_rank + qk_rope_dim], dtype=input_dtype)

                offset = torch.zeros([s2_tile_cur], dtype=torch.int32)
                for cur_s2_idx in range(s2_tile_cur):
                    topk_index = topk_indices_tmp[cur_s2_idx]
                    block_idx_in_batch = topk_index // block_size
                    slc_block_idx = block_table[b_idx, block_idx_in_batch]
                    tail = topk_index % block_size
                    offset[cur_s2_idx] = slc_block_idx * block_size + tail

                for cur_s2_idx in range(s2_tile_cur):
                    slc_idx = offset[cur_s2_idx]
                    slc_nope[cur_s2_idx, :] = nope_cache_2d[slc_idx, :]

                slc_kv_int8 = slc_nope[:, :kv_lora_rank]
                slc_kv_scales_vint8 = slc_nope[:, kv_lora_rank + 2 * qk_rope_dim:]
                slc_kv_scales = slc_kv_scales_vint8.view(torch.float32).reshape(-1, 1)
                slc_kv_fp32 = slc_kv_int8.reshape(-1, 128).to(torch.float)
                slc_kv = slc_kv_fp32 * slc_kv_scales
                slc_kr_vin8 = slc_nope[:, kv_lora_rank:kv_lora_rank + 2 * qk_rope_dim]

                slc_kv_up[:, :kv_lora_rank] = slc_kv.to(input_dtype).reshape(-1, kv_lora_rank)
                slc_kv_up[:, kv_lora_rank:] = slc_kr_vin8.view(input_dtype)
                vj = slc_kv_up[:, :kv_lora_rank]

                # C1
                sij = torch.matmul(qi.to(torch.float32), slc_kv_up.transpose(1, 0).to(torch.float32)).to(torch.float32)

                # V1: online softmax
                sij_scale = sij * scalar
                tilda_mij = sij_scale.amax(dim=-1, keepdims=True)
                t_sub = sij_scale - tilda_mij
                tilda_pij = torch.exp(t_sub)
                tilda_lij = tilda_pij.sum(dim=-1, keepdims=True)
                tilda_pij_f16 = tilda_pij.to(input_dtype)

                # C2
                q1 = torch.matmul(tilda_pij_f16.to(torch.float32), vj.to(torch.float32)).to(torch.float32)

                # online softmax update
                if s2_idx == 0:
                    oi = q1
                    mi = tilda_mij
                    li = tilda_lij
                else:
                    mi_new = torch.maximum(mi, tilda_mij)
                    scale_old = torch.exp(mi - mi_new)
                    scale_new = torch.exp(tilda_mij - mi_new)
                    li = li * scale_old + tilda_lij * scale_new
                    oi = oi * scale_old + q1 * scale_new
                    mi = mi_new

            attention_output[b_idx, s1_idx, :, :] = (oi / li).to(input_dtype)

    return attention_output, tmp_out


def gen_gather_select_attention_golden_aq_950(dtype, bn1n2s1, is_kn_quant, actual_seq):
    # 默认 量化场景
    block_size = 128
    torch.manual_seed(42)
    b, n_q, n_kv, s_q = bn1n2s1  # 48, 128, 1, 1
    kv_lora_rank = 512
    qk_rope_dim = 64
    topk = 2048
    np.random.seed(None)

    # q head dim
    d_q = kv_lora_rank + qk_rope_dim

    # k head dim
    _d_k = kv_lora_rank + qk_rope_dim

    # v head dim
    _d_v = kv_lora_rank

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

    shape_kn = [block_num, block_size, n_kv, kv_lora_rank]
    shape_kr = [block_num, block_size, n_kv, qk_rope_dim]

    # 2、生成数据
    max_kv_seq = max(actual_seq)
    block_num, block_table, _ = gen_block_table(torch.tensor(actual_seq), block_size, s_q, need_indices=False)
    topk_indices = torch.zeros(b, s_q, topk).to(torch.int32)
    slc_actual_seq = []
    for i in range(b):
        slc_actual_seq.append(min(actual_seq[i], topk))

    for b_i in range(b):
        for s_q_i in range(s_q):
            if slc_actual_seq[b_i] < topk:
                topk_indices[b_i, s_q_i, :slc_actual_seq[b_i]] = torch.arange(0, slc_actual_seq[b_i])
            else:
                perm = torch.randperm(slc_actual_seq[b_i])
                topk_indices[b_i, s_q_i, :] = perm[:topk]

    topk_indices = topk_indices.reshape(b * s_q, n_kv * topk)

    q_bsnd = gen_uniform_data(shape_q, -1, 1, dtype)
    kn_bsnd = gen_uniform_data(shape_kn, -1, 1, dtype)

    kn_bsnd_reshape = kn_bsnd.reshape(block_num * block_size, 4, 128).to(torch.float32)
    kn_scales = kn_bsnd_reshape.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    kn_quant_fp32 = kn_bsnd.reshape(block_num * block_size, 4, 128) / kn_scales
    kn_quant = torch.round(kn_quant_fp32).clamp(-128, 127).to(torch.int8)

    kr = gen_uniform_data(shape_kr, -1, 1, dtype)

    # 2D
    kn_quant = kn_quant.reshape(block_num * block_size, kv_lora_rank)
    kn_scales = kn_scales.reshape(block_num * block_size, 4)
    kr = kr.reshape(block_num * block_size, qk_rope_dim)

    # nope_cache: kv尾轴512 int8， kr尾轴64 bf16/fp16，kv scale尾轴4 fp32，共656
    nope_cache_2d = torch.zeros([block_num * block_size, kv_lora_rank + qk_rope_dim * 2 + 4 * 4], dtype=torch.int8)

    # [:, 0:512]
    nope_cache_2d[:, :kv_lora_rank] = kn_quant

    # [:, 512:640]
    nope_cache_2d[:, kv_lora_rank:kv_lora_rank + qk_rope_dim * 2] = kr.view(torch.int8)

    # [:, 640:656]
    nope_cache_2d[:, kv_lora_rank + qk_rope_dim * 2:] = kn_scales.view(torch.int8)

    # q split to [nope + rope]
    q_nope = q_bsnd[:, :, :, :kv_lora_rank]
    q_rope = q_bsnd[:, :, :, kv_lora_rank:]
    q_nope = q_nope.reshape(b * s_q * n_q, kv_lora_rank)
    q_rope = q_rope.reshape(b * s_q * n_q, qk_rope_dim)

    # 3. 计算attention
    params = [n_q, block_size, scalar, topk, kv_lora_rank, qk_rope_dim]
    input_data = [q_nope, q_rope, nope_cache_2d, topk_indices, block_table, actual_seq]

    s2_tile = 512
    atten_out, tmp_out = compute_attention_aq_950(input_data, params, s2_tile)

    # input params
    input_params = [b, s_q, n_q, n_kv, max_kv_seq, kv_lora_rank, qk_rope_dim, block_num, block_size, topk, scalar]
    input_data_map = [q_nope, q_rope, nope_cache_2d, topk_indices, block_table, actual_seq]
    return input_params, input_data_map, atten_out


def do_test_sparse_attention_func_aq_950(bn1n2s1, actual_seq, input_params, input_data, atten_out):
    b, n1, n2, s1 = bn1n2s1

    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tile_config = SaTileShapeConfig(
        g_tile=128,
        s_kv_tile=512,
        c1_tile_shape=[128, 128, 256, 256, 128, 128],
        v1_tile_shape=[64, 128],
        c2_tile_shape=[128, 128, 128, 128, 256, 256],
        v2_tile_shape=[128, 128]
    )

    b, s1, n_q, n_kv, max_kv_seq, kv_lora_rank, qk_rope_dim, block_num, block_size, topk, \
        softmax_scale = input_params
    q_nope, q_rope, nope_cache_2d, topk_indices, block_table, kv_actual_seqs = input_data
    kv_act_seqs = torch.tensor(actual_seq, dtype=torch.int32)

    total_rows = b * s1 * n_q
    calc_attention_out = torch.zeros([total_rows, kv_lora_rank], dtype=torch.bfloat16)
    calc_attention_out_npu = calc_attention_out.npu()

    q_nope_npu = q_nope.npu()
    q_rope_npu = q_rope.npu()
    nope_cache_npu = nope_cache_2d.npu()
    topk_indices_npu = topk_indices.npu()
    block_table_npu = block_table.npu()
    kv_act_seqs_npu = kv_act_seqs.npu()

    pto_inputs = [q_nope_npu, q_rope_npu, nope_cache_npu, topk_indices_npu, block_table_npu, kv_act_seqs_npu]
    pto_outputs = [calc_attention_out_npu]

    max_blocknum_perbatch = math.ceil(max_kv_seq / block_size)

    logging.info(f"q_nope_npu {q_nope_npu.shape} {q_nope_npu.dtype}")
    logging.info(f"q_rope_npu {q_rope_npu.shape} {q_rope_npu.dtype}")
    logging.info(f"nope_cache_npu {nope_cache_npu.shape} {nope_cache_npu.dtype}")
    logging.info(f"topk_indices_npu {topk_indices_npu.shape} {topk_indices_npu.dtype}")
    logging.info(f"block_table_npu {block_table_npu.shape} {block_table_npu.dtype}")
    logging.info(f"kv_act_seqs_npu {kv_act_seqs_npu.shape} {kv_act_seqs_npu.dtype}")
    logging.info(f"calc_attention_out_npu {calc_attention_out_npu.shape} {calc_attention_out_npu.dtype}")
    logging.info(f"{n_q, n_kv, softmax_scale, topk, block_size, max_blocknum_perbatch, tile_config}")
    for _ in range(1):
        a = torch.randn((int(192 * 1024 * 1024 * 2.5))).to(torch.float32).npu()
        for _ in range(100):
            _a_max = torch.max(a)
        sparse_attention_antiquant_d_950(*pto_inputs, *pto_outputs, n_q, n_kv, softmax_scale, topk, block_size, \
            max_blocknum_perbatch, tile_config)
    torch_npu.npu.synchronize()

    calc_attention_out_npu = calc_attention_out_npu.reshape(b, s1, n_q, kv_lora_rank)
    out_cpu = calc_attention_out_npu.cpu()
    compare(out_cpu, atten_out, "atten_out", atol=0.0001, rtol=0.005, max_error_count=100)


def get_case_config(case_name: str):
    # case参数配置字典，key为case名称，value为对应的参数元组(bn1n2s1, is_kn_quant, actual_seq)
    test_case_config = {
        "sfa_bf16_b4_s2_seq64K_total_int8_d": ((4, 128, 1, 2), 1, [65536, 16381, 666, 15]),
        "sfa_bf16_b4_s2_seq64K_per_int8_d": ((4, 128, 1, 2), 1, [65536] * 4),
        "sfa_bf16_b1_s256_seq64K_int8_p": ((1, 128, 1, 256), 1, [65536]),
        "sfa_bf16_b64_s2_seq64K_uniform_per_int8_d_950": ((64, 128, 1, 2), 1, [16384] * 8 + [32768] * 8 + \
            [49152] * 8 + [65536] * 16 + [81920] * 8 + [98304] * 8 + [114688] * 8),
        "sfa_bf16_b4_s2_seq64K_per_int8_d_950": ((4, 128, 1, 2), 1, [65536] * 4),
        "sfa_bf16_b64_s2_seq64K_per_int8_d_950": ((64, 128, 1, 2), 1, [65536] * 64),
    }
    case_config = test_case_config.get(case_name)
    return case_config


def do_test_sfa_entry(case_name: str, is_p: bool):
    case_config = get_case_config(case_name)
    if not case_config:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    bn1n2s1, is_kn_quant, actual_seq = case_config

    input_params, input_data, atten_out = gen_gather_select_attention_golden_aq(
        torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq
    )
    do_test_sparse_attention_func_aq(bn1n2s1, actual_seq, input_params, input_data, atten_out, is_p)
    return True


def do_test_sfa_entry_950(case_name: str):
    case_config = get_case_config(case_name)
    if not case_config:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    bn1n2s1, is_kn_quant, actual_seq = case_config

    logging.info(f"sfa_quant golden begin...")
    input_params, input_data, atten_out = gen_gather_select_attention_golden_aq_950(
        torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq
    )
    logging.info(f"sfa_quant golden end")

    do_test_sparse_attention_func_aq_950(
        bn1n2s1, actual_seq, input_params, input_data, atten_out)
    return True


@pytest.mark.soc("950", "910")
def test_sfa_bf16_b4_s2_seq64k_total_int8_d():
    '''
    sfa decode测试函数
    '''
    do_test_sfa_entry("sfa_bf16_b4_s2_seq64K_total_int8_d", is_p=False)


@pytest.mark.soc("950", "910")
@pytest.mark.skip(reason="perf")
def test_sfa_bf16_b4_s2_seq64k_per_int8_d():
    '''
    sfa decode测试函数
    '''
    do_test_sfa_entry("sfa_bf16_b4_s2_seq64K_per_int8_d", is_p=False)


@pytest.mark.soc("950", "910")
@pytest.mark.skip(reason="large test case")
def test_sfa_bf16_b1_s256_seq64k_int8_p():
    '''
    sfa prefill测试函数
    '''
    do_test_sfa_entry("sfa_bf16_b1_s256_seq64K_int8_p", is_p=True)


@pytest.mark.soc("950")
@pytest.mark.skip(reason="large test case")
def test_sfa_bf16_b64_s2_seq64k_uniform_per_int8_d_950():
    '''
    sfa decode测试函数
    '''
    do_test_sfa_entry_950("sfa_bf16_b64_s2_seq64K_uniform_per_int8_d_950")


@pytest.mark.soc("950")
@pytest.mark.skip(reason="large test case")
def test_sfa_bf16_b4_s2_seq64k_per_int8_d_950():
    '''
    sfa decode测试函数
    '''
    do_test_sfa_entry_950("sfa_bf16_b4_s2_seq64K_per_int8_d_950")


@pytest.mark.soc("950")
def test_sfa_bf16_b64_s2_seq64k_per_int8_d_950():
    '''
    sfa decode测试函数
    '''
    do_test_sfa_entry_950("sfa_bf16_b64_s2_seq64K_per_int8_d_950")


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    test_sfa_bf16_b4_s2_seq64k_total_int8_d()
    test_sfa_bf16_b4_s2_seq64k_per_int8_d()
    test_sfa_bf16_b1_s256_seq64k_int8_p()
    logging.info(f"sfa_quant_int8 910 case PASS")
    if pypto.platform.npuarch == 'DAV_3510':
        test_sfa_bf16_b64_s2_seq64k_uniform_per_int8_d_950()
        test_sfa_bf16_b4_s2_seq64k_per_int8_d_950()
        test_sfa_bf16_b64_s2_seq64k_per_int8_d_950()
        logging.info(f"sfa_quant_int8 950 case PASS")
