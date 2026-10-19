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
Sparse Attention Antiquant KV Split 算子测试
"""

from dataclasses import dataclass
import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

import math
import logging
from typing import Any
import torch
import torch_npu  # noqa: F401
import numpy as np
import pytest

from common_utils import compare
from experimental.ops_transformer.sparse_attention_antiquant_kv_split.sparse_attention_antiquant_kv_split_impl import (
    sparse_attention_antiquant_kv_split_d,
    sparse_attention_antiquant_kv_split_p,
    SaTileShapeConfig,
)


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=dtype)
    if dtype == torch.bool:
        return torch.randint(0, 2, data_shape, dtype=dtype)
    if torch.is_floating_point(torch.tensor(0, dtype=dtype)):
        return min_value + (max_value - min_value) * torch.rand(data_shape, dtype=dtype)
    else:
        return torch.randint(low=min_value, high=max_value, size=data_shape, dtype=dtype)


@dataclass
class _GatherKvSlicesInputs:
    topk_indices: Any
    topk_indices_tmp: Any
    s2_start: Any
    s2_tile_cur: Any
    kn_quant: Any
    kr: Any
    kn_scales: Any
    block_table: Any
    block_size: Any
    b_idx: Any
    kv_lora_rank: Any
    qk_rope_dim: Any
    input_dtype: Any


def _gather_kv_slices(inputs: _GatherKvSlicesInputs):


    """Gather KV slices for a tile: compute offsets from inputs.block_table, then fetch kn/inputs.kr/scales."""
    offset = torch.zeros([inputs.s2_tile_cur], dtype=torch.int32)
    for cur_s2_idx in range(inputs.s2_tile_cur):
        s2_idx_tmp = inputs.s2_start + cur_s2_idx
        topk_index = inputs.topk_indices_tmp[s2_idx_tmp]
        block_idx_in_batch = topk_index // inputs.block_size
        slc_block_idx = inputs.block_table[inputs.b_idx, block_idx_in_batch]
        tail = topk_index % inputs.block_size
        offset[cur_s2_idx] = slc_block_idx * inputs.block_size + tail

    kn_quant_slice = torch.zeros([inputs.s2_tile_cur, inputs.kv_lora_rank], dtype=torch.float8_e4m3fn)
    kr_slice = torch.zeros([inputs.s2_tile_cur, inputs.qk_rope_dim], dtype=torch.bfloat16)
    kn_scales_slice = torch.zeros([inputs.s2_tile_cur, 4], dtype=torch.float32)
    for cur_s2_idx in range(inputs.s2_tile_cur):
        slc_idx = offset[cur_s2_idx]
        kn_quant_slice[cur_s2_idx, :] = inputs.kn_quant[slc_idx, :]
        kr_slice[cur_s2_idx, :] = inputs.kr[slc_idx, :]
        kn_scales_slice[cur_s2_idx, :] = inputs.kn_scales[slc_idx, :]
    return kn_quant_slice, kr_slice, kn_scales_slice


def _dequant_and_attention(qi, kn_quant_slice, kr_slice, kn_scales_slice,
                            s2_tile_cur, kv_lora_rank, qk_rope_dim, scalar, input_dtype):
    """Dequant KV and compute attention scores + softmax for a tile."""
    kn_quant_fp32 = kn_quant_slice.to(torch.float32)
    kn_quant_fp32_reshape = kn_quant_fp32.reshape(s2_tile_cur * 4, 128)
    kn_scales_slice_reshape = kn_scales_slice.reshape(s2_tile_cur * 4, 1)
    slc_kv = kn_quant_fp32_reshape * kn_scales_slice_reshape

    slc_kv_up = torch.zeros([s2_tile_cur, kv_lora_rank + qk_rope_dim], dtype=input_dtype)
    slc_kv_up[:, :kv_lora_rank] = slc_kv.reshape(s2_tile_cur, kv_lora_rank).to(input_dtype)
    slc_kv_up[:, kv_lora_rank:] = kr_slice
    slc_kv_reshape = slc_kv.reshape(s2_tile_cur, kv_lora_rank).to(input_dtype)

    sij = torch.matmul(qi.to(torch.float32), slc_kv_up.transpose(1, 0).to(torch.float32)).to(torch.float32)
    sij_scale = sij * scalar
    tilda_mij = sij_scale.amax(dim=-1, keepdims=True)
    t_sub = sij_scale - tilda_mij
    tilda_pij = torch.exp(t_sub)
    tilda_lij_reduce = tilda_pij.sum(dim=-1, keepdims=True)
    t_softmax = tilda_pij / tilda_lij_reduce
    tilda_pij_f16 = t_softmax.to(input_dtype)
    q1 = torch.matmul(tilda_pij_f16.to(torch.float32), slc_kv_reshape.to(torch.float32)).to(torch.float32)
    return q1


def compute_attention_aq(input_data, params, s2_tile):
    """SA, 存8算16, Page nope cache, 计算流非FA, 使用PyTorch实现"""
    q_nope, q_rope, kn_quant, kr, kn_scales, topk_indices, block_table, actual_seq = input_data
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

            for s2_idx in range(bn_per_batch):
                s2_tile_cur = min(s2_tile, cur_seq - s2_idx * s2_tile)
                s2_start = s2_tile * s2_idx

                topk_indices_tmp = topk_indices[b_idx * s1 + s1_idx, s2_start:s2_start + s2_tile_cur]
                kn_quant_slice, kr_slice, kn_scales_slice = _gather_kv_slices(
                    _GatherKvSlicesInputs(
                        topk_indices, topk_indices_tmp, s2_start, s2_tile_cur,
                        kn_quant, kr, kn_scales, block_table, block_size, b_idx,
                        kv_lora_rank, qk_rope_dim, input_dtype))

                q1 = _dequant_and_attention(qi, kn_quant_slice, kr_slice, kn_scales_slice,
                                             s2_tile_cur, kv_lora_rank, qk_rope_dim,
                                             scalar, input_dtype)

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


def _generate_topk_indices(b, s_q, actual_seq, topk, slc_actual_seq):
    """Generate topk_indices for golden computation."""
    topk_indices = torch.zeros(b, s_q, topk).to(torch.int32)
    for b_i in range(b):
        for s_q_i in range(s_q):
            if slc_actual_seq[b_i] < topk:
                topk_indices[b_i, s_q_i, :slc_actual_seq[b_i]] = torch.arange(0, slc_actual_seq[b_i])
            else:
                perm = torch.randperm(slc_actual_seq[b_i])
                topk_indices[b_i, s_q_i, :] = perm[:topk]
    return topk_indices


def _generate_quantized_kv(shape_kn, dtype, block_size, kv_lora_rank, qk_rope_dim, block_num, shape_kr):
    """Generate quantized key/value tensors."""
    kn_bsnd = gen_uniform_data(shape_kn, -1, 1, dtype)
    kn_bsnd_reshape = kn_bsnd.reshape(block_num * block_size, 4, 128).to(torch.float32)
    kn_scales = kn_bsnd_reshape.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    kn_quant_fp32 = kn_bsnd.reshape(block_num * block_size, 4, 128) / kn_scales
    kn_quant = torch.round(kn_quant_fp32).clamp(-128, 127).to(torch.float8_e4m3fn)
    kr = gen_uniform_data(shape_kr, -1, 1, dtype)
    kn_quant = kn_quant.reshape(block_num * block_size, kv_lora_rank)
    kn_scales = kn_scales.reshape(block_num * block_size, 4)
    kr = kr.reshape(block_num * block_size, qk_rope_dim).to(torch.bfloat16)
    return kn_quant, kn_scales, kr


def gen_gather_select_attention_golden_aq(dtype, bn1n2s1, is_kn_quant, actual_seq):
    block_size = 128
    torch.manual_seed(42)
    b, n_q, n_kv, s_q = bn1n2s1
    kv_lora_rank = 512
    qk_rope_dim = 64
    topk = 2048
    np.random.seed(None)

    d_q = kv_lora_rank + qk_rope_dim
    d_k = kv_lora_rank + qk_rope_dim
    d_v = kv_lora_rank

    scalar = d_q ** -0.5
    if isinstance(actual_seq, int):
        actual_seq = [actual_seq] * b
    elif not isinstance(actual_seq, list):
        raise RuntimeError("unsupported actual_seq data type")

    shape_q = [b, s_q, n_q, d_q]

    block_num = 0
    block_num_per_batch = []
    for actual_seq_tmp in actual_seq:
        block_num_per_batch.append(math.ceil(actual_seq_tmp / block_size))
        block_num += math.ceil(actual_seq_tmp / block_size)

    shape_kn = [block_num, block_size, n_kv, kv_lora_rank]
    shape_kr = [block_num, block_size, n_kv, qk_rope_dim]

    max_kv_seq = max(actual_seq)
    block_num, block_table, _ = gen_block_table(torch.tensor(actual_seq), block_size, s_q, need_indices=False)
    slc_actual_seq = [min(actual_seq[i], topk) for i in range(b)]

    topk_indices = _generate_topk_indices(b, s_q, actual_seq, topk, slc_actual_seq)
    topk_indices = topk_indices.reshape(b * s_q, n_kv * topk)

    q_bsnd = gen_uniform_data(shape_q, -1, 1, dtype)

    kn_quant, kn_scales, kr = _generate_quantized_kv(
        shape_kn, dtype, block_size, kv_lora_rank, qk_rope_dim, block_num, shape_kr)

    q_nope = q_bsnd[:, :, :, :kv_lora_rank]
    q_rope = q_bsnd[:, :, :, kv_lora_rank:]
    q_nope = q_nope.reshape(b * s_q * n_q, kv_lora_rank)
    q_rope = q_rope.reshape(b * s_q * n_q, qk_rope_dim)

    params = [n_q, block_size, scalar, topk, kv_lora_rank, qk_rope_dim]
    input_data = [q_nope, q_rope, kn_quant, kr, kn_scales, topk_indices, block_table, actual_seq]

    s2_tile = 2048
    atten_out, tmp_out = compute_attention_aq(input_data, params, s2_tile)

    input_params = [b, s_q, n_q, n_kv, max_kv_seq, kv_lora_rank, qk_rope_dim, block_num, block_size, topk, scalar]
    input_data_map = [q_nope, q_rope, kn_quant, kr, kn_scales, topk_indices, block_table, actual_seq]

    return input_params, input_data_map, atten_out


def do_test_sparse_attention_func_aq(bn1n2s1, actual_seq, input_params, input_data, atten_out, is_p):
    b, n1, n2, s1 = bn1n2s1

    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tile_config = SaTileShapeConfig(
        g_tile=128,
        s_kv_tile=2048,
        c1_tile_shape=[128, 128, 128, 128, 128, 128],
        v1_tile_shape=[8, 2048],
        c2_tile_shape=[128, 128, 128, 128, 128, 128],
        v2_tile_shape=[64, 128]
    )

    b, s1, n_q, n_kv, max_kv_seq, kv_lora_rank, qk_rope_dim, block_num, block_size, topk, \
        softmax_scale = input_params
    q_nope, q_rope, kn_quant, kr, kn_scales, topk_indices, block_table, kv_actual_seqs = input_data
    kv_act_seqs = torch.tensor(actual_seq, dtype=torch.int32)

    calc_attention_out = torch.zeros([b * s1 * n_q, kv_lora_rank], dtype=torch.bfloat16)
    calc_attention_out_npu = calc_attention_out.npu()

    q_nope_npu = q_nope.npu()
    q_rope_npu = q_rope.npu()
    kn_quant_npu = kn_quant.npu()
    kr_npu = kr.npu()
    kn_scales_npu = kn_scales.npu()
    topk_indices_npu = topk_indices.npu()
    block_table_npu = block_table.npu()
    kv_act_seqs_npu = kv_act_seqs.npu()

    pto_inputs = [q_nope_npu, q_rope_npu, kn_quant_npu, kr_npu, kn_scales_npu,
                   topk_indices_npu, block_table_npu, kv_act_seqs_npu]
    pto_outputs = [calc_attention_out_npu]

    max_blocknum_perbatch = math.ceil(max_kv_seq / block_size)

    if is_p:
        sparse_attention_antiquant_kv_split_p(*pto_inputs, *pto_outputs, n_q, n_kv, softmax_scale, topk,
                                              block_size, max_blocknum_perbatch, tile_config)
    else:
        sparse_attention_antiquant_kv_split_d(*pto_inputs, *pto_outputs, n_q, n_kv, softmax_scale, topk,
                                              block_size, max_blocknum_perbatch, tile_config)
    calc_attention_out_npu = calc_attention_out_npu.reshape(b, s1, n_q, kv_lora_rank)
    torch_npu.npu.synchronize()
    compare(calc_attention_out_npu.cpu(), atten_out, "atten_out", atol=0.0001, rtol=0.005, max_error_count=100)


def get_case_config(case_name: str):
    test_case_config = {
        "sfa_bf16_b4_s2_seq64K_total_fp8_d": (
            (4, 128, 1, 2), 1, [65536, 16381, 666, 15]
        ),
        "sfa_bf16_b4_s2_seq64K_per_fp8_d": (
            (4, 128, 1, 2), 1, [65536] * 4
        ),
        "sfa_bf16_b1_s256_seq64K_fp8_p": (
            (1, 128, 1, 256), 1, [65536]
        ),
    }
    return test_case_config.get(case_name)


def do_test_sfa_entry(case_name: str, is_p: bool):
    case_config = get_case_config(case_name)
    if not case_config:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    bn1n2s1, is_kn_quant, actual_seq = case_config

    input_params, input_data, atten_out = gen_gather_select_attention_golden_aq(
        torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq
    )
    do_test_sparse_attention_func_aq(
        bn1n2s1, actual_seq, input_params, input_data, atten_out, is_p
    )
    return True


@pytest.mark.soc("950")
def test_sfa_bf16_b4_s2_seq64k_total_fp8_d():
    do_test_sfa_entry("sfa_bf16_b4_s2_seq64K_total_fp8_d", is_p=False)


@pytest.mark.soc("950")
@pytest.mark.skip(reason="perf")
def test_sfa_bf16_b4_s2_seq64k_per_fp8_d():
    do_test_sfa_entry("sfa_bf16_b4_s2_seq64K_per_fp8_d", is_p=False)


@pytest.mark.soc("950")
@pytest.mark.skip(reason="large test case")
def test_sfa_bf16_b1_s256_seq64k_fp8_p():
    do_test_sfa_entry("sfa_bf16_b1_s256_seq64K_fp8_p", is_p=True)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO
    )
    test_sfa_bf16_b4_s2_seq64k_total_fp8_d()
    test_sfa_bf16_b4_s2_seq64k_per_fp8_d()
    test_sfa_bf16_b1_s256_seq64k_fp8_p()
