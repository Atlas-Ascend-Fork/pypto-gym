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
import logging
import math
from typing import Any
import pytest
import torch
import torch_npu
import numpy as np
import pypto
import sys

_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

from common_utils import compare
from deepseek_v32_exp.lightning_indexer_quant_impl import LightningIndexerConfigs


def gen_cache_tensor(k_tensor, block_table, block_num, block_size, b):
    logging.info("Entering into gen_cache_tensor!")
    dtype = k_tensor.dtype
    b, s, n, d = k_tensor.shape
    k_cache = torch.zeros([block_num, block_size, n * d], dtype=dtype)
    k_tensor_bsh_raw = k_tensor.reshape(b, s, n * d)
    k_tensor_bsh = torch.zeros(
        (b, block_table.shape[1] * block_size, n * d), dtype=dtype)
    k_tensor_bsh[:, : k_tensor_bsh_raw.shape[1], :] = k_tensor_bsh_raw[:, :, :]
    for b_idx in range(b):
        for block_idx, cache_block_idx in enumerate(block_table[b_idx]):
            block_offset = block_idx * block_size
            if cache_block_idx != -1:
                k_cache[cache_block_idx, :, :] = k_tensor_bsh[b_idx,
                                                              block_offset: (block_offset + block_size), :]
    k_cache = k_cache.reshape(block_num, block_size, n, d)
    return k_cache


def gen_block_table(b, block_size, max_kv, act_kv):
    logging.info("Entering into gen_block_table!")
    block_num = 0
    block_num_each = []
    for cur_s in act_kv:
        cur_block_num = math.ceil(cur_s / block_size)
        block_num_each.append(cur_block_num)
        block_num += cur_block_num
    shape_bt = [b, math.ceil(max_kv / block_size)]
    block_idx_list = np.arange(0, block_num, 1)
    block_idx_list = np.random.permutation(block_idx_list).astype(np.int32)
    block_table = [-1] * shape_bt[1]
    block_table = np.tile(block_table, (shape_bt[0], 1)).astype(np.int32)
    block_table_bidx = 0
    block_idx = 0
    for cur_block in block_num_each:
        for j in range(cur_block):
            block_table[block_table_bidx][j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_bidx += 1
    return block_num, block_table


def gen_data_for_compute(params, is_quant: bool):
    b = params.get("b")
    s1 = params.get("s1")
    n1 = params.get("n1")
    n2 = params.get("n2")
    d = params.get("d")
    dtype = params.get("dtype")
    s2 = params.get("s2")
    act_seq_len = params.get("act_seq")
    block_size = params.get("block_size")
    block_num = params.get("block_num")
    selected_count = params.get("selected_count")

    query = torch.randn([b, s1, n1, d]).to(torch.int8)
    weights = torch.randn([b, s1, n1], dtype=dtype).to(torch.float16)
    k_bsnd = torch.randn([b, s2, n2, d]).to(torch.int8)
    _, block_table_list = gen_block_table(b, block_size, s2, act_seq_len)
    block_table = torch.tensor(block_table_list, dtype=torch.int32)
    act_seq = torch.tensor(act_seq_len, dtype=torch.int32)
    topk_res = torch.ones([b, s1, n2, selected_count], dtype=torch.int32)
    key = gen_cache_tensor(k_bsnd, block_table_list, block_num, block_size, b)

    input_data_map = {}
    if is_quant:
        q_scale = (query.abs().max(dim=-1, keepdim=True).values / 127).\
                    to(dtype=torch.float16).maximum(torch.tensor(1e-3))
        k_scale = (key.abs().max(dim=-1, keepdim=True).values / 127).to(dtype=torch.float16).maximum(torch.tensor(1e-3))
        query = torch.round(query / q_scale).clip(-127, 127).to(dtype=torch.int8)
        key = torch.round(key / k_scale).clip(-127, 127).to(dtype=torch.int8)
        input_data_map["query"] = query
        input_data_map["key"] = key
        input_data_map["q_scale"] = q_scale
        input_data_map["k_scale"] = k_scale
    else:
        input_data_map["query"] = query
        input_data_map["key"] = key

    input_data_map["weights"] = weights
    input_data_map["act_seq"] = act_seq
    input_data_map["block_table"] = block_table
    input_data_map["selected_count"] = selected_count
    input_data_map["topk_res"] = topk_res
    return input_data_map


def _init_compute_tensors(input_data_map, params):
    """Extract params, reshape tensors, and allocate output buffers for lightning_indexer_compute."""
    block_size = params.get("block_size")
    selected_count = params.get("selected_count")
    b = params.get("b")
    s1 = params.get("s1")
    n1 = params.get("n1")
    d = params.get("d")
    block_num = params.get("block_num")
    max_block_num = params.get("max_block_num")

    query = input_data_map.get("query")
    key = input_data_map.get("key")
    q_scale = input_data_map.get("q_scale")
    k_scale = input_data_map.get("k_scale")
    weights = input_data_map.get("weights")
    act_seq = input_data_map.get("act_seq")
    block_table = input_data_map.get("block_table")

    topk_res = torch.zeros([b * s1, 1, selected_count], dtype=torch.int32)
    first_mm = torch.zeros(b * s1 * n1, max_block_num * block_size, dtype=torch.float16)
    mm_out = torch.zeros([b * s1 * 1, max_block_num * block_size], dtype=torch.float32)
    avoid_fp32_to_fp16_overflow_scale = 1.0 / 2048

    query = query.reshape(b * s1 * n1, d)
    q_scale = q_scale.reshape(b * s1, 1, n1)
    key = key.reshape(block_num * block_size, d)
    k_scale = k_scale.reshape(block_num, block_size)
    weights = weights.reshape(b * s1, 1, n1)

    return (b, s1, n1, d, block_size, block_num, max_block_num, selected_count,
            query, key, q_scale, k_scale, weights, act_seq, block_table,
            topk_res, first_mm, mm_out, avoid_fp32_to_fp16_overflow_scale)


@dataclass
class _LightningBatchBlockLoopInputs:
    b: Any
    s1: Any
    n1: Any
    d: Any
    block_size: Any
    block_num: Any
    query: Any
    key: Any
    q_scale: Any
    k_scale: Any
    weights: Any
    act_seq: Any
    block_table: Any
    first_mm: Any
    mm_out: Any
    avoid_fp32_to_fp16_overflow_scale: Any


def _lightning_batch_block_loop(inputs: _LightningBatchBlockLoopInputs):
    """Main batch+block loop for lightning indexer golden compute."""
    for b_idx in range(inputs.b):
        cur_seq = inputs.act_seq[b_idx]
        cur_block = (cur_seq + inputs.block_size - 1) // inputs.block_size
        cur_qs = inputs.q_scale[b_idx * inputs.s1:(b_idx + 1) * inputs.s1, :, :]
        cur_w = inputs.weights[b_idx * inputs.s1:(b_idx + 1) * inputs.s1, :, :]
        w_scale = cur_qs * cur_w

        for block_idx in range(cur_block):
            cur_q = inputs.query[b_idx * inputs.s1 * inputs.n1: (b_idx + 1) * inputs.s1 * inputs.n1, :]
            cur_block_idx = inputs.block_table[b_idx][block_idx]
            tail_seq = min(inputs.block_size, cur_seq - inputs.block_size * block_idx)
            cur_k = inputs.key[cur_block_idx * inputs.block_size: (cur_block_idx * inputs.block_size + tail_seq), :]
            qk_dot = torch.matmul(cur_q.to(torch.int32),
                                  cur_k.transpose(1, 0).to(torch.int32)).to(torch.float32).relu()
            qk_dot = qk_dot * inputs.avoid_fp32_to_fp16_overflow_scale
            qk_dot = qk_dot.to(torch.float16)
            start_row = b_idx * inputs.s1 * inputs.n1
            end_row = (b_idx + 1) * inputs.s1 * inputs.n1
            start_col = block_idx * inputs.block_size
            inputs.first_mm[start_row:end_row, start_col:(start_col + tail_seq)] = qk_dot
            qk_dot = qk_dot.reshape(inputs.s1, inputs.n1, tail_seq)
            cur_ks = inputs.k_scale[cur_block_idx:(cur_block_idx + 1), :tail_seq]
            cur_ks = cur_ks.to(torch.float32)
            w_qk = torch.bmm(w_scale.to(torch.float32), qk_dot.to(torch.float32))
            w_qk = w_qk.reshape(inputs.s1, tail_seq)
            k_res = w_qk * cur_ks
            row_s = b_idx * inputs.s1
            col_s = block_idx * inputs.block_size
            inputs.mm_out[row_s:(b_idx + 1) * inputs.s1, col_s:(col_s + tail_seq)] = k_res


def _lightning_topk_select(b, s1, selected_count, act_seq, mm_out, topk_res):
    """Top-k selection for all batches and sequences."""
    for b_idx in range(b):
        cur_seq = act_seq[b_idx]
        for s_idx in range(s1):
            eff_seq = cur_seq - (s1 - s_idx - 1)
            topk_in = mm_out[(b_idx * s1 + s_idx):(b_idx * s1 + s_idx + 1), :eff_seq]
            if eff_seq < selected_count:
                cur_res, cur_idx = torch.topk(topk_in, k=eff_seq, dim=-1)
                pad_res = torch.full((1, selected_count - eff_seq), float("-inf"), dtype=torch.float32)
                pad_idx = torch.full((1, selected_count - eff_seq), -1, dtype=torch.int32)
                cur_res = torch.cat([cur_res, pad_res], dim=1)
                cur_idx = torch.cat([cur_idx, pad_idx], dim=1)
                topk_res[(b_idx * s1 + s_idx):(b_idx * s1 + s_idx + 1), :, :] = cur_idx.reshape(1, 1, selected_count)
            else:
                cur_res, cur_idx = torch.topk(topk_in, k=selected_count, dim=-1)
                topk_res[(b_idx * s1 + s_idx):(b_idx * s1 + s_idx + 1), :, :] = cur_idx.reshape(1, 1, selected_count)


def lightning_indexer_compute(input_data_map, params):
    (b, s1, n1, d, block_size, block_num, max_block_num, selected_count,
     query, key, q_scale, k_scale, weights, act_seq, block_table,
     topk_res, first_mm, mm_out, avoid_fp32_to_fp16_overflow_scale) = _init_compute_tensors(input_data_map, params)

    _lightning_batch_block_loop(_LightningBatchBlockLoopInputs(
        b, s1, n1, d, block_size, block_num,
        query, key, q_scale, k_scale, weights,
        act_seq, block_table, first_mm, mm_out,
        avoid_fp32_to_fp16_overflow_scale))
    _lightning_topk_select(b, s1, selected_count, act_seq, mm_out, topk_res)
    return topk_res


def topk_idx_compare(t: torch.Tensor, t_ref: torch.Tensor, name, atol, error_count_threshold):
    part_result_dict = {}
    err_msg = None
    for idx, (act, exp) in enumerate(zip(t.flatten().tolist(), t_ref.flatten().tolist())):
        part_index = idx // error_count_threshold
        if exp != act:
            if part_index not in part_result_dict:
                part_result_dict[part_index] = {
                    "exp": [],
                    "act": []
                }
            part_result_dict[part_index]["exp"].append(exp)
            part_result_dict[part_index]["act"].append(act)

    precision = "PASS"
    for idx_index in part_result_dict.keys():
        exp_list = part_result_dict[idx_index]["exp"]
        act_list = part_result_dict[idx_index]["act"]
        exp_list.sort()
        act_list.sort()
        error_count = 0
        for topk_id in exp_list:
            if topk_id not in act_list:
                error_count += 1
        if error_count > int(error_count_threshold * atol):
            precision = "FAIL"
            err_msg = f"compare fail: {name}, error_count: {error_count}, \
                        error_count_threshold: {int(error_count_threshold * atol)}"
            break
    assert precision == "PASS", err_msg


@dataclass
class _BuildLightningParamsInputs:
    b: Any
    s1: Any
    n1: Any
    n2: Any
    d: Any
    dtype: Any
    s2: Any
    act_seq: Any
    block_size: Any
    block_num: Any
    max_block_num: Any
    selected_count: Any


def _build_lightning_params(inputs: _BuildLightningParamsInputs):
    """Build the params dictionary for lightning_indexer."""
    params = {
        "b": inputs.b,
        "s1": inputs.s1,
        "n1": inputs.n1,
        "n2": inputs.n2,
        "d": inputs.d,
        "dtype": inputs.dtype,
        "s2": inputs.s2,
        "act_seq": inputs.act_seq,
        "block_size": inputs.block_size,
        "block_num": inputs.block_num,
        "max_block_num": inputs.max_block_num,
        "selected_count": inputs.selected_count
    }
    return params


def _execute_lightning_kernel(b, s1, n1, block_num, block_size, selected_count,
                               input_data_map, unroll_list, configs):
    """Build NPU inputs, run kernel, synchronize, and return result."""
    from deepseek_v32_exp.lightning_indexer_quant_impl import lightning_indexer_decode
    idx_query_npu = input_data_map["query"].reshape(b * s1, n1, 128).npu()
    idx_query_scale_npu = input_data_map["q_scale"].reshape(b * s1, n1).npu()
    idx_key_cache_npu = input_data_map["key"].npu()
    idx_key_scale_npu = input_data_map["k_scale"].reshape(block_num, block_size, 1).npu()
    idx_weight_npu = input_data_map["weights"].reshape(b * s1, n1).npu()
    act_seq_key_npu = input_data_map["act_seq"].npu()
    block_table_npu = input_data_map["block_table"].npu()

    topk_res_out = torch.zeros([b * s1, 1, selected_count], dtype=torch.int32)
    topk_res_npu = topk_res_out.npu()

    lightning_indexer_decode(idx_query_npu, idx_query_scale_npu, idx_key_cache_npu, idx_key_scale_npu,
                          idx_weight_npu, act_seq_key_npu, block_table_npu, topk_res_npu,
                          unroll_list, configs, selected_count)
    torch_npu.npu.synchronize()
    return topk_res_npu


def lightning_indexer(case_name: str) -> bool:
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    n1, d = 64, 128
    n2 = 1
    block_size = 128
    dtype = torch.float16

    if case_name == "LightningIndexerSTest.lightning_indexer_quant_4_b_2_s1_64k_s2":
        b, s1 = 4, 2
        act_seq = [64 * 1024, 971, 32 * 1024 + 101, 16 * 1024 - 1]
    elif case_name == "LightningIndexerSTest.lightning_indexer_quant_8_b_2_s1_64k_s2":
        b, s1 = 8, 2
        act_seq = [32767, 32656, 384, 2000, 64 * 1024, 971, 32 * 1024 + 101, 129090]
    elif case_name == "LightningIndexerSTest.lightning_indexer_quant_4_b_2_s1_64k_s2_perf":
        b, s1 = 4, 2
        act_seq = [64 * 1024] * b
    else:
        logging.error("Fail to gen golden for Case(%s)", case_name)
        return False

    s2 = max(act_seq)
    block_num = sum([(s + block_size - 1) // block_size for s in act_seq])
    max_block_num = (s2 + block_size - 1) // block_size
    selected_count = 2048

    params = _build_lightning_params(_BuildLightningParamsInputs(
        b, s1, n1, n2, d, dtype, s2, act_seq,
        block_size, block_num, max_block_num, selected_count))
    input_data_map = gen_data_for_compute(params, is_quant=True)

    unroll_list = [128, 64, 32, 16, 8, 4, 1]
    configs = LightningIndexerConfigs()
    topk_res_npu = _execute_lightning_kernel(b, s1, n1, block_num, block_size, selected_count,
                                              input_data_map, unroll_list, configs)

    topk_res_golden = lightning_indexer_compute(input_data_map, params)
    topk_idx_compare(topk_res_npu.cpu(), topk_res_golden.cpu(), "topk_res", 5e-4, selected_count)
    return True


@pytest.mark.skip(reason="large shape")
def test_lightning_indexer_topk_quant_4_b_2_s1_64k_s2():
    lightning_indexer("LightningIndexerSTest.lightning_indexer_quant_4_b_2_s1_64k_s2")


@pytest.mark.skip(reason="large shape")
def test_lightning_indexer_topk_quant_8_b_2_s1_64k_s2():
    lightning_indexer("LightningIndexerSTest.lightning_indexer_quant_8_b_2_s1_64k_s2")


@pytest.mark.skip(reason="large shape")
def test_lightning_indexer_topk_quant_4_b_2_s1_64k_s2_perf():
    lightning_indexer("LightningIndexerSTest.lightning_indexer_quant_4_b_2_s1_64k_s2_perf")


if __name__ == "__main__":
    test_lightning_indexer_topk_quant_4_b_2_s1_64k_s2()
    test_lightning_indexer_topk_quant_8_b_2_s1_64k_s2()
    test_lightning_indexer_topk_quant_4_b_2_s1_64k_s2_perf()
