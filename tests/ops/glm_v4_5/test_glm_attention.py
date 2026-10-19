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
import os
import math
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
from torch._subclasses.fake_tensor import FakeTensor
from numpy.testing import assert_allclose
import pypto
from common_utils import detailed_allclose_manual as compare
from glm_v4_5.glm_attention_impl import (
    attention, attention_for_950, attention_for_950_high_through, attention_for_910_high_performance, \
    attention_for_910_s1_range_high_performance, IfaTileShapeConfig, IfaConfig
)


np.random.seed(0)
torch.manual_seed(0)
np.set_printoptions(formatter={'float': '{:.6f}'.format})


def get_case_config(case_name: str):
    m_tile = 128
    cube_tile = 128
    test_case_config = {
        "ifa_b16_s1_1_s2_8k": {
            "b": 16,
            "s1": 1,
            "s2": 8192,
            "nq": 12,
            "nkv": 1,
            "qd": 128,
            "block_size": 128,
            "tile_config": IfaTileShapeConfig(
                g_tile=12,
                s2_tile=1024,
                c1_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
                v1_tile_shape=[m_tile, 512],
                c2_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
                v2_tile_shape=[m_tile, cube_tile],
            ),
        },
        "ifa_b64_s1_2_s2_8k": {
            "b": 64,
            "s1": 2,
            "s2": 8192,
            "nq": 12,
            "nkv": 1,
            "qd": 128,
            "block_size": 128,
            "tile_config": IfaTileShapeConfig(
                g_tile=12,
                s2_tile=1024,
                c1_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
                v1_tile_shape=[m_tile, 512],
                c2_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
                v2_tile_shape=[m_tile, cube_tile],
            ),
        },
        "ifa_b64_s1_2_s2_8k_uniform": {
            "b": 64,
            "s1": 2,
            "s2": [2048] * 12 + [4096] * 12 + [8192] * 16 + [12288] * 12 + [16384] * 12,
            "nq": 12,
            "nkv": 1,
            "qd": 128,
            "block_size": 128,
            "tile_config": IfaTileShapeConfig(
                g_tile=12,
                s2_tile=1024,
                c1_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
                v1_tile_shape=[m_tile, 512],
                c2_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
                v2_tile_shape=[m_tile, cube_tile],
            ),
        },
        "ifa_b8_s1_1_s2_16k": {
            "b": 8,
            "s1": 1,
            "s2": 16384,
            "nq": 12,
            "nkv": 1,
            "qd": 128,
            "block_size": 128,
            "tile_config": IfaTileShapeConfig(
                g_tile=12,
                s2_tile=512,
                c1_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
                v1_tile_shape=[m_tile, 512],
                c2_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
                v2_tile_shape=[m_tile, cube_tile],
            ),
        },
        "ifa_b16_s1_1_s2_16k_nkv_2": {
            "b": 16,
            "s1": 1,
            "s2": 16384,
            "nq": 12,
            "nkv": 2,
            "qd": 128,
            "block_size": 128,
            "tile_config": IfaTileShapeConfig(
                g_tile=6,
                s2_tile=512,
                c1_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
                v1_tile_shape=[m_tile, 512],
                c2_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
                v2_tile_shape=[m_tile, cube_tile],
            ),
        },
        "ifa_950_b16_s1_1_s2_8k_nkv_2": {
            "b": 16,
            "s1": 1,
            "s2": 8192,
            "nq": 12,
            "nkv": 2,
            "qd": 128,
            "block_size": 128,
            "tile_config": IfaTileShapeConfig(
                g_tile=6,
                s2_tile=1024,
                c1_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
                v1_tile_shape=[m_tile, 1024],
                c2_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
                v2_tile_shape=[m_tile, cube_tile],
            ),
        },
        "ifa_950_b16_s1_1_s2_8k": {
            "b": 16,
            "s1": 1,
            "s2": 8192,
            "nq": 12,
            "nkv": 1,
            "qd": 128,
            "block_size": 128,
            "tile_config": IfaTileShapeConfig(
                g_tile=12,
                s2_tile=1024,
                c1_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
                v1_tile_shape=[m_tile, 1024],
                c2_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
                v2_tile_shape=[m_tile, cube_tile],
            ),
        },
        "ifa_950_b64_s1_1_s2_8k": {
            "b": 64,
            "s1": 1,
            "s2": 8192,
            "nq": 12,
            "nkv": 1,
            "qd": 128,
            "block_size": 128,
            "tile_config": IfaTileShapeConfig(
                g_tile=12,
                s2_tile=1024,
                c1_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
                v1_tile_shape=[m_tile, 1024],
                c2_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
                v2_tile_shape=[m_tile, cube_tile],
            ),
        },
        "ifa_950_b16_s1_1_s2_16k": {
            "b": 16,
            "s1": 1,
            "s2": 16384,
            "nq": 12,
            "nkv": 1,
            "qd": 128,
            "block_size": 128,
            "tile_config": IfaTileShapeConfig(
                g_tile=12,
                s2_tile=1024,
                c1_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
                v1_tile_shape=[m_tile, 1024],
                c2_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
                v2_tile_shape=[m_tile, cube_tile],
            ),
        },
        "ifa_950_b64_s1_2_s2_8k_high_through": {
            "b": 64,
            "s1": 2,
            "s2": 8192,
            "nq": 12,
            "nkv": 1,
            "qd": 128,
            "block_size": 128,
            "tile_config": IfaTileShapeConfig(
                g_tile=12,
                s2_tile=1024,
                c1_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
                v1_tile_shape=[m_tile, 1024],
                c2_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
                v2_tile_shape=[m_tile, cube_tile],
            ),
        },
    }
    return test_case_config.get(case_name)


def build_ifa_config(case_config):
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    device = f'npu:{device_id}'

    b = case_config["b"]
    s1 = case_config["s1"]
    s2 = case_config["s2"]
    nq = case_config["nq"]
    nkv = case_config["nkv"]
    qd = case_config["qd"]
    block_size = case_config["block_size"]
    kv_layout = "PA_BSND"
    softmax_scale = qd ** -0.5
    block_table_batch = b

    if isinstance(s2, int):
        actual_seq_values = [s2] * b
    elif isinstance(s2, list):
        if len(s2) == b:
            actual_seq_values = s2
            s2 = max(s2)
        else:
            raise RuntimeError("unsupported actual_seq list length")
    else:
        raise RuntimeError("unsupported actual_seq data type")

    kv_num_blocks = b * ((s2 + block_size - 1) // block_size)

    actual_seq_tensor = torch.tensor(actual_seq_values, dtype=torch.int32, device=device)

    atten_cfg = IfaConfig(
        b=b,
        s1=s1,
        s2=s2,
        nq=nq,
        nkv=nkv,
        qd=qd,
        kvd=qd,
        block_size=block_size,
        softmax_scale=softmax_scale,
        kv_layout=kv_layout,
        block_table_batch=block_table_batch,
        kv_num_blocks=kv_num_blocks,
        actual_seq=actual_seq_tensor,
    )
    atten_cfg.max_num_blocks_per_query = (s2 + block_size - 1) // block_size

    return atten_cfg, case_config["tile_config"]


def gen_block_table(actual_seq_len, block_size, block_table_shape):
    block_num_per_batch = []
    block_num = 0

    if isinstance(actual_seq_len, torch.Tensor):
        if actual_seq_len.device.type != 'cpu':
            actual_seq_len_cpu = actual_seq_len.cpu()
        else:
            actual_seq_len_cpu = actual_seq_len

        for actual_seq in actual_seq_len_cpu:
            block_num_per_batch.append(math.ceil(actual_seq.item() / block_size))
            block_num += math.ceil(actual_seq.item() / block_size)
    else:
        for actual_seq in actual_seq_len:
            block_num_per_batch.append(math.ceil(actual_seq / block_size))
            block_num += math.ceil(actual_seq / block_size)

    block_idx_list = torch.arange(0, block_num, dtype=torch.int32)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))]

    block_table = torch.full(block_table_shape, -1, dtype=torch.int32)
    block_idx = 0
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_batch_idx += 1
    return block_table


def kv_cache_concat_bsnd(kr_cache_out, kv_cache_out, block_table, atten_config):
    b = atten_config.b
    nkv = atten_config.nkv
    kv_lora_rank = atten_config.qd
    rope_dim = atten_config.kvd
    block_size = atten_config.block_size
    kv_cache_actual_seq = atten_config.actual_seq
    dtype = kv_cache_out.dtype

    if isinstance(kv_cache_actual_seq, torch.Tensor):
        if kv_cache_actual_seq.device.type != 'cpu':
            kv_cache_actual_seq_cpu = kv_cache_actual_seq.cpu()
        else:
            kv_cache_actual_seq_cpu = kv_cache_actual_seq
        kv_max = (torch.max(kv_cache_actual_seq_cpu).item() + block_size - 1) // block_size * block_size
    else:
        kv_max = (max(kv_cache_actual_seq) + block_size - 1) // block_size * block_size

    device = kr_cache_out.device
    k_cache = torch.zeros([b, kv_max, nkv, kv_lora_rank], dtype=dtype, device=device)
    v_cache = torch.zeros([b, kv_max, nkv, rope_dim], dtype=dtype, device=device)

    for b_idx in range(b):
        block_list = block_table[b_idx]
        kv_nope_temp_tensor = torch.zeros([1, kv_max, nkv, kv_lora_rank], dtype=dtype, device=device)
        kv_rope_temp_tensor = torch.zeros([1, kv_max, nkv, rope_dim], dtype=dtype, device=device)
        s_idx = 0

        for _, block_idx in enumerate(block_list):
            if block_idx == -1:
                break
            start_idx = s_idx * block_size
            end_idx = (s_idx + 1) * block_size

            kv_nope_temp_tensor[:, start_idx:end_idx, :, :] = kv_cache_out[block_idx:block_idx + 1, :, :, :]
            kv_rope_temp_tensor[:, start_idx:end_idx, :, :] = kr_cache_out[block_idx:block_idx + 1, :, :, :]
            s_idx += 1

        v_cache[b_idx:b_idx + 1, :, :, :] = kv_nope_temp_tensor
        k_cache[b_idx:b_idx + 1, :, :, :] = kv_rope_temp_tensor

    return k_cache, v_cache


def get_special_array(m, n):
    q_shape = [m, n]

    base = np.arange(1, m + 1)

    q = base[:, np.newaxis]
    q = np.broadcast_to(q, q_shape)

    q = q.astype(np.float16)
    return q


def softmax(x, is_fp16=False):
    if is_fp16:
        original_dtype = x.dtype
        x = x.float()
    x_max = x.max(dim=-1, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)
    ans = y / x_sum
    if is_fp16:
        ans = ans.to(original_dtype)
        x_max = x_max.to(original_dtype)
        x_sum = x_sum.to(original_dtype)

    return ans, x_max, x_sum



def ifa(atten_cfg, tile_config, is_950=False, is_high_through=False, is_high_precision=True):
    np.random.seed(0)
    torch.manual_seed(0)
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch_dtype = torch.bfloat16
    torch.npu.set_device(int(device_id))

    b = atten_cfg.b
    s1 = atten_cfg.s1
    d = atten_cfg.qd
    nq = atten_cfg.nq
    nkv = atten_cfg.nkv

    block_size = atten_cfg.block_size
    max_num_blocks_per_query = atten_cfg.max_num_blocks_per_query

    kv_cache_actual_seq = atten_cfg.actual_seq

    q_shape = [b * s1, nq, d]
    kv_shape = [atten_cfg.kv_num_blocks, block_size, nkv, d]
    block_table_shape = [atten_cfg.block_table_batch, max_num_blocks_per_query]

    device = f'npu:{device_id}'
    q = torch.empty(q_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    k = torch.empty(kv_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    v = torch.empty(kv_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    attention_output = torch.zeros(q_shape, dtype=torch_dtype).to(device=device)

    block_table = gen_block_table(kv_cache_actual_seq, block_size, block_table_shape)

    k_cache_bsnd, v_cache_bsnd = kv_cache_concat_bsnd(k, v, block_table, atten_cfg)

    block_table_torch = block_table.to(dtype=torch.int32, device=device)
    act_seq_torch = kv_cache_actual_seq.to(dtype=torch.int32, device=device)

    out_torch = torch.zeros(q_shape, dtype=torch_dtype).to(device=device)

    if is_high_precision:
        group = nq // nkv
        for i in range(b):
            for j in range(s1):
                for n2_idx in range(nkv):
                    kv_seq_len = kv_cache_actual_seq[i].item()
                    seq_len = kv_seq_len - s1 + 1 + j
                    q_group = q[i * s1 + j, n2_idx * group:(n2_idx + 1) * group]
                    k_bs = k_cache_bsnd[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
                    v_bs = v_cache_bsnd[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
                    qk_bmm_res = torch.matmul(q_group, k_bs.transpose(1, 0))
                    qk_ele_res = qk_bmm_res * atten_cfg.softmax_scale
                    softmax_res, _, _ = softmax(qk_ele_res, True)
                    bmm2_res = torch.matmul(softmax_res, v_bs)
                    attention_output[i * s1 + j, n2_idx * group:(n2_idx + 1) * group] = bmm2_res
    else:
        ifa_flash_torch(q=q, k=k, v=v, block_table=block_table_torch, kv_act_seqs=act_seq_torch, out=attention_output)

    inputs = [q, k, v, block_table_torch, act_seq_torch, out_torch]
    if is_950:
        if is_high_through:
            attention_for_950_high_through(*inputs, atten_cfg.softmax_scale, tile_config)
        else:
            attention_for_950(*inputs, atten_cfg.softmax_scale, tile_config)
    else:
        if is_high_through:
            if atten_cfg.s1 == 2:
                attention_for_910_s1_range_high_performance(*inputs, atten_cfg.softmax_scale, tile_config)
            else:
                attention_for_910_high_performance(*inputs, atten_cfg.softmax_scale, tile_config)
        else:
            attention(*inputs, atten_cfg.softmax_scale, tile_config)

    compare(
        np.array(attention_output.cpu().flatten().tolist()),
        np.array(out_torch.cpu().flatten().tolist()),
        "out_torch",
        rtol=0.0078125,
        atol=0.0001,
    )


def matmul_proxy(left, right):
    torch_fp32 = torch.float32
    return torch.matmul(left.to(torch_fp32), right.to(torch_fp32))


def ifa_flash_torch(q, k, v, block_table, kv_act_seqs, out, is_fp32=False):
    """
    PyTorch版本的ifa_flash_torch（修正原NumPy代码的关键bug，适配张量操作）
    参数说明：
        q: torch.Tensor, shape [b*s1, n1, d]  # b: batch数, s1: query序列长度, n1: query头数, d: 头维度
        k: torch.Tensor, shape [block_num, block_size, n2, d]
        # block_num: block总数, block_size: 每个block的长度, n2: key/value头数
        v: torch.Tensor, shape [block_num, block_size, n2, d]
        block_table: torch.Tensor, shape [b, max_block]  # 每个样本的block索引表
        kv_act_seqs: torch.Tensor, shape [b]  # 每个样本的kv有效序列长度
        out: torch.Tensor, shape [b*s1, n1, d]  # 输出注意力结果（需预先分配空间）
    """
    torch_fp32 = torch.float32
    if is_fp32:
        q = q.to(torch_fp32)
        k = k.to(torch_fp32)
        v = v.to(torch_fp32)

    q_shape = q.shape
    bs1, n1, d = q_shape[0], q_shape[1], q_shape[2]
    b = kv_act_seqs.shape[0]
    s1 = bs1 // b
    k_shape = k.shape
    block_num, block_size, n2, _ = k_shape
    g = n1 // n2
    g_tile = g

    k_2d = k.reshape(block_num * block_size, n2 * d)
    v_2d = v.reshape(block_num * block_size, n2 * d)
    q_2d = q.reshape(-1, d)

    for b_idx in range(b):
        for s1_idx in range(s1):
            cur_seq = kv_act_seqs[b_idx] - (s1 - 1 - s1_idx)
            cur_seq = max(cur_seq.item(), 0)
            s2_loop = math.ceil(cur_seq / block_size)

            for n2_idx in range(n2):
                for g_idx in range(g // g_tile):
                    device = q.device
                    dtype = q.dtype
                    oi_upd = torch.zeros((g_tile, d), device=device, dtype=torch_fp32)
                    li_upd = torch.zeros(g_tile, device=device, dtype=torch_fp32)
                    mi_upd = torch.zeros(g_tile, device=device, dtype=torch_fp32)

                    for s2_idx in range(s2_loop):
                        block_idx = block_table[b_idx][s2_idx].item()

                        bs_ofs = b_idx * s1 + s1_idx
                        n2g_ofs = n2_idx * g + g_idx * g_tile
                        actual_s2_tile = min(block_size, cur_seq - s2_idx * block_size)

                        qi_start = bs_ofs * n1 + n2g_ofs
                        qi_end = qi_start + g_tile
                        qi = q_2d[qi_start:qi_end, :]

                        kj_start = block_idx * block_size
                        kj_end = kj_start + actual_s2_tile
                        kj = k_2d[kj_start:kj_end, n2_idx * d:(n2_idx + 1) * d]

                        vj = v_2d[kj_start:kj_end, n2_idx * d:(n2_idx + 1) * d]

                        mm1 = matmul_proxy(qi, kj.t()).to(torch_fp32)
                        muls_res = mm1 * (d ** -0.5)
                        tilda_mij, _ = torch.max(muls_res, dim=-1, keepdim=True)
                        tsub = muls_res - tilda_mij
                        tilda_pij = torch.exp(tsub)
                        tilda_lij = torch.sum(tilda_pij, dim=-1, keepdim=True)

                        if s2_idx == 0:
                            oi_tmp = matmul_proxy(tilda_pij.to(dtype), vj).to(torch_fp32)
                            if s2_idx == s2_loop - 1:
                                oi_upd = oi_tmp / tilda_lij
                                out[bs_ofs:bs_ofs + 1, n2g_ofs:n2g_ofs + g_tile, :] = oi_upd.unsqueeze(0).to(dtype)
                            else:
                                oi_upd = oi_tmp
                            li_upd = tilda_lij.squeeze(-1)
                            mi_upd = tilda_mij.squeeze(-1)
                        else:
                            oi = oi_upd
                            li = li_upd.unsqueeze(-1)
                            mi = mi_upd.unsqueeze(-1)

                            mi_new, _ = torch.max(torch.cat([mi, tilda_mij], dim=-1), dim=-1, keepdim=True)
                            t1 = mi - mi_new
                            t2 = torch.exp(t1)
                            t3 = tilda_mij - mi_new
                            t4 = torch.exp(t3)
                            t5 = t4 * tilda_lij
                            t6 = t2 * li
                            li_new = t6 + t5
                            q3 = oi * t2
                            q1 = matmul_proxy(tilda_pij.to(dtype), vj).to(torch_fp32)
                            q2 = q1 * t4
                            oi_tmp = q3 + q2

                            if s2_idx == s2_loop - 1:
                                oi_upd = oi_tmp / li_new
                                oi_upd_3d = oi_upd.unsqueeze(0)
                                attn_out_start_col = n2g_ofs
                                attn_out_end_col = n2g_ofs + g_tile
                                if attn_out_end_col > out.shape[1]:
                                    attn_out_end_col = out.shape[1]
                                    attn_out_start_col = attn_out_end_col - g_tile
                                out[bs_ofs:bs_ofs + 1, attn_out_start_col:attn_out_end_col, :] = oi_upd_3d.to(dtype)
                            else:
                                oi_upd = oi_tmp
                            li_upd = li_new.squeeze(-1)
                            mi_upd = mi_new.squeeze(-1)
    return out


@pytest.mark.soc("950")
def test_ifa_for_950():
    case_names = [
        "ifa_950_b16_s1_1_s2_8k_nkv_2",
        "ifa_950_b16_s1_1_s2_8k",
        "ifa_950_b16_s1_1_s2_16k",
        "ifa_950_b64_s1_1_s2_8k",
    ]
    for case_name in case_names:
        case_config = get_case_config(case_name)
        atten_cfg, tile_config = build_ifa_config(case_config)

        assert atten_cfg.b == len(atten_cfg.actual_seq), (
            f'{atten_cfg.b} {atten_cfg.actual_seq} B的大小必须和actual_seq长度相等'
        )

        if atten_cfg.actual_seq.device.type != 'cpu':
            actual_seq_cpu = atten_cfg.actual_seq.cpu()
        else:
            actual_seq_cpu = atten_cfg.actual_seq

        assert all(x <= atten_cfg.s2 for x in actual_seq_cpu), "所有值都必须小于s2"
        ifa(atten_cfg, tile_config, is_950=True, is_high_through=False, is_high_precision=False)


@pytest.mark.soc("950")
def test_ifa_for_950_high_through():
    case_names = [
        "ifa_950_b64_s1_2_s2_8k_high_through",
    ]
    for case_name in case_names:
        case_config = get_case_config(case_name)
        atten_cfg, tile_config = build_ifa_config(case_config)

        assert atten_cfg.b == len(atten_cfg.actual_seq), (
            f'{atten_cfg.b} {atten_cfg.actual_seq} B的大小必须和actual_seq长度相等'
        )

        if atten_cfg.actual_seq.device.type != 'cpu':
            actual_seq_cpu = atten_cfg.actual_seq.cpu()
        else:
            actual_seq_cpu = atten_cfg.actual_seq

        assert all(x <= atten_cfg.s2 for x in actual_seq_cpu), "所有值都必须小于s2"
        ifa(atten_cfg, tile_config, is_950=True, is_high_through=True, is_high_precision=False)


@pytest.mark.soc("950", "910")
def test_ifa():
    case_names = [
        "ifa_b8_s1_1_s2_16k",
        "ifa_b16_s1_1_s2_16k_nkv_2",
    ]
    for case_name in case_names:
        case_config = get_case_config(case_name)
        atten_cfg, tile_config = build_ifa_config(case_config)

        assert atten_cfg.b == len(atten_cfg.actual_seq), (
            f'{atten_cfg.b} {atten_cfg.actual_seq} B的大小必须和actual_seq长度相等'
        )

        if atten_cfg.actual_seq.device.type != 'cpu':
            actual_seq_cpu = atten_cfg.actual_seq.cpu()
        else:
            actual_seq_cpu = atten_cfg.actual_seq

        assert all(x <= atten_cfg.s2 for x in actual_seq_cpu), "所有值都必须小于s2"
        ifa(atten_cfg, tile_config, is_high_precision=False)


@pytest.mark.soc("950", "910")
def test_ifa_910_high_performance():
    case_names = [
        "ifa_b16_s1_1_s2_8k",
        "ifa_b64_s1_2_s2_8k",
        "ifa_b64_s1_2_s2_8k_uniform",
    ]
    for case_name in case_names:
        case_config = get_case_config(case_name)
        atten_cfg, tile_config = build_ifa_config(case_config)

        assert atten_cfg.b == len(atten_cfg.actual_seq), (
            f'{atten_cfg.b} {atten_cfg.actual_seq} B的大小必须和actual_seq长度相等'
        )

        if atten_cfg.actual_seq.device.type != 'cpu':
            actual_seq_cpu = atten_cfg.actual_seq.cpu()
        else:
            actual_seq_cpu = atten_cfg.actual_seq

        assert all(x <= atten_cfg.s2 for x in actual_seq_cpu), "所有值都必须小于s2"
        ifa(atten_cfg, tile_config, is_high_through=True, is_high_precision=False)



if __name__ == "__main__":
    test_ifa()
    test_ifa_910_high_performance()
    if pypto.platform.npuarch == 'DAV_3510':
        test_ifa_for_950()
        test_ifa_for_950_high_through()
