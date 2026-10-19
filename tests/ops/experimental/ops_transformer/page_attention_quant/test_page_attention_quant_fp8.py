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
GLM-4.5 Attention Module

This module implements the Attention mechanism for GLM-4.5 model, which uses
a paged memory management approach similar to operating systems to efficiently
handle variable-length sequences and dynamic batch sizes in attention computation.

Main Functions:
    - attention: Main attention function with Attention support
    - ifa_func: JIT compiled kernel implementing Flash Attention with paged KV cache
    - gen_block_table: Generate block mapping table for Attention
    - kv_cache_concat_bsnd: Convert paged KV cache to BSND format
"""
import os
import math
import enum
from dataclasses import dataclass
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
import numpy as np
from numpy.testing import assert_allclose
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph

from experimental.ops_transformer.page_attention_quant.page_attention_quant_fp8_impl import (
    get_case_config, build_pfa_config, pfa_func_kernel_v2_bound
)
import pypto

np.random.seed(0)
torch.manual_seed(0)
np.set_printoptions(formatter={'float': '{:.6f}'.format})
torch.set_printoptions(threshold=float('inf'))


def check_cond(cond, msg):
    if not cond:
        raise ValueError(msg)


class TileOpFormat(enum.Enum):
    ND = "ND"
    NZ = "NZ"


def get_format(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("input type error")
    if not tensor.is_contiguous():
        raise TypeError("input type error")

    tile_op_format = TileOpFormat.ND.value
    if tensor.device.type == "npu":
        if torch_npu.get_npu_format(tensor) == 29:
            tile_op_format = TileOpFormat.NZ.value
    return tile_op_format


def check_args(
    query,
    key_cache,
    value_cache,
    block_tables,
    actual_seqs,
    attn_res
):
    check_cond(query.dim() == 3, "invalid query dim.")
    check_cond(get_format(query) == 'ND', "invalid query format.")
    check_cond(query.dtype == torch.float8_e4m3fn, "invalid query dtype.")
    check_cond(key_cache.dim() == 4, "invalid key_cache dim.")
    check_cond(get_format(key_cache) == 'ND', "invalid key_cache format.")
    check_cond(key_cache.dtype == torch.float8_e4m3fn, "invalid key_cache dtype.")
    check_cond(value_cache.dim() == 4, "invalid value_cache dim.")
    check_cond(get_format(value_cache) == 'ND', "invalid value_cache format.")
    check_cond(value_cache.dtype == torch.float8_e4m3fn, "invalid value_cache dtype.")
    check_cond(block_tables.dim() == 2, "invalid block_tables dim.")
    check_cond(get_format(block_tables) == 'ND', "invalid block_tables format.")
    check_cond(block_tables.dtype == torch.int32, "invalid block_tables dtype.")
    check_cond(actual_seqs.dim() == 1, "invalid actual_seqs dim.")
    check_cond(get_format(actual_seqs) == 'ND', "invalid actual_seqs format.")
    check_cond(actual_seqs.dtype == torch.int32, "invalid actual_seqs dtype.")
    check_cond(attn_res.dim() == 3, "invalid attn_res dim.")
    check_cond(get_format(attn_res) == 'ND', "invalid attn_res format.")
    check_cond(attn_res.dtype == torch.bfloat16, "invalid attn_res dtype.")


def gen_block_table(actual_seq_len, block_size, block_table_shape):
    block_num_per_batch = []
    block_num = 0

    # 处理 torch tensor 类型的 actual_seq_len
    if isinstance(actual_seq_len, torch.Tensor):
        # 如果 tensor 在 GPU/NPU 上，先移动到 CPU
        if actual_seq_len.device.type != 'cpu':
            actual_seq_len_cpu = actual_seq_len.cpu()
        else:
            actual_seq_len_cpu = actual_seq_len

        # 转换为 numpy 数组进行处理，或者直接使用 torch 操作
        for actual_seq in actual_seq_len_cpu:
            block_num_per_batch.append(math.ceil(actual_seq.item() / block_size))
            block_num += math.ceil(actual_seq.item() / block_size)
    else:
        # 保持对 list 的兼容
        for actual_seq in actual_seq_len:
            block_num_per_batch.append(math.ceil(actual_seq / block_size))
            block_num += math.ceil(actual_seq / block_size)

    # 使用 torch 替换 numpy
    block_idx_list = torch.arange(0, block_num, dtype=torch.int32)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))]

    # 创建 block_table 张量
    block_table = torch.full(block_table_shape, -1, dtype=torch.int32)
    block_idx = 0
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_batch_idx += 1
    return block_table


def kv_cache_concat_bsnd(kr_cache_out, kv_cache_out, k_scale, block_table, atten_config):
    b = atten_config.b
    nkv = atten_config.nkv
    kv_lora_rank = atten_config.qd
    rope_dim = atten_config.kvd
    block_size = atten_config.block_size
    kv_cache_actual_seq = atten_config.actual_seq
    dtype = kr_cache_out.dtype
    scale_dtype = k_scale.dtype

    # 处理 torch tensor 类型的 kv_cache_actual_seq
    if isinstance(kv_cache_actual_seq, torch.Tensor):
        if kv_cache_actual_seq.device.type != 'cpu':
            kv_cache_actual_seq_cpu = kv_cache_actual_seq.cpu()
        else:
            kv_cache_actual_seq_cpu = kv_cache_actual_seq
        kv_max = (torch.max(kv_cache_actual_seq_cpu).item() + block_size - 1) // block_size * block_size
    else:
        kv_max = (max(kv_cache_actual_seq) + block_size - 1) // block_size * block_size

    # 使用 torch 创建张量，保持在同一设备上
    device = kr_cache_out.device
    k_cache = torch.zeros([b, kv_max, nkv, kv_lora_rank], dtype=dtype, device=device)
    k_sclae_cache = torch.zeros([b, kv_max, nkv, 1], dtype=scale_dtype, device=device)
    v_cache = torch.zeros([b, kv_max, nkv, rope_dim], dtype=kv_cache_out.dtype, device=device)

    for b_idx in range(b):
        block_list = block_table[b_idx]
        kv_nope_temp_tensor = torch.zeros([1, kv_max, nkv, kv_lora_rank], dtype=kv_cache_out.dtype, device=device)
        kv_rope_temp_tensor = torch.zeros([1, kv_max, nkv, rope_dim], dtype=dtype, device=device)
        k_scale_temp_tensor = torch.zeros([1, kv_max, nkv, 1], dtype=scale_dtype, device=device)
        s_idx = 0

        for _, block_idx in enumerate(block_list):
            if block_idx == -1:
                break
            # 使用 torch 的切片操作
            start_idx = s_idx * block_size
            end_idx = (s_idx + 1) * block_size

            kv_nope_temp_tensor[:, start_idx:end_idx, :, :] = kv_cache_out[block_idx:block_idx + 1, :, :, :]
            kv_rope_temp_tensor[:, start_idx:end_idx, :, :] = kr_cache_out[block_idx:block_idx + 1, :, :, :]
            k_scale_temp_tensor[:, start_idx:end_idx, :, :] = k_scale[block_idx:block_idx + 1, :, :, :]
            s_idx += 1

        v_cache[b_idx:b_idx + 1, :, :, :] = kv_nope_temp_tensor
        k_cache[b_idx:b_idx + 1, :, :, :] = kv_rope_temp_tensor
        k_sclae_cache[b_idx:b_idx + 1, :, :, :] = k_scale_temp_tensor

    return k_cache, v_cache, k_sclae_cache


def get_special_array(m, n):
    q_shape = [m, n]

    # 生成递增的行值
    base = np.arange(1, m + 1)  # 生成 [1, 2, ..., m]

    # 将 base 扩展到二维形状 [m, n]
    q = base[:, np.newaxis]  # 增加一个新维度，形状变为 [m, 1]
    q = np.broadcast_to(q, q_shape)  # 广播到目标形状 [m, n]

    # 转换为 float16 类型
    q = q.astype(np.float16)
    return q


def softmax(x, is_fp16=False):
    # 使用 torch 的 softmax 实现
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


def quant_fp8e4m3_per_token(x: torch.Tensor):
    # perblock
    x_fp32 = x.to(torch.float32)
    max_value = torch.amax(torch.abs(x_fp32), dim=-1, keepdim=True)
    scale_quant = 448.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_fp32 = y_fp32.view(x.shape)
    y_fp8e4m3 = y_fp32.to(torch.float8_e4m3fn)
    scale_dequant = 1.0 / scale_quant
    # shape是(b, s, n, d) fp8e4m3, (b, s, n, 1) fp32
    return y_fp8e4m3, scale_dequant


def quant_fp8e4m3_per_token_key(x: torch.Tensor):
    # perblock
    x_fp32 = x.to(torch.float32)
    max_value = torch.amax(torch.abs(x_fp32), dim=-1, keepdim=True)
    scale_quant = 448.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_fp32 = y_fp32.view(x.shape)
    y_fp8e4m3 = y_fp32.to(torch.float8_e4m3fn)
    scale_dequant = 1.0 / scale_quant
    # shape是(b, s, n, d) fp8e4m3, (b, s, n, 1) fp32
    return y_fp8e4m3, scale_dequant


def quant_fp8e4m3_per_channel_value(x: torch.Tensor):
    # perblock
    x_fp32 = x.to(torch.float32)
    max_value = torch.amax(torch.abs(x_fp32), dim=1, keepdim=True)
    scale_quant = 448.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_fp32 = y_fp32.view(x.shape)
    y_fp8e4m3 = y_fp32.to(torch.float8_e4m3fn)
    scale_dequant = 1.0 / scale_quant
    # shape是(b, s, n, d) fp8e4m3, (b, 1, n, d) fp32
    return y_fp8e4m3, scale_dequant


def fp8_bsnd_to_pa_format(tensor_bsnd, block_table, actual_seq, block_size, device):
    """
    转换为最终 PA 格式：shape [total_blocks, 128, n, d]
    每个全局 block 存储 128 个 token 位置，有效 token 填充，无效位置补 0
    """
    b, s, n, d = tensor_bsnd.shape
    num_blocks_per_batch = s // block_size
    total_blocks = num_blocks_per_batch * b

    # 2. 初始化 PA 张量（核心：shape [total_blocks, 128, n, d]，全 0 填充）
    pa_tensor = torch.zeros((total_blocks, block_size, n, d), dtype=tensor_bsnd.dtype, device="cpu")

    # 3. 逐 Batch + 逐逻辑 Block 填充
    for batch_idx in range(b):
        curr_actual_seq = actual_seq[batch_idx].item()
        if curr_actual_seq <= 0:
            continue
        # 截断有效序列长度到总长度以内
        curr_actual_seq = min(curr_actual_seq, s)

        # 当前 batch 的原始数据和 Block Table
        curr_tokens = tensor_bsnd[batch_idx]  # [s, n, d]
        curr_global_block_ids = block_table[batch_idx]  # [num_blocks_per_batch]：每个逻辑 block 对应的全局 ID

        # 按 128 切分逻辑 block，逐个处理
        for logical_block_idx in range(num_blocks_per_batch):
            # 步骤 1：获取当前逻辑 block 对应的全局 PA block ID（唯一）
            global_pa_block_id = curr_global_block_ids[logical_block_idx].item()
            # 校验全局 ID 合法性
            if global_pa_block_id < 0 or global_pa_block_id >= total_blocks:
                raise ValueError(f"全局 Block ID {global_pa_block_id} 超出范围 [0, {total_blocks-1}]")

            # 步骤 2：计算当前逻辑 block 的 token 范围
            token_start = logical_block_idx * block_size  # 逻辑 block 起始 token
            token_end = min((logical_block_idx + 1) * block_size, curr_actual_seq)  # 结束 token（不超过有效长度）
            # 无有效 token，跳过
            if token_start >= token_end:
                continue

            # 步骤 3：填充当前逻辑 block 的有效 token 到全局 PA block 中
            # token_offset：token 在 block 内的偏移（0~127）
            for token_in_block_offset in range(token_end - token_start):
                src_token_idx = token_start + token_in_block_offset  # 原始张量的 token 索引
                # 填充：PA[全局block_id, 块内偏移, :, :] = 原始token数据
                pa_tensor[global_pa_block_id, token_in_block_offset] = curr_tokens[src_token_idx]
    return pa_tensor.to(device)


def _prepare_kv_data_for_pfa(atten_cfg, device, block_table_shape):
    """Prepare Q, K, V tensors for page flash attention."""
    torch_dtype = torch.bfloat16
    b = atten_cfg.b
    s1 = atten_cfg.s1
    d = atten_cfg.qd
    nq = atten_cfg.nq
    nkv = atten_cfg.nkv
    block_size = atten_cfg.block_size
    kv_actual_seq = atten_cfg.actual_seq

    q_shape = [b * s1, nq, d]
    kv_shape_1 = [atten_cfg.kv_num_blocks * block_size * nkv * d]
    kv_shape = [atten_cfg.kv_num_blocks, block_size, nkv, d]

    q = torch.empty(q_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    q_fp8_e4m3, q_scale = quant_fp8e4m3_per_token(q)
    k1 = torch.empty(kv_shape_1, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    k = k1.reshape(kv_shape)
    k_fp8_e4m3, k_scale = quant_fp8e4m3_per_token_key(k)
    v1 = torch.empty(kv_shape_1, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    v = v1.reshape(kv_shape)

    block_table = gen_block_table(kv_actual_seq, block_size, block_table_shape)
    k_cache_bsnd, v_cache_bsnd, k_sclae_bsnd = kv_cache_concat_bsnd(k_fp8_e4m3, v, k_scale, block_table, atten_cfg)
    v_fp8_e4m3_bsnd, v_scale = quant_fp8e4m3_per_channel_value(v_cache_bsnd)
    v_fp8_e4m3 = fp8_bsnd_to_pa_format(v_fp8_e4m3_bsnd, block_table, kv_actual_seq, block_size, device)
    v_scale = v_scale.reshape(b * 1, nkv, d)

    block_table_torch = block_table.to(dtype=torch.int32, device=device)
    act_seq_torch = kv_actual_seq.to(dtype=torch.int32, device=device)
    return q_fp8_e4m3, q_scale, k_fp8_e4m3, k_scale, v_fp8_e4m3, v_scale, block_table_torch, act_seq_torch


def pfa(atten_cfg, tile_config):
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch_dtype = torch.bfloat16
    torch.npu.set_device(int(device_id))
    block_size = atten_cfg.block_size
    max_num_blocks_per_query = atten_cfg.max_num_blocks_per_query
    block_table_shape = [atten_cfg.block_table_batch, max_num_blocks_per_query]
    q_shape = [atten_cfg.b * atten_cfg.s1, atten_cfg.nq, atten_cfg.qd]
    device = f'npu:{device_id}'

    q_fp8_e4m3, q_scale, k_fp8_e4m3, k_scale, v_fp8_e4m3, v_scale, block_table_torch, act_seq_torch = \
        _prepare_kv_data_for_pfa(atten_cfg, device, block_table_shape)

    attention_output = torch.zeros(q_shape, dtype=torch_dtype).to(device=device)
    out_torch = torch.zeros(q_shape, dtype=torch_dtype).to(device=device)

    pfa_flash_torch(PfaFlashTorchInputs(
        q=q_fp8_e4m3, q_scale=q_scale, k=k_fp8_e4m3, k_sclae_bsnd=k_scale, v=v_fp8_e4m3,
        v_scale=v_scale, block_table=block_table_torch, kv_act_seqs=act_seq_torch,
        out=attention_output, atten_cfg=atten_cfg, tile_config=tile_config))

    attention(AttentionInputs(
        query=q_fp8_e4m3, query_scale=q_scale, key_cache=k_fp8_e4m3,
        key_cache_scale=k_scale, value_cache=v_fp8_e4m3,
        value_cache_sclae=v_scale, block_tables=block_table_torch,
        actual_seqs=act_seq_torch, attn_res=out_torch,
        softmax_scale=atten_cfg.softmax_scale, tile_config=tile_config))

    assert_allclose(np.array(attention_output.cpu().flatten().tolist()),
                    np.array(out_torch.cpu().flatten().tolist()),
                    rtol=0.0078125, atol=0.0001)


def matmul_proxy(left, right):
    torch_fp32 = torch.float32
    return torch.matmul(left.to(torch_fp32), right.to(torch_fp32))


@dataclass
class _AssembleKvForS2TileInputs:
    k_2d: Any
    k_scale_2d: Any
    v_2d: Any
    block_table: Any
    b_idx: Any
    idx: Any
    actual_block_num: Any
    s2_tile: Any
    block_size: Any
    n2_idx: Any
    d: Any


def _assemble_kv_for_s2_tile(inputs: _AssembleKvForS2TileInputs):
    """Assemble K, K-scale, and V tensors for an S2 tile in the attention loop."""
    kj_assemble = torch.zeros((inputs.s2_tile, inputs.d), dtype=inputs.k_2d.dtype, device=inputs.k_2d.device)
    kj_scale_assemble = torch.zeros((inputs.s2_tile, 1), dtype=inputs.k_scale_2d.dtype, device=inputs.k_scale_2d.device)
    vj_assemble = torch.zeros((inputs.s2_tile, inputs.d), dtype=inputs.v_2d.dtype, device=inputs.v_2d.device)
    for i in range(inputs.actual_block_num):
        block_idx = inputs.block_table[inputs.b_idx, inputs.idx + i].item()
        block_idx_valid = max(block_idx, 0)
        kj_assemble[i * inputs.block_size:(i + 1) * inputs.block_size, :] = \
            inputs.k_2d[block_idx_valid * inputs.block_size:(block_idx_valid + 1) * inputs.block_size,
                 inputs.n2_idx * inputs.d:(inputs.n2_idx + 1) * inputs.d]
        kj_scale_assemble[i * inputs.block_size:(i + 1) * inputs.block_size, :] = \
            inputs.k_scale_2d[block_idx_valid * inputs.block_size:(block_idx_valid + 1) * inputs.block_size,
                       inputs.n2_idx * 1:(inputs.n2_idx + 1) * 1]
        vj_assemble[i * inputs.block_size:(i + 1) * inputs.block_size, :] = \
            inputs.v_2d[block_idx_valid * inputs.block_size:(block_idx_valid + 1) * inputs.block_size,
                 inputs.n2_idx * inputs.d:(inputs.n2_idx + 1) * inputs.d]
    return kj_assemble, kj_scale_assemble, vj_assemble


@dataclass
class _ProcessS2TileInputs:
    qi: Any
    qi_scale: Any
    kj_assemble: Any
    kj_scale_assemble: Any
    vj_assemble: Any
    v_scale_2d: Any
    b_idx: Any
    n2_idx: Any
    d: Any
    softmax_scale: Any
    actual_s2_tile: Any


def _process_s2_tile(inputs: _ProcessS2TileInputs):
    """Process a single S2 tile: matmul -> softmax -> matmul."""
    torch_fp32 = torch.float32
    mm1_quant = matmul_proxy(inputs.qi, inputs.kj_assemble.t()).to(torch_fp32)
    mm1_fp32 = mm1_quant * inputs.qi_scale * inputs.kj_scale_assemble.t()
    sij_scale = mm1_fp32 * inputs.softmax_scale
    tilda_mij, _ = torch.max(sij_scale, dim=-1, keepdim=True)
    tsub = sij_scale - tilda_mij
    vec1_res = torch.exp(tsub)
    sum_local = torch.sum(vec1_res, dim=-1, keepdim=True)
    tilda_pij_f8, tilda_pij_scale = quant_fp8e4m3_per_token(vec1_res)
    v_scale_bs = inputs.v_scale_2d[
        inputs.b_idx, inputs.n2_idx * inputs.d:(inputs.n2_idx + 1) * inputs.d
    ].reshape(1, inputs.d)
    mm2_quant = matmul_proxy(tilda_pij_f8, inputs.vj_assemble).to(torch_fp32)
    mm2_res = mm2_quant * tilda_pij_scale * v_scale_bs
    return mm2_res, sum_local, tilda_mij


@dataclass
class _FlashUpdateStateInputs:
    oi_upd: Any
    sum_upd: Any
    max_upd: Any
    mm2_res: Any
    sum_local: Any
    tilda_mij: Any
    s2_idx: Any
    s2_loop: Any
    dtype_out: Any
    out: Any
    bs_ofs: Any
    n1g_ofs: Any
    g_tile: Any


def _flash_update_state(inputs: _FlashUpdateStateInputs):
    """Update flash attention running state for current S2 tile."""
    oi_upd = inputs.oi_upd
    sum_upd = inputs.sum_upd
    max_upd = inputs.max_upd
    if inputs.s2_idx == 0:
        oi_tmp = inputs.mm2_res
        if inputs.s2_idx == inputs.s2_loop - 1:
            oi_upd = oi_tmp / inputs.sum_local
            inputs.out[
                inputs.bs_ofs:inputs.bs_ofs + 1,
                inputs.n1g_ofs:inputs.n1g_ofs + inputs.g_tile, :
            ] = oi_upd.unsqueeze(0).to(inputs.dtype_out)
        else:
            oi_upd = oi_tmp.clone()
            sum_upd = inputs.sum_local.clone()
            max_upd = inputs.tilda_mij.clone()
        oi_tmp_out = oi_tmp
    else:
        max_new, _ = torch.max(torch.cat([max_upd, inputs.tilda_mij], dim=-1), dim=-1, keepdim=True)
        t2 = torch.exp(max_upd - max_new)
        t4 = torch.exp(inputs.tilda_mij - max_new)
        sum_upd = (t4 * inputs.sum_local + t2 * sum_upd).clone()
        max_upd = max_new.clone()
        oi_tmp = oi_upd * t2 + inputs.mm2_res * t4
        if inputs.s2_idx == inputs.s2_loop - 1:
            oi_upd = oi_tmp / sum_upd
            inputs.out[
                inputs.bs_ofs:inputs.bs_ofs + 1,
                inputs.n1g_ofs:inputs.n1g_ofs + inputs.g_tile, :
            ] = oi_upd.unsqueeze(0).to(inputs.dtype_out)
        else:
            oi_upd = oi_tmp.clone()
        oi_tmp_out = oi_tmp
    return oi_tmp_out, oi_upd, sum_upd, max_upd


def _init_pfa_2d_tensors(q, q_scale, k, k_sclae_bsnd, v, v_scale, kv_act_seqs):
    """Initialize 2D tensors and compute shapes for pfa_flash_torch."""
    q_shape = q.shape
    bs1, n1, d = q_shape[0], q_shape[1], q_shape[2]
    b = kv_act_seqs.shape[0]
    s1 = bs1 // b
    k_shape = k.shape
    block_num, block_size, n2, _ = k_shape
    g = n1 // n2
    softmax_scale = d ** -0.5
    k_2d = k.reshape((block_num * block_size, n2 * d))
    k_scale_2d = k_sclae_bsnd.reshape((block_num * block_size, n2 * 1))
    v_2d = v.reshape((block_num * block_size, n2 * d))
    q_2d = q.reshape((b * s1 * n1, d))
    q_scale_2d = q_scale.reshape((b * s1 * n1, 1))
    v_scale_2d = v_scale.reshape((b * 1, n2 * d))
    return (b, s1, n1, n2, d, g, block_size, softmax_scale,
            k_2d, k_scale_2d, v_2d, q_2d, q_scale_2d, v_scale_2d)


@dataclass
class PfaFlashTorchInputs:
    q: Any
    q_scale: Any
    k: Any
    k_sclae_bsnd: Any
    v: Any
    v_scale: Any
    block_table: Any
    kv_act_seqs: Any
    out: Any
    atten_cfg: Any
    tile_config: Any


def pfa_flash_torch(inputs: PfaFlashTorchInputs):
    """PyTorch FP8 quant flash attention golden: q[b*s1,n1,d], k/v[block_num,bs,n2,d], out[b*s1,n1,d]."""
    torch_fp32 = torch.float32
    (b, s1, n1, n2, d, g, block_size, softmax_scale,
     k_2d, k_scale_2d, v_2d, q_2d, q_scale_2d, v_scale_2d) = (
        _init_pfa_2d_tensors(inputs.q, inputs.q_scale, inputs.k,
                             inputs.k_sclae_bsnd, inputs.v,
                             inputs.v_scale, inputs.kv_act_seqs))
    g_tile = g
    s2_tile = inputs.tile_config.s2_tile
    device = inputs.q.device
    dtype_out = inputs.out.dtype
    block_num_per_tile = s2_tile // block_size

    for b_idx in range(b):
        for s1_idx in range(s1):
            cur_seq = inputs.kv_act_seqs[b_idx] - (s1 - 1 - s1_idx)
            cur_seq = max(cur_seq.item(), 0)
            s2_loop = (cur_seq + s2_tile - 1) // s2_tile
            for n2_idx in range(n2):
                for g_idx in range(g // g_tile):
                    oi_upd = torch.zeros((g_tile, d), device=device, dtype=torch_fp32)
                    sum_upd = torch.zeros((g_tile, 1), device=device, dtype=torch_fp32)
                    max_upd = torch.zeros((g_tile, 1), device=device, dtype=torch_fp32)
                    for s2_idx in range(s2_loop):
                        idx = s2_idx * block_num_per_tile
                        bs_ofs = b_idx * s1 + s1_idx
                        n1g_ofs = n2_idx * g + g_idx * g_tile
                        actual_s2_tile = min(cur_seq - s2_idx * s2_tile, s2_tile)
                        qi_start = bs_ofs * n1 + n1g_ofs
                        qi = q_2d[qi_start:qi_start + g_tile, :]
                        qi_scale = q_scale_2d[qi_start:qi_start + g_tile, :]

                        actual_block_num = (actual_s2_tile + block_size - 1) // block_size
                        kj_assemble, kj_scale_assemble, vj_assemble = _assemble_kv_for_s2_tile(
                            _AssembleKvForS2TileInputs(
                                k_2d=k_2d, k_scale_2d=k_scale_2d, v_2d=v_2d,
                                block_table=inputs.block_table, b_idx=b_idx, idx=idx,
                                actual_block_num=actual_block_num, s2_tile=s2_tile,
                                block_size=block_size, n2_idx=n2_idx, d=d))
                        kj_assemble = kj_assemble[:actual_s2_tile, :]
                        kj_scale_assemble = kj_scale_assemble[:actual_s2_tile, :]
                        vj_assemble = vj_assemble[:actual_s2_tile, :]

                        mm2_res, sum_local, tilda_mij = _process_s2_tile(_ProcessS2TileInputs(
                            qi=qi, qi_scale=qi_scale, kj_assemble=kj_assemble,
                            kj_scale_assemble=kj_scale_assemble, vj_assemble=vj_assemble,
                            v_scale_2d=v_scale_2d, b_idx=b_idx, n2_idx=n2_idx, d=d,
                            softmax_scale=softmax_scale, actual_s2_tile=actual_s2_tile))

                        _, oi_upd, sum_upd, max_upd = _flash_update_state(
                            _FlashUpdateStateInputs(
                                oi_upd=oi_upd, sum_upd=sum_upd, max_upd=max_upd,
                                mm2_res=mm2_res, sum_local=sum_local, tilda_mij=tilda_mij,
                                s2_idx=s2_idx, s2_loop=s2_loop, dtype_out=dtype_out,
                                out=inputs.out, bs_ofs=bs_ofs, n1g_ofs=n1g_ofs, g_tile=g_tile))
    return inputs.out


def pfa_test_impl(case_name):
    case_config = get_case_config(case_name)
    atten_cfg, tile_config = build_pfa_config(case_config)

    check_cond(atten_cfg.b == len(atten_cfg.actual_seq), \
               f'{atten_cfg.b} {atten_cfg.actual_seq} B的大小必须和actual_seq长度相等')

    if atten_cfg.actual_seq.device.type != 'cpu':
        actual_seq_cpu = atten_cfg.actual_seq.cpu()
    else:
        actual_seq_cpu = atten_cfg.actual_seq

    check_cond(all(x <= atten_cfg.s2 for x in actual_seq_cpu), "所有值都必须小于s2")
    pfa(atten_cfg, tile_config)


@pytest.mark.soc("950")
def test_pfa_for_950():
    case_names = [
        "pfa_fp8_b1_s1_2_s2_2048_nkv_8",
        "pfa_fp8_b16_s1_1_s2_8195_nkv_2",
        "pfa_fp8_b16_s1_1_s2_8k_nkv_1",
        "pfa_fp8_b16_s1_1_s2_8k_nkv_2",
        "pfa_fp8_b2_s1_1_s2_1k",
        "pfa_fp8_b16_s1_1_s2_256_nkv_4",
    ]
    for case_name in case_names:
        case_config = get_case_config(case_name)
        atten_cfg, tile_config = build_pfa_config(case_config)

        assert atten_cfg.b == len(
            atten_cfg.actual_seq), f'{atten_cfg.b} {atten_cfg.actual_seq} B的大小必须和actual_seq长度相等'

        if atten_cfg.actual_seq.device.type != 'cpu':
            actual_seq_cpu = atten_cfg.actual_seq.cpu()
        else:
            actual_seq_cpu = atten_cfg.actual_seq
        assert all(x <= atten_cfg.s2 for x in actual_seq_cpu), "所有值都必须小于s2"
        pfa(atten_cfg, tile_config)


@dataclass
class AttentionInputs:
    query: Any
    query_scale: Any
    key_cache: Any
    key_cache_scale: Any
    value_cache: Any
    value_cache_sclae: Any
    block_tables: Any
    actual_seqs: Any
    attn_res: Any
    softmax_scale: Any
    tile_config: Any


@allow_in_graph
def attention(inputs: AttentionInputs):
    """
    Main attention function with Attention support.

    This function implements scaled dot-product attention using Attention
    mechanism, which efficiently handles variable-length sequences and dynamic
    batch sizes by managing KV cache in non-contiguous blocks.

    Args:
        query: Query tensor with shape [num_tokens, num_head, head_size]
        key_cache: Key cache tensor with shape [num_blocks, block_size, kv_head_num, head_size]
        value_cache: Value cache tensor with shape [num_blocks, block_size, kv_head_num, head_size]
        block_tables: Block mapping table with shape [batch_size, max_num_blocks_per_query]
        actual_seqs: Actual sequence lengths with shape [batch_size]
        attn_res: Output attention tensor with shape [num_tokens, num_head, head_size]
        softmax_scale: Scaling factor for attention scores
        tile_config: PfaTileShapeConfig object containing tiling parameters

    Note:
        This function is decorated with @allow_in_graph to enable integration
        with PyTorch's compilation graph.
    """
    if isinstance(inputs.query, FakeTensor):
        return
    check_args(
        inputs.query,
        inputs.key_cache,
        inputs.value_cache,
        inputs.block_tables,
        inputs.actual_seqs,
        inputs.attn_res
    )

    kernel_inputs = [inputs.query, inputs.query_scale, inputs.key_cache, inputs.key_cache_scale,
                     inputs.value_cache, inputs.value_cache_sclae, inputs.block_tables,
                     inputs.actual_seqs, inputs.attn_res]
    for _ in range(1):
        pfa_func_kernel_v2_bound(*kernel_inputs, inputs.softmax_scale, inputs.tile_config)


if __name__ == "__main__":
    if pypto.platform.npuarch == 'DAV_3510':
        test_pfa_for_950()