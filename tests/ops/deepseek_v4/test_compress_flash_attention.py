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
deepseekv4 Attention Module

This module implements the Attention mechanism for deepseekv4 model, which uses
a paged memory management approach similar to operating systems to efficiently
handle variable-length sequences and dynamic batch sizes in cfa_attention computation.

Main Functions:
    - cfa_attention: Main cfa_attention function with Attention support
    - ifa_flash: JIT compiled kernel implementing Flash Attention with paged KV cache
    - gen_block_table: Generate block mapping table for Attention
    - kv_cache_concat_bsnd: Convert paged KV cache to BSND format
"""

import os
import math
from dataclasses import dataclass

from typing import Any
import torch
import torch_npu
import pytest

import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

import numpy as np
import pypto

from deepseek_v4.compress_flash_attention_impl import cfa_attention, cfa_graph

np.random.seed(0)
torch.manual_seed(0)
np.set_printoptions(formatter={'float': '{:.6f}'.format})


@dataclass
class AttentionConfig:
    b: int
    s1: int
    s2: int
    n1: int
    n2: int
    q_d: int
    kv_d: int
    block_size: int = 128
    cmp_ratio: int = 128
    max_blocks: int = 0
    actual_seq: torch.Tensor = None  # 改为 torch.Tensor 类型
    block_table_batch: int = 0
    kv_num_blocks: int = 0


def gen_block_table(actual_seq_len, block_size, block_table_shape, cmp_ratio=128, enable_win=False):
    block_num_per_batch = []
    block_num = 0

    if enable_win:
        cmp_ratio = 1
    # 处理 torch tensor 类型的 actual_seq_len
    for actual_seq in actual_seq_len:
        block_num_per_batch.append(math.ceil(actual_seq.item() // cmp_ratio / block_size))
        block_num += math.ceil(actual_seq.item() / block_size)

    # 使用 torch 替换 numpy
    block_idx_list = torch.arange(0, block_num, dtype=torch.int32)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))]  # 随机排列

    # 创建 cmp_block_table 张量
    cmp_block_table = torch.full(
        block_table_shape, -1, dtype=torch.int32, device=actual_seq_len.device
    )
    block_idx = 0
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            cmp_block_table[block_table_batch_idx][j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_batch_idx += 1
    return cmp_block_table


def get_decode_case(device="cpu"):
    b = 64
    s1 = 2
    s2 = 8 * 1024
    q_d = 512
    nq = 64
    nkv = 1
    block_table_batch = b
    block_size = 128
    cmp_ratio = 128
    kv_num_blocks = b * ((s2 + block_size - 1) // block_size)
    actual_seq_values = [s2] * b
    actual_seq_tensor = torch.tensor(actual_seq_values, dtype=torch.int32, device=device)
    attn_cfg = AttentionConfig(b=b, s1=s1, s2=s2, n1=nq, n2=nkv,
                               q_d=q_d, kv_d=q_d, block_size=block_size, block_table_batch=block_table_batch,
                               kv_num_blocks=kv_num_blocks, actual_seq=actual_seq_tensor, cmp_ratio=cmp_ratio)
    attn_cfg.max_blocks = (s2 + block_size - 1) // block_size
    return attn_cfg


class MM(torch.nn.Module):
    def forward(
            self,
            q: torch.Tensor,
            cmp_kv: torch.Tensor,
            sinks: torch.Tensor,
            cmp_block_table: torch.Tensor,
            seqused_kv: torch.Tensor,
            ori_kv: torch.Tensor,
            ori_block_table: torch.Tensor,
            cmp_ratio: int = 1,
    ):
        return cfa_graph(q, cmp_kv, sinks, cmp_block_table, seqused_kv, ori_kv, \
                                            ori_block_table, cmp_ratio)


def softmax(x, sinks, is_fp16=False, is_new_sink=False):
    # 使用 torch 的 softmax 实现
    if is_fp16:
        original_dtype = x.dtype
        x = x.float()
    x_max = x.max(dim=-1, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)
    if sinks is not None:
        if not is_new_sink:
            x_sum += sinks.unsqueeze(-1)
        else:
            x_sum += torch.exp(sinks.unsqueeze(-1) - x_max)
    ans = y / x_sum
    if is_fp16:
        ans = ans.to(original_dtype)
        x_max = x_max.to(original_dtype)
        x_sum = x_sum.to(original_dtype)

    return ans, x_max, x_sum


def matmul_proxy(left, right):
    fp32 = torch.float32
    return torch.matmul(left.to(fp32), right.to(fp32)).to(left.dtype)


def get_block_kv(kv_2d, cmp_block_table, b_idx, s2_idx, block_size, cur_seq):
    block_idx = cmp_block_table[b_idx][s2_idx]
    block_idx_valid = max(block_idx, 0)
    actual_s2_tile = min(block_size, cur_seq - s2_idx * block_size)
    kj_start = block_idx_valid * block_size
    kj_end = kj_start + actual_s2_tile
    kvj = kv_2d[kj_start:kj_end, :]
    return kvj


def flash_end(out, sinks, li_upd, mi_upd, oi_upd, n2g_ofs, g_tile, bs_ofs, dtype, is_new_sink=False):
    li = li_upd.unsqueeze(-1)
    if sinks is not None:
        if not is_new_sink:
            li += sinks.unsqueeze(-1)
        else:
            li += torch.exp(sinks - mi_upd).unsqueeze(-1)
    oi_final = oi_upd / li
    oi_upd_3d = oi_final.unsqueeze(0)
    attn_out_start = n2g_ofs
    attn_out_end = n2g_ofs + g_tile
    if attn_out_end > out.shape[1]:
        attn_out_end = out.shape[1]
        attn_out_start = attn_out_end - g_tile
    out[bs_ofs: bs_ofs + 1, attn_out_start:attn_out_end, :] = (
        oi_upd_3d.to(dtype)
    )


def kv_cache_concat_bsnd(kv_cache_out, cmp_block_table, actual_seqs):
    b = actual_seqs.shape[0]
    n2 = kv_cache_out.shape[2]
    d = kv_cache_out.shape[3]
    block_size = kv_cache_out.shape[1]
    dtype = kv_cache_out.dtype

    # 处理 torch tensor 类型的 kv_cache_actual_seq
    kv_max = (torch.max(actual_seqs).item() + block_size - 1) // block_size * block_size

    # 使用 torch 创建张量，保持在同一设备上
    cmp_kv = torch.zeros([b, kv_max, n2, d], dtype=dtype).to(kv_cache_out.device)

    for b_idx in range(b):
        block_list = cmp_block_table[b_idx]
        kv_nope_temp_tensor = torch.zeros([1, kv_max, n2, d], dtype=dtype)
        s_idx = 0

        for _, block_idx in enumerate(block_list):
            if block_idx == -1:
                break
            # 使用 torch 的切片操作
            start_idx = s_idx * block_size
            end_idx = (s_idx + 1) * block_size

            kv_nope_temp_tensor[:, start_idx:end_idx, :, :] = kv_cache_out[
                block_idx: block_idx + 1, :, :, :
            ]
            s_idx += 1

        cmp_kv[b_idx: b_idx + 1, :, :, :] = kv_nope_temp_tensor

    return cmp_kv


@dataclass
class _FlashOriWindowInputs:
    qi: Any
    kv_win_2d: Any
    ori_block_table: Any
    b_idx: Any
    s1: Any
    s1_idx: Any
    win: Any
    block_size: Any
    seqused_kv: Any
    dtype: Any
    scale: Any
    d: Any


def _flash_ori_window(inputs: _FlashOriWindowInputs):
    """Compute attention for the original window (ori_kv) portion in ifa_flash_torch."""
    fp32 = torch.float32
    valid_len = inputs.seqused_kv[inputs.b_idx] - (inputs.s1 - inputs.s1_idx - 1)
    valid_win_len = min(valid_len, inputs.win)
    valid_start_pos = valid_len - valid_win_len
    valid_end_pos = valid_len - 1
    start_offset = valid_start_pos % inputs.block_size

    start_block = valid_start_pos // inputs.block_size
    end_block = valid_end_pos // inputs.block_size
    kv_list = []

    for block_idx in range(start_block, end_block + 1):
        block_idx_valid = max(inputs.ori_block_table[inputs.b_idx, block_idx], 0)
        block_offset = block_idx_valid * inputs.block_size
        kv_block = inputs.kv_win_2d[block_offset: block_offset + inputs.block_size, :]
        kv_list.append(kv_block)

    kv_cur = torch.cat(kv_list, axis=0)
    kv_cur = kv_cur[start_offset: start_offset + valid_win_len, :]

    mm1 = matmul_proxy(inputs.qi, kv_cur.t())
    muls_res = mm1 * inputs.scale
    tilda_mij, _ = torch.max(muls_res, dim=-1, keepdim=True)
    tsub = muls_res - tilda_mij
    tilda_pij = torch.exp(tsub)
    tilda_lij = torch.sum(tilda_pij, dim=-1, keepdim=True)
    oi_tmp = matmul_proxy(tilda_pij.to(inputs.dtype), kv_cur)
    oi_upd = oi_tmp
    li_upd = tilda_lij.squeeze(-1)
    mi_upd = tilda_mij.squeeze(-1)
    return oi_upd, li_upd, mi_upd


def _flash_first_kv(qi, kvj, scale, dtype):
    """First KV tile: initialize accumulators."""
    mm1 = matmul_proxy(qi, kvj.t())
    muls_res = mm1 * scale
    tilda_mij, _ = torch.max(muls_res, dim=-1, keepdim=True)
    tsub = muls_res - tilda_mij
    tilda_pij = torch.exp(tsub)
    tilda_lij = torch.sum(tilda_pij, dim=-1, keepdim=True)
    oi_tmp = matmul_proxy(tilda_pij.to(dtype), kvj)
    return oi_tmp, tilda_lij.squeeze(-1), tilda_mij.squeeze(-1)


def _flash_subsequent_kv(qi, kvj, oi_upd, li_upd, mi_upd, scale, dtype):
    """Subsequent KV tile: online softmax update."""
    mi = mi_upd.unsqueeze(-1)
    mm1 = matmul_proxy(qi, kvj.t())
    muls_res = mm1 * scale
    tilda_mij, _ = torch.max(muls_res, dim=-1, keepdim=True)
    max_new, _ = torch.max(torch.cat([mi, tilda_mij], dim=-1), dim=-1, keepdim=True)
    tsub = muls_res - max_new
    tilda_pij = torch.exp(tsub)
    tilda_lij = torch.sum(tilda_pij, dim=-1, keepdim=True)
    tsub2 = torch.sub(mi, max_new)
    mi_upd = max_new.squeeze(-1)
    update_mul = torch.exp(tsub2)
    li = li_upd.unsqueeze(-1)
    sum_new = li * update_mul + tilda_lij
    li_upd = sum_new.squeeze(-1)
    q1 = matmul_proxy(tilda_pij.to(dtype), kvj)
    oi_upd = oi_upd * update_mul + q1
    return oi_upd, li_upd, mi_upd


@dataclass
class _ProcessGroupBlockInputs:
    b_idx: Any
    s1_idx: Any
    g_idx: Any
    g_tile: Any
    d: Any
    device: Any
    fp32: Any
    n1: Any
    q_2d: Any
    kv_2d: Any
    cmp_block_table: Any
    block_size: Any
    seqused_kv: Any
    s1: Any
    s2_loop: Any
    cur_seq: Any
    scale: Any
    dtype: Any
    win: Any
    ori_kv: Any
    ori_block_table: Any
    sinks: Any
    output_flash: Any
    is_new_sink: Any


def _process_group_block(inputs: _ProcessGroupBlockInputs):
    """Process one group-block: init accumulators, run ori window + s2 loop."""
    oi_upd = torch.zeros((inputs.g_tile, inputs.d), device=inputs.device, dtype=inputs.fp32)
    li_upd = torch.zeros(inputs.g_tile, device=inputs.device, dtype=inputs.fp32)
    mi_upd = torch.zeros(inputs.g_tile, device=inputs.device, dtype=inputs.fp32)
    bs_ofs = inputs.b_idx * inputs.s1 + inputs.s1_idx
    n2g_ofs = inputs.g_idx * inputs.g_tile
    qi_start = bs_ofs * inputs.n1 + n2g_ofs
    qi_end = qi_start + inputs.g_tile
    qi = inputs.q_2d[qi_start:qi_end, :]
    if inputs.ori_kv is not None and inputs.ori_block_table is not None:
        kv_win_2d = inputs.ori_kv.reshape(-1, inputs.d)
        oi_upd, li_upd, mi_upd = _flash_ori_window(
            _FlashOriWindowInputs(
            qi=qi, kv_win_2d=kv_win_2d, ori_block_table=inputs.ori_block_table,
            b_idx=inputs.b_idx, s1=inputs.s1, s1_idx=inputs.s1_idx, win=inputs.win,
            block_size=inputs.block_size, seqused_kv=inputs.seqused_kv,
            dtype=inputs.dtype, scale=inputs.scale, d=inputs.d))
        if inputs.s2_loop == 0:
            flash_end(inputs.output_flash, inputs.sinks, li_upd, mi_upd,
                    oi_upd, n2g_ofs, inputs.g_tile, bs_ofs, inputs.dtype,
                    is_new_sink=inputs.is_new_sink)
    for s2_idx in range(inputs.s2_loop):
        kvj = get_block_kv(
            inputs.kv_2d, inputs.cmp_block_table, inputs.b_idx, s2_idx,
            inputs.block_size, inputs.cur_seq)
        if s2_idx == 0 and inputs.ori_kv is None:
            oi_upd, li_upd, mi_upd = _flash_first_kv(qi, kvj, inputs.scale, inputs.dtype)
        else:
            oi_upd, li_upd, mi_upd = _flash_subsequent_kv(qi, kvj, oi_upd, li_upd, mi_upd, inputs.scale, inputs.dtype)
        if s2_idx == inputs.s2_loop - 1:
            flash_end(inputs.output_flash, inputs.sinks, li_upd, mi_upd,
                    oi_upd, n2g_ofs, inputs.g_tile, bs_ofs, inputs.dtype,
                    is_new_sink=inputs.is_new_sink)


@dataclass
class IfaFlashTorchInputs:
    q: Any
    cmp_kv: Any
    sinks: Any
    cmp_block_table: Any
    seqused_kv: Any
    output_flash: Any
    tmp_out: Any
    cmp_ratio: Any = 128
    is_new_sink: Any = False
    ori_kv: Any = None
    ori_block_table: Any = None


def ifa_flash_torch(inputs: IfaFlashTorchInputs):
    """
    Args:
        inputs.q: Query [batch_size * s1, num_head, head_size]
        k: Key cache [num_blocks, block_size, kv_head_num, head_size]
        v: Value cache [num_blocks, block_size, kv_head_num, head_size]
         inputs.cmp_block_table: Block mapping for compress cache
            [batch_size, max_num_blocks_per_query]
        start_pos: Actual start position [batch_size], satisify start_pos + s1 = original actual seq
        out: Output [batch_size * s1, num_head, head_size]
    """
    fp32 = torch.float32
    device = inputs.q.device
    dtype = inputs.q.dtype
    q_shape = inputs.q.shape
    bs1, n1, d = q_shape[0], q_shape[1], q_shape[2]
    b = inputs.seqused_kv.shape[0]
    s1 = bs1 // b
    k_shape = inputs.cmp_kv.shape
    _, block_size, n2, _ = k_shape
    g = n1 // n2
    g_tile = g
    kv_2d = inputs.cmp_kv.reshape(-1, d)
    q_2d = inputs.q.reshape(-1, d)
    scale = d ** -0.5
    win = 128

    for b_idx in range(b):
        for s1_idx in range(s1):
            cur_seq = (inputs.seqused_kv[b_idx] - (s1 - 1 - s1_idx)) // inputs.cmp_ratio
            cur_seq = max(cur_seq, 0)
            s2_loop = math.ceil(cur_seq / block_size)
            for g_idx in range(g // g_tile):
                _process_group_block(
                    _ProcessGroupBlockInputs(
                    b_idx=b_idx, s1_idx=s1_idx, g_idx=g_idx, g_tile=g_tile, d=d,
                    device=device, fp32=fp32, n1=n1, q_2d=q_2d, kv_2d=kv_2d,
                    cmp_block_table=inputs.cmp_block_table, block_size=block_size,
                    seqused_kv=inputs.seqused_kv, s1=s1, s2_loop=s2_loop, cur_seq=cur_seq,
                    scale=scale, dtype=dtype, win=win, ori_kv=inputs.ori_kv,
                    ori_block_table=inputs.ori_block_table, sinks=inputs.sinks,
                    output_flash=inputs.output_flash, is_new_sink=inputs.is_new_sink))
    return output_flash


def _golden_no_flash(q, cmp_kv, sinks, cmp_block_table, seqused_kv, output_flash, cmp_ratio, is_new_sink,
                     ori_kv, ori_block_table):
    """Golden no-flash path: per-batch per-head full attention compute."""
    fp64 = torch.float64
    b = seqused_kv.shape[0]
    bs = q.shape[0]
    s1 = bs // b
    nkv = cmp_kv.shape[2]
    d = cmp_kv.shape[3]
    softmax_scale = d**-0.5
    compress_actual_seqs = seqused_kv // cmp_ratio
    kv_bsnd = kv_cache_concat_bsnd(cmp_kv, cmp_block_table, compress_actual_seqs)
    if ori_kv is not None and ori_block_table is not None:
        k_cfa_bsnd = kv_cache_concat_bsnd(cmp_kv, cmp_block_table, compress_actual_seqs)
        k_win_bsnd = kv_cache_concat_bsnd(ori_kv, ori_block_table, seqused_kv)
        kv_bsnd = torch.cat([k_cfa_bsnd], dim=1)
    for i in range(b):
        for j in range(s1):
            for n2_idx in range(nkv):
                seq_end = seqused_kv[i] - (s1 - 1 - j)
                seq_len = seq_end // cmp_ratio
                q_bs = q[i * s1 + j]
                kv_win_view = k_win_bsnd[i, max(seq_end-128, 0):seq_end, :, :].reshape(-1, d)
                kv_bs = kv_bsnd[i, :seq_len, n2_idx: n2_idx + 1].reshape(seq_len, d)
                kv_bs = torch.cat([kv_win_view, kv_bs], dim=0)
                q_bs = q_bs.to(fp64)
                kv_bs_64 = kv_bs.to(fp64)
                qk_bmm_res = matmul_proxy(q_bs, kv_bs_64.transpose(1, 0))
                qk_ele_res = qk_bmm_res * softmax_scale
                softmax_res, _, _ = softmax(qk_ele_res, sinks, True, is_new_sink=is_new_sink)
                bmm2_res = matmul_proxy(softmax_res.to(output_flash.dtype), kv_bs.to(output_flash.dtype))
                output_flash[i * s1 + j] = bmm2_res.to(output_flash.dtype)
    return output_flash, kv_bs


@dataclass
class IfaGoldenInputs:
    q: Any
    cmp_kv: Any
    sinks: Any
    cmp_block_table: Any
    seqused_kv: Any
    output_flash: Any
    tmp_out: Any
    enable_flash: Any = True
    cmp_ratio: Any = 1
    is_new_sink: Any = True
    ori_kv: Any = None
    ori_block_table: Any = None


def ifa_golden(inputs: IfaGoldenInputs):

    if not inputs.enable_flash:
        return _golden_no_flash(inputs.q, inputs.cmp_kv, inputs.sinks, inputs.cmp_block_table,
                                 inputs.seqused_kv, inputs.output_flash, inputs.cmp_ratio,
                                 inputs.is_new_sink, inputs.ori_kv, inputs.ori_block_table)
    else:
        output_flash = ifa_flash_torch(
            IfaFlashTorchInputs(q=inputs.q, cmp_kv=inputs.cmp_kv, sinks=inputs.sinks,
            cmp_block_table=inputs.cmp_block_table, seqused_kv=inputs.seqused_kv,
            output_flash=inputs.output_flash, tmp_out=inputs.tmp_out,
            cmp_ratio=inputs.cmp_ratio, is_new_sink=inputs.is_new_sink,
            ori_kv=inputs.ori_kv, ori_block_table=inputs.ori_block_table)
        )
        return output_flash


@dataclass
class _C128CreateTensorsOutputs:
    q: Any
    cmp_kv: Any
    sinks: Any
    ori_kv: Any
    ori_block_table: Any
    tmp_out_golden: Any
    output_flash: Any


def _c128_create_tensors(attn_cfg, torch_dtype, device):
    """Create all tensors needed for the c128 test."""
    b = attn_cfg.b
    s1 = attn_cfg.s1
    d = attn_cfg.q_d
    nq = attn_cfg.n1
    nkv = attn_cfg.n2
    cmp_ratio = attn_cfg.cmp_ratio
    block_size = attn_cfg.block_size
    max_blocks = attn_cfg.max_blocks
    seqused_kv = attn_cfg.actual_seq

    q_shape = [b * s1, nq, d]
    cmp_kv_shape = [attn_cfg.kv_num_blocks, block_size, nkv, d]
    cmp_blk_tbl_shape = [attn_cfg.block_table_batch, max_blocks]
    max_actual_seq = max(seqused_kv)
    win_max_actual_seq = max(max_actual_seq, block_size + s1 - 1)
    win_max_blocks = math.ceil(win_max_actual_seq / block_size)
    ori_kv_shape = [b * win_max_blocks, block_size, nkv, d]
    ori_blk_tbl_shape = cmp_blk_tbl_shape

    empty_kwargs = {"dtype": torch_dtype, "device": device}
    q = torch.empty(q_shape, **empty_kwargs).uniform_(-1, 1)
    cmp_kv = torch.empty(cmp_kv_shape, **empty_kwargs).uniform_(-1, 1)
    sinks = torch.empty(nq, dtype=torch.float32, device=device).uniform_(-1, 1)
    ori_kv = torch.empty(ori_kv_shape, **empty_kwargs).uniform_(-1, 1)
    ori_block_table = gen_block_table(seqused_kv, block_size, ori_blk_tbl_shape, cmp_ratio=cmp_ratio, enable_win=True)

    tmp_out_golden = torch.zeros((b * s1 * 2 * block_size, q_shape[2]), **empty_kwargs) + 1
    output_flash = torch.zeros(q_shape, **empty_kwargs)

    return _C128CreateTensorsOutputs(q, cmp_kv, sinks, ori_kv, ori_block_table, tmp_out_golden, output_flash)


def c128(enable_flash: bool, enable_high_perf: bool, enable_graph: bool, device: str, attn_cfg: AttentionConfig):
    torch_dtype = torch.bfloat16
    cmp_ratio = attn_cfg.cmp_ratio
    seqused_kv = attn_cfg.actual_seq
    q_shape = [attn_cfg.b * attn_cfg.s1, attn_cfg.n1, attn_cfg.q_d]

    _t = _c128_create_tensors(attn_cfg, torch_dtype, device)
    q, cmp_kv, sinks, ori_kv, ori_block_table, tmp_out_golden, output_flash = (
        _t.q, _t.cmp_kv, _t.sinks, _t.ori_kv,
        _t.ori_block_table, _t.tmp_out_golden, _t.output_flash)

    cmp_block_table = gen_block_table(seqused_kv, attn_cfg.block_size,
        [attn_cfg.block_table_batch, attn_cfg.max_blocks], cmp_ratio=cmp_ratio)
    attention_out = torch.zeros(q_shape, **{"dtype": torch_dtype, "device": device})

    ifa_golden(IfaGoldenInputs(q=q, cmp_kv=cmp_kv, sinks=sinks,
        cmp_block_table=cmp_block_table, seqused_kv=seqused_kv,
        output_flash=output_flash, tmp_out=tmp_out_golden, enable_flash=False,
        cmp_ratio=cmp_ratio, is_new_sink=True, ori_kv=ori_kv,
        ori_block_table=ori_block_table))

    # acl graph
    if enable_graph:
        import torchair as tng
        from torchair.configs.compiler_config import CompilerConfig
        compiler_config = CompilerConfig()
        compiler_config.mode = "reduce-overhead"
        npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
        model = torch.compile(MM(), dynamic=False, fullgraph=True, backend=npu_backend)
        for _ in range(10):
            attention_out = model(q, cmp_kv, sinks, cmp_block_table, seqused_kv, ori_kv, ori_block_table, cmp_ratio)
            pypto.runtime._device_synchronize()  # 内部接口，不推荐使用
    else:
        for _ in range(10):
            attention_out = cfa_attention(q, cmp_kv, sinks, cmp_block_table, seqused_kv, ori_kv,
                ori_block_table, cmp_ratio)

    from tests.ops.utils import compare
    compare.compare(output_flash, attention_out, "golden vs npu", rtol=0.0078125, atol=0.0001)


@pytest.mark.soc("950", "910")
def test_c128_decode(enable_flash: bool = False, enable_high_perf: bool = False, enable_graph: bool = False, \
                    device_id: int = 0):
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', device_id))
    torch.npu.set_device(device_id)
    device = f'npu:{device_id}'
    attn_cfg = get_decode_case(device=device)
    c128(enable_flash=enable_flash, enable_high_perf=enable_high_perf, enable_graph=enable_graph, device=device, \
        attn_cfg=attn_cfg)


if __name__ == "__main__":
    import argparse as ap
    p = ap.ArgumentParser(description="参数配置")
    p.add_argument("-f", "--enable-flash", action="store_true", help="开启flash模式")
    p.add_argument("-p", "--high-perf", action="store_true", help="启用高性能模式")
    p.add_argument("-g", "--enable-graph", action="store_true", help="启用高性能模式")
    p.add_argument("-c", "--device-id", type=int, default=0, help="显卡序号，默认0")
    p.add_argument("-u", "--upper", type=int, default=6000, help="融合上限法")
    args = p.parse_args()
    test_c128_decode(enable_flash=args.enable_flash, enable_high_perf=args.high_perf, enable_graph=args.enable_graph,
             device_id=args.device_id)
