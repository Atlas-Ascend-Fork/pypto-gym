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
PFA (Prompt Flash Attention) Optimized Implementation

优化项：
1. 块级循环重构 (Block-level Loop Refactoring) - 将s1循环改为块级处理
2. K/V同时组装 (K/V Joint Assembly) - 合并K和V的组装操作
3. 静态轴合并 (Static Axis Merge) - 合并n2和group循环，减少嵌套深度

"""

from dataclasses import dataclass, replace

import torch
from torch._dynamo import allow_in_graph

import pypto


@dataclass
class AttentionTileConfig:
    g_tile: int
    s2_tile: int
    c1_tile: list
    v1_tile: list
    c2_tile: list
    v2_tile: list


@dataclass
class LoopOfs:
    bs_ofs: int = 0
    n1g_ofs: int = 0
    out_ofs: int = 0


@dataclass
class LoopTensor:
    q_2d: pypto.Tensor = None
    k_2d: pypto.Tensor = None
    v_2d: pypto.Tensor = None
    block_table: pypto.Tensor = None
    kv_act_seqs: pypto.Tensor = None
    atten_out: pypto.Tensor = None


@dataclass
class LoopIndex:
    b_idx: int = 0
    s1_idx: int = 0
    n2_idx: int = 0
    group_idx: int = 0
    s2_idx: int = 0


@dataclass
class LoopSize:
    group_loop: int = 0
    s2_loop: int = 0


@dataclass
class TempUpdateTensor:
    out_update: pypto.Tensor = None
    sum_update: pypto.Tensor = None
    max_update: pypto.Tensor = None


@dataclass
class PFAKernelParams:
    n1: int
    d: int
    block_num: int
    n2: int
    block_size: int
    b: int
    s1: int
    group: int
    softmax_scale: float


@dataclass
class ContextParams:
    kernel_params: PFAKernelParams = None
    tile_cfg: AttentionTileConfig = None
    loop_tensors: LoopTensor = None
    loop_index: LoopIndex = None
    loop_size: LoopSize = None
    loop_ofs: LoopOfs = None
    temp_update_tensors: TempUpdateTensor = None


def get_pfa_tile_cfg():
    """获取PFA Tile配置"""
    m_tile = 256
    k_tile = 128
    n_tile = 128
    s2_tile = 1024

    tile_cfg = AttentionTileConfig(
        g_tile=12,
        s2_tile=s2_tile,
        c1_tile=[[m_tile, m_tile], [k_tile, k_tile], [n_tile, n_tile]],
        v1_tile=[128, s2_tile],
        c2_tile=[[m_tile, m_tile], [k_tile, k_tile], [n_tile, n_tile]],
        v2_tile=[128, m_tile]
    )
    return tile_cfg


def assemble_kv_j(idx, actual_s2_tile, ctx_params):
    k_2d = ctx_params.loop_tensors.k_2d
    v_2d = ctx_params.loop_tensors.v_2d
    block_table = ctx_params.loop_tensors.block_table

    s2_tile = ctx_params.tile_cfg.s2_tile
    block_size = ctx_params.kernel_params.block_size
    d = ctx_params.kernel_params.d
    b_idx = ctx_params.loop_index.b_idx

    block_num = s2_tile // block_size

    kj_assemble = pypto.tensor([s2_tile, d], k_2d.dtype, "kj_assemble")
    vj_assemble = pypto.tensor([s2_tile, d], v_2d.dtype, "vj_assemble")

    for i in range(block_num):
        block_idx = block_table[b_idx, idx + i]
        block_idx_valid = block_idx.max(0)
        base_offset = block_idx_valid * block_size

        kj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
            pypto.view(k_2d, [block_size, d], [base_offset, 0])
        vj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
            pypto.view(v_2d, [block_size, d], [base_offset, 0])

    kj_assemble = pypto.view(kj_assemble, [s2_tile, d], [0, 0], valid_shape=[actual_s2_tile, d])
    vj_assemble = pypto.view(vj_assemble, [s2_tile, d], [0, 0], valid_shape=[actual_s2_tile, d])

    return kj_assemble, vj_assemble


def compute_c1(qi, kj_assemble, actual_s2_tile, tile_cfg):
    """计算第一个矩阵乘法: Q x K^T"""
    c1_tile = tile_cfg.c1_tile
    g_tile = tile_cfg.g_tile
    s2_tile = tile_cfg.s2_tile

    pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
    sij = pypto.matmul(qi, kj_assemble, pypto.DT_FP32, a_trans=False, b_trans=True)
    sij = pypto.view(sij, [g_tile, s2_tile], [0, 0], valid_shape=[g_tile, actual_s2_tile])
    return sij


def compute_first_tile(sij, vj_assemble, dtype, ctx_params):
    """计算第一个tile的attention"""
    softmax_scale = ctx_params.kernel_params.softmax_scale
    c2_tile = ctx_params.tile_cfg.c2_tile
    v2_tile = ctx_params.tile_cfg.v2_tile
    out_update = ctx_params.temp_update_tensors.out_update
    sum_update = ctx_params.temp_update_tensors.sum_update
    max_update = ctx_params.temp_update_tensors.max_update

    sij_scale = pypto.mul(sij, softmax_scale)
    tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
    tsub = pypto.sub(sij_scale, tilda_mij)
    tilda_pij = pypto.exp(tsub)
    tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)

    sum_update[:] = pypto.sum(tilda_pij, dim=-1, keepdim=True)
    max_update[:] = tilda_mij

    pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
    oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)

    pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
    out_update[:] = oi_tmp


def compute_other_tile(sij, vj_assemble, dtype, ctx_params):
    """计算后续tile的attention（online softmax更新）"""
    softmax_scale = ctx_params.kernel_params.softmax_scale
    c2_tile = ctx_params.tile_cfg.c2_tile
    v2_tile = ctx_params.tile_cfg.v2_tile
    out_update = ctx_params.temp_update_tensors.out_update
    sum_update = ctx_params.temp_update_tensors.sum_update
    max_update = ctx_params.temp_update_tensors.max_update

    pypto.set_pass_options(sg_set_scope=1)
    sij_scale = pypto.mul(sij, softmax_scale)
    tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
    max_new = pypto.maximum(max_update, tilda_mij)
    tsub = pypto.sub(sij_scale, max_new)
    tilda_pij = pypto.exp(tsub)
    tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
    sum_local = pypto.sum(tilda_pij, dim=-1, keepdim=True)
    pypto.set_pass_options(sg_set_scope=-1)

    pypto.set_pass_options(sg_set_scope=2)
    tsub2 = pypto.sub(max_update, max_new)
    max_update[:] = max_new
    update_mul = pypto.exp(tsub2)
    sum_update[:] = sum_update * update_mul + sum_local
    pypto.set_pass_options(sg_set_scope=-1)

    pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
    oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)

    pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
    out_update[:] = out_update * update_mul + oi_tmp


def finalize_output(out_ofs, dtype, ctx_params):
    """完成输出并写入结果"""
    d = ctx_params.kernel_params.d
    v2_tile = ctx_params.tile_cfg.v2_tile
    g_tile = ctx_params.tile_cfg.g_tile
    out_update = ctx_params.temp_update_tensors.out_update
    sum_update = ctx_params.temp_update_tensors.sum_update
    atten_out = ctx_params.loop_tensors.atten_out

    oi_final = pypto.div(out_update, sum_update)
    pypto.set_vec_tile_shapes(16, v2_tile[0], v2_tile[1])
    oi_final_3d = pypto.cast(
        pypto.reshape(oi_final, [1, g_tile, d]), dtype)
    pypto.assemble(oi_final_3d, out_ofs, atten_out)


def init_kernel_params(q, k, block_table):
    """初始化kernel参数"""
    bs, n1, d = q.shape
    block_num, n2, block_size, _ = k.shape
    block_table_shape = block_table.shape
    b = block_table_shape[0]
    s1 = bs // b
    group = n1 // n2
    softmax_scale = d ** -0.5
    kernel_params = PFAKernelParams(
        n1=n1, d=d, block_num=block_num, n2=n2, block_size=block_size,
        b=b, s1=s1, group=group, softmax_scale=softmax_scale
    )
    return kernel_params


def reshape_qkv_to_2d(q, k, v, kernel_params):
    """将Q, K, V reshape为2D"""
    b = kernel_params.b
    s1 = kernel_params.s1
    n1 = kernel_params.n1
    d = kernel_params.d
    block_num = kernel_params.block_num
    block_size = kernel_params.block_size
    n2 = kernel_params.n2

    q_2d_shape = (b * s1 * n1, d)
    kv_2d_shape = (block_num * block_size * n2, d)

    q_2d = pypto.reshape(q, q_2d_shape, inplace=True)
    k_2d = pypto.reshape(k, kv_2d_shape, inplace=True)
    v_2d = pypto.reshape(v, kv_2d_shape, inplace=True)
    return q_2d, k_2d, v_2d


def compute_loop_b_optimized(dtype, ctx_params):
    """使用块级s1循环"""
    s1 = ctx_params.kernel_params.s1
    s2_tile = ctx_params.tile_cfg.s2_tile
    kv_act_seqs = ctx_params.loop_tensors.kv_act_seqs
    b_idx = ctx_params.loop_index.b_idx
    loop_size = ctx_params.loop_size

    s1_block_step = 16

    s1_block_num = (s1 + s1_block_step - 1) // s1_block_step

    for s1_block_idx in pypto.loop(s1_block_num, name="LOOP_s1_block", idx_name="s1_block_idx"):
        s1_start = s1_block_idx * s1_block_step
        s1_end = (s1_block_idx + 1) * s1_block_step
        actual_s1_in_block = (s1 - s1_start).min(s1_block_step)

        s2_max_for_block = s1_start + actual_s1_in_block

        s2_loop_for_block = pypto.ceildiv(
            kv_act_seqs[b_idx] - (s1 - s2_max_for_block),
            s2_tile
        )

        loop_size = replace(loop_size, s2_loop=s2_loop_for_block)

        for s1_offset in pypto.loop(actual_s1_in_block, name="LOOP_s1_offset", idx_name="s1_offset"):
            s1_idx = s1_start + s1_offset

            cur_seq_len = kv_act_seqs[b_idx] - (s1 - 1 - s1_idx)

            bs_ofs = b_idx * s1 + s1_idx
            loop_ofs = LoopOfs(bs_ofs=bs_ofs)

            ctx_params = replace(
                ctx_params,
                loop_size=loop_size,
                loop_ofs=loop_ofs,
                loop_index=replace(ctx_params.loop_index, s1_idx=s1_idx)
            )
            compute_loop_n2_group_merged(ctx_params, cur_seq_len, dtype, s2_loop_for_block)


def compute_loop_n2_group_merged(ctx_params, cur_seq_len, dtype, s2_loop_for_block):
    """静态轴合并后的 n2+group 循环"""
    g_tile = ctx_params.tile_cfg.g_tile
    group = ctx_params.kernel_params.group
    n2 = ctx_params.kernel_params.n2
    d = ctx_params.kernel_params.d
    loop_ofs = ctx_params.loop_ofs
    bs_ofs = loop_ofs.bs_ofs
    group_loop = ctx_params.loop_size.group_loop

    g_loop_merged = group_loop * n2

    for g_idx_merged in pypto.loop(g_loop_merged, name="LOOP_n2_group_merged", idx_name="g_idx_merged"):
        n2_idx = g_idx_merged // group_loop
        group_idx = g_idx_merged % group_loop

        n1g_ofs = n2_idx * group + group_idx * g_tile
        out_ofs = [bs_ofs, n1g_ofs, 0]
        loop_ofs_merged = replace(loop_ofs, n1g_ofs=n1g_ofs, out_ofs=out_ofs)

        out_update = pypto.tensor([g_tile, d], pypto.DT_FP32, "out_update")
        sum_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update")
        max_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update")
        temp_update_tensors = TempUpdateTensor(out_update, sum_update, max_update)

        loop_index = replace(ctx_params.loop_index, n2_idx=n2_idx, group_idx=group_idx)
        ctx_params_merged = replace(
            ctx_params,
            loop_index=loop_index,
            temp_update_tensors=temp_update_tensors,
            loop_ofs=loop_ofs_merged
        )

        compute_loop_s2_optimized(ctx_params_merged, cur_seq_len, dtype, s2_loop_for_block)


def compute_loop_s2_optimized(ctx_params, cur_seq_len, dtype, s2_loop_for_block):
    """s2循环 - 使用K/V同时组装"""
    tile_cfg = ctx_params.tile_cfg
    s2_tile = tile_cfg.s2_tile
    v1_tile = tile_cfg.v1_tile
    g_tile = tile_cfg.g_tile
    block_size = ctx_params.kernel_params.block_size
    d = ctx_params.kernel_params.d
    n1 = ctx_params.kernel_params.n1
    q_2d = ctx_params.loop_tensors.q_2d
    bs_ofs = ctx_params.loop_ofs.bs_ofs
    n1g_ofs = ctx_params.loop_ofs.n1g_ofs
    out_ofs = ctx_params.loop_ofs.out_ofs
    s2_idx = ctx_params.loop_index.s2_idx

    block_num = s2_tile // block_size

    for s2_idx in pypto.loop(s2_loop_for_block, name="LOOP_s2", idx_name="s2_idx", unroll_list=[8, 4, 2, 1]):
        idx = s2_idx * block_num

        actual_s2_tile = (cur_seq_len - s2_idx * s2_tile).min(s2_tile)

        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
        qi = pypto.view(q_2d, [g_tile, d], [bs_ofs * n1 + n1g_ofs, 0])

        kj_assemble, vj_assemble = assemble_kv_j(idx, actual_s2_tile, ctx_params)
        sij = compute_c1(qi, kj_assemble, actual_s2_tile, tile_cfg)

        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
        if pypto.cond(pypto.is_loop_begin(s2_idx)):
            compute_first_tile(sij, vj_assemble, dtype, ctx_params)
        else:
            compute_other_tile(sij, vj_assemble, dtype, ctx_params)

        if pypto.cond(pypto.is_loop_end(s2_idx)):
            finalize_output(out_ofs, dtype, ctx_params)


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 128,
        "device_sched_mode": 1
    },
    pass_options={
        "cube_l1_reuse_setting": {-1: 16},
        "vec_nbuffer_setting": {0: 8}
    }
)
def pfa_optimized_kernel(
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    block_table: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_INT32),
    kv_act_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    atten_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16)
):
    """PFA Kernel"""

    dtype = q.dtype
    kernel_params = init_kernel_params(q, k, block_table)
    tile_cfg = get_pfa_tile_cfg()
    q_2d, k_2d, v_2d = reshape_qkv_to_2d(q, k, v, kernel_params)
    loop_tensors = LoopTensor(q_2d, k_2d, v_2d, block_table, kv_act_seqs, atten_out)

    group_loop = kernel_params.group // tile_cfg.g_tile
    loop_size = LoopSize(group_loop=group_loop)

    ctx_params = ContextParams(
        kernel_params=kernel_params, tile_cfg=tile_cfg, loop_tensors=loop_tensors,
        loop_size=loop_size
    )

    for b_idx in pypto.loop(kernel_params.b, name="LOOP_b", idx_name="b_idx"):
        loop_index = LoopIndex(b_idx=b_idx)
        ctx_params = replace(ctx_params, loop_index=loop_index)
        compute_loop_b_optimized(dtype, ctx_params)


@allow_in_graph
def prompt_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    actual_seq_lengths: torch.Tensor,
    block_table: torch.Tensor,
):
    """Prompt Flash Attention入口函数"""
    atten_out = torch.empty_like(query)
    atten_out.fill_(0)
    pfa_optimized_kernel(query, key, value, block_table, actual_seq_lengths, atten_out)
    return atten_out
