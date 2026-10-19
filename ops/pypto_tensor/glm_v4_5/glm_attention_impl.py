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
    - ifa_func_kernel: JIT compiled kernel implementing Flash Attention with paged KV cache
    - ifa_func_kernel_for_950: JIT compiled kernel optimized for Ascend 950
"""
import os
import math
from dataclasses import dataclass
from typing import List, Union
import torch
import pypto
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
from .utils.get_format import get_format


def check_args(query, key_cache, value_cache, block_tables, actual_seqs, attn_res):
    assert query.dim() == 3
    assert get_format(query) == 'ND'
    assert query.dtype == torch.bfloat16
    assert key_cache.dim() == 4
    assert get_format(key_cache) == 'ND'
    assert key_cache.dtype == torch.bfloat16
    assert value_cache.dim() == 4
    assert get_format(value_cache) == 'ND'
    assert value_cache.dtype == torch.bfloat16
    assert block_tables.dim() == 2
    assert get_format(block_tables) == 'ND'
    assert block_tables.dtype == torch.int32
    assert actual_seqs.dim() == 1
    assert get_format(actual_seqs) == 'ND'
    assert actual_seqs.dtype == torch.int32
    assert attn_res.dim() == 3
    assert get_format(attn_res) == 'ND'
    assert attn_res.dtype == torch.bfloat16


@dataclass
class IfaTileShapeConfig:
    g_tile: int
    s2_tile: int
    c1_tile_shape: list
    v1_tile_shape: list
    c2_tile_shape: list
    v2_tile_shape: list


@dataclass
class IfaConfig:
    b: int
    s1: int
    s2: Union[int, List[int]]
    nq: int
    nkv: int
    qd: int
    kvd: int
    block_size: int
    max_num_blocks_per_query: int = 0
    softmax_scale: float = 1.0
    kv_layout: str = "PA_BSND"
    actual_seq: torch.Tensor = None
    block_table_batch: int = 0
    kv_num_blocks: int = 0


@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 1024, "ready_on_host_tensors": ["block_table", "kv_act_seqs"]},
    pass_options={"cube_l1_reuse_setting": {0: 8}, "cube_nbuffer_setting": {-1: 8}},
)
def ifa_func_kernel(
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    block_table: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_INT32),
    kv_act_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    atten_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    softmax_scale,
    tile_config,
):
    pypto.experimental.set_operation_options(combine_axis=True)

    shape_q = q.shape
    shape_k = k.shape
    bs_scalar = shape_q[0]
    nq = shape_q[1]
    block_num_scalar = shape_k[0]
    block_size = shape_k[1]
    nkv = shape_k[2]
    dn = shape_k[3]
    b_scalar = kv_act_seqs.shape[0]

    dtype = q.dtype
    group = nq // nkv
    n2_sym = nkv

    g_tile = tile_config.g_tile
    s2_tile = tile_config.s2_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape

    s1_scalar = bs_scalar // b_scalar
    g = nq // nkv
    g_loop = g // g_tile

    k_2d_shape = (block_num_scalar * block_size, n2_sym * dn)
    q_2d_shape = (b_scalar * s1_scalar * nq, dn)

    k_2d = pypto.reshape(k, k_2d_shape, inplace=True)
    v_2d = pypto.reshape(v, k_2d_shape, inplace=True)
    q_2d = pypto.reshape(q, q_2d_shape, inplace=True)
    for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"):
        for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx"):
            cur_seq = kv_act_seqs[b_idx] - (s1_scalar - 1 - s1_idx)
            s2_loop = (cur_seq + s2_tile - 1) // s2_tile
            for n2_idx in pypto.loop(n2_sym, name="LOOP_n2", idx_name="n2_idx"):
                for g_idx in pypto.loop(g_loop, name="LOOP_g", idx_name="g_idx"):
                    oi_update = pypto.tensor([g_tile, dn], pypto.DT_FP32, "oi_update")
                    sum_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update")
                    max_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update")
                    for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx", unroll_list=[8, 4, 2, 1]):
                        block_num = s2_tile // block_size
                        idx = s2_idx * block_num
                        bs_ofs = b_idx * s1_scalar + s1_idx
                        n1g_ofs = n2_idx * group + g_idx * g_tile
                        actual_s2_tile = (cur_seq - s2_idx * s2_tile).min(s2_tile)
                        oi_ofs = [bs_ofs, n1g_ofs, 0]
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        qi = pypto.view(q_2d, [g_tile, dn], [bs_ofs * nq + n1g_ofs, 0])

                        kj_assemble = pypto.tensor([s2_tile, dn], k_2d.dtype, "kj_assemble")
                        for i in range(block_num):
                            block_idx = block_table[b_idx, idx + i]
                            block_idx_valid = block_idx.max(0)
                            kj_assemble[i * block_size:(i + 1) * block_size, 0:] = pypto.view(
                                k_2d, [block_size, dn], [block_idx_valid * block_size, n2_idx * dn]
                            )
                        kj_assemble = pypto.view(kj_assemble, [s2_tile, dn], [0, 0], valid_shape=[s2_tile, dn])

                        pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
                        sij = pypto.matmul(qi, kj_assemble, pypto.DT_FP32, a_trans=False, b_trans=True)
                        sij = pypto.view(sij, [g_tile, s2_tile], [0, 0], valid_shape=[g_tile, actual_s2_tile])
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        if pypto.is_loop_begin(s2_idx):
                            sij_scale = pypto.mul(sij, softmax_scale)
                            tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)

                            tsub = pypto.sub(sij_scale, tilda_mij)
                            tilda_pij = pypto.exp(tsub)
                            tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
                            sum_update[:] = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                            max_update[:] = tilda_mij

                            vj_assemble = pypto.tensor([s2_tile, dn], v_2d.dtype, "vj_assemble")
                            for i in range(block_num):
                                block_idx = block_table[b_idx, idx + i]
                                block_idx_valid = block_idx.max(0)
                                vj_assemble[i * block_size:(i + 1) * block_size, 0:] = pypto.view(
                                    v_2d, [block_size, dn], [block_idx_valid * block_size, n2_idx * dn]
                                )
                            vj_assemble = pypto.view(
                                vj_assemble, [s2_tile, dn], [0, 0], valid_shape=[actual_s2_tile, dn]
                            )
                            pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                            oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)

                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_update[:] = oi_tmp
                        else:
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

                            vj_assemble = pypto.tensor([s2_tile, dn], v_2d.dtype, "vj_assemble")
                            for i in range(block_num):
                                block_idx = block_table[b_idx, idx + i]
                                block_idx_valid = block_idx.max(0)
                                vj_assemble[i * block_size:(i + 1) * block_size, 0:] = pypto.view(
                                    v_2d, [block_size, dn], [block_idx_valid * block_size, n2_idx * dn]
                                )
                            vj_assemble = pypto.view(
                                vj_assemble, [s2_tile, dn], [0, 0], valid_shape=[actual_s2_tile, dn]
                            )
                            pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                            oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)

                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_update[:] = oi_update * update_mul + oi_tmp
                        if pypto.is_loop_end(s2_idx):
                            oi_final = pypto.div(oi_update, sum_update, precision_type=pypto.PrecisionType.INTRINSIC)
                            pypto.set_vec_tile_shapes(16, v2_tile[0], v2_tile[1])
                            oi_final_3d = pypto.cast(pypto.reshape(oi_final, [1, g_tile, dn]), dtype)
                            pypto.assemble(oi_final_3d, oi_ofs, atten_out)


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 512,
        "device_sched_mode": 2,
        "ready_on_host_tensors": ["block_table", "kv_act_seqs"],
        "max_workspace_kb": 2000000,
    },
    pass_options={
        "cube_l1_reuse_setting": {0: 16, 1: 8},
        "cube_nbuffer_setting": {0: 2, 1: 4},
        "vec_nbuffer_setting": {-2: 1, 0: 1, 1: 1},
    },
    host_options={"compile_monitor_enable": 0},
)
def ifa_func_kernel_for_910_high_performance(
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    block_table: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_INT32),
    kv_act_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    atten_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    softmax_scale,
    tile_config,
):
    pypto.experimental.set_operation_options(combine_axis=True)

    shape_q = q.shape
    shape_k = k.shape
    shape_act_seqs = kv_act_seqs.shape
    bs_scalar = shape_q[0]
    nq = shape_q[1]
    block_num_scalar = shape_k[0]
    block_size = shape_k[1]
    nkv = shape_k[2]
    dn = shape_k[3]
    b_scalar = shape_act_seqs[0]

    dtype = q.dtype
    group = nq // nkv
    n2_sym = nkv

    g_tile = tile_config.g_tile
    s2_tile = tile_config.s2_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape

    s1_scalar = bs_scalar // b_scalar
    g = nq // nkv
    g_loop = g // g_tile

    k_2d_shape = (block_num_scalar * block_size, n2_sym * dn)
    q_2d_shape = (b_scalar * s1_scalar * nq, dn)

    k_2d = pypto.reshape(k, k_2d_shape, inplace=True)
    v_2d = pypto.reshape(v, k_2d_shape, inplace=True)
    q_2d = pypto.reshape(q, q_2d_shape, inplace=True)
    for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"):
        for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx"):
            cur_seq = kv_act_seqs[b_idx] - (s1_scalar - 1 - s1_idx)
            s2_loop = (cur_seq + s2_tile - 1) // s2_tile
            for n2_idx in pypto.loop(n2_sym, name="LOOP_n2", idx_name="n2_idx"):
                for g_idx in pypto.loop(g_loop, name="LOOP_g", idx_name="g_idx"):
                    oi_update = pypto.tensor([g_tile, dn], pypto.DT_FP32, "oi_update")
                    sum_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update")
                    max_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update")
                    for s2_idx in pypto.loop(
                        s2_loop, name="LOOP_s2", idx_name="s2_idx", unroll_list=[32, 16, 12, 8, 4, 2, 1]
                    ):
                        block_num = s2_tile // block_size
                        idx = s2_idx * block_num
                        bs_ofs = b_idx * s1_scalar + s1_idx
                        n1g_ofs = n2_idx * group + g_idx * g_tile
                        actual_s2_tile = (cur_seq - s2_idx * s2_tile).min(s2_tile)
                        oi_ofs = [bs_ofs, n1g_ofs, 0]
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        qi = pypto.view(q_2d, [g_tile, dn], [bs_ofs * nq + n1g_ofs, 0])
                        kj_assemble = pypto.tensor([s2_tile, dn], k_2d.dtype, "kj_assemble")
                        for i in range(block_num):
                            block_idx = block_table[b_idx, idx + i]
                            block_idx_vaild = block_idx.max(0)
                            kj_assemble[i * block_size:(i + 1) * block_size, 0:] = pypto.view(
                                k_2d, [block_size, dn], [block_idx_vaild * block_size, n2_idx * dn]
                            )
                        kj_assemble = pypto.view(kj_assemble, [s2_tile, dn], [0, 0], valid_shape=[s2_tile, dn])

                        pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
                        sij = pypto.matmul(qi, kj_assemble, pypto.DT_FP32, a_trans=False, b_trans=True)
                        sij = pypto.view(sij, [g_tile, s2_tile], [0, 0], valid_shape=[g_tile, actual_s2_tile])
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        pypto.set_pass_options(sg_set_scope=2)
                        sij_scale = pypto.mul(sij, softmax_scale)
                        amax_ij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                        tsub = pypto.sub(sij_scale, amax_ij)
                        vec1_res = pypto.exp(tsub)
                        vec1_res_fp16 = pypto.cast(vec1_res, dtype)
                        sum_local = pypto.sum(vec1_res, dim=-1, keepdim=True)

                        vj_assemble = pypto.tensor([s2_tile, dn], v_2d.dtype, "vj_assemble")
                        for i in range(block_num):
                            block_idx = block_table[b_idx, idx + i]
                            block_idx_vaild = block_idx.max(0)
                            vj_assemble[i * block_size:(i + 1) * block_size, 0:] = pypto.view(
                                v_2d, [block_size, dn], [block_idx_vaild * block_size, n2_idx * dn]
                            )
                        vj_assemble = pypto.view(vj_assemble, [s2_tile, dn], [0, 0], valid_shape=[actual_s2_tile, dn])
                        pypto.set_pass_options(sg_set_scope=-1)
                        pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                        mm2_res = pypto.matmul(vec1_res_fp16, vj_assemble, pypto.DT_FP32)

                        pypto.set_pass_options(sg_set_scope=1)
                        if pypto.is_loop_begin(s2_idx):
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_tmp = mm2_res
                            oi_update[:] = pypto.tensor(oi_tmp.shape, pypto.DT_FP32, "oi_update")
                            if pypto.is_loop_end(s2_idx):
                                oi_update[:] = pypto.div(
                                    oi_tmp, sum_local, precision_type=pypto.PrecisionType.INTRINSIC
                                )
                                oi_update_3d = pypto.reshape(oi_update, [1, g_tile, dn])
                                pypto.set_vec_tile_shapes(1, v2_tile[0], v2_tile[1])
                                oi_update_3d = pypto.cast(oi_update_3d, dtype)
                                pypto.assemble(oi_update_3d, oi_ofs, atten_out)
                            else:
                                oi_update[:] = oi_tmp
                                sum_update[:] = sum_local
                                max_update[:] = amax_ij
                        else:
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            max_new = pypto.maximum(max_update, amax_ij)
                            t1 = pypto.sub(max_update, max_new)
                            t2 = pypto.exp(t1)
                            t6 = pypto.mul(t2, sum_update)
                            t3 = pypto.sub(amax_ij, max_new)
                            t4 = pypto.exp(t3)
                            t5 = pypto.mul(t4, sum_local)
                            sum_new = pypto.add(t6, t5)
                            sum_update[:] = sum_new
                            max_update[:] = max_new

                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_last = pypto.mul(oi_update, t2)
                            oi_flash = pypto.mul(mm2_res, t4)
                            oi_tmp = pypto.add(oi_last, oi_flash)
                            if pypto.is_loop_end(s2_idx):
                                pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                oi_update_tmp = pypto.div(
                                    oi_tmp, sum_update, precision_type=pypto.PrecisionType.INTRINSIC
                                )
                                oi_update_tmp_3d = pypto.reshape(oi_update_tmp, [1, g_tile, dn])
                                pypto.set_vec_tile_shapes(1, v2_tile[0], v2_tile[1])
                                oi_update_3d = pypto.cast(oi_update_tmp_3d, dtype)
                                pypto.assemble(oi_update_3d, oi_ofs, atten_out)
                            else:
                                oi_update[:] = oi_tmp
                        pypto.set_pass_options(sg_set_scope=-1)


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 512,
        "device_sched_mode": 2,
        "ready_on_host_tensors": ["block_table", "kv_act_seqs"],
        "max_workspace_kb": 2000000,
    },
    pass_options={
        "cube_l1_reuse_setting": {0: 16, 1: 8, 2: 1},
        "cube_nbuffer_setting": {0: 2, 1: 4, 2: 4},
        "vec_nbuffer_setting": {-2: 1, 0: 1, 1: 1},
    },
    host_options={"compile_monitor_enable": 0},
)
def ifa_func_kernel_for_910_s1_range_high_performance(
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    block_table: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_INT32),
    kv_act_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    atten_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    softmax_scale, tile_config
):
    pypto.experimental.set_operation_options(combine_axis=True)
    shape_q = q.shape
    shape_k = k.shape
    shape_act_seqs = kv_act_seqs.shape
    bs_scalar = shape_q[0]
    nq = shape_q[1]
    block_num_scalar = shape_k[0]
    block_size = shape_k[1]
    nkv = shape_k[2]
    dn = shape_k[3]
    b_scalar = shape_act_seqs[0]

    dtype = q.dtype
    group = nq // nkv
    n2_sym = nkv

    g_tile = tile_config.g_tile
    s2_tile = tile_config.s2_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape

    s1_scalar = bs_scalar // b_scalar
    g = nq // nkv
    g_loop = g // g_tile

    k_2d_shape = (block_num_scalar * block_size, n2_sym * dn)
    q_2d_shape = (b_scalar * s1_scalar * nq, dn)

    k_2d = pypto.reshape(k, k_2d_shape, inplace=True)
    v_2d = pypto.reshape(v, k_2d_shape, inplace=True)
    q_2d = pypto.reshape(q, q_2d_shape, inplace=True)

    for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"):
        for n2_idx in pypto.loop(n2_sym, name="LOOP_n2", idx_name="n2_idx"):
            for g_idx in pypto.loop(g_loop, name="LOOP_g", idx_name="g_idx"):
                oi_update_0 = pypto.tensor([g_tile, dn], pypto.DT_FP32, "oi_update_0")
                sum_update_0 = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update_0")
                max_update_0 = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update_0")
                oi_update_1 = pypto.tensor([g_tile, dn], pypto.DT_FP32, "oi_update_1")
                sum_update_1 = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update_1")
                max_update_1 = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update_1")

                max_cur_seq = kv_act_seqs[b_idx]
                s2_loop = (max_cur_seq + s2_tile - 1) // s2_tile

                for s2_idx in pypto.loop(
                    s2_loop, name="LOOP_s2", idx_name="s2_idx", unroll_list=[32, 16, 12, 8, 4, 2, 1]):
                    block_num = s2_tile // block_size
                    idx = s2_idx * block_num
                    n1g_ofs = n2_idx * group + g_idx * g_tile

                    kj_assemble = pypto.tensor([s2_tile, dn], k_2d.dtype, "kj_assemble")
                    for i in range(block_num):
                        block_idx = block_table[b_idx, idx + i]
                        block_idx_vaild = block_idx.max(0)
                        kj_assemble[i * block_size:(i + 1) * block_size, 0:] = pypto.view(k_2d,
                            [block_size, dn], [block_idx_vaild * block_size, n2_idx * dn])
                    kj_assemble = pypto.view(kj_assemble, [s2_tile, dn], [0, 0],
                                            valid_shape=[s2_tile, dn])

                    vj_assemble = pypto.tensor([s2_tile, dn], v_2d.dtype, "vj_assemble")
                    for i in range(block_num):
                        block_idx = block_table[b_idx, idx + i]
                        block_idx_vaild = block_idx.max(0)
                        vj_assemble[i * block_size:(i + 1) * block_size, 0:] = pypto.view(v_2d,
                            [block_size, dn], [block_idx_vaild * block_size, n2_idx * dn])

                    # ===== merged Q: stack two s1 q into one matmul =====
                    cur_seq_0 = kv_act_seqs[b_idx] - (s1_scalar - 1)
                    actual_s2_tile_0 = (cur_seq_0 - s2_idx * s2_tile).min(s2_tile).max(0)
                    bs_ofs_0 = b_idx * s1_scalar + 0
                    oi_ofs_0 = [bs_ofs_0, n1g_ofs, 0]

                    cur_seq_1 = kv_act_seqs[b_idx]
                    actual_s2_tile_1 = (cur_seq_1 - s2_idx * s2_tile).min(s2_tile)
                    bs_ofs_1 = b_idx * s1_scalar + 1
                    oi_ofs_1 = [bs_ofs_1, n1g_ofs, 0]

                    pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                    qi_merged = pypto.view(q_2d, [2 * g_tile, dn], [bs_ofs_0 * nq + n1g_ofs, 0])

                    pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
                    sij_merged = pypto.matmul(qi_merged, kj_assemble, pypto.DT_FP32, a_trans=False, b_trans=True)

                    sij_0 = pypto.view(sij_merged, [g_tile, s2_tile], [0, 0],
                            valid_shape=[g_tile, actual_s2_tile_0])
                    sij_1 = pypto.view(sij_merged, [g_tile, s2_tile], [g_tile, 0],
                            valid_shape=[g_tile, actual_s2_tile_1])

                    # ===== s1_idx = 0 softmax + V matmul =====
                    pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                    pypto.set_pass_options(sg_set_scope=2)
                    sij_scale_0 = pypto.mul(sij_0, softmax_scale)
                    amax_ij_0 = pypto.amax(sij_scale_0, dim=-1, keepdim=True)
                    tsub_0 = pypto.sub(sij_scale_0, amax_ij_0)
                    vec1_res_0 = pypto.exp(tsub_0)
                    vec1_res_fp16_0 = pypto.cast(vec1_res_0, dtype)
                    sum_local_0 = pypto.sum(vec1_res_0, dim=-1, keepdim=True)

                    vj_assemble_view_0 = pypto.view(vj_assemble, [s2_tile, dn], [0, 0],
                                    valid_shape=[actual_s2_tile_0, dn])
                    pypto.set_pass_options(sg_set_scope=-1)

                    pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                    mm2_res_0 = pypto.matmul(vec1_res_fp16_0, vj_assemble_view_0, pypto.DT_FP32)

                    safe_sum_0 = pypto.maximum(sum_local_0, 1.0)
                    pypto.set_pass_options(sg_set_scope=1)
                    if pypto.is_loop_begin(s2_idx):
                        pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                        oi_tmp_0 = mm2_res_0
                        oi_update_0[:] = pypto.tensor(oi_tmp_0.shape, pypto.DT_FP32, "oi_update_0_init")
                        if pypto.is_loop_end(s2_idx):
                            oi_update_0[:] = pypto.div(oi_tmp_0, safe_sum_0,
                                                    precision_type=pypto.PrecisionType.INTRINSIC)
                            oi_update_3d_0 = pypto.reshape(oi_update_0, [1, g_tile, dn])
                            pypto.set_vec_tile_shapes(1, v2_tile[0], v2_tile[1])
                            oi_update_3d_0 = pypto.cast(oi_update_3d_0, dtype)
                            pypto.assemble(oi_update_3d_0, oi_ofs_0, atten_out)
                        else:
                            oi_update_0[:] = oi_tmp_0
                            sum_update_0[:] = sum_local_0
                            max_update_0[:] = amax_ij_0
                    else:
                        pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                        max_new_0 = pypto.maximum(max_update_0, amax_ij_0)
                        t1_0 = pypto.sub(max_update_0, max_new_0)
                        t2_0 = pypto.exp(t1_0)
                        t6_0 = pypto.mul(t2_0, sum_update_0)
                        t3_0 = pypto.sub(amax_ij_0, max_new_0)
                        t4_0 = pypto.exp(t3_0)
                        t5_0 = pypto.mul(t4_0, sum_local_0)
                        sum_new_0 = pypto.add(t6_0, t5_0)
                        safe_sum_new_0 = pypto.maximum(sum_new_0, 1.0)

                        pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                        oi_last_0 = pypto.mul(oi_update_0, t2_0)
                        oi_flash_0 = pypto.mul(mm2_res_0, t4_0)
                        oi_tmp_0 = pypto.add(oi_last_0, oi_flash_0)
                        if pypto.is_loop_end(s2_idx):
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_update_tmp_0 = pypto.div(
                                oi_tmp_0, safe_sum_new_0, precision_type=pypto.PrecisionType.INTRINSIC
                            )
                            oi_update_tmp_3d_0 = pypto.reshape(oi_update_tmp_0, [1, g_tile, dn])
                            pypto.set_vec_tile_shapes(1, v2_tile[0], v2_tile[1])
                            oi_update_3d_0 = pypto.cast(oi_update_tmp_3d_0, dtype)
                            pypto.assemble(oi_update_3d_0, oi_ofs_0, atten_out)
                        else:
                            oi_update_0[:] = oi_tmp_0
                            sum_update_0[:] = sum_new_0
                            max_update_0[:] = max_new_0
                    pypto.set_pass_options(sg_set_scope=-1)

                    # ===== s1_idx = 1 softmax + V matmul =====
                    pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                    pypto.set_pass_options(sg_set_scope=2)
                    sij_scale_1 = pypto.mul(sij_1, softmax_scale)
                    amax_ij_1 = pypto.amax(sij_scale_1, dim=-1, keepdim=True)
                    tsub_1 = pypto.sub(sij_scale_1, amax_ij_1)
                    vec1_res_1 = pypto.exp(tsub_1)
                    vec1_res_fp16_1 = pypto.cast(vec1_res_1, dtype)
                    sum_local_1 = pypto.sum(vec1_res_1, dim=-1, keepdim=True)

                    vj_assemble_view_1 = pypto.view(vj_assemble, [s2_tile, dn], [0, 0],
                                    valid_shape=[actual_s2_tile_1, dn])
                    pypto.set_pass_options(sg_set_scope=-1)

                    pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                    mm2_res_1 = pypto.matmul(vec1_res_fp16_1, vj_assemble_view_1, pypto.DT_FP32)

                    pypto.set_pass_options(sg_set_scope=1)
                    if pypto.is_loop_begin(s2_idx):
                        pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                        oi_tmp_1 = mm2_res_1
                        oi_update_1[:] = pypto.tensor(oi_tmp_1.shape, pypto.DT_FP32, "oi_update_1_init")
                        if pypto.is_loop_end(s2_idx):
                            oi_update_1[:] = pypto.div(oi_tmp_1, sum_local_1,
                                                    precision_type=pypto.PrecisionType.INTRINSIC)
                            oi_update_3d_1 = pypto.reshape(oi_update_1, [1, g_tile, dn])
                            pypto.set_vec_tile_shapes(1, v2_tile[0], v2_tile[1])
                            oi_update_3d_1 = pypto.cast(oi_update_3d_1, dtype)
                            pypto.assemble(oi_update_3d_1, oi_ofs_1, atten_out)
                        else:
                            oi_update_1[:] = oi_tmp_1
                            sum_update_1[:] = sum_local_1
                            max_update_1[:] = amax_ij_1
                    else:
                        pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                        max_new_1 = pypto.maximum(max_update_1, amax_ij_1)
                        t1_1 = pypto.sub(max_update_1, max_new_1)
                        t2_1 = pypto.exp(t1_1)
                        t6_1 = pypto.mul(t2_1, sum_update_1)
                        t3_1 = pypto.sub(amax_ij_1, max_new_1)
                        t4_1 = pypto.exp(t3_1)
                        t5_1 = pypto.mul(t4_1, sum_local_1)
                        sum_new_1 = pypto.add(t6_1, t5_1)

                        pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                        oi_last_1 = pypto.mul(oi_update_1, t2_1)
                        oi_flash_1 = pypto.mul(mm2_res_1, t4_1)
                        oi_tmp_1 = pypto.add(oi_last_1, oi_flash_1)
                        if pypto.is_loop_end(s2_idx):
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_update_tmp_1 = pypto.div(
                                oi_tmp_1, sum_new_1, precision_type=pypto.PrecisionType.INTRINSIC
                            )
                            oi_update_tmp_3d_1 = pypto.reshape(oi_update_tmp_1, [1, g_tile, dn])
                            pypto.set_vec_tile_shapes(1, v2_tile[0], v2_tile[1])
                            oi_update_3d_1 = pypto.cast(oi_update_tmp_3d_1, dtype)
                            pypto.assemble(oi_update_3d_1, oi_ofs_1, atten_out)
                        else:
                            oi_update_1[:] = oi_tmp_1
                            sum_update_1[:] = sum_new_1
                            max_update_1[:] = max_new_1
                    pypto.set_pass_options(sg_set_scope=-1)


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 512,
        "device_sched_mode": 1,
        "ready_on_host_tensors": ["block_table", "kv_act_seqs"],
    },
    pass_options={"cube_l1_reuse_setting": {-1: 8}, "cube_nbuffer_setting": {-1: 4}},
)
def ifa_func_kernel_for_950(
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    block_table: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_INT32),
    kv_act_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    atten_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    softmax_scale,
    tile_config,
):
    pypto.experimental.set_operation_options(combine_axis=True)

    shape_q = q.shape
    shape_k = k.shape
    shape_act_seqs = kv_act_seqs.shape
    bs_scalar = shape_q[0]
    nq = shape_q[1]
    block_num_scalar = shape_k[0]
    block_size = shape_k[1]
    nkv = shape_k[2]
    dn = shape_k[3]
    b_scalar = shape_act_seqs[0]

    dtype = q.dtype
    group = nq // nkv
    n2_sym = nkv

    g_tile = tile_config.g_tile
    s2_tile = tile_config.s2_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape

    s1_scalar = bs_scalar // b_scalar
    g = nq // nkv
    g_loop = g // g_tile

    k_2d_shape = (block_num_scalar * block_size, n2_sym * dn)
    q_2d_shape = (b_scalar * s1_scalar * nq, dn)

    k_2d = pypto.reshape(k, k_2d_shape, inplace=True)
    v_2d = pypto.reshape(v, k_2d_shape, inplace=True)
    q_2d = pypto.reshape(q, q_2d_shape, inplace=True)
    for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"):
        for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx"):
            cur_seq = kv_act_seqs[b_idx] - (s1_scalar - 1 - s1_idx)
            s2_loop = (cur_seq + s2_tile - 1) // s2_tile
            for n2_idx in pypto.loop(n2_sym, name="LOOP_n2", idx_name="n2_idx"):
                for g_idx in pypto.loop(g_loop, name="LOOP_g", idx_name="g_idx"):
                    oi_update = pypto.tensor([g_tile, dn], pypto.DT_FP32, "oi_update")
                    sum_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update")
                    max_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update")
                    for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx", unroll_list=[8, 4, 2, 1]):
                        block_num = s2_tile // block_size
                        idx = s2_idx * block_num
                        bs_ofs = b_idx * s1_scalar + s1_idx
                        n1g_ofs = n2_idx * group + g_idx * g_tile
                        actual_s2_tile = (cur_seq - s2_idx * s2_tile).min(s2_tile)
                        oi_ofs = [bs_ofs, n1g_ofs, 0]
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        qi = pypto.view(q_2d, [g_tile, dn], [bs_ofs * nq + n1g_ofs, 0])
                        kj_assemble = pypto.tensor([s2_tile, dn], k_2d.dtype, "kj_assemble")
                        for i in range(block_num):
                            block_idx = block_table[b_idx, idx + i]
                            block_idx_vaild = block_idx.max(0)
                            kj_assemble[i * block_size:(i + 1) * block_size, 0:] = pypto.view(
                                k_2d, [block_size, dn], [block_idx_vaild * block_size, n2_idx * dn]
                            )
                        kj_assemble = pypto.view(kj_assemble, [s2_tile, dn], [0, 0], valid_shape=[s2_tile, dn])

                        pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
                        pypto.set_pass_options(sg_set_scope=5001)
                        sij = pypto.matmul(qi, kj_assemble, pypto.DT_FP32, a_trans=False, b_trans=True)
                        sij = pypto.view(sij, [g_tile, s2_tile], [0, 0], valid_shape=[g_tile, actual_s2_tile])
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        sij_scale = pypto.mul(sij, softmax_scale)
                        amax_ij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                        tsub = pypto.sub(sij_scale, amax_ij)
                        vec1_res = pypto.exp(tsub)
                        vec1_res_fp16 = pypto.cast(vec1_res, dtype)
                        sum_local = pypto.sum(vec1_res, dim=-1, keepdim=True)

                        vj_assemble = pypto.tensor([s2_tile, dn], v_2d.dtype, "vj_assemble")
                        for i in range(block_num):
                            block_idx = block_table[b_idx, idx + i]
                            block_idx_vaild = block_idx.max(0)
                            vj_assemble[i * block_size:(i + 1) * block_size, 0:] = pypto.view(
                                v_2d, [block_size, dn], [block_idx_vaild * block_size, n2_idx * dn]
                            )
                        vj_assemble = pypto.view(vj_assemble, [s2_tile, dn], [0, 0], valid_shape=[actual_s2_tile, dn])
                        pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                        mm2_res = pypto.matmul(vec1_res_fp16, vj_assemble, pypto.DT_FP32)
                        pypto.set_pass_options(sg_set_scope=-1)

                        if pypto.is_loop_begin(s2_idx):
                            pypto.set_pass_options(sg_set_scope=2)
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_tmp = mm2_res
                            oi_update[:] = pypto.tensor(oi_tmp.shape, pypto.DT_FP32, "oi_update")
                            if pypto.is_loop_end(s2_idx):
                                oi_update[:] = pypto.div(
                                    oi_tmp, sum_local, precision_type=pypto.PrecisionType.INTRINSIC
                                )
                                pypto.set_vec_tile_shapes(16, v2_tile[0], v2_tile[1])
                                oi_update_3d = pypto.cast(pypto.reshape(oi_update, [1, g_tile, dn]), dtype)
                                pypto.assemble(oi_update_3d, oi_ofs, atten_out)
                            else:
                                oi_update[:] = oi_tmp
                                sum_update[:] = sum_local
                                max_update[:] = amax_ij
                            pypto.set_pass_options(sg_set_scope=-1)
                        else:
                            pypto.set_pass_options(sg_set_scope=1)
                            pypto.set_vec_tile_shapes(v2_tile[0], 128)
                            max_new = pypto.maximum(max_update, amax_ij)
                            t1 = pypto.sub(max_update, max_new)
                            t2 = pypto.exp(t1)
                            t6 = pypto.mul(t2, sum_update)
                            t3 = pypto.sub(amax_ij, max_new)
                            t4 = pypto.exp(t3)
                            t5 = pypto.mul(t4, sum_local)
                            sum_new = pypto.add(t6, t5)
                            sum_update[:] = sum_new
                            max_update[:] = max_new

                            pypto.set_vec_tile_shapes(v2_tile[0], 128)
                            oi_last = pypto.mul(oi_update, t2)
                            oi_flash = pypto.mul(mm2_res, t4)
                            oi_tmp = pypto.add(oi_last, oi_flash)
                            if pypto.is_loop_end(s2_idx):
                                pypto.set_vec_tile_shapes(16, v2_tile[0], v2_tile[1])
                                oi_update_tmp = pypto.div(
                                    oi_tmp, sum_update, precision_type=pypto.PrecisionType.INTRINSIC
                                )
                                oi_update_3d = pypto.cast(pypto.reshape(oi_update_tmp, [1, g_tile, dn]), dtype)
                                pypto.assemble(oi_update_3d, oi_ofs, atten_out)
                            else:
                                oi_update[:] = oi_tmp
                            pypto.set_pass_options(sg_set_scope=-1)


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 512,
        "device_sched_mode": 1,
        "ready_on_host_tensors": ["block_table", "kv_act_seqs"],
        "max_workspace_kb": 428672,
    },
    pass_options={
        "cube_l1_reuse_setting": {0: 16, 1: 8},
        "cube_nbuffer_setting": {0: 2, 1: 4},
        "vec_nbuffer_setting": {-2: 1, 0: 1, 1: 1},
    },
    host_options={
        "compile_monitor_enable": 0,
    },
)
def ifa_func_kernel_for_950_high_through(
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    block_table: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_INT32),
    kv_act_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    atten_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    softmax_scale,
    tile_config,
):
    pypto.experimental.set_operation_options(combine_axis=True)

    shape_q = q.shape
    shape_k = k.shape
    shape_act_seqs = kv_act_seqs.shape
    bs_scalar = shape_q[0]
    nq = shape_q[1]
    block_num_scalar = shape_k[0]
    block_size = shape_k[1]
    nkv = shape_k[2]
    dn = shape_k[3]
    b_scalar = shape_act_seqs[0]

    dtype = q.dtype
    group = nq // nkv
    n2_sym = nkv

    g_tile = tile_config.g_tile
    s2_tile = tile_config.s2_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape

    s1_scalar = bs_scalar // b_scalar
    g = nq // nkv
    g_loop = g // g_tile

    k_2d_shape = (block_num_scalar * block_size, n2_sym * dn)
    q_2d_shape = (b_scalar * s1_scalar * nq, dn)

    k_2d = pypto.reshape(k, k_2d_shape, inplace=True)
    v_2d = pypto.reshape(v, k_2d_shape, inplace=True)
    q_2d = pypto.reshape(q, q_2d_shape, inplace=True)
    for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"):
        for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx"):
            cur_seq = kv_act_seqs[b_idx] - (s1_scalar - 1 - s1_idx)
            s2_loop = (cur_seq + s2_tile - 1) // s2_tile
            for n2_idx in pypto.loop(n2_sym, name="LOOP_n2", idx_name="n2_idx"):
                for g_idx in pypto.loop(g_loop, name="LOOP_g", idx_name="g_idx"):
                    oi_update = pypto.tensor([g_tile, dn], pypto.DT_FP32, "oi_update")
                    sum_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update")
                    max_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update")
                    for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx", unroll_list=[8, 4, 2, 1]):
                        block_num = s2_tile // block_size
                        idx = s2_idx * block_num
                        bs_ofs = b_idx * s1_scalar + s1_idx
                        n1g_ofs = n2_idx * group + g_idx * g_tile
                        actual_s2_tile = (cur_seq - s2_idx * s2_tile).min(s2_tile)
                        oi_ofs = [bs_ofs, n1g_ofs, 0]
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        qi = pypto.view(q_2d, [g_tile, dn], [bs_ofs * nq + n1g_ofs, 0])
                        kj_assemble = pypto.tensor([s2_tile, dn], k_2d.dtype, "kj_assemble")
                        for i in range(block_num):
                            block_idx = block_table[b_idx, idx + i]
                            block_idx_vaild = block_idx.max(0)
                            kj_assemble[i * block_size:(i + 1) * block_size, 0:] = pypto.view(
                                k_2d, [block_size, dn], [block_idx_vaild * block_size, n2_idx * dn]
                            )
                        kj_assemble = pypto.view(kj_assemble, [s2_tile, dn], [0, 0], valid_shape=[s2_tile, dn])

                        pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
                        pypto.set_pass_options(sg_set_scope=5001)
                        sij = pypto.matmul(qi, kj_assemble, pypto.DT_FP32, a_trans=False, b_trans=True)
                        sij = pypto.view(sij, [g_tile, s2_tile], [0, 0], valid_shape=[g_tile, actual_s2_tile])
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        sij_scale = pypto.mul(sij, softmax_scale)
                        amax_ij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                        tsub = pypto.sub(sij_scale, amax_ij)
                        vec1_res = pypto.exp(tsub)
                        vec1_res_fp16 = pypto.cast(vec1_res, dtype)
                        sum_local = pypto.sum(vec1_res, dim=-1, keepdim=True)

                        vj_assemble = pypto.tensor([s2_tile, dn], v_2d.dtype, "vj_assemble")
                        for i in range(block_num):
                            block_idx = block_table[b_idx, idx + i]
                            block_idx_vaild = block_idx.max(0)
                            vj_assemble[i * block_size:(i + 1) * block_size, 0:] = pypto.view(
                                v_2d, [block_size, dn], [block_idx_vaild * block_size, n2_idx * dn]
                            )
                        vj_assemble = pypto.view(vj_assemble, [s2_tile, dn], [0, 0], valid_shape=[actual_s2_tile, dn])
                        pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                        mm2_res = pypto.matmul(vec1_res_fp16, vj_assemble, pypto.DT_FP32)

                        if pypto.is_loop_begin(s2_idx):
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_tmp = mm2_res
                            oi_update[:] = pypto.tensor(oi_tmp.shape, pypto.DT_FP32, "oi_update")
                            if pypto.is_loop_end(s2_idx):
                                oi_update[:] = pypto.div(
                                    oi_tmp, sum_local, precision_type=pypto.PrecisionType.INTRINSIC
                                )
                                oi_update_3d = pypto.reshape(oi_update, [1, g_tile, dn])
                                pypto.set_vec_tile_shapes(1, v2_tile[0], v2_tile[1])
                                oi_update_3d = pypto.cast(oi_update_3d, dtype)
                                pypto.assemble(oi_update_3d, oi_ofs, atten_out)
                            else:
                                oi_update[:] = oi_tmp
                                sum_update[:] = sum_local
                                max_update[:] = amax_ij
                        else:
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            max_new = pypto.maximum(max_update, amax_ij)
                            t1 = pypto.sub(max_update, max_new)
                            t2 = pypto.exp(t1)
                            t6 = pypto.mul(t2, sum_update)
                            t3 = pypto.sub(amax_ij, max_new)
                            t4 = pypto.exp(t3)
                            t5 = pypto.mul(t4, sum_local)
                            sum_new = pypto.add(t6, t5)
                            sum_update[:] = sum_new
                            max_update[:] = max_new

                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_last = pypto.mul(oi_update, t2)
                            oi_flash = pypto.mul(mm2_res, t4)
                            oi_tmp = pypto.add(oi_last, oi_flash)
                            if pypto.is_loop_end(s2_idx):
                                pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                oi_update_tmp = pypto.div(
                                    oi_tmp, sum_update, precision_type=pypto.PrecisionType.INTRINSIC
                                )
                                oi_update_tmp_3d = pypto.reshape(oi_update_tmp, [1, g_tile, dn])
                                pypto.set_vec_tile_shapes(1, v2_tile[0], v2_tile[1])
                                oi_update_3d = pypto.cast(oi_update_tmp_3d, dtype)
                                pypto.assemble(oi_update_3d, oi_ofs, atten_out)
                            else:
                                oi_update[:] = oi_tmp
                        pypto.set_pass_options(sg_set_scope=-1)


@allow_in_graph
def attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    actual_seqs: torch.Tensor,
    attn_res: torch.Tensor,
    softmax_scale,
    tile_config,
) -> None:
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
        tile_config: IfaTileShapeConfig object containing tiling parameters

    Note:
        This function is decorated with @allow_in_graph to enable integration
        with PyTorch's compilation graph.
    """
    if isinstance(query, FakeTensor):
        return
    check_args(query, key_cache, value_cache, block_tables, actual_seqs, attn_res)

    inputs = [query, key_cache, value_cache, block_tables, actual_seqs, attn_res]
    for _ in range(1):
        ifa_func_kernel(*inputs, softmax_scale, tile_config)


@allow_in_graph
def attention_for_910_high_performance(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    actual_seqs: torch.Tensor,
    attn_res: torch.Tensor,
    softmax_scale,
    tile_config,
) -> None:
    if isinstance(query, FakeTensor):
        return
    check_args(query, key_cache, value_cache, block_tables, actual_seqs, attn_res)

    inputs = [query, key_cache, value_cache, block_tables, actual_seqs, attn_res]
    for _ in range(1):
        a = torch.randn((int(192 * 1024 * 1024 * 2.5))).to(torch.float32).npu()
        for _ in range(10):
            _a_max = torch.max(a)
        ifa_func_kernel_for_910_high_performance(*inputs, softmax_scale, tile_config)


@allow_in_graph
def attention_for_910_s1_range_high_performance(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    actual_seqs: torch.Tensor,
    attn_res: torch.Tensor,
    softmax_scale,
    tile_config,
) -> None:
    if isinstance(query, FakeTensor):
        return
    check_args(query, key_cache, value_cache, block_tables, actual_seqs, attn_res)

    inputs = [query, key_cache, value_cache, block_tables, actual_seqs, attn_res]
    for _ in range(1):
        a = torch.randn((int(192 * 1024 * 1024 * 2.5))).to(torch.float32).npu()
        for _ in range(10):
            _a_max = torch.max(a)
        ifa_func_kernel_for_910_s1_range_high_performance(*inputs, softmax_scale, tile_config)


@allow_in_graph
def attention_for_950(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    actual_seqs: torch.Tensor,
    attn_res: torch.Tensor,
    softmax_scale,
    tile_config,
) -> None:
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
        tile_config: IfaTileShapeConfig object containing tiling parameters

    Note:
        This function is decorated with @allow_in_graph to enable integration
        with PyTorch's compilation graph.
    """
    if isinstance(query, FakeTensor):
        return
    check_args(query, key_cache, value_cache, block_tables, actual_seqs, attn_res)

    inputs = [query, key_cache, value_cache, block_tables, actual_seqs, attn_res]
    for _ in range(1):
        ifa_func_kernel_for_950(*inputs, softmax_scale, tile_config)


@allow_in_graph
def attention_for_950_high_through(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    actual_seqs: torch.Tensor,
    attn_res: torch.Tensor,
    softmax_scale,
    tile_config,
) -> None:
    if isinstance(query, FakeTensor):
        return
    check_args(query, key_cache, value_cache, block_tables, actual_seqs, attn_res)

    inputs = [query, key_cache, value_cache, block_tables, actual_seqs, attn_res]
    for _ in range(1):
        ifa_func_kernel_for_950_high_through(*inputs, softmax_scale, tile_config)
