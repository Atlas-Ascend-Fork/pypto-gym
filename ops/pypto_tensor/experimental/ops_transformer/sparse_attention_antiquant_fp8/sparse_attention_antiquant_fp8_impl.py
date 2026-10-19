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
from dataclasses import dataclass

import pypto
from pypto.experimental import gather_in_ub


@dataclass
class SaTileShapeConfig:
    g_tile: int
    s_kv_tile: int
    c1_tile_shape: list
    v1_tile_shape: list
    c2_tile_shape: list
    v2_tile_shape: list


def sparse_attention_antiquant_compute_950(
    query_nope,
    query_rope,
    nope_cache,
    topk_indices,
    block_table,
    kv_act_seqs,
    attention_out,
    nq,
    n_kv,
    softmax_scale,
    topk,
    block_size,
    max_blocknum_perbatch,
    tile_config,
):
    """Compute sparse flash attention with quantization support.

    Performs attention computation on top-k selected key-value pairs from cache.
    The function processes queries and keys in batches, computing attention scores
    and aggregating values. Supports both quantized (FP8) and non-quantized keys.

    Args:
        query_nope: Query tensor without RoPE, shape (t * n_q, kv_lora_rank), dtype BF16
        query_rope: Query tensor with RoPE, shape (t * n_q, rope_dim), dtype BF16
        nope_cache: Key tensor without RoPE, Key tensor with RoPE, Dequantization scales for quantized keys,
                    shape (block_num * block_size, kv_lora_rank + rope_dim*2 + 4*4),
                    dtype FP8
        topk_indices: Top-k indices for each query token, shape (t, n_kv * topk), dtype INT32
        block_table: Block mapping table for PagedAttention, shape (b, max_blocknum_perbatch),
                     dtype INT32
        kv_act_seqs: Actual sequence lengths for each batch, shape (b,), dtype INT32
        attention_out: Output attention tensor, shape (b, s, n_q, kv_lora_rank), dtype BF16
        nq: Number of query heads
        n_kv: Number of key-value heads
        softmax_scale: Scaling factor for attention scores, typically 1/sqrt(head_dim)
        topk: Number of top-k keys to attend to
        block_size: Size of each block in PagedAttention
        max_blocknum_perbatch: Maximum number of blocks per batch
        tile_config: SaTileShapeConfig object containing tiling parameters:
            - g_tile: Group tile size
            - s_kv_tile: Key-value sequence tile size
            - c1_tile_shape: Cube tile shape for first matmul
            - v1_tile_shape: Vector tile shape for softmax
            - c2_tile_shape: Cube tile shape for second matmul

    Note:
        The function uses nested loops to process batches, sequences, heads, and groups.
        For quantized keys, it performs dequantization before attention computation.
        The attention computation uses standard softmax normalization.
    """
    dtype = query_nope.dtype
    dn = query_nope.shape[1]
    dr = query_rope.shape[1]
    group = nq // n_kv
    group_tile = tile_config.g_tile
    s2_tile = tile_config.s_kv_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape
    v2_2_tile = [32, 512]

    n_kv_sym = n_kv

    batch_size_sym = kv_act_seqs.shape[0]

    s1_n2_gsym = query_nope.shape[0] // batch_size_sym
    s1_sym = s1_n2_gsym // nq

    g_loop_sym = (group + group_tile - 1) // group_tile

    for batch_idx in pypto.loop(0, batch_size_sym, 1, name="LOOP_L0_idx", idx_name="bIdx"):
        cur_act_seq = kv_act_seqs[batch_idx]
        for slc_idx in pypto.loop(0, s1_sym, 1, name="LOOP_L1_s1_SA", idx_name="s1Idx"):
            cur_seq = (cur_act_seq - s1_sym + 1 + slc_idx).max(0).min(topk)
            bn_per_batch = (cur_seq + s2_tile - 1) // s2_tile

            for n_kv_idx in pypto.loop(0, n_kv_sym, 1, name="LOOP_L2_n_kv_SA", idx_name="n_kvIdx"):
                for group_idx in pypto.loop(0, g_loop_sym, 1, name="LOOP_L3_g_SA", idx_name="gIdx"):
                    cur_group_tile = pypto.min(group, group_tile)
                    cur_offset = batch_idx * s1_n2_gsym + slc_idx * nq + n_kv_idx * group + group_idx * cur_group_tile

                    oi_update = pypto.tensor([cur_group_tile, dn], pypto.DT_FP32, "oi_update")
                    sum_update = pypto.tensor([cur_group_tile, 1], pypto.DT_FP32, "sum_update")
                    max_update = pypto.tensor([cur_group_tile, 1], pypto.DT_FP32, "max_update")

                    for s2_idx in pypto.loop(bn_per_batch, name="LOOP_L4_s2_SA", idx_name="s2_idx", unroll_list=[4]):
                        cur_s2_tile = s2_tile

                        pypto.set_pass_options(sg_set_scope=20001)
                        cur_topk_indices = pypto.view(topk_indices, [1, cur_s2_tile],
                                                [batch_idx * s1_sym + slc_idx, s2_idx * cur_s2_tile],
                                                valid_shape=[1, (cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile)])
                        cur_block_table = pypto.view(block_table, [1, max_blocknum_perbatch], [batch_idx, 0])

                        pypto.set_vec_tile_shapes(32, 656)
                        cache_view = pypto.view(nope_cache, [nope_cache.shape[0], dn + dr * 2 + 4 * 4],
                            [0, 0], valid_shape=[nope_cache.shape[0], dn + dr * 2 + 4 * 4])
                        gathered = gather_in_ub(cache_view, cur_topk_indices, cur_block_table, block_size, -2)

                        kn_quant = pypto.view(gathered, [s2_tile, dn], [0, 0],
                            valid_shape=[s2_tile, dn])

                        pypto.set_vec_tile_shapes(32, 512)
                        kn_quant_fp32 = pypto.cast(kn_quant, pypto.DT_FP32)
                        kn_quant_fp32_tmp = pypto.reshape(kn_quant_fp32, [s2_tile * 4, 128])

                        kn_scale_vfp8 = pypto.view(gathered, [s2_tile, 4 * 4], [0, dn + dr * 2],
                            valid_shape=[s2_tile, 4 * 4])
                        kn_scale = pypto.view(kn_scale_vfp8, dtype=pypto.DT_FP32)
                        kn_scale_tmp = pypto.reshape(kn_scale, [s2_tile * 4, 1])

                        pypto.set_vec_tile_shapes(128, 128)
                        kn_fp32 = pypto.mul(kn_quant_fp32_tmp, kn_scale_tmp)
                        kn_fp32_reshape = pypto.reshape(kn_fp32, [s2_tile, dn])
                        pypto.set_vec_tile_shapes(32, 512)
                        cur_kn_fp32 = pypto.view(kn_fp32_reshape, [cur_s2_tile, dn], [0, 0],
                            valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn])
                        kn = pypto.cast(cur_kn_fp32, dtype)

                        kr_vfp8 = pypto.view(gathered, [s2_tile, dr * 2], [0, dn],
                            valid_shape=[s2_tile, dr * 2])
                        kr = pypto.view(kr_vfp8, dtype=dtype)

                        kj = pypto.Tensor([cur_s2_tile, dn + dr], dtype, "kj")
                        pypto.assemble(kn, [0, 0], kj)
                        pypto.assemble(pypto.clone(kr), [0, dn], kj)
                        kj_view = pypto.view(kj, [cur_s2_tile, dn + dr], [0, 0],
                            valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn + dr])

                        qn = pypto.view(query_nope, [cur_group_tile, dn], [cur_offset, 0],
                            valid_shape=[cur_group_tile, dn])
                        qr = pypto.view(query_rope, [cur_group_tile, dr], [cur_offset, 0],
                            valid_shape=[cur_group_tile, dr])
                        qi = pypto.Tensor([cur_group_tile, dn + dr], dtype, "qi")
                        pypto.assemble(qn, [0, 0], qi)
                        pypto.assemble(qr, [0, dn], qi)

                        # C1
                        pypto.set_cube_tile_shapes([c1_tile[0],
                            c1_tile[1]], [c1_tile[2], c1_tile[3]], [c1_tile[4], c1_tile[5]])
                        sij = pypto.matmul(qi, kj_view, pypto.DT_FP32, a_trans=False, b_trans=True)

                        # V1: online softmax (no div)
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        sij_scale = pypto.mul(sij, softmax_scale)
                        tilda_mij_reduce = pypto.amax(sij_scale, dim=-1, keepdim=True)
                        t_sub = pypto.sub(sij_scale, tilda_mij_reduce)
                        tilda_pij = pypto.exp(t_sub)
                        tilda_pij_f16 = pypto.cast(tilda_pij, dtype)
                        sum_local = pypto.sum(tilda_pij, dim=-1, keepdim=True)

                        # C2
                        pypto.set_cube_tile_shapes([c2_tile[0],
                            c2_tile[1]], [c2_tile[2], c2_tile[3]], [c2_tile[4], c2_tile[5]])
                        pypto.set_matrix_size([tilda_pij_f16.shape[0], tilda_pij_f16.shape[1], kn.shape[1]])
                        vj = pypto.view(kj_view, [cur_s2_tile, dn], [0, 0],
                            valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn])
                        q1 = pypto.matmul(tilda_pij_f16, vj, pypto.DT_FP32)
                        pypto.set_pass_options(sg_set_scope=-1)

                        # V2: online softmax update
                        pypto.set_pass_options(sg_set_scope=1)
                        if pypto.cond(pypto.is_loop_begin(s2_idx)):
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_tmp = q1
                            if pypto.cond(pypto.is_loop_end(s2_idx)):
                                oi_update[:] = pypto.div(oi_tmp, sum_local, pypto.PrecisionType.INTRINSIC)
                                oi_final = pypto.cast(oi_update, dtype)
                                pypto.assemble(oi_final, [cur_offset, 0], attention_out)
                            else:
                                oi_update[:] = oi_tmp
                                sum_update[:] = sum_local
                                max_update[:] = tilda_mij_reduce
                        else:
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            max_new = pypto.maximum(max_update, tilda_mij_reduce)

                            t1 = pypto.sub(max_update, max_new)
                            t2 = pypto.exp(t1)
                            t6 = pypto.mul(t2, sum_update)
                            t3 = pypto.sub(tilda_mij_reduce, max_new)

                            pypto.set_vec_tile_shapes(v2_2_tile[0], v2_2_tile[1])
                            t4 = pypto.exp(t3)
                            t5 = pypto.mul(t4, sum_local)
                            sum_new = pypto.add(t6, t5)
                            sum_update[:] = sum_new
                            max_update[:] = max_new

                            oi_last = pypto.mul(oi_update, t2)
                            oi_flash = pypto.mul(q1, t4)
                            oi_tmp = pypto.add(oi_last, oi_flash)
                            if pypto.cond(pypto.is_loop_end(s2_idx)):
                                oi_update[:] = pypto.div(oi_tmp, sum_update, pypto.PrecisionType.INTRINSIC)
                                oi_final = pypto.cast(oi_update, dtype)
                                pypto.assemble(oi_final, [cur_offset, 0], attention_out)
                            else:
                                oi_update[:] = oi_tmp
                        pypto.set_pass_options(sg_set_scope=-1)


@pypto.frontend.jit(
    pass_options={
                "ooo_sched_mode": "GAPMIN",
                "vec_nbuffer_setting": {"DEFAULT": 1},
                "cube_l1_reuse_setting": {-1: 1},
                "cube_nbuffer_setting": {-1: 1},
            },
    runtime_options={
                "stitch_function_max_num": 128,
                "device_sched_mode": 1,
                "ready_on_host_tensors": ["block_table", "kv_act_seqs"],
                "max_workspace_kb": 1607648,
            },
    host_options={"compile_monitor_enable": 0},
)
def sparse_attention_antiquant_fp8_high(
    query_nope: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    query_rope: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    nope_cache: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP8E4M3),
    topk_indices: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32),
    block_table: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32),
    kv_act_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    attention_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),

    nq, n_kv, softmax_scale, topk, block_size, max_blocknum_perbatch, tile_config
):
    pypto.experimental.set_operation_options(combine_axis=True)

    sparse_attention_antiquant_compute_950(
        query_nope,
        query_rope,
        nope_cache,
        topk_indices,
        block_table,
        kv_act_seqs,
        attention_out,
        nq,
        n_kv,
        softmax_scale,
        topk,
        block_size,
        max_blocknum_perbatch,
        tile_config,
    )
