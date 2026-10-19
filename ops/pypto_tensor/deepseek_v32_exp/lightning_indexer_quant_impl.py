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
Lightning Indexer Quantization Module

This module implements lightning indexer with quantization support for DeepSeek V32.
It is a high-performance tensor computation function based on the PyPTO framework,
primarily used for optimizing index calculations during the decoding phase.

Main Functions:
    - lightning_indexer_decode_compute: JIT-compiled decode version

Example:
    See deepseekv32_lightning_indexer_quant.py for usage examples.
"""
import sys
from dataclasses import dataclass
import pypto


@dataclass
class LightningIndexerConfigs:
    # graph optimization params
    # used for copy in merge graph
    mg_copy_in_upper_bound = 2 * 1024 * 1024
    # l1 reuse merge params
    cube_l1_reuse_setting = {
        0: 16
    }
    # vector graph fuse optimization
    vec_merge_mode = 2
    vec_nbuffer_setting = {
        -1: 16
    }
    # tile params
    s1_tile = 2
    topk_tile = 8192
    # set the tileshape size in cube computation
    c1_tile = [64, 64, 128, 128, 128, 128]  # (m, M), (k, K), (n, N)
    c2_tile = [128, 128, 64, 64, 128, 128]  # (m, M), (k, K), (n, N)
    # matmul relu fuse params
    extend_param = {'scale': 1 / 2048.0, 'relu_type': pypto.ReLuType.RELU}


MAX_LI_S1 = 4
MAX_LI_S2 = 128 * 1024

SHAPE_DIM1 = 1
SHAPE_DIM2 = 2


def lightning_indexer_decode_compute(
    idx_query: pypto.tensor,
    idx_query_scale: pypto.tensor,
    idx_key_cache: pypto.tensor,
    idx_key_scale: pypto.tensor,
    idx_weight: pypto.tensor,
    act_seq_key: pypto.tensor,
    block_table: pypto.tensor,
    topk_res: pypto.tensor,
    unroll_list: list,
    configs: LightningIndexerConfigs,
    selected_count: int):

    """Compute lightning indexer with quantization support.
    It obtains the top-k positions corresponding to each token based on a series of operations.
    Args:
        idx_query: Non-contiguous data is not supported, shape (t, n_q, idx_head_dim), dtype INT8.
        idx_query_scale: It represents the scaling factor for idx_query. shape (t, n_q, idx_head_dim), dtype FP16.
        idx_key_cache: Non-contiguous data is not supported, shape (t, n_kv, idx_head_dim), dtype INT8.
        idx_key_scale: It represents the scaling factor for idx_key_cache, shape (t, n_kv, idx_head_dim), dtype FP16.
        idx_weight: Non-contiguous data is not supported. The data format supports ND, shape (t, n_q), dtype FP16.
        act_seq_key: It represents the number of valid tokens for `key` in different batches. shape (b), dtype INT32.
        block_table: It represents the block mapping table used for KV storage in PageAttention,
                    shape (b, ceilDiv(max(s2), block_size), dtype INT32.
        unroll_list: It represents the multi-level tiling configuration.
        configs: It is a LightningIndexerConfigs configuration structure that represents
                tiling configuration and optimization options.
        selected_count: Required parameter. It represents the number of topk selections, with a default value of 2048.
    """

    # vector graph fuse optimization
    pypto.set_pass_options(vec_nbuffer_setting=configs.vec_nbuffer_setting)

    # cube graph fuse optimization
    pypto.set_pass_options(cube_l1_reuse_setting=configs.cube_l1_reuse_setting)

    # get tile params from configs
    s1_tile = configs.s1_tile # s1 need to be divided by s1_tile
    topk_tile = configs.topk_tile
    c1_tile = configs.c1_tile
    c2_tile = configs.c2_tile

    # symbolization of params
    t = idx_query.shape[0]
    b = act_seq_key.shape[0]
    block_num = idx_key_cache.shape[0]

    idx_n_heads = idx_query.shape[SHAPE_DIM1]
    index_d = idx_query.shape[SHAPE_DIM2]
    block_size = idx_key_cache.shape[SHAPE_DIM1]
    qk_dtype = idx_query.dtype
    scale_dtype = idx_query_scale.dtype
    w_dtype = idx_weight.dtype

    s1 = t // b # s1 of each batch is equal in decode process
    s1_loop = (s1 + s1_tile - 1) // s1_tile

    xdtype = pypto.DT_FP32
    dxdtype = pypto.DT_INT32
    pad_idx_value = -1
    descending = True
    pad_value = -sys.float_info.max if descending else sys.float_info.max

    query_2d = pypto.tensor([t * idx_n_heads, index_d], qk_dtype, "query_2d")
    q_scale_3d = pypto.tensor([t, 1, idx_n_heads], scale_dtype, "q_scale_3d")
    key_2d = pypto.tensor([block_num * block_size, index_d], qk_dtype, "key_2d")
    k_scale_2d = pypto.tensor([block_num, block_size], scale_dtype, "k_scale_2d")
    weight_3d = pypto.tensor([t, 1, idx_n_heads], w_dtype, "weight_3d")

    # reshape will not generate real datamove
    query_2d = pypto.reshape(idx_query, [t * idx_n_heads, index_d], inplace=True)
    q_scale_3d = pypto.reshape(idx_query_scale, [t, 1, idx_n_heads], inplace=True)
    key_2d = pypto.reshape(idx_key_cache, [block_num * block_size, index_d], inplace=True)
    weight_3d = pypto.reshape(idx_weight, [t, 1, idx_n_heads], inplace=True)
    k_scale_2d = pypto.reshape(idx_key_scale, [block_num, block_size], inplace=True)

    for b_idx in pypto.loop(0, b, 1, name="LI_LOOP_BATCH", idx_name="b_idx"):
        cur_seq = act_seq_key[b_idx]
        cur_block = (cur_seq + block_size - 1) // block_size
        _last_seq = cur_seq - (cur_block - 1) * block_size
        # static tensor for rawShape assemble
        max_tensor = pypto.tensor([MAX_LI_S1, MAX_LI_S2], pypto.DT_FP32, "max_tensor")
        for s1_tile_idx in pypto.loop(0, s1_loop, 1, name="LI_LOOP_S1", idx_name="s1_loop"):
            w_scale = pypto.tensor([s1_tile, 1, idx_n_heads], pypto.DT_FP16, "w_scale")
            pypto.set_vec_tile_shapes(s1_tile, 1, idx_n_heads)
            cur_qs = pypto.view(q_scale_3d, [s1_tile, 1, idx_n_heads],
                                [b_idx * s1 + s1_tile * s1_tile_idx, 0, 0])
            cur_w = pypto.view(weight_3d, [s1_tile, 1, idx_n_heads],
                                [b_idx * s1 + s1_tile * s1_tile_idx, 0, 0])
            w_scale = pypto.mul(cur_qs, cur_w) # (s1_tile, 1, idx_n_heads), fp16 * fp16
            q_offset = b_idx * s1 * idx_n_heads + s1_tile_idx * s1_tile * idx_n_heads
            for bn_idx, unroll_loop in pypto.loop_unroll(
                0, cur_block, 1, name="LOOP_BLOCK_NUM", idx_name="bn_idx", unroll_list=unroll_list,):
                # static unroll into bigger block to reduce tasks
                first_mm_collect = pypto.tensor([s1_tile * idx_n_heads, block_size * unroll_loop],
                                                pypto.DT_FP16, "first_mm_collect")
                for sub_bn_idx in range(unroll_loop):
                    idx_in_block = bn_idx + sub_bn_idx
                    cur_block_idx = block_table[b_idx, idx_in_block]
                    tail_seq = pypto.min(block_size, cur_seq - (idx_in_block * block_size))
                    cur_q = pypto.view(query_2d, [s1_tile * idx_n_heads, index_d], [q_offset, 0])
                    k_block = pypto.view(key_2d, [block_size, index_d], [cur_block_idx * block_size, 0],
                        valid_shape=[tail_seq, index_d]) # (blockSize, indexD)
                    pypto.set_cube_tile_shapes([c1_tile[0], c1_tile[1]], [c1_tile[2],
                                                                          c1_tile[3]], [c1_tile[4], c1_tile[5]])
                    # use fixpipe
                    qk_dot = pypto.matmul(cur_q, k_block, pypto.DT_FP16, a_trans=False, b_trans=True,
                                            extend_params=configs.extend_param) # (s1Tile * idxNHeads, blockSize)
                    pypto.assemble(qk_dot, [0, sub_bn_idx * block_size], first_mm_collect)

                pypto.set_vec_tile_shapes(pypto.min(s1_tile * idx_n_heads, block_size), block_size)
                qk_3d = pypto.reshape(first_mm_collect, [s1_tile, idx_n_heads, unroll_loop * block_size],
                                        inplace=True)
                pypto.set_cube_tile_shapes(
                    [c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]], [c2_tile[4], c2_tile[5]])
                w_qk = pypto.matmul(w_scale, qk_3d, pypto.DT_FP32, a_trans=False, b_trans=False)
                second_mm = pypto.reshape(w_qk, [s1_tile, unroll_loop * block_size], inplace=True)

                ks_assemble = pypto.tensor([1, unroll_loop * block_size], pypto.DT_FP16, "ks_assemble")
                pypto.set_vec_tile_shapes(1, block_size)

                for idx in range(unroll_loop):
                    cur_block_idx = block_table[b_idx, bn_idx + idx]
                    k_s_block = pypto.view(k_scale_2d, [1, block_size], [cur_block_idx, 0],
                            valid_shape=[1, pypto.min(block_size, cur_seq - bn_idx * block_size)])
                    pypto.assemble(pypto.clone(k_s_block), [0, idx * block_size], ks_assemble)
                    pypto.set_vec_tile_shapes(1, 16 * block_size)

                k_res = pypto.mul(second_mm, pypto.cast(ks_assemble, pypto.DT_FP32))
                pypto.assemble(k_res, [s1_tile_idx * s1_tile, bn_idx * block_size], max_tensor)

        for s1_idx in pypto.loop(0, s1, 1, name="LOOP_TOPK_S1", idx_name="s1_idx"):
            # TopK Process
            pypto.set_vec_tile_shapes(1, topk_tile)
            casual_offset = s1 - s1_idx - 1
            eff_seq = cur_seq - casual_offset
            src_offset = s1_idx
            dst_offset = b_idx * s1 + s1_idx
            pad_sc = pypto.tensor([1, selected_count], xdtype, "pad_sc")

            for _ in pypto.loop(eff_seq < selected_count, name="TOPK_LT_SC", idx_name="un_used"):
                # input pad -inf; res_value pad -inf; res_index pad -1
                pypto.set_pass_options(sg_set_scope=1)
                pypto.set_vec_tile_shapes(1, selected_count)
                eff_in = pypto.view(max_tensor, [1, selected_count], [src_offset, 0], valid_shape=[1, eff_seq])
                ax = pypto.view(eff_in, [1, selected_count], [0, 0], valid_shape=[1, eff_seq])
                bx = pypto.full([1, selected_count], pad_value, pypto.DT_FP32,
                                valid_shape=[1, selected_count - eff_seq])
                pypto.assemble(pypto.clone(ax), [0, 0], pad_sc)
                pypto.assemble(bx, [0, eff_seq], pad_sc)
                pypto.set_pass_options(sg_set_scope=-1)
                _, res_index = pypto.topk(pad_sc, k=selected_count, dim=-1, largest=True)
                index_valid = pypto.view(res_index, [1, selected_count], [0, 0], valid_shape=[1, eff_seq])
                pypto.set_vec_tile_shapes(1, 1, selected_count)
                index_3d = pypto.reshape(index_valid, [1, 1, selected_count], valid_shape=[1, 1, eff_seq])
                index_pad = pypto.full([1, 1, selected_count], pad_idx_value, dxdtype,
                            valid_shape=[1, 1, selected_count - eff_seq])
                pypto.assemble(pypto.clone(index_3d), [dst_offset, 0, 0], topk_res)
                pypto.assemble(index_pad, [dst_offset, 0, eff_seq], topk_res)

            for _ in pypto.loop(eff_seq >= selected_count, name="TOPK_GE_SC", idx_name="un_used"):
                pypto.set_vec_tile_shapes(1, topk_tile)
                eff_in = pypto.view(max_tensor, [1, MAX_LI_S2], [src_offset, 0], valid_shape=[1, eff_seq])
                _, res_index = pypto.topk(eff_in, k=selected_count, dim=-1, largest=True)
                pypto.set_vec_tile_shapes(1, 1, topk_tile)
                eff_3d = pypto.reshape(res_index, [1, 1, selected_count])
                pypto.assemble(pypto.clone(eff_3d), [dst_offset, 0, 0], topk_res)


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 128,
        "device_sched_mode": 1,
        "ready_on_host_tensors": ["act_seq_key", "block_table"]
    }
)
def lightning_indexer_decode(
    idx_query: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    idx_query_scale: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP16),
    idx_key_cache: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    idx_key_scale: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    idx_weight: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP16),
    act_seq_key: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    block_table: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32),
    topk_res: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    unroll_list, configs, selected_count
):
    """JIT-compiled Lightning Indexer for decode phase.

    Args:
        idx_query: (t, idx_n_heads, idx_head_dim), dtype INT8
            Query indices for lightning indexing
        idx_query_scale: (t, idx_n_heads, idx_head_dim), dtype FP16
            Quantization scale for idx_query
        idx_key_cache: (block_num, block_size, 1, idx_head_dim), dtype INT8
            Key cache in PageAttention format
        idx_key_scale: (block_num, block_size, 1, idx_head_dim), dtype FP16
            Quantization scale for idx_key_cache
        idx_weight: (t, idx_n_heads), dtype FP16
            Attention weights for indexing
        act_seq_key: (b,), dtype INT32
            Actual sequence length per batch
        block_table: (b, max_blocks), dtype INT32
            Block mapping table for PageAttention
        topk_res: (t, 1, selected_count), dtype INT32
            TopK indices for sparse attention
        unroll_list: Multi-level tiling configuration
        configs: LightningIndexerConfigs configuration
        selected_count: Number of topk selections
    """
    # Call original compute function
    lightning_indexer_decode_compute(
        idx_query,
        idx_query_scale,
        idx_key_cache,
        idx_key_scale,
        idx_weight,
        act_seq_key,
        block_table,
        topk_res,
        unroll_list,
        configs,
        selected_count
    )
