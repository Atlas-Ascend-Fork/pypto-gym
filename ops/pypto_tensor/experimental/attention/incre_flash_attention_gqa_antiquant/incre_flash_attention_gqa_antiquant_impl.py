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

from dataclasses import dataclass, replace

import torch
from torch._dynamo import allow_in_graph

import pypto


@dataclass
class AttentionTileConfig:
    """
    Configuration for tile sizes used in attention computation.

    Tiling is used to break large computations into smaller, cache-friendly chunks.

    Attributes:
        g_tile: Tile size for group dimension
        s2_tile: Tile size for kv sequence dimension
        c1_tile: Tile configuration for first matrix multiplication (Q x K^T)
        v1_tile: Tile configuration for vector operations in first MM
        c2_tile: Tile configuration for second matrix multiplication (Softmax x V)
        v2_tile: Tile configuration for vector operations in second MM
        unroll_list: unroll_list
    """
    g_tile: int
    s2_tile: int
    v0_tile: list
    c1_tile: list
    v1_tile: list
    c2_tile: list
    v2_tile: list
    unroll_list: list


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 513,
        "device_sched_mode": 1,
        "ready_on_host_tensors": ["block_table", "kv_actual_seqs"]
    },
    pass_options={
        "cube_l1_reuse_setting": {-1: 16},
        "vec_nbuffer_setting": {-2: 1, -1: 16},
        "ooo_sched_mode": "GAPMIN",
        "sg_set_tunevf_mode": 1,
    }
)
def incre_flash_attention_gqa_antiquant_kernel(
    query: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    key: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP8E4M3),
    value: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP8E4M3),
    key_antiquant_scale: pypto.Tensor([...], pypto.DT_BF16),
    value_antiquant_scale: pypto.Tensor([...], pypto.DT_BF16),
    kv_actual_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    block_table: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_INT32),
    atten_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    tile_cfg
):
    pypto.experimental.set_operation_options(combine_axis=True)

    dtype = query.dtype
    b, n1, s1, d = query.shape
    block_num, n2, block_size, _ = key.shape
    block_table_shape = block_table.shape
    b = block_table_shape[0]
    group = n1 // n2
    softmax_scale = d ** -0.5

    q_2d_shape = (b * n1, s1 * d)
    kv_2d_shape = (block_num * block_size * n2, d)

    q_2d = pypto.reshape(query, q_2d_shape, inplace=True)
    k_2d = pypto.reshape(key, kv_2d_shape, inplace=True)
    v_2d = pypto.reshape(value, kv_2d_shape, inplace=True)

    g_tile = tile_cfg.g_tile
    group_loop = group // g_tile
    s2_tile = tile_cfg.s2_tile
    v0_tile = tile_cfg.v0_tile
    c1_tile = tile_cfg.c1_tile
    v1_tile = tile_cfg.v1_tile
    c2_tile = tile_cfg.c2_tile
    v2_tile = tile_cfg.v2_tile
    unroll_list = tile_cfg.unroll_list

    block_num = s2_tile // block_size

    for b_idx in pypto.loop(b, name="LOOP_b", idx_name="b_idx", parallel=True):
        for s1_idx in pypto.loop(s1, name="LOOP_s1", idx_name="s1_idx"):
            cur_seq_len = kv_actual_seqs[b_idx] - (s1 - 1 - s1_idx)
            s2_loop = pypto.ceildiv(cur_seq_len, s2_tile)
            bs_ofs = b_idx * s1 + s1_idx
            for n2_idx in pypto.loop(n2, name="LOOP_n2", idx_name="n2_idx"):
                for group_idx in pypto.loop(group_loop, name="LOOP_group_idx", idx_name="group_idx"):
                    out_update = pypto.tensor([g_tile, d], pypto.DT_FP32, "out_update")
                    sum_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update")
                    max_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update")
                    n1g_ofs = n2_idx * group + group_idx * g_tile
                    pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                    for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx", unroll_list=unroll_list):
                        pypto.set_pass_options(sg_set_scope=10001)
                        qi = pypto.view(q_2d, [g_tile, d], [b_idx * n1 + n1g_ofs, s1_idx * d])
                        actual_s2_tile = (cur_seq_len - s2_idx * s2_tile).min(s2_tile)
                        kj_assemble = pypto.tensor([s2_tile, d], k_2d.dtype, "kj_assemble")
                        vj_assemble = pypto.tensor([s2_tile, d], v_2d.dtype, "vj_assemble")
                        idx = s2_idx * block_num
                        for i in range(block_num):
                            block_idx = block_table[b_idx, idx + i]
                            block_idx_valid = block_idx.max(0)
                            kj_assemble[i * block_size:(i + 1) * block_size, 0:] = pypto.view(
                                k_2d, [block_size, d], [(block_idx_valid * n2 + n2_idx) * block_size, 0]
                            )
                            vj_assemble[i * block_size:(i + 1) * block_size, 0:] = pypto.view(
                                v_2d, [block_size, d], [(block_idx_valid * n2 + n2_idx) * block_size, 0]
                            )
                        pypto.set_pass_options(sg_set_ooo_scope=4)
                        kj_assemble = pypto.view(kj_assemble, [s2_tile, d], [0, 0], valid_shape=[actual_s2_tile, d])
                        vj_assemble = pypto.view(vj_assemble, [s2_tile, d], [0, 0], valid_shape=[actual_s2_tile, d])
                        pypto.set_vec_tile_shapes(v0_tile[0], v0_tile[1])
                        kj_fp32 = pypto.cast(kj_assemble, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
                        cur_key_antiquant_scale = key_antiquant_scale[n2_idx]
                        kj_antiquant_scale_fp32 = pypto.cast(
                            cur_key_antiquant_scale, pypto.DT_FP32, pypto.CastMode.CAST_NONE
                        )
                        out_data_fp32 = pypto.mul(kj_fp32, kj_antiquant_scale_fp32)
                        kj_assemble_antiquanted = pypto.cast(out_data_fp32, pypto.DT_BF16, pypto.CastMode.CAST_NONE)
                        vj_fp32 = pypto.cast(vj_assemble, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
                        cur_value_antiquant_scale = value_antiquant_scale[n2_idx]
                        vj_antiquant_scale_fp32 = pypto.cast(
                            cur_value_antiquant_scale, pypto.DT_FP32, pypto.CastMode.CAST_NONE
                        )
                        vj_out_data_fp32 = pypto.mul(vj_fp32, vj_antiquant_scale_fp32)
                        vj_assemble_antiquanted = pypto.cast(vj_out_data_fp32, pypto.DT_BF16, pypto.CastMode.CAST_NONE)
                        pypto.set_pass_options(sg_set_ooo_scope=1)
                        pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
                        sij = pypto.matmul(qi, kj_assemble_antiquanted, pypto.DT_FP32, a_trans=False, b_trans=True)
                        pypto.set_vec_tile_shapes(v2_tile[0], s2_tile)
                        pypto.set_pass_options(sg_set_ooo_scope=1)
                        sij = pypto.view(sij, [g_tile, s2_tile], [0, 0], valid_shape=[g_tile, actual_s2_tile])
                        sij_scale = pypto.mul(sij, softmax_scale)
                        tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                        tsub = pypto.sub(sij_scale, tilda_mij)
                        tilda_pij = pypto.exp(tsub)
                        tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
                        sum_local = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                        pypto.set_pass_options(sg_set_ooo_scope=-1)
                        pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                        mm2_res = pypto.matmul(tilda_pij_fp16, vj_assemble_antiquanted, pypto.DT_FP32)
                        if pypto.is_loop_begin(s2_idx):
                            pypto.set_pass_options(sg_set_ooo_scope=2)
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_tmp = mm2_res
                            out_update[:] = pypto.tensor(oi_tmp.shape, pypto.DT_FP32, "out_update")
                            if pypto.is_loop_end(s2_idx):
                                out_update[:] = pypto.div(oi_tmp, sum_local,
                                                        precision_type=pypto.PrecisionType.INTRINSIC)
                                pypto.set_vec_tile_shapes(1, g_tile, 1, d)
                                oi_final_4d = pypto.cast(pypto.reshape(out_update, [1, g_tile, 1, d]), dtype)
                                out_ofs = [b_idx, n1g_ofs, s1_idx, 0]
                                pypto.assemble(oi_final_4d, out_ofs, atten_out)
                                pypto.set_pass_options(sg_set_ooo_scope=-1)
                            else:
                                out_update[:] = oi_tmp
                                sum_update[:] = sum_local
                                max_update[:] = tilda_mij
                                pypto.set_pass_options(sg_set_ooo_scope=-1)
                        else:
                            pypto.set_pass_options(sg_set_ooo_scope=3)
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            max_new = pypto.maximum(max_update, tilda_mij)
                            t1 = pypto.sub(max_update, max_new)
                            t2 = pypto.exp(t1)
                            t6 = pypto.mul(t2, sum_update)
                            t3 = pypto.sub(tilda_mij, max_new)
                            t4 = pypto.exp(t3)
                            t5 = pypto.mul(t4, sum_local)
                            sum_new = pypto.add(t6, t5)
                            sum_update[:] = sum_new
                            max_update[:] = max_new
                            oi_last = pypto.mul(out_update, t2)
                            oi_flash = pypto.mul(mm2_res, t4)
                            oi_tmp = pypto.add(oi_last, oi_flash)
                            if pypto.is_loop_end(s2_idx):
                                oi_update_tmp = pypto.div(oi_tmp, sum_update,
                                                          precision_type=pypto.PrecisionType.INTRINSIC)
                                pypto.set_vec_tile_shapes(1, v2_tile[0], 1, d)
                                oi_final_4d = pypto.cast(pypto.reshape(oi_update_tmp, [1, g_tile, 1, d]), dtype)
                                out_ofs = [b_idx, n1g_ofs, s1_idx, 0]
                                pypto.assemble(oi_final_4d, out_ofs, atten_out)
                                pypto.set_pass_options(sg_set_ooo_scope=-1)
                            else:
                                out_update[:] = oi_tmp
                                pypto.set_pass_options(sg_set_ooo_scope=-1)
                        pypto.set_pass_options(sg_set_scope=-1)


@allow_in_graph
def incre_flash_attention_gqa_antiquant(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_antiquant_scale: torch.Tensor,
    value_antiquant_scale: torch.Tensor,
    kv_actual_seqs: torch.Tensor,
    block_table: torch.Tensor,
    tile_config
):
    atten_out = torch.zeros_like(query)
    input_values = [query, key, value, key_antiquant_scale, value_antiquant_scale, kv_actual_seqs,
                    block_table, atten_out, tile_config]
    incre_flash_attention_gqa_antiquant_kernel(*input_values)
    return atten_out