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
Sparse compress Flash Attention Module

This module implements sparse flash attention with quantization support for DeepSeek V4.
It performs attention computation on top-k selected key-value pairs from cache,
supporting both standard and flash attention algorithms.

Main Functions:
    - sparse_compress_flash_attention_compute: Standard sparse attention computation
    - sparse_compress_flash_attention_flash: Flash attention variant with online softmax
    - sparse_compress_flash_attention_d: JIT-compiled decode version

Example:
    See test_sparse_compress_flash_attention.py for usage examples.
"""
import logging
from dataclasses import dataclass
import collections
import pypto
import torch
from pypto.experimental import gather_in_ub
from torch._dynamo import allow_in_graph
from torch._subclasses.fake_tensor import FakeTensor

MAX_S2 = 131072



SCFANpuInputs = collections.namedtuple('SCFANpuInputs', [
    'query_npu', 'q_act_seqs_npu', 'ori_kv_npu', 'cmp_kv_npu',
    'ori_block_table_npu', 'cmp_block_table_npu', 'atten_sink_npu',
    'seqused_kv_npu', 'cmp_sparse_indices_npu', 'softmax_scale',
    'win_size', 'cmp_ratio',
])


@dataclass
class SCFATileShapeConfig:
    g_tile: int
    c1_tile_shape: list
    v1_tile_shape: list
    c2_tile_shape: list


def sparse_compress_flash_attention_compute(query, actual_seq_q, ori_kv, cmp_kv, ori_block_table,
                                            cmp_block_table, atten_sink, seqused_kv,
                                            cmp_sparse_indices, attention_out,
                                            nq, n_kv, softmax_scale, topk,
                                            block_size, win_size, cmp_ratio, tile_config):
    """Compute sparse compress flash attention for prefill.
    """
    dtype = query.dtype
    d = query.shape[1]
    group_tile = tile_config.g_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape

    batch_size_sym = seqused_kv.shape[0]
    topk_tile = topk
    sel_tile = win_size * 2 + topk_tile
    kv_tile = win_size + topk_tile
    pypto.experimental.set_operation_options(combine_axis=True)

    for batch_idx in pypto.loop(0, batch_size_sym, 1, name="LOOP_L0_idx", idx_name="bIdx"):
        cur_s1 = actual_seq_q[batch_idx + 1] - actual_seq_q[batch_idx]
        ori_act_seq = seqused_kv[batch_idx]
        for slc_idx in pypto.loop(0, cur_s1, 1, name="LOOP_L1_s1_SA", idx_name="s1Idx"):
            cur_len = pypto.max(ori_act_seq - cur_s1 + 1 + slc_idx, 0)
            cur_win_size = pypto.min(cur_len, win_size)
            cur_valid_start_pos = cur_len - cur_win_size
            cur_valid_end_pos = cur_len - 1
            start_block = cur_valid_start_pos // block_size
            start_offset = cur_valid_start_pos % block_size
            end_block = cur_valid_end_pos // block_size
            physical_block_id_0 = ori_block_table[batch_idx, start_block]
            physical_block_id_1 = ori_block_table[batch_idx, end_block]

            cur_topk_size = (ori_act_seq // cmp_ratio - cur_s1 + 1 + slc_idx).max(0).min(topk)
            cur_s2_tile = cur_win_size + cur_topk_size
            cur_group_tile = group_tile

            t_idx = actual_seq_q[batch_idx] + slc_idx
            cur_offset = t_idx * nq

            # V0
            pypto.set_semantic_label("Sa_V0")
            # ---- window select: GM --> UB  [win_tile, d]
            pypto.set_vec_tile_shapes(128, 512)
            kj = pypto.tensor([sel_tile, d], dtype, "kj")
            cur_kv_block_0_size = cur_win_size - start_offset
            kv_block_0 = pypto.view(ori_kv, [win_size, d], [physical_block_id_0 * block_size + start_offset, 0], \
                valid_shape=[cur_kv_block_0_size, d])
            pypto.assemble(pypto.clone(kv_block_0), [0, 0], kj)
            if pypto.cond(start_block < end_block):
                pypto.set_vec_tile_shapes(128, 512)
                cur_kv_block_1_size = cur_win_size - cur_kv_block_0_size
                kv_block_1 = pypto.view(ori_kv, [win_size, d], [physical_block_id_1 * block_size, 0], \
                    valid_shape=[cur_kv_block_1_size, d])
                pypto.assemble(pypto.clone(kv_block_1), [cur_kv_block_0_size, 0], kj)

            # ---- gather: GM --> UB  [topk_tile, d]
            pypto.set_vec_tile_shapes(128, 512)
            cur_cmp_sparse_indices = pypto.view(cmp_sparse_indices, [1, topk_tile], [t_idx, 0], \
                valid_shape=[1, cur_topk_size])
            cur_block_table = pypto.view(cmp_block_table, [1, MAX_S2 // block_size], [batch_idx, 0])
            cmp_kv_view = pypto.view(cmp_kv, [topk_tile, d], [0, 0], valid_shape=[cur_topk_size, d])
            compress_kv = gather_in_ub(cmp_kv_view, cur_cmp_sparse_indices, cur_block_table, block_size, -2)
            pypto.assemble(compress_kv, [cur_win_size, 0], kj)

            # C1
            pypto.set_semantic_label("Sa_C1")
            pypto.set_cube_tile_shapes([c1_tile[0], c1_tile[1]], [c1_tile[2], c1_tile[3]],
                                        [c1_tile[4], c1_tile[5]])
            qv = pypto.view(query, [cur_group_tile, d], [cur_offset, 0], valid_shape=[cur_group_tile, d])
            kv_after_gather = pypto.view(kj, [kv_tile, d], [0, 0], valid_shape=[cur_s2_tile, d])
            sij = pypto.matmul(qv, kv_after_gather, pypto.DT_FP32, a_trans=False, b_trans=True)

            # V1
            pypto.set_semantic_label("Sa_V1")
            pypto.set_pass_options(sg_set_scope=2)
            pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
            sij_scale = pypto.mul(sij, softmax_scale)
            tilda_mij_reduce = pypto.amax(sij_scale, dim=-1, keepdim=True)
            t_sub = pypto.sub(sij_scale, tilda_mij_reduce)
            tilda_pij = pypto.exp(t_sub)
            tilda_lij_reduce = pypto.sum(tilda_pij, dim=-1, keepdim=True)
            atten_sink_2d = pypto.reshape(atten_sink, [atten_sink.shape[0], 1], inplace=True)
            sink_sub_res = pypto.sub(atten_sink_2d, tilda_mij_reduce)
            sink_exp_res = pypto.exp(sink_sub_res)
            tilda_lij_reduce = pypto.add(tilda_lij_reduce, sink_exp_res)
            t_softmax = pypto.div(tilda_pij, tilda_lij_reduce, precision_type=pypto.PrecisionType.INTRINSIC)
            tilda_pij_f16 = pypto.cast(t_softmax, dtype)
            pypto.set_pass_options(sg_set_scope=-1)

            # C2
            pypto.set_semantic_label("Sa_C2")
            pypto.set_cube_tile_shapes([c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]],
                                        [c2_tile[4], c2_tile[5]])
            pypto.set_matrix_size([tilda_pij_f16.shape[0], tilda_pij_f16.shape[1], kj.shape[1]])
            vj = pypto.view(kj, [kv_tile, d], [0, 0], valid_shape=[cur_s2_tile, d])
            q1 = pypto.matmul(tilda_pij_f16, vj, dtype)

            pypto.assemble(q1, [cur_offset, 0], attention_out)


@pypto.frontend.jit(
    pass_options={
        "cube_l1_reuse_setting": {-1: 2, 0: 8},
        "cube_nbuffer_setting": {-1: 2},
        "vec_nbuffer_setting": {-2: 1, -1: 8},
    },
    runtime_options={
        "stitch_function_max_num": 1024,
        "device_sched_mode": 0
    }
)
def sparse_compress_flash_attention_kernel(
    query: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    actual_seq_q: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32), 
    ori_kv: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16), 
    cmp_kv: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16), 
    ori_block_table: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32), 
    cmp_block_table: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32), 
    atten_sink: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),
    seqused_kv: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32), 
    cmp_sparse_indices: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32),
    attention_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    nq, n_kv, softmax_scale, topk, block_size, win_size, cmp_ratio, tile_config):
    """JIT-compiled sparse compress flash attention for decode phase.
    """
    pypto.experimental.set_operation_options(combine_axis=True)

    sparse_compress_flash_attention_compute(query, actual_seq_q, ori_kv, cmp_kv, ori_block_table,
                                            cmp_block_table, atten_sink, seqused_kv,
                                            cmp_sparse_indices, attention_out,
                                            nq, n_kv, softmax_scale, topk,
                                            block_size, win_size, cmp_ratio, tile_config)


def check_input_output_shape_dtype(query_npu, q_act_seqs_npu, ori_kv_npu, cmp_kv_npu, atten_sink_npu, 
                                   cmp_sparse_indices_npu):
    assert q_act_seqs_npu is not None and q_act_seqs_npu.dim() == 1, \
        f"q_act_seqs_npu dim num is {q_act_seqs_npu.dim()}, expected 1"
    assert query_npu.dim() == 2 and query_npu.size(1) == 512 and query_npu.dtype == torch.bfloat16, \
        f"query dim num is {query_npu.dim()}, query axis 1 is {query_npu.size(1)}, \
            query dtype is {query_npu.dtype}, expected 2, 512, torch.bfloat16"
    assert ori_kv_npu.dim() == 2 and ori_kv_npu.size(1) == 512 and ori_kv_npu.dtype == torch.bfloat16, \
        f"ori_kv_npu dim num is {ori_kv_npu.dim()}, ori_kv_npu axis 1 is {ori_kv_npu.size(1)}, \
            ori_kv_npu dtype is {ori_kv_npu.dtype}, expected 2, 512, torch.bfloat16"
    assert cmp_kv_npu.dim() == 2 and cmp_kv_npu.size(1) == 512 and cmp_kv_npu.dtype == torch.bfloat16, \
        f"cmp_kv_npu dim num is {cmp_kv_npu.dim()}, cmp_kv_npu axis 1 is {cmp_kv_npu.size(1)}, \
            cmp_kv_npu dtype is {cmp_kv_npu.dtype}, expected 2, 512, torch.bfloat16"
    assert atten_sink_npu.dim() == 1 and atten_sink_npu.dtype == torch.float32, \
        f"atten_sink_npu dim num is {atten_sink_npu.dim()}, atten_sink_npu dtype is {atten_sink_npu.dtype}, \
            expected 1, torch.float32"
    assert cmp_sparse_indices_npu.dim() == 2, f"cmp_sparse_indices_npu dim num is {cmp_sparse_indices_npu.dim()}, \
        expected 2"


@allow_in_graph
def npu_sparse_compress_flash_attention(inputs: SCFANpuInputs):
    query_npu = inputs.query_npu
    q_act_seqs_npu = inputs.q_act_seqs_npu
    ori_kv_npu = inputs.ori_kv_npu
    cmp_kv_npu = inputs.cmp_kv_npu
    ori_block_table_npu = inputs.ori_block_table_npu
    cmp_block_table_npu = inputs.cmp_block_table_npu
    atten_sink_npu = inputs.atten_sink_npu
    seqused_kv_npu = inputs.seqused_kv_npu
    cmp_sparse_indices_npu = inputs.cmp_sparse_indices_npu
    softmax_scale = inputs.softmax_scale
    win_size = inputs.win_size
    cmp_ratio = inputs.cmp_ratio

    assert not isinstance(query_npu, FakeTensor), f"query_npu is FakeTensor"
    check_input_output_shape_dtype(query_npu, q_act_seqs_npu, ori_kv_npu, cmp_kv_npu, 
                                   atten_sink_npu, cmp_sparse_indices_npu)

    tile_config = SCFATileShapeConfig(
        g_tile=64,
        c1_tile_shape=[64, 64, 128, 512, 128, 128],
        v1_tile_shape=[32, 640],
        c2_tile_shape=[64, 64, 128, 640, 256, 256]
    )

    attention_out_npu = torch.zeros([query_npu.size(0), query_npu.size(1)], \
        dtype=query_npu.dtype, device=f'{query_npu.device}')

    nq = query_npu.size(0) // cmp_sparse_indices_npu.size(0)
    n_kv = 1
    topk = cmp_sparse_indices_npu.size(1)
    block_size = 128

    tensors = [query_npu, q_act_seqs_npu, ori_kv_npu, cmp_kv_npu, ori_block_table_npu, cmp_block_table_npu, 
        atten_sink_npu, seqused_kv_npu, cmp_sparse_indices_npu, attention_out_npu]

    sparse_compress_flash_attention_kernel(*tensors, nq, n_kv, softmax_scale, topk, block_size, 
                                           win_size, cmp_ratio, tile_config)

    return attention_out_npu


pyptolib = torch.library.Library("pypto", "FRAGMENT")
pyptolib.define("sparse_compress_flash_attention(Tensor query_npu, Tensor q_act_seqs_npu, Tensor ori_kv_npu, \
    Tensor cmp_kv_npu, Tensor ori_block_table_npu,\
    Tensor cmp_block_table_npu, Tensor atten_sink_npu, Tensor seqused_kv_npu, Tensor cmp_sparse_indices_npu,\
    float softmax_scale, int win_size, int cmp_ratio) -> (Tensor)")


@torch.library.impl(pyptolib, "sparse_compress_flash_attention", "Meta")
def sparse_compress_flash_attention(query_npu, q_act_seqs_npu, ori_kv_npu, cmp_kv_npu, ori_block_table_npu,
                                    cmp_block_table_npu, atten_sink_npu,
                                    seqused_kv_npu, cmp_sparse_indices_npu, softmax_scale, win_size, cmp_ratio):
    y = torch.empty([ori_block_table_npu.size(0), cmp_sparse_indices_npu.size(0) // ori_block_table_npu.size(0), \
        query_npu.size(0) // cmp_sparse_indices_npu.size(0), query_npu.size(1)], \
        dtype=query_npu.dtype, device=query_npu.device)
    return y


try:
    @torch.library.impl(pyptolib, "sparse_compress_flash_attention", "NPU")
    def sparse_compress_flash_attention(query_npu, q_act_seqs_npu, ori_kv_npu, cmp_kv_npu, ori_block_table_npu,
                                        cmp_block_table_npu, atten_sink_npu,
                                        seqused_kv_npu, cmp_sparse_indices_npu, softmax_scale, win_size, cmp_ratio):
        inputs = SCFANpuInputs(
            query_npu=query_npu, q_act_seqs_npu=q_act_seqs_npu,
            ori_kv_npu=ori_kv_npu, cmp_kv_npu=cmp_kv_npu,
            ori_block_table_npu=ori_block_table_npu,
            cmp_block_table_npu=cmp_block_table_npu,
            atten_sink_npu=atten_sink_npu, seqused_kv_npu=seqused_kv_npu,
            cmp_sparse_indices_npu=cmp_sparse_indices_npu,
            softmax_scale=softmax_scale, win_size=win_size,
            cmp_ratio=cmp_ratio,
        )
        return npu_sparse_compress_flash_attention(inputs)
except Exception as e:
    if "could not parse dispatch key: NPU" in str(e):
        logging.warning(
            "Skip: torchair not installed, skip NPU registration "
            "for operator 'sparse_compress_flash_attention'"
        )
    else:
        logging.warning(f"Skip: Unexpected error: {e}")


def sparse_compress_flash_attention_graph(inputs: SCFANpuInputs):
    return torch.ops.pypto.sparse_compress_flash_attention(
        inputs.query_npu, inputs.q_act_seqs_npu, inputs.ori_kv_npu,
        inputs.cmp_kv_npu, inputs.ori_block_table_npu,
        inputs.cmp_block_table_npu, inputs.atten_sink_npu,
        inputs.seqused_kv_npu, inputs.cmp_sparse_indices_npu,
        inputs.softmax_scale, inputs.win_size, inputs.cmp_ratio)
