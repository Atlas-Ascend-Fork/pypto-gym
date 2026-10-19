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
class MlaConfig:
    """
    Configuration parameters for mla computation.

    Attributes:
        layout: Input layout
        b: Batch size
        n1: Number of query heads
        s1: Query sequence length
        q_d: Query head dimension
        q_rope_d: Query rope dimension
        n2: Number of key/value heads (for grouped query attention)
        s2: Key/Value sequence length. int means all batches share the same length;
            list means per-batch KV sequence lengths (s2 must contain the max value).
        kv_d: Key/Value head dimension
        k_rope_d: Key rope dimension
        block_size: Size of each block in paged KV cache (default: 128)
        softmax_scale: Scaling factor for softmax (default: 1.0)
        group: Group nums. (n1 // n2)
    """
    layout: str = "BNSD"
    b: int = 32
    n1: int = 128
    s1: int = 1
    q_d: int = 512
    q_rope_d: int = 64
    n2: int = 1
    s2: int | list = 4 * 1024
    kv_d: int = 512
    k_rope_d: int = 64
    block_size: int = 128
    softmax_scale: float = 576 ** -0.5
    group: int = 128


@dataclass
class AttentionTileConfig:
    """
    Configuration for tile sizes used in attention computation.

    Tiling is used to break large computations into smaller, cache-friendly chunks.

    Attributes:
        g_tile: Tile size for group dimension
        s2_tile: Tile size for kv sequence dimension
        v0_tile: Tile configuration for vector operations assemble compute block
        c1_tile: Tile configuration for first matrix multiplication (Q x K^T)
        v1_tile: Tile configuration for vector operations in first MM
        c2_tile: Tile configuration for second matrix multiplication (Softmax x V)
        v2_tile: Tile configuration for vector operations in second MM
        v2_update_tile: Tile configuration for vector operations in second update MM
    """
    g_tile: int
    s2_tile: int
    v0_tile: list
    c1_tile: list
    v1_tile: list
    c2_tile: list
    v2_tile: list
    v2_update_tile: list


@dataclass
class LoopTensor:
    """
    Tensors used during loop iterations.

    These are the main data structures accessed during attention computation.

    Attributes:
        query_2d: Query tensor reshaped to 2D (b*n1*s1, d)
        knope_2d: Key tensor reshaped to 2D (block_num*block_size*n2, d)
        query_rope_2d: Query rope tensor reshaped to 2D (b*n1*s1, dr)
        key_rope_2d: Key rope tensor reshaped to 2D (block_num*block_size*n2, dr)
        block_table: Mapping from logical block indices to physical block indices
        kv_act_seqs: Key/Value actual sequence lengths for each batch element
        attention_output: Output tensor for attention results
    """
    query_2d: pypto.Tensor = None
    key_2d: pypto.Tensor = None
    query_rope_2d: pypto.Tensor = None
    key_rope_2d: pypto.Tensor = None
    block_table: pypto.Tensor = None
    kv_actual_seqs: pypto.Tensor = None
    attention_output: pypto.Tensor = None


@dataclass
class TempUpdateTensor:
    """
    Temporary tensors for online softmax computation.

    These tensors accumulate results across sequence tiles using the
    online softmax algorithm (Welford's algorithm variant).

    Attributes:
        out_update: Accumulated output (weighted sum of values)
        sum_update: Accumulated softmax denominator
        max_update: Accumulated softmax maximum value
    """
    out_update: pypto.Tensor = None
    sum_update: pypto.Tensor = None
    max_update: pypto.Tensor = None


@dataclass
class ContextParams:
    """
    Container for all context parameters passed between functions.

    This dataclass groups all the parameters needed for attention computation
    to avoid passing many individual arguments.

    Attributes:
        b_idx: Batch index
        s1_idx: Query sequence index
        n2_idx: Key/Value head index
        group_idx: Group index within heads
        s2_idx: Key/Value sequence tile index
        s2_loop: Number of sequence tiles to iterate
        kernel_config: Kernel computation config
        tile_config: Tile configuration
        loop_tensors: Tensors used in loops
        temp_update_tensors: Temporary update tensors
    """
    b_idx: int = 0
    s1_idx: int = 0
    n2_idx: int = 0
    group_idx: int = 0
    s2_idx: int = 0
    s2_loop: int = 0
    seq: object = None
    kernel_config: MlaConfig = None
    tile_config: AttentionTileConfig = None
    loop_tensors: LoopTensor = None
    temp_update_tensors: TempUpdateTensor = None


def reshape_qkv_to_2d(query, key, query_rope, key_rope, kp):
    """
    Reshape Q, K, V tensors to 2D

    Args:
        query: Query tensor of shape [b, s1, n1, d]
        key: Key cache of shape [block_num, n2, block_size, d]
        query_rope: Query rope tensor of shape [b, s1, n1, dr]
        key_rope: Key rope cache of shape [block_num, n2, block_size, dr]
        kp: Kernel config

    Returns:
        tuple: (qnope_2d, knope_2d, qrope_2d, krope_2d) reshaped to 2D
    """
    b = kp.b
    s1 = kp.s1
    n1 = kp.n1
    q_d = kp.q_d
    q_rope_d = kp.q_rope_d
    dr = kp.q_rope_d

    block_num = key.shape[0]
    block_size = kp.block_size
    n2 = kp.n2
    kv_d = kp.kv_d
    k_rope_d = kp.k_rope_d

    qnope_2d_shape = (b * s1 * n1, q_d)
    qrope_2d_shape = (b * s1 * n1, q_rope_d)
    knope_2d_shape = (block_num * block_size * n2, kv_d)
    krope_2d_shape = (block_num * block_size * n2, k_rope_d)

    qnope_2d = pypto.reshape(query, qnope_2d_shape, inplace=True)
    qrope_2d = pypto.reshape(query_rope, qrope_2d_shape, inplace=True)
    knope_2d = pypto.reshape(key, knope_2d_shape, inplace=True)
    krope_2d = pypto.reshape(key_rope, krope_2d_shape, inplace=True)
    return qnope_2d, knope_2d, qrope_2d, krope_2d


def assemble_key_blocks(ctx, actual_s2_tile):
    """
    Assemble K tensor for current tile from paged blocks.

    Args:
        ctx: Context parameters containing tensors and config
        actual_s2_tile: current kv_act_seqs

    Returns:
        pypto.Tensor: Assembled K tensor of shape [s2_tile, d]
    """
    kernel_config = ctx.kernel_config
    tile_config = ctx.tile_config
    loop_tensors = ctx.loop_tensors
    knope_2d = loop_tensors.key_2d
    krope_2d = loop_tensors.key_rope_2d
    block_table = loop_tensors.block_table
    dtype = knope_2d.dtype
    block_size = kernel_config.block_size
    kv_d = kernel_config.kv_d
    k_rope_d = kernel_config.k_rope_d
    n2 = kernel_config.n2

    s2_tile = tile_config.s2_tile
    v0_tile = tile_config.v0_tile

    pypto.set_semantic_label("MLA_V0")
    pypto.set_vec_tile_shapes(v0_tile[0], v0_tile[1])
    key_assemble = pypto.tensor([s2_tile, kv_d + k_rope_d], dtype, "key_assemble")
    kn_assemble = pypto.tensor([s2_tile, kv_d], dtype, "kn_assemble")
    kr_assemble = pypto.tensor([s2_tile, k_rope_d], dtype, "kr_assemble")

    block_num = s2_tile // block_size
    base_idx = ctx.s2_idx * block_num
    for i in range(block_num):
        b_idx = (block_table[ctx.b_idx, base_idx + i].max(0) * n2 + ctx.n2_idx) * block_size
        kn_view = pypto.view(knope_2d, [block_size, kv_d], [b_idx, 0], valid_shape=[actual_s2_tile, kv_d])
        pypto.assemble(kn_view, [i * block_size, 0], kn_assemble)
        pypto.assemble(kn_view, [i * block_size, 0], key_assemble)
        kr_view = pypto.view(krope_2d, [block_size, k_rope_d], [b_idx, 0])
        pypto.assemble(kr_view, [i * block_size, 0], kr_assemble)
        pypto.assemble(kr_view, [i * block_size, kv_d], key_assemble)
    return key_assemble, kn_assemble


def assemble_query_blocks(ctx):
    """
    Assemble query tensor for current tile.

    Args:
        ctx: Context parameters containing tensors and config

    Returns:
        pypto.Tensor: Assembled K tensor of shape [g_tile, d]
    """
    b_idx = ctx.b_idx
    s1_idx = ctx.s1_idx
    n2_idx = ctx.n2_idx
    group_idx = ctx.group_idx
    kernel_config = ctx.kernel_config
    g_tile = ctx.tile_config.g_tile
    q_d = kernel_config.q_d
    q_rope_d = kernel_config.q_rope_d
    n1 = kernel_config.n1
    s1 = kernel_config.s1
    group = kernel_config.group
    loop_tensors = ctx.loop_tensors
    qnope_2d = loop_tensors.query_2d
    qrope_2d = loop_tensors.query_rope_2d
    dtype = qnope_2d.dtype

    query_assemble = pypto.tensor([g_tile, q_d + q_rope_d], dtype, "query_assemble")
    offset = (b_idx * s1 + s1_idx) * n1 + n2_idx * group + group_idx * g_tile
    qn_view = pypto.view(qnope_2d, [g_tile, q_d], [offset, 0])
    qr_view = pypto.view(qrope_2d, [g_tile, q_rope_d], [offset, 0])
    pypto.assemble(qn_view, [0, 0], query_assemble)
    pypto.assemble(qr_view, [0, q_d], query_assemble)
    return query_assemble


def compute_c1(query_assemble, key_assemble, ctx, actual_s2_tile):
    """
    Compute first matrix multiplication: Q x K^T.

    Args:
        query_assemble: Query tensor for current head group, shape [g_tile, d]
        kj_assemble: Assembled K tensor for current tile, shape [s2_tile, d]
        ctx: Context parameters containing tensors and config
        actual_s2_tile: current kv_act_seqs

    Returns:
        pypto.Tensor: QK^T scores of shape [g_tile, actual_s2_tile]
    """
    pypto.set_semantic_label("MLA_C1")
    v0_tile = ctx.tile_config.v0_tile
    c1_tile = ctx.tile_config.c1_tile
    g_tile = ctx.tile_config.g_tile
    s2_tile = ctx.tile_config.s2_tile

    pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
    score = pypto.matmul(query_assemble, key_assemble, pypto.DT_FP32, a_trans=False, b_trans=True)
    pypto.set_vec_tile_shapes(v0_tile[0], v0_tile[1])
    score = pypto.view(score, [g_tile, s2_tile], [0, 0], valid_shape=[g_tile, actual_s2_tile])
    return score


def compute_v1(score, ctx):
    """
    Compute softmax.

    Args:
        score: QK^T of shape [g_tile, actual_s2_tile]
        ctx: Context parameters containing tensors and config

    Returns:
        prob_fp16, lse_reduce, lse, max_score
    """
    kernel_config = ctx.kernel_config
    dtype = ctx.loop_tensors.query_2d.dtype
    tile_config = ctx.tile_config
    v1_tile = tile_config.v1_tile
    g_tile = tile_config.g_tile

    pypto.set_semantic_label("MLA_V1")
    pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
    scaled = pypto.mul(score, kernel_config.softmax_scale)
    max_reduce = pypto.amax(scaled, dim=-1, keepdim=True)
    max_score = pypto.reshape(max_reduce, [1, g_tile])
    diff = pypto.sub(scaled, max_reduce)
    prob = pypto.exp(diff)
    prob_fp16 = pypto.cast(prob, dtype)
    lse_reduce = pypto.sum(prob, dim=-1, keepdim=True)
    lse = pypto.reshape(lse_reduce, [1, g_tile])
    return prob_fp16, lse_reduce, lse, max_score


def compute_c2(prob_fp16, v_assemble, ctx):
    """
    Compute softmax result x V.

    Args:
        prob_fp16: softmax result
        v_assemble: value block
        ctx: Context parameters containing tensors and config

    Returns:
        prob_fp16, lse_reduce, lse, max_score
    """
    tc = ctx.tile_config
    pypto.set_semantic_label("MLA_C2")
    pypto.set_cube_tile_shapes(tc.c2_tile[0], tc.c2_tile[1], tc.c2_tile[2])
    pypto.set_matrix_size([prob_fp16.shape[0], prob_fp16.shape[1], v_assemble.shape[1]])
    wv = pypto.matmul(prob_fp16, v_assemble, pypto.DT_FP32)
    return wv


def handle_first_tile(wv, lse_reduce, lse, max_score, ctx):
    """
    Compute attention for the first tile, computes the initial max, sum, and output values.
    """
    kp = ctx.kernel_config
    tc = ctx.tile_config
    lt = ctx.loop_tensors
    tt = ctx.temp_update_tensors
    dtype = lt.query_2d.dtype

    pypto.set_vec_tile_shapes(tc.v2_tile[0], tc.v2_tile[1])
    if pypto.cond(pypto.is_loop_end(ctx.s2_idx)):
        pypto.set_semantic_label("MLA_V2")
        tt.out_update[:] = wv / lse_reduce
        pypto.set_vec_tile_shapes(1, tc.v2_tile[0], 1, tc.v2_tile[1])
        out_4d = pypto.cast(pypto.reshape(tt.out_update, [1, 1, tc.g_tile, kp.q_d]), dtype)
        out_off = [ctx.b_idx, ctx.s1_idx, ctx.n2_idx * kp.group + ctx.group_idx * tc.g_tile, 0]
        pypto.assemble(out_4d, out_off, lt.attention_output)
    else:
        tt.out_update[:] = wv
    pypto.set_vec_tile_shapes(tc.v2_tile[0], tc.v2_tile[1])
    tt.sum_update[:] = lse
    tt.max_update[:] = max_score


def handle_other_tile(wv, lse, max_score, ctx):
    """
    Compute attention for subsequent sequence tiles.

    """
    kp = ctx.kernel_config
    tc = ctx.tile_config
    tt = ctx.temp_update_tensors
    dtype = ctx.loop_tensors.query_2d.dtype
    lt = ctx.loop_tensors

    pypto.set_semantic_label("MLA_UpdateVec2")
    pypto.set_vec_tile_shapes(tc.v2_update_tile[0], tc.v2_update_tile[1])
    if pypto.platform.npuarch != 'DAV_3510':
        pypto.set_pass_options(sg_set_scope=1)
    new_max = pypto.maximum(tt.max_update, max_score)
    old_diff = pypto.sub(tt.max_update, new_max)
    exp_old = pypto.exp(old_diff)
    new_diff = pypto.sub(max_score, new_max)
    exp_new = pypto.exp(new_diff)
    new_lse = pypto.add(pypto.mul(exp_old, tt.sum_update), pypto.mul(exp_new, lse))

    old_scaled = pypto.mul(tt.out_update, pypto.reshape(exp_old, [tc.g_tile, 1]))
    new_scaled = pypto.mul(wv, pypto.reshape(exp_new, [tc.g_tile, 1]))
    out_tmp = pypto.add(old_scaled, new_scaled)

    if pypto.cond(pypto.is_loop_end(ctx.s2_idx)):
        tt.out_update[:] = pypto.div(out_tmp, pypto.reshape(new_lse, [tc.g_tile, 1]),
                                     precision_type=pypto.PrecisionType.INTRINSIC)
        pypto.set_vec_tile_shapes(1, 1, tc.v2_tile[0], tc.v2_tile[1])
        out_4d = pypto.cast(pypto.reshape(tt.out_update, [1, 1, tc.g_tile, kp.q_d]), dtype)
        out_off = [ctx.b_idx, ctx.s1_idx, ctx.n2_idx * kp.group + ctx.group_idx * tc.g_tile, 0]
        pypto.assemble(out_4d, out_off, lt.attention_output)
    else:
        tt.out_update[:] = out_tmp

    tt.sum_update[:] = new_lse
    tt.max_update[:] = new_max
    if pypto.platform.npuarch != 'DAV_3510':
        pypto.set_pass_options(sg_set_scope=-1)


def compute_loop_s2(ctx):
    """
    Compute attention loop over sequence tiles.

    Args:
        ctx: Context parameters
    """
    kp = ctx.kernel_config
    tc = ctx.tile_config
    seq = ctx.seq
    s2_idx = ctx.s2_idx
    s2_tile = tc.s2_tile
    g_tile = tc.g_tile
    kv_d = kp.kv_d

    actual_s2_tile = (seq - s2_idx * s2_tile).min(s2_tile)
    key_assemble, kn_assemble = assemble_key_blocks(ctx, actual_s2_tile)

    query_assemble = assemble_query_blocks(ctx)
    score = compute_c1(query_assemble, key_assemble, ctx, actual_s2_tile)
    prob_fp16, lse_reduce, lse, max_score = compute_v1(score, ctx)
    wv = compute_c2(prob_fp16, kn_assemble, ctx)

    if pypto.cond(pypto.is_loop_begin(ctx.s2_idx)):
        handle_first_tile(wv, lse_reduce, lse, max_score, ctx)
    else:
        handle_other_tile(wv, lse, max_score, ctx)


def compute_loop_group(ctx):
    """
    Compute attention loop over groups.

    Args:
        ctx: Context parameters
    """
    kp = ctx.kernel_config
    tc = ctx.tile_config
    for g_idx in pypto.loop(kp.group // tc.g_tile, name="LOOP_group", idx_name="gIdx"):
        ctx = replace(ctx, group_idx=g_idx)
        tt = TempUpdateTensor(
            pypto.tensor([tc.g_tile, kp.q_d], pypto.DT_FP32, "out_update"),
            pypto.tensor([1, tc.g_tile], pypto.DT_FP32, "sum_update"),
            pypto.tensor([1, tc.g_tile], pypto.DT_FP32, "max_update")
        )
        ctx = replace(ctx, temp_update_tensors=tt)
        for s2_idx in pypto.loop(ctx.s2_loop, name="FLASH_LOOP_L4_s2_SA",
                                 idx_name="s2_idx", unroll_list=[8, 2, 1]):
            ctx = replace(ctx, s2_idx=s2_idx)
            compute_loop_s2(ctx)


def compute_loop_n2(ctx):
    """
    Compute attention loop over key/value heads.

    Args:
        ctx: Context parameters
    """
    kp = ctx.kernel_config
    for n2_idx in pypto.loop(kp.n2, name="LOOP_n2", idx_name="n2Idx"):
        ctx = replace(ctx, n2_idx=n2_idx)
        compute_loop_group(ctx)


def compute_loop_s1(ctx):
    """
    Compute attention loop over query sequence positions.

    Args:
        ctx: Context parameters
    """
    kp = ctx.kernel_config
    tc = ctx.tile_config
    lt = ctx.loop_tensors
    cur_seq = lt.kv_actual_seqs[ctx.b_idx]
    for s1_idx in pypto.loop(kp.s1, name="LOOP_s1", idx_name="s1Idx"):
        seq = (cur_seq - kp.s1 + 1 + s1_idx)
        ctx = replace(ctx, s1_idx=s1_idx, s2_loop=(seq + tc.s2_tile - 1) // tc.s2_tile, seq=seq)
        compute_loop_n2(ctx)


def compute_loop_b(ctx):
    """
    Compute attention loop over batch dimension.

    Args:
        ctx: Context parameters
    """
    kp = ctx.kernel_config
    for b_idx in pypto.loop(kp.b, name="LOOP_b", idx_name="bIdx"):
        ctx = replace(ctx, b_idx=b_idx)
        compute_loop_s1(ctx)


@pypto.frontend.jit(
    pass_options={
        "vec_nbuffer_setting": {-1: 2, 0: 8},
        "cube_l1_reuse_setting": {-1: 2, 1: 1},
        "cube_nbuffer_setting": {-1: 2}
    },
    runtime_options={
        "stitch_function_max_num": 256,
        "device_sched_mode": 3,
        "ready_on_host_tensors": ["block_table", "kv_actual_seqs"]
    }
)
def incre_flash_attention_mla_kernel(
    query: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    key: pypto.Tensor([...], pypto.DT_BF16),
    value: pypto.Tensor([...], pypto.DT_BF16),
    query_rope: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    key_rope: pypto.Tensor([...], pypto.DT_BF16),
    block_table: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_INT32),
    kv_actual_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    attention_output: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    kernel_config, tile_config
):
    pypto.experimental.set_operation_options(combine_axis=True)

    qnope_2d, knope_2d, qrope_2d, krope_2d = reshape_qkv_to_2d(
        query, key, query_rope, key_rope, kernel_config
    )
    loop_tensors = LoopTensor(
        query_2d=qnope_2d, key_2d=knope_2d, query_rope_2d=qrope_2d, key_rope_2d=krope_2d,
        block_table=block_table, kv_actual_seqs=kv_actual_seqs,
        attention_output=attention_output
    )
    ctx = ContextParams(
        kernel_config=kernel_config, tile_config=tile_config, loop_tensors=loop_tensors
    )
    compute_loop_b(ctx)


@allow_in_graph
def incre_flash_attention_mla(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
    kv_actual_seqs: torch.Tensor,
    block_table: torch.Tensor,
    kernel_config, tile_config
):
    out_shape = (kernel_config.b, kernel_config.s1, kernel_config.n1, kernel_config.q_d)
    attn_out = torch.zeros(out_shape, dtype=query.dtype, device=query.device)
    values = [query, key, value, query_rope, key_rope, block_table, kv_actual_seqs,
              attn_out, kernel_config, tile_config]
    incre_flash_attention_mla_kernel(*values)
    return attn_out
