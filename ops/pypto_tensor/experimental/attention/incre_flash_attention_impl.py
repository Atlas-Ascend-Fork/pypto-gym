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
Incremental Flash Attention (IFA) Implementation

This module implements the Incremental Flash Attention algorithm using PyPTO
"""

import enum
from dataclasses import dataclass, replace

import torch
from torch._dynamo import allow_in_graph

import pypto


class TileOpFormat(enum.Enum):
    ND = "ND"
    NZ = "NZ"


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
    """
    g_tile: int
    s2_tile: int
    c1_tile: list
    v1_tile: list
    c2_tile: list
    v2_tile: list


@dataclass
class LoopOfs:
    """
    Offset parameters for loop iterations.

    Used to track current positions in the output tensor during nested loops.

    Attributes:
        bs_ofs: Batch-sequence offset (b_idx * s1 + s1_idx)
        n1g_ofs: Head-group offset (n2_idx * group + group_idx * g_tile)
        out_ofs: Output tensor offset [bs_ofs, n1g_ofs, 0]
    """
    bs_ofs: int = 0
    n1g_ofs: int = 0
    out_ofs: int = 0


@dataclass
class LoopTensor:
    """
    Tensors used during loop iterations.

    These are the main data structures accessed during attention computation.

    Attributes:
        q_2d: Query tensor reshaped to 2D (bs*n1, d)
        k_2d: Key tensor reshaped to 2D (block_num*block_size*n2, d)
        v_2d: Value tensor reshaped to 2D (block_num*block_size*n2, d)
        block_table: Mapping from logical block indices to physical block indices
        kv_act_seqs: Key/Value actual sequence lengths for each batch element
        atten_out: Output tensor for attention results
    """
    q_2d: pypto.Tensor = None
    k_2d: pypto.Tensor = None
    v_2d: pypto.Tensor = None
    block_table: pypto.Tensor = None
    kv_act_seqs: pypto.Tensor = None
    atten_out: pypto.Tensor = None


@dataclass
class LoopIndex:
    """
    Current indices for nested loop iterations.

    Attributes:
        b_idx: Batch index
        s1_idx: Query sequence index
        n2_idx: Key/Value head index
        group_idx: Group index within heads
        s2_idx: Key/Value sequence tile index
    """
    b_idx: int = 0
    s1_idx: int = 0
    n2_idx: int = 0
    group_idx: int = 0
    s2_idx: int = 0


@dataclass
class LoopSize:
    """
    Loop iteration counts.

    Attributes:
        group_loop: Number of groups to iterate (group_num // g_tile ——> (n1 // n2) // g_tile)
        s2_loop: Number of sequence tiles to iterate
    """
    group_loop: int = 0
    s2_loop: int = 0


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
class IFAKernelParams:
    """
    Parameters for the IFA kernel computation.

    Attributes:
        n1: Number of query heads
        d: Head dimension
        block_num: Total number of KV blocks
        n2: Number of key/value heads
        block_size: Size of each block
        b: Batch size
        s1: Query sequence length
        group: Number of head groups (n1 // n2)
        softmax_scale: Softmax scaling factor
    """
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
    """
    Container for all context parameters passed between functions.

    This dataclass groups all the parameters needed for attention computation
    to avoid passing many individual arguments.

    Attributes:
        kernel_params: Kernel computation parameters
        tile_cfg: Tile configuration
        loop_tensors: Tensors used in loops
        loop_index: Current loop indices
        loop_size: Loop iteration counts
        loop_ofs: Loop offsets
        temp_update_tensors: Temporary update tensors
    """
    kernel_params: IFAKernelParams = None
    tile_cfg: AttentionTileConfig = None
    loop_tensors: LoopTensor = None
    loop_index: LoopIndex = None
    loop_size: TempUpdateTensor = None
    loop_ofs: LoopOfs = None
    temp_update_tensors: TempUpdateTensor = None


def get_format(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("input type error")
    if not tensor.is_contiguous():
        raise TypeError("input type error")

    tile_op_format = TileOpFormat.ND.value
    if tensor.device.type == "npu":
        import torch_npu
        if torch_npu.get_npu_format(tensor) == 29:
            tile_op_format = TileOpFormat.NZ.value
    return tile_op_format


def check_equal(param_name, param_value, expect):
    if param_value != expect:
        raise ValueError(f"{param_name} must be equal to {expect}, but got {param_value}.")


def check_min_max(param_name, param_value, min_max):
    """
    Args:
        min_max: [min, max]
    """
    min_value = min_max[0]
    max_value = min_max[1]

    if param_value < min_value or param_value > max_value:
        raise ValueError(
            f"{param_name} must be greater than or equal to {min_value} "
            f"and less than or equal to {max_value}, but got {param_value}."
        )


def check_query(inputs):
    # query shape is [T, n1, d]
    query = inputs.get("query")

    # Check query dim
    check_equal("query dim", query.dim(), 3)

    # Check query bs
    bs = query.shape[0]
    check_min_max("query dim 0", bs, [1, 16])

    # Check query s: [s1]
    block_table = inputs.get("block_table")
    b = block_table.shape[0]
    s1 = bs // b
    check_equal("s1", s1, 1)

    # Check query head num: [n1]
    n1 = query.shape[1]
    check_equal("query dim 1", n1, 12)

    # Check query head dim: [d]
    d = query.shape[2]
    check_equal("query dim 2", d, 128)

    # Check query dtype
    check_equal("query dtype", query.dtype, torch.bfloat16)

    # Check query format
    check_equal("query format", get_format(query), "ND")


def check_key(key):
    # key shape is [block_num, n2, block_size, d]
    # Check key dim
    check_equal(f"key dim", key.dim(), 4)

    # Check key head num: [n2]
    n2 = key.shape[1]
    check_equal("key dim 1", n2, 1)

    # Check block size.
    block_size = key.shape[2]
    check_equal("key dim 2", block_size, 128)

    # Check key head dim: [d]
    d = key.shape[3]
    check_equal("key dim 3", d, 128)

    # Check key dtype
    check_equal("key dtype", key.dtype, torch.bfloat16)

    # Check key format
    check_equal("key format", get_format(key), "ND")


def check_value(value):
    # value shape is [block_num, n2, block_size, d]
    # Check value dim
    check_equal(f"value dim", value.dim(), 4)

    # Check value head num: [n2]
    n2 = value.shape[1]
    check_equal("value dim 1", n2, 1)

    # Check block size.
    block_size = value.shape[2]
    check_equal("value dim 2", block_size, 128)

    # Check value head dim: [d]
    d = value.shape[3]
    check_equal("value dim 3", d, 128)

    # Check value dtype
    check_equal("value dtype", value.dtype, torch.bfloat16)

    # Check value format
    check_equal("value format", get_format(value), "ND")


def check_key_value_shape(inputs):
    key = inputs.get("key")
    value = inputs.get("value")

    key_shape = key.shape
    value_shape = value.shape

    if key_shape != value_shape:
        raise ValueError(
            f"key and value shape must be equal, "
            f"but got key shape is {key_shape} and value shape is {value_shape}"
        )


def check_block_table(block_table):
    # block_table shape is [b, max_block_num_per_query]
    # Check block_table dim
    check_equal(f"block table dim", block_table.dim(), 2)

    # Check block_table dtype
    check_equal("block table dtype", block_table.dtype, torch.int32)

    # Check block_table format
    check_equal("block table format", get_format(block_table), "ND")


def check_actual_seq_lengths(actual_seq_lengths):
    # actual_seq_lengths shape is [b]
    # Check actual_seq_lengths dim
    check_equal(f"actual_seq_lengths dim", actual_seq_lengths.dim(), 1)

    # Check actual_seq_lengths dtype
    check_equal("actual_seq_lengths dtype", actual_seq_lengths.dtype, torch.int32)

    # Check actual_seq_lengths format
    check_equal("actual_seq_lengths format", get_format(actual_seq_lengths), "ND")


def check_inputs(inputs):
    check_query(inputs)
    check_key(inputs.get("key"))
    check_value(inputs.get("value"))
    check_key_value_shape(inputs)
    check_block_table(inputs.get("block_table"))
    check_actual_seq_lengths(inputs.get("actual_seq_lengths"))


def get_ifa_tile_cfg():
    """
    Get tile configuration for IFA computation.

    Returns:
        AttentionTileConfig: Tile configuration with optimal sizes
    """
    m_tile = 256
    k_tile = 128
    n_tile = 128
    s2_tile = 2048

    tile_cfg = AttentionTileConfig(
        g_tile=12,
        s2_tile=s2_tile,
        c1_tile=[[m_tile, m_tile], [k_tile, k_tile], [n_tile, n_tile]],
        v1_tile=[6, s2_tile],
        c2_tile=[[m_tile, m_tile], [k_tile, k_tile], [n_tile, n_tile]],
        v2_tile=[64, m_tile]
    )
    return tile_cfg


def assemble_kj(idx, ctx_params):
    """
    Assemble K tensor for current tile from paged blocks.

    Args:
        idx: Starting block index for this tile
        ctx_params: Context parameters containing tensors and config

    Returns:
        pypto.Tensor: Assembled K tensor of shape [s2_tile, d]
    """
    # Get needed tensors
    k_2d = ctx_params.loop_tensors.k_2d
    block_table = ctx_params.loop_tensors.block_table

    # Get needed tile cfg
    s2_tile = ctx_params.tile_cfg.s2_tile

    # Get needed kernel params
    block_size = ctx_params.kernel_params.block_size
    d = ctx_params.kernel_params.d

    # Get needed loop index
    b_idx = ctx_params.loop_index.b_idx

    block_num = s2_tile // block_size

    # Create assembled tensor
    kj_assemble = pypto.tensor([s2_tile, d], k_2d.dtype, "kj_assemble")

    # Copy blocks from 2D K tensor according to block table
    for i in range(block_num):
        block_idx = block_table[b_idx, idx + i]
        block_idx_valid = block_idx.max(0)
        src_view = pypto.view(k_2d, [block_size, d], [block_idx_valid * block_size, 0])
        pypto.assemble(src_view, [i * block_size, 0], kj_assemble)

    # Set valid shape (may be smaller than allocated size)
    kj_assemble = pypto.view(kj_assemble, [s2_tile, d], [0, 0], valid_shape=[s2_tile, d])
    return kj_assemble


def assemble_vj(idx, actual_s2_tile, ctx_params):
    """
    Assemble V tensor for current tile from paged blocks.

    Args:
        idx: Starting block index for this tile
        actual_s2_tile: Actual sequence length in this tile (may be smaller)
        ctx_params: Context parameters containing tensors and config

    Returns:
        pypto.Tensor: Assembled V tensor of shape [actual_s2_tile, d]
    """
    # Get needed tensors
    v_2d = ctx_params.loop_tensors.v_2d
    block_table = ctx_params.loop_tensors.block_table

    # Get needed cfg
    s2_tile = ctx_params.tile_cfg.s2_tile

    # Get needed kernel params
    block_size = ctx_params.kernel_params.block_size
    d = ctx_params.kernel_params.d

    # Get needed loop index
    b_idx = ctx_params.loop_index.b_idx

    block_num = s2_tile // block_size

    # Create assembled tensor
    vj_assemble = pypto.tensor([s2_tile, d], v_2d.dtype, "vj_assemble")

    # Copy blocks from 2D V tensor according to block table
    for i in range(block_num):
        block_idx = block_table[b_idx, idx + i]
        block_idx_valid = block_idx.max(0)
        src_view = pypto.view(v_2d, [block_size, d], [block_idx_valid * block_size, 0])
        pypto.assemble(src_view, [i * block_size, 0], vj_assemble)

    # Set valid shape to actual sequence length
    vj_assemble = pypto.view(vj_assemble, [s2_tile, d], [0, 0], valid_shape=[actual_s2_tile, d])
    return vj_assemble


def compute_loop_b(dtype, ctx_params):
    """
    Compute attention loop over batch dimension.

    Args:
        dtype: Data type for computation
        ctx_params: Context parameters
    """
    # Get needed kernel params
    s1 = ctx_params.kernel_params.s1

    # Get needed tile cfg
    s2_tile = ctx_params.tile_cfg.s2_tile

    # Get needed loop tensors
    kv_act_seqs = ctx_params.loop_tensors.kv_act_seqs

    # Get needed loop index
    b_idx = ctx_params.loop_index.b_idx

    loop_size = ctx_params.loop_size

    # Loop over query sequence positions
    for s1_idx in pypto.loop(s1, name="LOOP_s1", idx_name="s1_idx"):
        # Calculate effective sequence length
        cur_seq_len = kv_act_seqs[b_idx] - (s1 - 1 - s1_idx)

        s2_loop = pypto.ceildiv(cur_seq_len, s2_tile)
        loop_size = replace(loop_size, s2_loop=s2_loop)
        bs_ofs = b_idx * s1 + s1_idx
        loop_ofs = LoopOfs(bs_ofs=bs_ofs)
        ctx_params = replace(ctx_params, loop_size=loop_size, loop_ofs=loop_ofs)
        compute_loop_s1(ctx_params, cur_seq_len, dtype)


def compute_loop_s1(ctx_params, cur_seq_len, dtype):
    """
    Compute attention loop over query sequence positions.

    Args:
        ctx_params: Context parameters
        cur_seq_len: Current sequence length
        dtype: Data type for computation
    """
    n2 = ctx_params.kernel_params.n2
    loop_index = ctx_params.loop_index

    for n2_idx in pypto.loop(n2, name="LOOP_n2", idx_name="n2_idx"):
        loop_index = replace(loop_index, n2_idx=n2_idx)
        ctx_params = replace(ctx_params, loop_index=loop_index)
        compute_loop_n2(ctx_params, cur_seq_len, dtype)


def compute_loop_n2(ctx_params, cur_seq_len, dtype):
    """
    Compute attention loop over key/value heads.

    Args:
        ctx_params: Context parameters
        cur_seq_len: Current sequence length
        dtype: Data type for computation
    """
    loop_index = ctx_params.loop_index
    group_loop = ctx_params.loop_size.group_loop

    for group_idx in pypto.loop(group_loop, name="LOOP_group_idx", idx_name="group_idx"):
        loop_index = replace(loop_index, group_idx=group_idx)
        ctx_params = replace(ctx_params, loop_index=loop_index)
        compute_loop_group(ctx_params, cur_seq_len, dtype)


def compute_loop_group(ctx_params, cur_seq_len, dtype):
    """
    Compute attention loop over groups.

    Args:
        ctx_params: Context parameters
        cur_seq_len: Current sequence length
        dtype: Data type for computation
    """
    # Get needed tile cfg
    g_tile = ctx_params.tile_cfg.g_tile

    # Get needed kernel params
    group = ctx_params.kernel_params.group
    d = ctx_params.kernel_params.d

    # Get needed loop index params
    loop_index = ctx_params.loop_index
    n2_idx = loop_index.n2_idx
    group_idx = loop_index.group_idx

    # Get needed loop offset params
    loop_ofs = ctx_params.loop_ofs
    bs_ofs = loop_ofs.bs_ofs

    # Get needed loop params
    s2_loop = ctx_params.loop_size.s2_loop

    # Calculate offset for current group
    n1g_ofs = n2_idx * group + group_idx * g_tile
    out_ofs = [bs_ofs, n1g_ofs, 0]
    loop_ofs = replace(loop_ofs, n1g_ofs=n1g_ofs, out_ofs=out_ofs)

    # Initialize temporary tensors for online softmax
    out_update = pypto.tensor([g_tile, d], pypto.DT_FP32, "out_update")
    sum_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update")
    max_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update")
    temp_update_tensors = TempUpdateTensor(out_update, sum_update, max_update)

    # Loop over sequence tiles
    for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx", unroll_list=[8, 2, 1]):
        loop_index = replace(loop_index, s2_idx=s2_idx)
        ctx_params = replace(ctx_params, loop_index=loop_index,
                            temp_update_tensors=temp_update_tensors, loop_ofs=loop_ofs)
        compute_loop_s2(ctx_params, cur_seq_len, dtype)


def compute_loop_s2(ctx_params, cur_seq_len, dtype):
    """
    Compute attention loop over sequence tiles.

    Args:
        ctx_params: Context parameters
        cur_seq_len: Current sequence length
        dtype: Data type for computation
    """
    # Get needed tile cfg
    tile_cfg = ctx_params.tile_cfg
    s2_tile = tile_cfg.s2_tile
    v1_tile = tile_cfg.v1_tile
    g_tile = tile_cfg.g_tile

    # Get needed kernel params
    block_size = ctx_params.kernel_params.block_size
    d = ctx_params.kernel_params.d
    n1 = ctx_params.kernel_params.n1

    # Get needed loop tensors
    q_2d = ctx_params.loop_tensors.q_2d

    # Get needed loop offset params
    bs_ofs = ctx_params.loop_ofs.bs_ofs
    n1g_ofs = ctx_params.loop_ofs.n1g_ofs
    out_ofs = ctx_params.loop_ofs.out_ofs

    # Get needed loop index params
    s2_idx = ctx_params.loop_index.s2_idx

    block_num = s2_tile // block_size
    idx = s2_idx * block_num

    # Calculate actual sequence length in this tile
    actual_s2_tile = (cur_seq_len - s2_idx * s2_tile).min(s2_tile)

    # Get query for current head group
    pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
    qi = pypto.view(q_2d, [g_tile, d], [bs_ofs * n1 + n1g_ofs, 0])

    # Assemble K and V for current tile
    kj_assemble = assemble_kj(idx, ctx_params)
    vj_assemble = assemble_vj(idx, actual_s2_tile, ctx_params)

    # Compute sij for this tile
    sij = compute_c1(qi, kj_assemble, actual_s2_tile, tile_cfg)

    # Compute attention for this tile
    pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
    if pypto.cond(pypto.is_loop_begin(s2_idx)):
        compute_first_tile(sij, vj_assemble, dtype, ctx_params)
    else:
        compute_other_tile(sij, vj_assemble, dtype, ctx_params)

    # Finalize output on last tile
    if pypto.cond(pypto.is_loop_end(s2_idx)):
        finalize_output(out_ofs, dtype, ctx_params)


def compute_c1(qi, kj_assemble, actual_s2_tile, tile_cfg):
    """
    Compute first matrix multiplication: Q x K^T.

    Args:
        qi: Query tensor for current head group, shape [g_tile, d]
        kj_assemble: Assembled K tensor for current tile, shape [s2_tile, d]
        actual_s2_tile: Actual sequence length in this tile
        tile_cfg: Tile configuration

    Returns:
        pypto.Tensor: QK^T scores of shape [g_tile, actual_s2_tile]
    """
    c1_tile = tile_cfg.c1_tile
    g_tile = tile_cfg.g_tile
    s2_tile = tile_cfg.s2_tile

    # Compute Q x K^T
    pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
    sij = pypto.matmul(qi, kj_assemble, pypto.DT_FP32, a_trans=False, b_trans=True)

    # Set valid shape to actual sequence length
    sij = pypto.view(sij, [g_tile, s2_tile], [0, 0], valid_shape=[g_tile, actual_s2_tile])
    return sij


def compute_first_tile(sij, vj_assemble, dtype, ctx_params):
    """
    Compute attention for the first tile, computes the initial max, sum, and output values.

    Args:
        sij: QK^T scores for current tile
        vj_assemble: V tensor for current tile
        dtype: Data type for computation
        ctx_params: Context parameters
    """
    softmax_scale = ctx_params.kernel_params.softmax_scale
    c2_tile = ctx_params.tile_cfg.c2_tile
    v2_tile = ctx_params.tile_cfg.v2_tile
    out_update = ctx_params.temp_update_tensors.out_update
    sum_update = ctx_params.temp_update_tensors.sum_update
    max_update = ctx_params.temp_update_tensors.max_update

    # Scale scores by softmax scale factor
    sij_scale = pypto.mul(sij, softmax_scale)

    # Compute maximum score for this tile
    tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)

    # Compute exp(scores - max) for numerical stability
    tsub = pypto.sub(sij_scale, tilda_mij)
    tilda_pij = pypto.exp(tsub)
    tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)

    # Initialize sum and max for online softmax
    sum_update[:] = pypto.sum(tilda_pij, dim=-1, keepdim=True)
    max_update[:] = tilda_mij

    # Compute weighted sum of values: exp(QK^T) x V
    pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
    oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)

    # Store initial output
    pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
    out_update[:] = oi_tmp


def compute_other_tile(sij, vj_assemble, dtype, ctx_params):
    """
    Compute attention for subsequent sequence tiles.

    Args:
        sij: QK^T scores for current tile
        vj_assemble: V tensor for current tile
        dtype: Data type for computation
        ctx_params: Context parameters
    """
    softmax_scale = ctx_params.kernel_params.softmax_scale
    c2_tile = ctx_params.tile_cfg.c2_tile
    v2_tile = ctx_params.tile_cfg.v2_tile
    out_update = ctx_params.temp_update_tensors.out_update
    sum_update = ctx_params.temp_update_tensors.sum_update
    max_update = ctx_params.temp_update_tensors.max_update

    # Compute scores for current tile
    sij_scale = pypto.mul(sij, softmax_scale)
    tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)

    # Update global maximum
    max_new = pypto.maximum(max_update, tilda_mij)

    # Compute exp(scores - max_new)
    tsub = pypto.sub(sij_scale, max_new)
    tilda_pij = pypto.exp(tsub)
    tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
    sum_local = pypto.sum(tilda_pij, dim=-1, keepdim=True)

    # Update sum using online algorithm
    tsub2 = pypto.sub(max_update, max_new)
    max_update[:] = max_new
    update_mul = pypto.exp(tsub2)
    sum_update[:] = sum_update * update_mul + sum_local

    # Update output using online algorithm
    pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
    oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)

    # Store output
    pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
    out_update[:] = out_update * update_mul + oi_tmp


def finalize_output(out_ofs, dtype, ctx_params):
    """
    Finalize and write attention output.

    This function divides the accumulated output by the sum to get
    the final attention result and writes it to the output tensor.

    Args:
        out_ofs: Offset in output tensor
        dtype: Output data type
        ctx_params: Context parameters
    """

    d = ctx_params.kernel_params.d
    v2_tile = ctx_params.tile_cfg.v2_tile
    g_tile = ctx_params.tile_cfg.g_tile
    out_update = ctx_params.temp_update_tensors.out_update
    sum_update = ctx_params.temp_update_tensors.sum_update
    atten_out = ctx_params.loop_tensors.atten_out

    # Divide by sum to get final attention result
    oi_final = pypto.div(out_update, sum_update)

    # Reshape and cast to output format
    pypto.set_vec_tile_shapes(16, v2_tile[0], v2_tile[1])
    oi_final_3d = pypto.cast(
        pypto.reshape(oi_final, [1, g_tile, d]), dtype)

    # Write result to output tensor
    pypto.assemble(oi_final_3d, out_ofs, atten_out)


def init_kernel_params(q, k, block_table):
    """
    Initialize kernel parameters from input tensors.

    This function extracts and computes all the parameters needed
    for the IFA kernel computation.

    Args:
        q: Query tensor
        k: Key cache tensor
        block_table: Block table

    Returns:
        IFAKernelParams: Initialized kernel parameters
    """
    bs, n1, d = q.shape
    block_num, n2, block_size, _ = k.shape
    block_table_shape = block_table.shape
    b = block_table_shape[0]
    s1 = bs // b
    group = n1 // n2
    softmax_scale = d ** -0.5
    kernel_params = IFAKernelParams(
        n1=n1, d=d, block_num=block_num, n2=n2, block_size=block_size,
        b=b, s1=s1, group=group, softmax_scale=softmax_scale
    )
    return kernel_params


def reshape_qkv_to_2d(q, k, v, kernel_params):
    """
    Reshape Q, K, V tensors to 2D

    Args:
        q: Query tensor of shape [b*s1, n1, d]
        k: Key cache of shape [block_num, n2, block_size, d]
        v: Value cache of shape [block_num, n2, block_size, d]
        kernel_params: Kernel parameters

    Returns:
        tuple: (q_2d, k_2d, v_2d) reshaped to 2D
    """
    b = kernel_params.b
    s1 = kernel_params.s1
    n1 = kernel_params.n1
    d = kernel_params.d
    block_num = kernel_params.block_num
    block_size = kernel_params.block_size
    n2 = kernel_params.n2
    d = kernel_params.d

    q_2d_shape = (b * s1 * n1, d)
    kv_2d_shape = (block_num * block_size * n2, d)

    q_2d = pypto.reshape(q, q_2d_shape, inplace=True)
    k_2d = pypto.reshape(k, kv_2d_shape, inplace=True)
    v_2d = pypto.reshape(v, kv_2d_shape, inplace=True)
    return q_2d, k_2d, v_2d


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 128,
        "device_sched_mode": 1
    },
    pass_options={
        "cube_l1_reuse_setting": {0: 8},
        "vec_nbuffer_setting": {0: 2}
    }
)
def ifa_func_kernel(
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    block_table: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_INT32),
    kv_act_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    atten_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16)
):
    # Step 1: Initialize kernel parameters
    dtype = q.dtype
    kernel_params = init_kernel_params(q, k, block_table)

    # Step 2: Get tile configuration
    tile_cfg = get_ifa_tile_cfg()

    # Step 3: Reshape Q, K, V to 2D
    q_2d, k_2d, v_2d = reshape_qkv_to_2d(q, k, v, kernel_params)
    loop_tensors = LoopTensor(q_2d, k_2d, v_2d, block_table, kv_act_seqs, atten_out)

    # Calculate number of groups to iterate
    group_loop = kernel_params.group // tile_cfg.g_tile
    loop_size = LoopSize(group_loop=group_loop)

    # Create context parameters
    ctx_params = ContextParams(
        kernel_params=kernel_params, tile_cfg=tile_cfg, loop_tensors=loop_tensors,
        loop_size=loop_size
    )

    # Step 4: Implement kernel logic with nested loops
    # Loop over batch dimension
    for b_idx in pypto.loop(kernel_params.b, name="LOOP_b", idx_name="b_idx"):
        loop_index = LoopIndex(b_idx=b_idx)
        ctx_params = replace(ctx_params, loop_index=loop_index)
        compute_loop_b(dtype, ctx_params)


@allow_in_graph
def incre_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    actual_seq_lengths: torch.Tensor,
    block_table: torch.Tensor,
):
    inputs = dict(
        query=query,
        key=key,
        value=value,
        block_table=block_table,
        actual_seq_lengths=actual_seq_lengths
    )
    check_inputs(inputs)
    atten_out = torch.zeros_like(query)
    input_values = [query, key, value, block_table, actual_seq_lengths, atten_out]
    ifa_func_kernel(*input_values)
    return atten_out
