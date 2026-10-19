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
"""
import os
from typing import Optional
from dataclasses import dataclass
import torch
import pypto
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph


@dataclass
class AttentionTileConfig:
    g_tile: int
    s2_tile: int
    c1_tile_shape: list
    v1_tile_shape: list
    c2_tile_shape: list
    v2_tile_shape: list


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
    max_num_blocks_per_query: int = 0
    softmax_scale: float = 1.0
    kv_layout: str = "PA_BSND"
    actual_seq: torch.Tensor = None  # 改为 torch.Tensor 类型
    hidden_size: int = 0
    block_table_batch: int = 0
    kv_num_blocks: int = 0
    eps: float = 1e-05


# pypto
def rms_norm_bias(tensor_value, gamma, bias, mean_coff, eps, tile_shape):
    input_dtype = tensor_value.dtype
    # cast
    pypto.set_vec_tile_shapes(*tile_shape)
    tensor_value_fp32 = pypto.cast(tensor_value, pypto.DT_FP32)

    # square
    square = pypto.mul(tensor_value_fp32, tensor_value_fp32)

    # mean_res
    mean_res = pypto.mul(square, mean_coff)

    # reduce sum
    reduce_asum = pypto.sum(mean_res, -1, keepdim=True)
    reduce_sum = pypto.add(reduce_asum, eps)

    # sqrt
    reduce_sqrt = pypto.sqrt(reduce_sum)

    # div
    res_div = pypto.div(tensor_value_fp32, reduce_sqrt)

    # gamma mul
    res = pypto.mul(res_div, gamma)
    res_add = pypto.add(res, bias)

    # cast
    y_bf16 = pypto.cast(res_add, input_dtype)

    return y_bf16


def rope_data(x1, x2, cos, sin, tile_shape):
    pypto.set_vec_tile_shapes(*tile_shape)
    o1 = pypto.sub(pypto.mul(x1, cos), pypto.mul(x2, sin))
    o2 = pypto.add(pypto.mul(x2, cos), pypto.mul(x1, sin))
    # concat
    res = pypto.concat([o1, o2], 2)

    # cast
    y_bf16 = pypto.cast(res, pypto.DT_BF16)
    return y_bf16


@allow_in_graph
def attention(
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        input_layernorm_weight: torch.Tensor,
        input_layernorm_bias: torch.Tensor,
        qkv_proj_scale: torch.Tensor,
        qkv_proj_offset: torch.Tensor,
        qkv_proj_weight: torch.Tensor,
        qkv_proj_quant_bias: torch.Tensor,
        qkv_proj_deq_scale: torch.Tensor,
        q_norm_weight: torch.Tensor,
        q_norm_bias: torch.Tensor,
        k_norm_weight: torch.Tensor,
        k_norm_bias: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        actual_seq_lens: torch.Tensor,
        slot_mapping: torch.Tensor,
        enable_residual: bool,
        eps: float,
        num_decode_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    for item in [hidden_states,
                 residual,
                 input_layernorm_weight,
                 input_layernorm_bias,
                 qkv_proj_scale,
                 qkv_proj_offset,
                 qkv_proj_weight,
                 qkv_proj_quant_bias,
                 qkv_proj_deq_scale,
                 q_norm_weight,
                 q_norm_bias,
                 k_norm_weight,
                 k_norm_bias,
                 cos,
                 sin,
                 key_cache,
                 value_cache,
                 block_tables,
                 actual_seq_lens,
                 slot_mapping]:
        if isinstance(item, FakeTensor):
            return None, None

    dtype, device = hidden_states.dtype, hidden_states.device
    bs = hidden_states.shape[0]
    total_head_size = qkv_proj_weight.shape[1]
    head_size = q_norm_weight.shape[0]
    n1 = total_head_size // head_size - 2
    q_shape = (bs, n1, head_size)
    out_torch = torch.empty(q_shape, dtype=dtype, device=device)
    q_tmp = torch.empty((128 * 1, n1 * head_size), dtype=dtype, device=device)
    k_tmp = torch.empty((bs, head_size), dtype=dtype, device=device)
    v_tmp = torch.empty((bs, head_size), dtype=dtype, device=device)
    hidden_size = qkv_proj_scale.shape[0]
    residual_tmp = torch.empty((bs, hidden_size), dtype=dtype, device=device)

    debug_str = os.environ.get("HIGH_PERFORMANCE", "True").lower()
    unroll_list = []
    if debug_str in ["true", "1", "yes"]:
        unroll_list = [8, 4, 2, 1]


    inputs = [
        block_tables,
        actual_seq_lens,
        slot_mapping,
        hidden_states,
        residual,
        input_layernorm_weight,
        input_layernorm_bias,
        qkv_proj_scale,
        qkv_proj_offset,
        qkv_proj_weight,
        qkv_proj_quant_bias,
        qkv_proj_deq_scale,
        q_norm_weight,
        q_norm_bias,
        k_norm_weight,
        k_norm_bias,
        cos,
        sin,
        out_torch,
        q_tmp,
        k_tmp,
        v_tmp,
        residual_tmp,
        key_cache,
        value_cache
    ]


    ifa_func_kernel(*inputs, unroll_list, eps)
    return out_torch, residual_tmp



@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128,
                     "ready_on_host_tensors": ["block_table", "kv_act_seqs"]
                    },
    host_options={"compile_monitor_enable": 0},
)
def ifa_func_kernel(
    block_table: pypto.Tensor(),
    kv_act_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    index: pypto.Tensor(),
    x: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    residual_input: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    x_gamma: pypto.Tensor(),
    x_bias: pypto.Tensor(),
    x_scale: pypto.Tensor(),
    x_offset: pypto.Tensor(),
    weight: pypto.Tensor(),
    quant_bias: pypto.Tensor(),
    deq_scale: pypto.Tensor(),
    q_gamma: pypto.Tensor(),
    q_bias: pypto.Tensor(),
    k_gamma: pypto.Tensor(),
    k_bias: pypto.Tensor(),
    cos: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    atten_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    q_tmp: pypto.Tensor([...], pypto.DT_BF16),
    k_tmp: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    v_tmp: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    residual: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    key_cache: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    value_cache: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    unroll_list,
    eps,
):
    bs_tile = 8
    pypto.experimental.set_operation_options(combine_axis=True)
    # 4. 得到动态tensor的shape
    bs = x.shape[0]
    hidden_size = x.shape[1]
    total_head_size = weight.shape[1]

    head_size = 128
    x_mean_coff = 1.0 / x.shape[-1]
    qk_mean_coff = 1.0 / head_size
    half_rotary_dim = cos.shape[-1]
    rotary_dim = cos.shape[-1] * 2
    stay_dim = head_size - rotary_dim

    q_size = q_tmp.shape[-1]
    kv_size = k_tmp.shape[-1]
    q_num_head = q_size // head_size
    kv_num_head = kv_size // head_size
    kv_index = q_num_head + kv_num_head

    bs_loop = (bs + bs_tile - 1) // bs_tile
    calc_dtype = pypto.DT_FP32
    input_dtype = x.dtype
    tiling_value = 128
    vec_tile_value = 5120
    q_batch_tile = 4

    # 4. 定义动态函数
    shape_k = key_cache.shape
    shape_act_seqs = kv_act_seqs.shape
    shape_hidden_states = x.shape
    weigh_shape = weight.shape
    q_gamma_shape = q_gamma.shape

    atten_cfg, tile_cfg = get_qwen_common_config()
    softmax_scale = atten_cfg.softmax_scale

    bs_scalar = shape_hidden_states[0]
    n1 = weigh_shape[1] // q_gamma_shape[0] - 2
    block_num_scalar = shape_k[0]
    block_size = shape_k[1]
    n2 = shape_k[2]
    dn = shape_k[3]
    b_scalar = shape_act_seqs[0]

    dtype = key_cache.dtype
    group = n1 // n2

    g_tile = tile_cfg.g_tile
    s2_tile = tile_cfg.s2_tile
    c1_tile = tile_cfg.c1_tile_shape
    v1_tile = tile_cfg.v1_tile_shape
    c2_tile = tile_cfg.c2_tile_shape
    v2_tile = tile_cfg.v2_tile_shape

    # 5. 得到动态tensor的shape
    s1_scalar = bs_scalar // b_scalar
    g = n1 // n2
    g_loop = g // g_tile

    q_2d_shape = (128 * 1 * n1, dn)
    kv_cache_2d_shape = (block_num_scalar * block_size, n2 * dn)

    pypto.set_vec_tile_shapes(5120)
    x_gamma_2d = pypto.reshape(x_gamma, [1, 5120], inplace=True)
    x_bias_2d = pypto.reshape(x_bias, [1, 5120], inplace=True)
    x_scale_2d = pypto.reshape(x_scale, [1, 5120], inplace=True)
    x_offset_2d = pypto.reshape(x_offset, [1, 5120], inplace=True)
    quant_bias_2d = pypto.reshape(quant_bias, [1, 1792], inplace=True)
    deq_scale_2d = pypto.reshape(deq_scale, [1, 1792], inplace=True)
    q_gamma_2d = pypto.reshape(q_gamma, [1, 1, head_size], inplace=True)
    q_bias_2d = pypto.reshape(q_bias, [1, 1, head_size], inplace=True)
    k_gamma_2d = pypto.reshape(k_gamma, [1, 1, head_size], inplace=True)
    k_bias_2d = pypto.reshape(k_bias, [1, 1, head_size], inplace=True)

    pypto.set_vec_tile_shapes(1, 1, head_size)
    q_gamma_2d_fp32 = pypto.cast(q_gamma_2d, calc_dtype)
    q_bias_2d_fp32 = pypto.cast(q_bias_2d, calc_dtype)
    k_gamma_2d_fp32 = pypto.cast(k_gamma_2d, calc_dtype)
    k_bias_2d_fp32 = pypto.cast(k_bias_2d, calc_dtype)
    q_gamma_expand = pypto.expand_clone(q_gamma_2d_fp32, [1, q_num_head, head_size])
    q_bias_expand = pypto.expand_clone(q_bias_2d_fp32, [1, q_num_head, head_size])
    k_gamma_expand = pypto.expand_clone(k_gamma_2d_fp32, [1, kv_num_head, head_size])
    k_bias_expand = pypto.expand_clone(k_bias_2d_fp32, [1, kv_num_head, head_size])
    key_cache_2d = pypto.reshape(key_cache, kv_cache_2d_shape, inplace=True)
    value_cache_2d = pypto.reshape(value_cache, kv_cache_2d_shape, inplace=True)

    for bs_idx in pypto.loop(bs_loop, name="LOOP_ATT_PRE_L0", idx_name="bs_idx"):
        act_bs_tile = (bs - bs_idx * bs_tile).min(bs_tile)

        # rms norm
        x_tile = pypto.view(x, [bs_tile, hidden_size], [bs_idx * bs_tile, 0],
                                                valid_shape=[act_bs_tile, hidden_size])
        # init
        pypto.set_vec_tile_shapes(1, vec_tile_value)
        x_tile_fp32 = pypto.cast(x_tile, calc_dtype)
        # add
        residual_input_tile = pypto.view(residual_input, [bs_tile, hidden_size], [bs_idx * bs_tile, 0],
                                        valid_shape=[act_bs_tile, hidden_size])
        residual_input_tile_fp32 = pypto.cast(residual_input_tile, calc_dtype)
        x_f32 = pypto.add(residual_input_tile_fp32, x_tile_fp32)  # tile_x

        square = pypto.mul(x_f32, x_f32)  # square
        mean_res = pypto.mul(square, x_mean_coff)  # mean_res = square * mean_coff
        reduce_asum = pypto.sum(mean_res, -1, keepdim=True)  # reduce_asum = mean_res.sum(dim=-1, keepdim=True)
        reduce_sum = pypto.add(reduce_asum, eps)  # reduce_sum = reduce_asum + eps
        reduce_sqrt = pypto.sqrt(reduce_sum)  # reduce_sqrt = torch.sqrt(reduce_sum)
        res_div = pypto.div(x_f32, reduce_sqrt)  # res_div = x_f32 / reduce_sqrt
        residual_bf16 = pypto.cast(x_f32, input_dtype)
        x_int8 = pypto.tensor([bs_tile, hidden_size], pypto.DT_INT8, "x_int8")

        for tmp_idx in range(bs_tile):
            pypto.set_vec_tile_shapes(1, vec_tile_value)
            x_gamma_2d_fp32 = pypto.cast(x_gamma_2d, calc_dtype)
            x_bias_2d_fp32 = pypto.cast(x_bias_2d, calc_dtype)
            x_scale_2d_fp32 = pypto.cast(x_scale_2d, calc_dtype)
            x_offset_2d_fp32 = pypto.cast(x_offset_2d, calc_dtype)

            res_div_single = pypto.view(res_div, [1, hidden_size], [tmp_idx, 0])

            res = pypto.mul(res_div_single, x_gamma_2d_fp32)  # res = res_div * weight
            res_add = pypto.add(res, x_bias_2d_fp32)
            x_norm = pypto.cast(res_add, input_dtype)

            # x quant
            pypto.set_vec_tile_shapes(1, vec_tile_value)
            x_norm_fp32 = pypto.cast(x_norm, calc_dtype)  # bf16 -> fp32
            x_mul = pypto.mul(x_norm_fp32, x_scale_2d_fp32)
            x_add = pypto.add(x_mul, x_offset_2d_fp32)
            x_int32 = pypto.cast(x_add, pypto.DT_INT32, pypto.CastMode.CAST_RINT)  # Align ascendC
            x_fp16 = pypto.cast(x_int32, pypto.DT_FP16)
            x_int8[tmp_idx:tmp_idx + 1, 0:] = pypto.cast(x_fp16, pypto.DT_INT8, satmode=pypto.SaturationMode.ON)

        pypto.set_cube_tile_shapes([32, 32], [256, 512], [256, 256])
        tmp_c = pypto.matmul(x_int8, weight, pypto.DT_INT32)
        pypto.set_vec_tile_shapes(bs_tile, total_head_size)
        mm_add = pypto.add(tmp_c, quant_bias_2d)
        mm_fp32 = pypto.cast(mm_add, calc_dtype)  # int32 -> fp32
        mm_deq_scale = pypto.mul(mm_fp32, deq_scale_2d)
        mm_bf16 = pypto.cast(mm_deq_scale, input_dtype)  # fp32 -> bf16

        pypto.set_vec_tile_shapes(bs_tile, head_size)
        mm_3d = pypto.reshape(mm_bf16, [bs_tile, total_head_size // head_size, head_size], inplace=True)
        pypto.set_vec_tile_shapes(bs_tile, tiling_value, head_size)

        # split
        q_tile = pypto.view(mm_3d, [bs_tile, q_num_head, head_size], [0, 0, 0],
                            valid_shape=[act_bs_tile, q_num_head, head_size])
        k_tile = pypto.view(mm_3d, [bs_tile, kv_num_head, head_size], [0, q_num_head, 0],
                            valid_shape=[act_bs_tile, kv_num_head, head_size])
        v_tile = pypto.view(mm_3d, [bs_tile, kv_num_head, head_size], [0, kv_index, 0],
                            valid_shape=[act_bs_tile, kv_num_head, head_size])

        # rms norm
        q_norm = rms_norm_bias(q_tile, q_gamma_expand, q_bias_expand, qk_mean_coff, eps,
                            [q_batch_tile, q_num_head, head_size])
        k_norm = rms_norm_bias(k_tile, k_gamma_expand, k_bias_expand, qk_mean_coff, eps,
                            [q_batch_tile, kv_num_head, head_size])

        q_rot = pypto.view(q_norm, [bs_tile, q_num_head, rotary_dim], [0, 0, 0],
                        valid_shape=[act_bs_tile, q_num_head, rotary_dim])
        q_pass = pypto.view(q_norm, [bs_tile, q_num_head, stay_dim], [0, 0, rotary_dim],
                            valid_shape=[act_bs_tile, q_num_head, stay_dim])

        k_rot = pypto.view(k_norm, [bs_tile, kv_num_head, rotary_dim], [0, 0, 0],
                        valid_shape=[act_bs_tile, kv_num_head, rotary_dim])
        k_pass = pypto.view(k_norm, [bs_tile, kv_num_head, stay_dim], [0, 0, rotary_dim],
                            valid_shape=[act_bs_tile, kv_num_head, stay_dim])

        # apply rope
        # cast
        pypto.set_vec_tile_shapes(q_batch_tile, q_num_head, head_size)
        cos_tile = pypto.view(cos, [bs_tile, 1, half_rotary_dim], [bs_idx * bs_tile, 0, 0],
                            valid_shape=[act_bs_tile, 1, half_rotary_dim])
        sin_tile = pypto.view(sin, [bs_tile, 1, half_rotary_dim], [bs_idx * bs_tile, 0, 0],
                            valid_shape=[act_bs_tile, 1, half_rotary_dim])
        q_fp32 = pypto.cast(q_rot, calc_dtype)
        k_fp32 = pypto.cast(k_rot, calc_dtype)
        cos_fp32 = pypto.cast(cos_tile, calc_dtype)
        sin_fp32 = pypto.cast(sin_tile, calc_dtype)

        # q split
        q1 = pypto.view(q_fp32, [bs_tile, q_num_head, half_rotary_dim], [0, 0, 0],
                        valid_shape=[act_bs_tile, q_num_head, half_rotary_dim])
        q2 = pypto.view(q_fp32, [bs_tile, q_num_head, half_rotary_dim], [0, 0, half_rotary_dim],
                        valid_shape=[act_bs_tile, q_num_head, half_rotary_dim])

        # rope data
        q_rope = rope_data(q1, q2, cos_fp32, sin_fp32, [q_batch_tile, q_num_head, half_rotary_dim])
        q_cat = pypto.concat([q_rope, q_pass], 2)

        # k split
        k1 = pypto.view(k_fp32, [bs_tile, kv_num_head, half_rotary_dim], [0, 0, 0],
                        valid_shape=[act_bs_tile, kv_num_head, half_rotary_dim])
        k2 = pypto.view(k_fp32, [bs_tile, kv_num_head, half_rotary_dim], [0, 0, half_rotary_dim],
                        valid_shape=[act_bs_tile, kv_num_head, half_rotary_dim])

        # rope data
        k_rope = rope_data(k1, k2, cos_fp32, sin_fp32, [q_batch_tile, kv_num_head, half_rotary_dim])
        k_cat = pypto.concat([k_rope, k_pass], 2)

        # post process
        q_res = pypto.reshape(q_cat, [bs_tile, q_size], valid_shape=[act_bs_tile, q_size])
        k_res = pypto.reshape(k_cat, [bs_tile, kv_size], valid_shape=[act_bs_tile, kv_size])
        v_res = pypto.reshape(v_tile, [bs_tile, kv_size], valid_shape=[act_bs_tile, kv_size])

        # # 9. 将结果搬运到输出tensor上
        # # update output
        q_tmp[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = q_res

        residual[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = residual_bf16

        b_ofs = bs_idx * bs_tile
        b_valid = (b_scalar - bs_idx * bs_tile).min(bs_tile)
        index_view = pypto.view(index, [bs_tile], [b_ofs], valid_shape=[b_valid])
        index_view = pypto.reshape(index_view, [bs_tile, 1], valid_shape=[b_valid, 1])
        pypto.set_vec_tile_shapes(bs_tile, 128)
        key_cache.move(pypto.scatter_update(key_cache_2d, -2, index_view, k_res))
        value_cache.move(pypto.scatter_update(value_cache_2d, -2, index_view, v_res))

    q_2d = pypto.reshape(q_tmp, q_2d_shape, inplace=True)
    # Q常驻，0代表第一组mmad，4代表4次matmul合并
    pypto.set_pass_options(cube_l1_reuse_setting={0: 4})
    # 6. 实现kernel逻辑，循环展开B动态轴
    for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"):
        for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx"):
            cur_seq = kv_act_seqs[b_idx] - (s1_scalar - 1 - s1_idx)
            s2_loop = (cur_seq + s2_tile - 1) // s2_tile
            for n2_idx in pypto.loop(n2, name="LOOP_n2", idx_name="n2_idx"):
                for g_idx in pypto.loop(g_loop, name="LOOP_g", idx_name="g_idx"):
                    oi_update = pypto.tensor([g_tile, dn], pypto.DT_FP32, "oi_update")
                    sum_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update")
                    max_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update")
                    for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx", unroll_list=unroll_list):
                        block_num = s2_tile // block_size
                        idx = s2_idx * block_num
                        bs_ofs = b_idx * s1_scalar + s1_idx
                        n1g_ofs = n2_idx * group + g_idx * g_tile
                        actual_s2_tile = (cur_seq - s2_idx * s2_tile).min(s2_tile)
                        oi_ofs = [bs_ofs, n1g_ofs, 0]
                        # 5. 按照计算图实现运算逻辑，设置set_vec_tile_shapes时应尽可能用满UB，但不要超过UB的大小。
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        qi = pypto.view(q_2d, [g_tile, dn], [bs_ofs * n1 + n1g_ofs, 0])
                        kj_assemble = pypto.tensor([s2_tile, dn], dtype, "kj_assemble")
                        for i in range(block_num):
                            block_idx = block_table[b_idx, idx + i]
                            block_idx_valid = block_idx.max(0)
                            kj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                pypto.view(key_cache_2d, [block_size, dn], [block_idx_valid * block_size, 0])
                        kj_assemble = pypto.view(kj_assemble, [s2_tile, dn], [0, 0], valid_shape=[s2_tile, dn])

                        # c1
                        # 6. 下面是flash attention的计算逻辑
                        pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
                        sij = pypto.matmul(qi, kj_assemble, pypto.DT_FP32, a_trans=False,
                                        b_trans=True)
                        sij = pypto.view(sij, [g_tile, s2_tile], [0, 0],
                                        valid_shape=[g_tile, actual_s2_tile])
                        # v1
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        if pypto.is_loop_begin(s2_idx):
                            pypto.set_pass_options(sg_set_scope=3)
                            sij_scale = pypto.mul(sij, softmax_scale)
                            tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)

                            tsub = pypto.sub(sij_scale, tilda_mij)
                            tilda_pij = pypto.exp(tsub)
                            tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
                            sum_update[:] = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                            max_update[:] = tilda_mij
                            pypto.set_pass_options(sg_set_scope=-1)

                            # c2
                            vj_assemble = pypto.tensor([s2_tile, dn], dtype, "vj_assemble")
                            for i in range(block_num):
                                block_idx = block_table[b_idx, idx + i]
                                block_idx_valid = block_idx.max(0)
                                vj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                    pypto.view(value_cache_2d, [block_size, dn], [block_idx_valid * block_size, 0])
                            vj_assemble = pypto.view(vj_assemble, [s2_tile, dn],
                                                    [0, 0], valid_shape=[actual_s2_tile, dn])
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

                            # c2
                            vj_assemble = pypto.tensor([s2_tile, dn], dtype, "vj_assemble")
                            for i in range(block_num):
                                block_idx = block_table[b_idx, idx + i]
                                block_idx_valid = block_idx.max(0)
                                vj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                    pypto.view(value_cache_2d, [block_size, dn],
                                                [block_idx_valid * block_size, 0])
                            vj_assemble = pypto.view(vj_assemble, [s2_tile, dn],
                                                    [0, 0], valid_shape=[actual_s2_tile, dn])
                            pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                            oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)

                            # v2
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_update[:] = oi_update * update_mul + oi_tmp
                        if pypto.is_loop_end(s2_idx):
                            oi_final = pypto.div(oi_update, sum_update)
                            pypto.set_vec_tile_shapes(16, v2_tile[0], v2_tile[1])
                            oi_final_3d = pypto.cast(
                                pypto.reshape(oi_final, [1, g_tile, dn]), dtype)
                            # 7. 将结果搬运到输出tensor上
                            pypto.assemble(oi_final_3d, oi_ofs, atten_out)



def get_qwen_common_config(device="cpu"):
    b = 16
    s1 = 1
    s2 = 8192
    q_d = 128
    n1 = 12
    n2 = 1
    kv_layout = "PA_BSND"
    softmax_scale = q_d ** -0.5
    block_table_batch = b
    block_size = 128
    kv_num_blocks = b * ((s2 + block_size - 1) // block_size)
    hidden_size = 5120

    # 创建 torch tensor 类型的 actual_seq
    actual_seq_values = [8, 6, 6, 6, 8, 6, 6, 6, 8, 6, 6, 6, 8, 6, 6, 6, 8, 6, 6, 6, 8, 6, 6, 6]
    actual_seq_values = [s2] * b
    actual_seq_tensor = torch.tensor(actual_seq_values, dtype=torch.int32, device=device)
    atten_cfg = AttentionConfig(b=b, s1=s1, s2=s2, n1=n1, n2=n2, softmax_scale=softmax_scale, kv_layout=kv_layout,
                                q_d=q_d, kv_d=q_d, block_table_batch=block_table_batch, kv_num_blocks=kv_num_blocks,
                                actual_seq=actual_seq_tensor, hidden_size=hidden_size)  # 传入 tensor
    atten_cfg.max_num_blocks_per_query = (s2 + block_size - 1) // block_size
    cube_tile = 128
    vector_tile = 128
    s2_tile = 512
    tile_cfg = AttentionTileConfig(
        n1,
        s2_tile,
        [[cube_tile, cube_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
        [vector_tile, s2_tile],
        [[cube_tile, cube_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
        [vector_tile, vector_tile])
    return atten_cfg, tile_cfg