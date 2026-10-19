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
MLA Indexer Prolog Quantization Module

This module implements fused MLA Prolog and Lightning Indexer Prolog computation
for DeepSeek V32 model. It combines both operators to enable pipeline parallelism
and improve overall performance.

Main Functions:
    - mla_indexer_prolog_quant_p: Fused computation for prefill phase
    - mla_indexer_prolog_quant_d: Fused computation for decode phase

Example:
    See deepseekv32_mla_indexer_prolog_quant.py for usage examples.
"""

import math
import pypto
from .lightning_indexer_prolog_quant_impl import rope_3d, quant_layer_norm, prolog_quant, quant_rope_2d
from .mla_prolog_quant_impl import quant, dequant, MlaTileConfig, rms_norm, rope_3d_v2, rope_v2, \
                                    MlaQuantInputs, k_nope_quant


L0M_INDEX = 0
L1M_INDEX = 1
L0K_INDEX = 2
L1K_INDEX = 3
L0N_INDEX = 4
L1N_INDEX = 5
SCATTER_DIM = -2

VEC_TILE_4 = 4
VEC_TILE_32 = 32


def pre_compute_2d(
    token_x: pypto.Tensor,
    w_dq: pypto.Tensor,
    w_uq_qr: pypto.Tensor,
    w_dkv_kr: pypto.Tensor,
    gamma_cq: pypto.Tensor,
    epsilon_cq: float,
    quant_inputs: MlaQuantInputs,
    tile_config: MlaTileConfig
) -> pypto.Tensor:
    """Pre-compute query and key-value projections with optional quantization.

    Performs the initial computation steps for MLA prolog:
    1. Query path: token_x -> w_dq -> RMSNorm -> w_uq_qr
    2. Key-value path: token_x -> w_dkv_kr

    Supports optional quantization at different stages (quant_a and quant_b).

    Args:
        token_x: Input token tensor, shape (bs, h)
        w_dq: Down-projection weight for query, shape (h, q_lora_rank)
        w_uq_qr: Up-projection weight for query and RoPE, shape (q_lora_rank, n*q_head_dim)
        w_dkv_kr: Down-projection weight for key-value and RoPE, shape (h, kv_lora_rank+rope_dim)
        gamma_cq: RMSNorm scale parameter for query, shape (q_lora_rank,)
        epsilon_cq: RMSNorm epsilon parameter
        quant_inputs: MlaQuantInputs object containing quantization scales:
            - dequant_scale_w_dq: Dequantization scale for w_dq (if quant_a)
            - dequant_scale_w_dkv_kr: Dequantization scale for w_dkv_kr (if quant_a)
            - dequant_scale_w_uq_qr: Dequantization scale for w_uq_qr (if quant_b)
            - smooth_scales_cq: Smooth quantization factor (if has_smooth)
        tile_config: MlaTileConfig object containing tiling parameters

    Returns:
        List containing:
            - q_b_proj: Query projection result, shape (bs, n*q_head_dim)
            - compressed_kv: Compressed key-value result, shape (bs, kv_lora_rank+rope_dim)
            - q_norm or norm_res: Normalized query (quantized or not)
            - q_norm_scale or None: Quantization scale (if quant_b) or None

    Note:
        The function supports three quantization modes:
        - quant_a: Quantize input and weights w_dq, w_dkv_kr
        - quant_b: Quantize normalized query and weight w_uq_qr
        - smooth: Apply smooth quantization factor before quant_b
    """
    dequant_scale_w_dq = quant_inputs.dequant_scale_w_dq
    dequant_scale_w_dkv_kr = quant_inputs.dequant_scale_w_dkv_kr
    dequant_scale_w_uq_qr = quant_inputs.dequant_scale_w_uq_qr

    is_quant_a = (dequant_scale_w_dq is not None) and (dequant_scale_w_dkv_kr is not None)
    is_quant_b = dequant_scale_w_uq_qr is not None

    smooth_scales_cq = quant_inputs.smooth_scales_cq
    is_smooth = smooth_scales_cq is not None

    bs = token_x.shape[0]
    k = token_x.shape[1]
    q_lora_rank = w_dq.shape[1]

    dtype = token_x.dtype
    dtype_quant_a_out = pypto.DT_INT32 if is_quant_a else dtype
    dtype_quant_b_out = pypto.DT_INT32 if is_quant_b else dtype
    qkv_pre_res = []

    pypto.set_semantic_label("pre_reshape")

    mv = tile_config.mv_tile

    if is_quant_a:
        pypto.set_vec_tile_shapes(mv, q_lora_rank)
        pypto.set_cube_tile_shapes([tile_config.pre_quant_cube_tile[0], tile_config.pre_quant_cube_tile[1]],
                                   [256, 256], [256, 256])
        pypto.set_semantic_label("Quant_x")
        quant_res = quant(token_x)
        input_quant = quant_res[0]
        input_quant_scale = quant_res[1]
        pypto.set_semantic_label("QuantMatmul_qa")
        q_a_proj = pypto.matmul(input_quant, w_dq, dtype_quant_a_out)
        pypto.set_semantic_label("Dequant_qa")
        q_a_proj[:] = dequant(dtype, q_a_proj, input_quant_scale, dequant_scale_w_dq)
    else:
        pypto.set_cube_tile_shapes([tile_config.pre_quant_cube_tile[0], tile_config.pre_quant_cube_tile[1]],
                                   [tile_config.pre_quant_cube_tile[2], tile_config.pre_quant_cube_tile[3]],
                                   [tile_config.pre_quant_cube_tile[4], tile_config.pre_quant_cube_tile[5]])
        pypto.set_semantic_label("Matmul_qa")
        q_a_proj = pypto.matmul(token_x, w_dq, pypto.DT_BF16)

    pypto.set_vec_tile_shapes(mv, q_lora_rank)
    pypto.set_semantic_label("RmsNorm_qa")
    norm_res = rms_norm(q_a_proj, gamma_cq, epsilon_cq)

    if is_quant_b:
        pypto.set_vec_tile_shapes(mv, q_lora_rank)
        pypto.set_semantic_label("Quant_qMnRes")
        if is_smooth:
            quant_res = quant(norm_res, True, True, smooth_scales_cq)
        else:
            quant_res = quant(norm_res, True, False)
        norm_quant = quant_res[0]
        norm_quant_scale = quant_res[1]
        pypto.set_semantic_label("QuantMatmul_qb")
        pypto.set_cube_tile_shapes([tile_config.cube_qb_tile[0], tile_config.cube_qb_tile[1]],
                                   [tile_config.cube_qb_tile[2], tile_config.cube_qb_tile[3]],
                                   [tile_config.cube_qb_tile[4], tile_config.cube_qb_tile[5]])
        q_b_proj_tmp = pypto.matmul(norm_quant, w_uq_qr, dtype_quant_b_out)
        pypto.set_semantic_label("Dequant_qb")
        q_b_proj = dequant(dtype, q_b_proj_tmp, norm_quant_scale, dequant_scale_w_uq_qr)
    else:
        pypto.set_cube_tile_shapes([tile_config.cube_qb_tile[0], tile_config.cube_qb_tile[1]],
                                   [tile_config.cube_qb_tile[2], tile_config.cube_qb_tile[3]],
                                   [tile_config.cube_qb_tile[4], tile_config.cube_qb_tile[5]])
        pypto.set_semantic_label("Matmul_qb")
        q_b_proj = pypto.matmul(norm_res, w_uq_qr, dtype)

    qkv_pre_res.append(q_b_proj)

    ####### kv ##########
    if is_quant_a:
        pypto.set_vec_tile_shapes(mv, q_lora_rank)
        pypto.set_cube_tile_shapes(tile_config.m_tile, [256, 256], [256, 256])
        pypto.set_semantic_label("QuantMatmul_kva")
        compressed_kv = pypto.matmul(input_quant, w_dkv_kr, dtype_quant_a_out)
        pypto.set_semantic_label("Dequant_kva")
        compressed_kv[:] = dequant(dtype, compressed_kv, input_quant_scale, dequant_scale_w_dkv_kr)
    else:
        pypto.set_cube_tile_shapes([tile_config.pre_quant_cube_tile[0], tile_config.pre_quant_cube_tile[1]],
                                   [tile_config.pre_quant_cube_tile[2], tile_config.pre_quant_cube_tile[3]],
                                   [tile_config.pre_quant_cube_tile[4], tile_config.pre_quant_cube_tile[5]])
        pypto.set_semantic_label("Matmul_kva")
        compressed_kv = pypto.matmul(token_x, w_dkv_kr, dtype)

    qkv_pre_res.append(compressed_kv)
    if is_quant_b:
        qkv_pre_res.append(norm_quant)
        qkv_pre_res.append(norm_quant_scale)
    else:
        qkv_pre_res.append(norm_res)
    return qkv_pre_res


def mla_indexer_prolog_quant_compute(
    token_x, mla_w_dq, mla_w_uq_qr, mla_dequant_scale, mla_w_uk, mla_w_dkv_kr, mla_gamma_cq,
    mla_gamma_ckv, cos, sin, cache_index, mla_kv_cache, mla_kr_cache,
    mla_k_scale_cache, ip_w_qb_in, ip_w_qb_scale_in, ip_wk_in, ip_w_proj_in,
    ip_ln_gamma_k_in, ip_ln_beta_k_in, ip_hadamard_q_in, ip_hadamard_k_in,
    ip_k_cache, ip_k_cache_scale, mla_query_nope_out, mla_query_rope_out,
    mla_q_norm_out, mla_q_norm_scale_out,
    mla_kv_cache_out, mla_kr_cache_out,
    mla_k_scale_cache_out, ip_q_int8_out, ip_q_scale_out, ip_k_int8_out,
    ip_k_scale_out, ip_weights_out, mla_epsilon_cq, mla_epsilon_ckv,
    mla_cache_mode, mla_tile_config,
    ip_attrs, ip_configs, rope_cfg
):
    dtype = token_x.dtype
    h = token_x.shape[1]
    n1 = mla_w_uk.shape[0]
    q_lora_rank = mla_w_dq.shape[1]
    qk_nope_head_dim = mla_w_uk.shape[1]
    kv_lora_rank = mla_w_uk.shape[2]
    qk_rope_head_dim = sin.shape[1]
    head_num = ip_w_proj_in.shape[1]
    head_dim = ip_hadamard_q_in.shape[0]
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    t = token_x.shape[0]
    quant_inputs = MlaQuantInputs()

    k_cache_index_2d = pypto.reshape(cache_index, [t, 1], inplace=True)
    w_qb_scale = pypto.reshape(ip_w_qb_scale_in, [1, head_num * head_dim], inplace=True)
    gamma_2d = pypto.reshape(ip_ln_gamma_k_in, [1, ip_ln_gamma_k_in.shape[0]], inplace=True)
    beta_2d = pypto.reshape(ip_ln_beta_k_in, [1, ip_ln_beta_k_in.shape[0]], inplace=True)
    if mla_dequant_scale is not None:
        dequant_scale_wuqr_reshape = pypto.reshape(mla_dequant_scale, [1, n1 * q_head_dim], inplace=True)
        quant_inputs.dequant_scale_w_uq_qr = dequant_scale_wuqr_reshape

    unroll_list = mla_tile_config.unroll_list
    for bs_offset, unroll_length in pypto.loop_unroll(0, t, 1, name="MLA_BS_LOOP", idx_name="bs_offset",
                                                      unroll_list=unroll_list, ):
        tile_bs = unroll_length
        output_offset = [bs_offset, 0, 0]

        pypto.set_vec_tile_shapes(tile_bs, 128)
        x_view = pypto.view(token_x, [tile_bs, h], [bs_offset, 0])
        q_kv = pre_compute_2d(x_view, mla_w_dq, mla_w_uq_qr, mla_w_dkv_kr, mla_gamma_cq, \
                            mla_epsilon_cq, quant_inputs, mla_tile_config)
        q = q_kv[0]
        kv_tmp = q_kv[1]

        ############# q_norm #############
        pypto.set_semantic_label("Assemble_qNorm")
        q_norm = q_kv[2]
        pypto.set_vec_tile_shapes(tile_bs, q_lora_rank)
        pypto.assemble(q_norm, [bs_offset, 0], mla_q_norm_out)
        q_norm_scale = q_kv[3]
        pypto.set_vec_tile_shapes(tile_bs, 1)
        pypto.assemble(q_norm_scale, [bs_offset, 0], mla_q_norm_scale_out)

        ########### q ##############
        q_tmp = pypto.reshape(q, [tile_bs, n1, q_head_dim])
        pypto.set_semantic_label("Prepare_qNope")
        q_nope = pypto.view(q_tmp, [tile_bs, n1, qk_nope_head_dim], [0, 0, 0])
        tile_shape = [min(16, tile_bs), 32, qk_nope_head_dim]
        pypto.set_vec_tile_shapes(*tile_shape)
        q_nope_trans = pypto.transpose(q_nope, 0, 1)

        m = mla_tile_config.m_tile
        pypto.set_semantic_label("Matmul_qNope_wUk")
        pypto.set_cube_tile_shapes([m, m], [128, 128], [128, 128])
        q_nope_new = pypto.matmul(q_nope_trans, mla_w_uk, dtype)

        tile_shape = [1, min(32, tile_bs), kv_lora_rank]
        pypto.set_vec_tile_shapes(*tile_shape)
        q_nope_new_trans = pypto.transpose(q_nope_new, 0, 1)

        pypto.set_semantic_label("Assemble_queryOut")
        pypto.set_vec_tile_shapes(mla_tile_config.q_vec_tile0, mla_tile_config.q_vec_tile1, 128)
        pypto.assemble(q_nope_new_trans, output_offset, mla_query_nope_out)

        if tile_bs >= 128:
            pypto.set_vec_tile_shapes(mla_tile_config.q_vec_tile0, mla_tile_config.q_vec_tile1, 64)
        q_pe_view = pypto.view(q_tmp, [tile_bs, n1, qk_rope_head_dim], [0, 0, qk_nope_head_dim])
        cos_2d_view = pypto.view(cos, [tile_bs, qk_rope_head_dim], [bs_offset, 0])
        sin_2d_view = pypto.view(sin, [tile_bs, qk_rope_head_dim], [bs_offset, 0])
        pypto.set_semantic_label("Rope_qRope")
        q_rope_view = rope_3d_v2(q_pe_view, cos_2d_view, sin_2d_view)
        pypto.set_semantic_label("Assemble_qRope")
        pypto.set_vec_tile_shapes(mla_tile_config.q_vec_tile0, mla_tile_config.q_vec_tile1, 64)
        pypto.assemble(q_rope_view, output_offset, mla_query_rope_out)

        ########### RoPE #################
        pypto.set_vec_tile_shapes(mla_tile_config.k_vec_tile0, mla_tile_config.k_vec_tile1)
        pypto.set_semantic_label("RotaryPosEmb")
        k_pe_view = pypto.view(kv_tmp, [tile_bs, qk_rope_head_dim], [0, kv_lora_rank])

        k_rope_2d = rope_v2(k_pe_view, cos_2d_view, sin_2d_view, rope_cfg)

        ############### kNope ##############

        compressed_kv = pypto.view(kv_tmp, [tile_bs, kv_lora_rank], [0, 0])
        pypto.set_semantic_label("RmsNorm_compressedkv")
        pypto.set_vec_tile_shapes(mla_tile_config.k_vec_tile0, mla_tile_config.k_vec_tile1)
        k_nope = rms_norm(compressed_kv, mla_gamma_ckv, mla_epsilon_ckv)

        ########### kNope Quant ############
        pypto.set_semantic_label("Quant_knope")
        pypto.set_vec_tile_shapes(32, kv_lora_rank)
        k_nope_split = pypto.reshape(k_nope, [tile_bs, 4, kv_lora_rank // 4])
        pypto.set_vec_tile_shapes(32, 4, kv_lora_rank // 4)
        k_nope_quant_res = k_nope_quant(k_nope_split)
        k_nope_quant_tensor = k_nope_quant_res[0]
        k_nope_scale = k_nope_quant_res[1]

        pypto.set_vec_tile_shapes(32, 4, kv_lora_rank // 4)
        k_nope_2d = pypto.reshape(k_nope_quant_tensor, [tile_bs, kv_lora_rank])
        k_scale_2d = pypto.reshape(k_nope_scale, [tile_bs, 4])

        k_rope_4d = pypto.reshape(k_rope_2d, [tile_bs, 1, 1, qk_rope_head_dim], inplace=True)
        k_nope_4d = pypto.reshape(k_nope_2d, [tile_bs, 1, 1, kv_lora_rank], inplace=True)
        k_scale_4d = pypto.reshape(k_scale_2d, [tile_bs, 1, 1, 4], inplace=True)
        index = pypto.view(k_cache_index_2d, [tile_bs, 1], [bs_offset, 0])
        pypto.set_semantic_label("ScatterUpdate_krCache")
        pypto.set_vec_tile_shapes(32, 1, 1, qk_rope_head_dim)
        mla_kr_cache_out[:] = pypto.scatter_update(mla_kr_cache, -2, index, k_rope_4d)
        pypto.set_semantic_label("ScatterUpdate_kvCache")
        pypto.set_vec_tile_shapes(32, 1, 1, kv_lora_rank)
        mla_kv_cache_out[:] = pypto.scatter_update(mla_kv_cache, -2, index, k_nope_4d)
        pypto.set_semantic_label("ScatterUpdate_kScaleCache")
        pypto.set_vec_tile_shapes(32, 1, 1, 4)
        mla_k_scale_cache_out[:] = pypto.scatter_update(mla_k_scale_cache, -2, index, k_scale_4d)

        q_linear = ip_configs.q_linear
        q_hd = ip_configs.q_hd

        pypto.set_semantic_label("Query-Linear")
        pypto.set_cube_tile_shapes([q_linear[L0M_INDEX], q_linear[L1M_INDEX]],
                                [q_linear[L0K_INDEX], q_linear[L1K_INDEX]],
                                [q_linear[L0N_INDEX], q_linear[L1N_INDEX]])
        q_s32 = pypto.matmul(q_norm, ip_w_qb_in, pypto.DT_INT32)  # (tile_bs, head_num * head_dim)

        pypto.set_semantic_label("Query-Dequant")

        pypto.set_vec_tile_shapes(ip_configs.t_sub_tile, head_num * head_dim // ip_configs.chunk_size)
        q_f32 = pypto.cast(q_s32, pypto.DT_FP32)  # (tile_bs, head_num * head_dim), fp32
        q_f32 = q_f32 * q_norm_scale  # (tile_bs, head_num * head_dim), fp32
        q_f32 = q_f32 * w_qb_scale  # (tile_bs, head_num * head_dim), fp32
        q_cast = pypto.cast(q_f32, dtype)

        q_bf16 = pypto.reshape(q_cast, [tile_bs, head_num, head_dim], valid_shape=[tile_bs, head_num, head_dim])
        # UB view
        q_rope = pypto.view(q_bf16, [tile_bs, head_num, qk_rope_head_dim], [0, 0, 0],
                            valid_shape=[tile_bs, head_num, qk_rope_head_dim])
        q_nope = pypto.view(q_bf16, [tile_bs, head_num, head_dim - qk_rope_head_dim], [0, 0, qk_rope_head_dim],
                            valid_shape=[tile_bs, head_num, head_dim - qk_rope_head_dim])

        q_roped = rope_3d(q_rope, cos_2d_view, sin_2d_view, ip_configs)  # [tile_bs, head_num, qk_rope_head_dim]
        pypto.set_vec_tile_shapes(ip_configs.t_sub_tile, head_num // ip_configs.chunk_size, head_dim)
        q_nope = pypto.cast(pypto.cast(q_nope, pypto.DT_FP32), q_bf16.dtype)
        q_cat = pypto.concat([q_roped, q_nope], -1)  # [tile_bs, head_num, head_dim]
        hadamard_q = pypto.reshape(ip_hadamard_q_in, [1, head_dim, head_dim], valid_shape=[1, head_dim, head_dim])

        pypto.set_semantic_label("Query-Hadamard")
        cur_max_unroll = 32
        q_hd_m_tile = cur_max_unroll if tile_bs < cur_max_unroll else q_hd[L0M_INDEX]
        pypto.set_cube_tile_shapes([q_hd_m_tile, q_hd_m_tile], [q_hd[L0K_INDEX], q_hd[L1K_INDEX]],
                                [q_hd[L0N_INDEX], q_hd[L1N_INDEX]])
        q_hadamard = pypto.matmul(q_cat, hadamard_q, dtype)  # (tile_bs, head_num, head_dim)

        pypto.set_semantic_label("Query-Quant")
        pypto.set_vec_tile_shapes(ip_configs.t_sub_tile, head_num // ip_configs.chunk_size, head_dim)
        q_res = prolog_quant(q_hadamard)
        q_scale = pypto.cast(q_res[1], pypto.DT_FP16)

        pypto.assemble(q_res[0], [bs_offset, 0, 0], ip_q_int8_out)
        pypto.assemble(q_scale, [bs_offset, 0, 0], ip_q_scale_out)

        # 获取key计算的各阶段Tile参数
        k_linear = ip_configs.k_linear
        pypto.set_semantic_label("Key-Linear")
        pypto.set_cube_tile_shapes([k_linear[L0M_INDEX], k_linear[L1M_INDEX]],
                                [k_linear[L0K_INDEX], k_linear[L1K_INDEX]],
                                [k_linear[L0N_INDEX], k_linear[L1N_INDEX]])
        k = pypto.matmul(x_view, ip_wk_in, pypto.DT_FP32)  # (tile_bs, head_dim)

        if tile_bs <= 32:
            pypto.set_vec_tile_shapes(min(tile_bs, VEC_TILE_4), head_dim)
        else:
            pypto.set_vec_tile_shapes(min(tile_bs, VEC_TILE_32), head_dim)
        k_bf16 = pypto.cast(quant_layer_norm(k, gamma_2d, beta_2d, -1, ip_attrs.eps), dtype)

        k_rope = pypto.view(k_bf16, [tile_bs, qk_rope_head_dim], [0, 0], valid_shape=[tile_bs, qk_rope_head_dim])
        k_nope = pypto.view(k_bf16, [tile_bs, head_dim - qk_rope_head_dim], [0, qk_rope_head_dim],
                            valid_shape=[tile_bs, head_dim - qk_rope_head_dim])
        k_roped = quant_rope_2d(k_rope, cos_2d_view, sin_2d_view)  # (tile_bs, qk_rope_head_dim)
        pypto.set_vec_tile_shapes(tile_bs, head_dim)
        k_nope = pypto.cast(pypto.cast(k_nope, pypto.DT_FP32), k_bf16.dtype)
        k_concat = pypto.concat([k_roped, k_nope], -1)
        pypto.set_semantic_label("Key-Hadamard")
        hadamard_k = pypto.matmul(k_concat, ip_hadamard_k_in, dtype)  # (tile_bs, head_dim), bf16
        pypto.set_semantic_label("Key-Quant")
        k_res = prolog_quant(hadamard_k)
        k_cache_4d = pypto.reshape(k_res[0], [tile_bs, 1, 1, head_dim], valid_shape=[tile_bs, 1, 1, head_dim])
        k_scale_4d = pypto.reshape(pypto.cast(k_res[1], pypto.DT_FP16), [tile_bs, 1, 1, 1],
                                valid_shape=[tile_bs, 1, 1, 1])


        pypto.set_vec_tile_shapes(tile_bs, 1, 1, head_dim)
        ip_k_int8_out.move(pypto.scatter_update(ip_k_cache, SCATTER_DIM, index, k_cache_4d))
        ip_k_scale_out.move(pypto.scatter_update(ip_k_cache_scale, SCATTER_DIM, index, k_scale_4d))

        pypto.set_semantic_label("Weight-Linear")
        w_linear = ip_configs.w_linear
        pypto.set_cube_tile_shapes([w_linear[L0M_INDEX], w_linear[L1M_INDEX]],
                                [w_linear[L0K_INDEX], w_linear[L1K_INDEX]],
                                [w_linear[L0N_INDEX], w_linear[L1N_INDEX]])
        pypto.set_vec_tile_shapes(tile_bs, head_num)
        weights = pypto.cast(pypto.matmul(x_view, ip_w_proj_in, dtype), pypto.DT_FP32)
        weights = pypto.mul(weights, 1.0 / (math.sqrt(head_num) * math.sqrt(head_dim)))
        weights_f16 = pypto.cast(weights, pypto.DT_FP16)
        pypto.assemble(weights_f16, [bs_offset, 0], ip_weights_out)


@pypto.frontend.jit(
    # prefill版本融合算子优化参数
    pass_options={
        "cube_l1_reuse_setting": {-1: 4},
    },
    runtime_options={"stitch_function_max_num": 128,
                    "device_sched_mode": 2},
)
def mla_indexer_prolog_quant_p(
    token_x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    mla_w_dq: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_NZ),
    mla_w_uq_qr: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT8, format=pypto.TileOpFormat.TILEOP_NZ),
    mla_dequant_scale: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    mla_w_uk: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    mla_w_dkv_kr: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_NZ),
    mla_gamma_cq: pypto.Tensor([pypto.STATIC], pypto.DT_BF16),
    mla_gamma_ckv: pypto.Tensor([pypto.STATIC], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    cache_index: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT64),
    mla_kv_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    mla_kr_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    mla_k_scale_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    ip_w_qb_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT8, format=pypto.TileOpFormat.TILEOP_NZ),
    ip_w_qb_scale_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    ip_wk_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_NZ),
    ip_w_proj_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_NZ),
    ip_ln_gamma_k_in: pypto.Tensor([pypto.STATIC], pypto.DT_BF16),
    ip_ln_beta_k_in: pypto.Tensor([pypto.STATIC], pypto.DT_BF16),
    ip_hadamard_q_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    ip_hadamard_k_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    ip_k_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    ip_k_cache_scale: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),

    mla_query_nope_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    mla_query_rope_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    mla_kv_cache_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    mla_kr_cache_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    mla_k_scale_cache_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    ip_q_int8_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    ip_q_scale_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    ip_k_int8_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    ip_k_scale_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    ip_weights_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP16),

    mla_epsilon_cq,
    mla_epsilon_ckv,
    mla_cache_mode,
    mla_tile_config,
    ip_attrs,
    ip_configs,
    rope_cfg
):

    """Fused MLA and Indexer Prolog quantization for prefill phase.

    Combines MLA Prolog and Lightning Indexer Prolog computations in a single
    fused operator for prefill phase. This enables pipeline parallelism and
    reduces memory transfers between operators.

    The computation flow:
    1. MLA Prolog: Computes MLA query, key, and value projections
    2. Indexer Prolog: Uses MLA's q_norm output to compute indexer query, key, and weights

    Args:
        token_x: Input token tensor, shape (t, h), dtype BF16
        mla_w_dq: MLA down-projection weight for query, NZ format
        mla_w_uq_qr: MLA up-projection weight for query and RoPE, NZ format
        mla_dequant_scale: MLA dequantization scale, FP32
        mla_w_uk: MLA up-projection weight for key, BF16
        mla_w_dkv_kr: MLA down-projection weight for key-value and RoPE, NZ format
        mla_gamma_cq: MLA RMSNorm scale for query, BF16
        mla_gamma_ckv: MLA RMSNorm scale for key-value, BF16
        cos: Cosine values for RoPE, BF16
        sin: Sine values for RoPE, BF16
        cache_index: Cache index for scatter update, INT64
        mla_kv_cache: MLA key-value cache input/output, INT8
        mla_kr_cache: MLA key RoPE cache input/output, BF16
        mla_k_scale_cache: MLA key scale cache input/output, FP16
        ip_w_qb_in: Indexer query projection weight, INT8, NZ format
        ip_w_qb_scale_in: Indexer query weight dequantization scale, FP32
        ip_wk_in: Indexer key projection weight, BF16, NZ format
        ip_w_proj_in: Indexer weight projection matrix, BF16, NZ format
        ip_ln_gamma_k_in: Indexer LayerNorm scale for key, BF16
        ip_ln_beta_k_in: Indexer LayerNorm shift for key, BF16
        ip_hadamard_q_in: Indexer Hadamard matrix for query, BF16
        ip_hadamard_k_in: Indexer Hadamard matrix for key, BF16
        ip_k_cache: Indexer key cache input/output, INT8
        ip_k_cache_scale: Indexer key cache scale input/output, FP16
        mla_query_nope_out: Output MLA query without RoPE, BF16
        mla_query_rope_out: Output MLA query with RoPE, BF16
        mla_kv_cache_out: Output MLA key-value cache
        mla_kr_cache_out: Output MLA key RoPE cache
        mla_k_scale_cache_out: Output MLA key scale cache
        ip_q_int8_out: Output indexer quantized query, INT8
        ip_q_scale_out: Output indexer query quantization scale, FP16
        ip_k_int8_out: Output indexer key cache
        ip_k_scale_out: Output indexer key cache scale
        ip_weights_out: Output indexer weights, FP16
        mla_epsilon_cq: MLA RMSNorm epsilon for query
        mla_epsilon_ckv: MLA RMSNorm epsilon for key-value
        mla_cache_mode: MLA cache mode
        mla_tile_config: MlaTileConfig object for MLA computation
        ip_attrs: IndexerPrologQuantAttr object for indexer computation
        ip_configs: IndexerPrologQuantip_configs object for indexer computation
        rope_cfg: RopeTileShapeConfig object for RoPE computation

    Note:
        The function creates intermediate tensors (mla_q_norm_out, mla_q_norm_scale_out)
        to pass data from MLA Prolog to Indexer Prolog. Pipeline parallelism is
        enabled through device_sched_mode=2.
    """
    t = token_x.shape[0]
    actual_q_lora_rank = ip_w_qb_in.shape[0]
    mla_q_norm_out = pypto.Tensor((t, actual_q_lora_rank), pypto.DT_INT8)
    mla_q_norm_scale_out = pypto.Tensor((t, 1), pypto.DT_FP32)
    mla_indexer_prolog_quant_compute(
        token_x, mla_w_dq, mla_w_uq_qr, mla_dequant_scale, mla_w_uk, mla_w_dkv_kr, mla_gamma_cq,
        mla_gamma_ckv, cos, sin, cache_index, mla_kv_cache, mla_kr_cache,
        mla_k_scale_cache, ip_w_qb_in, ip_w_qb_scale_in, ip_wk_in, ip_w_proj_in,
        ip_ln_gamma_k_in, ip_ln_beta_k_in, ip_hadamard_q_in, ip_hadamard_k_in,
        ip_k_cache, ip_k_cache_scale, mla_query_nope_out, mla_query_rope_out,
        mla_q_norm_out, mla_q_norm_scale_out,
        mla_kv_cache_out, mla_kr_cache_out,
        mla_k_scale_cache_out, ip_q_int8_out, ip_q_scale_out, ip_k_int8_out,
        ip_k_scale_out, ip_weights_out, mla_epsilon_cq, mla_epsilon_ckv,
        mla_cache_mode, mla_tile_config,
        ip_attrs, ip_configs, rope_cfg
    )


@pypto.frontend.jit(
    pass_options={
        "cube_l1_reuse_setting": {-1: 4, 0: 1, 1: 1, 2: 1},
        "cube_nbuffer_setting": {-1: 6, 0: 1, 1: 1, 2: 1},
    },
    runtime_options={"device_sched_mode": 2}
)
def mla_indexer_prolog_quant_d(
    token_x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    mla_w_dq: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_NZ),
    mla_w_uq_qr: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT8, format=pypto.TileOpFormat.TILEOP_NZ),
    mla_dequant_scale: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    mla_w_uk: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    mla_w_dkv_kr: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_NZ),
    mla_gamma_cq: pypto.Tensor([pypto.STATIC], pypto.DT_BF16),
    mla_gamma_ckv: pypto.Tensor([pypto.STATIC], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    cache_index: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT64),
    mla_kv_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    mla_kr_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    mla_k_scale_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    ip_w_qb_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT8, format=pypto.TileOpFormat.TILEOP_NZ),
    ip_w_qb_scale_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    ip_wk_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_NZ),
    ip_w_proj_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_NZ),
    ip_ln_gamma_k_in: pypto.Tensor([pypto.STATIC], pypto.DT_BF16),
    ip_ln_beta_k_in: pypto.Tensor([pypto.STATIC], pypto.DT_BF16),
    ip_hadamard_q_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    ip_hadamard_k_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    ip_k_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    ip_k_cache_scale: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),

    mla_query_nope_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    mla_query_rope_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    mla_kv_cache_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    mla_kr_cache_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    mla_k_scale_cache_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    ip_q_int8_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    ip_q_scale_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    ip_k_int8_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    ip_k_scale_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    ip_weights_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP16),

    mla_epsilon_cq,
    mla_epsilon_ckv,
    mla_cache_mode,
    mla_tile_config,
    ip_attrs,
    ip_configs,
    rope_cfg
):

    """Fused MLA and Indexer Prolog quantization for decode phase.

    Combines MLA Prolog and Lightning Indexer Prolog computations in a single
    fused operator for decode phase. This enables pipeline parallelism and
    reduces memory transfers between operators.

    The computation flow:
    1. MLA Prolog: Computes MLA query, key, and value projections
    2. Indexer Prolog: Uses MLA's q_norm output to compute indexer query, key, and weights

    Args:
        token_x: Input token tensor, shape (t, h), dtype BF16
        mla_w_dq: MLA down-projection weight for query, NZ format
        mla_w_uq_qr: MLA up-projection weight for query and RoPE, NZ format
        mla_dequant_scale: MLA dequantization scale, FP32
        mla_w_uk: MLA up-projection weight for key, BF16
        mla_w_dkv_kr: MLA down-projection weight for key-value and RoPE, NZ format
        mla_gamma_cq: MLA RMSNorm scale for query, BF16
        mla_gamma_ckv: MLA RMSNorm scale for key-value, BF16
        cos: Cosine values for RoPE, BF16
        sin: Sine values for RoPE, BF16
        cache_index: Cache index for scatter update, INT64
        mla_kv_cache: MLA key-value cache input/output, INT8
        mla_kr_cache: MLA key RoPE cache input/output, BF16
        mla_k_scale_cache: MLA key scale cache input/output, FP16
        ip_w_qb_in: Indexer query projection weight, INT8, NZ format
        ip_w_qb_scale_in: Indexer query weight dequantization scale, FP32
        ip_wk_in: Indexer key projection weight, BF16, NZ format
        ip_w_proj_in: Indexer weight projection matrix, BF16, NZ format
        ip_ln_gamma_k_in: Indexer LayerNorm scale for key, BF16
        ip_ln_beta_k_in: Indexer LayerNorm shift for key, BF16
        ip_hadamard_q_in: Indexer Hadamard matrix for query, BF16
        ip_hadamard_k_in: Indexer Hadamard matrix for key, BF16
        ip_k_cache: Indexer key cache input/output, INT8
        ip_k_cache_scale: Indexer key cache scale input/output, FP16
        mla_query_nope_out: Output MLA query without RoPE, BF16
        mla_query_rope_out: Output MLA query with RoPE, BF16
        mla_kv_cache_out: Output MLA key-value cache
        mla_kr_cache_out: Output MLA key RoPE cache
        mla_k_scale_cache_out: Output MLA key scale cache
        ip_q_int8_out: Output indexer quantized query, INT8
        ip_q_scale_out: Output indexer query quantization scale, FP16
        ip_k_int8_out: Output indexer key cache
        ip_k_scale_out: Output indexer key cache scale
        ip_weights_out: Output indexer weights, FP16
        mla_epsilon_cq: MLA RMSNorm epsilon for query
        mla_epsilon_ckv: MLA RMSNorm epsilon for key-value
        mla_cache_mode: MLA cache mode
        mla_tile_config: MlaTileConfig object for MLA computation
        ip_attrs: IndexerPrologQuantAttr object for indexer computation
        ip_configs: IndexerPrologQuantip_configs object for indexer computation
        rope_cfg: RopeTileShapeConfig object for RoPE computation

    Note:
        The function creates intermediate tensors (mla_q_norm_out, mla_q_norm_scale_out)
        to pass data from MLA Prolog to Indexer Prolog. Pipeline parallelism is
        enabled through device_sched_mode=2.
    """
    t = token_x.shape[0]
    actual_q_lora_rank = ip_w_qb_in.shape[0]
    mla_q_norm_out = pypto.Tensor((t, actual_q_lora_rank), pypto.DT_INT8)
    mla_q_norm_scale_out = pypto.Tensor((t, 1), pypto.DT_FP32)
    mla_indexer_prolog_quant_compute(
        token_x, mla_w_dq, mla_w_uq_qr, mla_dequant_scale, mla_w_uk, mla_w_dkv_kr, mla_gamma_cq,
        mla_gamma_ckv, cos, sin, cache_index, mla_kv_cache, mla_kr_cache,
        mla_k_scale_cache, ip_w_qb_in, ip_w_qb_scale_in, ip_wk_in, ip_w_proj_in,
        ip_ln_gamma_k_in, ip_ln_beta_k_in, ip_hadamard_q_in, ip_hadamard_k_in,
        ip_k_cache, ip_k_cache_scale, mla_query_nope_out, mla_query_rope_out,
        mla_q_norm_out, mla_q_norm_scale_out,
        mla_kv_cache_out, mla_kr_cache_out,
        mla_k_scale_cache_out, ip_q_int8_out, ip_q_scale_out, ip_k_int8_out,
        ip_k_scale_out, ip_weights_out, mla_epsilon_cq, mla_epsilon_ckv,
        mla_cache_mode, mla_tile_config,
        ip_attrs, ip_configs, rope_cfg
    )
