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
    - pfa_func: JIT compiled kernel implementing Flash Attention with paged KV cache
    - gen_block_table: Generate block mapping table for Attention
    - kv_cache_concat_bsnd: Convert paged KV cache to BSND format
"""
import os
from dataclasses import dataclass
from typing import Tuple
import torch
import pypto


@dataclass
class PfaTileShapeConfig:
    g_tile: int
    s2_tile: int
    c1_tile_shape: list
    v1_tile_shape: list
    c2_tile_shape: list
    v2_tile_shape: list


@dataclass
class PfaConfig:
    b: int
    s1: int
    s2: int
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
    s2_tile: int = 1024


def create_config(b, s1, s2, nq, nkv, qd, block_size):
    m_tile = 128
    cube_tile = 128
    s2_tile = 128
    v2_tile = 256
    return {
        "b": b, "s1": s1, "s2": s2, "nq": nq, "nkv": nkv, "qd": qd, "block_size": block_size,
        "tile_config": PfaTileShapeConfig(
            g_tile=nq // nkv,  # 动态计算
            s2_tile=1024 if s2 == 8192 else s2_tile,
            c1_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
            v1_tile_shape=[m_tile, s2_tile],
            c2_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
            v2_tile_shape=[m_tile, v2_tile],
        ),
    }


def get_case_config(case_name: str):

    test_case_config = {
        "pfa_fp8_b1_s1_2_s2_2048_nkv_8": create_config(
            b=1, s1=2, s2=2048, nq=48, nkv=8, qd=128, 
            block_size=128
        ),
        "pfa_fp8_b16_s1_1_s2_8195_nkv_2": create_config(
            b=16, s1=1, s2=8195, nq=12, nkv=2, qd=128, 
            block_size=128
        ),
        "pfa_fp8_b16_s1_1_s2_8k_nkv_1": create_config(
            b=16, s1=1, s2=8192, nq=12, nkv=1, qd=128, 
            block_size=128
        ),
        "pfa_fp8_b16_s1_1_s2_8k_nkv_2": create_config(
            b=16, s1=1, s2=8192, nq=12, nkv=2, qd=128, 
            block_size=128
        ),
        "pfa_fp8_b2_s1_1_s2_1k": create_config(
            b=2, s1=1, s2=1024, nq=12, nkv=1, qd=128, 
            block_size=128
        ),
        "pfa_fp8_b16_s1_1_s2_256_nkv_4": create_config(
            b=16, s1=1, s2=256, nq=16, nkv=4, qd=128, 
            block_size=128
        ),
    }
    return test_case_config.get(case_name)


def build_pfa_config(case_config):
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    device = f'npu:{device_id}'

    b = case_config["b"]
    s1 = case_config["s1"]
    s2 = case_config["s2"]
    nq = case_config["nq"]
    nkv = case_config["nkv"]
    qd = case_config["qd"]
    block_size = case_config["block_size"]
    kv_layout = "PA_BSND"
    softmax_scale = qd ** -0.5
    block_table_batch = b
    s2_tile = case_config["tile_config"].s2_tile
    kv_num_blocks = b * ((s2 + block_size - 1) // block_size)

    actual_seq_values = [s2] * b
    actual_seq_tensor = torch.tensor(actual_seq_values, dtype=torch.int32, device=device)

    atten_cfg = PfaConfig(
        b=b, s1=s1, s2=s2, nq=nq, nkv=nkv, qd=qd, kvd=qd,
        block_size=block_size, softmax_scale=softmax_scale, kv_layout=kv_layout,
        block_table_batch=block_table_batch, kv_num_blocks=kv_num_blocks,
        actual_seq=actual_seq_tensor, s2_tile=s2_tile
    )
    atten_cfg.max_num_blocks_per_query = (s2 + block_size - 1) // block_size

    return atten_cfg, case_config["tile_config"]


def dequant_dynamic(in_tensor, scale_1, scale_2):
    """
    Perform dynamic dequantization using two scale factors.

    Args:
        in_tensor: Quantized input tensor
        scale_1: First scale factor
        scale_2: Second scale factor

    Returns:
        Dequantized tensor
    """
    in_tensor_fp32 = pypto.cast(in_tensor, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    scale_1_fp32 = pypto.cast(scale_1, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    scale_2_fp32 = pypto.cast(scale_2, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    out_scale_2 = pypto.mul(in_tensor_fp32, scale_2_fp32)
    out = pypto.mul(out_scale_2, scale_1_fp32)
    return out


def symmetric_quantization_per_token_fp8_e4m3(input_tensor) -> Tuple:
    """
    Perform symmetric quantization per token (per row).

    Args:
        input_tensor: Input tensor to quantize

    Returns:
        Tuple of (quantized_f8_e4m3_tensor, dequantization_scale)
    """
    fp8_e4m3_max_value = 448.0
    x_fp32 = pypto.cast(input_tensor, pypto.DT_FP32)
    x_abs = pypto.abs(x_fp32)
    x_max = pypto.amax(x_abs, -1, True)
    shape_0, shape_1 = x_max.shape[:2]
    x_scale = pypto.div(pypto.full([shape_0, shape_1], fp8_e4m3_max_value, pypto.DT_FP32),
                        x_max, precision_type=pypto.PrecisionType.INTRINSIC)
    x_mul = pypto.mul(x_fp32, x_scale)
    x_fp8_e4m3 = pypto.cast(x_mul, pypto.DT_FP8E4M3)
    x_scale_quant = pypto.div(pypto.full([shape_0, shape_1], 1.0, pypto.DT_FP32),
                              x_scale, precision_type=pypto.PrecisionType.INTRINSIC)
    return x_fp8_e4m3, x_scale_quant


@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 512, "device_sched_mode": 0},
    # 当子图大小达到上界不允许与其他子图合并
    pass_options={
        # Q常驻，0代表第一组mmad，4代表4次matmul合并
        "cube_l1_reuse_setting": {-1: 8},
        "vec_nbuffer_setting": {-1: 4},
        "cube_nbuffer_setting": {-1: 4}
    },
    verify_options={
        "enable_pass_verify": False,
        "pass_verify_save_tensor": False,
    },
)
def pfa_func_kernel_v2_bound(
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP8E4M3),
    q_scale: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP8E4M3),
    k_scale: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP8E4M3),
    v_scale: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    block_table: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_INT32),
    kv_act_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    atten_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    softmax_scale, tile_config
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

    dtype = atten_out.dtype
    group = nq // nkv
    n2_sym = nkv

    g_tile = tile_config.g_tile
    s2_tile = tile_config.s2_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape

    # 3. 得到动态tensor的shape
    s1_scalar = bs_scalar // b_scalar
    g_loop = group // g_tile

    k_2d_shape = (block_num_scalar * block_size, n2_sym * dn)
    q_2d_shape = (b_scalar * s1_scalar * nq, dn)
    q_scale_2d_shape = (b_scalar * s1_scalar * nq, 1)
    k_scale_2d_shape = (block_num_scalar * block_size, n2_sym * 1)
    v_scale_2d_shape = (b_scalar * 1, n2_sym * dn)

    k_2d = pypto.reshape(k, k_2d_shape, inplace=True)
    k_scale_2d = pypto.reshape(k_scale, k_scale_2d_shape, inplace=True)
    v_2d = pypto.reshape(v, k_2d_shape, inplace=True)
    v_scale_2d = pypto.reshape(v_scale, v_scale_2d_shape, inplace=True)
    q_2d = pypto.reshape(q, q_2d_shape, inplace=True)
    q_scale_2d = pypto.reshape(q_scale, q_scale_2d_shape, inplace=True)

    # 4. 实现kernel逻辑，循环展开B动态轴
    for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"):
        for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx"):
            cur_seq = (kv_act_seqs[b_idx] - (s1_scalar - 1 - s1_idx)).max(0)
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
                        # 5. 按照计算图实现运算逻辑，设置set_vec_tile_shapes时应尽可能用满UB，但不要超过UB的大小。
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        qi = pypto.view(q_2d, [g_tile, dn], [bs_ofs * nq + n1g_ofs, 0])
                        qi_scale = pypto.view(q_scale_2d, [g_tile, 1], [bs_ofs * nq + n1g_ofs, 0])

                        kj_assemble = pypto.tensor([s2_tile, dn], k_2d.dtype, "kj_assemble")
                        kj_sclae_assemble = pypto.tensor([s2_tile, 1], k_scale_2d.dtype, "kj_sclae_assemble")
                        for i in range(block_num):
                            block_idx = block_table[b_idx, idx + i]
                            block_idx_valid = block_idx.max(0)
                            kj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                pypto.view(k_2d, [block_size, dn], [block_idx_valid * block_size, n2_idx * dn])
                            kj_sclae_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                pypto.view(k_scale_2d, [block_size, 1], [block_idx_valid * block_size, n2_idx * 1])

                        kj_assemble = pypto.view(kj_assemble, [s2_tile, dn], [0, 0], valid_shape=[actual_s2_tile, dn])
                        kj_sclae_assemble = pypto.view(kj_sclae_assemble, [s2_tile, 1], [0, 0], 
                                            valid_shape=[actual_s2_tile, 1])

                        # c1
                        # 6. 下面是flash attention的计算逻辑
                        pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
                        sij_quant = pypto.matmul(qi, kj_assemble, pypto.DT_FP32, a_trans=False, b_trans=True)
                        # dequant
                        pypto.set_pass_options(sg_set_scope=3)
                        kj_sclae_assemble_t = pypto.transpose(kj_sclae_assemble, 0, 1)
                        sij = dequant_dynamic(sij_quant, qi_scale, kj_sclae_assemble_t)
                        sij = pypto.view(sij, [g_tile, s2_tile], [0, 0],
                                            valid_shape=[g_tile, actual_s2_tile])
                        # v1
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        sij_scale = pypto.mul(sij, softmax_scale)
                        tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)

                        tsub = pypto.sub(sij_scale, tilda_mij)
                        vec1_res = pypto.exp(tsub)
                        sum_local = pypto.sum(vec1_res, dim=-1, keepdim=True)

                        #  quant
                        tilda_pij_fp8_e4m3, tilda_pij_scale = symmetric_quantization_per_token_fp8_e4m3(vec1_res)
                        pypto.set_pass_options(sg_set_scope=-1)
                        # c2
                        vj_assemble = pypto.tensor([s2_tile, dn], v_2d.dtype, "vj_assemble_2")

                        for i in range(block_num):
                            block_idx = block_table[b_idx, idx + i]
                            block_idx_valid = block_idx.max(0)
                            vj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                pypto.view(v_2d, [block_size, dn], [block_idx_valid * block_size, n2_idx * dn])

                        vj_assemble = pypto.view(vj_assemble, [s2_tile, dn],
                                                    [0, 0], valid_shape=[actual_s2_tile, dn])

                        pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                        oi_tmp_quant = pypto.matmul(tilda_pij_fp8_e4m3, vj_assemble, pypto.DT_FP32)
                        # dequant
                        pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                        vj_scale_assemble = pypto.view(v_scale_2d, [1, dn], [b_idx, n2_idx * dn])
                        mm2_res = dequant_dynamic(oi_tmp_quant, tilda_pij_scale, vj_scale_assemble)

                        # # v2
                        if pypto.is_loop_begin(s2_idx):
                            pypto.set_pass_options(sg_set_scope=2)
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_tmp = mm2_res
                            oi_update[:] = pypto.tensor(oi_tmp.shape, pypto.DT_FP32, "oi_update")
                            if pypto.is_loop_end(s2_idx):
                                oi_update[:] = pypto.div(
    oi_tmp, sum_local, precision_type=pypto.PrecisionType.INTRINSIC)
                                oi_update_3d_tmp = pypto.reshape(oi_update, [1, g_tile, dn])
                                pypto.set_vec_tile_shapes(1, v2_tile[0], v2_tile[1])
                                oi_update_3d = pypto.cast(oi_update_3d_tmp, dtype)
                                # 10. 将结果搬运到输出tensor上
                                pypto.assemble(oi_update_3d, oi_ofs, atten_out)
                            else:
                                oi_update[:] = oi_tmp
                                sum_update[:] = sum_local
                                max_update[:] = tilda_mij
                            pypto.set_pass_options(sg_set_scope=-1)
                        else:
                            pypto.set_pass_options(sg_set_scope=1)
                            pypto.set_vec_tile_shapes(v2_tile[0], 128)
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

                            pypto.set_vec_tile_shapes(v2_tile[0], 128)
                            oi_last = pypto.mul(oi_update, t2)
                            oi_flash = pypto.mul(mm2_res, t4)
                            oi_tmp = pypto.add(oi_last, oi_flash)
                            if pypto.is_loop_end(s2_idx):
                                oi_update_tmp = pypto.div(oi_tmp, sum_update,
                                                          precision_type=pypto.PrecisionType.INTRINSIC)
                                oi_update_tmp_3d = pypto.reshape(oi_update_tmp, [1, g_tile, dn])
                                pypto.set_vec_tile_shapes(1, v2_tile[0], v2_tile[1])
                                oi_update_3d = pypto.cast(oi_update_tmp_3d, dtype)
                                # 11. 将结果搬运到输出tensor上
                                pypto.assemble(oi_update_3d, oi_ofs, atten_out)
                            else:
                                oi_update[:] = oi_tmp
                            pypto.set_pass_options(sg_set_scope=-1)
