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
Test for SFA Forward TND v2

v2 interface:
    - 3D input tensors
    - S = Q_nope @ KV^T + Q_pe @ K_pe^T
    - softmax_max/sum: [N2, T1, group]
    - outer B loop, inner s loop (B from npu_actual_q_len shape)
    - npu_actual_q_len is prefix-sum, npu_actual_kv_len is per-batch
"""
import os
import math
import logging
from dataclasses import dataclass
import torch
import torch_npu

import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

import numpy as np
import pytest

from experimental.ops_transformer.sparse_attention_tnd.sfa_forward_tnd_impl import sfa_forward_tnd, SaTileShapeConfig

from common_utils import compare


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=dtype)
    if torch.is_floating_point(torch.tensor(0, dtype=dtype)):
        return min_value + (max_value - min_value) * torch.rand(data_shape, dtype=dtype)
    else:
        return torch.randint(low=min_value, high=max_value, size=data_shape, dtype=dtype)


def compute_attention_tnd_golden(q_nope, compressed_kv_norm, value, topk_indices,
                                q_pe, k_pe, scale, npu_actual_q_len_list,
                                npu_actual_kv_len_list, sparse_size):
    t1, n1, d = q_nope.shape
    d_rope = q_pe.shape[-1]
    t2, n2, _ = compressed_kv_norm.shape
    group = n1 // n2
    input_dtype = q_nope.dtype

    npu_actual_q_len_list = torch.diff(npu_actual_q_len_list, prepend=torch.tensor([0]))
    npu_actual_kv_len_list = torch.diff(npu_actual_kv_len_list, prepend=torch.tensor([0]))

    core_attn_out = torch.zeros([t1, n1, d], dtype=input_dtype)
    softmax_max_out = torch.zeros([n2, t1, group], dtype=torch.float32)
    softmax_sum_out = torch.zeros([n2, t1, group], dtype=torch.float32)

    # Build token-index → batch-index mapping
    if npu_actual_q_len_list is not None and npu_actual_kv_len_list is not None:
        token_to_batch = []
        for b_i, q_len in enumerate(npu_actual_q_len_list):
            token_to_batch.extend([b_i] * q_len)
    else:
        token_to_batch = None

    for t_idx in range(t1):
        # Determine effective topk width for this token's batch
        if token_to_batch is not None:
            b_i = token_to_batch[t_idx]
            kv_len = npu_actual_kv_len_list[b_i]

            eff_topk = min(max((kv_len - (npu_actual_q_len_list[0: b_i + 1].sum() - t_idx) + 1), 0), sparse_size)
        else:
            eff_topk = sparse_size
        
        for kv_head_idx in range(n2):
            # topk indices for this (token, kv_head) - all sparse_size used
            cur_topk = topk_indices[t_idx, kv_head_idx, :eff_topk]  # (sparse_size,)

            # Gather KV nope: compressed_kv_norm[cur_topk, kv_head_idx, :]
            kv_sel = compressed_kv_norm[cur_topk, kv_head_idx, :]  # (sparse_size, D)

            # Gather K_pe: k_pe[cur_topk, kv_head_idx, :]
            k_pe_sel = k_pe[cur_topk, kv_head_idx, :]  # (sparse_size, d_rope)

            # Q slices for group query heads
            q_head_start = kv_head_idx * group
            q_head_end = q_head_start + group
            qi_nope = q_nope[t_idx, q_head_start:q_head_end, :]  # (group, D)
            qi_pe = q_pe[t_idx, q_head_start:q_head_end, :]      # (group, d_rope)

            # Q_nope @ KV^T + Q_pe @ K_pe^T
            s_nope = torch.matmul(qi_nope.to(torch.float32), kv_sel.transpose(0, 1).to(torch.float32))
            s_rope = torch.matmul(qi_pe.to(torch.float32), k_pe_sel.transpose(0, 1).to(torch.float32))
            sij = s_nope + s_rope

            # Softmax
            sij_scale = sij * scale
            tilda_mij = sij_scale.amax(dim=-1, keepdims=True)
            t_sub = sij_scale - tilda_mij
            tilda_pij = torch.exp(t_sub)
            tilda_lij = tilda_pij.sum(dim=-1, keepdims=True)
            tmp_softmax = (tilda_pij / tilda_lij).to(input_dtype)

            # Store softmax stats
            softmax_max_out[kv_head_idx, t_idx, :] = tilda_mij.squeeze(-1).to(torch.float32)
            softmax_sum_out[kv_head_idx, t_idx, :] = tilda_lij.squeeze(-1).to(torch.float32)

            # Softmax @ V
            vj = kv_sel
            atten_part = torch.matmul(tmp_softmax.to(torch.float32), vj.to(torch.float32))
            core_attn_out[t_idx, q_head_start:q_head_end, :] = atten_part.to(input_dtype)

    return core_attn_out, softmax_max_out, softmax_sum_out


def gen_topk_indices(b, n_kv, npu_actual_q_len_list, npu_actual_kv_len_list, sparse_size, t1):
    topk_indices = torch.zeros(t1, n_kv, sparse_size, dtype=torch.int32)

    q_offsets = [0]
    kv_offsets = [0]
    for i in range(b):
        q_offsets.append(q_offsets[-1] + npu_actual_q_len_list[i])
        kv_offsets.append(kv_offsets[-1] + npu_actual_kv_len_list[i])
    import torch.nn.functional as F
    for b_i in range(b):
        kv_start = kv_offsets[b_i]
        kv_len = npu_actual_kv_len_list[b_i]
        s1 = npu_actual_q_len_list[b_i]

        for s_q_i in range(s1):
            t_idx = q_offsets[b_i] + s_q_i
            kv_valid = kv_len - s1 + s_q_i + 1
            for kv_h in range(n_kv):
                # Always fill all sparse_size positions
                perm = torch.randperm(kv_valid)
                pad_size = sparse_size - kv_valid
                if pad_size > 0:
                    perm = F.pad(perm, (0, pad_size), value=-1)
                topk_indices[t_idx, kv_h, :] = (perm[:sparse_size] + kv_start).to(torch.int32)
                
    return topk_indices


def gen_sfa_tnd_golden(dtype, b, nq, n_kv, npu_actual_q_len_list, npu_actual_kv_len_list,
                       sparse_size=2048):
    """Generate test data and golden output for SFA TND v2."""
    torch.manual_seed(42)

    kv_lora_rank = 512
    qk_rope_dim = 64
    scale = 0.07216878364870322
    group = nq // n_kv

    t1 = sum(npu_actual_q_len_list)
    t2 = sum(npu_actual_kv_len_list)

    # Generate 3D inputs
    q_nope = gen_uniform_data([t1, nq, kv_lora_rank], -1, 1, dtype)
    q_pe = gen_uniform_data([t1, nq, qk_rope_dim], -1, 1, dtype)
    compressed_kv_norm = gen_uniform_data([t2, n_kv, kv_lora_rank], -1, 1, dtype)
    k_pe = gen_uniform_data([t2, n_kv, qk_rope_dim], -1, 1, dtype)

    # Generate topk_indices: [T1, N2, sparse_size]
    # All sparse_size positions filled with valid indices
    topk_indices = gen_topk_indices(b, n_kv, npu_actual_q_len_list, npu_actual_kv_len_list, sparse_size, t1)
    # npu_actual_q_len is prefix-sum: [q0, q0+q1, q0+q1+q2, ...]
    q_prefix_sum = []
    running_sum = 0
    for ql in npu_actual_q_len_list:
        running_sum += ql
        q_prefix_sum.append(running_sum)
    npu_actual_q_len = torch.tensor(q_prefix_sum, dtype=torch.int32)
    # npu_actual_kv_len is also prefix-sum: [kv0, kv0+kv1, kv0+kv1+kv2, ...]
    kv_prefix_sum = []
    running_sum = 0
    for kvl in npu_actual_kv_len_list:
        running_sum += kvl
        kv_prefix_sum.append(running_sum)
    npu_actual_kv_len = torch.tensor(kv_prefix_sum, dtype=torch.int32)

    # Compute golden
    core_attn_golden, sm_max_golden, sm_sum_golden = compute_attention_tnd_golden(
        q_nope, compressed_kv_norm, compressed_kv_norm, topk_indices,
        q_pe, k_pe, scale, npu_actual_q_len, npu_actual_kv_len, sparse_size
    )
    
    input_params = {
        'b': b, 'nq': nq, 'n_kv': n_kv,
        'kv_lora_rank': kv_lora_rank, 'qk_rope_dim': qk_rope_dim,
        'sparse_size': sparse_size, 'scale': scale,
        'T1': t1, 'T2': t2, 'group': group,
    }
    input_tensors = {
        'q_nope': q_nope,
        'compressed_kv_norm': compressed_kv_norm,
        'topk_indices': topk_indices,
        'q_pe': q_pe,
        'k_pe': k_pe,
        'npu_actual_q_len': npu_actual_q_len,
        'npu_actual_kv_len': npu_actual_kv_len,
    }

    return input_params, input_tensors, core_attn_golden, sm_max_golden, sm_sum_golden


def _run_sfa_kernel_on_npu(input_params, input_tensors):
    """Move inputs to NPU, run SFA kernel, return outputs."""
    p = input_params
    t = input_tensors
    tile_config = SaTileShapeConfig(
        s_kv_tile=2048,
        c1_tile_shape=[128, 128, 128, 256, 256, 256],
        v1_tile_shape=[8, 2048],
        c2_tile_shape=[128, 128, 128, 256, 128, 128],
        v2_tile_shape=[64, 128]
    )
    q_npe_npu = t['q_nope'].npu()
    compressed_kv_norm_npu = t['compressed_kv_norm'].npu()
    topk_indices_npu = t['topk_indices'].npu()
    q_pe_npu = t['q_pe'].npu()
    k_pe_npu = t['k_pe'].npu()
    npu_actual_q_len_npu = t['npu_actual_q_len'].npu()
    npu_actual_kv_len_npu = t['npu_actual_kv_len'].npu()

    t1, nq, n_kv = p['T1'], p['nq'], p['n_kv']
    kv_lora_rank, group = p['kv_lora_rank'], p['group']
    core_attn_out = torch.empty([t1, nq, kv_lora_rank], dtype=t['q_nope'].dtype).npu()
    softmax_max_out = torch.empty([n_kv, t1, group], dtype=torch.float32).npu()
    softmax_sum_out = torch.empty([n_kv, t1, group], dtype=torch.float32).npu()

    pto_inputs = [
        q_npe_npu, compressed_kv_norm_npu, topk_indices_npu,
        q_pe_npu, k_pe_npu,
        npu_actual_q_len_npu, npu_actual_kv_len_npu,
        core_attn_out, softmax_max_out, softmax_sum_out
    ]
    logging.info("Running SFA Forward TND v2 kernel on NPU...")
    sfa_forward_tnd(*pto_inputs,
        nq=p['nq'], n_kv=p['n_kv'], scale=p['scale'],
        sparse_size=p['sparse_size'], tile_config=tile_config)
    torch_npu.npu.synchronize()
    return core_attn_out, softmax_max_out, softmax_sum_out


def _compare_sfa_vs_golden(core_attn_out, softmax_max_out, softmax_sum_out,
                           core_attn_golden, sm_max_golden, sm_sum_golden):
    """Compare NPU outputs against golden reference."""
    compare(core_attn_out.cpu(), core_attn_golden, "core_attn_out",
            atol=0.0001, rtol=0.005, max_error_count=100)
    logging.info("core_attn_out comparison PASSED!")
    compare(softmax_max_out.cpu(), sm_max_golden, "softmax_max",
            atol=0.001, rtol=0.01, max_error_count=100)
    logging.info("softmax_max comparison PASSED!")
    compare(softmax_sum_out.cpu(), sm_sum_golden, "softmax_sum",
            atol=0.001, rtol=0.01, max_error_count=100)
    logging.info("softmax_sum comparison PASSED!")
    logging.info("All comparisons PASSED!")


def do_test_sfa_tnd(input_params, input_tensors, core_attn_golden,
                    sm_max_golden, sm_sum_golden):
    """Run SFA TND v2 kernel on NPU and compare with golden."""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    logging.info(f"device_id:{device_id}")

    core_attn_out, softmax_max_out, softmax_sum_out = \
        _run_sfa_kernel_on_npu(input_params, input_tensors)
    _compare_sfa_vs_golden(core_attn_out, softmax_max_out, softmax_sum_out,
                           core_attn_golden, sm_max_golden, sm_sum_golden)


def get_case_config(case_name: str):
    test_case_config = {
        "sfa_tnd_v2_bf16_b1_s2_seq4K": (
            1, 128, 1, [2], [4096], 2048
        ),
        "sfa_tnd_v2_bf16_b2_s3_seq4K": (
            2, 48, 1, [2, 1], [4096, 4096], 2048
        ),
        "sfa_tnd_v2_bf16_b1_s2_seq64K": (
            1, 128, 1, [2], [65536], 2048
        ),
        "sfa_tnd_v2_bf16_b4_s4_seq128K": (
            4, 48, 1,
            [4, 4, 4, 3],
            [128 * 1024, 128 * 1024, 128 * 1024, 128 * 1024],
            2048
        ),
        "sfa_tnd_v2_bf16_b1_seq_4k": (
            1, 2, 1,
            [512],
            [4096],
            2048
        ),
        "sfa_tnd_v2_bf16_b1_s1k_seq_32k": (
            1, 2, 1,
            [1024],
            [32768],
            2048
        ),
        "sfa_tnd_v2_bf16_b1_s256_seq_32k": (
            1, 64, 1,
            [256],
            [32768],
            2048
        ),
        "sfa_tnd_v2_bf16_b1_s128_seq_32k": (
            1, 32, 1,
            [128],
            [32768],
            2048
        )
        
    }
    return test_case_config.get(case_name)


def do_test_sfa_tnd_entry(case_name: str):
    case_config = get_case_config(case_name)
    if not case_config:
        logging.error("Can't find test case config for Case(%s)", case_name)
        return False

    b, nq, n_kv, npu_actual_q_len_list, npu_actual_kv_len_list, sparse_size = case_config

    input_params, input_tensors, core_attn_golden, sm_max_golden, sm_sum_golden = \
        gen_sfa_tnd_golden(torch.bfloat16, b, nq, n_kv,
                           npu_actual_q_len_list, npu_actual_kv_len_list,
                           sparse_size=sparse_size)

    do_test_sfa_tnd(input_params, input_tensors, core_attn_golden,
                    sm_max_golden, sm_sum_golden)
    return True


def get_data(case_name: str):
    case_config = get_case_config(case_name)
    if not case_config:
        logging.error("Can't find test case config for Case(%s)", case_name)
        return None, None, None, None, None

    b, nq, n_kv, npu_actual_q_len_list, npu_actual_kv_len_list, sparse_size = case_config

    input_params, input_tensors, core_attn_golden, sm_max_golden, sm_sum_golden = \
        gen_sfa_tnd_golden(torch.bfloat16, b, nq, n_kv,
                           npu_actual_q_len_list, npu_actual_kv_len_list,
                           sparse_size=sparse_size)

    return input_params, input_tensors, core_attn_golden, sm_max_golden, sm_sum_golden


@pytest.mark.skip(reason="large test case")
def test_sfa_tnd_v2_bf16_b1_s2_seq4k():
    do_test_sfa_tnd_entry("sfa_tnd_v2_bf16_b1_s2_seq4K")


@pytest.mark.skip(reason="large test case")
def test_sfa_tnd_v2_bf16_b2_s3_seq4k():
    do_test_sfa_tnd_entry("sfa_tnd_v2_bf16_b2_s3_seq4K")


@pytest.mark.skip(reason="large test case")
def test_sfa_tnd_v2_bf16_b1_s2_seq64k():
    do_test_sfa_tnd_entry("sfa_tnd_v2_bf16_b1_s2_seq64K")


@pytest.mark.skip(reason="large test case")
def test_sfa_tnd_v2_bf16_b4_s4_seq128k():
    do_test_sfa_tnd_entry("sfa_tnd_v2_bf16_b4_s4_seq128K")


@pytest.mark.skip(reason="large test case")
def test_sfa_tnd_v2_bf16_b1_seq_4k():
    do_test_sfa_tnd_entry("sfa_tnd_v2_bf16_b1_seq_4k")
    

@pytest.mark.skip(reason="large test case")   
def test_sfa_tnd_v2_bf16_b1_s1k_seq_32k():
    do_test_sfa_tnd_entry("sfa_tnd_v2_bf16_b1_s1k_seq_32k")
    

@pytest.mark.skip(reason="large test case")    
def test_sfa_tnd_v2_bf16_b1_s256_seq_32k():
    do_test_sfa_tnd_entry("sfa_tnd_v2_bf16_b1_s256_seq_32k")


def test_sfa_tnd_v2_bf16_b1_s128_seq_32k():
    do_test_sfa_tnd_entry("sfa_tnd_v2_bf16_b1_s128_seq_32k")


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO
    )
    test_sfa_tnd_v2_bf16_b1_s1k_seq_32k()

