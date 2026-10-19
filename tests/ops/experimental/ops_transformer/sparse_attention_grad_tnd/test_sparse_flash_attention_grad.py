#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# ... (license unchanged)
# -----------------------------------------------------------------------------------------------------------
"""
Sparse Flash Attention Grad - Test & Golden (TND format, nope/rope split)
"""
from dataclasses import dataclass
import os
import math
import random
import logging
from typing import Any
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

from common_utils import compare
import pytest


def generate_random_sequence(length, total_sum):
    sequence = [1] * length
    remaining = total_sum - length
    for i in range(length):
        if remaining <= 0:
            break
        if remaining <= total_sum // 8:
            max_ = remaining
        else:
            max_ = remaining // 4
        add = random.randint(1, max_)
        sequence[i] += add
        remaining -= add
    prefix_sums = []
    current_sum = 0
    for num in sequence:
        current_sum += num
        prefix_sums.append(current_sum)
    logging.info(f"原序列：{sequence}")
    logging.info(f"前缀和序列：{prefix_sums}")
    assert sum(sequence) == total_sum, "总和不正确"
    return sequence


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=dtype)
    if dtype == torch.bool:
        return torch.randint(0, 2, data_shape, dtype=dtype)
    if torch.is_floating_point(torch.tensor(0, dtype=dtype)):
        return min_value + (max_value - min_value) * torch.rand(data_shape, dtype=dtype)
    else:
        return torch.randint(low=min_value, high=max_value, size=data_shape, dtype=dtype)


@dataclass
class _SfaGradSingleHeadInputs:
    t_idx: Any
    n2: Any
    sparse_indices_tnd: Any
    k_nope_tnd: Any
    k_pe_tnd: Any
    value_tnd: Any
    q_nope_tnd: Any
    q_pe_tnd: Any
    d_out_tnd: Any
    out_tnd: Any
    sm_max_tnd: Any
    sm_sum_tnd: Any
    scale_value: Any
    group: Any
    slc_kv_len: Any


@dataclass


class _SfaGradSingleHeadOutputs:
    dq_nope_part: Any
    dq_pe_part: Any
    dk_nope_part: Any
    dk_pe_part: Any
    dv_part: Any
    indices: Any


def _sfa_grad_single_head(inputs: _SfaGradSingleHeadInputs):


    """Compute SFA grad for a single (inputs.t_idx, inputs.n2) pair."""
    d = inputs.q_nope_tnd.shape[-1]
    dr = inputs.q_pe_tnd.shape[-1]
    indices = inputs.sparse_indices_tnd[inputs.t_idx, inputs.n2, :inputs.slc_kv_len].long()
    sel_k_nope = inputs.k_nope_tnd[indices, inputs.n2, :].float()
    sel_k_pe = inputs.k_pe_tnd[indices, inputs.n2, :].float()
    sel_k = torch.cat([sel_k_nope, sel_k_pe], dim=-1)
    sel_v = inputs.value_tnd[indices, inputs.n2, :].float()

    dq_nope_part = torch.zeros(inputs.group, d, dtype=torch.float32)
    dq_pe_part = torch.zeros(inputs.group, dr, dtype=torch.float32)
    dk_nope_part = torch.zeros(inputs.slc_kv_len, d, dtype=torch.float32)
    dk_pe_part = torch.zeros(inputs.slc_kv_len, dr, dtype=torch.float32)
    dv_part = torch.zeros(inputs.slc_kv_len, d, dtype=torch.float32)

    for g in range(inputs.group):
        n1 = inputs.n2 * inputs.group + g
        q_nope_vec = inputs.q_nope_tnd[inputs.t_idx, n1, :].float()
        q_pe_vec = inputs.q_pe_tnd[inputs.t_idx, n1, :].float()
        q_vec = torch.cat([q_nope_vec, q_pe_vec]).to(torch.bfloat16).to(torch.float32)
        do_vec = inputs.d_out_tnd[inputs.t_idx, n1, :].float()
        o_vec = inputs.out_tnd[inputs.t_idx, n1, :].float()
        mi = inputs.sm_max_tnd[inputs.n2, inputs.t_idx, g].float()
        li = inputs.sm_sum_tnd[inputs.n2, inputs.t_idx, g].float()

        s_scores = (q_vec.unsqueeze(0) @ sel_k.T).squeeze(0) * inputs.scale_value
        s_shifted = s_scores - mi
        p_vec = (torch.exp(s_shifted) / li)
        dp_vec = (do_vec.unsqueeze(0) @ sel_v.T).squeeze(0)
        dv_local = p_vec.unsqueeze(1).to(torch.bfloat16).to(torch.float32) * do_vec.unsqueeze(0)
        d_val = (do_vec * o_vec).sum()
        ds_vec = (p_vec * (dp_vec - d_val)).to(torch.bfloat16).to(torch.float32)

        dq_full = (ds_vec.unsqueeze(0) @ sel_k).squeeze(0) * inputs.scale_value
        dq_nope_part[g, :] = dq_full[:d]
        dq_pe_part[g, :] = dq_full[d:]

        dk_full = (ds_vec.unsqueeze(1) * q_vec.unsqueeze(0)) * inputs.scale_value
        for ki in range(inputs.slc_kv_len):
            idx = indices[ki]
            dk_nope_part[ki, :] += dk_full[ki, :d]
            dk_pe_part[ki, :] += dk_full[ki, d:]
            dv_part[ki, :] += dv_local[ki]

    return _SfaGradSingleHeadOutputs(dq_nope_part, dq_pe_part, dk_nope_part, dk_pe_part, dv_part, indices)


@dataclass
class SfaGradGoldenTndInputs:
    q_nope_tnd: Any
    q_pe_tnd: Any
    k_nope_tnd: Any
    k_pe_tnd: Any
    value_tnd: Any
    sparse_indices_tnd: Any
    d_out_tnd: Any
    out_tnd: Any
    sm_max_tnd: Any
    sm_sum_tnd: Any
    actual_q_lens: Any
    actual_kv_lens: Any
    scale_value: Any


def sfa_grad_golden_tnd(inputs: SfaGradGoldenTndInputs):


    t_1, n_1, d = inputs.q_nope_tnd.shape
    dr = inputs.q_pe_tnd.shape[-1]
    t_2, n_2, _ = inputs.k_nope_tnd.shape
    k = inputs.sparse_indices_tnd.shape[-1]
    group = n_1 // n_2
    batch = len(inputs.actual_q_lens)

    dq_nope = torch.zeros(t_1, n_1, d, dtype=torch.float32)
    dq_pe = torch.zeros(t_1, n_1, dr, dtype=torch.float32)
    dk_nope = torch.zeros(t_2, n_2, d, dtype=torch.float32)
    dk_pe = torch.zeros(t_2, n_2, dr, dtype=torch.float32)
    dv = torch.zeros(t_2, n_2, d, dtype=torch.float32)

    q_offset = 0
    for b in range(batch):
        s_count = inputs.actual_q_lens[b]
        kv_len = inputs.actual_kv_lens[b]
        for s in range(s_count):
            t_idx = q_offset + s
            for n2 in range(n_2):
                slc_kv_len = min(max(kv_len - s_count + 1 + s, 0), k)
                grad_out = _sfa_grad_single_head(
                    _SfaGradSingleHeadInputs(
                        t_idx, n2, inputs.sparse_indices_tnd, inputs.k_nope_tnd,
                        inputs.k_pe_tnd, inputs.value_tnd,
                        inputs.q_nope_tnd, inputs.q_pe_tnd,
                        inputs.d_out_tnd, inputs.out_tnd,
                        inputs.sm_max_tnd, inputs.sm_sum_tnd,
                        inputs.scale_value, group, slc_kv_len))
                dqn, dqp, dkn, dkp, dvp, indices = (
                    grad_out.dq_nope_part, grad_out.dq_pe_part,
                    grad_out.dk_nope_part, grad_out.dk_pe_part,
                    grad_out.dv_part, grad_out.indices)
                for g in range(group):
                    n1 = n2 * group + g
                    dq_nope[t_idx, n1, :] += dqn[g, :]
                    dq_pe[t_idx, n1, :] += dqp[g, :]
                for ki in range(slc_kv_len):
                    idx = indices[ki]
                    dk_nope[idx, n2, :] += dkn[ki, :]
                    dk_pe[idx, n2, :] += dkp[ki, :]
                    dv[idx, n2, :] += dvp[ki]
        q_offset += s_count

    dtype = inputs.q_nope_tnd.dtype
    return (dq_nope.to(dtype), dq_pe.to(dtype),
            (dk_nope + dv).to(dtype), dk_pe.to(dtype), dv.to(dtype))


def sfa_forward_golden_tnd(q_nope_tnd, q_pe_tnd, k_nope_tnd, k_pe_tnd, value_tnd,
                           sparse_indices_tnd, actual_q_lens, actual_kv_lens, scale_value):
    t_1, n_1, d = q_nope_tnd.shape
    dr = q_pe_tnd.shape[-1]
    t_2, n_2, _ = k_nope_tnd.shape
    k = sparse_indices_tnd.shape[-1]
    group = n_1 // n_2
    batch = len(actual_q_lens)

    out = torch.zeros(t_1, n_1, d, dtype=q_nope_tnd.dtype)
    softmax_max = torch.zeros(n_2, t_1, group, dtype=torch.float32)
    softmax_sum = torch.zeros(n_2, t_1, group, dtype=torch.float32)

    q_offset = 0
    for b in range(batch):
        s_len = actual_q_lens[b]
        kv_len = actual_kv_lens[b]
        for s in range(s_len):
            t_idx = q_offset + s
            for n2 in range(n_2):
                slc_kv_len = min(max(kv_len - s_len + 1 + s, 0), k)
                indices = sparse_indices_tnd[t_idx, n2, :slc_kv_len].long()
                sel_k_nope = k_nope_tnd[indices, n2, :].float()
                sel_k_pe = k_pe_tnd[indices, n2, :].float()
                sel_k = torch.cat([sel_k_nope, sel_k_pe], dim=-1)
                sel_v = value_tnd[indices, n2, :].float()
                for g in range(group):
                    n1 = n2 * group + g
                    q_nope_vec = q_nope_tnd[t_idx, n1, :].float()
                    q_pe_vec = q_pe_tnd[t_idx, n1, :].float()
                    q_vec = torch.cat([q_nope_vec, q_pe_vec])
                    s_scores = (q_vec.unsqueeze(0) @ sel_k.T).squeeze(0) * scale_value
                    mi = s_scores.max()
                    s_shifted = s_scores - mi
                    exp_s = torch.exp(s_shifted)
                    li = exp_s.sum()
                    p = exp_s / li
                    o = (p.unsqueeze(0) @ sel_v).squeeze(0)
                    out[t_idx, n1, :] = o.to(q_nope_tnd.dtype)
                    softmax_max[n2, t_idx, g] = mi
                    softmax_sum[n2, t_idx, g] = li
        q_offset += s_len

    return out, softmax_max, softmax_sum


def _gen_sparse_indices(t_1, n_2, k, actual_q_lens, actual_kv_lens, batch):
    """Generate sparse indices for TND test data."""
    sparse_indices = torch.zeros(t_1, n_2, k, dtype=torch.int64) - 1
    q_offset = 0
    for b in range(batch):
        s_count = actual_q_lens[b]
        kv_len = actual_kv_lens[b]
        for s in range(s_count):
            t_idx = q_offset + s
            for n2 in range(n_2):
                slc_kv_len = min(max(kv_len - s_count + 1 + s, 0), k)
                perm = torch.randperm(slc_kv_len)
                sparse_indices[t_idx, n2, :slc_kv_len] = perm
        q_offset += s_count
    return sparse_indices


def gen_test_data_tnd(actual_q_lens, actual_kv_lens, n_1, n_2, d, dr, k,
                      dtype=torch.bfloat16, seed=42):
    torch.manual_seed(seed)
    batch = len(actual_q_lens)
    group = n_1 // n_2
    d_full = d + dr
    scale_value = 1.0 / math.sqrt(d_full)

    t_1 = sum(actual_q_lens)
    t_2 = sum(actual_kv_lens)

    q_nope = torch.randn(t_1, n_1, d, dtype=dtype) * 0.5 + 0.5
    q_pe = torch.randn(t_1, n_1, dr, dtype=dtype) * 0.5 + 0.5
    k_nope = torch.randn(t_2, n_2, d, dtype=dtype) * 0.5 + 0.5
    k_pe = torch.randn(t_2, n_2, dr, dtype=dtype) * 0.5 + 0.5
    value = k_nope.clone()
    d_out = torch.randn(t_1, n_1, d, dtype=dtype) * 0.5 + 0.5

    sparse_indices = _gen_sparse_indices(t_1, n_2, k, actual_q_lens, actual_kv_lens, batch)

    actual_seq_qlen = torch.tensor(actual_q_lens, dtype=torch.int32).cumsum(0).to(torch.int32)
    actual_seq_kvlen = torch.tensor(actual_kv_lens, dtype=torch.int32).cumsum(0).to(torch.int32)

    out, sm_max, sm_sum = sfa_forward_golden_tnd(
        q_nope, q_pe, k_nope, k_pe, value,
        sparse_indices, actual_q_lens, actual_kv_lens, scale_value
    )
    logging.info("sfa_forward_golden_tnd success!!")

    dq_nope_g, dq_pe_g, dk_nope_g, dk_pe_g, dv_g = sfa_grad_golden_tnd(
        SfaGradGoldenTndInputs(
            q_nope, q_pe, k_nope, k_pe, value,
            sparse_indices, d_out, out,
            sm_max, sm_sum,
            actual_q_lens, actual_kv_lens, scale_value
        )
    )

    return {
        'q_nope': q_nope, 'q_pe': q_pe,
        'k_nope': k_nope, 'k_pe': k_pe,
        'value': value,
        'sparse_indices': sparse_indices,
        'd_out': d_out, 'out': out,
        'sm_max': sm_max, 'sm_sum': sm_sum,
        'actual_seq_qlen': actual_seq_qlen,
        'actual_seq_kvlen': actual_seq_kvlen,
        'scale_value': scale_value,
        't_1': t_1, 't_2': t_2, 'batch': batch,
        'dq_nope_golden': dq_nope_g, 'dq_pe_golden': dq_pe_g,
        'dk_nope_golden': dk_nope_g, 'dk_pe_golden': dk_pe_g,
        'dv_golden': dv_g,
    }


def do_test_sfa_grad_npu_eager(case_name, actual_q_lens, actual_kv_lens,
                      n_1, n_2, d, dr, k, seed=42):
    logging.info("=" * 50)
    logging.info(f"Test: SFA grad NPU ({case_name})")
    logging.info("=" * 50)

    torch.npu.set_device(int(os.environ.get('TILE_FWK_DEVICE_ID', 0)))

    data = gen_test_data_tnd(actual_q_lens, actual_kv_lens, n_1, n_2, d, dr, k, seed=seed)
    scale_value = data['scale_value']
    t_1 = data['t_1']
    t_2 = data['t_2']
    group = n_1 // n_2

    from experimental.ops_transformer.sparse_attention_grad_tnd.sparse_flash_attention_grad_impl import (
        npu_sfa_sparse_attention_grad
    )

    q_nope_npu = data['q_nope'].npu()
    q_pe_npu = data['q_pe'].npu()
    k_nope_npu = data['k_nope'].npu()
    k_pe_npu = data['k_pe'].npu()
    value_npu = data['value'].npu()
    sparse_idx_npu = data['sparse_indices'].to(torch.int32).npu()
    d_out_npu = data['d_out'].npu()
    out_npu = data['out'].npu()
    sm_max_npu = data['sm_max'].npu()
    sm_sum_npu = data['sm_sum'].npu()
    actual_seq_qlen_npu = data['actual_seq_qlen'].npu()
    actual_seq_kvlen_npu = data['actual_seq_kvlen'].npu()

    dq_nope_out_npu, dq_pe_out_npu, dk_nope_out_npu, dk_pe_out_npu, dv_out_npu = npu_sfa_sparse_attention_grad(
        q_nope_npu, q_pe_npu, k_nope_npu, k_pe_npu, value_npu, sparse_idx_npu, d_out_npu, out_npu,
        sm_max_npu, sm_sum_npu, actual_seq_qlen_npu, actual_seq_kvlen_npu, scale_value)

    torch_npu.npu.synchronize()
    compare(dq_nope_out_npu.cpu(), data['dq_nope_golden'], "dQ_nope",
            atol=0.0001, rtol=0.0078125, max_error_count=100)
    compare(dq_pe_out_npu.cpu(), data['dq_pe_golden'], "dQ_pe",
            atol=0.0001, rtol=0.0078125, max_error_count=100)
    compare(dk_nope_out_npu.cpu(), data['dk_nope_golden'], "dK_nope",
            atol=0.0001, rtol=0.0078125, max_error_count=100)
    compare(dk_pe_out_npu.cpu(), data['dk_pe_golden'], "dK_pe",
            atol=0.0001, rtol=0.0078125, max_error_count=100)
    logging.info(f"Test {case_name} PASSED!")


def _save_golden_data(data, case_name, batch, t_1):
    """Save golden data to disk for later use."""
    path = f"./golden/B_{batch}_T1_{t_1}/"
    os.makedirs(path, exist_ok=True)
    for name, tensor in data.items():
        torch.save(tensor, path + f"{name}.pt")
    logging.info("save success !!!")


def _move_inputs_to_npu(data):
    """Move all input tensors to NPU."""
    return {
        'q_pe': data['q_pe'].npu(), 'k_nope': data['k_nope'].npu(),
        'q_nope': data['q_nope'].npu(), 'k_pe': data['k_pe'].npu(),
        'value': data['value'].npu(),
        'actual_seq_qlen': data['actual_seq_qlen'].npu(),
        'actual_seq_kvlen': data['actual_seq_kvlen'].npu(),
        'd_out': data['d_out'].npu(),
        'sparse_idx': data['sparse_indices'].to(torch.int32).npu(),
        'sm_max': data['sm_max'].npu(), 'sm_sum': data['sm_sum'].npu(),
        'out': data['out'].npu(),
    }


@dataclass
class _CreateOutputBuffersOutputs:
    dq_nope_out: Any
    dq_pe_out: Any
    dk_nope_out: Any
    dk_pe_out: Any
    dv_out: Any
    dk_out: Any


def _create_output_buffers(t_1, t_2, n_1, n_2, d, dr, dtype):


    """Create output buffers for SFA grad NPU."""
    dq_nope_out = torch.empty((t_1 * n_1, d), dtype=dtype).npu()
    dq_pe_out = torch.empty((t_1 * n_1, dr), dtype=dtype).npu()
    dk_nope_out = torch.zeros((t_2 * n_2, d), dtype=torch.float32).npu()
    dk_pe_out = torch.zeros((t_2 * n_2, dr), dtype=torch.float32).npu()
    dv_out = torch.zeros((t_2 * n_2, d), dtype=torch.float32).npu()
    dk_out = torch.zeros((t_2 * n_2, d + dr), dtype=torch.float32).npu()
    return _CreateOutputBuffersOutputs(dq_nope_out, dq_pe_out, dk_nope_out, dk_pe_out, dv_out, dk_out)


def do_test_sfa_grad_npu(case_name, actual_q_lens, actual_kv_lens,
                      n_1, n_2, d, dr, k, seed=42):
    logging.info("=" * 60)
    logging.info(f"Test: SFA grad NPU ({case_name})")
    logging.info("=" * 60)

    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    data = gen_test_data_tnd(actual_q_lens, actual_kv_lens, n_1, n_2, d, dr, k, seed=seed)
    scale_value = data['scale_value']
    batch = data['batch']
    t_1 = data['t_1']
    t_2 = data['t_2']
    group = n_1 // n_2

    _save_golden_data(data, case_name, batch, t_1)

    from experimental.ops_transformer.sparse_attention_grad_tnd.sparse_flash_attention_grad_impl import (
        sparse_flash_attention_grad
    )

    npu_data = _move_inputs_to_npu(data)
    dtype = data['q_nope'].dtype
    bufs = _create_output_buffers(t_1, t_2, n_1, n_2, d, dr, dtype)
    dq_nope_out = bufs.dq_nope_out
    dq_pe_out = bufs.dq_pe_out
    dk_nope_out = bufs.dk_nope_out
    dk_pe_out = bufs.dk_pe_out
    dv_out = bufs.dv_out
    dk_out = bufs.dk_out

    sparse_flash_attention_grad(
        npu_data['q_nope'], npu_data['q_pe'], npu_data['k_nope'], npu_data['k_pe'], npu_data['value'],
        npu_data['sparse_idx'], npu_data['d_out'], npu_data['out'],
        npu_data['sm_max'], npu_data['sm_sum'],
        npu_data['actual_seq_qlen'], npu_data['actual_seq_kvlen'],
        dq_nope_out, dq_pe_out, dk_nope_out, dk_pe_out, dv_out,
        dk_nope_out, dk_pe_out, dk_out, dk_out,
        n_1, n_2, d, dr, k, group, scale_value
    )
    torch_npu.npu.synchronize()

    dk_nope_out_slice = dk_out.cpu()[:, :d]
    dk_pe_out_slice = dk_out.cpu()[:, d:]

    compare(dq_nope_out.cpu(), data['dq_nope_golden'].reshape(t_1 * n_1, d), "dQ_nope",
            atol=0.0001, rtol=0.0078125, max_error_count=100)
    compare(dq_pe_out.cpu(), data['dq_pe_golden'].reshape(t_1 * n_1, dr), "dQ_pe",
            atol=0.0001, rtol=0.0078125, max_error_count=100)
    compare(dk_nope_out_slice.to(dtype), data['dk_nope_golden'].reshape(t_2 * n_2, d), "dK_nope",
            atol=0.0001, rtol=0.0078125, max_error_count=100)
    compare(dk_pe_out_slice.to(dtype), data['dk_pe_golden'].reshape(t_2 * n_2, dr), "dK_pe",
            atol=0.0001, rtol=0.0078125, max_error_count=100)

    logging.info(f"Test {case_name} PASSED!")


def test_level0_tiny():
    do_test_sfa_grad_npu("level0_tiny",
                      actual_q_lens=[1], actual_kv_lens=[16],
                      n_1=2, n_2=1, d=8, dr=8, k=8)


@pytest.mark.skip(reason="large test case")
def test_level1_small():
    do_test_sfa_grad_npu("level1_b1_s1",
                      actual_q_lens=[1], actual_kv_lens=[2048],
                      n_1=16, n_2=1, d=512, dr=64, k=1024)


@pytest.mark.skip(reason="large test case")
def test_level2_medium():
    do_test_sfa_grad_npu("level2_b1_s4",
                      actual_q_lens=[128], actual_kv_lens=[32768],
                      n_1=2, n_2=1, d=512, dr=64, k=2048, seed=123)


@pytest.mark.skip(reason="large test case")
def test_level2_t1k():
    do_test_sfa_grad_npu("level2_b1_s4",
                      actual_q_lens=[1024], actual_kv_lens=[32768],
                      n_1=2, n_2=1, d=512, dr=64, k=2048, seed=123)


@pytest.mark.skip(reason="large test case")
def test_level2_t256():
    do_test_sfa_grad_npu("level2_b1_s4",
                      actual_q_lens=[256], actual_kv_lens=[32768],
                      n_1=64, n_2=1, d=512, dr=64, k=2048, seed=123)


@pytest.mark.skip(reason="large test case")
def test_level2_t128():
    do_test_sfa_grad_npu("level2_b1_s4",
                      actual_q_lens=[128], actual_kv_lens=[32768],
                      n_1=32, n_2=1, d=512, dr=64, k=2048, seed=123)


@pytest.mark.skip(reason="large test case")
def test_eager():
    do_test_sfa_grad_npu_eager("level_eager",
                      actual_q_lens=[128, ], actual_kv_lens=[32768, ],
                      n_1=2, n_2=1, d=512, dr=64, k=2048, seed=123)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO
    )

    logging.info("Starting SFA Grad (TND, nope/rope split) tests...")
    test_level0_tiny()
    logging.info("All tests completed!")
