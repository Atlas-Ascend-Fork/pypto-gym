#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance of the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Flash Attention Forward with Dynamic Variable Length Sequences
"""
import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

import collections
import logging
from dataclasses import dataclass

import torch
import torch_npu

import numpy as np
import pytest

from experimental.ops_transformer.flash_attention_mha.flash_attention_mha_impl import (
    flash_attention_varlen_forward_kernel,
    FlashAttentionTileShapeConfig,
    flash_attention_varlen_forward_kernel_910,
)

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)

NUM_HEADS = 8
HEAD_DIM = 64
HIDDEN_DIM = NUM_HEADS * HEAD_DIM
Q_TILE = 320
K_TILE = 320

MhaInputs = collections.namedtuple("MhaInputs",
    ["q", "k", "v", "cu_seqlens_q", "cu_seqlens_k", "q_seqlens", "kv_seqlens"])
AttentionForwardOutput = collections.namedtuple("AttentionForwardOutput", ["o", "m", "l"])


def create_inputs(batch_size, s1_size, s2_size, num_heads, head_dim, device):
    q_seqlens = [s1_size] * batch_size
    kv_seqlens = [s2_size] * batch_size
    total_q = sum(q_seqlens)
    total_kv = sum(kv_seqlens)

    torch.manual_seed(42)
    q = torch.randn(total_q, num_heads, head_dim, dtype=torch.bfloat16, device=device) + 0.5
    k = torch.randn(total_kv, num_heads, head_dim, dtype=torch.bfloat16, device=device) + 0.5
    v = torch.randn(total_kv, num_heads, head_dim, dtype=torch.bfloat16, device=device) + 0.5

    cu_seqlens_q = torch.tensor([0] + list(np.cumsum(q_seqlens)), dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0] + list(np.cumsum(kv_seqlens)), dtype=torch.int32, device=device)

    return MhaInputs(q=q, k=k, v=v,
                     cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                     q_seqlens=q_seqlens, kv_seqlens=kv_seqlens)


def attention_forward_golden_noflash(q, k, v, scale):
    scores = torch.matmul(q.cpu().to(torch.float32), k.cpu().transpose(1, 0).to(torch.float32)) * scale
    m = scores.amax(dim=-1, keepdim=True)
    p_unnorm = torch.exp(scores - m)
    l = p_unnorm.sum(dim=-1, keepdim=True)
    p_norm = p_unnorm / l
    o = torch.matmul(p_norm.to(torch.bfloat16).to(torch.float32), v.cpu().to(torch.float32)).to(torch.bfloat16)
    return AttentionForwardOutput(o, m, l)


def _kv_tile_iter_body(q_tile_view, k_tile_view, v_tile_view, scale,
                        oi_upd, li_upd, mi_upd, q_tile_len, k_tile_idx, k_tile_count):
    """One KV tile iteration: compute scores, softmax, matmul, and update accumulators."""
    scores = torch.matmul(q_tile_view, k_tile_view.T) * scale
    mij = scores.amax(dim=-1, keepdim=True)
    s_shifted = scores - mij
    pij = torch.exp(s_shifted)
    lij = pij.sum(dim=-1, keepdim=True)
    p_bf16 = pij.to(torch.bfloat16)
    oij = torch.matmul(p_bf16, v_tile_view)

    if k_tile_idx == 0:
        if k_tile_count == 1:
            return None, None, None, None, None, None, True, None, None, None, \
                lij[:q_tile_len, :], mij[:q_tile_len, :], oij[:q_tile_len, :], scores, pij
        oi_upd = torch.zeros(q_tile_view.shape[0], q_tile_view.shape[1], dtype=torch.float32)
        li_upd = torch.zeros(q_tile_view.shape[0], 1, dtype=torch.float32)
        mi_upd = torch.full((q_tile_view.shape[0], 1), float('-inf'), dtype=torch.float32)
        oi_upd[:q_tile_len, :] = oij[:q_tile_len, :]
        li_upd[:q_tile_len, :] = lij[:q_tile_len, :]
        mi_upd[:q_tile_len, :] = mij[:q_tile_len, :]
        return oi_upd, li_upd, mi_upd, lij, mij, oij, False, scores, pij, lij, mij, oij, scores, pij
    else:
        mi = mi_upd[:q_tile_len, :]
        li = li_upd[:q_tile_len, :]
        oi = oi_upd[:q_tile_len, :]
        mi_new = torch.maximum(mi, mij[:q_tile_len, :])
        t1 = torch.exp(mi - mi_new)
        t2 = torch.exp(mij[:q_tile_len, :] - mi_new)
        li_new = t1 * li + t2 * lij[:q_tile_len, :]
        oi_tmp = t1 * oi + t2 * oij[:q_tile_len, :]
        if k_tile_idx == k_tile_count - 1:
            return None, None, None, None, None, None, True, None, None, None, \
                li_new, mi_new, oi_tmp, scores, pij
        oi_upd[:q_tile_len, :] = oi_tmp
        li_upd[:q_tile_len, :] = li_new
        mi_upd[:q_tile_len, :] = mi_new
        return oi_upd, li_upd, mi_upd, lij, mij, oij, False, scores, pij, li_new, mi_new, oi_tmp, scores, pij


@dataclass
class ProcessKvTileInputs:
    q_tile_view: torch.Tensor
    k_tile_view: torch.Tensor
    v_tile_view: torch.Tensor
    scale: float
    q_tile_len: int
    k_tile_idx: int
    k_tile_count: int
    oi_upd: torch.Tensor
    li_upd: torch.Tensor
    mi_upd: torch.Tensor
    q_tile_start: int
    q_tile_end: int
    o_out: torch.Tensor
    l_out: torch.Tensor
    m_out: torch.Tensor


def _process_kv_tile(inputs: ProcessKvTileInputs):
    """Process one KV tile: compute scores, softmax, matmul, update accumulators."""
    scores = torch.matmul(inputs.q_tile_view, inputs.k_tile_view.T) * inputs.scale
    mij = scores.amax(dim=-1, keepdim=True)
    s_shifted = scores - mij
    pij = torch.exp(s_shifted)
    lij = pij.sum(dim=-1, keepdim=True)
    p_bf16 = pij.to(torch.bfloat16)
    oij = torch.matmul(p_bf16, inputs.v_tile_view)

    if inputs.k_tile_idx == 0:
        if inputs.k_tile_idx == inputs.k_tile_count - 1:
            pij_div = pij / lij
            pij_bf16 = pij_div.to(torch.bfloat16)
            out_bf16 = torch.matmul(pij_bf16, inputs.v_tile_view)
            inputs.o_out[inputs.q_tile_start:inputs.q_tile_end, :] = out_bf16[:inputs.q_tile_len, :]
            inputs.l_out[inputs.q_tile_start:inputs.q_tile_end, :] = lij[:inputs.q_tile_len, :]
            inputs.m_out[inputs.q_tile_start:inputs.q_tile_end, :] = mij[:inputs.q_tile_len, :]
        else:
            inputs.oi_upd[:inputs.q_tile_len, :] = oij[:inputs.q_tile_len, :]
            inputs.li_upd[:inputs.q_tile_len, :] = lij[:inputs.q_tile_len, :]
            inputs.mi_upd[:inputs.q_tile_len, :] = mij[:inputs.q_tile_len, :]
    else:
        mi = inputs.mi_upd[:inputs.q_tile_len, :]
        li = inputs.li_upd[:inputs.q_tile_len, :]
        oi = inputs.oi_upd[:inputs.q_tile_len, :]
        mi_new = torch.maximum(mi, mij[:inputs.q_tile_len, :])
        t1 = torch.exp(mi - mi_new)
        t2 = torch.exp(mij[:inputs.q_tile_len, :] - mi_new)
        li_new = t1 * li + t2 * lij[:inputs.q_tile_len, :]
        oi_tmp = t1 * oi + t2 * oij[:inputs.q_tile_len, :]
        if inputs.k_tile_idx == inputs.k_tile_count - 1:
            out_fp32 = oi_tmp / li_new
            out_bf16 = out_fp32.to(torch.bfloat16)
            inputs.o_out[inputs.q_tile_start:inputs.q_tile_end, :] = out_bf16[:inputs.q_tile_len, :]
            inputs.l_out[inputs.q_tile_start:inputs.q_tile_end, :] = li_new[:inputs.q_tile_len, :]
            inputs.m_out[inputs.q_tile_start:inputs.q_tile_end, :] = mi_new[:inputs.q_tile_len, :]
        else:
            inputs.oi_upd[:inputs.q_tile_len, :] = oi_tmp
            inputs.li_upd[:inputs.q_tile_len, :] = li_new
            inputs.mi_upd[:inputs.q_tile_len, :] = mi_new


def attention_forward_golden(q, k, v, scale):
    s1_size, head_dim = q.shape
    s2_size = k.shape[0]
    q_f = q.cpu()
    k_f = k.cpu()
    v_f = v.cpu()
    q_tile = Q_TILE
    k_tile = K_TILE
    q_tile_count = (s1_size + q_tile - 1) // q_tile
    k_tile_count = (s2_size + k_tile - 1) // k_tile

    o_out = torch.empty(s1_size, head_dim, dtype=torch.bfloat16)
    l_out = torch.empty(s1_size, 1, dtype=torch.float32)
    m_out = torch.empty(s1_size, 1, dtype=torch.float32)

    for q_tile_idx in range(q_tile_count):
        q_tile_start = q_tile_idx * q_tile
        q_tile_end = min(q_tile_start + q_tile, s1_size)
        q_tile_len = q_tile_end - q_tile_start
        q_tile_view = q_f[q_tile_start:q_tile_end, :]

        oi_upd = torch.empty(q_tile, head_dim, dtype=torch.float32)
        li_upd = torch.empty(q_tile, 1, dtype=torch.float32)
        mi_upd = torch.empty(q_tile, 1, dtype=torch.float32)

        for k_tile_idx in range(k_tile_count):
            k_tile_start = k_tile_idx * k_tile
            k_tile_end = min(k_tile_start + k_tile, s2_size)
            k_tile_view = k_f[k_tile_start:k_tile_end, :]
            v_tile_view = v_f[k_tile_start:k_tile_end, :]
            scores = torch.matmul(q_tile_view.to(torch.float32), k_tile_view.to(torch.float32).T) * scale

            mij = scores.amax(dim=-1, keepdim=True)
            s_shifted = scores - mij
            pij = torch.exp(s_shifted)
            lij = pij.sum(dim=-1, keepdim=True)

            p_bf16 = pij.to(torch.bfloat16)
            oij = torch.matmul(p_bf16.to(torch.float32), v_tile_view.to(torch.float32))

            if k_tile_idx == 0:
                if k_tile_idx == k_tile_count - 1:
                    pij_div = pij / lij
                    pij_bf16 = pij_div.to(torch.bfloat16)
                    out_bf16 = torch.matmul(pij_bf16.to(torch.float32),
                                            v_tile_view.to(torch.float32)).to(torch.bfloat16)

                    o_out[q_tile_start:q_tile_end, :] = out_bf16[:q_tile_len, :]
                    l_out[q_tile_start:q_tile_end, :] = lij[:q_tile_len, :]
                    m_out[q_tile_start:q_tile_end, :] = mij[:q_tile_len, :]
                else:
                    oi_upd[:q_tile_len, :] = oij[:q_tile_len, :]
                    li_upd[:q_tile_len, :] = lij[:q_tile_len, :]
                    mi_upd[:q_tile_len, :] = mij[:q_tile_len, :]
            else:
                mi = mi_upd[:q_tile_len, :]
                li = li_upd[:q_tile_len, :]
                oi = oi_upd[:q_tile_len, :]

                mi_new = torch.maximum(mi, mij[:q_tile_len, :])
                t1 = torch.exp(mi - mi_new)
                t2 = torch.exp(mij[:q_tile_len, :] - mi_new)

                li_new = t1 * li + t2 * lij[:q_tile_len, :]
                oi_tmp = t1 * oi + t2 * oij[:q_tile_len, :]

                if k_tile_idx == k_tile_count - 1:
                    out_fp32 = oi_tmp / li_new
                    out_bf16 = out_fp32.to(torch.bfloat16)

                    o_out[q_tile_start:q_tile_end, :] = out_bf16[:q_tile_len, :]
                    l_out[q_tile_start:q_tile_end, :] = li_new[:q_tile_len, :]
                    m_out[q_tile_start:q_tile_end, :] = mi_new[:q_tile_len, :]
                else:
                    oi_upd[:q_tile_len, :] = oi_tmp
                    li_upd[:q_tile_len, :] = li_new
                    mi_upd[:q_tile_len, :] = mi_new

    return o_out, m_out, l_out


def _compute_golden_outputs(inputs, batch_size, num_heads, dim, s1_size, scale, no_flash):
    """Compute golden O/M/L outputs for all batches and heads."""
    out_golden = torch.empty(batch_size * s1_size, num_heads * dim, dtype=torch.bfloat16)
    l_golden = torch.empty(batch_size * s1_size, num_heads, dtype=torch.float32)
    m_golden = torch.empty(batch_size * s1_size, num_heads, dtype=torch.float32)

    q_off, k_off = 0, 0
    for b in range(batch_size):
        sq, sk = inputs.q_seqlens[b], inputs.kv_seqlens[b]
        for h in range(num_heads):
            h_off = h * dim
            q_h = inputs.q[q_off:q_off + sq, h, :]
            k_h = inputs.k[k_off:k_off + sk, h, :]
            v_h = inputs.v[k_off:k_off + sk, h, :]
            if no_flash:
                golden_o, golden_m, golden_l = attention_forward_golden_noflash(q_h, k_h, v_h, scale)
            else:
                golden_o, golden_m, golden_l = attention_forward_golden(q_h, k_h, v_h, scale)
            out_golden[q_off:q_off + sq, h_off:h_off + dim] = golden_o
            m_golden[q_off:q_off + sq, h:h + 1] = golden_m
            l_golden[q_off:q_off + sq, h:h + 1] = golden_l
        q_off += sq
        k_off += sk
    return out_golden, l_golden, m_golden


def _run_kernel_and_verify(inputs, out_npu, l_out_npu, m_out_npu,
                            out_golden, l_golden, m_golden, tile_config, perf_910):
    """Run the kernel and verify outputs against golden."""
    logging.info("  Running kernel...")
    if perf_910:
        flash_attention_varlen_forward_kernel_910(
            inputs.q, inputs.k, inputs.v, out_npu, l_out_npu, m_out_npu,
            inputs.cu_seqlens_q, inputs.cu_seqlens_k, tile_config)
    else:
        flash_attention_varlen_forward_kernel(
            inputs.q, inputs.k, inputs.v, out_npu, l_out_npu, m_out_npu,
            inputs.cu_seqlens_q, inputs.cu_seqlens_k, tile_config)

    torch.set_printoptions(precision=6)
    passed = True
    for name, npu_tensor, golden_tensor, rtol, atol in [
        ("O", out_npu, out_golden, 0.0078125, 0.0001),
        ("L", l_out_npu, l_golden, 0.005, 0.000025),
        ("M", m_out_npu, m_golden, 0.005, 0.000025),
    ]:
        npu_np = npu_tensor.cpu().float().numpy()
        golden_np = golden_tensor.float().numpy()
        max_diff = np.abs(npu_np - golden_np).max()
        try:
            from common_utils import compare
            compare(npu_tensor.cpu(), golden_tensor, name, atol=atol, rtol=rtol, max_error_count=10)
            logging.info(f"  {name}: PASSED (max_diff={max_diff:.6f}, rtol={rtol}, atol={atol})")
        except AssertionError as e:
            logging.info(f"  {name}: FAILED (max_diff={max_diff:.6f})")
            logging.info(f"    {e}")
            passed = False
    return passed


def _get_default_tile_config(perf_910):
    """Get default tile config if none provided."""
    if perf_910:
        return FlashAttentionTileShapeConfig(
            q_tile=2048, k_tile=2048,
            c1_cube_tile=[[128, 128], [256, 512], [128, 128]],
            v1_tile=[8, 1024],
            c2_cube_tile=[[128, 128], [256, 512], [128, 128]],
            v2_tile=[64, 128])
    return FlashAttentionTileShapeConfig(
        q_tile=Q_TILE, k_tile=K_TILE,
        c1_cube_tile=[[128, 128], [128, 256], [128, 128]],
        v1_tile=[64, 512],
        c2_cube_tile=[[128, 512], [256, 512], [64, 64]],
        v2_tile=[512, 64])


def run_test(batch_size=None, num_heads=None, s1_size=None,
             s2_size=None, dim=None, tile_config=None, perf_910=False, no_flash=False):
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    device = f'npu:{device_id}'

    if batch_size is None:
        batch_size = 1
    if num_heads is None:
        num_heads = NUM_HEADS
    if s1_size is None:
        s1_size = 320
    if s2_size is None:
        s2_size = s1_size
    if dim is None:
        dim = HEAD_DIM
    if tile_config is None:
        tile_config = _get_default_tile_config(perf_910)

    hidden_dim = num_heads * dim
    scale = 1.0 / (dim ** 0.5)

    logging.info("=" * 60)
    logging.info(f"Test Case: batch={batch_size}, heads={num_heads}, "
                 f"Q:s1_size={s1_size}, KV:s2_size={s2_size}, dim={dim}")
    logging.info(f"  hidden_dim={hidden_dim}, scale={scale:.6f}")
    logging.info("=" * 60)

    inputs = create_inputs(batch_size, s1_size, s2_size, num_heads, dim, device)
    total_q = batch_size * s1_size
    total_kv = batch_size * s2_size

    out_npu = torch.empty(total_q, hidden_dim, dtype=torch.bfloat16, device=device)
    l_out_npu = torch.empty(total_q, num_heads, dtype=torch.float32, device=device)
    m_out_npu = torch.empty(total_q, num_heads, dtype=torch.float32, device=device)

    out_golden, l_golden, m_golden = _compute_golden_outputs(
        inputs, batch_size, num_heads, dim, s1_size, scale, no_flash)

    passed = _run_kernel_and_verify(inputs, out_npu, l_out_npu, m_out_npu,
                                     out_golden, l_golden, m_golden, tile_config, perf_910)

    logging.info(f"  {'PASSED' if passed else 'FAILED'}")
    logging.info("")

    assert passed, (f"Accuracy Compare Failed!")

    return passed


@pytest.mark.skip(reason="large test case")
def test_00_910_1b():
    return run_test(batch_size=1, num_heads=8, s1_size=4096, s2_size=4096, dim=128, perf_910=True, no_flash=False)


@pytest.mark.skip(reason="large test case")
def test_00_910_2b():
    return run_test(batch_size=2, num_heads=8, s1_size=4096, s2_size=4096, dim=128, perf_910=True, no_flash=False)


@pytest.mark.skip(reason="large test case")
def test_00_910_8b():
    return run_test(batch_size=8, num_heads=8, s1_size=4096, s2_size=4096, dim=128, perf_910=True, no_flash=False)


def test_01():
    return run_test(batch_size=8, num_heads=8, s1_size=320, s2_size=320, dim=64)


@pytest.mark.soc("950")
def test_02():
    config = FlashAttentionTileShapeConfig(
        q_tile=128, k_tile=128,
        c1_cube_tile=[[128, 128], [128, 128], [128, 128]],
        v1_tile=[128, 64],
        c2_cube_tile=[[128, 128], [128, 128], [128, 128]],
        v2_tile=[128, 64])
    return run_test(batch_size=1, num_heads=8, s1_size=4096, s2_size=4096, dim=128, tile_config=config)


@pytest.mark.soc("950")
def test_03():
    return run_test(batch_size=8, num_heads=16, s1_size=32, s2_size=32, dim=32)


@pytest.mark.soc("950")
def test_04():
    return run_test(batch_size=8, num_heads=16, s1_size=64, s2_size=64, dim=32)


@pytest.mark.soc("950")
def test_05():
    return run_test(batch_size=8, num_heads=8, s1_size=32, s2_size=32, dim=64)


@pytest.mark.soc("950")
def test_06():
    return run_test(batch_size=8, num_heads=4, s1_size=64, s2_size=64, dim=128)


def main():
    logging.info("\n" + "=" * 60)
    logging.info("Flash Attention Forward (4-loop, Q+KV tiling)")
    logging.info("=" * 60 + "\n")

    test_funcs = [test_00_910_1b, test_00_910_2b, test_00_910_8b, test_01, test_02, test_03, test_04, test_05, test_06]

    results = []
    for i, fn in enumerate(test_funcs):
        logging.info(f"[{i+1}/{len(test_funcs)}] {fn.__name__}: {fn.__doc__}")
        try:
            passed = fn()
            results.append((fn.__name__, fn.__doc__, passed))
        except Exception as e:
            logging.info(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((fn.__name__, fn.__doc__, False))
            logging.info("")

    logging.info("=" * 60)
    logging.info("Summary:")
    logging.info("-" * 60)
    all_passed = True
    for name, desc, passed in results:
        status = "PASSED" if passed else "FAILED"
        logging.info(f"  {name}: {desc}  => {status}")
        if not passed:
            all_passed = False
    logging.info("-" * 60)
    logging.info(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
