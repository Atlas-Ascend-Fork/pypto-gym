#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Flash Attention HiFP8 Forward Test

Golden: dequant HF8 → FP32 (per-token scales), then online softmax flash attention
with P (attention probabilities) quantized to HF8 via p_scale. p_scale cancels in O = O / L.
"""

import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

import logging
from dataclasses import dataclass

import torch
import torch_npu

import numpy as np
import pytest

from experimental.ops_transformer.flash_attention_fp8.flash_attention_fp8_impl \
    import flash_attention_fp8_varlen_forward_kernel


logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)


NUM_HEADS = 8
HEAD_DIM = 64
HIDDEN_DIM = NUM_HEADS * HEAD_DIM

Q_TILE = 320
K_TILE = 320


@dataclass
class TileConfig:
    q_tile: int = Q_TILE
    k_tile: int = K_TILE


def create_inputs(batch_size, s1_size, s2_size, num_heads, head_dim, device):
    q_seqlens = [s1_size] * batch_size
    kv_seqlens = [s2_size] * batch_size
    total_q = sum(q_seqlens)
    total_kv = sum(kv_seqlens)

    torch.manual_seed(42)
    q_fp32 = torch.randn(total_q, num_heads, head_dim, dtype=torch.float32, device='cpu') * 0.5
    k_fp32 = torch.randn(total_kv, num_heads, head_dim, dtype=torch.float32, device='cpu') * 0.5
    v_fp32 = torch.randn(total_kv, num_heads, head_dim, dtype=torch.float32, device='cpu') * 0.5

    q_hf8 = torch_npu.npu_dtype_cast(q_fp32.to(device), torch_npu.hifloat8)
    k_hf8 = torch_npu.npu_dtype_cast(k_fp32.to(device), torch_npu.hifloat8)
    v_hf8 = torch_npu.npu_dtype_cast(v_fp32.to(device), torch_npu.hifloat8)

    d_scale_q = torch.empty(total_q, num_heads, 1, dtype=torch.float32, device=device).uniform_(0.5, 2.0)
    d_scale_k = torch.empty(total_kv, num_heads, 1, dtype=torch.float32, device=device).uniform_(0.5, 2.0)
    d_scale_v = torch.empty(total_kv, num_heads, 1, dtype=torch.float32, device=device).uniform_(0.5, 2.0)

    p_scale = torch.tensor([16.0], dtype=torch.float32, device=device)

    cu_seqlens_q = torch.tensor([0] + list(np.cumsum(q_seqlens)), dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0] + list(np.cumsum(kv_seqlens)), dtype=torch.int32, device=device)

    return q_hf8, k_hf8, v_hf8, d_scale_q, d_scale_k, d_scale_v, p_scale, cu_seqlens_q, \
            cu_seqlens_k, q_seqlens, kv_seqlens


def attention_forward_golden(q_hf8, k_hf8, v_hf8, dscale_q, dscale_k, dscale_v, p_scale_val, scale):
    """
    Golden: dequant HF8 → FP32 (per-token multiplicative scales), then online softmax
    flash attention with P quantized to HF8 via p_scale. p_scale cancels in O = O / L.
    All tensors on NPU.
    """
    s1_size, head_dim = q_hf8.shape
    s2_size = k_hf8.shape[0]
    device = q_hf8.device

    q_f = torch_npu.npu_dtype_cast(q_hf8, torch.float32, input_dtype=torch_npu.hifloat8) * dscale_q
    k_f = torch_npu.npu_dtype_cast(k_hf8, torch.float32, input_dtype=torch_npu.hifloat8) * dscale_k
    v_f = torch_npu.npu_dtype_cast(v_hf8, torch.float32, input_dtype=torch_npu.hifloat8) * dscale_v

    q_tile = Q_TILE
    k_tile = K_TILE

    q_tile_count = (s1_size + q_tile - 1) // q_tile
    k_tile_count = (s2_size + k_tile - 1) // k_tile

    o_out = torch.zeros(s1_size, head_dim, dtype=torch.bfloat16, device=device)
    l_out = torch.zeros(s1_size, 1, dtype=torch.float32, device=device)
    m_out = torch.zeros(s1_size, 1, dtype=torch.float32, device=device)

    for q_tile_idx in range(q_tile_count):
        q_tile_start = q_tile_idx * q_tile
        q_tile_end = min(q_tile_start + q_tile, s1_size)
        q_tile_len = q_tile_end - q_tile_start

        q_tile_view = q_f[q_tile_start:q_tile_end, :]

        oi_update = torch.zeros(q_tile, head_dim, dtype=torch.float32, device=device)
        li_update = torch.zeros(q_tile, 1, dtype=torch.float32, device=device)
        mi_update = torch.full((q_tile, 1), float('-inf'), dtype=torch.float32, device=device)

        for k_tile_idx in range(k_tile_count):
            k_tile_start = k_tile_idx * k_tile
            k_tile_end = min(k_tile_start + k_tile, s2_size)
            k_tile_len = k_tile_end - k_tile_start

            k_tile_view = k_f[k_tile_start:k_tile_end, :]
            v_tile_view = v_f[k_tile_start:k_tile_end, :]

            scores = torch.matmul(q_tile_view, k_tile_view.T) * scale

            mij = scores.amax(dim=-1, keepdim=True)
            s_shifted = scores - mij
            pij = torch.exp(s_shifted)

            pij_scaled = pij * p_scale_val
            pij_hf8 = torch_npu.npu_dtype_cast(pij_scaled, torch_npu.hifloat8)
            pij = torch_npu.npu_dtype_cast(pij_hf8, torch.float32, input_dtype=torch_npu.hifloat8)

            lij = pij.sum(dim=-1, keepdim=True)

            oij = torch.matmul(pij, v_tile_view)

            if k_tile_idx == 0:
                if k_tile_idx == k_tile_count - 1:
                    out_bf16 = (oij / lij).to(torch.bfloat16)

                    o_out[q_tile_start:q_tile_end, :] = out_bf16[:q_tile_len, :]
                    l_out[q_tile_start:q_tile_end, :] = lij[:q_tile_len, :]
                    m_out[q_tile_start:q_tile_end, :] = mij[:q_tile_len, :]
                else:
                    oi_update[:q_tile_len, :] = oij[:q_tile_len, :]
                    li_update[:q_tile_len, :] = lij[:q_tile_len, :]
                    mi_update[:q_tile_len, :] = mij[:q_tile_len, :]
            else:
                mi = mi_update[:q_tile_len, :]
                li = li_update[:q_tile_len, :]
                oi = oi_update[:q_tile_len, :]

                mi_new = torch.maximum(mi, mij[:q_tile_len, :])
                t1 = torch.exp(mi - mi_new)
                t2 = torch.exp(mij[:q_tile_len, :] - mi_new)

                li_new = t1 * li + t2 * lij[:q_tile_len, :]
                oi_tmp = t1 * oi + t2 * oij[:q_tile_len, :]

                if k_tile_idx == k_tile_count - 1:
                    out_bf16 = (oi_tmp / li_new).to(torch.bfloat16)

                    o_out[q_tile_start:q_tile_end, :] = out_bf16[:q_tile_len, :]
                    l_out[q_tile_start:q_tile_end, :] = li_new[:q_tile_len, :]
                    m_out[q_tile_start:q_tile_end, :] = mi_new[:q_tile_len, :]
                else:
                    oi_update[:q_tile_len, :] = oi_tmp
                    li_update[:q_tile_len, :] = li_new
                    mi_update[:q_tile_len, :] = mi_new

    return o_out, m_out, l_out


def run_test(batch_size=None, num_heads=None, s1_size=None,
             s2_size=None, dim=None, tile_config=None):
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
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
        tile_config = TileConfig()

    hidden_dim = num_heads * dim
    scale = 1.0 / (dim ** 0.5)

    logging.info("=" * 60)
    logging.info(f"HiFP8 Test Case: batch={batch_size}, heads={num_heads}, "
                 f"Q:s1={s1_size}, KV:s2={s2_size}, dim={dim}")
    logging.info(f"  hidden_dim={hidden_dim}, scale={scale:.6f}")
    logging.info("=" * 60)

    q_hf8, k_hf8, v_hf8, d_scale_q, d_scale_k, d_scale_v, p_scale, \
        cu_seqlens_q, cu_seqlens_k, q_seqlens, kv_seqlens = create_inputs(
        batch_size, s1_size, s2_size, num_heads, dim, device)

    total_q = batch_size * s1_size
    total_kv = batch_size * s2_size

    out_npu = torch.empty(total_q, hidden_dim, dtype=torch.bfloat16, device=device)
    l_out_npu = torch.empty(total_q, num_heads, dtype=torch.float32, device=device)
    m_out_npu = torch.empty(total_q, num_heads, dtype=torch.float32, device=device)

    out_golden = torch.empty(total_q, hidden_dim, dtype=torch.bfloat16, device=device)
    l_golden = torch.empty(total_q, num_heads, dtype=torch.float32, device=device)
    m_golden = torch.empty(total_q, num_heads, dtype=torch.float32, device=device)

    p_scale_val = p_scale.item()

    q_off, k_off = 0, 0
    for b in range(batch_size):
        sq, sk = q_seqlens[b], kv_seqlens[b]

        for h in range(num_heads):
            h_off = h * dim

            q_h = q_hf8[q_off:q_off + sq, h, :].contiguous()
            k_h = k_hf8[k_off:k_off + sk, h, :].contiguous()
            v_h = v_hf8[k_off:k_off + sk, h, :].contiguous()
            dsq = d_scale_q[q_off:q_off + sq, h, :].contiguous()
            dsk = d_scale_k[k_off:k_off + sk, h, :].contiguous()
            dsv = d_scale_v[k_off:k_off + sk, h, :].contiguous()

            golden_o, golden_m, golden_l = attention_forward_golden(
                q_h, k_h, v_h, dsq, dsk, dsv, p_scale_val, scale)

            out_golden[q_off:q_off + sq, h_off:h_off + dim] = golden_o
            m_golden[q_off:q_off + sq, h:h + 1] = golden_m
            l_golden[q_off:q_off + sq, h:h + 1] = golden_l
        q_off += sq
        k_off += sk

    logging.info("  Running kernel...")
    for _ in range(1):
        a = torch.randn((int(192 * 1024 * 1024 * 2.5))).to(torch.float32).npu()
        for _ in range(100):
            _a_max = torch.max(a)
        flash_attention_fp8_varlen_forward_kernel(
            q_hf8, k_hf8, v_hf8, d_scale_q, d_scale_k, d_scale_v, p_scale,
            out_npu, l_out_npu, m_out_npu, cu_seqlens_q, cu_seqlens_k)

    torch.set_printoptions(precision=6)
    passed = True
    for name, npu_tensor, golden_tensor, rtol, atol in [
        ("O", out_npu, out_golden, 0.05, 0.005),
        ("L", l_out_npu, l_golden, 0.05, 0.005),
        ("M", m_out_npu, m_golden, 0.01, 0.001),
    ]:
        npu_np = npu_tensor.cpu().float().numpy()
        golden_np = golden_tensor.cpu().float().numpy()
        max_diff = np.abs(npu_np - golden_np).max()
        mean_diff = np.abs(npu_np - golden_np).mean()

        close = np.allclose(npu_np, golden_np, rtol=rtol, atol=atol)
        if not close:
            passed = False

        logging.info(f"  {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
                     f"rtol={rtol}, atol={atol} => {'PASS' if close else 'FAIL'}")

    logging.info(f"  {'PASSED' if passed else 'FAILED'}")
    logging.info("")
    return passed


@pytest.mark.soc("950")
def test_01():
    return run_test(batch_size=8, num_heads=8, s1_size=320, s2_size=320, dim=64)


@pytest.mark.soc("950")
def test_02():
    return run_test(batch_size=1, num_heads=8, s1_size=4096, s2_size=4096, dim=128)


@pytest.mark.soc("950")
def test_03():
    return run_test(batch_size=2, num_heads=8, s1_size=4096, s2_size=4096, dim=128)


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
    logging.info("Flash Attention HiFP8 Forward (4-loop, Q+KV tiling, P quantized to HF8)")
    logging.info("=" * 60 + "\n")

    test_funcs = [
        # test_01,
        # test_02,
        test_03,
        # test_04,
        # test_05,
        # test_06,
    ]

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
