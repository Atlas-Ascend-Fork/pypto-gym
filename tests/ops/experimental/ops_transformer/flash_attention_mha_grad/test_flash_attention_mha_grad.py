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
Flash Attention Backward with Dynamic Variable Length Sequences

语义约定:
  - Q 侧: s1_size (Q seqlen), 张量包括 Q/O/dO/L/M/dQ
  - KV 侧: s2_size (KV seqlen), 张量包括 K/V/dK/dV
  - S2_TILE: KV 序列维度的分块大小 (将 s2_size 切分为多个 tile 迭代)

3 loops: batch + head + kv_tile (KV sequence tiling).
Tiles KV sequence dimension by S2_TILE to reduce intermediate attention matrix
from [s1_size, s2_size] to [s1_size, S2_TILE] per iteration.
dK and dV are accumulated across kv tiles.
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

from typing import Any
import numpy as np
from numpy.testing import assert_allclose
import pytest
import torch
import torch_npu


from experimental.ops_transformer.flash_attention_mha_grad.flash_attention_mha_grad_impl \
    import flash_attention_mha_grad_kernel_impl, FlashAttentionGradTileShapeConfig


logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)


NUM_HEADS_DEFAULT = 8
HEAD_DIM_DEFAULT = 64
S1_TILE = 512
S2_TILE = 512

MhaGradInputs = collections.namedtuple("MhaGradInputs", \
    ["q", "k", "v", "actual_q", "actual_kv", "q_seqlens", "kv_seqlens"])
AttentionBackwardOutput = collections.namedtuple("AttentionBackwardOutput", ["dq", "dk", "dv"])


def get_device_id():
    return int(os.environ.get('TILE_FWK_DEVICE_ID', 0))


@pytest.fixture(scope="module")
def device():
    device_id = get_device_id()
    torch.npu.set_device(device_id)
    return f'npu:{device_id}'


@dataclass
class TileConfig:
    """分块配置结构体，封装 kernel 所需的 tile 参数。"""
    s2_tile: int = S2_TILE


def create_inputs(batch_size, s1_size, s2_size, num_heads, head_dim, device,
                   q_seqlens=None, kv_seqlens=None):
    if q_seqlens is None:
        q_seqlens = [s1_size] * batch_size
    if kv_seqlens is None:
        kv_seqlens = [s2_size] * batch_size
    assert len(q_seqlens) == batch_size
    assert len(kv_seqlens) == batch_size
    total_q = sum(q_seqlens)
    total_kv = sum(kv_seqlens)
    torch.manual_seed(42)
    q = torch.randn(total_q, num_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(total_kv, num_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    v = torch.randn(total_kv, num_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    q_cumsum = [0]
    for sq in q_seqlens:
        q_cumsum.append(q_cumsum[-1] + sq)
    actual_q = torch.tensor(q_cumsum, dtype=torch.int32, device=device)
    kv_cumsum = [0]
    for skv in kv_seqlens:
        kv_cumsum.append(kv_cumsum[-1] + skv)
    actual_kv = torch.tensor(kv_cumsum, dtype=torch.int32, device=device)
    return MhaGradInputs(
        q=q, k=k, v=v, actual_q=actual_q, actual_kv=actual_kv,
        q_seqlens=q_seqlens, kv_seqlens=kv_seqlens)


def attention_backward_golden(q, k, v, o_input, do_t, scale):
    scores = torch.matmul(q.float(), k.float().T) * scale
    p = torch.softmax(scores, dim=-1)
    d = (o_input.float() * do_t.float()).sum(dim=-1, keepdim=True)
    dp = torch.matmul(do_t.float(), v.float().T)
    ds = p * (dp - d)
    ds_half = ds.to(torch.bfloat16)
    p_half = p.to(torch.bfloat16)
    dk_fp32 = torch.matmul(ds_half.float().T, q.float()) * scale
    dk = dk_fp32.to(torch.float32)
    dv = torch.matmul(p_half.float().T, do_t.float()).to(torch.float32)
    dq_fp32 = torch.matmul(ds_half.float(), k.float()) * scale
    dq = dq_fp32.to(torch.float32)
    return AttentionBackwardOutput(dq, dk, dv)


def compute_l_m_o(q, k, v, scale):
    scores = torch.matmul(q.float(), k.float().T) * scale
    m = scores.max(dim=-1, keepdim=True)[0]
    p = torch.exp(scores - m)
    l_val = p.sum(dim=-1, keepdim=True)
    o = torch.matmul(p / l_val, v.float())
    return l_val, m, o.to(torch.bfloat16)


@dataclass
class _ResolveParamsOutputs:
    batch_size: Any
    num_heads: Any
    s1_size: Any
    s2_size: Any
    dim: Any
    q_seqlens: Any
    kv_seqlens: Any


def _resolve_params(batch_size, num_heads, s1_size, s2_size, dim, q_seqlens, kv_seqlens):
    if batch_size is None:
        batch_size = 1
    if num_heads is None:
        num_heads = NUM_HEADS_DEFAULT
    if s1_size is None:
        s1_size = 320
    if s2_size is None:
        s2_size = s1_size
    if dim is None:
        dim = HEAD_DIM_DEFAULT
    if q_seqlens is not None:
        batch_size = len(q_seqlens)
        s1_size = max(q_seqlens)
    if kv_seqlens is not None:
        assert len(kv_seqlens) == batch_size, "kv_seqlens must match batch_size"
        s2_size = max(kv_seqlens)
    return _ResolveParamsOutputs(batch_size, num_heads, s1_size, s2_size, dim, q_seqlens, kv_seqlens)


def _default_tile_config():
    return FlashAttentionGradTileShapeConfig(
        s1_tile=S1_TILE,
        s2_tile=S2_TILE,
        c_tile_mm=[[128, 128], [128, 256], [256, 256]],
        c_tile_dq=[[128, 128], [128, 256], [128, 128]],
        c_tile_dkv=[[256, 256], [64, 128], [128, 128]],
        v_tile_d=[64, 256],
        v_tile_q=[64, 128],
        v_tile_kv=[128, 128],
    )


def _setup_device():
    device_id = get_device_id()
    torch.npu.set_device(device_id)
    return f'npu:{device_id}', device_id


def _log_case(batch_size, num_heads, s1_size, s2_size, dim, hidden_dim, scale,
              tile_config, q_seqlens, kv_seqlens):
    logging.info("=" * 60)
    logging.info(f"Test Case: batch={batch_size}, heads={num_heads}, "
                 f"Q:s1_size={s1_size}, KV:s2_size={s2_size}, dim={dim}")
    if q_seqlens is not None or kv_seqlens is not None:
        logging.info(f"  varlen: q_seqlens={q_seqlens}, kv_seqlens={kv_seqlens}")
    logging.info(f"  hidden_dim={hidden_dim}, scale={scale:.6f}, "
                 f"s2_tile={tile_config.s2_tile}")
    logging.info("=" * 60)


@dataclass
class _PrecomputeLMOInputs:
    batch_size: Any
    num_heads: Any
    q: Any
    k: Any
    v: Any
    q_cumsum: Any
    kv_cumsum: Any
    l_out: Any
    m_out: Any
    o_out: Any
    scale: Any


def _precompute_l_m_o(inputs: _PrecomputeLMOInputs):
    for b in range(inputs.batch_size):
        q_off = inputs.q_cumsum[b]
        kv_off = inputs.kv_cumsum[b]
        sq = inputs.q_cumsum[b + 1] - q_off
        skv = inputs.kv_cumsum[b + 1] - kv_off
        for h in range(inputs.num_heads):
            l_h, m_h, o_h = compute_l_m_o(
                inputs.q[q_off: q_off + sq, h, :],
                inputs.k[kv_off: kv_off + skv, h, :],
                inputs.v[kv_off: kv_off + skv, h, :],
                inputs.scale)
            inputs.l_out[q_off: q_off + sq, h, :] = l_h
            inputs.m_out[q_off: q_off + sq, h, :] = m_h
            inputs.o_out[q_off: q_off + sq, h, :] = o_h


@dataclass
class _ComputeGoldenInputs:
    batch_size: Any
    num_heads: Any
    dim: Any
    q: Any
    k: Any
    v: Any
    o_out: Any
    do_t: Any
    q_cumsum: Any
    kv_cumsum: Any
    total_q: Any
    total_kv: Any
    device: Any
    scale: Any


def _compute_golden(inputs: _ComputeGoldenInputs):
    dq_golden = torch.empty(inputs.total_q, inputs.num_heads * inputs.dim, dtype=torch.float32, device=inputs.device)
    dk_golden = torch.empty(inputs.total_kv, inputs.num_heads * inputs.dim, dtype=torch.float32, device=inputs.device)
    dv_golden = torch.empty(inputs.total_kv, inputs.num_heads * inputs.dim, dtype=torch.float32, device=inputs.device)
    for b in range(inputs.batch_size):
        q_off = inputs.q_cumsum[b]
        kv_off = inputs.kv_cumsum[b]
        sq = inputs.q_cumsum[b + 1] - q_off
        skv = inputs.kv_cumsum[b + 1] - kv_off
        for h in range(inputs.num_heads):
            h_off = h * inputs.dim
            dq_g, dk_g, dv_g = attention_backward_golden(
                inputs.q[q_off: q_off + sq, h, :],
                inputs.k[kv_off: kv_off + skv, h, :],
                inputs.v[kv_off: kv_off + skv, h, :],
                inputs.o_out[q_off: q_off + sq, h, :],
                inputs.do_t[q_off: q_off + sq, h, :],
                inputs.scale)
            dq_golden[q_off: q_off + sq, h_off: h_off + inputs.dim] = dq_g
            dk_golden[kv_off: kv_off + skv, h_off: h_off + inputs.dim] = dk_g
            dv_golden[kv_off: kv_off + skv, h_off: h_off + inputs.dim] = dv_g
    return dq_golden, dk_golden, dv_golden


@dataclass
class _RunKernelInputs:
    q: Any
    k: Any
    v: Any
    o_out: Any
    do_t: Any
    l_out: Any
    m_out: Any
    dq_out: Any
    dk_out: Any
    dv_out: Any
    actual_q: Any
    actual_kv: Any
    tile_config: Any


def _run_kernel(inputs: _RunKernelInputs):
    logging.info("  Running kernel...")
    import time
    start_time = time.time()
    flash_attention_mha_grad_kernel_impl(
        inputs.q, inputs.k, inputs.v, inputs.o_out, inputs.do_t,
        inputs.l_out, inputs.m_out, inputs.dq_out, inputs.dk_out, inputs.dv_out,
        inputs.actual_q, inputs.actual_kv, inputs.tile_config)
    elapsed = time.time() - start_time
    logging.info(f"  Kernel time: {elapsed * 1000:.2f} ms")


def _verify_precision(dq_out, dk_out, dv_out, dq_golden, dk_golden, dv_golden):
    rtol = 0.005
    atol = 0.000025
    passed = True
    for name, npu_tensor, golden_tensor in [
        ("dQ", dq_out, dq_golden),
        ("dK", dk_out, dk_golden),
        ("dV", dv_out, dv_golden),
    ]:
        npu_np = npu_tensor.float().cpu().numpy()
        golden_np = golden_tensor.float().cpu().numpy()
        max_diff = np.abs(npu_np - golden_np).max()
        try:
            assert_allclose(npu_np, golden_np, rtol=rtol, atol=atol)
            logging.info(f"  {name}: PASSED (max_diff={max_diff:.6f}, rtol={rtol}, atol={atol})")
        except AssertionError as e:
            logging.info(f"  {name}: FAILED (max_diff={max_diff:.6f}, rtol={rtol}, atol={atol})")
            logging.info(f"    {e}")
            passed = False
    logging.info(f"  {'PASSED' if passed else 'FAILED'}")
    return passed


def run_test(batch_size=None, num_heads=None, s1_size=None,
             s2_size=None, dim=None, tile_config=None,
             q_seqlens=None, kv_seqlens=None):
    """运行单个测试用例: 构造输入 → 调用 kernel → 与 golden 对比。"""
    device, device_id = _setup_device()
    if device is None:
        return None

    params = _resolve_params(batch_size, num_heads, s1_size, s2_size, dim, q_seqlens, kv_seqlens)
    batch_size = params.batch_size
    num_heads = params.num_heads
    s1_size = params.s1_size
    s2_size = params.s2_size
    dim = params.dim
    q_seqlens = params.q_seqlens
    kv_seqlens = params.kv_seqlens
    if tile_config is None:
        tile_config = _default_tile_config()

    hidden_dim = num_heads * dim
    scale = 1.0 / (dim ** 0.5)
    _log_case(batch_size, num_heads, s1_size, s2_size, dim, hidden_dim, scale,
              tile_config, q_seqlens, kv_seqlens)

    torch.manual_seed(2026)
    inputs = create_inputs(
        batch_size, s1_size, s2_size, num_heads, dim, device,
        q_seqlens=q_seqlens, kv_seqlens=kv_seqlens)
    q, k, v = inputs.q, inputs.k, inputs.v
    actual_q, actual_kv = inputs.actual_q, inputs.actual_kv
    q_seqlens, kv_seqlens = inputs.q_seqlens, inputs.kv_seqlens

    q_cumsum = actual_q.cpu().tolist()
    kv_cumsum = actual_kv.cpu().tolist()
    total_q = q_cumsum[-1]
    total_kv = kv_cumsum[-1]

    do_t = torch.randn(total_q, num_heads, dim, dtype=torch.bfloat16, device=device) * 0.1
    l_out = torch.empty(total_q, num_heads, 1, dtype=torch.float32, device=device)
    m_out = torch.empty(total_q, num_heads, 1, dtype=torch.float32, device=device)
    o_out = torch.empty(total_q, num_heads, dim, dtype=torch.bfloat16, device=device)

    _precompute_l_m_o(_PrecomputeLMOInputs(
        batch_size=batch_size, num_heads=num_heads, q=q, k=k, v=v,
        q_cumsum=q_cumsum, kv_cumsum=kv_cumsum,
        l_out=l_out, m_out=m_out, o_out=o_out, scale=scale))

    dq_out = torch.zeros(total_q, hidden_dim, dtype=torch.float32, device=device)
    dk_out = torch.zeros(total_kv, hidden_dim, dtype=torch.float32, device=device)
    dv_out = torch.zeros(total_kv, hidden_dim, dtype=torch.float32, device=device)

    dq_golden, dk_golden, dv_golden = _compute_golden(_ComputeGoldenInputs(
        batch_size=batch_size, num_heads=num_heads, dim=dim, q=q, k=k, v=v,
        o_out=o_out, do_t=do_t,
        q_cumsum=q_cumsum, kv_cumsum=kv_cumsum,
        total_q=total_q, total_kv=total_kv, device=device, scale=scale))

    _run_kernel(_RunKernelInputs(
        q=q, k=k, v=v, o_out=o_out, do_t=do_t, l_out=l_out, m_out=m_out,
        dq_out=dq_out, dk_out=dk_out, dv_out=dv_out,
        actual_q=actual_q, actual_kv=actual_kv, tile_config=tile_config))
    return _verify_precision(dq_out, dk_out, dv_out, dq_golden, dk_golden, dv_golden)


########################################################################
# 独立测试用例
########################################################################


@pytest.mark.soc("950")
def test_01():
    """ 用例规格信息：batch=8, heads=8, s1=320, s2=320, dim=64"""
    return run_test(batch_size=8, num_heads=8, s1_size=320, s2_size=320, dim=64)


@pytest.mark.skip(reason="large test case")
def test_02():
    """ 用例规格信息：batch=1, heads=8, s1=4096, s2=4096, dim=128"""
    return run_test(batch_size=1, num_heads=8, s1_size=4096, s2_size=4096, dim=128)


@pytest.mark.soc("950")
@pytest.mark.skip(reason="large test case")
def test_03():
    """ 用例规格信息：batch=8, heads=16, s1=32, s2=32, dim=32"""
    return run_test(batch_size=8, num_heads=16, s1_size=32, s2_size=32, dim=32)


@pytest.mark.soc("950")
@pytest.mark.skip(reason="large test case")
def test_04():
    """ 用例规格信息：batch=8, heads=16, s1=64, s2=64, dim=32"""
    return run_test(batch_size=8, num_heads=16, s1_size=64, s2_size=64, dim=32)


@pytest.mark.soc("950")
@pytest.mark.skip(reason="large test case")
def test_05():
    """ 用例规格信息：batch=8, heads=8, s1=32, s2=32, dim=64"""
    return run_test(batch_size=8, num_heads=8, s1_size=32, s2_size=32, dim=64)


@pytest.mark.skip("950")
@pytest.mark.skip(reason="large test case")
def test_06():
    """ 用例规格信息：batch=8, heads=4, s1=64, s2=64, dim=128"""
    return run_test(batch_size=8, num_heads=4, s1_size=64, s2_size=64, dim=128)


@pytest.mark.skip(reason="large test case")
def test_07_varlen_small_seq():
    """ 用例规格信息：batch=4, heads=8, q_seqlens=[64,128,192,256], kv_seqlens=[64,128,192,256], dim=64 """
    return run_test(num_heads=8, dim=64,
                    q_seqlens=[64, 128, 192, 256],
                    kv_seqlens=[64, 128, 192, 256])


@pytest.mark.soc("950")
@pytest.mark.skip(reason="large test case")
def test_08_varlen_long_seq():
    """ 用例规格信息: batch=2, heads=8, q_seqlens=[384,512], kv_seqlens=[384,512], dim=64 """
    return run_test(num_heads=8, dim=64,
                    q_seqlens=[384, 512],
                    kv_seqlens=[384, 512])


@pytest.mark.soc("950")
@pytest.mark.skip(reason="large test case")
def test_09_varlen_cross_attn():
    """ 用例规格信息：batch=3, heads=8, q_seqlens=[128,64,192], kv_seqlens=[96,256,128], dim=64 """
    return run_test(num_heads=8, dim=64,
                    q_seqlens=[128, 64, 192],
                    kv_seqlens=[96, 256, 128])


def main():
    logging.info("Flash Attention Backward (3-loop, KV tiling)")
    test_funcs = [
        test_01, test_02, test_03, test_04, test_05, test_06,
        test_07_varlen_small_seq, test_08_varlen_long_seq, test_09_varlen_cross_attn,
    ]
    results = []
    for fn in test_funcs:
        try:
            passed = fn()
            results.append((fn.__name__, fn.__doc__, passed))
        except Exception as e:
            logging.info(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((fn.__name__, fn.__doc__, False))
    logging.info("=" * 60)
    logging.info("Summary:")
    logging.info("-" * 60)
    all_passed = True
    for name, desc, passed in results:
        status = "PASSED" if passed else "FAILED"
        logging.info(f"  {name}: {desc}  => {status}")
        if not passed:
            all_passed = False
    logging.info(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
