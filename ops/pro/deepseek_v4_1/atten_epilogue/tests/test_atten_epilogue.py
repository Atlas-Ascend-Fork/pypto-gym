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
"""AttenEpilogue precompiled-Torch-binding tests (kernel: atten_epilogue/op_kernel).

Same acceptance criterion as the binary it exercises: bit-exact equality
(rtol=0, atol=0) against the torch_npu native reference, plus an independent
CPU oracle.  Cases are split across the two formal entries
(``pypto_gym.atten_epilogue`` and ``torch.ops.pypto_gym.atten_epilogue``);
every case performs exactly one operator call per invocation and the output is
allocated inside the binding.  The entries are not controlled by ``--mode``.

Requires the .run package with precompiled Torch bindings installed into the
active CANN environment (``build_ci.py --pkg-type run --stest`` does this
automatically):

    pytest ops/pro/deepseek_v4_1/atten_epilogue/tests -q
"""

import logging
import math
import os

import pytest
import torch

from atten_epilogue_golden_cpu import atten_epilogue_golden_cpu, inverse_rope, quantize_mx
from atten_epilogue_kernel_consts import (
    GEMM1_K, GEMM2_K, GROUPS, GROUP_N, HEADS, HEAD_DIM, OUT_FEATURES,
    WORKSPACE_BYTES_PER_ROW, padded_tokens, workspace_bytes,
)
from atten_epilogue_precision import assert_exact, assert_payload

import pypto_gym

# pypto_gym is lazy; register torch.ops before calling it directly.
pypto_gym.load()

pytestmark = pytest.mark.soc("950")

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def entry(api):
    """Return the formal entry selected for one case."""
    if api == "python":
        return pypto_gym.atten_epilogue
    return torch.ops.pypto_gym.atten_epilogue


# T values covering the three capacity profiles (T<=64 / <=128 / >128), the
# 16-row padding granularity and the scale factors that drive the MX saturating
# path.  `api` alternates between the two formal Torch entries.
SUPPLIED_CASES = [(1, 1.0, "python"), (33, 1.0, "torch"), (72, 1.0, "python"),
                  (129, 1.0, "torch"), (193, 1.0, "python"), (255, 1.0, "torch"),
                  (72, 0.0, "python"), (256, 1e-37, "torch")]
# Deployment scenarios as (batch, seq) pairs: T = B * S. B=16/S=6 (T=96) is a
# mid-profile point (64 < T <= 128, already 16-row aligned) that none of the
# token lists names explicitly.
BATCH_SEQ_CASES = [(16, 6, "torch")]
# Full boundary sweep of the source operator plus 129: 16-row padding edges,
# the three profile switch points (64/128) and the large-T tail where the flat
# GM workspace grows to 42240 * padded bytes.
BOUNDARY_TOKENS = [2, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 129, 192, 256,
                   257, 272, 383, 384, 385, 511, 512, 513, 769, 1023, 1024, 1793, 2048, 2049, 4096, 16384]
BOUNDARY_CASES = [(t, "python" if i % 2 == 0 else "torch")
                  for i, t in enumerate(BOUNDARY_TOKENS)]
# Repeat-call sweep (the source operator's test_public_tp1 token list).
REPEAT_CASES = [(1, "python"), (72, "torch"), (256, "python"), (257, "torch"), (2048, "python")]


def input_fixture(t, scale=1.0):
    gen = torch.Generator().manual_seed(2026)
    return (torch.randn(t, HEADS, HEAD_DIM, generator=gen) * (0.3 * scale)).to(torch.bfloat16)


def rope_tables(t):
    j = torch.arange(0, HEADS, 2, dtype=torch.float32)
    angles = torch.outer(torch.arange(t, dtype=torch.float32), 1.0 / (10000.0 ** (j / HEADS)))
    return angles.sin().repeat_interleave(2, -1), angles.cos().repeat_interleave(2, -1)


def make_weights():
    import torch_npu
    gen = torch.Generator().manual_seed(2026)
    wa = (torch.randn(GROUPS, GROUP_N, GEMM1_K, generator=gen) * 0.3).to(torch.bfloat16)
    wb = (torch.randn(OUT_FEATURES, GEMM2_K, generator=gen) * 0.2).to(torch.bfloat16)
    aq, asc = torch_npu.npu_dynamic_mx_quant(wa.reshape(GROUPS * GROUP_N, GEMM1_K).npu(), dst_type=292)
    bq, bsc = torch_npu.npu_dynamic_mx_quant(wb.npu(), dst_type=292)
    e8 = torch.float8_e8m0fnu
    return (aq.reshape(GROUPS, GROUP_N, GEMM1_K), bq,
            asc.view(e8).reshape(GROUPS, GROUP_N, GEMM1_K // 64, 2), bsc.view(e8))


def native_reference(o, woa, wob, sa, sb, sin, cos):
    """Use the same native arithmetic and intermediate rounding as upstream."""
    import torch_npu
    t = o.shape[0]
    rotated = inverse_rope(o.cpu(), sin.cpu(), cos.cpu()).reshape(t, GROUPS, GEMM1_K)
    aq, asc = torch_npu.npu_dynamic_mx_quant(rotated.npu(), dst_type=292)
    y = torch_npu.npu_transpose_quant_batchmatmul(
        aq, woa, 15, x1_scale=asc.view(torch.float8_e8m0fnu), x2_scale=sa,
        group_sizes=[1, 1, 32], perm_x1=[1, 0, 2], perm_x2=[0, 2, 1], perm_y=[1, 0, 2],
    ).reshape(t, GEMM2_K)
    yq, ys = torch_npu.npu_dynamic_mx_quant(y, dst_type=292)
    out = torch_npu.npu_quant_matmul(
        yq, wob.t(), sb.transpose(0, 1), pertoken_scale=ys.view(torch.float8_e8m0fnu),
        output_dtype=torch.bfloat16, group_sizes=[1, 1, 32],
    )
    return (out.cpu(), aq.view(torch.uint8).cpu(),
            asc.view(torch.uint8).reshape(t, GROUPS, GEMM1_K // 32).cpu(),
            yq.view(torch.uint8).cpu(), ys.view(torch.uint8).reshape(t, GEMM2_K // 32).cpu())


@pytest.fixture(scope="module")
def weights():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend 950 NPU required")
    torch.npu.set_device(ST_DEVICE_ID)
    return make_weights()


def arguments(weights, t, scale=1.0):
    x = input_fixture(t, scale)
    sin, cos = rope_tables(t)
    return (x.to(ST_DEVICE), *weights, sin.to(ST_DEVICE), cos.to(ST_DEVICE))


def guarded(value):
    """Place one input inside sentinel-padded storage to catch stray writes."""
    shape = tuple(value.shape)
    base = torch.full((math.prod(shape) + 256,), 37, dtype=value.dtype,
                      device=value.device)
    view = base[128:-128].view(shape)
    view.copy_(value)
    return view, base


def check_guards(bases):
    for base in bases:
        assert bool((base[:128] == 37).all()), "prefix guard overwritten"
        assert bool((base[-128:] == 37).all()), "suffix guard overwritten"


def test_comparator_rejects_corruption():
    """CPU-only: the exact comparator must not silently accept a perturbed result."""
    a = torch.ones((2, 3), dtype=torch.bfloat16)
    b = a.clone()
    b[1, 2] += 1
    with pytest.raises(AssertionError):
        assert_exact(a, b)
    with pytest.raises(AssertionError):
        assert_payload(torch.tensor([1, 2], dtype=torch.uint8), torch.tensor([1, 3], dtype=torch.uint8))


def test_workspace_budget_covers_payloads():
    """CPU-only: the flat workspace budget must equal the four GM payload sizes.

    Guards the three-way contract between the host tiling (bytes it reports), the
    kernel (pl.addptr offsets it carves) and the payload shapes both derive from.
    """
    per_row = GROUPS * GEMM1_K + GROUPS * GEMM1_K // 32 + GEMM2_K + GEMM2_K // 32
    assert per_row == WORKSPACE_BYTES_PER_ROW
    for t in (1, 16, 72, 256, 2048):
        padded = padded_tokens(t)
        assert padded % 16 == 0 and padded >= t
        assert workspace_bytes(t) == per_row * padded
        # every payload offset stays 512-byte aligned (MTE burst granularity)
        for offset in (0, GROUPS * GEMM1_K, GROUPS * GEMM1_K + GROUPS * GEMM1_K // 32):
            assert offset * padded % 512 == 0


@pytest.mark.parametrize("t,scale,api", SUPPLIED_CASES)
def test_supplied_native_exact(weights, t, scale, api):
    """Bit-exact parity with the torch_npu native reference (upstream acceptance).

    Exactly one formal-entry call per case; the binding allocates the output.
    The per-token inputs sit inside sentinel-guarded storage, so the same case
    proves the kernel leaves every input untouched (no overrun into the guards).
    """
    torch.npu.set_device(ST_DEVICE_ID)
    args = arguments(weights, t, scale)
    expected = native_reference(*args)[0]
    guarded_pairs = [guarded(value) for value in (args[0], args[5], args[6])]
    call_args = (guarded_pairs[0][0], *args[1:5],
                 guarded_pairs[1][0], guarded_pairs[2][0])
    out = entry(api)(*call_args)
    torch.npu.synchronize()
    assert out.shape == (t, OUT_FEATURES) and out.dtype == torch.bfloat16
    assert out.is_contiguous()
    assert_exact(out, expected, f"native T={t} scale={scale} api={api}")
    for (value, _), original in zip(guarded_pairs, (args[0], args[5], args[6])):
        torch.testing.assert_close(value.cpu(), original.cpu(), rtol=0, atol=0)
    check_guards([base for _, base in guarded_pairs])
    logging.info("  [PASS] native exact T=%d scale=%g api=%s", t, scale, api)


@pytest.mark.parametrize("b,s,api", BATCH_SEQ_CASES)
def test_batch_seq_scenario(weights, b, s, api):
    """B=16/S=6 (T=96) deployment scenario stays bit-exact on a formal entry."""
    torch.npu.set_device(ST_DEVICE_ID)
    t = b * s
    args = arguments(weights, t)
    expected = native_reference(*args)[0]
    out = entry(api)(*args)
    torch.npu.synchronize()
    assert out.shape == (t, OUT_FEATURES) and out.dtype == torch.bfloat16
    assert_exact(out, expected, f"scenario B={b} S={s} T={t} api={api}")


@pytest.mark.parametrize("t,api", REPEAT_CASES)
def test_repeat_call_exact(weights, t, api):
    """Two back-to-back calls (second on a negated activation) are both bit-exact.

    Exercises binding/workspace reuse across launches: every call allocates a
    fresh output and fully rewrites the internal payloads, so a stale-payload
    leak would break the second call.
    """
    torch.npu.set_device(ST_DEVICE_ID)
    args = arguments(weights, t)
    expected = [native_reference(*args)[0], native_reference(-args[0], *args[1:])[0]]
    run = entry(api)
    first = run(*args)
    torch.npu.synchronize()
    assert_exact(first, expected[0], f"repeat T={t} call#1")
    args[0].neg_()
    second = run(*args)
    torch.npu.synchronize()
    assert second.data_ptr() != first.data_ptr()
    assert_exact(second, expected[1], f"repeat T={t} call#2")
    assert_exact(first, expected[0], f"repeat T={t} call#1 clobbered")


@pytest.mark.parametrize("t,api", BOUNDARY_CASES)
def test_token_boundaries(weights, t, api):
    """Row-padding and profile-switch boundaries stay bit-exact."""
    torch.npu.set_device(ST_DEVICE_ID)
    args = arguments(weights, t)
    assert_exact(entry(api)(*args), native_reference(*args)[0],
                 f"boundary T={t} api={api}")


def test_dynamic_token_range(weights):
    """Every T in [1, 256] is bit-exact: the full sweep across all three profiles."""
    torch.npu.set_device(ST_DEVICE_ID)
    run = entry("python")
    for t in range(1, 257):
        args = arguments(weights, t)
        assert_exact(run(*args), native_reference(*args)[0], f"runtime T={t}")


def test_independent_cpu_projection():
    """Bit-exact parity with the independent FP64 CPU oracle (no torch_npu ops)."""
    torch.npu.set_device(ST_DEVICE_ID)
    x = ((torch.arange(HEADS * HEAD_DIM).float() % 127 - 63) / 128).reshape(
        1, HEADS, HEAD_DIM).bfloat16()
    sin, cos = rope_tables(1)
    codes = torch.tensor([1.0, -1.0]).to(torch.float8_e4m3fn).view(torch.uint8)
    wa = torch.zeros((GROUPS, GROUP_N, GEMM1_K), dtype=torch.uint8)
    rows = torch.arange(GROUP_N)
    for group in range(GROUPS):
        wa[group, rows, (rows * 3 + group * 17) % GEMM1_K] = codes[group % 2]
    wb = torch.zeros((OUT_FEATURES, GEMM2_K), dtype=torch.uint8)
    rows = torch.arange(OUT_FEATURES)
    wb[rows, (rows * 37 + 13) % GEMM2_K] = codes[rows % 2]
    wa = wa.view(torch.float8_e4m3fn)
    wb = wb.view(torch.float8_e4m3fn)
    sa = torch.full((GROUPS, GROUP_N, GEMM1_K // 64, 2), 127,
                    dtype=torch.uint8).view(torch.float8_e8m0fnu)
    sb = torch.full((OUT_FEATURES, GEMM2_K // 64, 2), 127,
                    dtype=torch.uint8).view(torch.float8_e8m0fnu)
    cpu_args = (x, wa, wb, sa, sb, sin, cos)
    expected = atten_epilogue_golden_cpu(*cpu_args)
    npu_args = tuple(v.to(ST_DEVICE) for v in cpu_args)
    assert_exact(entry("torch")(*npu_args), expected, "independent CPU projection")


def test_cpu_quantization_matches_native():
    """The CPU MX quantizer used by the oracle matches the native quantizer byte-for-byte."""
    import torch_npu
    torch.npu.set_device(ST_DEVICE_ID)
    for scale in (0.0, 1e-37, 1.0, 1000.0):
        x = input_fixture(1, scale).reshape(GROUPS, GEMM1_K)
        q, s = quantize_mx(x)
        nq, ns = torch_npu.npu_dynamic_mx_quant(x.npu(), dst_type=292)
        assert_payload(q, nq.cpu())
        assert_payload(s, ns.cpu())


def test_binding_rejects_invalid_inputs(weights):
    """The binding validates contract faults on the caller thread, before ACLNN work."""
    import torch_npu
    torch.npu.set_device(ST_DEVICE_ID)
    t = 72
    args = arguments(weights, t)
    run = torch.ops.pypto_gym.atten_epilogue
    faults = [
        (0, args[0].reshape(t * HEADS, HEAD_DIM)),          # wrong rank/shape
        (0, args[0].cpu()),                                  # host tensor
        (0, torch.empty((0, HEADS, HEAD_DIM), dtype=torch.bfloat16, device=ST_DEVICE)),
        (1, weights[0].reshape(GROUPS * GROUP_N, GEMM1_K)),  # wrong weight shape
        (3, weights[2].view(torch.uint8)),                   # wrong scale dtype
        (6, args[6][:, ::2]),                                # non-contiguous
        (6, args[6].to(torch.float16)),                      # wrong table dtype
        (2, torch_npu.npu_format_cast(weights[1], 29)),      # private NZ storage
    ]
    for index, value in faults:
        bad = list(args)
        bad[index] = value
        with pytest.raises(RuntimeError):
            run(*bad)
        torch.npu.synchronize()
