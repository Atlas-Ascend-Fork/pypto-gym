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
"""AttenEpilogue delivery-form test (kernel: atten_epilogue/op_kernel).

Same acceptance criterion as the source operator it was ported from: bit-exact
equality (rtol=0, atol=0) against the torch_npu native reference, plus an
independent CPU oracle.  Runs unchanged in both modes:

    pytest ops/pro/deepseek_v4_1/atten_epilogue/tests/test_atten_epilogue.py --mode jit
    pytest ops/pro/deepseek_v4_1/atten_epilogue/tests/test_atten_epilogue.py --mode aclnn

The aclnn mode exercises the built-in binary path (host tiling -> pre-compiled
kernel .o -> framework workspace); jit-only cases additionally inspect the GM
payloads inside that workspace and the Python host contract.
"""

import ast
import inspect
import logging
import os

import pytest
import torch

from op_caller import get_mode, get_op_caller
from atten_epilogue_call import make_output, run_epilogue
from atten_epilogue_golden_cpu import atten_epilogue_golden_cpu, inverse_rope, quantize_mx
from atten_epilogue_kernel_consts import (
    GEMM1_K, GEMM2_K, GROUPS, GROUP_N, HEADS, HEAD_DIM, OUT_FEATURES, PROFILE_COUNT,
    WORKSPACE_BYTES_PER_ROW, WIDE_CUBE_CORES, padded_tokens, workspace_bytes,
)
from atten_epilogue_precision import assert_exact, assert_payload

atten_epilogue = get_op_caller("atten_epilogue")

pytestmark = pytest.mark.soc("950")

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# T values covering the three capacity profiles (T<=64 / <=128 / >128), the 16-row
# padding granularity and the scale factors that drive the MX saturating path.
SUPPLIED_CASES = [(1, 1.0), (33, 1.0), (72, 1.0), (129, 1.0), (193, 1.0),
                  (255, 1.0), (72, 0.0), (256, 1e-37)]
# Full boundary sweep of the source operator (ops/pypto_pro/deepseek_v4_1/attn_epilogue)
# plus 129: 16-row padding edges, the three profile switch points (64/128) and the
# large-T tail where the flat GM workspace grows to 42240 * padded bytes.
BOUNDARY_TOKENS = [2, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 129, 192, 256,
                   257, 272, 511, 512, 513, 769, 1024, 1793, 2048, 2049, 4096, 16384]
# Repeat-call sweep (the source operator's test_public_tp1 token list).
REPEAT_TOKENS = [1, 72, 256, 257, 2048]


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


@pytest.mark.parametrize("t,scale", SUPPLIED_CASES)
def test_supplied_native_exact(weights, t, scale):
    """Bit-exact parity with the torch_npu native reference (upstream acceptance).

    The output is a T-row slice of a sentinel-filled padded storage with 32-element
    guard bands on both sides, so the same case also proves the kernel writes
    exactly the T valid rows: no overrun into the guards, no write into the
    16-row padding tail (the delivery form derives every GM address from the
    framework workspace, so a wrong pl.addptr offset would show up here).
    """
    torch.npu.set_device(ST_DEVICE_ID)
    args = arguments(weights, t, scale)
    expected = native_reference(*args)[0]
    padded = padded_tokens(t)
    storage = torch.full((padded * OUT_FEATURES + 64,), 19.0, dtype=torch.bfloat16,
                         device=args[0].device)
    full = storage[32:-32].view(padded, OUT_FEATURES)
    out = run_epilogue(atten_epilogue, args, out=full[:t])
    torch.npu.synchronize()
    assert out.shape == (t, OUT_FEATURES) and out.dtype == torch.bfloat16
    assert out.is_contiguous()
    assert_exact(out, expected, f"native T={t} scale={scale}")
    assert bool((storage[:32] == 19).all()) and bool((storage[-32:] == 19).all()), \
        "guard band overwritten"
    assert bool((full[t:] == 19).all()), "padded rows were written"
    logging.info("  [PASS] native exact T=%d scale=%g", t, scale)


@pytest.mark.parametrize("t", REPEAT_TOKENS)
def test_repeat_call_exact(weights, t):
    """Two back-to-back calls (second on a negated activation) are both bit-exact.

    Exercises workspace/payload reuse across launches: the aq/asc/yq/ys slices are
    fully rewritten every call, so a stale-payload leak would break the second one.
    """
    torch.npu.set_device(ST_DEVICE_ID)
    args = arguments(weights, t)
    expected = [native_reference(*args)[0], native_reference(-args[0], *args[1:])[0]]
    first = run_epilogue(atten_epilogue, args)
    torch.npu.synchronize()
    assert_exact(first, expected[0], f"repeat T={t} call#1")
    args[0].neg_()
    second = run_epilogue(atten_epilogue, args)
    torch.npu.synchronize()
    assert second.data_ptr() != first.data_ptr()
    assert_exact(second, expected[1], f"repeat T={t} call#2")
    assert_exact(first, expected[0], f"repeat T={t} call#1 clobbered")


@pytest.mark.parametrize("t", BOUNDARY_TOKENS)
def test_token_boundaries(weights, t):
    """Row-padding and profile-switch boundaries stay bit-exact."""
    torch.npu.set_device(ST_DEVICE_ID)
    args = arguments(weights, t)
    assert_exact(run_epilogue(atten_epilogue, args), native_reference(*args)[0],
                 f"boundary T={t}")


def test_dynamic_token_range(weights):
    """Every T in [1, 256] is bit-exact: the full sweep across all three profiles."""
    torch.npu.set_device(ST_DEVICE_ID)
    for t in range(1, 257):
        args = arguments(weights, t)
        assert_exact(run_epilogue(atten_epilogue, args), native_reference(*args)[0],
                     f"runtime T={t}")


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
    assert_exact(run_epilogue(atten_epilogue, npu_args), expected, "independent CPU projection")


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


@pytest.mark.jit_only
@pytest.mark.parametrize("t,scale", SUPPLIED_CASES)
def test_gm_payloads_exact(weights, t, scale):
    """jit only: the workspace payloads (aq/asc/yq/ys) are byte-exact vs the native reference.

    The built-in path hides the workspace inside the framework allocation, so this
    intermediate check drives launch_kernel() with a caller-owned buffer instead.
    It is the strongest guard on the pl.addptr carve-out: a wrong offset or size
    shows up as a payload mismatch even when `out` still looks plausible.
    """
    import atten_epilogue as op
    torch.npu.set_device(ST_DEVICE_ID)
    args = arguments(weights, t, scale)
    expected = native_reference(*args)
    padded = padded_tokens(t)
    workspace = torch.full((workspace_bytes(t),), 7, dtype=torch.uint8, device=args[0].device)
    out = make_output(args[0])
    op.launch_kernel(*args, workspace, out)
    torch.npu.synchronize()
    assert_exact(out, expected[0], f"payloads T={t} scale={scale} output")
    aq = workspace[:op.ASC_OFFSET_PER_ROW * padded].view(GROUPS, padded, GEMM1_K)[:, :t]
    asc = workspace[op.ASC_OFFSET_PER_ROW * padded:op.YQ_OFFSET_PER_ROW * padded].view(
        GROUPS, padded, GEMM1_K // 32)[:, :t]
    yq = workspace[op.YQ_OFFSET_PER_ROW * padded:op.YS_OFFSET_PER_ROW * padded].view(
        padded, GEMM2_K)[:t]
    ys = workspace[op.YS_OFFSET_PER_ROW * padded:].view(padded, GEMM2_K // 32)[:t]
    assert_payload(aq, expected[1].permute(1, 0, 2), f"T={t} activation bytes")
    assert_payload(asc, expected[2].permute(1, 0, 2), f"T={t} activation scales")
    assert_payload(yq, expected[3], f"T={t} Y bytes")
    assert_payload(ys, expected[4], f"T={t} Y scales")


@pytest.mark.jit_only
def test_workspace_cache_reuse(weights):
    """jit only: repeated calls reuse one cached workspace and stay bit-exact.

    Note on graph capture: the delivery form passes a tiling *struct*, and the
    PyPTO jit launcher serializes it into a fresh device buffer per launch
    (_pack_tiling_arg).  That H2D copy is rejected inside a GLOBAL-mode capture,
    so jit-mode graph capture is not available for tiling-struct kernels (same as
    flex_attention).  The built-in aclnn path builds the tiling buffer in the
    framework and has no such copy.
    """
    import atten_epilogue as op
    torch.npu.set_device(ST_DEVICE_ID)
    t = 72
    args = arguments(weights, t)
    expected = [native_reference(*args)[0], native_reference(-args[0], *args[1:])[0]]
    out = make_output(args[0])
    op.atten_epilogue(*args, out)
    torch.npu.synchronize()
    assert_exact(out, expected[0])
    cache_ids = {key: value.data_ptr() for key, value in op.WORKSPACE_CACHE.items()}
    args[0].neg_()
    other = make_output(args[0])
    op.atten_epilogue(*args, other)
    torch.npu.synchronize()
    assert other.data_ptr() != out.data_ptr()
    assert_exact(other, expected[1], "second call")
    assert cache_ids == {key: value.data_ptr() for key, value in op.WORKSPACE_CACHE.items()}


@pytest.mark.jit_only
def test_bounded_tiling_profiles():
    """jit only: the tiling key schema stays enumerable and the selector stays inside it."""
    from pypto_pro.runtime.tilingkey import TilingKeySchema
    import atten_epilogue as op
    schema = TilingKeySchema(op.EpilogueKey)
    profiles = set(schema.enumerate_valid())
    assert len(profiles) == PROFILE_COUNT
    combined = set()
    for cores in (28, WIDE_CUBE_CORES):
        selected = {tuple(op.tiling_key_for(t, cores)[name] for name in schema.field_names())
                    for t in [*range(1, 257), *BOUNDARY_TOKENS]}
        assert len(selected) == PROFILE_COUNT // 2
        assert selected <= profiles
        combined.update(selected)
    assert combined == profiles


@pytest.mark.jit_only
def test_host_contract_validation(weights):
    """jit only: the Python host entry rejects shape/dtype/device/contiguity faults."""
    import torch_npu
    import atten_epilogue as op
    torch.npu.set_device(ST_DEVICE_ID)
    args = arguments(weights, 72)
    full = list(args) + [make_output(args[0])]
    faults = [(0, args[0].reshape(72 * HEADS, HEAD_DIM)),
              (1, weights[0].reshape(GROUPS * GROUP_N, GEMM1_K)),
              (3, weights[2].view(torch.uint8)),
              (6, args[6].cpu()),
              (6, args[6][:, ::2]),
              (6, args[6].to(torch.float16)),
              (7, make_output(args[0], rows=64)),
              (2, torch_npu.npu_format_cast(weights[1], 29))]
    for index, value in faults:
        bad = list(full)
        bad[index] = value
        with pytest.raises((ValueError, TypeError)):
            op.atten_epilogue(*bad)
    empty_o = torch.empty((0, HEADS, HEAD_DIM), dtype=torch.bfloat16, device=args[0].device)
    with pytest.raises(ValueError):
        op.atten_epilogue(empty_o, *args[1:], make_output(empty_o))


def _pl_attribute(node):
    """Return True for a `pl.<attr>` access in the kernel source AST."""
    if not isinstance(node, ast.Attribute):
        return False
    if not isinstance(node.value, ast.Name):
        return False
    return node.value.id == "pl"


@pytest.mark.jit_only
def test_primitive_only_implementation():
    """jit only: the kernel goes through the tile/primitive API only."""
    import atten_epilogue as op
    tree = ast.parse(inspect.getsource(op))
    accesses = {node.attr for node in ast.walk(tree) if _pl_attribute(node)}
    assert not accesses.intersection({"gemm", "gemm_mx", "mem_space", "allocate_tile_group"})
    assert {"make_tile_group", "matmul_mx", "matmul_mx_acc"} <= accesses


def test_mode_is_dual():
    """Sanity: the shared entry resolved for the running mode (informational)."""
    logging.info("  [INFO] test mode = %s, entry = %r", get_mode(), atten_epilogue)
