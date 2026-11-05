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

"""A5 AscendC/Pro MX full-quantization accuracy test; run directly, without pytest.

Run from the repository root. The target NPU is selected by the
TILE_FWK_DEVICE_ID environment variable, falling back to logical device 0:

    python ops/pypto_pro/experimental/quant_bmm_mxfp8/tests/test_quant_batch_matmul_mxfp8.py
    python ops/pypto_pro/experimental/quant_bmm_mxfp8/tests/test_quant_batch_matmul_mxfp8.py --self-test

Fixed shape: Batch=8, M=96, N=1024, K=4096 (the order is B,M,N,K).
Default: E5M2/E5M2 and E4M3FN/E4M3FN -> BF16, NN storage,
FP32 bias [N], [B,1,N] and no bias: six ND plus three E4M3FN NZ cases.
Input shapes are [B,M,K] and [B,K,N]. NZ cases use WeightNZ x2;
x1/x1_scale/x2_scale/bias remain ND. --formats nd restores ND only.

MX scales are E8M0 bytes: value = 2**(byte - 127), one per 32 K values.
Packed scale shapes:
  x1_scale [B,M,ceil(K/64),2], x2_scale [B,ceil(K/64),N,2].
The last axis selects the first/second group of 32 inside a group of 64.

Golden independently decodes actual FP8/E8M0 bytes on CPU, dequantizes
in FP64, performs FP64 matmul, adds FP32 bias, then rounds once to BF16.
It never calls an NPU matmul/quantizer to produce the reference.
The data tests the operator on already quantized inputs, rather than
comparing to the pre-quantization FP32 source (which includes quantization error).

References in this operator directory: README.md, DESIGN.md and RESULTS.md.
The CPU FP64 golden and the all-elements comparator are imported from
tests/assets/; neither calls an NPU matmul or quantizer.
torch_npu API/mapping checked against:
https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_matmul.md
https://github.com/Ascend/op-plugin/blob/master/op_plugin/ops/opapi/QuantMatmulKernelNpuOpApi.cpp

Requirements on board: compatible A5 CANN + torch + torch_npu, numpy.
Pro uses the frozen 9589 nine-case tiling: BM/BN/BK=96/256/128,
KL1=256, scaleKL1=4096, four L1 data buffers and a 32-Cube launch.
The test does not calculate tiling or read platform configuration files.
Pro mode additionally requires pypto/pypto_pro and the
op_kernel/quant_batch_matmul_mxfp8.py implementation module. Every case feeds
the same device inputs to both implementations and compares AscendC/golden,
Pro/golden and Pro/AscendC with the unchanged tolerances below. BF16 bitwise
agreement is required by default; --no-require-bitwise disables that condition.
Native E8M0 tensors carry exact scale bits on recent torch versions. For ND,
older versions use uint8 plus explicit E8M0 dtype attributes as a fallback.
NZ cases format-cast only x2, check its ND/NZ round trip and retain ND scales.
pertoken_scale and group/output attributes express MX; scale dtype overrides
are only used with the uint8 fallback. Optional offset/y_scale and input
dtype overrides are omitted.
The default tolerances (rtol=0.008, atol=0.001) are configurable test
thresholds, not a claim about official CANN acceptance criteria. All elements
must pass; no allowed mismatch ratio. Exit 0=pass, 1=accuracy fail, 2=runtime error.
"""

import argparse
import gc
import logging
import os
from pathlib import Path
import sys
import traceback

import numpy as np

_HERE = Path(__file__).resolve().parent
_OP_DIR = _HERE.parent
for _path in (_HERE, _HERE / "assets", _OP_DIR / "op_kernel"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from quant_batch_matmul_mxfp8_golden_cpu import (
    decode_e8m0, fp8_table, golden_matmul, make_fp8_bytes, round_bf16,
)
from quant_batch_matmul_mxfp8_precision import compare

_LOG = logging.getLogger("quant_bmm_mxfp8")
# Own handler only: never configure the root logger. pypto_pro logs through
# `logging.info(...)` on the root, so a basicConfig here would surface the
# framework's internal INFO lines (kernel generation, env dumps) that were
# previously dropped for lack of a root handler. The formatter is message-only
# and the stream is stdout, so the emitted text is byte-identical to print().
_LOG.setLevel(logging.INFO)
if not _LOG.handlers:
    _stream = logging.StreamHandler(sys.stdout)
    _stream.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(_stream)
_LOG.propagate = False


BATCH, M, N, K = 8, 96, 1024, 4096
GROUP_K = 32

# Device selection follows the repository convention used by
# quant_sparse_lightning_indexer: read TILE_FWK_DEVICE_ID, default to logical 0.
DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))


def matrix_to_npu(raw, dtype, device, torch):
    return torch.from_numpy(np.ascontiguousarray(raw)).view(dtype).to(device)


def scale_to_npu(raw, device, torch, native_dtype=None):
    tensor = torch.from_numpy(np.ascontiguousarray(raw))
    if native_dtype is not None:
        # Reinterpret bytes, never numerically cast exponent codes to FP8.
        tensor = tensor.view(native_dtype)
    return tensor.to(device)


def nz_inputs(b, sb, x2_bits, scale_bits, torch, torch_npu):
    """Convert x2 to WeightNZ; preserve the original ND x2Scale tensor.

    The installed reference requires ND scale. Both kernels receive this
    identical WeightNZ + ND-scale combination, including the uint8 E8M0
    carrier path when native E8M0 is unavailable.
    """
    b_nz = torch_npu.npu_format_cast(b, 29)
    if torch_npu.get_npu_format(b_nz) != 29 or tuple(b_nz.shape) != x2_bits.shape:
        raise AssertionError("x2 must preserve logical shape and become WeightNZ=29")
    restored = torch_npu.npu_format_cast(b_nz, 2).view(torch.uint8).cpu().numpy()
    np.testing.assert_array_equal(restored, x2_bits, err_msg="x2 ND/WeightNZ round trip")
    scale_format = torch_npu.get_npu_format(sb)
    if scale_format not in (0, 2) or tuple(sb.shape) != scale_bits.shape:
        raise AssertionError("x2Scale must retain its original logical shape and ND format")
    np.testing.assert_array_equal(sb.view(torch.uint8).cpu().numpy(), scale_bits,
                                  err_msg="ND x2Scale payload")
    for name, tensor, fmt in (("x2", b_nz, "WeightNZ(29)"), ("scale", sb, f"ND({scale_format})")):
        _LOG.info(f"  {name}: format={fmt} logical={tuple(tensor.shape)} dtype={tensor.dtype} "
                  f"stride={tuple(tensor.stride())} offset={tensor.storage_offset()} "
                  f"storage_bytes={tensor.untyped_storage().nbytes()}")
    return b_nz, sb


def print_reference_library_paths():
    """Read loaded-library paths after reference failure; no NPU/API calls.

    Repository features do not prove the process loaded matching
    op-api/tiling libraries. /proc/self/maps records actual mapped libraries.
    """
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return
    try:
        paths = sorted({line.split()[-1] for line in maps.read_text().splitlines()
                        if any(name in line for name in
                               ("libopapi", "libcust_opapi", "libnnopbase", "liboptiling"))})
        _LOG.info("  Loaded reference libraries (actual process mappings):")
        for path in paths:
            _LOG.info(f"    {path}")
        if not paths:
            _LOG.info("    No matching path found in /proc/self/maps")
    except OSError as exc:
        _LOG.info(f"  Cannot read process library mappings: {exc}")


def check_nz_copy_layout():
    """CPU address-map check of native WeightNZ GM-to-L1 NZ windows.

    This tests packing/index arithmetic, not the NPU format-cast or codegen.
    Include StreamK nonzero starts, K=256 tail, and full scale-window scheduling.
    """
    rng = np.random.default_rng(61)
    for k, kl1, skl1, starts in ((5120, 512, 1024, (0, 1280, 2560, 3840)),
                                (1280, 256, 1280, (0,)),
                                (8192, 256, 4096, (0,)),
                                (4096, 256, 4096, (0,))):
        n, bn = 512, 256
        x = rng.integers(0, 256, (k, n), dtype=np.uint8)
        packed_b = x.reshape(k // 16, 16, n // 32, 32).transpose(2, 0, 1, 3).copy()
        for start in starts:
            local_k = 1280 if k == 5120 else k
            for n0 in (0, 256):
                for sk in range(0, local_k, skl1):
                    slen = min(skl1, local_k - sk)
                    for dk in range(sk, sk + slen, kl1):
                        dlen = min(kl1, local_k - dk)
                        kd = start + dk
                        # Native NZ offsets are logical (kd,n0); the physical
                        # N1/K1 window preserves complete 16x32 fractals.
                        received = packed_b[n0 // 32:(n0 + bn) // 32,
                                            kd // 16:(kd + dlen) // 16].copy()
                        element_offset = (n0 // 32) * k * 32 + (kd // 16) * 512
                        assert packed_b.reshape(-1)[element_offset] == x[kd, n0]
                        restored = received.transpose(1, 2, 0, 3).reshape(dlen, bn)
                        np.testing.assert_array_equal(restored, x[kd:kd + dlen, n0:n0 + bn])
    _LOG.info("CPU native WeightNZ layout self-test PASS")


def self_test():
    """CPU-only sanity checks for decoding, scale pairing, bias and comparator."""
    np.testing.assert_equal(fp8_table("e5m2")[[0, 60, 188, 1]], [0, 1, -1, 2 ** -16])
    np.testing.assert_equal(fp8_table("e4m3fn")[[0, 56, 184, 1, 126]], [0, 1, -1, 2 ** -9, 448])
    assert np.isinf(fp8_table("e5m2")[124])
    assert np.isnan(fp8_table("e4m3fn")[127])
    np.testing.assert_equal(decode_e8m0(np.array([0, 126, 127, 128, 254], np.uint8)),
                            [2.0 ** -127, 0.5, 1, 2, 2.0 ** 127])
    for kind, one in (("e5m2", 60), ("e4m3fn", 56)):
        a = np.full((2, 3, 64), one, np.uint8)
        b = np.full((2, 64, 4), one, np.uint8)
        sa = np.full((2, 3, 1, 2), 127, np.uint8)
        sb = np.full((2, 1, 4, 2), 127, np.uint8)
        sa[..., 1] = 128  # 1,2 across the two consecutive K groups
        sb[..., 0] = 129  # 4,1 across the two consecutive K groups
        sb[1] += 1  # Different batch must use different scales
        sb[:, :, 1, :] += 1  # Different column must use different scales
        sa[:, 2, :, :] += 1  # Different row must use different scales
        expected = np.full((2, 3, 4), 32 * 1 * 4 + 32 * 2 * 1, np.float64)
        expected[1] *= 2
        expected[:, :, 1] *= 2
        expected[:, 2] *= 2
        np.testing.assert_equal(golden_matmul(a, b, sa, sb, kind), expected)
        np.testing.assert_equal(round_bf16(golden_matmul(a, b, sa, sb, kind)), expected)
        for bias in (16 * np.arange(4, dtype=np.float32),
                     16 * np.arange(8, dtype=np.float32).reshape(2, 1, 4)):
            np.testing.assert_equal(round_bf16(golden_matmul(a, b, sa, sb, kind) + bias), expected + bias)
    # Halfway FP32 values must use even BF16 mantissas.
    np.testing.assert_equal(round_bf16(np.array([0x3F808000, 0x3F818000], np.uint32).view(np.float32)),
                            [1.0, 1.015625])
    assert compare(np.array([1.0]), np.array([1.0]), 0, 0)
    assert not compare(np.array([2.0]), np.array([1.0]), 0.008, 0.001)
    assert not compare(np.array([np.nan]), np.array([1.0]), 0.008, 0.001)
    _LOG.info("CPU self-test PASS (intentional mismatch diagnostics above)")


def run(args):
    import torch
    import torch_npu
    from quant_batch_matmul_mxfp8 import quant_batch_matmul_mxfp8

    if not torch.npu.is_available():
        raise RuntimeError("NPU unavailable; source the A5 CANN environment first")
    native_e8m0 = getattr(torch, "float8_e8m0fnu", None)
    e8m0 = native_e8m0
    if e8m0 is None:
        e8m0 = getattr(torch_npu, "float8_e8m0fnu", None)
        if isinstance(e8m0, torch.dtype):
            native_e8m0 = e8m0
    if e8m0 is None:
        raise RuntimeError("torch/torch_npu lacks float8_e8m0fnu; install an A5-compatible build")
    device = f"npu:{DEVICE_ID}"
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    _LOG.info(f"torch={torch.__version__} torch_npu={torch_npu.__version__} "
              f"device={torch.npu.get_device_name(DEVICE_ID)} seed={args.seed}")
    _LOG.info(f"B,M,N,K={BATCH},{M},{N},{K}; rtol={args.rtol}, atol={args.atol}; "
              "all elements must pass")
    _LOG.info(f"Scale representation: {'native E8M0' if native_e8m0 is not None else 'uint8 + E8M0 dtype overrides'}")
    scale_dtype_override = None if native_e8m0 is not None else e8m0
    kinds = ("e5m2", "e4m3fn") if args.dtype == "all" else (args.dtype,)
    passed = failed = errors = 0
    for kind in kinds:
        formats = [f for f in args.formats if f == "nd" or kind == "e4m3fn"]
        if not formats:
            continue
        # Same seed/data family per dtype; every batch/row/column/group varies.
        rng = np.random.default_rng(args.seed)
        x1 = make_fp8_bytes(rng, (BATCH, M, K), kind)
        x2 = make_fp8_bytes(rng, (BATCH, K, N), kind)
        s1 = rng.integers(124, 129, (BATCH, M, K // 64, 2), dtype=np.uint8)
        s2 = rng.integers(124, 129, (BATCH, K // 64, N, 2), dtype=np.uint8)
        biases = {"vector": rng.uniform(-2, 2, (N,)).astype(np.float32),
                  "batch": rng.uniform(-2, 2, (BATCH, 1, N)).astype(np.float32),
                  "none": None}
        _LOG.info(f"Computing independent CPU FP64 golden: {kind} ...")
        base = golden_matmul(x1, x2, s1, s2, kind)
        dtype = getattr(torch, "float8_" + kind)
        for input_format, bias_layout in ((fmt, bl) for fmt in formats for bl in args.bias_layouts):
            label = f"{kind}_no_bias" if bias_layout == "none" else f"{kind}_{bias_layout}_bias"
            label += "_nz" if input_format == "nz" else ""
            _LOG.info(f"CASE {label}")
            bias = biases[bias_layout]
            golden_fp64 = base if bias is None else base + bias
            expected = round_bf16(golden_fp64)
            stage = "input preparation"
            try:
                with torch.inference_mode():
                    a = matrix_to_npu(x1, dtype, device, torch)
                    b = matrix_to_npu(x2, dtype, device, torch)
                    sa = scale_to_npu(s1, device, torch, native_e8m0)
                    sb = scale_to_npu(s2, device, torch, native_e8m0)
                    if input_format == "nz":
                        stage = "NZ format conversion/input checks"
                        b, sb = nz_inputs(b, sb, x2, s2, torch, torch_npu)
                    bias_npu = None if bias is None else torch.from_numpy(bias).to(device)
                    # Both scales are mathematically required for MX.
                    # No offset, y_scale, x1_dtype or x2_dtype is supplied.
                    stage = "AscendC launch/synchronize"
                    out = torch_npu.npu_quant_matmul(
                        a, b, sb,
                        pertoken_scale=sa,
                        bias=bias_npu,
                        output_dtype=torch.bfloat16,
                        group_sizes=[1, 1, GROUP_K],
                        pertoken_scale_dtype=scale_dtype_override,
                        scale_dtype=scale_dtype_override,
                    )
                    torch.npu.synchronize()
                    stage = "Pro launch/synchronize"
                    pro_out = quant_batch_matmul_mxfp8(
                        a, b, sb,
                        pertoken_scale=sa,
                        bias=bias_npu,
                        output_dtype=torch.bfloat16,
                        group_sizes=[1, 1, GROUP_K],
                        pertoken_scale_dtype=scale_dtype_override,
                        scale_dtype=scale_dtype_override,
                    )
                    torch.npu.synchronize()
                    stage = "output transfer"
                    if out.dtype != torch.bfloat16:
                        raise AssertionError(f"Output dtype {out.dtype}; expected BF16")
                    if pro_out.dtype != torch.bfloat16:
                        raise AssertionError(f"Pro output dtype {pro_out.dtype}; expected BF16")
                    actual = out.cpu().float().numpy()
                    pro_actual = pro_out.cpu().float().numpy()
                stage = "precision comparison"
                _LOG.info("  AscendC vs CPU golden:")
                ascendc_ok = compare(actual, expected, args.rtol, args.atol)
                _LOG.info("  Pro vs CPU golden:")
                pro_ok = compare(pro_actual, expected, args.rtol, args.atol)
                _LOG.info("  Pro vs AscendC:")
                pair_ok = compare(pro_actual, actual, args.rtol, args.atol)
                bit_mismatches = int(np.count_nonzero(
                    pro_actual.view(np.uint32) != actual.view(np.uint32)))
                _LOG.info(f"  Pro/AscendC BF16 bit mismatches={bit_mismatches}/{actual.size}")
                ok = ascendc_ok and pro_ok and pair_ok
                if args.require_bitwise:
                    ok = ok and bit_mismatches == 0
                passed += int(ok)
                failed += int(not ok)
                _LOG.info(f"  {'PASS' if ok else 'FAIL'}")
                if not ok and args.dump_dir:
                    args.dump_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(args.dump_dir / f"{label}_seed{args.seed}.npz",
                                        x1_bits=x1, x2_bits=x2, x1_scale_bits=s1,
                                        x2_scale_bits=s2,
                                        bias=np.empty((0,), dtype=np.float32) if bias is None else bias,
                                        has_bias=bias is not None, bias_layout=bias_layout, actual=actual,
                                        pro_actual=pro_actual,
                                        golden=expected, golden_fp64=golden_fp64,
                                        seed=args.seed, kind=kind, input_format=input_format)
                del a, b, sa, sb, bias_npu, out, pro_out
            except Exception:
                errors += 1
                traceback.print_exc()
                if input_format == "nz" and stage == "AscendC launch/synchronize":
                    print_reference_library_paths()
                _LOG.info(f"  ERROR {label}; stage={stage}")
                _LOG.info(f"RESULT: PASS={passed} FAIL={failed} ERROR={errors}")
                _LOG.info("Stopping on execution error; no further NPU calls or empty_cache(). "
                          "Rerun the isolated case in a fresh process after device recovery, if required.")
                return 2
            gc.collect()
    _LOG.info(f"RESULT: PASS={passed} FAIL={failed} ERROR={errors}")
    return 2 if errors else (1 if failed else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # No --device flag: the target NPU comes from TILE_FWK_DEVICE_ID (see DEVICE_ID).
    parser.add_argument("--formats", nargs="+", choices=("nd", "nz"), default=["nd", "nz"],
                        help="x2 formats; NZ means WeightNZ x2 + ND scales, E4M3FN only (default: nd nz)")
    parser.add_argument("--dtype", choices=("all", "e5m2", "e4m3fn"), default="all")
    parser.add_argument("--bias-layouts", nargs="+", choices=("vector", "batch", "none"),
                        default=["vector", "batch", "none"])
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--rtol", type=float, default=0.008)
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--dump-dir", type=Path, help="Save inputs and outputs of accuracy failures as NPZ")
    parser.add_argument("--require-bitwise", action=argparse.BooleanOptionalAction, default=True,
                        help="Require bitwise-identical BF16 Pro/AscendC outputs (default: enabled)")
    parser.add_argument("--self-test", action="store_true", help="Validate CPU golden only; no torch/NPU required")
    args = parser.parse_args()
    if not np.isfinite([args.rtol, args.atol]).all() or args.rtol < 0 or args.atol < 0:
        parser.error("rtol and atol must be finite and nonnegative")
    if args.self_test:
        check_nz_copy_layout()
        self_test()
        return 0
    if args.dtype == "e5m2" and "nd" not in args.formats:
        parser.error("NZ cases support E4M3FN only; use --dtype e4m3fn or --formats nd")
    return run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
