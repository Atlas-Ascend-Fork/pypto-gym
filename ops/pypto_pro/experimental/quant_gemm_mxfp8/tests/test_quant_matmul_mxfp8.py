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

"""A5 MX QuantMatmul: independent CPU FP64 / AscendC / native Pro comparison.

Run from the repository root. The target NPU is selected by the
TILE_FWK_DEVICE_ID environment variable, falling back to logical device 0:

    python ops/pypto_pro/experimental/quant_gemm_mxfp8/tests/test_quant_matmul_mxfp8.py
    python ops/pypto_pro/experimental/quant_gemm_mxfp8/tests/test_quant_matmul_mxfp8.py \
        --shape 96x1792x5120 --dtype e5m2
    python ops/pypto_pro/experimental/quant_gemm_mxfp8/tests/test_quant_matmul_mxfp8.py --self-test

Eighteen cases: twelve ND + six E4M3FN WeightNZ + ND scale, three shapes x vector/no bias.
Rank-2 NN logical inputs; x1 and both scales remain ND,
native E8M0 scales [M,K/64,2] / [K/64,N,2], optional FP32 bias [N], BF16 output.
The uint8 + explicit E8M0 dtype-attribute fallback covers older torch versions.
No optional offset, y_scale or input dtype overrides are supplied.
The same exact device tensors feed both implementations; no reconstruction
or requantization occurs between calls. Golden/comparator are imported from
tests/assets/, adapted to rank 2 and bounded CPU column-block allocation.
Default rtol=0.008, atol=0.001, ALL elements must pass. BF16 bitwise equality
between Pro and AscendC is additionally REQUIRED by default, never inferred
from the two tolerance comparisons. No performance optimization/timing here.
Exit 0=all pass, 1=numerical/bitwise failure, 2=environment/execution error.
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

from quant_matmul_mxfp8_golden_cpu import (
    decode_e8m0, fp8_table, golden_matmul, make_fp8_bytes, round_bf16,
)
from quant_matmul_mxfp8_precision import compare

_LOG = logging.getLogger("quant_gemm_mxfp8")
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

SHAPES = ((96, 1792, 5120), (96, 32768, 1280), (96, 5120, 8192))
SHAPE_NAMES = tuple("x".join(str(v) for v in shape) for shape in SHAPES)
GROUP_K = 32

# Device selection follows the repository convention used by
# quant_sparse_lightning_indexer: read TILE_FWK_DEVICE_ID, default to logical 0.
DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))


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
    Include split-K nonzero starts, K=256 tail, and full scale-window scheduling.
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
    """No Torch/NPU imports: check decoding, scale order, chunking and criteria."""
    np.testing.assert_equal(fp8_table("e5m2")[[0, 60, 188, 1]], [0, 1, -1, 2 ** -16])
    np.testing.assert_equal(fp8_table("e4m3fn")[[0, 56, 184, 1, 126]], [0, 1, -1, 2 ** -9, 448])
    assert np.isinf(fp8_table("e5m2")[124])
    assert np.isnan(fp8_table("e4m3fn")[127])
    np.testing.assert_equal(decode_e8m0(np.array([0, 126, 127, 128, 254], np.uint8)),
                            [2.0 ** -127, 0.5, 1, 2, 2.0 ** 127])
    for kind, one in (("e5m2", 60), ("e4m3fn", 56)):
        a = np.full((3, 128), one, np.uint8)
        b = np.full((128, 5), one, np.uint8)
        sa = np.full((3, 2, 2), 127, np.uint8)
        sb = np.full((2, 5, 2), 127, np.uint8)
        sa[..., 1] = 128
        sa[:, 1, :] += 1  # Different K/64 window; catches reset at K=64.
        sb[..., 0] = 129
        sb[:, 1, :] += 1
        sa[2, :, :] += 1
        expected = np.full((3, 5), (32 * 1 * 4 + 32 * 2 * 1) * 3, np.float64)
        expected[:, 1] *= 2
        expected[2] *= 2
        for block in (1, 2, 5, 512):
            np.testing.assert_equal(golden_matmul(a, b, sa, sb, kind, block), expected)
        np.testing.assert_equal(round_bf16(golden_matmul(a, b, sa, sb, kind)), expected)
        bias = 16 * np.arange(5, dtype=np.float32)
        np.testing.assert_equal(round_bf16(golden_matmul(a, b, sa, sb, kind) + bias), expected + bias)
        # Coordinate-distinct signed random values checked against explicit
        # scalar indexing, independently of reshape/transpose/repeat logic.
        rng = np.random.default_rng(17)
        a = make_fp8_bytes(rng, (3, 128), kind)
        b = make_fp8_bytes(rng, (128, 5), kind)
        table = fp8_table(kind)
        scalar = np.zeros((3, 5), np.float64)
        for row in range(3):
            for col in range(5):
                for kk in range(128):
                    s_a = 2.0 ** (int(sa[row, kk // 64, (kk // 32) % 2]) - 127)
                    s_b = 2.0 ** (int(sb[kk // 64, col, (kk // 32) % 2]) - 127)
                    scalar[row, col] += table[a[row, kk]] * s_a * table[b[kk, col]] * s_b
        np.testing.assert_equal(golden_matmul(a, b, sa, sb, kind, 2), scalar)
    np.testing.assert_equal(round_bf16(np.array([0x3F808000, 0x3F818000], np.uint32).view(np.float32)),
                            [1.0, 1.015625])
    assert compare(np.array([1.0]), np.array([1.0]), 0, 0)
    assert not compare(np.array([2.0]), np.array([1.0]), 0.008, 0.001)
    assert not compare(np.array([np.nan]), np.array([1.0]), 0.008, 0.001)
    _LOG.info("CPU self-test PASS (intentional mismatch diagnostics above)")


def run(args):
    import torch
    import torch_npu
    from quant_matmul_mxfp8 import quant_matmul_mxfp8

    if not torch.npu.is_available():
        raise RuntimeError("NPU unavailable; source the A5 CANN environment first")
    native_e8m0 = getattr(torch, "float8_e8m0fnu", None)
    e8m0 = native_e8m0
    if e8m0 is None:
        e8m0 = getattr(torch_npu, "float8_e8m0fnu", None)
        if isinstance(e8m0, torch.dtype):
            native_e8m0 = e8m0
    if e8m0 is None:
        raise RuntimeError("torch/torch_npu lacks float8_e8m0fnu")
    scale_dtype_override = None if native_e8m0 is not None else e8m0
    device = f"npu:{DEVICE_ID}"
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    _LOG.info(f"torch={torch.__version__} torch_npu={torch_npu.__version__} "
              f"device={torch.npu.get_device_name(DEVICE_ID)} seed={args.seed}")
    _LOG.info(f"rtol={args.rtol}, atol={args.atol}; all elements must pass; "
              f"require_bitwise={args.require_bitwise}")
    _LOG.info(f"Scale representation: {'native E8M0' if native_e8m0 is not None else 'uint8 + E8M0 dtype overrides'}")
    kinds = ("e5m2", "e4m3fn") if args.dtype == "all" else (args.dtype,)
    passed = failed = 0
    for shape_name in args.shape:
        m, n, k = SHAPES[SHAPE_NAMES.index(shape_name)]
        for kind in kinds:
            formats = [f for f in args.formats if f == "nd" or kind == "e4m3fn"]
            if not formats:
                continue
            rng = np.random.default_rng(args.seed)
            x1 = make_fp8_bytes(rng, (m, k), kind)
            x2 = make_fp8_bytes(rng, (k, n), kind)
            s1 = rng.integers(124, 129, (m, k // 64, 2), dtype=np.uint8)
            s2 = rng.integers(124, 129, (k // 64, n, 2), dtype=np.uint8)
            vector_bias = rng.uniform(-2, 2, (n,)).astype(np.float32)
            _LOG.info("  Computing independent CPU FP64 golden ...")
            base = golden_matmul(x1, x2, s1, s2, kind)
            for input_format, has_bias in ((fmt, hb) for fmt in formats for hb in (True, False)):
                bias = vector_bias if has_bias else None
                label = f"{shape_name}_{kind}_bias" if has_bias else f"{shape_name}_{kind}_no_bias"
                label += "_nz" if input_format == "nz" else ""
                _LOG.info(f"CASE {label}")
                golden_fp64 = base + bias if bias is not None else base
                expected = round_bf16(golden_fp64)
                stage = "input preparation"
                try:
                    with torch.inference_mode():
                        dtype = getattr(torch, "float8_" + kind)
                        a = torch.from_numpy(x1).view(dtype).to(device)
                        b = torch.from_numpy(x2).view(dtype).to(device)
                        sa = torch.from_numpy(s1)
                        sb = torch.from_numpy(s2)
                        if native_e8m0 is not None:
                            sa = sa.view(native_e8m0)
                            sb = sb.view(native_e8m0)
                        sa = sa.to(device)
                        sb = sb.to(device)
                        if input_format == "nz":
                            stage = "NZ format conversion/input checks"
                            b, sb = nz_inputs(b, sb, x2, s2, torch, torch_npu)
                        bias_npu = None if bias is None else torch.from_numpy(bias).to(device)
                        stage = "AscendC launch/synchronize"
                        actual_out = torch_npu.npu_quant_matmul(
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
                        pro_out = quant_matmul_mxfp8(
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
                        for output in (actual_out, pro_out):
                            if tuple(output.shape) != (m, n) or output.dtype != torch.bfloat16:
                                raise AssertionError(f"expected BF16 [{m},{n}], got {output.shape}/{output.dtype}")
                            if not output.is_contiguous():
                                raise AssertionError("expected contiguous ND output")
                        actual_bits = actual_out.view(torch.int16).cpu().numpy().view(np.uint16)
                        pro_bits = pro_out.view(torch.int16).cpu().numpy().view(np.uint16)
                        actual = actual_out.cpu().float().numpy()
                        pro_actual = pro_out.cpu().float().numpy()
                    stage = "precision comparison"
                    _LOG.info("  AscendC vs CPU golden:")
                    ascendc_ok = compare(actual, expected, args.rtol, args.atol)
                    _LOG.info("  Pro vs CPU golden:")
                    pro_ok = compare(pro_actual, expected, args.rtol, args.atol)
                    _LOG.info("  Pro vs AscendC:")
                    pair_ok = compare(pro_actual, actual, args.rtol, args.atol)
                    bit_mismatches = int(np.count_nonzero(pro_bits != actual_bits))
                    _LOG.info(f"  Pro/AscendC BF16 bit mismatches={bit_mismatches}/{actual.size}")
                    ok = ascendc_ok and pro_ok and pair_ok and (not args.require_bitwise or bit_mismatches == 0)
                    passed += int(ok)
                    failed += int(not ok)
                    _LOG.info(f"  {'PASS' if ok else 'FAIL'}")
                    if not ok and args.dump_dir:
                        args.dump_dir.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(args.dump_dir / f"{label}_seed{args.seed}.npz",
                                            x1_bits=x1, x2_bits=x2, x1_scale_bits=s1, x2_scale_bits=s2,
                                            bias=np.empty((0,), dtype=np.float32) if bias is None else bias,
                                            has_bias=has_bias,
                                            actual=actual, pro_actual=pro_actual,
                                            actual_bf16_bits=actual_bits, pro_bf16_bits=pro_bits,
                                            golden=expected, golden_fp64=golden_fp64,
                                            seed=args.seed, kind=kind, input_format=input_format,
                                            mnk=np.array([m, n, k]))
                    del a, b, sa, sb, bias_npu, actual_out, pro_out
                except Exception:
                    traceback.print_exc()
                    if input_format == "nz" and stage == "AscendC launch/synchronize":
                        print_reference_library_paths()
                    _LOG.info(f"  ERROR {label}; stage={stage}")
                    _LOG.info(f"RESULT: PASS={passed} FAIL={failed} ERROR=1")
                    _LOG.info("Stopping on execution error; no further NPU calls or empty_cache().")
                    return 2
                del golden_fp64, expected
                gc.collect()
            del x1, x2, s1, s2, base, vector_bias
    _LOG.info(f"RESULT: PASS={passed} FAIL={failed} ERROR=0")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # No --device flag: the target NPU comes from TILE_FWK_DEVICE_ID (see DEVICE_ID).
    parser.add_argument("--formats", nargs="+", choices=("nd", "nz"), default=["nd", "nz"],
                        help="x2 formats; NZ means WeightNZ x2 + ND scales, E4M3FN only (default: nd nz)")
    parser.add_argument("--shape", nargs="+", choices=SHAPE_NAMES, default=list(SHAPE_NAMES))
    parser.add_argument("--dtype", choices=("all", "e5m2", "e4m3fn"), default="all")
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--rtol", type=float, default=0.008)
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--require-bitwise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--self-test", action="store_true", help="CPU golden/comparator checks; no Torch/NPU")
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
