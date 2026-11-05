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

"""A5 Pro MXFP8 batch matmul shape-generalization test; run directly, without pytest.

Run from the repository root. The target NPU is selected by the
TILE_FWK_DEVICE_ID environment variable, falling back to logical device 0:

    python ops/pypto_pro/experimental/quant_bmm_mxfp8/tests/test_quant_batch_matmul_mxfp8_general.py
    python ops/pypto_pro/experimental/quant_bmm_mxfp8/tests/test_quant_batch_matmul_mxfp8_general.py --self-test

Companion to test_quant_batch_matmul_mxfp8.py, which covers the measured shape
Batch=8, M=96, N=1024, K=4096 and requires a bit-exact match against the AscendC
reference. This file covers the rest of the supported domain:

  * BATCH, M and N are arbitrary positive integers, including M % 96 != 0 and
    N % 256 != 0 tail blocks and tile counts that need several grid-stride waves;
  * K is an arbitrary multiple of BK=128, including K < SCALE_KL1 (a partial
    scale window), K == BK (a single accumulation step) and K > SCALE_KL1
    (several scale windows);
  * both FP8 dtypes, all three bias layouts, and ND or WeightNZ x2.

Acceptance differs from the frozen suite on purpose. Every element must satisfy
the independent CPU FP64 golden within rtol/atol; a bit-exact match against
torch_npu.npu_quant_matmul is reported but NOT required off the frozen shape,
because the reference may split K differently there and then rounds differently.
The measured shape is included as a regression and still must be bit-exact. One
kernel now serves the whole domain, and at Batch=8/M=96/N=1024/K=4096 the tile
count is 8*1*4 = 32, so the grid-stride loop runs exactly one wave and each
worker owns one [96,256] tile -- the same work split the performance cases use.

--self-test checks the host-side admission rules only: no torch, no NPU.
"""

import argparse
import gc
import logging
import os
import sys
import traceback
from collections import namedtuple
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OP_DIR = _HERE.parent
for _path in (_HERE, _HERE / "assets", _OP_DIR / "op_kernel"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np

from quant_batch_matmul_mxfp8_golden_cpu import golden_matmul, round_bf16
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

DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))

# Tile geometry is frozen in the kernel; these mirror it so the case table can
# state how many output tiles and grid-stride waves each shape produces.
BM, BN, BK = 96, 256, 128
FROZEN = (8, 96, 1024, 4096)

# Each CASES entry holds batch, m, n, k, dtype, bias, x2_format and a rationale string, in that order.
CASES = [
    (1, 96, 256, 256, "e4m3fn", "none", "nd", "single full tile; smallest K that fills one KL1 window"),
    (1, 96, 256, 256, "e4m3fn", "vector", "nd", "single tile + FP32 [N] bias broadcast over B and M"),
    (1, 96, 256, 256, "e4m3fn", "batch", "nd", "single tile + per-batch bias row offset"),
    (1, 16, 32, 128, "e4m3fn", "none", "nd", "M and N each one 16-wide fractal; K == BK, a single accumulation step"),
    (1, 96, 256, 128, "e4m3fn", "vector", "nd", "K == BK with bias: first step is also the last, so it carries Final"),
    (2, 50, 100, 384, "e4m3fn", "vector", "nd", "M tail (50) + N tail (100) + K = 1.5 x KL1"),
    (2, 50, 100, 384, "e5m2", "none", "nd", "same tails under E5M2, no bias"),
    (1, 200, 300, 640, "e4m3fn", "batch", "nd", "M spans 2 tiles with a tail (200 = 2*96 + 8); N tail; K = 2.5 x KL1"),
    (3, 192, 512, 512, "e4m3fn", "none", "nd", "M exactly 2 tiles, no tail; BATCH = 3"),
    (1, 96, 1024, 5120, "e4m3fn", "vector", "nd", "K > SCALE_KL1: two scale windows (4096 + 1024)"),
    (1, 96, 256, 4224, "e4m3fn", "none", "nd", "second scale window holds only 128 (4224 = 4096 + 128)"),
    (5, 97, 257, 1152, "e4m3fn", "vector", "nd", "M and N both tailed, odd BATCH, K = 4.5 x KL1"),
    (16, 192, 512, 128, "e4m3fn", "none", "nd", "64 output tiles over 32 cores: two grid-stride waves"),
    (1, 96, 288, 256, "e4m3fn", "none", "nz", "WeightNZ with an N tail (288 = 256 + 32, still 32-aligned)"),
    (1, 96, 256, 256, "e4m3fn", "vector", "nz", "WeightNZ full tile + vector bias"),
    (1, 96, 32, 256, "e4m3fn", "none", "nz", "WeightNZ at the 32-column fractal minimum"),
    (2, 96, 256, 256, "e5m2", "batch", "nd", "E5M2 + per-batch bias"),
    FROZEN + ("e4m3fn", "vector", "nd", "measured-shape regression: must stay bit-exact"),
    FROZEN + ("e5m2", "none", "nd", "measured-shape regression: must stay bit-exact"),
]

# Each REJECTS entry holds batch, m, n, k, x2_format and the expected exception substring.
REJECTS = [
    (1, 96, 256, 64, "nd", "multiple of BK"),
    (1, 96, 256, 192, "nd", "multiple of BK"),
    (1, 96, 256, 100, "nd", "multiple of BK"),
    (1, 96, 100, 256, "nz", "multiple of 32"),
    (1, 96, 48, 256, "nz", "multiple of 32"),
]


def tiles_and_waves(batch, m, n, cores=32):
    tiles = batch * -(-m // BM) * -(-n // BN)
    return tiles, -(-tiles // cores)


def self_test():
    """CPU-only checks of the case table and the tile/wave arithmetic."""
    assert BK == 128 and BM == 96 and BN == 256
    seen = set()
    for batch, m, n, k, kind, bias, fmt, _ in CASES:
        ident = (batch, m, n, k, kind, bias, fmt)
        assert ident not in seen, f"duplicate case {ident}"
        seen.add(ident)
        assert batch >= 1 and m >= 1 and n >= 1, f"non-positive extent in {ident}"
        assert k % BK == 0, f"K={k} is not a multiple of BK in {ident}"
        assert kind in ("e5m2", "e4m3fn"), f"bad dtype in {ident}"
        assert bias in ("none", "vector", "batch"), f"bad bias layout in {ident}"
        assert fmt in ("nd", "nz"), f"bad x2 format in {ident}"
        if fmt == "nz":
            assert kind == "e4m3fn", f"WeightNZ needs E4M3FN in {ident}"
            assert n % 32 == 0, f"WeightNZ N={n} not 32-aligned in {ident}"
        tiles, waves = tiles_and_waves(batch, m, n)
        assert tiles == batch * -(-m // BM) * -(-n // BN)
        assert waves >= 1
    # The table must actually exercise the boundaries it claims to.
    assert any(c[3] == BK for c in CASES), "no K == BK single-step case"
    assert any(c[3] > 4096 for c in CASES), "no multi-scale-window case"
    assert any(c[3] % 256 != 0 for c in CASES), "no K that is not a multiple of KL1"
    assert any(c[1] % BM for c in CASES), "no M tail case"
    assert any(c[2] % BN for c in CASES), "no N tail case"
    assert any(tiles_and_waves(c[0], c[1], c[2])[1] > 1 for c in CASES), "no multi-wave case"
    assert any(c[6] == "nz" and c[2] % BN for c in CASES), "no WeightNZ N-tail case"
    assert len({c[4] for c in CASES}) == 2, "both FP8 dtypes must appear"
    assert {c[5] for c in CASES} == {"none", "vector", "batch"}, "all bias layouts must appear"
    assert sum(1 for c in CASES if (c[0], c[1], c[2], c[3]) == FROZEN) == 2, \
        "frozen-domain regression must appear exactly twice"
    for batch, m, n, k, fmt, needle in REJECTS:
        if fmt == "nd":
            assert k % BK != 0, f"{(batch, m, n, k)} should violate the K rule"
        else:
            assert n % 32 != 0, f"{(batch, m, n, k)} should violate the WeightNZ N rule"
        assert needle
    _LOG.info(f"CPU self-test PASS ({len(CASES)} device cases, {len(REJECTS)} rejection cases)")


# Named container for one case's inputs: the raw NumPy bytes that feed the CPU
# golden, plus the staged device tensors handed to the kernel.
BuiltInputs = namedtuple("BuiltInputs", "x1 x2 s1 s2 bias_np a b sa sb bias_t")


def build(batch, m, n, k, kind, bias_layout, fmt, device, torch, torch_npu, native):
    """Construct inputs with the frozen suite's generator so bytes are comparable."""
    import test_quant_batch_matmul_mxfp8 as frozen_suite
    rng = np.random.default_rng(20261103)
    x1 = frozen_suite.make_fp8_bytes(rng, (batch, m, k), kind)
    x2 = frozen_suite.make_fp8_bytes(rng, (batch, k, n), kind)
    s1 = rng.integers(124, 129, (batch, m, k // 64, 2), dtype=np.uint8)
    s2 = rng.integers(124, 129, (batch, k // 64, n, 2), dtype=np.uint8)
    biases = {"vector": rng.uniform(-2, 2, (n,)).astype(np.float32),
              "batch": rng.uniform(-2, 2, (batch, 1, n)).astype(np.float32),
              "none": None}
    dtype = getattr(torch, "float8_" + kind)
    a = torch.from_numpy(x1).view(dtype).to(device)
    b = torch.from_numpy(x2).view(dtype).to(device)
    sa, sb = torch.from_numpy(s1), torch.from_numpy(s2)
    if native is not None:
        sa, sb = sa.view(native), sb.view(native)
    sa, sb = sa.to(device), sb.to(device)
    if fmt == "nz":
        b, sb = frozen_suite.nz_inputs(b, sb, x2, s2, torch, torch_npu)
    bias_np = biases[bias_layout]
    bias_t = None if bias_np is None else torch.from_numpy(bias_np).to(device)
    return BuiltInputs(x1, x2, s1, s2, bias_np, a, b, sa, sb, bias_t)


def run(args):
    import torch
    import torch_npu
    from quant_batch_matmul_mxfp8 import quant_batch_matmul_mxfp8

    if not torch.npu.is_available():
        raise RuntimeError("NPU unavailable; source the A5 CANN environment first")
    native = getattr(torch, "float8_e8m0fnu", None)
    e8m0 = native
    if e8m0 is None:
        e8m0 = getattr(torch_npu, "float8_e8m0fnu", None)
        if isinstance(e8m0, torch.dtype):
            native = e8m0
    if e8m0 is None:
        raise RuntimeError("torch/torch_npu lacks float8_e8m0fnu; install an A5-compatible build")
    override = None if native is not None else e8m0
    device = f"npu:{DEVICE_ID}"
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    _LOG.info(f"torch={torch.__version__} torch_npu={torch_npu.__version__} "
              f"device={torch.npu.get_device_name(DEVICE_ID)}")
    _LOG.info(f"frozen domain={FROZEN}  BM/BN/BK={BM}/{BN}/{BK}  rtol={args.rtol} atol={args.atol}")
    _LOG.info(f"scale carrier: {'native E8M0' if native is not None else 'uint8 + E8M0 override'}")

    cases = CASES if not args.only else [CASES[i] for i in args.only]
    passed = failed = errors = 0
    bit_exact = bit_diff = 0
    for batch, m, n, k, kind, bias_layout, fmt, why in cases:
        label = f"({batch},{m},{n},{k}) {kind} {bias_layout} {fmt}"
        frozen = (batch, m, n, k) == FROZEN
        tag = "frozen" if frozen else "general"
        tiles, waves = tiles_and_waves(batch, m, n)
        _LOG.info(f"CASE {label}  [{tag}] tiles={tiles} waves={waves}  # {why}")
        try:
            x1, x2, s1, s2, bias_np, a, b, sa, sb, bias_t = build(
                batch, m, n, k, kind, bias_layout, fmt, device, torch, torch_npu, native)
            with torch.inference_mode():
                base = golden_matmul(x1, x2, s1, s2, kind)
                expected = round_bf16(base if bias_np is None else base + bias_np)
                kw = dict(pertoken_scale=sa, bias=bias_t, output_dtype=torch.bfloat16,
                          group_sizes=[1, 1, 32], pertoken_scale_dtype=override,
                          scale_dtype=override)
                pro = quant_batch_matmul_mxfp8(a, b, sb, **kw)
                torch.npu.synchronize()
                got = pro.cpu().float().numpy()
                ref_np = None
                if not args.skip_reference:
                    try:
                        ref = torch_npu.npu_quant_matmul(a, b, sb, **kw)
                        torch.npu.synchronize()
                        ref_np = ref.cpu().float().numpy()
                    except Exception as exc:
                        _LOG.info(f"  reference unavailable: {type(exc).__name__}: {str(exc)[:90]}")
            _LOG.info("  Pro vs CPU golden:")
            ok_golden = compare(got, expected, args.rtol, args.atol)
            mismatches = None
            ok_ref = True
            if ref_np is not None:
                mismatches = int(np.count_nonzero(got.view(np.uint32) != ref_np.view(np.uint32)))
                _LOG.info(f"  Pro vs AscendC: bit mismatches={mismatches}/{got.size}")
                _LOG.info("  Pro vs AscendC (tolerance):")
                ok_ref = compare(got, ref_np, args.rtol, args.atol)
                if mismatches == 0:
                    bit_exact += 1
                else:
                    bit_diff += 1
            # Off the frozen shape a bit difference is acceptable: the reference may
            # split K differently and then round differently. On it, it is not.
            ok = ok_golden and ok_ref and (not frozen or mismatches == 0)
            if frozen and mismatches is None:
                ok = False
                _LOG.info("  frozen case requires the reference for the bit-exact check")
            passed += int(ok)
            failed += int(not ok)
            _LOG.info(f"  {'PASS' if ok else 'FAIL'}")
            del a, b, sa, sb, bias_t, pro
            if ref_np is not None:
                del ref
        except Exception:
            errors += 1
            traceback.print_exc()
            _LOG.info(f"  ERROR {label}")
            _LOG.info("RESULT: PASS=%d FAIL=%d ERROR=%d" % (passed, failed, errors))
            return 2
        gc.collect()

    _LOG.info(f"\nrejection contract ({len(REJECTS)} host-side rules):")
    for batch, m, n, k, fmt, needle in (REJECTS if not args.only else []):
        try:
            # Build the offending shape literally. K need not be a multiple of 64
            # here: the packed scale window is k // 64, and the BK rule is
            # strictly stronger than the K % 64 rule, so it fires first.
            u8 = lambda *s: torch.from_numpy(np.full(s, 56, np.uint8))
            a = u8(batch, m, k).view(torch.float8_e4m3fn).to(device)
            b = u8(batch, k, n).view(torch.float8_e4m3fn).to(device)
            if fmt == "nz":
                b = torch_npu.npu_format_cast(b, 29)
            sa = u8(batch, m, k // 64, 2).view(native).to(device)
            sb = u8(batch, k // 64, n, 2).view(native).to(device)
            with torch.inference_mode():
                quant_batch_matmul_mxfp8(a, b, sb, pertoken_scale=sa,
                                         output_dtype=torch.bfloat16,
                                         group_sizes=[1, 1, 32])
            _LOG.info(f"  FAIL ({batch},{m},{n},{k}) {fmt} was admitted; expected rejection "
                      f"containing {needle!r}")
            failed += 1
        except (ValueError, TypeError) as exc:
            text = str(exc)
            if needle in text:
                _LOG.info(f"  PASS ({batch},{m},{n},{k}) {fmt} rejected: {text[:88]}")
                passed += 1
            else:
                _LOG.info(f"  FAIL ({batch},{m},{n},{k}) {fmt} rejected for the wrong reason: "
                          f"{text[:88]}")
                failed += 1
        except Exception:
            errors += 1
            traceback.print_exc()
        gc.collect()

    _LOG.info(f"RESULT: PASS={passed} FAIL={failed} ERROR={errors}")
    _LOG.info(f"  device cases: {len(cases)}  (bit-exact vs reference {bit_exact}, "
              f"within tolerance but not bit-exact {bit_diff})")
    return 2 if errors else (1 if failed else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # No --device flag: the target NPU comes from TILE_FWK_DEVICE_ID (see DEVICE_ID).
    parser.add_argument("--rtol", type=float, default=0.008)
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--only", type=int, nargs="+",
                        help="run just these indices into the case table")
    parser.add_argument("--skip-reference", action="store_true",
                        help="skip torch_npu.npu_quant_matmul; golden-only checking")
    parser.add_argument("--self-test", action="store_true",
                        help="validate the case table on CPU only; no torch/NPU required")
    args = parser.parse_args()
    if not np.isfinite([args.rtol, args.atol]).all() or args.rtol < 0 or args.atol < 0:
        parser.error("rtol and atol must be finite and nonnegative")
    if args.self_test:
        self_test()
        return 0
    if args.only and any(i < 0 or i >= len(CASES) for i in args.only):
        parser.error(f"--only indices must be within 0..{len(CASES) - 1}")
    return run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
