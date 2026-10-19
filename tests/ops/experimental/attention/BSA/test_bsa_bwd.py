#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
BSA Backward Unified Test & Benchmark

Modes:
  --mode precision     Precision test (default)
  --mode perf          Swimlane perf benchmark (direct swimlane parsing)
  --mode perf-table    Swimlane perf benchmark with clean table output
  --mode perf-wall     Wall-clock timing only (no swimlane, works with cached kernels)
  --mode non-aligned   Non-aligned/variable-length precision

Usage:
  # For precision and wall-clock modes, run from BSA directory:
  cd tests/ops/experimental/attention/BSA
  python3 test_bsa_bwd.py                     # precision
  python3 test_bsa_bwd.py --mode perf-wall    # wall-clock perf
  python3 test_bsa_bwd.py --mode non-aligned  # non-aligned precision

  # For swimlane perf modes, MUST run from parent directory
  # (PyPTO trace post-processing requires CWD at parent level):
  cd tests/ops/experimental/attention
  python3 BSA/test_bsa_bwd.py --mode perf         # swimlane perf
  python3 BSA/test_bsa_bwd.py --mode perf-table   # swimlane perf table
"""

import argparse
import sys
import os
import time
import torch
import pytest


# Pre-parse mode argument to set BSA_RUNTIME_DEBUG_MODE before importing
# bsa_common (which reads the env var at module load time).
_pre_mode = 'precision'
for _i, _arg in enumerate(sys.argv):
    if _arg == '--mode' and _i + 1 < len(sys.argv):
        _pre_mode = sys.argv[_i + 1]
        break
if _pre_mode in ('perf', 'perf-table'):
    os.environ['BSA_RUNTIME_DEBUG_MODE'] = '1'


def _check_project_root_candidate(d, test_file_dir, bsa_marker):
    """Check if directory d is a project root containing BSA impl under src layout."""
    if not (os.path.isdir(os.path.join(d, 'tests')) and os.path.isdir(os.path.join(d, 'src'))):
        return None
    tests_dir = os.path.join(d, 'tests')
    rel_from_tests = os.path.relpath(test_file_dir, tests_dir)
    # Map tests/ops/<rest> -> ops/pypto_tensor/<rest>
    if rel_from_tests.startswith('ops/'):
        rest = rel_from_tests[len('ops/'):]
        candidate = os.path.join(d, 'ops', 'pypto_tensor', rest)
        if all(os.path.isfile(os.path.join(candidate, m)) for m in bsa_marker):
            return candidate
    return None


def _resolve_bsa_root():
    """Locate BSA root directory (containing common/, FWD/, BWD/).

    Search strategy:
      1. BSA_ROOT environment variable (if set and valid).
      2. Walk upward from this file's directory, checking each parent
         for marker files (BSA_README.md, common/bsa_common.py).
      3. At each parent that has both 'tests/' and 'src/' subdirectories
         (project-root candidate), check whether the src layout contains
         the BSA implementation under ops/pypto_tensor/ with
         the same relative path as under tests/ops/.
    """
    _BSA_MARKER = ("BSA_README.md", os.path.join("common", "bsa_common.py"))

    env_root = os.environ.get("BSA_ROOT")
    if env_root and all(os.path.isfile(os.path.join(env_root, m)) for m in _BSA_MARKER):
        return os.path.abspath(env_root)

    test_file_dir = os.path.dirname(os.path.abspath(__file__))
    d = test_file_dir
    for _ in range(20):
        # Standard upward search: check if markers exist here
        if all(os.path.isfile(os.path.join(d, m)) for m in _BSA_MARKER):
            return d

        # Project-root search: check if BSA impl is under src layout
        candidate = _check_project_root_candidate(d, test_file_dir, _BSA_MARKER)
        if candidate:
            return candidate

        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent

    raise RuntimeError(
        "Cannot locate BSA root directory.\n"
        "  Set BSA_ROOT to the path containing common/, FWD/, BWD/.")


_bsa_root = _resolve_bsa_root()
_bsa_root = os.path.abspath(_bsa_root)
# _test_dir is the directory containing this test file — used for __file__-relative
# path resolution instead of CWD-dependent os.path.abspath('.').
_test_dir = os.path.dirname(os.path.abspath(__file__))
for _sub in ('common', 'FWD', 'BWD'):
    _p = os.path.join(_bsa_root, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bsa_test_utils import (  # noqa: huawei-wrong-import-position
    logger, cfg, get_device,
    gen_inputs, gen_inputs_non_aligned, GenInputsConfig, GenInputsNonAlignedConfig,
    _compare_grads, _compare_grads_non_aligned,
    SwimlaneQueryConfig,
    _take_perf_timestamp, _record_perf_from_dirs,
    _print_perf_summary, _perf_records,
    _find_and_parse_swimlane, _save_perf_case_data,
    _clean_output_dirs, _print_perf_table,
    _run_test_cases, _run_non_aligned_test_cases, _print_test_results,
    BwdCaseConfig, BwdNonAlignedCaseConfig, _print_mode_results,
    BSARunnerConfig, _measure_wall_clock_time,
)
from bsa_fwd_golden import (  # noqa: huawei-wrong-import-position
    bsa_forward_golden, BSAForwardInputs)
from bsa_fwd_impl import (  # noqa: huawei-wrong-import-position
    block_sparse_attention_forward, BSAForwardCallInputs)
from bsa_bwd_golden import (  # noqa: huawei-wrong-import-position
    bsa_backward_golden, BSABackwardInputs)
from bsa_bwd_impl import (  # noqa: huawei-wrong-import-position
    block_sparse_attention_backward, BSABackwardCallInputs,
    _last_backward_perf_dirs)

# Test Cases

_BWD_SWIMLANE_SEARCH_BASES = [
    os.path.abspath(os.path.join(_test_dir, 'output')),
    os.path.abspath(os.path.join(os.path.dirname(_test_dir), 'output')),
    os.path.abspath(os.path.join(_bsa_root, 'output')),
    os.path.abspath(os.path.join(_bsa_root, 'BWD', 'output')),
]
# _BWD_CLEAN_DIRS must cover all directories where PyPTO runtime may write
# swimlane output — same as search bases plus FWD output (shared workspace).
_BWD_CLEAN_DIRS = _BWD_SWIMLANE_SEARCH_BASES + [
    os.path.abspath(os.path.join(_bsa_root, 'FWD', 'output')),
]

ALL_CASES = [
    ("S256 Sparse50", 1, 4, 2, 256, 256, 0.5),
    ("S512 Sparse70", 1, 4, 4, 512, 512, 0.7),
    ("MHA Dense", 1, 4, 4, 256, 256, 1.0),
    ("GQA", 1, 8, 2, 256, 512, 0.5),
    ("MHA sparse0.3", 1, 4, 4, 256, 256, 0.3),
    ("MHA sparse0.7", 1, 4, 4, 256, 256, 0.7),
    ("MHA S512", 1, 4, 4, 512, 512, 0.7),
    ("B2 MHA S256", 2, 4, 4, 256, 256, 0.5),
    ("B2 GQA S256", 2, 8, 2, 256, 512, 0.5),
    ("B2 Dense S256", 2, 4, 4, 256, 256, 1.0),
    ("B2 Sparse S512", 2, 4, 4, 512, 512, 0.7),
    ("B4 MHA S256", 4, 4, 4, 256, 256, 0.5),
]

QUICK_CASES = [
    ("S256 Sparse50", 1, 4, 2, 256, 256, 0.5),
    ("S512 Sparse70", 1, 4, 4, 512, 512, 0.7),
    ("B2 MHA S256", 2, 4, 4, 256, 256, 0.5),
]

LONG_CASES = [
    ("S1024 Sparse50", 1, 4, 2, 1024, 1024, 0.5),
    ("S1024 Dense", 1, 4, 4, 1024, 1024, 1.0),
    ("S2048 Sparse50", 1, 4, 2, 2048, 2048, 0.5),
]

NON_ALIGNED_CASES = [
    ("S300 NonAligned", 1, 4, 2, 512, 512, 0.5, [300], [300]),
    ("B2 VarLen S256-300", 2, 4, 4, 512, 512, 0.5, [256, 300], [256, 300]),
    ("S400 NonAligned", 1, 8, 2, 512, 1024, 0.4, [400], [800]),
    ("B2 MixAligned", 2, 4, 4, 512, 512, 0.6, [256, 400], [256, 400]),
    ("B2 VarLen S2264-3603", 2, 2, 2, 3840, 4096, 0.5, [2264, 3603], [2264, 3603]),
]


# --- Precision test mode ---

def _run_fwd_impl(inputs):
    """Run FWD impl and return (o_p, lse_p)."""
    q_inp, k_inp, v_inp = inputs.q, inputs.k, inputs.v
    mask, asq, askv = inputs.mask, inputs.asq, inputs.askv
    fwd_result = block_sparse_attention_forward(BSAForwardCallInputs(
        query=q_inp, key=k_inp, value=v_inp, block_sparse_mask=mask,
        actual_seq_lengths=asq, actual_seq_lengths_kv=askv,
        block_shape=None, cfg=cfg))
    return fwd_result.o, fwd_result.lse


def _run_fwd_golden(inputs, device):
    """Run FWD golden and return (o_g, lse_g) on device."""
    q_inp, k_inp, v_inp = inputs.q, inputs.k, inputs.v
    mask, asq, askv = inputs.mask, inputs.asq, inputs.askv
    fwd_result = bsa_forward_golden(BSAForwardInputs(
        query=q_inp.cpu(), key=k_inp.cpu(), value=v_inp.cpu(), block_sparse_mask=mask.cpu(),
        block_shape_x=None, block_shape_y=None,
        actual_seq_lengths=asq.cpu(), actual_seq_lengths_kv=askv.cpu(),
        scale_value=None, cfg=cfg))
    return fwd_result.o.to(device), fwd_result.lse.to(device)


def _run_bwd_golden(inputs, o_g, lse_g):
    """Run BWD golden and return (dq_g, dk_g, dv_g)."""
    q_inp, k_inp, v_inp, d_o = inputs.q, inputs.k, inputs.v, inputs.d_o
    mask, asq, askv = inputs.mask, inputs.asq, inputs.askv
    bwd_result = bsa_backward_golden(BSABackwardInputs(
        dout=d_o.cpu(), query=q_inp.cpu(), key=k_inp.cpu(), value=v_inp.cpu(),
        attention_out=o_g.cpu(), softmax_lse=lse_g.cpu(),
        block_sparse_mask=mask.cpu(),
        block_shape_x=None, block_shape_y=None,
        actual_seq_lengths=asq.cpu(), actual_seq_lengths_kv=askv.cpu(),
        scale_value=None, cfg=cfg))
    return bwd_result.d_q, bwd_result.d_k, bwd_result.d_v


def _run_bwd_impl(inputs, o_p, lse_p):
    """Run BWD impl and return (dq_p, dk_p, dv_p)."""
    q_inp, k_inp, v_inp, d_o = inputs.q, inputs.k, inputs.v, inputs.d_o
    mask, asq, askv = inputs.mask, inputs.asq, inputs.askv
    bwd_result = block_sparse_attention_backward(BSABackwardCallInputs(
        dout=d_o, query=q_inp, key=k_inp, value=v_inp,
        attention_out=o_p, softmax_lse=lse_p,
        block_sparse_mask=mask,
        actual_seq_lengths=asq, actual_seq_lengths_kv=askv,
        block_shape=None, cfg=cfg))
    return bwd_result.d_q, bwd_result.d_k, bwd_result.d_v


def do_precision_test(case_cfg):
    """Run forward+backward precision test for one configuration."""
    name = case_cfg.name
    b = case_cfg.b
    hq = case_cfg.hq
    hkv = case_cfg.hkv
    sq = case_cfg.sq
    skv = case_cfg.skv
    sparsity = case_cfg.sparsity
    device = get_device()
    torch.npu.set_device(device)
    logger.info(f"  [BWD] {name}: b={b} hq={hq} hkv={hkv} sq={sq} skv={skv} sp={sparsity}")
    inputs = gen_inputs(GenInputsConfig(
        b=b, hq=hq, hkv=hkv, sq=sq, skv=skv, sparsity=sparsity, device=device, seed=42))

    o_p, lse_p = _run_fwd_impl(inputs)
    o_g, lse_g = _run_fwd_golden(inputs, device)
    dq_g, dk_g, dv_g = _run_bwd_golden(inputs, o_g, lse_g)

    _take_perf_timestamp()
    dq_p, dk_p, dv_p = _run_bwd_impl(inputs, o_p, lse_p)

    tracked_dirs = [d for d in _last_backward_perf_dirs.values() if d is not None]
    _record_perf_from_dirs(f"{name} [b={b},hq={hq}]", tracked_dirs, kernel_filter=("dQ", "dK/dV"))

    _compare_grads(
        [("dQ", dq_g, dq_p), ("dK", dk_g, dk_p), ("dV", dv_g, dv_p)],
        cfg.bwd_atol, cfg.bwd_rtol)

    logger.info(f"  [BWD] {name} precision: PASS")


def do_non_aligned_precision_test(case_cfg):
    """Run forward+backward precision test for non-aligned/variable-length sequences."""
    device = get_device()
    torch.npu.set_device(device)
    logger.info(f"  [BWD-NA] {case_cfg.name}: b={case_cfg.b} hq={case_cfg.hq} hkv={case_cfg.hkv} "
                f"sq_max={case_cfg.sq_max} skv_max={case_cfg.skv_max} "
                f"asq={case_cfg.asq_list} askv={case_cfg.askv_list} sp={case_cfg.sparsity}")
    inputs = gen_inputs_non_aligned(GenInputsNonAlignedConfig(
        b=case_cfg.b, hq=case_cfg.hq, hkv=case_cfg.hkv,
        sq_max=case_cfg.sq_max, skv_max=case_cfg.skv_max,
        actual_seq_lengths_q=case_cfg.asq_list, actual_seq_lengths_kv=case_cfg.askv_list,
        sparsity=case_cfg.sparsity, device=device, seed=42))

    o_p, lse_p = _run_fwd_impl(inputs)
    o_g, lse_g = _run_fwd_golden(inputs, device)
    dq_g, dk_g, dv_g = _run_bwd_golden(inputs, o_g, lse_g)

    _take_perf_timestamp()
    dq_p, dk_p, dv_p = _run_bwd_impl(inputs, o_p, lse_p)

    tracked_dirs = [out_dir for out_dir in _last_backward_perf_dirs.values() if out_dir is not None]
    _record_perf_from_dirs(f"{case_cfg.name} [b={case_cfg.b},hq={case_cfg.hq},NA]",
                           tracked_dirs, kernel_filter=("dQ", "dK/dV"))

    _compare_grads_non_aligned(
        [("dQ", dq_g, dq_p), ("dK", dk_g, dk_p), ("dV", dv_g, dv_p)],
        cfg.bwd_atol, cfg.bwd_rtol, inputs.asq, inputs.askv)
    logger.info(f"  [BWD-NA] {case_cfg.name} precision: PASS (non-aligned)")


def do_perf_bench(case_cfg):
    """Run one BWD kernel call with BSA_RUNTIME_DEBUG_MODE=1 and parse swimlane metrics.

    NOTE: Does NOT clean output dirs per-case. Swimlane data (merged_swimlane.json)
    is created during kernel compilation, not per-invocation. Since BWD uses
    dynamic dimensions, all shapes share the same compiled kernel and thus
    produce identical compilation-phase metrics (TaskTime, AICoreTime, Cores,
    Tasks). Per-case cleaning would remove the only swimlane source, causing
    N/A results. Clean once at the start of the perf test instead.
    """
    name = case_cfg.name
    b = case_cfg.b
    hq = case_cfg.hq
    hkv = case_cfg.hkv
    sq = case_cfg.sq
    skv = case_cfg.skv
    sparsity = case_cfg.sparsity
    device = get_device()
    torch.npu.set_device(device)
    logger.info(f"  [BWD-perf] {name}: b={b} hq={hq} hkv={hkv} sq={sq} skv={skv}")
    inputs = gen_inputs(GenInputsConfig(
        b=b, hq=hq, hkv=hkv, sq=sq, skv=skv, sparsity=sparsity, device=device, seed=42))

    o_p, lse_p = _run_fwd_impl(inputs)
    torch.npu.synchronize()

    _take_perf_timestamp()
    start = time.time()
    dq_p, dk_p, dv_p = _run_bwd_impl(inputs, o_p, lse_p)
    torch.npu.synchronize()
    wall_ms = (time.time() - start) * 1000

    # Find and parse swimlane metrics
    row, swim_path = _find_and_parse_swimlane(SwimlaneQueryConfig(
        search_bases=_BWD_SWIMLANE_SEARCH_BASES, tag="BWD-perf", name=name,
        b=b, hq=hq, sq=sq, wall_ms=wall_ms))

    # Save per-case swimlane data (with kernel_aicore for BWD merged kernels)
    _save_perf_case_data(
        swim_path, os.path.join('output_bwd', name.replace(' ', '_')),
        extra_copy_dirs=[('kernel_aicore', 'kernel_aicore')])

    torch.npu.empty_cache()
    return row


def do_perf_wall(case_cfg):
    """Run BWD kernel call WITHOUT debug mode — pure wall-clock timing."""
    device = get_device()
    torch.npu.set_device(device)
    logger.info(f"  [BWD-wall] {case_cfg.name}: b={case_cfg.b} hq={case_cfg.hq} "
                f"hkv={case_cfg.hkv} sq={case_cfg.sq} skv={case_cfg.skv}")
    inputs = gen_inputs(GenInputsConfig(
        b=case_cfg.b, hq=case_cfg.hq, hkv=case_cfg.hkv, sq=case_cfg.sq,
        skv=case_cfg.skv, sparsity=case_cfg.sparsity, device=device, seed=42))

    o_out, lse = _run_fwd_impl(inputs)
    median_ms, mean_ms = _measure_wall_clock_time(
        lambda: _run_bwd_impl(inputs, o_out, lse))

    logger.info(f"  [BWD-wall] {case_cfg.name}: wall_median={median_ms:.2f}ms "
                f"wall_mean={mean_ms:.2f}ms (3 runs)")
    torch.npu.empty_cache()
    return {'name': case_cfg.name, 'BH': f"b={case_cfg.b},hq={case_cfg.hq}",
            'Sq': case_cfg.sq, 'wall_ms': median_ms}


# --- Pytest entry points for CI smoke test ---
# These thin wrappers allow pytest to collect and run BSA BWD precision tests.
# The argparse/main() entry point is still available for standalone execution.

def test_bsa_bwd_s256_sparse50():
    do_precision_test(BwdCaseConfig("S256 Sparse50", 1, 4, 2, 256, 256, 0.5))


@pytest.mark.skip(reason="large test case")
def test_bsa_bwd_s512_sparse70():
    do_precision_test(BwdCaseConfig("S512 Sparse70", 1, 4, 4, 512, 512, 0.7))


@pytest.mark.skip(reason="large test case")
def test_bsa_bwd_b2_mha_s256():
    do_precision_test(BwdCaseConfig("B2 MHA S256", 2, 4, 4, 256, 256, 0.5))


# Main

def _run_cases(cases, mode):
    """Run all cases for given mode, returning (passed, failed, perf_rows)."""
    runner_cfg = BSARunnerConfig(
        mode=mode, precision_fn=do_precision_test,
        perf_fn=do_perf_bench, perf_wall_fn=do_perf_wall,
        case_cfg_type=BwdCaseConfig)
    return _run_test_cases(cases, runner_cfg)


def _run_non_aligned_cases(cases):
    """Run all non-aligned cases, returning (passed, failed)."""
    return _run_non_aligned_test_cases(cases, do_non_aligned_precision_test,
                                        case_cfg_type=BwdNonAlignedCaseConfig)


def main():
    parser = argparse.ArgumentParser(description="BSA Backward Unified Test & Benchmark")
    parser.add_argument("--mode", choices=["precision", "perf", "perf-table", "perf-wall", "non-aligned"],
                        default="precision",
                        help="precision=baseline, perf=swimlane benchmark, "
                             "perf-table=swimlane benchmark with table, "
                             "perf-wall=wall-clock timing (no recompile), "
                             "non-aligned=non-aligned/variable-length precision")
    parser.add_argument("--cases", choices=["all", "quick", "long", "non-aligned"], default="all",
                        help="all=full suite, quick=S256+S512+B2S256, "
                             "long=S1024+S2048 (long-sequence perf), "
                             "non-aligned=non-aligned/variable-length cases")
    args = parser.parse_args()

    # BSA_RUNTIME_DEBUG_MODE is already set before module import if needed
    _perf_records.clear()

    passed = 0
    failed = 0
    perf_rows = []

    if args.mode == "non-aligned" or args.cases == "non-aligned":
        logger.info("\n" + "=" * 70)
        logger.info(f"BSA Backward — mode=non-aligned, cases=non-aligned")
        logger.info(f"  dtype: {cfg.torch_dtype}  head_dim: {cfg.head_dim}  "
                    f"block_shape: ({cfg.block_shape_x}, {cfg.block_shape_y})")
        logger.info(f"  bwd atol={cfg.bwd_atol}, rtol={cfg.bwd_rtol}")
        logger.info("=" * 70 + "\n")

        passed, failed = _run_non_aligned_cases(NON_ALIGNED_CASES)
        logger.info("=" * 70)
        logger.info(f"Non-aligned Results: {passed}/{passed + failed} PASSED")
        logger.info("=" * 70)
        _print_perf_summary()
        return 0 if failed == 0 else 1

    cases = {"all": ALL_CASES, "quick": QUICK_CASES, "long": LONG_CASES}.get(args.cases, ALL_CASES)

    logger.info("\n" + "=" * 70)
    logger.info(f"BSA Backward — mode={args.mode}, cases={args.cases}")
    logger.info(f"  dtype: {cfg.torch_dtype}  head_dim: {cfg.head_dim}  "
                f"block_shape: ({cfg.block_shape_x}, {cfg.block_shape_y})")
    if args.mode == "precision":
        logger.info(f"  bwd atol={cfg.bwd_atol}, rtol={cfg.bwd_rtol}")
    if args.mode in ("perf", "perf-table"):
        logger.info(f"  BSA_RUNTIME_DEBUG_MODE=1")
        # Clean stale output dirs once before perf cases start.
        # Do NOT clean per-case — swimlane data is created during kernel
        # compilation (not per-invocation) and must persist across cases.
        _clean_output_dirs(_BWD_CLEAN_DIRS)
    elif args.mode == "perf-wall":
        logger.info(f"  Wall-clock timing only (no swimlane)")
    logger.info("=" * 70 + "\n")

    passed, failed, perf_rows = _run_cases(cases, args.mode)

    return _print_mode_results(passed, failed, perf_rows, args.mode)


if __name__ == "__main__":
    sys.exit(main())
