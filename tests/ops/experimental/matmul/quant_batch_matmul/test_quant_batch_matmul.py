# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------



import argparse
import os
import random
import sys
from dataclasses import dataclass

import pytest
import torch
import torch_npu  # noqa: F401  # must import before pypto kernel collection
from numpy.testing import assert_allclose

_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, "src")):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, "ops", "pypto_tensor"))

import pypto
from experimental.matmul.quant_batch_matmul.quant_batch_matmul_impl import (
    QuantBatchMatmulConfig,
    QuantBatchMatmulInputs,
    get_x1_scale_shape,
    get_x1_shape,
    get_x2_scale_shape,
    get_x2_shape,
    quant_batch_matmul,
)
from quant_batch_matmul_golden import gen_golden


@dataclass(frozen=True)
class MatmulCase:
    batch: int
    m: int
    k: int
    n: int
    with_x1_scale: bool
    description: str


# Per-point relative error: |result - golden| / max(|golden|, 1) <= 1e-3
_PER_POINT_REL_RTOL = 1e-3

MATMUL_CASES = [
    MatmulCase(4, 4, 7168, 2048, True, "F-T FP8E4M3->INT8"),
    MatmulCase(8, 128, 7168, 2048, True, "F-T FP8E4M3->INT8"),
    MatmulCase(16, 8, 6144, 2048, True, "F-T FP8E4M3->INT8"),
    MatmulCase(32, 64, 6144, 2048, True, "F-T FP8E4M3->INT8"),
]

# Set by main() before each kernel run; used in precision failure logs.
_CURRENT_CASE_LOG = ""


def _case_tag(case_idx: int, total: int) -> str:
    return f"[case {case_idx:02d}/{total:02d}]"


def _case_shape(case: MatmulCase) -> str:
    x1_scale = "yes" if case.with_x1_scale else "null"
    return f"batch={case.batch} m={case.m} k={case.k} n={case.n} x1Scale={x1_scale}"


def _case_tiling(case: MatmulCase) -> str:
    m_tile = _cube_tile_m(case.m)
    return f"m_tile={m_tile} k_tile=[128,512] n_tile=[512,512] n_block=512 vec_tile=[{m_tile[-1]}, 512]"


def _print_case_plan(cases: list[MatmulCase]) -> None:
    total = len(cases)
    print(f"cases: {total}")
    for idx, case in enumerate(cases, start=1):
        print(f"  {idx:02d}. {_case_shape(case)}")


def _cube_tile_m(m: int) -> list[int]:
    """Cube M tile; kernel LOOP_M uses m_tile[-1] as block size (keep small for correctness)."""
    if m >= 4:
        return [4, 4]
    return [4, 4]


def _torch_fp8_dtype():
    return getattr(torch, "float8_e4m3fn", torch.float16)


def _make_config(case: MatmulCase) -> QuantBatchMatmulConfig:
    n_block = 512
    m_tile = _cube_tile_m(case.m)
    # n_tile 第 0 维须 >= n_block；若与 m_tile 同为 128，cube 输出 N 可能被截断为 128
    return QuantBatchMatmulConfig(
        ori_shape=[case.batch, case.m, case.k, case.n],
        m_tile_shape=m_tile,
        k_tile_shape=[128, 512],
        n_tile_shape=[n_block, n_block],
        vector_tile_shape=[4, n_block],
        n_block_size=n_block,
        in_dtype=pypto.DT_FP8E4M3,
        out_dtype=pypto.DT_INT8,
        description=case.description,
    )


def _make_inputs(config: QuantBatchMatmulConfig, with_x1_scale: bool) -> QuantBatchMatmulInputs:
    torch.manual_seed(0)
    fp8_dtype = _torch_fp8_dtype()
    x1 = torch.randn(get_x1_shape(config), dtype=torch.float32).uniform_(-1.0, 1.0).to(fp8_dtype)
    x2 = torch.randn(get_x2_shape(config), dtype=torch.float32).uniform_(-1.0, 1.0).to(fp8_dtype)
    x1_scale = None
    if with_x1_scale:
        x1_scale = torch.randn(get_x1_scale_shape(), dtype=torch.float32).uniform_(0.25, 1.0)
    x2_scale = torch.randn(get_x2_scale_shape(), dtype=torch.float32).uniform_(0.25, 1.0)
    return QuantBatchMatmulInputs(x1=x1, x2=x2, x1_scale=x1_scale, x2_scale=x2_scale)


@pytest.mark.soc("950", "910")
@pytest.mark.parametrize("case", MATMUL_CASES)
def test_golden_quant_batch_matmul_layouts(case):
    config = _make_config(case)
    inputs = _make_inputs(config, case.with_x1_scale)
    golden = gen_golden(inputs, config)

    assert tuple(golden.shape) == (case.batch, case.m, case.n)
    assert golden.dtype == torch.int8


@pytest.mark.soc("950")
@pytest.mark.parametrize("case", MATMUL_CASES)
def test_quant_batch_matmul(case):
    config = _make_config(case)
    inputs = _make_inputs(config, case.with_x1_scale)

    golden = gen_golden(inputs, config)
    result = quant_batch_matmul(inputs, config)

    result_f = result.cpu().float().numpy()
    golden_f = golden.cpu().float().numpy()
    diff = abs(result_f - golden_f)
    rel_err = diff / (abs(golden_f) + 1e-6)
    if rel_err.size and rel_err.max() > _PER_POINT_REL_RTOL:
        prefix = f"{_CURRENT_CASE_LOG} " if _CURRENT_CASE_LOG else ""
        print(
            f"{prefix}precision stats: max_abs={diff.max():.6f}, "
            f"max_rel={rel_err.max():.6f}, "
            f"mismatch_ratio={(rel_err > _PER_POINT_REL_RTOL).mean():.4f} "
            f"(threshold={_PER_POINT_REL_RTOL})"
        )
    assert_allclose(result_f, golden_f, rtol=_PER_POINT_REL_RTOL, atol=0)


def _run_golden_case(case_idx: int, total: int, case: MatmulCase) -> None:
    tag = _case_tag(case_idx, total)
    print(f"{tag} [golden] START | {_case_shape(case)}")
    test_golden_quant_batch_matmul_layouts(case)
    print(f"{tag} [golden] PASSED")


def _run_kernel_case(case_idx: int, total: int, case: MatmulCase) -> None:
    global _CURRENT_CASE_LOG
    tag = _case_tag(case_idx, total)
    _CURRENT_CASE_LOG = f"{tag} {_case_shape(case)}"
    print(f"{tag} [kernel] START | {_case_shape(case)} | {_case_tiling(case)}")
    print(f"{tag} [kernel] running (compile monitor / swimlane may follow) ...")
    test_quant_batch_matmul(case)
    print(f"{tag} [kernel] PASSED")
    _CURRENT_CASE_LOG = ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Quant batch matmul F-T FP8E4M3->INT8 tests")
    parser.add_argument(
        "--golden-only",
        action="store_true",
        help="Only run CPU golden reference checks (no NPU kernel)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable swimlane: compile monitor + runtime_debug_mode (see output/)",
    )
    args = parser.parse_args()

    if args.profile:
        pypto.set_host_options(
            compile_monitor_enable=1,
            compile_monitor_print_interval=2,
        )
        pypto.set_debug_options(runtime_debug_mode=1, compile_debug_mode=1)

    cases = MATMUL_CASES

    total = len(cases)
    print("=" * 60)
    print("quant_batch_matmul FP8E4M3 tests")
    print("=" * 60)
    _print_case_plan(cases)
    print("=" * 60)

    for case_idx, case in enumerate(cases, start=1):
        if case_idx > 1:
            print("-" * 60)
        _run_golden_case(case_idx, total, case)
        if not args.golden_only:
            _run_kernel_case(case_idx, total, case)

    print("=" * 60)
    print(f"ALL PASSED ({total} case(s))")
    print("=" * 60)


if __name__ == "__main__":
    main()
