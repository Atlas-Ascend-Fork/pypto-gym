#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------
import argparse
import logging
import os
from dataclasses import dataclass

import numpy as np
import pypto
import torch
from numpy.testing import assert_allclose

try:
    import torch_npu
except ImportError:
    torch_npu = None


LOGGER = logging.getLogger(__name__)

FP32_ALIGN = 8
CHANNEL_TILE = 16
DEFAULT_TEST_SHAPES = [
    (2400, 88, 1, 1),
    (400, 8, 1, 1),
    (800, 40, 2, 1),
    (900, 45, 1, 1),
]


@dataclass(frozen=True)
class BnReduceConfig:
    tile_n: int
    tile_c: int
    tile_hw: int
    tile_reduce: int
    out_tile_c: int


def get_device_id() -> int:
    return int(os.environ.get('TILE_FWK_DEVICE_ID', 0))


def align_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def select_bn_reduce_tiles(shape: tuple[int, int, int, int]) -> BnReduceConfig:
    n, c, h, w = shape
    hw = h * w
    tile_hw = max(FP32_ALIGN, align_up(hw, FP32_ALIGN))
    target_flat_tile = 512 if n >= 512 else 256
    tile_n = max(32, target_flat_tile // tile_hw)
    if tile_n >= 128:
        tile_n = 128
    elif tile_n >= 64:
        tile_n = 64
    else:
        tile_n = 32
    tile_reduce = tile_n * tile_hw
    out_tile_c = 64 if c > 64 else 16
    return BnReduceConfig(tile_n, CHANNEL_TILE, tile_hw, tile_reduce, out_tile_c)


@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "stitch_function_max_num": 128,
    }
)
def bn_reduce_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    sum_out: pypto.Tensor([], pypto.DT_FP32),
    sq_sum_out: pypto.Tensor([], pypto.DT_FP32),
    config: BnReduceConfig,
):
    n, c, h, w = x.shape
    reduce_axis_size = n * h * w

    v1 = pypto.reshape(x, [n, c, h * w], inplace=True)
    pypto.set_vec_tile_shapes(config.tile_n, config.tile_c, config.tile_hw)
    v1_sq = v1 * v1

    v2 = pypto.transpose(v1, 1, 2)
    v2_sq = pypto.transpose(v1_sq, 1, 2)
    v3 = pypto.reshape(v2, [reduce_axis_size, c], inplace=True)
    v3_sq = pypto.reshape(v2_sq, [reduce_axis_size, c], inplace=True)

    pypto.set_vec_tile_shapes(config.tile_reduce, config.tile_c)
    sum_raw = pypto.sum(v3, dim=0, keepdim=True)
    sq_sum_raw = pypto.sum(v3_sq, dim=0, keepdim=True)

    pypto.set_vec_tile_shapes(1, config.out_tile_c, 1, 8)
    sum_out.move(pypto.reshape(sum_raw, [1, c, 1, 1], inplace=True))
    sq_sum_out.move(pypto.reshape(sq_sum_raw, [1, c, 1, 1], inplace=True))


def prepare_device(run_mode: str) -> str:
    if run_mode != "npu":
        raise ValueError(f"Unsupported run_mode: {run_mode}")
    device_id = get_device_id()
    if torch_npu is None:
        raise RuntimeError("torch_npu is required when TILE_FWK_DEVICE_ID is set.")
    torch.npu.set_device(device_id)
    return f"npu:{device_id}"


def run_bn_reduce_case(case_index: int, shape: tuple[int, int, int, int], device: str) -> None:
    config = select_bn_reduce_tiles(shape)
    x_torch = torch.rand(shape, dtype=torch.float32, device=device)

    channel_size = shape[1]
    output_shape = (1, channel_size, 1, 1)
    sum_out = torch.empty(output_shape, dtype=torch.float32, device=device)
    sq_sum_out = torch.empty(output_shape, dtype=torch.float32, device=device)

    bn_reduce_kernel(x_torch, sum_out, sq_sum_out, config)

    golden_sum = torch.sum(x_torch, dim=(0, 2, 3), keepdim=True)
    golden_sq_sum = torch.sum(x_torch * x_torch, dim=(0, 2, 3), keepdim=True)
    actual_sum_np = sum_out.cpu().numpy()
    golden_sum_np = golden_sum.cpu().numpy()
    actual_sq_sum_np = sq_sum_out.cpu().numpy()
    golden_sq_sum_np = golden_sq_sum.cpu().numpy()

    sum_diff = np.max(np.abs(actual_sum_np - golden_sum_np))
    sq_sum_diff = np.max(np.abs(actual_sq_sum_np - golden_sq_sum_np))
    assert_allclose(actual_sum_np, golden_sum_np, rtol=1e-3, atol=1e-3)
    assert_allclose(actual_sq_sum_np, golden_sq_sum_np, rtol=1e-3, atol=1e-3)
    LOGGER.info("Case %s: sum_diff=%.6e, sq_sum_diff=%.6e", case_index, sum_diff, sq_sum_diff)


def summarize_failed_cases(failed_cases: list[tuple[int, tuple[int, int, int, int]]]) -> None:
    LOGGER.info("")
    LOGGER.info("%s Test Summary %s", "=" * 20, "=" * 20)
    if not failed_cases:
        LOGGER.info("All tests passed.")
        return
    LOGGER.warning("⚠️ %s failed test cases:", len(failed_cases))
    for case_index, shape in failed_cases:
        LOGGER.warning("   - Case %s: %s", case_index, shape)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_mode", type=str, default="npu", choices=["npu"])
    args = parser.parse_args()

    device = prepare_device(args.run_mode)
    failed_cases: list[tuple[int, tuple[int, int, int, int]]] = []
    for case_index, shape in enumerate(DEFAULT_TEST_SHAPES, start=1):
        try:
            run_bn_reduce_case(case_index, shape, device)
        except Exception as exc:
            LOGGER.error("  ✗ Case %s Failed: %s", case_index, exc)
            failed_cases.append((case_index, shape))
    summarize_failed_cases(failed_cases)


if __name__ == "__main__":
    main()
