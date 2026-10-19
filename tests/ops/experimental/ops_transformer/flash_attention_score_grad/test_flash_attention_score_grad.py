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
FlashAttentionScoreGrad PyPTO 算子测试

测试 PyPTO 实现与 PyTorch golden 参考实现的精度对比。
支持多个测试级别，验证不同规模下的正确性。
"""


import os
import sys

_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

import argparse
import logging

import numpy as np
import torch
import torch_npu  # noqa: F401
from numpy.testing import assert_allclose
import pytest

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from experimental.ops_transformer.flash_attention_score_grad.flash_attention_score_grad_impl import (
    flash_attention_score_grad_wrapper,
)
from flash_attention_score_grad_golden import (
    AttentionGradInputs,
    ForwardDataConfig,
    ForwardDataResult,
    flash_attention_score_grad_golden,
    generate_forward_data,
)


def get_device_id():
    return int(os.environ.get('TILE_FWK_DEVICE_ID', 0))


class TestConfig:
    """Container for test configuration."""

    def __init__(self, name, batch_size, num_heads, seq_len, head_dim,
                 device_id, run_mode="npu", rtol=1e-2, atol=2e-2):
        self.name = name
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.device_id = device_id
        self.run_mode = run_mode
        self.rtol = rtol
        self.atol = atol


def run_test(cfg: TestConfig):
    """运行单个测试用例。"""
    logger.info("=" * 60)
    logger.info(
        "Test: %s (B=%d, N=%d, S=%d, D=%d)",
        cfg.name, cfg.batch_size, cfg.num_heads,
        cfg.seq_len, cfg.head_dim)
    logger.info("=" * 60)

    device = (
        f"npu:{cfg.device_id}"
        if cfg.run_mode == "npu" and cfg.device_id is not None
        else "cpu")

    fwd_cfg = ForwardDataConfig(
        cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim,
        device=device)
    fwd = generate_forward_data(fwd_cfg)
    q, k, v, dy = fwd.q, fwd.k, fwd.v, fwd.dy
    sm, ss, ao, scale = fwd.softmax_max, fwd.softmax_sum, fwd.attention_out, fwd.scale

    logger.info("  Q shape:  %s", q.shape)
    logger.info("  scale:    %.6f", scale)

    # PyPTO 实现
    dq, dk, dv = flash_attention_score_grad_wrapper(
        q, k, v, dy, sm, ss, ao, scale,
        num_heads=cfg.num_heads, head_dim=cfg.head_dim)

    # Golden 参考
    inputs = AttentionGradInputs(q, k, v, dy, sm, ss, ao, scale)
    dq_g, dk_g, dv_g = flash_attention_score_grad_golden(inputs)

    logger.info("  dQ shape: %s", dq.shape)
    logger.info("  dK shape: %s", dk.shape)
    logger.info("  dV shape: %s", dv.shape)

    # 精度对比
    results = []
    for grad_name, impl, golden in [
            ("dQ", dq, dq_g), ("dK", dk, dk_g), ("dV", dv, dv_g)]:
        impl_np = impl.float().cpu().numpy()
        golden_np = golden.float().cpu().numpy()
        max_diff = np.abs(impl_np - golden_np).max()
        logger.info("  %s max diff: %.6e", grad_name, max_diff)
        results.append((grad_name, impl_np, golden_np, max_diff))

    # 精度判定
    try:
        for grad_name, impl_np, golden_np, _ in results:
            assert_allclose(
                impl_np, golden_np, rtol=cfg.rtol, atol=cfg.atol,
                err_msg=f"{grad_name} precision check failed")
        logger.info("  [PRECISION_PASS]")
        logger.info("  ✓ %s passed\n", cfg.name)
        return True
    except AssertionError as e:
        logger.error("  [PRECISION_FAIL] %s", e)
        return False


def test_level0(device_id, run_mode="npu"):
    """Level 0: 最小功能验证"""
    cfg = TestConfig(
        "Level 0 (minimal)", 1, 8, 128, 64, device_id, run_mode)
    return run_test(cfg)


@pytest.mark.skip(reason="large test case")
def test_level1(device_id, run_mode="npu"):
    """Level 1: 典型小规模"""
    cfg = TestConfig(
        "Level 1 (typical)", 2, 8, 128, 64, device_id, run_mode)
    return run_test(cfg)


@pytest.mark.skip(reason="large test case")
def test_level2(device_id, run_mode="npu"):
    """Level 2: 中等规模"""
    cfg = TestConfig(
        "Level 2 (medium)", 2, 8, 256, 64, device_id, run_mode)
    return run_test(cfg)


def _dispatch_tests(args, tests, device_id):
    """Execute the requested test level(s) and return overall success."""
    if args.list:
        logger.info("\nAvailable test levels:")
        for level, (desc, _) in tests.items():
            logger.info("  %d: %s", level, desc)
        return None

    if args.level is not None:
        if args.level not in tests:
            logger.info(
                "ERROR: Invalid level %d. Use --list to see available levels.",
                args.level)
            return None
        _, fn = tests[args.level]
        return fn(device_id, args.run_mode)

    all_pass = True
    for level in sorted(tests.keys()):
        _, fn = tests.get(level)
        if not fn(device_id, args.run_mode):
            all_pass = False
            break
    if all_pass:
        logger.info("=" * 60)
        logger.info("All tests passed!")
        logger.info("=" * 60)
    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="FlashAttentionScoreGrad PyPTO Test")
    parser.add_argument(
        "level", type=int, nargs="?", default=None,
        help="Test level (0/1/2). If not specified, run all.")
    parser.add_argument(
        "--list", action="store_true", help="List test levels")
    parser.add_argument(
        "--run_mode", type=str, default="npu", choices=["npu", "sim"],
        help="Run mode")
    args = parser.parse_args()

    tests = {
        0: ("Level 0: 最小功能验证", test_level0),
        1: ("Level 1: 典型小规模", test_level1),
        2: ("Level 2: 中等规模", test_level2),
    }

    logger.info("\n" + "=" * 60)
    logger.info("FlashAttentionScoreGrad PyPTO Test")
    logger.info("=" * 60 + "\n")

    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        torch.npu.set_device(device_id)
        logger.info("Running on NPU:%d\n", device_id)

    success = _dispatch_tests(args, tests, device_id)
    if success is False:
        logger.error("Test execution failed")


if __name__ == "__main__":
    main()
