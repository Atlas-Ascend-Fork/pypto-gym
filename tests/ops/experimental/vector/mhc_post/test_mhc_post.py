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

"""PyPTO mhc_post operator test.

测试说明：
  - 本文件测试 mhc_post 算子的精度验证
  - golden 实现来自 mhc_post_golden.py
  - kernel 实现来自 mhc_post_impl.py
  - 精度对比使用 numpy.testing.assert_allclose
  - 输出三态标记：[PRECISION_PASS] 或 [PRECISION_FAIL]
"""


import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

import argparse

import torch
import torch_npu  # noqa: F401
import pypto
import numpy as np
from numpy.testing import assert_allclose

import pytest

# golden 文件从同级目录导入
from mhc_post_golden import mhc_post_golden
# impl 文件使用包导入方式（假设 pypto_gym 已通过 pip install -e . 安装）
from experimental.vector.mhc_post.mhc_post_impl import mhc_post_wrapper, MhcPostConfig

# 精度容差
RTOL = 0.0078125
ATOL = 0.0001

# ─────────────────────────────────────────────
# 1. 环境工具
# ─────────────────────────────────────────────


def get_device_id():
    return int(os.environ.get('TILE_FWK_DEVICE_ID', 0))


def setup_npu(device_id):
    """设置 NPU 设备。"""
    torch.npu.set_device(device_id)


# ─────────────────────────────────────────────
# 2. 测试函数
# ─────────────────────────────────────────────

def _run_golden_and_compare(x, h_res, h_out, h_post, result, bs, N, D, run_mode):
    """Execute golden reference on CPU and compare precision with kernel result."""
    B = 1
    S = bs

    # 执行 golden（在 CPU 上）
    x_cpu = x.cpu()
    h_res_cpu = h_res.cpu()
    h_out_cpu = h_out.cpu()
    h_post_cpu = h_post.cpu()

    # golden 使用 [B*S, N, D] 格式，需要 reshape
    x_golden = x_cpu.view(bs, N, D).contiguous()
    h_res_golden = h_res_cpu.view(bs, N, N).contiguous()
    h_out_golden = h_out_cpu.view(bs, D).contiguous()
    h_post_golden_input = h_post_cpu.view(bs, N).contiguous()

    golden_output = mhc_post_golden(x_golden, h_res_golden, h_out_golden, h_post_golden_input)
    golden_output = golden_output.view(B, S, N, D)  # reshape 回 [B, S, N, D]

    # 精度对比
    print(f"  Input x shape    : {x.shape}")
    print(f"  Input h_res shape: {h_res.shape}")
    print(f"  Input h_out shape: {h_out.shape}")
    print(f"  Input h_post shape: {h_post.shape}")
    print(f"  Output shape     : {result.shape}")

    # BF16 数据转 FP32 再转 numpy（避免 dtype 问题）
    result_np = result.cpu().float().numpy()
    golden_np = golden_output.float().numpy()

    max_diff = np.abs(result_np - golden_np).max()
    mean_diff = np.abs(result_np - golden_np).mean()
    print(f"  Max diff         : {max_diff:.6e}")
    print(f"  Mean diff        : {mean_diff:.6e}")

    # 三态判定
    if run_mode == "npu":
        try:
            assert_allclose(result_np, golden_np, rtol=RTOL, atol=ATOL)
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"[PRECISION_FAIL] {e}", file=sys.stderr)
            raise
        except Exception as e:
            print(f"Runtime error: {e}", file=sys.stderr)
            raise

    print("  ✓ Passed\n")


def run_mhc_post_test(bs, N, D, device_id=None, run_mode="npu", test_name=None):
    """通用测试函数：执行 mhc_post 算子精度验证

    Args:
        bs: batch size (B*S)
        N: 注意力头数
        D: 隐藏层维度
        device_id: NPU 设备 ID
        run_mode: 运行模式 ("npu" 或 "sim")
        test_name: 测试名称（可选，用于日志输出）
    """
    if test_name is None:
        test_name = f"B*S = {bs}"

    print("=" * 60)
    print(f"Test: mhc_post {test_name}")
    print("=" * 60)

    # 设置设备
    if run_mode == "npu":
        if device_id is None:
            device_id = get_device_id()
        setup_npu(device_id)
        device = f"npu:{device_id}"
    else:
        device = "cpu"

    # 固定随机种子以保证可重复性
    torch.manual_seed(42)

    # 准备测试数据（使用 [B, S, N, D] 格式，wrapper 会 reshape）
    B = 1
    S = bs

    x = torch.randn(B, S, N, D, dtype=torch.bfloat16, device=device)
    h_res = torch.randn(B, S, N, N, dtype=torch.float32, device=device)
    h_out = torch.randn(B, S, D, dtype=torch.bfloat16, device=device)
    h_post = torch.randn(B, S, N, dtype=torch.float32, device=device)

    if pypto.platform.npuarch == 'DAV_3510':
        if bs <= 32:
            config = MhcPostConfig(vec_nbuffer=1)
        elif bs <= 64:
            config = MhcPostConfig(vec_nbuffer=2)
        else:
            config = MhcPostConfig(vec_nbuffer=4)
    else:
        if bs <= 32:
            config = MhcPostConfig(vec_nbuffer=1)
        elif bs <= 64:
            config = MhcPostConfig(vec_nbuffer=4)
        else:
            config = MhcPostConfig(vec_nbuffer=8)

    # 执行 kernel wrapper
    result = mhc_post_wrapper(x, h_res, h_out, h_post, config)

    _run_golden_and_compare(x, h_res, h_out, h_post, result, bs, N, D, run_mode)


@pytest.mark.soc("950", "910")
def test_mhc_post_bs8_n4_d128(device_id=None, run_mode="npu"):
    run_mhc_post_test(bs=8, N=4, D=128, device_id=device_id, run_mode=run_mode, test_name="B*S = 8, N = 4, D = 128")


@pytest.mark.soc("950", "910")
def test_mhc_post_bs256_n4_d128(device_id=None, run_mode="npu"):
    run_mhc_post_test(bs=256, N=4, D=128, device_id=device_id, run_mode=run_mode, test_name="B*S = 256, N = 4, D = 128")


@pytest.mark.soc("950", "910")
@pytest.mark.skip(reason="large test case")
def test_mhc_post_bs1024_n4_d5120(device_id=None, run_mode="npu"):
    run_mhc_post_test(
    bs=1024,
    N=4,
    D=5120,
    device_id=device_id,
    run_mode=run_mode,
     test_name="B*S = 1024, N = 4, D = 5120")


@pytest.mark.soc("950", "910")
@pytest.mark.skip(reason="large test case")
def test_mhc_post_bs4096_n4_d2560(device_id=None, run_mode="npu"):
    run_mhc_post_test(
    bs=4096,
    N=4,
    D=2560,
    device_id=device_id,
    run_mode=run_mode,
     test_name="B*S = 4096, N = 4, D = 2560")


# ─────────────────────────────────────────────
# 4. CLI 入口
# ─────────────────────────────────────────────

EXAMPLES = {
    "mhc_post::test_mhc_post_bs8_n4_d128": {
        "name": "mhc_post B*S = 8, N = 4, D = 128",
        "description": "最小规模精度验证",
        "function": test_mhc_post_bs8_n4_d128,
    },
    "mhc_post::test_mhc_post_bs256_n4_d128": {
        "name": "mhc_post B*S = 256, N = 4, D = 128",
        "description": "小规模精度验证",
        "function": test_mhc_post_bs256_n4_d128,
    },
    "mhc_post::test_mhc_post_bs1024_n4_d5120": {
        "name": "mhc_post B*S = 1024, N = 4, D = 5120",
        "description": "基础精度验证（大 D）",
        "function": test_mhc_post_bs1024_n4_d5120,
    },
    "mhc_post::test_mhc_post_bs4096_n4_d2560": {
        "name": "mhc_post B*S = 4096, N = 4, D = 2560",
        "description": "大规模精度验证",
        "function": test_mhc_post_bs4096_n4_d2560,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO mhc_post operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s mhc_post::test_mhc_post_bs1024_n4_d2560    Run B*S=1024 test
  %(prog)s --list                                     List all cases
        """,
    )
    parser.add_argument("example_id", type=str, nargs="?", help="Case ID to run")
    parser.add_argument("--list", action="store_true", help="List available cases")
    parser.add_argument(
        "--run_mode", "--run-mode",
        type=str, default="npu", choices=["npu", "sim"],
        help="Run mode (default: npu)",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable cases:\n")
        for key, info in sorted(EXAMPLES.items()):
            print(f"  {key}  — {info['description']}")
        return

    if args.example_id:
        if args.example_id not in EXAMPLES:
            print(f"ERROR: unknown case '{args.example_id}'")
            print(f"Valid: {', '.join(sorted(EXAMPLES))}")
            raise RuntimeError("Test execution failed")
        to_run = [(args.example_id, EXAMPLES[args.example_id])]
    else:
        to_run = list(sorted(EXAMPLES.items()))

    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()

    try:
        for key, info in to_run:
            print(f"\n▸ Running {key}: {info['name']}")
            info["function"](device_id, args.run_mode)
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()