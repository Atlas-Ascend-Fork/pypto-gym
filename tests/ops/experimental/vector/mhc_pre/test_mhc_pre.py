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

"""PyPTO mhc_pre operator test.

MHC Pre-processing 算子精度验证测试

测试配置:
  - 测试 B*S = 1024, 2048, 4096
  - atol=0.0001, rtol=0.0078125
  - 使用 NPU 模式验证（若可用）

三态标记:
  - [PRECISION_PASS]: 精度验证通过
  - [PRECISION_FAIL]: 精度验证失败
  - 无标记 + exit ≠ 0: 功能问题
"""


import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

import argparse
from dataclasses import dataclass

import torch
import torch_npu  # noqa: F401

import pytest

# golden 文件从同级目录导入
from mhc_pre_golden import mhc_pre_golden
# impl 文件使用包导入方式（假设 pypto_gym 已通过 pip install -e . 安装）
from experimental.vector.mhc_pre.mhc_pre_impl import mhc_pre_wrapper


# 精度容差（来自 SPEC.md）
RTOL = 0.0078125
ATOL = 0.0001

# 失败个数比例阈值
MAX_ERROR_RATIO = 0.0001


# ─────────────────────────────────────────────
# 0. 精度对比工具
# ─────────────────────────────────────────────
@dataclass
class CompareConfig:
    """精度对比配置（封装容差与误差阈值相关参数）。"""
    atol: float
    rtol: float
    max_error_ratio: float = MAX_ERROR_RATIO
    max_error_count: int = 10


def compare(t: torch.Tensor, t_ref: torch.Tensor, name: str, config: CompareConfig):
    """比较两个张量的差异，超过阈值时打印错误点并抛出断言错误。

    Args:
        t: 待比较张量
        t_ref: 参考张量
        name: 张量名称（用于日志）
        config: 精度对比配置（atol / rtol / max_error_ratio / max_error_count）
    """
    atol = config.atol
    rtol = config.rtol
    max_error_ratio = config.max_error_ratio
    max_error_count = config.max_error_count

    def check_is_nan_inf():
        # 检测t中的NaN和Inf并直接报错
        nan_mask = torch.isnan(t)
        nan_count = nan_mask.sum().item()

        inf_mask = torch.isinf(t)
        inf_count = inf_mask.sum().item()

        if nan_count > 0 or inf_count > 0:
            error_msg = f"\n========== 张量 {name} 检测到非法值（禁止存在NaN/Inf）=========="

            if nan_count > 0:
                nan_positions = torch.nonzero(nan_mask, as_tuple=False)
                show_nan_count = min(nan_count, max_error_count)
                error_msg += f"\n- NaN数量：{nan_count}，前 {show_nan_count} 个位置："
                for i in range(show_nan_count):
                    pos_tuple = tuple(p.item() for p in nan_positions[i])
                    error_msg += f"\n  位置 {pos_tuple}"

            if inf_count > 0:
                inf_positions = torch.nonzero(inf_mask, as_tuple=False)
                show_inf_count = min(inf_count, max_error_count)
                error_msg += f"\n- Inf数量：{inf_count}，前 {show_inf_count} 个位置（值类型）："
                for i in range(show_inf_count):
                    pos = inf_positions[i]
                    pos_tuple = tuple(p.item() for p in pos)
                    inf_val = t[pos_tuple].item()
                    inf_type = "+Inf" if inf_val == float('inf') else "-Inf"
                    error_msg += f"\n  位置 {pos_tuple}：{inf_type}"
            error_msg += "\n" + "=" * 80 + "\n"

            assert False, error_msg

    check_is_nan_inf()

    # 验证张量基本属性一致
    assert t.shape == t_ref.shape, f"张量形状不一致：t.shape={t.shape}, t_ref.shape={t_ref.shape}"
    assert t.dtype == t_ref.dtype, f"张量数据类型不一致：t.dtype={t.dtype}, t_ref.dtype={t_ref.dtype}"
    assert t.device == t_ref.device, f"张量设备不一致：t.device={t.device}, t_ref.device={t_ref.device}"

    # 误差点数量阈值（按比例计算）
    error_count_threshold = round(max_error_ratio * t_ref.numel())

    # 计算误差掩码（超过阈值的位置为True）
    diff_abs = (t - t_ref).abs()
    tolerance = atol + rtol * t_ref.abs()
    diff_mask = diff_abs > tolerance
    error_count = diff_mask.sum().item()

    # 最大误差及其位置
    max_diff, flat_max_pos = torch.max(diff_abs.flatten(), dim=0)
    max_pos = torch.unravel_index(flat_max_pos, t.shape)
    max_pos = tuple(idx.item() for idx in max_pos)

    if error_count > 0:
        print(f"\n========== 张量 {name} 存在 {error_count} 个误差点（阈值：{error_count_threshold}）==========")

        error_positions = torch.nonzero(diff_mask, as_tuple=False)

        show_count = min(error_count, max_error_count)
        print(f"显示前 {show_count} 个误差点（位置 | 待比较值 | 参考值 | 绝对误差 | 允许阈值）：")

        for i in range(show_count):
            pos = error_positions[i]
            pos_tuple = tuple(p.item() for p in pos)
            t_val = t[pos_tuple].item()
            t_ref_val = t_ref[pos_tuple].item()
            diff_val = diff_abs[pos_tuple].item()
            tol_val = tolerance[pos_tuple].item()
            print(f"  位置 {pos_tuple}: {t_val:.8f} vs {t_ref_val:.8f} | 误差={diff_val:.8f} | 阈值={tol_val:.8f}")

        print(f"\n最大误差点：位置 {max_pos} | 误差={max_diff.item():.8f} | 阈值={tolerance[max_pos].item():.8f}")
        print("=" * 80 + "\n")

    assert error_count <= error_count_threshold, \
        (f"compare fail: {name}, max diff: {max_diff.item():.8f} at {max_pos}, "
         f"error_count: {error_count}, error_count_threshold: {error_count_threshold}")

    print("compare success !!!!")

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

def run_mhc_pre_test(bs, N, D, device_id=None, run_mode="npu", test_name=None):
    """通用测试函数：执行 mhc_pre 算子精度验证
    
    Args:
        bs: batch size
        N: 注意力头数
        D: 隐藏层维度
        device_id: NPU 设备 ID
        run_mode: 运行模式 ("npu" 或 "sim")
        test_name: 测试名称（可选，用于日志输出）
    """
    if test_name is None:
        test_name = f"B*S = {bs}"
    
    print("=" * 60)
    print(f"Test: mhc_pre {test_name}")
    print("=" * 60)

    # 设置设备
    if run_mode == "npu":
        if device_id is None:
            device_id = get_device_id()
        setup_npu(device_id)
        device = f"npu:{device_id}"
    else:
        device = "cpu"

    # 计算派生参数
    N_SQUARED_PLUS_2N = N * N + 2 * N

    # 固定随机种子以保证可重复性
    torch.manual_seed(1)

    # 准备测试数据
    x = torch.randn(bs, N, D, dtype=torch.bfloat16, device=device)
    phi = torch.randn(N_SQUARED_PLUS_2N, N * D, dtype=torch.float32, device=device)
    alpha = torch.randn(3, dtype=torch.float32, device=device)
    bias = torch.randn(N_SQUARED_PLUS_2N, dtype=torch.float32, device=device)

    # 执行 impl
    h_in_impl, h_post_impl, h_res_impl = mhc_pre_wrapper(x, phi, alpha, bias)

    # 执行 golden
    h_in_golden, h_post_golden, h_res_golden = mhc_pre_golden(x.cpu(), phi.cpu(), alpha.cpu(), bias.cpu())

    # 精度对比
    print(f"  Input shape : x {x.shape}, phi {phi.shape}")
    print(f"  Output shape: h_in {h_in_impl.shape}, h_post {h_post_impl.shape}, h_res {h_res_impl.shape}")

    # 三态判定
    if run_mode == "npu":
        cmp_cfg = CompareConfig(atol=ATOL, rtol=RTOL, max_error_ratio=MAX_ERROR_RATIO)
        try:
            compare(h_in_impl.cpu().float(), h_in_golden.cpu().float(), "h_in", cmp_cfg)
            compare(h_post_impl.cpu().float(), h_post_golden.cpu().float(), "h_post", cmp_cfg)
            compare(h_res_impl.cpu().float(), h_res_golden.cpu().float(), "h_res", cmp_cfg)
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"[PRECISION_FAIL] {e}", file=sys.stderr)
            raise
        except Exception as e:
            print(f"Runtime error: {e}", file=sys.stderr)
            raise

    print("  ✓ Passed\n")


@pytest.mark.soc("950", "910")
def test_mhc_pre_bs128_n4_d5120(device_id=None, run_mode="npu"):
    """测试 case: B*S = 128"""
    run_mhc_pre_test(bs=128, N=4, D=5120, device_id=device_id, run_mode=run_mode, test_name="B*S = 128")


@pytest.mark.soc("950", "910")
def test_mhc_pre_bs8(device_id=None, run_mode="npu"):
    """测试 case: B*S = 8"""
    run_mhc_pre_test(bs=8, N=4, D=128, device_id=device_id, run_mode=run_mode, test_name="B*S = 8")


@pytest.mark.soc("950", "910")
@pytest.mark.skip(reason="large test case")
def test_mhc_pre_bs256(device_id=None, run_mode="npu"):
    """测试 case: B*S = 256"""
    run_mhc_pre_test(bs=256, N=4, D=128, device_id=device_id, run_mode=run_mode, test_name="B*S = 256")


@pytest.mark.soc("950", "910")
def test_mhc_pre_bs1024(device_id=None, run_mode="npu"):
    """测试 case: B*S = 1024"""
    run_mhc_pre_test(bs=1024, N=4, D=5120, device_id=device_id, run_mode=run_mode, test_name="B*S = 1024")


@pytest.mark.soc("950", "910")
@pytest.mark.skip(reason="large test case")
def test_mhc_pre_bs4096(device_id=None, run_mode="npu"):
    """测试 case: B*S = 4096"""
    run_mhc_pre_test(bs=4096, N=4, D=2560, device_id=device_id, run_mode=run_mode, test_name="B*S = 4096")


# ─────────────────────────────────────────────
# 3. CLI 入口
# ─────────────────────────────────────────────

EXAMPLES = {
    "mhc_pre::test_mhc_pre_bs128_n4_d5120": {
        "name": "mhc_pre B*S = 128",
        "description": "泛化用例-1精度验证",
        "function": test_mhc_pre_bs128_n4_d5120,
    },
    "mhc_pre::test_mhc_pre_bs8": {
        "name": "mhc_pre B*S = 8",
        "description": "泛化用例-2精度验证",
        "function": test_mhc_pre_bs8,
    },
    "mhc_pre::test_mhc_pre_bs256": {
        "name": "mhc_pre B*S = 256",
        "description": "泛化用例-3精度验证",
        "function": test_mhc_pre_bs256,
    },
    "mhc_pre::test_mhc_pre_bs1024": {
        "name": "mhc_pre B*S = 1024",
        "description": "中等规模精度验证",
        "function": test_mhc_pre_bs1024,
    },
    "mhc_pre::test_mhc_pre_bs4096": {
        "name": "mhc_pre B*S = 4096",
        "description": "大规模精度验证",
        "function": test_mhc_pre_bs4096,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO mhc_pre operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s mhc_pre::test_mhc_pre_bs1024    Run B*S=1024 test
  %(prog)s --list                          List all cases
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