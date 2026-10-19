# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""PyPTO scatter_pa_kv_cache operator test.

测试说明：
  - 本文件测试 scatter_pa_kv_cache 算子的精度验证
  - golden 实现来自 scatter_pa_kv_cache_golden.py
  - kernel 实现来自 scatter_pa_kv_cache_impl.py
  - 精度对比使用 numpy.testing.assert_allclose
  - 输出三态标记：[PRECISION_PASS] 或 [PRECISION_FAIL]
"""

import os
import sys

# 设置 Python 路径，使 experimental 包可导入
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

import argparse
import pytest

# 注意：torch、numpy、golden、impl 等模块将在测试函数中延迟导入
# 这样可以避免在 --list 或 --help 时卡住

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
    import torch_npu  # noqa: F401
    import torch
    torch.npu.set_device(device_id)


# ─────────────────────────────────────────────
# 2. 测试函数
# ─────────────────────────────────────────────

def _create_test_tensors(num_tokens, num_blocks, block_size, num_heads, head_size, seed, device):
    """Create input tensors for the test."""
    import torch
    torch.manual_seed(seed)
    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16, device=device)
    key_cache = torch.randn(
        num_blocks, block_size, num_heads, head_size, dtype=torch.bfloat16, device=device)

    max_slots = num_blocks * block_size
    if num_tokens <= max_slots:
        slot_mapping = torch.randperm(max_slots)[:num_tokens].to(torch.int32).to(device)
    else:
        slot_mapping = torch.randint(0, max_slots, (num_tokens,), dtype=torch.int32, device=device)

    value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16, device=device)
    value_cache = torch.randn(
        num_blocks, block_size, num_heads, head_size, dtype=torch.bfloat16, device=device)

    return key, key_cache, slot_mapping, value, value_cache


def _run_kernel(key, key_cache, slot_mapping, value, value_cache):
    """Execute the scatter_pa_kv_cache kernel."""
    from experimental.vector.scatter_pa_kv_cache.scatter_pa_kv_cache_impl import scatter_pa_kv_cache_wrapper
    return scatter_pa_kv_cache_wrapper(key, key_cache, slot_mapping, value, value_cache)


def _compute_golden(key_cache_orig, value_cache_orig, key_cpu, value_cpu,
                    slot_mapping_cpu, block_size):
    """Compute golden reference results for key_cache and value_cache."""
    import gc

    block_indices = slot_mapping_cpu // block_size
    block_offsets = slot_mapping_cpu % block_size

    golden_key_cache = key_cache_orig.clone()
    golden_key_cache[block_indices, block_offsets, :, :] = key_cpu
    del key_cache_orig, key_cpu
    gc.collect()

    golden_value_cache = value_cache_orig.clone()
    golden_value_cache[block_indices, block_offsets, :, :] = value_cpu
    del value_cache_orig, value_cpu, block_indices, block_offsets, slot_mapping_cpu
    gc.collect()

    return golden_key_cache, golden_value_cache


def _move_kernel_inputs_to_cpu(key, slot_mapping, value, run_mode):
    """Move kernel inputs to CPU and release NPU memory."""
    import gc
    key_cpu = key.cpu()
    slot_mapping_cpu = slot_mapping.cpu()
    value_cpu = value.cpu()
    del key, slot_mapping, value
    gc.collect()
    if run_mode == "npu":
        import torch
        torch.npu.empty_cache()
    return key_cpu, slot_mapping_cpu, value_cpu


def _move_kernel_outputs_to_cpu(result_key_cache, result_value_cache, key_cache, value_cache, run_mode):
    """Move kernel outputs to CPU and release NPU memory."""
    import gc
    result_key_cache_cpu = result_key_cache.cpu()
    result_value_cache_cpu = result_value_cache.cpu()
    del result_key_cache, result_value_cache, key_cache, value_cache
    gc.collect()
    if run_mode == "npu":
        import torch
        torch.npu.empty_cache()
    return result_key_cache_cpu, result_value_cache_cpu


def _log_shapes(num_tokens, num_heads, head_size, num_blocks, block_size,
                result_key_cache_cpu, result_value_cache_cpu):
    """Log input/output shapes."""
    print(f"  Input shapes:")
    print(f"    key:           torch.Size([{num_tokens}, {num_heads}, {head_size}])")
    print(f"    key_cache:     torch.Size([{num_blocks}, {block_size}, {num_heads}, {head_size}])")
    print(f"    slot_mapping:  torch.Size([{num_tokens}])")
    print(f"    value:         torch.Size([{num_tokens}, {num_heads}, {head_size}])")
    print(f"    value_cache:   torch.Size([{num_blocks}, {block_size}, {num_heads}, {head_size}])")
    print(f"  Output shapes:")
    print(f"    key_cache:     {result_key_cache_cpu.shape}")
    print(f"    value_cache:   {result_value_cache_cpu.shape}")


def _compare_results(result_key_cache_cpu, result_value_cache_cpu,
                     golden_key_cache, golden_value_cache, run_mode):
    """Compare kernel results with golden and report precision."""
    import numpy as np
    import gc
    from numpy.testing import assert_allclose

    result_key_np = result_key_cache_cpu.float().numpy()
    del result_key_cache_cpu
    gc.collect()
    golden_key_np = golden_key_cache.float().numpy()
    del golden_key_cache
    gc.collect()

    result_value_np = result_value_cache_cpu.float().numpy()
    del result_value_cache_cpu
    gc.collect()
    golden_value_np = golden_value_cache.float().numpy()
    del golden_value_cache
    gc.collect()

    max_diff_key = np.abs(result_key_np - golden_key_np).max()
    mean_diff_key = np.abs(result_key_np - golden_key_np).mean()
    max_diff_value = np.abs(result_value_np - golden_value_np).max()
    mean_diff_value = np.abs(result_value_np - golden_value_np).mean()
    print(f"  Max diff (key_cache):    {max_diff_key:.6e}")
    print(f"  Mean diff (key_cache):   {mean_diff_key:.6e}")
    print(f"  Max diff (value_cache):  {max_diff_value:.6e}")
    print(f"  Mean diff (value_cache): {mean_diff_value:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(result_key_np, golden_key_np, rtol=RTOL, atol=ATOL)
            assert_allclose(result_value_np, golden_value_np, rtol=RTOL, atol=ATOL)
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"[PRECISION_FAIL] {e}", file=sys.stderr)
            raise
        except Exception as e:
            print(f"Runtime error: {e}", file=sys.stderr)
            raise

    print("  \u2713 Passed\n")


def run_scatter_pa_kv_cache_test(
    num_tokens, num_blocks, block_size, num_heads, head_size,
    seed=42,
    device_id=None, run_mode="npu", test_name=None
):
    """通用测试函数：执行 scatter_pa_kv_cache 算子精度验证

    Args:
        num_tokens: 当前 step 处理的 token 数量
        num_blocks: cache 中的总块数
        block_size: 每个块的元素数
        num_heads: head 数量
        head_size: 每个 head 的维度
        seed: 随机种子
        device_id: NPU 设备 ID
        run_mode: 运行模式 ("npu" 或 "sim")
        test_name: 测试名称（可选，用于日志输出）
    """
    import torch
    import gc

    if test_name is None:
        test_name = (
            f"num_tokens={num_tokens}, num_blocks={num_blocks}, "
            f"block_size={block_size}, num_heads={num_heads}, head_size={head_size}"
        )

    print("=" * 60)
    print(f"Test: scatter_pa_kv_cache {test_name}")
    print("=" * 60)

    if run_mode == "npu":
        if device_id is None:
            device_id = get_device_id()
        setup_npu(device_id)
        device = f"npu:{device_id}"
    else:
        device = "cpu"

    key, key_cache, slot_mapping, value, value_cache = _create_test_tensors(
        num_tokens, num_blocks, block_size, num_heads, head_size, seed, device)

    # 保存原始 cache 用于 golden 对比（在 CPU 端保存副本，减少峰值内存）
    key_cache_orig = key_cache.cpu().clone()
    value_cache_orig = value_cache.cpu().clone()

    result_key_cache, result_value_cache = _run_kernel(key, key_cache, slot_mapping, value, value_cache)

    key_cpu, slot_mapping_cpu, value_cpu = _move_kernel_inputs_to_cpu(key, slot_mapping, value, run_mode)

    golden_key_cache, golden_value_cache = _compute_golden(
        key_cache_orig, value_cache_orig, key_cpu, value_cpu, slot_mapping_cpu, block_size)

    result_key_cache_cpu, result_value_cache_cpu = _move_kernel_outputs_to_cpu(
        result_key_cache, result_value_cache, key_cache, value_cache, run_mode)

    # 日志输出形状
    _log_shapes(num_tokens, num_heads, head_size, num_blocks, block_size,
                result_key_cache_cpu, result_value_cache_cpu)

    _compare_results(result_key_cache_cpu, result_value_cache_cpu,
                     golden_key_cache, golden_value_cache, run_mode)


# ─────────────────────────────────────────────
# 3. Level 0 测试：SPEC.md P0 配置
# ─────────────────────────────────────────────

@pytest.mark.soc("950", "910")
def test_config1_performance_p0(device_id=None, run_mode="npu"):
    """配置1_性能P0: num_tokens=8, 适中序列长度性能测试。"""
    run_scatter_pa_kv_cache_test(
        num_tokens=8, num_blocks=512, block_size=128, num_heads=1, head_size=512,
        seed=42, device_id=device_id, run_mode=run_mode,
        test_name="num_tokens=8, num_blocks=512, block_size=128, num_heads=1, head_size=512 (配置1_性能P0)",
    )


@pytest.mark.soc("950", "910")
@pytest.mark.skip(reason="large test case")
def test_config2_function_p0(device_id=None, run_mode="npu"):
    """配置2_功能P0: num_tokens=128, 较长序列功能验证。"""
    run_scatter_pa_kv_cache_test(
        num_tokens=128, num_blocks=512, block_size=128, num_heads=4, head_size=128,
        seed=42, device_id=device_id, run_mode=run_mode,
        test_name="num_tokens=128, num_blocks=512, block_size=128, num_heads=4, head_size=128 (配置2_功能P0)",
    )


@pytest.mark.soc("950", "910")
@pytest.mark.skip(reason="large test case")
def test_config3_boundary_p0(device_id=None, run_mode="npu"):
    """配置3_边界P0: num_tokens=4096, 最大序列长度边界测试。"""
    run_scatter_pa_kv_cache_test(
        num_tokens=4096, num_blocks=512, block_size=128, num_heads=64, head_size=64,
        seed=42, device_id=device_id, run_mode=run_mode,
        test_name="num_tokens=4096, num_blocks=512, block_size=128, num_heads=64, head_size=64 (配置3_边界P0)",
    )


# ─────────────────────────────────────────────
# 4. Level 1 测试：小规模 & 泛化验证
# ─────────────────────────────────────────────

@pytest.mark.soc("950", "910")
def test_small_scale(device_id=None, run_mode="npu"):
    """小规模验证: num_tokens=8, 确保基础功能正确。"""
    run_scatter_pa_kv_cache_test(
        num_tokens=8, num_blocks=100, block_size=128, num_heads=2, head_size=256,
        seed=100, device_id=device_id, run_mode=run_mode,
        test_name="num_tokens=8, num_blocks=100 (小规模验证)",
    )


@pytest.mark.soc("950", "910")
@pytest.mark.skip(reason="large test case")
def test_min_tokens(device_id=None, run_mode="npu"):
    """最小 tokens: num_tokens=1, 边界最低值测试。"""
    run_scatter_pa_kv_cache_test(
        num_tokens=1, num_blocks=100, block_size=128, num_heads=2, head_size=256,
        seed=200, device_id=device_id, run_mode=run_mode,
        test_name="num_tokens=1, num_blocks=100 (最小 tokens)",
    )


@pytest.mark.soc("950", "910")
@pytest.mark.skip(reason="large test case")
def test_medium_scale(device_id=None, run_mode="npu"):
    """中等规模: num_tokens=500, 中间范围验证。"""
    run_scatter_pa_kv_cache_test(
        num_tokens=500, num_blocks=500, block_size=128, num_heads=2, head_size=256,
        seed=300, device_id=device_id, run_mode=run_mode,
        test_name="num_tokens=500, num_blocks=500 (中等规模)",
    )


# ─────────────────────────────────────────────
# 5. CLI 入口
# ─────────────────────────────────────────────

EXAMPLES = {
    "scatter_pa_kv_cache::test_config1_performance_p0": {
        "name": "scatter_pa_kv_cache 配置1_性能P0",
        "description": "test_config1_performance_p0 (num_tokens=2633, num_blocks=100)",
        "function": test_config1_performance_p0,
    },
    "scatter_pa_kv_cache::test_config2_function_p0": {
        "name": "scatter_pa_kv_cache 配置2_功能P0",
        "description": "test_config2_function_p0 (num_tokens=7902, num_blocks=200)",
        "function": test_config2_function_p0,
    },
    "scatter_pa_kv_cache::test_config3_boundary_p0": {
        "name": "scatter_pa_kv_cache 配置3_边界P0",
        "description": "test_config3_boundary_p0 (num_tokens=16384, num_blocks=200)",
        "function": test_config3_boundary_p0,
    },
    "scatter_pa_kv_cache::test_small_scale": {
        "name": "scatter_pa_kv_cache 小规模验证",
        "description": "test_small_scale",
        "function": test_small_scale,
    },
    "scatter_pa_kv_cache::test_min_tokens": {
        "name": "scatter_pa_kv_cache 最小 tokens",
        "description": "test_min_tokens",
        "function": test_min_tokens,
    },
    "scatter_pa_kv_cache::test_medium_scale": {
        "name": "scatter_pa_kv_cache 中等规模",
        "description": "test_medium_scale",
        "function": test_medium_scale,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO scatter_pa_kv_cache operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s scatter_pa_kv_cache::test_config1_performance_p0    Run 配置1_性能P0
  %(prog)s scatter_pa_kv_cache::test_small_scale               Run 小规模验证
  %(prog)s --list                                                List all cases

Note: Before running, please set environment:
  source /mnt/workspace/gitCode/cann/pypto/env_setup.sh
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
        print("\nNote: Before running, please set environment:")
        print("  source /mnt/workspace/gitCode/cann/pypto/env_setup.sh")
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
