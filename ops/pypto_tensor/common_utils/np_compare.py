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
import logging
import numpy as np
from numpy.testing import assert_allclose


class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'


def detailed_allclose_manual(cpu, npu, name, *, rtol=1e-3, atol=1e-3, max_prints=50, force_print_first_n=5):
    """
    手动实现 np.allclose 的详细比较，打印超出容差和 NaN 的值

    参数:
        cpu: CPU端的数组
        npu: NPU端的数组
        rtol: 相对容差
        atol: 绝对容差
        max_prints: 最大打印数量
        force_print_first_n: 强制打印前n个元素的值, 无论是否异常
    """
    # 检查形状是否一致
    if cpu.shape != npu.shape:
        logging.error(f"Error: shape mismatch - cpu {cpu.shape} vs npu {npu.shape}")
        return False

    total_elements = cpu.size
    abnormal_count = 0
    nan_count = 0
    exceed_tolerance_count = 0

    logging.info(f"Comparing arrays, shape: {cpu.shape}, total elements: {total_elements}")
    logging.info(f"Tolerance: rtol={rtol}, atol={atol}")
    logging.info("=" * 80)

    # 将数组展平以便遍历，但记录原始索引
    cpu_flat = cpu.reshape(-1)
    npu_flat = npu.reshape(-1)

    # 获取多维索引的函数
    def get_multi_index(flat_index, shape):
        indices = []
        remaining = flat_index
        for dim in reversed(shape):
            indices.append(remaining % dim)
            remaining = remaining // dim
        return tuple(reversed(indices))

    # 强制打印前n个元素
    _print_first_n(cpu_flat, npu_flat, get_multi_index, force_print_first_n)

    # 遍历所有元素查找异常
    for flat_idx in range(total_elements):
        cpu_val = cpu_flat[flat_idx]
        npu_val = npu_flat[flat_idx]
        # 获取多维索引
        multi_idx = get_multi_index(flat_idx, cpu.shape)

        # 检查是否为 NaN (npu 中有 NaN 就认为是异常)
        if _is_nan(npu_val, npu_val):
            abnormal_count += 1
            nan_count += 1
            if abnormal_count <= max_prints:
                _log_nan_error(multi_idx, cpu_val, npu_val)
        # 检查是否超出容差
        elif _is_above_tolerance(cpu_val, npu_val, rtol, atol):
            # CPU 有 NaN 但 NPU 没有，也是异常
            abnormal_count += 1
            exceed_tolerance_count += 1
            if abnormal_count <= max_prints:
                _log_tolerance_error(multi_idx, cpu_val, npu_val, rtol, atol)

    _print_summary(abnormal_count, nan_count, exceed_tolerance_count, total_elements, name)

    # 检查是否通过 allclose 条件
    is_allclose = abnormal_count == 0
    logging.info(f"\nnp.allclose equivalent result: {is_allclose}")

    assert_allclose(cpu, npu, rtol, atol)

    if abnormal_count > max_prints:
        logging.info(f"\nNote: only showing first {max_prints} anomalies, total {abnormal_count} anomalous elements")
    assert_allclose(cpu, npu, rtol, atol)

    return is_allclose


def _is_nan(cpu_val, npu_val):
    return np.isnan(cpu_val) or np.isnan(npu_val)


def _is_above_tolerance(cpu_val, npu_val, rtol, atol):
    if np.isnan(cpu_val) or np.isnan(npu_val):
        return False
    abs_diff = np.abs(cpu_val - npu_val)
    allowed_diff = atol + rtol * np.abs(npu_val)
    return abs_diff > allowed_diff


def _print_first_n(cpu_flat, npu_flat, get_multi_index, n):
    if n <= 0:
        return
    logging.info(f"{Colors.YELLOW}Force printing first {n} elements:{Colors.RESET}")
    for flat_idx in range(min(n, len(cpu_flat))):
        cpu_val = cpu_flat[flat_idx]
        npu_val = npu_flat[flat_idx]
        multi_idx = get_multi_index(flat_idx, cpu_flat.shape)
        # 格式化值
        cpu_str = "NaN" if np.isnan(cpu_val) else f"{cpu_val:.6e}"
        npu_str = "NaN" if np.isnan(npu_val) else f"{npu_val:.6e}"
        diff_str = "NaN" if np.isnan(cpu_val) or np.isnan(npu_val) else f"{np.abs(cpu_val - npu_val):.6e}"
        logging.info(f"{Colors.YELLOW}index {multi_idx}: cpu={cpu_str}, npu={npu_str}, diff={diff_str}{Colors.RESET}")
    logging.info("-" * 80)


def _log_nan_error(multi_idx, cpu_val, npu_val):
    cpu_str = "NaN" if np.isnan(cpu_val) else f"{cpu_val:.6e}"
    npu_str = "NaN" if np.isnan(npu_val) else f"{npu_val:.6e}"
    logging.info(f"index {multi_idx}: cpu={cpu_str}, npu={npu_str}, diff=NaN")


def _log_tolerance_error(multi_idx, cpu_val, npu_val, rtol, atol):
    abs_diff = np.abs(cpu_val - npu_val)
    allowed_diff = atol + rtol * np.abs(npu_val)
    logging.info(
        f"index {multi_idx}: cpu={cpu_val:.6e}, npu={npu_val:.6e}, "
        f"diff={abs_diff:.6e}(exceeds tolerance {allowed_diff:.6e})"
    )


def _print_summary(abnormal_count, nan_count, exceed_tolerance_count, total_elements, name):
    # 统计信息
    logging.info("=" * 80)
    logging.info(f"{Colors.BOLD}{Colors.PURPLE}{name} comparison statistics:{Colors.RESET}")
    logging.info(f"Total elements: {total_elements}")
    logging.info(f"Anomalous elements: {abnormal_count}")
    logging.info(f"  - NaN count: {nan_count}")
    logging.info(f"  - Exceeds tolerance count: {exceed_tolerance_count}")
    logging.info(f"Anomaly ratio: {abnormal_count / total_elements * 100:.4f}%")
