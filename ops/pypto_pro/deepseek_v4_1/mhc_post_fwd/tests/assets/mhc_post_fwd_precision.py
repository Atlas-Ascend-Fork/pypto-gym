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

"""mhc_post_fwd 的精度比较口径。

两个层次：

* **v.s. BF16 数学参考**（CPU golden 或 torch ``einsum`` 参考）：输出为 BF16，允许一个
  BF16 ulp 级别的差异。4 项 FP32 求和在不同归约顺序下会产生不同的 BF16 舍入，因此这里用
  ``torch.testing.assert_close`` 的 BF16 默认容差，并额外报告 max-abs-diff 便于回归对比。
* **同序累加核验**：每 64 lane 的 FMA 顺序固定（先 ``post*x``，再按 ``mi = 0,1,2,3`` 累加），
  据此要求**逐 bit 相同**。这是判断"实现是否遵循约定累加顺序"最直接的证据，比容差比较强得多。
"""

import torch

MHC = 4

# BF16 输出的验收容差：沿用 torch.testing.assert_close 的 BF16 默认值。
RTOL = 1.6e-2
ATOL = 1e-5


def _cpu_f32(t):
    """Device-agnostic float32 view on CPU (the golden is computed on CPU)."""
    return t.detach().float().cpu()


def max_abs_diff(a, b) -> float:
    return (_cpu_f32(a) - _cpu_f32(b)).abs().max().item()


def count_mismatch(a, b, atol: float = 0.05) -> int:
    return int(((_cpu_f32(a) - _cpu_f32(b)).abs() > atol).sum())


def assert_matches_reference(out, ref, name: str = "", rtol: float = RTOL,
                             atol: float = ATOL):
    """BF16 容差比较；失败信息里带上 max-abs-diff 与超差元素数。"""
    out_c, ref_c = _cpu_f32(out), _cpu_f32(ref)
    try:
        torch.testing.assert_close(out_c, ref_c, rtol=rtol, atol=atol)
    except AssertionError as exc:
        raise AssertionError(
            f"{name}: mhc_post_fwd output mismatch "
            f"(max_abs_diff={max_abs_diff(out_c, ref_c):.4g}, "
            f"n_gt_{atol}={count_mismatch(out_c, ref_c, atol)}/{out_c.numel()})\n{exc}") from exc


def assert_bit_identical(out, ref, name: str = ""):
    """逐 bit 相同（用于核验固定累加顺序的实现一致性）。"""
    if tuple(out.shape) != tuple(ref.shape):
        raise AssertionError(f"{name}: shape {tuple(out.shape)} != {tuple(ref.shape)}")
    if not torch.equal(_cpu_f32(out), _cpu_f32(ref)):
        d = (_cpu_f32(out) - _cpu_f32(ref)).abs()
        raise AssertionError(
            f"{name}: not bit-identical (max_abs_diff={d.max().item():.4g}, "
            f"n_diff={int((d > 0).sum())}/{out.numel()})")


def assert_written(out, sentinel: float = -1.0):
    """哨兵检查：确认输出真的被写入（防止"解析成功但没发工作"的空过）。"""
    if out.numel() == 0:
        return
    if bool((_cpu_f32(out) == sentinel).all()):
        raise AssertionError("output is entirely the sentinel: kernel emitted no work")
