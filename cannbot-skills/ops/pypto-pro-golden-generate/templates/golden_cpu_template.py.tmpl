#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro {op} CPU golden reference (higher precision).

基于 NPU golden ({op}_golden.py) 的数学逻辑，在 CPU 上以更高精度（FP32）计算。
供 test_{op}.py 的精度校验使用（方案A混合容差标准，见 precision_compare.py）。

与 NPU golden 的差异:
  - 不使用 NPU 设备，纯 CPU 计算
  - 不 import torch_npu
  - FP16/BF16 输入提升至 FP32 计算，**不降回原 dtype**（返回 FP32 = 更高精度）
  - 数学逻辑与 NPU golden 完全一致

生成方式: 从 {op}_golden.py 复制数学逻辑，移除 .to(device) 和末尾 .to(out_dtype)。
"""

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────
# CPU Golden 参考实现（更高精度）
# ─────────────────────────────────────────────

def {op}_golden_cpu(x: torch.Tensor) -> torch.Tensor:
    """CPU 更高精度 golden 参考实现。

    与 {op}_golden 数学逻辑一致，但:
      - 在 CPU 上执行
      - FP16/BF16 输入提升至 FP32 计算，不降回原 dtype
      - 返回 FP32 tensor

    Args:
        x: 输入 tensor（CPU，任意 dtype）。

    Returns:
        计算结果 tensor（CPU，FP32）。
    """
    # TODO: 从 {op}_golden.py 复制数学逻辑，移除 .to(device)
    # FP16/BF16 → FP32（不降回）
    x_fp32 = x.to(torch.float32) if x.dtype in (torch.float16, torch.bfloat16) else x
    # TODO: 替换为实际 golden 逻辑（与 {op}_golden 相同的数学公式）
    # 示例（SiLU）:  return x_fp32 * torch.sigmoid(x_fp32)
    return x_fp32


# ==========================================
# 验证
# ==========================================

def _validate():
    """简单验证 golden_cpu 函数可正常运行。"""
    print("=" * 60)
    print("{op}_golden_cpu 验证报告")
    print("=" * 60)

    all_pass = True

    # 基本功能验证
    x = torch.randn(4, 8, dtype=torch.float32)
    out = {op}_golden_cpu(x)
    shape_ok = out.shape == x.shape
    dtype_ok = out.dtype == torch.float32
    finite_ok = torch.isfinite(out).all().item()
    print(f"  float32 input: shape={tuple(out.shape)}, dtype={out.dtype}, "
          f"finite={finite_ok} ... {'PASS' if (shape_ok and dtype_ok and finite_ok) else 'FAIL'}")
    if not (shape_ok and dtype_ok and finite_ok):
        all_pass = False

    # FP16 输入验证
    x_fp16 = torch.randn(4, 8, dtype=torch.float16)
    out_fp16 = {op}_golden_cpu(x_fp16)
    dtype_fp16_ok = out_fp16.dtype == torch.float32  # 必须返回 FP32
    finite_fp16_ok = torch.isfinite(out_fp16).all().item()
    print(f"  float16 input: shape={tuple(out_fp16.shape)}, dtype={out_fp16.dtype}, "
          f"finite={finite_fp16_ok} ... {'PASS' if (dtype_fp16_ok and finite_fp16_ok) else 'FAIL'}")
    if not (dtype_fp16_ok and finite_fp16_ok):
        all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("所有验证通过")
    else:
        print("存在验证失败项")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    ok = _validate()
    if not ok:
        exit(1)
