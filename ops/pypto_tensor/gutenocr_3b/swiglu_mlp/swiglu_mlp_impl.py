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
SwiGLU MLP PyPTO融合算子实现

数学结构:
  gate = silu(gate_proj(x))  # Linear(H→I)
  up = up_proj(x)            # Linear(H→I)
  hidden = gate * up         # Element-wise mul
  output = down_proj(hidden) # Linear(I→H)
"""

import logging
import os

import pypto
import torch

_logger = logging.getLogger(__name__)

HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 11008


@pypto.frontend.jit
def swiglu_mlp_fused(
    x: pypto.Tensor([1, HIDDEN_SIZE], pypto.DT_BF16),
    gate_weight: pypto.Tensor([HIDDEN_SIZE, INTERMEDIATE_SIZE], pypto.DT_BF16),
    gate_bias: pypto.Tensor([INTERMEDIATE_SIZE], pypto.DT_BF16),
    up_weight: pypto.Tensor([HIDDEN_SIZE, INTERMEDIATE_SIZE], pypto.DT_BF16),
    up_bias: pypto.Tensor([INTERMEDIATE_SIZE], pypto.DT_BF16),
    down_weight: pypto.Tensor([INTERMEDIATE_SIZE, HIDDEN_SIZE], pypto.DT_BF16),
    down_bias: pypto.Tensor([HIDDEN_SIZE], pypto.DT_BF16),
    output: pypto.Tensor([1, HIDDEN_SIZE], pypto.DT_BF16)):
    """
    SwiGLU MLP融合实现

    Args:
        x: 输入tensor [batch, hidden_size]
        gate_weight: gate projection权重
        gate_bias: gate projection bias
        up_weight: up projection权重  
        up_bias: up projection bias
        down_weight: down projection权重
        down_bias: down projection bias
        output: 输出tensor [batch, hidden_size]
    """
    pypto.set_vec_tile_shapes(64, 512)

    # Gate projection: Linear + SiLU
    gate = pypto.add(pypto.matmul(x, gate_weight), gate_bias)
    gate = pypto.silu(gate)

    # Up projection: Linear
    up = pypto.add(pypto.matmul(x, up_weight), up_bias)

    # Element-wise multiplication
    hidden = pypto.mul(gate, up)

    # Down projection: Linear
    result = pypto.add(pypto.matmul(hidden, down_weight), down_bias)

    output[:] = result


def swiglu_mlp_fused_static(x, gate_weight, gate_bias, up_weight, up_bias, 
                            down_weight, down_bias, batch_size_static=4):
    """
    静态batch size版本的SwiGLU MLP (需要reshape)
    """
    hidden_size = 2048
    intermediate_size = 11008

    batch_sz = batch_size_static
    hidden_sz = hidden_size
    inter_sz = intermediate_size

    @pypto.frontend.jit
    def swiglu_kernel(
        x: pypto.Tensor([batch_sz, hidden_sz], pypto.DT_BF16),
        gate_weight: pypto.Tensor([hidden_sz, inter_sz], pypto.DT_BF16),
        gate_bias: pypto.Tensor([inter_sz], pypto.DT_BF16),
        up_weight: pypto.Tensor([hidden_sz, inter_sz], pypto.DT_BF16),
        up_bias: pypto.Tensor([inter_sz], pypto.DT_BF16),
        down_weight: pypto.Tensor([inter_sz, hidden_sz], pypto.DT_BF16),
        down_bias: pypto.Tensor([hidden_sz], pypto.DT_BF16),
        output: pypto.Tensor([batch_sz, hidden_sz], pypto.DT_BF16)):
        pypto.set_vec_tile_shapes(64, 512)

        gate = pypto.add(pypto.matmul(x, gate_weight), gate_bias)
        gate_activated = pypto.mul(gate, pypto.sigmoid(gate))

        up = pypto.add(pypto.matmul(x, up_weight), up_bias)

        hidden = pypto.mul(gate_activated, up)

        result = pypto.add(pypto.matmul(hidden, down_weight), down_bias)

        output[:] = result

    output = torch.empty(batch_sz, hidden_sz, dtype=x.dtype, device=x.device)
    swiglu_kernel(x, gate_weight, gate_bias, up_weight, up_bias, 
                  down_weight, down_bias, output)
    return output


def test_swiglu_precision():
    """Accuracy test"""
    _logger.info("%s", "=" * 60)
    _logger.info("SwiGLU MLP Accuracy Test")
    _logger.info("%s", "=" * 60)

    torch.manual_seed(42)
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    device = f'npu:{device_id}'

    batch_size = 4
    hidden_size = 2048
    intermediate_size = 11008

    # 创建测试数据
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)

    # 创建权重（使用随机初始化）
    gate_weight = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device=device)
    gate_bias = torch.randn(intermediate_size, dtype=torch.bfloat16, device=device)
    up_weight = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device=device)
    up_bias = torch.randn(intermediate_size, dtype=torch.bfloat16, device=device)
    down_weight = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device=device)
    down_bias = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)

    # PyPTO融合版本
    try:
        output_pypto = swiglu_mlp_fused_static(
            x, gate_weight, gate_bias, up_weight, up_bias,
            down_weight, down_bias, batch_size_static=batch_size
        )
        _logger.info("✓ PyPTO fused kernel executed successfully")
        _logger.info("  Input shape: %s", x.shape)
        _logger.info("  Output shape: %s", output_pypto.shape)
    except Exception as e:
        _logger.error("✗ PyPTO fused kernel failed: %s", e)
        import traceback
        traceback.print_exc()
        return False

    # Torch baseline (SwiGLU结构)
    gate = torch.nn.functional.linear(x, gate_weight.T, gate_bias)
    gate = torch.nn.functional.silu(gate)

    up = torch.nn.functional.linear(x, up_weight.T, up_bias)

    hidden = gate * up

    output_torch = torch.nn.functional.linear(hidden, down_weight.T, down_bias)

    # 精度对比
    diff = torch.abs(output_torch - output_pypto).max().item()
    mean_diff = torch.abs(output_torch - output_pypto).mean().item()

    _logger.info("Accuracy comparison:")
    _logger.info("  Max diff: %.6f", diff)
    _logger.info("  Mean diff: %.6f", mean_diff)

    if diff < 0.1:
        _logger.info("✓ Accuracy passed (diff < 0.1)")
        return True
    else:
        _logger.error("✗ Accuracy check failed (diff > 0.1)")
        return False


def benchmark_swiglu_performance():
    """Performance test"""
    _logger.info("\n%s", "=" * 60)
    _logger.info("SwiGLU MLP Performance Test")
    _logger.info("%s", "=" * 60)

    import time

    torch.manual_seed(42)
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    device = f'npu:{device_id}'

    test_configs = [
        (1, 2048, 11008),
        (4, 2048, 11008),
        (8, 2048, 11008),
        (16, 2048, 11008),
    ]

    iterations = 100

    for batch_size, hidden_size, intermediate_size in test_configs:
        _logger.info("\n[Batch=%s, Hidden=%s, Inter=%s]", batch_size, hidden_size, intermediate_size)

        # 准备数据
        x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
        gate_weight = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device=device)
        gate_bias = torch.randn(intermediate_size, dtype=torch.bfloat16, device=device)
        up_weight = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device=device)
        up_bias = torch.randn(intermediate_size, dtype=torch.bfloat16, device=device)
        down_weight = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device=device)
        down_bias = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)

        # Torch baseline
        def torch_swiglu(x, gw, gb, uw, ub, dw, db):
            gate = torch.nn.functional.linear(x, gw.T, gb)
            gate = torch.nn.functional.silu(gate)
            up = torch.nn.functional.linear(x, uw.T, ub)
            hidden = gate * up
            output = torch.nn.functional.linear(hidden, dw.T, db)
            return output

        # Warmup
        for _ in range(10):
            _ = torch_swiglu(x, gate_weight, gate_bias, up_weight, up_bias, 
                            down_weight, down_bias)
        torch.npu.synchronize()

        # Measure
        start = time.time()
        for _ in range(iterations):
            output_torch = torch_swiglu(x, gate_weight, gate_bias, up_weight, up_bias,
                                       down_weight, down_bias)
        torch.npu.synchronize()
        torch_time = (time.time() - start) / iterations * 1000

        _logger.info("  Torch baseline: %.3f ms", torch_time)

        # PyPTO fused
        try:
            # Warmup
            for _ in range(10):
                _ = swiglu_mlp_fused_static(
                    x, gate_weight, gate_bias, up_weight, up_bias,
                    down_weight, down_bias, batch_size_static=batch_size
                )
            torch.npu.synchronize()

            # Measure
            start = time.time()
            for _ in range(iterations):
                output_pypto = swiglu_mlp_fused_static(
                    x, gate_weight, gate_bias, up_weight, up_bias,
                    down_weight, down_bias, batch_size_static=batch_size
                )
            torch.npu.synchronize()
            pypto_time = (time.time() - start) / iterations * 1000

            _logger.info("  PyPTO fused: %.3f ms", pypto_time)

            speedup = torch_time / pypto_time
            if speedup >= 1.0:
                _logger.info("  ✓ PyPTO faster: %.2fx", speedup)
            else:
                _logger.info("  ✗ PyPTO slower: %.2fx", speedup)

            # 精度验证
            diff = torch.abs(output_torch - output_pypto).max().item()
            _logger.info("  Max diff: %.6f", diff)

        except Exception as e:
            _logger.error("  ✗ PyPTO failed: %s", e)


if __name__ == "__main__":
    import sys

    # 测试精度
    precision_ok = test_swiglu_precision()

    if precision_ok:
        # 测试性能
        benchmark_swiglu_performance()

    sys.exit(0 if precision_ok else 1)