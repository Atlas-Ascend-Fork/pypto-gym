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
AttentionWorkerCombine PyPTO Kernel - Dynamic Shape Implementation

支持动态 batch (bs) 维度
h (hidden) 维度通过参数传入

动态轴说明：
- 第0维 (batch): 使用 pypto.DYNAMIC 标记，支持运行时变化
"""

import os
import logging
import torch
import torch_npu
import pypto

logging.basicConfig(level=logging.INFO, format="%(message)s")


def get_device_id():
    return int(os.environ.get('TILE_FWK_DEVICE_ID', 0))


k = 2


@pypto.frontend.jit
def attention_worker_combine_splitbs_kernel(
    token_data: pypto.Tensor([pypto.DYNAMIC, k + 1, pypto.STATIC], pypto.DT_BF16),
    expert_scales: pypto.Tensor([pypto.DYNAMIC, k], pypto.DT_FP32),
    y: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    h: int,
    tile_bs: int = 8,
):
    """
    SplitBS: 向量化实现 (支持动态 batch 维度)
    """
    bs_dyn = token_data.shape[0]

    bs_loops = (bs_dyn + tile_bs - 1) // tile_bs

    pypto.set_vec_tile_shapes(tile_bs, k + 1, h)

    for bs_idx in pypto.loop(bs_loops):
        bs_offset = bs_idx * tile_bs
        bs_end = min(bs_offset + tile_bs, bs_dyn)
        valid_bs = bs_end - bs_offset

        token_view = pypto.view(token_data, [tile_bs, k + 1, h],
                                [bs_offset, 0, 0], valid_shape=[valid_bs, k + 1, h])
        scales_view = pypto.view(expert_scales, [tile_bs, k],
                                 [bs_offset, 0], valid_shape=[valid_bs, k])

        token_fp32 = pypto.cast(token_view, pypto.DT_FP32)
        token_routed = token_fp32[:, 0:k, :]
        token_shared = token_fp32[:, k, :]

        scales_3d = pypto.unsqueeze(scales_view, dim=2)

        weighted = pypto.mul(token_routed, scales_3d)
        weighted_sum = pypto.sum(weighted, dim=1, keepdim=True)

        result = pypto.add(weighted_sum[:, 0, :], token_shared)
        y_tile = pypto.cast(result, pypto.DT_BF16)

        pypto.assemble(y_tile, [bs_offset, 0], y)


@pypto.frontend.jit()
def attention_worker_combine_splith_kernel(
    token_data: pypto.Tensor([pypto.DYNAMIC, k + 1, pypto.STATIC], pypto.DT_BF16),
    expert_scales: pypto.Tensor([pypto.DYNAMIC, k], pypto.DT_FP32),
    y: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    h: int,
    tile_bs: int = 8,
):
    """
    SplitH: 按 hidden 维度切分 (支持动态 batch 维度)
    """
    h_tile = 16
    bs_dyn = token_data.shape[0]

    pypto.set_vec_tile_shapes(tile_bs, k + 1, h_tile)

    bs_loops = (bs_dyn + tile_bs - 1) // tile_bs
    h_loops = h // h_tile

    for bs_idx in pypto.loop(bs_loops):
        bs_offset = bs_idx * tile_bs
        bs_end = min(bs_offset + tile_bs, bs_dyn)
        valid_bs = bs_end - bs_offset

        scales_view = pypto.view(expert_scales, [tile_bs, k],
                                 [bs_offset, 0], valid_shape=[valid_bs, k])
        scales_3d = pypto.unsqueeze(scales_view, dim=2)

        for h_idx in pypto.loop(h_loops):
            h_start = h_idx * h_tile
            h_end = h_start + h_tile

            token_view = pypto.view(token_data, [tile_bs, k + 1, h_tile],
                                    [bs_offset, 0, h_start], valid_shape=[valid_bs, k + 1, h_tile])

            token_h_fp32 = pypto.cast(token_view, pypto.DT_FP32)

            token_routed = token_h_fp32[:, 0:k, :]
            token_shared = token_h_fp32[:, k, :]

            weighted = pypto.mul(token_routed, scales_3d)
            weighted_sum = pypto.sum(weighted, dim=1, keepdim=True)

            result = pypto.add(weighted_sum[:, 0, :], token_shared)
            y_h = pypto.cast(result, pypto.DT_BF16)

            pypto.assemble(y_h, [bs_offset, h_start], y)


def test_kernel(kernel_func, kernel_name, test_bs=8, test_h=32):
    """Test a single kernel"""
    logging.info(f"\n--- Testing {kernel_name} (bs={test_bs}, h={test_h}) ---")

    device_id = get_device_id()
    torch.npu.set_device(device_id)
    device = f'npu:{device_id}'

    token_data = torch.randn(test_bs, k + 1, test_h, dtype=torch.bfloat16, device=device)
    expert_scales = torch.rand(test_bs, k, dtype=torch.float32, device=device)
    y = torch.zeros(test_bs, test_h, dtype=torch.bfloat16, device=device)

    logging.info(f"  Input: token_data={token_data.shape}, expert_scales={expert_scales.shape}")

    try:
        kernel_func(token_data, expert_scales, y, h=test_h, tile_bs=8)
        logging.info(f"  ✓ Kernel executed successfully")

        golden = (token_data[:, :k, :].float() * expert_scales.unsqueeze(-1)).sum(1) + token_data[:, k, :].float()
        golden = golden.to(torch.bfloat16)

        max_diff = (y - golden).abs().max().item()
        logging.info(f"  Max diff: {max_diff:.6f}")
        (logging.error if max_diff >= 0.01 else logging.info)(f"  Result: {'PASS' if max_diff < 0.01 else 'FAIL'}")
        return max_diff < 0.01

    except Exception as e:
        logging.error(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    logging.info("=" * 70)
    logging.info("AttentionWorkerCombine PyPTO Kernel Tests (Dynamic Shape)")
    logging.info("=" * 70)

    test_cases = [
        (8, 32),
        (16, 32),
        (4, 32),
    ]

    results = []

    for test_bs, test_h in test_cases:
        logging.info(f"\n{'='*70}")
        logging.info(f"Testing with bs={test_bs}, h={test_h}")
        logging.info(f"{'='*70}")
        results.append((f"SplitBS (bs={test_bs}, h={test_h})",
                       test_kernel(attention_worker_combine_splitbs_kernel, "SplitBS", test_bs, test_h)))
        results.append((f"SplitH (bs={test_bs}, h={test_h})",
                       test_kernel(attention_worker_combine_splith_kernel, "SplitH", test_bs, test_h)))

    logging.info("\n" + "=" * 70)
    logging.info("Summary")
    logging.info("=" * 70)
    for name, passed in results:
        (logging.warning if not passed else logging.info)(f"  {name}: {'✓ PASS' if passed else '✗ FAIL'}")

    total = sum(1 for _, p in results if p)
    logging.info(f"\nTotal: {total}/{len(results)} passed")


if __name__ == "__main__":
    run_all_tests()
