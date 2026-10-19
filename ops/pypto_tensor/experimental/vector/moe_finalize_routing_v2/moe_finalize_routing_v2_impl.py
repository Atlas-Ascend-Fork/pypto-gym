# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------



import sys
from typing import Optional
import pypto
import torch


# 从环境变量获取 run_mode
def _get_run_mode():
    """从命令行参数获取 run_mode。"""
    for arg in sys.argv:
        if arg == "--run_mode=sim" or arg == "--run-mode=sim":
            return pypto.RunMode.SIM
        if arg == "--run_mode=npu" or arg == "--run-mode=npu":
            return pypto.RunMode.NPU
    # 默认返回 NPU
    return pypto.RunMode.NPU


global_run_mode = _get_run_mode()
kernel_run_mode = pypto.RunMode.NPU  # 始终使用 NPU 模式


# ─────────────────────────────────────────────
# JIT Kernel - 计算逻辑全部在 kernel 内完成
# ─────────────────────────────────────────────

@pypto.frontend.jit(
    runtime_options={"run_mode": kernel_run_mode},
    pass_options={"vec_nbuffer_setting": {-2: 1, -1: 2}}
)
def moe_finalize_routing_v2_kernel(
expanded_x: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    expanded_row_idx: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    bias: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    expert_idx: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_INT32),
    scales: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    x1: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    x2: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    K: int,
    H: int,
    NUM_ROWS: int,
    NUM_ROWS_K: int,
    drop_pad_mode: int,
    has_x1: bool,
    has_x2: bool,
    has_bias: bool,
    has_scales: bool,
):
    """MoE 路由聚合 kernel - 使用 SymbolicScalar 动态偏移完成索引查找。

    计算公式：
      out[i, :] = x1[i, :] + x2[i, :] + Σ_k(scales[i,k] * (expanded_x[row_idx[i,k]] + bias[expert_id[i,k]]))

    无效索引（-1）通过 row_idx.max(0) clamp 为 0，保证 view offset 安全。
    """

    tile_h = H
    pypto.set_vec_tile_shapes(1, tile_h)

    for i in pypto.loop(NUM_ROWS, name="rows_loop", idx_name="i", unroll_list=[16, 4, 1]):
        out_row_fp32 = pypto.tensor([1, tile_h], pypto.DT_FP32, "out_row_fp32")

        pypto.set_pass_options(sg_set_scope=1)
        for k in range(K):
            # 根据 drop_pad_mode 计算 idx_pos（编译期分支）
            if drop_pad_mode == 0 or drop_pad_mode == 1:
                idx_pos = k * NUM_ROWS + i
            else:
                idx_pos = i * K + k

            # 从 expanded_row_idx 读取行索引 → SymbolicScalar（1D 索引）
            row_idx = expanded_row_idx[idx_pos]
            row_idx_safe = row_idx.max(0)

            # 读取专家输出行（使用 SymbolicScalar 动态偏移）
            dst_row_bf16 = pypto.view(expanded_x, [1, tile_h], [row_idx_safe, 0], valid_shape=[1, H])
            dst_row_fp32 = pypto.cast(dst_row_bf16, pypto.DT_FP32)

            # 添加专家偏置（如果 has_bias）
            if has_bias:
                expert_id = expert_idx[i, k]
                bias_row_bf16 = pypto.view(bias, [1, tile_h], [expert_id, 0], valid_shape=[1, H])
                bias_row_fp32 = pypto.cast(bias_row_bf16, pypto.DT_FP32)
                dst_row_fp32 = pypto.add(dst_row_fp32, bias_row_fp32)

            # 应用路由权重（如果 has_scales）
            if has_scales:
                scale_bf16 = pypto.view(scales, [1, 1], [i, k], valid_shape=[1, 1])
                scale_fp32 = pypto.cast(scale_bf16, pypto.DT_FP32)
                dst_row_fp32 = pypto.mul(dst_row_fp32, scale_fp32)

            # 累加：必须使用 "if k==0: assign; else: accumulate" 模式
            # k==0 时：x1/x2 加入 dst_row（不受 scales 影响），然后赋值初始化 out_row
            if k == 0:
                if has_x1:
                    x1_row_bf16 = pypto.view(x1, [1, tile_h], [i, 0], valid_shape=[1, H])
                    x1_row_fp32 = pypto.cast(x1_row_bf16, pypto.DT_FP32)
                    dst_row_fp32 = pypto.add(dst_row_fp32, x1_row_fp32)

                if has_x2:
                    x2_row_bf16 = pypto.view(x2, [1, tile_h], [i, 0], valid_shape=[1, H])
                    x2_row_fp32 = pypto.cast(x2_row_bf16, pypto.DT_FP32)
                    dst_row_fp32 = pypto.add(dst_row_fp32, x2_row_fp32)

                out_row_fp32[:] = dst_row_fp32
            else:
                out_row_fp32[:] = out_row_fp32 + dst_row_fp32

        out_row_bf16 = pypto.cast(out_row_fp32, pypto.DT_BF16)
        pypto.assemble(out_row_bf16, [i, 0], out)
        pypto.set_pass_options(sg_set_scope=-1)


# ─────────────────────────────────────────────
# Wrapper 函数 - 仅负责 tensor 创建和参数校验
# ─────────────────────────────────────────────

def moe_finalize_routing_v2_wrapper(
    expanded_x: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    x1: Optional[torch.Tensor] = None,
    x2: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    expert_idx: Optional[torch.Tensor] = None,
    drop_pad_mode: int = 2,
) -> torch.Tensor:
    """算子 wrapper - 仅负责参数校验、维度计算和默认值填充。

    Args:
        expanded_x: [NUM_ROWS*K, H] BF16 - MoE FFN 输出
        expanded_row_idx: [NUM_ROWS*K] INT32 - 行索引（1D，无需 reshape）
        x1: 残差连接1，可选，shape=(NUM_ROWS, H)，BF16
        x2: 残差连接2，可选，shape=(NUM_ROWS, H)，BF16
        bias: 专家偏置，可选，shape=(E, H)，BF16
        scales: 路由权重，可选，shape=(NUM_ROWS, K)，BF16
        expert_idx: 专家索引，可选，shape=(NUM_ROWS, K)，INT32
        drop_pad_mode: 索引排列方式（0-3）

    Returns:
        输出 [NUM_ROWS, H] BF16
    """
    # ========================================
    # 1. 参数校验
    # ========================================

    # 约束3：x1Optional 未提供时，x2Optional 不能提供
    if x1 is None and x2 is not None:
        raise ValueError("x2 cannot be provided when x1 is None (constraint #3)")

    # 约束5：biasOptional 存在时，expertIdxOptional 必须存在
    if bias is not None and expert_idx is None:
        raise ValueError("expert_idx must be provided when bias is provided (constraint #5)")

    # ========================================
    # 2. 计算 K 和 NUM_ROWS
    # ========================================

    bsk = expanded_row_idx.shape[0]
    H = expanded_x.shape[-1]

    # 确定 K 值和 NUM_ROWS（与 golden 保持一致的推导逻辑）
    if scales is not None:
        K = scales.shape[1]
        NUM_ROWS = bsk // K
    elif x1 is not None:
        NUM_ROWS = x1.shape[0]
        K = bsk // NUM_ROWS
    elif x2 is not None:
        NUM_ROWS = x2.shape[0]
        K = bsk // NUM_ROWS
    elif expert_idx is not None:
        NUM_ROWS = expert_idx.shape[0]
        K = expert_idx.shape[1]
    else:
        # 无任何辅助信息，默认 K=1
        K = 1
        NUM_ROWS = bsk

    NUM_ROWS_K = NUM_ROWS * K

    # ========================================
    # 3. 创建输出 tensor
    # ========================================

    output = torch.zeros((NUM_ROWS, H), dtype=torch.bfloat16, device=expanded_x.device)

    # ========================================
    # 4. 为可选参数创建默认值 tensor（最小尺寸，has_*_False 时 kernel 不访问）
    # ========================================

    bias_param = bias if bias is not None else torch.zeros((1, H), dtype=torch.bfloat16, device=expanded_x.device)
    expert_idx_param = (expert_idx if expert_idx is not None
                        else torch.zeros((1, 1), dtype=torch.int32, device=expanded_x.device))
    scales_param = scales if scales is not None else torch.ones((1, 1), dtype=torch.bfloat16, device=expanded_x.device)
    x1_param = x1 if x1 is not None else torch.zeros((1, H), dtype=torch.bfloat16, device=expanded_x.device)
    x2_param = x2 if x2 is not None else torch.zeros((1, H), dtype=torch.bfloat16, device=expanded_x.device)

    # 判断标志
    has_x1_param = (x1 is not None)
    has_x2_param = (x2 is not None)
    has_bias_param = (bias is not None)
    has_scales_param = (scales is not None)

    # ========================================
    # 5. 调用 kernel（所有计算逻辑在 kernel 内完成，expanded_row_idx 直接传入 1D）
    # ========================================

    moe_finalize_routing_v2_kernel(
        expanded_x, expanded_row_idx, bias_param, expert_idx_param,
        scales_param, x1_param, x2_param, output,
        K, H, NUM_ROWS, NUM_ROWS_K, drop_pad_mode,
        has_x1_param, has_x2_param, has_bias_param, has_scales_param,
    )

    return output