#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use the file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
mhc_post 算子实现 — loop_unroll(bs) + Python for(k) + pypto.view + axpy_ 逐 k 累积

核心优化策略（对标 C++ AscendC USE_PERMANENT_X=1 路径）：
1. loop_unroll 处理 bs 维度（高效调度，对标原版）
2. Python for 循环展开 k=0..3（N=4 很小，展开无循环开销）
3. pypto.view 取 h_res 和 x 的第 k 行（避免降维 slice 导致 tile shape 不匹配）
4. axpy_ 原地累加替代 sum(dim=1)+add 两步操作（对标 C++ Axpy 融合 mul+add）
5. 消除中间大 tensor weighted[unroll_length,N,N,D] 和 sum(dim=1) 归约

公式：output[b*s, n, d] = h_post[b*s, n] * h_out[b*s, d] + sum_{k=0}^{N-1} h_res[b*s, k, n] * x[b*s, k, d]
"""

from dataclasses import dataclass
import pypto
import torch
import torch_npu


@dataclass
class MhcPostConfig:
    vec_nbuffer: int


@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 1024})
def mhc_post_kernel_bf16(
    x: pypto.Tensor([pypto.DYNAMIC, 4, pypto.STATIC], pypto.DT_BF16),       # [B*S, N, D] BF16
    h_res: pypto.Tensor([pypto.DYNAMIC, 4, 4], pypto.DT_FP32),              # [B*S, N, N] FP32
    h_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),      # [B*S, D] BF16
    h_post: pypto.Tensor([pypto.DYNAMIC, 4], pypto.DT_FP32),                # [B*S, N] FP32
    output: pypto.Tensor([pypto.DYNAMIC, 4, pypto.STATIC], pypto.DT_BF16),  # [B*S, N, D] BF16
    mhc_post_config: MhcPostConfig,
):
    """mhc_post kernel — loop_unroll(bs) + Python for(k) + pypto.view + axpy_。

    计算流程（每个 bs unroll_length）：
    1. cast x 和 h_out 到 FP32（USE_PERMANENT_X=1）
    2. mul(h_post, h_out) 初始化 result（对标 Muls）
    3. 逐 k=0..3 Python for 循环：
       - pypto.view 取 h_res[:, k, :] → reshape [unroll_length, N, 1]
       - pypto.view 取 x[:, k, :] → [unroll_length, 1, D]
       - mul → term_k [unroll_length, N, D]
       - axpy_(result, term_k) → 原地累加（对标 Axpy）
    4. cast result → BF16，assemble 写回
    """
    pypto.experimental.set_operation_options(combine_axis=True)

    vec_nbuffer_value = mhc_post_config.vec_nbuffer
    pypto.set_pass_options(vec_nbuffer_setting={-2: 1, -1: vec_nbuffer_value})

    BS = x.shape[0]
    N = 4
    D = x.shape[2]

    # reshape 以支持广播
    h_post1 = pypto.reshape(h_post, [BS, N, 1], inplace=True)    # [BS, N, 1]
    h_out1 = pypto.reshape(h_out, [BS, 1, D], inplace=True)      # [BS, 1, D]

    # bs 分块 — 使用 loop_unroll（高效调度）
    for bs_idx, unroll_length in pypto.loop_unroll(
        0, BS, 1, name="LOOP_BS", idx_name="bs_idx", unroll_list=[16, 8, 4, 2, 1]):
        x_slice = x[bs_idx: bs_idx + unroll_length, :, :]
        h_res_slice = h_res[bs_idx: bs_idx + unroll_length, :, :]
        h_out_slice = h_out1[bs_idx: bs_idx + unroll_length, :, :]
        h_post_slice = h_post1[bs_idx: bs_idx + unroll_length, :, :]

        # USE_PERMANENT_X=1: cast x 和 h_out 到 FP32
        pypto.set_vec_tile_shapes(1, N, 2048)
        x_fp32 = pypto.cast(x_slice, pypto.DT_FP32)

        h_out_fp32 = pypto.cast(h_out_slice, pypto.DT_FP32)

        # Muls 等价: h_post * h_out 初始化 result
        result = pypto.mul(h_post_slice, h_out_fp32)               # [unroll_length, N, D] FP32

        # Axpy 等价: 逐 k=0..3 累积 — Python for（展开为 4 个独立 op 组）
        for k in range(N):
            # pypto.view 取 h_res 的第 k 行（避免降维 slice）
            # h_res[:, k, :] → [unroll_length, 1, N] → reshape [unroll_length, N, 1]
            h_res_k_view = pypto.view(h_res_slice, [unroll_length, 1, N], [0, k, 0])
            h_res_k = pypto.reshape(h_res_k_view, [unroll_length, N, 1], inplace=True)  # [unroll_length, N, 1] FP32

            # pypto.view 取 x 的第 k 行（避免降维 slice）
            x_k = pypto.view(x_fp32, [unroll_length, 1, D], [0, k, 0])    # [unroll_length, 1, D] FP32

            # mul: h_res_k * x_k → term_k [unroll_length, N, D]（广播乘法）
            term_k = pypto.mul(h_res_k, x_k)                        # [unroll_length, N, D] FP32

            # axpy_: result += term_k（alpha=1.0, 原地累加, 融合 add）
            result.axpy_(term_k, alpha=1.0)

        # 写回 BF16
        result_bf16 = pypto.cast(result, pypto.DT_BF16)
        pypto.assemble(result_bf16, [bs_idx, 0, 0], output)


# ─────────────────────────────────────────────
# Wrapper 函数（导出接口）
# ─────────────────────────────────────────────


def mhc_post_wrapper(
    x: torch.Tensor,
    h_res: torch.Tensor,
    h_out: torch.Tensor,
    h_post: torch.Tensor,
    mhc_post_config: MhcPostConfig,
    output: torch.Tensor = None,
) -> torch.Tensor:
    """算子 wrapper，供 test_mhc_post.py 调用。

    负责：
    1. 验证输入 shape/dtype
    2. 将输入 reshape 为 [B*S, ...] 格式
    3. 调用 JIT kernel（直接传递 torch tensor）
    4. 将输出 reshape 回原始格式

    Args:
        x: [B, S, N, D] 输入 tensor (BF16)
           N 固定为 4，D 为 2560 或 5120
        h_res: [B, S, N, N] 流间混合权重矩阵 (FP32)
        h_out: [B, S, D] 输出项数据 (BF16)
        h_post: [B, S, N] 后处理权重 (FP32)
        mhc_post_config: 算子配置（含 vec_nbuffer）
        output: 可选输出 tensor，如未提供则自动构造

    Returns:
        output: [B, S, N, D] 融合计算结果 (BF16)
    """
    assert x.is_contiguous(), "x must be contiguous"
    assert h_res.is_contiguous(), "h_res must be contiguous"
    assert h_out.is_contiguous(), "h_out must be contiguous"
    assert h_post.is_contiguous(), "h_post must be contiguous"

    B = x.shape[0]
    S = x.shape[1]
    N = x.shape[2]
    D = x.shape[3]
    assert N == 4, f"N must be 4, got {N}"

    BS = B * S
    x_reshaped = x.view(BS, N, D).contiguous()
    h_res_reshaped = h_res.view(BS, N, N).contiguous()
    h_out_reshaped = h_out.view(BS, D).contiguous()
    h_post_reshaped = h_post.view(BS, N).contiguous()

    if output is None:
        output_reshaped = torch.empty(BS, N, D, dtype=torch.bfloat16, device=x.device)
    else:
        output_reshaped = output.view(BS, N, D).contiguous()

    mhc_post_kernel_bf16(
        x_reshaped, h_res_reshaped, h_out_reshaped, h_post_reshaped,
        output_reshaped, mhc_post_config
    )

    return output_reshaped.view(B, S, N, D)