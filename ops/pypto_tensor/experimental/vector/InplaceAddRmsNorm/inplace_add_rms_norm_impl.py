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
PyPTO inplace_add_rms_norm 算子实现

功能：融合 elementwise add + RMSNorm
公式：
    x_add = x1 + x2                              # [B,S,H]
    ms    = mean(x_add^2, dim=-1, keepdim=True)  # [B,S,1]
    rstd  = 1 / sqrt(ms + eps)                   # [B,S,1]
    y     = x_add * rstd * gamma                 # [B,S,H]

实现策略：
    PyPTO kernel层不支持对输入tensor进行inplace修改（会导致循环依赖），
    因此采用以下策略：
    1. Kernel层面：输入tensor只读，创建独立输出tensor
    2. Wrapper层面：使用torch.copy_()实现inplace语义（保证data_ptr不变）

Inplace语义验证：
    - x1/x2 的 data_ptr 在调用前后不变
    - x1/x2 的内容被计算结果覆写
    - rstd 是新建的输出tensor

测试验证：所有测试级别（level0-level5）均通过精度验证。
"""

import pypto
import torch
import torch_npu


@pypto.frontend.jit(
    pass_options={
        "vec_nbuffer_setting": {-2: 1, -1: 8}
    }
)
def inplace_add_rms_norm_kernel_bf16(
    x1: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16), 
    x2: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16), 
    gamma: pypto.Tensor([pypto.STATIC], pypto.DT_BF16), 
    y_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16), 
    x_add_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16), 
    rstd_out: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_BF16), 
    eps: float,
):

    BS = x1.shape[0]  
    H = x1.shape[1]   

    mean_coeff = 1.0 / H

    pypto.set_vec_tile_shapes(1, H)
    gamma_2d = pypto.reshape(gamma, [1, H], inplace=True)  
    gamma_fp32 = pypto.cast(gamma_2d, pypto.DT_FP32)

    bs_tile = 2
    if pypto.platform.npuarch == 'DAV_3510':
        bs_tile = 1

    for bs_idx, unroll_length in pypto.loop_unroll(
    0, BS, 1, name="LOOP_BS", idx_name="bs_idx", unroll_list=[
        64, 16, 4, 2, 1]):
        x1_row = x1[bs_idx:bs_idx + unroll_length, :]
        x2_row = x2[bs_idx:bs_idx + unroll_length, :]

        pypto.set_vec_tile_shapes(bs_tile, H)
        x1_fp32 = pypto.cast(x1_row, pypto.DT_FP32)
        x2_fp32 = pypto.cast(x2_row, pypto.DT_FP32)

        x_add_fp32 = pypto.add(x1_fp32, x2_fp32)
        square = pypto.mul(x_add_fp32, x_add_fp32)
        square_sum = pypto.sum(square, dim=-1, keepdim=True)

        pypto.set_vec_tile_shapes(bs_tile, 1)
        mean_square = pypto.mul(square_sum, mean_coeff)

        ms_plus_eps = pypto.add(mean_square, eps)
        rstd_fp32 = pypto.rsqrt(ms_plus_eps)
        rstd_bf16 = pypto.cast(rstd_fp32, pypto.DT_BF16)

        pypto.set_vec_tile_shapes(bs_tile, H)
        y_fp32 = pypto.mul(x_add_fp32, rstd_fp32)
        y_fp32_scaled = pypto.mul(y_fp32, gamma_fp32)

        y_bf16 = pypto.cast(y_fp32_scaled, pypto.DT_BF16)
        x_add_bf16 = pypto.cast(x_add_fp32, pypto.DT_BF16)

        pypto.assemble(y_bf16, [bs_idx, 0], y_out)
        pypto.assemble(x_add_bf16, [bs_idx, 0], x_add_out)
        pypto.assemble(rstd_bf16, [bs_idx, 0], rstd_out)


def npu_inplace_add_rms_norm(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1.0e-6,
):

    assert x1.is_contiguous(), "x1 must be contiguous"
    assert x2.is_contiguous(), "x2 must be contiguous"
    assert gamma.is_contiguous(), "gamma must be contiguous"

    B = x1.shape[0]
    S = x1.shape[1]
    H = x1.shape[2]

    assert x2.shape == (B, S, H), \
        f"x2 shape mismatch: expected {(B, S, H)}, got {x2.shape}"
    assert gamma.shape == (H,), \
        f"gamma shape mismatch: expected {(H,)}, got {gamma.shape}"
    assert x1.dtype == torch.bfloat16, \
        f"x1 dtype must be bfloat16, got {x1.dtype}"
    assert x2.dtype == torch.bfloat16, \
        f"x2 dtype must be bfloat16, got {x2.dtype}"
    assert gamma.dtype == torch.bfloat16, \
        f"gamma dtype must be bfloat16, got {gamma.dtype}"

    BS = B * S
    x1_reshaped = x1.view(BS, H).contiguous()
    x2_reshaped = x2.view(BS, H).contiguous()

    y_out_reshaped = torch.empty(BS, H, dtype=torch.bfloat16, device=x1.device)
    x_add_out_reshaped = torch.empty(BS, H, dtype=torch.bfloat16, device=x1.device)
    rstd_reshaped = torch.empty(BS, 1, dtype=torch.bfloat16, device=x1.device)

    inplace_add_rms_norm_kernel_bf16(
        x1_reshaped, x2_reshaped, gamma,
        y_out_reshaped, x_add_out_reshaped, rstd_reshaped,
        eps
    )

    x1.copy_(y_out_reshaped.view(B, S, H))       
    x2.copy_(x_add_out_reshaped.view(B, S, H))   
    rstd = rstd_reshaped.view(B, S, 1)

    return x1, x2, rstd
