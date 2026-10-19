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

"""PyPTO mhc_pre kernel implementation.

MHC Pre-processing 算子实现 - Multi-Head Context 前处理

计算流程:
    1. RMSNorm: 对输入 x 进行归一化 
    2. MatMul: 与权重矩阵 phi_T 进行矩阵乘法 
    3. Split: 将结果分流为三路（Pre、Post、Comb）
    4. 三分枝处理:
       - Branch Pre: sigmoid + 加权求和 → h_in (BF16) 
       - Branch Post: sigmoid + 缩放 → h_post (FP32) 
       - Branch Comb: 加权 → h_res (FP32) 
"""

from typing import Tuple
import pypto
import torch


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def compute_rmsnorm_rsqrt(X_flat: pypto.Tensor, N_D: int, norm_eps: float) -> pypto.Tensor:
    """计算 RMSNorm 的平方根倒数归一化系数
    
    RMSNorm 公式: rsqrt(mean(X²) + eps)
    
    Args:
        X_flat: 输入 tensor，shape [unroll_length, N_D]，dtype FP32
        N_D: 特征维度大小（N*D），用于计算均值系数
        norm_eps: 防止除零的 epsilon 值
    
    Returns:
        rsqrt_val: 归一化系数，shape [unroll_length, 1]，dtype FP32
    """
    # Step 2.1: 计算平方
    X_sq = pypto.mul(X_flat, X_flat)  # [unroll_length, N_D] FP32
    
    # Step 2.2: 沿尾轴求和
    mean_val = pypto.sum(X_sq, -1, keepdim=True)  # [unroll_length, 1] FP32
    
    # Step 2.3: 计算均值（使用 sum + div 替代 mean API）
    mean_coeff = 1.0 / N_D  # Python float（均值系数 = 1/N_D）
    variance = pypto.mul(mean_val, mean_coeff)  # [unroll_length, 1] FP32（均值）
    
    # Step 2.4: 加 epsilon 防止除零
    variance_eps = pypto.add(variance, norm_eps)  # [unroll_length, 1] FP32
    
    # Step 2.5: 计算平方根倒数
    rsqrt_val = pypto.rsqrt(variance_eps)  # [unroll_length, 1] FP32
    
    return rsqrt_val


# ─────────────────────────────────────────────
# JIT Kernel
# ─────────────────────────────────────────────

@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 128, 
        "device_sched_mode": 1}, 
    pass_options={"cube_nbuffer_setting": {-1: 4}})
def mhc_pre_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    phi_T: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),           # [N*D, N²+2N] 固定值
    bias_pre: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),               # [N] - 已切片（Step 5 使用）
    bias_post: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),              # [N] - 已切片（Step 6 使用）
    bias_comb: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),              # [N*N] - 已切片（Step 7 使用）
    h_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),  # [B*S, D]
    h_post: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),   # [B*S, N]
    h_res: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),  # [B*S, N, N] 3D 输出
    alpha_0: float = 1.0,  # Python float scalar (Step 5 使用)
    alpha_1: float = 1.0,  # Python float scalar (Step 6 使用)
    alpha_2: float = 1.0,  # Python float scalar (Step 7 使用)
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
):
    """mhc_pre kernel - Multi-Head Context 前处理算子

    注意:
        - phi_T 是预处理后的转置权重
        - B*S 是动态轴
        - N=8, D=5120 使用固定值（替代 STATIC 以解决 tile shape 兼容性）
        - 使用 loop_unroll 处理 BS 轴，unroll_length=128，tileshape=1
    """
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_pass_options(vec_nbuffer_setting={"DEFAULT": 4, "func8_3": 1})
    # ─────────────────────────────────────────────
    # 从 x.shape 获取维度（动态轴 vs 静态轴）
    # ─────────────────────────────────────────────
    BS = x.shape[0]      # 动态轴，SymbolicScalar（B*S）

    # N 和 D 从 x.shape 获取（静态轴，编译期常量）
    N = x.shape[1]       # 静态轴，pypto.STATIC 类型（如 N=8）
    D = x.shape[2]       # 静态轴，pypto.STATIC 类型（如 D=5120）
    N_D = N * D          # 派生常量（如 N_D=40960）
    N_SQUARED_PLUS_2N = N * N + 2 * N  # 派生常量（如 80）

    # ─────────────────────────────────────────
    # 预处理静态参数（loop 外）
    # ─────────────────────────────────────────
    # bias 已经在 wrapper 中切片，直接 reshape 为 2D（用于广播）
    # Step 5 使用 bias_pre
    bias_pre_2d = pypto.reshape(bias_pre, [1, N], inplace=True)   # [1, N] FP32

    # Step 6 使用 bias_post
    bias_post_2d = pypto.reshape(bias_post, [1, N], inplace=True)  # [1, N] FP32

    # Step 7 使用 bias_comb
    bias_comb_2d = pypto.reshape(bias_comb, [1, N * N], inplace=True)  # [1, N*N] FP32

    # ─────────────────────────────────────────
    # 在 loop 外部 reshape tensor（参考 mhc_post_impl.py）
    # ─────────────────────────────────────────
    # x: [BS, N, D] BF16 → x_flat: [BS, N*D] BF16
    x_flat = pypto.reshape(x, [BS, N_D], inplace=True)  # [BS, N*D] BF16

    # ─────────────────────────────────────────
    # loop_unroll 处理 BS 轴（参考 mhc_post_impl.py）
    # ─────────────────────────────────────────
    bs_tile = 1
    bs_tile_2 = 1
    D_tile = 2048

    if D < 2048:
        bs_tile = 8
        D_tile = 128
    else:
        bs_tile = 1
        D_tile = 2048
        
    for bs_idx, unroll_length in pypto.loop_unroll(0, BS, 1, name="LOOP_BS", idx_name="bs_idx", unroll_list=[16, 8, 1]):
        # 提取当前 slice（从已 reshape 的 tensor 切片）
        x_slice_flat = x_flat[bs_idx: bs_idx + unroll_length, :]  # [unroll_length, N*D] BF16
        x_slice_3d = x[bs_idx: bs_idx + unroll_length, :, :]  # [unroll_length, N, D] BF16 (用于 Step 5)

        # Step 1: cast to FP32（用于 RMSNorm 和 MatMul）
        pypto.set_vec_tile_shapes(bs_tile * 8, D_tile)
        pypto.set_pass_options(sg_set_scope=1)
        X_flat = pypto.cast(x_slice_flat, pypto.DT_FP32)  # [unroll_length, N*D] FP32
        pypto.set_pass_options(sg_set_scope=-1)

        # ─────────────────────────────────────────
        # Step 2: RMSNorm（Root Mean Square Layer Normalization）
        # ─────────────────────────────────────────
        # 计算 rsqrt = 1 / sqrt(mean(X_flat²) + norm_eps)
        pypto.set_pass_options(sg_set_scope=2)
        pypto.set_vec_tile_shapes(bs_tile, D_tile)  # tileshape=1
        rsqrt_val = compute_rmsnorm_rsqrt(X_flat, N_D, norm_eps)  # [unroll_length, 1] FP32
        pypto.set_pass_options(sg_set_scope=-1)

        # ─────────────────────────────────────────
        # Step 3: MatMul with Normalization
        # ─────────────────────────────────────────
        # matmul 需要正确设置 vec tile shapes 和 cube tile shapes
        # ✅ 使用 FP32 MatMul（符合 DESIGN.md 约束）
        # DESIGN.md 约束: matmul 两侧 dtype 一致，X_flat 和 phi_T 都是 FP32

        # MatMul: [unroll_length, N_D] @ [N_D, N_SQUARED_PLUS_2N] -> [unroll_length, N_SQUARED_PLUS_2N]
        # A: X_flat [unroll_length, N_D] FP32
        # B: phi_T [N_D, N_SQUARED_PLUS_2N] FP32
        # 输出: FP32
        pypto.set_cube_tile_shapes([16, 16], [512, 1024], [128, 128], enable_split_k=True)
        X_hat = pypto.matmul(X_flat, phi_T, pypto.DT_FP32)  # [unroll_length, N_SQUARED_PLUS_2N] FP32

        pypto.set_pass_options(sg_set_scope=3)
        pypto.set_vec_tile_shapes(bs_tile, D_tile)  # tileshape=1
        X_hat_norm = pypto.mul(X_hat, rsqrt_val)  # [unroll_length, N_SQUARED_PLUS_2N] FP32（归一化）

        # ─────────────────────────────────────────
        # Step 4: Split（分流为三路）
        # ─────────────────────────────────────────
        # 使用切片替代 torch.split
        X_pre = X_hat_norm[:, 0:N]  # [unroll_length, N] FP32
        X_post = X_hat_norm[:, N:2 * N]  # [unroll_length, N] FP32
        X_comb = X_hat_norm[:, 2 * N:2 * N + N * N]  # [unroll_length, N²] FP32

        
        # ─────────────────────────────────────────
        # Step 5: Branch Pre
        # ─────────────────────────────────────────

        # 注意：pypto.mul 需要 (Tensor, float) 顺序，不支持 (float, Tensor)
        pypto.set_vec_tile_shapes(bs_tile, N)  # tileshape=1
        scaled_X_pre = pypto.mul(X_pre, alpha_0)  # [unroll_length, N] FP32

        # add bias
        X_pre_bias = pypto.add(scaled_X_pre, bias_pre_2d)  # [unroll_length, N] FP32（bias 广播）

        # sigmoid
        H_pre = pypto.sigmoid(X_pre_bias)  # [unroll_length, N] FP32

        # add hc_eps
        H_pre_eps = pypto.add(H_pre, hc_eps)  # [unroll_length, N] FP32

        # reshape to 3D (equivalent to unsqueeze at -1)
        h_pre_expanded = pypto.reshape(H_pre_eps, [unroll_length, N, 1])  # [unroll_length, N, 1] FP32

        # weighted_x
        pypto.set_vec_tile_shapes(bs_tile, N, D_tile)  # tileshape=1
        x_fp32_3d = pypto.cast(x_slice_3d, pypto.DT_FP32)  # [unroll_length, N, D] FP32
        weighted_x = pypto.mul(h_pre_expanded, x_fp32_3d)  # [unroll_length, N, D] FP32

        # sum
        h_in_fp32 = pypto.sum(weighted_x, 1)  # [unroll_length, D] FP32

        # cast to BF16
        pypto.set_vec_tile_shapes(bs_tile, D_tile)  # tileshape=1
        h_in_tile = pypto.cast(h_in_fp32, pypto.DT_BF16)  # [unroll_length, D] BF16

        # ─────────────────────────────────────────
        # Step 6: Branch Post
        # ─────────────────────────────────────────

        pypto.set_vec_tile_shapes(bs_tile_2, N)  # tileshape=1
        scaled_X_post = pypto.mul(X_post, alpha_1)  # [unroll_length, N] FP32

        X_post_bias = pypto.add(scaled_X_post, bias_post_2d)  # [unroll_length, N] FP32

        H_post = pypto.sigmoid(X_post_bias)  # [unroll_length, N] FP32

        h_post_tile = pypto.mul(H_post, 2.0)  # [unroll_length, N] FP32
        
        # ─────────────────────────────────────────
        # Step 7: Branch Comb
        # ─────────────────────────────────────────

        # 注意：X_comb 是 [unroll_length, N*N] 2D tensor
        # bias_comb_2d 是 [1, N*N] 2D tensor（已 reshape）
        # 先计算为 2D [unroll_length, N*N]，然后 reshape 为 3D [unroll_length, N, N]

        pypto.set_vec_tile_shapes(bs_tile_2, N * N)  # tileshape=1
        scaled_X_comb = pypto.mul(X_comb, alpha_2)  # [unroll_length, N*N] FP32

        # add bias（使用 2D bias 广播）
        h_res_2d = pypto.add(scaled_X_comb, bias_comb_2d)  # [unroll_length, N*N] FP32
        
        # reshape to 3D [unroll_length, N, N]（非 inplace，避免内存问题）
        # 关键：tile shape 最后一维必须满足 32 字节对齐（参考 hc_pre_impl.py）
        # 对于 FP32，需要 last_dim * 4 bytes >= 32 bytes，即 last_dim >= 8
        # 但参考 hc_split_sinkhorn，设置 last_dim=32 可满足所有情况
        pypto.set_vec_tile_shapes(bs_tile_2, 16, 32)  # 使用固定的对齐 tile shape
        h_res_tile = pypto.reshape(h_res_2d, [unroll_length, N, N])  # [unroll_length, N, N] FP32

        # ─────────────────────────────────────────
        # Step 8: 输出写回（使用 assemble）
        # ─────────────────────────────────────────
        pypto.assemble(h_in_tile, [bs_idx, 0], h_in)
        pypto.assemble(h_post_tile, [bs_idx, 0], h_post)
        pypto.assemble(h_res_tile, [bs_idx, 0, 0], h_res)  # 3D tensor 需要 3 个索引
        pypto.set_pass_options(sg_set_scope=-1)

# ─────────────────────────────────────────────
# Wrapper 函数（导出接口）
# ─────────────────────────────────────────────


def mhc_pre_wrapper(
    x: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 1. phi 转置：[N²+2N, N*D] = [80, 40960] → [N*D, N²+2N] = [40960, 80]
    # 注意：需要 contiguous，否则 PyPTO 会报错
    phi_T = phi.T.contiguous()
    
    # 2. 提取 alpha 值为 Python float（避免 tensor tile shape 问题）
    alpha_0 = float(alpha[0].item())
    alpha_1 = float(alpha[1].item())  # Step 6 使用
    alpha_2 = float(alpha[2].item())  # Step 7 使用
    
    # 3. 从 bias 中切片出 bias_pre、bias_post 和 bias_comb（在 Python 层完成，避免 PyPTO view 问题）
    N = x.shape[1]  # 从输入获取 N
    bias_pre = bias[:N].contiguous()          # [N] FP32（Step 5 使用）
    bias_post = bias[N:2 * N].contiguous()      # [N] FP32（Step 6 使用）
    bias_comb = bias[2 * N:].contiguous()       # [N*N] FP32（Step 7 使用）
    
    # 4. 创建输出 tensor（确保在同一设备上）
    bs = x.shape[0]
    D = x.shape[2]  # 从输入获取 D
    device = x.device
    h_in = torch.empty(bs, D, dtype=torch.bfloat16, device=device)
    h_post = torch.empty(bs, N, dtype=torch.float32, device=device)
    h_res = torch.empty(bs, N, N, dtype=torch.float32, device=device)  # 3D 输出 [B*S, N, N]
    
    # 5. 调用 kernel（传入已切片的 bias_pre、bias_post 和 bias_comb）
    mhc_pre_kernel(x, phi_T, bias_pre, bias_post, bias_comb,
                   h_in, h_post, h_res, 
                   alpha_0, alpha_1, alpha_2, norm_eps, hc_eps)
    
    return h_in, h_post, h_res