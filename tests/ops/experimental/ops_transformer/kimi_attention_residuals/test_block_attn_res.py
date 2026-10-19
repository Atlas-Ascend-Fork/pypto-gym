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
"""Block Attention Residuals 正反向级联精度测试 (pytest format).

架构：唯一正向参考实现 + 唯一反向 autograd 参考实现 + kernel 级联链.
  - backward_reference 内部通过 autograd.grad 同时产出正向 h 和所有梯度，
    golden / benchmark 链无需单独计算正向，避免重复。

三方对比（三个独立计算链，共享同一份随机输入）:
  - cpu_golden : CPU fp32 autograd 正反向（真值基准）
  - benchmark  : NPU torch autograd 正反向（小算子拼接对照）
  - kernel     : PyPTO forward kernel（产出 rms/alpha cache）-> backward kernel（消费 cache）

注意：golden/benchmark 的反向使用 autograd 自动微分，内部从原始 block 重算正向，
不依赖外部 cache。cache 仅由 kernel 链的 forward kernel 产出，backward kernel 直接消费。

精度等级 L0: mare <= 10, mere <= 2, rmse <= 2

泛化规格：
  dtype: float16/bfloat16
  N: 1-127, B: 1-8, T: 1-32K
  D: 1536/2048/2560/4096/5120/6144
"""

import gc
import logging
import math
import os
import sys
from typing import List, Optional, Tuple

_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, "src")):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, "ops", "pypto_tensor"))

from experimental.ops_transformer.kimi_attention_residuals.block_attn_res_impl import (
    ai_infra_block_attn_res,
    ai_infra_block_attn_res_backward,
)
import pytest
import torch
import torch.nn.functional as functional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 精度标准 2.1 对比模块
# ---------------------------------------------------------------------------

small_value_thres_dict = {
    torch.float16: 2**-11,
    torch.bfloat16: 2**-8,
    torch.float32: 2**-14,
}

small_value_error_thres_dict = {
    torch.float16: 2**-16,
    torch.bfloat16: 2**-16,
    torch.float32: 2**-30,
}


def _get_split_index(golden_data, dtype):
    thres = small_value_thres_dict[dtype]
    large_mask = torch.abs(golden_data) >= thres
    small_mask = torch.abs(golden_data) < thres
    return large_mask, small_mask, thres


def _compute_small_value(input_data, golden_data, dtype, small_mask):
    if not torch.any(small_mask):
        return 0
    thres = small_value_error_thres_dict[dtype]
    error_count = torch.sum(torch.abs(input_data[small_mask] - golden_data[small_mask]) > thres).item()
    return error_count


def _compute_large_value(input_data, golden_data, large_mask):
    if not torch.any(large_mask):
        return 0, 0, 0
    input_large = input_data[large_mask]
    golden_large = golden_data[large_mask]
    abs_diff = torch.abs(input_large - golden_large)
    relative_error = abs_diff / (torch.abs(golden_large) + 1e-7)
    mare = torch.max(relative_error).item()
    mere = torch.mean(relative_error).item()
    rmse = torch.sqrt(torch.mean((input_large - golden_large) ** 2)).item()
    return mare, mere, rmse


def _compute_re(input_value, bm_value, small_value_thres):
    if math.isinf(bm_value) or math.isnan(bm_value):
        return 1
    if math.isinf(input_value) or math.isnan(input_value):
        return 1000
    return input_value / max(bm_value, small_value_thres)


def precision_compare_triple(npu_data, bm_data, golden_data, thres=(2, 1.2, 1.2)):
    """三方精度对比。

    Args:
        npu_data: PyPTO kernel 输出
        bm_data: NPU benchmark 输出
        golden_data: CPU fp64/fp32 golden 输出
        thres: (mare_thres, mere_thres, rmse_thres)

    Returns:
        result: "PASS" / "FAILED"
        mare_matrix, mere_matrix, rmse_matrix, small_value_matrix
    """
    dtype = npu_data.dtype

    npu_fp32 = npu_data.to(torch.float32).cpu()
    bm_fp32 = bm_data.to(torch.float32).cpu()
    golden_fp32 = golden_data.to(torch.float32).cpu()

    large_idx, small_idx, sv_thres = _get_split_index(golden_fp32, dtype)

    npu_err_count = _compute_small_value(npu_fp32, golden_fp32, dtype, small_idx)
    bm_err_count = _compute_small_value(bm_fp32, golden_fp32, dtype, small_idx)
    small_value_matrix = npu_err_count / max(bm_err_count, 1)

    mare_npu, mere_npu, rmse_npu = _compute_large_value(npu_fp32, golden_fp32, large_idx)
    mare_bm, mere_bm, rmse_bm = _compute_large_value(bm_fp32, golden_fp32, large_idx)

    mare_matrix = _compute_re(mare_npu, mare_bm, sv_thres)
    mere_matrix = _compute_re(mere_npu, mere_bm, sv_thres)
    rmse_matrix = _compute_re(rmse_npu, rmse_bm, sv_thres)

    is_pass = (
        small_value_matrix <= 2 and mare_matrix <= thres[0] and mere_matrix <= thres[1] and rmse_matrix <= thres[2]
    )

    result = "PASS" if is_pass else "FAILED"
    return result, mare_matrix, mere_matrix, rmse_matrix, small_value_matrix


def compare(npu_data, bm_data, golden_data, name=""):
    """三方精度对比包装，失败时抛出异常。"""
    result, mare, mere, rmse, sv = precision_compare_triple(npu_data, bm_data, golden_data)
    logger.info(f"  {name}: MARE={mare:.4f} MERE={mere:.4f} RMSE={rmse:.4f} SmallVal={sv:.4f} [{result}]")
    if result != "PASS":
        raise Exception(f"fail precision check: {name}")
    return result, mare, mere, rmse, sv


# ---------------------------------------------------------------------------
# Forward & Backward 参考实现
# ---------------------------------------------------------------------------
#
# 架构说明：
#   - block_attn_res_forward_reference：唯一正向参考实现
#     - CPU golden 调它（fp64 升精度）
#     - NPU benchmark 调它（保持 bf16/fp16）
#     - backward_reference 内部通过 autograd 重算（不读外部 cache）
#   - block_attn_res_backward_reference：唯一反向参考实现
#     - 内部调用 forward_reference + autograd 自动微分
#     - rms_cache/alpha_cache 参数均忽略
#   - NPU kernel 链的 cache 由真实 forward kernel 产出，不经参考实现


def block_attn_res_forward_reference(
    blocks: List[torch.Tensor],
    proj_weight: torch.Tensor,
    partial_block: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    rmsnorm_eps: float = 1e-6,
    rmsnorm_gamma: Optional[torch.Tensor] = None,
    enable_rmsnorm: bool = True,
):
    """正向参考实现。

    用途：
      - CPU golden 正向（调用侧升精度到 fp64）
      - NPU benchmark 正向（调用侧保持 bf16/fp16）
      - backward_reference 内部通过 autograd 重算（不依赖此函数返回值）
    """
    tensors = blocks + ([partial_block] if partial_block is not None else [])
    v = torch.stack(tensors, dim=2)  # [B, T, L, D]

    rms = None
    if enable_rmsnorm:
        rms = torch.sqrt(torch.mean(v**2, dim=-1, keepdim=True) + rmsnorm_eps)
        k = v / rms
        if rmsnorm_gamma is not None:
            k = k * rmsnorm_gamma
    else:
        k = v

    logits = torch.matmul(k, proj_weight)
    if not math.isclose(scale, 1.0):
        logits = logits * scale
    alpha = functional.softmax(logits, dim=2)
    h = torch.matmul(alpha.unsqueeze(-2), v).squeeze(-2)
    return h, rms, alpha


def block_attn_res_backward_reference(
    grad_h: torch.Tensor,
    blocks: List[torch.Tensor],
    partial_block: torch.Tensor,
    proj_weight: torch.Tensor,
    rmsnorm_gamma: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    rmsnorm_eps: float = 1e-6,
    enable_rmsnorm: bool = True,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """反向传播参考实现（PyTorch autograd 自动微分）。

    内部执行正向 -> autograd.grad 反向，正向输出 h 一并返回，避免调用方重复计算。

    Returns:
        h:                 正向输出 [B, T, D]
        grad_blocks:       各 block 的梯度列表
        grad_partial_block: partial_block 的梯度
        grad_proj_weight:  proj_weight 的梯度
        grad_rmsnorm_gamma: rmsnorm_gamma 的梯度（可选）
    """
    _device = grad_h.device
    dtype = grad_h.dtype

    # 正向输入副本（requires_grad=True 使 autograd 可追踪）
    fw_blocks = [b.clone().requires_grad_(True) for b in blocks]
    fw_partial_block = partial_block.clone().requires_grad_(True)
    fw_proj_weight = proj_weight.clone().requires_grad_(True)
    fw_rmsnorm_gamma = rmsnorm_gamma.clone().requires_grad_(True) if rmsnorm_gamma is not None else None

    h, rms, alpha = block_attn_res_forward_reference(
        fw_blocks,
        proj_weight=fw_proj_weight,
        partial_block=fw_partial_block,
        rmsnorm_gamma=fw_rmsnorm_gamma,
        scale=scale,
        rmsnorm_eps=rmsnorm_eps,
        enable_rmsnorm=enable_rmsnorm,
    )
    h_value = h.detach().to(dtype)  # 保存正向输出，autograd.grad 后释放图
    rms_value = rms.detach()
    alpha_value = alpha.detach()

    inputs = fw_blocks + [fw_partial_block, fw_proj_weight]
    if fw_rmsnorm_gamma is not None:
        inputs.append(fw_rmsnorm_gamma)

    grads = torch.autograd.grad(
        outputs=h,
        inputs=inputs,
        grad_outputs=grad_h,
        retain_graph=False,
        create_graph=False,
    )

    grad_blocks = [g.detach().to(dtype) for g in grads[:len(blocks)]]
    grad_partial_block = grads[len(blocks)].detach().to(dtype)
    grad_proj_weight = grads[len(blocks) + 1].detach().to(dtype)
    grad_rmsnorm_gamma = grads[len(blocks) + 2].detach().to(dtype) if fw_rmsnorm_gamma is not None else None

    return h_value, rms_value, alpha_value, grad_blocks, grad_partial_block, grad_proj_weight, grad_rmsnorm_gamma


# ---------------------------------------------------------------------------
# 测试辅助 — 输入生成
# ---------------------------------------------------------------------------


def _make_inputs(b, t, n, d, dtype, seed=42):
    """在 CPU 上生成测试输入数据。

    Returns:
        blocks, partial_block, proj_weight, rmsnorm_gamma, grad_h
    """
    torch.manual_seed(seed)
    blocks = [torch.randn(b, t, d, dtype=dtype) for _ in range(n)]
    partial_block = torch.randn(b, t, d, dtype=dtype)
    grad_h = torch.randn(b, t, d, dtype=dtype)

    # proj_weight 使用较小数值避免 softmax 过度不均匀
    proj_weight = torch.randn(d, dtype=dtype) / math.sqrt(d * 4)
    rmsnorm_gamma = torch.ones(d, dtype=dtype)

    return blocks, partial_block, proj_weight, rmsnorm_gamma, grad_h


# ---------------------------------------------------------------------------
# 级联测试执行器
# ---------------------------------------------------------------------------


def run_cascade_test(
    case_name,
    b,
    t,
    n,
    d,
    dtype_str,
    device_id=None,
    enable_rmsnorm=True,
    scale=1.0,
    has_partial_block=True,
    rms_out_flag=True,
    alpha_out_flag=True,
    thres=(2, 1.2, 1.2),
):
    """执行正反向级联测试。

    级联流程：
      1. 生成输入数据
      2. CPU golden 正向 (fp64) -> CPU golden 反向 (fp32 autograd + caches)
      3. NPU benchmark 正向 -> NPU benchmark 反向
      4. NPU kernel 正向 (产出 caches) -> NPU kernel 反向 (使用 caches)
      5. 精度对比：forward output + backward gradients
    """
    logger.info("=" * 80)
    logger.info(f"[Block Attn Res Cascade Test 2.1] {case_name}")
    logger.info(
        f"  B={b}, T={t}, N={n}, D={d}, dtype={dtype_str}, "
        f"rmsnorm={enable_rmsnorm}, scale={scale}, partial_block={has_partial_block}"
    )
    logger.info("=" * 80)

    npu_device = f"npu:{device_id}" if device_id is not None else "cpu"
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16
    _l = n + 1 if has_partial_block else n

    # -- 在 CPU 上生成输入 --
    blocks, partial_block, proj_weight, rmsnorm_gamma, grad_h = _make_inputs(b, t, n, d, dtype)
    if not has_partial_block:
        partial_block = None

    # ----------------------------------------
    # Step 1: CPU golden 正向 + 反向 (autograd, 正向 h 一并返回)
    # ----------------------------------------
    partial_for_ref = partial_block if partial_block is not None else torch.zeros(b, t, d)
    cpu_h, cpu_rms, cpu_alpha, cpu_grad_blocks, cpu_grad_partial, cpu_grad_proj, cpu_grad_gamma = (
        block_attn_res_backward_reference(
            grad_h.to(torch.float32).cpu(),
            [block.to(torch.float32).cpu() for block in blocks],
            partial_for_ref.to(torch.float32).cpu(),
            proj_weight.to(torch.float32).cpu(),
            rmsnorm_gamma.to(torch.float32).cpu() if enable_rmsnorm else None,
            scale=scale,
            enable_rmsnorm=enable_rmsnorm,
        )
    )

    # ----------------------------------------
    # Step 2: 转移到 NPU
    # ----------------------------------------
    blocks_npu = [block.to(npu_device) for block in blocks]
    partial_blk_npu = partial_block.to(npu_device) if partial_block is not None else None
    proj_weight_npu = proj_weight.to(npu_device)
    rmsnorm_gamma_npu = rmsnorm_gamma.to(npu_device) if enable_rmsnorm else None
    grad_h_npu = grad_h.to(npu_device)

    # ----------------------------------------
    # Step 3: NPU benchmark 正向 + 反向 (autograd, h 一并返回)
    # ----------------------------------------
    partial_for_bm = (
        partial_blk_npu if partial_blk_npu is not None else torch.zeros(b, t, d, dtype=dtype, device=npu_device)
    )
    bm_h, bm_rms, bm_alpha, bm_grad_blocks, bm_grad_partial, bm_grad_proj, bm_grad_gamma = (
        block_attn_res_backward_reference(
            grad_h_npu,
            blocks_npu,
            partial_for_bm,
            proj_weight_npu,
            rmsnorm_gamma_npu,
            scale=scale,
            enable_rmsnorm=enable_rmsnorm,
        )
    )

    # ----------------------------------------
    # Step 4: NPU kernel 级联 — 正向产出 cache -> 反向消费 cache
    # ----------------------------------------
    fwd_out = ai_infra_block_attn_res(
        blocks_npu,
        proj_weight_npu,
        partial_block=partial_blk_npu,
        scale=scale,
        rmsnorm_eps=1e-6,
        rmsnorm_gamma=rmsnorm_gamma_npu,
        enable_rmsnorm=enable_rmsnorm,
        rms_out_flag=rms_out_flag,
        alpha_out_flag=alpha_out_flag,
    )
    npu_h = fwd_out[0]
    npu_rms_cache = fwd_out[1] if enable_rmsnorm and rms_out_flag else None
    npu_alpha_cache = fwd_out[2] if alpha_out_flag else None

    npu_grad = ai_infra_block_attn_res_backward(
        grad_h_npu,
        blocks_npu,
        proj_weight_npu,
        npu_alpha_cache,
        partial_block=partial_blk_npu,
        rmsnorm_gamma=rmsnorm_gamma_npu,
        rms_cache=npu_rms_cache if enable_rmsnorm else None,
        scale=scale,
        enable_rmsnorm=enable_rmsnorm,
    )
    # 解包 kernel 梯度: (grad_blocks, grad_partial, grad_proj, grad_gamma?)
    npu_grad_blocks = npu_grad[0]
    npu_grad_partial = npu_grad[1]
    npu_grad_proj = npu_grad[2]
    npu_grad_gamma = npu_grad[3] if (enable_rmsnorm and len(npu_grad) > 3) else None

    # 释放 NPU 数据
    del blocks_npu, partial_blk_npu, proj_weight_npu, rmsnorm_gamma_npu, grad_h_npu
    torch.npu.empty_cache()
    gc.collect()

    # ========================================
    # Step 5: 精度对比（统一转为 fp32 在 CPU 上比较）
    # ========================================

    # -- 5.1 forward output --
    npu_h_cmp = npu_h.to(torch.float32).cpu()
    bm_h_cmp = bm_h.to(torch.float32).cpu()
    cpu_h_cmp = cpu_h.to(torch.float32).cpu()
    compare(npu_h_cmp, bm_h_cmp, cpu_h_cmp, name="fwd_h")
    del npu_h_cmp, bm_h_cmp, cpu_h_cmp

    if enable_rmsnorm and rms_out_flag:
        npu_rms_cmp = npu_rms_cache.to(torch.float32).cpu()
        bm_rms_cmp = bm_rms.to(torch.float32).cpu()
        cpu_rms_cmp = cpu_rms.to(torch.float32).cpu()
        compare(npu_rms_cmp, bm_rms_cmp, cpu_rms_cmp, name="fwd_rms_cache")
        del npu_rms_cmp, bm_rms_cmp, cpu_rms_cmp

    if alpha_out_flag:
        npu_alpha_cmp = npu_alpha_cache.to(torch.float32).cpu()
        bm_alpha_cmp = bm_alpha.to(torch.float32).cpu()
        cpu_alpha_cmp = cpu_alpha.to(torch.float32).cpu()
        compare(npu_alpha_cmp, bm_alpha_cmp, cpu_alpha_cmp, name="fwd_alpha_cache")
        del npu_alpha_cmp, bm_alpha_cmp, cpu_alpha_cmp

    # -- 5.2 grad_partial_block --
    if has_partial_block:
        npu_pb = npu_grad_partial.to(torch.float32).cpu()
        bm_pb = bm_grad_partial.to(torch.float32).cpu()
        cpu_pb = cpu_grad_partial.to(torch.float32).cpu()
        compare(npu_pb, bm_pb, cpu_pb, name="bwd_grad_partial_block")
        del npu_pb, bm_pb, cpu_pb

    # -- 5.3 grad_block[0] --
    npu_b0 = npu_grad_blocks[0].to(torch.float32).cpu()
    bm_b0 = bm_grad_blocks[0].to(torch.float32).cpu()
    cpu_b0 = cpu_grad_blocks[0].to(torch.float32).cpu()
    compare(npu_b0, bm_b0, cpu_b0, name="bwd_grad_block[0]")
    del npu_b0, bm_b0, cpu_b0

    # -- 5.4 grad_proj_weight --
    npu_pw = npu_grad_proj.to(torch.float32).cpu()
    bm_pw = bm_grad_proj.to(torch.float32).cpu()
    cpu_pw = cpu_grad_proj.to(torch.float32).cpu()
    compare(npu_pw, bm_pw, cpu_pw, name="bwd_grad_proj_weight")
    del npu_pw, bm_pw, cpu_pw

    # -- 5.5 grad_gamma --
    if enable_rmsnorm and cpu_grad_gamma is not None:
        npu_gg = npu_grad_gamma.to(torch.float32).cpu()
        bm_gg = bm_grad_gamma.to(torch.float32).cpu()
        cpu_gg = cpu_grad_gamma.to(torch.float32).cpu()
        compare(npu_gg, bm_gg, cpu_gg, name="bwd_grad_gamma")
        del npu_gg, bm_gg, cpu_gg

    # 释放所有大张量
    del npu_h, bm_h, cpu_h
    del npu_grad_blocks, npu_grad_partial, npu_grad_proj, npu_grad, npu_grad_gamma
    del cpu_grad_blocks, cpu_grad_partial, cpu_grad_proj, cpu_grad_gamma
    del bm_grad_blocks, bm_grad_partial, bm_grad_proj, bm_grad_gamma
    del npu_rms_cache, npu_alpha_cache
    torch.npu.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# pytest 测试用例
# ---------------------------------------------------------------------------

@pytest.mark.soc("950", "910")
@pytest.mark.parametrize(
    ("b", "t", "n", "d", "dtype_str", "enable_rmsnorm", "scale", "has_partial_block"),
    [
        pytest.param(2, 4096, 25, 512, "bf16", True, 1.0, True, id="b2_t4096_n25_d512_bf16"),
        pytest.param(1, 1023, 32, 512, "bf16", True, 1.0, True, id="b1_t1023_n32_d512_bf16"),
    ],
)
def test_block_attn_res(b, t, n, d, dtype_str, enable_rmsnorm, scale, has_partial_block):
    """Block Attention Residuals 正反向级联精度测试。"""
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    case_name = f"b{b}_t{t}_n{n}_d{d}_{dtype_str}"
    if not enable_rmsnorm:
        case_name += "_no_rmsnorm"
    if scale != 1.0:
        case_name += f"_scale{scale}"
    if not has_partial_block:
        case_name += "_no_partial"

    try:
        run_cascade_test(
            case_name=case_name,
            b=b,
            t=t,
            n=n,
            d=d,
            dtype_str=dtype_str,
            device_id=device_id,
            enable_rmsnorm=enable_rmsnorm,
            scale=scale,
            has_partial_block=has_partial_block,
        )
    finally:
        torch.npu.empty_cache()
        gc.collect()


if __name__ == "__main__":
    test_block_attn_res(1, 1024, 25, 512, "bf16", True, 1.0, True)
